"""FastAPI service for RLCD task classification and model routing."""

import hashlib
import os
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from core.schema import StructuredSchema
from core.engine import (
    MODEL_ID,
    list_loaded_models,
    run_parallel_generation,
    run_parallel_for_model,
    run_text_generation,
)

from router.router_schema import ROUTER_SCHEMA
from router.signals import extract_signals
from router.classifier import RouterClassifier, bypass_classification
from router.policy import choose_route
from router.telemetry import TelemetryLog
from router.config import TIER_DEFINITIONS, MODEL_PROFILES, CONFIDENCE_THRESHOLDS

app = FastAPI(title="RLCD Multi-Model Router")


def _prompt_hash(prompt: str) -> str:
    """Return a stable, non-reversible identifier for telemetry correlation."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

_cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "ROUTER_CORS_ORIGINS",
        "http://127.0.0.1:8001,http://localhost:8001",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health():
    """Identify this process so clients do not mistake another service for the router."""
    return {"status": "ok", "service": "rlcd-router"}


# Router state (lazy-loaded on first route call)
_router_classifier: Optional[RouterClassifier] = None
_telemetry_log = TelemetryLog(
    log_path=os.environ.get(
        "ROUTER_TELEMETRY_PATH",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "router_events.jsonl"),
    )
)


class PredictRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    context: str
    schema_def: Dict[str, Any] = Field(..., alias="schema")
    temperature: Optional[float] = None

@app.post("/api/run-parallel")
@app.post("/api/run-rlcd")
def api_run_parallel(req: PredictRequest):
    try:
        schema = StructuredSchema(req.schema_def)
        temp = req.temperature if req.temperature is not None else 1.0
        res = run_parallel_generation(req.context, schema, temperature=temp)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ===================================================================
# ROUTER API
# ===================================================================

class RouteRequest(BaseModel):
    prompt: str
    runtime: Dict[str, Any] = {}
    options: Dict[str, Any] = {}
    model_override: Optional[str] = None  # user-requested model profile name


class OutcomeReport(BaseModel):
    event_id: str
    status: str          # "success" | "failed"
    failure_reason: str = ""
    verification: str = ""
    actual_model: str = ""
    latency_ms: float = 0.0
    cost: float = 0.0


def _get_router_classifier() -> RouterClassifier:
    """Lazily build the classifier.

    Single-stage: Qwen3-Coder-30B-A3B-Instruct-4bit (~900 ms)
    Uses the full 11-field schema.  The 1.5B model was tested but
    can't reliably classify 11 fields (mean confidence ~0.73).
    The Coder at ~900 ms is fast enough for routing.

    The two-stage infrastructure (FAST_ROUTER_SCHEMA, etc.) is
    preserved for future use with a better small model.
    """
    global _router_classifier
    if _router_classifier is not None:
        return _router_classifier

    # Use the Coder for classification (single stage)
    model_id = MODEL_PROFILES["rlcd_smart"].model_id

    def run_rlcd(context: str, schema_dict: dict) -> dict:
        """RLCD classification using Qwen3-Coder-30B-A3B."""
        schema = StructuredSchema(schema_dict)
        return run_parallel_for_model(model_id, context, schema)

    _router_classifier = RouterClassifier(
        run_fast=run_rlcd,
        run_smart=None,  # single stage
    )
    return _router_classifier


def _tier_for_profile(profile_name: str, profile: Any = None) -> str:
    """Find the best matching tier for a given model profile.

    Checks tier definitions first; if the profile appears in a tier's
    preferred list, return that tier.  Otherwise infer from the profile's
    provider and capabilities.
    """
    # Check if any tier lists this profile
    for tier_name, td in TIER_DEFINITIONS.items():
        if profile_name in td.preferred_model_profiles:
            return tier_name
    # Infer from profile metadata
    if profile is None:
        profile = MODEL_PROFILES.get(profile_name)
    if profile:
        if profile.billing == "local":
            if "coding" in profile.capabilities:
                return "local_coding"
            return "local_general"
        if "research" in profile.capabilities:
            return "cloud_research"
        if "coding" in profile.capabilities:
            return "cloud_coding"
        if "reasoning" in profile.capabilities:
            return "cloud_reasoning"
        return "cloud_standard"
    return "cloud_standard"


@app.post("/api/route")
def api_route(req: RouteRequest):
    """Classify a prompt and return a routing decision.

    This is the primary entry point for PI to decide which model
    tier should handle a given task.  Returns:
    - classification (what the task needs)
    - tier (which routing tier to use)
    - requirements (what the chosen tier must support)
    - escalation config (what to try if it fails)
    """
    try:
        # --- User model override: skip classification entirely ---
        if req.model_override:
            profile_name = req.model_override
            executor_profiles = sorted(
                name for name, candidate in MODEL_PROFILES.items()
                if candidate.role == "executor"
            )
            if profile_name not in MODEL_PROFILES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown model profile: {profile_name}. "
                           f"Available: {', '.join(executor_profiles)}",
                )
            profile = MODEL_PROFILES[profile_name]
            if profile.role != "executor":
                raise HTTPException(
                    status_code=400,
                    detail=f"Override must name an executor model profile. "
                           f"Available: {', '.join(executor_profiles)}",
                )
            # Pick the tier that contains this profile, or infer from profile
            tier = _tier_for_profile(profile_name, profile)
            cls_result = bypass_classification("user_model_override")
            signals = extract_signals(
                req.prompt,
                prior_attempts=req.runtime.get("prior_attempts", 0),
                prior_failures=req.runtime.get("prior_failures", 0),
                tool_calls_so_far=req.runtime.get("tool_calls_so_far", 0),
                deadline=req.runtime.get("deadline", "normal"),
                user_hint=req.runtime.get("user_hint", ""),
                repo_root=req.runtime.get("repo_root"),
            )
            from router.decision import RouteDecision, TierRequirement
            td = TIER_DEFINITIONS.get(tier)
            req_obj = TierRequirement(
                minimum_capability=td.minimum_capability if td else "general",
                requires_tools=td.supports_tools if td else (),
                requires_verification=False,
                allow_cloud=not (td.is_local if td else False),
                allow_weak_models=True,
                requires_web=td.supports_web if td else False,
            )
            decision = RouteDecision(
                tier=tier,
                eligible_models=(profile_name,),
                requirements=req_obj,
                classifier_used="user_model_override",
                confidence=1.0,
                critical_confidence=1.0,
                reasons=("user_requested_model", f"override:{profile_name}"),
                escalation_config={},
            )
            event = _telemetry_log.new_event(
                prompt_hash=_prompt_hash(req.prompt),
                prompt_chars=signals.prompt_chars,
                estimated_tokens=signals.estimated_tokens,
                classifier_used="user_model_override",
                classification=cls_result.fields,
                field_confidence={},
                overall_confidence=1.0,
                critical_confidence=1.0,
                abstain=False,
                selected_tier=tier,
                reasons=["user_requested_model", f"override:{profile_name}"],
                requirements=req_obj.to_dict(),
                classifier_latency_ms=0.0,
            )
            _telemetry_log.record(event)
            return {
                "route_id": event.event_id,
                "classification": cls_result.to_dict(),
                "signals": signals.to_dict(),
                "decision": decision.to_dict(),
            }

        signals = extract_signals(
            req.prompt,
            prior_attempts=req.runtime.get("prior_attempts", 0),
            prior_failures=req.runtime.get("prior_failures", 0),
            tool_calls_so_far=req.runtime.get("tool_calls_so_far", 0),
            deadline=req.runtime.get("deadline", "normal"),
            user_hint=req.runtime.get("user_hint", ""),
            repo_root=req.runtime.get("repo_root"),
        )

        # Skip model loading if caller explicitly requests bypass
        if req.options.get("bypass", False):
            cls_result = bypass_classification("caller_requested_bypass")
        else:
            classifier = _get_router_classifier()
            cls_result = classifier.classify(req.prompt, signals)

        decision = choose_route(
            classification=cls_result.fields,
            signals=signals,
            overall_confidence=cls_result.overall_confidence,
            critical_confidence=cls_result.critical_confidence,
            classifier_used=cls_result.classifier_used,
            abstain=cls_result.abstain,
        )

        # Record telemetry
        event = _telemetry_log.new_event(
            prompt_hash=_prompt_hash(req.prompt),
            prompt_chars=signals.prompt_chars,
            estimated_tokens=signals.estimated_tokens,
            classifier_used=cls_result.classifier_used,
            classification=cls_result.fields,
            field_confidence={k: v for k, v in cls_result.field_confidence.items()},
            overall_confidence=cls_result.overall_confidence,
            critical_confidence=cls_result.critical_confidence,
            abstain=cls_result.abstain,
            selected_tier=decision.tier,
            reasons=list(decision.reasons),
            requirements=decision.requirements.to_dict(),
            classifier_latency_ms=cls_result.latency_ms,
        )
        _telemetry_log.record(event)

        return {
            "route_id": event.event_id,
            "classification": cls_result.to_dict(),
            "signals": signals.to_dict(),
            "decision": decision.to_dict(),
        }
    except HTTPException:
        raise  # re-raise 4xx errors as-is
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/route/outcome")
def api_report_outcome(report: OutcomeReport):
    """Report the outcome of a routing decision.

    PI calls this after the chosen model completes (or fails) to
    provide feedback for telemetry and escalation decisions.
    """
    # Find the event and update it
    for event in _telemetry_log.events:
        if event.event_id == report.event_id:
            event.status = report.status
            event.failure_reason = report.failure_reason
            event.verification = report.verification
            event.actual_model = report.actual_model
            event.latency_ms += report.latency_ms
            event.cost += report.cost
            _telemetry_log.record(event)  # re-record with updates
            return {"status": "recorded", "event_id": report.event_id}

    raise HTTPException(status_code=404, detail=f"event_id {report.event_id} not found")


@app.get("/api/route/config")
def api_router_config():
    """Return the current router configuration: tiers, models, thresholds."""
    return {
        "tiers": {k: v.to_dict() for k, v in TIER_DEFINITIONS.items()},
        "models": {k: v.to_dict() for k, v in MODEL_PROFILES.items()},
        "thresholds": CONFIDENCE_THRESHOLDS.to_dict(),
        "schema_fields": list(ROUTER_SCHEMA.keys()),
    }


@app.get("/api/route/telemetry")
def api_router_telemetry():
    """Return aggregate telemetry metrics."""
    return _telemetry_log.summary()


# ===================================================================
# EXECUTE API – run tasks through local models
# ===================================================================

# Default system prompts per tier
TIER_SYSTEM_PROMPTS: Dict[str, str] = {
    "local_fast": "You are a helpful assistant. Be concise and direct.",
    "local_general": "You are a helpful assistant with strong reasoning skills.",
    "local_coding": (
        "You are an expert software engineer. Write clean, correct, "
        "well-tested code. Follow best practices. When editing existing "
        "code, match its style. Output only the necessary code and "
        "brief explanations."
    ),
    "local_reasoning": (
        "You are a careful analytical thinker. Reason step by step. "
        "Consider edge cases and trade-offs before concluding."
    ),
}


class ExecuteRequest(BaseModel):
    prompt: str
    tier: str = "local_coding"
    model_profile: Optional[str] = None   # override: explicit profile name
    model_id: Optional[str] = None        # override: explicit model path/ID
    max_tokens: int = 2048
    temperature: float = 0.2
    system_prompt: Optional[str] = None


@app.post("/api/execute")
def api_execute(req: ExecuteRequest):
    """Execute a task through a local model.

    PI calls this after routing to run the actual task.  The tier
    determines which local model handles the request:

      - local_fast   → Qwen2.5-1.5B
      - local_coding → Qwen3-Coder-30B-A3B
      - local_general / local_reasoning → Qwen3.8-27B

    Cloud tiers are NOT handled here — PI routes those directly
    through its own provider layer.

    Returns:
      - model_id, model_name, tier
      - text (the generated response)
      - elapsed_ms, total_tokens, tokens_per_second
    """
    try:
        # Resolve which model to use
        if req.model_id:
            model_id = req.model_id
        elif req.model_profile and req.model_profile in MODEL_PROFILES:
            model_id = MODEL_PROFILES[req.model_profile].model_id
        elif req.tier in TIER_DEFINITIONS and TIER_DEFINITIONS[req.tier].is_local:
            td = TIER_DEFINITIONS[req.tier]
            if td.preferred_model_profiles and td.preferred_model_profiles[0] in MODEL_PROFILES:
                model_id = MODEL_PROFILES[td.preferred_model_profiles[0]].model_id
            else:
                model_id = MODEL_ID
        else:
            model_id = MODEL_ID

        # Resolve system prompt
        sys_prompt = req.system_prompt or TIER_SYSTEM_PROMPTS.get(req.tier, "")

        # Run generation
        result = run_text_generation(
            model_id=model_id,
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            system_prompt=sys_prompt if sys_prompt else None,
        )

        return {
            "model_id": model_id,
            "model_name": os.path.basename(model_id.rstrip("/")),
            "tier": req.tier,
            "text": result["text"],
            "elapsed_ms": result["elapsed_ms"],
            "total_tokens": result["total_tokens"],
            "tokens_per_second": result["tokens_per_second"],
            "mode": result["mode"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RouteAndExecuteRequest(BaseModel):
    """Combined: classify the prompt, then execute through the chosen tier."""
    prompt: str
    runtime: Dict[str, Any] = {}
    options: Dict[str, Any] = {}
    model_override: Optional[str] = None  # user-requested model profile name
    max_tokens: int = 2048
    temperature: float = 0.2
    system_prompt: Optional[str] = None


@app.post("/api/route-and-execute")
def api_route_and_execute(req: RouteAndExecuteRequest):
    """Full pipeline: classify → select tier → execute locally.

    If the selected tier is a cloud tier, returns the routing
    decision without executing (PI handles cloud tiers itself).
    """
    try:
        # --- User model override: skip classification ---
        if req.model_override:
            route_resp = api_route(RouteRequest(
                prompt=req.prompt,
                runtime=req.runtime,
                options=req.options,
                model_override=req.model_override,
            ))
            decision_dict = route_resp["decision"]
            tier = decision_dict["tier"]
            profile_name = req.model_override
            profile = MODEL_PROFILES.get(profile_name)
            td = TIER_DEFINITIONS.get(tier)

            execution_result = None
            if td and td.is_local and profile:
                model_id = profile.model_id
                sys_prompt = req.system_prompt or TIER_SYSTEM_PROMPTS.get(tier, "")
                exec_result = run_text_generation(
                    model_id=model_id,
                    prompt=req.prompt,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    system_prompt=sys_prompt if sys_prompt else None,
                )
                execution_result = {
                    "model_id": model_id,
                    "model_name": os.path.basename(model_id.rstrip("/")),
                    "text": exec_result["text"],
                    "elapsed_ms": exec_result["elapsed_ms"],
                    "total_tokens": exec_result["total_tokens"],
                    "tokens_per_second": exec_result["tokens_per_second"],
                }

            return {
                "route_id": route_resp["route_id"],
                "classification": route_resp["classification"],
                "signals": route_resp["signals"],
                "decision": decision_dict,
                "execution": execution_result,
            }

        # --- Stage 1: Route ---
        signals = extract_signals(
            req.prompt,
            prior_attempts=req.runtime.get("prior_attempts", 0),
            prior_failures=req.runtime.get("prior_failures", 0),
            tool_calls_so_far=req.runtime.get("tool_calls_so_far", 0),
            deadline=req.runtime.get("deadline", "normal"),
            user_hint=req.runtime.get("user_hint", ""),
            repo_root=req.runtime.get("repo_root"),
        )

        if req.options.get("bypass", False):
            cls_result = bypass_classification("caller_requested_bypass")
        else:
            classifier = _get_router_classifier()
            cls_result = classifier.classify(req.prompt, signals)

        decision = choose_route(
            classification=cls_result.fields,
            signals=signals,
            overall_confidence=cls_result.overall_confidence,
            critical_confidence=cls_result.critical_confidence,
            classifier_used=cls_result.classifier_used,
            abstain=cls_result.abstain,
        )

        route_event = _telemetry_log.new_event(
            prompt_hash=_prompt_hash(req.prompt),
            prompt_chars=signals.prompt_chars,
            estimated_tokens=signals.estimated_tokens,
            classifier_used=cls_result.classifier_used,
            classification=cls_result.fields,
            field_confidence={k: v for k, v in cls_result.field_confidence.items()},
            overall_confidence=cls_result.overall_confidence,
            critical_confidence=cls_result.critical_confidence,
            abstain=cls_result.abstain,
            selected_tier=decision.tier,
            reasons=list(decision.reasons),
            requirements=decision.requirements.to_dict(),
            classifier_latency_ms=cls_result.latency_ms,
        )

        # --- Stage 2: Execute (local tiers only) ---
        execution_result = None
        if decision.tier in TIER_DEFINITIONS and TIER_DEFINITIONS[decision.tier].is_local:
            td = TIER_DEFINITIONS[decision.tier]
            profile_name = td.preferred_model_profiles[0] if td.preferred_model_profiles else None
            model_id = MODEL_PROFILES[profile_name].model_id if profile_name in MODEL_PROFILES else MODEL_ID
            sys_prompt = req.system_prompt or TIER_SYSTEM_PROMPTS.get(decision.tier, "")

            exec_result = run_text_generation(
                model_id=model_id,
                prompt=req.prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                system_prompt=sys_prompt if sys_prompt else None,
            )

            execution_result = {
                "model_id": model_id,
                "model_name": os.path.basename(model_id.rstrip("/")),
                "text": exec_result["text"],
                "elapsed_ms": exec_result["elapsed_ms"],
                "total_tokens": exec_result["total_tokens"],
                "tokens_per_second": exec_result["tokens_per_second"],
            }

            route_event.status = "success"
            route_event.actual_model = model_id
            route_event.latency_ms = cls_result.latency_ms + exec_result["elapsed_ms"]
        else:
            # Cloud tier – return decision, PI handles execution
            route_event.status = "routed_to_cloud"
            route_event.latency_ms = cls_result.latency_ms

        _telemetry_log.record(route_event)

        return {
            "route_id": route_event.event_id,
            "classification": cls_result.to_dict(),
            "signals": signals.to_dict(),
            "decision": decision.to_dict(),
            "execution": execution_result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models")
def api_loaded_models():
    """List all models currently loaded in Apple Silicon memory."""
    loaded = list_loaded_models()
    return {
        "loaded": loaded,
        "available_profiles": {k: v.to_dict() for k, v in MODEL_PROFILES.items()},
    }


@app.get("/api/models/overrides")
def api_model_overrides():
    """List model profiles available for user override, grouped by billing."""
    groups: Dict[str, list] = {}
    for name, profile in MODEL_PROFILES.items():
        if profile.role == "classifier":
            continue  # don't expose classifiers as override targets
        billing = profile.billing
        if billing not in groups:
            groups[billing] = []
        groups[billing].append({
            "profile_name": name,
            "model_id": profile.model_id,
            "provider": profile.provider,
            "capabilities": list(profile.capabilities),
            "billing": billing,
            "notes": profile.notes,
        })
    return {
        "billing_order": ["local", "subscription_unlimited", "subscription_quota", "paid"],
        "models": groups,
    }

