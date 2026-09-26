"""FastAPI service for deterministic task classification and model routing."""

import hashlib
import logging
import os
import time
import uuid
from collections import OrderedDict
from dataclasses import replace
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

# --- production router interfaces (no MLX, no network required) ---
from router.availability import (
    AvailabilityRecord,
    AvailabilityStore,
    FailureCategory,
)
from router.config import (
    OLLAYA_CONFIG,
    TIER_DEFINITIONS,
    MODEL_PROFILES,
    CONFIDENCE_THRESHOLDS,
    AccessClass,
    Capability,
)
from router.decision import Candidate, RouteDecision, TierRequirement
from router.deterministic import TaskClassification, TaskRelationship
from router.ollaya import classify_task
from router.policy import (
    EXCL_UNAVAILABLE,
    RouteRequirements,
    _MIN_CAPABILITY,
    _hard_filter,
    build_route,
    choose_route,
)

# --- legacy router interfaces (kept during migration) ---
from router.router_schema import ROUTER_SCHEMA
from router.signals import TurnContext, extract_signals
from router.classifier import RouterClassifier, bypass_classification
from router.policy import choose_route
from router.telemetry import TelemetryLog
from router.fallback import recommend_subscription_fallback

# NOTE: ``core.engine`` (MLX) is deliberately NOT imported at module
# scope.  The production /api/route path never touches it; the legacy
# /execute, /route-and-execute, and /run-parallel endpoints import it
# lazily inside their handlers so importing this module — or serving a
# production route — never loads MLX.

app = FastAPI(title="RLCD Multi-Model Router")

logger = logging.getLogger(__name__)

# Optional per-request cost estimate (USD) for the paid/metered alternative.
# When configured, routes that selected a subscription/local candidate while
# a paid candidate was eligible record the avoided spend estimate.  Default
# 0.0 disables estimation (no cost metadata configured).
_ESTIMATED_PAID_REQUEST_COST = float(
    os.getenv("ROUTER_ESTIMATED_PAID_REQUEST_COST", "0.0")
)


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

# Long-lived, process-shared availability state.  Created lazily on
# first use and NEVER recreated per request: cooldowns recorded by
# /api/route/outcome must be visible to subsequent /api/route and
# /api/router/status calls.
_availability_store: Optional[AvailabilityStore] = None

# Compact plain-dict snapshot of the most recent production route
# (served by /api/router/status; never a live record).
_last_route: Optional[Dict[str, Any]] = None


def _get_availability_store() -> AvailabilityStore:
    """The process-wide availability store (lazily constructed)."""
    global _availability_store
    if _availability_store is None:
        _availability_store = AvailabilityStore()
    return _availability_store


def _availability_dict(profile_id: str) -> Dict[str, Any]:
    """Plain-dict availability view of one profile (copy, never a record)."""
    record = _get_availability_store().snapshot().get(profile_id)
    if record is None:
        return AvailabilityRecord(profile_id=profile_id).to_dict()
    return record.to_dict()


def _availability_snapshot() -> Dict[str, Dict[str, Any]]:
    """Plain-dict availability snapshot of every tracked profile."""
    return {
        profile_id: record.to_dict()
        for profile_id, record in _get_availability_store().snapshot().items()
    }


# Bounded in-memory route outcome index.  Outcome reporting must never
# depend on TelemetryLog: telemetry recording is fail-open, so a route
# whose telemetry write failed must still be reportable.  The index is
# capped (oldest entry evicted first, deterministically) so a long-lived
# process cannot grow it without bound.
_ROUTE_INDEX_MAX = 1024
_route_index: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()


def _register_route_index(
    route_id: str, eligible_models: list
) -> None:
    """Register route context for outcome reporting (telemetry-independent)."""
    _route_index[route_id] = {
        "route_id": route_id,
        "timestamp": time.time(),
        "eligible_models": list(eligible_models),
        "status": "pending",
    }
    while len(_route_index) > _ROUTE_INDEX_MAX:
        _route_index.popitem(last=False)  # evict the oldest entry


class PredictRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    context: str
    schema_def: Dict[str, Any] = Field(..., alias="schema")
    temperature: Optional[float] = None

@app.post("/api/run-parallel")
@app.post("/api/run-rlcd")
def api_run_parallel(req: PredictRequest):
    # Legacy/evaluation endpoint: MLX is imported only when explicitly
    # called, never on the production route path.
    from core.schema import StructuredSchema
    from core.engine import run_parallel_generation

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

class TurnContextInput(BaseModel):
    """Bounded per-session turn context supplied by the extension.

    This is compact session state, not a transcript: every field is
    bounded, and ``previous_prompt`` is additionally capped by
    :class:`router.signals.TurnContext` itself.
    """

    model_config = ConfigDict(extra="ignore")

    previous_prompt: str = Field(default="", max_length=65_536)
    previous_capability: Optional[str] = Field(default=None, max_length=16)
    previous_model: str = Field(default="", max_length=256)
    previous_route_reason: list[str] = Field(default_factory=list, max_length=64)
    task_depth: int = Field(default=0, ge=0, le=10_000)
    files_involved: int = Field(default=0, ge=0, le=100_000)
    task_failures: int = Field(default=0, ge=0, le=1_000)

    def to_context(self) -> TurnContext:
        capability = None
        if self.previous_capability:
            try:
                capability = Capability[self.previous_capability.strip().upper()]
            except KeyError:
                # Unknown label: treat as missing history (conservative).
                capability = None
        return TurnContext(
            previous_prompt=self.previous_prompt,
            previous_capability=capability,
            previous_model=self.previous_model,
            previous_route_reason=tuple(self.previous_route_reason),
            task_depth=self.task_depth,
            files_involved=self.files_involved,
            task_failures=self.task_failures,
        )


class RouteRequirementsInput(BaseModel):
    """Hard execution requirements a candidate profile must satisfy."""

    model_config = ConfigDict(extra="ignore")

    supports_tools: bool = False
    supports_thinking: bool = False
    minimum_context: int = Field(default=0, ge=0, le=10_000_000)
    local_only: bool = False

    def to_requirements(self) -> RouteRequirements:
        return RouteRequirements(
            supports_tools=self.supports_tools,
            supports_thinking=self.supports_thinking,
            minimum_context=self.minimum_context,
            local_only=self.local_only,
        )


class RouteRequest(BaseModel):
    """Production routing request.

    Legacy fields (``runtime``, ``options``, ``model_override``) stay
    accepted during migration; ``turn_context`` and ``requirements``
    are the new bounded production inputs.
    """

    prompt: str
    runtime: Dict[str, Any] = {}
    options: Dict[str, Any] = {}
    model_override: Optional[str] = None  # user-requested model profile name
    turn_context: Optional[TurnContextInput] = None
    requirements: Optional[RouteRequirementsInput] = None


class OutcomeReport(BaseModel):
    """Outcome of one routed attempt.

    Legacy fields (``failure_reason``, ``actual_model``) stay accepted;
    ``failure_category``, ``retry_after``, and ``candidate_profile``
    drive the availability/task-outcome distinction.
    """

    event_id: str
    status: str          # "success" | "failed"
    failure_reason: str = ""
    verification: str = ""
    actual_model: str = ""
    latency_ms: float = 0.0
    cost: float = 0.0
    # --- production fields ---
    failure_category: Optional[str] = None   # FailureCategory value
    retry_after: Optional[float] = None      # provider Retry-After seconds
    candidate_profile: Optional[str] = None  # profile the attempt used


class FallbackRecommendationRequest(BaseModel):
    current_model: str
    eligible_models: list[str] = Field(default_factory=list)


def _get_router_classifier() -> RouterClassifier:
    """Lazily build the legacy RLCD classifier (MLX-backed).

    Only the deprecated legacy endpoints (/api/route-and-execute) use
    this; the production /api/route path never calls it.  MLX is
    imported here — inside the function — so a production route never
    loads it.

    Single-stage: Qwen3-Coder-30B-A3B-Instruct-4bit (~900 ms)
    Uses the full 11-field schema.  The 1.5B model was tested but
    can't reliably classify 11 fields (mean confidence ~0.73).
    """
    global _router_classifier
    if _router_classifier is not None:
        return _router_classifier

    from core.schema import StructuredSchema
    from core.engine import run_parallel_for_model

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
    """Classify a prompt and return an ordered, executable route.

    Production path (no MLX, no network required):

      signal extraction -> deterministic / optional-Ollaya
      classification -> policy candidate construction -> telemetry start

    Returns:
    - ``route_id`` (also the telemetry event id for /api/route/outcome)
    - ``classification`` (router-owned TaskClassification dict)
    - ``signals`` (deterministic runtime facts)
    - ``decision`` (ordered candidates, exclusions with stable reason
      codes, and legacy compatibility fields)
    - ``override`` (present when ``model_override`` was requested)

    Internal failures never 500 when a safe subscription fallback can be
    resolved: the response carries a ``classifier="fallback"`` decision
    containing the configured safe subscription profile.
    """
    try:
        return _route_request(req)
    except HTTPException:
        raise  # re-raise 4xx errors as-is
    except Exception as exc:
        return _route_fallback(req, exc)


def _turn_context_from_request(req: RouteRequest) -> TurnContext:
    """Build the bounded TurnContext from new or legacy request fields."""
    if req.turn_context is not None:
        return req.turn_context.to_context()
    runtime = req.runtime or {}
    # Legacy migration: derive bounded context from legacy runtime fields.
    # ``prior_failures`` becomes ``task_failures`` (the deterministic
    # classifier's escalation counter); other legacy fields feed the
    # legacy signals only.
    try:
        task_failures = min(max(int(runtime.get("prior_failures", 0)), 0), 1000)
    except (TypeError, ValueError):
        task_failures = 0
    return TurnContext(task_failures=task_failures)


def _requirements_from_request(req: RouteRequest) -> RouteRequirements:
    if req.requirements is not None:
        return req.requirements.to_requirements()
    return RouteRequirements()


def _set_last_route(
    route_id: str,
    classification: TaskClassification,
    decision: RouteDecision,
) -> None:
    """Store a compact plain-dict snapshot of the most recent route."""
    global _last_route
    _last_route = {
        "route_id": route_id,
        "classifier": classification.classifier,
        "capability": classification.capability.name.lower(),
        "confidence": classification.confidence,
        "candidates": [c.profile_id for c in decision.candidates],
        "reasons": list(decision.reasons),
        "timestamp": time.time(),
    }


def _ollaya_attempt(classification: TaskClassification) -> tuple[bool, str]:
    """Ollaya invocation outcome for telemetry (no prompt data, no errors).

    ``classify_task`` consults Ollaya only for ambiguous deterministic
    results, and converts every adapter failure into the conservative
    deterministic fallback.  A returned ambiguous result while the
    adapter is enabled therefore means the Ollaya attempt was made and
    failed — recorded as ``error_fallback`` so later failure analysis
    can see the attempt without seeing any prompt.
    """
    if classification.classifier == "ollaya":
        return True, "classified"
    if classification.classifier != "deterministic":
        # Fallback / bypass / override classifications never consulted
        # Ollaya, even when they carry ambiguous=True.
        return False, "not_consulted"
    invoked = OLLAYA_CONFIG.enabled and classification.ambiguous
    return invoked, ("error_fallback" if invoked else "not_consulted")


def _record_route_event(
    req: RouteRequest,
    signals,
    classification: TaskClassification,
    decision: RouteDecision,
) -> str:
    """Open the single telemetry record for a route and return its route_id.

    Task 8: the event stays PENDING in memory (no JSONL write) until the
    route settles through /api/route/outcome — first success or candidate
    exhaustion — which finalizes exactly one record with an attempts
    array.  Telemetry must never fail a route: exceptions are swallowed
    with debug logging.
    """
    route_id = uuid.uuid4().hex[:16]
    # Outcome context is registered BEFORE the best-effort telemetry
    # write so a telemetry failure can never prevent outcome reporting.
    _register_route_index(route_id, list(decision.eligible_models))
    try:
        top = decision.candidates[0] if decision.candidates else None
        paid_accesses = (
            AccessClass.METERED_CHEAP, AccessClass.METERED_EXPENSIVE
        )
        paid_candidate_available = any(
            c.access_class in paid_accesses for c in decision.candidates
        )
        ollaya_invoked, ollaya_outcome = _ollaya_attempt(classification)
        _telemetry_log.start_event(
            event_id=route_id,
            prompt_hash=_prompt_hash(req.prompt),
            prompt_chars=signals.prompt_chars,
            estimated_tokens=signals.estimated_tokens,
            classifier_used=classification.classifier,
            classification=classification.to_dict(),
            field_confidence={},
            overall_confidence=classification.confidence,
            critical_confidence=classification.confidence,
            abstain=classification.ambiguous,
            selected_tier=decision.tier,
            selected_profile=top.profile_id if top else "",
            access_class=(
                top.access_class.value if top is not None else ""
            ),
            eligible_models=list(decision.eligible_models),
            reasons=list(decision.reasons),
            requirements=decision.requirements.to_dict(),
            ollaya_invoked=ollaya_invoked,
            ollaya_outcome=ollaya_outcome,
            ollaya_confidence=(
                classification.confidence
                if classification.classifier == "ollaya" else 0.0
            ),
            ollaya_latency_ms=(
                classification.latency_ms
                if classification.classifier == "ollaya" else 0.0
            ) or 0.0,
            paid_selection_reasons=(
                list(decision.reasons)
                if top is not None and top.access_class in paid_accesses
                else []
            ),
            estimated_spend_avoided=(
                _ESTIMATED_PAID_REQUEST_COST
                if (
                    _ESTIMATED_PAID_REQUEST_COST > 0.0
                    and top is not None
                    and top.access_class not in paid_accesses
                    and paid_candidate_available
                )
                else 0.0
            ),
            classifier_latency_ms=classification.latency_ms or 0.0,
        )
    except Exception:
        logger.debug("route telemetry start failed", exc_info=True)
    _set_last_route(route_id, classification, decision)
    return route_id


def _route_response(
    route_id: str,
    signals,
    classification: TaskClassification,
    decision: RouteDecision,
    override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    response = {
        "route_id": route_id,
        "classification": classification.to_dict(),
        "signals": signals.to_dict(),
        "decision": decision.to_dict(),
    }
    if override is not None:
        response["override"] = override
    return response


def _route_request(req: RouteRequest) -> Dict[str, Any]:
    store = _get_availability_store()
    context = _turn_context_from_request(req)
    requirements = _requirements_from_request(req)
    signals = extract_signals(
        req.prompt,
        prior_attempts=req.runtime.get("prior_attempts", 0),
        prior_failures=req.runtime.get("prior_failures", 0),
        tool_calls_so_far=req.runtime.get("tool_calls_so_far", 0),
        deadline=req.runtime.get("deadline", "normal"),
        user_hint=req.runtime.get("user_hint", ""),
        repo_root=req.runtime.get("repo_root"),  # legacy, ignored (no-op)
    )

    # --- explicit override ---------------------------------------------
    if req.model_override:
        name = req.model_override
        executor_profiles = sorted(
            profile_name for profile_name, candidate in MODEL_PROFILES.items()
            if candidate.role == "executor"
        )
        if name not in MODEL_PROFILES:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model profile: {name}. "
                       f"Available: {', '.join(executor_profiles)}",
            )
        profile = MODEL_PROFILES[name]
        if profile.role != "executor":
            raise HTTPException(
                status_code=400,
                detail=f"Override must name an executor model profile. "
                       f"Available: {', '.join(executor_profiles)}",
            )

        available = store.is_available(name)
        incompatibility = None
        if available:
            # The override must satisfy the caller's actual hard
            # requirements (privacy, tools, thinking, context) — an
            # available but incompatible override is NOT returned.
            reason = _hard_filter(
                name, profile, Capability.BASIC, requirements,
                False, store, (profile.access_class,),
            )
            if reason == EXCL_UNAVAILABLE:
                # Cooldown raced in: treat exactly like an unavailable
                # override (fall back for this turn, keep the preference).
                available = False
            elif reason is not None:
                incompatibility = reason

        if available and incompatibility is None:
            classification = TaskClassification(
                capability=profile.capability,
                confidence=1.0,
                relationship=TaskRelationship.NEW_SUBSTANTIAL,
                reasons=("user_requested_model", f"override:{name}"),
                ambiguous=False,
                classifier="user_model_override",
            )
            decision = _override_decision(name, profile, requirements)
            route_id = _record_route_event(req, signals, classification, decision)
            return _route_response(
                route_id, signals, classification, decision,
                override={"requested": name, "available": True},
            )

        # Unavailable or incompatible override: route this turn from the
        # normal pool.  The user's stored preference lives client-side
        # (PI); the server never clears it, and the profile's cooldown /
        # configuration state is left untouched.
        classification = classify_task(req.prompt, context)
        try:
            decision = build_route(classification, MODEL_PROFILES, store, requirements)
        except Exception as exc:
            return _route_fallback(req, exc, classification)
        if not available:
            decision = replace(
                decision,
                reasons=tuple(decision.reasons)
                + (f"override_unavailable:{name}",),
            )
            override = {"requested": name, "available": False}
        else:
            decision = replace(
                decision,
                reasons=tuple(decision.reasons)
                + (f"override_incompatible:{incompatibility}",),
            )
            override = {
                "requested": name,
                "available": True,
                "compatible": False,
                "reason": incompatibility,
            }
        route_id = _record_route_event(req, signals, classification, decision)
        return _route_response(
            route_id, signals, classification, decision, override=override,
        )

    # --- automatic production path ---------------------------------------
    if req.options.get("bypass", False):
        # Legacy option kept: skip classification, stay conservative.
        classification = TaskClassification(
            capability=Capability.NORMAL,
            confidence=0.0,
            relationship=TaskRelationship.AMBIGUOUS,
            reasons=("caller_requested_bypass",),
            ambiguous=True,
            classifier="bypass",
        )
    else:
        classification = classify_task(req.prompt, context)

    try:
        decision = build_route(classification, MODEL_PROFILES, store, requirements)
    except Exception as exc:
        # The policy failed after classification: the fallback keeps the
        # task's required capability as its floor.
        return _route_fallback(req, exc, classification)
    route_id = _record_route_event(req, signals, classification, decision)
    return _route_response(route_id, signals, classification, decision)


def _override_decision(
    name: str, profile: Any, requirements: RouteRequirements,
) -> RouteDecision:
    """Single-candidate decision for a compatible explicit override.

    Built directly (legacy semantics): an explicit override bypasses
    access-class ordering, and its tier label comes from the tier
    definitions, exactly as before the candidate migration.
    """
    return RouteDecision(
        tier=_tier_for_profile(name, profile),
        eligible_models=(name,),
        requirements=TierRequirement(
            minimum_capability=_MIN_CAPABILITY.get(profile.capability, "general"),
            requires_tools=("shell",) if requirements.supports_tools else (),
            requires_verification=False,
            allow_cloud=profile.access_class is not AccessClass.LOCAL,
            allow_weak_models=True,
            requires_web=False,
        ),
        classifier_used="user_model_override",
        confidence=1.0,
        critical_confidence=1.0,
        reasons=("user_requested_model", f"override:{name}"),
        escalation_config={},
        capability=profile.capability,
        candidates=(
            Candidate(
                profile_id=name,
                provider=profile.provider,
                model=profile.model_id,
                capability=profile.capability,
                access_class=profile.access_class,
                priority=profile.priority,
            ),
        ),
        excluded=(),
    )


def _fallback_profile(
    store: AvailabilityStore,
    requirements: RouteRequirements,
    capability: Capability,
) -> Optional[tuple[str, Any]]:
    """Resolve the configured safe fallback profile, or None.

    Reuses the policy's own hard-filter predicate so the fallback can
    never be less capable than the policy's eligibility rules: role,
    enabled flag, minimum task capability, tools, thinking, context,
    local-only, and availability all apply.  Nothing is fabricated —
    only real, configured, currently-available executor profiles
    qualify.  The fallback deliberately does NOT call build_route, so a
    failure in the orchestration path cannot break the fallback too.
    """
    preferred = (
        AccessClass.LOCAL if requirements.local_only
        else AccessClass.SUBSCRIPTION
    )
    pool = []
    for name, profile in MODEL_PROFILES.items():
        reason = _hard_filter(
            name, profile, capability, requirements,
            True,  # conservative: paid access is never a safe fallback
            store,
            (preferred,),
        )
        if reason is None:
            pool.append((name, profile))
    if not pool:
        return None
    return min(pool, key=lambda item: (item[1].priority, item[0]))


def _route_fallback(
    req: RouteRequest, exc: Exception,
    classification: Optional[TaskClassification] = None,
) -> Dict[str, Any]:
    """Safe fallback after an internal route failure.

    Returns a ``classifier="fallback"`` decision containing the configured
    safe subscription (or local-only) profile when one can be resolved.
    The minimum task capability is preserved: when classification
    succeeded before the failure, its capability is the fallback's floor;
    otherwise the conservative NORMAL floor applies.  Never fabricates
    unavailable or under-capable profiles; when nothing eligible exists
    the decision carries an empty candidate list (PI retains its active
    model).
    """
    try:
        store = _get_availability_store()
        requirements = _requirements_from_request(req)
        capability = (
            classification.capability
            if classification is not None
            else Capability.NORMAL
        )
        signals = extract_signals(
            req.prompt,
            prior_attempts=req.runtime.get("prior_attempts", 0),
            prior_failures=req.runtime.get("prior_failures", 0),
            tool_calls_so_far=req.runtime.get("tool_calls_so_far", 0),
            deadline=req.runtime.get("deadline", "normal"),
            user_hint=req.runtime.get("user_hint", ""),
        )
        fallback_classification = TaskClassification(
            capability=capability,
            confidence=0.0,
            relationship=TaskRelationship.AMBIGUOUS,
            reasons=("internal_error", f"error:{type(exc).__name__}"),
            ambiguous=True,
            classifier="fallback",
        )
        safe = _fallback_profile(store, requirements, capability)
        if safe is not None:
            name, profile = safe
            candidates = (
                Candidate(
                    profile_id=name,
                    provider=profile.provider,
                    model=profile.model_id,
                    capability=profile.capability,
                    access_class=profile.access_class,
                    priority=profile.priority,
                ),
            )
            reasons = fallback_classification.reasons + (
                "safe_subscription_default",
            )
        else:
            candidates = ()
            reasons = fallback_classification.reasons + (
                "no_eligible_candidates",
            )
        tier = (
            "local_general" if requirements.local_only
            else "cloud_standard"
        )
        decision = RouteDecision(
            tier=tier,
            eligible_models=tuple(c.profile_id for c in candidates),
            requirements=TierRequirement(
                minimum_capability=_MIN_CAPABILITY.get(capability, "general"),
                requires_tools=(),
                requires_verification=False,
                allow_cloud=not requirements.local_only,
                allow_weak_models=False,
                requires_web=False,
            ),
            classifier_used="fallback",
            confidence=0.0,
            critical_confidence=0.0,
            reasons=reasons,
            escalation_config={},
            capability=capability,
            candidates=candidates,
            excluded=(),
        )
        route_id = _record_route_event(
            req, signals, fallback_classification, decision
        )
        return _route_response(
            route_id, signals, fallback_classification, decision
        )
    except Exception:
        raise HTTPException(status_code=500, detail=f"route failed: {exc}")


def _profile_name_for_model(model: str) -> str:
    if model in MODEL_PROFILES:
        return model
    for name, profile in MODEL_PROFILES.items():
        if profile.model_id == model:
            return name
    return model


def _coerce_failure_category(raw: str) -> FailureCategory:
    """Explicitly coerce a JSON string failure category to FailureCategory.

    Invalid values fail safely with a machine-readable 400 error —
    never a crash, never a silent conservative cooldown.
    """
    try:
        return FailureCategory(str(raw).strip().lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_failure_category",
                "value": raw,
                "valid_categories": [c.value for c in FailureCategory],
            },
        )


@app.post("/api/route/outcome")
def api_report_outcome(report: OutcomeReport):
    """Record an attempt outcome and update availability or task outcome.

    Route resolution goes through the bounded in-memory route index
    first (telemetry-independent), with the pending/finalized telemetry
    event as a legacy fallback.  Telemetry stays best-effort/fail-open:
    the outcome appends an attempt to the route's single pending record
    and finalizes it on settlement (first success or candidate
    exhaustion); repeated reports for the same route/attempt identity
    are idempotent, and a telemetry failure can never prevent cooldown
    or outcome handling.

    Availability categories (rate limit, quota, auth, capacity,
    provider-unknown) cool the provider down; task-side categories
    (task failure, local tool failure) never touch provider
    availability — the provider was available, the work failed.
    Legacy ``failure_reason`` values that are not FailureCategory values
    (e.g. "subscription_exhausted") keep their legacy behavior: no
    availability change, plus the model-switch recommendation.
    """
    entry = _route_index.get(report.event_id)
    event = _telemetry_log.get_event(report.event_id)
    if entry is None and event is None:
        raise HTTPException(
            status_code=404,
            detail=f"event_id {report.event_id} not found",
        )

    # --- category resolution ---------------------------------
    category: Optional[FailureCategory] = None
    if report.failure_category:
        category = _coerce_failure_category(report.failure_category)
    elif report.failure_reason:
        try:
            category = FailureCategory(report.failure_reason.strip().lower())
        except ValueError:
            category = None  # legacy free-text reason: no cooldown

    profile_name = (
        report.candidate_profile
        or (report.actual_model and _profile_name_for_model(report.actual_model))
        or None
    )

    # --- best-effort telemetry: append attempt, settle once ----
    try:
        eligible_models = (
            entry["eligible_models"] if entry is not None
            else (event.eligible_models if event is not None else [])
        )
        exhausted = (
            report.status == "failed"
            and bool(eligible_models)
            and profile_name == eligible_models[-1]
        )
        metered = bool(
            profile_name and profile_name in MODEL_PROFILES
            and MODEL_PROFILES[profile_name].access_class in (
                AccessClass.METERED_CHEAP, AccessClass.METERED_EXPENSIVE
            )
        )
        _telemetry_log.report_outcome(
            report.event_id,
            profile_id=profile_name or report.actual_model or "",
            status=report.status,
            failure_category=(category.value if category else ""),
            failure_reason=report.failure_reason,
            verification=report.verification,
            actual_model=report.actual_model,
            latency_ms=report.latency_ms,
            cost=report.cost,
            estimated_spend_incurred=(
                report.cost if metered else 0.0
            ),
            exhausted=exhausted,
        )
    except Exception:
        logger.debug("outcome telemetry update failed", exc_info=True)

    if entry is not None:
        entry["status"] = report.status
        entry["failure_reason"] = report.failure_reason

    # --- availability / task-outcome update --------------------
    availability = None
    if profile_name:
        store = _get_availability_store()
        if report.status == "success":
            store.record_success(profile_name)
        elif report.status == "failed" and category is not None:
            store.record_failure(
                profile_name, category, report.retry_after
            )
        # failed without a recognizable category: conservative
        # no-op — availability is never changed on a guess.
        availability = _availability_dict(profile_name)

    eligible_models = (
        entry["eligible_models"] if entry is not None
        else (event.eligible_models if event is not None else [])
    )
    response = {
        "status": "recorded",
        "event_id": report.event_id,
        "failure_category": category.value if category else None,
        "availability": availability,
    }
    if report.failure_reason == "subscription_exhausted":
        response["model_switch"] = recommend_subscription_fallback(
            eligible_models,
            _profile_name_for_model(report.actual_model),
        )
    return response


@app.post("/api/models/recommend")
def api_recommend_fallback(req: FallbackRecommendationRequest):
    """Return a switch prompt for a subscription that has run out."""
    eligible = req.eligible_models or [
        name for name, profile in MODEL_PROFILES.items() if profile.role == "executor"
    ]
    return recommend_subscription_fallback(
        eligible,
        _profile_name_for_model(req.current_model),
    )


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


@app.get("/api/router/status")
def api_router_status():
    """Compact router status for /router-status.

    Returns model access/capability/availability (plain-dict copies —
    never live AvailabilityRecord objects), the classifier configuration
    including Ollaya enabled/health state, and the last route.
    """
    snapshot = _get_availability_store().snapshot()
    models: Dict[str, Dict[str, Any]] = {}
    for name, profile in MODEL_PROFILES.items():
        record = snapshot.get(name)
        models[name] = {
            "provider": profile.provider,
            "capability": profile.capability.name.lower(),
            "access_class": profile.access_class.value,
            "enabled": profile.enabled,
            "priority": profile.priority,
            "availability": (
                record.to_dict() if record is not None
                else AvailabilityRecord(profile_id=name).to_dict()
            ),
        }
    return {
        "service": "rlcd-router",
        "classifier": {
            "deterministic": True,
            "ollaya": {
                "enabled": OLLAYA_CONFIG.enabled,
                "base_url": OLLAYA_CONFIG.base_url,
                "model": OLLAYA_CONFIG.model,
                "timeout_ms": OLLAYA_CONFIG.timeout_ms,
                # No live probe on the status path (disabled by default;
                # health of an enabled adapter is refined in Task 8/9).
                "health": "disabled" if not OLLAYA_CONFIG.enabled else "unknown",
            },
        },
        "models": models,
        "last_route": dict(_last_route) if _last_route else None,
    }


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
        # Legacy execution endpoint: MLX is imported only when the
        # endpoint is explicitly called, never on the production path.
        from core.engine import MODEL_ID, run_text_generation

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
        # Legacy combined endpoint: MLX is imported only when the
        # endpoint is explicitly called, never on the production path.
        from core.engine import MODEL_ID, run_text_generation

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
            eligible_models=list(decision.eligible_models),
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
    # Legacy endpoint: MLX is imported only when explicitly called.
    from core.engine import list_loaded_models

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
        "billing_order": ["local", "subscription_quota", "paid"],
        "models": groups,
    }

