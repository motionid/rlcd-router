"""
Model profiles, tier definitions, and confidence thresholds.

All values are overridable via environment variables so the same
code works with different OMLX model paths or cloud providers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from enum import Enum, IntEnum
from typing import Any, Mapping


# Local models already downloaded and managed by oMLX (see ~/.omlx/settings.json
# -> model.model_dirs). Executor model_id values match Pi's dynamic oMLX
# registry; deprecated router-owned MLX execution uses each profile's separate
# legacy_model_id path under this directory. Override OMLX_MODELS_DIR or the
# LEGACY_LOCAL_* variables when the on-disk layout differs.
OMLX_MODELS_DIR = os.getenv("OMLX_MODELS_DIR", os.path.expanduser("~/.omlx/models"))


# ---------------------------------------------------------------------------
# Capability and access – two independent scales
#
# Capability describes what a task requires / a model can do:
#     BASIC < NORMAL < STRONG < REASONING (ordinal, comparable)
# Access class describes how using a model is granted or paid for:
#     local | subscription | metered-cheap | metered-expensive
#
# The two are orthogonal: a strong local model, a basic metered model,
# and a reasoning-class subscription model are all valid combinations.
# ---------------------------------------------------------------------------

class Capability(IntEnum):
    """Ordered capability scale.  Independent of access/cost."""

    BASIC = 0
    NORMAL = 1
    STRONG = 2
    REASONING = 3


class AccessClass(str, Enum):
    """How using a model is granted / paid for.  Independent of capability."""

    LOCAL = "local"
    SUBSCRIPTION = "subscription"
    METERED_CHEAP = "metered-cheap"
    METERED_EXPENSIVE = "metered-expensive"


# ---------------------------------------------------------------------------
# Model profiles – describes what each model *is* and what it can do
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    role: str                          # "classifier" | "executor"
    provider: str                      # provider key used by the host agent
    capabilities: tuple[str, ...]      # legacy: e.g. ("coding", "tools")
    latency_class: str                 # "fast" | "medium" | "slow"
    notes: str = ""
    # Optional mlx_lm path used only by deprecated router-owned execution.
    # Production candidates always serialize model_id for Pi's registry.
    legacy_model_id: str | None = None
    # --- capability/access registry fields ---
    capability: Capability = Capability.NORMAL
    access_class: AccessClass = AccessClass.LOCAL
    priority: int = 10                 # lower wins within an access class
    enabled: bool = True
    context_limit: int = 128_000      # max context tokens the model accepts
    supports_tools: bool = False
    supports_thinking: bool = False

    @property
    def billing(self) -> str:
        """Legacy billing label, derived from access_class (never drifts).

        Kept while compatibility consumers (override catalog grouping,
        telemetry) still speak the old ``local | subscription_quota | paid``
        vocabulary.
        """
        if self.access_class is AccessClass.LOCAL:
            return "local"
        if self.access_class is AccessClass.SUBSCRIPTION:
            return "subscription_quota"
        return "paid"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "role": self.role,
            "provider": self.provider,
            "capabilities": list(self.capabilities),
            "latency_class": self.latency_class,
            "billing": self.billing,
            "notes": self.notes,
            "capability": self.capability.name.lower(),
            "access_class": self.access_class.value,
            "priority": self.priority,
            "enabled": self.enabled,
            "context_limit": self.context_limit,
            "supports_tools": self.supports_tools,
            "supports_thinking": self.supports_thinking,
        }


def executor_candidates(
    profiles: Mapping[str, ModelProfile] | None = None,
) -> dict[str, ModelProfile]:
    """Enabled executor profiles eligible for execution.

    Classifier profiles can never become executor candidates, even when
    enabled: they label tasks, they must not execute them.
    """
    source = MODEL_PROFILES if profiles is None else profiles
    return {
        name: profile
        for name, profile in source.items()
        if profile.role == "executor" and profile.enabled
    }


def _truthy(raw: str | None) -> bool:
    """Environment boolean: 1/true/yes/on (case-insensitive)."""
    return bool(raw) and raw.strip().lower() in ("1", "true", "yes", "on")


def _priority(name: str, default: int) -> int:
    """Profile priority (lower wins within an access class).

    Overridable per profile via ``ROUTER_PRIORITY_<PROFILE_NAME_UPPER>``.
    """
    return int(os.getenv(f"ROUTER_PRIORITY_{name.upper()}", str(default)))


def _enabled(name: str, default: bool) -> bool:
    """Profile enablement (overridable per profile).

    ``ROUTER_ENABLE_<NAME>=1`` turns a profile on and
    ``ROUTER_DISABLE_<NAME>=1`` turns it off; disable wins over enable;
    unset variables keep the profile's declared default.  The metered
    example profiles below declare ``enabled=False`` so they stay off
    until real credentials and provider/model IDs are configured.
    """
    if _truthy(os.getenv(f"ROUTER_DISABLE_{name.upper()}")):
        return False
    if _truthy(os.getenv(f"ROUTER_ENABLE_{name.upper()}")):
        return True
    return default


def _build_model_profiles() -> dict[str, ModelProfile]:
    """Construct the profile registry (every knob env-overridable).

    Called once at import time to build ``MODEL_PROFILES``; also callable
    directly (e.g. from tests with ``monkeypatch``-ed environment) so the
    env-driven construction is verifiable without reloading the module
    (reloading would recreate the ``AccessClass``/``Capability`` enums and
    break identity comparisons in already-imported modules).
    """
    profiles: dict[str, ModelProfile] = {
    # ---- classifiers (RLCD models) ----
    "rlcd_fast": ModelProfile(
        model_id=os.getenv(
            "RLCD_FAST_MODEL",
            "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        ),
        role="classifier",
        provider="local_mlx",
        capabilities=("classification", "routing"),
        latency_class="fast",
        capability=Capability.BASIC,
        access_class=AccessClass.LOCAL,
        priority=_priority("rlcd_fast", 10),
        context_limit=32_768,
        notes="~70-500 ms, good for obvious tasks",
    ),
    "rlcd_smart": ModelProfile(
        model_id=os.getenv(
            "RLCD_SMART_MODEL",
            f"{OMLX_MODELS_DIR}/mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
        ),
        role="classifier",
        provider="omlx",
        capabilities=("classification", "routing", "coding", "reasoning"),
        latency_class="medium",
        capability=Capability.NORMAL,
        access_class=AccessClass.LOCAL,
        priority=_priority("rlcd_smart", 20),
        context_limit=65_536,
        notes="~0.5-1.6 s MoE, good for ambiguous or code-heavy tasks. Shares weights with local_coding.",
    ),
    # ---- executors (models PI actually runs) ----
    "local_general": ModelProfile(
        model_id=os.getenv(
            "LOCAL_GENERAL_MODEL",
            "Qwen3.8-27B-MLX-4bit",
        ),
        role="executor",
        provider="omlx",
        capabilities=("general", "reasoning"),
        latency_class="slow",
        legacy_model_id=os.getenv(
            "LEGACY_LOCAL_GENERAL_MODEL",
            f"{OMLX_MODELS_DIR}/lmstudio-community/Qwen3.8-27B-MLX-4bit",
        ),
        capability=Capability.STRONG,
        access_class=AccessClass.LOCAL,
        priority=_priority("local_general", 12),
        context_limit=65_536,
        notes="Qwen3.8 27B 4bit (General Reasoning) - matches installed oMLX model",
    ),
    "local_coding": ModelProfile(
        model_id=os.getenv(
            "LOCAL_CODING_MODEL",
            "Qwen3-Coder-30B-A3B-Instruct-4bit",
        ),
        role="executor",
        provider="omlx",
        capabilities=("coding", "tools", "shell"),
        latency_class="medium",
        legacy_model_id=os.getenv(
            "LEGACY_LOCAL_CODING_MODEL",
            f"{OMLX_MODELS_DIR}/mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
        ),
        capability=Capability.STRONG,
        access_class=AccessClass.LOCAL,
        priority=_priority("local_coding", 10),
        context_limit=65_536,
        supports_tools=True,
        notes="Qwen3 Coder 30B MoE (Fast Local Coding)",
    ),
    "splash_local": ModelProfile(
        model_id="incoai/Qwen3.8-27B-Splash",
        role="executor",
        provider=os.getenv("SPLASH_PROVIDER", "splash-local"),
        capabilities=("general", "coding", "reasoning", "tools"),
        latency_class="medium",
        capability=Capability.STRONG,
        access_class=AccessClass.LOCAL,
        priority=_priority("splash_local", 14),
        context_limit=65_536,
        supports_tools=True,
        notes="Splash local server at http://127.0.0.1:8080 (OpenAI-compatible)",
    ),
    # NOTE: Ornith-1.5-35B-A3B-oQ4e-mtp (LookUpMark/Ornith-1.5-35B-A3B-oQ4e-mtp)
    # is intentionally NOT registered as a router-executable profile. Its
    # architecture (qwen3_5_moe: hybrid linear-attention MoE + MTP head) is
    # only correctly implemented by oMLX's own server. The router's bundled
    # backend (core/engine_mlx.py -> plain mlx_lm 0.31.3) loads it without
    # erroring but produces incoherent/garbled output - verified by direct
    # test. Only use Ornith through oMLX's HTTP API (e.g. Pi's "omlx"
    # provider in models.json), never through this router's /api/execute.
    # local_reasoning falls back to local_general below until mlx_lm ships
    # native qwen3_5_moe support.
    # ---- Pi subscription providers ----
    # Provider/model pairs must exactly match Pi's model registry because the
    # extension resolves candidates with modelRegistry.find(provider, model).
    # GPT subscription models are exposed by openai-codex and Gemini by
    # antigravity on this installation. Anthropic-backed profiles are
    # intentionally not registered; use the OpenRouter/Gemini routes
    # configured in the host instead. Profiles absent from that registry are
    # omitted rather than retained as stale disabled entries.
    # Ordering within each capability group keeps cheaper/faster models first.

    # -- cheap / fast (routine work, treated as effectively unlimited) --
    "gpt_6_luna": ModelProfile(
        model_id="gpt-6-luna",
        role="executor",
        provider="openai-codex",
        capabilities=("general", "coding", "tools"),
        latency_class="fast",
        capability=Capability.NORMAL,
        access_class=AccessClass.SUBSCRIPTION,
        priority=_priority("gpt_6_luna", 10),
        supports_tools=True,
        notes="GPT-6 Luna via Pi's OpenAI Codex subscription provider",
    ),
    "gpt_5_6_luna": ModelProfile(
        model_id="gpt-5.6-luna",
        role="executor",
        provider="openai-codex",
        capabilities=("general", "coding", "tools"),
        latency_class="fast",
        capability=Capability.NORMAL,
        access_class=AccessClass.SUBSCRIPTION,
        priority=_priority("gpt_5_6_luna", 12),
        supports_tools=True,
        notes="GPT-5.6 Luna via Pi's OpenAI Codex subscription provider",
    ),
    "gemini_3_8_flash": ModelProfile(
        model_id="gemini-3.8-flash",
        role="executor",
        provider="antigravity",
        capabilities=("general", "coding", "tools", "research"),
        latency_class="fast",
        capability=Capability.NORMAL,
        access_class=AccessClass.SUBSCRIPTION,
        priority=_priority("gemini_3_8_flash", 14),
        context_limit=1_000_000,
        supports_tools=True,
        notes="Gemini 3.8 Flash via Pi's Antigravity provider (1M context)",
    ),

    # -- coding specialists (moderate cost, reserve for real code work) --
    "gpt_5_3_codex": ModelProfile(
        model_id="gpt-5.6-sol",
        role="executor",
        provider="openai-codex",
        capabilities=("coding", "tools", "reasoning"),
        latency_class="medium",
        capability=Capability.STRONG,
        access_class=AccessClass.SUBSCRIPTION,
        priority=_priority("gpt_5_3_codex", 99),
        supports_tools=True,
        notes="Manual-override compatibility alias for GPT-5.6 Sol; the canonical profile wins automatic deduplication",
    ),

    # -- general reasoning / balanced (subscription quota) --
    "gpt_5_6_sol": ModelProfile(
        model_id="gpt-5.6-sol",
        role="executor",
        provider="openai-codex",
        capabilities=("general", "coding", "reasoning", "research", "tools"),
        latency_class="fast",
        capability=Capability.STRONG,
        access_class=AccessClass.SUBSCRIPTION,
        priority=_priority("gpt_5_6_sol", 20),
        supports_tools=True,
        notes="GPT-5.6 Sol via Pi's OpenAI Codex subscription provider",
    ),

    # -- deep reasoning / flagship (expensive, reserve for high failure cost) --
    "gpt_5_6_terra": ModelProfile(
        model_id="gpt-5.6-terra",
        role="executor",
        provider="openai-codex",
        capabilities=("general", "coding", "reasoning", "research", "tools"),
        latency_class="slow",
        capability=Capability.REASONING,
        access_class=AccessClass.SUBSCRIPTION,
        priority=_priority("gpt_5_6_terra", 30),
        supports_tools=True,
        supports_thinking=True,
        notes="GPT-5.6 Terra via Pi's OpenAI Codex subscription provider",
    ),
    "gpt_6_astra": ModelProfile(
        model_id="gpt-6-astra",
        role="executor",
        provider="openai-codex",
        capabilities=("general", "coding", "reasoning", "research", "tools"),
        latency_class="slow",
        capability=Capability.REASONING,
        access_class=AccessClass.SUBSCRIPTION,
        priority=_priority("gpt_6_astra", 32),
        supports_tools=True,
        supports_thinking=True,
        notes="GPT-6 Astra via Pi's OpenAI Codex subscription provider",
    ),

    # ---- metered examples (DISABLED by default) ----
    # These two profiles demonstrate the metered access classes.  They
    # are NOT enabled: real per-token credentials and provider/model IDs
    # must be configured first (host provider config + the env vars
    # below + ROUTER_ENABLE_<NAME>=1).  Uncertain classifications never
    # select paid access at all (see router/policy.py), and the safe
    # fallback never resolves to paid access.
    "openrouter_qwen": ModelProfile(
        model_id=os.getenv("ROUTER_METERED_CHEAP_MODEL", "qwen/qwen3-coder"),
        role="executor",
        provider=os.getenv("ROUTER_METERED_CHEAP_PROVIDER", "openrouter"),
        capabilities=("general", "coding", "tools"),
        latency_class="medium",
        capability=Capability.STRONG,
        access_class=AccessClass.METERED_CHEAP,
        priority=_priority("openrouter_qwen", 10),
        enabled=False,  # example only: enable explicitly via env
        supports_tools=True,
        notes="Example metered-cheap OpenRouter/Qwen profile - disabled by "
              "default; configure credentials and model IDs before enabling",
    ),
    "premium_fallback": ModelProfile(
        model_id=os.getenv(
            "ROUTER_METERED_EXPENSIVE_MODEL", "z-ai/glm-5.3"
        ),
        role="executor",
        provider=os.getenv(
            "ROUTER_METERED_EXPENSIVE_PROVIDER", "openrouter"
        ),
        capabilities=("general", "coding", "reasoning", "research", "tools"),
        latency_class="slow",
        capability=Capability.REASONING,
        access_class=AccessClass.METERED_EXPENSIVE,
        priority=_priority("premium_fallback", 10),
        enabled=False,  # example only: enable explicitly via env
        supports_tools=True,
        supports_thinking=True,
        notes="Example metered-expensive premium fallback profile - disabled "
              "by default; configure credentials and model IDs before enabling",
    ),
    }

    # Per-profile enablement override, applied uniformly after construction:
    # ROUTER_ENABLE_<NAME>=1 forces a profile on (e.g. a metered example with
    # configured credentials), ROUTER_DISABLE_<NAME>=1 forces it off (e.g. a
    # subscription profile whose seat is suspended).  Disable wins; unset
    # variables keep the declared default above.
    return {
        name: replace(profile, enabled=_enabled(name, profile.enabled))
        for name, profile in profiles.items()
    }


MODEL_PROFILES: dict[str, ModelProfile] = _build_model_profiles()


# ---------------------------------------------------------------------------
# Availability cooldowns — failure-category cooldown durations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CooldownConfig:
    """Per-failure-category cooldown durations (seconds).

    Every value is overridable via environment variables so cooldown
    behavior can be tuned without code changes.  Durations are
    intentionally bounded: no category disables a profile permanently,
    and expiry always restores eligibility.
    """

    rate_limit: float = 60.0        # transient rate limit — short cooldown
    quota: float = 900.0            # usage/quota limit — medium cooldown
    auth: float = 300.0            # temporary authentication issue
    capacity: float = 120.0         # temporary capacity/overload issue
    unknown: float = 300.0         # unknown provider failure — conservative
    retry_after_cap: float = 3600.0  # safety bound for honored Retry-After

    def to_dict(self) -> dict[str, float]:
        """Plain-dict serialization (no secrets) for the config/status APIs."""
        return {
            "rate_limit": self.rate_limit,
            "quota": self.quota,
            "auth": self.auth,
            "capacity": self.capacity,
            "unknown": self.unknown,
            "retry_after_cap": self.retry_after_cap,
        }


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


COOLDOWN_CONFIG = CooldownConfig(
    rate_limit=_env_float("ROUTER_COOLDOWN_RATE_LIMIT", 60.0),
    quota=_env_float("ROUTER_COOLDOWN_QUOTA", 900.0),
    auth=_env_float("ROUTER_COOLDOWN_AUTH", 300.0),
    capacity=_env_float("ROUTER_COOLDOWN_CAPACITY", 120.0),
    unknown=_env_float("ROUTER_COOLDOWN_UNKNOWN", 300.0),
    retry_after_cap=_env_float("ROUTER_COOLDOWN_RETRY_AFTER_CAP", 3600.0),
)


# ---------------------------------------------------------------------------
# Ollaya — optional local ambiguity classifier (disabled by default)
#
# Live verification (Task 1, Ollaya 0.6.1 on Apple Silicon) showed ~205 ms
# warm / ~2.17 s cold decide latency.  The adapter is therefore enabled
# only explicitly, invoked only for ambiguous deterministic
# classifications, and never selects providers, models, access classes,
# or spending policy — it only answers task-classification questions.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OllayaConfig:
    """Settings for the optional Ollaya ambiguity classifier.

    Every value is overridable via environment variables
    (``ROUTER_OLLAYA_*``) so classification behavior can be tuned without
    code changes.

    ``enabled`` defaults to False: live evidence (task-1 report) passed the
    structure and latency gates but semantic accuracy is not yet calibrated
    against labeled fixtures, so Ollaya must be opt-in.

    ``basic_threshold`` is the probability ``required_capability == basic``
    must reach before the router accepts a local/basic downgrade.  The
    measured obvious-basic example scored only 0.4874, so the 0.90 default
    keeps uncalibrated answers on the subscription-safe path.

    ``continuation_threshold`` is the ``is_task_continuation`` probability
    required to treat a turn as a continuation (measured ambiguous
    continuation: 0.8152; obvious standalone: 0.1699).  The third question,
    ``is_independent_trivial_task``, is diagnostic telemetry only and never
    gates routing on its own.

    ``capability_threshold`` is the answer ``confidence`` an
    above-NORMAL capability choice (strong/reasoning) must reach before
    the router accepts the upclassification.  Measured confidences were
    low (0.3165 / 0.2047), so 0.60 keeps uncalibrated answers on the
    conservative NORMAL / inherited-continuation path; only confident
    answers may raise the capability.
    """

    enabled: bool = False
    base_url: str = "http://127.0.0.1:11435"
    model: str = "laya"
    timeout_ms: int = 500
    basic_threshold: float = 0.90
    continuation_threshold: float = 0.75
    capability_threshold: float = 0.60


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in (
        "1", "true", "yes", "on",
    )


OLLAYA_CONFIG = OllayaConfig(
    enabled=_env_bool("ROUTER_OLLAYA_ENABLED", False),
    base_url=os.getenv("ROUTER_OLLAYA_URL", "http://127.0.0.1:11435"),
    model=os.getenv("ROUTER_OLLAYA_MODEL", "laya"),
    timeout_ms=int(os.getenv("ROUTER_OLLAYA_TIMEOUT_MS", "500")),
    basic_threshold=_env_float("ROUTER_OLLAYA_BASIC_THRESHOLD", 0.90),
    continuation_threshold=_env_float(
        "ROUTER_OLLAYA_CONTINUATION_THRESHOLD", 0.75
    ),
    capability_threshold=_env_float(
        "ROUTER_OLLAYA_CAPABILITY_THRESHOLD", 0.60
    ),
)


# ---------------------------------------------------------------------------
# Tier definitions – abstract routing tiers that PI maps to models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TierDefinition:
    """An abstract routing tier.  PI's provider layer maps tier names
    to concrete model/provider combinations."""

    name: str
    is_local: bool
    minimum_capability: str            # "none" | "general" | "coding" | "research"
    preferred_model_profiles: tuple[str, ...] # ordered list of profile keys
    supports_tools: tuple[str, ...]
    supports_web: bool
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "is_local": self.is_local,
            "minimum_capability": self.minimum_capability,
            "preferred_model_profiles": list(self.preferred_model_profiles),
            "supports_tools": list(self.supports_tools),
            "supports_web": self.supports_web,
            "notes": self.notes,
        }


TIER_DEFINITIONS: dict[str, TierDefinition] = {
    "local_fast": TierDefinition(
        name="local_fast",
        is_local=True,
        minimum_capability="none",
        preferred_model_profiles=("rlcd_fast",),
        supports_tools=("read_only",),
        supports_web=False,
        notes="Trivial extraction, classification, small transforms",
    ),
    "local_general": TierDefinition(
        name="local_general",
        is_local=True,
        minimum_capability="general",
        preferred_model_profiles=("local_general",),
        supports_tools=("read_only", "shell"),
        supports_web=False,
        notes="Explanations, summarisation, non-code reasoning",
    ),
    "local_coding": TierDefinition(
        name="local_coding",
        is_local=True,
        minimum_capability="coding",
        preferred_model_profiles=("local_coding",),
        supports_tools=("shell", "read_only"),
        supports_web=False,
        notes="Code edits, debugging, test generation",
    ),
    "local_reasoning": TierDefinition(
        name="local_reasoning",
        is_local=True,
        minimum_capability="general",
        preferred_model_profiles=("local_general",),
        supports_tools=("shell", "read_only"),
        supports_web=False,
        notes="Deep reasoning, architecture decisions, multi-step planning. "
              "(Ornith 1.5 35B would be the ideal fit but is not usable via "
              "this router's bundled mlx_lm backend - see MODEL_PROFILES note.)",
    ),
    "cloud_standard": TierDefinition(
        name="cloud_standard",
        is_local=False,
        minimum_capability="general",
        preferred_model_profiles=("gpt_6_luna", "gpt_5_6_luna", "gemini_3_8_flash"),
        supports_tools=("shell", "read_only", "browser"),
        supports_web=True,
        notes="General tasks that benefit from a stronger subscription model",
    ),
    "cloud_coding": TierDefinition(
        name="cloud_coding",
        is_local=False,
        minimum_capability="coding",
        preferred_model_profiles=("gpt_5_6_sol", "gpt_6_astra"),
        supports_tools=("shell", "read_only", "browser"),
        supports_web=True,
        notes="Complex refactors, unfamiliar frameworks, high-stakes code",
    ),
    "cloud_reasoning": TierDefinition(
        name="cloud_reasoning",
        is_local=False,
        minimum_capability="general",
        preferred_model_profiles=("gpt_5_6_terra", "gpt_6_astra", "gpt_5_6_sol"),
        supports_tools=("shell", "read_only", "browser"),
        supports_web=True,
        notes="Adversarial reasoning, deep architecture, high failure cost",
    ),
    "cloud_research": TierDefinition(
        name="cloud_research",
        is_local=False,
        minimum_capability="research",
        preferred_model_profiles=("gpt_6_astra", "gemini_3_8_flash", "gpt_5_6_sol"),
        supports_tools=("shell", "read_only", "browser"),
        supports_web=True,
        notes="Tasks requiring current web data, long context, or external sources",
    ),
}


# ---------------------------------------------------------------------------
# Confidence thresholds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConfidenceThresholds:
    """Knobs for the legacy RLCD cascade — **evaluation-only**.

    These thresholds tune the deprecated fast→smart→bypass RLCD
    classification cascade, which is served by the explicit evaluation
    endpoints (``/api/run-parallel``) and the deprecated execution
    endpoints only.  Production routing never reads them: the
    deterministic classifier is rule-based and the optional Ollaya
    adapter carries its own thresholds (see :class:`OllayaConfig`).
    They remain serialized by ``/api/route/config`` so offline RLCD
    evaluation runs are reproducible."""

    fast_accept_overall: float = 0.85
    fast_accept_critical: float = 0.75
    conservative_floor: float = 0.65
    smart_bypass_floor: float = 0.70
    route_margin: float = 0.15

    def to_dict(self) -> dict[str, float]:
        return {
            "fast_accept_overall": self.fast_accept_overall,
            "fast_accept_critical": self.fast_accept_critical,
            "conservative_floor": self.conservative_floor,
            "smart_bypass_floor": self.smart_bypass_floor,
            "route_margin": self.route_margin,
        }


CONFIDENCE_THRESHOLDS = ConfidenceThresholds(
    fast_accept_overall=float(os.getenv("ROUTER_FAST_ACCEPT", "0.85")),
    fast_accept_critical=float(os.getenv("ROUTER_CRITICAL_FLOOR", "0.60")),
    conservative_floor=float(os.getenv("ROUTER_CONSERVATIVE_FLOOR", "0.50")),
    smart_bypass_floor=float(os.getenv("ROUTER_SMART_BYPASS", "0.50")),
    route_margin=float(os.getenv("ROUTER_ROUTE_MARGIN", "0.15")),
)
