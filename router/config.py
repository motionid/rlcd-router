"""
Model profiles, tier definitions, and confidence thresholds.

All values are overridable via environment variables so the same
code works with different OMLX model paths or cloud providers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Mapping


# Local models already downloaded and managed by oMLX (see ~/.omlx/settings.json
# -> model.model_dirs). Pointing model_id at these paths lets mlx_lm.load()
# reuse the exact weights oMLX serves instead of re-downloading a duplicate
# copy into ~/.cache/huggingface. Override with OMLX_MODELS_DIR if the layout
# differs on another machine (e.g. a home machine with a different model set).
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


def _priority(name: str, default: int) -> int:
    """Profile priority (lower wins within an access class).

    Overridable per profile via ``ROUTER_PRIORITY_<PROFILE_NAME_UPPER>``.
    """
    return int(os.getenv(f"ROUTER_PRIORITY_{name.upper()}", str(default)))


MODEL_PROFILES: dict[str, ModelProfile] = {
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
            f"{OMLX_MODELS_DIR}/mlx-community/Qwen3.8-27B-4bit",
        ),
        role="executor",
        provider="omlx",
        capabilities=("general", "reasoning"),
        latency_class="slow",
        capability=Capability.STRONG,
        access_class=AccessClass.LOCAL,
        priority=_priority("local_general", 12),
        context_limit=65_536,
        notes="Qwen3.8 27B 4bit (General Reasoning) - matches installed oMLX model",
    ),
    "local_coding": ModelProfile(
        model_id=os.getenv(
            "LOCAL_CODING_MODEL",
            f"{OMLX_MODELS_DIR}/mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
        ),
        role="executor",
        provider="omlx",
        capabilities=("coding", "tools", "shell"),
        latency_class="medium",
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
        provider=os.getenv("SPLASH_PROVIDER", "openai"),
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
    # ---- GitHub Copilot (single provider, single seat) ----
    # All executor models below are served through the GitHub Copilot API
    # (provider "github-copilot", see ~/.pi/agent/models-store.json for the
    # live catalogue). Verified by decoding the live Copilot token's claims
    # (~/.pi/agent/auth.json): sku="copilot_enterprise_seat_quota" - there is
    # NO unlimited tier on this seat. Every model draws against the same
    # metered enterprise quota, just at very different per-model multipliers
    # (reflected here by the $/M-token notes). All profiles are therefore
    # billing="subscription_quota" (registry: access_class=subscription);
    # ordering within each tier still puts the
    # cheapest/lowest-multiplier models first to conserve quota.

    # -- cheap / fast (routine work, treated as effectively unlimited) --
    "gpt_6_luna": ModelProfile(
        model_id="gpt-6-luna",
        role="executor",
        provider="github-copilot",
        capabilities=("general", "coding", "tools"),
        latency_class="fast",
        capability=Capability.NORMAL,
        access_class=AccessClass.SUBSCRIPTION,
        priority=_priority("gpt_6_luna", 10),
        supports_tools=True,
        notes="GPT-6 Luna via GitHub Copilot ($0.10 / $0.50) - cheapest general/coding model",
    ),
    "gpt_5_6_luna": ModelProfile(
        model_id="gpt-5.6-luna",
        role="executor",
        provider="github-copilot",
        capabilities=("general", "coding", "tools"),
        latency_class="fast",
        capability=Capability.NORMAL,
        access_class=AccessClass.SUBSCRIPTION,
        priority=_priority("gpt_5_6_luna", 12),
        supports_tools=True,
        notes="GPT-5.6 Luna via GitHub Copilot ($0.20 / $1.20)",
    ),
    "gemini_3_8_flash": ModelProfile(
        model_id="gemini-3.8-flash",
        role="executor",
        provider="github-copilot",
        capabilities=("general", "coding", "tools", "research"),
        latency_class="fast",
        capability=Capability.NORMAL,
        access_class=AccessClass.SUBSCRIPTION,
        priority=_priority("gemini_3_8_flash", 14),
        context_limit=1_000_000,
        supports_tools=True,
        notes="Gemini 3.8 Flash via GitHub Copilot ($0.75 / $3.75, 1M context)",
    ),
    "gpt_5_mini": ModelProfile(
        model_id="gpt-5-mini",
        role="executor",
        provider="github-copilot",
        capabilities=("general", "coding", "tools"),
        latency_class="fast",
        capability=Capability.NORMAL,
        access_class=AccessClass.SUBSCRIPTION,
        priority=_priority("gpt_5_mini", 16),
        supports_tools=True,
        notes="GPT-5 Mini via GitHub Copilot ($0.25 / $2.00)",
    ),
    "claude_haiku_4_5": ModelProfile(
        model_id="claude-haiku-4.5",
        role="executor",
        provider="github-copilot",
        capabilities=("general", "coding", "tools"),
        latency_class="fast",
        capability=Capability.NORMAL,
        access_class=AccessClass.SUBSCRIPTION,
        priority=_priority("claude_haiku_4_5", 18),
        supports_tools=True,
        notes="Claude Haiku 4.5 via GitHub Copilot ($1.00 / $5.00)",
    ),
    "mai_code_1_1_flash": ModelProfile(
        model_id="mai-code-1.1-flash",
        role="executor",
        provider="github-copilot",
        capabilities=("coding", "tools"),
        latency_class="fast",
        capability=Capability.NORMAL,
        access_class=AccessClass.SUBSCRIPTION,
        priority=_priority("mai_code_1_1_flash", 20),
        supports_tools=True,
        notes="MAI-Code-1.1-Flash via GitHub Copilot ($0.20 / $1.20)",
    ),

    # -- coding specialists (moderate cost, reserve for real code work) --
    "gpt_5_3_codex": ModelProfile(
        model_id="gpt-5.3-codex",
        role="executor",
        provider="github-copilot",
        capabilities=("coding", "tools", "reasoning"),
        latency_class="medium",
        capability=Capability.STRONG,
        access_class=AccessClass.SUBSCRIPTION,
        priority=_priority("gpt_5_3_codex", 10),
        supports_tools=True,
        notes="GPT-5.3 Codex via GitHub Copilot ($1.75 / $14.00) - purpose-built coding model",
    ),
    "kimi_k2_7_code": ModelProfile(
        model_id="kimi-k2.7-code",
        role="executor",
        provider="github-copilot",
        capabilities=("coding", "tools"),
        latency_class="medium",
        capability=Capability.STRONG,
        access_class=AccessClass.SUBSCRIPTION,
        priority=_priority("kimi_k2_7_code", 12),
        supports_tools=True,
        notes="Kimi K2.7 Code via GitHub Copilot ($0.95 / $4.00)",
    ),
    "mai_code_1_flash": ModelProfile(
        model_id="mai-code-1-flash-picker",
        role="executor",
        provider="github-copilot",
        capabilities=("coding", "tools"),
        latency_class="medium",
        capability=Capability.STRONG,
        access_class=AccessClass.SUBSCRIPTION,
        priority=_priority("mai_code_1_flash", 14),
        supports_tools=True,
        notes="MAI-Code-1-Flash via GitHub Copilot ($0.75 / $4.50)",
    ),

    # -- general reasoning / balanced (subscription quota) --
    "claude_sonnet_5": ModelProfile(
        model_id="claude-sonnet-5",
        role="executor",
        provider="github-copilot",
        capabilities=("general", "coding", "reasoning", "research", "tools"),
        latency_class="fast",
        capability=Capability.STRONG,
        access_class=AccessClass.SUBSCRIPTION,
        priority=_priority("claude_sonnet_5", 10),
        supports_tools=True,
        notes="Claude Sonnet 5 via GitHub Copilot ($2.00 / $10.00)",
    ),
    "grok_4_7": ModelProfile(
        model_id="grok-4.7",
        role="executor",
        provider="github-copilot",
        capabilities=("general", "coding", "reasoning", "tools"),
        latency_class="medium",
        capability=Capability.STRONG,
        access_class=AccessClass.SUBSCRIPTION,
        priority=_priority("grok_4_7", 12),
        supports_tools=True,
        notes="Grok 4.7 via GitHub Copilot ($2.00 / $6.00)",
    ),
    "gpt_5_6_sol": ModelProfile(
        model_id="gpt-5.6-sol",
        role="executor",
        provider="github-copilot",
        capabilities=("general", "coding", "reasoning", "research", "tools"),
        latency_class="fast",
        capability=Capability.STRONG,
        access_class=AccessClass.SUBSCRIPTION,
        priority=_priority("gpt_5_6_sol", 14),
        supports_tools=True,
        notes="GPT-5.6 Sol via GitHub Copilot ($4.00 / $20.00)",
    ),

    # -- deep reasoning / flagship (expensive, reserve for high failure cost) --
    "gpt_5_6_terra": ModelProfile(
        model_id="gpt-5.6-terra",
        role="executor",
        provider="github-copilot",
        capabilities=("general", "coding", "reasoning", "research", "tools"),
        latency_class="slow",
        capability=Capability.REASONING,
        access_class=AccessClass.SUBSCRIPTION,
        priority=_priority("gpt_5_6_terra", 10),
        supports_tools=True,
        supports_thinking=True,
        notes="GPT-5.6 Terra via GitHub Copilot ($2.00 / $12.00) - deep reasoning",
    ),
    "claude_opus_5_5": ModelProfile(
        model_id="claude-opus-5.5",
        role="executor",
        provider="github-copilot",
        capabilities=("general", "coding", "reasoning", "research", "tools"),
        latency_class="slow",
        capability=Capability.REASONING,
        access_class=AccessClass.SUBSCRIPTION,
        priority=_priority("claude_opus_5_5", 12),
        supports_tools=True,
        supports_thinking=True,
        notes="Claude Opus 5.5 via GitHub Copilot ($4.00 / $20.00)",
    ),
    "claude_opus_5": ModelProfile(
        model_id="claude-opus-5",
        role="executor",
        provider="github-copilot",
        capabilities=("general", "coding", "reasoning", "research", "tools"),
        latency_class="slow",
        capability=Capability.REASONING,
        access_class=AccessClass.SUBSCRIPTION,
        priority=_priority("claude_opus_5", 14),
        supports_tools=True,
        supports_thinking=True,
        notes="Claude Opus 5 via GitHub Copilot ($5.00 / $25.00)",
    ),
    "gpt_6_astra": ModelProfile(
        model_id="gpt-6-astra",
        role="executor",
        provider="github-copilot",
        capabilities=("general", "coding", "reasoning", "research", "tools"),
        latency_class="slow",
        capability=Capability.REASONING,
        access_class=AccessClass.SUBSCRIPTION,
        priority=_priority("gpt_6_astra", 16),
        supports_tools=True,
        supports_thinking=True,
        notes="GPT-6 Astra via GitHub Copilot ($10.00 / $50.00) - flagship",
    ),
    "claude_fable_5_1": ModelProfile(
        model_id="claude-fable-5.1",
        role="executor",
        provider="github-copilot",
        capabilities=("general", "reasoning", "research", "tools"),
        latency_class="slow",
        capability=Capability.REASONING,
        access_class=AccessClass.SUBSCRIPTION,
        priority=_priority("claude_fable_5_1", 18),
        context_limit=1_000_000,
        supports_tools=True,
        supports_thinking=True,
        notes="Claude Fable 5.1 via GitHub Copilot ($10.00 / $50.00, 1M context) - long-context research",
    ),
}


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
        preferred_model_profiles=("gpt_6_luna", "gpt_5_6_luna", "gemini_3_8_flash", "gpt_5_mini", "claude_haiku_4_5"),
        supports_tools=("shell", "read_only", "browser"),
        supports_web=True,
        notes="General tasks that benefit from a stronger model (cheap Copilot models)",
    ),
    "cloud_coding": TierDefinition(
        name="cloud_coding",
        is_local=False,
        minimum_capability="coding",
        preferred_model_profiles=("gpt_5_3_codex", "kimi_k2_7_code", "mai_code_1_1_flash", "claude_sonnet_5", "gpt_5_6_sol", "gpt_6_astra"),
        supports_tools=("shell", "read_only", "browser"),
        supports_web=True,
        notes="Complex refactors, unfamiliar frameworks, high-stakes code",
    ),
    "cloud_reasoning": TierDefinition(
        name="cloud_reasoning",
        is_local=False,
        minimum_capability="general",
        preferred_model_profiles=("claude_opus_5_5", "gpt_5_6_terra", "gpt_6_astra", "claude_opus_5", "claude_sonnet_5", "gpt_5_6_sol"),
        supports_tools=("shell", "read_only", "browser"),
        supports_web=True,
        notes="Adversarial reasoning, deep architecture, high failure cost",
    ),
    "cloud_research": TierDefinition(
        name="cloud_research",
        is_local=False,
        minimum_capability="research",
        preferred_model_profiles=("claude_fable_5_1", "gpt_6_astra", "gemini_3_8_flash", "gpt_5_6_sol"),
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
    """Knobs that control the fast→smart→bypass cascade."""

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
