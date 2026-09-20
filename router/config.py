"""
Model profiles, tier definitions, and confidence thresholds.

All values are overridable via environment variables so the same
code works with different OMLX model paths or cloud providers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Model profiles – describes what each model *is* and what it can do
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    role: str                          # "classifier" | "executor"
    provider: str                      # provider key used by the host agent
    capabilities: tuple[str, ...]      # e.g. ("coding", "tools")
    latency_class: str                 # "fast" | "medium" | "slow"
    billing: str = "local"             # local | subscription_unlimited | subscription_quota | paid
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "role": self.role,
            "provider": self.provider,
            "capabilities": list(self.capabilities),
            "latency_class": self.latency_class,
            "billing": self.billing,
            "notes": self.notes,
        }


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
        billing="local",
        notes="~70-500 ms, good for obvious tasks",
    ),
    "rlcd_smart": ModelProfile(
        model_id=os.getenv(
            "RLCD_SMART_MODEL",
            "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
        ),
        role="classifier",
        provider="omlx",
        capabilities=("classification", "routing", "coding", "reasoning"),
        latency_class="medium",
        billing="local",
        notes="~0.5-1.6 s MoE, good for ambiguous or code-heavy tasks",
    ),
    # ---- executors (models PI actually runs) ----
    "local_general": ModelProfile(
        model_id=os.getenv(
            "LOCAL_GENERAL_MODEL",
            "Qwen3.8-27B-MTP-bf16",
        ),
        role="executor",
        provider="omlx",
        capabilities=("general", "reasoning"),
        latency_class="slow",
        billing="local",
        notes="Qwen3.8 27B MTP bf16 (General Reasoning)",
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
        billing="local",
        notes="Qwen3 Coder 30B MoE (Fast Local Coding)",
    ),
    # ---- cloud subscriptions (UNLIMITED) ----
    "gemini_3_8_flash": ModelProfile(
        model_id="gemini-3.8-flash",
        role="executor",
        provider="antigravity",
        capabilities=("general", "coding", "tools"),
        latency_class="fast",
        billing="subscription_unlimited",
        notes="Google Gemini 3.8 Flash via Antigravity",
    ),
    "gpt_5_6_luna": ModelProfile(
        model_id="gpt-5.6-luna",
        role="executor",
        provider="openai",
        capabilities=("general", "coding", "tools"),
        latency_class="fast",
        billing="subscription_unlimited",
        notes="OpenAI GPT-5.6 Luna",
    ),
    # ---- cloud subscriptions (STRICT QUOTA) ----
    "gemini_3_1_pro": ModelProfile(
        model_id="gemini-3.1-pro",
        role="executor",
        provider="antigravity",
        capabilities=("general", "coding", "reasoning", "research", "tools"),
        latency_class="medium",
        billing="subscription_quota",
        notes="Google Gemini 3.1 Pro via Antigravity",
    ),
    "gpt_5_6_sol": ModelProfile(
        model_id="gpt-5.6-sol",
        role="executor",
        provider="openai",
        capabilities=("general", "coding", "reasoning", "research", "tools"),
        latency_class="fast",
        billing="subscription_quota",
        notes="OpenAI GPT-5.6 Sol",
    ),
    "claude_sonnet_4_6": ModelProfile(
        model_id="claude-sonnet-4-6",
        role="executor",
        provider="antigravity",
        capabilities=("general", "coding", "reasoning", "research", "tools"),
        latency_class="fast",
        billing="subscription_quota",
        notes="Anthropic Claude Sonnet 4.6 via Antigravity",
    ),
    # ---- cloud paid ----
    "gpt_5_6_luna_paid": ModelProfile(
        model_id="gpt-5.6-luna",
        role="executor",
        provider="openrouter",
        capabilities=("general", "coding", "tools"),
        latency_class="fast",
        billing="paid",
        notes="OpenAI Luna via OpenRouter ($0.20 / $1.20)",
    ),
    "qwen_3_8_max": ModelProfile(
        model_id="qwen3.8-max",
        role="executor",
        provider="qwen-cloud",
        capabilities=("general", "coding", "reasoning", "tools"),
        latency_class="medium",
        billing="paid",
        notes="Qwen 3.8 Max ($2.00 / $6.00)",
    ),
    "qwen_3_7_max": ModelProfile(
        model_id="qwen3.7-max",
        role="executor",
        provider="qwen-cloud",
        capabilities=("general", "coding", "reasoning", "tools"),
        latency_class="medium",
        billing="paid",
        notes="Qwen 3.7 Max ($1.48 / $4.43)",
    ),
    "claude_sonnet_5": ModelProfile(
        model_id="claude-sonnet-5",
        role="executor",
        provider="openrouter",
        capabilities=("general", "coding", "reasoning", "research", "tools"),
        latency_class="fast",
        billing="paid",
        notes="Anthropic Claude Sonnet 5 via OpenRouter ($2.00 / $10.00)",
    ),
    # ---- additional subscription models ----
    "claude_opus_4_6": ModelProfile(
        model_id="claude-opus-4-6",
        role="executor",
        provider="antigravity",
        capabilities=("general", "coding", "reasoning", "research", "tools"),
        latency_class="slow",
        billing="subscription_quota",
        notes="Anthropic Claude Opus 4.6 via Antigravity",
    ),
    "gpt_5_6_terra": ModelProfile(
        model_id="gpt-5.6-terra",
        role="executor",
        provider="openai",
        capabilities=("general", "coding", "reasoning", "research", "tools"),
        latency_class="slow",
        billing="subscription_quota",
        notes="OpenAI GPT-5.6 Terra (deep reasoning)",
    ),
    "gpt_oss_120b": ModelProfile(
        model_id="gpt-oss-120b",
        role="executor",
        provider="antigravity",
        capabilities=("general", "coding", "reasoning", "tools"),
        latency_class="medium",
        billing="subscription_unlimited",
        notes="OpenAI GPT-OSS 120B via Antigravity",
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
        notes="Deep reasoning, architecture decisions, multi-step planning",
    ),
    "cloud_standard": TierDefinition(
        name="cloud_standard",
        is_local=False,
        minimum_capability="general",
        preferred_model_profiles=("gpt_5_6_luna", "gemini_3_8_flash", "gpt_5_6_luna_paid", "qwen_3_8_max", "qwen_3_7_max"),
        supports_tools=("shell", "read_only", "browser"),
        supports_web=True,
        notes="General tasks that benefit from a stronger model",
    ),
    "cloud_coding": TierDefinition(
        name="cloud_coding",
        is_local=False,
        minimum_capability="coding",
        preferred_model_profiles=("gpt_5_6_sol", "gemini_3_1_pro", "claude_sonnet_4_6", "qwen_3_8_max", "qwen_3_7_max", "claude_sonnet_5"),
        supports_tools=("shell", "read_only", "browser"),
        supports_web=True,
        notes="Complex refactors, unfamiliar frameworks, high-stakes code",
    ),
    "cloud_reasoning": TierDefinition(
        name="cloud_reasoning",
        is_local=False,
        minimum_capability="general",
        preferred_model_profiles=("gpt_5_6_sol", "claude_sonnet_4_6", "gemini_3_1_pro", "qwen_3_8_max", "qwen_3_7_max", "claude_sonnet_5"),
        supports_tools=("shell", "read_only", "browser"),
        supports_web=True,
        notes="Adversarial reasoning, deep architecture, high failure cost",
    ),
    "cloud_research": TierDefinition(
        name="cloud_research",
        is_local=False,
        minimum_capability="research",
        preferred_model_profiles=("gemini_3_1_pro", "gpt_5_6_sol", "qwen_3_8_max", "qwen_3_7_max"),
        supports_tools=("shell", "read_only", "browser"),
        supports_web=True,
        notes="Tasks requiring current web data or external sources",
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
