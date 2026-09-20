"""
Routing decision data structures.

A ``RouteDecision`` is the immutable output of the policy engine.
It tells PI which tier to use and what constraints the chosen
tier must satisfy.  PI then maps the tier to an actual model
via its own provider configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class TierRequirement:
    """Minimum capabilities the selected model must provide."""

    minimum_capability: str          # "none" | "general" | "coding" | "research"
    requires_tools: tuple[str, ...]  # e.g. ("shell",)
    requires_verification: bool      # tests / lint / review gate
    allow_cloud: bool                # False → local-only execution
    allow_weak_models: bool          # False → skip fast/small local models
    requires_web: bool               # needs browser or search tool

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum_capability": self.minimum_capability,
            "requires_tools": list(self.requires_tools),
            "requires_verification": self.requires_verification,
            "allow_cloud": self.allow_cloud,
            "allow_weak_models": self.allow_weak_models,
            "requires_web": self.requires_web,
        }


@dataclass(frozen=True)
class RouteDecision:
    """The complete routing decision returned to PI."""

    tier: str                     # e.g. "local_coding"
    eligible_models: tuple[str, ...] # ordered list of model profile names
    requirements: TierRequirement
    classifier_used: str          # "rlcd_fast" | "rlcd_smart" | "bypass"
    confidence: float             # overall classifier confidence
    critical_confidence: float    # lowest confidence among critical fields
    reasons: tuple[str, ...]      # human-readable reason codes
    escalation_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "eligible_models": list(self.eligible_models),
            "requirements": self.requirements.to_dict(),
            "classifier_used": self.classifier_used,
            "confidence": round(self.confidence, 4),
            "critical_confidence": round(self.critical_confidence, 4),
            "reasons": list(self.reasons),
            "escalation_config": self.escalation_config,
        }


# ---------------------------------------------------------------------------
# Default escalation configs per tier (used by choose_route)
# ---------------------------------------------------------------------------

DEFAULT_ESCALATION: dict[str, dict[str, Any]] = {
    "local_fast": {
        "next_tier": "local_general",
        "max_attempts": 3,
        "allow_cloud_fallback": False,
    },
    "local_general": {
        "next_tier": "local_coding",
        "max_attempts": 3,
        "allow_cloud_fallback": True,
    },
    "local_coding": {
        "next_tier": "local_reasoning",
        "max_attempts": 3,
        "allow_cloud_fallback": True,
    },
    "local_reasoning": {
        "next_tier": "cloud_coding",
        "max_attempts": 2,
        "allow_cloud_fallback": True,
    },
    "cloud_standard": {
        "next_tier": "cloud_reasoning",
        "max_attempts": 2,
        "allow_cloud_fallback": True,
    },
    "cloud_coding": {
        "next_tier": "cloud_reasoning",
        "max_attempts": 2,
        "allow_cloud_fallback": True,
    },
    "cloud_reasoning": {
        "next_tier": "cloud_research",
        "max_attempts": 2,
        "allow_cloud_fallback": True,
    },
    "cloud_research": {
        "next_tier": None,  # terminal
        "max_attempts": 2,
        "allow_cloud_fallback": True,
    },
}
