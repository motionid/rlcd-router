"""User-facing fallback recommendations for exhausted subscriptions."""

from __future__ import annotations

from typing import Iterable

from router.config import MODEL_PROFILES


def _model_option(profile_name: str) -> dict[str, object]:
    profile = MODEL_PROFILES[profile_name]
    return {
        "profile_name": profile_name,
        "model_id": profile.model_id,
        "provider": profile.provider,
        "billing": profile.billing,
        "capabilities": list(profile.capabilities),
        "notes": profile.notes,
    }


def recommend_subscription_fallback(
    eligible_models: Iterable[str],
    exhausted_model: str = "",
) -> dict[str, object]:
    """Recommend the next executor after a subscription quota is exhausted.

    ``eligible_models`` is already ordered by the routing policy, so the first
    remaining executor is the best automatic recommendation.  Other remaining
    executors are offered as explicit alternatives.
    """
    ordered = [
        name for name in eligible_models
        if name in MODEL_PROFILES and MODEL_PROFILES[name].role == "executor"
    ]
    if exhausted_model in ordered:
        ordered = ordered[ordered.index(exhausted_model) + 1 :]

    recommendation = ordered[0] if ordered else None
    alternatives = [_model_option(name) for name in ordered[1:]]
    return {
        "reason": "subscription_exhausted",
        "exhausted_model": exhausted_model,
        "recommended_model": recommendation,
        "recommended": _model_option(recommendation) if recommendation else None,
        "alternatives": alternatives,
        "can_override": True,
        "message": (
            f"The subscription for {exhausted_model or 'the selected model'} is exhausted. "
            f"Switch to {recommendation}, or choose another model."
            if recommendation
            else "The subscription is exhausted and no eligible fallback remains. Choose another model."
        ),
    }
