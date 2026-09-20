"""
Deterministic routing policy.

Pure function:  (classification, signals) → RouteDecision

The policy has three layers executed in order:

  1. **Hard constraints** – safety and privacy rules that override
     everything (e.g. ``privacy=local_only`` blocks all cloud tiers).
  2. **Capability matching** – the task's functional requirements
     narrow the tier set (coding needs a coding tier, web data needs
     a research tier, etc.).
  3. **Optimisation** – among the remaining candidates, prefer local
     over cloud, then prefer the minimum capability that satisfies
     the task.

The goal:  *the weakest tier that can reliably complete the task*.
"""

from __future__ import annotations

from typing import Any

from router.signals import RuntimeSignals
from router.decision import RouteDecision, TierRequirement, DEFAULT_ESCALATION
from router.config import CONFIDENCE_THRESHOLDS, TIER_DEFINITIONS


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def choose_route(
    classification: dict[str, str],
    signals: RuntimeSignals,
    *,
    overall_confidence: float = 0.0,
    critical_confidence: float = 0.0,
    classifier_used: str = "bypass",
    abstain: bool = False,
) -> RouteDecision:
    """Map a classification + signals to a concrete routing decision.

    Parameters
    ----------
    classification:
        Mapping of schema field → chosen enum value.
    signals:
        Deterministic runtime observations.
    overall_confidence:
        Router's self-reported confidence in [0, 1].
    critical_confidence:
        Lowest confidence among critical fields.
    classifier_used:
        Which RLCD stage produced the classification.
    abstain:
        If True, the classifier was unsure; policy defaults to a
        safe conservative tier.
    """

    cls = classification  # shorthand
    reasons: list[str] = []

    # ------------------------------------------------------------------
    # 0.  Abstention → conservative default
    # ------------------------------------------------------------------
    if abstain:
        reasons.append("classifier_abstained")
        return _build(
            "local_general",
            cls,
            signals,
            reasons,
            overall_confidence,
            critical_confidence,
            classifier_used,
        )

    # ------------------------------------------------------------------
    # 1.  Hard constraints (privacy, data sensitivity)
    # ------------------------------------------------------------------
    local_only = cls.get("privacy") in ("local_only", "sensitive")
    if local_only:
        reasons.append("privacy_requires_local")

    # ------------------------------------------------------------------
    # 2.  Capability requirements
    # ------------------------------------------------------------------
    coding = cls.get("coding", "none")
    task_type = cls.get("task_type", "unknown")
    fresh_info = cls.get("fresh_information", "none")
    repo_access = cls.get("repo_access", "none")
    tool_use = cls.get("tool_use", "none")
    complexity = cls.get("complexity", "low")
    reasoning = cls.get("reasoning_depth", "direct")
    failure_cost = cls.get("failure_cost", "low")
    output_contract = cls.get("output_contract", "freeform")
    latency_pref = cls.get("latency", signals.deadline)

    # Does the task need coding capability?
    needs_coding = (
        coding in ("substantial", "dominant")
        or task_type in ("code_edit", "code_review", "debugging")
        or repo_access == "modify"
    )
    if needs_coding:
        reasons.append("needs_coding")

    # Does the task need web/research?
    needs_research = fresh_info == "required"
    if needs_research:
        reasons.append("needs_fresh_information")

    # Does the task need deep reasoning?
    needs_deep = (
        reasoning in ("deep", "adversarial")
        or complexity in ("high", "critical")
        or failure_cost == "very_high"
    )
    if needs_deep:
        reasons.append("needs_deep_reasoning")

    # Does the task need shell?
    needs_shell = (
        tool_use in ("shell", "multiple")
        or repo_access == "modify"
        or (needs_coding and output_contract in ("exact_patch", "verified_artifact"))
    )
    if needs_shell:
        reasons.append("needs_shell")

    # Does the task need browser?
    needs_browser = tool_use in ("browser", "multiple") and fresh_info in ("useful", "required")
    if needs_browser:
        reasons.append("needs_browser")

    # ------------------------------------------------------------------
    # 3.  Tier selection (lexicographic preference)
    # ------------------------------------------------------------------
    tier = _select_tier(
        local_only=local_only,
        needs_coding=needs_coding,
        needs_research=needs_research,
        needs_deep=needs_deep,
        complexity=complexity,
        latency_pref=latency_pref,
        failure_cost=failure_cost,
        prior_failures=signals.prior_failures,
        is_long_context=signals.is_long_context,
    )
    reasons.append(f"selected:{tier}")

    return _build(
        tier,
        cls,
        signals,
        reasons,
        overall_confidence,
        critical_confidence,
        classifier_used,
    )


# ---------------------------------------------------------------------------
# Tier selection logic
# ---------------------------------------------------------------------------

def _select_tier(
    *,
    local_only: bool,
    needs_coding: bool,
    needs_research: bool,
    needs_deep: bool,
    complexity: str,
    latency_pref: str,
    failure_cost: str,
    prior_failures: int,
    is_long_context: bool,
) -> str:
    """Pick the minimum viable tier."""

    # --- research always needs web → cloud ---
    if needs_research and not local_only:
        return "cloud_research"

    # --- if forced local, choose among local tiers ---
    if local_only:
        if needs_coding:
            if needs_deep or failure_cost == "very_high":
                return "local_reasoning"
            return "local_coding"
        if needs_deep:
            return "local_reasoning"
        if complexity in ("trivial", "low") and latency_pref == "interactive":
            return "local_fast"
        return "local_general"

    # --- not forced local: prefer local unless there is a reason ---
    if needs_coding:
        if complexity in ("trivial", "low") and prior_failures == 0:
            return "local_coding"
        if complexity == "medium" and failure_cost not in ("high", "very_high"):
            return "local_coding"
        if needs_deep or failure_cost in ("high", "very_high"):
            return "cloud_coding"
        if prior_failures >= 1:
            return "cloud_coding"
        return "local_coding"

    if needs_deep:
        if failure_cost == "very_high":
            return "cloud_reasoning"
        if complexity == "critical":
            return "cloud_reasoning"
        return "local_reasoning"

    # --- general task ---
    if complexity in ("trivial", "low") and latency_pref == "interactive":
        return "local_fast"

    if complexity == "medium":
        return "local_general"

    # high / critical general tasks
    return "cloud_standard"


# ---------------------------------------------------------------------------
# Build the decision with the right requirements
# ---------------------------------------------------------------------------

def _build(
    tier: str,
    cls: dict[str, str],
    signals: RuntimeSignals,
    reasons: list[str],
    overall_confidence: float,
    critical_confidence: float,
    classifier_used: str,
) -> RouteDecision:
    td = TIER_DEFINITIONS.get(tier)

    needs_shell = "needs_shell" in reasons
    needs_browser = "needs_browser" in reasons
    tools: list[str] = []
    if needs_shell:
        tools.append("shell")
    if needs_browser:
        tools.append("browser")
    if cls.get("output_contract") in ("exact_patch", "verified_artifact"):
        tools.append("read_only")

    req = TierRequirement(
        minimum_capability=td.minimum_capability if td else "general",
        requires_tools=tuple(tools),
        requires_verification=(
            cls.get("output_contract") in ("exact_patch", "verified_artifact")
            or cls.get("failure_cost") in ("high", "very_high")
        ),
        allow_cloud=td.is_local is False if td else True,
        allow_weak_models=(tier == "local_fast"),
        requires_web=td.supports_web if td else False,
    )

    esc = dict(DEFAULT_ESCALATION.get(tier, {}))

    # If privacy blocks cloud, strip cloud from escalation chain
    if cls.get("privacy") in ("local_only", "sensitive"):
        esc["allow_cloud_fallback"] = False

    return RouteDecision(
        tier=tier,
        eligible_models=td.preferred_model_profiles if td else (),
        requirements=req,
        classifier_used=classifier_used,
        confidence=overall_confidence,
        critical_confidence=critical_confidence,
        reasons=tuple(reasons),
        escalation_config=esc,
    )
