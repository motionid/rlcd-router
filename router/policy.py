"""
Capability/access candidate routing policy.

Production entry point:  ``build_route(classification, profiles,
availability, requirements) -> RouteDecision``

The policy is a pure function over caller-supplied inputs.  It executes
three stages, in order:

  1. **Hard filtering** — role, enabled flag, capability floor, tool /
     thinking / context requirements, local-only privacy, and
     availability (cooldown) state exclude profiles *before* any
     ordering happens.  Every exclusion is reported with a stable
     reason code.
  2. **Access ordering** — the surviving profiles are grouped by access
     class in a capability-specific order (see ``ACCESS_ORDER`` and the
     spec's candidate policy):

       * basic:    local → subscription → metered-cheap → metered-expensive
       * normal:   subscription → normal-capable local → metered-cheap
                   → metered-expensive
       * strong:   subscription → metered-cheap → metered-expensive
       * reasoning: subscription → metered-cheap → metered-expensive

     A suitable subscription therefore always precedes metered access,
     and classification uncertainty never selects paid access at all.
  3. **Priority ordering** — inside an access group, lower numeric
     profile priority wins.

If nothing survives, the policy falls back to a *safe subscription
default*: the lowest-priority eligible subscription profile, relaxing
only the capability floor (never the hard requirements or availability).
The fallback is itself eligible and subscription-backed; if no such
profile exists the decision returns an empty candidate list and the
caller (PI) retains its active model.

``choose_route`` (bottom of this module) is a **deprecated** legacy
adapter kept only for old RLCD-dictionary callers and tests; it maps the
legacy classification onto a conservative ``TaskClassification`` and
delegates to ``build_route``.  The new production API path must call
``build_route`` directly.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Mapping

from router.availability import AvailabilityStore
from router.config import AccessClass, Capability, ModelProfile
from router.decision import (
    Candidate,
    CandidateExclusion,
    DEFAULT_ESCALATION,
    RouteDecision,
    TierRequirement,
)
from router.deterministic import TaskClassification, TaskRelationship


# ---------------------------------------------------------------------------
# Route requirements
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RouteRequirements:
    """Hard execution requirements a candidate profile must satisfy.

    These filter profiles *before* any ordering happens; an ineligible
    profile is never a candidate, regardless of priority or cost.
    """

    supports_tools: bool = False
    supports_thinking: bool = False
    minimum_context: int = 0
    local_only: bool = False


# ---------------------------------------------------------------------------
# Capability-specific access ordering
# ---------------------------------------------------------------------------

#: Access-class order per required capability.  Module-level so the API
#: layer (and tests) can configure ordering without code changes; lower
#: rank = preferred access.  Access classes absent from a capability's
#: tuple are excluded from candidacy for that capability.
ACCESS_ORDER: dict[Capability, tuple[AccessClass, ...]] = {
    Capability.BASIC: (
        AccessClass.LOCAL,
        AccessClass.SUBSCRIPTION,
        AccessClass.METERED_CHEAP,
        AccessClass.METERED_EXPENSIVE,
    ),
    # Normal work prefers the subscription quota, then *explicitly
    # normal-capable* local models, then metered access.
    Capability.NORMAL: (
        AccessClass.SUBSCRIPTION,
        AccessClass.LOCAL,
        AccessClass.METERED_CHEAP,
        AccessClass.METERED_EXPENSIVE,
    ),
    # Strong and reasoning work is never delegated to local models.
    Capability.STRONG: (
        AccessClass.SUBSCRIPTION,
        AccessClass.METERED_CHEAP,
        AccessClass.METERED_EXPENSIVE,
    ),
    Capability.REASONING: (
        AccessClass.SUBSCRIPTION,
        AccessClass.METERED_CHEAP,
        AccessClass.METERED_EXPENSIVE,
    ),
}

#: Paid access classes — excluded whenever the classification is
#: uncertain, so uncertainty can never trigger metered spending.
_PAID_ACCESS: frozenset[AccessClass] = frozenset({
    AccessClass.METERED_CHEAP,
    AccessClass.METERED_EXPENSIVE,
})

# Stable exclusion reason codes.
EXCL_CLASSIFIER = "classifier_profile"
EXCL_DISABLED = "profile_disabled"
EXCL_CAPABILITY = "insufficient_capability"
EXCL_TOOLS = "tools_unsupported"
EXCL_THINKING = "thinking_unsupported"
EXCL_CONTEXT = "context_limit_exceeded"
EXCL_LOCAL_ONLY = "local_only_blocks_remote"
EXCL_UNAVAILABLE = "unavailable_cooldown"
EXCL_PAID_UNCERTAIN = "paid_excluded_uncertain"
EXCL_ACCESS_ORDER = "access_class_excluded"


# ---------------------------------------------------------------------------
# Production entry point
# ---------------------------------------------------------------------------

def build_route(
    classification: TaskClassification,
    profiles: Mapping[str, ModelProfile],
    availability: AvailabilityStore,
    requirements: RouteRequirements = RouteRequirements(),
) -> RouteDecision:
    """Build an ordered, executable candidate route for one turn.

    Parameters
    ----------
    classification:
        Deterministic (or fallback) task classification.  Its
        capability is the required floor; ``ambiguous`` classifications
        never select paid access.
    profiles:
        Model profile registry (may include classifier and disabled
        profiles; they are filtered out, with exclusion reasons).
    availability:
        Availability store consulted *after* hard-requirement
        filtering, before ordering.
    requirements:
        Hard execution requirements (tools / thinking / context /
        local-only).
    """
    required = classification.capability
    uncertain = classification.ambiguous
    access_order = ACCESS_ORDER.get(
        required,
        (AccessClass.SUBSCRIPTION, AccessClass.METERED_CHEAP,
         AccessClass.METERED_EXPENSIVE),
    )
    access_rank = {access: i for i, access in enumerate(access_order)}

    reasons: list[str] = list(classification.reasons)
    reasons.append(f"capability:{required.name.lower()}")
    if requirements.local_only:
        reasons.append("privacy_requires_local")
    if uncertain:
        reasons.append("uncertainty_subscription_only")

    # -- 1. hard filtering (before any ordering) --------------------------
    eligible: list[tuple[str, ModelProfile]] = []
    excluded: list[CandidateExclusion] = []

    for name, profile in profiles.items():
        reason = _hard_filter(
            name, profile, required, requirements,
            uncertain, availability, access_order,
        )
        if reason is None:
            eligible.append((name, profile))
        else:
            excluded.append(CandidateExclusion(profile_id=name, reason=reason))

    # -- 2./3. access-group ordering, then numeric priority -----------------
    ordered = _order(eligible, access_rank)

    # -- safe fallback ------------------------------------------------------
    if not ordered:
        fallback = _safe_subscription_default(
            profiles, availability, requirements
        )
        if fallback is not None:
            ordered = [fallback]
            reasons.append("safe_subscription_default")
        else:
            reasons.append("no_eligible_candidates")

    candidates = tuple(
        Candidate(
            profile_id=name,
            provider=profile.provider,
            model=profile.model_id,
            capability=profile.capability,
            access_class=profile.access_class,
            priority=profile.priority,
        )
        for name, profile in ordered
    )

    # -- compatibility fields -----------------------------------------------
    tier = _legacy_tier(required, candidates, requirements)
    return RouteDecision(
        tier=tier,
        eligible_models=tuple(c.profile_id for c in candidates),
        requirements=_tier_requirement(required, requirements),
        classifier_used=classification.classifier,
        confidence=classification.confidence,
        critical_confidence=classification.confidence,
        reasons=tuple(reasons),
        escalation_config=_escalation_config(tier, requirements),
        capability=required,
        candidates=candidates,
        excluded=tuple(excluded),
    )


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

def _hard_filter(
    name: str,
    profile: ModelProfile,
    required: Capability,
    requirements: RouteRequirements,
    uncertain: bool,
    availability: AvailabilityStore,
    access_order: tuple[AccessClass, ...],
) -> str | None:
    """Return the exclusion reason for ``profile``, or None if eligible.

    Filter order (binding): role, enabled, capability, tools, thinking,
    context, local-only, availability — all before any sorting.
    """
    if profile.role != "executor":
        return EXCL_CLASSIFIER
    if not profile.enabled:
        return EXCL_DISABLED
    if profile.capability < required:
        return EXCL_CAPABILITY
    if requirements.supports_tools and not profile.supports_tools:
        return EXCL_TOOLS
    if requirements.supports_thinking and not profile.supports_thinking:
        return EXCL_THINKING
    if requirements.minimum_context and profile.context_limit < requirements.minimum_context:
        return EXCL_CONTEXT
    if requirements.local_only and profile.access_class is not AccessClass.LOCAL:
        return EXCL_LOCAL_ONLY
    if uncertain and profile.access_class in _PAID_ACCESS:
        return EXCL_PAID_UNCERTAIN
    if profile.access_class not in access_order:
        return EXCL_ACCESS_ORDER
    if not availability.is_available(name):
        return EXCL_UNAVAILABLE
    return None


def _order(
    eligible: list[tuple[str, ModelProfile]],
    access_rank: dict[AccessClass, int],
) -> list[tuple[str, ModelProfile]]:
    """Sort eligible profiles by access-group rank, then priority."""
    return sorted(
        eligible,
        key=lambda item: (
            access_rank[item[1].access_class],
            item[1].priority,
            item[0],
        ),
    )


def _safe_subscription_default(
    profiles: Mapping[str, ModelProfile],
    availability: AvailabilityStore,
    requirements: RouteRequirements,
) -> tuple[str, ModelProfile] | None:
    """Safe fallback when no candidate survived filtering.

    The fallback relaxes only the capability floor; hard requirements
    (tools/thinking/context/local-only) and availability still apply, so
    the returned profile is itself eligible.  It is subscription-backed
    (local-backed when privacy forces local-only); paid access is never
    a safe default.
    """
    preferred = (
        AccessClass.LOCAL if requirements.local_only else AccessClass.SUBSCRIPTION
    )
    pool: list[tuple[str, ModelProfile]] = []
    for name, profile in profiles.items():
        if profile.role != "executor" or not profile.enabled:
            continue
        if requirements.supports_tools and not profile.supports_tools:
            continue
        if requirements.supports_thinking and not profile.supports_thinking:
            continue
        if requirements.minimum_context and profile.context_limit < requirements.minimum_context:
            continue
        if requirements.local_only and profile.access_class is not AccessClass.LOCAL:
            continue
        if not availability.is_available(name):
            continue
        if profile.access_class is not preferred:
            continue
        pool.append((name, profile))
    if not pool:
        return None
    return min(pool, key=lambda item: (item[1].priority, item[0]))


# ---------------------------------------------------------------------------
# Compatibility (legacy tier) fields
# ---------------------------------------------------------------------------

_MIN_CAPABILITY = {
    Capability.BASIC: "none",
    Capability.NORMAL: "general",
    Capability.STRONG: "coding",
    Capability.REASONING: "research",
}


def _tier_requirement(
    required: Capability, requirements: RouteRequirements
) -> TierRequirement:
    return TierRequirement(
        minimum_capability=_MIN_CAPABILITY.get(required, "general"),
        requires_tools=("shell",) if requirements.supports_tools else (),
        requires_verification=False,
        allow_cloud=not requirements.local_only,
        allow_weak_models=required <= Capability.BASIC,
        requires_web=False,
    )


def _escalation_config(
    tier: str, requirements: RouteRequirements
) -> dict:
    esc = dict(DEFAULT_ESCALATION.get(tier, {}))
    if requirements.local_only:
        esc["allow_cloud_fallback"] = False
    return esc


def _legacy_tier(
    required: Capability,
    candidates: tuple[Candidate, ...],
    requirements: RouteRequirements,
) -> str:
    """Derive the legacy tier name from the chosen candidates.

    Kept only so compatibility consumers (existing API tests, the
    route-and-execute executor, telemetry) keep working while the
    tier-based migration completes.
    """
    if not candidates:
        return "local_general" if requirements.local_only else "cloud_standard"
    top = candidates[0]
    if top.access_class is AccessClass.LOCAL:
        if required <= Capability.BASIC and not requirements.supports_tools:
            return "local_fast"
        if requirements.supports_tools:
            return "local_coding"
        if required >= Capability.REASONING:
            return "local_reasoning"
        return "local_general"
    if required >= Capability.REASONING:
        return "cloud_reasoning"
    if required >= Capability.STRONG:
        return "cloud_coding"
    return "cloud_standard"


# ---------------------------------------------------------------------------
# Deprecated legacy adapter
# ---------------------------------------------------------------------------

def choose_route(
    classification: dict[str, str],
    signals,
    *,
    overall_confidence: float = 0.0,
    critical_confidence: float = 0.0,
    classifier_used: str = "bypass",
    abstain: bool = False,
) -> RouteDecision:
    """**Deprecated:** map a legacy RLCD classification to a route.

    Old callers (and tests) pass the RLCD 11-field classification
    dictionary.  This adapter maps it onto a conservative
    :class:`~router.deterministic.TaskClassification` and delegates to
    :func:`build_route` with the default registry and a fresh
    availability store.

    New production code must call :func:`build_route` directly; this
    adapter exists only for the legacy API path during the tier→
    candidate migration and will be removed once consumers are
    migrated.
    """
    warnings.warn(
        "choose_route() is deprecated; call build_route() with a "
        "TaskClassification instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    from router.config import MODEL_PROFILES

    cls = classification  # shorthand
    legacy_reasons: list[str] = []

    requirements = _legacy_requirements(cls, signals, legacy_reasons)

    if abstain:
        # Conservative: never basic, never paid — the safe subscription
        # default path in build_route handles the rest.
        legacy_reasons.append("classifier_abstained")
        task_classification = TaskClassification(
            capability=Capability.NORMAL,
            confidence=overall_confidence,
            relationship=TaskRelationship.AMBIGUOUS,
            reasons=tuple(legacy_reasons),
            ambiguous=True,
            classifier=classifier_used,
        )
    else:
        task_classification = TaskClassification(
            capability=_legacy_capability(cls, signals, legacy_reasons),
            confidence=overall_confidence,
            relationship=TaskRelationship.NEW_SUBSTANTIAL,
            reasons=tuple(legacy_reasons),
            ambiguous=False,
            classifier=classifier_used,
        )

    decision = build_route(
        task_classification,
        MODEL_PROFILES,
        AvailabilityStore(),
        requirements,
    )

    # Preserve legacy confidence and verification semantics that
    # TaskClassification does not carry.
    decision = replace(decision, critical_confidence=critical_confidence)
    if _legacy_requires_verification(cls):
        decision = replace(
            decision,
            requirements=replace(
                decision.requirements, requires_verification=True
            ),
        )

    # Legacy research tier: fresh_information == "required" always
    # produced the cloud_research tier with a web requirement (privacy
    # outranking it, above).  build_route() has no research tier, so the
    # deprecated adapter re-derives it here — the new policy's
    # capability/access ordering and candidates are untouched.
    if cls.get("fresh_information") == "required" and not requirements.local_only:
        decision = replace(
            decision,
            tier="cloud_research",
            requirements=replace(
                decision.requirements,
                minimum_capability="research",
                requires_web=True,
            ),
            escalation_config=_escalation_config("cloud_research", requirements),
        )
    return decision


def _legacy_requirements(
    cls: dict[str, str], signals, legacy_reasons: list[str]
) -> RouteRequirements:
    """Map legacy classification fields to hard route requirements."""
    local_only = cls.get("privacy") in ("local_only", "sensitive")
    if local_only:
        legacy_reasons.append("privacy_requires_local")

    tools: list[str] = []
    if cls.get("tool_use") in ("shell", "multiple") or cls.get("repo_access") == "modify":
        tools.append("shell")
    if cls.get("tool_use") in ("browser", "multiple"):
        tools.append("browser")

    supports_tools = bool(tools)

    return RouteRequirements(
        supports_tools=supports_tools,
        supports_thinking=False,
        minimum_context=0,
        local_only=local_only,
    )


def _legacy_capability(
    cls: dict[str, str], signals, legacy_reasons: list[str]
) -> Capability:
    """Map legacy classification fields to a conservative capability.

    Conservative means: strong/reasoning signals map up eagerly, and a
    prior failure on trivial coding routes to the subscription-backed
    normal level (legacy "cloud_standard") rather than staying local.
    """
    coding = cls.get("coding", "none")
    task_type = cls.get("task_type", "unknown")
    complexity = cls.get("complexity", "low")
    reasoning = cls.get("reasoning_depth", "direct")
    failure_cost = cls.get("failure_cost", "low")
    repo_access = cls.get("repo_access", "none")
    tool_use = cls.get("tool_use", "none")
    fresh_info = cls.get("fresh_information", "none")
    latency_pref = cls.get("latency", getattr(signals, "deadline", "normal"))

    needs_coding = (
        coding in ("substantial", "dominant")
        or task_type in ("code_edit", "code_review", "debugging")
        or repo_access == "modify"
    )
    if needs_coding:
        legacy_reasons.append("needs_coding")

    if fresh_info == "required":
        legacy_reasons.append("needs_fresh_information")

    needs_deep = (
        reasoning in ("deep", "adversarial")
        or complexity in ("high", "critical")
        or failure_cost == "very_high"
    )
    if needs_deep:
        legacy_reasons.append("needs_deep_reasoning")

    if tool_use in ("shell", "multiple") or repo_access == "modify":
        legacy_reasons.append("needs_shell")
    if tool_use in ("browser", "multiple") and fresh_info in ("useful", "required"):
        legacy_reasons.append("needs_browser")

    if needs_deep:
        return Capability.REASONING

    if needs_coding:
        if complexity in ("trivial", "low") and failure_cost in ("low", "medium"):
            if getattr(signals, "prior_failures", 0) == 0:
                return Capability.BASIC
            # Prior failure: conservative subscription-backed level.
            return Capability.NORMAL
        return Capability.STRONG

    if fresh_info in ("useful", "required"):
        return Capability.NORMAL

    if complexity in ("trivial", "low") and latency_pref == "interactive":
        return Capability.BASIC

    return Capability.NORMAL


def _legacy_requires_verification(cls: dict[str, str]) -> bool:
    return (
        cls.get("output_contract") in ("exact_patch", "verified_artifact")
        or cls.get("failure_cost") in ("high", "very_high")
    )
