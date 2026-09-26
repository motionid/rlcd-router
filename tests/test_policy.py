"""
Tests for the capability/access candidate policy.

``build_route`` is the production policy: it filters a synthetic (or the
real) registry by hard requirements and availability, then orders the
survivors by the capability-specific access order and numeric priority.

``choose_route`` is the deprecated legacy adapter that maps old RLCD
classification dictionaries onto a conservative ``TaskClassification``;
its tests pin only the compatibility contract, not tier internals.

All tests use synthetic registries and an injected clock — no models,
no network.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from router.availability import AvailabilityStore, FailureCategory
from router.config import AccessClass, Capability, ModelProfile
from router.decision import RouteDecision
from router.deterministic import TaskClassification, TaskRelationship
from router.policy import (
    ACCESS_ORDER,
    RouteRequirements,
    build_route,
    choose_route,
)
from router.signals import RuntimeSignals


# ---------------------------------------------------------------------------
# Synthetic registry helpers
# ---------------------------------------------------------------------------

def _profile(
    profile_id: str,
    *,
    capability: Capability = Capability.NORMAL,
    access: AccessClass = AccessClass.SUBSCRIPTION,
    priority: int = 10,
    role: str = "executor",
    enabled: bool = True,
    tools: bool = False,
    thinking: bool = False,
    context_limit: int = 128_000,
) -> ModelProfile:
    return ModelProfile(
        model_id=f"model-{profile_id}",
        role=role,
        provider="synthetic",
        capabilities=(),
        latency_class="fast",
        capability=capability,
        access_class=access,
        priority=priority,
        enabled=enabled,
        context_limit=context_limit,
        supports_tools=tools,
        supports_thinking=thinking,
    )


def _classification(
    capability: Capability,
    *,
    ambiguous: bool = False,
    confidence: float = 0.9,
    reasons: tuple[str, ...] = ("synthetic",),
) -> TaskClassification:
    return TaskClassification(
        capability=capability,
        confidence=confidence,
        relationship=TaskRelationship.NEW_SUBSTANTIAL,
        reasons=reasons,
        ambiguous=ambiguous,
        classifier="deterministic",
    )


def _store(clock=None) -> AvailabilityStore:
    return AvailabilityStore(clock=clock) if clock else AvailabilityStore()


class _FakeClock:
    """Mutable injected clock for cooldown/recovery tests."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _ids(decision: RouteDecision) -> list[str]:
    return [c.profile_id for c in decision.candidates]


def _excluded(decision: RouteDecision, profile_id: str) -> str | None:
    for entry in decision.excluded:
        if entry.profile_id == profile_id:
            return entry.reason
    return None


# ---------------------------------------------------------------------------
# Capability-specific access ordering
# ---------------------------------------------------------------------------

class TestCapabilityAccessOrder:
    def test_basic_prefers_local(self):
        profiles = {
            "local_a": _profile("local_a", capability=Capability.NORMAL,
                                access=AccessClass.LOCAL, priority=30),
            "sub_a": _profile("sub_a", priority=10),
            "metered_a": _profile("metered_a",
                                  access=AccessClass.METERED_CHEAP),
        }
        decision = build_route(
            _classification(Capability.BASIC), profiles, _store()
        )
        # Basic: local → subscription → metered-cheap → metered-expensive.
        assert _ids(decision) == ["local_a", "sub_a", "metered_a"]

    def test_normal_prefers_subscription_before_capable_local(self):
        profiles = {
            # Priority 1 would win a pure priority sort — access order
            # must put the subscription first anyway.
            "local_strong": _profile(
                "local_strong", capability=Capability.STRONG,
                access=AccessClass.LOCAL, priority=1,
            ),
            # A basic local model is not "normal-capable": excluded.
            "local_basic": _profile(
                "local_basic", capability=Capability.BASIC,
                access=AccessClass.LOCAL, priority=2,
            ),
            "sub_normal": _profile("sub_normal", priority=99),
        }
        decision = build_route(
            _classification(Capability.NORMAL), profiles, _store()
        )
        assert _ids(decision)[0] == "sub_normal"
        assert "local_strong" in _ids(decision)
        assert _ids(decision).index("sub_normal") < _ids(decision).index("local_strong")
        assert _excluded(decision, "local_basic") == "insufficient_capability"

    def test_strong_prefers_strong_subscriptions(self):
        profiles = {
            # Strong tasks never use local models.
            "local_strong": _profile(
                "local_strong", capability=Capability.STRONG,
                access=AccessClass.LOCAL, priority=1,
            ),
            "sub_normal": _profile("sub_normal", priority=1),
            "sub_strong": _profile(
                "sub_strong", capability=Capability.STRONG, priority=10,
            ),
            "metered_strong": _profile(
                "metered_strong", capability=Capability.STRONG,
                access=AccessClass.METERED_CHEAP, priority=20,
            ),
        }
        decision = build_route(
            _classification(Capability.STRONG), profiles, _store()
        )
        assert _ids(decision) == ["sub_strong", "metered_strong"]
        assert decision.candidates[0].access_class is AccessClass.SUBSCRIPTION
        assert _excluded(decision, "local_strong") == "access_class_excluded"
        assert _excluded(decision, "sub_normal") == "insufficient_capability"

    def test_reasoning_prefers_reasoning_subscription(self):
        profiles = {
            "sub_reasoning": _profile(
                "sub_reasoning", capability=Capability.REASONING, priority=10,
            ),
            "sub_strong": _profile(
                "sub_strong", capability=Capability.STRONG, priority=1,
            ),
            "metered_reasoning": _profile(
                "metered_reasoning", capability=Capability.REASONING,
                access=AccessClass.METERED_CHEAP, priority=5,
            ),
        }
        decision = build_route(
            _classification(Capability.REASONING), profiles, _store()
        )
        assert _ids(decision) == ["sub_reasoning", "metered_reasoning"]
        assert _excluded(decision, "sub_strong") == "insufficient_capability"


# ---------------------------------------------------------------------------
# Availability interaction
# ---------------------------------------------------------------------------

class TestAvailability:
    def test_unavailable_first_subscription_falls_back_to_next_subscription(self):
        store = _store()
        store.record_failure("sub_first", FailureCategory.RATE_LIMIT)
        profiles = {
            "sub_first": _profile("sub_first", priority=10),
            "sub_second": _profile("sub_second", priority=20),
            "metered_a": _profile(
                "metered_a", access=AccessClass.METERED_CHEAP, priority=1,
            ),
        }
        decision = build_route(
            _classification(Capability.NORMAL), profiles, store
        )
        assert _ids(decision)[0] == "sub_second"
        assert _excluded(decision, "sub_first") == "unavailable_cooldown"

    def test_all_subscriptions_unavailable_uses_metered_cheap(self):
        store = _store()
        store.record_failure("sub_a", FailureCategory.QUOTA)
        store.record_failure("sub_b", FailureCategory.QUOTA)
        profiles = {
            "sub_a": _profile("sub_a", priority=10),
            "sub_b": _profile("sub_b", priority=20),
            "metered_cheap": _profile(
                "metered_cheap", access=AccessClass.METERED_CHEAP, priority=10,
            ),
            "metered_expensive": _profile(
                "metered_expensive", access=AccessClass.METERED_EXPENSIVE,
                priority=1,
            ),
        }
        decision = build_route(
            _classification(Capability.NORMAL), profiles, store
        )
        assert _ids(decision) == ["metered_cheap", "metered_expensive"]
        assert decision.candidates[0].access_class is AccessClass.METERED_CHEAP

    def test_recovered_subscription_reenters_before_metered(self):
        clock = _FakeClock()
        store = _store(clock=clock)
        store.record_failure("sub_a", FailureCategory.RATE_LIMIT)
        profiles = {
            "sub_a": _profile("sub_a", priority=10),
            "metered_a": _profile(
                "metered_a", access=AccessClass.METERED_CHEAP, priority=1,
            ),
        }

        during_cooldown = build_route(
            _classification(Capability.NORMAL), profiles, store
        )
        assert _ids(during_cooldown) == ["metered_a"]

        # Advance past the (rate-limit) cooldown: the subscription must
        # re-enter ahead of the metered profile.
        clock.now += 61.0
        recovered = build_route(
            _classification(Capability.NORMAL), profiles, store
        )
        assert _ids(recovered) == ["sub_a", "metered_a"]


# ---------------------------------------------------------------------------
# Paid-access protection
# ---------------------------------------------------------------------------

class TestPaidAccessProtection:
    def test_paid_not_selected_when_subscription_is_suitable(self):
        profiles = {
            "sub_a": _profile("sub_a", capability=Capability.STRONG, priority=10),
            "metered_cheap": _profile(
                "metered_cheap", capability=Capability.STRONG,
                access=AccessClass.METERED_CHEAP, priority=1,
            ),
            "metered_expensive": _profile(
                "metered_expensive", capability=Capability.REASONING,
                access=AccessClass.METERED_EXPENSIVE, priority=1,
            ),
        }
        decision = build_route(
            _classification(Capability.STRONG), profiles, _store()
        )
        assert _ids(decision)[0] == "sub_a"
        access_classes = [c.access_class for c in decision.candidates]
        assert AccessClass.METERED_EXPENSIVE in access_classes
        # Subscription always precedes metered.
        assert access_classes.index(AccessClass.SUBSCRIPTION) < min(
            i for i, ac in enumerate(access_classes)
            if ac in (AccessClass.METERED_CHEAP, AccessClass.METERED_EXPENSIVE)
        )

    def test_uncertainty_never_selects_paid_access(self):
        store = _store()
        store.record_failure("sub_a", FailureCategory.QUOTA)
        profiles = {
            "sub_a": _profile("sub_a", priority=10),
            "metered_a": _profile(
                "metered_a", access=AccessClass.METERED_CHEAP, priority=1,
            ),
        }
        decision = build_route(
            _classification(Capability.NORMAL, ambiguous=True),
            profiles,
            store,
        )
        assert _ids(decision) == []
        assert _excluded(decision, "metered_a") == "paid_excluded_uncertain"
        assert "uncertainty_subscription_only" in decision.reasons

    def test_uncertain_work_still_routes_to_subscription(self):
        profiles = {
            "sub_a": _profile("sub_a", priority=10),
            "metered_a": _profile(
                "metered_a", access=AccessClass.METERED_CHEAP, priority=1,
            ),
        }
        decision = build_route(
            _classification(Capability.NORMAL, ambiguous=True),
            profiles,
            _store(),
        )
        assert _ids(decision) == ["sub_a"]


# ---------------------------------------------------------------------------
# Priority and hard-requirement filtering
# ---------------------------------------------------------------------------

class TestOrderingAndFiltering:
    def test_candidate_order_respects_configured_priority(self):
        profiles = {
            "sub_c": _profile("sub_c", priority=30),
            "sub_a": _profile("sub_a", priority=10),
            "sub_b": _profile("sub_b", priority=20),
            "local_a": _profile(
                "local_a", access=AccessClass.LOCAL, priority=1,
            ),
        }
        decision = build_route(
            _classification(Capability.NORMAL), profiles, _store()
        )
        # Subscription group first, ordered by numeric priority.
        assert _ids(decision) == ["sub_a", "sub_b", "sub_c", "local_a"]

    def test_tools_and_context_filter_ineligible_profiles(self):
        profiles = {
            "sub_no_tools": _profile("sub_no_tools", priority=1),
            "sub_small_context": _profile(
                "sub_small_context", priority=2, tools=True, context_limit=128_000,
            ),
            "sub_capable": _profile(
                "sub_capable", priority=3, tools=True, context_limit=1_000_000,
            ),
        }
        decision = build_route(
            _classification(Capability.NORMAL),
            profiles,
            _store(),
            requirements=RouteRequirements(
                supports_tools=True, minimum_context=200_000,
            ),
        )
        assert _ids(decision) == ["sub_capable"]
        assert _excluded(decision, "sub_no_tools") == "tools_unsupported"
        assert _excluded(decision, "sub_small_context") == "context_limit_exceeded"

    def test_thinking_requirement_filters_non_thinking_profiles(self):
        profiles = {
            "sub_plain": _profile(
                "sub_plain", capability=Capability.REASONING, priority=1,
            ),
            "sub_thinking": _profile(
                "sub_thinking", capability=Capability.REASONING,
                priority=2, thinking=True,
            ),
        }
        decision = build_route(
            _classification(Capability.REASONING),
            profiles,
            _store(),
            requirements=RouteRequirements(supports_thinking=True),
        )
        assert _ids(decision) == ["sub_thinking"]
        assert _excluded(decision, "sub_plain") == "thinking_unsupported"

    def test_local_only_requirement_excludes_all_remote_profiles(self):
        profiles = {
            "local_a": _profile(
                "local_a", access=AccessClass.LOCAL, priority=20,
            ),
            "sub_a": _profile("sub_a", priority=1),
            "metered_a": _profile(
                "metered_a", access=AccessClass.METERED_CHEAP, priority=1,
            ),
        }
        decision = build_route(
            _classification(Capability.NORMAL),
            profiles,
            _store(),
            requirements=RouteRequirements(local_only=True),
        )
        assert _ids(decision) == ["local_a"]
        assert _excluded(decision, "sub_a") == "local_only_blocks_remote"
        assert _excluded(decision, "metered_a") == "local_only_blocks_remote"
        assert decision.requirements.allow_cloud is False
        assert decision.escalation_config.get("allow_cloud_fallback") is False

    def test_classifier_and_disabled_profiles_are_excluded(self):
        profiles = {
            "classifier_a": _profile("classifier_a", role="classifier"),
            "disabled_a": _profile("disabled_a", enabled=False),
            "sub_a": _profile("sub_a", priority=10),
        }
        decision = build_route(
            _classification(Capability.NORMAL), profiles, _store()
        )
        assert _ids(decision) == ["sub_a"]
        assert _excluded(decision, "classifier_a") == "classifier_profile"
        assert _excluded(decision, "disabled_a") == "profile_disabled"


# ---------------------------------------------------------------------------
# Safe fallback
# ---------------------------------------------------------------------------

class TestSafeFallback:
    def test_policy_failure_returns_safe_subscription_default(self):
        # Nothing survives the normal filters: the only reasoning-capable
        # profile is cooling down and the subscription is under-capability.
        store = _store()
        store.record_failure("metered_reasoning", FailureCategory.QUOTA)
        profiles = {
            "sub_normal": _profile("sub_normal", priority=10),
            "metered_reasoning": _profile(
                "metered_reasoning", capability=Capability.REASONING,
                access=AccessClass.METERED_CHEAP, priority=1,
            ),
        }
        decision = build_route(
            _classification(Capability.REASONING), profiles, store
        )
        assert _ids(decision) == ["sub_normal"]
        assert decision.candidates[0].access_class is AccessClass.SUBSCRIPTION
        assert "safe_subscription_default" in decision.reasons
        # The fallback itself must be eligible (available).
        assert store.is_available("sub_normal") is True
        assert _excluded(decision, "metered_reasoning") == "unavailable_cooldown"

    def test_safe_fallback_never_picks_unavailable_or_paid_profiles(self):
        store = _store()
        store.record_failure("sub_a", FailureCategory.QUOTA)
        store.record_failure("sub_b", FailureCategory.QUOTA)
        profiles = {
            "sub_a": _profile("sub_a", priority=10),
            "sub_b": _profile("sub_b", priority=20),
            "metered_a": _profile(
                "metered_a", access=AccessClass.METERED_CHEAP, priority=1,
            ),
        }
        decision = build_route(
            _classification(Capability.NORMAL), profiles, store
        )
        # Subscriptions are down; metered stays a legitimate candidate for
        # *certain* classifications, but the safe fallback itself never
        # triggers here because candidates exist.
        assert _ids(decision) == ["metered_a"]
        assert "safe_subscription_default" not in decision.reasons

    def test_no_eligible_profiles_returns_empty_candidates(self):
        profiles = {"sub_a": _profile("sub_a", enabled=False)}
        decision = build_route(
            _classification(Capability.NORMAL), profiles, _store()
        )
        assert decision.candidates == ()
        assert "no_eligible_candidates" in decision.reasons


# ---------------------------------------------------------------------------
# Decision shape
# ---------------------------------------------------------------------------

class TestDecisionShape:
    def test_decision_reports_capability_and_serializes_candidates(self):
        profiles = {"sub_a": _profile("sub_a", priority=10)}
        decision = build_route(
            _classification(Capability.NORMAL), profiles, _store()
        )
        assert decision.capability is Capability.NORMAL
        assert decision.tier in (
            "local_fast", "local_general", "local_coding", "local_reasoning",
            "cloud_standard", "cloud_coding", "cloud_reasoning", "cloud_research",
        )
        assert list(decision.eligible_models) == _ids(decision)
        payload = decision.to_dict()
        assert payload["candidates"][0]["profile_id"] == "sub_a"
        assert payload["candidates"][0]["access_class"] == "subscription"

    def test_access_order_is_configurable(self):
        original = dict(ACCESS_ORDER)
        try:
            ACCESS_ORDER[Capability.NORMAL] = (
                AccessClass.LOCAL,
                AccessClass.SUBSCRIPTION,
                AccessClass.METERED_CHEAP,
                AccessClass.METERED_EXPENSIVE,
            )
            profiles = {
                "local_a": _profile(
                    "local_a", access=AccessClass.LOCAL, priority=30,
                ),
                "sub_a": _profile("sub_a", priority=10),
            }
            decision = build_route(
                _classification(Capability.NORMAL), profiles, _store()
            )
            assert _ids(decision)[0] == "local_a"
        finally:
            ACCESS_ORDER.clear()
            ACCESS_ORDER.update(original)


# ---------------------------------------------------------------------------
# Legacy ``choose_route`` adapter (deprecated)
# ---------------------------------------------------------------------------

def _signals(**overrides) -> RuntimeSignals:
    defaults = dict(
        prompt_chars=500,
        estimated_tokens=125,
        attached_file_count=0,
        repo_size_mb=5.0,
        repo_files_referenced=0,
        has_git_diff=False,
        has_test_command=False,
        prior_attempts=0,
        prior_failures=0,
        tool_calls_so_far=0,
        deadline="normal",
        user_hint="",
    )
    defaults.update(overrides)
    return RuntimeSignals(**defaults)


def _default_classification(**overrides) -> dict[str, str]:
    defaults = {
        "task_type": "explanation",
        "complexity": "low",
        "coding": "none",
        "reasoning_depth": "direct",
        "fresh_information": "none",
        "repo_access": "none",
        "tool_use": "none",
        "privacy": "ordinary",
        "latency": "normal",
        "failure_cost": "low",
        "output_contract": "freeform",
    }
    defaults.update(overrides)
    return defaults


class TestLegacyAdapter:
    def test_choose_route_is_deprecated(self):
        with pytest.warns(DeprecationWarning):
            choose_route(_default_classification(), _signals())

    def test_trivial_coding_routes_local_coding_with_candidates(self):
        cls = _default_classification(
            task_type="code_edit",
            complexity="low",
            coding="dominant",
            repo_access="modify",
            tool_use="shell",
            failure_cost="medium",
            output_contract="exact_patch",
        )
        with pytest.warns(DeprecationWarning):
            decision = choose_route(cls, _signals(), overall_confidence=0.95)
        assert decision.tier == "local_coding"
        assert "needs_coding" in decision.reasons
        # API-compatible decisions carry executable candidates.
        assert decision.candidates
        assert decision.candidates[0].profile_id == "local_coding"
        assert decision.eligible_models
        assert decision.eligible_models[0] == "local_coding"

    def test_prior_failure_escalates_to_cloud_standard(self):
        cls = _default_classification(
            task_type="code_edit",
            complexity="low",
            coding="dominant",
            repo_access="modify",
            tool_use="shell",
            failure_cost="medium",
        )
        with pytest.warns(DeprecationWarning):
            decision = choose_route(
                cls,
                _signals(prior_failures=1),
                overall_confidence=0.95,
            )
        assert decision.tier == "cloud_standard"
        assert decision.candidates
        assert decision.candidates[0].access_class is AccessClass.SUBSCRIPTION

    def test_local_only_stays_local(self):
        cls = _default_classification(
            privacy="local_only", coding="dominant", repo_access="modify",
        )
        with pytest.warns(DeprecationWarning):
            decision = choose_route(cls, _signals(), overall_confidence=0.95)
        assert decision.tier.startswith("local_")
        assert not decision.requirements.allow_cloud
        assert decision.escalation_config.get("allow_cloud_fallback") is False
        assert "privacy_requires_local" in decision.reasons
        assert all(
            c.access_class is AccessClass.LOCAL for c in decision.candidates
        )

    def test_abstain_is_conservative_subscription(self):
        with pytest.warns(DeprecationWarning):
            decision = choose_route(
                _default_classification(),
                _signals(),
                abstain=True,
                overall_confidence=0.0,
            )
        assert "classifier_abstained" in decision.reasons
        assert decision.candidates
        assert decision.candidates[0].access_class is AccessClass.SUBSCRIPTION
        assert decision.capability is Capability.NORMAL

    def test_deep_reasoning_routes_to_reasoning_subscription(self):
        cls = _default_classification(
            task_type="planning",
            reasoning_depth="deep",
            complexity="high",
        )
        with pytest.warns(DeprecationWarning):
            decision = choose_route(cls, _signals(), overall_confidence=0.95)
        assert decision.capability is Capability.REASONING
        assert decision.candidates
        assert decision.candidates[0].capability >= Capability.REASONING
        assert "needs_deep_reasoning" in decision.reasons
