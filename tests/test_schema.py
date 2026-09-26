"""
Tests for the router schema definition and validation.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from dataclasses import FrozenInstanceError

from router.router_schema import (
    ROUTER_SCHEMA,
    CRITICAL_FIELDS,
    ROUTER_FIELD_ORDER,
    validate_router_output,
    build_router_preset,
)
from router.config import (
    AccessClass,
    Capability,
    MODEL_PROFILES,
    ModelProfile,
    executor_candidates,
)
from router.decision import (
    Candidate,
    CandidateExclusion,
    RouteDecision,
    TierRequirement,
)


class TestRouterSchema:
    def test_has_expected_fields(self):
        expected = {
            "task_type", "complexity", "coding", "reasoning_depth",
            "fresh_information", "repo_access", "tool_use", "privacy",
            "latency", "failure_cost", "output_contract",
        }
        assert set(ROUTER_SCHEMA.keys()) == expected

    def test_all_fields_are_enum(self):
        for name, spec in ROUTER_SCHEMA.items():
            assert spec["type"] == "enum", f"{name} should be enum"

    def test_all_have_choices(self):
        for name, spec in ROUTER_SCHEMA.items():
            assert "choices" in spec, f"{name} missing choices"
            assert len(spec["choices"]) > 1, f"{name} has only {len(spec['choices'])} choice(s)"

    def test_all_have_descriptions(self):
        for name, spec in ROUTER_SCHEMA.items():
            assert spec.get("description"), f"{name} missing description"

    def test_no_choice_exceeds_255(self):
        for name, spec in ROUTER_SCHEMA.items():
            assert len(spec["choices"]) <= 255, f"{name} exceeds 255 choices"

    def test_field_order_matches_schema(self):
        assert ROUTER_FIELD_ORDER == list(ROUTER_SCHEMA.keys())


class TestCriticalFields:
    def test_critical_fields_exist_in_schema(self):
        for f in CRITICAL_FIELDS:
            assert f in ROUTER_SCHEMA, f"{f} not in schema"

    def test_critical_fields_are_subset(self):
        assert CRITICAL_FIELDS.issubset(set(ROUTER_SCHEMA.keys()))


class TestValidateRouterOutput:
    def test_valid_output(self):
        raw = {
            "task_type": "code_edit",
            "complexity": "low",
            "coding": "dominant",
            "reasoning_depth": "direct",
            "fresh_information": "none",
            "repo_access": "modify",
            "tool_use": "shell",
            "privacy": "ordinary",
            "latency": "normal",
            "failure_cost": "low",
            "output_contract": "exact_patch",
        }
        validated, errors = validate_router_output(raw)
        assert len(errors) == 0
        assert validated == raw

    def test_missing_field(self):
        raw = {"task_type": "code_edit"}
        validated, errors = validate_router_output(raw)
        assert len(errors) == 10  # 10 missing fields
        assert "task_type" in validated

    def test_invalid_choice(self):
        raw = {
            "task_type": "INVALID_VALUE",
            "complexity": "low",
            "coding": "dominant",
            "reasoning_depth": "direct",
            "fresh_information": "none",
            "repo_access": "modify",
            "tool_use": "shell",
            "privacy": "ordinary",
            "latency": "normal",
            "failure_cost": "low",
            "output_contract": "exact_patch",
        }
        validated, errors = validate_router_output(raw)
        assert len(errors) == 1
        assert "task_type" in errors[0]
        assert "task_type" not in validated


class TestBuildRouterPreset:
    def test_preset_structure(self):
        preset = build_router_preset("Fix the login bug")
        assert preset["id"] == "pi_router"
        assert preset["context"] == "Fix the login bug"
        assert preset["schema"] == ROUTER_SCHEMA


# ---------------------------------------------------------------------------
# Capability / access registry (Task 2)
#
# Capability (what a task/model can do) and access class (how using a model
# is granted/paid for) are independent concepts:
#   basic < normal < strong < reasoning
#   local | subscription | metered-cheap | metered-expensive
# ---------------------------------------------------------------------------


def _synthetic_profile(
    capability=Capability.STRONG,
    access_class=AccessClass.SUBSCRIPTION,
    priority=20,
    role="executor",
    enabled=True,
):
    """A synthetic registry profile; production metered profiles require
    real provider/model configuration and must not be invented by tests."""
    return ModelProfile(
        model_id="synthetic/model",
        role=role,
        provider="synthetic",
        capabilities=("general", "coding"),
        latency_class="fast",
        capability=capability,
        access_class=access_class,
        priority=priority,
        enabled=enabled,
    )


class TestCapabilityAccessRegistry:
    def test_capability_ordering(self):
        assert (
            Capability.BASIC < Capability.NORMAL < Capability.STRONG < Capability.REASONING
        )

    def test_capability_serializes_as_lowercase_name(self):
        for cap in Capability:
            assert cap.name.lower() in {"basic", "normal", "strong", "reasoning"}

    def test_access_class_serialization_values(self):
        assert {c.value for c in AccessClass} == {
            "local",
            "subscription",
            "metered-cheap",
            "metered-expensive",
        }

    def test_capability_and_access_class_are_independent(self):
        strong_local = _synthetic_profile(access_class=AccessClass.LOCAL)
        basic_metered = _synthetic_profile(
            capability=Capability.BASIC,
            access_class=AccessClass.METERED_CHEAP,
        )
        assert strong_local.capability is Capability.STRONG
        assert strong_local.access_class is AccessClass.LOCAL
        assert basic_metered.capability is Capability.BASIC
        assert basic_metered.access_class is AccessClass.METERED_CHEAP

    def test_profile_serializes_access_class(self):
        profile = _synthetic_profile(
            capability=Capability.STRONG,
            access_class=AccessClass.SUBSCRIPTION,
            priority=20,
        )
        assert profile.to_dict()["access_class"] == "subscription"

    def test_profile_serializes_capability_and_registry_fields(self):
        d = _synthetic_profile().to_dict()
        assert d["capability"] == "strong"
        assert d["priority"] == 20
        assert d["enabled"] is True
        assert isinstance(d["context_limit"], int)
        assert isinstance(d["supports_tools"], bool)
        assert isinstance(d["supports_thinking"], bool)

    def test_profile_preserves_legacy_serialization(self):
        subscription = _synthetic_profile(access_class=AccessClass.SUBSCRIPTION)
        assert subscription.to_dict()["billing"] == "subscription_quota"
        assert subscription.to_dict()["capabilities"] == ["general", "coding"]
        local = _synthetic_profile(access_class=AccessClass.LOCAL)
        assert local.billing == "local"
        assert local.to_dict()["billing"] == "local"
        metered = _synthetic_profile(access_class=AccessClass.METERED_EXPENSIVE)
        assert metered.billing == "paid"
        assert metered.to_dict()["billing"] == "paid"

    def test_synthetic_metered_profile_serializes_access_class(self):
        assert (
            _synthetic_profile(access_class=AccessClass.METERED_CHEAP).to_dict()[
                "access_class"
            ]
            == "metered-cheap"
        )
        assert (
            _synthetic_profile(access_class=AccessClass.METERED_EXPENSIVE).to_dict()[
                "access_class"
            ]
            == "metered-expensive"
        )

    def test_classifier_profiles_are_never_executor_candidates(self):
        candidates = executor_candidates()
        assert candidates, "registry should expose executor candidates"
        for name, profile in MODEL_PROFILES.items():
            if profile.role == "classifier":
                assert name not in candidates

        # Even an explicitly enabled synthetic classifier profile is excluded.
        synthetic = {
            "synthetic_classifier": _synthetic_profile(role="classifier"),
            "synthetic_executor": _synthetic_profile(role="executor"),
        }
        assert set(executor_candidates(synthetic)) == {"synthetic_executor"}

    def test_disabled_profile_is_not_an_executor_candidate(self):
        synthetic = {
            "disabled": _synthetic_profile(enabled=False),
            "enabled": _synthetic_profile(),
        }
        assert set(executor_candidates(synthetic)) == {"enabled"}

    def test_registry_profiles_declare_capability_and_access_class(self):
        for name, profile in MODEL_PROFILES.items():
            assert isinstance(profile.capability, Capability), name
            assert isinstance(profile.access_class, AccessClass), name
            d = profile.to_dict()
            assert d["capability"] in {"basic", "normal", "strong", "reasoning"}, name
            assert d["access_class"] in {
                "local",
                "subscription",
                "metered-cheap",
                "metered-expensive",
            }, name

    def test_registry_metered_profiles_stay_disabled_without_configuration(self):
        # Metered access requires real provider credentials and model IDs
        # before it can spend money.  The registry may carry *example*
        # metered profiles (Task 11: openrouter_qwen / premium_fallback,
        # provider/model IDs env-overridable), but they must stay disabled
        # until explicitly enabled; paid-path behavior is tested with
        # synthetic metered profiles.
        metered = (
            AccessClass.METERED_CHEAP,
            AccessClass.METERED_EXPENSIVE,
        )
        for name, profile in MODEL_PROFILES.items():
            if profile.access_class in metered:
                assert profile.enabled is False, name
        assert "openrouter_qwen" in MODEL_PROFILES
        assert "premium_fallback" in MODEL_PROFILES

    def test_registry_migration_assigns_capability_and_access(self):
        assert MODEL_PROFILES["rlcd_fast"].role == "classifier"
        assert MODEL_PROFILES["rlcd_smart"].role == "classifier"
        assert MODEL_PROFILES["local_coding"].access_class is AccessClass.LOCAL
        assert MODEL_PROFILES["local_general"].access_class is AccessClass.LOCAL
        assert MODEL_PROFILES["gpt_6_luna"].capability is Capability.NORMAL
        assert MODEL_PROFILES["gpt_6_luna"].access_class is AccessClass.SUBSCRIPTION
        assert MODEL_PROFILES["gpt_5_3_codex"].capability is Capability.STRONG
        assert MODEL_PROFILES["claude_sonnet_5"].capability is Capability.STRONG
        assert MODEL_PROFILES["claude_sonnet_5"].access_class is AccessClass.SUBSCRIPTION
        assert MODEL_PROFILES["claude_opus_5"].capability is Capability.REASONING
        assert MODEL_PROFILES["claude_opus_5"].access_class is AccessClass.SUBSCRIPTION

    def test_registry_priorities_are_positive_ints(self):
        for name, profile in MODEL_PROFILES.items():
            assert isinstance(profile.priority, int) and profile.priority > 0, name


class TestCandidateRecords:
    def _candidate(self, **overrides):
        fields = dict(
            profile_id="gpt_6_luna",
            provider="github-copilot",
            model="gpt-6-luna",
            capability=Capability.NORMAL,
            access_class=AccessClass.SUBSCRIPTION,
            priority=10,
        )
        fields.update(overrides)
        return Candidate(**fields)

    def test_candidate_capability_and_access_class_serialization(self):
        assert self._candidate().to_dict() == {
            "profile_id": "gpt_6_luna",
            "provider": "github-copilot",
            "model": "gpt-6-luna",
            "capability": "normal",
            "access_class": "subscription",
            "priority": 10,
        }

    def test_candidate_records_are_immutable(self):
        candidate = self._candidate()
        with pytest.raises(FrozenInstanceError):
            candidate.profile_id = "other"

    def test_candidate_exclusion_serializes_profile_and_reason(self):
        exclusion = CandidateExclusion(
            profile_id="rlcd_smart", reason="classifier_profile"
        )
        assert exclusion.to_dict() == {
            "profile_id": "rlcd_smart",
            "reason": "classifier_profile",
        }
        with pytest.raises(FrozenInstanceError):
            exclusion.reason = "other"


class TestRouteDecisionCapabilityFields:
    def _decision(self, **overrides):
        fields = dict(
            tier="cloud_coding",
            eligible_models=("gpt_5_3_codex",),
            requirements=TierRequirement(
                minimum_capability="coding",
                requires_tools=("shell",),
                requires_verification=True,
                allow_cloud=True,
                allow_weak_models=False,
                requires_web=False,
            ),
            classifier_used="rlcd_fast",
            confidence=0.9,
            critical_confidence=0.85,
            reasons=("needs_coding",),
        )
        fields.update(overrides)
        return RouteDecision(**fields)

    def test_route_decision_defaults_capability_and_candidates(self):
        decision = self._decision()
        assert decision.capability is Capability.NORMAL
        assert decision.candidates == ()
        assert decision.excluded == ()

    def test_route_decision_serializes_capability_candidates_and_exclusions(self):
        decision = self._decision(
            capability=Capability.STRONG,
            candidates=(
                Candidate(
                    profile_id="gpt_5_3_codex",
                    provider="github-copilot",
                    model="gpt-5.3-codex",
                    capability=Capability.STRONG,
                    access_class=AccessClass.SUBSCRIPTION,
                    priority=10,
                ),
            ),
            excluded=(
                CandidateExclusion(
                    profile_id="rlcd_fast", reason="classifier_profile"
                ),
            ),
        )
        serialized = decision.to_dict()
        assert serialized["capability"] == "strong"
        assert serialized["candidates"] == [
            {
                "profile_id": "gpt_5_3_codex",
                "provider": "github-copilot",
                "model": "gpt-5.3-codex",
                "capability": "strong",
                "access_class": "subscription",
                "priority": 10,
            }
        ]
        assert serialized["excluded"] == [
            {"profile_id": "rlcd_fast", "reason": "classifier_profile"}
        ]
        # Compatibility fields retained until migration completes.
        assert serialized["tier"] == "cloud_coding"
        assert serialized["eligible_models"] == ["gpt_5_3_codex"]
