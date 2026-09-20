"""
Tests for the deterministic routing policy.

These use synthetic classifications and signals — no model calls.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.policy import choose_route
from router.signals import RuntimeSignals


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


# ---------------------------------------------------------------------------
# Hard constraints
# ---------------------------------------------------------------------------

class TestPrivacyHardConstraint:
    def test_local_only_blocks_cloud(self):
        cls = _default_classification(privacy="local_only", coding="dominant")
        decision = choose_route(cls, _signals(), overall_confidence=0.95)
        assert decision.tier in ("local_coding", "local_reasoning")
        assert not decision.requirements.allow_cloud
        assert "privacy_requires_local" in decision.reasons

    def test_sensitive_data_stays_local(self):
        cls = _default_classification(privacy="sensitive", coding="substantial")
        decision = choose_route(cls, _signals(), overall_confidence=0.95)
        assert decision.tier.startswith("local_")


# ---------------------------------------------------------------------------
# Coding tasks
# ---------------------------------------------------------------------------

class TestCodingRouting:
    def test_trivial_code_edit_goes_local(self):
        cls = _default_classification(
            task_type="code_edit",
            complexity="low",
            coding="dominant",
            repo_access="modify",
            output_contract="exact_patch",
        )
        decision = choose_route(cls, _signals(), overall_confidence=0.95)
        assert decision.tier == "local_coding"
        assert "needs_coding" in decision.reasons
        assert "needs_shell" in decision.reasons

    def test_complex_code_edit_goes_cloud(self):
        cls = _default_classification(
            task_type="code_edit",
            complexity="high",
            coding="dominant",
            repo_access="modify",
            failure_cost="very_high",
            output_contract="exact_patch",
        )
        decision = choose_route(cls, _signals(), overall_confidence=0.95)
        assert decision.tier == "cloud_coding"
        assert decision.requirements.requires_verification is True

    def test_prior_failure_escalates_to_cloud(self):
        cls = _default_classification(
            task_type="code_edit",
            complexity="low",
            coding="dominant",
            repo_access="modify",
        )
        decision = choose_route(
            cls,
            _signals(prior_failures=1),
            overall_confidence=0.95,
        )
        assert decision.tier == "cloud_coding"

    def test_medium_coding_stays_local(self):
        cls = _default_classification(
            task_type="debugging",
            complexity="medium",
            coding="substantial",
            failure_cost="medium",
        )
        decision = choose_route(cls, _signals(), overall_confidence=0.95)
        assert decision.tier == "local_coding"


# ---------------------------------------------------------------------------
# Research / web tasks
# ---------------------------------------------------------------------------

class TestResearchRouting:
    def test_fresh_information_required_goes_cloud_research(self):
        cls = _default_classification(
            task_type="research",
            fresh_information="required",
        )
        decision = choose_route(cls, _signals(), overall_confidence=0.95)
        assert decision.tier == "cloud_research"
        assert decision.requirements.requires_web is True

    def test_fresh_information_useful_stays_local(self):
        cls = _default_classification(
            task_type="research",
            fresh_information="useful",
            complexity="medium",
        )
        decision = choose_route(cls, _signals(), overall_confidence=0.95)
        assert decision.tier != "cloud_research"


# ---------------------------------------------------------------------------
# Deep reasoning
# ---------------------------------------------------------------------------

class TestDeepReasoning:
    def test_deep_reasoning_goes_local_reasoning(self):
        cls = _default_classification(
            task_type="planning",
            reasoning_depth="deep",
            complexity="high",
        )
        decision = choose_route(cls, _signals(), overall_confidence=0.95)
        assert decision.tier == "local_reasoning"

    def test_very_high_failure_cost_goes_cloud_reasoning(self):
        cls = _default_classification(
            task_type="planning",
            reasoning_depth="deep",
            failure_cost="very_high",
        )
        decision = choose_route(cls, _signals(), overall_confidence=0.95)
        assert decision.tier == "cloud_reasoning"


# ---------------------------------------------------------------------------
# Abstention / low confidence
# ---------------------------------------------------------------------------

class TestAbstention:
    def test_abstain_defaults_to_local_general(self):
        cls = _default_classification()
        decision = choose_route(
            cls, _signals(), abstain=True, overall_confidence=0.0,
        )
        assert decision.tier == "local_general"
        assert "classifier_abstained" in decision.reasons


# ---------------------------------------------------------------------------
# Verification requirements
# ---------------------------------------------------------------------------

class TestVerification:
    def test_exact_patch_requires_verification(self):
        cls = _default_classification(
            task_type="code_edit",
            coding="dominant",
            repo_access="modify",
            output_contract="exact_patch",
        )
        decision = choose_route(cls, _signals(), overall_confidence=0.95)
        assert decision.requirements.requires_verification is True

    def test_high_failure_cost_requires_verification(self):
        cls = _default_classification(
            task_type="planning",
            failure_cost="high",
        )
        decision = choose_route(cls, _signals(), overall_confidence=0.95)
        assert decision.requirements.requires_verification is True


# ---------------------------------------------------------------------------
# Interactive latency
# ---------------------------------------------------------------------------

class TestLatencyPreference:
    def test_interactive_trivial_uses_local_fast(self):
        cls = _default_classification(
            task_type="classification",
            complexity="trivial",
            latency="interactive",
        )
        decision = choose_route(cls, _signals(), overall_confidence=0.95)
        assert decision.tier == "local_fast"

    def test_interactive_does_not_override_coding(self):
        cls = _default_classification(
            task_type="code_edit",
            complexity="trivial",
            coding="dominant",
            latency="interactive",
        )
        decision = choose_route(cls, _signals(), overall_confidence=0.95)
        assert decision.tier == "local_coding"


# ---------------------------------------------------------------------------
# Escalation config
# ---------------------------------------------------------------------------

class TestEscalationConfig:
    def test_local_only_blocks_cloud_escalation(self):
        cls = _default_classification(
            privacy="local_only",
            coding="dominant",
        )
        decision = choose_route(cls, _signals(), overall_confidence=0.95)
        assert decision.escalation_config.get("allow_cloud_fallback") is False

    def test_ordinary_privacy_allows_cloud_escalation(self):
        cls = _default_classification(
            task_type="code_edit",
            complexity="high",
            coding="dominant",
        )
        decision = choose_route(cls, _signals(), overall_confidence=0.95)
        assert decision.escalation_config.get("allow_cloud_fallback") is True


# ---------------------------------------------------------------------------
# Reason codes
# ---------------------------------------------------------------------------

class TestReasonCodes:
    def test_coding_task_includes_needs_coding(self):
        cls = _default_classification(
            task_type="debugging",
            coding="substantial",
        )
        decision = choose_route(cls, _signals(), overall_confidence=0.95)
        assert "needs_coding" in decision.reasons

    def test_selected_tier_in_reasons(self):
        cls = _default_classification()
        decision = choose_route(cls, _signals(), overall_confidence=0.95)
        selected_reasons = [r for r in decision.reasons if r.startswith("selected:")]
        assert len(selected_reasons) == 1
        assert decision.tier in selected_reasons[0]
