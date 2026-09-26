"""
Tests for the escalation state machine.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.availability import FailureCategory
from router.escalation import (
    ExecutionState,
    should_escalate,
    escalate,
    record_attempt,
    outcome_category,
)


class TestShouldEscalate:
    def test_success_no_escalation(self):
        state = ExecutionState(selected_tiers=["local_coding"])
        ok, reason = should_escalate(state, {"status": "success"})
        assert ok is False
        assert reason == "success"

    def test_tests_failed_triggers_escalation(self):
        state = ExecutionState(
            selected_tiers=["local_coding"],
            max_attempts=3,
        )
        ok, reason = should_escalate(
            state,
            {"status": "failed", "failure_reason": "tests_failed"},
        )
        assert ok is True
        assert reason == "tests_failed"

    def test_patch_not_applied_triggers_escalation(self):
        state = ExecutionState(
            selected_tiers=["local_coding"],
            max_attempts=3,
        )
        ok, reason = should_escalate(
            state,
            {"status": "failed", "failure_reason": "patch_not_applied"},
        )
        assert ok is True
        assert reason == "patch_not_applied"

    def test_exhausted_no_escalation(self):
        state = ExecutionState(
            selected_tiers=["local_coding", "cloud_coding", "cloud_reasoning"],
            max_attempts=3,
        )
        ok, reason = should_escalate(
            state,
            {"status": "failed", "failure_reason": "tests_failed"},
        )
        assert ok is False
        assert reason == "max_attempts_exhausted"

    def test_generic_failure_triggers_escalation(self):
        state = ExecutionState(
            selected_tiers=["local_coding"],
            max_attempts=3,
        )
        ok, reason = should_escalate(state, {"status": "failed"})
        assert ok is True
        assert reason == "generic_failure"


class TestEscalateChain:
    def test_local_coding_escalates_to_cloud_standard(self):
        state = ExecutionState()
        next_tier = escalate("local_coding", state)
        assert next_tier == "cloud_standard"

    def test_local_reasoning_escalates_to_cloud_reasoning(self):
        state = ExecutionState()
        next_tier = escalate("local_reasoning", state)
        assert next_tier == "cloud_reasoning"

    def test_force_local_stops_at_local_reasoning(self):
        state = ExecutionState()
        next_tier = escalate("local_reasoning", state, force_local=True)
        assert next_tier is None

    def test_no_cloud_fallback_stops_at_boundary(self):
        state = ExecutionState(allow_cloud_fallback=False)
        next_tier = escalate("local_reasoning", state)
        assert next_tier is None

    def test_cloud_research_is_terminal(self):
        state = ExecutionState()
        next_tier = escalate("cloud_research", state)
        assert next_tier is None

    def test_full_local_chain(self):
        """Walk the full local escalation chain."""
        chain = []
        tier = "local_fast"
        state = ExecutionState(allow_cloud_fallback=False)

        while tier:
            chain.append(tier)
            tier = escalate(tier, state, force_local=True)

        assert chain == [
            "local_fast",
            "local_general",
        ]

    def test_full_chain_with_cloud(self):
        """Walk the full chain including cloud fallback."""
        chain = []
        tier = "local_fast"
        state = ExecutionState(allow_cloud_fallback=True)

        while tier:
            chain.append(tier)
            tier = escalate(tier, state)

        assert "cloud_standard" in chain
        assert "cloud_reasoning" in chain
        assert chain[-1] == "cloud_research"


class TestOutcomeCategory:
    """Failure-category classification of provider outcomes."""

    def test_success_has_no_category(self):
        assert outcome_category({"status": "success"}) is None

    def test_task_quality_reasons_map_to_task_failure(self):
        for reason in ("tests_failed", "patch_not_applied", "invalid_output"):
            assert outcome_category(
                {"status": "failed", "failure_reason": reason}
            ) is FailureCategory.TASK_FAILURE

    def test_subscription_exhausted_maps_to_quota(self):
        assert outcome_category(
            {"status": "failed", "failure_reason": "subscription_exhausted"}
        ) is FailureCategory.QUOTA

    def test_rate_limited_maps_to_rate_limit(self):
        assert outcome_category(
            {"status": "failed", "failure_reason": "rate_limited"}
        ) is FailureCategory.RATE_LIMIT

    def test_tool_error_maps_to_local_tool_failure(self):
        assert outcome_category(
            {"status": "failed", "failure_reason": "tool_error"}
        ) is FailureCategory.LOCAL_TOOL_FAILURE

    def test_unknown_failure_is_conservative(self):
        assert outcome_category({"status": "failed"}) is (
            FailureCategory.PROVIDER_UNKNOWN
        )

    def test_explicit_category_is_consumed(self):
        outcome = {
            "status": "failed",
            "failure_category": FailureCategory.CAPACITY,
        }
        assert outcome_category(outcome) is FailureCategory.CAPACITY


class TestFailureCategorySeparation:
    """Task failures raise the task counter; availability failures do not."""

    def test_task_failure_increments_task_failure_count(self):
        state = ExecutionState()
        record_attempt(
            state,
            "local_coding",
            {"status": "failed", "failure_reason": "tests_failed"},
        )
        record_attempt(
            state,
            "cloud_coding",
            {"status": "failed", "failure_reason": "patch_not_applied"},
        )
        assert state.task_failure_count == 2

    def test_availability_failure_advances_without_task_capability(self):
        state = ExecutionState()
        record_attempt(
            state,
            "cloud_coding",
            {"status": "failed", "failure_reason": "subscription_exhausted"},
        )
        record_attempt(
            state,
            "cloud_reasoning",
            {"status": "failed", "failure_reason": "rate_limited"},
        )
        # The candidate advanced, but the task was never blamed.
        assert state.task_failure_count == 0

    def test_local_tool_failure_does_not_blame_task_or_provider(self):
        state = ExecutionState()
        record_attempt(
            state,
            "local_coding",
            {"status": "failed", "failure_reason": "tool_error"},
        )
        assert state.task_failure_count == 0

    def test_success_does_not_touch_task_failure_count(self):
        state = ExecutionState(task_failure_count=1)
        record_attempt(state, "cloud_coding", {"status": "success"})
        assert state.task_failure_count == 1

    def test_task_failure_count_serializes(self):
        state = ExecutionState()
        record_attempt(
            state,
            "local_coding",
            {"status": "failed", "failure_reason": "tests_failed"},
        )
        assert state.to_dict()["task_failure_count"] == 1


class TestRecordAttempt:
    def test_record_accumulates(self):
        state = ExecutionState()
        state = record_attempt(state, "local_coding", {"status": "failed"}, latency_ms=500)
        state = record_attempt(state, "cloud_coding", {"status": "success"}, latency_ms=2000)

        assert state.attempts == 2
        assert state.selected_tiers == ["local_coding", "cloud_coding"]
        assert state.elapsed_ms == 2500
        assert state.last_tier == "cloud_coding"
        assert not state.exhausted

    def test_exhausted_after_max(self):
        state = ExecutionState(max_attempts=2)
        state = record_attempt(state, "local_coding", {"status": "failed"})
        state = record_attempt(state, "cloud_coding", {"status": "failed"})
        assert state.exhausted is True
