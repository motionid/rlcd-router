"""
Tests for in-memory availability cooldowns and recovery.

Pins the binding cooldown-boundary decision: a profile is excluded
strictly *before* its ``cooldown_until`` and becomes eligible *at or
after* it.  All time is injected — no sleeps, no background timers.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.availability import (
    AvailabilityRecord,
    AvailabilityStore,
    FailureCategory,
)
from router.config import CooldownConfig, COOLDOWN_CONFIG


class FakeClock:
    """Deterministic injectable clock."""

    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_store(
    cooldowns: CooldownConfig | None = None,
    clock: FakeClock | None = None,
) -> tuple[AvailabilityStore, FakeClock]:
    clock = clock or FakeClock()
    return AvailabilityStore(cooldowns or COOLDOWN_CONFIG, clock), clock


class TestFailureCategories:
    def test_categories_are_string_enum(self):
        assert FailureCategory.RATE_LIMIT.value == "rate_limit"
        assert FailureCategory.QUOTA.value == "quota"
        assert FailureCategory.AUTH.value == "auth"
        assert FailureCategory.CAPACITY.value == "capacity"
        assert FailureCategory.PROVIDER_UNKNOWN.value == "provider_unknown"
        assert FailureCategory.TASK_FAILURE.value == "task_failure"
        assert FailureCategory.LOCAL_TOOL_FAILURE.value == "local_tool_failure"


class TestCooldownRecording:
    def test_rate_limit_uses_short_cooldown(self):
        store, clock = make_store()
        record = store.record_failure("gpt_6_luna", FailureCategory.RATE_LIMIT)

        assert isinstance(record, AvailabilityRecord)
        assert record.reason == FailureCategory.RATE_LIMIT
        assert record.cooldown_until == clock.now + COOLDOWN_CONFIG.rate_limit
        assert record.status == "cooldown"
        assert record.failure_count == 1
        assert record.last_failure_at == clock.now

    def test_rate_limit_is_available_after_expiry(self):
        store, clock = make_store()
        store.record_failure("gpt_6_luna", FailureCategory.RATE_LIMIT)

        assert store.is_available("gpt_6_luna") is False
        clock.advance(COOLDOWN_CONFIG.rate_limit - 0.001)
        assert store.is_available("gpt_6_luna") is False
        clock.advance(0.001)
        assert store.is_available("gpt_6_luna") is True

    def test_retry_after_is_respected(self):
        store, clock = make_store()
        record = store.record_failure(
            "gpt_6_luna",
            FailureCategory.RATE_LIMIT,
            retry_after=30.0,
        )
        assert record.cooldown_until == clock.now + 30.0
        assert record.retry_after == 30.0

        clock.advance(29.999)
        assert store.is_available("gpt_6_luna") is False
        clock.advance(0.001)
        assert store.is_available("gpt_6_luna") is True

    def test_retry_after_is_capped_at_safety_bound(self):
        store, clock = make_store()
        record = store.record_failure(
            "gpt_6_luna",
            FailureCategory.RATE_LIMIT,
            retry_after=10_000_000.0,
        )
        assert record.cooldown_until == clock.now + COOLDOWN_CONFIG.retry_after_cap

    def test_retry_after_shorter_than_category_cooldown_wins(self):
        store, clock = make_store()
        record = store.record_failure(
            "gpt_6_luna",
            FailureCategory.QUOTA,
            retry_after=5.0,
        )
        assert record.cooldown_until == clock.now + 5.0

    def test_quota_uses_medium_cooldown(self):
        store, clock = make_store()
        record = store.record_failure("claude_sonnet_5", FailureCategory.QUOTA)

        assert record.cooldown_until == clock.now + COOLDOWN_CONFIG.quota
        assert COOLDOWN_CONFIG.quota > COOLDOWN_CONFIG.rate_limit

    def test_auth_and_capacity_use_configured_cooldowns(self):
        store, clock = make_store()
        auth = store.record_failure("claude_opus_5", FailureCategory.AUTH)
        assert auth.cooldown_until == clock.now + COOLDOWN_CONFIG.auth

        capacity = store.record_failure(
            "claude_opus_5", FailureCategory.CAPACITY
        )
        assert capacity.cooldown_until == clock.now + COOLDOWN_CONFIG.capacity

    def test_unknown_failure_recovers_after_conservative_cooldown(self):
        store, clock = make_store()
        store.record_failure(
            "splash_local", FailureCategory.PROVIDER_UNKNOWN
        )

        clock.advance(COOLDOWN_CONFIG.unknown - 0.001)
        assert store.is_available("splash_local") is False
        clock.advance(0.001)
        assert store.is_available("splash_local") is True

    def test_repeated_failures_extend_cooldown_without_permanent_disable(self):
        store, clock = make_store()
        store.record_failure("gpt_6_luna", FailureCategory.RATE_LIMIT)
        first_until = store.snapshot()["gpt_6_luna"].cooldown_until

        clock.advance(1.0)
        store.record_failure("gpt_6_luna", FailureCategory.RATE_LIMIT)
        record = store.snapshot()["gpt_6_luna"]

        assert record.cooldown_until > first_until
        assert record.failure_count == 2
        # Still just a cooldown — expiry always restores eligibility.
        clock.advance(COOLDOWN_CONFIG.rate_limit)
        assert store.is_available("gpt_6_luna") is True


class TestCooldownBoundary:
    def test_subscription_is_excluded_before_cooldown_boundary(self):
        store, clock = make_store()
        store.record_failure("claude_sonnet_5", FailureCategory.QUOTA)

        clock.advance(COOLDOWN_CONFIG.quota - 0.001)
        assert store.is_available("claude_sonnet_5") is False

    def test_subscription_is_eligible_at_cooldown_boundary(self):
        store, clock = make_store()
        store.record_failure("claude_sonnet_5", FailureCategory.QUOTA)

        clock.advance(COOLDOWN_CONFIG.quota)
        assert store.is_available("claude_sonnet_5") is True

    def test_unknown_profile_is_available(self):
        store, _ = make_store()
        assert store.is_available("never_recorded") is True

    def test_expired_record_becomes_eligible_without_background_timer(self):
        store, clock = make_store()
        store.record_failure("gpt_6_luna", FailureCategory.RATE_LIMIT)

        # No expiry sweep, no timer call: eligibility is computed lazily.
        clock.advance(COOLDOWN_CONFIG.rate_limit)
        assert store.is_available("gpt_6_luna") is True
        assert store.snapshot()["gpt_6_luna"].status == "available"


class TestFailureCategorySeparation:
    def test_task_failure_does_not_mark_provider_unavailable(self):
        store, clock = make_store()
        record = store.record_failure(
            "local_coding", FailureCategory.TASK_FAILURE
        )

        assert record.failure_count == 1
        assert record.cooldown_until is None
        assert record.status == "available"
        assert store.is_available("local_coding") is True

    def test_local_tool_failure_does_not_mark_provider_unavailable(self):
        store, _ = make_store()
        record = store.record_failure(
            "local_coding", FailureCategory.LOCAL_TOOL_FAILURE
        )

        assert record.cooldown_until is None
        assert record.status == "available"
        assert store.is_available("local_coding") is True

    def test_task_failure_does_not_lift_active_provider_cooldown(self):
        store, clock = make_store()
        store.record_failure("gpt_6_luna", FailureCategory.RATE_LIMIT)
        cooldown_until = store.snapshot()["gpt_6_luna"].cooldown_until

        clock.advance(1.0)
        store.record_failure("gpt_6_luna", FailureCategory.TASK_FAILURE)

        record = store.snapshot()["gpt_6_luna"]
        assert record.cooldown_until == cooldown_until
        assert record.status == "cooldown"
        assert store.is_available("gpt_6_luna") is False

    def test_task_failure_then_provider_failure_only_cooldowns_provider(self):
        store, clock = make_store()
        store.record_failure("local_coding", FailureCategory.TASK_FAILURE)
        assert store.is_available("local_coding") is True

        store.record_failure("local_coding", FailureCategory.CAPACITY)
        assert store.is_available("local_coding") is False

        # Provider cooldown expires; the task failure never did anything.
        clock.advance(COOLDOWN_CONFIG.capacity)
        assert store.is_available("local_coding") is True


class TestRecovery:
    def test_success_clears_transient_failure_count(self):
        store, _ = make_store()
        store.record_failure("gpt_6_luna", FailureCategory.RATE_LIMIT)
        store.record_failure("gpt_6_luna", FailureCategory.RATE_LIMIT)

        record = store.record_success("gpt_6_luna")
        assert record.failure_count == 0
        assert record.status == "available"
        assert record.cooldown_until is None
        assert store.is_available("gpt_6_luna") is True

    def test_success_restores_profile_mid_cooldown(self):
        store, clock = make_store()
        store.record_failure("gpt_6_luna", FailureCategory.QUOTA)
        assert store.is_available("gpt_6_luna") is False

        clock.advance(1.0)  # still well inside the cooldown
        store.record_success("gpt_6_luna")
        assert store.is_available("gpt_6_luna") is True

    def test_snapshot_exposes_all_records(self):
        store, _ = make_store()
        store.record_failure("gpt_6_luna", FailureCategory.RATE_LIMIT)
        store.record_success("claude_sonnet_5")

        snap = store.snapshot()
        assert set(snap) == {"gpt_6_luna", "claude_sonnet_5"}
        assert snap["gpt_6_luna"].reason == FailureCategory.RATE_LIMIT
        assert snap["claude_sonnet_5"].status == "available"

    def test_records_serialize_to_plain_values(self):
        store, _ = make_store()
        store.record_failure(
            "gpt_6_luna", FailureCategory.RATE_LIMIT, retry_after=30.0
        )
        data = store.snapshot()["gpt_6_luna"].to_dict()

        assert data["profile_id"] == "gpt_6_luna"
        assert data["reason"] == "rate_limit"
        assert isinstance(data["cooldown_until"], float)
