"""
In-memory model/provider availability, failure categories, and cooldowns.

Binding decisions pinned here (see the simplified-turn-aware-router spec,
section 12, and the plan's review-focus ruling 3):

* A profile is excluded *strictly before* its ``cooldown_until`` and
  becomes eligible *at or after* it — eligibility is exactly at the
  boundary.
* Cooldown expiry is computed lazily at read time; there is no
  background timer and no reaper thread.
* No single failure — of any category — permanently disables a
  profile.  Every cooldown is a finite duration.
* Task-quality failures and local tool failures are *separate* failure
  categories: they never create a provider cooldown, because the
  provider was available; the work failed.  Conversely, provider
  availability failures never raise task capability — that separation
  lives in :mod:`router.escalation` and the deterministic classifier's
  ``task_failures`` counter.
* An explicit provider ``Retry-After`` is honored, clamped to a
  configurable safety bound.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from router.config import CooldownConfig, COOLDOWN_CONFIG

__all__ = [
    "FailureCategory",
    "AvailabilityRecord",
    "AvailabilityStore",
]


# ---------------------------------------------------------------------------
# Failure categories
# ---------------------------------------------------------------------------

class FailureCategory(str, Enum):
    """Why an attempt failed.

    Availability categories (rate limit … provider unknown) describe the
    *provider* being unwilling or unable to serve a request right now and
    therefore justify a cooldown.  Task-quality and local-tool failures
    describe the *work* or the local environment and never cool a
    provider down.
    """

    RATE_LIMIT = "rate_limit"
    QUOTA = "quota"
    AUTH = "auth"
    CAPACITY = "capacity"
    PROVIDER_UNKNOWN = "provider_unknown"
    TASK_FAILURE = "task_failure"
    LOCAL_TOOL_FAILURE = "local_tool_failure"


#: Categories that justify a provider cooldown.
AVAILABILITY_CATEGORIES: frozenset[FailureCategory] = frozenset({
    FailureCategory.RATE_LIMIT,
    FailureCategory.QUOTA,
    FailureCategory.AUTH,
    FailureCategory.CAPACITY,
    FailureCategory.PROVIDER_UNKNOWN,
})

#: Categories that never create a provider cooldown.
TASK_SIDE_CATEGORIES: frozenset[FailureCategory] = frozenset({
    FailureCategory.TASK_FAILURE,
    FailureCategory.LOCAL_TOOL_FAILURE,
})


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class AvailabilityRecord:
    """Mutable per-profile availability state (spec section 12)."""

    profile_id: str
    status: str = "available"          # "available" | "cooldown"
    cooldown_until: float | None = None
    reason: FailureCategory | None = None
    failure_count: int = 0
    last_failure_at: float | None = None
    retry_after: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "status": self.status,
            "cooldown_until": self.cooldown_until,
            "reason": self.reason.value if self.reason else None,
            "failure_count": self.failure_count,
            "last_failure_at": self.last_failure_at,
            "retry_after": self.retry_after,
        }


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class AvailabilityStore:
    """In-memory availability state for model profiles.

    The clock is injected (default: :func:`time.time`) so tests pin the
    cooldown boundary deterministically.  All state is ephemeral: nothing
    is persisted across restarts by design.
    """

    def __init__(
        self,
        cooldowns: CooldownConfig = COOLDOWN_CONFIG,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.cooldowns = cooldowns
        self._clock = clock
        self._records: dict[str, AvailabilityRecord] = {}

    # -- internal helpers --------------------------------------------------

    def _record(self, profile_id: str) -> AvailabilityRecord:
        record = self._records.get(profile_id)
        if record is None:
            record = AvailabilityRecord(profile_id=profile_id)
            self._records[profile_id] = record
        return record

    def _cooldown_duration(
        self,
        category: FailureCategory,
        retry_after: float | None,
    ) -> float | None:
        """Cooldown seconds for a category, honoring an explicit Retry-After.

        Returns ``None`` when the category must not cool the provider
        down (task-side failures).  An explicit ``retry_after`` wins over
        the category default but is clamped to ``[0, retry_after_cap]``.
        """
        if category in TASK_SIDE_CATEGORIES:
            return None

        defaults = {
            FailureCategory.RATE_LIMIT: self.cooldowns.rate_limit,
            FailureCategory.QUOTA: self.cooldowns.quota,
            FailureCategory.AUTH: self.cooldowns.auth,
            FailureCategory.CAPACITY: self.cooldowns.capacity,
            FailureCategory.PROVIDER_UNKNOWN: self.cooldowns.unknown,
        }
        duration = defaults[category]
        if retry_after is not None:
            duration = min(
                max(float(retry_after), 0.0),
                self.cooldowns.retry_after_cap,
            )
        return duration

    # -- public API ----------------------------------------------------------

    def record_failure(
        self,
        profile_id: str,
        category: FailureCategory,
        retry_after: float | None = None,
    ) -> AvailabilityRecord:
        """Record a failure for ``profile_id`` and return its record.

        Task-side categories are recorded (failure count, timestamps) but
        never set a cooldown: the provider stays available.  Availability
        categories start (or extend) a cooldown.  No category disables the
        profile permanently.
        """
        record = self._record(profile_id)
        now = self._clock()

        record.failure_count += 1
        record.last_failure_at = now

        duration = self._cooldown_duration(category, retry_after)
        if duration is None:
            # Task-side failure: cooldown/availability state is untouched —
            # in particular a task failure must never lift (or set) a
            # provider cooldown on the same profile.
            return record

        record.reason = category
        record.retry_after = retry_after
        record.cooldown_until = now + duration
        record.status = "cooldown"
        return record

    def record_success(self, profile_id: str) -> AvailabilityRecord:
        """Record a success: clear any cooldown and transient failure count."""
        record = self._record(profile_id)
        record.status = "available"
        record.cooldown_until = None
        record.reason = None
        record.failure_count = 0
        record.last_failure_at = None
        record.retry_after = None
        return record

    def is_available(self, profile_id: str) -> bool:
        """Eligibility check, computed lazily (no background timer).

        A profile with no record is available.  A record is unavailable
        strictly before its ``cooldown_until`` and eligible at or after
        it.
        """
        record = self._records.get(profile_id)
        if record is None:
            return True
        if record.cooldown_until is None:
            return True
        if self._clock() >= record.cooldown_until:
            # Lazy expiry: eligible again without any timer involvement.
            record.status = "available"
            return True
        return False

    def snapshot(self) -> dict[str, AvailabilityRecord]:
        """Point-in-time view of all records (for telemetry / debug APIs)."""
        # Lazily expire before reporting so status never lies about the
        # clock, even though expiry is read-driven.
        for record in self._records.values():
            if (
                record.status == "cooldown"
                and record.cooldown_until is not None
                and self._clock() >= record.cooldown_until
            ):
                record.status = "available"
        return dict(self._records)
