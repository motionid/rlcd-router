"""
Escalation / fallback state machine.

Tracks what has been tried, detects concrete failure signals, and
produces the next escalation step.  Escalation is always *evidence-
driven* — the same model is never retried with the same prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from router.availability import FailureCategory
from router.config import TIER_DEFINITIONS
from router.decision import DEFAULT_ESCALATION


# ---------------------------------------------------------------------------
# Execution state (mutable, accumulates across attempts)
# ---------------------------------------------------------------------------

@dataclass
class ExecutionState:
    """Tracks the full history of attempts for one routing decision."""

    selected_tiers: list[str] = field(default_factory=list)
    attempt_results: list[dict[str, Any]] = field(default_factory=list)
    max_attempts: int = 3
    allow_cloud_fallback: bool = True
    elapsed_ms: float = 0.0
    total_cost: float = 0.0
    # Consecutive *task-quality* failures only.  Provider availability
    # failures (rate limits, quota, outages) advance the candidate cursor
    # but must never inflate this counter — availability failures must not
    # raise task capability.
    task_failure_count: int = 0

    @property
    def attempts(self) -> int:
        return len(self.selected_tiers)

    @property
    def last_tier(self) -> Optional[str]:
        return self.selected_tiers[-1] if self.selected_tiers else None

    @property
    def exhausted(self) -> bool:
        return self.attempts >= self.max_attempts

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_tiers": list(self.selected_tiers),
            "attempt_results": list(self.attempt_results),
            "max_attempts": self.max_attempts,
            "allow_cloud_fallback": self.allow_cloud_fallback,
            "attempts": self.attempts,
            "exhausted": self.exhausted,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "total_cost": round(self.total_cost, 6),
            "task_failure_count": self.task_failure_count,
        }


# ---------------------------------------------------------------------------
# Failure detection
# ---------------------------------------------------------------------------

FAILURE_REASONS = frozenset({
    "tests_failed",
    "patch_not_applied",
    "invalid_output",
    "tool_error",
    "no_progress",
    "model_uncertain",
    "timeout",
    "schema_violation",
    "semantic_error",
    "scope_exceeded",
    "subscription_exhausted",
})


def should_escalate(
    state: ExecutionState,
    result: dict[str, Any],
) -> tuple[bool, str]:
    """Decide whether the latest attempt warrants escalation.

    Returns ``(should_escalate, reason)``.  ``reason`` is one of
    ``FAILURE_REASONS`` or ``"success"``.
    """
    if state.exhausted:
        return False, "max_attempts_exhausted"

    status = result.get("status", "unknown")

    if status == "success":
        return False, "success"

    reason = result.get("failure_reason", "unknown")
    if reason in FAILURE_REASONS:
        return True, reason

    # Generic failure without a specific reason
    if status == "failed":
        return True, "generic_failure"

    return False, "success"


# ---------------------------------------------------------------------------
# Escalation step
# ---------------------------------------------------------------------------

def escalate(
    current_tier: str,
    state: ExecutionState,
    *,
    force_local: bool = False,
) -> Optional[str]:
    """Return the next tier to try, or ``None`` if the chain is exhausted.

    The escalation chain is:
        local_fast → local_general → cloud_standard
        local_coding → cloud_coding
        local_reasoning → cloud_reasoning → cloud_research

    If ``force_local`` is True (e.g. privacy constraint), the chain
    stops at ``local_reasoning``.
    """
    esc = DEFAULT_ESCALATION.get(current_tier, {})
    next_tier = esc.get("next_tier")

    if next_tier is None:
        return None

    if force_local:
        td = TIER_DEFINITIONS.get(next_tier)
        if td and not td.is_local:
            return None

    if not state.allow_cloud_fallback:
        td = TIER_DEFINITIONS.get(next_tier)
        if td and not td.is_local:
            return None

    return next_tier


# ---------------------------------------------------------------------------
# Convenience: record an attempt
# ---------------------------------------------------------------------------

def record_attempt(
    state: ExecutionState,
    tier: str,
    result: dict[str, Any],
    *,
    latency_ms: float = 0.0,
    cost: float = 0.0,
) -> ExecutionState:
    """Append an attempt and return the mutated state.

    Task-quality failures increment ``state.task_failure_count`` (the
    counter that may later raise required capability).  Availability and
    local-tool failures do not: they advance the candidate/tier chain
    without ever blaming the task.
    """
    state.selected_tiers.append(tier)
    state.attempt_results.append(result)
    state.elapsed_ms += latency_ms
    state.total_cost += cost

    category = outcome_category(result)
    if category is FailureCategory.TASK_FAILURE:
        state.task_failure_count += 1

    return state


# ---------------------------------------------------------------------------
# Failure-category classification of provider outcomes
# ---------------------------------------------------------------------------

# Legacy/internal reason codes mapped to availability failure categories.
# Unlisted failed outcomes fall back to PROVIDER_UNKNOWN (conservative
# availability handling, never a task-capability increase).
_AVAILABILITY_REASON_CATEGORY: dict[str, FailureCategory] = {
    "subscription_exhausted": FailureCategory.QUOTA,
    "rate_limited": FailureCategory.RATE_LIMIT,
    "auth_failed": FailureCategory.AUTH,
    "timeout": FailureCategory.CAPACITY,
}

# Task-quality reasons: the provider served the request; the work failed.
_TASK_REASON_CATEGORY: dict[str, FailureCategory] = {
    "tests_failed": FailureCategory.TASK_FAILURE,
    "patch_not_applied": FailureCategory.TASK_FAILURE,
    "invalid_output": FailureCategory.TASK_FAILURE,
    "schema_violation": FailureCategory.TASK_FAILURE,
    "semantic_error": FailureCategory.TASK_FAILURE,
    "scope_exceeded": FailureCategory.TASK_FAILURE,
    "no_progress": FailureCategory.TASK_FAILURE,
    "model_uncertain": FailureCategory.TASK_FAILURE,
    "tool_error": FailureCategory.LOCAL_TOOL_FAILURE,
}

FAILURE_REASON_CATEGORY: dict[str, FailureCategory] = {
    **_AVAILABILITY_REASON_CATEGORY,
    **_TASK_REASON_CATEGORY,
}


def outcome_category(result: dict[str, Any]) -> FailureCategory | None:
    """Classify a provider outcome into a :class:`FailureCategory`.

    Returns ``None`` for successful outcomes.  An explicit
    ``failure_category`` in the outcome wins (the caller — e.g. Task 7's
    outcome API — may carry richer provider metadata); otherwise the
    reason code is mapped, and unknown failed outcomes map to
    ``PROVIDER_UNKNOWN`` so availability handles them conservatively
    without blaming the task.
    """
    if result.get("status") == "success":
        return None

    explicit = result.get("failure_category")
    if isinstance(explicit, FailureCategory):
        return explicit

    reason = result.get("failure_reason")
    return FAILURE_REASON_CATEGORY.get(reason, FailureCategory.PROVIDER_UNKNOWN)
