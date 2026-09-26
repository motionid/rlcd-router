"""
Structured telemetry for routing decisions and outcomes.

Task 8 record shape: one routed turn produces exactly ONE finalized
JSONL record.  The lifecycle is:

1. ``start_event`` opens the route's record as a *pending* event held in
   memory (bounded, oldest evicted) — nothing is written to disk.
2. Candidate executions are appended to the event's ``attempts`` array
   as outcomes are reported (``report_outcome`` / ``add_attempt``).
3. The record *settles* on the first success or on candidate exhaustion
   (caller flags the last eligible candidate failed); settlement writes
   exactly one finalized JSONL line and moves the event out of the
   pending index.  Repeated outcome reports are idempotent by
   route/attempt identity: they never append a duplicate record and
   never double-count aggregates.

Aggregate metrics (``TelemetryLog.summary``) reduce finalized records
only.  Telemetry is a feedback channel, never a routing dependency: the
API layer swallows every telemetry exception, and outcome/availability
handling stays telemetry-independent (see ``server/app.py``).

Privacy: events carry a prompt *hash* and bounded routing state only —
never prompt text.
"""

from __future__ import annotations

import json
import time
import uuid
from collections import OrderedDict, Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


# Bound on simultaneously pending (unsettled) route events.  Routes whose
# outcome is never reported are dropped oldest-first, silently: telemetry
# must not become a memory leak in a long-lived router process.
ACTIVE_EVENT_MAX = 1024


@dataclass
class AttemptEvent:
    """One candidate execution attempt inside a route event.

    Identity for idempotent settlement is ``(profile_id, status)``: the
    same candidate reporting the same status twice is a duplicate, not a
    second attempt.
    """

    attempt_index: int = 0
    profile_id: str = ""
    model: str = ""                 # model id the host actually ran
    access_class: str = ""          # "local" | "subscription" | metered…
    status: str = "pending"        # pending | success | failed
    latency_ms: float = 0.0
    cost: float = 0.0
    failure_category: str = ""     # FailureCategory value, when classified
    failure_reason: str = ""
    verification: str = ""
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def identity(self) -> tuple[str, str]:
        """Idempotency key: which candidate reported which status."""
        return (self.profile_id, self.status)


@dataclass
class RoutingEvent:
    """One complete routing decision plus its execution attempts.

    The route fields are filled when the route is served; the execution
    fields are reduced from the appended attempts at settlement.
    """

    # --- identity ---
    event_id: str = ""
    timestamp: float = 0.0

    # --- input (hash/bounded state only, never prompt text) ---
    prompt_hash: str = ""
    prompt_chars: int = 0
    estimated_tokens: int = 0

    # --- classification ---
    classifier_used: str = ""
    classification: dict[str, str] = field(default_factory=dict)
    field_confidence: dict[str, float] = field(default_factory=dict)
    overall_confidence: float = 0.0
    critical_confidence: float = 0.0
    abstain: bool = False

    # --- decision ---
    selected_tier: str = ""
    selected_profile: str = ""      # top candidate profile id
    access_class: str = ""          # top candidate access class
    eligible_models: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    requirements: dict[str, Any] = field(default_factory=dict)
    paid_selection_reasons: list[str] = field(default_factory=list)

    # --- optional Ollaya evidence (invocation/failure analysis) ---
    ollaya_invoked: bool = False
    ollaya_outcome: str = "not_consulted"  # not_consulted|classified|error_fallback
    ollaya_confidence: float = 0.0
    ollaya_latency_ms: float = 0.0

    # --- execution ---
    actual_model: str = ""
    status: str = "pending"          # pending | success | failed | escalated
    verification: str = ""
    latency_ms: float = 0.0
    classifier_latency_ms: float = 0.0
    cost: float = 0.0
    escalated: bool = False
    escalation_chain: list[str] = field(default_factory=list)
    failure_reason: str = ""
    attempts: list[AttemptEvent] = field(default_factory=list)

    # --- optional cost estimation (caller-supplied estimates) ---
    estimated_spend_incurred: float = 0.0
    estimated_spend_avoided: float = 0.0

    # --- lifecycle ---
    finalized: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TelemetryLog:
    """Append-only JSONL log of *finalized* routing events.

    Each event is one line containing the route decision and its
    attempts array.  Pending events live in memory only.  The log is
    designed for offline replay and batch analysis, not real-time
    querying.
    """

    def __init__(self, log_path: Optional[str] = None):
        self._log_path = Path(log_path) if log_path else None
        self._events: list[RoutingEvent] = []      # finalized only
        self._finalized_ids: set[str] = set()
        self._active: "OrderedDict[str, RoutingEvent]" = OrderedDict()
        self._active_max = ACTIVE_EVENT_MAX

    # ------------------------------------------------------------------
    # Pending route lifecycle
    # ------------------------------------------------------------------

    def start_event(self, **kwargs: Any) -> RoutingEvent:
        """Open one pending route event (in memory, nothing on disk).

        Raises ``ValueError`` on a duplicate event id — callers swallow
        it (telemetry must never fail a route).
        """
        kwargs.setdefault("event_id", uuid.uuid4().hex[:16])
        kwargs.setdefault("timestamp", time.time())
        kwargs.setdefault("status", "pending")
        event = RoutingEvent(**kwargs)
        event_id = event.event_id
        if event_id and (
            event_id in self._active or event_id in self._finalized_ids
        ):
            raise ValueError(f"route event {event_id!r} already exists")
        self._active[event_id] = event
        while len(self._active) > self._active_max:
            self._active.popitem(last=False)  # drop oldest pending event
        return event

    def get_event(self, event_id: str) -> Optional[RoutingEvent]:
        """Find a route event, pending or finalized."""
        if event_id in self._active:
            return self._active[event_id]
        if event_id and event_id in self._finalized_ids:
            for event in self._events:
                if event.event_id == event_id:
                    return event
        return None

    def add_attempt(self, event_id: str, attempt: AttemptEvent) -> bool:
        """Append one attempt unless the same identity was recorded.

        Idempotent by (profile_id, status): a repeated outcome report for
        the same candidate and status is a duplicate, not a new attempt.
        """
        event = self.get_event(event_id)
        if event is None or event.finalized:
            return False
        for existing in event.attempts:
            if existing.identity == attempt.identity:
                return False
        attempt.attempt_index = len(event.attempts)
        event.attempts.append(attempt)
        return True

    def finalize_event(
        self, event_id: str, **updates: Any
    ) -> Optional[RoutingEvent]:
        """Settle a pending event: write exactly one finalized record.

        Idempotent: finalizing an unknown or already-finalized event is a
        no-op (never a second JSONL line).
        """
        event = self._active.pop(event_id, None)
        if event is None:
            return None
        for key, value in updates.items():
            if hasattr(event, key):
                setattr(event, key, value)
        if len(event.attempts) > 1:
            event.escalated = True
            event.escalation_chain = [a.profile_id for a in event.attempts]
        event.finalized = True
        self._append_finalized(event)
        return event

    def report_outcome(
        self,
        event_id: str,
        profile_id: str,
        status: str,
        *,
        failure_category: str = "",
        failure_reason: str = "",
        verification: str = "",
        actual_model: str = "",
        latency_ms: float = 0.0,
        cost: float = 0.0,
        estimated_spend_incurred: float = 0.0,
        exhausted: bool = False,
    ) -> Optional[RoutingEvent]:
        """Record one candidate outcome on the route's single record.

        - Appends the attempt (idempotent by candidate/status identity).
        - Finalizes on settlement: first success, or failure flagged as
          candidate exhaustion.
        - No-ops for unknown or already-settled routes.
        """
        event = self.get_event(event_id)
        if event is None or event.finalized:
            return event  # unknown route or late report: idempotent no-op

        added = self.add_attempt(event_id, AttemptEvent(
            profile_id=profile_id,
            model=actual_model,
            access_class=event.access_class,
            status=status,
            latency_ms=latency_ms,
            cost=cost,
            failure_category=failure_category,
            failure_reason=failure_reason,
            verification=verification,
            timestamp=time.time(),
        ))
        if not added:
            return event  # duplicate report: never double-count

        event.latency_ms += latency_ms
        event.cost += cost
        event.estimated_spend_incurred += estimated_spend_incurred
        if actual_model:
            event.actual_model = actual_model
        if status == "failed":
            event.failure_reason = failure_reason

        if status == "success":
            return self.finalize_event(
                event_id, status="success", verification=verification,
            )
        if exhausted:
            return self.finalize_event(event_id, status="failed")
        return event

    # ------------------------------------------------------------------
    # Legacy record path (route-and-execute): finalize-and-append
    # ------------------------------------------------------------------

    def record(self, event: RoutingEvent) -> None:
        """Append one finalized event (legacy compatibility path).

        Idempotent for non-empty event ids: re-recording a settled event
        replaces it in memory and never writes a second JSONL line.
        """
        event_id = event.event_id
        if event_id and event_id in self._active:
            self._active.pop(event_id)
        if event_id and event_id in self._finalized_ids:
            for i, existing in enumerate(self._events):
                if existing.event_id == event_id:
                    self._events[i] = event
                    return
        event.finalized = True
        self._append_finalized(event)

    def new_event(self, **kwargs: Any) -> RoutingEvent:
        """Create a new unfinalized event with defaults filled in.

        Legacy factory for the route-and-execute path; the event is NOT
        registered as pending — the caller passes it to ``record``.
        """
        kwargs.setdefault("event_id", uuid.uuid4().hex[:16])
        kwargs.setdefault("timestamp", time.time())
        return RoutingEvent(**kwargs)

    def _append_finalized(self, event: RoutingEvent) -> None:
        self._events.append(event)
        if event.event_id:
            self._finalized_ids.add(event.event_id)
        if self._log_path is not None:
            try:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._log_path, "a") as f:
                    f.write(json.dumps(event.to_dict(), default=str) + "\n")
            except OSError:
                # Disk telemetry is best-effort; the in-memory record
                # still counts.  Never raise into the routing path.
                pass

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @property
    def events(self) -> list[RoutingEvent]:
        """Finalized events only (pending ones are not yet records)."""
        return list(self._events)

    def filter_by(self, **kwargs: Any) -> list[RoutingEvent]:
        """Return finalized events matching all field=value constraints."""
        result = []
        for ev in self._events:
            d = ev.to_dict()
            if all(d.get(k) == v for k, v in kwargs.items()):
                result.append(ev)
        return result

    # ------------------------------------------------------------------
    # Aggregate metrics (finalized records only)
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Aggregate metrics across finalized records."""
        events = self._events
        if not events:
            return self._empty_summary()

        total = len(events)
        successes = sum(1 for e in events if e.status == "success")
        failures = sum(1 for e in events if e.status == "failed")
        escalated = sum(1 for e in events if e.escalated)
        abstained = sum(1 for e in events if e.abstain)

        tier_counts: dict[str, int] = {}
        classifier_counts: dict[str, int] = {}
        access_counts: dict[str, int] = {}
        model_counts: dict[str, int] = {}
        paid_reason_counts: dict[str, int] = {}
        failure_category_counts: dict[str, int] = {}

        total_latency = 0.0
        total_cost = 0.0
        total_cls_latency = 0.0
        spend_incurred = 0.0
        spend_avoided = 0.0
        ollaya_latencies: list[float] = []
        ollaya_confidences: list[float] = []
        ollaya_error_fallbacks = 0
        ollaya_count = 0
        local_events = 0
        local_failures = 0
        local_escalations = 0

        for ev in events:
            tier_counts[ev.selected_tier] = (
                tier_counts.get(ev.selected_tier, 0) + 1
            )
            classifier_counts[ev.classifier_used] = (
                classifier_counts.get(ev.classifier_used, 0) + 1
            )
            access = ev.access_class or "unknown"
            access_counts[access] = access_counts.get(access, 0) + 1
            model = ev.selected_profile or ev.actual_model or "unknown"
            model_counts[model] = model_counts.get(model, 0) + 1
            for reason in ev.paid_selection_reasons:
                paid_reason_counts[reason] = (
                    paid_reason_counts.get(reason, 0) + 1
                )
            for attempt in ev.attempts:
                if attempt.failure_category:
                    failure_category_counts[attempt.failure_category] = (
                        failure_category_counts.get(attempt.failure_category, 0) + 1
                    )
            total_latency += ev.latency_ms
            total_cost += ev.cost
            total_cls_latency += ev.classifier_latency_ms
            spend_incurred += ev.estimated_spend_incurred
            spend_avoided += ev.estimated_spend_avoided

            if ev.ollaya_invoked or ev.classifier_used == "ollaya":
                ollaya_count += 1
                if ev.ollaya_outcome == "error_fallback":
                    ollaya_error_fallbacks += 1
                if ev.ollaya_latency_ms:
                    ollaya_latencies.append(ev.ollaya_latency_ms)
                if ev.ollaya_confidence:
                    ollaya_confidences.append(ev.ollaya_confidence)

            if access == "local":
                local_events += 1
                if ev.status == "failed":
                    local_failures += 1
                if ev.escalated:
                    local_escalations += 1

        return {
            "total": total,
            "success_rate": round(successes / total, 4),
            "failure_rate": round(failures / total, 4),
            "escalation_rate": round(escalated / total, 4),
            "abstention_rate": round(abstained / total, 4),
            "tier_distribution": tier_counts,
            "classifier_distribution": classifier_counts,
            "avg_latency_ms": round(total_latency / total, 2),
            "avg_classifier_latency_ms": round(total_cls_latency / total, 2),
            "total_cost": round(total_cost, 6),
            "avg_cost": round(total_cost / total, 6),
            # --- Task 8 metrics ---
            "access_distribution": access_counts,
            "model_distribution": model_counts,
            "paid_selection_reasons": paid_reason_counts,
            "deterministic_count": classifier_counts.get("deterministic", 0),
            "ollaya_count": ollaya_count,
            "ollaya_error_fallback_count": ollaya_error_fallbacks,
            "avg_ollaya_latency_ms": (
                round(sum(ollaya_latencies) / len(ollaya_latencies), 2)
                if ollaya_latencies else 0.0
            ),
            "avg_ollaya_confidence": (
                round(sum(ollaya_confidences) / len(ollaya_confidences), 4)
                if ollaya_confidences else 0.0
            ),
            "local_event_count": local_events,
            "local_failure_rate": (
                round(local_failures / local_events, 4) if local_events else 0.0
            ),
            "local_escalation_rate": (
                round(local_escalations / local_events, 4) if local_events else 0.0
            ),
            "failure_categories": failure_category_counts,
            "estimated_spend_incurred": round(spend_incurred, 6),
            "estimated_spend_avoided": round(spend_avoided, 6),
        }

    def _empty_summary(self) -> dict[str, Any]:
        return {
            "total": 0,
            "success_rate": 0.0,
            "failure_rate": 0.0,
            "escalation_rate": 0.0,
            "abstention_rate": 0.0,
            "tier_distribution": {},
            "classifier_distribution": {},
            "avg_latency_ms": 0.0,
            "avg_classifier_latency_ms": 0.0,
            "total_cost": 0.0,
            "avg_cost": 0.0,
            "access_distribution": {},
            "model_distribution": {},
            "paid_selection_reasons": {},
            "deterministic_count": 0,
            "ollaya_count": 0,
            "ollaya_error_fallback_count": 0,
            "avg_ollaya_latency_ms": 0.0,
            "avg_ollaya_confidence": 0.0,
            "local_event_count": 0,
            "local_failure_rate": 0.0,
            "local_escalation_rate": 0.0,
            "failure_categories": {},
            "estimated_spend_incurred": 0.0,
            "estimated_spend_avoided": 0.0,
        }

    def under_routing_events(self) -> list[RoutingEvent]:
        """Events where the selected tier failed but a stronger tier
        might have succeeded (escalated after failure)."""
        return [
            ev for ev in self._events
            if ev.escalated and ev.failure_reason != ""
        ]

    def over_routing_events(self) -> list[RoutingEvent]:
        """Events routed to a stronger tier where the task was trivial.

        Heuristic: success on first attempt at a cloud tier for a
        task classified as low complexity.
        """
        return [
            ev for ev in self._events
            if (
                ev.status == "success"
                and not ev.escalated
                and ev.selected_tier.startswith("cloud_")
                and ev.classification.get("complexity") in ("trivial", "low")
            )
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> list[RoutingEvent]:
        """Load finalized events from the JSONL file."""
        if self._log_path is None or not self._log_path.exists():
            return []

        events = []
        with open(self._log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                attempts = [
                    AttemptEvent(**attempt)
                    for attempt in data.pop("attempts", [])
                ]
                data.pop("finalized", None)
                events.append(RoutingEvent(**data, attempts=attempts))
        self._events = events
        self._finalized_ids = {
            e.event_id for e in events if e.event_id
        }
        return events
