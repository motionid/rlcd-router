"""
Structured telemetry for routing decisions and outcomes.

Every routing decision produces a ``RoutingEvent`` that PI logs.
A ``TelemetryLog`` collects events for later replay and analysis.

This is the primary feedback channel for tuning the router:
  - under-routing detection
  - over-routing detection
  - confidence calibration
  - escalation frequency
  - cost attribution
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


@dataclass
class RoutingEvent:
    """One complete routing decision + execution outcome."""

    # --- identity ---
    event_id: str = ""
    timestamp: float = 0.0

    # --- input ---
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
    reasons: list[str] = field(default_factory=list)
    requirements: dict[str, Any] = field(default_factory=dict)

    # --- execution ---
    actual_model: str = ""
    status: str = "pending"          # pending | success | failed | escalated
    verification: str = ""           # e.g. "tests_passed", "patch_applied"
    latency_ms: float = 0.0
    classifier_latency_ms: float = 0.0
    cost: float = 0.0
    escalated: bool = False
    escalation_chain: list[str] = field(default_factory=list)
    failure_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TelemetryLog:
    """Append-only JSONL log of routing events.

    Each event is one line.  The log is designed for offline replay
    and batch analysis, not real-time querying.
    """

    def __init__(self, log_path: Optional[str] = None):
        self._log_path = Path(log_path) if log_path else None
        self._events: list[RoutingEvent] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, event: RoutingEvent) -> None:
        """Append an event to the in-memory list and optionally to disk."""
        self._events.append(event)

        if self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_path, "a") as f:
                f.write(json.dumps(event.to_dict(), default=str) + "\n")

    def new_event(self, **kwargs: Any) -> RoutingEvent:
        """Create a new event with defaults filled in."""
        kwargs.setdefault("event_id", uuid.uuid4().hex[:16])
        kwargs.setdefault("timestamp", time.time())
        return RoutingEvent(**kwargs)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @property
    def events(self) -> list[RoutingEvent]:
        return list(self._events)

    def filter_by(self, **kwargs: Any) -> list[RoutingEvent]:
        """Return events matching all given field=value constraints."""
        result = []
        for ev in self._events:
            d = ev.to_dict()
            if all(d.get(k) == v for k, v in kwargs.items()):
                result.append(ev)
        return result

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Compute aggregate metrics across all recorded events."""
        if not self._events:
            return {"total": 0}

        total = len(self._events)
        successes = sum(1 for e in self._events if e.status == "success")
        failures = sum(1 for e in self._events if e.status == "failed")
        escalated = sum(1 for e in self._events if e.escalated)
        abstained = sum(1 for e in self._events if e.abstain)

        tier_counts: dict[str, int] = {}
        classifier_counts: dict[str, int] = {}
        total_latency = 0.0
        total_cost = 0.0
        total_cls_latency = 0.0

        for ev in self._events:
            tier_counts[ev.selected_tier] = tier_counts.get(ev.selected_tier, 0) + 1
            classifier_counts[ev.classifier_used] = classifier_counts.get(ev.classifier_used, 0) + 1
            total_latency += ev.latency_ms
            total_cost += ev.cost
            total_cls_latency += ev.classifier_latency_ms

        return {
            "total": total,
            "success_rate": round(successes / total, 4) if total else 0.0,
            "failure_rate": round(failures / total, 4) if total else 0.0,
            "escalation_rate": round(escalated / total, 4) if total else 0.0,
            "abstention_rate": round(abstained / total, 4) if total else 0.0,
            "tier_distribution": tier_counts,
            "classifier_distribution": classifier_counts,
            "avg_latency_ms": round(total_latency / total, 2),
            "avg_classifier_latency_ms": round(total_cls_latency / total, 2),
            "total_cost": round(total_cost, 6),
            "avg_cost": round(total_cost / total, 6),
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
        """Load events from the JSONL file."""
        if self._log_path is None or not self._log_path.exists():
            return []

        events = []
        with open(self._log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    events.append(RoutingEvent(**data))
        self._events = events
        return events
