"""
Tests for the telemetry log.
"""

from __future__ import annotations

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.telemetry import RoutingEvent, TelemetryLog


def _make_event(**overrides) -> RoutingEvent:
    defaults = dict(
        classifier_used="rlcd_fast",
        selected_tier="local_coding",
        status="success",
        latency_ms=500.0,
        classifier_latency_ms=80.0,
        cost=0.0,
    )
    defaults.update(overrides)
    return RoutingEvent(**defaults)


class TestTelemetryLog:
    def test_record_and_summary(self):
        log = TelemetryLog()
        log.record(_make_event(status="success"))
        log.record(_make_event(status="failed"))
        log.record(_make_event(status="success", selected_tier="cloud_coding"))

        s = log.summary()
        assert s["total"] == 3
        assert abs(s["success_rate"] - 2 / 3) < 0.001
        assert abs(s["failure_rate"] - 1 / 3) < 0.001

    def test_empty_summary(self):
        log = TelemetryLog()
        assert log.summary()["total"] == 0

    def test_escalation_rate(self):
        log = TelemetryLog()
        log.record(_make_event(escalated=True))
        log.record(_make_event(escalated=False))
        s = log.summary()
        assert s["escalation_rate"] == 0.5

    def test_tier_distribution(self):
        log = TelemetryLog()
        log.record(_make_event(selected_tier="local_coding"))
        log.record(_make_event(selected_tier="local_coding"))
        log.record(_make_event(selected_tier="cloud_reasoning"))
        s = log.summary()
        assert s["tier_distribution"]["local_coding"] == 2
        assert s["tier_distribution"]["cloud_reasoning"] == 1

    def test_under_routing_events(self):
        log = TelemetryLog()
        log.record(_make_event(escalated=True, failure_reason="tests_failed"))
        log.record(_make_event(escalated=False))
        assert len(log.under_routing_events()) == 1

    def test_over_routing_events(self):
        log = TelemetryLog()
        log.record(_make_event(
            selected_tier="cloud_coding",
            status="success",
            classification={"complexity": "trivial"},
        ))
        log.record(_make_event(
            selected_tier="local_coding",
            status="success",
            classification={"complexity": "trivial"},
        ))
        over = log.over_routing_events()
        assert len(over) == 1
        assert over[0].selected_tier == "cloud_coding"

    def test_jsonl_persistence(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            # Write
            log1 = TelemetryLog(log_path=path)
            log1.record(_make_event(selected_tier="local_fast"))
            log1.record(_make_event(selected_tier="cloud_coding"))

            # Read back
            log2 = TelemetryLog(log_path=path)
            events = log2.load()
            assert len(events) == 2
            assert events[0].selected_tier == "local_fast"
            assert events[1].selected_tier == "cloud_coding"
        finally:
            os.unlink(path)

    def test_filter_by(self):
        log = TelemetryLog()
        log.record(_make_event(selected_tier="local_coding", status="success"))
        log.record(_make_event(selected_tier="cloud_coding", status="failed"))
        log.record(_make_event(selected_tier="local_coding", status="failed"))

        failures = log.filter_by(status="failed")
        assert len(failures) == 2

        local_successes = log.filter_by(selected_tier="local_coding", status="success")
        assert len(local_successes) == 1
