"""
Tests for the telemetry log and the labeled routing replay fixture.

Task 8 shape: one routed turn produces exactly ONE finalized JSONL
record with an appendable ``attempts`` array.  Pending events live in
memory only; settlement (first success or candidate exhaustion) writes
the single record.  Repeated outcome reports for the same route/attempt
identity are idempotent — never a duplicate record, never a
double-counted aggregate.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.telemetry import AttemptEvent, RoutingEvent, TelemetryLog

FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "routing_cases.json"
)


def _make_event(**overrides) -> RoutingEvent:
    """A finalized-style event for direct-record aggregate tests."""
    defaults = dict(
        event_id="",
        classifier_used="deterministic",
        selected_tier="local_coding",
        access_class="local",
        selected_profile="local_coding",
        status="success",
        latency_ms=500.0,
        classifier_latency_ms=0.5,
        cost=0.0,
    )
    defaults.update(overrides)
    return RoutingEvent(**defaults)


def _start(log: TelemetryLog, event_id: str, **overrides) -> str:
    """Open one pending route event with sane defaults."""
    defaults = dict(
        event_id=event_id,
        classifier_used="deterministic",
        selected_tier="cloud_standard",
        access_class="subscription",
        selected_profile="sub_normal_a",
        eligible_models=["sub_normal_a", "sub_normal_b"],
        overall_confidence=0.9,
    )
    defaults.update(overrides)
    log.start_event(**defaults)
    return event_id


def _jsonl_lines(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class TestSingleRecordLifecycle:
    """One route record, appendable attempts, finalize on settlement."""

    def test_pending_event_is_memory_only(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            log = TelemetryLog(log_path=path)
            _start(log, "r1")
            # Pending: no finalized record, not in aggregates.
            assert log.events == []
            assert log.summary()["total"] == 0
            assert _jsonl_lines(path) == []
        finally:
            os.unlink(path)

    def test_outcome_finalizes_exactly_one_record(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            log = TelemetryLog(log_path=path)
            _start(log, "r1")
            ev = log.report_outcome(
                "r1", profile_id="sub_normal_a", status="success",
                latency_ms=120.0, verification="tests_passed",
            )
            assert ev is not None
            assert ev.status == "success"
            assert len(ev.attempts) == 1
            assert ev.attempts[0].profile_id == "sub_normal_a"
            assert ev.attempts[0].status == "success"
            assert ev.attempts[0].verification == "tests_passed"

            # Exactly one finalized record, in memory and on disk.
            assert len(log.events) == 1
            assert log.summary()["total"] == 1
            lines = _jsonl_lines(path)
            assert len(lines) == 1
            assert lines[0]["event_id"] == "r1"
            assert len(lines[0]["attempts"]) == 1
        finally:
            os.unlink(path)

    def test_reporting_outcome_does_not_append_duplicate_event(self):
        """RED against the old duplicate mutable-event recording."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            log = TelemetryLog(log_path=path)
            _start(log, "r1")
            log.report_outcome(
                "r1", profile_id="sub_normal_a", status="success",
                latency_ms=100.0,
            )
            # PI retries the same outcome report (idempotency key:
            # route + attempt identity): no second record, no
            # double-counted latency.
            log.report_outcome(
                "r1", profile_id="sub_normal_a", status="success",
                latency_ms=100.0,
            )
            assert len(log.events) == 1
            assert len(log.events[0].attempts) == 1
            assert log.events[0].latency_ms == 100.0
            assert log.summary()["total"] == 1
            assert len(_jsonl_lines(path)) == 1
        finally:
            os.unlink(path)

    def test_report_after_settlement_is_idempotent_noop(self):
        log = TelemetryLog()
        _start(log, "r1")
        log.report_outcome("r1", profile_id="sub_normal_a", status="success")
        # A late/contradictory report must not resurrect or duplicate.
        ev = log.report_outcome(
            "r1", profile_id="sub_normal_b", status="failed",
        )
        assert ev is not None
        assert ev.status == "success"
        assert len(ev.attempts) == 1
        assert len(log.events) == 1

    def test_unknown_route_outcome_is_noop(self):
        log = TelemetryLog()
        assert log.report_outcome(
            "nope", profile_id="x", status="success"
        ) is None

    def test_multi_attempt_escalation_single_record(self):
        log = TelemetryLog()
        _start(log, "r1", eligible_models=["a", "b"])
        log.report_outcome(
            "r1", profile_id="a", status="failed",
            failure_category="rate_limit", latency_ms=50.0,
        )
        # Still pending (candidate "b" remains): memory only.
        assert log.events == []
        ev = log.report_outcome(
            "r1", profile_id="b", status="success", latency_ms=80.0,
        )
        assert ev.status == "success"
        assert len(ev.attempts) == 2
        assert [a.profile_id for a in ev.attempts] == ["a", "b"]
        assert ev.escalated is True
        assert ev.escalation_chain == ["a", "b"]
        assert len(log.events) == 1

    def test_candidate_exhaustion_finalizes(self):
        log = TelemetryLog()
        _start(log, "r1", eligible_models=["only"])
        ev = log.report_outcome(
            "r1", profile_id="only", status="failed",
            failure_reason="tests_failed", exhausted=True,
        )
        assert ev is not None
        assert ev.status == "failed"
        assert ev.failure_reason == "tests_failed"
        assert len(ev.attempts) == 1
        assert len(log.events) == 1
        s = log.summary()
        assert s["total"] == 1
        assert s["failure_rate"] == 1.0

    def test_start_event_rejects_duplicate_route_id(self):
        log = TelemetryLog()
        _start(log, "r1")
        try:
            _start(log, "r1")
            raise AssertionError("duplicate route id must be rejected")
        except ValueError:
            pass

    def test_active_index_is_bounded(self):
        log = TelemetryLog()
        log._active_max = 4
        for i in range(10):
            _start(log, f"r{i}")
        assert len(log._active) == 4
        assert "r0" not in log._active
        assert "r9" in log._active

    def test_get_event_finds_pending_and_finalized(self):
        log = TelemetryLog()
        _start(log, "r1")
        assert log.get_event("r1") is not None
        assert log.get_event("r1").status == "pending"
        log.report_outcome("r1", profile_id="a", status="success")
        assert log.get_event("r1").status == "success"


class TestAttemptEvent:
    def test_attempt_shape(self):
        a = AttemptEvent(
            profile_id="local_coding", status="failed",
            failure_category="rate_limit", latency_ms=12.0,
        )
        d = a.to_dict()
        assert d["profile_id"] == "local_coding"
        assert d["status"] == "failed"
        assert d["failure_category"] == "rate_limit"

    def test_event_serializes_attempts(self):
        ev = _make_event(event_id="e1")
        ev.attempts.append(AttemptEvent(profile_id="x", status="success"))
        d = ev.to_dict()
        assert isinstance(d["attempts"], list)
        assert d["attempts"][0]["profile_id"] == "x"


class TestAggregateMetrics:
    def test_summary_reduces_finalized_only(self):
        log = TelemetryLog()
        _start(log, "pending1")
        log.record(_make_event(status="success"))
        log.record(_make_event(status="failed"))
        s = log.summary()
        assert s["total"] == 2
        assert abs(s["success_rate"] - 0.5) < 0.001
        assert abs(s["failure_rate"] - 0.5) < 0.001

    def test_empty_summary(self):
        log = TelemetryLog()
        s = log.summary()
        assert s["total"] == 0
        assert s["access_distribution"] == {}

    def test_tier_distribution(self):
        log = TelemetryLog()
        log.record(_make_event(selected_tier="local_coding"))
        log.record(_make_event(selected_tier="local_coding"))
        log.record(_make_event(selected_tier="cloud_reasoning"))
        s = log.summary()
        assert s["tier_distribution"]["local_coding"] == 2
        assert s["tier_distribution"]["cloud_reasoning"] == 1

    def test_access_and_model_distribution(self):
        log = TelemetryLog()
        log.record(_make_event(
            access_class="subscription", selected_profile="sub_a",
        ))
        log.record(_make_event(
            access_class="subscription", selected_profile="sub_a",
        ))
        log.record(_make_event(
            access_class="local", selected_profile="local_x",
        ))
        s = log.summary()
        assert s["access_distribution"] == {"subscription": 2, "local": 1}
        assert s["model_distribution"] == {"sub_a": 2, "local_x": 1}

    def test_paid_selection_reasons(self):
        log = TelemetryLog()
        log.record(_make_event(
            access_class="metered-cheap",
            paid_selection_reasons=["capability:strong", "no_subscription"],
        ))
        log.record(_make_event(
            access_class="metered-cheap",
            paid_selection_reasons=["capability:strong"],
        ))
        log.record(_make_event(access_class="subscription"))
        s = log.summary()
        assert s["paid_selection_reasons"] == {
            "capability:strong": 2, "no_subscription": 1,
        }

    def test_deterministic_vs_ollaya_counts_and_ollaya_latency(self):
        log = TelemetryLog()
        log.record(_make_event(
            classifier_used="ollaya", ollaya_invoked=True,
            ollaya_outcome="classified",
            ollaya_confidence=0.82, ollaya_latency_ms=205.0,
        ))
        log.record(_make_event(
            classifier_used="deterministic", ollaya_invoked=True,
            ollaya_outcome="error_fallback",
            ollaya_confidence=0.0, ollaya_latency_ms=500.0,
        ))
        log.record(_make_event(classifier_used="deterministic"))
        s = log.summary()
        assert s["ollaya_count"] == 2
        assert s["deterministic_count"] == 2
        assert s["ollaya_error_fallback_count"] == 1
        assert abs(s["avg_ollaya_latency_ms"] - 352.5) < 0.01
        # Confidence averages over *measured* classifications only; the
        # error-fallback attempt recorded no confidence.
        assert abs(s["avg_ollaya_confidence"] - 0.82) < 0.001

    def test_local_failure_and_escalation_rate(self):
        log = TelemetryLog()
        log.record(_make_event(access_class="local", status="failed"))
        log.record(_make_event(access_class="local", status="success"))
        log.record(_make_event(access_class="local", escalated=True))
        log.record(_make_event(access_class="subscription", status="failed"))
        s = log.summary()
        assert s["local_event_count"] == 3
        assert abs(s["local_failure_rate"] - 1 / 3) < 0.001
        assert abs(s["local_escalation_rate"] - 1 / 3) < 0.001

    def test_failure_categories_from_attempts(self):
        log = TelemetryLog()
        ev = _make_event(status="failed", access_class="subscription")
        ev.attempts = [
            AttemptEvent(profile_id="a", status="failed",
                         failure_category="rate_limit"),
            AttemptEvent(profile_id="b", status="failed",
                         failure_category="task_failure"),
            AttemptEvent(profile_id="c", status="failed",
                         failure_category="task_failure"),
        ]
        log.record(ev)
        s = log.summary()
        assert s["failure_categories"] == {
            "rate_limit": 1, "task_failure": 2,
        }

    def test_estimated_spend_incurred_and_avoided(self):
        log = TelemetryLog()
        log.record(_make_event(
            access_class="metered-cheap",
            estimated_spend_incurred=0.04, estimated_spend_avoided=0.0,
        ))
        log.record(_make_event(
            access_class="subscription",
            estimated_spend_incurred=0.0, estimated_spend_avoided=0.06,
        ))
        s = log.summary()
        assert abs(s["estimated_spend_incurred"] - 0.04) < 1e-9
        assert abs(s["estimated_spend_avoided"] - 0.06) < 1e-9

    def test_spend_accumulates_across_attempt_outcomes(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            log = TelemetryLog(log_path=path)
            _start(log, "r1", access_class="metered-cheap")
            log.report_outcome(
                "r1", profile_id="paid_a", status="failed",
                latency_ms=10.0, cost=0.02,
                estimated_spend_incurred=0.02,
            )
            log.report_outcome(
                "r1", profile_id="paid_b", status="success",
                latency_ms=10.0, cost=0.03,
                estimated_spend_incurred=0.03,
            )
            s = log.summary()
            assert abs(s["total_cost"] - 0.05) < 1e-9
            assert abs(s["estimated_spend_incurred"] - 0.05) < 1e-9
        finally:
            os.unlink(path)

    def test_escalation_rate(self):
        log = TelemetryLog()
        log.record(_make_event(escalated=True))
        log.record(_make_event(escalated=False))
        s = log.summary()
        assert s["escalation_rate"] == 0.5


class TestLegacyRecord:
    """The legacy record() path stays a finalize-and-append."""

    def test_record_is_finalized_append(self):
        log = TelemetryLog()
        log.record(_make_event(event_id="e1"))
        assert log.events[0].finalized is True

    def test_record_with_same_event_id_is_idempotent(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            log = TelemetryLog(log_path=path)
            ev = _make_event(event_id="e1", status="success")
            log.record(ev)
            ev.status = "failed"
            log.record(ev)  # duplicate settlement: replace, never append
            assert len(log.events) == 1
            assert len(_jsonl_lines(path)) == 1
            assert log.events[0].status == "failed"
        finally:
            os.unlink(path)

    def test_record_pops_active_pending_event(self):
        log = TelemetryLog()
        _start(log, "r1")
        ev = log.get_event("r1")
        log.record(ev)  # legacy route-and-execute style finalize
        assert log.get_event("r1").status == "pending"  # replaced in place
        assert "r1" not in log._active
        assert len(log.events) == 1

    def test_jsonl_roundtrip_with_attempts(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            log1 = TelemetryLog(log_path=path)
            _start(log1, "r1", eligible_models=["a", "b"])
            log1.report_outcome(
                "r1", profile_id="a", status="failed",
                failure_category="rate_limit",
            )
            log1.report_outcome("r1", profile_id="b", status="success")

            log2 = TelemetryLog(log_path=path)
            events = log2.load()
            assert len(events) == 1
            assert events[0].event_id == "r1"
            assert [a.profile_id for a in events[0].attempts] == ["a", "b"]
            assert events[0].attempts[0].failure_category == "rate_limit"
            assert log2.summary()["total"] == 1
        finally:
            os.unlink(path)

    def test_filter_by(self):
        log = TelemetryLog()
        log.record(_make_event(selected_tier="local_coding", status="success"))
        log.record(_make_event(selected_tier="cloud_coding", status="failed"))
        log.record(_make_event(selected_tier="local_coding", status="failed"))
        assert len(log.filter_by(status="failed")) == 2
        assert len(log.filter_by(selected_tier="local_coding", status="success")) == 1

    def test_under_routing_events(self):
        log = TelemetryLog()
        log.record(_make_event(escalated=True, failure_reason="tests_failed"))
        log.record(_make_event(escalated=False))
        assert len(log.under_routing_events()) == 1

    def test_over_routing_events(self):
        log = TelemetryLog()
        log.record(_make_event(
            selected_tier="cloud_coding", status="success",
            classification={"complexity": "trivial"},
        ))
        log.record(_make_event(
            selected_tier="local_coding", status="success",
            classification={"complexity": "trivial"},
        ))
        over = log.over_routing_events()
        assert len(over) == 1
        assert over[0].selected_tier == "cloud_coding"


class TestRoutingCasesFixture:
    """The labeled replay fixture covers all 22 required routing cases."""

    def _load(self):
        data = json.loads(FIXTURE_PATH.read_text())
        return data

    def _benchmark_module(self):
        scripts_dir = str(
            Path(__file__).resolve().parent.parent / "scripts"
        )
        sys.path.insert(0, scripts_dir)
        import benchmark_routing  # noqa: PLC0415
        return benchmark_routing

    def test_fixture_exists_and_has_at_least_22_cases(self):
        data = self._load()
        cases = data["cases"]
        assert len(cases) >= 22, f"only {len(cases)} cases"
        ids = [c["id"] for c in cases]
        assert len(ids) == len(set(ids)), "case ids must be unique"
        # The 22 spec-mandated scenarios are all present and labeled.
        spec_ids = {c["spec_case"] for c in cases if c.get("spec_case")}
        assert spec_ids == set(range(1, 23))

    def test_every_case_declares_required_fields(self):
        for case in self._load()["cases"]:
            for field in (
                "id", "prompt", "turn_context", "expected",
            ):
                assert field in case, f"{case.get('id')} missing {field}"
            for field in (
                "min_capability", "access_class", "ollaya_expected",
            ):
                assert field in case["expected"], (
                    f"{case['id']} missing expected.{field}"
                )
            assert case["expected"]["min_capability"] in (
                "basic", "normal", "strong", "reasoning",
            )
            assert case["expected"]["access_class"] in (
                "local", "subscription", "metered-cheap",
                "metered-expensive",
            )
            assert isinstance(case["expected"]["ollaya_expected"], bool)

    def test_registry_covers_all_access_classes(self):
        data = self._load()
        access = {p["access_class"] for p in data["registry"]}
        assert access == {
            "local", "subscription", "metered-cheap", "metered-expensive",
        }

    def test_deterministic_replay_matches_every_expectation(self):
        """Full offline replay: classification, policy selection, and
        Ollaya-invocation expectations all hold, deterministically."""
        bench = self._benchmark_module()
        data = self._load()
        result = bench.replay(data, ollaya=None)
        rows = result["rows"]
        assert len(rows) == len(data["cases"])
        assert result["mismatches"] == [], (
            f"fixture expectation mismatches: {result['mismatches']}"
        )

    def test_replay_records_latency_and_escalation(self):
        bench = self._benchmark_module()
        data = self._load()
        result = bench.replay(data, ollaya=None)
        for row in result["rows"]:
            assert row["classifier_latency_ms"] >= 0.0
            assert "simulated_escalation" in row
            assert row["simulated_escalation"]["outcome"] in (
                "escalated", "exhausted", "single_candidate", "no_candidates",
            )
        summary = result["summary"]
        assert summary["total_cases"] == len(data["cases"])
        assert summary["ollaya_consulted_expected"] == sum(
            1 for c in data["cases"]
            if c["expected"]["ollaya_expected"]
        )
