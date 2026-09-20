"""
Tests for the confidence-gated classifier cascade.

Uses mock RLCD callables so no models are loaded.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.classifier import RouterClassifier, ClassificationResult, bypass_classification
from router.signals import RuntimeSignals
from router.config import ConfidenceThresholds


def _signals(**overrides) -> RuntimeSignals:
    defaults = dict(
        prompt_chars=500,
        estimated_tokens=125,
        prior_attempts=0,
        prior_failures=0,
        deadline="normal",
    )
    defaults.update(overrides)
    return RuntimeSignals(**defaults)


def _make_rlcd_output(fields: dict[str, tuple[str, float]]) -> dict:
    """Build a fake RLCD engine output."""
    parsed = {}
    for name, (value, prob) in fields.items():
        parsed[name] = {"value": value, "prob": prob}
    return {
        "parsed_json": parsed,
        "elapsed_ms": 100.0,
        "is_valid_json": True,
        "schema_match": True,
    }


# Standard high-confidence output for all 11 router fields
_HIGH_CONF_FIELDS = {
    "task_type": ("explanation", 0.95),
    "complexity": ("low", 0.92),
    "coding": ("none", 0.97),
    "reasoning_depth": ("direct", 0.93),
    "fresh_information": ("none", 0.96),
    "repo_access": ("none", 0.98),
    "tool_use": ("none", 0.94),
    "privacy": ("ordinary", 0.97),
    "latency": ("normal", 0.90),
    "failure_cost": ("low", 0.91),
    "output_contract": ("freeform", 0.93),
}


class TestFastAcceptance:
    def test_high_confidence_accepts_fast(self):
        run_fast = lambda ctx, schema: _make_rlcd_output(_HIGH_CONF_FIELDS)
        classifier = RouterClassifier(run_fast=run_fast)
        result = classifier.classify("Explain recursion", _signals())
        assert result.classifier_used == "rlcd_fast"
        assert result.abstain is False
        assert result.fields["task_type"] == "explanation"

    def test_low_overall_confidence_triggers_smart(self):
        low_conf = {k: (v, p * 0.6) for k, (v, p) in _HIGH_CONF_FIELDS.items()}
        run_fast = lambda ctx, schema: _make_rlcd_output(low_conf)
        run_smart = lambda ctx, schema: _make_rlcd_output(_HIGH_CONF_FIELDS)
        classifier = RouterClassifier(run_fast=run_fast, run_smart=run_smart)
        result = classifier.classify("Explain recursion", _signals())
        assert result.classifier_used == "rlcd_smart"

    def test_low_critical_confidence_triggers_smart(self):
        fields = dict(_HIGH_CONF_FIELDS)
        fields["privacy"] = ("sensitive", 0.40)  # critical field, low conf
        run_fast = lambda ctx, schema: _make_rlcd_output(fields)
        run_smart = lambda ctx, schema: _make_rlcd_output(_HIGH_CONF_FIELDS)
        classifier = RouterClassifier(run_fast=run_fast, run_smart=run_smart)
        result = classifier.classify("Process private data", _signals())
        assert result.classifier_used == "rlcd_smart"

    def test_prior_failures_triggers_smart(self):
        run_fast = lambda ctx, schema: _make_rlcd_output(_HIGH_CONF_FIELDS)
        run_smart = lambda ctx, schema: _make_rlcd_output(_HIGH_CONF_FIELDS)
        classifier = RouterClassifier(run_fast=run_fast, run_smart=run_smart)
        result = classifier.classify(
            "Fix the bug again",
            _signals(prior_failures=1),
        )
        assert result.classifier_used == "rlcd_smart"


class TestSmartAbstention:
    def test_smart_also_uncertain_abstains(self):
        low_conf = {k: (v, p * 0.5) for k, (v, p) in _HIGH_CONF_FIELDS.items()}
        run_fast = lambda ctx, schema: _make_rlcd_output(low_conf)
        run_smart = lambda ctx, schema: _make_rlcd_output(low_conf)
        classifier = RouterClassifier(run_fast=run_fast, run_smart=run_smart)
        result = classifier.classify("Something ambiguous", _signals())
        assert result.abstain is True

    def test_no_smart_classifier_abstains_on_low_confidence(self):
        # confidence * 0.4 → mean ~0.38, below conservative_floor (0.50)
        very_low = {k: (v, p * 0.4) for k, (v, p) in _HIGH_CONF_FIELDS.items()}
        run_fast = lambda ctx, schema: _make_rlcd_output(very_low)
        classifier = RouterClassifier(run_fast=run_fast, run_smart=None)
        result = classifier.classify("Something", _signals())
        assert result.abstain is True

    def test_medium_confidence_no_smart_no_abstain(self):
        # confidence * 0.6 → mean ~0.56, above conservative_floor (0.50)
        # Without smart classifier, accepted but not via fast-accept path
        med_conf = {k: (v, p * 0.6) for k, (v, p) in _HIGH_CONF_FIELDS.items()}
        run_fast = lambda ctx, schema: _make_rlcd_output(med_conf)
        classifier = RouterClassifier(run_fast=run_fast, run_smart=None)
        result = classifier.classify("Something", _signals())
        assert result.abstain is False
        assert result.classifier_used == "rlcd_fast"


class TestInteractiveSkipSmart:
    def test_interactive_deadline_skips_smart(self):
        low_conf = {k: (v, p * 0.7) for k, (v, p) in _HIGH_CONF_FIELDS.items()}
        run_fast = lambda ctx, schema: _make_rlcd_output(low_conf)
        smart_called = [False]

        def run_smart(ctx, schema):
            smart_called[0] = True
            return _make_rlcd_output(_HIGH_CONF_FIELDS)

        classifier = RouterClassifier(run_fast=run_fast, run_smart=run_smart)
        result = classifier.classify("Quick question", _signals(deadline="interactive"))
        assert result.classifier_used == "rlcd_fast"
        assert smart_called[0] is False


class TestBypass:
    def test_bypass_sets_abstain(self):
        result = bypass_classification("no model loaded")
        assert result.abstain is True
        assert result.classifier_used == "bypass"
        assert result.overall_confidence == 0.0

    def test_bypass_has_all_fields(self):
        result = bypass_classification()
        assert len(result.fields) == 11


class TestConfidenceThresholds:
    def test_custom_thresholds(self):
        # 0.78 is below fast_accept_overall (0.85) so fast is rejected,
        # but above conservative_floor (0.65) so without a smart classifier
        # it still accepts the fast result (just not via the fast-accept path).
        low_conf = {k: (v, 0.78) for k, (v, _) in _HIGH_CONF_FIELDS.items()}
        run_fast = lambda ctx, schema: _make_rlcd_output(low_conf)

        c1 = RouterClassifier(run_fast=run_fast, run_smart=None)
        r1 = c1.classify("test", _signals())
        # Below fast_accept → not fast-accepted, but above conservative_floor → not abstained
        assert r1.classifier_used == "rlcd_fast"
        assert r1.abstain is False

        # Very low confidence → below conservative_floor (0.50) → abstain
        very_low = {k: (v, 0.40) for k, (v, _) in _HIGH_CONF_FIELDS.items()}
        run_fast_low = lambda ctx, schema: _make_rlcd_output(very_low)
        c_low = RouterClassifier(run_fast=run_fast_low, run_smart=None)
        r_low = c_low.classify("test", _signals())
        assert r_low.abstain is True

        # Relaxed thresholds: 0.70 overall → 0.78 is fast-accepted
        relaxed = ConfidenceThresholds(fast_accept_overall=0.70, fast_accept_critical=0.60)
        c2 = RouterClassifier(run_fast=run_fast, run_smart=None, thresholds=relaxed)
        r2 = c2.classify("test", _signals())
        assert r2.abstain is False
