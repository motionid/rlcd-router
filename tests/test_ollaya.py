"""
Tests for the optional Ollaya ambiguity-classification adapter.

The adapter is disabled by default, is consulted only for ambiguous
deterministic classifications, and must never be a single point of
failure: timeouts, malformed responses, and disabled configuration
all fall back to the deterministic classification.

All non-live tests use an injected fake transport — no network, no
Ollaya service, no downloaded models.  The optional live smoke test
is gated behind ``OLLAYA_LIVE_TEST=1``.

Mocked response bodies mirror the raw wire format captured during
Task 1's live verification of Ollaya 0.6.1 (see the task-1 report).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.config import Capability, OllayaConfig
from router.deterministic import TaskRelationship
from router.signals import TurnContext
from router.ollaya import (
    OLLAYA_REASON_CODES,
    OLLAYA_STATE_CHAR_LIMIT,
    APPROVED_QUESTIONS,
    OllayaClassifier,
    OllayaDisabledError,
    OllayaError,
    OllayaParseError,
    OllayaUnavailableError,
    classify_task,
)


# ---------------------------------------------------------------------------
# Wire-format fixtures (verbatim shapes from the Task 1 live verification)
# ---------------------------------------------------------------------------

# Task 1 Example 1 — obvious basic request.
BASIC_RESPONSE: dict = {
    "model": "laya:en",
    "answers": {
        "required_capability": {
            "type": "choice",
            "choice": "basic",
            "confidence": 0.3165,
            "probabilities": {
                "basic": 0.4874,
                "normal": 0.3241,
                "strong": 0.0983,
                "reasoning": 0.0902,
            },
        },
        "is_task_continuation": {"type": "noul", "noul": 0.1699},
        "is_independent_trivial_task": {"type": "noul", "noul": 0.1508},
    },
    "usage": {"input_tokens": 223, "output_tokens": 0},
    "state_truncated": False,
    "done_reason": "decide",
    "total_duration": 155527833,
    "load_duration": 0,
    "eval_duration": 155357334,
}

# Task 1 Example 2 — ambiguous continuation after a strong task.
CONTINUATION_RESPONSE: dict = {
    "model": "laya:en",
    "answers": {
        "required_capability": {
            "type": "choice",
            "choice": "strong",
            "confidence": 0.2047,
            "probabilities": {
                "basic": 0.0937,
                "normal": 0.3248,
                "strong": 0.4035,
                "reasoning": 0.1779,
            },
        },
        "is_task_continuation": {"type": "noul", "noul": 0.8152},
        "is_independent_trivial_task": {"type": "noul", "noul": 0.628},
    },
    "usage": {"input_tokens": 346, "output_tokens": 0},
    "state_truncated": False,
    "done_reason": "decide",
    "total_duration": 191266792,
    "load_duration": 0,
    "eval_duration": 191112833,
}


def confident_basic_response(basic_prob: float = 0.95) -> dict:
    """A basic answer whose probability clears the 0.90 basic threshold."""
    rest = round(1.0 - basic_prob, 4) / 3
    response = dict(BASIC_RESPONSE)
    response["answers"] = {
        "required_capability": {
            "type": "choice",
            "choice": "basic",
            "confidence": 0.95,
            "probabilities": {
                "basic": basic_prob,
                "normal": rest,
                "strong": rest,
                "reasoning": rest,
            },
        },
        "is_task_continuation": {"type": "noul", "noul": 0.05},
        "is_independent_trivial_task": {"type": "noul", "noul": 0.9},
    }
    return response


def continuation_context() -> TurnContext:
    return TurnContext(
        previous_prompt=(
            "The user asked the assistant to refactor the authentication "
            "module of a Python service. The assistant analyzed the code, "
            "proposed splitting token validation into a separate module, "
            "and asked: 'Shall I proceed with the refactor?'"
        ),
        previous_capability=Capability.STRONG,
        previous_model="local_coding",
    )


def fake_transport(response: dict | None = None, calls: list | None = None,
                   error: Exception | None = None):
    """Build an injectable transport capturing the request payload."""
    def transport(url: str, payload: dict, timeout_ms: int) -> dict:
        if calls is not None:
            calls.append({"url": url, "payload": payload,
                          "timeout_ms": timeout_ms})
        if error is not None:
            raise error
        assert response is not None
        return response
    return transport


def enabled_classifier(response: dict | None = None, calls: list | None = None,
                       error: Exception | None = None,
                       config: OllayaConfig | None = None) -> OllayaClassifier:
    return OllayaClassifier(
        config=config or OllayaConfig(enabled=True),
        transport=fake_transport(response, calls, error),
    )


# ---------------------------------------------------------------------------
# Invocation gate — deterministic first
# ---------------------------------------------------------------------------

class TestInvocationGate:
    def test_ollaya_is_not_called_for_obvious_rename(self):
        calls: list = []
        result = classify_task(
            "rename the variable x to user_count in parser.py",
            TurnContext(),
            ollaya=enabled_classifier(response=BASIC_RESPONSE, calls=calls),
        )
        # Deterministic, unambiguous result: adapter never invoked.
        assert calls == []
        assert result.classifier == "deterministic"
        assert result.capability == Capability.BASIC
        assert result.ambiguous is False

    def test_obvious_strong_request_never_invokes_ollaya(self):
        calls: list = []
        result = classify_task(
            "debug the race condition across the repository",
            TurnContext(),
            ollaya=enabled_classifier(response=CONTINUATION_RESPONSE,
                                      calls=calls),
        )
        assert calls == []
        assert result.classifier == "deterministic"
        assert result.capability >= Capability.STRONG

    def test_ambiguous_prompt_without_ollaya_returns_conservative_deterministic(self):
        result = classify_task("why?", TurnContext(), ollaya=None)
        assert result.classifier == "deterministic"
        assert result.ambiguous is True
        assert result.capability >= Capability.NORMAL

    def test_ollaya_can_be_disabled(self):
        calls: list = []
        classifier = enabled_classifier(
            response=BASIC_RESPONSE,
            calls=calls,
            config=OllayaConfig(enabled=False),
        )
        result = classify_task("why?", TurnContext(), ollaya=classifier)
        assert calls == []
        assert result.classifier == "deterministic"
        assert result.ambiguous is True
        assert result.capability >= Capability.NORMAL

    def test_disabled_classifier_direct_call_raises_disabled_error(self):
        classifier = OllayaClassifier(config=OllayaConfig(enabled=False),
                                      transport=fake_transport(BASIC_RESPONSE))
        with pytest.raises(OllayaDisabledError):
            classifier.classify("why?", TurnContext())

    def test_config_defaults_to_disabled(self):
        assert OllayaConfig().enabled is False


# ---------------------------------------------------------------------------
# Ambiguous classification through the adapter
# ---------------------------------------------------------------------------

class TestAmbiguousClassification:
    def test_ollaya_can_classify_ambiguous_continuation(self):
        calls: list = []
        result = classify_task(
            "yes, go ahead",
            continuation_context(),
            ollaya=enabled_classifier(response=CONTINUATION_RESPONSE,
                                      calls=calls),
        )
        assert len(calls) == 1
        assert result.classifier == "ollaya"
        assert result.relationship is TaskRelationship.CONTINUATION
        assert result.ambiguous is False
        # Inherited STRONG continuation capability, never paid access.
        assert result.capability == Capability.STRONG
        assert "ollaya_continuation" in result.reasons
        assert "ollaya_inherited_capability" in result.reasons

    def test_low_confidence_basic_falls_back_to_subscription_safe_capability(self):
        # Task 1 live evidence: an obvious-basic answer carried only 0.4874
        # basic probability and 0.3165 confidence — far below the 0.90 basic
        # threshold.  It must land on subscription-safe NORMAL, never BASIC.
        result = classify_task(
            "What is the capital of France?",
            TurnContext(),
            ollaya=enabled_classifier(response=BASIC_RESPONSE),
        )
        assert result.classifier == "ollaya"
        assert result.capability == Capability.NORMAL
        assert "ollaya_basic_below_threshold" in result.reasons

    def test_high_confidence_basic_is_permitted(self):
        result = classify_task(
            "What is the capital of France?",
            TurnContext(),
            ollaya=enabled_classifier(response=confident_basic_response()),
        )
        assert result.classifier == "ollaya"
        assert result.capability == Capability.BASIC
        assert result.relationship is TaskRelationship.INDEPENDENT
        assert "ollaya_basic_confident" in result.reasons

    def test_independent_trivial_noul_never_alone_permits_basic_downgrade(self):
        # Binding ruling: is_independent_trivial_task is diagnostic
        # telemetry only.  Even a 0.99 noul must not downgrade to basic
        # when the capability evidence is below threshold.
        response = confident_basic_response(basic_prob=0.50)
        result = classify_task(
            "What is the capital of France?",
            TurnContext(),
            ollaya=enabled_classifier(response=response),
        )
        assert result.capability != Capability.BASIC
        assert result.capability >= Capability.NORMAL
        assert "ollaya_independent_trivial_diagnostic" in result.reasons

    def test_continuation_below_threshold_is_not_a_continuation(self):
        result = classify_task(
            "yes, go ahead",
            continuation_context(),
            ollaya=enabled_classifier(
                response=dict(CONTINUATION_RESPONSE,
                              **{"answers": dict(
                                  CONTINUATION_RESPONSE["answers"],
                                  **{"is_task_continuation":
                                     {"type": "noul", "noul": 0.42}})}),
            ),
        )
        assert result.relationship is TaskRelationship.NEW_SUBSTANTIAL
        assert "ollaya_continuation" not in result.reasons

    def test_continuation_without_history_floors_at_normal(self):
        response = dict(CONTINUATION_RESPONSE)
        response["answers"] = dict(
            CONTINUATION_RESPONSE["answers"],
            **{
                "required_capability": {
                    "type": "choice", "choice": "normal",
                    "confidence": 0.7,
                    "probabilities": {"basic": 0.05, "normal": 0.75,
                                       "strong": 0.15, "reasoning": 0.05},
                },
            },
        )
        result = classify_task(
            "yes, go ahead",
            TurnContext(),  # no previous capability to inherit
            ollaya=enabled_classifier(response=response),
        )
        assert result.relationship is TaskRelationship.CONTINUATION
        assert result.capability == Capability.NORMAL
        assert "ollaya_missing_history" in result.reasons


# ---------------------------------------------------------------------------
# Capability-confidence gate — upclassification must be confident
# ---------------------------------------------------------------------------

def strong_response(confidence: float, continuation: float) -> dict:
    """A strong-choice answer with explicit confidence/continuation."""
    response = dict(CONTINUATION_RESPONSE)
    response["answers"] = dict(
        CONTINUATION_RESPONSE["answers"],
        **{
            "required_capability": {
                "type": "choice",
                "choice": "strong",
                "confidence": confidence,
                "probabilities": {
                    "basic": 0.09, "normal": 0.32,
                    "strong": 0.40, "reasoning": 0.19,
                },
            },
            "is_task_continuation": {"type": "noul", "noul": continuation},
        },
    )
    return response


class TestCapabilityConfidenceGate:
    def test_capability_threshold_defaults_to_060(self):
        assert OllayaConfig().capability_threshold == 0.60

    def test_low_confidence_strong_non_continuation_stays_normal(self):
        # Task 1 live evidence: the ambiguous continuation answer carried
        # only 0.2047 confidence.  A strong label at that confidence must
        # not upclassify — conservative NORMAL instead.
        result = classify_task(
            "What is the capital of France?",
            TurnContext(),
            ollaya=enabled_classifier(
                response=strong_response(confidence=0.2047,
                                         continuation=0.10)),
        )
        assert result.classifier == "ollaya"
        assert result.capability == Capability.NORMAL
        assert "ollaya_low_confidence_capability" in result.reasons

    def test_low_confidence_reasoning_non_continuation_stays_normal(self):
        response = strong_response(confidence=0.30, continuation=0.10)
        response["answers"]["required_capability"]["choice"] = "reasoning"
        result = classify_task(
            "What is the capital of France?",
            TurnContext(),
            ollaya=enabled_classifier(response=response),
        )
        assert result.capability == Capability.NORMAL
        assert "ollaya_low_confidence_capability" in result.reasons

    def test_low_confidence_strong_continuation_inherits_not_max(self):
        # High-confidence continuation, low-confidence capability: the
        # turn inherits the previous NORMAL capability — it must not be
        # max()'d upward by the untrusted strong label.
        context = TurnContext(
            previous_prompt="add input validation to the forms",
            previous_capability=Capability.NORMAL,
            previous_model="gpt_6_luna",
        )
        result = classify_task(
            "yes, go ahead",
            context,
            ollaya=enabled_classifier(
                response=strong_response(confidence=0.2047,
                                         continuation=0.85)),
        )
        assert result.relationship is TaskRelationship.CONTINUATION
        assert result.capability == Capability.NORMAL
        assert "ollaya_low_confidence_capability" in result.reasons
        assert "ollaya_inherited_capability" in result.reasons

    def test_high_confidence_strong_continuation_conservatively_raises(self):
        context = TurnContext(
            previous_prompt="add input validation to the forms",
            previous_capability=Capability.NORMAL,
            previous_model="gpt_6_luna",
        )
        result = classify_task(
            "yes, go ahead",
            context,
            ollaya=enabled_classifier(
                response=strong_response(confidence=0.90,
                                         continuation=0.85)),
        )
        assert result.relationship is TaskRelationship.CONTINUATION
        assert result.capability == Capability.STRONG
        assert "ollaya_low_confidence_capability" not in result.reasons

    def test_high_confidence_strong_non_continuation_is_accepted(self):
        result = classify_task(
            "What is the capital of France?",
            TurnContext(),
            ollaya=enabled_classifier(
                response=strong_response(confidence=0.90,
                                         continuation=0.10)),
        )
        assert result.capability == Capability.STRONG
        assert result.relationship is TaskRelationship.NEW_SUBSTANTIAL
        assert "ollaya_low_confidence_capability" not in result.reasons

    def test_capability_threshold_is_configurable(self):
        config = OllayaConfig(enabled=True, capability_threshold=0.10)
        result = classify_task(
            "What is the capital of France?",
            TurnContext(),
            ollaya=enabled_classifier(
                response=strong_response(confidence=0.2047,
                                         continuation=0.10),
                config=config,
            ),
        )
        # With the threshold lowered, 0.2047 confidence is accepted.
        assert result.capability == Capability.STRONG


class TestStateTruncation:
    def test_state_truncated_true_returns_deterministic_fallback(self):
        truncated = dict(BASIC_RESPONSE, state_truncated=True)
        result = classify_task(
            "why?",
            TurnContext(),
            ollaya=enabled_classifier(response=truncated),
        )
        assert result.classifier == "deterministic"
        assert result.ambiguous is True
        assert result.capability >= Capability.NORMAL

    def test_state_truncated_true_raises_parse_error(self):
        truncated = dict(BASIC_RESPONSE, state_truncated=True)
        classifier = enabled_classifier(response=truncated)
        with pytest.raises(OllayaParseError):
            classifier.classify("why?", TurnContext())

    def test_state_truncated_false_is_accepted(self):
        # The Task 1 wire format always carries the flag; only `true` is a
        # fallback condition.
        result = classify_task(
            "yes, go ahead",
            continuation_context(),
            ollaya=enabled_classifier(response=CONTINUATION_RESPONSE),
        )
        assert result.classifier == "ollaya"


# ---------------------------------------------------------------------------
# Failure behavior — Ollaya is never a single point of failure
# ---------------------------------------------------------------------------

class TestFallbacks:
    def test_unavailable_ollaya_returns_deterministic_fallback(self):
        calls: list = []
        result = classify_task(
            "why?",
            TurnContext(),
            ollaya=enabled_classifier(error=ConnectionError("refused"),
                                      calls=calls),
        )
        assert len(calls) == 1
        assert result.classifier == "deterministic"
        assert result.ambiguous is True
        assert result.capability >= Capability.NORMAL

    def test_transport_failure_raises_unavailable_error(self):
        classifier = enabled_classifier(error=TimeoutError("timed out"))
        with pytest.raises(OllayaUnavailableError):
            classifier.classify("why?", TurnContext())

    def test_malformed_ollaya_response_returns_deterministic_fallback(self):
        for malformed in (
            {},                                   # empty body
            {"answers": {}},                      # missing everything
            {"answers": {"required_capability": {"type": "choice",
                                                 "choice": "ultra"}}},  # bad label
            {"answers": {
                "required_capability": {"type": "choice", "choice": "basic"},
                "is_task_continuation": {"type": "noul", "noul": "high"},
            }},                                    # non-numeric noul
        ):
            result = classify_task(
                "why?",
                TurnContext(),
                ollaya=enabled_classifier(response=malformed),
            )
            assert result.classifier == "deterministic", malformed
            assert result.ambiguous is True
            assert result.capability >= Capability.NORMAL

    def test_malformed_response_raises_parse_error(self):
        classifier = enabled_classifier(response={"nonsense": True})
        with pytest.raises(OllayaParseError):
            classifier.classify("why?", TurnContext())

    def test_ollaya_error_hierarchy(self):
        assert issubclass(OllayaDisabledError, OllayaError)
        assert issubclass(OllayaParseError, OllayaError)
        assert issubclass(OllayaUnavailableError, OllayaError)


# ---------------------------------------------------------------------------
# Wire format — payload construction and result evidence
# ---------------------------------------------------------------------------

class TestWireFormat:
    def test_state_payload_is_bounded_and_contains_prior_context(self):
        calls: list = []
        context = continuation_context()
        long_prompt = "do something with " + "x" * (OLLAYA_STATE_CHAR_LIMIT + 500)
        classifier = enabled_classifier(response=CONTINUATION_RESPONSE,
                                        calls=calls)
        classifier.classify(long_prompt, context)
        assert len(calls) == 1
        state = calls[0]["payload"]["state"]
        # Prior context is carried and both fields are hard-bounded.
        assert state["previous_context"] == context.previous_prompt
        assert len(state["previous_context"]) <= OLLAYA_STATE_CHAR_LIMIT
        assert state["current_turn"] == long_prompt[:OLLAYA_STATE_CHAR_LIMIT]
        assert len(state["current_turn"]) <= OLLAYA_STATE_CHAR_LIMIT

    def test_payload_sends_three_approved_questions_and_keep_alive(self):
        calls: list = []
        config = OllayaConfig(enabled=True, model="laya", timeout_ms=500)
        classify_task(
            "yes, go ahead",
            continuation_context(),
            ollaya=enabled_classifier(response=CONTINUATION_RESPONSE,
                                      calls=calls, config=config),
        )
        payload = calls[0]["payload"]
        assert payload["model"] == "laya"
        assert payload["keep_alive"] == -1  # warm preload per Task 1 ruling
        assert payload["questions"] == APPROVED_QUESTIONS
        assert set(payload["questions"]) == {
            "required_capability",
            "is_task_continuation",
            "is_independent_trivial_task",
        }
        assert calls[0]["url"] == "http://127.0.0.1:11435/api/decide"
        assert calls[0]["timeout_ms"] == 500

    def test_ollaya_result_contains_latency_and_probabilities(self):
        result = classify_task(
            "yes, go ahead",
            continuation_context(),
            ollaya=enabled_classifier(response=CONTINUATION_RESPONSE),
        )
        assert result.classifier == "ollaya"
        assert result.latency_ms is not None
        assert result.latency_ms >= 0.0
        probabilities = dict(result.probabilities)
        assert probabilities["basic"] == pytest.approx(0.0937)
        assert probabilities["strong"] == pytest.approx(0.4035)
        assert result.continuation_probability == pytest.approx(0.8152)
        assert result.independent_trivial_probability == pytest.approx(0.628)

        as_dict = result.to_dict()
        assert as_dict["classifier"] == "ollaya"
        assert as_dict["probabilities"]["strong"] == pytest.approx(0.4035)
        assert as_dict["continuation_probability"] == pytest.approx(0.8152)
        assert as_dict["independent_trivial_probability"] == pytest.approx(
            0.628)
        assert as_dict["latency_ms"] == result.latency_ms

    def test_deterministic_result_dict_has_no_ollaya_evidence(self):
        result = classify_task(
            "rename the variable x to user_count in parser.py",
            TurnContext(),
        )
        as_dict = result.to_dict()
        assert "probabilities" not in as_dict
        assert "latency_ms" not in as_dict
        assert "continuation_probability" not in as_dict

    def test_ollaya_reasons_are_bounded_codes(self):
        for response in (BASIC_RESPONSE, CONTINUATION_RESPONSE,
                         confident_basic_response()):
            result = classify_task(
                "yes, go ahead",
                continuation_context(),
                ollaya=enabled_classifier(response=response),
            )
            for reason in result.reasons:
                assert reason in OLLAYA_REASON_CODES, reason


# ---------------------------------------------------------------------------
# Optional live smoke test — explicitly enabled only
# ---------------------------------------------------------------------------

LIVE_TEST_ENABLED = os.getenv("OLLAYA_LIVE_TEST", "") == "1"


@pytest.mark.skipif(
    not LIVE_TEST_ENABLED,
    reason="live Ollaya service required; set OLLAYA_LIVE_TEST=1 to run",
)
class TestLiveOllaya:
    """Schema-shape and latency smoke test against a running Ollaya.

    Asserts structure and bounded latency only — never semantic
    perfection (accuracy calibration is deferred to labeled fixtures,
    per the Task 1 gate decision).
    """

    def test_live_decide_schema_and_latency(self):
        classifier = OllayaClassifier(config=OllayaConfig(enabled=True))
        result = classifier.classify(
            "yes, go ahead", continuation_context()
        )
        assert result.classifier == "ollaya"
        assert result.ambiguous is False
        assert result.capability in (
            Capability.BASIC, Capability.NORMAL,
            Capability.STRONG, Capability.REASONING,
        )
        assert result.latency_ms is not None
        assert 0.0 < result.latency_ms < 10_000.0
        probabilities = dict(result.probabilities)
        assert 0.0 <= min(probabilities.values())
        assert max(probabilities.values()) <= 1.0
        print(f"\nlive ollaya decide: latency_ms={result.latency_ms:.1f} "
              f"capability={result.capability.name.lower()} "
              f"continuation={result.continuation_probability}")
