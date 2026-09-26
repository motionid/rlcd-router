"""
Optional Ollaya ambiguity-classification adapter.

Ollaya (https://ollaya.dev) is a small local router model that answers
narrow, pre-declared classification questions over a bounded state
payload.  This adapter wires it into the router behind three binding
rules from the Task 1 live verification (Ollaya 0.6.1, Apple Silicon):

1. **Disabled by default.**  Structure and latency gates passed
   (~205 ms warm decide), but semantic accuracy is not calibrated
   against labeled fixtures, so the adapter is strictly opt-in
   (``ROUTER_OLLAYA_ENABLED=1``).
2. **Ambiguity-only.**  ``classify_task`` is deterministic-first: it
   returns the deterministic classification immediately whenever that
   classification is unambiguous.  Ollaya is consulted only for turns
   the deterministic classifier marks ``ambiguous`` — obvious cases
   never pay the ~200 ms or the ~1.4 GiB resident model.
3. **Never a single point of failure, never a spending decision.**
   Disabled configuration, timeouts, connection errors, and malformed
   responses all raise :class:`OllayaError` subclasses that
   ``classify_task`` converts into the conservative deterministic
   fallback (subscription-safe NORMAL or inherited continuation
   capability).  The adapter classifies tasks only; it never selects
   providers, models, access classes, or spending policy.

All Ollaya wire-format knowledge — the ``POST /api/decide`` payload,
the three approved questions, and the answer schema — is isolated in
this file.  The rest of the router only ever sees router-owned
:class:`~router.deterministic.TaskClassification` values.

The third question, ``is_independent_trivial_task``, is recorded as
diagnostic telemetry only: live evidence showed it contradicting
``is_task_continuation`` in the same forward pass (0.628 vs 0.8152),
so it must never alone permit a local/basic downgrade until labeled
calibration validates it.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from router.config import (
    Capability,
    OLLAYA_CONFIG,
    OllayaConfig,
)
from router.deterministic import (
    TaskClassification,
    TaskRelationship,
    classify_deterministically,
)
from router.signals import TurnContext

# ---------------------------------------------------------------------------
# Bounded state payload
# ---------------------------------------------------------------------------

# Hard bound on each state field sent to Ollaya.  ``TurnContext`` already
# caps ``previous_prompt`` at PREVIOUS_PROMPT_CHAR_LIMIT; this is defense
# in depth so no unbounded prompt can ever reach the wire.
OLLAYA_STATE_CHAR_LIMIT = int(
    os.getenv("ROUTER_OLLAYA_STATE_CHAR_LIMIT", "2048")
)

# ---------------------------------------------------------------------------
# The three approved questions (verbatim from the design spec §11 and the
# Task 1 live verification — do not edit without a reviewed calibration
# round; answers are sensitive to the exact wording).
# ---------------------------------------------------------------------------

APPROVED_QUESTIONS: dict[str, dict[str, Any]] = {
    "required_capability": {
        "type": "choice",
        "instructions": (
            "Which model capability does answering the current turn "
            "require?"
        ),
        "criteria": {
            "basic": (
                "A trivial question answerable in one sentence with no "
                "tools or context"
            ),
            "normal": (
                "Routine work such as short edits, lookups, or drafting"
            ),
            "strong": (
                "Substantial multi-file or multi-step engineering work"
            ),
            "reasoning": (
                "Deep analysis, complex debugging, or hard architectural "
                "decisions"
            ),
        },
    },
    "is_task_continuation": {
        "type": "noul",
        "instructions": (
            "The current turn continues the previous task or conversation."
        ),
        "criteria": {
            "true": (
                "The current turn continues or builds on the previous task"
            ),
            "false": (
                "The current turn starts a new, independent request"
            ),
        },
    },
    "is_independent_trivial_task": {
        "type": "noul",
        "instructions": (
            "The current turn is a standalone trivial task needing no "
            "prior context."
        ),
        "criteria": {
            "true": "The current turn is independent and trivial",
            "false": (
                "The current turn needs context or is not trivial"
            ),
        },
    },
}

# ---------------------------------------------------------------------------
# Bounded reason codes — every Ollaya ``reasons`` entry comes from this set.
# ---------------------------------------------------------------------------

OLLAYA_REASON_CODES: frozenset[str] = frozenset({
    "ollaya_classified",
    "ollaya_continuation",
    "ollaya_inherited_capability",
    "ollaya_missing_history",
    "ollaya_basic_confident",
    "ollaya_basic_below_threshold",
    # Diagnostic telemetry only — never gates routing (Task 1 ruling).
    "ollaya_independent_trivial_diagnostic",
})

_CHOICE_TO_CAPABILITY: dict[str, Capability] = {
    "basic": Capability.BASIC,
    "normal": Capability.NORMAL,
    "strong": Capability.STRONG,
    "reasoning": Capability.REASONING,
}


# ---------------------------------------------------------------------------
# Errors — all adapter failures are router-owned types
# ---------------------------------------------------------------------------

class OllayaError(Exception):
    """Base class for every recoverable Ollaya adapter failure."""


class OllayaDisabledError(OllayaError):
    """The adapter is disabled (configuration), not malfunctioning."""


class OllayaUnavailableError(OllayaError):
    """Transport-level failure: connection refused, timeout, HTTP error."""


class OllayaParseError(OllayaError):
    """The response body does not match the verified answer schema."""


# ---------------------------------------------------------------------------
# Transport
#
# A transport is an injectable callable (url, payload, timeout_ms) ->
# parsed JSON dict.  The default uses only the standard library — no new
# dependency is introduced for an optional feature (the plan forbids an
# Ollaya SDK; stdlib urllib suffices for one small POST per ambiguous
# turn).
# ---------------------------------------------------------------------------

Transport = Callable[[str, dict, int], dict]


def _default_transport(url: str, payload: dict, timeout_ms: int) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request, timeout=timeout_ms / 1000.0
        ) as response:
            raw = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise OllayaUnavailableError(
            f"Ollaya decide request failed: {exc!r}"
        ) from exc
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OllayaParseError(
            f"Ollaya returned a non-JSON body: {exc!r}"
        ) from exc
    if not isinstance(parsed, dict):
        raise OllayaParseError(
            f"Ollaya response is not a JSON object: {type(parsed).__name__}"
        )
    return parsed


# ---------------------------------------------------------------------------
# Response parsing (all wire-format knowledge lives below this line)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ParsedDecision:
    capability: Capability
    confidence: float
    probabilities: tuple[tuple[str, float], ...]
    continuation: float
    independent_trivial: float | None


def _require_mapping(value: Any, where: str) -> dict:
    if not isinstance(value, dict):
        raise OllayaParseError(f"{where} is not an object")
    return value


def _require_prob(value: Any, where: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise OllayaParseError(f"{where} is not a number")
    return float(value)


def _parse_response(response: Mapping[str, Any]) -> _ParsedDecision:
    answers = _require_mapping(response.get("answers"), "answers")

    capability_answer = _require_mapping(
        answers.get("required_capability"),
        "answers.required_capability",
    )
    choice = capability_answer.get("choice")
    if choice not in _CHOICE_TO_CAPABILITY:
        raise OllayaParseError(
            f"unknown required_capability choice: {choice!r}"
        )
    confidence = _require_prob(
        capability_answer.get("confidence"),
        "answers.required_capability.confidence",
    )
    probabilities_raw = _require_mapping(
        capability_answer.get("probabilities"),
        "answers.required_capability.probabilities",
    )
    probabilities = tuple(
        (str(label), _require_prob(prob, f"probabilities[{label!r}]"))
        for label, prob in probabilities_raw.items()
    )

    continuation_answer = _require_mapping(
        answers.get("is_task_continuation"),
        "answers.is_task_continuation",
    )
    continuation = _require_prob(
        continuation_answer.get("noul"),
        "answers.is_task_continuation.noul",
    )

    # Diagnostic-only question: tolerate absence, never gate on it.
    independent_answer = answers.get("is_independent_trivial_task")
    independent_trivial: float | None = None
    if isinstance(independent_answer, dict):
        try:
            independent_trivial = _require_prob(
                independent_answer.get("noul"),
                "answers.is_independent_trivial_task.noul",
            )
        except OllayaParseError:
            independent_trivial = None

    return _ParsedDecision(
        capability=_CHOICE_TO_CAPABILITY[choice],
        confidence=confidence,
        probabilities=probabilities,
        continuation=continuation,
        independent_trivial=independent_trivial,
    )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class OllayaClassifier:
    """Classify ambiguous turns through a local Ollaya ``/api/decide``.

    ``transport`` is injectable so tests never touch the network.  The
    default transport performs one POST with the configured timeout and
    sends ``keep_alive: -1`` — Task 1 measured ~2.17 s cold vs ~205 ms
    warm, so an enabled adapter keeps the model resident.
    """

    def __init__(
        self,
        config: OllayaConfig | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.config = config if config is not None else OLLAYA_CONFIG
        self._transport = (
            transport if transport is not None else _default_transport
        )

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    # -- payload construction -----------------------------------------------

    def _build_payload(self, prompt: str, context: TurnContext) -> dict:
        return {
            "model": self.config.model,
            # keep_alive: -1 keeps the model resident: cold decide was
            # ~2.17 s vs ~205 ms warm (Task 1 ruling).
            "keep_alive": -1,
            "state": {
                "previous_context": context.previous_prompt[
                    :OLLAYA_STATE_CHAR_LIMIT
                ],
                "current_turn": prompt[:OLLAYA_STATE_CHAR_LIMIT],
            },
            "questions": APPROVED_QUESTIONS,
        }

    # -- classification ------------------------------------------------------

    def classify(
        self, prompt: str, context: TurnContext
    ) -> TaskClassification:
        """Classify one turn; raise :class:`OllayaError` on any failure.

        Callers that cannot tolerate a raise should use
        :func:`classify_task`, which converts every adapter failure into
        the conservative deterministic fallback.
        """
        if not self.config.enabled:
            raise OllayaDisabledError(
                "Ollaya classification is disabled "
                "(set ROUTER_OLLAYA_ENABLED=1 to enable)"
            )

        url = f"{self.config.base_url.rstrip('/')}/api/decide"
        payload = self._build_payload(prompt, context)

        start = time.perf_counter()
        try:
            response = self._transport(url, payload, self.config.timeout_ms)
        except OllayaError:
            raise
        except Exception as exc:  # noqa: BLE001
            # Any transport failure (injected or stdlib) degrades to a
            # router-owned error; Ollaya is never a single point of
            # failure and never crashes routing.
            raise OllayaUnavailableError(
                f"Ollaya transport failed: {exc!r}"
            ) from exc
        latency_ms = (time.perf_counter() - start) * 1000.0

        parsed = _parse_response(response)
        return self._to_classification(parsed, context, latency_ms)

    def _to_classification(
        self,
        parsed: _ParsedDecision,
        context: TurnContext,
        latency_ms: float,
    ) -> TaskClassification:
        """Map a parsed decision onto router-owned types.

        Conservative mapping rules (binding rulings from Task 1):

        - ``is_independent_trivial_task`` is telemetry only; it never
          alone permits a local/basic downgrade.
        - A basic downgrade requires the basic *probability* to clear
          ``basic_threshold`` and the turn to not be a continuation.
          Otherwise the turn lands on subscription-safe NORMAL.
        - A confirmed continuation inherits the previous capability
          (floored at NORMAL when history is missing).
        """
        capability = parsed.capability
        reasons: list[str] = ["ollaya_classified"]

        if parsed.continuation >= self.config.continuation_threshold:
            relationship = TaskRelationship.CONTINUATION
            reasons.append("ollaya_continuation")
            if context.previous_capability is not None:
                if capability < context.previous_capability:
                    capability = context.previous_capability
                reasons.append("ollaya_inherited_capability")
            else:
                capability = max(capability, Capability.NORMAL)
                reasons.append("ollaya_missing_history")
        else:
            if capability is Capability.BASIC:
                relationship = TaskRelationship.INDEPENDENT
                basic_prob = dict(parsed.probabilities).get("basic", 0.0)
                if basic_prob >= self.config.basic_threshold:
                    reasons.append("ollaya_basic_confident")
                else:
                    # Uncalibrated "basic" answers stay subscription-safe.
                    capability = Capability.NORMAL
                    reasons.append("ollaya_basic_below_threshold")
            else:
                relationship = TaskRelationship.NEW_SUBSTANTIAL

        if parsed.independent_trivial is not None:
            # Recorded for telemetry/calibration only — never a gate.
            reasons.append("ollaya_independent_trivial_diagnostic")

        return TaskClassification(
            capability=capability,
            confidence=max(0.0, min(parsed.confidence, 1.0)),
            relationship=relationship,
            reasons=tuple(reasons),
            ambiguous=False,
            classifier="ollaya",
            probabilities=parsed.probabilities,
            continuation_probability=parsed.continuation,
            independent_trivial_probability=parsed.independent_trivial,
            latency_ms=latency_ms,
        )


# ---------------------------------------------------------------------------
# Deterministic-first orchestration
# ---------------------------------------------------------------------------

_default_classifier: OllayaClassifier | None = None


def _default_ollaya() -> OllayaClassifier | None:
    """The process-wide default classifier, or None when disabled.

    Constructed lazily so merely importing this module never opens a
    socket; tests that never set ``ROUTER_OLLAYA_ENABLED`` stay offline.
    """
    global _default_classifier
    if not OLLAYA_CONFIG.enabled:
        return None
    if _default_classifier is None:
        _default_classifier = OllayaClassifier(OLLAYA_CONFIG)
    return _default_classifier


def classify_task(
    prompt: str,
    context: TurnContext,
    ollaya: OllayaClassifier | None = None,
) -> TaskClassification:
    """Classify a turn: deterministic first, Ollaya only for ambiguity.

    1. Run the deterministic classifier.  Unambiguous results return
       immediately — obvious work never pays model latency.
    2. Ambiguous turns consult Ollaya when an enabled adapter exists
       (an explicit ``ollaya`` argument, otherwise the env-configured
       default, otherwise none).
    3. Every adapter failure — disabled, timeout, unavailable,
       malformed — falls back to the deterministic classification,
       which for ambiguous turns is already the conservative
       subscription-safe choice (NORMAL floor or inherited
       continuation capability, never paid access, never basic).

    Ollaya never selects providers, models, access classes, or
    spending policy; it only refines the *task capability* label that
    the downstream policy consumes.
    """
    base = classify_deterministically(prompt, context)
    if not base.ambiguous:
        return base

    if ollaya is None:
        ollaya = _default_ollaya()
    if ollaya is None or not ollaya.enabled:
        return base

    try:
        return ollaya.classify(prompt, context)
    except OllayaError:
        return base
