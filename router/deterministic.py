"""
Deterministic turn-aware task classification.

This is the primary hot path: pure string/regex analysis of the prompt
plus bounded ``TurnContext`` facts.  No models, no filesystem access,
no network — every input is caller-supplied, so the same prompt and
context always produce the same classification.

Precedence rules (binding decisions):

1. Strong/reasoning signals outrank trivial/basic verbs — a prompt that
   says "rename this while debugging the race across the repository" is
   strong, never basic.
2. Prompt length is supporting evidence only; short prompts are *not*
   automatically basic or unambiguous.
3. Clear continuations ("fix it", "continue", ...) inherit the previous
   task's capability; explicitly standalone commit-message / README /
   summarize work is independent basic work even after a strong task.
4. Repeated *task* failures raise the required capability one level per
   configured threshold, capped at reasoning.  Provider availability
   failures are never represented by this counter.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum

from router.config import Capability
from router.signals import TurnContext

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Every TASK_FAILURE_ESCALATION_THRESHOLD consecutive task failures raise
# the required capability by one level (capped at reasoning).
TASK_FAILURE_ESCALATION_THRESHOLD = int(
    os.getenv("ROUTER_TASK_FAILURE_THRESHOLD", "2")
)

# Prompts shorter than this many words with no explicit signal are marked
# ambiguous (and conservatively floor at normal, never basic).
SHORT_PROMPT_WORDS = 6

# Context fact: this many distinct files involved makes the work multi-file.
MULTI_FILE_THRESHOLD = 3

# Context fact: this deep into a task, work is at least normal.
DEEP_TASK_DEPTH = 3


# ---------------------------------------------------------------------------
# Relationship
# ---------------------------------------------------------------------------

class TaskRelationship(str, Enum):
    """How this turn relates to the previous turn's task."""

    INDEPENDENT = "independent"          # standalone work (typ. trivial)
    CONTINUATION = "continuation"        # "fix it", "continue", ...
    NEW_SUBSTANTIAL = "new_substantial"  # a fresh, self-directed task
    AMBIGUOUS = "ambiguous"              # cannot tell from the evidence


# ---------------------------------------------------------------------------
# Signal patterns — ordered, bounded, stable
# ---------------------------------------------------------------------------

# Reasoning signals are checked first (highest precedence).
_REASONING_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("reasoning_signal_root_cause", re.compile(
        r"\broot[ -]?cause\b", re.IGNORECASE)),
    ("reasoning_signal_architecture_decision", re.compile(
        r"\barchitect\w* (decision|review|trade[- ]?offs?)\b"
        r"|\barchitectural (decision|review)\b", re.IGNORECASE)),
    ("reasoning_signal_deep_analysis", re.compile(
        r"\bdeep (dive|analysis|reasoning)\b|\binvestigate deeply\b"
        r"|\bdeeply (analyze|investigate)\b", re.IGNORECASE)),
    ("reasoning_signal_proof", re.compile(
        r"\bprove (that|correctness|the invariant)\b|\bformal(ly)? proof\b"
        r"|\bderivation of\b", re.IGNORECASE)),
    ("reasoning_signal_adversarial", re.compile(
        r"\badversarial\b|\bsecurity audit\b|\bthreat model\w*\b",
        re.IGNORECASE)),
)

# Strong signals outrank every basic pattern.
_STRONG_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("strong_signal_debug", re.compile(
        r"\b(debug\w*|troubleshoot\w*|diagnose\w*)\b", re.IGNORECASE)),
    ("strong_signal_concurrency", re.compile(
        r"\b(race condition\w*|data race|deadlock\w*|concurren\w+"
        r"|thread[- ]safety)\b", re.IGNORECASE)),
    ("strong_signal_repository_scope", re.compile(
        r"\b(across|throughout) the (repo\w*|repository|codebase|project)\b"
        r"|\b(whole|entire) (repo\w*|repository|codebase|project)\b",
        re.IGNORECASE)),
    ("strong_signal_refactor", re.compile(
        r"\brefactor\w*\b|\bre-?architect\w*\b|\bredesign\w*\b",
        re.IGNORECASE)),
    ("strong_signal_architecture", re.compile(
        r"\barchitect\w+\b", re.IGNORECASE)),
    ("strong_signal_multi_file", re.compile(
        r"\bmulti[- ]?file\b|\bacross (the )?files\b|\bseveral files\b",
        re.IGNORECASE)),
    ("strong_signal_flaky", re.compile(
        r"\b(heisenbug|intermittent\w*|flaky)\b", re.IGNORECASE)),
    ("strong_signal_nontrivial_bug", re.compile(
        r"\bnon[- ]?(trivial|obvious) bug\w*\b|\bcomplex bug\w*\b",
        re.IGNORECASE)),
)

# Bounded basic patterns — only reached when no strong/reasoning signal hit.
_BASIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("basic_signal_rename", re.compile(
        r"\brename\b", re.IGNORECASE)),
    ("basic_signal_typo", re.compile(
        r"\bfix (a |the )?typo\w*\b|\bcorrect the spelling\b"
        r"|\bmisspell\w*\b", re.IGNORECASE)),
    ("basic_signal_comment", re.compile(
        r"\b(add|update|fix|change|improve|edit) (a |the )?comment\w*\b",
        re.IGNORECASE)),
    ("basic_signal_readme", re.compile(
        r"\b(update|fix|improve|change|reword|edit|polish) (the )?"
        r"(readme\w*|docs?) (wording|text|paragraph|section|sentence)\b"
        r"|\b(readme|doc) wording\b", re.IGNORECASE)),
    ("basic_signal_format", re.compile(
        r"\bformat (the |this |these )?(code|file|data|output|list)\b"
        r"|\brun (the )?formatter\b|\bre-?indent\b", re.IGNORECASE)),
    ("basic_signal_summarize", re.compile(
        r"\bsummar(ize|ise|y) (this|the) (output|result\w*|log\w*|text"
        r"|error\w*|diff)\b", re.IGNORECASE)),
    ("basic_signal_commit_message", re.compile(
        r"\b(write|draft|compose|generate|suggest) (a |the )?"
        r"commit message\w*\b", re.IGNORECASE)),
)

# Continuation phrases — matched against the whole (cleaned) prompt.
_CONTINUATION_PHRASES: tuple[str, ...] = (
    "fix it", "fix that", "fix them", "apply that", "apply it",
    "continue", "keep going", "go ahead", "do it", "do that", "do so",
    "try again", "again", "and again",
    "why did that fail", "why did it fail", "why did that break",
    "why did it break", "what went wrong", "why did that happen",
    "why did that not work", "do the same for the rest",
    "same for the rest", "finish it", "finish that", "carry on",
)

# Bounded reason codes — every ``reasons`` entry comes from this set.
REASON_CODES: frozenset[str] = frozenset(
    code
    for patterns in (_REASONING_PATTERNS, _STRONG_PATTERNS, _BASIC_PATTERNS)
    for code, _ in patterns
) | {
    "continuation_phrase",
    "continuation_inherit_capability",
    "continuation_missing_history",
    "task_failure_escalation",
    "deep_task_context",
    "ambiguous_short_prompt",
    "default_normal",
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskClassification:
    """Deterministic classification of one turn's task.

    The deterministic classifier populates only the core fields.  The
    optional Ollaya adapter (``router/ollaya.py``) additionally records
    its evidence — capability probabilities, the two ``noul``
    probabilities, and measured latency — so telemetry can audit model
    assisted decisions.  The fields stay ``None``/empty on the pure
    deterministic hot path.
    """

    capability: Capability
    confidence: float
    relationship: TaskRelationship
    reasons: tuple[str, ...]
    ambiguous: bool
    classifier: str = "deterministic"
    # --- optional evidence from model-assisted classification ---
    # Frozen/hashable: probabilities are a tuple of (label, p) pairs.
    probabilities: tuple[tuple[str, float], ...] = ()
    continuation_probability: float | None = None
    independent_trivial_probability: float | None = None
    latency_ms: float | None = None

    def to_dict(self) -> dict:
        d = {
            "capability": self.capability.name.lower(),
            "confidence": self.confidence,
            "relationship": self.relationship.value,
            "reasons": list(self.reasons),
            "ambiguous": self.ambiguous,
            "classifier": self.classifier,
        }
        if self.probabilities:
            d["probabilities"] = dict(self.probabilities)
        if self.continuation_probability is not None:
            d["continuation_probability"] = self.continuation_probability
        if self.independent_trivial_probability is not None:
            d["independent_trivial_probability"] = (
                self.independent_trivial_probability
            )
        if self.latency_ms is not None:
            d["latency_ms"] = self.latency_ms
        return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _matching_codes(
    prompt: str,
    patterns: tuple[tuple[str, re.Pattern[str]], ...],
) -> tuple[str, ...]:
    return tuple(code for code, pattern in patterns if pattern.search(prompt))


def _continuation_match(prompt: str) -> str | None:
    """Return the matched continuation phrase, if the prompt is one.

    The whole prompt (stripped, lowercased, trailing punctuation removed)
    must equal a phrase or extend it with more words — "continue with the
    migration" is a continuation, but "can you continue to annoy me while
    renaming" is not forced into this branch by a buried keyword.
    """
    cleaned = prompt.strip().lower().rstrip("!.? \t")
    for phrase in _CONTINUATION_PHRASES:
        if cleaned == phrase or cleaned.startswith(phrase + " "):
            return phrase
    return None


def _word_count(prompt: str) -> int:
    return len(prompt.split())


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def classify_deterministically(
    prompt: str, context: TurnContext
) -> TaskClassification:
    """Classify a turn deterministically.

    Order of evaluation (each step only runs if earlier steps did not
    decide):

    1. reasoning signals (highest precedence)
    2. strong signals — including context facts (multi-file, deep task)
    3. bounded basic patterns (standalone trivial work)
    4. continuation phrases (inherit previous capability)
    5. defaults: unmatched work is normal; short vague prompts are
       ambiguous and *never* basic
    6. task-failure escalation applied on top, capped at reasoning
    """
    # -- 1. reasoning -------------------------------------------------------
    reasoning_codes = _matching_codes(prompt, _REASONING_PATTERNS)
    if reasoning_codes:
        return TaskClassification(
            capability=Capability.REASONING,
            confidence=0.9,
            relationship=TaskRelationship.NEW_SUBSTANTIAL,
            reasons=reasoning_codes,
            ambiguous=False,
        )

    # -- 2. strong ----------------------------------------------------------
    strong_codes = list(_matching_codes(prompt, _STRONG_PATTERNS))
    if context.files_involved >= MULTI_FILE_THRESHOLD:
        strong_codes.append("strong_signal_multi_file")
    if strong_codes:
        capability = Capability.STRONG
        if context.task_failures >= TASK_FAILURE_ESCALATION_THRESHOLD:
            strong_codes.append("task_failure_escalation")
            capability = Capability.REASONING
        return TaskClassification(
            capability=capability,
            confidence=0.85,
            relationship=TaskRelationship.NEW_SUBSTANTIAL,
            reasons=tuple(strong_codes),
            ambiguous=False,
        )

    # -- 3. bounded basic patterns ------------------------------------------
    basic_codes = _matching_codes(prompt, _BASIC_PATTERNS)
    if basic_codes:
        return TaskClassification(
            capability=Capability.BASIC,
            confidence=0.9,
            relationship=TaskRelationship.INDEPENDENT,
            reasons=basic_codes,
            ambiguous=False,
        )

    # -- 4. continuation -----------------------------------------------------
    continuation = _continuation_match(prompt)
    if continuation is not None:
        reasons = ["continuation_phrase"]
        if context.previous_capability is not None:
            capability = context.previous_capability
            reasons.append("continuation_inherit_capability")
            ambiguous = False
            confidence = 0.85
        else:
            # Nothing to inherit; ambiguity falls back conservatively to a
            # subscription-capable level (normal), never basic.
            capability = Capability.NORMAL
            reasons.append("continuation_missing_history")
            ambiguous = True
            confidence = 0.5
        if (
            context.task_failures >= TASK_FAILURE_ESCALATION_THRESHOLD
            and capability < Capability.REASONING
        ):
            reasons.append("task_failure_escalation")
            capability = Capability(capability + 1)
        if context.task_depth >= DEEP_TASK_DEPTH:
            reasons.append("deep_task_context")
        return TaskClassification(
            capability=capability,
            confidence=confidence,
            relationship=TaskRelationship.CONTINUATION,
            reasons=tuple(reasons),
            ambiguous=ambiguous,
        )

    # -- 5. defaults --------------------------------------------------------
    reasons: list[str] = []
    capability = Capability.NORMAL
    ambiguous = False
    confidence = 0.7
    relationship = TaskRelationship.NEW_SUBSTANTIAL

    if _word_count(prompt) <= SHORT_PROMPT_WORDS:
        # Short and no explicit signal: ambiguous, conservatively normal.
        ambiguous = True
        confidence = 0.4
        relationship = TaskRelationship.AMBIGUOUS
        reasons.append("ambiguous_short_prompt")
    else:
        reasons.append("default_normal")

    if context.task_depth >= DEEP_TASK_DEPTH:
        reasons.append("deep_task_context")

    # -- 6. task-failure escalation (capped at reasoning) --------------------
    if context.task_failures >= TASK_FAILURE_ESCALATION_THRESHOLD:
        levels = context.task_failures // TASK_FAILURE_ESCALATION_THRESHOLD
        capability = Capability(
            min(int(capability) + levels, int(Capability.REASONING))
        )
        reasons.append("task_failure_escalation")
        if capability == Capability.REASONING:
            ambiguous = False

    return TaskClassification(
        capability=capability,
        confidence=confidence,
        relationship=relationship,
        reasons=tuple(reasons),
        ambiguous=ambiguous,
    )
