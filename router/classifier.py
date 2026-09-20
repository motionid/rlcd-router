"""
Two-stage RLCD classifier with confidence handling.

Wraps the existing RLCD engine to produce a ``ClassificationResult``
that the policy engine can consume.  Implements the fast→smart cascade:

  1.  Run RLCD-fast (Qwen2.5 1.5B, ~70-500 ms).
  2.  If confidence is high and no critical field is uncertain, accept.
  3.  Otherwise, run RLCD-smart (Qwen3-Coder-30B-A3B, ~0.5-1.6 s).
  4.  If smart confidence is still low, set ``abstain=True``.

The classifier does NOT call models directly — it delegates to a
callable ``run_fn`` so it stays testable without loading MLX models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from router.router_schema import (
    ROUTER_SCHEMA,
    FAST_ROUTER_SCHEMA,
    FAST_SCHEMA_DEFAULTS,
    CRITICAL_FIELDS,
    FAST_CRITICAL_FIELDS,
    validate_router_output,
)
from router.signals import RuntimeSignals
from router.config import CONFIDENCE_THRESHOLDS


# ---------------------------------------------------------------------------
# Classification result
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """Output of the classifier cascade."""

    fields: dict[str, str]                          # field → chosen value
    field_confidence: dict[str, float]              # field → probability
    overall_confidence: float
    critical_confidence: float
    abstain: bool
    classifier_used: str                            # "rlcd_fast" | "rlcd_smart" | "bypass"
    latency_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fields": dict(self.fields),
            "field_confidence": {k: round(v, 4) for k, v in self.field_confidence.items()},
            "overall_confidence": round(self.overall_confidence, 4),
            "critical_confidence": round(self.critical_confidence, 4),
            "abstain": self.abstain,
            "classifier_used": self.classifier_used,
            "latency_ms": round(self.latency_ms, 2),
            "errors": list(self.errors),
        }


# ---------------------------------------------------------------------------
# RLCD output parser
# ---------------------------------------------------------------------------

def _parse_rlcd_output(
    raw: dict[str, Any],
) -> tuple[dict[str, str], dict[str, float], list[str]]:
    """Convert raw RLCD engine output into (fields, confidences, errors).

    The RLCD engine returns ``parsed_json`` with per-field dicts:
        {"field_name": {"value": "enum_val", "prob": 0.95}}
    """
    parsed = raw.get("parsed_json", {})
    fields: dict[str, str] = {}
    confs: dict[str, float] = {}
    errors: list[str] = []

    for field_name, entry in parsed.items():
        if isinstance(entry, dict):
            val = str(entry.get("value", ""))
            prob = float(entry.get("prob", 0.0))
        else:
            val = str(entry)
            prob = 0.0

        fields[field_name] = val
        confs[field_name] = prob

    # Validate against schema
    validated, val_errors = validate_router_output(fields)
    errors.extend(val_errors)

    # Only keep validated fields
    fields = validated

    return fields, confs, errors


def _compute_critical_confidence(confs: dict[str, float]) -> float:
    """Minimum confidence among critical fields."""
    vals = [confs.get(f, 0.0) for f in CRITICAL_FIELDS if f in confs]
    return min(vals) if vals else 0.0


def _compute_fast_critical_confidence(confs: dict[str, float]) -> float:
    """Minimum confidence among fast-critical fields."""
    vals = [confs.get(f, 0.0) for f in FAST_CRITICAL_FIELDS if f in confs]
    return min(vals) if vals else 0.0


def _compute_overall_confidence(confs: dict[str, float]) -> float:
    """Mean confidence across all fields."""
    if not confs:
        return 0.0
    return sum(confs.values()) / len(confs)


def _apply_fast_heuristics(
    merged: dict[str, str],
    fast_fields: dict[str, str],
) -> dict[str, str]:
    """Derive reasonable defaults for the 7 missing fields based on
    the 4 fast-classified fields.

    These are deterministic heuristics — the smart classifier will
    override them if it runs.
    """
    coding = fast_fields.get("coding", "none")
    repo = fast_fields.get("repo_access", "none")
    fresh = fast_fields.get("fresh_information", "none")
    privacy = fast_fields.get("privacy", "ordinary")

    # task_type
    if coding in ("substantial", "dominant"):
        merged.setdefault("task_type", "code_edit")
    elif fresh == "required":
        merged.setdefault("task_type", "research")
    elif coding == "none":
        merged.setdefault("task_type", "explanation")

    # tool_use
    if fresh == "required":
        merged["tool_use"] = "browser"
    elif repo == "modify":
        merged["tool_use"] = "shell"
    elif coding in ("substantial", "dominant"):
        merged["tool_use"] = "read_only"

    # output_contract
    if coding in ("substantial", "dominant") and repo in ("inspect", "modify"):
        merged["output_contract"] = "exact_patch"
    elif coding == "none" and fresh == "none":
        merged["output_contract"] = "freeform"

    # reasoning_depth
    if coding in ("substantial", "dominant"):
        merged["reasoning_depth"] = "multi_step"

    # failure_cost
    if privacy in ("sensitive", "local_only"):
        merged["failure_cost"] = "high"

    return merged


# ---------------------------------------------------------------------------
# RouterClassifier
# ---------------------------------------------------------------------------

# Type alias for the RLCD call function.
# Signature: (context: str, schema: dict) -> dict[str, Any]
RlcdCallFn = Callable[[str, dict[str, Any]], dict[str, Any]]


class RouterClassifier:
    """Two-stage RLCD classifier with confidence-gated escalation.

    Parameters
    ----------
    run_fast:
        Callable that invokes RLCD-fast and returns the raw engine dict.
    run_smart:
        Callable that invokes RLCD-smart and returns the raw engine dict.
        May be ``None`` to disable the second stage.
    thresholds:
        Override confidence thresholds (defaults to global config).
    """

    def __init__(
        self,
        run_fast: RlcdCallFn,
        run_smart: Optional[RlcdCallFn] = None,
        thresholds=None,
    ):
        self._run_fast = run_fast
        self._run_smart = run_smart
        self._thresholds = thresholds or CONFIDENCE_THRESHOLDS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        prompt: str,
        signals: RuntimeSignals,
    ) -> ClassificationResult:
        """Run the full cascade and return a classification.

        Both stages use the full 11-field ROUTER_SCHEMA.  The fast
        stage (Coder-30B-A3B) handles classification directly.  The
        smart stage re-classifies only when the fast stage is
        uncertain or blocked by prior failures.

        The slim FAST_ROUTER_SCHEMA is reserved for future use with
        smaller models that can't reliably classify 11 fields.
        """

        # Stage 1: primary classifier
        fast_raw = self._run_fast(prompt, ROUTER_SCHEMA)
        fast_fields, fast_confs, fast_errors = _parse_rlcd_output(fast_raw)
        fast_overall = _compute_overall_confidence(fast_confs)
        fast_critical = _compute_critical_confidence(fast_confs)
        fast_latency = float(fast_raw.get("elapsed_ms", 0.0))

        # Accept fast if confident enough
        if self._accept_fast(fast_overall, fast_critical, signals):
            return ClassificationResult(
                fields=fast_fields,
                field_confidence=fast_confs,
                overall_confidence=fast_overall,
                critical_confidence=fast_critical,
                abstain=False,
                classifier_used="rlcd_fast",
                latency_ms=fast_latency,
                errors=fast_errors,
            )

        # Stage 2: re-classify (if available and not blocked)
        if self._run_smart is not None and not self._should_skip_smart(signals):
            smart_raw = self._run_smart(prompt, ROUTER_SCHEMA)
            smart_fields, smart_confs, smart_errors = _parse_rlcd_output(smart_raw)
            smart_overall = _compute_overall_confidence(smart_confs)
            smart_critical = _compute_critical_confidence(smart_confs)
            smart_latency = float(smart_raw.get("elapsed_ms", 0.0))

            abstain = smart_overall < self._thresholds.smart_bypass_floor

            return ClassificationResult(
                fields=smart_fields,
                field_confidence=smart_confs,
                overall_confidence=smart_overall,
                critical_confidence=smart_critical,
                abstain=abstain,
                classifier_used="rlcd_smart",
                latency_ms=fast_latency + smart_latency,
                errors=smart_errors,
            )

        # No smart classifier or smart skipped → fast result with abstain flag
        abstain = fast_overall < self._thresholds.conservative_floor
        return ClassificationResult(
            fields=fast_fields,
            field_confidence=fast_confs,
            overall_confidence=fast_overall,
            critical_confidence=fast_critical,
            abstain=abstain,
            classifier_used="rlcd_fast",
            latency_ms=fast_latency,
            errors=fast_errors,
        )

    # ------------------------------------------------------------------
    # Internal decisions
    # ------------------------------------------------------------------

    def _accept_fast(
        self,
        overall: float,
        critical: float,
        signals: RuntimeSignals,
    ) -> bool:
        """Should the fast classifier result be trusted?"""
        t = self._thresholds

        if overall < t.fast_accept_overall:
            return False

        if critical < t.fast_accept_critical:
            return False

        # If there have already been failures, don't trust the fast path
        if signals.prior_failures > 0:
            return False

        return True

    def _should_skip_smart(self, signals: RuntimeSignals) -> bool:
        """Skip the smart classifier in edge cases."""
        # If the user explicitly wants interactive speed, skip smart
        if signals.deadline == "interactive":
            return True
        return False


# ---------------------------------------------------------------------------
# Convenience: build a bypass result (no model calls)
# ---------------------------------------------------------------------------

def bypass_classification(reason: str = "no_classifier_available") -> ClassificationResult:
    """Return a classification that forces the policy to use a safe default."""
    return ClassificationResult(
        fields={
            "task_type": "unknown",
            "complexity": "medium",
            "coding": "none",
            "reasoning_depth": "direct",
            "fresh_information": "none",
            "repo_access": "none",
            "tool_use": "none",
            "privacy": "ordinary",
            "latency": "normal",
            "failure_cost": "low",
            "output_contract": "freeform",
        },
        field_confidence={},
        overall_confidence=0.0,
        critical_confidence=0.0,
        abstain=True,
        classifier_used="bypass",
        latency_ms=0.0,
        errors=[reason],
    )
