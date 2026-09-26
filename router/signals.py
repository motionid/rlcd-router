"""
Deterministic runtime signals extracted from the prompt and session
state.  These are *facts*, not model predictions, and are often more
reliable than what the classifier infers.

The classifier sees the prompt text.  The policy sees both the
classifier output *and* these signals.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from router.config import Capability

# ---------------------------------------------------------------------------
# Bounded previous-prompt state
#
# The extension sends compact session state; the previous prompt is capped
# at a configured character limit so one gigantic prompt can never bloat
# every subsequent turn's routing input.
# ---------------------------------------------------------------------------

PREVIOUS_PROMPT_CHAR_LIMIT = int(
    os.getenv("ROUTER_PREVIOUS_PROMPT_CHAR_LIMIT", "2048")
)


@dataclass(frozen=True)
class TurnContext:
    """Compact, bounded per-session routing state for one turn.

    Supplied by the PI extension; it is *not* the transcript.  ``task_failures``
    counts consecutive *task* failures only — provider availability failures
    (rate limits, outages, cooldowns) are tracked elsewhere and must never
    inflate this counter.
    """

    previous_prompt: str = ""
    previous_capability: Capability | None = None
    previous_model: str = ""
    previous_route_reason: tuple[str, ...] = ()
    task_depth: int = 0
    files_involved: int = 0
    task_failures: int = 0

    def __post_init__(self) -> None:
        capped = self.previous_prompt[:PREVIOUS_PROMPT_CHAR_LIMIT]
        if capped != self.previous_prompt:
            object.__setattr__(self, "previous_prompt", capped)

    def to_dict(self) -> dict[str, Any]:
        return {
            "previous_prompt": self.previous_prompt[:PREVIOUS_PROMPT_CHAR_LIMIT],
            "previous_capability": (
                self.previous_capability.name.lower()
                if self.previous_capability is not None
                else None
            ),
            "previous_model": self.previous_model,
            "previous_route_reason": list(self.previous_route_reason),
            "task_depth": self.task_depth,
            "files_involved": self.files_involved,
            "task_failures": self.task_failures,
        }


@dataclass
class RuntimeSignals:
    """Observable facts about the current request and session."""

    # --- prompt geometry ---
    prompt_chars: int = 0
    estimated_tokens: int = 0
    attached_file_count: int = 0

    # --- repository context ---
    repo_size_mb: float = 0.0
    repo_files_referenced: int = 0
    has_git_diff: bool = False
    has_test_command: bool = False

    # --- session history ---
    prior_attempts: int = 0
    prior_failures: int = 0
    tool_calls_so_far: int = 0

    # --- user / deadline hints ---
    deadline: str = "normal"  # interactive | normal | asynchronous
    user_hint: str = ""       # e.g. "cheap", "fast", "careful"

    # --- derived flags (computed, not set externally) ---
    is_long_context: bool = field(default=False, init=False)
    has_file_attachments: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.is_long_context = self.estimated_tokens > 8_000
        self.has_file_attachments = self.attached_file_count > 0

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Heuristic token estimation (≈4 chars per token for English code)
# ---------------------------------------------------------------------------

_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Rough token count; good enough for routing thresholds."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------
# Signal extraction from a raw PI routing request
# ---------------------------------------------------------------------------

_DIFF_RE = re.compile(r"^diff\s|^\+{3}\s|^@@\s", re.MULTILINE)
_TEST_RE = re.compile(
    r"\b(python\s+-m\s+pytest|npm\s+test|make\s+test|cargo\s+test"
    r"|go\s+test|bundle\s+exec\s+rspec|./gradlew\s+test)\b",
    re.IGNORECASE,
)
_FILE_PATH_RE = re.compile(
    r"(?:^|\s)([\w./\\-]+\.(?:py|js|ts|tsx|jsx|rs|go|java|c|cpp|h|hpp"
    r"|rb|sh|md|json|yaml|yml|toml|html|css))",
)


def measure_repo_size_mb(repo_root: str) -> float:
    """Opt-in repository size measurement (MB).

    Deliberately *not* called from the default per-turn path.  Callers that
    want repo size as a signal must measure it out-of-band (e.g. once per
    session) and supply the fact via ``extract_signals(repo_size_mb=...)``.
    """
    if not repo_root or not os.path.isdir(repo_root):
        return 0.0
    total = 0
    for dirpath, dirnames, filenames in os.walk(repo_root):
        # skip hidden dirs
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for f in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return round(total / (1024 * 1024), 2)


def extract_signals(
    prompt: str,
    *,
    attached_files: Optional[list[str]] = None,
    repo_size_mb: float = 0.0,
    prior_attempts: int = 0,
    prior_failures: int = 0,
    tool_calls_so_far: int = 0,
    deadline: str = "normal",
    user_hint: str = "",
    repo_root: Optional[str] = None,
) -> RuntimeSignals:
    """Build a ``RuntimeSignals`` from raw inputs.

    Parameters
    ----------
    prompt:
        The full user prompt (or system+user combined text).
    attached_files:
        Explicit list of file paths the caller attached.
    repo_size_mb:
        Caller-supplied repository size fact.  The default turn hot path
        no longer walks the filesystem; callers that care about repo size
        must measure it out-of-band via :func:`measure_repo_size_mb` (or any
        other source) and pass the number here.
    prior_attempts / prior_failures:
        Counters from the current PI session.
    tool_calls_so_far:
        How many tool calls have already happened this turn.
    deadline:
        ``interactive``, ``normal``, or ``asynchronous``.
    user_hint:
        Free-text hint like ``"cheap"`` or ``"fast"``.
    repo_root:
        Deprecated, accepted for backward compatibility with existing
        callers but *ignored* — measuring it here put ``os.walk()`` on the
        per-turn hot path.
    """
    attached = attached_files or []

    # file references inside prompt
    referenced = set(_FILE_PATH_RE.findall(prompt))

    return RuntimeSignals(
        prompt_chars=len(prompt),
        estimated_tokens=estimate_tokens(prompt),
        attached_file_count=len(attached),
        repo_size_mb=round(float(repo_size_mb), 2),
        repo_files_referenced=len(referenced),
        has_git_diff=bool(_DIFF_RE.search(prompt)),
        has_test_command=bool(_TEST_RE.search(prompt)),
        prior_attempts=prior_attempts,
        prior_failures=prior_failures,
        tool_calls_so_far=tool_calls_so_far,
        deadline=deadline,
        user_hint=user_hint,
    )
