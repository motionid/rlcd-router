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


def extract_signals(
    prompt: str,
    *,
    attached_files: Optional[list[str]] = None,
    repo_root: Optional[str] = None,
    prior_attempts: int = 0,
    prior_failures: int = 0,
    tool_calls_so_far: int = 0,
    deadline: str = "normal",
    user_hint: str = "",
) -> RuntimeSignals:
    """Build a ``RuntimeSignals`` from raw inputs.

    Parameters
    ----------
    prompt:
        The full user prompt (or system+user combined text).
    attached_files:
        Explicit list of file paths the caller attached.
    repo_root:
        If provided, the repo size is measured in MB.
    prior_attempts / prior_failures:
        Counters from the current PI session.
    tool_calls_so_far:
        How many tool calls have already happened this turn.
    deadline:
        ``interactive``, ``normal``, or ``asynchronous``.
    user_hint:
        Free-text hint like ``"cheap"`` or ``"fast"``.
    """
    attached = attached_files or []

    # repo size
    repo_mb = 0.0
    if repo_root and os.path.isdir(repo_root):
        total = 0
        for dirpath, dirnames, filenames in os.walk(repo_root):
            # skip hidden dirs
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for f in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, f))
                except OSError:
                    pass
        repo_mb = total / (1024 * 1024)

    # file references inside prompt
    referenced = set(_FILE_PATH_RE.findall(prompt))

    return RuntimeSignals(
        prompt_chars=len(prompt),
        estimated_tokens=estimate_tokens(prompt),
        attached_file_count=len(attached),
        repo_size_mb=round(repo_mb, 2),
        repo_files_referenced=len(referenced),
        has_git_diff=bool(_DIFF_RE.search(prompt)),
        has_test_command=bool(_TEST_RE.search(prompt)),
        prior_attempts=prior_attempts,
        prior_failures=prior_failures,
        tool_calls_so_far=tool_calls_so_far,
        deadline=deadline,
        user_hint=user_hint,
    )
