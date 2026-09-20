"""
Tests for runtime signal extraction.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.signals import extract_signals, estimate_tokens, RuntimeSignals


class TestEstimateTokens:
    def test_short_text(self):
        assert estimate_tokens("hello") == 1

    def test_longer_text(self):
        text = "word " * 400  # 2000 chars
        assert estimate_tokens(text) == 500

    def test_minimum_one(self):
        assert estimate_tokens("") == 1


class TestExtractSignals:
    def test_basic_extraction(self):
        signals = extract_signals("Hello world, fix the bug")
        assert signals.prompt_chars == 24
        assert signals.estimated_tokens == 6
        assert signals.has_git_diff is False
        assert signals.has_test_command is False

    def test_detects_diff(self):
        prompt = "Apply this diff:\ndiff --git a/foo.py b/foo.py\n+new line"
        signals = extract_signals(prompt)
        assert signals.has_git_diff is True

    def test_detects_test_command(self):
        prompt = "Run python -m pytest to verify"
        signals = extract_signals(prompt)
        assert signals.has_test_command is True

    def test_detects_npm_test(self):
        prompt = "Run npm test to check"
        signals = extract_signals(prompt)
        assert signals.has_test_command is True

    def test_counts_file_references(self):
        prompt = "Look at src/main.py and tests/test_auth.py"
        signals = extract_signals(prompt)
        assert signals.repo_files_referenced >= 2

    def test_attached_files(self):
        signals = extract_signals(
            "Review this",
            attached_files=["a.py", "b.py", "c.py"],
        )
        assert signals.attached_file_count == 3
        assert signals.has_file_attachments is True

    def test_long_context_flag(self):
        long_text = "x" * 40_000  # ~10k tokens
        signals = extract_signals(long_text)
        assert signals.is_long_context is True

    def test_short_context_flag(self):
        signals = extract_signals("short")
        assert signals.is_long_context is False

    def test_prior_failures(self):
        signals = extract_signals("try again", prior_failures=2)
        assert signals.prior_failures == 2

    def test_deadline(self):
        signals = extract_signals("hurry", deadline="interactive")
        assert signals.deadline == "interactive"


class TestRuntimeSignals:
    def test_to_dict(self):
        signals = RuntimeSignals(prompt_chars=100, estimated_tokens=25)
        d = signals.to_dict()
        assert d["prompt_chars"] == 100
        assert d["estimated_tokens"] == 25
        assert "is_long_context" in d
