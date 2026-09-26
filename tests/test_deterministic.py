"""
Tests for the deterministic turn-aware task classifier and bounded
turn context.

The deterministic classifier is the primary hot path: it must classify
obvious tasks with stable, bounded reason codes, inherit capability on
clear continuations, and never let a trivial verb mask repository-wide
debugging signals.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.config import Capability
from router.signals import (
    PREVIOUS_PROMPT_CHAR_LIMIT,
    TurnContext,
    extract_signals,
)
from router.deterministic import (
    REASON_CODES,
    TaskClassification,
    TaskRelationship,
    classify_deterministically,
)


def empty_context() -> TurnContext:
    return TurnContext()


def classification_reasons(result: TaskClassification) -> set[str]:
    return set(result.reasons)


# ---------------------------------------------------------------------------
# Obvious-task classification
# ---------------------------------------------------------------------------

class TestObviousTasks:
    def test_simple_rename_is_basic(self):
        result = classify_deterministically(
            "rename the variable x to user_count in parser.py",
            empty_context(),
        )
        assert result.capability == Capability.BASIC
        assert result.relationship is TaskRelationship.INDEPENDENT
        assert result.ambiguous is False
        assert "basic_signal_rename" in result.reasons
        assert 0.0 < result.confidence <= 1.0

    def test_readme_wording_is_basic(self):
        result = classify_deterministically(
            "update the README wording in the installation section",
            empty_context(),
        )
        assert result.capability == Capability.BASIC
        assert result.relationship is TaskRelationship.INDEPENDENT
        assert result.ambiguous is False
        assert "basic_signal_readme" in result.reasons

    def test_normal_function_work_is_normal(self):
        result = classify_deterministically(
            "add a function that validates email addresses in utils.py",
            empty_context(),
        )
        assert result.capability == Capability.NORMAL
        assert result.relationship is TaskRelationship.NEW_SUBSTANTIAL
        assert result.ambiguous is False
        assert "default_normal" in result.reasons

    def test_multi_file_debugging_is_strong(self):
        result = classify_deterministically(
            "debug the failing interaction between router/policy.py "
            "and router/decision.py",
            empty_context(),
        )
        assert result.capability >= Capability.STRONG
        assert result.relationship is TaskRelationship.NEW_SUBSTANTIAL
        assert result.ambiguous is False
        assert "strong_signal_debug" in result.reasons

    def test_repository_wide_scope_is_strong(self):
        result = classify_deterministically(
            "refactor the config loading across the repository",
            empty_context(),
        )
        assert result.capability >= Capability.STRONG
        assert "strong_signal_repository_scope" in result.reasons

    def test_root_cause_analysis_is_reasoning(self):
        result = classify_deterministically(
            "do a root cause analysis of the intermittent CI failures",
            empty_context(),
        )
        assert result.capability == Capability.REASONING
        assert "reasoning_signal_root_cause" in result.reasons


# ---------------------------------------------------------------------------
# Task continuity
# ---------------------------------------------------------------------------

class TestContinuity:
    def test_fix_it_inherits_previous_strong_task(self):
        context = TurnContext(
            previous_prompt="debug the race condition in the scheduler",
            previous_capability=Capability.STRONG,
            previous_model="local_coding",
        )
        result = classify_deterministically("fix it", context)
        assert result.capability == Capability.STRONG
        assert result.relationship is TaskRelationship.CONTINUATION
        assert result.ambiguous is False
        assert "continuation_phrase" in result.reasons
        assert "continuation_inherit_capability" in result.reasons

    def test_continue_inherits_previous_capability(self):
        context = TurnContext(previous_capability=Capability.NORMAL)
        result = classify_deterministically("continue", context)
        assert result.capability == Capability.NORMAL
        assert result.relationship is TaskRelationship.CONTINUATION

    def test_continuation_without_history_is_ambiguous_not_basic(self):
        result = classify_deterministically("try again", empty_context())
        assert result.capability >= Capability.NORMAL
        assert result.ambiguous is True
        assert result.relationship is TaskRelationship.CONTINUATION
        assert "continuation_missing_history" in result.reasons

    def test_commit_message_after_strong_task_is_independent_basic(self):
        context = TurnContext(
            previous_prompt="debug the race condition in the scheduler",
            previous_capability=Capability.STRONG,
            previous_model="local_coding",
            task_depth=4,
        )
        result = classify_deterministically(
            "write a commit message for the staged changes", context
        )
        assert result.capability == Capability.BASIC
        assert result.relationship is TaskRelationship.INDEPENDENT
        assert result.ambiguous is False
        assert "basic_signal_commit_message" in result.reasons

    def test_readme_wording_after_strong_task_is_independent_basic(self):
        context = TurnContext(previous_capability=Capability.STRONG)
        result = classify_deterministically(
            "fix the README wording in the intro paragraph", context
        )
        assert result.capability == Capability.BASIC
        assert result.relationship is TaskRelationship.INDEPENDENT
        assert "basic_signal_readme" in result.reasons


# ---------------------------------------------------------------------------
# Ambiguity and short prompts
# ---------------------------------------------------------------------------

class TestAmbiguity:
    def test_short_ambiguous_prompt_is_not_basic(self):
        result = classify_deterministically("why?", empty_context())
        assert result.capability > Capability.BASIC
        assert result.capability >= Capability.NORMAL
        assert result.ambiguous is True
        assert "ambiguous_short_prompt" in result.reasons

    def test_short_prompt_with_clear_basic_signal_is_basic(self):
        # Short is not automatically ambiguous when the signal is explicit.
        result = classify_deterministically("fix the typo", empty_context())
        assert result.capability == Capability.BASIC
        assert "basic_signal_typo" in result.reasons


# ---------------------------------------------------------------------------
# Failure-driven escalation
# ---------------------------------------------------------------------------

class TestFailureEscalation:
    def test_repeated_task_failures_raise_capability(self):
        result = classify_deterministically(
            "add a helper function that parses dates",
            TurnContext(task_failures=2),
        )
        # baseline is normal; two failures raise one level
        assert result.capability >= Capability.STRONG
        assert "task_failure_escalation" in result.reasons

    def test_escalation_is_capped_at_reasoning(self):
        result = classify_deterministically(
            "add a helper function that parses dates",
            TurnContext(task_failures=100),
        )
        assert result.capability == Capability.REASONING

    def test_no_failures_do_not_escalate(self):
        result = classify_deterministically(
            "add a helper function that parses dates",
            TurnContext(task_failures=0),
        )
        assert result.capability == Capability.NORMAL
        assert "task_failure_escalation" not in result.reasons


# ---------------------------------------------------------------------------
# Mixed-signal precedence
# ---------------------------------------------------------------------------

class TestMixedSignals:
    def test_mixed_trivial_and_repository_debug_signals_choose_strong(self):
        result = classify_deterministically(
            "rename this while debugging the race across the repository",
            empty_context(),
        )
        assert result.capability >= Capability.STRONG
        assert result.ambiguous is False
        # Strong evidence must outrank the trivial "rename" verb.
        assert "strong_signal_debug" in result.reasons
        assert "basic_signal_rename" not in result.reasons

    def test_multi_file_context_fact_raises_floor(self):
        result = classify_deterministically(
            "fix the typo",
            TurnContext(files_involved=3),
        )
        assert result.capability >= Capability.STRONG
        assert "strong_signal_multi_file" in result.reasons

    def test_deep_task_context_floors_at_normal(self):
        result = classify_deterministically(
            "why did that happen?",
            TurnContext(task_depth=4),
        )
        assert result.capability >= Capability.NORMAL
        assert "deep_task_context" in result.reasons


# ---------------------------------------------------------------------------
# Result shape and bounded reason codes
# ---------------------------------------------------------------------------

class TestContract:
    def test_classification_is_frozen_with_classifier_tag(self):
        result = classify_deterministically("hello there, please help", empty_context())
        assert isinstance(result, TaskClassification)
        assert result.classifier == "deterministic"

    def test_all_reasons_are_bounded_codes(self):
        prompts = [
            "rename the variable x to y",
            "debug the race across the repository",
            "fix it",
            "write a commit message",
            "why?",
            "add a helper function that parses dates",
            "do a root cause analysis of the deadlock",
        ]
        for prompt in prompts:
            result = classify_deterministically(prompt, empty_context())
            for reason in result.reasons:
                assert reason in REASON_CODES, reason

    def test_relationship_values(self):
        assert TaskRelationship.INDEPENDENT == "independent"
        assert TaskRelationship.CONTINUATION == "continuation"
        assert TaskRelationship.NEW_SUBSTANTIAL == "new_substantial"
        assert TaskRelationship.AMBIGUOUS == "ambiguous"


# ---------------------------------------------------------------------------
# TurnContext bounding and signal extraction
# ---------------------------------------------------------------------------

class TestTurnContext:
    def test_previous_prompt_is_capped(self):
        long_prompt = "x" * (PREVIOUS_PROMPT_CHAR_LIMIT + 500)
        context = TurnContext(previous_prompt=long_prompt)
        assert len(context.previous_prompt) == PREVIOUS_PROMPT_CHAR_LIMIT

    def test_short_previous_prompt_is_preserved(self):
        context = TurnContext(previous_prompt="debug the parser")
        assert context.previous_prompt == "debug the parser"

    def test_defaults(self):
        context = TurnContext()
        assert context.previous_prompt == ""
        assert context.previous_capability is None
        assert context.previous_model == ""
        assert context.previous_route_reason == ()
        assert context.task_depth == 0
        assert context.files_involved == 0
        assert context.task_failures == 0

    def test_to_dict_caps_previous_prompt(self):
        context = TurnContext(previous_prompt="y" * (PREVIOUS_PROMPT_CHAR_LIMIT + 10))
        d = context.to_dict()
        assert len(d["previous_prompt"]) == PREVIOUS_PROMPT_CHAR_LIMIT
        assert d["task_failures"] == 0


class TestSignalsRepoFacts:
    def test_repo_size_is_caller_supplied(self):
        signals = extract_signals("fix the bug", repo_size_mb=12.5)
        assert signals.repo_size_mb == 12.5

    def test_repo_root_is_accepted_without_walking(self, monkeypatch):
        # The default turn hot path must not walk the filesystem.
        def boom(_root):
            raise AssertionError("os.walk must not run in the default path")

        monkeypatch.setattr("os.walk", boom)
        signals = extract_signals("fix the bug", repo_root="/definitely/exists")
        assert signals.repo_size_mb == 0.0

    def test_existing_signal_fields_still_work(self):
        signals = extract_signals(
            "Look at src/main.py and run python -m pytest",
            attached_files=["a.py"],
            prior_failures=1,
        )
        assert signals.repo_files_referenced >= 1
        assert signals.has_test_command is True
        assert signals.attached_file_count == 1


# ---------------------------------------------------------------------------
# Benchmark helper (Step 5) — structural sanity only, no wall-clock assertion
# ---------------------------------------------------------------------------

def benchmark_deterministic(n: int = 10_000) -> dict:
    """Run n classifications over representative prompts, return timing stats.

    Uses time.perf_counter_ns() per classification; the caller (or a
    manual run) records median/p95.  Tests only assert structural
    sanity of this helper — never a wall-clock threshold.
    """
    prompts = [
        "rename the variable x to user_count in parser.py",
        "update the README wording in the installation section",
        "add a function that validates email addresses in utils.py",
        "debug the failing interaction between router/policy.py and router/decision.py",
        "rename this while debugging the race across the repository",
        "write a commit message for the staged changes",
        "fix it",
        "continue",
        "why?",
        "add a helper function that parses dates",
        "do a root cause analysis of the intermittent CI failures",
        "refactor the config loading across the repository",
    ]
    contexts = [
        TurnContext(),
        TurnContext(previous_capability=Capability.STRONG, task_depth=3),
        TurnContext(task_failures=2, files_involved=3),
    ]
    samples: list[int] = []
    for i in range(n):
        prompt = prompts[i % len(prompts)]
        context = contexts[i % len(contexts)]
        start = time.perf_counter_ns()
        classify_deterministically(prompt, context)
        samples.append(time.perf_counter_ns() - start)
    samples.sort()
    median_ns = samples[len(samples) // 2]
    p95_ns = samples[int(len(samples) * 0.95)]
    return {
        "iterations": n,
        "median_ns": median_ns,
        "p95_ns": p95_ns,
        "median_us": median_ns / 1000,
        "p95_us": p95_ns / 1000,
    }


class TestBenchmark:
    def test_benchmark_helper_runs(self):
        stats = benchmark_deterministic(10_000)
        assert stats["iterations"] == 10_000
        assert stats["median_ns"] > 0
        assert stats["p95_ns"] >= stats["median_ns"]
        print(
            f"\ndeterministic benchmark: median={stats['median_us']:.1f}us "
            f"p95={stats['p95_us']:.1f}us over {stats['iterations']} classifications"
        )
