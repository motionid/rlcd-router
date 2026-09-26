#!/usr/bin/env python3
"""Routing benchmark: deterministic replay vs optional live Ollaya.

Replays the labeled fixture ``tests/fixtures/routing_cases.json``:

- Default (no flags) is fully deterministic and offline: no network,
  no Ollaya, no MLX.  Each case replays deterministic classification
  plus the candidate policy against the fixture's synthetic registry,
  and prints classification, selected capability/model, classifier
  latency, and a simulated escalation outcome (what the next candidate
  would be if the top candidate failed with a rate limit).
- ``--ollaya`` additionally consults the local Ollaya adapter for the
  cases the deterministic classifier marks ambiguous, and records the
  Ollaya result, confidence, latency, and disagreement with the
  deterministic classification.  It records evidence — it does not
  judge semantic perfection.

Fixture expectation mismatches in the default mode exit 1 so CI can
gate on the replay; in ``--ollaya`` mode disagreements are the signal,
so the run exits 0 and reports them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from router.availability import AvailabilityStore, FailureCategory
from router.config import AccessClass, Capability, ModelProfile, OLLAYA_CONFIG
from router.deterministic import TaskClassification, classify_deterministically
from router.ollaya import OllayaClassifier, OllayaError
from router.policy import RouteRequirements, build_route
from router.signals import TurnContext

DEFAULT_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "routing_cases.json"

_CAPABILITY_NAMES = {c.name.lower(): c for c in Capability}


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

def load_fixture(path: Optional[str] = None) -> dict[str, Any]:
    """Load and structurally validate the labeled replay fixture."""
    fixture = Path(path) if path else DEFAULT_FIXTURE
    data = json.loads(fixture.read_text())
    if not isinstance(data.get("registry"), list) or not data["registry"]:
        raise ValueError(f"{fixture}: registry missing or empty")
    if not isinstance(data.get("cases"), list) or not data["cases"]:
        raise ValueError(f"{fixture}: cases missing or empty")
    return data


def build_registry(data: dict[str, Any]) -> dict[str, ModelProfile]:
    """Materialize the fixture's synthetic registry as ModelProfiles."""
    registry: dict[str, ModelProfile] = {}
    for spec in data["registry"]:
        registry[spec["name"]] = ModelProfile(
            model_id=spec.get("model_id", spec["name"]),
            role="executor",
            provider=spec.get("provider", "synthetic"),
            capabilities=(),
            latency_class="fast",
            capability=Capability[str(spec["capability"]).upper()],
            access_class=AccessClass(str(spec["access_class"])),
            priority=int(spec.get("priority", 10)),
            supports_tools=bool(spec.get("supports_tools", False)),
            supports_thinking=bool(spec.get("supports_thinking", False)),
            context_limit=int(spec.get("context_limit", 128_000)),
        )
    return registry


def build_turn_context(case: dict[str, Any]) -> TurnContext:
    tc = case.get("turn_context") or {}
    previous_capability = None
    if tc.get("previous_capability"):
        previous_capability = Capability[
            str(tc["previous_capability"]).upper()
        ]
    return TurnContext(
        previous_prompt=tc.get("previous_prompt", ""),
        previous_capability=previous_capability,
        previous_model=tc.get("previous_model", ""),
        task_depth=int(tc.get("task_depth", 0)),
        files_involved=int(tc.get("files_involved", 0)),
        task_failures=int(tc.get("task_failures", 0)),
    )


def build_requirements(case: dict[str, Any]) -> RouteRequirements:
    req = case.get("requirements") or {}
    return RouteRequirements(
        supports_tools=bool(req.get("supports_tools", False)),
        supports_thinking=bool(req.get("supports_thinking", False)),
        minimum_context=int(req.get("minimum_context", 0)),
        local_only=bool(req.get("local_only", False)),
    )


def _apply_unavailability(
    store: AvailabilityStore,
    registry: dict[str, ModelProfile],
    case: dict[str, Any],
) -> None:
    """Put the case's unavailable profiles/access classes on cooldown."""
    for name in case.get("unavailable_profiles", []):
        store.record_failure(name, FailureCategory.QUOTA)
    for access in case.get("unavailable_access_classes", []):
        access_class = AccessClass(str(access))
        for name, profile in registry.items():
            if profile.access_class is access_class:
                store.record_failure(name, FailureCategory.QUOTA)


def _safe_subscription_default(
    registry: dict[str, ModelProfile],
    store: AvailabilityStore,
    requirements: RouteRequirements,
) -> Optional[tuple[str, ModelProfile]]:
    """Mirror router.policy._safe_subscription_default (scenario replay).

    Relaxing only the capability floor; hard requirements and
    availability still apply; paid access is never a safe default.
    """
    preferred = (
        AccessClass.LOCAL if requirements.local_only
        else AccessClass.SUBSCRIPTION
    )
    pool = []
    for name, profile in registry.items():
        if requirements.supports_tools and not profile.supports_tools:
            continue
        if requirements.supports_thinking and not profile.supports_thinking:
            continue
        if (
            requirements.minimum_context
            and profile.context_limit < requirements.minimum_context
        ):
            continue
        if not store.is_available(name):
            continue
        if profile.access_class is not preferred:
            continue
        pool.append((name, profile))
    if not pool:
        return None
    return min(pool, key=lambda item: (item[1].priority, item[0]))


# ---------------------------------------------------------------------------
# Case replay
# ---------------------------------------------------------------------------

def _classification_dict(classification: TaskClassification) -> dict[str, Any]:
    return {
        "classifier": classification.classifier,
        "capability": classification.capability.name.lower(),
        "relationship": classification.relationship.value,
        "ambiguous": classification.ambiguous,
        "reasons": list(classification.reasons),
        "confidence": round(classification.confidence, 4),
    }


def replay_case(
    case: dict[str, Any],
    registry: dict[str, ModelProfile],
    ollaya: Optional[OllayaClassifier] = None,
) -> dict[str, Any]:
    prompt = case["prompt"]
    scenario = case.get("scenario", "classification")
    context = build_turn_context(case)
    requirements = build_requirements(case)

    store = AvailabilityStore()
    _apply_unavailability(store, registry, case)
    if scenario == "recovery":
        # Cooldown, then recovery: the profile must be eligible again.
        recover = case["recover_profile"]
        store.record_failure(recover, FailureCategory.QUOTA)
        store.record_success(recover)

    # Deterministic classification is always the baseline (and the
    # fallback for every Ollaya failure).
    start = time.perf_counter()
    det = classify_deterministically(prompt, context)
    det_latency_ms = (time.perf_counter() - start) * 1000.0

    classification = det
    ollaya_info: Optional[dict[str, Any]] = None
    if scenario == "ollaya_disabled":
        ollaya_info = {"invoked": False, "outcome": "disabled"}
    elif ollaya is not None:
        if det.ambiguous:
            try:
                ollaya_cls = ollaya.classify(prompt, context)
            except OllayaError as exc:
                # Never a routing failure: record the fallback attempt.
                ollaya_info = {
                    "invoked": True,
                    "outcome": "error_fallback",
                    "error": type(exc).__name__,
                }
            else:
                ollaya_info = {
                    "invoked": True,
                    "outcome": "classified",
                    "capability": ollaya_cls.capability.name.lower(),
                    "confidence": round(ollaya_cls.confidence, 4),
                    "latency_ms": round(ollaya_cls.latency_ms or 0.0, 2),
                    "disagreement": (
                        ollaya_cls.capability is not det.capability
                    ),
                }
                classification = ollaya_cls
        else:
            ollaya_info = {"invoked": False, "outcome": "not_ambiguous"}

    # --- routing per scenario -------------------------------------------
    candidates: list[str] = []
    selected: Optional[dict[str, Any]] = None
    if scenario == "override":
        name = case["model_override"]
        profile = registry[name]
        candidates = [name]
        selected = {
            "profile": name,
            "access_class": profile.access_class.value,
            "capability": profile.capability.name.lower(),
        }
    elif scenario == "fallback":
        fallback = _safe_subscription_default(registry, store, requirements)
        if fallback is not None:
            name, profile = fallback
            candidates = [name]
            selected = {
                "profile": name,
                "access_class": profile.access_class.value,
                "capability": profile.capability.name.lower(),
            }
    else:
        decision = build_route(classification, registry, store, requirements)
        candidates = list(decision.eligible_models)
        if decision.candidates:
            top = decision.candidates[0]
            selected = {
                "profile": top.profile_id,
                "access_class": top.access_class.value,
                "capability": top.capability.name.lower(),
            }

    # --- simulated escalation outcome ------------------------------------
    # If the top candidate fails with a provider rate limit, PI advances
    # to the next candidate; with no remaining candidate the route is
    # exhausted and PI retains its active model.
    if len(candidates) >= 2:
        escalation = {
            "failed_profile": candidates[0],
            "next_profile": candidates[1],
            "outcome": "escalated",
        }
    elif len(candidates) == 1:
        escalation = {
            "failed_profile": candidates[0],
            "next_profile": None,
            "outcome": "single_candidate",
        }
    else:
        escalation = {
            "failed_profile": None,
            "next_profile": None,
            "outcome": "no_candidates",
        }

    return {
        "id": case["id"],
        "scenario": scenario,
        "prompt": prompt,
        "deterministic": {
            **_classification_dict(det),
            "latency_ms": round(det_latency_ms, 3),
        },
        "classifier_latency_ms": round(det_latency_ms, 3),
        "classification": _classification_dict(classification),
        "selected": selected,
        "candidates": candidates,
        "ollaya": ollaya_info,
        "simulated_escalation": escalation,
    }


def validate_row(
    case: dict[str, Any],
    row: dict[str, Any],
) -> list[str]:
    """Compare one replay row against the case's labeled expectations."""
    exp = case["expected"]
    scenario = row["scenario"]
    mismatches: list[str] = []
    det = row["deterministic"]
    selected = row["selected"]

    if scenario in ("classification", "recovery", "ollaya_disabled"):
        if det["capability"] != exp["min_capability"]:
            mismatches.append(
                f"{case['id']}: capability {det['capability']!r} != "
                f"expected {exp['min_capability']!r}"
            )
        if not selected or selected["access_class"] != exp["access_class"]:
            got = selected["access_class"] if selected else None
            mismatches.append(
                f"{case['id']}: access {got!r} != expected "
                f"{exp['access_class']!r}"
            )
        if scenario in ("classification", "recovery"):
            if det["ambiguous"] != exp["ollaya_expected"]:
                mismatches.append(
                    f"{case['id']}: ollaya consultation {det['ambiguous']} "
                    f"!= expected {exp['ollaya_expected']}"
                )
        elif scenario == "ollaya_disabled":
            if row["ollaya"]["outcome"] != "disabled" or exp["ollaya_expected"]:
                mismatches.append(
                    f"{case['id']}: ollaya-disabled scenario recorded "
                    f"{row['ollaya']}"
                )
    elif scenario == "override":
        if not selected or selected["profile"] != case["model_override"]:
            mismatches.append(f"{case['id']}: override not selected")
        elif selected["access_class"] != exp["access_class"]:
            mismatches.append(
                f"{case['id']}: override access "
                f"{selected['access_class']!r} != expected "
                f"{exp['access_class']!r}"
            )
        if exp["ollaya_expected"]:
            mismatches.append(
                f"{case['id']}: override scenario must not expect Ollaya"
            )
    elif scenario == "fallback":
        if not selected or selected["access_class"] != exp["access_class"]:
            got = selected["access_class"] if selected else None
            mismatches.append(
                f"{case['id']}: fallback access {got!r} != expected "
                f"{exp['access_class']!r}"
            )
        elif (
            _CAPABILITY_NAMES[selected["capability"]]
            < _CAPABILITY_NAMES[exp["min_capability"]]
        ):
            mismatches.append(
                f"{case['id']}: fallback capability "
                f"{selected['capability']!r} below expected floor "
                f"{exp['min_capability']!r}"
            )
        if exp["ollaya_expected"]:
            mismatches.append(
                f"{case['id']}: fallback scenario must not expect Ollaya"
            )
    return mismatches


# ---------------------------------------------------------------------------
# Full replay
# ---------------------------------------------------------------------------

def replay(
    data: dict[str, Any],
    ollaya: Optional[OllayaClassifier] = None,
) -> dict[str, Any]:
    """Replay every fixture case and summarize."""
    registry = build_registry(data)
    rows = [replay_case(case, registry, ollaya) for case in data["cases"]]
    mismatches: list[str] = []
    for case, row in zip(data["cases"], rows):
        mismatches.extend(validate_row(case, row))

    capability_counts: dict[str, int] = {}
    access_counts: dict[str, int] = {}
    escalation_counts: dict[str, int] = {}
    total_det_latency = 0.0
    ollaya_consulted = 0
    ollaya_classified = 0
    ollaya_error_fallback = 0
    ollaya_disagreements = 0

    for row in rows:
        cap = row["classification"]["capability"]
        capability_counts[cap] = capability_counts.get(cap, 0) + 1
        total_det_latency += row["deterministic"]["latency_ms"]
        outcome = row["simulated_escalation"]["outcome"]
        escalation_counts[outcome] = escalation_counts.get(outcome, 0) + 1
        if row["selected"]:
            access = row["selected"]["access_class"]
            access_counts[access] = access_counts.get(access, 0) + 1
        info = row["ollaya"]
        if info and info.get("invoked"):
            ollaya_consulted += 1
            if info.get("outcome") == "classified":
                ollaya_classified += 1
                if info.get("disagreement"):
                    ollaya_disagreements += 1
            elif info.get("outcome") == "error_fallback":
                ollaya_error_fallback += 1

    summary = {
        "total_cases": len(rows),
        "capability_distribution": capability_counts,
        "access_distribution": access_counts,
        "total_classifier_latency_ms": round(total_det_latency, 3),
        "avg_classifier_latency_ms": (
            round(total_det_latency / len(rows), 3) if rows else 0.0
        ),
        "ollaya_consulted_expected": sum(
            1 for c in data["cases"] if c["expected"]["ollaya_expected"]
        ),
        "ollaya_consulted": ollaya_consulted,
        "ollaya_classified": ollaya_classified,
        "ollaya_error_fallback": ollaya_error_fallback,
        "ollaya_disagreements": ollaya_disagreements,
        "escalation_outcomes": escalation_counts,
        "mismatch_count": len(mismatches),
    }
    return {"rows": rows, "summary": summary, "mismatches": mismatches}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _print_console(result: dict[str, Any]) -> None:
    rows = result["rows"]
    header = (
        f"{'id':44} {'cap':10} {'access':16} {'top candidate':24} "
        f"{'cls ms':>7} {'ollaya':16} {'escalation':14}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        selected = row["selected"]
        ollaya = row["ollaya"]
        if ollaya is None:
            ollaya_txt = "-"
        elif ollaya["invoked"]:
            ollaya_txt = (
                f"{ollaya['outcome']}"
                + (
                    f" ({ollaya['confidence']}, {ollaya['latency_ms']}ms"
                    + (", disagree" if ollaya.get("disagreement") else "")
                    + ")"
                    if ollaya.get("outcome") == "classified"
                    else ""
                )
            )
        else:
            ollaya_txt = f"skip:{ollaya['outcome']}"
        esc = row["simulated_escalation"]
        print(
            f"{row['id']:44} "
            f"{row['classification']['capability']:10} "
            f"{(selected['access_class'] if selected else '-'):16} "
            f"{(selected['profile'] if selected else '-'):24} "
            f"{row['deterministic']['latency_ms']:7.2f} "
            f"{ollaya_txt:16} "
            f"{esc['outcome']:14}"
        )
    print()
    summary = result["summary"]
    print("summary:")
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")
    if result["mismatches"]:
        print("\nexpectation mismatches (deterministic baseline):")
        for mismatch in result["mismatches"]:
            print(f"  - {mismatch}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay labeled routing cases: deterministic by default, "
            "deterministic+Ollaya with --ollaya."
        )
    )
    parser.add_argument(
        "--ollaya", action="store_true",
        help=(
            "Also consult the local Ollaya adapter for ambiguous cases "
            "(requires a running Ollaya service; explicit opt-in)."
        ),
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON instead of console rows.",
    )
    parser.add_argument(
        "--fixture", default=None,
        help="Path to the routing cases fixture (default: tests/fixtures/routing_cases.json).",
    )
    parser.add_argument(
        "--case", default=None,
        help="Replay only the case with this id.",
    )
    args = parser.parse_args(argv)

    try:
        data = load_fixture(args.fixture)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"fixture error: {exc}", file=sys.stderr)
        return 2
    if args.case:
        data["cases"] = [c for c in data["cases"] if c["id"] == args.case]
        if not data["cases"]:
            print(f"no case with id {args.case!r}", file=sys.stderr)
            return 2

    ollaya = None
    if args.ollaya:
        # Explicit opt-in: force the adapter enabled for this run.
        ollaya = OllayaClassifier(replace(OLLAYA_CONFIG, enabled=True))

    result = replay(data, ollaya=ollaya)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_console(result)

    # Fixture expectations gate the deterministic baseline; the Ollaya
    # run records evidence (result/confidence/latency/disagreement)
    # rather than semantic perfection, so it never fails the exit code.
    if not args.ollaya and result["mismatches"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
