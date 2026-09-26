# Simplified Turn-Aware PI Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the existing RLCD router into a deterministic, turn-aware, subscription-first PI router with optional Ollaya ambiguity classification, expiring availability state, ordered candidates, and a new thin PI extension.

**Architecture:** The Python service owns deterministic task classification, optional Ollaya classification, the capability/access registry, availability cooldowns, candidate ordering, and telemetry. A new small PI extension owns branch-sensitive session continuity, model activation, provider outcome capture, bounded candidate retries, and manual auto/override controls. The existing RLCD/MLX classifier remains optional for offline evaluation and is removed from the required production route path.

**Tech Stack:** Python 3.13, dataclasses/enums, FastAPI/Pydantic, pytest, JSONL telemetry, TypeScript PI extension APIs, optional Ollaya HTTP API.

**Spec:** `docs/superpowers/specs/2026-09-26-simplified-turn-aware-router-design.md`

## Global Constraints

- Do not restore `pi-extension/rlcd-router.ts`; its deletion is intentional.
- Create a new extension at `pi-extension/turn-router.ts` and use the disabled backup only as reference.
- Preserve existing public APIs while migration tests still rely on them; deprecate before removal.
- Production routing must work with both Ollaya and RLCD disabled.
- Ollaya may classify tasks but must never select providers, models, access classes, or spending policy.
- Do not install Ollaya or download a model without explicit user approval at Task 1.
- Use only official Ollaya documentation/releases for installation and API behavior.
- Normal tests must not need Ollaya, RLCD weights, downloaded models, or network access.
- Metered models must not be selected while a suitable subscription is available.
- Router, telemetry, classifier, or availability failures must not break PI.
- Do not add embeddings, vector databases, remote classifiers, agent chains, or new persistence services.
- Do not overwrite or silently include the repository's pre-existing uncommitted changes in task commits.
- Execute in an isolated worktree after resolving how the current dirty working tree is represented there.

## Review Focus

1. **Mixed-signal prompts:** a request containing a trivial verb plus repository-wide debugging must route strong, not basic; pin in Task 3.
2. **Provider failure versus task failure:** quota/rate failures must create cooldowns without raising task capability, while unsuccessful task attempts may raise capability; pin in Tasks 4 and 7.
3. **Cooldown boundary:** a model is excluded strictly before `cooldown_until` and becomes eligible at or after it; pin in Task 4 with an injected clock.
4. **Manual versus automatic model changes:** `model_select` from the extension's own `pi.setModel()` must not become an override, while user selection must; pin in Task 9.
5. **Fallback recursion:** repeated candidate failures or settlement continuations must stop at the configured attempt/candidate limit; pin in Task 9.

---

## Execution Prerequisite

Before Task 1, invoke `superpowers:using-git-worktrees`. The current main working tree contains intentional uncommitted changes, including the deleted old extension. Ask the user whether those non-spec changes should be checkpointed, excluded, or supplied to the implementation worktree; do not infer an answer and do not stash, commit, or discard them without approval. Verify the implementation worktree starts from the approved spec/plan commits and contains the intended baseline before changing product code.

## File Map

### Create

- `router/availability.py` — in-memory model/provider availability, failure categories, cooldown calculation, and recovery.
- `router/deterministic.py` — deterministic task/relationship classifier and bounded continuity input types.
- `router/ollaya.py` — optional local HTTP adapter and Ollaya-independent classification result mapping.
- `pi-extension/turn-router.ts` — new thin PI lifecycle/model-switching integration.
- `pi-extension/router-state.ts` — pure, PI-independent session/override/candidate state transitions.
- `tests/test_pi_extension_state.mjs` — Node tests for the pure TypeScript state transitions.
- `tests/test_deterministic.py` — obvious task, ambiguity, continuity, and failure-depth classification.
- `tests/test_availability.py` — cooldown, Retry-After, and recovery behavior.
- `tests/test_ollaya.py` — mocked adapter behavior, confidence handling, disabled/unavailable behavior.
- `tests/fixtures/routing_cases.json` — small labeled deterministic/Ollaya replay dataset.
- `scripts/benchmark_routing.py` — deterministic versus optional live-Ollaya comparison.

### Modify

- `router/config.py` — capability/access enums, routing-focused model profile fields, configurable priorities, Ollaya/cooldown settings.
- `router/signals.py` — bounded `TurnContext`, file/tool/context facts; remove repository walking from the turn hot path.
- `router/decision.py` — capability-based `RouteDecision`, candidate and exclusion records while retaining compatibility serialization.
- `router/policy.py` — registry filtering and capability-specific access ordering.
- `router/escalation.py` — distinguish task failures from availability failures and support candidate-cursor outcomes.
- `router/telemetry.py` — one route record with attempts and new aggregate metrics.
- `server/app.py` — new production route pipeline, availability/outcome/status APIs, fail-safe behavior, optional legacy endpoints.
- `router/__init__.py` — export new production interfaces without importing MLX.
- `tests/test_policy.py` — capability/access candidate-order tests.
- `tests/test_escalation.py` — candidate/failure-category behavior.
- `tests/test_telemetry.py` — single-record attempts and access/cost aggregates.
- `tests/test_router_api.py` — new route/outcome/status and fail-safe integration tests.
- `tests/test_pi_extension_structure.py` — target the new extension and pin lifecycle/state/override contracts.
- `tests/conftest.py` — shared synthetic registry/clock only if duplication justifies it.
- `README.md` — local/subscription/metered/Ollaya configuration and PI commands.
- `requirements.txt` — add no Ollaya SDK; continue using the existing HTTP client dependency.

### Preserve Initially

- `router/router_schema.py`, `router/classifier.py`, `core/` — optional RLCD evaluation path.
- `server/app.py` legacy `/api/execute` and `/api/route-and-execute` — mark deprecated; remove only after consumer checks.
- `pi-extension/rlcd-router.ts.disabled.backup` — untouched reference artifact.

---

### Task 1: Phase 3 Ollaya Approval and Live Verification

**Files:**
- No repository changes required
- Reference: `docs/superpowers/specs/2026-09-26-simplified-turn-aware-router-design.md`

**Interfaces:**
- Consumes: official Ollaya install command and API documented at `https://ollaya.dev/download` and `https://ollaya.dev/docs/api`.
- Produces: verified CLI version, server endpoint, installed model name, real request/response fixture, and measured cold/warm latency for Task 6.

- [ ] **Step 1: Ask for explicit installation approval**

Present the exact official Apple Silicon command and state that the default `laya` router pulls its English and multilingual target checkpoints:

```bash
curl -fsSL https://ollaya.dev/install.sh | sh
```

Do not continue until the user approves installation and the initial model download.

- [ ] **Step 2: Verify host compatibility again immediately before installation**

Run:

```bash
uname -m
sw_vers -productVersion
command -v ollaya || true
```

Expected: `arm64`; if Ollaya appeared since Phase 1, inspect it rather than reinstalling.

- [ ] **Step 3: Install only after approval and inspect the version**

Run the approved official installer, then:

```bash
ollaya -v
```

Expected: client/server version output from the installed release. Record the exact version.

- [ ] **Step 4: Start and verify the documented local service**

Run `ollaya serve` through the documented macOS lifecycle, then:

```bash
curl -fsS http://127.0.0.1:11435/
curl -fsS http://127.0.0.1:11435/api/version
```

Expected: liveness text and a JSON version response. Do not guess or substitute another port.

- [ ] **Step 5: Pull/run only the approved initial Laya model**

Default path:

```bash
ollaya run laya --preset router "fix it"
```

If the user chooses the smallest English-only experiment instead, use `laya:en` and a direct `/api/decide` request. Do not download `decider`, `nli`, `gliclass`, or additional checkpoints.

- [ ] **Step 6: Execute a real routing decision and capture probabilities**

Use `POST /api/decide` with exactly three questions: `required_capability`, `is_task_continuation`, and `is_independent_trivial_task`. Test one obvious basic request and one ambiguous continuation state containing previous strong-task context.

Expected: typed answers, probabilities, model name, `state_truncated: false`, and timing fields.

- [ ] **Step 7: Measure local cold and warm latency**

Run at least one cold request and ten warm requests using `/usr/bin/time` plus Ollaya's `total_duration`/`eval_duration`. Record median and maximum wall-clock latency. Do not use published GPU latency as the local result.

- [ ] **Step 8: Record the gate result in the implementation session notes**

Decision:

- proceed with Ollaya adapter if responses are structurally correct and latency is interactive;
- implement adapter disabled-by-default if accuracy is uncertain;
- skip live integration and retain mocked/disabled behavior if the service is unsuitable.

No application integration code is written during this task.

---

### Task 2: Capability and Access Registry

**Files:**
- Modify: `router/config.py`
- Modify: `router/decision.py`
- Modify: `tests/test_schema.py`
- Modify: `tests/test_router_api.py`

**Interfaces:**
- Consumes: existing `ModelProfile`, `MODEL_PROFILES`, and serialized route/config API.
- Produces: `Capability`, `AccessClass`, expanded `ModelProfile`, `Candidate`, `CandidateExclusion`, and compatibility serialization used by Tasks 4–10.

- [ ] **Step 1: Write failing enum and profile serialization tests**

Add tests asserting:

```python
assert Capability.BASIC < Capability.NORMAL < Capability.STRONG < Capability.REASONING
assert ModelProfile(..., capability=Capability.STRONG,
                    access_class=AccessClass.SUBSCRIPTION,
                    priority=20).to_dict()["access_class"] == "subscription"
```

Also assert classifier profiles are never executor candidates and access classes serialize as `local`, `subscription`, `metered-cheap`, or `metered-expensive`.

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```bash
.venv/bin/pytest -q tests/test_schema.py tests/test_router_api.py -k 'capability or access_class or profile'
```

Expected: FAIL because the new enums/fields do not exist.

- [ ] **Step 3: Add exact registry types**

In `router/config.py`, add:

```python
class Capability(IntEnum):
    BASIC = 0
    NORMAL = 1
    STRONG = 2
    REASONING = 3

class AccessClass(str, Enum):
    LOCAL = "local"
    SUBSCRIPTION = "subscription"
    METERED_CHEAP = "metered-cheap"
    METERED_EXPENSIVE = "metered-expensive"
```

Extend `ModelProfile` with `capability`, `access_class`, `priority`, `enabled`, `context_limit`, `supports_tools`, and `supports_thinking`. Preserve old `billing`/`capabilities` serialization temporarily where compatibility tests require it.

- [ ] **Step 4: Migrate existing executor profiles explicitly**

Assign local profiles as local access; routine Copilot profiles as normal subscription; coding/general reasoning profiles as strong subscription; flagship/deep models as reasoning subscription. Keep RLCD profiles role=`classifier` and excluded from execution. Add no enabled metered production profile without real provider/model configuration; tests use synthetic metered profiles.

- [ ] **Step 5: Add candidate decision records**

In `router/decision.py`, define immutable records:

```python
@dataclass(frozen=True)
class Candidate:
    profile_id: str
    provider: str
    model: str
    capability: Capability
    access_class: AccessClass
    priority: int

@dataclass(frozen=True)
class CandidateExclusion:
    profile_id: str
    reason: str
```

Extend `RouteDecision` with `capability`, `candidates`, and `excluded`, retaining `tier` and `eligible_models` as compatibility fields until migration completes.

- [ ] **Step 6: Run focused tests and verify pass**

Run the Step 2 command.

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add router/config.py router/decision.py tests/test_schema.py tests/test_router_api.py
git commit -m "refactor: separate model capability from access class"
```

---

### Task 3: Deterministic Signals and Task Continuity

**Files:**
- Create: `router/deterministic.py`
- Create: `tests/test_deterministic.py`
- Modify: `router/signals.py`
- Modify: `router/__init__.py`

**Interfaces:**
- Consumes: `Capability` from Task 2 and compact state sent by the extension.
- Produces: `TurnContext`, `TaskRelationship`, `TaskClassification`, `classify_deterministically(prompt, context)`, and bounded reason codes used by Tasks 5–10.

- [ ] **Step 1: Write the requested obvious-task and continuity tests**

Create tests named:

```text
test_simple_rename_is_basic
test_readme_wording_is_basic
test_normal_function_work_is_normal
test_multi_file_debugging_is_strong
test_fix_it_inherits_previous_strong_task
test_commit_message_after_strong_task_is_independent_basic
test_short_ambiguous_prompt_is_not_basic
test_repeated_task_failures_raise_capability
test_mixed_trivial_and_repository_debug_signals_choose_strong
```

Assert capability, relationship, ambiguity, and stable reason codes. The mixed-signal review test must classify `rename this while debugging the race across the repository` as at least strong.

- [ ] **Step 2: Run tests and verify failure**

```bash
.venv/bin/pytest -q tests/test_deterministic.py
```

Expected: import or assertion failures because the classifier is absent.

- [ ] **Step 3: Add bounded turn context types**

In `router/signals.py`, define:

```python
@dataclass(frozen=True)
class TurnContext:
    previous_prompt: str = ""
    previous_capability: Capability | None = None
    previous_model: str = ""
    previous_route_reason: tuple[str, ...] = ()
    task_depth: int = 0
    files_involved: int = 0
    task_failures: int = 0
```

Cap `previous_prompt` at a configured character limit during construction/serialization. Remove `os.walk()` repo-size measurement from the default per-turn path; accept caller-supplied repository facts instead.

- [ ] **Step 4: Implement deterministic classification interfaces**

In `router/deterministic.py`, define:

```python
class TaskRelationship(str, Enum): ...

@dataclass(frozen=True)
class TaskClassification:
    capability: Capability
    confidence: float
    relationship: TaskRelationship
    reasons: tuple[str, ...]
    ambiguous: bool
    classifier: str = "deterministic"

def classify_deterministically(prompt: str, context: TurnContext) -> TaskClassification: ...
```

Use ordered strong/reasoning checks before bounded basic patterns. Continuation inheritance occurs before prompt-length defaults. Repeated task failures raise capability one level per configured threshold, capped at reasoning.

- [ ] **Step 5: Benchmark the deterministic path in a unit test helper**

Run 10,000 classifications over representative prompts with `time.perf_counter_ns()` outside CI assertions. Record median/p95 in session notes; do not add a flaky wall-clock assertion to tests.

- [ ] **Step 6: Run deterministic and existing signal tests**

```bash
.venv/bin/pytest -q tests/test_deterministic.py tests/test_signals.py
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add router/deterministic.py router/signals.py router/__init__.py tests/test_deterministic.py tests/test_signals.py
git commit -m "feat: add deterministic turn-aware task classification"
```

---

### Task 4: Availability, Cooldowns, and Recovery

**Files:**
- Create: `router/availability.py`
- Create: `tests/test_availability.py`
- Modify: `router/config.py`
- Modify: `router/escalation.py`
- Modify: `tests/test_escalation.py`

**Interfaces:**
- Consumes: model profile IDs and provider outcome metadata.
- Produces: `FailureCategory`, `AvailabilityRecord`, `AvailabilityStore`, `CooldownConfig`, `record_failure()`, `record_success()`, and `is_available()` for Tasks 5, 7, and 9.

- [ ] **Step 1: Write failing cooldown tests with an injected clock**

Cover:

```text
test_rate_limit_uses_short_cooldown
test_retry_after_is_respected
test_quota_uses_medium_cooldown
test_unknown_failure_recovers_after_conservative_cooldown
test_subscription_is_excluded_before_cooldown_boundary
test_subscription_is_eligible_at_cooldown_boundary
test_task_failure_does_not_mark_provider_unavailable
test_success_clears_transient_failure_count
```

- [ ] **Step 2: Run tests and verify failure**

```bash
.venv/bin/pytest -q tests/test_availability.py tests/test_escalation.py
```

Expected: FAIL because availability types are absent.

- [ ] **Step 3: Implement availability types and clock injection**

Define:

```python
class FailureCategory(str, Enum):
    RATE_LIMIT = "rate_limit"
    QUOTA = "quota"
    AUTH = "auth"
    CAPACITY = "capacity"
    PROVIDER_UNKNOWN = "provider_unknown"
    TASK_FAILURE = "task_failure"
    LOCAL_TOOL_FAILURE = "local_tool_failure"

class AvailabilityStore:
    def __init__(self, cooldowns: CooldownConfig, clock: Callable[[], float] = time.time): ...
    def record_failure(self, profile_id: str, category: FailureCategory,
                       retry_after: float | None = None) -> AvailabilityRecord: ...
    def record_success(self, profile_id: str) -> AvailabilityRecord: ...
    def is_available(self, profile_id: str) -> bool: ...
    def snapshot(self) -> dict[str, AvailabilityRecord]: ...
```

Task/local-tool failures do not create provider cooldowns. Expired records become eligible without a background timer.

- [ ] **Step 4: Make escalation consume failure categories**

Update `router/escalation.py` so task-quality failures affect task failure count while provider availability failures advance candidates without increasing task capability.

- [ ] **Step 5: Run tests and verify pass**

Run the Step 2 command.

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add router/availability.py router/config.py router/escalation.py tests/test_availability.py tests/test_escalation.py
git commit -m "feat: add expiring model availability cooldowns"
```

---

### Task 5: Capability and Cost Candidate Policy

**Files:**
- Modify: `router/policy.py`
- Modify: `router/decision.py`
- Modify: `tests/test_policy.py`

**Interfaces:**
- Consumes: `TaskClassification`, `ModelProfile`, and `AvailabilityStore` from Tasks 2–4.
- Produces: `build_route(classification, profiles, availability, requirements) -> RouteDecision` and configurable access ordering used by the API.

- [ ] **Step 1: Replace tier-only tests with the required candidate tests**

Add synthetic-registry tests:

```text
test_basic_prefers_local
test_normal_prefers_subscription_before_capable_local
test_strong_prefers_strong_subscriptions
test_reasoning_prefers_reasoning_subscription
test_unavailable_first_subscription_falls_back_to_next_subscription
test_all_subscriptions_unavailable_uses_metered_cheap
test_recovered_subscription_reenters_before_metered
test_paid_not_selected_when_subscription_is_suitable
test_candidate_order_respects_configured_priority
test_tools_and_context_filter_ineligible_profiles
test_local_only_requirement_excludes_all_remote_profiles
test_policy_failure_returns_safe_subscription_default
```

- [ ] **Step 2: Run policy tests and verify failure**

```bash
.venv/bin/pytest -q tests/test_policy.py
```

Expected: FAIL against the current tier selector.

- [ ] **Step 3: Implement capability-specific access ordering**

Add:

```python
@dataclass(frozen=True)
class RouteRequirements:
    supports_tools: bool = False
    supports_thinking: bool = False
    minimum_context: int = 0
    local_only: bool = False

def build_route(
    classification: TaskClassification,
    profiles: Mapping[str, ModelProfile],
    availability: AvailabilityStore,
    requirements: RouteRequirements = RouteRequirements(),
) -> RouteDecision: ...
```

Filter hard requirements first, then availability, then sort by the spec's capability-specific access order and numeric profile priority. Populate exclusions with stable reason codes.

- [ ] **Step 4: Keep a compatibility `choose_route()` adapter**

Map legacy RLCD classification dictionaries to a conservative `TaskClassification` only for old callers/tests. Mark the adapter deprecated; do not use it in the new production API path.

- [ ] **Step 5: Run policy tests and verify pass**

Run the Step 2 command plus:

```bash
.venv/bin/pytest -q tests/test_confidence.py
```

Expected: PASS, with legacy compatibility tests still green.

- [ ] **Step 6: Commit**

```bash
git add router/policy.py router/decision.py tests/test_policy.py tests/test_confidence.py
git commit -m "feat: order candidates by capability availability and access"
```

---

### Task 6: Optional Ollaya Ambiguity Adapter

**Files:**
- Create: `router/ollaya.py`
- Create: `tests/test_ollaya.py`
- Modify: `router/config.py`
- Modify: `router/deterministic.py`

**Interfaces:**
- Consumes: verified API behavior from Task 1 and `TaskClassification`/`TurnContext` from Task 3.
- Produces: `OllayaConfig`, `OllayaClassifier.classify()`, and `classify_task()` orchestration for Task 7.

- [ ] **Step 1: Write mocked adapter and invocation-gate tests**

Cover:

```text
test_ollaya_is_not_called_for_obvious_rename
test_ollaya_can_classify_ambiguous_continuation
test_low_confidence_basic_falls_back_to_subscription_safe_capability
test_unavailable_ollaya_returns_deterministic_fallback
test_malformed_ollaya_response_returns_deterministic_fallback
test_ollaya_can_be_disabled
test_state_payload_is_bounded_and_contains_prior_context
test_ollaya_result_contains_latency_and_probabilities
```

Mock HTTP responses; no live service is required.

- [ ] **Step 2: Run tests and verify failure**

```bash
.venv/bin/pytest -q tests/test_ollaya.py tests/test_deterministic.py
```

Expected: FAIL because the adapter/orchestrator is absent.

- [ ] **Step 3: Add configuration and adapter abstraction**

Define:

```python
@dataclass(frozen=True)
class OllayaConfig:
    enabled: bool = False
    base_url: str = "http://127.0.0.1:11435"
    model: str = "laya"
    timeout_ms: int = 500
    basic_threshold: float = 0.90
    continuation_threshold: float = 0.75

class OllayaClassifier:
    def classify(self, prompt: str, context: TurnContext) -> TaskClassification: ...
```

Use `POST /api/decide`; isolate all Ollaya wire parsing in this file. Send exactly the three approved questions. Return router-owned types.

- [ ] **Step 4: Add deterministic-first orchestration**

Define:

```python
def classify_task(prompt: str, context: TurnContext,
                  ollaya: OllayaClassifier | None = None) -> TaskClassification: ...
```

Return deterministic results immediately when `ambiguous` is false. On low confidence or adapter error, return conservative normal/previous-continuation capability suitable for subscription—not a paid-access decision.

- [ ] **Step 5: Run tests and verify pass**

Run the Step 2 command.

Expected: PASS.

- [ ] **Step 6: Run one optional live test only when explicitly enabled**

Gate with `OLLAYA_LIVE_TEST=1`; otherwise skip. Assert schema shape and record latency, not semantic perfection.

- [ ] **Step 7: Defer Modelfile creation unless replay evidence supports it**

If direct-question definitions are stable and useful, add a separately reviewed `router/ollaya/Modelfile`; otherwise keep questions centralized in `router/ollaya.py` for this iteration. Do not create a Modelfile merely because Ollaya supports it.

- [ ] **Step 8: Commit**

```bash
git add router/ollaya.py router/config.py router/deterministic.py tests/test_ollaya.py
git commit -m "feat: add optional Ollaya ambiguity classification"
```

---

### Task 7: Production Route, Outcome, and Status APIs

**Files:**
- Modify: `server/app.py`
- Modify: `tests/test_router_api.py`
- Modify: `router/__init__.py`

**Interfaces:**
- Consumes: `classify_task()`, `build_route()`, `AvailabilityStore`, and route records.
- Produces: new `POST /api/route`, `POST /api/route/outcome`, and `GET /api/router/status` response contracts used by the extension.

- [ ] **Step 1: Write failing API tests for the complete production path**

Add tests:

```text
test_route_obvious_request_does_not_load_rlcd
test_route_ambiguous_request_works_with_ollaya_disabled
test_route_returns_ordered_candidates_and_exclusions
test_outcome_rate_limit_updates_availability_not_task_capability
test_outcome_task_failure_updates_task_failure_not_availability
test_status_reports_classifier_and_cooldowns
test_router_exception_returns_safe_fallback_decision
test_no_network_classifier_is_required
test_explicit_override_returns_only_requested_available_profile
test_unavailable_override_falls_back_without_clearing_preference
```

Patch new interfaces rather than importing/loading `core.engine`.

- [ ] **Step 2: Run API tests and verify failure**

```bash
.venv/bin/pytest -q tests/test_router_api.py
```

Expected: FAIL on new response contracts and current eager RLCD path.

- [ ] **Step 3: Make production startup independent of MLX**

Remove top-level `core.engine` imports from the production route path. Import RLCD/MLX lazily only inside legacy/evaluation endpoints. Construct deterministic/Ollaya/availability services lazily with failure-safe defaults.

- [ ] **Step 4: Define request/response models**

Add Pydantic models for bounded turn context, route requirements, outcome category/status/Retry-After, ordered candidates, and compact status. Preserve accepted legacy request fields during migration.

- [ ] **Step 5: Implement production route orchestration**

`/api/route` performs signal extraction → deterministic/optional Ollaya classification → policy candidate construction → telemetry start. Catch internal failures and return a `classifier="fallback"` decision containing the configured safe subscription profile when resolvable.

- [ ] **Step 6: Implement outcome and status behavior**

`/api/route/outcome` records attempts and updates either availability or task outcome according to `FailureCategory`. `/api/router/status` returns model access/capability/availability, Ollaya enabled/health state, and last route.

- [ ] **Step 7: Run API and compatibility tests**

```bash
.venv/bin/pytest -q tests/test_router_api.py tests/test_execute.py tests/test_subscription_fallback.py
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add server/app.py router/__init__.py tests/test_router_api.py
git commit -m "feat: expose deterministic route availability and status APIs"
```

---

### Task 8: Single-Record Telemetry and Routing Benchmark

**Files:**
- Modify: `router/telemetry.py`
- Modify: `tests/test_telemetry.py`
- Create: `tests/fixtures/routing_cases.json`
- Create: `scripts/benchmark_routing.py`

**Interfaces:**
- Consumes: route decisions, candidate attempts, classification source/latency/confidence, optional cost metadata.
- Produces: one immutable route event with appendable attempts, aggregate metrics, and deterministic/Ollaya replay output.

- [ ] **Step 1: Write failing telemetry representation tests**

Cover one JSONL route record, multiple attempt entries, access/model counts, paid-selection reasons, Ollaya count/latency, local escalation rate, and optional estimated spend incurred/avoided. Assert reporting an outcome does not append a duplicate pending event.

- [ ] **Step 2: Run tests and verify failure**

```bash
.venv/bin/pytest -q tests/test_telemetry.py
```

Expected: FAIL against duplicate mutable event recording.

- [ ] **Step 3: Implement route plus attempts event shape**

Define an `AttemptEvent` and make `RoutingEvent.attempts` a list. Keep the active route event in memory and append exactly one finalized JSONL record at settlement or candidate exhaustion. `summary()` reduces finalized records only. Telemetry exceptions are swallowed at the API boundary with debug logging.

- [ ] **Step 4: Add labeled replay fixtures**

Include at least the 22 required routing cases with prompt, compact prior state, expected minimum capability, expected access class, and whether Ollaya is expected to be called.

- [ ] **Step 5: Add benchmark script**

`scripts/benchmark_routing.py` runs deterministic-only by default and adds Ollaya results only with `--ollaya`. Output JSON/console rows containing classification, selected capability/model, classifier latency, Ollaya confidence, and simulated/recorded escalation outcome.

- [ ] **Step 6: Run tests and deterministic benchmark**

```bash
.venv/bin/pytest -q tests/test_telemetry.py
.venv/bin/python scripts/benchmark_routing.py
```

Expected: tests PASS; benchmark completes without network access.

- [ ] **Step 7: Commit**

```bash
git add router/telemetry.py tests/test_telemetry.py tests/fixtures/routing_cases.json scripts/benchmark_routing.py
git commit -m "feat: record routing attempts and benchmark classifiers"
```

---

### Task 9: New Thin PI Extension

**Files:**
- Create: `pi-extension/turn-router.ts`
- Create: `pi-extension/router-state.ts`
- Create: `tests/test_pi_extension_state.mjs`
- Modify: `tests/test_pi_extension_structure.py`

**Interfaces:**
- Consumes: Task 7 route/outcome/status APIs and PI's documented lifecycle/model APIs.
- Produces: turn-aware model selection, bounded fallback attempts, persisted continuity, manual override detection, `/router-auto`, and `/router-status`.

- [ ] **Step 1: Rewrite structural tests to target only the new extension**

Assert the source uses:

```text
before_agent_start
after_provider_response
agent_before_settle
model_select
session_start
session_shutdown
pi.setModel
pi.appendEntry
ctx.sessionManager.getBranch
/router-auto
/router-status
```

Assert it does not import or reference the deleted `pi-extension/rlcd-router.ts` and does not call RLCD classification tools.

- [ ] **Step 2: Add pure state-machine tests using Node 25's TypeScript support**

Create `tests/test_pi_extension_state.mjs` using `node:test` and import pure exports from `pi-extension/router-state.ts`. Pin:

```text
user model selection creates override
automatic setModel event does not create override
router-auto clears override
candidate activation failure advances cursor
provider 429 + Retry-After reports rate limit and advances cursor
task failure reports task failure
aborted turn does not penalize model
candidate exhaustion stops continuation
router outage retains active model or configured subscription fallback
branch state reconstructs from latest active-branch router entry
```

The pure module exports versioned state creation/restoration, manual-selection handling, guarded automatic-selection handling, candidate advancement, and continuation-limit decisions. It imports no PI runtime modules.

- [ ] **Step 3: Run extension tests and verify failure**

```bash
.venv/bin/pytest -q tests/test_pi_extension_structure.py
node --test tests/test_pi_extension_state.mjs
```

Expected: FAIL because the new extension and pure state module are absent.

- [ ] **Step 4: Implement pure compact session state transitions**

In `pi-extension/router-state.ts`, define a versioned `RouterSessionState` with bounded previous prompt, capability/model/reasons, task depth/failures, route/candidate cursor, override, automatic-switch guard, and maximum attempts. Export pure functions exercised by Step 2.

- [ ] **Step 5: Wire persistence into the PI extension**

Rebuild state only from `ctx.sessionManager.getBranch()` on `session_start`; append versioned state with `pi.appendEntry()` after each route, override change, and settled outcome.

- [ ] **Step 6: Implement routing and candidate activation**

In `before_agent_start`, skip slash commands/empty prompts, call `/api/route` with a short timeout, resolve candidates against `ctx.modelRegistry`, and set the first activatable model. Use an internal boolean/token guard around `pi.setModel()` so `model_select` does not create an override.

- [ ] **Step 7: Implement bounded outcome fallback**

Capture status/headers in `after_provider_response`. At `agent_before_settle`, classify provider versus task failure, report it, and advance at most through the returned candidates. Return `continue: true` only when a next candidate was activated and the per-route attempt limit permits it.

- [ ] **Step 8: Implement manual controls and compact UI**

- `/router-auto`: clear the session override and append state.
- `/router-status`: fetch status and display auto/override, capability, Ollaya state, availability/cooldowns, and last route.
- Normal status: one compact `ROUTE capability · classifier · profile · reason` line.
- Guard UI-only behavior using PI mode/UI capabilities.

- [ ] **Step 9: Add optional service startup without coupling factory initialization**

If auto-start remains configured, perform health check/spawn only in `session_start`, use documented cleanup in `session_shutdown`, and fail open. Do not start processes in the extension factory.

- [ ] **Step 10: Run extension tests**

Run the Step 3 commands.

Expected: PASS, including manual/automatic model-change and fallback-loop review tests.

- [ ] **Step 11: Commit**

```bash
git add pi-extension/turn-router.ts pi-extension/router-state.ts tests/test_pi_extension_structure.py tests/test_pi_extension_state.mjs
git commit -m "feat: add thin turn-aware PI router extension"
```

---

### Task 10: Optional RLCD Evaluation and Legacy Endpoint Deprecation

**Files:**
- Modify: `router/classifier.py`
- Modify: `server/app.py`
- Modify: `tests/test_confidence.py`
- Modify: `tests/test_execute.py`

**Interfaces:**
- Consumes: existing RLCD engine and new benchmark/classification records.
- Produces: explicitly optional RLCD evaluation path and deprecation signals for router-owned execution endpoints.

- [ ] **Step 1: Add tests that production routing never imports or invokes MLX**

Patch module import/call sites to raise and assert `/api/route` still succeeds. Add tests that explicit RLCD evaluation still uses the existing injected callable.

- [ ] **Step 2: Run tests and verify failure where production remains coupled**

```bash
.venv/bin/pytest -q tests/test_confidence.py tests/test_router_api.py -k 'rlcd or mlx or production'
```

Expected: FAIL until imports are isolated.

- [ ] **Step 3: Isolate RLCD under explicit configuration/evaluation entry points**

Keep `RouterClassifier` and schema validation for offline use. Remove misleading production fast/smart comments or mark the cascade experimental. Do not load the 30B model from normal `/api/route`.

- [ ] **Step 4: Deprecate local execution endpoints without deleting them**

Add response headers/docs marking `/api/execute` and `/api/route-and-execute` deprecated. Search repository and installed extension references before scheduling later removal.

- [ ] **Step 5: Run compatibility tests**

```bash
.venv/bin/pytest -q tests/test_confidence.py tests/test_execute.py tests/test_router_api.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add router/classifier.py server/app.py tests/test_confidence.py tests/test_execute.py tests/test_router_api.py
git commit -m "refactor: make RLCD classification optional"
```

---

### Task 11: Configuration and Documentation

**Files:**
- Modify: `README.md`
- Modify: `router/config.py`
- Modify: `.gitignore` only if live Ollaya artifacts need exclusion
- Test: `tests/test_router_api.py`

**Interfaces:**
- Consumes: final registry, API, extension, and Ollaya settings.
- Produces: reproducible configuration instructions without hard-coded provider preference in policy logic.

- [ ] **Step 1: Add configuration serialization assertions**

Assert `/api/route/config` or `/api/router/status` exposes capability, access class, priority, enabled state, Ollaya enabled/model/endpoint (without secrets), and cooldown durations.

- [ ] **Step 2: Run focused test and verify failure**

```bash
.venv/bin/pytest -q tests/test_router_api.py -k 'config or status'
```

Expected: FAIL for any undocumented/unserialized field.

- [ ] **Step 3: Finalize environment-driven settings**

Document and implement exact environment variables for:

```text
local/OMLX model IDs
subscription profile enablement/priorities
metered-cheap OpenRouter/Qwen profile examples
metered-expensive fallback examples
Ollaya enabled/base URL/model/timeout/thresholds
cooldown durations
safe fallback subscription
telemetry/debug mode
```

Example metered profiles remain `enabled=False` until real credentials and provider/model IDs are configured.

- [ ] **Step 4: Rewrite README routing sections**

Describe deterministic fast path, continuity, candidate order, recovery, manual `/model` override behavior, `/router-auto`, `/router-status`, Ollaya-disabled mode, optional RLCD evaluation, and deprecated execution endpoints. Correct all references to the intentionally deleted old extension.

- [ ] **Step 5: Run documentation/config tests**

Run the Step 2 command plus `git diff --check`.

Expected: PASS and no whitespace errors.

- [ ] **Step 6: Commit**

```bash
git add README.md router/config.py tests/test_router_api.py .gitignore
git commit -m "docs: explain turn-aware router configuration"
```

Omit `.gitignore` from `git add` if unchanged.

---

### Task 12: Full Verification, Cleanup, and Review

**Files:**
- Modify only files required to fix verified regressions
- Review all files changed since Task 2

**Interfaces:**
- Consumes: all prior task deliverables.
- Produces: verified implementation, measured latency report, and no premature dead-code deletion.

- [ ] **Step 1: Run the complete test suite**

```bash
.venv/bin/pytest -q
```

Expected: all normal tests PASS; live Ollaya tests SKIP unless explicitly enabled.

- [ ] **Step 2: Run deterministic benchmark**

```bash
.venv/bin/python scripts/benchmark_routing.py
```

Expected: all labeled cases produce results without Ollaya/network; record median/p95 routing latency.

- [ ] **Step 3: Run optional live Ollaya benchmark if approved and installed**

```bash
.venv/bin/python scripts/benchmark_routing.py --ollaya
```

Compare classification disagreements, local/subscription selection, confidence, escalation simulation, and measured latency. Do not enable Ollaya by default unless the evidence supports it.

- [ ] **Step 4: Verify fail-open behavior manually**

With router running, route an obvious rename. Then stop Ollaya and confirm ambiguous routing still returns subscription candidates. Stop the router and confirm the extension retains PI usability. Restore services afterward.

- [ ] **Step 5: Inspect final diff and dead-code references**

```bash
rg -n "FAST_ROUTER_SCHEMA|route-and-execute|api/execute|rlcd-router.ts" . --glob '!*.pyc'
git diff --check
git status --short --branch
```

Do not remove referenced legacy code. Record candidates for a separate cleanup only after usage is proven absent.

- [ ] **Step 6: Request independent code review**

Use `superpowers:requesting-code-review` with the approved spec, plan, commit range, test output, dirty-tree caveat, and Ollaya benchmark result. Fix only technically verified findings.

- [ ] **Step 7: Re-run verification after review fixes**

```bash
.venv/bin/pytest -q
.venv/bin/python scripts/benchmark_routing.py
```

Expected: PASS.

- [ ] **Step 8: Commit any final verified fixes**

If review produced changes, list the exact modified paths with `git status --short`, stage each reviewed path explicitly, inspect `git diff --cached`, then run:

```bash
git commit -m "fix: address turn-aware router review findings"
```

Skip this commit if no changes were required.

- [ ] **Step 9: Use the branch-finishing workflow**

Invoke `superpowers:finishing-a-development-branch` and present merge/PR/keep/discard options. Do not merge, push, or discard without user choice.
