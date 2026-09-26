# Simplified Turn-Aware PI Router Design

**Date:** 2026-09-26  
**Status:** Approved in conversation; pending written-spec review

## 1. Purpose

Refactor the existing RLCD router incrementally into a lightweight, turn-aware router for the PI coding harness.

The router selects the lowest-cost available model whose capability satisfies the current task:

```text
minimize access cost
subject to model capability >= task capability
```

The production hot path must be deterministic for obvious requests. Ollaya may classify ambiguous requests locally, but it never chooses providers or controls spending. The existing 30B RLCD classifier remains optional for evaluation and research rather than being required for interactive routing.

## 2. Goals

- Route obvious basic work to capable local models.
- Prefer already-paid subscription models for substantive coding and reasoning.
- Use metered APIs only when suitable subscriptions are unavailable or incapable.
- Reconsider model selection on every PI turn without losing task continuity.
- Return an ordered candidate list so failures do not require reclassification.
- Detect subscription/provider failures and temporarily remove affected models.
- Automatically restore recovered subscriptions to the candidate pool.
- Keep manual model selection available until explicitly returned to auto mode.
- Make every route concise, explainable, measurable, and fail-open.
- Work with both Ollaya and the RLCD classifier disabled.

## 3. Non-goals

- Predicting the theoretically best model for every prompt.
- Calling a generative LLM for every routing decision.
- Letting Ollaya select providers or spending classes.
- Solving coding tasks inside the router.
- Persisting full prompts, repository contents, or conversation transcripts.
- Adding embeddings, vector databases, remote classification, or agent chains.
- Restoring the intentionally deleted PI extension.
- Rewriting the repository from scratch.

## 4. Current Code Reuse

Retain and simplify these existing components:

- `router/policy.py`: deterministic policy boundary.
- `router/config.py`: model-profile registry concept.
- `router/signals.py`: deterministic fact extraction.
- `router/decision.py`: immutable decision and ordered candidate output.
- `router/escalation.py`: failure and escalation concepts.
- `router/telemetry.py`: lightweight JSONL telemetry.
- `server/app.py`: local HTTP service and existing compatibility endpoints.

Preserve for optional evaluation rather than production routing:

- `router/router_schema.py`
- `router/classifier.py`
- `core/engine.py`
- `core/engine_mlx.py`
- the 30B RLCD classifier and related schemas

The old disabled PI extension is reference material only. A new, smaller extension will be designed and implemented independently.

## 5. Architecture

```text
PI before_agent_start
        |
        v
thin PI extension builds TurnContext
        |
        v
Python router deterministic signals
        |
   clear|ambiguous
        |     |
        |     v
        |  optional Ollaya adapter
        |     |
        +-----+
        |
        v
minimum task capability
        |
        v
registry filters:
capability + tools + context + availability
        |
        v
access policy + configured priority
        |
        v
ordered candidates
        |
        v
PI resolves and activates first usable model
        |
 success|failure
        |     |
        |     v
        |  availability update + next candidate
        v
telemetry + compact session state
```

### Responsibility split

#### Thin PI extension

- Observe each user turn.
- Maintain compact, branch-sensitive session continuity state.
- Send `TurnContext` to the Python router.
- Resolve profile IDs against PI's model registry.
- Call `pi.setModel()` before execution.
- Capture provider status, Retry-After, tool failures, and final outcome.
- Advance through the returned candidate list with bounded retries.
- Persist session state using PI custom entries.
- Provide `/router-auto` and `/router-status`.
- Fail open if any routing component fails.

#### Python router

- Extract deterministic task signals.
- Classify obvious tasks without model inference.
- Invoke Ollaya only for ambiguous requests when enabled.
- Determine minimum task capability.
- Maintain model/provider availability and cooldown state.
- Filter and order eligible model candidates.
- Record telemetry and expose compact status.
- Provide optional RLCD evaluation interfaces.

#### Ollaya

- Answer narrow task-classification questions.
- Receive only bounded task/session state.
- Never choose a provider, model, access class, or spending policy.
- Remain optional and non-blocking.

## 6. Task Capability

Use one ordered capability scale:

```text
basic < normal < strong < reasoning
```

### Basic

Examples:

- variable rename
- typo, comment, or README wording
- formatting
- summarizing command output
- commit-message writing
- simple text or shell operations
- trivial code edits

### Normal

Examples:

- typical functions
- ordinary unit tests
- straightforward bug fixes
- routine implementation work

### Strong

Examples:

- multi-file debugging
- architecture changes
- non-obvious bugs
- substantial refactors
- repository-wide reasoning

### Reasoning

Examples:

- difficult unresolved failures
- repeated failed attempts
- deep architecture reasoning
- complex debugging after weaker attempts failed

Capability describes task requirements only. It does not encode local/cloud location, subscription status, or price.

## 7. Model Registry

Each enabled profile contains only routing-relevant data:

```text
id
provider
model
capability
access_class
priority
enabled
local
context_limit
supports_tools
supports_thinking
optional coding-strength metadata
optional cost metadata
optional provider metadata
```

Access classes are:

```text
local
subscription
metered-cheap
metered-expensive
```

Capability is ordinal. A model is eligible only when its declared capability is at least the required task capability and it satisfies hard requirements such as tools and context size.

Provider and model preference is configuration, not routing code. Lower numeric priority wins within an access-policy group.

## 8. Candidate Policy

### Basic tasks

```text
local
subscription
metered-cheap
metered-expensive
```

### Normal tasks

```text
subscription
explicitly normal-capable local
metered-cheap
metered-expensive
```

### Strong tasks

```text
strong subscription models
capable metered-cheap models
metered-expensive models
```

### Reasoning tasks

```text
strongest suitable subscription models
capable metered models
```

The policy excludes disabled, cooling-down, unavailable, under-capable, or requirement-incompatible profiles before returning candidates.

Classification uncertainty must not directly select paid API access. Uncertain basic/normal work falls back to a suitable subscription. Metered access is selected only because subscriptions are unavailable or incapable.

## 9. Deterministic Classification

The deterministic classifier is the primary hot path.

### Basic signals

High-confidence patterns include bounded forms of:

- rename X to Y
- fix typo
- update comment
- update README wording
- format data
- summarize supplied output
- write a commit message

These patterns must be constrained so words such as `refactor` or `debug` elsewhere cannot be hidden by a basic verb.

### Strong signals

Examples include:

- architecture or design changes
- root-cause analysis
- race conditions or concurrency
- trace, investigate, or debug across files
- repository-wide scope
- multi-file changes
- unresolved tests or previous failures

### Supporting signals

- referenced or attached file count
- tool requirements
- repository-wide wording
- previous task capability
- task depth
- prior task and provider failures
- test failures
- context size
- explicit user hint

Prompt length is supporting evidence only. Short prompts are not assumed to be basic.

### Deterministic result

The classifier returns:

```text
capability
confidence
relationship: independent | continuation | new_substantial | ambiguous
reasons
ambiguous: boolean
```

## 10. Task Continuity

The extension keeps bounded per-session state:

```text
previous request text or compact bounded summary
current task capability
previous model
previous route reason
current task depth
files involved
consecutive task failures
last route id
candidate cursor
manual override
```

It does not store the full transcript for routing.

### Continuations

Phrases such as:

- fix it
- continue
- do that
- try again
- why did that fail?
- do the same for the rest

normally inherit the current task capability.

### Independent trivial work

Explicitly independent outputs such as:

- write a commit message
- update README wording
- summarize this output

may downgrade immediately even after a strong task.

### Ambiguity

If deterministic evidence cannot distinguish continuation from an independent request, optional Ollaya may decide. Without Ollaya, ambiguity falls back conservatively to a subscription-capable tier rather than local or paid API.

### Failure effects

- A task/execution failure may increase minimum capability by one level.
- Repeated failures are capped at `reasoning`.
- Provider availability failures do not increase task capability.
- Successful independent work resets task depth.
- Successful continuations retain bounded continuity state.

## 11. Ollaya Adapter

Provide an optional adapter with a router-owned interface such as:

```text
classify(turn_context) -> ClassificationResult
```

The rest of the router does not depend on Ollaya response structures.

The initial question set is intentionally small:

1. `required_capability`: basic / normal / strong / reasoning
2. `is_task_continuation`: probability
3. `is_independent_trivial_task`: probability

A separate `safe_for_local` question is added only if evaluation proves it provides independent value.

### Invocation rules

Do not invoke Ollaya for deterministic high-confidence requests. Invoke it only for ambiguous classifications and only when enabled.

### Confidence rules

- High-confidence basic may permit local routing.
- Low or conflicting confidence falls back to an appropriate subscription.
- Ollaya uncertainty never directly selects metered access.
- Router-owned configurable thresholds interpret probabilities.

### Failure behavior

Timeout, connection error, unavailable service, invalid schema, malformed answer, or disabled configuration immediately falls back to deterministic classification. Ollaya is never a single point of failure.

### Initial model verification

Before integration:

1. Install Ollaya only after explicit user approval.
2. Verify the official CLI and `GET /api/version`.
3. Test the official default `laya` model or, if minimizing the initial download is preferred, one `laya:en` checkpoint.
4. Run real routing examples and measure warm and cold Apple Silicon CPU latency.
5. Inspect probabilities and confidence behavior.
6. Compare decisions against labeled fixtures.

Do not assume published GPU latency applies to this machine.

### Modelfile

Start with direct `/api/decide` requests while evaluating questions. Create a project-owned Modelfile only after the question set proves useful. A Modelfile should then hold stable questions and, later, project-specific calibration. Do not add calibration without labeled validation data.

## 12. Availability and Cooldowns

The Python service keeps small in-memory records per model or provider:

```text
status
cooldown_until
reason
failure_count
last_failure_at
optional retry_after
```

Initial status is `available` unless configuration disables the profile.

### Failure categories

- transient rate limit: short configurable cooldown
- explicit Retry-After: honor provider value within safety bounds
- usage/quota limit: medium configurable cooldown
- temporary authentication/capacity issue: configurable cooldown
- unknown provider failure: conservative short cooldown
- model execution/task-quality failure: no provider cooldown unless availability evidence exists

Cooldown expiry automatically returns a profile to eligibility. No single failure permanently disables a subscription.

Initial availability persistence across router restarts is unnecessary. Configuration remains durable; runtime cooldowns are ephemeral.

## 13. Ordered Attempts and Escalation

A route returns one already-filtered ordered candidate list. Classification runs once.

The extension maintains a bounded candidate cursor:

1. Resolve the next profile against PI's model registry.
2. Attempt `pi.setModel()`.
3. Execute the turn.
4. On retryable activation/provider/tool failure, report the outcome.
5. Move to the next candidate.
6. At `agent_before_settle`, request one guarded continuation when appropriate.

Attempt count and candidate exhaustion prevent loops. A user-aborted turn is not a model failure.

A failed local tool attempt may advance to subscription without increasing provider cooldown. A provider quota/rate failure both advances the cursor and updates availability.

## 14. Explicit Overrides

Manual PI model selection, including `/model` and cycling, becomes a session override when the selected model maps to a configured profile.

Automatic `pi.setModel()` calls use an internal guard so their resulting `model_select` events are not mistaken for user overrides.

The override persists for the session until `/router-auto` clears it. While active, automatic classification is bypassed.

If the override becomes unavailable:

- preserve the user's stored preference;
- warn concisely;
- use automatic candidates for that turn;
- retry the override on a future turn after cooldown expiry unless the user clears it.

## 15. New PI Extension

The new extension is deliberately small and does not restore the old deleted file.

It uses documented PI APIs:

- `before_agent_start` for per-turn routing before execution;
- `pi.setModel()` for session model activation;
- `after_provider_response` for status and response headers;
- `agent_before_settle` for bounded fallback continuation;
- `model_select` for manual override detection;
- `pi.appendEntry()` for non-context session state;
- `ctx.sessionManager.getBranch()` to rebuild branch-sensitive state;
- `ctx.ui.setStatus()` and `notify()` for compact feedback.

The extension must work without terminal UI and guard UI behavior where necessary.

Long-lived resources start from `session_start`, not extension factory initialization. Cleanup is idempotent in `session_shutdown`.

## 16. API

### `POST /api/route`

Input includes the current prompt and compact `TurnContext`.

Output includes:

```text
route_id
capability
relationship
classifier source
confidence
reasons
ordered candidate profile ids
availability exclusions
```

### `POST /api/route/outcome`

Input includes:

```text
route_id
candidate profile
status
failure category
provider status
retry_after
latency
optional cost/usage
```

The endpoint updates availability and telemetry.

### `GET /api/router/status`

Returns compact information for `/router-status`:

- auto or override mode supplied by the extension
- current classifier configuration
- Ollaya health/configuration
- model availability and cooldowns
- last route and reasons

Existing APIs remain temporarily for compatibility while consumers are checked. `/api/execute` and `/api/route-and-execute` are deprecated before any removal.

## 17. Observability and Telemetry

Normal route feedback is compact, for example:

```text
ROUTE basic · deterministic · qwen_local · simple_edit
```

Debug mode may show signals, excluded candidates, confidence, thresholds, and availability decisions.

Each JSONL route record contains one route and an attempts array rather than appending duplicate mutated events.

Required aggregate metrics:

- local, subscription, metered-cheap, and metered-expensive usage
- usage per model
- paid-selection reason
- deterministic versus Ollaya classification count
- Ollaya latency and confidence
- escalation rate
- local failure and escalation rate
- availability failure categories
- estimated spend incurred
- estimated spend avoided when configured cost metadata permits estimation

Telemetry failure never blocks routing.

## 18. Fail-Safe Behavior

If Ollaya fails, deterministic classification continues.

If classification fails, route to a configurable safe subscription default.

If availability state fails, treat configured enabled models as available for the current decision.

If telemetry fails, return the route normally.

If the Python router is unreachable, the extension tries the configured fallback subscription; if that cannot be resolved, it retains PI's active model and allows the turn to proceed.

No minor router failure may break the coding session.

## 19. Performance

- Deterministic signal extraction and policy target: sub-millisecond to a few milliseconds.
- No repository scan on each turn.
- No network classifier.
- Ollaya invoked only on ambiguity.
- HTTP calls use short configurable timeouts.
- Model configuration may be cached by the extension and invalidated explicitly.
- Actual deterministic and Ollaya latency must be measured on this machine.

## 20. Test Strategy

Normal tests use no downloaded models, Ollaya service, RLCD model, or network access.

Required routing tests:

1. Simple rename routes local.
2. README change routes local.
3. Normal coding routes to an available subscription.
4. Multi-file debugging routes to a strong subscription.
5. `fix it` after strong debugging retains strong capability.
6. Commit-message work after a strong task can downgrade to local.
7. Unavailable Claude subscription falls back to the next subscription.
8. All subscriptions unavailable fall back to cheap metered API.
9. Subscription recovery makes it eligible again.
10. Explicit model override bypasses routing.
11. Paid API is not selected while an appropriate subscription is available.
12. Local failure advances to the next candidate.
13. Repeated task failures increase required capability.
14. Router failure falls back safely.
15. Candidate ordering follows configuration.
16. Short ambiguous prompts are not automatically basic.
17. Ollaya unavailability does not prevent routing.
18. Obvious trivial requests do not call Ollaya.
19. Ollaya can classify an ambiguous continuation.
20. Low Ollaya confidence falls back to subscription, not paid API.
21. Ollaya can be disabled completely.
22. Normal routing requires no network classifier.

Additional coverage:

- failure-category cooldown durations
- Retry-After handling
- cooldown expiry and subscription re-entry
- provider failure versus task failure distinction
- context/tool capability filtering
- manual PI model-selection override detection
- internal automatic-switch guard
- branch-sensitive session-state reconstruction
- bounded continuation and attempt-loop prevention
- telemetry route/attempt representation
- optional live Ollaya smoke and latency tests

A small labeled replay fixture compares deterministic-only and deterministic-plus-Ollaya classifications, selected capability, selected model, Ollaya latency/confidence, and eventual escalation outcome.

## 21. Incremental Migration

1. Add new capability/access schemas alongside existing tiers.
2. Add deterministic classification tests and implementation.
3. Add registry filtering, ordering, and availability tests.
4. Add optional Ollaya adapter with mocked tests.
5. Simplify `/api/route` while preserving compatibility.
6. Add the new thin PI extension.
7. Add outcome-driven cooldown and bounded candidate retries.
8. Fix telemetry representation and add status output.
9. Verify Ollaya live only after explicit installation approval.
10. Benchmark and compare deterministic versus Ollaya routing.
11. Deprecate unused execution endpoints.
12. Remove dead code only after usage checks and full regression tests.
13. Update README and configuration examples.

## 22. Documentation Deliverables

Document configuration for:

- OMLX/local executor profiles
- subscription profiles and priority
- OpenRouter/Qwen or other cheap metered profiles
- premium paid fallbacks
- capability declarations
- access classes
- cooldowns and thresholds
- Ollaya endpoint/model/timeout/confidence settings
- Ollaya-disabled mode
- optional RLCD evaluation mode
- `/router-auto` and `/router-status`

## 23. Acceptance Criteria

The design is complete when implementation can demonstrate:

- model choice changes appropriately between independent PI turns;
- ambiguous continuation preserves difficult-task capability;
- obvious trivial work downgrades without Ollaya;
- appropriate subscriptions prevent paid API selection;
- failures advance through ordered candidates without reclassification;
- subscription cooldowns expire and subscriptions re-enter automatically;
- manual selection persists until auto mode is restored;
- all routing remains usable with Ollaya and RLCD disabled;
- routing and telemetry failures do not break PI;
- deterministic routing overhead meets the local latency target;
- the requested routing test matrix passes.
