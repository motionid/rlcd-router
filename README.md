# RLCD Multi-Model Router

A local-first, turn-aware routing service for AI coding agents. Each turn is
classified by a **deterministic** rule-based classifier (optionally refined by
a local [Ollaya](https://ollaya.dev) classifier for ambiguous turns), a
capability/access policy turns that into an **ordered list of eligible model
candidates**, and the host agent (Pi) executes the task with the first
activatable candidate and reports the outcome back.

The router never executes substantive work itself and never chooses provider
names from the classifier: the classifier answers one question — *what
capability does this turn require?* (`basic | normal | strong | reasoning`) —
and a separate deterministic policy selects candidates by capability floor,
access class, and priority.

## Why this exists

A single default model is often wasteful:

- routine work is better served by cheap subscription or local models;
- difficult coding or reasoning may need stronger models;
- paid per-token APIs should normally be last resorts;
- private tasks must never leave the machine (`local_only` requirement);
- an explicit user model request should override automatic selection.

## How routing works

```text
Prompt + bounded turn context (previous prompt/capability, task depth,
                                     consecutive task failures)
        │
        ▼
Deterministic classifier            ← production hot path: pure rules,
(task capability + relationship)      no models, no network, no filesystem
        │  ambiguous only, and only when explicitly enabled
        ▼
Optional Ollaya refinement          ← disabled by default; on any failure
(three gated questions)                the deterministic result stands
        │
        ▼
Capability/access policy
(hard filter → access-class order → priority)
        │
        ▼
Ordered candidates + per-profile exclusions (stable reason codes)
        │
        ▼
Host agent (Pi) activates the first activatable candidate, executes,
and reports the outcome → availability cooldowns / task escalation
```

Deterministic precedence rules (binding):

1. strong/reasoning signals outrank trivial verbs — "rename this while
   debugging the race across the repository" is strong, never basic;
2. prompt length is supporting evidence only — short prompts are not
   automatically basic or unambiguous (short, signal-free prompts are marked
   *ambiguous* and conservatively floor at `normal`);
3. clear continuations ("fix it", "continue") inherit the previous task's
   capability; explicitly standalone trivial work is independent;
4. repeated **task** failures raise the required capability one level per
   threshold (default: every 2 consecutive task failures), capped at
   `reasoning`. Provider availability failures never raise capability.

### Capability and access — two independent scales

- **Capability** (`basic < normal < strong < reasoning`): what the task
  requires and the model can do.
- **Access class** (`local | subscription | metered-cheap |
  metered-expensive`): how using a model is granted or paid for.

Every profile registers both. Candidate ordering per required capability:

| Required capability | Access-class order |
|---|---|
| basic | local → subscription → metered-cheap → metered-expensive |
| normal | subscription → local (normal-capable) → metered-cheap → metered-expensive |
| strong | subscription → metered-cheap → metered-expensive |
| reasoning | subscription → metered-cheap → metered-expensive |

A suitable subscription profile therefore always precedes metered (paid)
access, and an **uncertain classification never selects paid access at all**.
Within an access class, lower numeric `priority` wins.

All of this — capability, access class, priority, and enabled state — is
serialized by [`GET /api/route/config`](#api) and
[`GET /api/router/status`](#api).

### The profile registry

Routing configuration lives in [`router/config.py`](router/config.py) and is
environment-tunable (see [Configuration](#configuration)). Provider/model
pairs must exactly match Pi's registry. The shipped registry contains:

- **OMLX/local executors** — e.g. `local_coding` (Qwen3-Coder-30B-A3B via
  oMLX) and `local_general` (Qwen3.8-27B), using the IDs registered by the
  `omlx-models` Pi extension.
- **Subscription executors** — OpenAI Codex GPT models and Antigravity
  Gemini models. Anthropic-backed profiles are intentionally removed; use
  OpenRouter or Gemini-hosted alternatives when Claude-family behavior is
  needed. Profiles absent from the installed Pi registry (Kimi, MAI, Grok,
  and unavailable model versions) are omitted rather than retained as stale
  disabled entries.
- **Disabled metered examples** — `openrouter_qwen` (metered-cheap
  OpenRouter/Qwen example) and `premium_fallback` (metered-expensive premium
  fallback example). Both ship `enabled=False`; see
  [Metered example profiles](#metered-example-profiles) before enabling.
- **Classifier profiles** — `rlcd_fast` / `rlcd_smart` (RLCD models,
  evaluation-only; never executable candidates).

The legacy `gpt_5_3_codex` profile remains as a manual-override compatibility
alias for `gpt_5_6_sol`. Automatic routes and the override catalog expose only
the canonical profile, and cooldowns apply to their shared provider/model pair.

## Manual control (Pi extension)

The optional Pi extension ([`pi-extension/turn-router.ts`](pi-extension/turn-router.ts))
provides manual control commands:

- **`/model <model>`** (Pi's own command, plus model cycling): creates a
  **persistent override**. The override is respected even when no router
  profile maps the selected model — the active model is kept as-is
  (unknown model preserved), no routing happens, and the override survives
  session reloads (restore adopts the model without creating a new
  override). Automatic candidate switches never create or clear an
  override. An override lasts until cleared.
- **`/router-auto`**: clears the manual override and resumes automatic
  routing. No-op (reported) when already automatic.
- **`/router-status`**: shows mode (auto / override), last capability,
  Ollaya enabled/health, active per-profile cooldowns with remaining
  seconds, and a compact last-route line. Server-unreachable is reported
  inline.

The server also accepts `model_override` directly on `POST /api/route`; an
unavailable (cooled-down) or requirement-incompatible override routes the
current turn from the normal pool while the client-side preference is kept —
the server never clears a user's stored preference.

## Failure handling

- **Bounded candidate retries.** A route carries a fixed ordered candidate
  list and a per-route attempt limit (default 3, hard-bounded 1–16). After
  an explicit provider failure the extension activates the *next* candidate
  without reclassifying; activation failures advance the cursor too, so no
  loop can form. At exhaustion the active model is retained.
- **Availability cooldowns.** Provider-side failure categories (`rate_limit`,
  `quota`, `auth`, `capacity`, `provider_unknown`) cool the profile down
  for a configured duration; no category disables a profile permanently.
  Task-side failures (`task_failure`, `local_tool_failure`) never cool a
  provider — the provider was available; the work failed. Cooldown state is
  visible in `/api/router/status`.
- **Retry-After.** A numeric (seconds) `Retry-After` header on a provider
  failure is honored, clamped to the configured safety cap (default 3600 s).
  HTTP-date `Retry-After` values are not parsed; they fall back to the
  category's default cooldown.
- **Escalation.** Consecutive task failures raise the next turn's required
  capability (see precedence rule 4 above); a provider success resets the
  counter.
- **Safe fallback.** If routing itself fails internally, the server never
  500s when it can resolve a safe **subscription** (or, for `local_only`
  requests, local) profile that satisfies the request — paid access is never
  a safe fallback, and an unresolvable fallback returns an empty candidate
  list rather than fabricating a profile.
- **Fail-open active-model behavior.** Every extension↔server interaction
  fails open: an unreachable router, a route timeout, an empty candidate
  list, or a failed outcome report retains the active model and never
  blocks a turn.

## Installation

Requires Python 3.10+. The production routing path uses **no MLX and no
network**; MLX (Apple Silicon) is only needed for the optional RLCD
evaluation endpoints and deprecated legacy execution endpoints.

```bash
git clone <your-repository-url> rlcd-router
cd rlcd-router

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

./run.sh
```

Verify:

```bash
curl http://127.0.0.1:8001/health
# {"status":"ok","service":"rlcd-router"}
```

## Quick start

Route a task:

```bash
curl -s http://127.0.0.1:8001/api/route \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Fix the authentication race, add a regression test, and run the test suite.",
    "turn_context": {
      "previous_prompt": "",
      "previous_model": "",
      "task_depth": 0,
      "task_failures": 0
    }
  }' | python3 -m json.tool
```

The response includes:

```json
{
  "route_id": "...",
  "classification": {
    "capability": "normal",
    "confidence": 0.7,
    "relationship": "new_substantial",
    "ambiguous": false,
    "classifier": "deterministic",
    "reasons": ["..."]
  },
  "signals": {"prompt_chars": 75, "estimated_tokens": 18},
  "decision": {
    "capability": "normal",
    "candidates": [
      {
        "profile_id": "gpt_6_luna",
        "provider": "openai-codex",
        "model": "gpt-6-luna",
        "capability": "normal",
        "access_class": "subscription",
        "priority": 10
      },
      { "profile_id": "gpt_5_6_luna", "...": "..." }
    ],
    "excluded": [
      {"profile_id": "rlcd_fast", "reason": "classifier_profile"},
      {"profile_id": "rlcd_smart", "reason": "classifier_profile"},
      {"profile_id": "gpt_5_3_codex", "reason": "duplicate_provider_model"}
    ],
    "eligible_models": ["gpt_6_luna", "gpt_5_6_luna", "..."],
    "requirements": {"minimum_capability": "general", "allow_cloud": true}
  }
}
```

(Truncated for readability: the candidate/exclusion lists and `signals` are partial, and `requirements` is shown partially.)

Candidates are ordered: activate the first one your host can resolve, then
the next, and so on. Hard caller requirements are optional fields on the
same request:

```json
"requirements": {
  "supports_tools": true,
  "supports_thinking": false,
  "minimum_context": 0,
  "local_only": false
}
```

`local_only: true` keeps execution local and blocks every cloud candidate.

Report the attempt outcome so cooldowns and telemetry work:

```bash
curl -s http://127.0.0.1:8001/api/route/outcome \
  -H 'Content-Type: application/json' \
  -d '{
    "event_id": "<route_id>",
    "status": "failed",
    "failure_category": "rate_limit",
    "candidate_profile": "gpt_6_luna",
    "retry_after": 30
  }' | python3 -m json.tool
```

Valid `failure_category` values: `rate_limit`, `quota`, `auth`, `capacity`,
`provider_unknown`, `task_failure`, `local_tool_failure` (anything else is a
machine-readable 400). A success clears any cooldown on the profile.

## Pi extension

The optional extension is [`pi-extension/turn-router.ts`](pi-extension/turn-router.ts)
(with its pure state module
[`pi-extension/router-state.ts`](pi-extension/router-state.ts)).

It wires Pi's lifecycle to the router:

- `before_agent_start`: routes each non-command turn (skipped while a manual
  override is active) with bounded turn context, resolves every candidate
  through Pi's model registry up front, and activates the first activatable
  one. Guarded automatic switches are never mistaken for user overrides.
- `after_provider_response` / `agent_before_settle`: classify the attempt
  (provider availability failure vs task failure), report the outcome,
  and run the bounded candidate failover described above.
- `model_select`: user `/model` selections create persistent overrides;
  restore and automatic switches never do.
- Commands: `/router-auto`, `/router-status`.

### Install

```bash
# Install both files as one multi-file extension directory. Remove the old
# single-file link first if upgrading from an earlier installation.
rm -f ~/.pi/agent/extensions/turn-router.ts
mkdir -p ~/.pi/agent/extensions/turn-router
ln -sf "$(pwd)/pi-extension/turn-router.ts" \
  ~/.pi/agent/extensions/turn-router/index.ts
ln -sf "$(pwd)/pi-extension/router-state.ts" \
  ~/.pi/agent/extensions/turn-router/router-state.ts
```

Then restart Pi or run `/reload`. Keep both files together if you copy the
extension instead of symlinking it. The extension resolves the router
directory through its entry point's real path; `run.sh` is located there for
auto-start.

> **Migration note:** the previous extension (`rlcd-router.ts`), its
> `/route`, `/model-override`, and `/model-exhausted` commands, and the
> `rlcd_classify` tool were removed intentionally. Use `/router-auto`,
> `/router-status`, Pi's own `/model`, and `POST /api/run-parallel` for
> explicit RLCD evaluation instead.

### Extension settings

| Variable | Default | Meaning |
|---|---|---|
| `RLCD_ROUTER_URL` | `http://127.0.0.1:8001` | Router server base URL |
| `RLCD_ROUTER_TIMEOUT_MS` | `2000` | `/api/route` timeout — an outage blocks the turn by at most this long, then fails open |
| `RLCD_ROUTER_STATUS_TIMEOUT_MS` | `2000` | `/api/router/status` timeout (`/router-status`) |
| `RLCD_ROUTER_HEALTH_TIMEOUT_MS` | `1000` | `/health` probe timeout (auto-start) |
| `RLCD_ROUTER_AUTO_START` | off | **Server auto-start is off by default.** When enabled, `session_start` health-probes first and spawns `run.sh` only if no server answers |

Auto-start ownership limitations: the extension only ever stops a server
process **it** spawned in the current runtime; a pre-existing (shared)
server is never killed. Consequently a spawned server can outlive a session
(or be re-spawned) across extension reload / new-session boundaries — the
default (off) avoids this entirely.

## Optional Ollaya ambiguity classifier

[Ollaya](https://ollaya.dev) is a small local classifier service that the
router consults **only** for turns the deterministic classifier marks
ambiguous, and only when explicitly enabled (`ROUTER_OLLAYA_ENABLED=1`;
disabled by default pending threshold calibration against labeled fixtures).
It never selects providers, models, access classes, or spending policy —
it only refines the task capability label. Every adapter failure (disabled,
timeout, unreachable, malformed response) falls back to the deterministic
result, which is already the conservative subscription-safe choice.

Install (official command) and start:

```bash
curl -fsSL https://ollaya.dev/install.sh | sh
ollaya serve          # binds 127.0.0.1:11435 (macOS auto-starts on any CLI use)
```

The adapter is a direct HTTP client (`router/ollaya.py`) against
`POST /api/decide`; it sends `keep_alive: -1` so the model stays resident,
and asks three gated questions:

| Question | Gate (`OllayaConfig` field) | Default |
|---|---|---:|
| `required_capability == basic` (local downgrade) | `basic_threshold` | `0.90` |
| `is_task_continuation` (inherit previous capability) | `continuation_threshold` | `0.75` |
| above-`normal` capability upgrade | `capability_threshold` | `0.60` |

(The third question, `is_independent_trivial_task`, is diagnostic telemetry
only and never gates routing on its own.)

Measured on this machine (Ollaya 0.6.1, Apple Silicon, CPU): **~205 ms warm /
~2.17 s cold** decide latency — machine-specific local evidence, not
published figures. With `keep_alive: -1` the cold cost is paid once per
server lifetime. The default `timeout_ms` of 500 accommodates the warm case;
a cold first request exceeds it and safely falls back to the deterministic
classification.

## RLCD: evaluation-only

The original RLCD model cascade is **evaluation-only**. It is reachable
only through explicit endpoints — `POST /api/run-parallel` (alias
`/api/run-rlcd`) — and the deprecated execution endpoints below. The
production `/api/route` path never loads MLX, never builds the RLCD
classifier, and never consults RLCD. The `ConfidenceThresholds` in
`router/config.py` (serialized by `/api/route/config`) tune only that legacy
cascade and exist so offline evaluation runs are reproducible.

### Deprecated execution endpoints

`POST /api/execute` and `POST /api/route-and-execute` are **deprecated**:
router-owned execution is migrating to the host (Pi), which always owned
cloud execution. They remain functional during the migration window and
carry RFC 9745 `Deprecation: true` plus an RFC 8594 `Sunset` header
(provisional removal horizon, default `Fri, 31 Dec 2027 23:59:59 GMT`):

- `ROUTER_LEGACY_SUNSET` overrides the Sunset date (an arbitrary string is
  passed through verbatim).
- An **empty** value omits the `Sunset` header entirely (deprecation
  signaled by `Deprecation: true` alone).

New integrations must use `POST /api/route` and execute through the host.

## Configuration

Priorities, enablement, local model IDs, Ollaya, cooldowns, and metered
examples are environment-driven; shipped provider/model defaults match the
Pi registry used by this installation.

### OMLX / local model IDs

| Variable | Default | Meaning |
|---|---|---|
| `OMLX_MODELS_DIR` | `~/.omlx/models` | oMLX model store used by RLCD and deprecated MLX execution |
| `RLCD_FAST_MODEL` | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | `rlcd_fast` classifier model ID |
| `RLCD_SMART_MODEL` | `$OMLX_MODELS_DIR/.../Qwen3-Coder-30B-A3B-Instruct-4bit` | `rlcd_smart` classifier model ID |
| `LOCAL_GENERAL_MODEL` | `Qwen3.8-27B-MLX-4bit` | `local_general` Pi/oMLX registry ID |
| `LOCAL_CODING_MODEL` | `Qwen3-Coder-30B-A3B-Instruct-4bit` | `local_coding` Pi/oMLX registry ID |
| `LEGACY_LOCAL_GENERAL_MODEL` | `$OMLX_MODELS_DIR/lmstudio-community/Qwen3.8-27B-MLX-4bit` | Filesystem path for deprecated MLX execution only |
| `LEGACY_LOCAL_CODING_MODEL` | `$OMLX_MODELS_DIR/mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` | Filesystem path for deprecated MLX execution only |
| `SPLASH_PROVIDER` | `splash-local` | `splash_local` provider key |

### Subscription / all profile enablement and priorities

| Variable | Example | Meaning |
|---|---|---|
| `ROUTER_PRIORITY_<PROFILE>` | `ROUTER_PRIORITY_GPT_5_MINI=5` | Per-profile priority (lower wins within an access class) |
| `ROUTER_ENABLE_<PROFILE>` | `ROUTER_ENABLE_LOCAL_CODING=1` | Force a profile on (e.g. re-enable a metered example after configuring credentials) |
| `ROUTER_DISABLE_<PROFILE>` | `ROUTER_DISABLE_GPT_6_LUNA=1` | Force a profile off (e.g. a suspended seat). Disable wins over enable; unset keeps the declared default |

Truthy values: `1`, `true`, `yes`, `on` (case-insensitive). Profiles disabled
because their provider/model pair is absent from Pi should only be re-enabled
after their configured pair exists in Pi's model registry.

> **Migration:** `LOCAL_GENERAL_MODEL` and `LOCAL_CODING_MODEL` now mean Pi
> registry IDs, not filesystem paths. Remove old path-style exports or replace
> them with the IDs above. Use `LEGACY_LOCAL_*` only if the deprecated MLX
> execution endpoints need a non-default on-disk path.

### Metered example profiles

`openrouter_qwen` (metered-cheap) and `premium_fallback` (metered-expensive)
are registered as **disabled examples**: the router ships without metered
credentials, and these profiles must stay off until real provider/model IDs
are configured. To use one, set its IDs, configure credentials with the
host's provider settings (never in this repository), and enable it:

```bash
export ROUTER_METERED_CHEAP_MODEL="qwen/qwen3-coder"        # or your ID
export ROUTER_METERED_CHEAP_PROVIDER="openrouter"
export ROUTER_ENABLE_OPENROUTER_QWEN=1
# or, for the premium fallback:
export ROUTER_METERED_EXPENSIVE_MODEL="z-ai/glm-5.3"
export ROUTER_METERED_EXPENSIVE_PROVIDER="openrouter"
export ROUTER_ENABLE_PREMIUM_FALLBACK=1
```

Even when enabled, metered access is only reachable after all suitable
subscription candidates (per the capability-specific access order), and
uncertain classifications never select it.

**Disabled profiles are not valid router override choices**: while a
profile is disabled it does not appear in `GET /api/models/overrides`, and
a direct `model_override` naming it is reported incompatible
(`profile_disabled`) with the turn routed from the normal pool — the
preference is kept in metadata but the disabled profile never becomes the
selected candidate. Enable a profile (`ROUTER_ENABLE_<NAME>=1`) to make it
an override choice again.

### Ollaya

| Variable | Default | Meaning |
|---|---|---|
| `ROUTER_OLLAYA_ENABLED` | off | Opt-in ambiguity refinement |
| `ROUTER_OLLAYA_URL` | `http://127.0.0.1:11435` | Ollaya endpoint |
| `ROUTER_OLLAYA_MODEL` | `laya` | Ollaya router model |
| `ROUTER_OLLAYA_TIMEOUT_MS` | `500` | Decide timeout (warm case) |
| `ROUTER_OLLAYA_BASIC_THRESHOLD` | `0.90` | `basic` downgrade acceptance probability |
| `ROUTER_OLLAYA_CONTINUATION_THRESHOLD` | `0.75` | Continuation acceptance probability |
| `ROUTER_OLLAYA_CAPABILITY_THRESHOLD` | `0.60` | Above-normal upgrade acceptance confidence |

### Cooldowns

| Variable | Default (s) | Category |
|---|---:|---|
| `ROUTER_COOLDOWN_RATE_LIMIT` | `60` | transient rate limit |
| `ROUTER_COOLDOWN_QUOTA` | `900` | usage/quota limit |
| `ROUTER_COOLDOWN_AUTH` | `300` | temporary authentication issue |
| `ROUTER_COOLDOWN_CAPACITY` | `120` | capacity/overload |
| `ROUTER_COOLDOWN_UNKNOWN` | `300` | unknown provider failure (conservative) |
| `ROUTER_COOLDOWN_RETRY_AFTER_CAP` | `3600` | safety bound for honored numeric `Retry-After` |

### Safe fallback

| Variable | Default | Meaning |
|---|---|---|
| `ROUTER_SAFE_FALLBACK_PROFILE` | *(unset)* | Pin the internal-error safe fallback to a profile (e.g. `gpt_5_6_sol`). The pin is honored only when it passes the same hard filter (enabled, capability floor, requirements, availability, non-paid access); an ineligible or unknown pin is ignored and automatic resolution (lowest-priority eligible subscription profile) takes over |

### Telemetry, cost estimates, and debug mode

| Variable | Default | Meaning |
|---|---|---|
| `ROUTER_TELEMETRY_PATH` | `data/router_events.jsonl` | Telemetry log (one JSONL record per route with an attempts array; prompt text is never written, only a hash) |
| `ROUTER_DEBUG` | off | `1` enables DEBUG logging (telemetry failures, fallback paths) |
| `ROUTER_ESTIMATED_PAID_REQUEST_COST` | `0.0` | Optional per-request USD estimate: when set, a route that chose a non-paid candidate while a paid candidate was eligible records the avoided spend (`estimated_spend_avoided`); `0.0` disables estimation |

### Server settings

| Variable | Default | Meaning |
|---|---|---|
| `RLCD_ROUTER_HOST` / `RLCD_ROUTER_PORT` | `127.0.0.1` / `8001` | `run.sh` bind address |
| `RLCD_PYTHON` | `<repo>/.venv/bin/python` | Interpreter `run.sh` uses |
| `ROUTER_CORS_ORIGINS` | same-origin defaults | Comma-separated allowlist for browser clients |
| `ROUTER_TASK_FAILURE_THRESHOLD` | `2` | Consecutive task failures per capability escalation level |

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Identify the service and confirm readiness |
| `POST` | `/api/route` | **Production route**: classify a turn, return ordered candidates and exclusions |
| `POST` | `/api/route/outcome` | Record an attempt outcome (availability cooldown vs task failure) |
| `GET` | `/api/route/config` | Profiles (capability/access/priority/enabled), tiers, evaluation-only thresholds, cooldown durations |
| `GET` | `/api/router/status` | Compact status: per-profile access/capability/enabled/availability, Ollaya config (no secrets), cooldown durations, last route |
| `GET` | `/api/route/telemetry` | Aggregate routing metrics |
| `POST` | `/api/models/recommend` | Subscription switch recommendation after quota exhaustion |
| `GET` | `/api/models/overrides` | Valid override targets grouped by billing (enabled executors only — disabled profiles, e.g. the metered examples, are excluded until enabled) |
| `POST` | `/api/run-parallel` (alias `/api/run-rlcd`) | **RLCD evaluation only**: arbitrary constrained classification (loads MLX) |
| `POST` | `/api/execute` | *Deprecated* legacy local execution |
| `POST` | `/api/route-and-execute` | *Deprecated* legacy route + local execution |
| `GET` | `/api/models` | *Legacy*: list models loaded in memory (loads MLX) |

Interactive API documentation is available at `http://127.0.0.1:8001/docs`.

## Project layout

```text
core/                   RLCD/MLX engine (evaluation + deprecated legacy endpoints only)
  engine.py             Public API for the bundled MLX backend
  engine_mlx.py         MLX RLCD and local text generation
  schema.py             Structured schema compilation and calibration
router/
  deterministic.py      Production rule-based classifier (hot path)
  ollaya.py             Optional Ollaya HTTP adapter (direct HTTP, no SDK)
  policy.py             Capability/access candidate policy
  config.py             Env-driven profiles, cooldowns, Ollaya and thresholds
  availability.py       Failure categories and cooldown store
  escalation.py         Task-failure capability escalation
  fallback.py           Subscription switch recommendations
  decision.py           Route decision / candidate / exclusion structures
  signals.py            Bounded runtime signals
  telemetry.py          Single-record JSONL events and aggregates
  classifier.py         Legacy RLCD cascade (evaluation-only)
  router_schema.py      Legacy task-requirement schema
server/
  app.py                FastAPI service
pi-extension/
  turn-router.ts        Pi extension (routing, overrides, bounded failover)
  router-state.ts       Pure extension state machine (no runtime imports)
tests/                  Unit, API, and replay tests
```

## Testing

```bash
pytest -q
npm run test:extension
```

The root `package.json` marks the Pi extension sources as ES modules and
provides the explicit Node test command (Node does not discover these `.mjs`
files reliably from the directory alone).

The suites mock model execution and network access — production routing
tests run without MLX and without a live Ollaya server. The one live
Ollaya test is gated behind `OLLAYA_LIVE_TEST=1` and skips otherwise.
Tests that do exercise the legacy MLX endpoints require `mlx`/`mlx-lm`
(Apple Silicon); the pure `router/` modules remain reusable elsewhere.

## Security and privacy

- The service binds to `127.0.0.1` by default; do not expose it publicly
  without authentication and strict CORS settings.
- `local_only` is a hard policy constraint, not a preference.
- Prompt text is never written to telemetry — only a hash and aggregate
  metadata; the Ollaya adapter sends only the bounded turn context it
  needs to the local endpoint.
- No metered credentials are stored in this repository; provider
  credentials belong in the host agent's provider configuration, and the
  metered example profiles stay disabled until they are configured.
- Keep API keys out of the repository; `.env` is git-ignored.

## Current limitations

- The bundled MLX backend (evaluation/legacy endpoints) requires Apple
  Silicon; production routing does not.
- The profile registry reflects one specific Pi provider catalog and local
  oMLX installation; adjust `router/config.py` or the environment variables
  for your environment.
- Status Ollaya `health` is a static label (`disabled`/`unknown`); there is
  no per-route live probe by design.
- Provider failures without an HTTP response are conservatively treated as
  task failures (never a provider cooldown) unless the status/headers were
  captured.
- The route index, availability store, and telemetry globals are lock-free
  under FastAPI's threadpool concurrency (acceptable for a single-process
  local service).

## Attribution

This router builds on the RLCD model and engine work in the original
[Qwen-2.5-1B-RLCD project by harshatheg](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD).

## License

Add the license appropriate for your project before publishing.
