# RLCD Multi-Model Router

A local-first routing service for AI coding agents. It classifies what a task requires, applies deterministic policy, and returns an ordered list of eligible model profiles.

The classifier never chooses provider names directly. It predicts requirements such as coding intensity, reasoning depth, privacy, tool use, and failure cost. A separate policy then selects the least-expensive tier likely to complete the task.

## Why this exists

A single default model is often wasteful:

- small local models are enough for routine work;
- difficult coding or reasoning may need stronger subscription models;
- paid APIs should normally be fallbacks;
- private tasks must never leave the machine;
- an explicit user model request should override automatic selection.

This project implements those rules behind a small FastAPI service and includes an optional Pi extension.

## Routing strategy

```text
Prompt + runtime signals
        │
        ▼
Local RLCD classifier
(task requirements, not model names)
        │
        ▼
Deterministic policy
(hard constraints → capability → cost)
        │
        ▼
Tier + ordered eligible model profiles
        │
        ▼
Host agent executes and reports outcome
```

The default escalation chain is:

```text
local_fast
  → local_general
  → local_coding
  → local_reasoning
  → cloud_coding
  → cloud_reasoning
  → cloud_research
```

`privacy=local_only` or `privacy=sensitive` blocks cloud routing and cloud fallback.

## Features

- **Parallel constrained classification:** all routing fields are evaluated in one RLCD pass.
- **Deterministic policy:** model selection is auditable and independently testable.
- **Local-first routing:** local execution is preferred when capability permits.
- **Billing-aware fallbacks:** subscription models precede paid-per-token models.
- **User overrides:** callers can request a specific model profile for one task.
- **Ordered fallbacks:** decisions contain `eligible_models`, not only one model.
- **Escalation:** failed attempts can move to stronger tiers.
- **Telemetry:** route decisions, confidence, latency, outcomes, and cost can be recorded.
- **Multi-session-safe startup:** the Pi extension checks the router health endpoint before starting another process.

## Requirements

- Python 3.10+
- Apple Silicon for the included MLX inference backend
- An MLX-compatible instruction model
- Approximately 16–24 GB of available unified memory for the default 30B MoE classifier

The routing policy itself is pure Python and can be used without MLX. Replacing the classifier backend only requires supplying a compatible callable to `RouterClassifier`.

## Installation

```bash
git clone <your-repository-url> rlcd-router
cd rlcd-router

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure the local classifier model with a Hugging Face model ID or local path:

```bash
export RLCD_SMART_MODEL="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"
export RLCD_DEFAULT_MODEL="$RLCD_SMART_MODEL"
```

Start the service:

```bash
./run.sh
```

Verify it:

```bash
curl http://127.0.0.1:8001/health
```

Expected response:

```json
{"status":"ok","service":"rlcd-router"}
```

Models are loaded lazily on the first classification request.

## Quick start

Route a task:

```bash
curl -s http://127.0.0.1:8001/api/route \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Fix the authentication race, add a regression test, and run the test suite.",
    "runtime": {
      "repo_root": ".",
      "prior_failures": 0,
      "deadline": "normal"
    }
  }' | python3 -m json.tool
```

The response includes:

```json
{
  "route_id": "...",
  "classification": {
    "fields": {
      "task_type": "debugging",
      "coding": "dominant",
      "repo_access": "modify",
      "tool_use": "shell"
    }
  },
  "decision": {
    "tier": "local_coding",
    "eligible_models": ["local_coding"],
    "reasons": ["needs_coding", "needs_shell", "selected:local_coding"]
  }
}
```

The host agent resolves each profile name through `GET /api/route/config` and tries the returned profiles in order.

## Explicit model override

Pass a profile key in `model_override` to bypass classification for a specific task:

```bash
curl -s http://127.0.0.1:8001/api/route \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Review this migration plan.",
    "model_override": "gpt_5_6_terra"
  }' | python3 -m json.tool
```

The decision will have:

- `classifier_used: "user_model_override"`
- confidence `1.0`
- one entry in `eligible_models`
- reason `user_requested_model`

Invalid profile names return HTTP 400 and list the available profile keys.

List valid override targets:

```bash
curl -s http://127.0.0.1:8001/api/models/overrides | python3 -m json.tool
```

## Pi extension

The optional extension is in [`pi-extension/rlcd-router.ts`](pi-extension/rlcd-router.ts).

It provides:

- router startup on `session_start`;
- health checking so multiple Pi sessions share one server;
- the `rlcd_classify` constrained-classification tool;
- `/route <task>` to inspect a routing decision;
- `/model-override` to select a one-shot override for the next `/route` inspection;
- `/model-override clear` to return to automatic selection.

### Install using a symlink

A symlink lets the extension derive the router directory from its own location:

```bash
mkdir -p ~/.pi/agent/extensions
ln -s "$(pwd)/pi-extension/rlcd-router.ts" \
  ~/.pi/agent/extensions/rlcd-router.ts
```

Then restart Pi or run `/reload`.

### Install by copying

If you copy the extension instead of symlinking it, set the project location explicitly:

```bash
export RLCD_ROUTER_DIR="/path/to/rlcd-router"
```

Optional settings:

```bash
export RLCD_ROUTER_HOST="127.0.0.1"
export RLCD_ROUTER_PORT=8001
export RLCD_ROUTER_URL="http://127.0.0.1:8001"
export RLCD_PYTHON="$(pwd)/.venv/bin/python"
```

For a browser client on a different local origin, set a comma-separated allowlist:

```bash
export ROUTER_CORS_ORIGINS="http://127.0.0.1:3000,http://localhost:3000"
```

> The included extension starts the service and exposes routing commands/tools. Execution remains the host agent's responsibility: it should resolve the selected profile to a configured provider/model and try `eligible_models` in order.

## Configuration

Routing configuration lives in [`router/config.py`](router/config.py).

### Model profiles

A profile describes:

```python
ModelProfile(
    model_id="provider-model-id",
    role="executor",
    provider="provider-name",
    capabilities=("general", "coding", "reasoning", "tools"),
    latency_class="medium",
    billing="subscription_quota",
)
```

Supported billing categories are:

1. `local`
2. `subscription_unlimited`
3. `subscription_quota`
4. `paid`

Profile keys are stable routing identifiers. `model_id` and `provider` should match the host agent's model registry.

### Tiers

A tier specifies minimum capability and an ordered profile list:

```python
TierDefinition(
    name="cloud_coding",
    is_local=False,
    minimum_capability="coding",
    preferred_model_profiles=(
        "subscription_coding_model",
        "paid_coding_fallback",
    ),
    supports_tools=("shell", "read_only", "browser"),
    supports_web=True,
)
```

Edit the example profiles and tier ordering to match the models available in your environment.

### Confidence thresholds

These environment variables tune classifier acceptance:

| Variable | Default | Meaning |
|---|---:|---|
| `ROUTER_FAST_ACCEPT` | `0.85` | Overall confidence required for fast acceptance |
| `ROUTER_CRITICAL_FLOOR` | `0.60` | Minimum confidence for critical fields |
| `ROUTER_CONSERVATIVE_FLOOR` | `0.50` | Conservative confidence floor |
| `ROUTER_SMART_BYPASS` | `0.50` | Smart-classifier bypass floor |
| `ROUTER_ROUTE_MARGIN` | `0.15` | Required margin between route candidates |

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Identify the service and confirm readiness |
| `POST` | `/api/run-parallel` | Run arbitrary constrained RLCD classification |
| `POST` | `/api/route` | Classify a task and select a tier |
| `POST` | `/api/route-and-execute` | Route and execute local tiers; cloud execution is delegated to the host |
| `POST` | `/api/execute` | Execute directly with a local model profile |
| `POST` | `/api/route/outcome` | Record success, failure, latency, and cost |
| `GET` | `/api/route/config` | Return profiles, tiers, thresholds, and schema fields |
| `GET` | `/api/route/telemetry` | Return aggregate routing metrics |
| `GET` | `/api/models` | Return loaded models and configured profiles |
| `GET` | `/api/models/overrides` | Return valid override targets grouped by billing |

Interactive API documentation is available at `http://127.0.0.1:8001/docs`.

## Project layout

```text
core/
  engine.py          Public API for the bundled MLX backend
  engine_mlx.py      MLX RLCD and local text generation
  schema.py          Structured schema compilation and calibration
router/
  router_schema.py   Task-requirement schema
  classifier.py      Confidence and abstention handling
  signals.py         Deterministic runtime signals
  policy.py          Pure tier-selection policy
  config.py          Model profiles, tiers, and thresholds
  decision.py        Route decision structures
  escalation.py      Failure-driven escalation
  telemetry.py       JSONL events and aggregate metrics
server/
  app.py             FastAPI routing service
pi-extension/
  rlcd-router.ts     Optional Pi integration
presets/
  pi_router.json     Example labeled classification case
tests/
  ...                Unit and API tests
```

## Testing

```bash
pytest -q
```

The tests mock model execution, so the routing suite does not load a large model. The complete suite currently imports the bundled MLX backend and therefore requires `mlx` and `mlx-lm`; the pure `router/` policy modules can be reused independently on other platforms.

## Security and privacy

- The service binds to `127.0.0.1` by default.
- Do not bind it publicly without authentication and strict CORS settings.
- Prompt text is not written to telemetry; only a hash and aggregate metadata are recorded.
- Telemetry defaults to `data/router_events.jsonl`. Override it with `ROUTER_TELEMETRY_PATH`.
- `local_only` is enforced as a policy constraint, not merely a preference.
- Keep API keys in provider configuration or environment variables; this repository does not require cloud credentials.

## Current limitations

- The included inference backend requires MLX and therefore Apple Silicon.
- Hybrid/linear-attention models may not support the batched cache shape used by parallel constrained decoding.
- Cloud execution is intentionally delegated to the host agent.
- Model availability and subscription limits are host-specific; update `router/config.py` accordingly.
- The default model profiles are examples and may refer to models not available from every provider.

## Attribution

This router builds on the RLCD model and engine work in the original
[Qwen-2.5-1B-RLCD project by harshatheg](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD).

## License

Add the license appropriate for your project before publishing.
