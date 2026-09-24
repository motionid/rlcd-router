# Router Bundle Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the bundle's validated catalog and automatic Pi enforcement fixes while preserving a clear rollback path and fail-open behavior.

**Architecture:** Keep the existing FastAPI router contracts and update the model catalog/configuration in place. Extend the Pi adapter at its lifecycle boundaries: resolve an eligible executor before each turn, then report outcomes after settlement. Validate the Python API directly and statically verify the TypeScript adapter because no TypeScript harness exists in the repository.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic, pytest, TypeScript Pi extension, Git bundle/history.

**Spec:** `docs/superpowers/specs/2026-09-24-router-bundle-integration-design.md`

## Global Constraints

- Keep `OMLX_MODELS_DIR` configurable and avoid hard-coding a user-specific path beyond the existing default convention.
- Treat router model IDs and Pi registry IDs as separate representations; local oMLX profiles may use a full filesystem path while Pi resolves the basename.
- Only executor profiles may be selected as user overrides or activated for a turn.
- Do not reintroduce `subscription_unlimited`; the catalog uses `local`, `subscription_quota`, and `paid`.
- Preserve classifier-only profiles as non-executable profiles.
- Keep all routing failures fail-open and never block a Pi turn.
- Do not install MLX or alter host-level Pi credentials/model configuration.

## Review Focus

- **Rollback safety:** integration must have a named pre-change backup ref and separately committed changes; verify with a clean-tree rollback rehearsal using a temporary worktree.
- **Machine-specific local paths:** changing `OMLX_MODELS_DIR` or profile IDs must preserve environment overrides and produce stable serialized config.
- **Classifier-only profiles:** overrides and Pi activation must reject/skip `rlcd_fast` and `rlcd_smart`.
- **Unavailable Pi models:** a missing registry model must fall through to the next eligible executor or configured next tier, not block the turn.
- **Quota exhaustion:** only a reported 429/subscription exhaustion should schedule the server-recommended one-shot fallback; aborted turns must not escalate.

---

### Task 1: Create the rollback checkpoint and inspect the exact integration delta

**Files:**
- No product files created or modified.
- Git ref created from the `backup_ref` shell variable.

**Interfaces:**
- Consumes: current `HEAD`, clean working tree, `rlcd-router.bundle`.
- Produces: a named backup ref and an auditable diff range for Tasks 2–4.

- [ ] **Step 1: Confirm the working tree contains only the expected bundle/spec artifacts**

Run:

```bash
git status --short
git diff --check
git rev-parse HEAD
```

Expected: no product-code modifications and no whitespace errors.

- [ ] **Step 2: Create the rollback branch before touching product code**

Run:

```bash
backup_ref="backup/router-pre-bundle-$(date +%Y%m%d-%H%M%S)"
git branch "$backup_ref" HEAD
printf '%s\n' "$backup_ref" > .router-integration-backup-ref
```

The backup ref must point to the current pre-integration commit. The ref file is local operational metadata and must not be committed if the repository policy excludes it; otherwise add it to the integration documentation only.

- [ ] **Step 3: Verify the bundle remains independently readable**

Run:

```bash
git bundle verify ./rlcd-router.bundle
git bundle list-heads ./rlcd-router.bundle
```

Expected: bundle is valid and advertises `2389afa... refs/heads/main`.

- [ ] **Step 4: Record the source diff without applying it**

Run:

```bash
tmp_repo=$(mktemp -d /tmp/rlcd-router-bundle-XXXXXX)
git clone --quiet ./rlcd-router.bundle "$tmp_repo"
cd "$tmp_repo"
git diff --stat 85e3481..2389afa
git diff --name-status 85e3481..2389afa
```

Expected: only `router/config.py`, `server/app.py`, and `pi-extension/rlcd-router.ts` are changed by the bundle delta.

- [ ] **Step 5: Commit the rollback metadata only if it is repository-approved**

If `.router-integration-backup-ref` is not ignored by repository policy, remove it before product commits and retain the backup branch name in the implementation commit message. Do not commit generated temporary paths or the bundle itself.

---

### Task 2: Integrate and test the model catalog/API configuration

**Files:**
- Modify: `router/config.py`
- Modify: `server/app.py`
- Test: `tests/test_router_api.py`
- Test: `tests/test_subscription_fallback.py`

**Interfaces:**
- Consumes: existing `ModelProfile`, `TIER_DEFINITIONS`, `/api/route/config`, and override/fallback APIs.
- Produces: configurable oMLX model paths, the corrected billing categories/catalog, and JSON shapes consumed by the Pi extension.

- [ ] **Step 1: Add failing assertions for catalog invariants**

Extend the existing `TestRouterAPI` methods, which use `self.client`, with these exact assertions:

```python
def test_route_config_exposes_catalog_invariants(self):
    import router.config as config
    assert config.MODEL_PROFILES["local_coding"].role == "executor"
    assert config.MODEL_PROFILES["rlcd_smart"].role == "classifier"
    assert config.MODEL_PROFILES["local_coding"].billing == "local"


def test_override_catalog_uses_supported_billing_order(self):
    response = self.client.get("/api/models/overrides")
    assert response.status_code == 200
    body = response.json()
    profile_names = {
        model["profile_name"]
        for models in body["models"].values()
        for model in models
    }
    assert "rlcd_fast" not in profile_names
    assert "rlcd_smart" not in profile_names
    assert body["billing_order"] == ["local", "subscription_quota", "paid"]
```

Place these methods alongside the existing override/catalog tests.

- [ ] **Step 2: Run the focused tests and confirm the new expectations expose the old catalog**

Run:

```bash
pytest -q tests/test_router_api.py -k 'override_catalog or route_config' tests/test_subscription_fallback.py
```

Expected: the new catalog-specific checks fail before the bundle catalog changes are integrated, while unrelated existing tests may pass.

- [ ] **Step 3: Integrate the bundle catalog changes**

Apply the `19d5a09` changes to `router/config.py` and `server/app.py`, preserving the existing API and environment variable overrides. Confirm that:

- `OMLX_MODELS_DIR` is read from the environment with the existing default convention.
- local executor/classifier IDs use that directory where required;
- cloud profiles use the `github-copilot` provider and `subscription_quota` billing;
- classifier profiles remain non-executable;
- `/api/models/overrides` reports only executor profiles and the supported billing order.

- [ ] **Step 4: Run focused tests and inspect API payloads**

Run:

```bash
pytest -q tests/test_router_api.py -k 'override_catalog or route_config' tests/test_subscription_fallback.py
python - <<'PY'
from router.config import MODEL_PROFILES, TIER_DEFINITIONS
assert MODEL_PROFILES["rlcd_smart"].role == "classifier"
assert "subscription_unlimited" not in {p.billing for p in MODEL_PROFILES.values()}
assert MODEL_PROFILES["rlcd_fast"].role == "classifier"
assert MODEL_PROFILES["rlcd_smart"].role == "classifier"
assert "local_coding" in TIER_DEFINITIONS
print("catalog invariants ok")
PY
```

Expected: focused tests pass; the invariant script prints `catalog invariants ok`.

- [ ] **Step 5: Commit the catalog/API integration**

```bash
git add router/config.py server/app.py tests/test_router_api.py tests/test_subscription_fallback.py
git commit -m "Integrate router model catalog fixes"
```

---

### Task 3: Integrate automatic Pi enforcement and outcome reporting

**Files:**
- Modify: `pi-extension/rlcd-router.ts`
- Create: `tests/test_pi_extension_structure.py`

**Interfaces:**
- Consumes: `/api/route`, `/api/route/config`, `/api/route/outcome`, `decision.eligible_models`, `decision.escalation_config.next_tier`, and Pi `modelRegistry`/`setModel` APIs.
- Produces: executor profile resolution, automatic model switching, failure escalation state, quota fallback state, and fail-open lifecycle handling.

- [ ] **Step 1: Add structural tests for required extension behavior**

Create a Python source-structure test that reads `pi-extension/rlcd-router.ts` and asserts the integration contains these exact markers:

```python
from pathlib import Path

SOURCE = Path("pi-extension/rlcd-router.ts").read_text()


def test_extension_enforces_and_reports_routes():
    for marker in (
        'pi.on("before_agent_start"',
        'pi.setModel(',
        'pi.on("after_provider_response"',
        'pi.on("agent_before_settle"',
        'POST',
        "/api/route/outcome",
        'failure_reason === "subscription_exhausted"',
        "priorFailures",
        "Fail open",
    ):
        assert marker in SOURCE


def test_extension_skips_non_executors_and_resolves_local_basename():
    assert 'profile.role !== "executor"' in SOURCE
    assert 'provider === "omlx"' in SOURCE
    assert "parts[parts.length - 1]" in SOURCE
```

- [ ] **Step 2: Run the structural tests before the extension change and confirm they fail**

Run:

```bash
pytest -q tests/test_pi_extension_structure.py
```

Expected: FAIL because the current extension does not contain automatic enforcement/outcome hooks.

- [ ] **Step 3: Integrate the bundle extension changes**

Apply `2389afa` to `pi-extension/rlcd-router.ts`. Preserve the existing slash commands and ensure:

- `before_agent_start` ignores empty prompts and slash commands;
- route/config/model failures only set warnings and return;
- profile resolution skips classifier-only and unavailable models;
- one configured `next_tier` fallback is attempted;
- `modelOverride` is consumed once by automatic routing or `/route` consistently;
- successful turns reset `priorFailures`, failed turns increment it;
- aborted turns do not report failure;
- only HTTP 429 is converted into `subscription_exhausted`;
- outcome fallback sets a one-shot override rather than permanently changing user preference.

- [ ] **Step 4: Run structural and syntax checks available in the environment**

Run:

```bash
pytest -q tests/test_pi_extension_structure.py
if command -v npx >/dev/null 2>&1; then
  npx --yes prettier --check pi-extension/rlcd-router.ts
else
  printf '%s\n' 'No npx available; source-structure tests are the TypeScript verification.'
fi
```

Expected: structural tests pass. Formatting is either validated or explicitly reported unavailable.

- [ ] **Step 5: Commit the extension integration**

```bash
git add pi-extension/rlcd-router.ts tests/test_pi_extension_structure.py
git commit -m "Enforce router decisions in Pi extension"
```

---

### Task 4: Run full verification and rehearse rollback

**Files:**
- Modify: `docs/superpowers/specs/2026-09-24-router-bundle-integration-design.md` only if verification discovers a design correction.
- No product-code changes expected.

**Interfaces:**
- Consumes: the backup ref from Task 1 and the separate commits from Tasks 2–3.
- Produces: test evidence, final diff review, and a documented rollback command.

- [ ] **Step 1: Run the full Python suite**

Run:

```bash
pytest -q
```

Expected: all non-MLX tests pass. If MLX imports are unavailable, record those failures as environment-blocked with the exact missing dependency; do not claim a clean full suite.

- [ ] **Step 2: Verify the server/extension response contract**

Run:

```bash
python - <<'PY'
from router.config import MODEL_PROFILES, TIER_DEFINITIONS
for tier_name, tier in TIER_DEFINITIONS.items():
    assert tier_name
    assert tier.preferred_model_profiles
    for profile_name in tier.preferred_model_profiles:
        assert profile_name in MODEL_PROFILES
print("tier/profile contract ok")
PY
```

Expected: prints `tier/profile contract ok`.

- [ ] **Step 3: Review the final diff and commit graph**

Run:

```bash
backup_ref=$(git for-each-ref --format='%(refname:short)' refs/heads/backup/router-pre-bundle-* | tail -1)
git diff --check "$backup_ref"..HEAD
git log --oneline --decorate --graph -6
git status --short
```

Expected: only intended integration commits and documentation are present; the original bundle remains uncommitted unless explicitly requested.

- [ ] **Step 4: Rehearse rollback in a temporary worktree**

Run:

```bash
backup_ref=$(git for-each-ref --format='%(refname:short)' refs/heads/backup/router-pre-bundle-* | tail -1)
rollback_dir=$(mktemp -d /tmp/rlcd-router-rollback-XXXXXX)
git worktree add "$rollback_dir" "$backup_ref"
test "$(git -C "$rollback_dir" rev-parse HEAD)" = "$(git rev-parse "$backup_ref")"
git worktree remove "$rollback_dir"
```

Expected: temporary worktree checks out the exact pre-integration commit, proving the backup ref is usable without altering the active worktree.

- [ ] **Step 5: Document operational rollback commands in the final report**

Use these commands if the integration must be undone while preserving history:

```bash
# Preferred: use the actual hashes printed by `git log`, newest first.
git revert ACTUAL_EXTENSION_COMMIT ACTUAL_CATALOG_COMMIT

# Emergency local rollback before publishing the branch.
git reset --hard "$backup_ref"
```

The final report must identify the actual backup ref and actual commit hashes. Never delete the backup ref until the integrated extension has been validated in the target Pi environment.
