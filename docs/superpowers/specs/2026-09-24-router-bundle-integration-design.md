# Router Bundle Integration Design

## Goal

Integrate the fixes contained in `rlcd-router.bundle` into the current RLCD router without blindly replacing the repository. Preserve the bundle's useful routing behavior, validate its machine-specific model catalog assumptions, and keep failures fail-open for Pi users.

## Current state and source history

The bundle is a complete Git bundle whose `main` ref is `2389afa`. The current repository `HEAD` (`85e3481`) is an ancestor, so the source history is compatible.

The bundle adds two functional changes:

- `19d5a09`: updates the model catalog to the observed GitHub Copilot/oMLX inventory, uses configurable oMLX model paths, and removes the obsolete `subscription_unlimited` billing category.
- `2389afa`: makes the Pi extension enforce routing with `pi.setModel()` before a turn, reports outcomes, escalates consecutive failures, and automatically prepares a fallback after subscription exhaustion.

## Integration approach

Use the bundle commits as the implementation source, but integrate and validate the affected boundaries rather than accepting a blind fast-forward:

1. Apply the catalog/configuration changes in `router/config.py` and `server/app.py`.
2. Apply the automatic enforcement and outcome-reporting changes in `pi-extension/rlcd-router.ts`.
3. Preserve existing public route, fallback, telemetry, and override contracts.
4. Add or adjust focused tests for catalog serialization, executor-only overrides, route outcomes, and fallback behavior where coverage is missing or stale.
5. Add a lightweight static/structural verification for the TypeScript extension because this repository has no TypeScript package or Pi extension test harness.
6. Keep all routing failures fail-open: an unavailable router, unavailable model, or failed model switch must not block a Pi turn.

## Catalog compatibility rules

- Keep `OMLX_MODELS_DIR` configurable and avoid hard-coding a user-specific path beyond the existing default convention.
- Treat router model IDs and Pi registry IDs as separate representations; local oMLX profiles may use a full filesystem path while Pi resolves the basename.
- Only executor profiles may be selected as user overrides or activated for a turn.
- Do not reintroduce `subscription_unlimited`; the catalog uses `local`, `subscription_quota`, and `paid`.
- Preserve the classifier-only profiles as non-executable profiles.

## Extension behavior

For each non-command turn:

1. Send the prompt and prior failure count to `/api/route`.
2. Fetch and cache `/api/route/config`.
3. Resolve the decision's ordered eligible profiles against Pi's model registry, skipping classifier-only or unavailable profiles.
4. If the selected tier has no usable executor, try its configured `next_tier` once.
5. Call `pi.setModel()` only when the selected model differs from the active model.
6. If any router/config/model operation fails, show a status warning and allow the existing model/turn to continue.

After a turn:

1. Capture provider status when available.
2. Report success/failure to `/api/route/outcome`.
3. Track consecutive failures for the next route request.
4. On HTTP 429/subscription exhaustion, set a one-shot override to the server-recommended fallback and notify the user.
5. Do not treat an explicitly aborted/user-cancelled turn as a model failure.

## Testing and verification

- Run the complete Python test suite.
- If MLX is unavailable, record those tests as environment-blocked and still run all independent tests.
- Check that route config and outcome response shapes match what the extension reads.
- Perform static TypeScript checks available in the environment, or use targeted source assertions if no compiler/runtime is installed.
- Inspect the final diff and confirm only intended files and tests changed.

## Out of scope

- Replacing the router architecture or classifier backend.
- Installing MLX or changing host-level Pi model/auth configuration.
- Guaranteeing that every catalog model exists in every user's live Copilot account; the extension must continue to filter through Pi's registry.
