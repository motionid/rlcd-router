// Pure state-machine tests for pi-extension/router-state.ts.
//
// The module under test imports no PI runtime; Node 25's native
// TypeScript support (type stripping) executes it directly.
//
// Run: node --test tests/test_pi_extension_state.mjs

import test from "node:test";
import assert from "node:assert/strict";

import {
  ROUTER_STATE_ENTRY,
  ROUTER_STATE_VERSION,
  DEFAULT_MAX_ATTEMPTS,
  advanceCandidate,
  beginRoute,
  clearOverride,
  classifyProviderFailure,
  createRouterState,
  flattenOverrideCatalog,
  outageDecision,
  resolveOverrideProfile,
  restoreFromBranch,
  settleAttempt,
  withAutomaticSelection,
  withModelSelection,
} from "../pi-extension/router-state.ts";

function routedState(candidates, activeIndex = 0, maxAttempts = DEFAULT_MAX_ATTEMPTS) {
  const base = createRouterState();
  return beginRoute(base, {
    routeId: "r1",
    candidates,
    activeIndex,
    prompt: "fix the failing test",
    capability: "coding",
    reasons: ["code_signal"],
    maxAttempts,
  });
}

test("createRouterState produces a fresh versioned state", () => {
  const state = createRouterState();
  assert.equal(state.version, ROUTER_STATE_VERSION);
  assert.equal(state.overrideModel, null);
  assert.equal(state.route, null);
  assert.equal(state.taskFailures, 0);
  assert.equal(state.activeModel, "");
});

test("user model selection creates an override (set and cycle)", () => {
  const base = createRouterState();
  const set = withModelSelection(base, "github-copilot/claude-sonnet-5", "set");
  assert.equal(set.overrideModel, "github-copilot/claude-sonnet-5");
  assert.equal(set.activeModel, "github-copilot/claude-sonnet-5");
  const cycle = withModelSelection(base, "omlx/qwen-coder", "cycle");
  assert.equal(cycle.overrideModel, "omlx/qwen-coder");
});

test("session restore does not create an override", () => {
  const base = createRouterState();
  const restored = withModelSelection(base, "omlx/qwen-coder", "restore");
  assert.equal(restored.overrideModel, null);
  assert.equal(restored.activeModel, "omlx/qwen-coder");
  // A pre-existing persisted override survives a restore untouched.
  const overridden = withModelSelection(base, "a/b", "set");
  const reRestored = withModelSelection(overridden, "c/d", "restore");
  assert.equal(reRestored.overrideModel, "a/b");
});

test("automatic setModel event does not create an override", () => {
  const base = createRouterState();
  const automatic = withAutomaticSelection(base, "github-copilot/gpt-6-luna");
  assert.equal(automatic.overrideModel, null);
  assert.equal(automatic.activeModel, "github-copilot/gpt-6-luna");
  // An existing manual override is never consumed by an automatic switch.
  const overridden = withModelSelection(base, "a/b", "set");
  const stillOverridden = withAutomaticSelection(overridden, "c/d");
  assert.equal(stillOverridden.overrideModel, "a/b");
});

test("router-auto clears the override", () => {
  const overridden = withModelSelection(createRouterState(), "a/b", "set");
  const cleared = clearOverride(overridden);
  assert.equal(cleared.overrideModel, null);
  // Idempotent: clearing again is a no-op.
  assert.equal(clearOverride(cleared).overrideModel, null);
});

test("candidate activation failure advances the cursor without reclassification", () => {
  const state = routedState(["alpha", "beta", "gamma"], 0);
  assert.equal(state.route.cursor, 0);
  const first = advanceCandidate(state);
  assert.equal(first.nextProfileId, "beta");
  assert.equal(first.state.route.cursor, 1);
  assert.equal(first.state.route.candidates, state.route.candidates, "candidates are never reclassified");
  const second = advanceCandidate(first.state);
  assert.equal(second.nextProfileId, "gamma");
  assert.equal(second.state.route.cursor, 2);
  const exhausted = advanceCandidate(second.state);
  assert.equal(exhausted.nextProfileId, null, "past the last candidate there is no next profile");
  assert.equal(exhausted.state.route.cursor, 3);
});

test("provider 429 with Retry-After reports rate limit and advances the cursor", () => {
  const failure = classifyProviderFailure(429, { "Retry-After": "30" });
  assert.deepEqual(failure, { category: "rate_limit", retryAfter: 30 });

  // Header lookup is case-insensitive; non-numeric values are ignored.
  assert.equal(classifyProviderFailure(429, { "retry-after": "45" }).retryAfter, 45);
  assert.equal(classifyProviderFailure(429, { "retry-after": "soon" }).retryAfter, null);

  const state = routedState(["alpha", "beta", "gamma"], 0);
  const settled = settleAttempt(state, {
    outcome: "error",
    providerStatus: 429,
    providerHeaders: { "Retry-After": "30" },
  });
  assert.equal(settled.penalizeProvider, true);
  assert.deepEqual(settled.report, {
    status: "failed",
    failureCategory: "rate_limit",
    retryAfter: 30,
  });
  assert.equal(settled.nextProfileId, "beta");
  assert.equal(settled.state.route.cursor, 1);
  assert.equal(settled.continueTurn, true);
});

test("task failure reports task failure and never penalizes the provider", () => {
  const state = routedState(["alpha", "beta"], 0);
  const settled = settleAttempt(state, {
    outcome: "error",
    providerStatus: 200,
    providerHeaders: {},
  });
  assert.deepEqual(settled.report, {
    status: "failed",
    failureCategory: "task_failure",
    retryAfter: null,
  });
  assert.equal(settled.penalizeProvider, false);
  assert.equal(settled.state.taskFailures, 1, "task failures feed the next-turn escalation counter");
  assert.equal(settled.nextProfileId, null, "a task failure does not advance candidates");
  assert.equal(settled.continueTurn, false);
  assert.equal(settled.state.route, null);
});

test("aborted turn does not penalize the model", () => {
  const state = routedState(["alpha", "beta"], 0);
  const settled = settleAttempt(state, {
    outcome: "aborted",
    providerStatus: 429,
    providerHeaders: { "Retry-After": "30" },
  });
  assert.equal(settled.report, null, "no outcome is reported for an aborted turn");
  assert.equal(settled.penalizeProvider, false);
  assert.equal(settled.continueTurn, false);
  assert.equal(settled.nextProfileId, null);
  assert.equal(settled.state.taskFailures, 0);
  assert.equal(settled.state.route, null);
});

test("successful turn reports success and clears the escalation counter", () => {
  const state = routedState(["alpha"], 0);
  const settled = settleAttempt(state, {
    outcome: "completed",
    providerStatus: 200,
    providerHeaders: {},
  });
  assert.deepEqual(settled.report, {
    status: "success",
    failureCategory: null,
    retryAfter: null,
  });
  assert.equal(settled.state.taskFailures, 0);
  assert.equal(settled.state.route, null);
  assert.equal(settled.continueTurn, false);
});

test("candidate exhaustion stops continuation", () => {
  // Single candidate: nothing left to try.
  const single = settleAttempt(routedState(["alpha"], 0), {
    outcome: "error",
    providerStatus: 429,
    providerHeaders: {},
  });
  assert.equal(single.nextProfileId, null);
  assert.equal(single.continueTurn, false);

  // Attempt limit reached: a next candidate exists but may not be used.
  const limited = settleAttempt(routedState(["alpha", "beta"], 0, 2), {
    outcome: "error",
    providerStatus: 429,
    providerHeaders: {},
  });
  assert.equal(limited.nextProfileId, "beta");
  assert.equal(limited.state.route.attempts, 2);
  assert.equal(limited.continueTurn, false, "maxAttempts bounds the fallback loop");
});

test("router outage retains the active model or a configured subscription fallback", () => {
  const state = createRouterState({ activeModel: "omlx/qwen-coder" });
  const retained = outageDecision(state, null);
  assert.deepEqual(retained, { model: "omlx/qwen-coder", retainedActive: true });

  const fallback = outageDecision(state, "github-copilot/gpt-6-luna");
  assert.deepEqual(fallback, { model: "github-copilot/gpt-6-luna", retainedActive: false });
});

test("branch state reconstructs from the latest compatible active-branch entry", () => {
  const entries = [
    { type: "user", data: "noise" },
    { type: "custom", customType: "other-extension", data: { version: 1 } },
    {
      type: "custom",
      customType: ROUTER_STATE_ENTRY,
      data: {
        version: ROUTER_STATE_VERSION,
        overrideModel: "a/old",
        activeModel: "a/old",
        taskDepth: 3,
      },
    },
    { type: "assistant", data: "noise" },
    {
      type: "custom",
      customType: ROUTER_STATE_ENTRY,
      data: {
        version: ROUTER_STATE_VERSION,
        overrideModel: "a/new",
        activeModel: "a/new",
        taskFailures: 2,
        previousPrompt: "x".repeat(100000),
      },
    },
  ];
  const restored = restoreFromBranch(entries);
  assert.ok(restored);
  assert.equal(restored.overrideModel, "a/new", "the latest entry wins");
  assert.equal(restored.taskFailures, 2);
  assert.ok(restored.previousPrompt.length <= 4096, "restored previous prompt is bounded");
  assert.equal(restored.route, null, "an ephemeral server route is never restored");

  // Incompatible versions are skipped, not restored.
  const incompatible = restoreFromBranch([
    { type: "custom", customType: ROUTER_STATE_ENTRY, data: { version: 99, overrideModel: "a/future" } },
  ]);
  assert.equal(incompatible, null);

  // No router entry at all: fresh state must be created by the caller.
  assert.equal(restoreFromBranch([{ type: "user" }]), null);
  assert.equal(restoreFromBranch([]), null);
});

test("beginRoute bounds the persisted state", () => {
  const state = beginRoute(createRouterState(), {
    routeId: "r".repeat(500),
    candidates: ["a", "b", "c", "", null],
    activeIndex: 1,
    prompt: "p".repeat(100000),
    capability: "reasoning",
    reasons: ["r1", "r2", ...Array(50).fill("x")],
    maxAttempts: 3,
  });
  assert.ok(state.route.routeId.length <= 64);
  assert.deepEqual(state.route.candidates, ["a", "b", "c"], "empty candidate ids are dropped");
  assert.equal(state.route.cursor, 1);
  assert.equal(state.route.attempts, 1);
  assert.equal(state.route.maxAttempts, 3);
  assert.ok(state.previousPrompt.length <= 4096);
  assert.ok(state.previousRouteReason.length <= 16);
  assert.equal(state.taskDepth, 1);
});

// --- round 2: known-override mapping and persistence -------------------------

test("override catalog flattens the server's billing-grouped response", () => {
  const catalog = flattenOverrideCatalog({
    billing_order: ["local", "subscription_quota", "paid"],
    models: {
      subscription_quota: [
        {
          profile_name: "gpt_6_luna",
          model_id: "gpt-6-luna",
          provider: "github-copilot",
          capabilities: ["general"],
          billing: "subscription_quota",
          notes: "",
        },
        { profile_name: "claude_sonnet_5", model_id: "claude-sonnet-5", provider: "github-copilot" },
      ],
      local: [{ profile_name: "local_coding", model_id: "/omlx/models/qwen3-coder", provider: "omlx" }],
      paid: "not-an-array", // malformed billing group: skipped, not fatal
    },
  });
  assert.equal(catalog.length, 3);
  assert.ok(catalog.some((e) => e.profile_name === "gpt_6_luna" && e.provider === "github-copilot"));
  assert.ok(catalog.some((e) => e.profile_name === "local_coding"));
  // Unusable shapes are null, never guessed from.
  assert.equal(flattenOverrideCatalog({ models: null }), null);
  assert.equal(flattenOverrideCatalog("junk"), null);
  assert.equal(flattenOverrideCatalog(null), null);
});

test("known manual model maps to its router profile by exact provider/model match", () => {
  const entries = [
    { profile_name: "local_coding", model_id: "/omlx/models/qwen3-coder", provider: "omlx" },
    { profile_name: "gpt_6_luna", model_id: "gpt-6-luna", provider: "github-copilot" },
  ];
  assert.equal(resolveOverrideProfile("github-copilot/gpt-6-luna", entries), "gpt_6_luna");
  // No exact match: never guess (fuzzy, prefix, basename, wrong provider).
  assert.equal(resolveOverrideProfile("github-copilot/gpt-6", entries), null);
  assert.equal(resolveOverrideProfile("openai/gpt-6-luna", entries), null);
  assert.equal(resolveOverrideProfile("omlx/qwen3-coder", entries), null);
  assert.equal(resolveOverrideProfile("omlx//omlx/models/qwen3-coder", entries), "local_coding");
  // Unknown or malformed inputs preserve the override untouched.
  assert.equal(resolveOverrideProfile("no-slash", entries), null);
  assert.equal(resolveOverrideProfile("github-copilot/", entries), null);
  assert.equal(resolveOverrideProfile("github-copilot/gpt-6-luna", null), null);
  assert.equal(resolveOverrideProfile("github-copilot/gpt-6-luna", undefined), null);
});

test("manual override persists through automatic fallback and bounded continuation", () => {
  const base = withModelSelection(createRouterState(), "github-copilot/gpt-6-luna", "set");
  // Unavailable override: the server routes from the normal pool; the
  // persisted manual preference survives so it can be retried after cooldown.
  const routed = beginRoute(base, {
    routeId: "r1",
    candidates: ["claude_sonnet_5", "gpt_5_3_codex"],
    activeIndex: 0,
    prompt: "task",
    capability: "strong",
    reasons: ["strong_signal", "override_unavailable:gpt_6_luna"],
  });
  assert.equal(routed.overrideModel, "github-copilot/gpt-6-luna");
  // Automatic candidate activation never replaces the override.
  const afterAuto = withAutomaticSelection(routed, "github-copilot/claude-sonnet-5");
  assert.equal(afterAuto.overrideModel, "github-copilot/gpt-6-luna");
  // Provider failure on the fallback advances within the bound, override intact.
  const settled = settleAttempt(afterAuto, { outcome: "error", providerStatus: 429, providerHeaders: {} });
  assert.equal(settled.continueTurn, true);
  assert.equal(settled.nextProfileId, "gpt_5_3_codex");
  assert.equal(settled.state.overrideModel, "github-copilot/gpt-6-luna");
  // Only /router-auto clears it.
  assert.equal(clearOverride(settled.state).overrideModel, null);
});

test("available-override route failure has no fallback candidates", () => {
  const routed = beginRoute(withModelSelection(createRouterState(), "a/b", "set"), {
    routeId: "r2",
    candidates: ["gpt_6_luna"],
    activeIndex: 0,
    prompt: "p",
    capability: "normal",
    reasons: ["user_requested_model", "override:gpt_6_luna"],
  });
  const settled = settleAttempt(routed, { outcome: "error", providerStatus: 429, providerHeaders: {} });
  assert.equal(settled.nextProfileId, null);
  assert.equal(settled.continueTurn, false);
  assert.equal(settled.state.overrideModel, "a/b");
});
