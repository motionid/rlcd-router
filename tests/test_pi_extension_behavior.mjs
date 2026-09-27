// Behavioral tests for pi-extension/turn-router.ts.
//
// The extension is loaded through jiti (the same loader Pi uses for
// TypeScript extensions) and driven through a stubbed ExtensionAPI and a
// stubbed fetch layer.  No network and no live server: the stub responses
// mirror the Task 7/8 server contracts exactly (/api/route with
// model_override, /api/route/outcome, /api/models/overrides).
//
// Run: node --test tests/test_pi_extension_behavior.mjs

import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const PI_PACKAGE_JSON =
  process.env.PI_PACKAGE_JSON ??
  "/opt/homebrew/lib/node_modules/@earendil-works/pi-coding-agent/package.json";

// jiti resolves relative to cwd; pin the repo root from this file's location.
const REPO_ROOT = fileURLToPath(new URL("..", import.meta.url));
const EXTENSION = `${REPO_ROOT}pi-extension/turn-router.ts`;

let loadExtension;
try {
  const require = createRequire(PI_PACKAGE_JSON);
  const jiti = require("jiti");
  const jitiLoad = jiti(REPO_ROOT, { interopDefault: true });
  loadExtension = () => {
    const mod = jitiLoad(EXTENSION);
    return typeof mod === "function" ? mod : mod.default;
  };
} catch {
  loadExtension = null;
}

const maybeTest = loadExtension
  ? test
  : (name, fn) => test(name, { skip: "pi package (jiti) not installed" }, fn);

// ---------------------------------------------------------------------------
// Server contract stubs
// ---------------------------------------------------------------------------

const CATALOG = {
  billing_order: ["local", "subscription_quota", "paid"],
  models: {
    subscription_quota: [
      { profile_name: "gpt_6_luna", model_id: "gpt-6-luna", provider: "github-copilot", billing: "subscription_quota" },
      { profile_name: "claude_sonnet_5", model_id: "claude-sonnet-5", provider: "github-copilot", billing: "subscription_quota" },
      { profile_name: "gpt_5_3_codex", model_id: "gpt-5.3-codex", provider: "github-copilot", billing: "subscription_quota" },
    ],
  },
};

const candidate = (profileId, provider, model) => ({
  profile_id: profileId,
  provider,
  model,
  capability: "normal",
  access_class: "subscription",
  priority: 10,
});

const overrideRouteResponse = {
  route_id: "route-override-1",
  classification: { classifier: "user_model_override", capability: "normal", confidence: 1.0, reasons: ["user_requested_model"], ambiguous: false },
  signals: {},
  decision: {
    tier: "cloud_standard",
    capability: "normal",
    reasons: ["user_requested_model", "override:gpt_6_luna"],
    candidates: [candidate("gpt_6_luna", "github-copilot", "gpt-6-luna")],
    excluded: [],
  },
  override: { requested: "gpt_6_luna", available: true },
};

const unavailableOverrideRouteResponse = {
  route_id: "route-fallback-1",
  classification: { classifier: "deterministic", capability: "strong", confidence: 0.9, reasons: ["strong_signal"], ambiguous: false },
  signals: {},
  decision: {
    tier: "cloud_coding",
    capability: "strong",
    reasons: ["strong_signal", "override_unavailable:gpt_6_luna"],
    candidates: [
      candidate("claude_sonnet_5", "github-copilot", "claude-sonnet-5"),
      candidate("gpt_5_3_codex", "github-copilot", "gpt-5.3-codex"),
    ],
    excluded: [],
  },
  override: { requested: "gpt_6_luna", available: false },
};

const liveRegistryAutoRouteResponse = {
  route_id: "route-live-registry-1",
  classification: { classifier: "deterministic", capability: "strong", confidence: 0.9, reasons: ["strong_signal_debug"], ambiguous: false },
  signals: {},
  decision: {
    tier: "cloud_coding",
    capability: "strong",
    reasons: ["strong_signal_debug", "capability:strong"],
    candidates: [
      candidate("claude_sonnet_5", "anthropic", "claude-sonnet-5"),
      candidate("gpt_5_3_codex", "openai-codex", "gpt-5.3-codex-spark"),
    ],
    excluded: [],
  },
};

// ---------------------------------------------------------------------------
// Stubbed PI runtime
// ---------------------------------------------------------------------------

function makeRuntime(router, { mode = "tui" } = {}) {
  const savedFetch = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (url, init = {}) => {
    const path = url.replace("http://router.test", "");
    const body = typeof init.body === "string" ? JSON.parse(init.body) : null;
    calls.push({ path, method: init.method, body });
    const handler = router[path];
    if (!handler) throw new Error(`no stub for ${path}`);
    const res = typeof handler === "function" ? handler(body) : handler;
    if (res instanceof Error) throw res;
    return { ok: true, status: 200, json: async () => res };
  };

  const handlers = new Map();
  const commands = new Map();
  const entries = [];
  const setModelCalls = [];
  const notifications = [];
  const statuses = [];
  const catalog = {
    "omlx/qwen3.8-27b": { provider: "omlx", id: "qwen3.8-27b" },
    "anthropic/claude-sonnet-5": { provider: "anthropic", id: "claude-sonnet-5" },
    "openai-codex/gpt-5.3-codex-spark": { provider: "openai-codex", id: "gpt-5.3-codex-spark" },
    "github-copilot/gpt-6-luna": { provider: "github-copilot", id: "gpt-6-luna" },
    "github-copilot/claude-sonnet-5": { provider: "github-copilot", id: "claude-sonnet-5" },
    "github-copilot/gpt-5.3-codex": { provider: "github-copilot", id: "gpt-5.3-codex" },
  };
  let currentModel = catalog["omlx/qwen3.8-27b"];

  const ctx = {
    get model() {
      return currentModel;
    },
    hasUI: true,
    mode,
    ui: {
      notify: (m) => notifications.push(m),
      setStatus: (k, t) => statuses.push([k, t]),
    },
    modelRegistry: {
      find: (provider, id) => catalog[`${provider}/${id}`],
      hasConfiguredAuth: () => true,
    },
    sessionManager: { getBranch: () => entries },
  };

  const pi = {
    on: (event, handler) => {
      if (!handlers.has(event)) handlers.set(event, []);
      handlers.get(event).push(handler);
    },
    registerCommand: (name, options) => commands.set(name, options),
    appendEntry: (customType, data) => {
      entries.push({ type: "custom", customType, data: structuredClone(data) });
    },
    setModel: async (model) => {
      setModelCalls.push(`${model.provider}/${model.id}`);
      const previous = currentModel;
      currentModel = model;
      for (const h of handlers.get("model_select") ?? []) {
        await h({ model, previousModel: previous, source: "set" }, ctx);
      }
      return true;
    },
  };

  async function emit(event, payload) {
    let last;
    for (const h of handlers.get(event) ?? []) last = await h(payload, ctx);
    return last;
  }

  /** Simulate a user model selection (/model, cycling) or a session restore. */
  async function select(modelRef, source = "set") {
    const sep = modelRef.indexOf("/");
    currentModel = { provider: modelRef.slice(0, sep), id: modelRef.slice(sep + 1) };
    await emit("model_select", { model: currentModel, previousModel: undefined, source });
  }

  loadExtension()(pi);
  return {
    emit,
    select,
    entries,
    calls,
    setModelCalls,
    notifications,
    statuses,
    commands,
    get currentModelRef() {
      return `${currentModel.provider}/${currentModel.id}`;
    },
    restoreFetch: () => {
      globalThis.fetch = savedFetch;
    },
  };
}

const routeCalls = (rt) => rt.calls.filter((c) => c.path === "/api/route");
const outcomeCalls = (rt) => rt.calls.filter((c) => c.path === "/api/route/outcome");
const catalogCalls = (rt) => rt.calls.filter((c) => c.path === "/api/models/overrides");
const lastState = (rt) => rt.entries[rt.entries.length - 1].data;

// ---------------------------------------------------------------------------
// Live provider/model compatibility
// ---------------------------------------------------------------------------

maybeTest("automatic route activates a model from the live Pi provider registry", async (t) => {
  process.env.RLCD_ROUTER_URL = "http://router.test";
  const rt = makeRuntime({
    "/api/route": liveRegistryAutoRouteResponse,
  });
  t.after(rt.restoreFetch);

  await rt.emit("before_agent_start", {
    type: "before_agent_start",
    prompt: "Debug the authentication race and add a regression test.",
  });

  assert.deepEqual(rt.setModelCalls, ["anthropic/claude-sonnet-5"]);
  assert.equal(lastState(rt).route.routeId, "route-live-registry-1");
  assert.ok(
    rt.statuses.some(([, line]) => line.includes("claude_sonnet_5")),
    "the activated profile is visible in Pi's router status line",
  );
});

// ---------------------------------------------------------------------------
// Round 2: known manual overrides reach the server's override contract
// ---------------------------------------------------------------------------

maybeTest("known override is routed with model_override and stays active when available", async (t) => {
  process.env.RLCD_ROUTER_URL = "http://router.test";
  const rt = makeRuntime({
    "/api/models/overrides": CATALOG,
    "/api/route": (body) => {
      assert.equal(body.model_override, "gpt_6_luna", "known override must be sent as model_override");
      return overrideRouteResponse;
    },
  });
  t.after(rt.restoreFetch);

  await rt.select("github-copilot/gpt-6-luna"); // user /model choice
  await rt.emit("before_agent_start", { type: "before_agent_start", prompt: "help me write a small function" });

  assert.equal(catalogCalls(rt).length, 1, "the override catalog must be consulted");
  assert.equal(routeCalls(rt).length, 1);
  assert.deepEqual(rt.setModelCalls, [], "available override is already active: no model churn");
  const state = lastState(rt);
  assert.equal(state.overrideModel, "github-copilot/gpt-6-luna");
  assert.deepEqual(state.route.candidates, ["gpt_6_luna"]);
  assert.ok(rt.statuses.some(([, line]) => line.startsWith("ROUTE ")), "compact route line shown");
});

maybeTest("unavailable known override activates the first pool candidate and keeps the preference", async (t) => {
  process.env.RLCD_ROUTER_URL = "http://router.test";
  const rt = makeRuntime({
    "/api/models/overrides": CATALOG,
    "/api/route": (body) => {
      assert.equal(body.model_override, "gpt_6_luna");
      return unavailableOverrideRouteResponse;
    },
  });
  t.after(rt.restoreFetch);

  await rt.select("github-copilot/gpt-6-luna");
  await rt.emit("before_agent_start", { type: "before_agent_start", prompt: "serious coding work" });

  assert.deepEqual(rt.setModelCalls, ["github-copilot/claude-sonnet-5"], "first pool candidate activated");
  const state = lastState(rt);
  assert.equal(
    state.overrideModel,
    "github-copilot/gpt-6-luna",
    "the manual preference must survive the automatic fallback (model_select guard)"
  );
  assert.deepEqual(state.route.candidates, ["claude_sonnet_5", "gpt_5_3_codex"]);
  assert.equal(state.route.cursor, 0);
  const lines = [...rt.notifications, ...rt.statuses.map(([, l]) => l)].join("\n");
  assert.match(lines, /override gpt_6_luna unavailable/, "compact override status surfaced");
});

maybeTest("override route permits bounded continuation and never clears the preference", async (t) => {
  process.env.RLCD_ROUTER_URL = "http://router.test";
  const rt = makeRuntime({
    "/api/models/overrides": CATALOG,
    "/api/route": unavailableOverrideRouteResponse,
  });
  t.after(rt.restoreFetch);

  await rt.select("github-copilot/gpt-6-luna");
  await rt.emit("before_agent_start", { type: "before_agent_start", prompt: "serious coding work" });

  // First pool candidate fails with a provider rate limit -> failover once.
  await rt.emit("after_provider_response", { type: "after_provider_response", status: 429, headers: { "Retry-After": "30" } });
  const settle1 = await rt.emit("agent_before_settle", { type: "agent_before_settle", outcome: "error" });
  assert.deepEqual(settle1, { continue: true }, "bounded continuation requested for the fallback route");
  assert.ok(rt.setModelCalls.includes("github-copilot/gpt-5.3-codex"), "next candidate activated");
  assert.equal(outcomeCalls(rt).length, 1);
  assert.equal(outcomeCalls(rt)[0].body.failure_category, "rate_limit");
  assert.equal(outcomeCalls(rt)[0].body.retry_after, 30);
  assert.equal(lastState(rt).overrideModel, "github-copilot/gpt-6-luna");

  // Continued attempt fails again: candidates exhausted, no further continuation.
  await rt.emit("after_provider_response", { type: "after_provider_response", status: 429, headers: {} });
  const settle2 = await rt.emit("agent_before_settle", { type: "agent_before_settle", outcome: "error" });
  assert.equal(settle2, undefined, "candidate exhaustion stops continuation");
  assert.equal(outcomeCalls(rt).length, 2);
  assert.equal(lastState(rt).overrideModel, "github-copilot/gpt-6-luna");
});

maybeTest("unknown manual model is preserved untouched and never routed", async (t) => {
  process.env.RLCD_ROUTER_URL = "http://router.test";
  const rt = makeRuntime({
    "/api/models/overrides": CATALOG,
    "/api/route": () => {
      throw new Error("must not route an unknown override");
    },
  });
  t.after(rt.restoreFetch);

  await rt.select("custom/mystery-model"); // not in the catalog
  await rt.emit("before_agent_start", { type: "before_agent_start", prompt: "some work" });

  assert.equal(catalogCalls(rt).length, 1);
  assert.equal(routeCalls(rt).length, 0, "no route request for an unknown override");
  assert.deepEqual(rt.setModelCalls, []);
  assert.equal(lastState(rt).overrideModel, "custom/mystery-model", "override retained until /router-auto");
});

maybeTest("catalog outage preserves the override untouched", async (t) => {
  process.env.RLCD_ROUTER_URL = "http://router.test";
  const rt = makeRuntime({
    "/api/models/overrides": new Error("catalog unavailable"),
    "/api/route": () => {
      throw new Error("must not route when the catalog is unreachable");
    },
  });
  t.after(rt.restoreFetch);

  await rt.select("github-copilot/gpt-6-luna");
  await rt.emit("before_agent_start", { type: "before_agent_start", prompt: "some work" });

  assert.equal(routeCalls(rt).length, 0, "mapping uncertainty never routes");
  assert.deepEqual(rt.setModelCalls, []);
  assert.equal(lastState(rt).overrideModel, "github-copilot/gpt-6-luna");
});

maybeTest("abort on an override route never reports or penalizes", async (t) => {
  process.env.RLCD_ROUTER_URL = "http://router.test";
  const rt = makeRuntime({
    "/api/models/overrides": CATALOG,
    "/api/route": overrideRouteResponse,
  });
  t.after(rt.restoreFetch);

  await rt.select("github-copilot/gpt-6-luna");
  await rt.emit("before_agent_start", { type: "before_agent_start", prompt: "some work" });
  await rt.emit("after_provider_response", { type: "after_provider_response", status: 429, headers: {} });
  const settle = await rt.emit("agent_before_settle", { type: "agent_before_settle", outcome: "aborted" });

  assert.equal(settle, undefined);
  assert.equal(outcomeCalls(rt).length, 0, "aborted turns never report");
  const state = lastState(rt);
  assert.equal(state.route, null);
  assert.equal(state.overrideModel, "github-copilot/gpt-6-luna");
});
