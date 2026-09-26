/**
 * Thin turn-aware router extension for Pi.
 *
 * All state transitions (override semantics, candidate advancement,
 * failure classification, continuation bounds) live in the pure,
 * runtime-free `router-state.ts`; this module only wires those
 * transitions into Pi's documented lifecycle, model, and persistence
 * APIs and talks to the router server's HTTP contract:
 *
 *   POST /api/route          - classify + ordered candidates
 *   POST /api/route/outcome  - attempt outcome (availability vs task)
 *   GET  /api/router/status  - compact server status (/router-status)
 *   GET  /health             - identity check for optional auto-start
 *
 * Design invariants:
 *   - The factory registers handlers only; processes are never started
 *     there (some invocations load extensions without a session).
 *   - Optional router auto-start happens only in session_start, after a
 *     /health probe; session_shutdown cleanup is idempotent and never
 *     kills a shared pre-existing server we did not spawn.
 *   - Every server interaction fails open: an unreachable router or a
 *     failed report retains the active model and never blocks a turn.
 *   - Automatic candidate activation is wrapped in a guard token so the
 *     resulting model_select event is not mistaken for a user override;
 *     user set/cycle *does* create a persistent override (respected
 *     even when no router profile maps the model) until /router-auto.
 *   - A *known* override (exact match in the server's
 *     /api/models/overrides catalog) is routed with model_override so
 *     the server can honor it or fall back to the normal pool during
 *     cooldown; the manual preference is never cleared by that
 *     fallback.  Mapping or catalog uncertainty retains the override
 *     untouched and never routes.
 */

import { spawn, type ChildProcess } from "node:child_process";
import { realpathSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

import {
  ROUTER_STATE_ENTRY,
  advanceCandidate,
  beginRoute,
  clearOverride,
  createRouterState,
  flattenOverrideCatalog,
  outageDecision,
  resolveOverrideProfile,
  restoreFromBranch,
  settleAttempt,
  withAutomaticSelection,
  withModelSelection,
  type RouterSessionState,
} from "./router-state.ts";

type PiModel = any; // Model<any> from pi-ai; structural use only

// ---------------------------------------------------------------------------
// Configuration (environment-overridable)
// ---------------------------------------------------------------------------

const ROUTER_URL = (
  process.env.RLCD_ROUTER_URL ?? "http://127.0.0.1:8001"
).replace(/\/+$/, "");
const ROUTE_TIMEOUT_MS = Number(process.env.RLCD_ROUTER_TIMEOUT_MS ?? 2000);
const STATUS_TIMEOUT_MS = Number(
  process.env.RLCD_ROUTER_STATUS_TIMEOUT_MS ?? 2000,
);
const HEALTH_TIMEOUT_MS = Number(
  process.env.RLCD_ROUTER_HEALTH_TIMEOUT_MS ?? 1000,
);
// Optional server auto-start: off unless explicitly enabled.
const AUTO_START = ["1", "true", "yes", "on"].includes(
  (process.env.RLCD_ROUTER_AUTO_START ?? "0").trim().toLowerCase(),
);

/** Repo root resolved through the symlink (the extension is symlinked into ~/.pi/agent/extensions). */
const ROUTER_DIR = dirname(dirname(realpathSync(fileURLToPath(import.meta.url))));

// ---------------------------------------------------------------------------
// Extension
// ---------------------------------------------------------------------------

export default function turnRouter(pi: ExtensionAPI) {
  // --- per-runtime mutable session state ---------------------------------
  let state: RouterSessionState = createRouterState();
  /** Guard token around automatic setModel calls. */
  let automaticSwitchToken = false;
  /** Resolved Pi models for the active route, by candidate profile id. */
  let routeModels = new Map<string, PiModel | undefined>();
  /** Provider evidence captured by after_provider_response this turn. */
  let providerStatus: number | null = null;
  let providerHeaders: Record<string, string> = {};
  /** Server process this runtime spawned (owned; never a shared one). */
  let ownedServer: ChildProcess | null = null;

  // --- helpers -------------------------------------------------------------
  const persist = (): void => pi.appendEntry(ROUTER_STATE_ENTRY, state);

  const notify = (
    ctx: { hasUI: boolean; ui: { notify: (m: string, t?: "info" | "warning" | "error") => void } },
    message: string,
    type: "info" | "warning" | "error" = "info",
  ): void => {
    if (ctx.hasUI) ctx.ui.notify(message, type);
  };

  /** One compact route line, guarded for non-UI modes. */
  const showRoute = (
    ctx: { mode: string; hasUI: boolean; ui: any },
    line: string,
  ): void => {
    if (ctx.mode === "tui") ctx.ui.setStatus("router", line);
    else if (ctx.hasUI) ctx.ui.notify(line, "info");
  };

  async function postJson<T = any>(
    path: string,
    body: unknown,
    timeoutMs: number,
  ): Promise<T> {
    const res = await fetch(`${ROUTER_URL}${path}`, {
      method: body === null ? "GET" : "POST",
      headers: { "Content-Type": "application/json" },
      body: body === null ? undefined : JSON.stringify(body),
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!res.ok) throw new Error(`router ${path} -> HTTP ${res.status}`);
    return (await res.json()) as T;
  }

  /**
   * Activate a candidate model.  The guard token ensures the
   * model_select event fired by our own setModel is classified as
   * automatic and never creates a user override.
   */
  async function activate(
    ctx: { modelRegistry: any },
    model: PiModel,
  ): Promise<boolean> {
    try {
      if (!ctx.modelRegistry.hasConfiguredAuth(model)) return false;
    } catch {
      return false;
    }
    automaticSwitchToken = true;
    try {
      return await pi.setModel(model);
    } finally {
      automaticSwitchToken = false;
    }
  }

  /** Resolve a server candidate (provider/model) through Pi's registry. */
  function resolveModel(
    ctx: { modelRegistry: any },
    candidate: { provider?: string; model?: string },
  ): PiModel | undefined {
    if (typeof candidate.provider !== "string" || typeof candidate.model !== "string") {
      return undefined;
    }
    try {
      return ctx.modelRegistry.find(candidate.provider, candidate.model);
    } catch {
      return undefined;
    }
  }

  /**
   * Map a manual override ("provider/model") to its router profile name
   * via the server's /api/models/overrides catalog.  Any uncertainty
   * (catalog outage, malformed response, no exact match) returns null,
   * which means: retain the override untouched and do not route.
   */
  async function lookupOverrideProfile(
    overrideModel: string,
  ): Promise<string | null> {
    let catalog: unknown;
    try {
      catalog = await postJson("/api/models/overrides", null, STATUS_TIMEOUT_MS);
    } catch {
      return null; // catalog outage: keep the override, do not route
    }
    return resolveOverrideProfile(overrideModel, flattenOverrideCatalog(catalog));
  }

  // --- session lifecycle ----------------------------------------------------

  pi.on("session_start", async (_event, ctx) => {
    // Rebuild branch-sensitive state from the latest compatible
    // active-branch router entry only; abandoned branches are
    // alternative histories and are never consulted.
    state = restoreFromBranch(ctx.sessionManager.getBranch()) ?? createRouterState();

    // Optional server auto-start: health-probe first (a shared
    // pre-existing server is left alone), spawn only when configured,
    // never in the factory.
    if (!AUTO_START) return;
    if (ownedServer && ownedServer.exitCode === null) return; // ours, alive
    try {
      await postJson("/health", null, HEALTH_TIMEOUT_MS);
      return; // a server is already running (shared): never ours to stop
    } catch {
      ownedServer = spawnRouter();
    }
  });

  pi.on("session_shutdown", () => {
    stopOwnedServer();
  });

  /** Spawn the router server (only ever from session_start). */
  function spawnRouter(): ChildProcess | null {
    try {
      const child = spawn("/bin/bash", [join(ROUTER_DIR, "run.sh")], {
        detached: true,
        stdio: "ignore",
        env: { ...process.env },
      });
      child.unref();
      return child;
    } catch {
      return null; // fail open: routing stays unavailable, turns proceed
    }
  }

  /** Idempotent cleanup; never kills a shared pre-existing server. */
  function stopOwnedServer(): void {
    const child = ownedServer;
    ownedServer = null;
    if (!child || child.exitCode !== null || child.pid === undefined) return;
    try {
      process.kill(-child.pid, "SIGTERM"); // detached group
    } catch {
      /* already gone */
    }
    try {
      child.kill("SIGTERM");
    } catch {
      /* already gone */
    }
  }

  // --- model selection: override semantics ---------------------------------

  pi.on("model_select", (event, ctx) => {
    const modelRef = `${event.model.provider}/${event.model.id}`;
    if (automaticSwitchToken) {
      // Guarded automatic switch (candidate activation): never an override.
      automaticSwitchToken = false;
      state = withAutomaticSelection(state, modelRef);
      persist();
      return;
    }
    if (event.source === "restore") {
      // Session restore: adopt the restored model, keep persisted override.
      state = withModelSelection(state, modelRef, "restore");
      persist();
      return;
    }
    // User /model or model cycling: a persistent override, respected even
    // when no router profile maps the model, until /router-auto clears it.
    state = withModelSelection(state, modelRef, event.source);
    persist();
    notify(ctx, `Router override: ${modelRef} (use /router-auto to clear)`);
  });

  // --- routing --------------------------------------------------------------

  pi.on("before_agent_start", async (event, ctx) => {
    // Each routed turn starts with fresh provider evidence and no stale route.
    providerStatus = null;
    providerHeaders = {};
    routeModels = new Map();
    if (state.route) state = { ...state, route: null };

    const prompt = typeof event.prompt === "string" ? event.prompt : "";
    const trimmed = prompt.trim();
    if (!trimmed || trimmed.startsWith("/")) return; // commands / empty: no routing

    // A manual override is kept until /router-auto.  A *known* override
    // (exact match in the server catalog) is routed with model_override so
    // the server can honor it — or, while it is in cooldown/incompatible,
    // fall back to the normal pool for this turn.  Unknown overrides and
    // catalog uncertainty retain the override untouched and never route.
    let overrideProfile: string | null = null;
    if (state.overrideModel) {
      overrideProfile = await lookupOverrideProfile(state.overrideModel);
      if (!overrideProfile) return; // retain the active model until /router-auto
    }

    let body: any;
    try {
      body = await postJson(
        "/api/route",
        {
          ...(overrideProfile ? { model_override: overrideProfile } : {}),
          prompt,
          turn_context: {
            previous_prompt: state.previousPrompt,
            previous_capability: state.previousCapability,
            previous_model: state.activeModel,
            previous_route_reason: state.previousRouteReason,
            task_depth: state.taskDepth,
            files_involved: 0,
            task_failures: state.taskFailures,
          },
        },
        ROUTE_TIMEOUT_MS,
      );
    } catch {
      // Router outage: retain the active model (or an explicitly resolved
      // fallback); fail open, never block the turn.
      const decision = outageDecision(state, null);
      showRoute(
        ctx,
        `ROUTE unreachable · retaining ${decision.model ?? "active model"}`,
      );
      return;
    }

    const routeDecision = body?.decision ?? {};
    const routeId = typeof body?.route_id === "string" ? body.route_id : "";
    const candidates: any[] = Array.isArray(routeDecision.candidates)
      ? routeDecision.candidates
      : [];

    // Compact override status (unavailable / incompatible) from the
    // server; routing continues from the normal pool while the manual
    // preference stays persisted for retry after cooldown.
    let overrideNote = "";
    const overrideInfo = body?.override;
    if (overrideProfile && overrideInfo && typeof overrideInfo === "object") {
      if (overrideInfo.available === false) {
        overrideNote = `override ${overrideProfile} unavailable`;
      } else if (overrideInfo.compatible === false) {
        overrideNote = `override ${overrideProfile} incompatible (${overrideInfo.reason ?? "requirements"})`;
      }
    }

    if (!routeId || candidates.length === 0) {
      // Empty candidate list (the server found nothing eligible, including
      // its own safe-subscription fallback resolution): retain active model.
      showRoute(
        ctx,
        `ROUTE ${overrideNote ? `${overrideNote} · ` : ""}no candidates · retaining active model`,
      );
      return;
    }

    // Resolve every ordered candidate through Pi's registry up front, then
    // activate the first activatable one; activation failures advance
    // without reclassification.
    for (const candidate of candidates) {
      if (typeof candidate?.profile_id === "string") {
        routeModels.set(candidate.profile_id, resolveModel(ctx, candidate));
      }
    }
    let activeIndex = -1;
    let activeProfile = "";
    for (let i = 0; i < candidates.length; i++) {
      const candidate = candidates[i] ?? {};
      const model = routeModels.get(String(candidate.profile_id ?? ""));
      if (!model) continue; // not in PI's registry: an activation failure
      // Already the active model (a known available override usually is):
      // keep it without a redundant setModel/model_select round trip.
      if (
        ctx.model !== undefined &&
        model.provider === ctx.model.provider &&
        model.id === ctx.model.id
      ) {
        activeIndex = i;
        activeProfile = String(candidate.profile_id ?? "");
        break;
      }
      if (await activate(ctx, model)) {
        activeIndex = i;
        activeProfile = String(candidate.profile_id ?? "");
        break;
      }
    }
    if (activeIndex < 0) {
      showRoute(ctx, "ROUTE no activatable candidate · retaining active model");
      return;
    }

    const capability = typeof routeDecision.capability === "string"
      ? routeDecision.capability
      : state.previousCapability;
    const classifier =
      typeof body?.classification?.classifier === "string"
        ? body.classification.classifier
        : "deterministic";
    const reasons: string[] = Array.isArray(routeDecision.reasons)
      ? routeDecision.reasons.map(String)
      : [];

    state = beginRoute(state, {
      routeId,
      candidates: candidates.map((c) => String(c?.profile_id ?? "")),
      activeIndex,
      prompt,
      capability: capability ?? null,
      reasons,
    });
    persist();

    // One compact route line: capability · classifier · profile · reason
    // (with the override status appended when the server reported one).
    showRoute(
      ctx,
      `ROUTE ${capability ?? "-"} · ${classifier} · ${activeProfile} · ${reasons[0] ?? "-"}${
        overrideNote ? ` · ${overrideNote}` : ""
      }`,
    );
  });

  // --- provider evidence ------------------------------------------------------

  pi.on("after_provider_response", (event) => {
    providerStatus = event.status;
    providerHeaders = event.headers ?? {};
  });

  // --- bounded outcome fallback -------------------------------------------------

  pi.on("agent_before_settle", async (event, ctx) => {
    const route = state.route;
    if (!route) return; // no routed attempt this turn (override, outage, command)

    const settled = settleAttempt(state, {
      outcome: event.outcome,
      providerStatus,
      providerHeaders,
    });
    state = settled.state;
    const activeProfile = route.candidates[route.cursor] ?? null;

    if (settled.report && activeProfile) {
      try {
        await postJson(
          "/api/route/outcome",
          {
            event_id: route.routeId,
            status: settled.report.status,
            failure_category: settled.report.failureCategory,
            retry_after: settled.report.retryAfter,
            candidate_profile: activeProfile,
            actual_model: state.activeModel,
          },
          STATUS_TIMEOUT_MS,
        );
      } catch {
        // Fail open: a failed report must never block settlement.
      }
    }
    persist();

    if (!settled.continueTurn) return; // aborted, task failure, exhausted, or at the attempt limit

    // Bounded fallback: activate the next candidate; activation failures
    // advance the cursor without reclassification, and continuation is
    // requested only when an activation succeeded and the attempt limit
    // permits — so no loop can form.
    let next = settled.nextProfileId;
    while (
      next !== null &&
      state.route !== null &&
      state.route.attempts < state.route.maxAttempts
    ) {
      const model = routeModels.get(next);
      let activated = false;
      if (model) {
        activated = await activate(ctx, model);
        if (activated) {
          showRoute(
            ctx,
            `ROUTE failover ${activeProfile} -> ${next} · ${
              settled.report?.failureCategory ?? "failure"
            }`,
          );
          persist();
          // Fresh evidence for the continued attempt: clear the failed
          // attempt's status/Retry-After so a continuation that errors
          // without any HTTP response cannot settle against stale
          // provider evidence (which would wrongly cool this candidate).
          providerStatus = null;
          providerHeaders = {};
          return { continue: true };
        }
      }
      const advanced = advanceCandidate(state);
      state = advanced.state;
      next = advanced.nextProfileId;
    }
    persist();
  });

  // --- manual controls -----------------------------------------------------------

  pi.registerCommand("router-auto", {
    description: "Clear the manual model override and resume automatic routing",
    handler: async (_args, ctx) => {
      const hadOverride = state.overrideModel !== null;
      state = clearOverride(state);
      persist();
      notify(
        ctx,
        hadOverride
          ? "Router: manual override cleared, automatic routing resumes"
          : "Router: already automatic",
      );
    },
  });

  pi.registerCommand("router-status", {
    description: "Show router mode, override, and compact server status",
    handler: async (_args, ctx) => {
      const lines: string[] = [];
      if (state.overrideModel) {
        lines.push(`mode: override ${state.overrideModel}`);
      } else {
        lines.push(`mode: auto (${state.activeModel || "active model"})`);
      }
      if (state.previousCapability) {
        lines.push(`last capability: ${state.previousCapability}`);
      }

      let server: any = null;
      try {
        server = await postJson("/api/router/status", null, STATUS_TIMEOUT_MS);
      } catch {
        lines.push(`server: unreachable (${ROUTER_URL})`);
      }

      if (server) {
        const ollaya = server?.classifier?.ollaya;
        lines.push(
          `ollaya: ${
            ollaya
              ? `${ollaya.enabled ? "enabled" : "disabled"} (${ollaya.health})`
              : "unknown"
          }`,
        );
        const cooldowns: string[] = [];
        for (const [name, info] of Object.entries<any>(server.models ?? {})) {
          const availability = info?.availability;
          if (availability?.status === "cooldown") {
            const left = Math.max(
              0,
              Math.round((availability.cooldown_until ?? 0) - Date.now() / 1000),
            );
            cooldowns.push(`${name}(${availability.reason ?? "?"} ${left}s)`);
          }
        }
        lines.push(
          cooldowns.length
            ? `cooldowns: ${cooldowns.join(", ")}`
            : "availability: all available",
        );
        const last = server.last_route;
        if (last) {
          lines.push(
            `last route: ${last.capability} · ${last.classifier} · ${
              Array.isArray(last.candidates) ? last.candidates[0] ?? "-" : "-"
            }`,
          );
        }
      }
      notify(ctx, lines.join("\n"));
    },
  });
}
