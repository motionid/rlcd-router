/**
 * RLCD Router Extension for Pi
 *
 * 1. On session_start: ensures the configured RLCD routing server is running.
 *    If it's already running (from another Pi session), does nothing.
 *    If not, spawns it as a detached background process.
 *
 * 2. On before_agent_start (every turn): classifies the incoming prompt via
 *    the router and calls pi.setModel() BEFORE the turn executes, so the
 *    routing decision actually controls which model runs -- not just what
 *    /route prints. This is the piece that protects Copilot quota/cost.
 *
 * 3. Registers a /route command so the user can ask "what model would handle this?"
 *
 * 4. Registers a one-shot /model-override command that changes the next
 *    routing decision (manual /route call, or the next automatic turn) --
 *    e.g. "/model-override claude_opus_4_6".
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { spawn } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROUTER_PORT = Number(process.env.RLCD_ROUTER_PORT ?? "8001");
const ROUTER_URL =
  process.env.RLCD_ROUTER_URL ?? `http://127.0.0.1:${ROUTER_PORT}`;
const EXTENSION_DIR = dirname(fileURLToPath(import.meta.url));
const ROUTER_DIR = process.env.RLCD_ROUTER_DIR ?? resolve(EXTENSION_DIR, "..");
const ROUTER_PYTHON =
  process.env.RLCD_PYTHON ?? resolve(ROUTER_DIR, ".venv/bin/python");

// ─── helpers ───────────────────────────────────────────────────────

async function checkRouter(): Promise<boolean> {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 750);
    const response = await fetch(`${ROUTER_URL}/health`, {
      signal: controller.signal,
    });
    clearTimeout(timeout);
    if (!response.ok) return false;
    const body = (await response.json()) as {
      status?: string;
      service?: string;
    };
    return body.status === "ok" && body.service === "rlcd-router";
  } catch {
    return false;
  }
}

async function waitForServer(timeoutMs = 30_000): Promise<boolean> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (await checkRouter()) return true;
    await new Promise((r) => setTimeout(r, 500));
  }
  return false;
}

interface OverrideCatalog {
  billing_order: string[];
  models: Record<string, any[]>;
}

async function fetchOverrideModels(): Promise<OverrideCatalog | null> {
  try {
    const resp = await fetch(`${ROUTER_URL}/api/models/overrides`);
    if (!resp.ok) return null;
    const data = (await resp.json()) as Partial<OverrideCatalog>;
    if (!data.models) return null;
    return {
      // Billing groups come from the server's billing_order so the
      // override catalog UI never drifts from the profiles the router
      // actually defines (stale hardcoded groups are gone). Fall back to
      // the response's own group keys if billing_order is absent.
      billing_order: data.billing_order ?? Object.keys(data.models),
      models: data.models,
    };
  } catch {
    return null;
  }
}

async function fetchFallbackRecommendation(
  currentModel: string,
): Promise<any | null> {
  try {
    const resp = await fetch(`${ROUTER_URL}/api/models/recommend`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ current_model: currentModel }),
    });
    if (!resp.ok) return null;
    return await resp.json();
  } catch {
    return null;
  }
}

// ─── model resolution (router profile -> Pi provider/model id) ───

interface RouterModelProfile {
  model_id: string;
  role: string;
  provider: string;
  billing: string;
}

interface RouterTierDefinition {
  name: string;
  preferred_model_profiles: string[];
}

interface RouterConfig {
  models: Record<string, RouterModelProfile>;
  tiers: Record<string, RouterTierDefinition>;
}

let routerConfigCache: RouterConfig | null = null;

async function fetchRouterConfig(): Promise<RouterConfig | null> {
  if (routerConfigCache) return routerConfigCache;
  try {
    const resp = await fetch(`${ROUTER_URL}/api/route/config`);
    if (!resp.ok) return null;
    routerConfigCache = (await resp.json()) as RouterConfig;
    return routerConfigCache;
  } catch {
    return null;
  }
}

/**
 * The router's local ("omlx") executor profiles store a full filesystem
 * path as model_id (needed by core/engine_mlx.py's own loader), but Pi's
 * "omlx" provider in models.json registers local models by basename only.
 * Cloud ("github-copilot") profiles already match Pi's catalogue ids
 * directly. Resolve accordingly.
 */
function resolvePiModelId(provider: string, modelId: string): string {
  if (provider === "omlx") {
    const parts = modelId.split("/").filter(Boolean);
    return parts[parts.length - 1] ?? modelId;
  }
  return modelId;
}

/**
 * Walk an ordered list of router profile names and return the first one
 * that is both (a) an executor profile and (b) resolves to a model Pi's
 * own registry actually has available (API key / local server present).
 * Non-executor profiles (e.g. "rlcd_fast", a classifier-only model with
 * no Pi provider) are skipped rather than treated as failures.
 */
function findExecutorModel(
  profileNames: string[],
  profiles: Record<string, RouterModelProfile>,
  ctx: any,
): { profileName: string; model: any } | null {
  for (const profileName of profileNames) {
    const profile = profiles[profileName];
    if (!profile || profile.role !== "executor") continue;
    const piModelId = resolvePiModelId(profile.provider, profile.model_id);
    const model = ctx.modelRegistry.find(profile.provider, piModelId);
    if (model) return { profileName, model };
  }
  return null;
}

/**
 * Resolve a routing decision to an actual Pi model, falling through to the
 * decision's own next_tier once if the primary tier has no profile Pi can
 * actually activate (e.g. local_fast's only profile is a classifier-only
 * model with no chat provider registered in Pi).
 */
async function resolveDecisionToModel(
  decision: {
    tier: string;
    eligible_models: string[];
    escalation_config?: { next_tier?: string };
  },
  config: RouterConfig,
  ctx: any,
): Promise<{ profileName: string; model: any } | null> {
  const primary = findExecutorModel(
    decision.eligible_models ?? [],
    config.models,
    ctx,
  );
  if (primary) return primary;

  const nextTier = decision.escalation_config?.next_tier;
  const nextProfiles = nextTier
    ? config.tiers[nextTier]?.preferred_model_profiles
    : undefined;
  if (nextProfiles) return findExecutorModel(nextProfiles, config.models, ctx);
  return null;
}

// ─── extension ─────────────────────────────────────────────────────

export default function (pi: ExtensionAPI) {
  // One-shot override consumed by the next /route command.
  let modelOverride: string | null = null;
  // route_id from the most recent automatic /api/route call, consumed by
  // the outcome-reporting hook below.
  let lastRouteId: string | null = null;
  // Consecutive routing failures, fed back into /api/route as
  // runtime.prior_failures so policy.py escalates to a bigger tier.
  // Reset to 0 on any successful turn.
  let priorFailures = 0;
  // HTTP status from the most recent provider response this turn, used to
  // tell a real quota/rate-limit failure (429) apart from other errors.
  let lastProviderStatus: number | null = null;

  // ── Lifecycle: start the router on session_start ──────────────

  pi.on("session_start", async (_event, ctx) => {
    const running = await checkRouter();

    if (running) {
      ctx.ui.setStatus("rlcd", "🧭 Router online");
      return;
    }

    ctx.ui.setStatus("rlcd", "🧭 Starting router…");

    const child = spawn(
      ROUTER_PYTHON,
      [
        "-m",
        "uvicorn",
        "server.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        String(ROUTER_PORT),
      ],
      {
        cwd: ROUTER_DIR,
        env: { ...process.env, PYTHONPATH: ROUTER_DIR },
        detached: true,
        stdio: "ignore",
      },
    );
    child.unref();

    const ready = await waitForServer(30_000);
    if (ready) {
      ctx.ui.setStatus("rlcd", "🧭 Router online");
    } else {
      ctx.ui.setStatus("rlcd", "⚠️ Router failed to start");
      ctx.ui.notify(`RLCD router failed to start at ${ROUTER_URL}`, "error");
    }
  });

  // ── Tool: rlcd_classify (LLM can call the router) ────────────

  pi.registerTool({
    name: "rlcd_classify",
    label: "RLCD Classify",
    description:
      "Use the local RLCD Qwen model for fast constrained classification, routing, or structured decisions.",
    parameters: Type.Object({
      context: Type.String({
        description: "Text or task that should be classified",
      }),
      schema: Type.Record(Type.String(), Type.Any(), {
        description:
          "Output fields to classify. Each field is either a list of allowed values or a field object.",
      }),
    }),
    async execute(_toolCallId, params, _signal) {
      try {
        const resp = await fetch(`${ROUTER_URL}/api/run-parallel`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            context: params.context,
            schema: params.schema,
          }),
        });
        if (!resp.ok) {
          const err = await resp.text();
          return {
            content: [
              { type: "text", text: `RLCD error (${resp.status}): ${err}` },
            ],
          };
        }
        const result = await resp.json();
        return {
          content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
          details: result,
        };
      } catch (e: any) {
        return {
          content: [
            {
              type: "text",
              text: `Failed to reach RLCD router at ${ROUTER_URL}: ${e.message}`,
            },
          ],
        };
      }
    },
  });

  // ── Hook: before_agent_start — actually enforce the routing decision ──
  //
  // This is the piece that closes the loop: without it, /route and
  // /model-override only ever *displayed* a recommendation while Sonnet
  // (or whatever was already active) kept running every turn.

  pi.on("before_agent_start", async (event, ctx) => {
    const prompt = event.prompt?.trim();
    if (!prompt || prompt.startsWith("/")) return; // let slash commands pass through untouched

    try {
      lastProviderStatus = null;
      const body: Record<string, any> = {
        prompt,
        runtime: { prior_failures: priorFailures },
      };
      // One-shot override: capture it for this turn, but only clear the
      // shared state once a model has actually been activated below. If
      // the /api/route fetch fails, the router config is unavailable, or
      // Pi cannot activate the chosen model, the override survives so
      // the next turn retries it instead of silently dropping it.
      const overrideForThisTurn = modelOverride;
      if (overrideForThisTurn) {
        body.model_override = overrideForThisTurn;
      }

      const resp = await fetch(`${ROUTER_URL}/api/route`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!resp.ok) {
        ctx.ui.setStatus("rlcd", "⚠️ Router error");
        return;
      }
      const data = await resp.json();
      const decision = data.decision as {
        tier: string;
        eligible_models: string[];
        confidence: number;
        escalation_config?: { next_tier?: string };
      };
      lastRouteId = data.route_id ?? null;

      const config = await fetchRouterConfig();
      if (!config) {
        // No model will be activated for this turn; suppress outcome
        // reporting so telemetry does not pretend routing succeeded.
        lastRouteId = null;
        ctx.ui.setStatus("rlcd", "⚠️ Router config unavailable");
        return;
      }

      const picked = await resolveDecisionToModel(decision, config, ctx);
      if (!picked) {
        // Same as above: the decision selected nothing Pi can run.
        lastRouteId = null;
        ctx.ui.setStatus(
          "rlcd",
          `⚠️ No available model for tier ${decision.tier}`,
        );
        return;
      }

      const alreadyActive =
        ctx.model &&
        ctx.model.provider === picked.model.provider &&
        ctx.model.id === picked.model.id;
      if (!alreadyActive) {
        const success = await pi.setModel(picked.model);
        if (!success) {
          // Activation failed: the routing decision never took effect,
          // so do not report an outcome for this route.
          lastRouteId = null;
          ctx.ui.notify(
            `Router picked ${picked.profileName} (${picked.model.provider}/${picked.model.id}) but Pi could not activate it (missing API key/server?).`,
            "warning",
          );
          ctx.ui.setStatus("rlcd", "⚠️ Switch failed");
          return;
        }
      }

      // The routing decision now actually controls this turn: consume
      // the one-shot override and report the outcome against this route.
      if (overrideForThisTurn) {
        modelOverride = null;
      }
      ctx.ui.setStatus("rlcd", `🧭 ${decision.tier} → ${picked.profileName}`);
    } catch (e: any) {
      // Fail open: never block a turn because the router is unreachable.
      // Also drop any pending route id -- if we got here mid-turn we
      // cannot trust that a model was activated for it.
      lastRouteId = null;
      ctx.ui.setStatus("rlcd", "⚠️ Router unreachable");
    }
  });

  // ── Hook: after_provider_response — remember HTTP status for outcome reporting ──

  pi.on("after_provider_response", async (event) => {
    lastProviderStatus = event.status;
  });

  // ── Hook: agent_before_settle — report outcome + drive escalation ──
  //
  // Closes the loop the other way: tells the router whether the turn it
  // routed actually succeeded, so a real quota-exhaustion (HTTP 429) can
  // trigger an automatic model switch, and any failure bumps
  // prior_failures so the *next* /api/route call escalates to a bigger tier.

  pi.on("agent_before_settle", async (event, ctx) => {
    if (!lastRouteId) return;
    const routeId = lastRouteId;
    lastRouteId = null;

    if (event.outcome === "aborted") return; // user-cancelled, not a model failure

    const status = event.outcome === "completed" ? "success" : "failed";
    let failure_reason = "";
    if (status === "failed") {
      // Only a real quota/rate-limit rejection (HTTP 429) counts as a
      // subscription exhaustion; every other failure is a generic agent
      // error and must not trigger an automatic model switch.
      failure_reason =
        lastProviderStatus === 429 ? "subscription_exhausted" : "agent_error";
    }
    const actualModel = ctx.model?.id ?? "";

    priorFailures = status === "success" ? 0 : priorFailures + 1;

    try {
      const resp = await fetch(`${ROUTER_URL}/api/route/outcome`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          event_id: routeId,
          status,
          failure_reason,
          actual_model: actualModel,
        }),
      });
      if (!resp.ok) return;
      const data = await resp.json();
      const modelSwitch = data.model_switch;
      if (
        failure_reason === "subscription_exhausted" &&
        modelSwitch?.recommended?.profile_name
      ) {
        // Auto-apply: quota is exhausted on the current model, so the very
        // next turn should use the recommendation without waiting for the
        // user to notice and run /model-exhausted manually.
        modelOverride = modelSwitch.recommended.profile_name;
        ctx.ui.notify(
          `${actualModel || "Current model"} looks quota-exhausted (HTTP 429). ` +
            `Switching to ${modelSwitch.recommended.profile_name} next turn.`,
          "warning",
        );
      }
    } catch {
      // Fail open: outcome reporting is best-effort, never blocks settlement.
    }
  });

  // ── Command: /route — show what the router would pick ─────────

  pi.registerCommand("route", {
    description: "Show what model the RLCD router would pick for a prompt",
    handler: async (args, ctx) => {
      if (!args) {
        ctx.ui.notify("Usage: /route <prompt to classify>", "info");
        return;
      }
      try {
        const body: Record<string, any> = { prompt: args, runtime: {} };
        if (modelOverride) {
          body.model_override = modelOverride;
        }
        const resp = await fetch(`${ROUTER_URL}/api/route`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!resp.ok) {
          ctx.ui.notify(`Router error: ${resp.status}`, "error");
          return;
        }
        const data = await resp.json();
        const d = data.decision;
        const models = (d.eligible_models || []).join(" → ");
        if (modelOverride) {
          modelOverride = null;
          ctx.ui.setStatus("rlcd", "🧭 Router online (auto)");
        }
        ctx.ui.notify(
          `Tier: ${d.tier}  |  Models: ${models}  |  Confidence: ${d.confidence}`,
          "info",
        );
      } catch (e: any) {
        ctx.ui.notify(`Router unreachable: ${e.message}`, "error");
      }
    },
  });

  // ── Command: /model-exhausted — choose a subscription fallback ──

  pi.registerCommand("model-exhausted", {
    description:
      "Choose a replacement when a model subscription is exhausted. Usage: /model-exhausted <profile>",
    handler: async (args, ctx) => {
      const currentModel = args.trim();
      if (!currentModel) {
        ctx.ui.notify(
          "Usage: /model-exhausted <profile name or model id>",
          "info",
        );
        return;
      }

      const recommendation = await fetchFallbackRecommendation(currentModel);
      if (!recommendation) {
        ctx.ui.notify(
          "Could not get a fallback recommendation from the router.",
          "error",
        );
        return;
      }

      const options: string[] = [];
      if (recommendation.recommended) {
        options.push(
          `Use recommended: ${recommendation.recommended.profile_name} (${recommendation.recommended.model_id})`,
        );
      }
      for (const alternative of recommendation.alternatives ?? []) {
        options.push(
          `Use alternative: ${alternative.profile_name} (${alternative.model_id})`,
        );
      }
      options.push("Choose another model…");

      const choice = await ctx.ui.select(
        recommendation.message ??
          "Subscription exhausted. Choose a replacement:",
        options,
      );
      if (!choice) return;

      let profileName: string | null = null;
      if (choice.startsWith("Use recommended: ")) {
        profileName = choice.slice("Use recommended: ".length).split(" ")[0];
      } else if (choice.startsWith("Use alternative: ")) {
        profileName = choice.slice("Use alternative: ".length).split(" ")[0];
      } else {
        const catalog = await fetchOverrideModels();
        if (!catalog) {
          ctx.ui.notify(
            "Could not fetch available models from the router.",
            "error",
          );
          return;
        }
        const allOptions: string[] = [];
        for (const billing of catalog.billing_order) {
          for (const model of catalog.models[billing] ?? []) {
            allOptions.push(
              `${model.profile_name} (${model.provider}/${model.model_id}) [${billing}]`,
            );
          }
        }
        const selected = await ctx.ui.select(
          "Choose a different model:",
          allOptions,
        );
        if (selected) profileName = selected.split(" ")[0];
      }

      if (profileName) {
        modelOverride = profileName;
        ctx.ui.setStatus("rlcd", `🧭 Override: ${profileName}`);
        ctx.ui.notify(`Model override set: ${profileName}`, "info");
      }
    },
  });

  // ── Command: /model-override — force a specific model ────────

  pi.registerCommand("model-override", {
    description:
      "Force a specific model in the next /route decision. Use without args to see available models, or 'clear' to reset.",
    handler: async (args, ctx) => {
      // Clear override
      if (args === "clear" || args === "reset" || args === "auto") {
        modelOverride = null;
        ctx.ui.setStatus("rlcd", "🧭 Router online (auto)");
        ctx.ui.notify(
          "Model override cleared. Router will auto-select.",
          "info",
        );
        return;
      }

      // No args → show available models and let user pick
      if (!args) {
        const catalog = await fetchOverrideModels();
        if (!catalog) {
          ctx.ui.notify(
            "Could not fetch models from router. Is it running?",
            "error",
          );
          return;
        }

        const options: string[] = [];
        for (const billing of catalog.billing_order) {
          const group = catalog.models[billing];
          if (!group) continue;
          for (const m of group) {
            const label = `${m.profile_name} (${m.provider}/${m.model_id}) [${billing}]`;
            options.push(label);
          }
        }

        if (options.length === 0) {
          ctx.ui.notify("No executor models available.", "error");
          return;
        }

        const choice = await ctx.ui.select(
          "Pick a model to force for routing:",
          options,
        );
        if (!choice) return;

        // Extract profile name (everything before the first space/paren)
        const profileName = choice.split(" ")[0];
        modelOverride = profileName;
        ctx.ui.setStatus("rlcd", `🧭 Override: ${profileName}`);
        ctx.ui.notify(`Model override set: ${profileName}`, "info");
        return;
      }

      // Direct profile name
      modelOverride = args.trim();
      ctx.ui.setStatus("rlcd", `🧭 Override: ${modelOverride}`);
      ctx.ui.notify(`Model override set: ${modelOverride}`, "info");
    },
  });
}
