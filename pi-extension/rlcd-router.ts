/**
 * RLCD Router Extension for Pi
 *
 * 1. On session_start: ensures the configured RLCD routing server is running.
 *    If it's already running (from another Pi session), does nothing.
 *    If not, spawns it as a detached background process.
 *
 * 2. Registers a /route command so the user can ask "what model would handle this?"
 *
 * 3. Registers a one-shot /model-override command that changes the next
 *    /route decision (e.g. "/model-override claude_opus_4_6").
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { spawn } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROUTER_PORT = Number(process.env.RLCD_ROUTER_PORT ?? "8001");
const ROUTER_URL = process.env.RLCD_ROUTER_URL ?? `http://127.0.0.1:${ROUTER_PORT}`;
const EXTENSION_DIR = dirname(fileURLToPath(import.meta.url));
const ROUTER_DIR = process.env.RLCD_ROUTER_DIR ?? resolve(EXTENSION_DIR, "..");
const ROUTER_PYTHON = process.env.RLCD_PYTHON ?? resolve(ROUTER_DIR, ".venv/bin/python");

// ─── helpers ───────────────────────────────────────────────────────

async function checkRouter(): Promise<boolean> {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 750);
    const response = await fetch(`${ROUTER_URL}/health`, { signal: controller.signal });
    clearTimeout(timeout);
    if (!response.ok) return false;
    const body = (await response.json()) as { status?: string; service?: string };
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

async function fetchOverrideModels(): Promise<Record<string, any[]> | null> {
  try {
    const resp = await fetch(`${ROUTER_URL}/api/models/overrides`);
    if (!resp.ok) return null;
    const data = (await resp.json()) as { models: Record<string, any[]> };
    return data.models;
  } catch {
    return null;
  }
}

async function fetchFallbackRecommendation(currentModel: string): Promise<any | null> {
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

// ─── extension ─────────────────────────────────────────────────────

export default function (pi: ExtensionAPI) {
  // One-shot override consumed by the next /route command.
  let modelOverride: string | null = null;

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
      ["-m", "uvicorn", "server.app:app", "--host", "127.0.0.1", "--port", String(ROUTER_PORT)],
      {
        cwd: ROUTER_DIR,
        env: { ...process.env, PYTHONPATH: ROUTER_DIR },
        detached: true,
        stdio: "ignore",
      }
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
      context: Type.String({ description: "Text or task that should be classified" }),
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
          body: JSON.stringify({ context: params.context, schema: params.schema }),
        });
        if (!resp.ok) {
          const err = await resp.text();
          return { content: [{ type: "text", text: `RLCD error (${resp.status}): ${err}` }] };
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
          "info"
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
        ctx.ui.notify("Usage: /model-exhausted <profile name or model id>", "info");
        return;
      }

      const recommendation = await fetchFallbackRecommendation(currentModel);
      if (!recommendation) {
        ctx.ui.notify("Could not get a fallback recommendation from the router.", "error");
        return;
      }

      const options: string[] = [];
      if (recommendation.recommended) {
        options.push(
          `Use recommended: ${recommendation.recommended.profile_name} (${recommendation.recommended.model_id})`
        );
      }
      for (const alternative of recommendation.alternatives ?? []) {
        options.push(`Use alternative: ${alternative.profile_name} (${alternative.model_id})`);
      }
      options.push("Choose another model…");

      const choice = await ctx.ui.select(
        recommendation.message ?? "Subscription exhausted. Choose a replacement:",
        options,
      );
      if (!choice) return;

      let profileName: string | null = null;
      if (choice.startsWith("Use recommended: ")) {
        profileName = choice.slice("Use recommended: ".length).split(" ")[0];
      } else if (choice.startsWith("Use alternative: ")) {
        profileName = choice.slice("Use alternative: ".length).split(" ")[0];
      } else {
        const models = await fetchOverrideModels();
        if (!models) {
          ctx.ui.notify("Could not fetch available models from the router.", "error");
          return;
        }
        const allOptions: string[] = [];
        for (const billing of ["local", "subscription_unlimited", "subscription_quota", "paid"]) {
          for (const model of models[billing] ?? []) {
            allOptions.push(`${model.profile_name} (${model.provider}/${model.model_id}) [${billing}]`);
          }
        }
        const selected = await ctx.ui.select("Choose a different model:", allOptions);
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
        ctx.ui.notify("Model override cleared. Router will auto-select.", "info");
        return;
      }

      // No args → show available models and let user pick
      if (!args) {
        const models = await fetchOverrideModels();
        if (!models) {
          ctx.ui.notify("Could not fetch models from router. Is it running?", "error");
          return;
        }

        const options: string[] = [];
        const order = ["local", "subscription_unlimited", "subscription_quota", "paid"];
        for (const billing of order) {
          const group = models[billing];
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
          options
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
