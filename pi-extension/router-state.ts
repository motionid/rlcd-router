/**
 * Pure state transitions for the turn-aware router PI extension.
 *
 * This module is deliberately dependency-free: it imports no PI runtime
 * modules and no server code, so it can be exercised directly by
 * `node --test tests/test_pi_extension_state.mjs`.  The PI extension
 * (`pi-extension/turn-router.ts`) composes these transitions inside
 * Pi's documented lifecycle hooks.
 *
 * State shape (versioned; persisted via pi.appendEntry):
 *   - bounded previous prompt / capability / model / route reasons
 *     (the compact turn context the server's /api/route consumes)
 *   - task depth and task failures (escalation counters)
 *   - the active route: ordered candidate profile ids, a cursor, a
 *     per-route attempt counter, and its maximum
 *   - the manual override model (provider/id), created only by user
 *     model selection and cleared only by /router-auto
 */

// ---------------------------------------------------------------------------
// Versioning / persistence identity
// ---------------------------------------------------------------------------

/** Current state schema version.  Incompatible entries are not restored. */
export const ROUTER_STATE_VERSION = 1;

/** The pi.appendEntry customType identifying this extension's entries. */
export const ROUTER_STATE_ENTRY = "rlcd-turn-router-state";

/** Default per-route attempt limit (bounded fallback loop). */
export const DEFAULT_MAX_ATTEMPTS = 3;

/** Bound on the persisted previous prompt (chars). */
export const PREVIOUS_PROMPT_LIMIT = 4096;

/** Bound on a persisted model reference "provider/id" (chars). */
export const MODEL_REF_LIMIT = 256;

/** Bound on persisted route reason codes. */
export const REASON_LIMIT = 128;
export const REASON_COUNT_LIMIT = 16;

/** Bounds for the server's TurnContextInput contract. */
export const TASK_DEPTH_LIMIT = 10_000;
export const TASK_FAILURES_LIMIT = 1_000;

/** Model selection sources, mirroring PI's ModelSelectEvent. */
export type ModelSelectionSource = "set" | "cycle" | "restore";

/** Agent activity outcomes, mirroring PI's AgentActivityOutcome. */
export type ActivityOutcome = "completed" | "aborted" | "error";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

export interface RouteProgress {
  routeId: string;
  /** Ordered candidate profile ids from the server's decision. */
  candidates: string[];
  /** Index of the active (or next-to-try) candidate. */
  cursor: number;
  /** Candidates consumed so far on this route. */
  attempts: number;
  /** Per-route attempt limit: continuation is refused at >= maxAttempts. */
  maxAttempts: number;
}

export interface RouterSessionState {
  version: number;
  /** Bounded previous prompt (never the full transcript). */
  previousPrompt: string;
  previousCapability: string | null;
  /** Last known active model as "provider/id". */
  activeModel: string;
  previousRouteReason: string[];
  taskDepth: number;
  /** Consecutive task failures (next-turn escalation counter). */
  taskFailures: number;
  /** Manual override: active until /router-auto clears it. */
  overrideModel: string | null;
  /** The in-flight route this turn, or null when idle. */
  route: RouteProgress | null;
}

/** Minimal structural shape of a PI session entry (no runtime import). */
export interface BranchEntryLike {
  type: string;
  customType?: string;
  data?: unknown;
}

// ---------------------------------------------------------------------------
// Sanitization helpers (every restore path is untrusted input)
// ---------------------------------------------------------------------------

function boundString(value: unknown, limit: number): string {
  if (typeof value !== "string") return "";
  return value.length > limit ? value.slice(0, limit) : value;
}

function boundCount(value: unknown, min: number, max: number): number {
  const n =
    typeof value === "number" && Number.isFinite(value) ? value : min;
  return Math.min(Math.max(Math.round(n), min), max);
}

function boundModelRef(value: unknown): string | null {
  if (typeof value !== "string" || value.length === 0) return null;
  return boundString(value, MODEL_REF_LIMIT);
}

function sanitizeRoute(raw: unknown): RouteProgress | null {
  if (raw === null || typeof raw !== "object") return null;
  const r = raw as Record<string, unknown>;
  if (typeof r.routeId !== "string" || !Array.isArray(r.candidates)) {
    return null;
  }
  const candidates = (r.candidates as unknown[])
    .map((c) => boundString(c, 64))
    .filter((c) => c.length > 0);
  if (candidates.length === 0) return null;
  return {
    routeId: boundString(r.routeId, 64),
    candidates,
    cursor: boundCount(r.cursor, 0, candidates.length - 1),
    attempts: boundCount(r.attempts, 1, TASK_FAILURES_LIMIT),
    maxAttempts: boundCount(r.maxAttempts, 1, 16),
  };
}

function sanitizeState(raw: unknown): RouterSessionState {
  const r =
    raw !== null && typeof raw === "object"
      ? (raw as Record<string, unknown>)
      : {};
  return {
    version: ROUTER_STATE_VERSION,
    previousPrompt: boundString(r.previousPrompt, PREVIOUS_PROMPT_LIMIT),
    previousCapability:
      typeof r.previousCapability === "string"
        ? boundString(r.previousCapability, 16)
        : null,
    activeModel: boundString(r.activeModel, MODEL_REF_LIMIT),
    previousRouteReason: Array.isArray(r.previousRouteReason)
      ? (r.previousRouteReason as unknown[])
          .slice(0, REASON_COUNT_LIMIT)
          .map((reason) => boundString(reason, REASON_LIMIT))
          .filter((reason) => reason.length > 0)
      : [],
    taskDepth: boundCount(r.taskDepth, 0, TASK_DEPTH_LIMIT),
    taskFailures: boundCount(r.taskFailures, 0, TASK_FAILURES_LIMIT),
    overrideModel: boundModelRef(r.overrideModel),
    route: sanitizeRoute(r.route),
  };
}

// ---------------------------------------------------------------------------
// Creation / restoration
// ---------------------------------------------------------------------------

export function createRouterState(
  init: Partial<RouterSessionState> = {},
): RouterSessionState {
  return sanitizeState({
    ...init,
    version: ROUTER_STATE_VERSION,
  });
}

/**
 * Reconstruct state from the latest *compatible* active-branch router
 * entry.  `entries` must be the active branch (PI's
 * `ctx.sessionManager.getBranch()`): abandoned branches are alternative
 * histories and are never consulted.  Scanning from the newest entry
 * backwards, the first entry with our customType and a compatible
 * version wins; anything else is skipped.  A restored route is never
 * continued (server route ids are process-ephemeral).
 */
export function restoreFromBranch(
  entries: readonly BranchEntryLike[],
): RouterSessionState | null {
  for (let i = entries.length - 1; i >= 0; i--) {
    const entry = entries[i];
    if (
      !entry ||
      entry.type !== "custom" ||
      entry.customType !== ROUTER_STATE_ENTRY
    ) {
      continue;
    }
    const data = entry.data;
    if (
      data === null ||
      typeof data !== "object" ||
      (data as Record<string, unknown>).version !== ROUTER_STATE_VERSION
    ) {
      continue; // incompatible or corrupt: skip, do not restore
    }
    const state = sanitizeState(data);
    state.route = null; // an ephemeral server route never survives a reload
    return state;
  }
  return null;
}

// ---------------------------------------------------------------------------
// Model selection (override semantics)
// ---------------------------------------------------------------------------

/**
 * Apply a model_select event to the state.
 *
 * - "set"/"cycle" (user /model, model cycling): a persistent override,
 *   respected even when no router profile maps the model, until
 *   /router-auto clears it.  Any in-flight route is dropped: the user
 *   owns the turn now.
 * - "restore" (session restore): adopt the restored model as active but
 *   never create an override — the persisted override field is the only
 *   source of override truth across reloads.
 */
export function withModelSelection(
  state: RouterSessionState,
  model: string,
  source: ModelSelectionSource,
): RouterSessionState {
  const ref = boundModelRef(model) ?? state.activeModel;
  if (source === "restore") {
    return { ...state, activeModel: ref, route: null };
  }
  return {
    ...state,
    activeModel: ref,
    overrideModel: ref,
    route: null,
  };
}

/**
 * Apply an *automatic* model switch (candidate activation guarded by
 * the extension around pi.setModel).  Automatic switches never create
 * or consume an override and never touch the in-flight route.
 */
export function withAutomaticSelection(
  state: RouterSessionState,
  model: string,
): RouterSessionState {
  return { ...state, activeModel: boundModelRef(model) ?? state.activeModel };
}

/** /router-auto: clear the manual override and resume automatic routing. */
export function clearOverride(state: RouterSessionState): RouterSessionState {
  return { ...state, overrideModel: null };
}

// ---------------------------------------------------------------------------
// Routing / candidate advancement
// ---------------------------------------------------------------------------

export interface BeginRouteInput {
  routeId: string;
  /** Ordered candidate profile ids from the server's decision. */
  candidates: readonly string[];
  /** Index of the candidate that was actually activated. */
  activeIndex: number;
  prompt: string;
  capability: string | null;
  reasons: readonly string[];
  maxAttempts?: number;
}

/**
 * Record a fresh route: the prompt becomes the bounded previous prompt
 * for the next turn, capability/reasons feed the next turn context, and
 * the candidate cursor starts at the activated candidate with one
 * attempt consumed.
 */
export function beginRoute(
  state: RouterSessionState,
  input: BeginRouteInput,
): RouterSessionState {
  const candidates = input.candidates
    .map((c) => boundString(c, 64))
    .filter((c) => c.length > 0);
  const activeIndex = Math.min(
    Math.max(Math.round(input.activeIndex) || 0, 0),
    Math.max(candidates.length - 1, 0),
  );
  return {
    ...state,
    previousPrompt: boundString(input.prompt, PREVIOUS_PROMPT_LIMIT),
    previousCapability:
      typeof input.capability === "string"
        ? boundString(input.capability, 16)
        : null,
    previousRouteReason: input.reasons
      .slice(0, REASON_COUNT_LIMIT)
      .map((r) => boundString(r, REASON_LIMIT))
      .filter((r) => r.length > 0),
    taskDepth: Math.min(state.taskDepth + 1, TASK_DEPTH_LIMIT),
    route:
      candidates.length > 0
        ? {
            routeId: boundString(input.routeId, 64),
            candidates,
            cursor: activeIndex,
            attempts: 1,
            maxAttempts: input.maxAttempts ?? DEFAULT_MAX_ATTEMPTS,
          }
        : null,
  };
}

/**
 * Advance to the next candidate *without reclassification*: the ordered
 * candidate list is fixed for the route; only the cursor and the attempt
 * counter move.  Returns the next profile id, or null when the
 * candidates are exhausted.
 */
export function advanceCandidate(state: RouterSessionState): {
  state: RouterSessionState;
  nextProfileId: string | null;
} {
  const route = state.route;
  if (!route) {
    return { state, nextProfileId: null };
  }
  const cursor = route.cursor + 1;
  const attempts = Math.min(route.attempts + 1, TASK_FAILURES_LIMIT);
  return {
    state: { ...state, route: { ...route, cursor, attempts } },
    nextProfileId: cursor < route.candidates.length ? route.candidates[cursor] : null,
  };
}

// ---------------------------------------------------------------------------
// Provider failure classification
// ---------------------------------------------------------------------------

export interface ProviderFailure {
  category: string;
  retryAfter: number | null;
}

function retryAfterFromHeaders(
  headers: Record<string, string> | null | undefined,
): number | null {
  if (!headers) return null;
  for (const [key, value] of Object.entries(headers)) {
    if (key.toLowerCase() !== "retry-after") continue;
    const seconds = Number(value);
    if (Number.isFinite(seconds) && seconds > 0) return seconds;
  }
  return null;
}

/**
 * Classify an explicit provider availability failure from the captured
 * response status and headers.
 *
 * Only a captured HTTP failure status is provider evidence: a missing
 * status (no after_provider_response event) is *not* classified as a
 * provider failure — without explicit availability evidence the
 * failure stays task-side and never penalizes the provider.
 */
export function classifyProviderFailure(
  status: number | null | undefined,
  headers: Record<string, string> | null | undefined,
): ProviderFailure | null {
  if (typeof status !== "number" || status < 400) return null;
  let category: string;
  if (status === 429) category = "rate_limit";
  else if (status === 402) category = "quota";
  else if (status === 401 || status === 403) category = "auth";
  else if (status === 408 || status === 502 || status === 503 || status === 504) {
    category = "capacity";
  } else {
    category = "provider_unknown";
  }
  return { category, retryAfter: retryAfterFromHeaders(headers) };
}

// ---------------------------------------------------------------------------
// Settlement: outcome classification + bounded continuation
// ---------------------------------------------------------------------------

export interface AttemptReport {
  status: "success" | "failed";
  failureCategory: string | null;
  retryAfter: number | null;
}

export interface SettleInput {
  outcome: ActivityOutcome;
  providerStatus?: number | null;
  providerHeaders?: Record<string, string> | null;
}

export interface SettleDecision {
  state: RouterSessionState;
  /** The /api/route/outcome report to send, or null (aborted turns never report). */
  report: AttemptReport | null;
  /** Next candidate profile id, already advanced past the failed one. */
  nextProfileId: string | null;
  /** True only when a next candidate exists and the attempt limit permits. */
  continueTurn: boolean;
  /** True only for explicit provider availability failures. */
  penalizeProvider: boolean;
}

/**
 * Classify a settled turn and decide the bounded fallback.
 *
 * - aborted: no report, no penalty, no advancement — an aborted turn
 *   must never penalize the active model.
 * - completed: success report; the task-failure escalation counter
 *   resets.
 * - error + explicit provider failure (e.g. 429 + Retry-After):
 *   availability-failure report; the cursor advances without
 *   reclassification; continuation is permitted while candidates and
 *   the attempt limit hold.
 * - error without provider evidence: task failure — reported as
 *   task_failure (the provider was available; the work failed), the
 *   escalation counter increments, and no fallback continuation runs.
 */
export function settleAttempt(
  state: RouterSessionState,
  input: SettleInput,
): SettleDecision {
  const route = state.route;
  if (!route) {
    return {
      state,
      report: null,
      nextProfileId: null,
      continueTurn: false,
      penalizeProvider: false,
    };
  }

  if (input.outcome === "aborted") {
    return {
      state: { ...state, route: null },
      report: null,
      nextProfileId: null,
      continueTurn: false,
      penalizeProvider: false,
    };
  }

  if (input.outcome === "completed") {
    return {
      state: { ...state, route: null, taskFailures: 0 },
      report: { status: "success", failureCategory: null, retryAfter: null },
      nextProfileId: null,
      continueTurn: false,
      penalizeProvider: false,
    };
  }

  // outcome === "error"
  const failure = classifyProviderFailure(
    input.providerStatus,
    input.providerHeaders,
  );
  if (failure) {
    const advanced = advanceCandidate(state);
    return {
      state: advanced.state,
      report: {
        status: "failed",
        failureCategory: failure.category,
        retryAfter: failure.retryAfter,
      },
      nextProfileId: advanced.nextProfileId,
      continueTurn:
        advanced.nextProfileId !== null &&
        advanced.state.route !== null &&
        advanced.state.route.attempts < advanced.state.route.maxAttempts,
      penalizeProvider: true,
    };
  }

  return {
    state: {
      ...state,
      route: null,
      taskFailures: Math.min(state.taskFailures + 1, TASK_FAILURES_LIMIT),
    },
    report: { status: "failed", failureCategory: "task_failure", retryAfter: null },
    nextProfileId: null,
    continueTurn: false,
    penalizeProvider: false,
  };
}

// ---------------------------------------------------------------------------
// Router outage / empty candidates
// ---------------------------------------------------------------------------

export interface OutageDecision {
  model: string | null;
  retainedActive: boolean;
}

/**
 * Resolve what to do when the router cannot produce a usable route
 * (network outage, or an empty candidate list with no explicit
 * fallback).  The active model is retained unless an explicitly
 * configured (and available) fallback subscription model was
 * resolved; the turn is never blocked either way.
 */
export function outageDecision(
  state: RouterSessionState,
  fallbackModel: string | null,
): OutageDecision {
  const fallback = boundModelRef(fallbackModel);
  if (fallback) {
    return { model: fallback, retainedActive: false };
  }
  return { model: state.activeModel || null, retainedActive: true };
}

// ---------------------------------------------------------------------------
// Manual override -> router profile mapping (server catalog)
// ---------------------------------------------------------------------------

/** One entry of the server's /api/models/overrides catalog. */
export interface OverrideCatalogEntry {
  profile_name: string;
  provider: string;
  model_id: string;
}

/**
 * Flatten the server's billing-grouped override catalog
 * (`{ billing_order: [...], models: { <billing>: [entries] } }`) into a
 * list of valid entries.  Returns null when the response is unusable;
 * malformed groups or entries are skipped, never guessed from.
 */
export function flattenOverrideCatalog(
  raw: unknown,
): OverrideCatalogEntry[] | null {
  if (raw === null || typeof raw !== "object") return null;
  const models = (raw as Record<string, unknown>).models;
  if (models === null || typeof models !== "object" || Array.isArray(models)) {
    return null;
  }
  const entries: OverrideCatalogEntry[] = [];
  for (const group of Object.values(models)) {
    if (!Array.isArray(group)) continue; // malformed billing group: skip
    for (const item of group) {
      if (item === null || typeof item !== "object") continue;
      const entry = item as Record<string, unknown>;
      if (
        typeof entry.profile_name === "string" && entry.profile_name.length > 0 &&
        typeof entry.provider === "string" && entry.provider.length > 0 &&
        typeof entry.model_id === "string" && entry.model_id.length > 0
      ) {
        entries.push({
          profile_name: boundString(entry.profile_name, 64),
          provider: boundString(entry.provider, 64),
          model_id: boundString(entry.model_id, MODEL_REF_LIMIT),
        });
      }
    }
  }
  return entries;
}

/**
 * Map a manual override ("provider/model_id") to its router profile name
 * by an *exact* provider and model_id match against the catalog.
 *
 * Never guesses: fuzzy, prefix, basename, or wrong-provider matches all
 * return null, as do missing catalogs and malformed references — every
 * null means "retain the override untouched, do not route".  Disabled
 * profiles never appear in the catalog (the server excludes them), so
 * they can never map; uncertainty never selects a paid profile because
 * uncertainty never routes at all.
 */
export function resolveOverrideProfile(
  overrideModel: string,
  entries: readonly OverrideCatalogEntry[] | null | undefined,
): string | null {
  if (typeof overrideModel !== "string" || !Array.isArray(entries)) {
    return null;
  }
  const sep = overrideModel.indexOf("/");
  if (sep <= 0 || sep === overrideModel.length - 1) return null;
  const provider = overrideModel.slice(0, sep);
  const modelId = overrideModel.slice(sep + 1);
  for (const entry of entries) {
    if (
      entry &&
      typeof entry === "object" &&
      entry.provider === provider &&
      entry.model_id === modelId &&
      typeof entry.profile_name === "string" &&
      entry.profile_name.length > 0
    ) {
      return entry.profile_name;
    }
  }
  return null;
}
