import Mime from "@effect/platform-node/Mime";
import {
  AuthOrchestrationOperateScope,
  AuthOrchestrationReadScope,
  EnvironmentHttpApi,
} from "@t3tools/contracts";
import { isDevProxiedPath } from "@t3tools/shared/devProxy";
import { decodeOtlpTraceRecords } from "@t3tools/shared/observability";
import * as Data from "effect/Data";
import * as Effect from "effect/Effect";
import * as FileSystem from "effect/FileSystem";
import * as Layer from "effect/Layer";
import * as Option from "effect/Option";
import * as Path from "effect/Path";
import * as Result from "effect/Result";
import * as Schedule from "effect/Schedule";
import * as Stream from "effect/Stream";
import { cast } from "effect/Function";
import {
  HttpBody,
  HttpClient,
  HttpClientResponse,
  HttpMiddleware,
  HttpRouter,
  HttpServerResponse,
  HttpServerRequest,
  HttpServerRespondable,
} from "effect/unstable/http";
import * as HttpApiBuilder from "effect/unstable/httpapi/HttpApiBuilder";
import { OtlpTracer } from "effect/unstable/observability";

import * as ServerConfig from "./config.ts";
import { ASSET_ROUTE_PREFIX, resolveAsset } from "./assets/AssetAccess.ts";
import * as BrowserTraceCollector from "./observability/BrowserTraceCollector.ts";
import * as EnvironmentAuth from "./auth/EnvironmentAuth.ts";
import { traceRelayRequest } from "./cloud/traceRelayRequest.ts";
import {
  annotateEnvironmentRequest,
  failEnvironmentScopeRequired,
  failEnvironmentAuthInvalid,
  failEnvironmentInternal,
} from "./auth/http.ts";
import * as ServerEnvironment from "./environment/ServerEnvironment.ts";
import { browserApiCorsAllowedHeaders, browserApiCorsAllowedMethods } from "./httpCors.ts";

const OTLP_TRACES_PROXY_PATH = "/api/observability/v1/traces";
const LOOPBACK_HOSTNAMES = new Set(["127.0.0.1", "::1", "localhost"]);
const DESKTOP_RENDERER_ORIGINS = ["t3code://app", "t3code-dev://app"];
const SVG_CONTENT_SECURITY_POLICY = "default-src 'none'; style-src 'unsafe-inline'; sandbox";
const LM_TOOLS_CAMPAIGN_ROUTE = "/api/lm-tools/campaign";
const LM_TOOLS_CAMPAIGN_EVENTS_ROUTE = "/api/lm-tools/campaign/events";

type JsonRecord = Record<string, unknown>;
type LmToolsObserverSnapshot = {
  configured: boolean;
  observer?: { kind: string; title: string };
  campaign: unknown;
  current: unknown;
  summaries?: unknown;
  definition?: unknown;
  controller?: unknown;
  evaluations?: unknown[];
  events: unknown[];
  stream?: { lastEventAt: string | null; bytes: number; lastEventMethod: string | null };
  inference?: unknown;
};

function jsonRecord(value: unknown): JsonRecord | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as JsonRecord)
    : null;
}

function redactCommand(value: string): string {
  const secretName = "(?:password|passwd|token|api[_-]?key|secret)";
  return value
    .replace(
      new RegExp(`(["']${secretName}["']\\s*:\\s*)["'][^"']*["']`, "gi"),
      '$1"***"',
    )
    .replace(
      new RegExp(`(\\b${secretName}\\s*=\\s*)(?:"[^"]*"|'[^']*'|[^\\s;]+)`, "gi"),
      "$1***",
    )
    .replace(/(authorization\s*:\s*bearer\s+)[^\s"']+/gi, "$1***")
    .replace(/(https?:\/\/[^\s/:@]+:)[^\s/@]+@/gi, "$1***@");
}

function readOptionalJson(fileSystem: FileSystem.FileSystem, path: string) {
  return fileSystem.readFileString(path).pipe(
    Effect.map((contents) => JSON.parse(contents) as unknown),
    Effect.orElseSucceed(() => null),
  );
}

function readEvaluationHistory(fileSystem: FileSystem.FileSystem) {
  const directory = process.env.LM_TOOLS_CAMPAIGN_CANDIDATES_DIR;
  if (!directory) return Effect.succeed([] as unknown[]);
  return fileSystem.readDirectory(directory, { recursive: false }).pipe(
    Effect.flatMap((entries) =>
      Effect.all(
        entries.map((entry) =>
          readOptionalJson(fileSystem, `${directory}/${entry}/results/evaluation.json`),
        ),
        { concurrency: 8 },
      ),
    ),
    Effect.map((entries) =>
      entries
        .filter((entry) => entry !== null)
        .sort((left, right) => {
          const timestamp = (value: unknown) =>
            value !== null &&
            typeof value === "object" &&
            !Array.isArray(value) &&
            "startedAt" in value &&
            typeof value.startedAt === "string"
              ? Date.parse(value.startedAt)
              : 0;
          return timestamp(left) - timestamp(right);
        })
        .slice(-100),
    ),
    Effect.orElseSucceed(() => [] as unknown[]),
  );
}

/**
 * Reads authoritative inference state from lm-tools proxies.
 *
 * Two proxies front the same llama.cpp process and each holds half the answer:
 * the stack that spawned llama owns its stderr and therefore the cumulative
 * prompt-eval fraction, while the per-environment proxy sees the billed usage
 * for that environment's own traffic. Merge whichever fields each reports.
 * Falls back to `/slots` when no proxy answers.
 */
function readProxyInference(httpClient: HttpClient.HttpClient) {
  return Effect.gen(function* () {
    const configured = (process.env.LM_TOOLS_INFERENCE_URL ?? "")
      .split(",")
      .map((entry) => entry.trim())
      .filter((entry) => entry.length > 0);
    if (configured.length === 0) return null;
    const snapshots: Record<string, unknown>[] = [];
    for (const base of configured) {
      const snapshot = yield* httpClient
        .get(new URL("/lm-tools/inference", base).toString())
        .pipe(
          Effect.flatMap(HttpClientResponse.filterStatusOk),
          Effect.flatMap((response) => response.json),
          Effect.timeout("750 millis"),
          Effect.orElseSucceed(() => null as unknown),
        );
      if (snapshot !== null && typeof snapshot === "object" && !Array.isArray(snapshot)) {
        snapshots.push(snapshot as Record<string, unknown>);
      }
    }
    if (snapshots.length === 0) return null;
    // Prompt evaluation is reported by whichever proxy owns llama's stderr.
    const prompting = snapshots.find((entry) => entry.phase === "prompt");
    // Context usage comes from the proxy that billed this environment's turn.
    // A proxy that has not yet seen a completed turn reports used=0; that is an
    // absence of data, not a measurement, so let llama's live slot fill it in.
    const withContext = snapshots.find((entry) => {
      const value = entry.context;
      if (value === null || typeof value !== "object") return false;
      const used = (value as Record<string, unknown>).used;
      return typeof used === "number" && used > 0;
    });
    const generating = snapshots.find((entry) => entry.phase === "generation");
    const first = snapshots[0];
    if (first === undefined) return null;
    const primary = prompting ?? generating ?? withContext ?? first;
    // The renderer reads these fields unconditionally, so never hand it a
    // partially shaped payload: a missing counter must render as zero, not
    // crash the observer.
    const num = (value: unknown): number =>
      typeof value === "number" && Number.isFinite(value) ? value : 0;
    const promptSource = (prompting ?? primary).prompt;
    const prompt = promptSource !== null && typeof promptSource === "object"
      ? (promptSource as Record<string, unknown>)
      : {};
    const contextSource = withContext?.context;
    const context = contextSource !== null && typeof contextSource === "object"
      ? (contextSource as Record<string, unknown>)
      : null;
    return {
      ...primary,
      active: snapshots.some((entry) => entry.active === true),
      phase: prompting ? "prompt" : generating ? "generation" : "idle",
      taskId: primary.taskId ?? null,
      prompt: {
        processed: num(prompt.processed),
        total: num(prompt.total),
        cached: num(prompt.cached),
        percent: num(prompt.percent),
      },
      context: context
        ? {
          used: num(context.used),
          limit: num(context.limit),
          percent: num(context.percent),
        }
        : null,
    };
  }).pipe(Effect.catchCause(() => Effect.succeed(null)));
}

/**
 * Coerces a merged snapshot into exactly the shape the observer renders.
 *
 * The renderer reads every field unconditionally (`context.limit`,
 * `generatedTokens.toLocaleString()`, ...), so a missing key is a crash, not a
 * blank. Normalizing on the single return path keeps that impossible.
 */
function normalizeInference(value: Record<string, unknown> | null): {
  active: boolean;
  phase: "prompt" | "generation" | "idle";
  taskId: number | null;
  context: { used: number; limit: number; percent: number };
  prompt: { processed: number; total: number; cached: number; percent: number };
  generatedTokens: number;
} | null {
  if (value === null) return null;
  const num = (input: unknown): number =>
    typeof input === "number" && Number.isFinite(input) ? input : 0;
  const record = (input: unknown): Record<string, unknown> =>
    input !== null && typeof input === "object" && !Array.isArray(input)
      ? (input as Record<string, unknown>)
      : {};
  const prompt = record(value.prompt);
  const context = record(value.context);
  const phase = value.phase === "prompt" || value.phase === "generation" ? value.phase : "idle";
  return {
    active: value.active === true,
    phase,
    taskId: typeof value.taskId === "number" ? value.taskId : null,
    context: {
      used: num(context.used),
      limit: num(context.limit),
      percent: num(context.percent),
    },
    prompt: {
      processed: num(prompt.processed),
      total: num(prompt.total),
      cached: num(prompt.cached),
      percent: num(prompt.percent),
    },
    generatedTokens: num(value.generatedTokens),
  };
}

function readModelInference(httpClient: HttpClient.HttpClient) {
  return Effect.gen(function* () {
    const fromProxy = yield* readProxyInference(httpClient);
    // The proxies hold the authoritative numbers but only after a turn has run.
    // llama's slot is always live, so it backfills whatever they cannot answer
    // yet -- notably context usage across a proxy restart.
    const fromSlots = yield* readSlotInference(httpClient);
    if (fromProxy === null) return normalizeInference(fromSlots);
    // The stderr scraper latches the last progress line and is only reset by
    // traffic through its own proxy, so a per-environment proxy's turns leave it
    // reporting a finished evaluation forever. llama's slot is the liveness
    // authority: if nothing is evaluating there, nothing is evaluating.
    const slotIsPrompting = fromSlots?.phase === "prompt";
    const merged = fromProxy.phase === "prompt" && !slotIsPrompting
      ? {
        ...fromProxy,
        active: fromSlots?.active ?? false,
        phase: fromSlots?.phase ?? "idle",
        prompt: fromSlots?.prompt ?? fromProxy.prompt,
      }
      : fromProxy;
    // Slot state backfills anything the proxies cannot answer yet: context usage
    // before a turn completes, and generated-token counts, which only llama has.
    // Cache hits are known to llama immediately but only reach the proxy with
    // end-of-turn usage, so prefer the live slot while a turn is in flight.
    const mergedPrompt = merged.prompt as Record<string, unknown> | undefined;
    const slotCached = fromSlots?.prompt?.cached ?? 0;
    const withGenerated = {
      generatedTokens: fromSlots?.generatedTokens ?? 0,
      ...merged,
      ...(mergedPrompt && !(typeof mergedPrompt.cached === "number" && mergedPrompt.cached > 0) &&
          slotCached > 0
        ? { prompt: { ...mergedPrompt, cached: slotCached } }
        : {}),
    };
    if (merged.context !== null || fromSlots === null) return normalizeInference(withGenerated);
    return normalizeInference({ ...withGenerated, context: fromSlots.context });
  }).pipe(Effect.catchCause(() => Effect.succeed(null)));
}

function readSlotInference(httpClient: HttpClient.HttpClient) {
  return Effect.gen(function* () {
    const modelUrl = process.env.LM_TOOLS_MODEL_URL;
    if (!modelUrl) return null;
    const slots = yield* httpClient.get(new URL("/slots", modelUrl).toString()).pipe(
      Effect.flatMap(HttpClientResponse.filterStatusOk),
      Effect.flatMap((response) => response.json),
      Effect.timeout("750 millis"),
    );
    if (!Array.isArray(slots)) return null;
    const candidates = slots.filter(
      (slot): slot is Record<string, unknown> =>
        slot !== null && typeof slot === "object" && !Array.isArray(slot),
    );
    const slot = candidates.find((candidate) => candidate.is_processing === true) ?? candidates[0];
    if (!slot) return null;
    const nextTokens = Array.isArray(slot.next_token) ? slot.next_token : [];
    const nextToken = nextTokens.find(
      (candidate): candidate is Record<string, unknown> =>
        candidate !== null && typeof candidate === "object" && !Array.isArray(candidate),
    );
    const contextLimit = typeof slot.n_ctx === "number" ? slot.n_ctx : 0;
    const promptTokens = typeof slot.n_prompt_tokens === "number" ? slot.n_prompt_tokens : 0;
    const cachedPromptTokens =
      typeof slot.n_prompt_tokens_cache === "number" ? slot.n_prompt_tokens_cache : 0;
    const processedPromptTokens =
      typeof slot.n_prompt_tokens_processed === "number" ? slot.n_prompt_tokens_processed : 0;
    const generatedTokens = typeof nextToken?.n_decoded === "number" ? nextToken.n_decoded : 0;
    // llama.cpp folds decoded output back into n_prompt_tokens while generating.
    // Subtract n_decoded to recover the actual uncached prompt work.
    const uncachedPromptTokens = Math.max(0, promptTokens - cachedPromptTokens - generatedTokens);
    const active = slot.is_processing === true;
    // n_prompt_tokens and n_prompt_tokens_processed are not an atomic snapshot:
    // llama.cpp updates them while decoded output is folded back into the slot.
    // Once this task has decoded a token it cannot return to prompt evaluation,
    // so n_decoded is the stable phase boundary.
    const phase = active ? (generatedTokens > 0 ? "generation" : "prompt") : "idle";
    const contextUsed = Math.min(contextLimit, promptTokens);
    return {
      active,
      phase,
      taskId: typeof slot.id_task === "number" ? slot.id_task : null,
      context: {
        used: contextUsed,
        limit: contextLimit,
        percent: contextLimit > 0 ? contextUsed / contextLimit : 0,
      },
      prompt: {
        processed: Math.min(processedPromptTokens, uncachedPromptTokens),
        total: uncachedPromptTokens,
        cached: cachedPromptTokens,
        percent:
          uncachedPromptTokens > 0 ? Math.min(1, processedPromptTokens / uncachedPromptTokens) : 1,
      },
      generatedTokens,
    };
  }).pipe(Effect.catchCause(() => Effect.succeed(null)));
}

const LM_TOOLS_DELTA_METHODS = new Set([
  "item/agentMessage/delta",
  "item/reasoning/summaryTextDelta",
  "item/reasoning/textDelta",
  "item/commandExecution/outputDelta",
]);

function compactCampaignEvents(events: unknown[]): unknown[] {
  const compacted: unknown[] = [];
  const deltas = new Map<string, number>();
  for (const candidate of events) {
    if (candidate === null || typeof candidate !== "object" || Array.isArray(candidate)) {
      compacted.push(candidate);
      continue;
    }
    const event = candidate as Record<string, unknown>;
    const params =
      event.params !== null && typeof event.params === "object" && !Array.isArray(event.params)
        ? (event.params as Record<string, unknown>)
        : null;
    const method = typeof event.method === "string" ? event.method : "";
    const itemId = typeof params?.itemId === "string" ? params.itemId : null;
    const delta = typeof params?.delta === "string" ? params.delta : null;
    if (!LM_TOOLS_DELTA_METHODS.has(method) || !itemId || delta === null) {
      compacted.push(candidate);
      continue;
    }
    const key = `${method}:${itemId}`;
    const existing = deltas.get(key);
    if (existing === undefined) {
      deltas.set(key, compacted.length);
      compacted.push(candidate);
      continue;
    }
    const previous = compacted[existing] as Record<string, unknown>;
    const previousParams = previous.params as Record<string, unknown>;
    compacted[existing] = {
      ...previous,
      params: { ...previousParams, delta: String(previousParams.delta ?? "") + delta },
    };
  }
  return compacted.slice(-2000);
}

function readEventTail(fileSystem: FileSystem.FileSystem, path: string | undefined) {
  if (!path) {
    return Effect.succeed({
      events: [] as unknown[],
      lastEventAt: null,
      bytes: 0,
      lastEventMethod: null,
    });
  }
  return Effect.scoped(
    Effect.gen(function* () {
      const file = yield* fileSystem.open(path, { flag: "r" });
      const stat = yield* file.stat;
      const maximumLength = 16n * 1024n * 1024n;
      const length = stat.size < maximumLength ? stat.size : maximumLength;
      yield* file.seek(stat.size - length, "start");
      const contents = yield* file.readAlloc(length);
      const bytes = Option.getOrElse(contents, () => new Uint8Array());
      const lines = new TextDecoder().decode(bytes).split("\n");
      if (stat.size > length) lines.shift();
      const rawEvents = lines.flatMap((line) => {
        try {
          return [JSON.parse(line) as unknown];
        } catch {
          return [];
        }
      });
      const lastEvent = rawEvents.at(-1);
      const lastEventMethod =
        lastEvent !== null &&
        typeof lastEvent === "object" &&
        !Array.isArray(lastEvent) &&
        typeof (lastEvent as Record<string, unknown>).method === "string"
          ? ((lastEvent as Record<string, unknown>).method as string)
          : null;
      const events = compactCampaignEvents(rawEvents);
      return {
        events,
        lastEventAt: Option.match(stat.mtime, {
          onNone: () => null,
          onSome: (modified) => modified.toISOString(),
        }),
        bytes: Number(stat.size),
        lastEventMethod,
      };
    }),
  ).pipe(
    Effect.orElseSucceed(() => ({
      events: [] as unknown[],
      lastEventAt: null,
      bytes: 0,
      lastEventMethod: null,
    })),
  );
}

function readProxyTraceTail(fileSystem: FileSystem.FileSystem) {
  return readEventTail(fileSystem, process.env.LM_TOOLS_PROXY_TRACE).pipe(
    Effect.map((tail) => {
      const events = tail.events.flatMap((candidate) => {
        const record = jsonRecord(candidate);
        const event = jsonRecord(record?.event);
        if (!event) return [];
        const item = jsonRecord(event.item);
        return [{
          ...event,
          ...(item && typeof item.command === "string"
            ? { item: { ...item, command: redactCommand(item.command) } }
            : {}),
        }];
      });
      const lastEvent = events.at(-1);
      return {
        ...tail,
        events: compactCampaignEvents(events),
        lastEventMethod: typeof jsonRecord(lastEvent)?.method === "string"
          ? jsonRecord(lastEvent)?.method as string
          : null,
      };
    }),
  );
}

/**
 * Agent-agnostic observer: everything rendered here already crosses the model
 * proxy, so the snapshot is built from the proxy trace plus live inference
 * state. No per-agent gateway API, and nothing to reimplement when the agent
 * on top changes.
 */
function readProxySnapshot(
  fileSystem: FileSystem.FileSystem,
  httpClient: HttpClient.HttpClient,
) {
  return Effect.gen(function* () {
    const title = process.env.LM_TOOLS_OBSERVER_TITLE ?? "Agent";
    const [inference, proxyTrace] = yield* Effect.all([
      readModelInference(httpClient),
      readProxyTraceTail(fileSystem),
    ]);
    const active = inference?.active === true;
    return {
      configured: true,
      observer: { kind: "proxy", title },
      campaign: null,
      current: {
        id: title,
        phase: "conversation",
        status: active ? "running" : "idle",
      },
      events: proxyTrace.events,
      stream: {
        lastEventAt: proxyTrace.lastEventAt,
        bytes: proxyTrace.bytes,
        lastEventMethod: proxyTrace.lastEventMethod,
      },
      inference,
    };
  }).pipe(
    Effect.catchCause(() =>
      Effect.succeed({
        configured: false,
        observer: { kind: "proxy", title: process.env.LM_TOOLS_OBSERVER_TITLE ?? "Agent" },
        campaign: null,
        current: null,
        events: [],
      })
    ),
  );
}

function readCampaignSnapshot(
  fileSystem: FileSystem.FileSystem,
  httpClient: HttpClient.HttpClient,
): Effect.Effect<LmToolsObserverSnapshot> {

  if (process.env.LM_TOOLS_OBSERVER_SOURCE === "proxy") {
    return readProxySnapshot(fileSystem, httpClient).pipe(
      Effect.map((snapshot) => snapshot as LmToolsObserverSnapshot),
    );
  }
  return Effect.gen(function* () {
    const directory = process.env.LM_TOOLS_CAMPAIGN_DIR;
    if (!directory) {
      return {
        configured: false,
        campaign: null,
        current: null,
        events: [],
      };
    }
    const [campaign, current, summaries, inference, definition, controller, evaluations] =
      yield* Effect.all([
        readOptionalJson(fileSystem, `${directory}/campaign.json`),
        readOptionalJson(fileSystem, `${directory}/current.json`),
        readOptionalJson(fileSystem, `${directory}/summaries.json`),
        readModelInference(httpClient),
        process.env.LM_TOOLS_ENVIRONMENT_CONFIG
          ? readOptionalJson(fileSystem, process.env.LM_TOOLS_ENVIRONMENT_CONFIG)
          : Effect.succeed(null),
        process.env.LM_TOOLS_CAMPAIGN_JOURNAL
          ? readOptionalJson(fileSystem, process.env.LM_TOOLS_CAMPAIGN_JOURNAL)
          : Effect.succeed(null),
        readEvaluationHistory(fileSystem),
      ]);
    const eventLog =
      current &&
      typeof current === "object" &&
      "eventLog" in current &&
      typeof current.eventLog === "string"
        ? current.eventLog
        : undefined;
    const tail = yield* readEventTail(fileSystem, eventLog);
    return {
      configured: true,
      observer: { kind: "campaign", title: "Qwen tinygrad optimization campaign" },
      campaign,
      current,
      summaries,
      definition,
      controller,
      evaluations,
      events: tail.events,
      stream: {
        lastEventAt: tail.lastEventAt,
        bytes: tail.bytes,
        lastEventMethod: tail.lastEventMethod,
      },
      inference,
    };
  }).pipe(Effect.map((snapshot) => snapshot as LmToolsObserverSnapshot));
}

export const lmToolsCampaignRouteLayer = HttpRouter.add(
  "GET",
  LM_TOOLS_CAMPAIGN_ROUTE,
  Effect.gen(function* () {
    const fileSystem = yield* FileSystem.FileSystem;
    const httpClient = yield* HttpClient.HttpClient;
    return HttpServerResponse.jsonUnsafe(yield* readCampaignSnapshot(fileSystem, httpClient));
  }),
);

export const lmToolsCampaignEventsRouteLayer = HttpRouter.add(
  "GET",
  LM_TOOLS_CAMPAIGN_EVENTS_ROUTE,
  Effect.gen(function* () {
    const fileSystem = yield* FileSystem.FileSystem;
    const httpClient = yield* HttpClient.HttpClient;
    let eventLog: string | undefined;
    let offset = 0n;
    let pending = "";
    let decoder = new TextDecoder();

    const encode = (event: string, payload: unknown) =>
      new TextEncoder().encode(`event: ${event}\ndata: ${JSON.stringify(payload)}\n\n`);
    const snapshot = readCampaignSnapshot(fileSystem, httpClient).pipe(
      Effect.map((value) => {
        const current = value.current;
        eventLog =
          current &&
          typeof current === "object" &&
          "eventLog" in current &&
          typeof current.eventLog === "string"
            ? current.eventLog
            : undefined;
        offset = BigInt(value.stream?.bytes ?? 0);
        pending = "";
        decoder = new TextDecoder();
        return encode("snapshot", value);
      }),
    );
    const readAppend = Effect.gen(function* () {
      const directory = process.env.LM_TOOLS_CAMPAIGN_DIR;
      if (!directory) return Option.none<Uint8Array>();
      const current = yield* readOptionalJson(fileSystem, `${directory}/current.json`);
      const nextEventLog =
        current &&
        typeof current === "object" &&
        "eventLog" in current &&
        typeof current.eventLog === "string"
          ? current.eventLog
          : undefined;
      if (nextEventLog !== eventLog) return Option.some(yield* snapshot);
      if (!eventLog) return Option.none<Uint8Array>();

      return yield* Effect.scoped(
        Effect.gen(function* () {
          const file = yield* fileSystem.open(eventLog!, { flag: "r" });
          const stat = yield* file.stat;
          if (stat.size < offset) return Option.some(yield* snapshot);
          if (stat.size === offset) return Option.none<Uint8Array>();
          yield* file.seek(offset, "start");
          const contents = yield* file.readAlloc(stat.size - offset);
          offset = stat.size;
          const bytes = Option.getOrElse(contents, () => new Uint8Array());
          pending += decoder.decode(bytes, { stream: true });
          const lines = pending.split("\n");
          pending = lines.pop() ?? "";
          const events = lines.flatMap((line) => {
            try {
              return [JSON.parse(line) as unknown];
            } catch {
              return [];
            }
          });
          if (events.length === 0) return Option.none<Uint8Array>();
          return Option.some(
            encode("append", {
              events,
              stream: {
                lastEventAt: Option.match(stat.mtime, {
                  onNone: () => null,
                  onSome: (modified) => modified.toISOString(),
                }),
                bytes: Number(stat.size),
                lastEventMethod: (() => {
                  const last = events.at(-1);
                  return last !== null &&
                    typeof last === "object" &&
                    !Array.isArray(last) &&
                    typeof (last as Record<string, unknown>).method === "string"
                    ? (last as Record<string, unknown>).method
                    : null;
                })(),
              },
            }),
          );
        }),
      ).pipe(Effect.orElseSucceed(() => Option.none<Uint8Array>()));
    });

    const snapshots = process.env.LM_TOOLS_OBSERVER_SOURCE === "proxy"
      ? Stream.concat(
        Stream.fromEffect(snapshot),
        Stream.fromSchedule(Schedule.spaced("1 second")).pipe(
          Stream.mapEffect(() => readCampaignSnapshot(fileSystem, httpClient)),
          Stream.map((value) => JSON.stringify(value)),
          Stream.changes,
          Stream.map((value) => encode("snapshot", JSON.parse(value) as unknown)),
        ),
      )
      : Stream.concat(
        Stream.fromEffect(snapshot),
        Stream.fromSchedule(Schedule.spaced("25 millis")).pipe(
          Stream.mapEffect(() => readAppend),
          Stream.filterMap((value) =>
            Option.match(value, {
              onNone: () => Result.failVoid,
              onSome: Result.succeed,
            }),
          ),
        ),
      );
    const inferenceUpdates = Stream.fromSchedule(Schedule.spaced("100 millis")).pipe(
      Stream.mapEffect(() => readModelInference(httpClient)),
      Stream.map((inference) => JSON.stringify(inference)),
      Stream.changes,
      Stream.map((inference) =>
        encode("inference", { inference: JSON.parse(inference) as unknown }),
      ),
    );
    return HttpServerResponse.stream(Stream.merge(snapshots, inferenceUpdates), {
      headers: {
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
        "Content-Type": "text/event-stream",
        "X-Accel-Buffering": "no",
      },
    });
  }),
);

export function assetResponseHeaders(filePath: string): Record<string, string> {
  return {
    "Cache-Control": "private, max-age=3600",
    "X-Content-Type-Options": "nosniff",
    ...(filePath.toLowerCase().endsWith(".svg")
      ? { "Content-Security-Policy": SVG_CONTENT_SECURITY_POLICY }
      : {}),
  };
}

export const httpCompressionLayer = HttpRouter.middleware(HttpMiddleware.compression(), {
  global: true,
});

export const browserApiCorsLayer = Layer.unwrap(
  Effect.gen(function* () {
    const config = yield* ServerConfig.ServerConfig;
    const devOrigin = config.devUrl?.origin;
    // Dev uses credentialed requests from Vite or the Electron custom origin, so both must be
    // explicit. Packaged desktop omits credentials and uses Effect's default wildcard origin.
    //
    // T3CODE_DEV_ALLOWED_ORIGINS covers dev servers reached from a second
    // origin — a tailnet name, a LAN IP, a phone. Browser dev normally proxies
    // through Vite and is same-origin (no preflight at all), so this is a
    // safety net for the desktop renderer and any direct-to-backend caller.
    return HttpRouter.cors({
      ...(devOrigin
        ? {
            allowedOrigins: [devOrigin, ...DESKTOP_RENDERER_ORIGINS, ...config.devAllowedOrigins],
            credentials: true,
          }
        : {}),
      allowedMethods: browserApiCorsAllowedMethods,
      allowedHeaders: browserApiCorsAllowedHeaders,
      maxAge: 600,
    });
  }),
);

export function isLoopbackHostname(hostname: string): boolean {
  const normalizedHostname = hostname
    .trim()
    .toLowerCase()
    .replace(/^\[(.*)\]$/, "$1");
  return LOOPBACK_HOSTNAMES.has(normalizedHostname);
}

export function resolveDevRedirectUrl(devUrl: URL, requestUrl: URL): string {
  const redirectUrl = new URL(devUrl.toString());
  redirectUrl.pathname = requestUrl.pathname;
  redirectUrl.search = requestUrl.search;
  redirectUrl.hash = requestUrl.hash;
  return redirectUrl.toString();
}

const authenticateRawRouteWithScope = (
  scope: typeof AuthOrchestrationReadScope | typeof AuthOrchestrationOperateScope,
) =>
  Effect.gen(function* () {
    const request = yield* HttpServerRequest.HttpServerRequest;
    const serverAuth = yield* EnvironmentAuth.EnvironmentAuth;
    const session = yield* serverAuth.authenticateHttpRequest(request).pipe(
      Effect.catchIf(EnvironmentAuth.isServerAuthCredentialError, (error) =>
        failEnvironmentAuthInvalid(EnvironmentAuth.serverAuthCredentialReason(error)),
      ),
      Effect.catchIf(EnvironmentAuth.isServerAuthInternalError, (error) =>
        failEnvironmentInternal("internal_error", error),
      ),
    );
    if (!session.scopes.includes(scope)) {
      return yield* failEnvironmentScopeRequired(scope);
    }
  });

export const serverEnvironmentHttpApiLayer = HttpApiBuilder.group(
  EnvironmentHttpApi,
  "metadata",
  Effect.fnUntraced(function* (handlers) {
    const serverEnvironment = yield* ServerEnvironment.ServerEnvironment;
    return handlers.handle(
      "descriptor",
      Effect.fn("environment.metadata.descriptor")(function* (args) {
        yield* annotateEnvironmentRequest(args.endpoint.name);
        return yield* serverEnvironment.getDescriptor;
      }, traceRelayRequest),
    );
  }),
);

class DecodeOtlpTraceRecordsError extends Data.TaggedError("DecodeOtlpTraceRecordsError")<{
  readonly cause: unknown;
  readonly bodyJson: OtlpTracer.TraceData;
}> {}

export const otlpTracesProxyRouteLayer = HttpRouter.add(
  "POST",
  OTLP_TRACES_PROXY_PATH,
  Effect.gen(function* () {
    yield* authenticateRawRouteWithScope(AuthOrchestrationOperateScope);
    const request = yield* HttpServerRequest.HttpServerRequest;
    const config = yield* ServerConfig.ServerConfig;
    const otlpTracesUrl = config.otlpTracesUrl;
    const browserTraceCollector = yield* BrowserTraceCollector.BrowserTraceCollector;
    const httpClient = yield* HttpClient.HttpClient;
    const bodyJson = cast<unknown, OtlpTracer.TraceData>(yield* request.json);

    yield* Effect.try({
      try: () => decodeOtlpTraceRecords(bodyJson),
      catch: (cause) => new DecodeOtlpTraceRecordsError({ cause, bodyJson }),
    }).pipe(
      Effect.flatMap((records) => browserTraceCollector.record(records)),
      Effect.catch((cause) =>
        Effect.logWarning("Failed to decode browser OTLP traces", {
          cause,
          bodyJson,
        }),
      ),
    );

    if (otlpTracesUrl === undefined) {
      return HttpServerResponse.empty({ status: 204 });
    }

    return yield* httpClient
      .post(otlpTracesUrl, {
        body: HttpBody.jsonUnsafe(bodyJson),
      })
      .pipe(
        Effect.flatMap(HttpClientResponse.filterStatusOk),
        Effect.as(HttpServerResponse.empty({ status: 204 })),
        Effect.tapError((cause) =>
          Effect.logWarning("Failed to export browser OTLP traces", {
            cause,
            otlpTracesUrl,
          }),
        ),
        Effect.orElseSucceed(() =>
          HttpServerResponse.text("Trace export failed.", { status: 502 }),
        ),
      );
  }).pipe(
    Effect.catchTags({
      EnvironmentAuthInvalidError: HttpServerRespondable.toResponse,
      EnvironmentInternalError: HttpServerRespondable.toResponse,
      EnvironmentScopeRequiredError: HttpServerRespondable.toResponse,
    }),
  ),
);

export const assetRouteLayer = HttpRouter.add(
  "GET",
  `${ASSET_ROUTE_PREFIX}/*`,
  Effect.gen(function* () {
    const request = yield* HttpServerRequest.HttpServerRequest;
    const url = HttpServerRequest.toURL(request);
    if (Option.isNone(url)) {
      return HttpServerResponse.text("Bad Request", { status: 400 });
    }

    const suffix = url.value.pathname.slice(`${ASSET_ROUTE_PREFIX}/`.length);
    const separatorIndex = suffix.indexOf("/");
    if (separatorIndex <= 0) {
      return HttpServerResponse.text("Not Found", { status: 404 });
    }

    const asset = yield* resolveAsset(
      suffix.slice(0, separatorIndex),
      suffix.slice(separatorIndex + 1),
    );
    if (!asset) {
      return HttpServerResponse.text("Not Found", { status: 404 });
    }
    return yield* HttpServerResponse.file(asset.path, {
      status: 200,
      headers: assetResponseHeaders(asset.path),
    }).pipe(
      Effect.orElseSucceed(() => HttpServerResponse.text("Internal Server Error", { status: 500 })),
    );
  }),
);

export const staticAndDevRouteLayer = HttpRouter.add(
  "GET",
  "*",
  Effect.gen(function* () {
    const request = yield* HttpServerRequest.HttpServerRequest;
    const url = HttpServerRequest.toURL(request);

    if (Option.isNone(url)) {
      return HttpServerResponse.text("Bad Request", { status: 400 });
    }

    const config = yield* ServerConfig.ServerConfig;
    if (config.devUrl && isDevProxiedPath(url.value.pathname)) {
      return HttpServerResponse.text("Not Found", { status: 404 });
    }

    if (config.devUrl && isLoopbackHostname(url.value.hostname)) {
      return HttpServerResponse.redirect(resolveDevRedirectUrl(config.devUrl, url.value), {
        status: 302,
      });
    }

    const staticDir =
      config.staticDir ?? (config.devUrl ? yield* ServerConfig.resolveStaticDir() : undefined);
    if (!staticDir) {
      return HttpServerResponse.text("No static directory configured and no dev URL set.", {
        status: 503,
      });
    }

    const fileSystem = yield* FileSystem.FileSystem;
    const path = yield* Path.Path;
    const staticRoot = path.resolve(staticDir);
    const staticRequestPath = url.value.pathname === "/" ? "/index.html" : url.value.pathname;
    const rawStaticRelativePath = staticRequestPath.replace(/^[/\\]+/, "");
    const hasRawLeadingParentSegment = rawStaticRelativePath.startsWith("..");
    const staticRelativePath = path.normalize(rawStaticRelativePath).replace(/^[/\\]+/, "");
    const hasPathTraversalSegment = staticRelativePath.startsWith("..");
    if (
      staticRelativePath.length === 0 ||
      hasRawLeadingParentSegment ||
      hasPathTraversalSegment ||
      staticRelativePath.includes("\0")
    ) {
      return HttpServerResponse.text("Invalid static file path", { status: 400 });
    }

    const isWithinStaticRoot = (candidate: string) =>
      candidate === staticRoot ||
      candidate.startsWith(staticRoot.endsWith(path.sep) ? staticRoot : `${staticRoot}${path.sep}`);

    let filePath = path.resolve(staticRoot, staticRelativePath);
    if (!isWithinStaticRoot(filePath)) {
      return HttpServerResponse.text("Invalid static file path", { status: 400 });
    }

    const ext = path.extname(filePath);
    if (!ext) {
      filePath = path.resolve(filePath, "index.html");
      if (!isWithinStaticRoot(filePath)) {
        return HttpServerResponse.text("Invalid static file path", { status: 400 });
      }
    }

    const fileInfo = yield* fileSystem.stat(filePath).pipe(Effect.orElseSucceed(() => null));
    if (!fileInfo || fileInfo.type !== "File") {
      const indexPath = path.resolve(staticRoot, "index.html");
      const indexData = yield* fileSystem
        .readFile(indexPath)
        .pipe(Effect.orElseSucceed(() => null));
      if (!indexData) {
        return HttpServerResponse.text("Not Found", { status: 404 });
      }
      return HttpServerResponse.uint8Array(indexData, {
        status: 200,
        contentType: "text/html; charset=utf-8",
      });
    }

    const contentType = Mime.getType(filePath) ?? "application/octet-stream";
    const data = yield* fileSystem.readFile(filePath).pipe(Effect.orElseSucceed(() => null));
    if (!data) {
      return HttpServerResponse.text("Internal Server Error", { status: 500 });
    }

    return HttpServerResponse.uint8Array(data, {
      status: 200,
      contentType,
    });
  }),
);
