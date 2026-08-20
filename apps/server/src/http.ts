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

function readOptionalJson(fileSystem: FileSystem.FileSystem, path: string) {
  return fileSystem.readFileString(path).pipe(
    Effect.map((contents) => JSON.parse(contents) as unknown),
    Effect.orElseSucceed(() => null),
  );
}

function readModelInference(httpClient: HttpClient.HttpClient) {
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
    const phase =
      active && processedPromptTokens < uncachedPromptTokens
        ? "prompt"
        : active
          ? "generation"
          : "idle";
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

function readCampaignSnapshot(
  fileSystem: FileSystem.FileSystem,
  httpClient: HttpClient.HttpClient,
) {
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
    const [campaign, current, inference] = yield* Effect.all([
      readOptionalJson(fileSystem, `${directory}/campaign.json`),
      readOptionalJson(fileSystem, `${directory}/current.json`),
      readModelInference(httpClient),
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
      campaign,
      current,
      events: tail.events,
      stream: {
        lastEventAt: tail.lastEventAt,
        bytes: tail.bytes,
        lastEventMethod: tail.lastEventMethod,
      },
      inference,
    };
  });
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

    const snapshots = Stream.concat(
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
