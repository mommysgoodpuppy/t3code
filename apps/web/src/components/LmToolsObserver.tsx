import { useEffect, useMemo, useRef, useState } from "react";

type JsonRecord = Record<string, unknown>;
type ObserverResponse = {
  configured: boolean;
  observer?: { kind: "campaign" | "proxy"; title: string };
  campaign: JsonRecord | null;
  current: JsonRecord | null;
  summaries?: JsonRecord | null;
  definition?: JsonRecord | null;
  controller?: JsonRecord | null;
  evaluations?: unknown[];
  events: JsonRecord[];
  stream?: { lastEventAt: string | null; bytes: number; lastEventMethod?: string | null };
  inference?: InferenceState | null;
};
type InferenceState = {
  active: boolean;
  phase: "prompt" | "generation" | "idle";
  taskId: number | null;
  context: { used: number; limit: number; percent: number };
  prompt: { processed: number; total: number; cached: number; percent: number };
  generatedTokens: number;
};
type ObserverAppend = {
  events: JsonRecord[];
  stream: NonNullable<ObserverResponse["stream"]>;
};

type TimelineEntry =
  | { key: string; kind: "user" | "assistant" | "reasoning" | "error"; text: string }
  | { key: string; kind: "plan"; items: Array<{ text: string; completed: boolean }> }
  | {
      key: string;
      kind: "command";
      command: string;
      output: string;
      status: "running" | "completed" | "failed";
      exitCode: number | null;
    };

const EMPTY: ObserverResponse = {
  configured: false,
  campaign: null,
  current: null,
  events: [],
};

const DELTA_METHODS = new Set([
  "item/agentMessage/delta",
  "item/reasoning/summaryTextDelta",
  "item/reasoning/textDelta",
  "item/commandExecution/outputDelta",
]);

function compactEvents(events: JsonRecord[]): JsonRecord[] {
  const compacted: JsonRecord[] = [];
  const deltas = new Map<string, number>();
  for (const event of events) {
    const params = record(event.params);
    const method = typeof event.method === "string" ? event.method : "";
    const itemId = typeof params?.itemId === "string" ? params.itemId : null;
    const delta = typeof params?.delta === "string" ? params.delta : null;
    if (!DELTA_METHODS.has(method) || !itemId || delta === null) {
      compacted.push(event);
      continue;
    }
    const key = `${method}:${itemId}`;
    const existing = deltas.get(key);
    if (existing === undefined) {
      deltas.set(key, compacted.length);
      compacted.push(event);
      continue;
    }
    const previous = compacted[existing];
    const previousParams = record(previous?.params);
    if (!previous || !previousParams) continue;
    compacted[existing] = {
      ...previous,
      params: { ...previousParams, delta: String(previousParams.delta ?? "") + delta },
    };
  }
  return compacted.slice(-2000);
}

function record(value: unknown): JsonRecord | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as JsonRecord)
    : null;
}

function number(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function compactCount(value: number): string {
  return new Intl.NumberFormat(undefined, {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}

function formatDuration(seconds: number): string {
  const safe = Math.max(0, Math.floor(seconds));
  if (safe < 60) return `${safe}s`;
  const minutes = Math.floor(safe / 60);
  if (minutes < 60) return `${minutes}m ${safe % 60}s`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m`;
}

function records(value: unknown): JsonRecord[] {
  return Array.isArray(value)
    ? value.flatMap((entry) => (record(entry) ? [entry as JsonRecord] : []))
    : [];
}

function strings(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((entry): entry is string => typeof entry === "string")
    : [];
}

function phaseDefinition(data: ObserverResponse, phaseName: string): JsonRecord | null {
  const worker = record(data.definition?.worker);
  return records(worker?.phases).find((phase) => phase.name === phaseName) ?? null;
}

function formatAge(timestamp: string | null | undefined, now: number): string {
  if (!timestamp) return "no events received";
  const ageSeconds = Math.max(0, Math.floor((now - Date.parse(timestamp)) / 1_000));
  if (ageSeconds < 5) return "just now";
  if (ageSeconds < 60) return `${ageSeconds}s ago`;
  const minutes = Math.floor(ageSeconds / 60);
  if (minutes < 60) return `${minutes}m ${ageSeconds % 60}s ago`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m ago`;
}

function useNow(): number {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1_000);
    return () => window.clearInterval(timer);
  }, []);
  return now;
}

function planItems(value: unknown): Array<{ text: string; completed: boolean }> {
  if (!Array.isArray(value)) return [];
  return value.flatMap((candidate) => {
    const item = record(candidate);
    if (!item) return [];
    const text =
      typeof item.text === "string" ? item.text : typeof item.step === "string" ? item.step : null;
    if (!text) return [];
    return [
      {
        text,
        completed: item.completed === true || item.status === "completed",
      },
    ];
  });
}

function itemText(value: unknown): string {
  if (typeof value === "string") return value;
  if (!Array.isArray(value)) return "";
  return value
    .flatMap((candidate) => {
      if (typeof candidate === "string") return [candidate];
      const part = record(candidate);
      if (!part) return [];
      if (typeof part.text === "string") return [part.text];
      return [];
    })
    .join("\n");
}

function itemType(value: unknown): string {
  return String(value ?? "")
    .replaceAll("_", "")
    .toLowerCase();
}

function timelineEntries(events: JsonRecord[]): TimelineEntry[] {
  const entries: TimelineEntry[] = [];
  const commands = new Map<string, number>();
  const messages = new Map<string, number>();
  const reasoning = new Map<string, number>();

  const upsertText = (
    map: Map<string, number>,
    key: string,
    kind: "user" | "assistant" | "reasoning",
    text: string,
    replace: boolean,
  ) => {
    const existing = map.get(key);
    if (existing === undefined) {
      map.set(key, entries.length);
      entries.push({ key, kind, text });
      return;
    }
    const entry = entries[existing];
    if (!entry || entry.kind !== kind) return;
    entries[existing] = { ...entry, text: replace ? text : entry.text + text };
  };

  for (let index = 0; index < events.length; index += 1) {
    const event = events[index];
    if (!event) continue;
    const method = typeof event.method === "string" ? event.method : null;
    const params = record(event.params);
    const item = record(event.item) ?? record(params?.item);
    const deltaId = String(params?.itemId ?? `event-${index}`);

    if (method === "item/agentMessage/delta" && typeof params?.delta === "string") {
      upsertText(messages, deltaId, "assistant", params.delta, false);
      continue;
    }
    if (
      (method === "item/reasoning/summaryTextDelta" || method === "item/reasoning/textDelta") &&
      typeof params?.delta === "string"
    ) {
      upsertText(reasoning, deltaId, "reasoning", params.delta, false);
      continue;
    }
    if (method === "item/commandExecution/outputDelta" && typeof params?.delta === "string") {
      const existing = commands.get(deltaId);
      if (existing === undefined) {
        commands.set(deltaId, entries.length);
        entries.push({
          key: deltaId,
          kind: "command",
          command: "command output (start event outside retained window)",
          output: params.delta,
          status: "running",
          exitCode: null,
        });
      } else {
        const entry = entries[existing];
        if (entry?.kind === "command") {
          entries[existing] = { ...entry, output: entry.output + params.delta };
        }
      }
      continue;
    }
    if (method === "turn/plan/updated") {
      const items = planItems(params?.plan);
      const existing = entries.findIndex((entry) => entry.kind === "plan");
      const plan: TimelineEntry = { key: `plan-${index}`, kind: "plan", items };
      if (existing >= 0) entries[existing] = plan;
      else entries.push(plan);
      continue;
    }
    if (method === "error") {
      const error = record(params?.error);
      const text = typeof error?.message === "string" ? error.message : "Unknown Codex error";
      entries.push({ key: `error-${index}`, kind: "error", text });
      continue;
    }
    if (!item) continue;
    const id = String(item.id ?? index);
    const type = itemType(item.type);

    if (type === "reasoning") {
      const text =
        typeof item.text === "string"
          ? item.text
          : itemText(item.summary) || itemText(item.content);
      if (text) upsertText(reasoning, id, "reasoning", text, method === "item/completed");
      continue;
    }
    if (type === "usermessage" && typeof item.text === "string") {
      upsertText(messages, id, "user", item.text, true);
      continue;
    }
    if (type === "agentmessage" && typeof item.text === "string") {
      upsertText(messages, id, "assistant", item.text, method === "item/completed");
      continue;
    }
    if (type === "error" && typeof item.message === "string") {
      entries.push({ key: id, kind: "error", text: item.message });
      continue;
    }
    if (type === "todolist" || type === "plan") {
      const items = planItems(item.items);
      const existing = entries.findIndex((entry) => entry.kind === "plan");
      const plan: TimelineEntry = { key: id, kind: "plan", items };
      if (existing >= 0) entries[existing] = plan;
      else entries.push(plan);
      continue;
    }
    if (type !== "commandexecution") continue;

    const completed = event.type === "item.completed" || method === "item/completed";
    const exitCode =
      typeof item.exit_code === "number"
        ? item.exit_code
        : typeof item.exitCode === "number"
          ? item.exitCode
          : null;
    const finalOutput =
      typeof item.aggregated_output === "string"
        ? item.aggregated_output
        : typeof item.aggregatedOutput === "string"
          ? item.aggregatedOutput
          : "";
    const commandText = typeof item.command === "string" ? item.command : "";
    const command: TimelineEntry = {
      key: id,
      kind: "command",
      command: commandText || "tool invocation outside retained proxy trace",
      output: finalOutput,
      status: completed ? (exitCode === 0 ? "completed" : "failed") : "running",
      exitCode,
    };
    const existing = commands.get(id);
    if (existing === undefined) {
      commands.set(id, entries.length);
      entries.push(command);
    } else {
      const previous = entries[existing];
      entries[existing] = {
        ...command,
        command: commandText || (previous?.kind === "command" ? previous.command : command.command),
        output: finalOutput || (previous?.kind === "command" ? previous.output : ""),
      };
    }
  }

  return entries.slice(-100);
}

function useObserverStream() {
  const [data, setData] = useState<ObserverResponse>(EMPTY);
  const [connection, setConnection] = useState<"connecting" | "live" | "reconnecting">(
    "connecting",
  );

  useEffect(() => {
    const source = new EventSource("/api/lm-tools/campaign/events");
    source.onopen = () => setConnection("live");
    const receiveSnapshot = (event: MessageEvent<string>) => {
      try {
        const snapshot = JSON.parse(event.data) as ObserverResponse;
        setData({ ...snapshot, events: compactEvents(snapshot.events) });
        setConnection("live");
      } catch {
        setConnection("reconnecting");
      }
    };
    const receiveAppend = (event: MessageEvent<string>) => {
      try {
        const append = JSON.parse(event.data) as ObserverAppend;
        setData((current) => ({
          ...current,
          events: compactEvents([...current.events, ...append.events]),
          stream: append.stream,
        }));
        setConnection("live");
      } catch {
        setConnection("reconnecting");
      }
    };
    const receiveInference = (event: MessageEvent<string>) => {
      try {
        const update = JSON.parse(event.data) as { inference: InferenceState | null };
        setData((current) => ({ ...current, inference: update.inference }));
        setConnection("live");
      } catch {
        setConnection("reconnecting");
      }
    };
    source.onmessage = receiveSnapshot;
    source.addEventListener("snapshot", receiveSnapshot);
    source.addEventListener("append", receiveAppend);
    source.addEventListener("inference", receiveInference);
    source.onerror = () => setConnection("reconnecting");
    return () => {
      source.removeEventListener("snapshot", receiveSnapshot);
      source.removeEventListener("append", receiveAppend);
      source.removeEventListener("inference", receiveInference);
      source.close();
    };
  }, []);

  return { data, connection };
}

export function LmToolsObserver() {
  const { data, connection } = useObserverStream();
  const [view, setView] = useState<"live" | "history">("live");
  const entries = useMemo(() => timelineEntries(data.events), [data.events]);
  const usage = record(data.campaign?.usage);
  // Proxy observers work for any agent whose traffic crosses the model proxy.
  // Anything else is a campaign run.
  const observerKind = data.observer?.kind;
  const conversation = observerKind === "proxy";
  const observerTitle = String(data.observer?.title ?? "Agent");
  const status = String(data.current?.status ?? "not started");
  const paused = data.controller?.phase === "PAUSED" || status === "paused";
  const running =
    !paused &&
    (status === "running" ||
      status === "starting" ||
      (conversation && data.inference?.active === true));
  const activeCommand = entries.some(
    (entry) => entry.kind === "command" && entry.status === "running",
  );
  const now = useNow();
  const lastMethod = data.stream?.lastEventMethod ?? "";
  const lastEventAge = data.stream?.lastEventAt
    ? Math.max(0, now - Date.parse(data.stream.lastEventAt))
    : Number.POSITIVE_INFINITY;
  const streamingDelta =
    lastEventAge < 2_000 &&
    (lastMethod === "item/agentMessage/delta" ||
      lastMethod === "item/reasoning/summaryTextDelta" ||
      lastMethod === "item/reasoning/textDelta" ||
      lastMethod === "item/commandExecution/outputDelta");
  const scroller = useRef<HTMLDivElement>(null);
  const [following, setFollowing] = useState(true);
  const phaseName = conversation && data.inference?.active
    ? data.inference.phase
    : paused
    ? "paused"
    : String(data.current?.phase ?? "between cycles").replace(":finalize", "");
  const configuredPhase = phaseDefinition(data, phaseName);
  const phaseTimeoutSeconds = number(configuredPhase?.timeoutSeconds);
  const phaseStartedAt = String(data.current?.phaseStartedAt ?? "");
  const phaseElapsedSeconds = phaseStartedAt
    ? Math.max(0, (now - Date.parse(phaseStartedAt)) / 1_000)
    : 0;

  useEffect(() => {
    if (!following || view !== "live") return;
    const frame = window.requestAnimationFrame(() => {
      const element = scroller.current;
      if (element) element.scrollTop = element.scrollHeight;
    });
    return () => window.cancelAnimationFrame(frame);
  }, [entries.length, data.stream?.bytes, following, view]);

  const updateFollowing = () => {
    const element = scroller.current;
    if (!element) return;
    setFollowing(element.scrollHeight - element.scrollTop - element.clientHeight < 96);
  };

  const resumeFollowing = () => {
    setFollowing(true);
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: "smooth" });
  };

  return (
    <main className="fixed inset-0 flex flex-col overflow-hidden bg-background text-foreground">
      <header className="border-b border-border/80 bg-background/95 px-4 py-3 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h1 className="truncate text-sm font-semibold">
                {data.observer?.title ?? "lm-tools agent observer"}
              </h1>
              <ConnectionBadge connection={connection} />
            </div>
            <p className="mt-0.5 truncate font-mono text-[11px] text-muted-foreground">
              {String(data.current?.id ?? "Waiting for a run")}
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-4 text-xs">
            <HeaderMetric label={conversation ? "State" : "Stage"} value={phaseName} />
            {!conversation ? <HeaderMetric label="Turns" value={String(number(data.campaign?.runs))} /> : null}
            <HeaderMetric label="Tokens" value={compactCount(number(usage?.totalTokens))} />
            <ContextMeter inference={data.inference} />
            <span
              className={paused ? "text-sky-300" : running ? "text-amber-400" : "text-emerald-400"}
            >
              ● {paused ? "paused · GPU released" : running ? (conversation ? "agent active" : "run active") : status}
            </span>
          </div>
        </div>
        <div className="mx-auto mt-3 flex max-w-6xl items-end justify-between gap-4">
          <nav className="flex rounded-lg bg-muted/50 p-1 text-xs">
            <ViewButton active={view === "live"} onClick={() => setView("live")}>
              Live stream
            </ViewButton>
            {!conversation ? (
              <ViewButton active={view === "history"} onClick={() => setView("history")}>
                Campaign map
              </ViewButton>
            ) : null}
          </nav>
          {!conversation ? (
            <PhaseClock
              phase={phaseName}
              elapsedSeconds={phaseElapsedSeconds}
              timeoutSeconds={phaseTimeoutSeconds}
              running={running}
            />
          ) : null}
        </div>
      </header>

      {view === "live" ? (
        <div ref={scroller} className="relative flex-1 overflow-y-auto" onScroll={updateFollowing}>
          <div className="mx-auto flex min-h-full max-w-4xl flex-col px-4 py-6 sm:px-8">
            {!conversation ? <TurnIntro current={data.current} phase={phaseName} /> : null}
            {!data.configured ? (
              <EmptyState conversation={conversation} />
            ) : (
              <div className="space-y-4">
                {entries.map((entry) => (
                  <TimelineRow key={entry.key} entry={entry} />
                ))}
                {data.inference?.active &&
                !activeCommand &&
                !streamingDelta &&
                ((data.inference.phase === "prompt" && data.inference.prompt.processed > 0) ||
                  (data.inference.phase === "generation" && data.inference.generatedTokens > 0)) ? (
                  <InferenceActivity inference={data.inference} />
                ) : null}
                {running && !activeCommand && !streamingDelta && !data.inference?.active ? (
                  <ActivityFooter
                    lastEventAt={data.stream?.lastEventAt}
                    now={now}
                    source={conversation ? observerTitle : "Codex"}
                  />
                ) : null}
              </div>
            )}
          </div>
          {!following ? (
            <button
              type="button"
              onClick={resumeFollowing}
              className="sticky bottom-4 left-1/2 -translate-x-1/2 rounded-full border border-border bg-background/95 px-3 py-1.5 text-xs shadow-lg backdrop-blur hover:bg-accent"
            >
              ↓ Follow live activity
            </button>
          ) : null}
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto">
          <CampaignMap data={data} now={now} />
        </div>
      )}
    </main>
  );
}

function ViewButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={
        active
          ? "rounded-md bg-background px-3 py-1.5 font-medium shadow-sm"
          : "rounded-md px-3 py-1.5 text-muted-foreground hover:text-foreground"
      }
    >
      {children}
    </button>
  );
}

function PhaseClock({
  phase,
  elapsedSeconds,
  timeoutSeconds,
  running,
}: {
  phase: string;
  elapsedSeconds: number;
  timeoutSeconds: number;
  running: boolean;
}) {
  const bounded = running && timeoutSeconds > 0;
  const remaining = Math.max(0, timeoutSeconds - elapsedSeconds);
  const percent = bounded ? Math.min(100, (elapsedSeconds / timeoutSeconds) * 100) : 0;
  return (
    <div className="w-64 max-w-[45vw]">
      <div className="flex justify-between text-[10px] text-muted-foreground">
        <span className="capitalize">{phase}</span>
        <span className="font-mono">
          {bounded
            ? `${formatDuration(elapsedSeconds)} / ${formatDuration(timeoutSeconds)} · ${formatDuration(remaining)} left`
            : "not timed"}
        </span>
      </div>
      <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-muted">
        <div
          className={percent > 85 ? "h-full bg-amber-400" : "h-full bg-sky-400"}
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}

function ConnectionBadge({ connection }: { connection: "connecting" | "live" | "reconnecting" }) {
  const live = connection === "live";
  return (
    <span
      className={
        live
          ? "rounded-full bg-emerald-500/10 px-2 py-0.5 text-[10px] font-medium text-emerald-400"
          : "rounded-full bg-amber-500/10 px-2 py-0.5 text-[10px] font-medium text-amber-400"
      }
    >
      <span className={live ? "" : "animate-pulse"}>●</span>{" "}
      {live ? "Stream connected" : connection === "connecting" ? "Connecting" : "Reconnecting"}
    </span>
  );
}

function HeaderMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="hidden text-right sm:block">
      <div className="text-[9px] uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className="font-mono text-[11px]">{value}</div>
    </div>
  );
}

function ContextMeter({ inference }: { inference: InferenceState | null | undefined }) {
  if (!inference || inference.context.limit <= 0) return null;
  const percent = Math.min(100, Math.max(0, inference.context.percent * 100));
  return (
    <div className="hidden w-32 sm:block">
      <div className="flex justify-between text-[9px] uppercase tracking-wider text-muted-foreground">
        <span>Context</span>
        <span>{percent.toFixed(0)}%</span>
      </div>
      <div className="mt-1 h-1 overflow-hidden rounded-full bg-muted">
        <div className="h-full rounded-full bg-sky-400" style={{ width: `${percent}%` }} />
      </div>
      <div className="mt-0.5 text-right font-mono text-[9px] text-muted-foreground">
        {compactCount(inference.context.used)} / {compactCount(inference.context.limit)}
      </div>
    </div>
  );
}

function InferenceActivity({ inference }: { inference: InferenceState }) {
  const promptPercent = Math.min(100, Math.max(0, inference.prompt.percent * 100));
  const prompt = inference.phase === "prompt";
  return (
    <section className="flex gap-3 text-sm text-muted-foreground">
      <div className="mt-0.5 flex size-6 shrink-0 items-center justify-center rounded-full border border-border bg-muted/40 text-[10px]">
        <span className="size-1.5 animate-pulse rounded-full bg-amber-400" />
      </div>
      <div className="min-w-0 flex-1 border-l border-border/60 pl-3">
        <div className="mb-1 text-[10px] font-medium uppercase tracking-wider opacity-70">
          {prompt ? "Evaluating prompt" : "Generating"}
        </div>
        {prompt ? (
          <>
            <div className="h-1.5 overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full bg-amber-400 transition-[width] duration-100"
                style={{ width: `${promptPercent}%` }}
              />
            </div>
            <p className="mt-1 font-mono text-[10px]">
              {inference.prompt.processed.toLocaleString()} /{" "}
              {inference.prompt.total.toLocaleString()} uncached tokens ·{" "}
              {inference.prompt.cached.toLocaleString()} cached
            </p>
          </>
        ) : (
          <p className="font-mono text-[10px]">
            {inference.generatedTokens.toLocaleString()} tokens generated in this inference
          </p>
        )}
      </div>
    </section>
  );
}

function TurnIntro({ current, phase }: { current: JsonRecord | null; phase: string }) {
  return (
    <section className="mb-7 flex justify-end">
      <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-accent px-4 py-3 text-sm shadow-sm">
        <p className="font-medium">Continue the Qwen tinygrad campaign · {phase}</p>
        <p className="mt-1 font-mono text-[10px] text-muted-foreground">
          Phase prompt: {String(current?.prompt ?? "waiting")}
        </p>
      </div>
    </section>
  );
}

function CampaignMap({ data, now }: { data: ObserverResponse; now: number }) {
  const runs = records(data.campaign?.history);
  const summaries = records(data.summaries?.entries);
  const summariesByRun = new Map(
    summaries.map((summary) => [String(summary.runId ?? ""), summary]),
  );
  const evaluations = records(data.evaluations);
  const nodes = [
    ...runs.flatMap((run, runIndex) => {
      const phases = records(run.phases);
      const displayed = phases.length > 0 ? phases : [{ name: "worker", ...run }];
      return displayed.map((phase, phaseIndex) => ({
        kind: "phase" as const,
        run,
        phase,
        cycleIndex: runIndex + 1,
        phaseIndex,
        lastPhase: phaseIndex === displayed.length - 1,
        time: Date.parse(String(phase.startedAt ?? run.startedAt ?? "")) || 0,
      }));
    }),
    ...evaluations.map((value) => ({
      kind: "evaluation" as const,
      value,
      time: Date.parse(String(value.startedAt ?? "")) || 0,
    })),
  ].sort((left, right) => left.time - right.time);
  const controller = data.controller;
  const accepted = record(controller?.accepted);
  const phaseControl = record(record(data.definition?.worker)?.phaseControl);

  return (
    <div className="mx-auto max-w-6xl px-4 py-6 sm:px-8">
      <section className="mb-7 grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
        <HistoryMetric label="Completed cycles" value={String(runs.length)} />
        <HistoryMetric label="Research summaries" value={String(summaries.length)} />
        <HistoryMetric label="Evaluations" value={String(evaluations.length)} />
        <HistoryMetric
          label="Accepted candidate"
          value={String(accepted?.hash ?? "baseline").slice(0, 10)}
          mono
        />
        <HistoryMetric
          label="Emergency ceiling"
          value={`${formatDuration(number(phaseControl?.maxWallSeconds))} · ${number(phaseControl?.maxTransitions)} transitions`}
        />
      </section>

      <div className="mb-5">
        <h2 className="text-base font-semibold">Research trajectory</h2>
        <p className="mt-1 text-xs text-muted-foreground">
          Each phase is one separately budgeted agent session. A cycle spans one or more phases and
          ends at a host evaluation boundary or worker failure.
        </p>
      </div>

      <div className="relative ml-3 border-l border-border/80 pl-7">
        {nodes.length === 0 ? <EmptyState /> : null}
        <div className="space-y-2.5">
          {nodes.map((node, index) =>
            node.kind === "phase" ? (
              <HistoryPhaseNode
                key={`phase-${String(node.run.id ?? index)}-${node.phaseIndex}`}
                run={node.run}
                phase={node.phase}
                summary={
                  node.lastPhase
                    ? (record(node.run.summary) ?? summariesByRun.get(String(node.run.id ?? "")))
                    : undefined
                }
                cycleIndex={node.cycleIndex}
                lastPhase={node.lastPhase}
              />
            ) : (
              <EvaluationNode
                key={`evaluation-${String(record(node.value.candidate)?.id ?? index)}`}
                evaluation={node.value}
              />
            ),
          )}
          {data.current?.status === "running" ||
          data.current?.status === "starting" ||
          data.current?.status === "summarizing" ? (
            <CurrentHistoryNode
              current={data.current}
              now={now}
              phaseLimit={number(
                phaseDefinition(data, String(data.current.phase ?? ""))?.timeoutSeconds,
              )}
            />
          ) : data.controller?.phase === "PAUSED" || data.current?.status === "paused" ? (
            <PausedHistoryNode current={data.current} />
          ) : null}
        </div>
      </div>
    </div>
  );
}

function HistoryMetric({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="rounded-xl border border-border bg-card/50 px-3 py-2.5">
      <div className="text-[9px] uppercase tracking-wider text-muted-foreground">{label}</div>
      <div
        className={mono ? "mt-1 truncate font-mono text-sm" : "mt-1 truncate text-sm font-medium"}
      >
        {value}
      </div>
    </div>
  );
}

function TimelineDot({
  tone = "default",
}: {
  tone?: "default" | "active" | "evaluation" | "failed";
}) {
  const color =
    tone === "active"
      ? "border-sky-400 bg-sky-400"
      : tone === "evaluation"
        ? "border-violet-400 bg-violet-400"
        : tone === "failed"
          ? "border-destructive bg-destructive"
          : "border-emerald-400 bg-background";
  return (
    <span className={`absolute -left-[2.12rem] top-4 size-3 rounded-full border-2 ${color}`} />
  );
}

function HistoryPhaseNode({
  run,
  phase,
  summary,
  cycleIndex,
  lastPhase,
}: {
  run: JsonRecord;
  phase: JsonRecord;
  summary: JsonRecord | undefined;
  cycleIndex: number;
  lastPhase: boolean;
}) {
  const failed = number(phase.exitCode) !== 0;
  const before = record(run.repositoriesBefore);
  const after = record(run.repositoriesAfter);
  const repositoryNames = [...new Set([...Object.keys(before ?? {}), ...Object.keys(after ?? {})])];
  return (
    <article className="relative rounded-lg border border-border bg-card/35 px-3 py-2.5">
      <TimelineDot tone={failed ? "failed" : "default"} />
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2">
          <span className="text-xs font-medium capitalize">{String(phase.name ?? "worker")}</span>
          <span className="font-mono text-[9px] text-muted-foreground">Cycle {cycleIndex}</span>
          {lastPhase ? (
            <span
              className={
                number(run.exitCode) !== 0
                  ? "rounded-full bg-destructive/10 px-2 py-0.5 text-[10px] text-destructive"
                  : "rounded-full bg-emerald-500/10 px-2 py-0.5 text-[10px] text-emerald-400"
              }
            >
              {number(run.exitCode) !== 0
                ? run.timedOut
                  ? "cycle timed out"
                  : "cycle failed"
                : "cycle completed"}
            </span>
          ) : null}
        </div>
        <div className="flex items-center gap-2 font-mono text-[9px] text-muted-foreground">
          <span>
            {failed ? "×" : "●"} {formatDuration(number(phase.durationSeconds))}
          </span>
          {phase.finalizationAttempted ? <span>finalization used</span> : null}
          {lastPhase ? (
            <span>{compactCount(number(record(run.usage)?.totalTokens))} cycle tokens</span>
          ) : null}
        </div>
      </div>

      {summary ? (
        <ResearchSummary summary={summary} />
      ) : lastPhase && typeof run.finalMessage === "string" && run.finalMessage.trim() ? (
        <details className="mt-2 rounded-md border border-border/70 bg-background/30">
          <summary className="cursor-pointer px-3 py-2 text-xs font-medium">
            Cycle result summary
          </summary>
          <p className="whitespace-pre-wrap border-t border-border/60 px-3 py-2.5 text-xs leading-5 text-muted-foreground">
            {run.finalMessage.trim()}
          </p>
        </details>
      ) : null}

      {lastPhase && repositoryNames.length > 0 ? (
        <details className="mt-2 rounded-lg border border-border/70 bg-background/30">
          <summary className="cursor-pointer px-3 py-2 text-xs font-medium">
            Workspace changes
          </summary>
          <div className="space-y-3 border-t border-border/60 px-3 py-2.5">
            {repositoryNames.map((name) => {
              const beforeRepo = record(before?.[name]);
              const afterRepo = record(after?.[name]);
              const beforeStatus = String(beforeRepo?.status ?? "");
              const afterStatus = String(afterRepo?.status ?? "");
              return (
                <div key={name}>
                  <div className="flex justify-between gap-2 text-[10px]">
                    <span className="truncate font-mono">{name}</span>
                    <span
                      className={
                        beforeStatus === afterStatus ? "text-muted-foreground" : "text-amber-400"
                      }
                    >
                      {beforeStatus === afterStatus ? "snapshot unchanged" : "changed during cycle"}
                    </span>
                  </div>
                  <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap rounded bg-black/20 p-2 font-mono text-[10px] text-muted-foreground">
                    {afterStatus || "clean"}
                  </pre>
                </div>
              );
            })}
          </div>
        </details>
      ) : null}
    </article>
  );
}

function ResearchSummary({ summary }: { summary: JsonRecord }) {
  const evidence = strings(summary.evidence);
  const changes = strings(summary.changes);
  return (
    <details className="mt-2 rounded-md border border-sky-400/20 bg-sky-500/[0.035]">
      <summary className="cursor-pointer px-3 py-2">
        <span className="text-xs font-medium">{String(summary.title ?? "Research summary")}</span>
        <span className="ml-2 rounded-full bg-sky-500/10 px-2 py-0.5 text-[9px] uppercase tracking-wide text-sky-300">
          {String(summary.status ?? "recorded")}
        </span>
      </summary>
      <div className="space-y-4 border-t border-sky-400/15 px-3 py-3 text-xs leading-5">
        <p className="whitespace-pre-wrap text-foreground/90">{String(summary.summary ?? "")}</p>
        <div>
          <div className="text-[9px] font-medium uppercase tracking-wider text-muted-foreground">
            Outcome
          </div>
          <p className="mt-1 text-muted-foreground">{String(summary.outcome ?? "Not recorded")}</p>
        </div>
        {evidence.length > 0 ? (
          <div>
            <div className="text-[9px] font-medium uppercase tracking-wider text-muted-foreground">
              Evidence retained
            </div>
            <ul className="mt-1 list-disc space-y-1 pl-4 text-muted-foreground">
              {evidence.map((item, index) => (
                <li key={`${item}-${index}`}>{item}</li>
              ))}
            </ul>
          </div>
        ) : null}
        {changes.length > 0 ? (
          <div>
            <div className="text-[9px] font-medium uppercase tracking-wider text-muted-foreground">
              Durable changes
            </div>
            <ul className="mt-1 list-disc space-y-1 pl-4 text-muted-foreground">
              {changes.map((item, index) => (
                <li key={`${item}-${index}`}>{item}</li>
              ))}
            </ul>
          </div>
        ) : null}
        <div className="rounded-md border border-border/60 bg-background/40 px-3 py-2">
          <div className="text-[9px] font-medium uppercase tracking-wider text-muted-foreground">
            Best next step
          </div>
          <p className="mt-1 text-foreground/90">{String(summary.nextStep ?? "Not recorded")}</p>
        </div>
        <div className="font-mono text-[9px] text-muted-foreground/70">
          {String(summary.source ?? "campaign")} · {String(summary.generatedAt ?? "")}
        </div>
      </div>
    </details>
  );
}

function EvaluationNode({ evaluation }: { evaluation: JsonRecord }) {
  const candidate = record(evaluation.candidate);
  const request = record(evaluation.request);
  const probe = record(evaluation.probe);
  const profile = record(evaluation.profile);
  const profileFamilies = records(profile?.families).slice(0, 3);
  const tiers = records(evaluation.tiers);
  const failed = typeof evaluation.failure === "string";
  const retainedProfile = failed && profile !== null && number(profile.kernelCount) > 0;
  const probeExecutionSucceeded = probe?.executionSucceeded ?? probe?.passed;
  return (
    <article className="relative rounded-lg border border-violet-400/25 bg-violet-500/[0.04] px-3 py-2.5">
      <TimelineDot tone={failed ? "failed" : "evaluation"} />
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2">
          <span className="text-xs font-medium text-violet-300">
            Host {String(request?.kind ?? "evaluation")}
          </span>
          <span className="truncate font-mono text-[9px] text-muted-foreground">
            {String(candidate?.id ?? "candidate")}
          </span>
        </div>
        <span
          className={
            evaluation.accepted
              ? "rounded-full bg-emerald-500/10 px-2 py-0.5 text-[10px] text-emerald-400"
              : failed
                ? "rounded-full bg-destructive/10 px-2 py-0.5 text-[10px] text-destructive"
                : "rounded-full bg-muted px-2 py-0.5 text-[10px] text-muted-foreground"
          }
        >
          {evaluation.accepted
            ? "accepted"
            : retainedProfile
              ? "profile retained · run faulted"
              : failed
                ? "failed"
                : String(request?.kind ?? "benchmark")}
        </span>
      </div>
      {profile ? (
        <div className="mt-2 flex flex-wrap items-center gap-1.5 text-[9px] text-muted-foreground">
          <span className="font-mono">
            {number(profile.kernelCount).toLocaleString()} kernels ·{" "}
            {formatDuration(number(profile.kernelTimeMs) / 1000)} GPU time
          </span>
          {profileFamilies.map((family) => (
            <span key={String(family.name)} className="rounded bg-violet-500/10 px-1.5 py-0.5">
              {String(family.name)} {number(family.percentKernelTime).toFixed(1)}%
            </span>
          ))}
        </div>
      ) : null}
      {probe ? (
        <p className="mt-3 text-xs text-muted-foreground">
          Probe{" "}
          <code>
            {String(probe.runner)}:{String(probe.path)}
          </code>{" "}
          · {formatDuration(number(probe.wallMs) / 1000)} ·{" "}
          {probeExecutionSucceeded ? "process completed" : "process failed"}
        </p>
      ) : null}
      {tiers.length > 0 ? (
        <div className="mt-3 flex flex-wrap gap-2">
          {tiers.map((tier) => (
            <span
              key={String(tier.name)}
              className="rounded-md border border-border bg-background/50 px-2 py-1 font-mono text-[10px]"
            >
              {String(tier.name)} {formatDuration(number(tier.medianWallMs) / 1000)} ·{" "}
              {number(tier.improvementPercent).toFixed(1)}%
            </span>
          ))}
        </div>
      ) : null}
      {failed ? (
        <details className="mt-2 text-[10px] text-destructive">
          <summary className="cursor-pointer">
            {retainedProfile
              ? "The diagnostic produced profile evidence but no valid model response"
              : "Evaluation failure"}
          </summary>
          <p className="mt-1 font-mono">{String(evaluation.failure)}</p>
        </details>
      ) : null}
    </article>
  );
}

function CurrentHistoryNode({
  current,
  now,
  phaseLimit,
}: {
  current: JsonRecord;
  now: number;
  phaseLimit: number;
}) {
  const cycleStarted = Date.parse(String(current.startedAt ?? ""));
  const phaseStarted = Date.parse(String(current.phaseStartedAt ?? current.startedAt ?? ""));
  const phaseElapsed = Number.isFinite(phaseStarted) ? (now - phaseStarted) / 1000 : 0;
  return (
    <article className="relative rounded-lg border border-sky-400/30 bg-sky-500/[0.04] px-3 py-2.5">
      <TimelineDot tone="active" />
      <div className="flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2">
          <span className="text-xs font-medium capitalize text-sky-300">
            {String(current.phase ?? current.status ?? "starting")}
          </span>
          <span className="truncate font-mono text-[9px] text-muted-foreground">
            Active cycle · {String(current.id ?? "starting")}
          </span>
        </div>
        <span className="animate-pulse font-mono text-[10px] text-sky-300">
          ● {formatDuration(phaseElapsed)}
          {phaseLimit > 0 ? ` / ${formatDuration(phaseLimit)}` : ""}
        </span>
      </div>
      <p className="mt-1 text-[10px] text-muted-foreground">
        Cycle active for{" "}
        {formatDuration(Number.isFinite(cycleStarted) ? (now - cycleStarted) / 1000 : 0)}. Live
        Stream contains model inference, reasoning, and commands.
      </p>
    </article>
  );
}

function PausedHistoryNode({ current }: { current: JsonRecord | null }) {
  return (
    <article className="relative rounded-lg border border-sky-400/30 bg-sky-500/[0.04] px-3 py-2.5">
      <TimelineDot tone="active" />
      <div className="flex items-center justify-between gap-3">
        <span className="text-xs font-medium text-sky-300">Campaign paused</span>
        <span className="font-mono text-[9px] text-muted-foreground">GPU services stopped</span>
      </div>
      <p className="mt-1 text-[10px] text-muted-foreground">
        Durable workspace state is preserved. Resume starts a fresh discovery phase.
        {current?.pausedAt ? ` Paused at ${String(current.pausedAt)}.` : ""}
      </p>
    </article>
  );
}

function TimelineRow({ entry }: { entry: TimelineEntry }) {
  if (entry.kind === "command") return <CommandCard command={entry} />;
  if (entry.kind === "plan") return <PlanCard items={entry.items} />;
  if (entry.kind === "error") {
    return (
      <div className="rounded-lg border border-destructive/40 bg-destructive/5 px-3 py-2 text-xs text-destructive">
        <span className="mr-2 font-semibold">Error</span>
        {entry.text}
      </div>
    );
  }
  if (entry.kind === "user") {
    return (
      <section className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-accent px-4 py-3 text-sm shadow-sm">
          <div className="mb-1 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
            User
          </div>
          <p className="whitespace-pre-wrap leading-relaxed">{entry.text.trim()}</p>
        </div>
      </section>
    );
  }
  if (entry.kind === "reasoning") {
    return (
      <section className="group flex gap-3 text-sm text-muted-foreground">
        <div className="mt-0.5 flex size-6 shrink-0 items-center justify-center rounded-full border border-border bg-muted/40 text-[10px]">
          ◇
        </div>
        <div className="min-w-0 flex-1 border-l border-border/60 pl-3">
          <div className="mb-1 text-[10px] font-medium uppercase tracking-wider opacity-70">
            Reasoning
          </div>
          <p className="whitespace-pre-wrap leading-relaxed">{entry.text.trim()}</p>
        </div>
      </section>
    );
  }
  return (
    <section className="flex gap-3">
      <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-foreground text-[10px] font-bold text-background">
        Q
      </div>
      <div className="min-w-0 flex-1">
        <div className="mb-1 text-[11px] font-semibold">Qwen</div>
        <div className="whitespace-pre-wrap text-sm leading-6">{entry.text.trim()}</div>
      </div>
    </section>
  );
}

function CommandCard({ command }: { command: Extract<TimelineEntry, { kind: "command" }> }) {
  const [expanded, setExpanded] = useState(true);
  const [followingOutput, setFollowingOutput] = useState(true);
  const outputScroller = useRef<HTMLPreElement>(null);
  const failed = command.status === "failed";

  useEffect(() => {
    if (!expanded || !followingOutput) return;
    const frame = window.requestAnimationFrame(() => {
      const element = outputScroller.current;
      if (element) element.scrollTop = element.scrollHeight;
    });
    return () => window.cancelAnimationFrame(frame);
  }, [command.output, command.status, expanded, followingOutput]);

  const updateOutputFollowing = () => {
    const element = outputScroller.current;
    if (!element) return;
    setFollowingOutput(element.scrollHeight - element.scrollTop - element.clientHeight < 40);
  };

  return (
    <section
      className={
        failed
          ? "overflow-hidden rounded-xl border border-destructive/40 bg-destructive/[0.03]"
          : "overflow-hidden rounded-xl border border-border bg-card/60"
      }
    >
      <button
        type="button"
        className="flex w-full items-center gap-2 px-3 py-2 text-left hover:bg-accent/50"
        onClick={() => setExpanded((value) => !value)}
      >
        <span
          className={
            command.status === "running"
              ? "size-2 animate-pulse rounded-full bg-amber-400"
              : failed
                ? "text-destructive"
                : "text-emerald-400"
          }
        >
          {command.status === "running" ? "" : failed ? "×" : "✓"}
        </span>
        <span className="text-xs font-medium">
          {command.status === "running"
            ? "Running command"
            : failed
              ? `Command failed · exit ${command.exitCode ?? "?"}`
              : "Command completed"}
        </span>
        <code className="min-w-0 flex-1 whitespace-pre-wrap break-words text-[11px] leading-relaxed text-muted-foreground">
          {command.command}
        </code>
        <span className="text-[10px] text-muted-foreground">{expanded ? "▾" : "▸"}</span>
      </button>
      {expanded ? (
        <div className="border-t border-border/70 bg-black/20">
          <div className="border-b border-border/50 px-3 py-1.5 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
            Invocation and output
          </div>
          <pre
            ref={outputScroller}
            onScroll={updateOutputFollowing}
            className="max-h-[34rem] overflow-auto whitespace-pre-wrap break-words p-3 font-mono text-[11px] leading-relaxed text-muted-foreground"
          >
            <span className="select-none text-muted-foreground">$ </span>
            <span className="text-foreground/70">{command.command}</span>
            {command.output ? (
              <span className="text-muted-foreground/80">{`\n\n${command.output}`}</span>
            ) : null}
            {command.status === "running" ? (
              <span className="text-amber-400">{"\n\n▌"}</span>
            ) : null}
          </pre>
        </div>
      ) : null}
    </section>
  );
}

function PlanCard({ items }: { items: Array<{ text: string; completed: boolean }> }) {
  const completed = items.filter((item) => item.completed).length;
  return (
    <section className="rounded-xl border border-border bg-card/40 p-3">
      <div className="mb-2 flex items-center justify-between text-xs font-medium">
        <span>Plan</span>
        <span className="text-muted-foreground">
          {completed}/{items.length}
        </span>
      </div>
      <div className="space-y-1.5">
        {items.map((item, index) => (
          <div key={`${item.text}-${index}`} className="flex gap-2 text-xs">
            <span className={item.completed ? "text-emerald-400" : "text-muted-foreground"}>
              {item.completed ? "✓" : "○"}
            </span>
            <span className={item.completed ? "text-muted-foreground line-through" : ""}>
              {item.text}
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}

function ActivityFooter({
  lastEventAt,
  now,
  source = "Codex",
}: {
  lastEventAt: string | null | undefined;
  now: number;
  source?: string;
}) {
  return (
    <div className="flex items-center gap-2 border-t border-border/50 py-3 text-[11px] text-muted-foreground">
      <span className="size-1.5 rounded-full bg-muted-foreground/50" />
      No active streamed operation · waiting for the next {source} event · last event{" "}
      {formatAge(lastEventAt, now)}
    </div>
  );
}

function EmptyState({ conversation = false, label = "the agent" }: {
  conversation?: boolean;
  label?: string;
}) {
  return (
    <div className="m-auto rounded-xl border border-dashed border-border p-8 text-center text-sm text-muted-foreground">
      {conversation
        ? `Waiting for the ${label} session…`
        : "Waiting for an lm-tools campaign stream…"}
    </div>
  );
}
