import { useEffect, useMemo, useRef, useState } from "react";

type JsonRecord = Record<string, unknown>;
type ObserverResponse = {
  configured: boolean;
  campaign: JsonRecord | null;
  current: JsonRecord | null;
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
  | { key: string; kind: "assistant" | "reasoning" | "error"; text: string }
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
    kind: "assistant" | "reasoning",
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
    const command: TimelineEntry = {
      key: id,
      kind: "command",
      command: String(item.command ?? "command"),
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
  const entries = useMemo(() => timelineEntries(data.events), [data.events]);
  const usage = record(data.campaign?.usage);
  const status = String(data.current?.status ?? "not started");
  const running = status === "running" || status === "starting";
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

  useEffect(() => {
    if (!following) return;
    const frame = window.requestAnimationFrame(() => {
      const element = scroller.current;
      if (element) element.scrollTop = element.scrollHeight;
    });
    return () => window.cancelAnimationFrame(frame);
  }, [entries.length, data.stream?.bytes, following]);

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
        <div className="mx-auto flex max-w-5xl items-center justify-between gap-4">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h1 className="truncate text-sm font-semibold">PS2 Linux autonomous agent</h1>
              <ConnectionBadge connection={connection} />
            </div>
            <p className="mt-0.5 truncate font-mono text-[11px] text-muted-foreground">
              {String(data.current?.id ?? "Waiting for a run")}
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-4 text-xs">
            <HeaderMetric label="Turns" value={String(number(data.campaign?.runs))} />
            <HeaderMetric label="Tokens" value={compactCount(number(usage?.totalTokens))} />
            <ContextMeter inference={data.inference} />
            <span className={running ? "text-amber-400" : "text-emerald-400"}>
              ● {running ? "run active" : status}
            </span>
          </div>
        </div>
      </header>

      <div ref={scroller} className="relative flex-1 overflow-y-auto" onScroll={updateFollowing}>
        <div className="mx-auto flex min-h-full max-w-4xl flex-col px-4 py-6 sm:px-8">
          <TurnIntro current={data.current} />
          {!data.configured ? (
            <EmptyState />
          ) : (
            <div className="space-y-4">
              {entries.map((entry) => (
                <TimelineRow key={entry.key} entry={entry} />
              ))}
              {data.inference?.active && (data.inference.phase === "prompt" || !streamingDelta) ? (
                <InferenceActivity inference={data.inference} />
              ) : null}
              {running && !activeCommand && !streamingDelta && !data.inference?.active ? (
                <ActivityFooter lastEventAt={data.stream?.lastEventAt} now={now} />
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
    </main>
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

function TurnIntro({ current }: { current: JsonRecord | null }) {
  return (
    <section className="mb-7 flex justify-end">
      <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-accent px-4 py-3 text-sm shadow-sm">
        <p className="font-medium">Continue the PS2 Linux kernel investigation autonomously.</p>
        <p className="mt-1 font-mono text-[10px] text-muted-foreground">
          Handoff: {String(current?.prompt ?? "waiting")}
        </p>
      </div>
    </section>
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
        <code className="min-w-0 flex-1 truncate text-[11px] text-muted-foreground">
          {command.command}
        </code>
        <span className="text-[10px] text-muted-foreground">{expanded ? "▾" : "▸"}</span>
      </button>
      {expanded ? (
        <div className="border-t border-border/70 bg-black/20">
          <div className="border-b border-border/50 px-3 py-1.5 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
            Shell
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
}: {
  lastEventAt: string | null | undefined;
  now: number;
}) {
  return (
    <div className="flex items-center gap-2 border-t border-border/50 py-3 text-[11px] text-muted-foreground">
      <span className="size-1.5 rounded-full bg-muted-foreground/50" />
      No active streamed operation · waiting for the next Codex event · last event{" "}
      {formatAge(lastEventAt, now)}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="m-auto rounded-xl border border-dashed border-border p-8 text-center text-sm text-muted-foreground">
      Waiting for an lm-tools campaign stream…
    </div>
  );
}
