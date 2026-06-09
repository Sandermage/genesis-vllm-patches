import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { ArrowRight, Bot, BookText, ChevronDown, CircleAlert, Copy, Database, Download, Heart, Loader2, Pencil, Plus, RefreshCw, Route, Search, Send, SlidersHorizontal, Sparkles, Square, TimerReset, Trash2, User, X, Zap } from "lucide-react";
import type { ReactNode } from "react";
import { EngineBenchResult, EngineChatResult, EngineMetrics, EngineStatus, HubModel, Job, ModelCacheReport, RagDoc, type RoutingActive, type RoutingClassify, type RoutingSignals, api } from "./api";
import { SkeletonMetrics } from "./Skeleton";
import { tr } from "./i18n";

function fmtCount(n: number | null): string {
  if (n === null || n === undefined) return "—";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

/** Poll a fetcher on an interval; returns latest data + a manual reload. */
function usePoll<T>(fetcher: () => Promise<T>, intervalMs: number, enabled = true) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const saved = useRef(fetcher);
  saved.current = fetcher;
  const reload = () => {
    saved.current().then((value) => { setData(value); setLoading(false); }).catch(() => setLoading(false));
  };
  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    const tick = () => saved.current().then((value) => { if (!cancelled) { setData(value); setLoading(false); } }).catch(() => { if (!cancelled) setLoading(false); });
    tick();
    const id = setInterval(tick, intervalMs);
    return () => { cancelled = true; clearInterval(id); };
  }, [intervalMs, enabled]);
  return { data, loading, reload };
}

/** Live engine reachability, loaded model and version. */
export function EngineStatusCard({ host }: { host?: string }) {
  const { data, loading } = usePoll<EngineStatus>(() => api.engineStatus(host), 5000);
  const up = Boolean(data?.reachable);
  return (
    <div className={`engine-status ${up ? "up" : "down"}`}>
      <div className="engine-status-head">
        <span className={`engine-dot ${up ? "up" : "down"}`} />
        <div>
          <strong>{loading && !data ? tr("Probing engine…") : up ? tr("Engine online") : tr("Engine offline")}</strong>
          <small>{data?.base_url ?? "—"}</small>
        </div>
        {up && data?.version && <span className="engine-version">v{data.version}</span>}
      </div>
      {up ? (
        <div className="engine-models">
          {(data?.models ?? []).length ? (
            (data?.models ?? []).map((model) => (
              <span className="chip" key={model}>{model}</span>
            ))
          ) : (
            <span className="muted">{tr("No model reported by /v1/models.")}</span>
          )}
        </div>
      ) : (
        <p className="muted">
          {data?.error ? `${data.error}. ` : ""}{tr("Start the runtime (Launch Plan) — this reads the live OpenAI server.")}
        </p>
      )}
    </div>
  );
}

const KPI_SPECS: Array<{ key: string; label: string; fmt: (v: number) => string; tone?: (v: number) => string }> = [
  { key: "requests_running", label: "Running", fmt: (v) => String(Math.round(v)) },
  { key: "requests_waiting", label: "Waiting", fmt: (v) => String(Math.round(v)), tone: (v) => (v > 0 ? "warn" : "") },
  { key: "kv_cache_usage", label: "KV cache", fmt: (v) => `${Math.round(v * 100)}%`, tone: (v) => (v > 0.9 ? "warn" : "") },
  { key: "generation_toks_per_s", label: "Throughput", fmt: (v) => `${v} tok/s` },
  { key: "ttft_avg_s", label: "TTFT avg", fmt: (v) => `${Math.round(v * 1000)} ms` },
  { key: "tpot_avg_s", label: "TPOT avg", fmt: (v) => `${(v * 1000).toFixed(1)} ms` },
  { key: "spec_decode_acceptance_rate", label: "Spec accept", fmt: (v) => `${Math.round(v * 100)}%`, tone: () => "ok" },
  { key: "requests_success_total", label: "Succeeded", fmt: (v) => v.toLocaleString() }
];

/** Dependency-free inline SVG sparkline. Null samples are skipped. */
function Sparkline({ values, color }: { values: Array<number | null>; color: string }) {
  const points = values.filter((v): v is number => v !== null && !Number.isNaN(v));
  if (points.length < 2) return <span className="sparkline-empty muted">{tr("collecting…")}</span>;
  const min = Math.min(...points);
  const max = Math.max(...points);
  const range = max - min || 1;
  const w = 120;
  const h = 26;
  const step = w / (points.length - 1);
  const d = points
    .map((v, i) => `${i === 0 ? "M" : "L"}${(i * step).toFixed(1)},${(h - ((v - min) / range) * h).toFixed(1)}`)
    .join(" ");
  return (
    <svg className="sparkline" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-hidden="true">
      <path d={d} fill="none" stroke={color} strokeWidth={1.6} vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function TrendRow({ label, values, unit, color }: { label: string; values: Array<number | null>; unit: string; color: string }) {
  const latest = [...values].reverse().find((v) => v !== null && !Number.isNaN(v as number));
  return (
    <div className="engine-trend">
      <div className="engine-trend-head">
        <span>{tr(label)}</span>
        <strong>{latest === undefined ? "—" : `${Math.round((latest as number) * 10) / 10}${unit}`}</strong>
      </div>
      <Sparkline values={values} color={color} />
    </div>
  );
}

/** Live Prometheus KPIs distilled for operators. */
export function EngineMetricsPanel({ host }: { host?: string }) {
  const { data, loading, reload } = usePoll<EngineMetrics>(() => api.engineMetrics(host), 3000);
  if (loading && !data) {
    return <SkeletonMetrics count={6} />;
  }
  if (!data?.reachable) {
    return (
      <div className="engine-offline">
        <CircleAlert size={18} />
        <div>
          <strong>{tr("Metrics unavailable")}</strong>
          <p className="muted">{data?.error ?? tr("Engine /metrics endpoint is not reachable.")}</p>
        </div>
      </div>
    );
  }
  const kpis = data.kpis ?? {};
  const tiles = KPI_SPECS.filter((spec) => kpis[spec.key] !== undefined);
  return (
    <div className="engine-metrics">
      <div className="engine-metrics-head">
        <span className="muted">{data.metric_families ?? 0} {tr("metric families")} · {tr("refreshing every 3s")}</span>
        <button className="ghost-button" onClick={reload}><RefreshCw size={13} /> {tr("Now")}</button>
      </div>
      <div className="engine-kpis">
        {tiles.map((spec) => {
          const value = kpis[spec.key];
          const tone = spec.tone ? spec.tone(value) : "";
          return (
            <div className={`engine-kpi ${tone}`} key={spec.key}>
              <span className="engine-kpi-value">{spec.fmt(value)}</span>
              <span className="engine-kpi-label">{tr(spec.label)}</span>
            </div>
          );
        })}
      </div>
      {(data.history?.length ?? 0) > 1 && (
        <div className="engine-trends">
          <TrendRow label="Throughput" unit=" tok/s" color="var(--accent)" values={(data.history ?? []).map((s) => s.throughput)} />
          <TrendRow label="KV cache" unit="%" color="var(--warn, #f59e0b)" values={(data.history ?? []).map((s) => (s.kv_cache === null ? null : s.kv_cache * 100))} />
          <TrendRow label="Queue" unit="" color="var(--info, #2563eb)" values={(data.history ?? []).map((s) => (s.running ?? 0) + (s.waiting ?? 0))} />
        </div>
      )}
    </div>
  );
}

/** Model checkpoint cache + queue a download (sndr model pull) as a job. */
export function ModelManagementPanel() {
  const [data, setData] = useState<ModelCacheReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [job, setJob] = useState<Job | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hubQuery, setHubQuery] = useState("");
  const [hubResults, setHubResults] = useState<HubModel[] | null>(null);
  const [hubBusy, setHubBusy] = useState(false);
  const reload = () => {
    setLoading(true);
    api.modelsCache().then((d) => { setData(d); setLoading(false); }).catch(() => setLoading(false));
  };
  useEffect(() => { reload(); }, []);

  const runDownload = async (fn: () => Promise<Job>, key: string) => {
    setBusyId(key);
    setError(null);
    setJob(null);
    try {
      setJob(await fn());
    } catch (err) {
      setError(err instanceof Error ? err.message : tr("Failed to queue download."));
    } finally {
      setBusyId(null);
    }
  };
  const download = (modelId: string) => runDownload(() => api.modelsDownload(modelId), modelId);
  const downloadRepo = (repoId: string) => runDownload(() => api.downloadRepo(repoId), repoId);

  const searchHub = async (event: FormEvent) => {
    event.preventDefault();
    setHubBusy(true);
    setError(null);
    try {
      setHubResults((await api.hubSearch(hubQuery.trim(), 20)).results);
    } catch (err) {
      setError(err instanceof Error ? err.message : tr("Hugging Face search failed."));
    } finally {
      setHubBusy(false);
    }
  };

  // Poll a running download job for live progress until it finishes.
  useEffect(() => {
    if (!job || job.status !== "running") return;
    const id = setInterval(async () => {
      try {
        const fresh = await api.job(job.job_id);
        setJob(fresh);
        if (fresh.status !== "running") { clearInterval(id); reload(); }
      } catch { /* keep polling */ }
    }, 1500);
    return () => clearInterval(id);
  }, [job?.job_id, job?.status]); // eslint-disable-line react-hooks/exhaustive-deps

  const models = data?.models ?? [];
  return (
    <div className="model-mgmt">
      <div className="model-mgmt-head">
        <span className="muted">
          {loading ? tr("Reading cache…") : `${data?.present_count ?? 0}/${models.length} ${tr("present on the daemon host")}`}
        </span>
        <button className="ghost-button" onClick={reload}><RefreshCw size={13} /> {tr("Refresh")}</button>
      </div>
      {error && <div className="login-error"><CircleAlert size={14} /> <span>{error}</span></div>}
      <table className="module-table model-mgmt-table">
        <thead>
          <tr><th>{tr("Model")}</th><th>{tr("Cache path")}</th><th>{tr("Size")}</th><th>{tr("Status")}</th><th /></tr>
        </thead>
        <tbody>
          {models.map((entry) => (
            <tr key={entry.model_id}>
              <td><strong>{entry.model_id}</strong></td>
              <td className="model-path">{entry.model_path}</td>
              <td>{entry.size_mib !== null ? `${(entry.size_mib / 1024).toFixed(1)} GiB` : "—"}</td>
              <td>
                <span className={`dl-status ${entry.present ? "present" : "absent"}`}>
                  {entry.present ? tr("present") : tr("absent")}
                </span>
              </td>
              <td className="preset-row-actions">
                {!entry.present && (
                  <button className="ghost-button" onClick={() => download(entry.model_id)} disabled={busyId === entry.model_id}>
                    {busyId === entry.model_id ? <Loader2 size={13} className="spin" /> : <Download size={13} />} {tr("Download")}
                  </button>
                )}
              </td>
            </tr>
          ))}
          {models.length === 0 && !loading && (
            <tr><td colSpan={5} className="muted">{tr("No models declared in the catalog.")}</td></tr>
          )}
        </tbody>
      </table>

      <div className="hub-search">
        <h4><Search size={14} /> {tr("Hugging Face Hub")}</h4>
        <form className="hub-search-bar" onSubmit={searchHub}>
          <input
            aria-label={tr("Search Hugging Face Hub")}
            value={hubQuery}
            onChange={(event) => setHubQuery(event.target.value)}
            placeholder={tr("Search models on huggingface.co (e.g. qwen3, llama)…")}
          />
          <button className="primary-button" type="submit" disabled={hubBusy}>
            {hubBusy ? <Loader2 size={14} className="spin" /> : <Search size={14} />} {tr("Search")}
          </button>
        </form>
        {hubResults && (
          hubResults.length ? (
            <div className="hub-results">
              {hubResults.map((model) => (
                <div className="hub-result" key={model.id}>
                  <div className="hub-result-id">
                    <a href={`https://huggingface.co/${model.id}`} target="_blank" rel="noreferrer">{model.id}</a>
                    {model.gated && <span className="chip hub-gated">{tr("gated")}</span>}
                  </div>
                  <div className="hub-result-meta">
                    <span><Download size={11} /> {fmtCount(model.downloads)}</span>
                    <span><Heart size={11} /> {fmtCount(model.likes)}</span>
                    {model.pipeline_tag && <span>{model.pipeline_tag}</span>}
                  </div>
                  <button className="ghost-button" onClick={() => downloadRepo(model.id)} disabled={busyId === model.id}>
                    {busyId === model.id ? <Loader2 size={13} className="spin" /> : <Download size={13} />} {tr("Download")}
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <p className="muted">{tr("No models found for")} “{hubQuery}”.</p>
          )
        )}
      </div>

      {job && (
        <div className="dl-job">
          <div className="dl-job-head">
            <strong>{job.job_id}</strong>
            <span className={`chip dl-${job.status}`}>
              {job.dry_run ? tr("dry-run queued") : job.status}
              {job.status === "running" && <Loader2 size={11} className="spin" />}
            </span>
          </div>
          {job.status === "running" && typeof job.progress === "number" && (
            <div className="dl-progress"><div className="dl-progress-bar" style={{ width: `${job.progress}%` }} /><span>{Math.round(job.progress)}%</span></div>
          )}
          <pre className="dl-job-cmd">{(job.dry_run ? job.cli_mirror : job.log)?.slice(-200).join("\n")}</pre>
          {job.note && <p className="muted">{job.note}</p>}
        </div>
      )}
    </div>
  );
}

/** A real chat smoke-test against the running engine. */
export function EnginePlayground({ host, models }: { host?: string; models?: string[] }) {
  const [prompt, setPrompt] = useState("Say hello in one short sentence.");
  const [model, setModel] = useState("");
  const [maxTokens, setMaxTokens] = useState(128);
  const [temperature, setTemperature] = useState(0.7);
  const [result, setResult] = useState<EngineChatResult | null>(null);
  const [streamText, setStreamText] = useState("");
  const [streamMeta, setStreamMeta] = useState<{ tokens?: number; latency_ms?: number; ttft_ms?: number } | null>(null);
  const [stream, setStream] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!model && models && models.length) setModel(models[0]);
  }, [models, model]);

  const send = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    setResult(null);
    setStreamText("");
    setStreamMeta(null);
    const payload = { messages: [{ role: "user", content: prompt }], model: model || undefined, max_tokens: maxTokens, temperature, host };
    try {
      if (stream) {
        await api.engineChatStream(payload, {
          onDelta: (text) => setStreamText((prev) => prev + text),
          onDone: (meta) => setStreamMeta(meta),
          onError: (msg) => setError(msg)
        });
      } else {
        setResult(await api.engineChat(payload));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : tr("Request failed."));
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className="engine-playground" onSubmit={send}>
      <label className="field">
        <span>{tr("Prompt")}</span>
        <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} rows={3} />
      </label>
      <div className="engine-pg-controls">
        {models && models.length > 0 && (
          <label className="field">
            <span>{tr("Model")}</span>
            <select value={model} onChange={(event) => setModel(event.target.value)}>
              {models.map((m) => <option key={m} value={m}>{m}</option>)}
            </select>
          </label>
        )}
        <label className="field">
          <span>{tr("Max tokens")}</span>
          <input type="number" min={1} max={4096} value={maxTokens} onChange={(event) => setMaxTokens(Number(event.target.value) || 128)} />
        </label>
        <label className="field">
          <span>{tr("Temperature")}</span>
          <input type="number" min={0} max={2} step={0.1} value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} />
        </label>
        <label className="pg-stream-toggle" title={tr("Stream tokens as they generate")}>
          <input type="checkbox" checked={stream} onChange={(e) => setStream(e.target.checked)} /> {tr("Stream")}
        </label>
        <button className="primary-button" type="submit" disabled={busy || !prompt.trim()}>
          {busy ? <Loader2 size={15} className="spin" /> : <Send size={15} />}
          {busy ? (stream ? tr("Streaming…") : tr("Sending…")) : tr("Send")}
        </button>
      </div>
      {error && <div className="login-error"><CircleAlert size={14} /> <span>{error}</span></div>}
      {(streamText || streamMeta || result) && (
        <div className="engine-pg-result">
          <div className="engine-pg-reply">
            {(stream ? streamText : result?.reply) || (busy ? <span className="muted">…</span> : <span className="muted">{tr("(empty response)")}</span>)}
            {stream && busy && <span className="pg-caret" />}
          </div>
          <div className="engine-pg-meta">
            {stream ? (
              streamMeta && (
                <>
                  {streamMeta.latency_ms !== undefined && <span><Zap size={12} /> {streamMeta.latency_ms} ms</span>}
                  {streamMeta.ttft_ms !== undefined && <span>{streamMeta.ttft_ms} ms TTFT</span>}
                  {streamMeta.tokens !== undefined && <span>{streamMeta.tokens} {tr("tokens")}</span>}
                </>
              )
            ) : result ? (
              <>
                <span><Zap size={12} /> {result.latency_ms} ms</span>
                {result.usage?.total_tokens !== undefined && <span>{result.usage.total_tokens} {tr("tokens")}</span>}
                {result.usage?.prompt_tokens !== undefined && <span>{result.usage.prompt_tokens} {tr("prompt")}</span>}
                {result.usage?.completion_tokens !== undefined && <span>{result.usage.completion_tokens} {tr("output")}</span>}
                {result.finish_reason && <span>{tr("finish")}: {result.finish_reason}</span>}
              </>
            ) : null}
          </div>
        </div>
      )}
    </form>
  );
}

// Bench metric rows: (label, getter, "higher" | "lower" = which direction is better).
const BENCH_ROWS: Array<{ label: string; get: (m: EngineBenchResult["metrics"]) => number | null; unit: string; better: "higher" | "lower" }> = [
  { label: "Throughput", get: (m) => m.throughput_tok_s, unit: "tok/s", better: "higher" },
  { label: "TTFT p50", get: (m) => m.ttft_p50_ms, unit: "ms", better: "lower" },
  { label: "TTFT p90", get: (m) => m.ttft_p90_ms, unit: "ms", better: "lower" },
  { label: "TPOT avg", get: (m) => m.tpot_avg_ms, unit: "ms", better: "lower" },
  { label: "CV", get: (m) => m.cv_pct, unit: "%", better: "lower" },
  { label: "Tokens", get: (m) => m.total_tokens, unit: "", better: "higher" }
];

function deltaCell(a: number | null, b: number | null, better: "higher" | "lower") {
  if (a === null || b === null || a === 0) return <td className="bench-delta muted">—</td>;
  const pct = ((b - a) / Math.abs(a)) * 100;
  const improved = better === "higher" ? pct > 0 : pct < 0;
  const tone = Math.abs(pct) < 0.5 ? "" : improved ? "good" : "bad";
  return <td className={`bench-delta ${tone}`}>{pct > 0 ? "+" : ""}{pct.toFixed(1)}%</td>;
}

/** Real micro-benchmark driven against the running engine, with A/B compare. */
export function EngineBenchPanel({ host, referenceTps }: { host?: string; referenceTps?: number | null }) {
  const [numRequests, setNumRequests] = useState(8);
  const [concurrency, setConcurrency] = useState(2);
  const [maxTokens, setMaxTokens] = useState(128);
  const [runA, setRunA] = useState<EngineBenchResult | null>(null);
  const [runB, setRunB] = useState<EngineBenchResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      const out = await api.engineBench({ num_requests: numRequests, concurrency, max_tokens: maxTokens, host });
      if (!runA) setRunA(out);
      else setRunB(out);
    } catch (err) {
      setError(err instanceof Error ? err.message : tr("Benchmark failed."));
    } finally {
      setBusy(false);
    }
  };
  const promoteBtoA = () => { if (runB) { setRunA(runB); setRunB(null); } };
  const reset = () => { setRunA(null); setRunB(null); setError(null); };

  return (
    <div className="engine-bench">
      <div className="bench-controls">
        <label className="field"><span>{tr("Requests")}</span>
          <input type="number" min={1} max={64} value={numRequests} onChange={(e) => setNumRequests(Number(e.target.value) || 8)} /></label>
        <label className="field"><span>{tr("Concurrency")}</span>
          <input type="number" min={1} max={16} value={concurrency} onChange={(e) => setConcurrency(Number(e.target.value) || 2)} /></label>
        <label className="field"><span>{tr("Max tokens")}</span>
          <input type="number" min={1} max={2048} value={maxTokens} onChange={(e) => setMaxTokens(Number(e.target.value) || 128)} /></label>
        <button className="primary-button" onClick={run} disabled={busy}>
          {busy ? <Loader2 size={15} className="spin" /> : <TimerReset size={15} />}
          {busy ? tr("Running…") : runA ? tr("Run B") : tr("Run benchmark")}
        </button>
        {(runA || runB) && <button className="ghost-button" onClick={reset}>{tr("Reset")}</button>}
      </div>
      <p className="bench-method muted">
        {tr("Live quick-bench against the running engine — real TTFT/TPOT/throughput. Not the canonical Wave suite (different prompt set); use A/B for like-for-like deltas.")}
        {typeof referenceTps === "number" && referenceTps > 0 && (
          <> {tr("Card reference:")} <strong>{referenceTps} tok/s</strong> {tr("(canonical suite)")}.</>
        )}
      </p>
      {error && <div className="login-error"><CircleAlert size={14} /> <span>{error}</span></div>}

      {runA && (
        <table className="bench-table">
          <thead>
            <tr>
              <th>{tr("Metric")}</th>
              <th>A {runA && <small>({runA.params.num_requests}×c{runA.params.concurrency})</small>}</th>
              {runB && <th>B <small>({runB.params.num_requests}×c{runB.params.concurrency})</small></th>}
              {runB && <th>Δ</th>}
            </tr>
          </thead>
          <tbody>
            {BENCH_ROWS.map((row) => {
              const a = row.get(runA.metrics);
              const b = runB ? row.get(runB.metrics) : null;
              return (
                <tr key={row.label}>
                  <td>{tr(row.label)}</td>
                  <td className="bench-val">{a ?? "—"}{a !== null && row.unit ? ` ${row.unit}` : ""}</td>
                  {runB && <td className="bench-val">{b ?? "—"}{b !== null && row.unit ? ` ${row.unit}` : ""}</td>}
                  {runB && deltaCell(a, b, row.better)}
                </tr>
              );
            })}
            <tr>
              <td>{tr("Requests")}</td>
              <td className="bench-val">{runA.metrics.requests_ok} {tr("ok")}{runA.metrics.requests_failed ? ` · ${runA.metrics.requests_failed} ${tr("fail")}` : ""}</td>
              {runB && <td className="bench-val">{runB.metrics.requests_ok} {tr("ok")}{runB.metrics.requests_failed ? ` · ${runB.metrics.requests_failed} ${tr("fail")}` : ""}</td>}
              {runB && <td className="bench-delta muted">—</td>}
            </tr>
          </tbody>
        </table>
      )}
      {runB && (
        <button className="ghost-button bench-promote" onClick={promoteBtoA}>
          <ArrowRight size={13} /> {tr("Use B as the new baseline (A)")}
        </button>
      )}
    </div>
  );
}


// ── Markdown-lite renderer (bold / italic / inline code / links) ──────────
function inlineMd(text: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*\s][^*]*\*|\[[^\]]+\]\([^)]+\))/g;
  let last = 0;
  let key = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) nodes.push(text.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith("**")) nodes.push(<strong key={key++}>{tok.slice(2, -2)}</strong>);
    else if (tok.startsWith("`")) nodes.push(<code className="md-inline" key={key++}>{tok.slice(1, -1)}</code>);
    else if (tok.startsWith("*")) nodes.push(<em key={key++}>{tok.slice(1, -1)}</em>);
    else { const mm = tok.match(/\[([^\]]+)\]\(([^)]+)\)/); if (mm) { const url = mm[2].trim(); const safe = /^https?:\/\//i.test(url); nodes.push(safe ? <a key={key++} href={url} target="_blank" rel="noreferrer">{mm[1]}</a> : <span key={key++}>{mm[1]}</span>); } }
    last = m.index + tok.length;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes;
}

function MarkdownLite({ text }: { text: string }) {
  const blocks = text.split(/```/);
  return (
    <>
      {blocks.map((block, bi) => {
        if (bi % 2 === 1) {
          const nl = block.indexOf("\n");
          const first = nl >= 0 ? block.slice(0, nl).trim() : "";
          const lang = first.length > 0 && first.length < 16 && !/\s/.test(first) ? first : "";
          const code = (lang ? block.slice(nl + 1) : block).replace(/\n$/, "");
          return (
            <div className="md-code" key={bi}>
              <div className="md-code-head"><span>{lang || tr("code")}</span><button className="icon-only" title={tr("Copy")} onClick={() => void navigator.clipboard?.writeText(code)}><Copy size={11} /></button></div>
              <pre>{code}</pre>
            </div>
          );
        }
        const lines = block.split("\n");
        const out: ReactNode[] = [];
        let list: string[] = [];
        const flush = (k: string) => { if (list.length) { out.push(<ul className="md-ul" key={k}>{list.map((li, i) => <li key={i}>{inlineMd(li)}</li>)}</ul>); list = []; } };
        lines.forEach((line, i) => {
          const h = line.match(/^(#{1,3})\s+(.*)/);
          const bullet = line.match(/^\s*[-*]\s+(.*)/);
          const num = line.match(/^\s*\d+\.\s+(.*)/);
          const quote = line.match(/^>\s?(.*)/);
          if (h) { flush(`fl${bi}-${i}`); out.push(<div className={`md-h md-h${h[1].length}`} key={`${bi}-${i}`}>{inlineMd(h[2])}</div>); }
          else if (bullet) { list.push(bullet[1]); }
          else if (num) { list.push(num[1]); }
          else if (quote) { flush(`fl${bi}-${i}`); out.push(<blockquote className="md-quote" key={`${bi}-${i}`}>{inlineMd(quote[1])}</blockquote>); }
          else if (line.trim()) { flush(`fl${bi}-${i}`); out.push(<p className="md-p" key={`${bi}-${i}`}>{inlineMd(line)}</p>); }
        });
        flush(`fl${bi}-end`);
        return <div key={bi}>{out}</div>;
      })}
    </>
  );
}

type ChatStat = { tokens?: number; tps?: number; ttft_ms?: number; latency_ms?: number };
type ChatMessage = { role: "user" | "assistant"; content: string; stat?: ChatStat; sources?: RagDoc[] };
type Conversation = { id: string; title: string; messages: ChatMessage[]; createdAt: number; updatedAt: number };
type ChatSettings = { host: string; port: number; model: string; apiKey: string; hostId: string; system: string; temperature: number; maxTokens: number; topP: number; presencePenalty: number; frequencyPenalty: number; stop: string; thinking: boolean; useProject: boolean; ragProject: boolean; ragVaults: string[]; workloadClass: string };

const CHAT_KEY = "sndr.chat.v1";
const DEFAULT_SETTINGS: ChatSettings = { host: "127.0.0.1", port: 8000, model: "", apiKey: "", hostId: "", system: "You are a helpful assistant.", temperature: 0.7, maxTokens: 512, topP: 1, presencePenalty: 0, frequencyPenalty: 0, stop: "", thinking: false, useProject: false, ragProject: true, ragVaults: [], workloadClass: "" };

// Build the grounding system message from retrieved project-knowledge docs.
function buildRagContext(docs: RagDoc[]): string {
  const block = docs.map((d, i) => `[${i + 1}] (${d.ref}) ${d.snippet}`).join("\n");
  return (
    "Project knowledge retrieved from the Genesis vLLM patch project " +
    "(patch registry, presets, configs). Ground your answer in these facts " +
    "and cite the [n] source labels where relevant. If they don't cover the " +
    "question, say so and answer from general knowledge.\n\n" + block
  );
}
const SUGGESTIONS = ["Explain what this model is good at.", "Write a Python function to parse JSON.", "Summarize the pros of speculative decoding.", "Give me a haiku about GPUs."];

function newConversation(): Conversation {
  return { id: `c-${Math.random().toString(36).slice(2, 9)}`, title: "New chat", messages: [], createdAt: Date.now(), updatedAt: Date.now() };
}
function loadChatState(): { conversations: Conversation[]; activeId: string; settings: ChatSettings } {
  try {
    const raw = window.localStorage.getItem(CHAT_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed.conversations) && parsed.conversations.length) {
        return { conversations: parsed.conversations, activeId: parsed.activeId || parsed.conversations[0].id, settings: { ...DEFAULT_SETTINGS, ...(parsed.settings || {}) } };
      }
    }
  } catch { /* fall through */ }
  const c = newConversation();
  return { conversations: [c], activeId: c.id, settings: { ...DEFAULT_SETTINGS } };
}

// Mirror App.tsx's toast bus without importing across the module boundary.
function chatToast(message: string, tone: "info" | "ok" | "warn" | "danger" = "info") {
  window.dispatchEvent(new CustomEvent("sndr-toast", { detail: { message, tone, id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}` } }));
}

const PROJECT_SUGGESTIONS = [
  "What does patch PN95 do?",
  "Which preset should I use for 27B long context?",
  "List the patches that touch the KV cache.",
  "Explain the MTP speculative-decoding setup.",
];

// Citation chips for RAG-grounded answers; click a chip to expand its snippet.
function SourcesRow({ docs }: { docs: RagDoc[] }) {
  const [open, setOpen] = useState<string | null>(null);
  const kindClass = (k: string) => (k === "patch" ? "src-patch" : k === "preset" ? "src-preset" : k === "note" ? "src-note" : "src-config");
  return (
    <div className="chat-sources">
      <span className="chat-sources-label"><BookText size={12} /> {tr("Grounded in")} {docs.length} {docs.length > 1 ? tr("project sources") : tr("project source")}</span>
      <div className="chat-sources-chips">
        {docs.map((d, i) => (
          <button key={d.ref + i} className={`chat-src ${kindClass(d.kind)} ${open === d.ref ? "open" : ""}`} onClick={() => setOpen(open === d.ref ? null : d.ref)} title={d.title}>
            <span className="chat-src-n">{i + 1}</span>{d.ref}
          </button>
        ))}
      </div>
      {open && (() => { const d = docs.find((x) => x.ref === open); return d ? <div className="chat-src-detail"><strong>{d.title}</strong><p>{d.snippet}</p></div> : null; })()}
    </div>
  );
}

export type ChatTarget = { host: string; port: number; apiKey?: string; hostId?: string; model?: string; nonce: number };

// Full-featured streaming chat for any running local vLLM model.
// Inline routing awareness for the chat: tag a workload class and see, live, how
// the active spec-decode profile would route it (same brain as the gateway). The
// whole strip renders nothing when routing is unavailable, so the chat is never
// affected on deployments without the spec_decode layer.
function ChatRoutingHint({ value, onChange }: { value: string; onChange: (w: string) => void }) {
  const [active, setActive] = useState<RoutingActive | null>(null);
  const [res, setRes] = useState<RoutingClassify | null>(null);

  useEffect(() => {
    let alive = true;
    api.routingActive().then((x) => { if (alive) setActive(x); }).catch(() => { if (alive) setActive({ available: false }); });
    return () => { alive = false; };
  }, []);

  const profile = active?.profile ?? undefined;
  const classes = active?.artifact?.workload_classes ?? [];

  useEffect(() => {
    if (!active?.available || !profile) { setRes(null); return; }
    const signals: RoutingSignals = value ? { workload_class: value } : {};
    let alive = true;
    api.routingClassify(signals, profile).then((r) => { if (alive) setRes(r); }).catch(() => {});
    return () => { alive = false; };
  }, [active, profile, value]);

  if (!active?.available || !profile) return null;
  const accepted = res?.accepted === true;
  const delta = res?.expected_delta_tps;
  return (
    <div className="chat-routing">
      <span className="chat-routing-l"><Route size={13} /> {tr("Workload")}</span>
      <select value={value} onChange={(e) => onChange(e.target.value)} title={tr("Tag this chat's workload class to preview spec-decode routing")}>
        <option value="">{tr("untagged")}</option>
        {classes.map((c) => <option key={c} value={c}>{c}</option>)}
      </select>
      {res && (
        <span className={`chat-routing-verdict ${accepted ? "ok" : "fallback"}`} title={res.reason}>
          <ArrowRight size={12} /> {accepted ? res.profile : tr("fallback (MTP off)")}
          {typeof delta === "number" && <em className={delta >= 0 ? "ok" : "hot"}>{delta > 0 ? "+" : ""}{(delta * 100).toFixed(0)}% TPS</em>}
        </span>
      )}
    </div>
  );
}

export function ChatConsole({ defaultHost, target }: { defaultHost?: string; target?: ChatTarget | null }) {
  const initial = useRef(loadChatState());
  const [conversations, setConversations] = useState<Conversation[]>(initial.current.conversations);
  const [activeId, setActiveId] = useState<string>(initial.current.activeId);
  const [settings, setSettings] = useState<ChatSettings>({ ...initial.current.settings, host: initial.current.settings.host === "127.0.0.1" && defaultHost ? defaultHost : initial.current.settings.host });
  const [status, setStatus] = useState<EngineStatus | null>(null);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [retrieving, setRetrieving] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [atBottom, setAtBottom] = useState(true);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const activeIdRef = useRef(activeId);
  activeIdRef.current = activeId;

  const active = conversations.find((c) => c.id === activeId) ?? conversations[0];
  // Stable ref (recomputes only when the active conversation changes) so the
  // scroll-to-bottom effect's dependency doesn't churn every render.
  const messages = useMemo(() => active?.messages ?? [], [active]);
  const set = (patch: Partial<ChatSettings>) => setSettings((prev) => ({ ...prev, ...patch }));

  useEffect(() => {
    // Never persist the engine API key to localStorage (XSS-extractable secret);
    // it lives only in memory for the session.
    try { window.localStorage.setItem(CHAT_KEY, JSON.stringify({ conversations, activeId, settings: { ...settings, apiKey: "" } })); } catch { /* quota */ }
  }, [conversations, activeId, settings]);

  async function refreshStatus(over?: { host?: string; port?: number; apiKey?: string; hostId?: string }) {
    const host = over?.host ?? settings.host;
    const port = over?.port ?? settings.port;
    const key = over?.apiKey ?? settings.apiKey;
    const hostId = over?.hostId ?? settings.hostId;
    try {
      // host_id lets the daemon resolve a key-protected engine's bearer
      // server-side; a manually typed key (no hostId) still wins via header.
      const result = await api.engineStatus(host, port, key || undefined, hostId || undefined);
      setStatus(result);
      if (result.models.length && !result.models.includes(settings.model)) set({ model: result.models[0] });
    } catch { setStatus(null); }
  }
  // eslint-disable-next-line react-hooks/exhaustive-deps -- re-probe on settings change; refreshStatus reads the latest settings via closure
  useEffect(() => { void refreshStatus(); }, [settings.host, settings.port, settings.apiKey, settings.hostId]);

  // A host card (or any caller) can hand us a target to connect to — apply its
  // host/port/key/model and let the status effect above re-probe.
  useEffect(() => {
    if (!target) return;
    // A registry host hands us its hostId (the daemon resolves the engine key
    // server-side) — clear any stale manually-typed key so it can't override.
    // An ad-hoc target without a hostId may carry an explicit apiKey instead.
    set({
      host: target.host,
      port: target.port,
      hostId: target.hostId ?? "",
      ...(target.hostId ? { apiKey: "" } : target.apiKey !== undefined ? { apiKey: target.apiKey } : {}),
      ...(target.model ? { model: target.model } : {}),
    });
    chatToast(`${tr("Connecting to")} ${target.host}:${target.port}…`, "info");
    // Always re-probe, even if host/port match the current settings (the
    // dep-based effect wouldn't fire then).
    void refreshStatus({ host: target.host, port: target.port, apiKey: target.hostId ? "" : target.apiKey, hostId: target.hostId ?? "" });
    /* eslint-disable-next-line react-hooks/exhaustive-deps */
  }, [target?.nonce]);
  useEffect(() => { const el = scrollRef.current; if (el && atBottom) el.scrollTop = el.scrollHeight; }, [messages, atBottom]);

  function patchActive(updater: (c: Conversation) => Conversation) {
    setConversations((prev) => prev.map((c) => (c.id === activeIdRef.current ? updater(c) : c)));
  }

  async function runTurn(convo: ChatMessage[]) {
    const titled = convo.find((m) => m.role === "user");
    patchActive((c) => ({ ...c, title: c.title === "New chat" && titled ? titled.content.slice(0, 40) : c.title, messages: [...convo, { role: "assistant", content: "" }], updatedAt: Date.now() }));
    setStreaming(true);
    setError(null);
    setAtBottom(true);

    // RAG: ground the answer in the project's own knowledge (read-only).
    let ragMessages: Array<{ role: string; content: string }> = [];
    if (settings.useProject) {
      const lastUser = [...convo].reverse().find((m) => m.role === "user");
      if (lastUser) {
        setRetrieving(true);
        try {
          const result = await api.chatRetrieve(lastUser.content, 6, { project: settings.ragProject, vaults: settings.ragVaults });
          const docs = result.docs ?? [];
          if (docs.length) {
            ragMessages = [{ role: "system", content: buildRagContext(docs) }];
            // Attach the cited sources to the assistant placeholder.
            patchActive((c) => { const msgs = c.messages.slice(); const last = msgs[msgs.length - 1]; if (last?.role === "assistant") msgs[msgs.length - 1] = { ...last, sources: docs }; return { ...c, messages: msgs }; });
          }
        } catch { /* retrieval is best-effort — fall back to ungrounded chat */ }
        setRetrieving(false);
      }
    }

    const controller = new AbortController();
    abortRef.current = controller;
    const payloadMessages = [...(settings.system.trim() ? [{ role: "system", content: settings.system }] : []), ...ragMessages, ...convo];
    const stopSeqs = settings.stop.split(",").map((s) => s.trim()).filter(Boolean);
    const started = Date.now();
    try {
      await api.engineChatStream(
        { messages: payloadMessages, model: settings.model || undefined, max_tokens: settings.maxTokens, temperature: settings.temperature, top_p: settings.topP, presence_penalty: settings.presencePenalty, frequency_penalty: settings.frequencyPenalty, stop: stopSeqs.length ? stopSeqs : undefined, host: settings.host, port: settings.port, apiKey: settings.apiKey || undefined, hostId: settings.hostId || undefined, chat_template_kwargs: settings.thinking ? { enable_thinking: true } : undefined },
        {
          onDelta: (text) => patchActive((c) => { const msgs = c.messages.slice(); const last = msgs[msgs.length - 1]; if (!last) return c; msgs[msgs.length - 1] = { ...last, content: (last.content ?? "") + text }; return { ...c, messages: msgs }; }),
          onDone: (meta) => patchActive((c) => { const msgs = c.messages.slice(); const last = msgs[msgs.length - 1]; if (!last) return c; const secs = (meta.latency_ms ?? (Date.now() - started)) / 1000; msgs[msgs.length - 1] = { ...last, stat: { tokens: meta.tokens, ttft_ms: meta.ttft_ms, latency_ms: meta.latency_ms, tps: meta.tokens && secs ? Math.round((meta.tokens / secs) * 10) / 10 : undefined } }; return { ...c, messages: msgs, updatedAt: Date.now() }; }),
          onError: (msg) => setError(msg)
        },
        controller.signal
      );
    } catch (err) {
      if ((err as { name?: string })?.name !== "AbortError") setError(err instanceof Error ? err.message : String(err));
    } finally {
      setStreaming(false);
      setRetrieving(false);
      abortRef.current = null;
    }
  }

  function send(text?: string) {
    const value = (text ?? input).trim();
    if (!value || streaming) return;
    setInput("");
    void runTurn([...messages, { role: "user", content: value }]);
  }
  function stop() { abortRef.current?.abort(); setStreaming(false); }
  function regenerate() {
    const idx = messages.map((m) => m.role).lastIndexOf("user");
    if (idx < 0 || streaming) return;
    void runTurn(messages.slice(0, idx + 1));
  }
  function editUser(index: number) {
    const msg = messages[index];
    if (!msg || streaming) return;
    setInput(msg.content);
    patchActive((c) => ({ ...c, messages: c.messages.slice(0, index) }));
  }
  function deleteTurn(index: number) {
    patchActive((c) => { const msgs = c.messages.slice(); msgs.splice(index, messages[index]?.role === "user" && msgs[index + 1]?.role === "assistant" ? 2 : 1); return { ...c, messages: msgs }; });
  }
  function startConversation() { const c = newConversation(); setConversations((prev) => [c, ...prev]); setActiveId(c.id); setError(null); }
  function deleteConversation(id: string) {
    setConversations((prev) => { const next = prev.filter((c) => c.id !== id); if (!next.length) { const c = newConversation(); setActiveId(c.id); return [c]; } if (id === activeIdRef.current) setActiveId(next[0].id); return next; });
  }
  function exportConversation() {
    const md = messages.map((m) => `**${m.role === "user" ? "You" : "Assistant"}:**\n\n${m.content}`).join("\n\n---\n\n");
    void navigator.clipboard?.writeText(md);
  }

  const [vaultInput, setVaultInput] = useState("");
  const [vaultBusy, setVaultBusy] = useState(false);
  async function addVault() {
    const path = vaultInput.trim();
    if (!path) return;
    if (settings.ragVaults.includes(path)) { chatToast(tr("That folder is already connected."), "warn"); return; }
    setVaultBusy(true);
    try {
      const info = await api.ragPreview(path);
      if (!info.ok) { chatToast(`${tr("Can't read folder:")} ${info.error}`, "danger"); return; }
      set({ ragVaults: [...settings.ragVaults, info.path || path] });
      setVaultInput("");
      chatToast(`${tr("Connected")} ${info.files} ${info.files === 1 ? tr("note") : tr("notes")} (${info.chunks} ${tr("chunks")}) ${tr("from this folder.")}`, "ok");
    } catch (err) {
      chatToast(err instanceof Error ? err.message : tr("Failed to read folder"), "danger");
    } finally { setVaultBusy(false); }
  }
  function removeVault(path: string) { set({ ragVaults: settings.ragVaults.filter((v) => v !== path) }); }

  const reachable = !!status?.reachable;
  return (
    <div className="chat2">
      <aside className="chat2-side">
        <button className="chat2-new" onClick={startConversation}><Plus size={15} /> {tr("New chat")}</button>
        <div className="chat2-list">
          {conversations.map((c) => (
            <div className={`chat2-item ${c.id === activeId ? "active" : ""}`} key={c.id}>
              <button type="button" className="chat2-item-title" aria-current={c.id === activeId ? "true" : undefined} onClick={() => setActiveId(c.id)}>{c.title || tr("Untitled")}</button>
              <button className="icon-only" title={tr("Delete chat")} aria-label={`${tr("Delete chat")}: ${c.title || tr("Untitled")}`} onClick={() => deleteConversation(c.id)}><X size={13} /></button>
            </div>
          ))}
        </div>
      </aside>

      <div className="chat2-main">
        <div className="chat-bar">
          <label className="chat-field"><span>{tr("Host")}</span><input value={settings.host} onChange={(e) => set({ host: e.target.value })} spellCheck={false} /></label>
          <label className="chat-field chat-field-port"><span>{tr("Port")}</span><input type="number" value={settings.port} onChange={(e) => set({ port: Number(e.target.value) || 8000 })} /></label>
          <label className="chat-field chat-field-model"><span>{tr("Model")}</span>
            <select value={settings.model} onChange={(e) => set({ model: e.target.value })}>
              {status?.models?.length ? status.models.map((m) => <option key={m} value={m}>{m}</option>) : <option value="">{reachable ? tr("default") : "—"}</option>}
            </select>
          </label>
          <label className="chat-field chat-field-key"><span>{tr("API key")}</span><input type="password" value={settings.apiKey} onChange={(e) => set({ apiKey: e.target.value })} placeholder={tr("if engine requires one")} autoComplete="off" spellCheck={false} /></label>
          <span className={`chat-status ${reachable ? (status && !status.models?.length ? "warn" : "ok") : "down"}`}><span className="chat-dot" />{reachable ? `${tr("up")}${status?.version ? ` · v${status.version}` : ""}${status && !status.models?.length ? ` · ${tr("no models (API key?)")}` : ""}` : tr("down")}</span>
          <span className="chat-bar-spacer" />
          <button className={`chat-rag-toggle ${settings.useProject ? "on" : ""}`} onClick={() => set({ useProject: !settings.useProject })} title={tr("Ground answers in your knowledge sources (project patches/presets/configs + connected Obsidian/notes folders). Configure sources in Params.")}><Database size={14} /> {tr("Project RAG")}{settings.useProject && settings.ragVaults.length ? ` · ${settings.ragVaults.length}` : ""}</button>
          <button className="ghost-button" onClick={() => void refreshStatus()} title={tr("Reconnect")}><RefreshCw size={14} /></button>
          <button className={`ghost-button ${showSettings ? "active" : ""}`} onClick={() => setShowSettings((v) => !v)}><SlidersHorizontal size={14} /> {tr("Params")}</button>
          {messages.length > 0 && <button className="ghost-button" onClick={exportConversation} title={tr("Copy conversation as markdown")}><Copy size={14} /></button>}
        </div>

        {showSettings && (
          <div className="chat-settings">
            <label className="chat-field chat-field-wide"><span>{tr("System prompt")}</span><textarea value={settings.system} onChange={(e) => set({ system: e.target.value })} rows={2} /></label>
            <label className="chat-field"><span>{tr("Temperature")}</span><input type="number" min={0} max={2} step={0.1} value={settings.temperature} onChange={(e) => set({ temperature: Number(e.target.value) })} /></label>
            <label className="chat-field"><span>{tr("Max tokens")}</span><input type="number" min={1} max={4096} value={settings.maxTokens} onChange={(e) => set({ maxTokens: Number(e.target.value) || 512 })} /></label>
            <label className="chat-field"><span>{tr("Top P")}</span><input type="number" min={0} max={1} step={0.05} value={settings.topP} onChange={(e) => set({ topP: Number(e.target.value) })} /></label>
            <label className="chat-field"><span>{tr("Presence")}</span><input type="number" min={-2} max={2} step={0.1} value={settings.presencePenalty} onChange={(e) => set({ presencePenalty: Number(e.target.value) })} /></label>
            <label className="chat-field"><span>{tr("Frequency")}</span><input type="number" min={-2} max={2} step={0.1} value={settings.frequencyPenalty} onChange={(e) => set({ frequencyPenalty: Number(e.target.value) })} /></label>
            <label className="chat-field chat-field-stop"><span>{tr("Stop (comma-sep)")}</span><input value={settings.stop} onChange={(e) => set({ stop: e.target.value })} placeholder="</s>, ###" /></label>
            <label className="chat-think"><input type="checkbox" checked={settings.thinking} onChange={(e) => set({ thinking: e.target.checked })} /> {tr("Thinking mode")} <span className="chat-think-hint">{tr("(enable_thinking — reasoning models render the <think> path)")}</span></label>
            {settings.thinking && /coder/i.test(settings.model) && <div className="chat-advisory"><CircleAlert size={13} /> {tr("With reasoning + tool-calls, the")} <code>qwen3_coder</code> {tr("streaming parser drops")} <code>delta.tool_calls</code> — {tr("serve with")} <code>--tool-call-parser qwen3_xml</code> {tr("for reliable streaming tool calls.")}</div>}
            <div className="chat-knowledge">
              <div className="chat-knowledge-head"><Database size={13} /> {tr("Knowledge sources (RAG)")}</div>
              <label className="chat-knowledge-toggle"><input type="checkbox" checked={settings.ragProject} onChange={(e) => set({ ragProject: e.target.checked })} /> {tr("Project — patches, presets & configs")}</label>
              {settings.ragVaults.map((v) => (
                <div className="chat-vault" key={v}>
                  <BookText size={12} /><span className="chat-vault-path" title={v}>{v}</span>
                  <button className="icon-only" title={tr("Disconnect folder")} onClick={() => removeVault(v)}><X size={12} /></button>
                </div>
              ))}
              <div className="chat-vault-add">
                <input aria-label={tr("Vault or notes folder path")} value={vaultInput} onChange={(e) => setVaultInput(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); void addVault(); } }} placeholder={tr("Path to an Obsidian vault or notes folder…")} spellCheck={false} />
                <button className="ghost-button" onClick={() => void addVault()} disabled={vaultBusy || !vaultInput.trim()}>{vaultBusy ? <Loader2 size={13} className="spin" /> : <Plus size={13} />} {tr("Connect")}</button>
              </div>
              <span className="chat-knowledge-hint">{tr("Notes (.md / .txt) in the folder are indexed locally and read-only. Turn on RAG with the")} <strong>{tr("Project RAG")}</strong> {tr("button above.")}</span>
            </div>
          </div>
        )}

        <div className="chat-messages" ref={scrollRef} onScroll={(e) => { const el = e.currentTarget; setAtBottom(el.scrollHeight - el.scrollTop - el.clientHeight < 60); }}>
          {messages.length === 0 ? (
            <div className="chat-empty">
              <span className="chat-empty-orb"><Sparkles size={26} /></span>
              <strong>{settings.useProject ? tr("Ask about this project") : tr("Chat with your local model")}</strong>
              <span>{settings.useProject
                ? `${tr("RAG is on — grounding in")} ${[settings.ragProject && tr("project knowledge"), settings.ragVaults.length && `${settings.ragVaults.length} ${settings.ragVaults.length === 1 ? tr("notes folder") : tr("notes folders")}`].filter(Boolean).join(" + ") || tr("no sources (enable some in Params)")}.`
                : reachable ? `${tr("Connected to")} ${settings.model || tr("the engine")} ${tr("on")} ${settings.host}:${settings.port}.` : `${tr("Point Host/Port at a running vLLM engine (e.g.")} ${settings.host}:8101), ${tr("then say hello.")}`}</span>
              <div className="chat-suggest">{(settings.useProject ? PROJECT_SUGGESTIONS : SUGGESTIONS).map((s) => <button key={s} onClick={() => send(s)}>{tr(s)}</button>)}</div>
            </div>
          ) : messages.map((msg, index) => (
            <div className={`chat-msg ${msg.role}`} key={index}>
              <span className="chat-avatar">{msg.role === "user" ? <User size={15} /> : <Bot size={15} />}</span>
              <div className="chat-col">
                <div className="chat-bubble">
                  {retrieving && index === messages.length - 1 && msg.role === "assistant" && !msg.content
                    ? <span className="chat-retrieving"><Database size={13} /> {tr("Searching project knowledge…")}</span>
                    : msg.content ? <MarkdownLite text={msg.content} /> : (streaming && index === messages.length - 1 ? <span className="chat-typing"><span /><span /><span /></span> : <em className="muted">{tr("(empty)")}</em>)}
                  {streaming && index === messages.length - 1 && msg.content && <span className="chat-cursor" />}
                </div>
                {msg.sources && msg.sources.length > 0 && <SourcesRow docs={msg.sources} />}
                <div className="chat-msg-foot">
                  {msg.stat && <span className="chat-stat">{msg.stat.tokens ?? 0} tok{msg.stat.tps ? ` · ${msg.stat.tps} tok/s` : ""}{msg.stat.ttft_ms ? ` · ${msg.stat.ttft_ms}ms TTFT` : ""}</span>}
                  <div className="chat-msg-actions">
                    <button className="icon-only" title={tr("Copy")} onClick={() => void navigator.clipboard?.writeText(msg.content)}><Copy size={12} /></button>
                    {msg.role === "user" && <button className="icon-only" title={tr("Edit & resend")} onClick={() => editUser(index)}><Pencil size={12} /></button>}
                    {msg.role === "assistant" && index === messages.length - 1 && !streaming && <button className="icon-only" title={tr("Regenerate")} onClick={regenerate}><RefreshCw size={12} /></button>}
                    <button className="icon-only" title={tr("Delete")} onClick={() => deleteTurn(index)}><Trash2 size={12} /></button>
                  </div>
                </div>
              </div>
            </div>
          ))}
          {!atBottom && messages.length > 0 && <button className="chat-scroll-btn" onClick={() => { const el = scrollRef.current; if (el) el.scrollTop = el.scrollHeight; setAtBottom(true); }}><ChevronDown size={16} /></button>}
        </div>

        {error && <div className="chat-error"><CircleAlert size={14} /> {error}</div>}

        <ChatRoutingHint value={settings.workloadClass} onChange={(w) => set({ workloadClass: w })} />

        <div className="chat-composer">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
            placeholder={reachable ? tr("Message… (Enter to send, Shift+Enter for newline)") : tr("Engine not reachable — set Host/Port above")}
            rows={2}
          />
          <div className="chat-composer-actions">
            {streaming
              ? <button className="primary-button chat-stop" onClick={stop}><Square size={14} /> {tr("Stop")}</button>
              : <button className="primary-button" onClick={() => send()} disabled={!input.trim()}><Send size={15} /> {tr("Send")}</button>}
          </div>
        </div>
      </div>
    </div>
  );
}
