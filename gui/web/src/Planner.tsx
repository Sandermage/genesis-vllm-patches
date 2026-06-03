import { Fragment, useEffect, useRef, useState } from "react";
import { AlertTriangle, CheckCircle2, CircleAlert, GitCompare, Save, Server, Sparkles, Trash2, TrendingDown, TrendingUp } from "lucide-react";
import { api, type CalcModels, type HostModelConfig, type HostProfile, type KvCalcResult, type KvEstimate, type BaselineDiff, type BaselineRec, type BaselineTrend } from "./api";

const fmtGb = (m: number) => (Math.abs(m) >= 1024 ? `${(m / 1024).toFixed(1)} GB` : `${Math.round(m)} MB`);
const fmtCtx = (c: number) => (c >= 1000 ? `${Math.round(c / 1000)}K` : String(c));

// Enterprise KPI tile — the key numbers, legible at a glance (app .metric look).
function Kpi({ label, value, sub, tone }: { label: string; value: string; sub?: string; tone?: "ok" | "bad" | "accent" }) {
  return (
    <div className={`kpi ${tone ?? ""}`}>
      <span className="kpi-label">{label}</span>
      <strong className="kpi-value">{value}</strong>
      {sub && <span className="kpi-sub">{sub}</span>}
    </div>
  );
}

// ── KV / VRAM fit calculator ────────────────────────────────────────────────
export function KvCalcPanel() {
  const [meta, setMeta] = useState<CalcModels | null>(null);
  const [hosts, setHosts] = useState<HostProfile[]>([]);
  const [hostId, setHostId] = useState("");
  const [modelId, setModelId] = useState("qwen3.6-27b-int4");
  const [ctx, setCtx] = useState(32768);
  const [conc, setConc] = useState(1);
  const [tp, setTp] = useState(2);
  const [vram, setVram] = useState(24564);
  const [util, setUtil] = useState(0.9);
  const [kvDtype, setKvDtype] = useState("fp8");
  const [gpuName, setGpuName] = useState("");
  const [measured, setMeasured] = useState("");
  const [real, setReal] = useState<HostModelConfig | null>(null);
  const [loadingReal, setLoadingReal] = useState(false);
  const [calc, setCalc] = useState<KvCalcResult | null>(null);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    api.calcModels().then((m) => {
      setMeta(m);
      // Keep the default model if the catalog actually has it; otherwise fall
      // back to the first known key. (Don't blindly overwrite with models[0] —
      // the API's first key isn't guaranteed to be the intended default.)
      setModelId((cur) => (m.models[cur] ? cur : Object.keys(m.models)[0] ?? cur));
    }).catch(() => {});
    api.hosts().then((h) => setHosts(h.hosts)).catch(() => {});
  }, []);

  // Pick a host → pull real VRAM / GPU count / arch from the discovered profile.
  function applyHostRig(id: string) {
    setHostId(id);
    setReal(null);
    const h = hosts.find((x) => x.id === id);
    if (h) {
      if (h.gpu_vram_mib && h.gpu_vram_mib > 0) setVram(h.gpu_vram_mib);
      if (h.gpus && h.gpus > 0) setTp(h.gpus);
      setGpuName(h.gpu_name || "");
    } else setGpuName("");
  }

  // Read the running model's *real* architecture (config.json + exact weights).
  async function loadReal() {
    if (!hostId) return;
    setLoadingReal(true);
    try {
      const m = await api.modelConfig(hostId);
      if (m.ok) { setReal(m); if (m.max_context) setCtx(Math.min(m.max_context, 131072)); }
      else setReal({ ...m, ok: false });
    } catch (e) { setReal({ ok: false, error: e instanceof Error ? e.message : String(e) }); }
    finally { setLoadingReal(false); }
  }

  // When real dims are loaded, drive the calc from them (overrides the curated model).
  useEffect(() => {
    if (timer.current) window.clearTimeout(timer.current);
    // realDims is derived from `real` (in deps) — computed inside the effect so
    // it isn't an unstable closure dependency that would re-run every render.
    const realDims = real && real.ok ? {
      name: `${real.model_type || "model"} (${real.model_path?.split("/").pop()})`,
      num_layers: real.num_layers, num_kv_heads: real.num_kv_heads, head_dim: real.head_dim,
      weights_bytes_total: real.weights_bytes || undefined, sliding_window: real.sliding_window || undefined,
      global_layers: real.global_layers ?? undefined, max_context: real.max_context, source: "host-config",
    } : null;
    timer.current = window.setTimeout(() => {
      const base = { context: ctx, concurrency: conc, tp, gpu_count: tp, gpu_vram_mib: vram, util, kv_dtype: kvDtype, gpu_name: gpuName || undefined, measured_total_mib: measured ? Number(measured) : undefined };
      api.calcKv(realDims ? { ...base, ...realDims } : { ...base, model_id: modelId }).then(setCalc).catch(() => {});
    }, 120);
  }, [modelId, ctx, conc, tp, vram, util, kvDtype, gpuName, measured, real]);

  const r = calc?.result;
  const budget = r?.budget_per_gpu_mib ?? 1;
  const rec = calc?.recommendation?.find((x) => x.recommended) || calc?.recommendation?.[0];
  const dtypeOpts = meta ? Array.from(new Map(Object.entries(meta.kv_dtypes).map(([k, v]) => [v, k])).values()) : [];

  return (
    <div className="kvcalc">
      <div className="kvcalc-controls">
        <label className="param-field"><span><Server size={11} /> Rig (from host card)</span>
          <select value={hostId} onChange={(e) => applyHostRig(e.target.value)}>
            <option value="">— manual / custom rig —</option>
            {hosts.map((h) => <option key={h.id} value={h.id}>{h.label}{h.gpu_vram_mib ? ` · ${h.gpus}× ${Math.round(h.gpu_vram_mib / 1024)}GB ${h.gpu_arch || ""}` : " · run Discover first"}</option>)}
          </select>
        </label>
        <label className="param-field"><span>Model {real?.ok && <em className="kvcalc-real-tag">real dims</em>}</span>
          {real?.ok
            ? <button className="ghost-button kvcalc-clear-real" onClick={() => setReal(null)} title="Back to curated model">{real.model_path?.split("/").pop()} ✕</button>
            : <select value={modelId} onChange={(e) => setModelId(e.target.value)}>
                {meta && Object.entries(meta.models).map(([k, m]) => <option key={k} value={k}>{m.name}{m.is_moe ? " · MoE" : ""}</option>)}
              </select>}
        </label>
        <label className="param-field"><span>Real architecture</span>
          <button className="ghost-button" onClick={() => void loadReal()} disabled={!hostId || loadingReal} title="SSH to the host and read the running model's real config.json + exact weight size">
            {loadingReal ? "Reading…" : "Load from engine"}
          </button>
        </label>
        <label className="param-field"><span>KV dtype</span>
          <select value={kvDtype} onChange={(e) => setKvDtype(e.target.value)}>
            {dtypeOpts.map((d) => <option key={d} value={d}>{d} ({meta!.kv_dtypes[d]} B/elem)</option>)}
          </select>
        </label>
        <label className="param-field"><span>Tensor parallel</span><input type="number" min={1} max={8} value={tp} onChange={(e) => setTp(Math.max(1, Number(e.target.value) || 1))} /></label>
        <label className="param-field"><span>VRAM / GPU (MiB)</span><input type="number" value={vram} onChange={(e) => { setVram(Number(e.target.value) || 24564); setHostId(""); }} /></label>
        <label className="param-field"><span>gpu_mem_util</span><input type="number" min={0.1} max={1} step={0.01} value={util} onChange={(e) => setUtil(Number(e.target.value) || 0.9)} /></label>
        <label className="param-field"><span>Concurrency</span><input type="number" min={1} value={conc} onChange={(e) => setConc(Math.max(1, Number(e.target.value) || 1))} /></label>
        <label className="param-field"><span title="Calibrate overhead to a real measured VRAM total">Measured GB (optional)</span><input type="number" step={0.1} value={measured} placeholder="e.g. 14.0" onChange={(e) => setMeasured(e.target.value ? String(Number(e.target.value) * 1024) : "")} /></label>
      </div>

      {real && !real.ok && <div className="kvcalc-real-err"><AlertTriangle size={13} /> Couldn't read the model: {real.error}. Pick a host that has run Discover and whose engine is up.</div>}
      {real?.ok && (
        <div className="kvcalc-real">
          <CheckCircle2 size={14} />
          <span><strong>Real dims from engine</strong> — {real.model_type} · {real.num_layers} layers · {real.num_kv_heads} KV-heads · head_dim {real.head_dim}{real.is_moe ? ` · MoE ${real.num_experts}e` : ""}{real.sliding_window ? ` · sliding window ${fmtCtx(real.sliding_window)} (${real.global_layers} global layers)` : ""} · weights {real.weights_bytes ? fmtGb(real.weights_bytes / (1024 * 1024)) : "?"} {real.quant_method} · native {fmtCtx(real.max_context || 0)}</span>
        </div>
      )}
      {rec && (
        <div className={`kvcalc-rec ${rec.fits ? "ok" : "bad"}`}>
          {rec.fits ? <Sparkles size={16} /> : <AlertTriangle size={16} />}
          <div className="kvcalc-rec-text">
            <strong>{rec.fits ? `Recommended: ${rec.kv_dtype} KV` : "Won't fit at this target"}</strong>
            <span>{rec.fits
              ? `${calc!.arch.name} on ${tp}× ${Math.round(vram / 1024)}GB at ${fmtCtx(ctx)} ctx · ${conc} conc → fits with ${fmtGb(rec.headroom_mib)} headroom (max ${fmtCtx(rec.max_context)}).`
              : `Even ${calc!.recommendation[calc!.recommendation.length - 1].kv_dtype} KV is over budget at ${fmtCtx(ctx)} ctx · ${conc} conc. Lower context/concurrency, add a GPU (TP), or use a smaller model.`}</span>
          </div>
          <div className="kvcalc-rec-opts">
            {calc!.recommendation.map((o) => (
              <button key={o.kv_dtype} className={`kvcalc-rec-opt ${o.kv_dtype === kvDtype ? "active" : ""} ${o.fits ? "fits" : "over"}`} onClick={() => setKvDtype(o.kv_dtype)} title={`${o.kv_dtype}: ${o.fits ? fmtGb(o.headroom_mib) + " free" : fmtGb(-o.headroom_mib) + " over"}`}>
                {o.fits ? <CheckCircle2 size={11} /> : <CircleAlert size={11} />}{o.kv_dtype}
              </button>
            ))}
          </div>
        </div>
      )}

      <label className="kvcalc-slider">
        <div className="kvcalc-slider-head"><span>Context</span><strong>{ctx.toLocaleString()} tokens</strong>{r && <span className={`kvcalc-verdict ${r.fits ? "ok" : "bad"}`}>{r.fits ? <><CheckCircle2 size={13} /> fits · {fmtGb(r.headroom_mib)} free</> : <><AlertTriangle size={13} /> {fmtGb(-r.headroom_mib)} over budget</>}</span>}</div>
        <input type="range" min={1024} max={Math.max(262144, (r?.max_context ?? 0) + 8192)} step={1024} value={ctx} onChange={(e) => setCtx(Number(e.target.value))} />
      </label>

      {calc && r && (
        <>
          <div className="kpi-strip">
            <Kpi label="Weights / GPU" value={fmtGb(r.weights_per_gpu_mib)} />
            <Kpi label="KV cache / GPU" value={fmtGb(r.kv_per_gpu_mib)} tone="accent" />
            <Kpi label="Overhead" value={fmtGb(r.overhead_mib)} />
            <Kpi label="Total / GPU" value={fmtGb(r.total_per_gpu_mib)} sub={`of ${fmtGb(r.budget_per_gpu_mib)} budget`} />
            <Kpi label="Headroom" value={r.fits ? fmtGb(r.headroom_mib) : `−${fmtGb(-r.headroom_mib)}`} tone={r.fits ? "ok" : "bad"} />
            <Kpi label="Max context" value={`${fmtCtx(r.max_context)} tok`} />
          </div>
          <VramChart curve={calc.curve} budget={budget} ctx={ctx} maxCtx={r.max_context} />
          <div className="kvcalc-grid">
            <VramDonut r={r} />
            <FitHeatmap env={calc.envelope} ctx={ctx} conc={conc} onPick={(c, k) => { setCtx(c); setConc(k); }} />
            <DtypeBars byDtype={calc.by_dtype} active={kvDtype} onPick={setKvDtype} />
            <TpScalingBars byTp={calc.by_tp} activeTp={tp} onPick={setTp} />
          </div>
          {calc.arch_advice && calc.arch_advice.recommendations.length > 0 && (
            <div className="kvcalc-advice">
              <span className="kvcalc-label"><AlertTriangle size={12} /> Arch-aware notes — {calc.arch_advice.arch}</span>
              {calc.arch_advice.recommendations.map((a, i) => <span key={i} className={`kvcalc-arch-rec ${a.level}`}>{a.level === "ok" ? <CheckCircle2 size={11} /> : <CircleAlert size={11} />} {a.text}</span>)}
            </div>
          )}
        </>
      )}
    </div>
  );
}

// Stacked VRAM-vs-context area chart: gradient layers, budget + max-ctx markers,
// and a live cursor callout showing the exact breakdown at the chosen context.
function VramChart({ curve, budget, ctx, maxCtx }: { curve: KvCalcResult["curve"]; budget: number; ctx: number; maxCtx: number }) {
  const W = 1000, H = 230, L = 50, R = 16, T = 16, B = 28;
  const maxX = curve[curve.length - 1]?.context || 1;
  const maxY = Math.max(budget * 1.12, ...curve.map((p) => p.total_mib));
  const x = (c: number) => L + (c / maxX) * (W - L - R);
  const y = (m: number) => H - B - (m / maxY) * (H - T - B);
  type P = KvCalcResult["curve"][0];
  const area = (sel: (p: P) => number, base: (p: P) => number) => {
    const top = curve.map((p) => `${x(p.context).toFixed(1)},${y(base(p) + sel(p)).toFixed(1)}`);
    const bot = [...curve].reverse().map((p) => `${x(p.context).toFixed(1)},${y(base(p)).toFixed(1)}`);
    return `M${top.join(" L")} L${bot.join(" L")} Z`;
  };
  const w = (p: P) => p.weights_mib;
  const o = (p: P) => p.overhead_mib;
  const xticks = [0, 0.25, 0.5, 0.75, 1].map((f) => Math.round(maxX * f));
  const yticks = [0, 0.25, 0.5, 0.75, 1].map((f) => Math.round(maxY * f));
  // Nearest sampled point to the cursor → exact values for the callout.
  const cur = curve.reduce((a, b) => Math.abs(b.context - ctx) < Math.abs(a.context - ctx) ? b : a, curve[0]) || curve[0];
  const over = cur ? cur.total_mib > budget : false;
  const cx = x(cur?.context ?? ctx);
  const calloutLeft = cx > W * 0.6;
  return (
    <div className="vram-chart-wrap">
      <div className="vram-chart-head">
        <span className="kvcalc-label">Per-GPU VRAM as context grows</span>
        <span className="vram-chart-sub">weights + KV + overhead vs the GPU budget</span>
      </div>
      <svg className="vram-chart" viewBox={`0 0 ${W} ${H}`}>
        <defs>
          <linearGradient id="g-weights" x1="0" x2="0" y1="0" y2="1"><stop offset="0" className="gs-weights-0" /><stop offset="1" className="gs-weights-1" /></linearGradient>
          <linearGradient id="g-overhead" x1="0" x2="0" y1="0" y2="1"><stop offset="0" className="gs-overhead-0" /><stop offset="1" className="gs-overhead-1" /></linearGradient>
          <linearGradient id="g-kv" x1="0" x2="0" y1="0" y2="1"><stop offset="0" className="gs-kv-0" /><stop offset="1" className="gs-kv-1" /></linearGradient>
        </defs>
        {yticks.map((t, i) => <g key={i}><line x1={L} y1={y(t)} x2={W - R} y2={y(t)} className="vc-grid" /><text x={L - 8} y={y(t) + 3} className="vc-ytick">{(t / 1024).toFixed(0)}G</text></g>)}
        {xticks.map((t, i) => <text key={i} x={x(t)} y={H - 9} className="vc-xtick">{fmtCtx(t)}</text>)}
        <path d={area(w, () => 0)} fill="url(#g-weights)" className="vc-line-weights" />
        <path d={area(o, w)} fill="url(#g-overhead)" className="vc-line-overhead" />
        <path d={area((p) => p.kv_mib, (p) => p.weights_mib + p.overhead_mib)} fill="url(#g-kv)" className="vc-line-kv" />
        <line x1={L} y1={y(budget)} x2={W - R} y2={y(budget)} className="vc-budget" />
        <text x={W - R} y={y(budget) - 5} className="vc-budget-lbl">budget {(budget / 1024).toFixed(1)}G</text>
        {maxCtx > 0 && maxCtx <= maxX && <><line x1={x(maxCtx)} y1={T} x2={x(maxCtx)} y2={H - B} className="vc-maxctx" /><text x={x(maxCtx) + 4} y={T + 11} className="vc-maxctx-lbl">max {fmtCtx(maxCtx)}</text></>}
        {/* cursor + point + callout */}
        <line x1={cx} y1={T} x2={cx} y2={H - B} className="vc-cursor" />
        {cur && <circle cx={cx} cy={y(cur.total_mib)} r={4} className={`vc-dot ${over ? "over" : "ok"}`} />}
        {cur && (
          <g transform={`translate(${calloutLeft ? cx - 156 : cx + 10}, ${T + 6})`}>
            <rect width={146} height={74} rx={7} className="vc-callout" />
            <text x={10} y={17} className="vc-callout-ctx">{fmtCtx(cur.context)} ctx · {(cur.total_mib / 1024).toFixed(1)}G total</text>
            <text x={10} y={33} className="vc-callout-row"><tspan className="vc-sw-weights">■</tspan> weights {(cur.weights_mib / 1024).toFixed(1)}G</text>
            <text x={10} y={48} className="vc-callout-row"><tspan className="vc-sw-kv">■</tspan> KV {(cur.kv_mib / 1024).toFixed(1)}G</text>
            <text x={10} y={63} className="vc-callout-row"><tspan className="vc-sw-overhead">■</tspan> overhead {(cur.overhead_mib / 1024).toFixed(1)}G</text>
          </g>
        )}
      </svg>
      <div className="vram-chart-legend"><span><i className="vc-weights" /> weights</span><span><i className="vc-kv" /> KV cache</span><span><i className="vc-overhead" /> overhead</span><span className="vc-legend-budget"><i /> budget</span></div>
    </div>
  );
}

// VRAM-composition donut at the current operating point — a different view from
// the area sweep: what *fills* the budget right now (weights / KV / overhead),
// with utilisation % and headroom in the centre.
function VramDonut({ r }: { r: KvEstimate }) {
  const segs = [
    { key: "weights", label: "weights", val: r.weights_per_gpu_mib, cls: "dw" },
    { key: "kv", label: "KV cache", val: r.kv_per_gpu_mib, cls: "dk" },
    { key: "overhead", label: "overhead", val: r.overhead_mib, cls: "do" },
  ];
  const used = Math.max(1, r.weights_per_gpu_mib + r.kv_per_gpu_mib + r.overhead_mib);
  const RAD = 52, C = 2 * Math.PI * RAD;
  const util = Math.round((r.total_per_gpu_mib / Math.max(1, r.budget_per_gpu_mib)) * 100);
  let acc = 0;
  return (
    <div className="vram-donut-wrap">
      <div className="kvcalc-label">VRAM composition at {fmtCtx(r.context)} ctx — what fills the budget</div>
      <div className="vram-donut">
        <svg viewBox="0 0 140 140" role="img" aria-label={`VRAM utilisation ${util}%`}>
          <g transform="rotate(-90 70 70)">
            <circle cx="70" cy="70" r={RAD} className="donut-track" />
            {segs.map((s) => {
              const frac = s.val / used;
              const dash = `${(frac * C).toFixed(1)} ${C.toFixed(1)}`;
              const off = (-acc * C).toFixed(1);
              acc += frac;
              return <circle key={s.key} cx="70" cy="70" r={RAD} className={`donut-seg ${s.cls}`} strokeDasharray={dash} strokeDashoffset={off} />;
            })}
          </g>
          <text x="70" y="66" className={`donut-pct ${r.fits ? "ok" : "bad"}`}>{util}%</text>
          <text x="70" y="84" className="donut-sub">{r.fits ? `${fmtGb(r.headroom_mib)} free` : `${fmtGb(-r.headroom_mib)} over`}</text>
        </svg>
        <div className="donut-legend">
          {segs.map((s) => <span key={s.key}><i className={s.cls} /> {s.label} <b>{fmtGb(s.val)}</b></span>)}
          <span><i className="db" /> budget <b>{fmtGb(r.budget_per_gpu_mib)}</b></span>
        </div>
      </div>
    </div>
  );
}

// Operating-envelope heatmap: concurrency × context, cell colour = headroom.
function FitHeatmap({ env, ctx, conc, onPick }: { env: KvCalcResult["envelope"]; ctx: number; conc: number; onPick: (c: number, k: number) => void }) {
  const cellColor = (h: number) => h < 0 ? "over" : h < 2048 ? "tight" : h < 6144 ? "ok" : "good";
  const nearestCtx = env.contexts.reduce((a, b) => Math.abs(b - ctx) < Math.abs(a - ctx) ? b : a, env.contexts[0]);
  return (
    <div className="heatmap">
      <div className="kvcalc-label">Operating envelope — does it fit? (concurrency × context)</div>
      <div className="heatmap-grid" style={{ gridTemplateColumns: `40px repeat(${env.contexts.length}, 1fr)` }}>
        <span className="heatmap-corner" />
        {env.contexts.map((c) => <span key={c} className="heatmap-xlabel">{fmtCtx(c)}</span>)}
        {[...env.grid].reverse().map((row, ri) => {
          const k = [...env.concurrencies].reverse()[ri];
          return (
            <Fragment key={`row-${k}`}>
              <span className="heatmap-ylabel">{k}×</span>
              {row.map((cell) => (
                <button key={`${k}-${cell.context}`} className={`heatmap-cell ${cellColor(cell.headroom_mib)} ${cell.context === nearestCtx && k === conc ? "current" : ""}`}
                  title={`${fmtCtx(cell.context)} ctx · ${k} conc → ${cell.fits ? fmtGb(cell.headroom_mib) + " free" : fmtGb(-cell.headroom_mib) + " over"}`}
                  onClick={() => onPick(cell.context, k)} />
              ))}
            </Fragment>
          );
        })}
      </div>
      <div className="heatmap-legend"><span className="good" /> roomy<span className="ok" /> fits<span className="tight" /> tight<span className="over" /> over</div>
    </div>
  );
}

function DtypeBars({ byDtype, active, onPick }: { byDtype: Record<string, number>; active: string; onPick: (d: string) => void }) {
  const entries = Array.from(new Map(Object.entries(byDtype).map(([k, v]) => [v, [k, v] as [string, number]])).values()).sort((a, b) => b[1] - a[1]);
  const max = entries[0]?.[1] || 1;
  return (
    <div className="dtype-bars">
      <div className="kvcalc-label">Max context by KV dtype</div>
      {entries.map(([d, mc]) => (
        <button key={d} className={`dtype-bar-row ${d === active ? "active" : ""}`} onClick={() => onPick(d)}>
          <span className="dtype-bar-name">{d}</span>
          <span className="dtype-bar-track"><span className="dtype-bar-fill" style={{ width: `${(mc / max) * 100}%` }} /></span>
          <span className="dtype-bar-val">{fmtCtx(mc)}</span>
        </button>
      ))}
    </div>
  );
}

// Max context vs tensor-parallel width — how adding GPUs grows the context
// budget (weights + KV both shard across GPUs). Click a row to switch TP.
function TpScalingBars({ byTp, activeTp, onPick }: { byTp: Record<string, number>; activeTp: number; onPick: (tp: number) => void }) {
  const entries = Object.entries(byTp).map(([k, v]) => [Number(k), v] as [number, number]).sort((a, b) => a[0] - b[0]);
  const max = Math.max(1, ...entries.map(([, v]) => v));
  return (
    <div className="dtype-bars tp-bars">
      <div className="kvcalc-label">Max context by GPU count (tensor-parallel)</div>
      {entries.map(([t, mc]) => (
        <button key={t} className={`dtype-bar-row ${t === activeTp ? "active" : ""}`} onClick={() => onPick(t)}>
          <span className="dtype-bar-name">{t}× GPU</span>
          <span className="dtype-bar-track"><span className="dtype-bar-fill" style={{ width: `${(mc / max) * 100}%` }} /></span>
          <span className="dtype-bar-val">{fmtCtx(mc)}</span>
        </button>
      ))}
    </div>
  );
}

// ── Baseline regression diff ────────────────────────────────────────────────
// Regression trend: one metric charted across saved baselines over time.
function BaselineTrendChart({ reloadKey }: { reloadKey: number }) {
  const [metric, setMetric] = useState<string>("");
  const [data, setData] = useState<BaselineTrend | null>(null);
  useEffect(() => {
    let alive = true;
    api.baselineTrend(metric || undefined)
      .then((t) => { if (alive) { setData(t); if (!metric && t.metric) setMetric(t.metric); } })
      .catch(() => {});
    return () => { alive = false; };
  }, [metric, reloadKey]);

  const pts = data?.points ?? [];
  const metrics = data?.metrics_available ?? [];
  if (pts.length < 2) {
    return (
      <div className="baseline-trend">
        <div className="baseline-trend-head"><strong>Regression trend</strong></div>
        <div className="baseline-trend-empty muted">Save at least 2 baselines to chart how {data?.metric || "a metric"} moves run-over-run.</div>
      </div>
    );
  }
  const vals = pts.map((p) => p.value);
  const min = Math.min(...vals), max = Math.max(...vals), span = max - min || 1;
  const W = 100, H = 38, step = W / (pts.length - 1);
  const xy = (v: number, i: number) => ({ x: i * step, y: H - ((v - min) / span) * H });
  const line = pts.map((p, i) => { const c = xy(p.value, i); return `${c.x.toFixed(1)},${c.y.toFixed(1)}`; }).join(" ");
  const first = vals[0], last = vals[vals.length - 1];
  const delta = first ? ((last - first) / first) * 100 : 0;
  const flat = Math.abs(delta) < 1;
  const better = data?.lower_is_better ? delta < 0 : delta > 0;
  const tone = flat ? "" : better ? "imp" : "reg";

  return (
    <div className="baseline-trend">
      <div className="baseline-trend-head">
        <strong>Regression trend</strong>
        <select value={metric} onChange={(e) => setMetric(e.target.value)} aria-label="metric">
          {metrics.map((m) => <option key={m} value={m}>{m}</option>)}
        </select>
        <span className={`baseline-trend-delta ${tone}`}>
          {delta > 0 ? "+" : ""}{delta.toFixed(1)}% over {pts.length} runs
        </span>
      </div>
      <svg className={`baseline-trend-svg ${tone}`} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
        <polyline points={line} />
      </svg>
      <div className="baseline-trend-foot">
        <span>{pts[0].label}: {first}</span>
        <span>{pts[pts.length - 1].label}: {last}</span>
      </div>
    </div>
  );
}

export function BaselinePanel() {
  const [list, setList] = useState<BaselineRec[]>([]);
  const [draft, setDraft] = useState('{"label":"new run","scenarios":[{"name":"code","metrics":{"tps":120,"ttft_ms":110,"tool_call_success":0.95}}]}');
  const [against, setAgainst] = useState("");
  const [diff, setDiff] = useState<BaselineDiff | null>(null);
  const [err, setErr] = useState<string | null>(null);

  function reload() { api.baselines().then((b) => { setList(b.baselines); if (!against && b.baselines[0]) setAgainst(b.baselines[0].id); }).catch(() => {}); }
  // eslint-disable-next-line react-hooks/exhaustive-deps -- one-shot load on mount
  useEffect(() => { reload(); }, []);

  function parsed(): unknown { try { setErr(null); return JSON.parse(draft); } catch (e) { setErr(e instanceof Error ? e.message : "invalid JSON"); return null; } }
  const msg = (e: unknown) => (e instanceof Error ? e.message : String(e));
  async function save() { const r = parsed(); if (!r) return; try { await api.baselineSave(r); reload(); } catch (e) { setErr(msg(e)); } }
  async function runDiff() { const r = parsed(); if (!r || !against) return; try { setDiff(await api.baselineDiff(r, against)); } catch (e) { setErr(msg(e)); } }
  async function remove(id: string) { try { setErr(null); await api.baselineDelete(id); if (against === id) setAgainst(""); reload(); } catch (e) { setErr(msg(e)); } }

  return (
    <div className="baseline-panel">
      <div className="baseline-bar">
        <label className="param-field"><span>Baseline to diff against</span>
          <select value={against} onChange={(e) => setAgainst(e.target.value)}>
            <option value="">— pick a saved baseline —</option>
            {list.map((b) => <option key={b.id} value={b.id}>{b.label} · {new Date(b.saved_at * 1000).toLocaleDateString()}</option>)}
          </select>
        </label>
        <button className="ghost-button" onClick={() => void save()} title="Save the current result JSON as a baseline"><Save size={14} /> Save as baseline</button>
        <button className="primary-action" onClick={() => void runDiff()} disabled={!against}><GitCompare size={14} /> Diff</button>
      </div>
      <label className="param-field"><span>Current result (bench/eval JSON)</span>
        <textarea className="baseline-draft" value={draft} onChange={(e) => setDraft(e.target.value)} rows={4} spellCheck={false} />
      </label>
      {err && <div className="baseline-err"><AlertTriangle size={13} /> {err}</div>}
      {list.length > 0 && (
        <div className="baseline-list">{list.map((b) => (
          <span key={b.id} className="baseline-chip">{b.label}<button className="icon-only" onClick={() => void remove(b.id)} title="Delete"><Trash2 size={11} /></button></span>
        ))}</div>
      )}
      {diff && (
        <div className={`baseline-diff ${diff.has_regression ? "regress" : diff.improved ? "improve" : "stable"}`}>
          <div className="baseline-verdict"><strong>{diff.verdict}</strong> · {diff.regressed} regressed · {diff.improved} improved · exit {diff.exit_code}</div>
          {diff.scenarios.map((s) => (
            <div className="baseline-scn" key={s.name}>
              <div className="baseline-scn-head">{s.name} {s.status !== "compared" && <em>({s.status})</em>}</div>
              {s.metrics.map((m) => (
                <div key={m.metric} className={`baseline-metric ${m.regression ? "reg" : m.improvement ? "imp" : ""}`}>
                  <span className="bm-name">{m.metric}</span>
                  <span className="bm-vals">{m.baseline} → {m.current}</span>
                  <span className="bm-delta">{m.regression ? <TrendingDown size={12} /> : m.improvement ? <TrendingUp size={12} /> : null} {m.pct > 0 ? "+" : ""}{m.pct}%</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      )}
      <BaselineTrendChart reloadKey={list.length} />
    </div>
  );
}

