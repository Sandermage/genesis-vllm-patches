import { useEffect, useState, type ReactNode } from "react";
import { AlertTriangle, ArrowRight, Ban, Check, Cpu, Route, Workflow, Info, ChevronDown, Activity, Database, Zap, Clock, Layers, Server } from "lucide-react";
import { api, type RoutingActive, type RoutingArtifact, type RoutingArtifacts, type RoutingClassify, type RoutingSignals, type EngineMetrics, type ExternalOverview } from "./api";
import { useLang, t, tr, type Lang } from "./i18n";
import { onKeyActivate } from "./dialog";

// Workload classes the router knows — used by the classifier when no bench
// artifact is loaded (the classify endpoint works against the active profile).
const DEFAULT_WORKLOADS = ["free_chat", "code_gen", "tool_calls", "structured_json", "summarization", "long_context"];

function RoutingIntro({ lang }: { lang: Lang }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="k8s-intro rt-intro">
      <div className="k8s-intro-head" role="button" tabIndex={0} onClick={() => setOpen((v) => !v)} onKeyDown={onKeyActivate(() => setOpen((v) => !v))}>
        <Info size={14} />
        <span><strong>{t(lang, "rt.title")}</strong> — {t(lang, "rt.intro")}</span>
        <ChevronDown size={15} className={open ? "rot" : ""} />
      </div>
      {open && <div className="k8s-intro-body"><p><b>{t(lang, "rt.how")}.</b> {t(lang, "rt.howBody")}</p></div>}
    </div>
  );
}

// Live online data — what the active engine is actually serving right now.
function RoutingLive({ lang }: { lang: Lang }) {
  const [m, setM] = useState<EngineMetrics | null>(null);
  useEffect(() => {
    let alive = true;
    const pull = () => { if (!document.hidden) api.engineMetrics().then((r) => { if (alive) setM(r); }).catch(() => {}); };
    pull();
    const id = window.setInterval(pull, 3000);
    return () => { alive = false; window.clearInterval(id); };
  }, []);
  if (!m?.reachable || Object.keys(m.kpis).length === 0) return null;
  const k = m.kpis;
  const sec = (v?: number) => (v == null ? "—" : v >= 1 ? `${v.toFixed(2)} s` : `${Math.round(v * 1000)} ms`);
  const accept = k.spec_decode_acceptance_rate;
  const cells: { icon: ReactNode; l: string; v: string; tone?: string }[] = [
    { icon: <Activity size={13} />, l: tr("Running"), v: String(Math.round(k.requests_running ?? 0)) },
    { icon: <Clock size={13} />, l: tr("Waiting"), v: String(Math.round(k.requests_waiting ?? 0)), tone: (k.requests_waiting ?? 0) > 0 ? "warn" : "" },
    { icon: <Database size={13} />, l: "KV-cache", v: k.kv_cache_usage != null ? `${(k.kv_cache_usage * 100).toFixed(0)}%` : "—" },
    { icon: <Zap size={13} />, l: "tok/s", v: k.generation_toks_per_s != null ? k.generation_toks_per_s.toFixed(0) : "—" },
    { icon: <Clock size={13} />, l: "TTFT", v: sec(k.ttft_avg_s) },
    { icon: <Layers size={13} />, l: tr("MTP accept"), v: accept != null ? `${(accept * 100).toFixed(0)}%` : "—" },
  ];
  return (
    <div className="rt-section rt-live">
      <div className="rt-section-t"><Activity size={13} /> {t(lang, "rt.live")} <span className="muted">· {t(lang, "rt.liveHelp")}</span></div>
      <div className="rt-live-grid">
        {cells.map((c) => <div key={c.l} className="rt-live-cell"><span className="rt-live-l">{c.icon} {c.l}</span><b className={`rt-live-v tone-${c.tone || "n"}`}>{c.v}</b></div>)}
      </div>
    </div>
  );
}

// Live status of the adjacent proxy + aggregator (separate external projects;
// SNDR only connects). Self-gating: the endpoint returns enabled=false without
// the SNDR_ENABLE_EXTERNAL_SERVICES key, and we render an enable hint instead of
// data — so the GUI only surfaces this functionality when the operator opts in.
function RoutingExternal() {
  const [data, setData] = useState<ExternalOverview | null>(null);
  const [loaded, setLoaded] = useState(false);
  useEffect(() => {
    let alive = true;
    api.externalOverview()
      .then((r) => { if (alive) { setData(r); setLoaded(true); } })
      .catch(() => { if (alive) setLoaded(true); });
    return () => { alive = false; };
  }, []);
  if (!loaded) return null;
  if (!data?.enabled) {
    return (
      <div className="rt-section rt-external">
        <div className="rt-section-t"><Server size={13} /> {tr("Adjacent services")} <span className="muted">· {tr("proxy + aggregator")}</span></div>
        <p className="muted"><Ban size={13} /> {tr("Disabled — set SNDR_ENABLE_EXTERNAL_SERVICES=1 to connect to the Genesis proxy + aggregator.")}</p>
      </div>
    );
  }
  const cell = (label: string, sec: Record<string, unknown> | undefined, key: string) => {
    const v = sec?.[key];
    const err = v && typeof v === "object" && "error" in (v as object) ? String((v as Record<string, unknown>).error) : null;
    return (
      <div key={key} className="rt-live-cell" title={err || tr("connected")}>
        <span className="rt-live-l"><Server size={13} /> {label}</span>
        <b className={`rt-live-v tone-${err ? "hot" : "ok"}`}>{err ? tr("down") : tr("connected")}</b>
      </div>
    );
  };
  return (
    <div className="rt-section rt-external">
      <div className="rt-section-t"><Server size={13} /> {tr("Adjacent services")} <span className="muted">· {tr("live proxy + aggregator status")}</span></div>
      <div className="rt-live-grid">
        {cell(tr("Proxy health"), data.proxy, "health")}
        {cell(tr("Proxy cost"), data.proxy, "cost")}
        {cell(tr("Proxy routing"), data.proxy, "routing")}
        {cell(tr("Agg signals"), data.aggregator, "signals")}
        {cell(tr("Agg anomalies"), data.aggregator, "anomalies")}
        {cell(tr("Agg patterns"), data.aggregator, "patterns")}
      </div>
    </div>
  );
}

const pctTone = (v: number) => (v > 0.01 ? "ok" : v < -0.01 ? "hot" : "");
const fmtPct = (v: number | null | undefined) => (typeof v === "number" ? `${v > 0 ? "+" : ""}${(v * 100).toFixed(0)}%` : "—");
const clamp = (v: number, lo = 0, hi = 100) => Math.max(lo, Math.min(hi, v));

function WorkloadRow({ wclass, art, maxTps }: { wclass: string; art: RoutingArtifact; maxTps: number }) {
  const tps = art.profile_tps_per_class[wclass];
  const base = art.baseline_tps_per_class[wclass];
  const delta = art.delta_tps_per_class[wclass];
  const allowed = art.allowed_workloads.includes(wclass);
  const denied = art.denied_workloads.includes(wclass);
  // Lead with measured throughput (the always-present signal); the delta vs a
  // baseline is shown only when a baseline run exists for this profile.
  const pct = typeof tps === "number" && maxTps > 0 ? clamp((tps / maxTps) * 100) : 0;
  const hasDelta = typeof delta === "number";
  return (
    <div className="rt-wl">
      <span className="rt-wl-name">{wclass}</span>
      <span className={`rt-wl-gate ${allowed ? "ok" : denied ? "deny" : "muted"}`}>
        {allowed ? <><Check size={11} /> {tr("allowed")}</> : denied ? <><Ban size={11} /> {tr("denied")}</> : "—"}
      </span>
      <div className="rt-tps-track"><span className={`rt-tps-fill ${denied ? "deny" : "ok"}`} style={{ width: `${pct}%` }} /></div>
      <span className="rt-wl-tps"><b>{typeof tps === "number" ? tps.toFixed(0) : "—"}</b> tok/s</span>
      <span className={`rt-wl-base ${hasDelta ? pctTone(delta as number) : ""}`}>
        {hasDelta ? `${fmtPct(delta)} ${tr("vs base")}` : typeof base === "number" ? `${tr("base")} ${base.toFixed(0)}` : ""}
      </span>
    </div>
  );
}

function Classifier({ profile, workloadClasses }: { profile: string; workloadClasses: string[] }) {
  const [rf, setRf] = useState("");
  const [tc, setTc] = useState("");
  const [wc, setWc] = useState("");
  const [res, setRes] = useState<RoutingClassify | null>(null);

  useEffect(() => {
    if (!profile) { setRes(null); return; }
    const signals: RoutingSignals = {};
    if (rf) signals.response_format = { type: rf };
    if (tc) signals.tool_choice = tc === "required" ? "required" : { type: "function" };
    if (wc) signals.workload_class = wc;
    let alive = true;
    api.routingClassify(signals, profile).then((r) => { if (alive) setRes(r); }).catch(() => {});
    return () => { alive = false; };
  }, [profile, rf, tc, wc]);

  const accepted = res?.accepted === true;
  return (
    <div className="rt-classifier">
      <div className="rt-cls-controls">
        <label className="rt-cls-field"><span>response_format</span>
          <select value={rf} onChange={(e) => setRf(e.target.value)}>
            <option value="">{tr("— none —")}</option>
            <option value="json_object">json_object</option>
            <option value="json_schema">json_schema</option>
          </select>
        </label>
        <label className="rt-cls-field"><span>tool_choice</span>
          <select value={tc} onChange={(e) => setTc(e.target.value)}>
            <option value="">{tr("— none —")}</option>
            <option value="required">required</option>
            <option value="function">function</option>
          </select>
        </label>
        <label className="rt-cls-field"><span>workload_class</span>
          <select value={wc} onChange={(e) => setWc(e.target.value)}>
            <option value="">{tr("— none —")}</option>
            {workloadClasses.map((w) => <option key={w} value={w}>{w}</option>)}
          </select>
        </label>
      </div>
      {res && (
        <div className={`rt-verdict ${accepted ? "ok" : "fallback"}`}>
          <div className="rt-verdict-head">
            <span className="rt-verdict-sig">{tr("signal")}: <strong>{res.signal}</strong></span>
            <ArrowRight size={14} />
            <span className={`rt-verdict-prof ${accepted ? "ok" : "fallback"}`}>
              {accepted ? <Check size={13} /> : <ArrowRight size={13} />}
              {res.profile}
            </span>
            {res.workload_class && <span className="rt-verdict-wc">{res.workload_class}</span>}
            {typeof res.expected_delta_tps === "number" && (
              <span className={`rt-verdict-delta ${pctTone(res.expected_delta_tps)}`}>{fmtPct(res.expected_delta_tps)} TPS</span>
            )}
          </div>
          <div className="rt-verdict-reason">{res.reason}</div>
        </div>
      )}
    </div>
  );
}

export function RoutingPanel() {
  const [lang] = useLang();
  const [arts, setArts] = useState<RoutingArtifacts | null>(null);
  const [active, setActive] = useState<RoutingActive | null>(null);
  const [profile, setProfile] = useState<string>("");
  const [setting, setSetting] = useState(false);

  useEffect(() => {
    api.routingArtifacts().then(setArts).catch(() => setArts({ available: false, reason: "daemon offline", artifacts: [] }));
    api.routingActive().then((a) => { setActive(a); if (a.profile) setProfile(a.profile); }).catch(() => {});
  }, []);

  if (arts && !arts.available) {
    return (
      <div className="rt-empty">
        <Workflow size={22} />
        <strong>{tr("Spec-decode routing unavailable")}</strong>
        <span>{arts.reason ?? tr("The spec_decode integration is not present in this deployment (the GUI ships with sndr_core; the engine/patch layer may be absent).")}</span>
      </div>
    );
  }

  const list = arts?.artifacts ?? [];
  const art = list.find((a) => a.profile === profile) ?? null;
  const candidates = active?.candidates ?? list.map((a) => a.profile);

  return (
    <div className="rt">
      <RoutingIntro lang={lang} />

      <div className="rt-bar-top">
        <label className="rt-source">
          <Route size={14} />
          <select aria-label={tr("Routing profile")} value={profile} onChange={(e) => setProfile(e.target.value)}>
            {candidates.length === 0 && <option value="">{tr("— no profiles —")}</option>}
            {candidates.map((p) => <option key={p} value={p}>{p}</option>)}
          </select>
        </label>
        <button className="rt-set-active" disabled={!profile || profile === active?.profile || setting}
          title={tr("Pin this profile as the daemon's active route (clears on restart; set SNDR_ACTIVE_PROFILE to persist for the gateway)")}
          onClick={async () => { setSetting(true); const r = await api.routingSetActive(profile).catch(() => null); if (r) setActive(r); setSetting(false); }}>
          {profile && profile === active?.profile ? <><Check size={13} /> {tr("active")}</> : tr("Set active")}
        </button>
        {active?.source && <span className="rt-source-tag">{tr("source")}: {active.source}</span>}
        {profile && <code className="rt-env-hint" title={tr("Set on the gateway process to persist across restarts")}>SNDR_ACTIVE_PROFILE={profile}</code>}
      </div>

      <RoutingLive lang={lang} />

      <RoutingExternal />

      {art && (
        <div className="rt-profile">
          <div className="rt-profile-head">
            <div className="rt-profile-id">
              <strong className="rt-profile-name"><Cpu size={15} /> {art.profile}</strong>
              <span className="rt-profile-meta">{art.model_id}{art.k != null ? ` · K=${art.k}` : ""} · {art.vllm_pin}</span>
            </div>
            <span className={`rt-decision ${art.decision.includes("conditional") ? "warn" : art.decision.includes("denied") ? "hot" : "ok"}`}>{art.decision}</span>
          </div>
          <div className="rt-profile-kpis">
            <div className="rt-kpi"><span className="rt-kpi-l">{tr("Global")} Δ</span><strong className={`rt-kpi-v ${pctTone(art.profile_delta_global ?? 0)}`}>{fmtPct(art.profile_delta_global)}</strong></div>
            <div className="rt-kpi"><span className="rt-kpi-l">{tr("Accept mean")}</span><strong className="rt-kpi-v">{art.acceptance_mean != null ? art.acceptance_mean.toFixed(2) : "—"}</strong></div>
            <div className="rt-kpi"><span className="rt-kpi-l">{tr("VRAM floor")}</span><strong className="rt-kpi-v">{art.vram_free_mib_min != null ? `${(art.vram_free_mib_min / 1024).toFixed(1)} ${tr("GB free")}` : "—"}</strong></div>
            <div className="rt-kpi"><span className="rt-kpi-l">{tr("Allowed")}</span><strong className="rt-kpi-v">{art.allowed_workloads.length}/{art.workload_classes.length}</strong></div>
          </div>
          {art.notes && <div className="rt-notes">{art.notes}</div>}
          <div className="rt-wl-head"><span>{tr("Workload")}</span><span>{tr("Gate")}</span><span>{tr("Measured throughput")}</span><span>tok/s</span><span>{tr("vs baseline")}</span></div>
          <div className="rt-wl-list">
            {(() => {
              const tpsVals = art.workload_classes.map((w) => art.profile_tps_per_class[w]).filter((v): v is number => typeof v === "number");
              const maxTps = tpsVals.length ? Math.max(...tpsVals) : 0;
              return art.workload_classes.map((w) => <WorkloadRow key={w} wclass={w} art={art} maxTps={maxTps} />);
            })()}
          </div>
        </div>
      )}

      {arts && list.length === 0 && (
        <div className="rt-note"><AlertTriangle size={14} /> <span>{tr("No bench-validated artifacts in this deployment — the per-workload TPS table appears once profiles are benched. The classifier below works live against the active profile regardless.")}</span></div>
      )}

      <div className="rt-section">
        <div className="rt-section-t"><Workflow size={13} /> {t(lang, "rt.classifier")} <span className="muted">· {t(lang, "rt.classifierHelp")}</span></div>
        <Classifier profile={art?.profile ?? profile} workloadClasses={art?.workload_classes ?? DEFAULT_WORKLOADS} />
      </div>
    </div>
  );
}
