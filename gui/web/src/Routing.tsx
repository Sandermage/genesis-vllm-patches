import { useEffect, useState } from "react";
import { AlertTriangle, ArrowRight, Ban, Check, Cpu, Route, Workflow } from "lucide-react";
import { api, type RoutingActive, type RoutingArtifact, type RoutingArtifacts, type RoutingClassify, type RoutingSignals } from "./api";

const pctTone = (v: number) => (v > 0.01 ? "ok" : v < -0.01 ? "hot" : "");
const fmtPct = (v: number | null | undefined) => (typeof v === "number" ? `${v > 0 ? "+" : ""}${(v * 100).toFixed(0)}%` : "—");
const clamp = (v: number, lo = 0, hi = 100) => Math.max(lo, Math.min(hi, v));

// Diverging bar: positive deltas grow right (green), negative left (red).
function DeltaBar({ delta }: { delta: number }) {
  const mag = clamp((Math.abs(delta) / 0.6) * 50); // 60% delta → full half-width
  const tone = pctTone(delta);
  return (
    <div className="rt-delta-track">
      <span className="rt-delta-mid" />
      <span className={`rt-delta-fill ${tone}`} style={delta >= 0 ? { left: "50%", width: `${mag}%` } : { right: "50%", width: `${mag}%` }} />
    </div>
  );
}

function WorkloadRow({ wclass, art }: { wclass: string; art: RoutingArtifact }) {
  const delta = art.delta_tps_per_class[wclass];
  const tps = art.profile_tps_per_class[wclass];
  const allowed = art.allowed_workloads.includes(wclass);
  const denied = art.denied_workloads.includes(wclass);
  return (
    <div className="rt-wl">
      <span className="rt-wl-name">{wclass}</span>
      <span className={`rt-wl-gate ${allowed ? "ok" : denied ? "deny" : "muted"}`}>
        {allowed ? <><Check size={11} /> allowed</> : denied ? <><Ban size={11} /> denied</> : "—"}
      </span>
      <DeltaBar delta={typeof delta === "number" ? delta : 0} />
      <span className={`rt-wl-delta ${pctTone(typeof delta === "number" ? delta : 0)}`}>{fmtPct(delta)}</span>
      <span className="rt-wl-tps">{typeof tps === "number" ? `${tps.toFixed(1)} tok/s` : "—"}</span>
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
            <option value="">— none —</option>
            <option value="json_object">json_object</option>
            <option value="json_schema">json_schema</option>
          </select>
        </label>
        <label className="rt-cls-field"><span>tool_choice</span>
          <select value={tc} onChange={(e) => setTc(e.target.value)}>
            <option value="">— none —</option>
            <option value="required">required</option>
            <option value="function">function</option>
          </select>
        </label>
        <label className="rt-cls-field"><span>workload_class</span>
          <select value={wc} onChange={(e) => setWc(e.target.value)}>
            <option value="">— none —</option>
            {workloadClasses.map((w) => <option key={w} value={w}>{w}</option>)}
          </select>
        </label>
      </div>
      {res && (
        <div className={`rt-verdict ${accepted ? "ok" : "fallback"}`}>
          <div className="rt-verdict-head">
            <span className="rt-verdict-sig">signal: <strong>{res.signal}</strong></span>
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
  const [arts, setArts] = useState<RoutingArtifacts | null>(null);
  const [active, setActive] = useState<RoutingActive | null>(null);
  const [profile, setProfile] = useState<string>("");

  useEffect(() => {
    api.routingArtifacts().then(setArts).catch(() => setArts({ available: false, reason: "daemon offline", artifacts: [] }));
    api.routingActive().then((a) => { setActive(a); if (a.profile) setProfile(a.profile); }).catch(() => {});
  }, []);

  if (arts && !arts.available) {
    return (
      <div className="rt-empty">
        <Workflow size={22} />
        <strong>Spec-decode routing unavailable</strong>
        <span>{arts.reason ?? "The spec_decode integration is not present in this deployment (the GUI ships with sndr_core; the engine/patch layer may be absent)."}</span>
      </div>
    );
  }

  const list = arts?.artifacts ?? [];
  const art = list.find((a) => a.profile === profile) ?? null;
  const candidates = active?.candidates ?? list.map((a) => a.profile);

  return (
    <div className="rt">
      <div className="rt-bar-top">
        <label className="rt-source">
          <Route size={14} />
          <select value={profile} onChange={(e) => setProfile(e.target.value)}>
            {candidates.length === 0 && <option value="">— no profiles —</option>}
            {candidates.map((p) => <option key={p} value={p}>{p}</option>)}
          </select>
        </label>
        {active?.source && <span className="rt-source-tag">active: {active.source}</span>}
      </div>

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
            <div className="rt-kpi"><span className="rt-kpi-l">Global Δ</span><strong className={`rt-kpi-v ${pctTone(art.profile_delta_global ?? 0)}`}>{fmtPct(art.profile_delta_global)}</strong></div>
            <div className="rt-kpi"><span className="rt-kpi-l">Accept mean</span><strong className="rt-kpi-v">{art.acceptance_mean != null ? art.acceptance_mean.toFixed(2) : "—"}</strong></div>
            <div className="rt-kpi"><span className="rt-kpi-l">VRAM floor</span><strong className="rt-kpi-v">{art.vram_free_mib_min != null ? `${(art.vram_free_mib_min / 1024).toFixed(1)} GB free` : "—"}</strong></div>
            <div className="rt-kpi"><span className="rt-kpi-l">Allowed</span><strong className="rt-kpi-v">{art.allowed_workloads.length}/{art.workload_classes.length}</strong></div>
          </div>
          {art.notes && <div className="rt-notes">{art.notes}</div>}
          <div className="rt-wl-head"><span>Workload</span><span>Gate</span><span>TPS delta vs baseline</span><span /><span /></div>
          <div className="rt-wl-list">
            {art.workload_classes.map((w) => <WorkloadRow key={w} wclass={w} art={art} />)}
          </div>
        </div>
      )}

      {art && (
        <div className="rt-section">
          <div className="rt-section-t"><Workflow size={13} /> Request classifier — predict routing for a request shape</div>
          <Classifier profile={art.profile} workloadClasses={art.workload_classes} />
        </div>
      )}

      {arts && list.length === 0 && (
        <div className="rt-empty"><AlertTriangle size={20} /><strong>No artifacts</strong><span>No bench-validated profiles found in the artifacts directory.</span></div>
      )}
    </div>
  );
}
