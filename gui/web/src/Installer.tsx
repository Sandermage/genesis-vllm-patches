import { useEffect, useState } from "react";
import { AlertTriangle, Box, CheckCircle2, Container, Copy, Cpu, HardDrive, Loader2, Lock, Package, Rocket, Server, ShieldAlert, Tag } from "lucide-react";
import { api, type InstallPlan, type InstallTarget, type InstallTargets, type PresetRecord } from "./api";

const TARGET_ICON: Record<string, JSX.Element> = {
  compose: <Container size={18} />, quadlet: <Box size={18} />, kubernetes: <Package size={18} />,
  systemd: <HardDrive size={18} />, bare_metal: <Cpu size={18} />, proxmox: <Server size={18} />, proxmox_vm: <Server size={18} />,
};

// First-run remote-install wizard: pick host → target → preset → review a
// dry-run plan (artifact + ordered steps, infra-mutating steps flagged). It
// renders and copies; it never executes (the gated SSH apply is a later phase).
export function InstallWizard({ initial }: { initial?: { hostId?: string; target?: string } }) {
  const [meta, setMeta] = useState<InstallTargets | null>(null);
  const [presets, setPresets] = useState<PresetRecord[]>([]);
  const [hostId, setHostId] = useState(initial?.hostId || "");
  const [target, setTarget] = useState(initial?.target || "compose");
  const [presetId, setPresetId] = useState("");
  const [pin, setPin] = useState("");
  const [withDaemon, setWithDaemon] = useState(false);
  const [plan, setPlan] = useState<InstallPlan | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [confirming, setConfirming] = useState(false);
  const [applying, setApplying] = useState(false);
  const [applied, setApplied] = useState<import("./api").InstallApplyResult | null>(null);

  useEffect(() => {
    api.installTargets().then((m) => { setMeta(m); if (!initial?.hostId && m.hosts[0]) setHostId(m.hosts[0].id); }).catch(() => {});
    api.presets({}).then((p) => { setPresets(p.presets); if (p.presets[0]) setPresetId(p.presets[0].id); }).catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function buildPlan() {
    if (!hostId || !presetId || !target) return;
    setLoading(true); setErr(null); setPlan(null); setApplied(null); setConfirming(false);
    try { setPlan(await api.installPlan(hostId, presetId, target, pin.trim() || undefined, withDaemon)); }
    catch (e) { setErr(e instanceof Error ? e.message : String(e)); }
    finally { setLoading(false); }
  }

  async function runApply() {
    setApplying(true); setErr(null); setConfirming(false);
    try { setApplied(await api.installApply(hostId, presetId, target, pin.trim() || undefined, withDaemon)); }
    catch (e) { setErr(e instanceof Error ? e.message : String(e)); }
    finally { setApplying(false); }
  }

  const targets = meta?.targets ?? [];
  const applyOn = meta?.apply_enabled ?? false;

  return (
    <div className="installer">
      <ol className="install-steps">
        <li><span className="install-num">1</span> Host</li>
        <li><span className="install-num">2</span> Target</li>
        <li><span className="install-num">3</span> Preset</li>
        <li><span className="install-num">4</span> Review &amp; install</li>
      </ol>

      <div className="install-row">
        <label className="param-field"><span><Server size={11} /> Host (from registry)</span>
          <select value={hostId} onChange={(e) => { setHostId(e.target.value); setPlan(null); }}>
            <option value="">— pick a host —</option>
            {meta?.hosts.map((h) => <option key={h.id} value={h.id}>{h.label} · {h.host}{h.gpu_arch ? ` · ${h.gpus}× ${h.gpu_arch}` : ""}</option>)}
          </select>
        </label>
        <label className="param-field"><span><Package size={11} /> Preset to install</span>
          <select value={presetId} onChange={(e) => { setPresetId(e.target.value); setPlan(null); }}>
            {presets.map((p) => <option key={p.id} value={p.id}>{p.id}</option>)}
          </select>
        </label>
        <label className="param-field"><span><Tag size={11} /> Engine pin / image <em className="install-opt">(optional)</em></span>
          <input value={pin} onChange={(e) => { setPin(e.target.value); setPlan(null); }}
            placeholder="preset default — e.g. vllm/vllm-openai:nightly-<sha> or @sha256:…"
            spellCheck={false} />
        </label>
      </div>

      <div className="install-targets">
        <span className="install-label">Install target</span>
        <div className="install-target-grid">
          {targets.map((t: InstallTarget) => (
            <button key={t.id} className={`install-target ${target === t.id ? "active" : ""}`} onClick={() => { setTarget(t.id); setPlan(null); }}>
              <span className="install-target-head">{TARGET_ICON[t.id] || <Box size={18} />} <strong>{t.label}</strong>{(t.id === "proxmox" || t.id === "proxmox_vm") && <ShieldAlert size={12} className="install-infra-tag" />}</span>
              <span className="install-target-sum">{t.summary}</span>
              {t.needs && <span className="install-needs">needs {t.needs}</span>}
            </button>
          ))}
        </div>
      </div>

      <label className="install-daemon-toggle">
        <input type="checkbox" checked={withDaemon} onChange={(e) => { setWithDaemon(e.target.checked); setPlan(null); }} />
        <Cpu size={13} /> Also install the SNDR management daemon (GUI + sndr_core) alongside the engine
        <span className="install-opt"> — one apply lands GUI + core + engine; sidecar for containers, native systemd for bare-metal, inside the guest for Proxmox</span>
      </label>

      <div className="install-actions">
        <button className="primary-action" onClick={() => void buildPlan()} disabled={!hostId || !presetId || loading}>
          {loading ? <Loader2 size={14} className="spin" /> : <Rocket size={14} />} Build install plan (dry-run)
        </button>
      </div>
      {err && <div className="install-err"><AlertTriangle size={14} /> {err}</div>}

      {plan && (
        <div className="install-plan">
          <div className="install-plan-head">
            <strong>{plan.target_label} → {plan.host.label}</strong>
            <span className="install-dry">dry-run · nothing executed</span>
            {plan.image_override && <span className="install-pin"><Tag size={12} /> pinned: {plan.image_override}</span>}
            {plan.with_daemon && <span className="install-pin"><Cpu size={12} /> + SNDR daemon</span>}
            {plan.danger_count > 0 && <span className="install-danger-count"><ShieldAlert size={12} /> {plan.danger_count} infra-mutating step{plan.danger_count > 1 ? "s" : ""}</span>}
          </div>
          {plan.provisions_infra && <div className="install-infra-warn"><ShieldAlert size={14} /> This target <b>provisions infrastructure</b> on the Proxmox host (creates a {plan.target === "proxmox_vm" ? "VM" : "container"}). Review every step; run only on the Proxmox node.</div>}

          <ol className="install-step-list">
            {plan.steps.map((s) => (
              <li key={s.order} className={`install-step ${s.danger ? "danger" : ""}`}>
                <span className={`install-step-kind k-${s.kind}`}>{s.kind}</span>
                <span className="install-step-title">{s.danger && <ShieldAlert size={11} />} {s.title}</span>
                {s.cmd && <button className="icon-only" title="Copy command" onClick={() => void navigator.clipboard?.writeText(s.cmd!)}><Copy size={12} /></button>}
              </li>
            ))}
          </ol>

          <div className="install-artifact">
            <div className="install-artifact-head"><span>{plan.artifact.filename} · {plan.artifact.kind}</span>
              <button className="ghost-button" onClick={() => void navigator.clipboard?.writeText(plan.artifact.content)}><Copy size={13} /> Copy artifact</button>
            </div>
            <pre className="install-artifact-body">{plan.artifact.content}</pre>
          </div>

          <div className="install-apply">
            {!confirming ? (
              <button className="primary-action" disabled={!applyOn || applying || !!applied}
                onClick={() => setConfirming(true)}
                title={applyOn ? "Run this plan on the host over SSH" : "Start the daemon with SNDR_ENABLE_APPLY=1 to enable remote execution"}>
                {applyOn ? <Rocket size={14} /> : <Lock size={14} />} Install over SSH
              </button>
            ) : (
              <span className="install-confirm">
                <ShieldAlert size={14} /> Run {plan.steps.filter((s) => s.kind === "remote-exec").length} command(s) on <b>{plan.host.label}</b> over SSH?
                <button className="primary-action danger" onClick={() => void runApply()} disabled={applying}>
                  {applying ? <Loader2 size={14} className="spin" /> : <CheckCircle2 size={14} />} Yes, run it
                </button>
                <button className="ghost-button" onClick={() => setConfirming(false)} disabled={applying}>Cancel</button>
              </span>
            )}
            {!applied && <span className="install-apply-note">{applyOn ? plan.notes : `${plan.notes} (daemon is read-only — set SNDR_ENABLE_APPLY=1)`}</span>}
          </div>

          {applied && (
            <div className={`install-result ${applied.ok ? "ok" : "fail"}`}>
              <div className="install-result-head">
                {applied.ok ? <CheckCircle2 size={15} /> : <AlertTriangle size={15} />}
                <strong>{applied.ok ? `Installed on ${plan.host.label}` : "Install failed"}</strong>
                {applied.error && <span className="install-result-err">{applied.error}</span>}
              </div>
              <ol className="install-result-steps">
                {applied.steps.map((s, i) => (
                  <li key={i} className={s.rc === 0 ? "ok" : "fail"}>
                    <code className="install-result-cmd">{s.rc === 0 ? "✓" : "✗"} {s.cmd}</code>
                    {s.output && <pre className="install-result-out">{s.output}</pre>}
                  </li>
                ))}
              </ol>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
