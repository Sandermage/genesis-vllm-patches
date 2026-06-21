// SPDX-License-Identifier: Apache-2.0
// Kubernetes mode (read-only, P1) — cluster status + nodes with GPU. Degrades to
// a clear "connect a cluster" card when no kubeconfig/client is configured (the
// operator's setup is Docker by default, so this is the common state).
import { useEffect, useState } from "react";
import { Boxes, Cpu, Loader2, AlertTriangle, ShieldCheck, ShieldAlert, Rocket, Copy, Layers, Package } from "lucide-react";
import { api, type K8sPod, type K8sEvent, type DeploymentPlan } from "../api";
import { tr } from "../i18n";

export function K8sDeploy() {
  const [presets, setPresets] = useState<{ id: string; label: string }[]>([]);
  const [preset, setPreset] = useState("");
  const [plan, setPlan] = useState<DeploymentPlan | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    api.presets({}).then((r) => {
      const items = r.presets.map((p) => ({ id: p.id, label: `${p.id} · ${p.model}` }));
      setPresets(items);
      if (items[0]) setPreset(items[0].id);
    }).catch((e) => setErr(e instanceof Error ? e.message : String(e)));
  }, []);
  async function generate() {
    if (!preset) return;
    setBusy(true); setErr(null); setPlan(null);
    try { setPlan(await api.deployPlan({ preset_id: preset, target: "kubernetes" })); }
    catch (e) { setErr(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }
  const yaml = plan?.artifact?.content ?? "";
  return (
    <div className="k8s-deploy">
      <div className="k8s-deploy-bar">
        <label>{tr("Preset")}
          <select value={preset} onChange={(e) => setPreset(e.target.value)} aria-label={tr("Preset to deploy")}>
            {presets.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
          </select>
        </label>
        <button className="primary-button" disabled={!preset || busy} onClick={() => void generate()}>
          {busy ? <Loader2 size={14} className="spin" /> : <Rocket size={14} />} {tr("Generate manifest")}
        </button>
      </div>
      {err && <div className="containers-err"><AlertTriangle size={13} /> {err}</div>}
      {plan && (
        <>
          <div className="k8s-deploy-cmds">
            <span className="muted">{tr("Apply & watch:")}</span>
            <code>kubectl apply -f {plan.artifact.filename}</code>
            <code>kubectl rollout status deploy/sndr-{plan.preset_id.slice(0, 30)}</code>
            <button className="ghost-button" onClick={() => { navigator.clipboard?.writeText(yaml); setCopied(true); window.setTimeout(() => setCopied(false), 1500); }}>
              {copied ? <ShieldCheck size={13} /> : <Copy size={13} />} {copied ? tr("copied") : tr("copy YAML")}
            </button>
          </div>
          <pre className="k8s-yaml"><code>{yaml}</code></pre>
        </>
      )}
    </div>
  );
}

export function PodRow({ p }: { p: K8sPod }) {
  const st = p.phase === "Running" && p.ready_ok ? "online" : p.phase === "Pending" || p.phase === "Unknown" ? "partial" : p.ready_ok ? "online" : "offline";
  return (
    <tr className={`crow ${st}`}>
      <td className="crow-name">
        <span className="k8s-pod-name"><span className={`container-dot ${st}`} />{p.name}</span>
        {p.sndr_managed && (
          <span className="k8s-sndr-id" title={p.sndr_patches.length ? `${tr("Enabled patches:")} ${p.sndr_patches.join(", ")}` : undefined}>
            <span className="k8s-sndr-chip preset"><Package size={9} /> {p.sndr_preset}</span>
            {p.sndr_pin && <span className="k8s-sndr-chip"><Boxes size={9} /> {p.sndr_pin}</span>}
            {p.sndr_patch_count != null && <span className="k8s-sndr-chip"><Layers size={9} /> {p.sndr_patch_count}p</span>}
          </span>
        )}
      </td>
      <td className="muted">{p.namespace}</td>
      <td>
        <span className={`container-badge ${st}`}>{p.phase}</span>
        {p.reason && <span className="k8s-pressure" title={p.reason}><ShieldAlert size={10} /> {p.reason}</span>}
      </td>
      <td className={p.ready_ok ? "" : "muted"}>{p.ready}</td>
      <td className={p.restarts > 0 ? "k8s-restarts" : "muted"}>{p.restarts}</td>
      <td>{p.gpu_request > 0 ? <span className="k8s-gpu free"><Cpu size={11} /> {p.gpu_request}</span> : <span className="muted">—</span>}</td>
      <td className="muted">{p.node ?? "—"}</td>
    </tr>
  );
}

export function EventRow({ e }: { e: K8sEvent }) {
  const warn = e.type === "Warning";
  return (
    <tr className={`crow ${warn ? "offline" : ""}`}>
      <td><span className={`container-badge ${warn ? "offline" : "online"}`}>{e.type}</span></td>
      <td className="muted">{e.reason}</td>
      <td className="muted">{e.object}</td>
      <td className="k8s-evmsg" title={e.message ?? ""}>{e.message}</td>
      <td className="muted">{e.count ?? ""}</td>
    </tr>
  );
}
