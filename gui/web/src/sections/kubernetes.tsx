// SPDX-License-Identifier: Apache-2.0
// Kubernetes mode (read-only, P1) — cluster status + nodes with GPU. Degrades to
// a clear "connect a cluster" card when no kubeconfig/client is configured (the
// operator's setup is Docker by default, so this is the common state).
import { useEffect, useState, type ReactNode } from "react";
import { Boxes, Cpu, RefreshCw, Loader2, AlertTriangle, ShieldCheck, ShieldAlert, Server, Info, ChevronDown, Rocket, Copy } from "lucide-react";
import { api, type K8sStatus, type K8sNode, type K8sPod, type K8sEvent, type DeploymentPlan } from "../api";
import { onKeyActivate } from "../dialog";

type K8sTab = "nodes" | "pods" | "events" | "deploy";

export function KubernetesPanel() {
  const [status, setStatus] = useState<K8sStatus | null>(null);
  const [nodes, setNodes] = useState<K8sNode[] | null>(null);
  const [pods, setPods] = useState<K8sPod[] | null>(null);
  const [events, setEvents] = useState<K8sEvent[] | null>(null);
  const [tab, setTab] = useState<K8sTab>("nodes");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function load() {
    setLoading(true); setErr(null);
    try {
      const s = await api.k8sStatus();
      setStatus(s);
      if (s.available) {
        const [n, p, e] = await Promise.all([api.k8sNodes(), api.k8sPods(), api.k8sEvents()]);
        setNodes(n.nodes); setPods(p.pods); setEvents(e.events);
      } else { setNodes(null); setPods(null); setEvents(null); }
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)); }
    finally { setLoading(false); }
  }
  useEffect(() => { void load(); }, []);

  const clusterGated = (body: ReactNode) =>
    !status ? <div className="containers-empty"><Loader2 size={20} className="spin" /></div>
      : !status.available ? <ConnectCard error={status.error} /> : body;

  return (
    <div className="k8s">
      <div className="section-head">
        <div><h2><Boxes size={18} /> Kubernetes</h2><p className="muted">Monitor your cluster and deploy vLLM to it — honours your kubeconfig &amp; RBAC.</p></div>
        <button className="ghost-button" onClick={() => void load()} disabled={loading}>
          {loading ? <Loader2 size={14} className="spin" /> : <RefreshCw size={14} />} Refresh
        </button>
      </div>

      <K8sIntro available={status?.available} />
      {err && <div className="containers-err"><AlertTriangle size={13} /> {err}</div>}

      {status?.available && (
        <div className="k8s-kpis">
          <Kpi label="Server" value={status.version ?? "—"} />
          <Kpi label="Nodes" value={`${status.nodes_ready ?? 0}/${status.node_count ?? 0} ready`} />
          <Kpi label="GPU nodes" value={String(status.gpu_node_count ?? 0)} accent />
          <Kpi label="Namespaces" value={String(status.namespace_count ?? 0)} />
        </div>
      )}

      <div className="k8s-tabs">
        {(["nodes", "pods", "events", "deploy"] as K8sTab[]).map((t) => (
          <button key={t} className={tab === t ? "active" : ""} onClick={() => setTab(t)}>
            {t === "deploy" ? "deploy vLLM" : t}
            {t === "nodes" && nodes ? ` (${nodes.length})` : ""}{t === "pods" && pods ? ` (${pods.length})` : ""}
            {t === "events" && events ? ` (${events.filter((e) => e.type === "Warning").length}⚠)` : ""}
          </button>
        ))}
      </div>

      {tab === "deploy" && <K8sDeploy />}

      {tab === "nodes" && clusterGated(
        <div className="containers-table-wrap">
          <table className="containers-table">
            <thead><tr><th>Node</th><th>Status</th><th>Roles</th><th>Kubelet</th><th>GPU (free / alloc)</th><th>GPU labels</th><th>Taints</th></tr></thead>
            <tbody>{(nodes ?? []).map((n) => <NodeRow key={n.name} n={n} />)}</tbody>
          </table>
          {nodes && nodes.length === 0 && <div className="containers-empty"><strong>Cluster has no nodes</strong></div>}
        </div>
      )}
      {tab === "pods" && clusterGated(
        <div className="containers-table-wrap">
          <table className="containers-table">
            <thead><tr><th>Pod</th><th>Namespace</th><th>Phase</th><th>Ready</th><th>Restarts</th><th>GPU</th><th>Node</th></tr></thead>
            <tbody>{(pods ?? []).map((p) => <PodRow key={`${p.namespace}/${p.name}`} p={p} />)}</tbody>
          </table>
          {pods && pods.length === 0 && <div className="containers-empty"><strong>No pods</strong></div>}
        </div>
      )}
      {tab === "events" && clusterGated(
        <div className="containers-table-wrap">
          <table className="containers-table">
            <thead><tr><th>Type</th><th>Reason</th><th>Object</th><th>Message</th><th>×</th></tr></thead>
            <tbody>{(events ?? []).map((e, i) => <EventRow key={i} e={e} />)}</tbody>
          </table>
          {events && events.length === 0 && <div className="containers-empty"><strong>No recent events</strong></div>}
        </div>
      )}
    </div>
  );
}

function ConnectCard({ error }: { error: string | null }) {
  return (
    <div className="k8s-disconnected">
      <Server size={24} />
      <strong>No cluster connected</strong>
      <p className="muted">{error}</p>
      <p className="upd-hint">Install the client and point the daemon at a kubeconfig (the Deploy tab works without a cluster):</p>
      <code className="apply-gate-cmd">pip install 'vllm-sndr-core[k8s]'  ·  export KUBECONFIG=/path/to/config</code>
    </div>
  );
}

// What k8s mode is + the GPU prerequisites — so it's a real, explained option.
function K8sIntro({ available }: { available?: boolean }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="k8s-intro">
      <div className="k8s-intro-head" role="button" tabIndex={0} onClick={() => setOpen((v) => !v)} onKeyDown={onKeyActivate(() => setOpen((v) => !v))}>
        <Info size={14} />
        <span><strong>Run vLLM on Kubernetes</strong> — {available ? "monitor this cluster (nodes · pods · events) and deploy presets to it." : "generate a ready-to-apply manifest from any preset; connect a kubeconfig to also monitor the cluster."}</span>
        <ChevronDown size={15} className={open ? "rot" : ""} />
      </div>
      {open && (
        <div className="k8s-intro-body">
          <p className="k8s-intro-lead">Kubernetes is a cluster orchestrator. Today you run vLLM as Docker containers you start/stop per host (the Containers tab). Kubernetes lets you treat <b>several GPU servers as one pool</b> and have it keep your engines running for you.</p>
          <p><b>What it gives you over plain Docker:</b></p>
          <ul className="k8s-intro-list">
            <li><b>Self-healing</b> — if an engine crashes or a node reboots, k8s restarts/reschedules it automatically. No manual SSH-and-restart.</li>
            <li><b>GPU-aware scheduling</b> — you declare "this engine needs 2 GPUs" and k8s places it on a node that has them free, across the whole fleet.</li>
            <li><b>Scale</b> — run N replicas of an engine behind one Service/endpoint; bump the count to add capacity.</li>
            <li><b>Rolling updates &amp; rollback</b> — swap the image with zero-downtime, roll back if a build is bad.</li>
            <li><b>Declarative</b> — the whole deployment is one YAML you version, review, and re-apply identically anywhere.</li>
          </ul>
          <p className="muted"><b>When NOT to bother:</b> for a single GPU box, plain Docker (the Containers tab) is simpler and enough. Kubernetes pays off once you have a <b>fleet</b> of GPU hosts to schedule across.</p>
          <hr className="k8s-intro-sep" />
          <p><b>This panel:</b> <b>Deploy</b> renders a ready manifest from any preset (no cluster needed to generate it) — <code>kubectl apply</code> and you're running. <b>Nodes/Pods/Events</b> read a connected cluster via your kubeconfig (your RBAC) to show GPU capacity/free per node and why a pod is pending (e.g. <code>Insufficient nvidia.com/gpu</code>).</p>
          <p className="muted"><b>GPU prerequisite:</b> the cluster needs the NVIDIA device plugin / gpu-operator so nodes advertise <code>nvidia.com/gpu</code> — otherwise GPU pods stay Pending (the Events tab shows exactly that).</p>
        </div>
      )}
    </div>
  );
}

function K8sDeploy() {
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
        <label>Preset
          <select value={preset} onChange={(e) => setPreset(e.target.value)} aria-label="Preset to deploy">
            {presets.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
          </select>
        </label>
        <button className="primary-button" disabled={!preset || busy} onClick={() => void generate()}>
          {busy ? <Loader2 size={14} className="spin" /> : <Rocket size={14} />} Generate manifest
        </button>
      </div>
      {err && <div className="containers-err"><AlertTriangle size={13} /> {err}</div>}
      {plan && (
        <>
          <div className="k8s-deploy-cmds">
            <span className="muted">Apply &amp; watch:</span>
            <code>kubectl apply -f {plan.artifact.filename}</code>
            <code>kubectl rollout status deploy/sndr-{plan.preset_id.slice(0, 30)}</code>
            <button className="ghost-button" onClick={() => { navigator.clipboard?.writeText(yaml); setCopied(true); window.setTimeout(() => setCopied(false), 1500); }}>
              {copied ? <ShieldCheck size={13} /> : <Copy size={13} />} {copied ? "copied" : "copy YAML"}
            </button>
          </div>
          <pre className="k8s-yaml"><code>{yaml}</code></pre>
        </>
      )}
    </div>
  );
}

function Kpi({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return <div className={`k8s-kpi${accent ? " accent" : ""}`}><span className="k8s-kpi-label">{label}</span><b>{value}</b></div>;
}

function NodeRow({ n }: { n: K8sNode }) {
  const gpuAlloc = n.gpu_allocatable ?? 0;
  const gpuFree = n.gpu_free ?? gpuAlloc;
  const hasGpu = gpuAlloc > 0;
  const st = n.ready ? (n.schedulable ? "online" : "partial") : "offline";
  return (
    <tr className={`crow ${st}`}>
      <td className="crow-name"><span className={`container-dot ${st}`} />{n.name}</td>
      <td>
        <span className={`container-badge ${st}`}>{n.ready ? (n.schedulable ? "Ready" : "Ready (cordoned)") : "NotReady"}</span>
        {n.pressures.map((p) => <span key={p} className="k8s-pressure" title={`${p} = True`}><ShieldAlert size={10} /> {p.replace("Pressure", "")}</span>)}
      </td>
      <td className="muted">{n.roles.join(", ") || "—"}</td>
      <td className="muted">{n.kubelet_version ?? "—"}</td>
      <td>
        {hasGpu
          ? <span className={`k8s-gpu ${gpuFree === 0 ? "full" : "free"}`}><Cpu size={11} /> {gpuFree} / {gpuAlloc}{n.gpu_requested ? ` · ${n.gpu_requested} used` : ""}</span>
          : <span className="muted">—</span>}
      </td>
      <td className="muted k8s-gpulabels" title={Object.entries(n.gpu_labels).map(([k, v]) => `${k}=${v}`).join("\n")}>
        {n.gpu_labels["nvidia.com/gpu.product"] ?? (Object.keys(n.gpu_labels).length ? `${Object.keys(n.gpu_labels).length} labels` : "—")}
      </td>
      <td className="muted">{n.taints.length ? n.taints.map((t) => t.key).join(", ") : <ShieldCheck size={12} />}</td>
    </tr>
  );
}

function PodRow({ p }: { p: K8sPod }) {
  const st = p.phase === "Running" && p.ready_ok ? "online" : p.phase === "Pending" || p.phase === "Unknown" ? "partial" : p.ready_ok ? "online" : "offline";
  return (
    <tr className={`crow ${st}`}>
      <td className="crow-name"><span className={`container-dot ${st}`} />{p.name}</td>
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

function EventRow({ e }: { e: K8sEvent }) {
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
