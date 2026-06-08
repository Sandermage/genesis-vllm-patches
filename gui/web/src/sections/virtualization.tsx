// SPDX-License-Identifier: Apache-2.0
// Virtualization — one pane over compute: Proxmox VE hosts & guests, KubeVirt
// VMs, and Kubernetes nodes, each linked back to the SNDR preset it runs.
// Read-only + graceful (every source degrades to a "connect / not installed"
// card). Bilingual via i18n. Mirrors the Kubernetes panel's posture.
import { useCallback, useEffect, useState, type ReactNode } from "react";
import {
  Server, Cpu, Boxes, Monitor, Layers, RefreshCw,
  Loader2, Package, ChevronDown, Plug, Info,
} from "lucide-react";
import {
  api, type ProxmoxStatus, type ProxmoxNode, type ProxmoxGuest,
  type KubeVirtVM, type KubeVirtResult, type K8sNode,
} from "../api";
import { useLang, t, type Lang } from "../i18n";
import { onKeyActivate } from "../dialog";

const GiB = 1024 ** 3;
function fmtBytes(n?: number | null): string {
  if (n == null) return "—";
  if (n >= GiB) return `${(n / GiB).toFixed(n >= 10 * GiB ? 0 : 1)} GiB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(0)} MiB`;
  return `${n} B`;
}
function fmtUptime(s?: number | null): string {
  if (!s || s <= 0) return "—";
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
  return d > 0 ? `${d}d ${h}h` : h > 0 ? `${h}h ${m}m` : `${m}m`;
}
const tone = (p?: number | null) => (p == null ? "ok" : p >= 90 ? "hot" : p >= 70 ? "warn" : "ok");

function Meter({ label, pct, text }: { label: string; pct: number | null; text: string }) {
  return (
    <div className="virt-meter">
      <div className="virt-meter-top"><span className="virt-meter-l">{label}</span><span className="virt-meter-v">{text}</span></div>
      <div className="virt-meter-track"><span className={`virt-meter-fill ${tone(pct)}`} style={{ width: `${Math.min(100, Math.max(0, pct ?? 0))}%` }} /></div>
    </div>
  );
}

export function VirtualizationPanel() {
  const [lang] = useLang();
  const [tab, setTab] = useState<"proxmox" | "kubevirt" | "k8s">("proxmox");
  const [status, setStatus] = useState<ProxmoxStatus | null>(null);
  const [nodes, setNodes] = useState<ProxmoxNode[]>([]);
  const [guests, setGuests] = useState<ProxmoxGuest[]>([]);
  const [kv, setKv] = useState<KubeVirtResult | null>(null);
  const [k8sNodes, setK8sNodes] = useState<K8sNode[] | null>(null);
  const [loading, setLoading] = useState(true);

  const reload = useCallback(async () => {
    setLoading(true);
    const [s, n, g, v, kn] = await Promise.allSettled([
      api.proxmoxStatus(), api.proxmoxNodes(), api.proxmoxGuests(), api.k8sKubevirt(), api.k8sNodes(),
    ]);
    if (s.status === "fulfilled") setStatus(s.value);
    if (n.status === "fulfilled") setNodes(n.value.nodes ?? []);
    if (g.status === "fulfilled") setGuests(g.value.guests ?? []);
    if (v.status === "fulfilled") setKv(v.value);
    if (kn.status === "fulfilled") setK8sNodes(kn.value.available ? kn.value.nodes : null);
    setLoading(false);
  }, []);
  useEffect(() => { void reload(); }, [reload]);

  const sndrManaged = (status?.sndr_managed ?? 0) + (kv?.vms ?? []).filter((v) => v.sndr_preset).length;

  return (
    <div className="virt">
      <VirtIntro lang={lang} />

      <div className="virt-summary">
        <SummaryCard icon={<Server size={15} />} value={status?.available ? `${status.nodes_online ?? 0}/${status.node_count ?? 0}` : "—"} label={t(lang, "virt.hosts")} sub="Proxmox" tone={status?.available ? "ok" : "muted"} />
        <SummaryCard icon={<Monitor size={15} />} value={status?.available ? `${status.vm_running ?? 0}/${status.vm_count ?? 0}` : "—"} label={t(lang, "virt.vms")} sub="Proxmox" />
        <SummaryCard icon={<Boxes size={15} />} value={status?.available ? `${status.lxc_running ?? 0}/${status.lxc_count ?? 0}` : "—"} label={t(lang, "virt.lxc")} sub="Proxmox" />
        <SummaryCard icon={<Layers size={15} />} value={kv?.installed ? String(kv.vms.length) : "—"} label={t(lang, "virt.kubevirt")} sub="k8s" />
        <SummaryCard icon={<Cpu size={15} />} value={k8sNodes ? String(k8sNodes.length) : "—"} label={t(lang, "virt.k8sNodes")} sub="k8s" />
        <SummaryCard icon={<Package size={15} />} value={String(sndrManaged)} label={t(lang, "virt.sndrManaged")} sub="" tone={sndrManaged > 0 ? "accent" : "muted"} />
      </div>

      <div className="virt-bar">
        <div className="k8s-tabs">
          <button className={tab === "proxmox" ? "active" : ""} onClick={() => setTab("proxmox")}><Server size={14} /> {t(lang, "virt.proxmox")}</button>
          <button className={tab === "kubevirt" ? "active" : ""} onClick={() => setTab("kubevirt")}><Layers size={14} /> {t(lang, "virt.kubevirt")}</button>
          <button className={tab === "k8s" ? "active" : ""} onClick={() => setTab("k8s")}><Cpu size={14} /> {t(lang, "virt.k8sNodes")}</button>
        </div>
        <button className="ghost-button" onClick={() => void reload()} disabled={loading}>
          {loading ? <Loader2 size={14} className="spin" /> : <RefreshCw size={14} />} {t(lang, "common.refresh")}
        </button>
      </div>

      {tab === "proxmox" && <ProxmoxView lang={lang} status={status} nodes={nodes} guests={guests} loading={loading} />}
      {tab === "kubevirt" && <KubeVirtView lang={lang} kv={kv} />}
      {tab === "k8s" && <K8sNodesView lang={lang} nodes={k8sNodes} />}
    </div>
  );
}

function SummaryCard({ icon, value, label, sub, tone: cardTone = "n" }: { icon: ReactNode; value: string; label: string; sub: string; tone?: string }) {
  return (
    <div className={`virt-sum ${cardTone}`}>
      <div className="virt-sum-h">{icon}<span>{label}</span></div>
      <div className="virt-sum-v">{value}{sub ? <em>{sub}</em> : null}</div>
    </div>
  );
}

function VirtIntro({ lang }: { lang: Lang }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="k8s-intro">
      <div className="k8s-intro-head" role="button" tabIndex={0} onClick={() => setOpen((v) => !v)} onKeyDown={onKeyActivate(() => setOpen((v) => !v))}>
        <Info size={14} />
        <span><strong>{t(lang, "virt.title")}</strong> — {t(lang, "virt.subtitle")}</span>
        <ChevronDown size={15} className={open ? "rot" : ""} />
      </div>
      {open && (
        <div className="k8s-intro-body">
          <p><b>{t(lang, "virt.value")}.</b> {t(lang, "virt.valueBody")}</p>
        </div>
      )}
    </div>
  );
}

function ConnectCard({ icon, title, body, cmds }: { icon: ReactNode; title: string; body: string; cmds?: string[] }) {
  return (
    <div className="virt-connect">
      <div className="empty-state-icon">{icon}</div>
      <strong>{title}</strong>
      <p className="empty-state-msg">{body}</p>
      {cmds ? <pre className="virt-connect-cmd"><code>{cmds.join("\n")}</code></pre> : null}
    </div>
  );
}

function ProxmoxView({ lang, status, nodes, guests, loading }: { lang: Lang; status: ProxmoxStatus | null; nodes: ProxmoxNode[]; guests: ProxmoxGuest[]; loading: boolean }) {
  if (loading && !status) return <SkeletonBlock />;
  if (!status?.available) {
    return (
      <ConnectCard icon={<Plug size={22} />} title={t(lang, "virt.proxmoxNotConfigured")} body={status?.error || t(lang, "virt.proxmoxConnectHelp")}
        cmds={["export SNDR_PROXMOX_HOST=https://pve.local:8006", "export SNDR_PROXMOX_TOKEN_ID='root@pam!sndr'", "export SNDR_PROXMOX_TOKEN_SECRET=<secret>"]} />
    );
  }
  return (
    <div className="virt-pane">
      <div className="virt-nodes">
        {nodes.map((n) => (
          <div key={n.name} className={`virt-node ${n.online ? "online" : "offline"}`}>
            <div className="virt-node-h">
              <span className={`container-dot ${n.online ? "online" : "offline"}`} />
              <strong>{n.name}</strong>
              <span className={`container-badge ${n.online ? "online" : "offline"}`}>{n.status}</span>
              {n.uptime ? <span className="virt-node-up">{fmtUptime(n.uptime)}</span> : null}
            </div>
            <div className="virt-node-meters">
              <Meter label={`${t(lang, "common.cpu")} · ${n.cpu_cores ?? "?"}c`} pct={n.cpu_pct} text={n.cpu_pct == null ? "—" : `${n.cpu_pct.toFixed(0)}%`} />
              <Meter label={t(lang, "common.memory")} pct={n.mem_pct} text={`${fmtBytes(n.mem_used)} / ${fmtBytes(n.mem_total)}`} />
              <Meter label={t(lang, "common.disk")} pct={n.disk_pct} text={`${fmtBytes(n.disk_used)} / ${fmtBytes(n.disk_total)}`} />
            </div>
          </div>
        ))}
      </div>

      {guests.length === 0 ? (
        <div className="empty-state"><div className="empty-state-icon"><Monitor size={20} /></div><p className="empty-state-msg">{t(lang, "virt.noGuests")}</p></div>
      ) : (
        <table className="containers-table virt-guests">
          <thead><tr>
            <th>{t(lang, "virt.guests")}</th><th></th><th>{t(lang, "common.cpu")}</th><th>{t(lang, "common.memory")}</th>
            <th>{t(lang, "virt.node")}</th><th>{t(lang, "common.uptime")}</th><th>SNDR</th>
          </tr></thead>
          <tbody>
            {guests.map((g) => (
              <tr key={`${g.kind}-${g.vmid}`} className={`crow ${g.running ? "online" : "offline"}${g.sndr_preset ? " virt-managed" : ""}`}>
                <td className="crow-name">
                  <span className={`container-dot ${g.running ? "online" : "offline"}`} />
                  <span className={`virt-kind ${g.kind}`}>{g.kind === "vm" ? "VM" : "LXC"}</span>
                  {g.name} <span className="muted">#{g.vmid}</span>
                </td>
                <td><span className={`container-badge ${g.running ? "online" : "offline"}`}>{g.status}</span></td>
                <td className="muted">{g.cpu_pct == null ? "—" : `${g.cpu_pct.toFixed(0)}%`}<span className="virt-dim"> /{g.cpu_cores ?? "?"}c</span></td>
                <td className="muted">{g.mem_pct == null ? "—" : `${g.mem_pct.toFixed(0)}%`}<span className="virt-dim"> {fmtBytes(g.mem_total)}</span></td>
                <td className="muted">{g.node ?? "—"}</td>
                <td className="muted">{fmtUptime(g.uptime)}</td>
                <td>{g.sndr_preset ? <span className="k8s-sndr-chip preset"><Package size={9} /> {g.sndr_preset}</span> : <span className="muted">—</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function KubeVirtView({ lang, kv }: { lang: Lang; kv: KubeVirtResult | null }) {
  if (!kv) return <SkeletonBlock />;
  if (!kv.available) return <ConnectCard icon={<Plug size={22} />} title={t(lang, "virt.k8sNotConnected")} body={kv.error || t(lang, "virt.k8sNotConnected")} />;
  if (kv.installed === false) return <ConnectCard icon={<Layers size={22} />} title={t(lang, "virt.kubevirtNotInstalled")} body={t(lang, "virt.kubevirtHelp")} />;
  if (kv.vms.length === 0) return <div className="empty-state"><div className="empty-state-icon"><Layers size={20} /></div><p className="empty-state-msg">{t(lang, "virt.kubevirtNotInstalled")}</p></div>;
  return (
    <table className="containers-table virt-guests">
      <thead><tr><th>VM</th><th></th><th>{t(lang, "virt.node")}</th><th>{t(lang, "common.cpu")}</th><th>{t(lang, "common.memory")}</th><th>GPU</th><th>IP</th><th>SNDR</th></tr></thead>
      <tbody>
        {kv.vms.map((v: KubeVirtVM) => (
          <tr key={`${v.namespace}/${v.name}`} className={`crow ${v.running ? "online" : "offline"}${v.sndr_preset ? " virt-managed" : ""}`}>
            <td className="crow-name"><span className={`container-dot ${v.running ? "online" : "offline"}`} /><span className="virt-kind vm">VM</span>{v.name}<span className="muted"> · {v.namespace}</span></td>
            <td><span className={`container-badge ${v.running ? "online" : "offline"}`}>{v.phase}</span></td>
            <td className="muted">{v.node ?? "—"}</td>
            <td className="muted">{v.cpu_cores ?? "—"}c</td>
            <td className="muted">{v.memory ?? "—"}</td>
            <td>{v.gpu_count > 0 ? <span className="k8s-gpu free"><Cpu size={11} /> {v.gpu_count}</span> : <span className="muted">—</span>}</td>
            <td className="muted">{v.ip ?? "—"}</td>
            <td>{v.sndr_preset ? <span className="k8s-sndr-chip preset"><Package size={9} /> {v.sndr_preset}</span> : <span className="muted">—</span>}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function K8sNodesView({ lang, nodes }: { lang: Lang; nodes: K8sNode[] | null }) {
  if (nodes == null) return <ConnectCard icon={<Plug size={22} />} title={t(lang, "virt.k8sNotConnected")} body={t(lang, "virt.k8sNotConnected")} />;
  return (
    <div className="virt-nodes">
      {nodes.map((n) => {
        const gpuFree = n.gpu_free, gpuAlloc = n.gpu_allocatable ?? 0;
        return (
          <div key={n.name} className={`virt-node ${n.ready ? "online" : "offline"}`}>
            <div className="virt-node-h">
              <span className={`container-dot ${n.ready ? "online" : "offline"}`} />
              <strong>{n.name}</strong>
              <span className={`container-badge ${n.ready ? "online" : "offline"}`}>{n.ready ? "Ready" : "NotReady"}</span>
              {gpuAlloc > 0 ? <span className={`k8s-gpu ${gpuFree === 0 ? "full" : "free"}`}><Cpu size={11} /> {gpuFree ?? "?"} / {gpuAlloc} GPU</span> : null}
            </div>
            <div className="virt-node-facts muted">
              <span>{n.kubelet_version ?? ""}</span>
              <span>{n.os_image ?? ""}</span>
              {n.roles.length ? <span>{n.roles.join(", ")}</span> : null}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function SkeletonBlock() {
  return <div className="virt-pane"><div className="virt-node skel" /><div className="virt-node skel" /></div>;
}
