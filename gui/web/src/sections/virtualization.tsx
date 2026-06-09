// SPDX-License-Identifier: Apache-2.0
// Virtualization — the unified compute control plane. Two providers under one
// roof: Proxmox VE (hosts + VMs/LXC) and Kubernetes (nodes + pods + events +
// KubeVirt VMs + deploy), each guest/pod linked to the SNDR preset it runs.
// Read-only + graceful per source. Bilingual (EN/RU).
import { useCallback, useEffect, useState, type ReactNode } from "react";
import { cachePeek, cacheSet } from "../lib/swr-cache";
import {
  Server, Cpu, Boxes, Monitor, Layers, RefreshCw, Loader2, Package, ChevronDown,
  Plug, Info, ShieldCheck, ShieldAlert, Activity, AlertTriangle, Rocket, Copy,
  HardDrive, Network,
} from "lucide-react";
import {
  api, type ProxmoxStatus, type ProxmoxNode, type ProxmoxGuest, type ProxmoxGuestDetail, type ProxmoxNodeDetail,
  type KubeVirtResult, type K8sStatus, type K8sNode, type K8sPod, type K8sEvent,
  type DeploymentPlan,
} from "../api";
import { useLang, t, tr, type Lang } from "../i18n";
import { onKeyActivate } from "../dialog";
import { K8sDeploy, PodRow, EventRow } from "./kubernetes";

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
const meterTone = (p?: number | null) => (p == null ? "ok" : p >= 90 ? "hot" : p >= 70 ? "warn" : "ok");

function Meter({ label, pct, text }: { label: string; pct: number | null; text: string }) {
  return (
    <div className="virt-meter">
      <div className="virt-meter-top"><span className="virt-meter-l">{label}</span><span className="virt-meter-v">{text}</span></div>
      <div className="virt-meter-track"><span className={`virt-meter-fill ${meterTone(pct)}`} style={{ width: `${Math.min(100, Math.max(0, pct ?? 0))}%` }} /></div>
    </div>
  );
}

type Provider = "proxmox" | "kubernetes";
type K8sTab = "nodes" | "pods" | "events" | "kubevirt" | "deploy";

// One snapshot of every virtualization surface, cached so re-opening the section
// paints instantly (stale-while-revalidate).
type VirtSnapshot = {
  pxStatus: ProxmoxStatus | null;
  pxNodes: ProxmoxNode[];
  pxGuests: ProxmoxGuest[];
  k8sStatus: K8sStatus | null;
  k8sNodes: K8sNode[] | null;
  k8sPods: K8sPod[] | null;
  k8sEvents: K8sEvent[] | null;
  kv: KubeVirtResult | null;
};
const VIRT_CACHE_KEY = "virt:snapshot";
const EMPTY_VIRT: VirtSnapshot = {
  pxStatus: null, pxNodes: [], pxGuests: [], k8sStatus: null,
  k8sNodes: null, k8sPods: null, k8sEvents: null, kv: null,
};

export function VirtualizationPanel() {
  const [lang] = useLang();
  const [provider, setProvider] = useState<Provider>("kubernetes");
  const [k8sTab, setK8sTab] = useState<K8sTab>("nodes");

  // Proxmox
  const [pxStatus, setPxStatus] = useState<ProxmoxStatus | null>(null);
  const [pxNodes, setPxNodes] = useState<ProxmoxNode[]>([]);
  const [pxGuests, setPxGuests] = useState<ProxmoxGuest[]>([]);
  // Kubernetes
  const [k8sStatus, setK8sStatus] = useState<K8sStatus | null>(null);
  const [k8sNodes, setK8sNodes] = useState<K8sNode[] | null>(null);
  const [k8sPods, setK8sPods] = useState<K8sPod[] | null>(null);
  const [k8sEvents, setK8sEvents] = useState<K8sEvent[] | null>(null);
  const [kv, setKv] = useState<KubeVirtResult | null>(null);
  const [loading, setLoading] = useState(true);

  const apply = useCallback((s: VirtSnapshot) => {
    setPxStatus(s.pxStatus); setPxNodes(s.pxNodes); setPxGuests(s.pxGuests);
    setK8sStatus(s.k8sStatus); setK8sNodes(s.k8sNodes); setK8sPods(s.k8sPods);
    setK8sEvents(s.k8sEvents); setKv(s.kv);
  }, []);

  const reload = useCallback(async () => {
    setLoading(true);
    // Seed from the last snapshot so a transient failure of one sub-request keeps
    // its previous value instead of blanking it.
    const snap: VirtSnapshot = { ...(cachePeek<VirtSnapshot>(VIRT_CACHE_KEY) ?? EMPTY_VIRT) };
    const [ps, pn, pg, ks, kev] = await Promise.allSettled([
      api.proxmoxStatus(), api.proxmoxNodes(), api.proxmoxGuests(), api.k8sStatus(), api.k8sKubevirt(),
    ]);
    if (ps.status === "fulfilled") snap.pxStatus = ps.value;
    if (pn.status === "fulfilled") snap.pxNodes = pn.value.nodes ?? [];
    if (pg.status === "fulfilled") snap.pxGuests = pg.value.guests ?? [];
    if (kev.status === "fulfilled") snap.kv = kev.value;
    if (ks.status === "fulfilled") {
      snap.k8sStatus = ks.value;
      if (ks.value.available) {
        const [n, p, e] = await Promise.all([api.k8sNodes(), api.k8sPods(), api.k8sEvents()]);
        snap.k8sNodes = n.nodes; snap.k8sPods = p.pods; snap.k8sEvents = e.events;
      } else { snap.k8sNodes = null; snap.k8sPods = null; snap.k8sEvents = null; }
    }
    cacheSet(VIRT_CACHE_KEY, snap);
    apply(snap);
    setLoading(false);
  }, [apply]);

  // Stale-while-revalidate: hydrate the last snapshot instantly so re-opening the
  // section isn't a skeleton over slow Proxmox/k8s calls; always refresh under it.
  useEffect(() => {
    const stale = cachePeek<VirtSnapshot>(VIRT_CACHE_KEY);
    if (stale) apply(stale);
    void reload();
  }, [reload, apply]);

  const sndrManaged = (pxStatus?.sndr_managed ?? 0)
    + (kv?.vms ?? []).filter((v) => v.sndr_preset).length
    + (k8sPods ?? []).filter((p) => p.sndr_managed).length;
  const warnEvents = (k8sEvents ?? []).filter((e) => e.type === "Warning").length;

  return (
    <div className="virt">
      <VirtIntro lang={lang} />

      <div className="virt-summary">
        <SummaryCard icon={<Server size={15} />} value={pxStatus?.available ? `${pxStatus.nodes_online ?? 0}/${pxStatus.node_count ?? 0}` : "—"} label={t(lang, "virt.hosts")} sub="Proxmox" tone={pxStatus?.available ? "ok" : "muted"} />
        <SummaryCard icon={<Monitor size={15} />} value={pxStatus?.available ? `${pxStatus.vm_running ?? 0}/${pxStatus.vm_count ?? 0}` : "—"} label={t(lang, "virt.vms")} sub="Proxmox" />
        <SummaryCard icon={<Boxes size={15} />} value={pxStatus?.available ? `${pxStatus.lxc_running ?? 0}/${pxStatus.lxc_count ?? 0}` : "—"} label={t(lang, "virt.lxc")} sub="Proxmox" />
        <SummaryCard icon={<Cpu size={15} />} value={k8sStatus?.available ? `${k8sStatus.nodes_ready ?? 0}/${k8sStatus.node_count ?? 0}` : "—"} label={t(lang, "virt.k8sNodes")} sub="k8s" tone={k8sStatus?.available ? "ok" : "muted"} />
        <SummaryCard icon={<Activity size={15} />} value={k8sPods ? String(k8sPods.length) : "—"} label={t(lang, "virt.pods")} sub="k8s" />
        <SummaryCard icon={<Layers size={15} />} value={kv?.installed ? String(kv.vms.length) : "—"} label={t(lang, "virt.kubevirt")} sub="k8s" />
        <SummaryCard icon={<Package size={15} />} value={String(sndrManaged)} label={t(lang, "virt.sndrManaged")} sub="" tone={sndrManaged > 0 ? "accent" : "muted"} />
      </div>

      <div className="virt-bar">
        <div className="virt-providers">
          <button className={provider === "kubernetes" ? "active" : ""} onClick={() => setProvider("kubernetes")}><Layers size={15} /> Kubernetes</button>
          <button className={provider === "proxmox" ? "active" : ""} onClick={() => setProvider("proxmox")}><Server size={15} /> {t(lang, "virt.proxmox")}</button>
        </div>
        <button className="ghost-button" onClick={() => void reload()} disabled={loading}>
          {loading ? <Loader2 size={14} className="spin" /> : <RefreshCw size={14} />} {t(lang, "common.refresh")}
        </button>
      </div>

      {provider === "proxmox" && <ProxmoxView lang={lang} status={pxStatus} nodes={pxNodes} guests={pxGuests} loading={loading} />}

      {provider === "kubernetes" && (
        <div className="virt-pane">
          <Explain text={t(lang, "virt.k8sAbout")} />
          <div className="k8s-tabs virt-subtabs">
            <button className={k8sTab === "nodes" ? "active" : ""} onClick={() => setK8sTab("nodes")}>{t(lang, "virt.nodes")}{k8sNodes ? ` (${k8sNodes.length})` : ""}</button>
            <button className={k8sTab === "pods" ? "active" : ""} onClick={() => setK8sTab("pods")}>{t(lang, "virt.pods")}{k8sPods ? ` (${k8sPods.length})` : ""}</button>
            <button className={k8sTab === "events" ? "active" : ""} onClick={() => setK8sTab("events")}>{t(lang, "virt.events")}{warnEvents ? ` (${warnEvents}⚠)` : ""}</button>
            <button className={k8sTab === "kubevirt" ? "active" : ""} onClick={() => setK8sTab("kubevirt")}>{t(lang, "virt.kubevirt")}</button>
            <button className={k8sTab === "deploy" ? "active" : ""} onClick={() => setK8sTab("deploy")}><Rocket size={13} /> {t(lang, "virt.deploy")}</button>
          </div>
          {k8sStatus?.available && (k8sTab === "nodes" || k8sTab === "pods" || k8sTab === "events") && (
            <Explain text={t(lang, k8sTab === "nodes" ? "virt.tabNodesHelp" : k8sTab === "pods" ? "virt.tabPodsHelp" : "virt.tabEventsHelp")} />
          )}

          {k8sTab === "deploy" ? <K8sDeploy />
            : !k8sStatus ? <SkeletonBlock />
            : !k8sStatus.available && k8sTab !== "kubevirt" ? <ConnectCard icon={<Plug size={22} />} title={t(lang, "virt.k8sNotConnected")} body={k8sStatus.error || ""} cmds={["pip install 'vllm-sndr-core[k8s]'", "export KUBECONFIG=/etc/rancher/k3s/k3s.yaml"]} />
            : k8sTab === "nodes" ? <K8sNodesView lang={lang} status={k8sStatus} nodes={k8sNodes} />
            : k8sTab === "pods" ? <PodsTable lang={lang} pods={k8sPods} />
            : k8sTab === "events" ? <EventsTable events={k8sEvents} />
            : <KubeVirtView lang={lang} kv={kv} />}
        </div>
      )}
    </div>
  );
}

function SummaryCard({ icon, value, label, sub, tone = "n" }: { icon: ReactNode; value: string; label: string; sub: string; tone?: string }) {
  return (
    <div className={`virt-sum ${tone}`}>
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
      {open && <div className="k8s-intro-body"><p><b>{t(lang, "virt.value")}.</b> {t(lang, "virt.valueBody")}</p></div>}
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

// A short, inline explanation line — what a surface is / how to act on it.
function Explain({ text }: { text: string }) {
  return <div className="virt-explain"><Info size={13} /><span>{text}</span></div>;
}

// ── Proxmox ──────────────────────────────────────────────────────────────────
type PxTab = "hosts" | "guests" | "create";
function ProxmoxView({ lang, status, nodes, guests, loading }: { lang: Lang; status: ProxmoxStatus | null; nodes: ProxmoxNode[]; guests: ProxmoxGuest[]; loading: boolean }) {
  const [pxTab, setPxTab] = useState<PxTab>("hosts");
  const connected = !!status?.available;
  // Create only needs a preset (it renders a provision script), so it stays
  // available even when read-only Proxmox monitoring isn't configured.
  const monitorBody = (() => {
    if (pxTab === "create") return <ProxmoxDeploy lang={lang} />;
    if (loading && !status) return <SkeletonBlock />;
    if (!connected) {
      return <ConnectCard icon={<Plug size={22} />} title={t(lang, "virt.proxmoxNotConfigured")} body={status?.error || t(lang, "virt.proxmoxConnectHelp")}
        cmds={["export SNDR_PROXMOX_HOST=https://pve.local:8006", "export SNDR_PROXMOX_TOKEN_ID='root@pam!sndr'", "export SNDR_PROXMOX_TOKEN_SECRET=<secret>"]} />;
    }
    if (pxTab === "hosts") return <><Explain text={t(lang, "virt.tabHostsHelp")} /><ProxmoxHosts lang={lang} nodes={nodes} /></>;
    return <><Explain text={t(lang, "virt.tabGuestsHelp")} /><ProxmoxGuests lang={lang} guests={guests} /></>;
  })();
  return (
    <div className="virt-pane">
      <Explain text={t(lang, "virt.proxmoxAbout")} />
      <div className="k8s-tabs virt-subtabs">
        <button className={pxTab === "hosts" ? "active" : ""} onClick={() => setPxTab("hosts")}>{t(lang, "virt.hosts")}{connected ? ` (${nodes.length})` : ""}</button>
        <button className={pxTab === "guests" ? "active" : ""} onClick={() => setPxTab("guests")}>{t(lang, "virt.guests")}{connected ? ` (${guests.length})` : ""}</button>
        <button className={pxTab === "create" ? "active" : ""} onClick={() => setPxTab("create")}><Rocket size={13} /> {t(lang, "virt.create")}</button>
      </div>
      {monitorBody}
    </div>
  );
}

function ProxmoxHosts({ lang, nodes }: { lang: Lang; nodes: ProxmoxNode[] }) {
  return (
    <div className="virt-pane">
      <div className="virt-nodes">
        {nodes.map((n) => <NodeCard key={n.name} n={n} lang={lang} />)}
      </div>
    </div>
  );
}

// A Proxmox host node: live CPU/RAM/disk meters plus, lazily, the rich facts
// (CPU model, kernel, PVE version, load average, swap and the GPUs present).
function NodeCard({ n, lang }: { n: ProxmoxNode; lang: Lang }) {
  const [d, setD] = useState<ProxmoxNodeDetail | null>(null);
  useEffect(() => {
    let alive = true;
    if (n.online && n.name) api.proxmoxNodeDetail(n.name).then((r) => alive && setD(r)).catch(() => {});
    return () => { alive = false; };
  }, [n.name, n.online]);
  const pve = d?.pve_version ? d.pve_version.replace("pve-manager/", "").split("/")[0] : null;
  return (
    <div className={`virt-node ${n.online ? "online" : "offline"}`}>
      <div className="virt-node-h">
        <span className={`container-dot ${n.online ? "online" : "offline"}`} /><strong>{n.name}</strong>
        <span className={`container-badge ${n.online ? "online" : "offline"}`}>{n.status}</span>
        {pve ? <span className="px-node-pve">PVE {pve}</span> : null}
        {n.uptime ? <span className="virt-node-up">{fmtUptime(n.uptime)}</span> : null}
      </div>
      <div className="virt-node-meters">
        <Meter label={`${t(lang, "common.cpu")} · ${n.cpu_cores ?? d?.cpu_threads ?? "?"}t`} pct={n.cpu_pct} text={n.cpu_pct == null ? "—" : `${n.cpu_pct.toFixed(0)}%`} />
        <Meter label={t(lang, "common.memory")} pct={n.mem_pct} text={`${fmtBytes(n.mem_used)} / ${fmtBytes(n.mem_total)}`} />
        <Meter label={t(lang, "common.disk")} pct={n.disk_pct} text={`${fmtBytes(n.disk_used)} / ${fmtBytes(n.disk_total)}`} />
      </div>
      {d?.available ? (
        <div className="px-node-facts">
          {d.cpu_model ? <span title="CPU"><Cpu size={11} /> {d.cpu_model.replace(/\s*\d+-Core Processor/i, "")}{d.cpu_cores ? ` · ${d.cpu_cores}c/${d.cpu_threads}t` : ""}</span> : null}
          {d.kernel ? <span title={tr("kernel")}><Layers size={11} /> {d.kernel}</span> : null}
          {d.loadavg.length === 3 ? <span title={tr("load 1 / 5 / 15 min")}><Activity size={11} /> {d.loadavg.join(" ")}</span> : null}
          {d.swap_total ? <span title={tr("swap")}>swap {fmtBytes(d.swap_used)} / {fmtBytes(d.swap_total)}</span> : null}
          {d.gpus.length > 0 ? <span title={tr("GPUs on host")}><Boxes size={11} /> {d.gpus.join(" · ")}</span> : null}
        </div>
      ) : null}
    </div>
  );
}

function ProxmoxGuests({ lang, guests }: { lang: Lang; guests: ProxmoxGuest[] }) {
  if (guests.length === 0) {
    return <div className="empty-state"><div className="empty-state-icon"><Monitor size={20} /></div><p className="empty-state-msg">{t(lang, "virt.noGuests")}</p></div>;
  }
  return <div className="px-guests">{guests.map((g) => <GuestCard key={`${g.kind}-${g.vmid}`} g={g} lang={lang} />)}</div>;
}

function osLabel(t?: string | null): string {
  if (!t) return "—";
  if (t.startsWith("l")) return "Linux";
  if (t.startsWith("w")) return "Windows";
  if (t === "solaris") return "Solaris";
  return t;
}
const fmtGiB = (mb?: number | null) => (mb == null ? "—" : `${(mb / 1024).toFixed(mb >= 10240 ? 0 : 1)} GiB`);
function GFact({ l, v }: { l: string; v: ReactNode }) {
  return <div className="px-fact"><span>{l}</span><strong>{v}</strong></div>;
}

// One Proxmox guest as an expandable card: live metrics + I/O always visible,
// rich config (GPU passthrough, OS, disks, networks, boot…) fetched on expand.
function GuestCard({ g, lang }: { g: ProxmoxGuest; lang: Lang }) {
  const [open, setOpen] = useState(false);
  const [detail, setDetail] = useState<ProxmoxGuestDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const toggle = () => {
    const next = !open;
    setOpen(next);
    if (next && !detail && g.node && g.vmid != null) {
      setLoading(true);
      api.proxmoxGuestDetail(g.node, g.kind, g.vmid).then(setDetail).catch(() => {}).finally(() => setLoading(false));
    }
  };
  return (
    <div className={`px-guest ${g.running ? "online" : "offline"}${g.sndr_preset ? " managed" : ""}`}>
      <div className="px-guest-head" role="button" tabIndex={0} aria-expanded={open} onClick={toggle} onKeyDown={onKeyActivate(toggle)}>
        <span className={`container-dot ${g.running ? "online" : "offline"}`} />
        <span className={`virt-kind ${g.kind}`}>{g.kind === "vm" ? "VM" : "LXC"}</span>
        <strong className="px-guest-name">{g.name}</strong><span className="muted">#{g.vmid}</span>
        <span className={`container-badge ${g.running ? "online" : "offline"}`}>{g.status}</span>
        {g.node ? <span className="px-guest-node muted">{g.node}</span> : null}
        {g.sndr_preset ? <span className="k8s-sndr-chip preset"><Package size={9} /> {g.sndr_preset}</span> : null}
        <ChevronDown size={15} className={`px-guest-caret ${open ? "rot" : ""}`} />
      </div>
      <div className="px-guest-metrics">
        <Meter label={`${t(lang, "common.cpu")} · ${g.cpu_cores ?? "?"}c`} pct={g.cpu_pct} text={g.cpu_pct == null ? "—" : `${g.cpu_pct.toFixed(0)}%`} />
        <Meter label={t(lang, "common.memory")} pct={g.mem_pct} text={`${fmtBytes(g.mem_used)} / ${fmtBytes(g.mem_total)}`} />
        <div className="px-guest-io">
          <span title={tr("network in / out")}><Network size={10} /> ↓{fmtBytes(g.net_in)} ↑{fmtBytes(g.net_out)}</span>
          <span title={tr("disk read / write")}><HardDrive size={10} /> r{fmtBytes(g.disk_read)} w{fmtBytes(g.disk_write)}</span>
          <span title="uptime">{fmtUptime(g.uptime)}</span>
        </div>
      </div>
      {open && (
        <div className="px-guest-detail">
          {loading && !detail ? <SkeletonBlock />
            : detail && detail.available ? <GuestDetailBody d={detail} />
            : <span className="muted">{tr("detail unavailable")}{detail?.error ? `: ${detail.error}` : ""}</span>}
        </div>
      )}
    </div>
  );
}

function GuestDetailBody({ d }: { d: ProxmoxGuestDetail }) {
  return (
    <>
      {(() => {
        const gpus = d.devices.filter((x) => x.kind === "gpu");
        const others = d.devices.filter((x) => x.kind !== "gpu");
        return (
          <>
            {gpus.length > 0 && (
              <div className="px-gpu-row"><Cpu size={13} /> <strong>GPU {tr("passthrough")}</strong>{gpus.map((g) => <code key={g.address} className="px-gpu-chip" title={g.address}>{g.name}</code>)}</div>
            )}
            {others.length > 0 && (
              <div className="px-detail-list"><span className="px-detail-l"><Plug size={11} /> {tr("Devices")}</span>{others.map((o) => <code key={o.address} title={o.address}>{o.kind}: {o.name}</code>)}</div>
            )}
          </>
        );
      })()}
      <div className="px-facts">
        <GFact l="CPU" v={`${d.cores ?? "?"} ${tr("cores")}${d.sockets && d.sockets > 1 ? ` × ${d.sockets}` : ""}${d.cpu_type ? ` · ${d.cpu_type}` : ""}`} />
        <GFact l={tr("Memory")} v={`${fmtGiB(d.memory_mb)}${d.swap_mb ? ` + ${fmtGiB(d.swap_mb)} swap` : ""}${d.balloon ? "" : d.kind === "vm" ? ` · ${tr("no balloon")}` : ""}`} />
        <GFact l="OS" v={osLabel(d.ostype)} />
        {d.kind === "vm" ? <GFact l={tr("Firmware")} v={`${d.bios ?? "—"}${d.machine ? ` · ${d.machine}` : ""}`} /> : null}
        {d.kind === "lxc" && d.unprivileged != null ? <GFact l={tr("Privilege")} v={d.unprivileged ? tr("unprivileged") : tr("privileged")} /> : null}
        <GFact l={tr("Boot")} v={`${d.onboot ? tr("on boot") : tr("manual")}${d.boot_order ? ` · ${d.boot_order.replace("order=", "")}` : ""}`} />
        {d.kind === "vm" ? <GFact l={tr("Guest agent")} v={d.agent_enabled ? (d.agent_ips.length ? d.agent_ips.join(", ") : tr("enabled")) : tr("off")} /> : null}
        {d.ha_managed != null ? <GFact l="HA" v={d.ha_managed ? tr("managed") : tr("no")} /> : null}
        {d.features ? <GFact l={tr("Features")} v={d.features} /> : null}
        {d.qmpstatus ? <GFact l={tr("State")} v={d.qmpstatus} /> : null}
      </div>
      {d.disks.length > 0 && (
        <div className="px-detail-list"><span className="px-detail-l"><HardDrive size={11} /> {tr("Disks")}</span>{d.disks.map((dk) => <code key={dk.id}>{dk.id}: {dk.storage ?? dk.volume}{dk.size ? ` · ${dk.size}` : ""}</code>)}</div>
      )}
      {d.networks.length > 0 && (
        <div className="px-detail-list"><span className="px-detail-l"><Network size={11} /> {tr("Net")}</span>{d.networks.map((n) => <code key={n.id}>{n.id}: {n.bridge ?? "?"}{n.model ? ` ${n.model}` : ""}{n.ip ? ` ${n.ip}` : ""}{n.mac ? ` ${n.mac}` : ""}</code>)}</div>
      )}
      {d.description ? <div className="px-detail-desc muted">{d.description}</div> : null}
      {d.tags.length > 0 ? <div className="px-detail-tags">{d.tags.map((tg) => <span key={tg} className="px-tag">{tg}</span>)}</div> : null}
    </>
  );
}

// Create a Proxmox guest: pick a preset + guest type, generate the provision
// script via the existing deploy machinery (pct create / qm create). Read-only —
// the operator runs the script on the node (or applies it over SSH from Install).
function ProxmoxDeploy({ lang }: { lang: Lang }) {
  const [presets, setPresets] = useState<{ id: string; label: string }[]>([]);
  const [preset, setPreset] = useState("");
  const [gtype, setGtype] = useState<"proxmox" | "proxmox_vm">("proxmox");
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
    try { setPlan(await api.deployPlan({ preset_id: preset, target: gtype })); }
    catch (e) { setErr(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }
  const script = plan?.artifact?.content ?? "";
  return (
    <div className="px-create">
      <Explain text={t(lang, "virt.pxCreateBody")} />
      <div className="px-create-form">
        <label className="px-field"><span>{t(lang, "virt.preset")}</span>
          <select value={preset} onChange={(e) => { setPreset(e.target.value); setPlan(null); }} aria-label={tr("Preset to provision")}>
            {presets.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
          </select>
        </label>
        <div className="px-gtype">
          <span className="px-field-l">{t(lang, "virt.guestType")}</span>
          <div className="px-gtype-opts">
            <button type="button" className={`px-gtype-opt ${gtype === "proxmox" ? "active" : ""}`} onClick={() => { setGtype("proxmox"); setPlan(null); }}>
              <span className="px-gtype-head"><Boxes size={15} /> {t(lang, "virt.pxLxc")}</span>
              <small>{t(lang, "virt.pxLxcHelp")}</small>
            </button>
            <button type="button" className={`px-gtype-opt ${gtype === "proxmox_vm" ? "active" : ""}`} onClick={() => { setGtype("proxmox_vm"); setPlan(null); }}>
              <span className="px-gtype-head"><Monitor size={15} /> {t(lang, "virt.pxVm")}</span>
              <small>{t(lang, "virt.pxVmHelp")}</small>
            </button>
          </div>
        </div>
        <button className="primary-button px-gen" disabled={!preset || busy} onClick={() => void generate()}>
          {busy ? <Loader2 size={14} className="spin" /> : <Rocket size={14} />} {t(lang, "virt.generate")}
        </button>
      </div>
      {err && <div className="containers-err"><AlertTriangle size={13} /> {err}</div>}
      {plan?.artifact && (
        <div className="px-plan">
          <div className="px-plan-head">
            <strong>{plan.artifact.filename}</strong>
            <span className="install-dry">dry-run · {tr("nothing executed here")}</span>
            <div className="px-plan-acts">
              <button className="ghost-button" onClick={() => { void navigator.clipboard?.writeText(script); setCopied(true); window.setTimeout(() => setCopied(false), 1500); }}>
                {copied ? <ShieldCheck size={13} /> : <Copy size={13} />} {copied ? t(lang, "virt.copied") : t(lang, "virt.copyScript")}
              </button>
              <button className="ghost-button" onClick={() => { window.location.hash = "setup"; }} title={t(lang, "virt.pxApplySsh")}><ShieldAlert size={13} /> {t(lang, "virt.pxApplySsh")}</button>
            </div>
          </div>
          {plan.commands && plan.commands.length > 0 && (
            <div className="px-cmds"><span className="muted">{t(lang, "virt.pxRunOnNode")}:</span>{plan.commands.map((c, i) => <code key={i}>{c}</code>)}</div>
          )}
          <pre className="k8s-yaml"><code>{script}</code></pre>
        </div>
      )}
    </div>
  );
}

// ── Kubernetes ───────────────────────────────────────────────────────────────
function K8sNodesView({ lang, status, nodes }: { lang: Lang; status: K8sStatus; nodes: K8sNode[] | null }) {
  if (nodes == null) return <SkeletonBlock />;
  if (nodes.length === 0) return <div className="empty-state"><div className="empty-state-icon"><Cpu size={20} /></div><p className="empty-state-msg">{tr("Cluster has no nodes")}</p></div>;
  return (
    <>
      <div className="virt-clusterline muted">
        <ShieldCheck size={12} /> {status.version ?? ""} · {status.namespace_count ?? 0} namespaces · {status.gpu_node_count ?? 0} GPU {t(lang, "common.nodes")}
      </div>
      <div className="virt-nodes">
        {nodes.map((n) => {
          const gpuAlloc = n.gpu_allocatable ?? 0, gpuFree = n.gpu_free ?? gpuAlloc;
          const gpuUsedPct = gpuAlloc > 0 ? (100 * (gpuAlloc - (gpuFree ?? gpuAlloc))) / gpuAlloc : null;
          const st = n.ready ? (n.schedulable ? "online" : "partial") : "offline";
          const product = n.gpu_labels["nvidia.com/gpu.product"];
          return (
            <div key={n.name} className={`virt-node ${st}`}>
              <div className="virt-node-h">
                <span className={`container-dot ${st}`} /><strong>{n.name}</strong>
                <span className={`container-badge ${st}`}>{n.ready ? (n.schedulable ? tr("Ready") : tr("Cordoned")) : tr("NotReady")}</span>
                {n.pressures.map((p) => <span key={p} className="k8s-pressure" title={`${p}=True`}><ShieldAlert size={10} /> {p.replace("Pressure", "")}</span>)}
                {n.roles.length ? <span className="virt-node-up">{n.roles.join(", ")}</span> : null}
              </div>
              {gpuAlloc > 0 ? (
                <div className="virt-node-meters one">
                  <Meter label={`GPU${product ? ` · ${product.replace(/^NVIDIA-?/, "")}` : ""}`} pct={gpuUsedPct} text={`${gpuFree} / ${gpuAlloc} ${tr("free")}${n.gpu_requested ? ` · ${n.gpu_requested} ${tr("used")}` : ""}`} />
                </div>
              ) : <div className="virt-node-facts muted"><span>{t(lang, "common.none")} GPU</span></div>}
              <div className="virt-node-facts muted">
                <span>{n.kubelet_version ?? ""}</span>
                {n.cpu_capacity ? <span>{n.cpu_capacity} CPU</span> : null}
                {n.mem_capacity ? <span>{n.mem_capacity}</span> : null}
                {n.taints.length ? <span title={n.taints.map((tt) => tt.key).join(", ")}>{n.taints.length} {tr("taints")}</span> : null}
              </div>
            </div>
          );
        })}
      </div>
    </>
  );
}

function PodsTable({ lang, pods }: { lang: Lang; pods: K8sPod[] | null }) {
  if (pods == null) return <SkeletonBlock />;
  if (pods.length === 0) return <div className="empty-state"><div className="empty-state-icon"><Activity size={20} /></div><p className="empty-state-msg">{tr("No pods")}</p></div>;
  return (
    <table className="containers-table virt-guests">
      <thead><tr><th>{t(lang, "virt.pods")}</th><th>NS</th><th>{tr("Phase")}</th><th>{tr("Ready")}</th><th>↻</th><th>GPU</th><th>{t(lang, "virt.node")}</th></tr></thead>
      <tbody>{pods.map((p) => <PodRow key={`${p.namespace}/${p.name}`} p={p} />)}</tbody>
    </table>
  );
}

function EventsTable({ events }: { events: K8sEvent[] | null }) {
  if (events == null) return <SkeletonBlock />;
  if (events.length === 0) return <div className="empty-state"><div className="empty-state-icon"><AlertTriangle size={20} /></div><p className="empty-state-msg">{tr("No recent events")}</p></div>;
  return (
    <table className="containers-table virt-guests">
      <thead><tr><th>{tr("Type")}</th><th>{tr("Reason")}</th><th>{tr("Object")}</th><th>{tr("Message")}</th><th>×</th></tr></thead>
      <tbody>{events.map((e, i) => <EventRow key={i} e={e} />)}</tbody>
    </table>
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
      <tbody>{kv.vms.map((v) => (
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
      ))}</tbody>
    </table>
  );
}

function SkeletonBlock() {
  return <div className="virt-pane"><div className="virt-node skel" /><div className="virt-node skel" /></div>;
}
