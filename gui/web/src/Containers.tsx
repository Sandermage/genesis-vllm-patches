import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity, AlertTriangle, ArrowDownUp, ArrowLeft, ArrowUp, Bell, Box, Boxes, ChevronRight,
  Clock, Copy, Cpu, Database, Download, DownloadCloud, File as FileIcon, FileArchive, FileCode,
  FileText, Folder, GitCompare, HardDrive, Heart, Home, Layers, Loader2, MemoryStick, MoreVertical,
  Network, Play, RefreshCw, RotateCw, Search, Send, Server, Settings, ShieldAlert, ShieldCheck,
  Square, TerminalSquare, Wrench, X,
} from "lucide-react";
import {
  api, type AlertConfig, type ContainerAction, type ContainerSource, type ContainerStats,
  type ContainerUpdatePlan, type DockerNetwork, type FsEntry, type ImageScan,
  type ManagedContainer, type SourceReport, type SystemDf,
} from "./api";
import { hashParam, buildHash, replaceHash } from "./route";
import { useDialogFocus, useEscapeKey } from "./dialog";

type HostOption = { id: string; label: string };
type NavFn = (section: string) => void;
type StateFilter = "all" | "running" | "stopped";

// ── Container source ⇄ hash-param helpers (deep-linking) ──────────────
// The hash encodes the source as `local` (daemon socket) or a host id.
// Resolve a `src` hash value to a source, or null when it can't be resolved yet
// (e.g. a host id whose profile hasn't loaded). null lets the caller fall back.
function sourceFromKey(key: string | null, hosts: HostOption[]): ContainerSource | null {
  if (key === "local") return { kind: "local" };
  if (key && hosts.some((h) => h.id === key)) return { kind: "host", hostId: key };
  return null;
}
function defaultSource(initialHostId: string | undefined, hosts: HostOption[]): ContainerSource {
  return initialHostId && hosts.some((h) => h.id === initialHostId) ? { kind: "host", hostId: initialHostId } : { kind: "local" };
}
// Fire a toast via the shared ToastHost (App listens for `sndr-toast`). Avoids
// importing from App.tsx, which would create a circular dependency.
function toast(message: string, tone: "success" | "error" | "info" = "info") {
  window.dispatchEvent(new CustomEvent("sndr-toast", { detail: { id: `${Date.now()}-${message.length}`, message, tone } }));
}
type SortKey = "name" | "cpu" | "mem" | "state";
type Inspect = Record<string, any>;

const HISTORY = 30;

function stateClass(state: string): "online" | "partial" | "offline" {
  const s = (state || "").toLowerCase();
  if (s === "running") return "online";
  if (s === "paused" || s === "restarting" || s === "created") return "partial";
  return "offline";
}
function fmtBytes(n: number | undefined): string {
  const v0 = n ?? 0;
  if (!v0) return "0 B";
  const u = ["B", "KiB", "MiB", "GiB", "TiB"];
  let i = 0, v = v0;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v < 10 && i > 0 ? 1 : 0)} ${u[i]}`;
}
function splitArgv(line: string): string[] {
  const m = line.match(/(?:[^\s'"]+|'[^']*'|"[^"]*")+/g) || [];
  return m.map((t) => t.replace(/^['"]|['"]$/g, ""));
}
// "Up 2 hours" → "Up 2h", "Exited (137) 3 minutes ago" → "Exited (137) 3m ago".
function compactStatus(s: string): string {
  return (s || "").replace(/\bAbout an?\b/i, "1").replace(/\ban?\b/gi, "1")
    .replace(/\s*hours?\b/gi, "h").replace(/\s*minutes?\b/gi, "m")
    .replace(/\s*seconds?\b/gi, "s").replace(/\s*days?\b/gi, "d")
    .replace(/\s*weeks?\b/gi, "w").replace(/\s*months?\b/gi, "mo");
}
function pctClass(p: number): "ok" | "warn" | "hot" {
  if (p >= 90) return "hot";
  if (p >= 70) return "warn";
  return "ok";
}

// Minimal ANSI SGR → HTML for the log view (escapes first, so it's XSS-safe).
const ANSI_FG: Record<number, string> = {
  30: "#666", 31: "#e06c75", 32: "#98c379", 33: "#d19a66", 34: "#61afef",
  35: "#c678dd", 36: "#56b6c2", 37: "#b8bfca", 90: "#7d8590", 91: "#ff7b72",
  92: "#7ee787", 93: "#f2cc60", 94: "#79c0ff", 95: "#d2a8ff", 96: "#56d4dd", 97: "#f0f6fc",
};
function escapeHtml(s: string): string {
  return s.replace(/[&<>]/g, (c) => (c === "&" ? "&amp;" : c === "<" ? "&lt;" : "&gt;"));
}
function ansiToHtml(text: string): string {
  const re = /\x1b\[([0-9;]*)m/g;
  let out = "", idx = 0, spanOpen = false;
  let m: RegExpExecArray | null;
  const close = () => { if (spanOpen) { out += "</span>"; spanOpen = false; } };
  while ((m = re.exec(text)) !== null) {
    out += escapeHtml(text.slice(idx, m.index));
    idx = m.index + m[0].length;
    const codes = m[1].split(";").filter(Boolean).map(Number);
    close();
    if (codes.length === 0 || codes.includes(0)) continue;
    let color = "", bold = false;
    for (const c of codes) { if (c === 1) bold = true; else if (ANSI_FG[c]) color = ANSI_FG[c]; }
    const style = `${color ? `color:${color};` : ""}${bold ? "font-weight:600;" : ""}`;
    if (style) { out += `<span style="${style}">`; spanOpen = true; }
  }
  out += escapeHtml(text.slice(idx));
  close();
  return out;
}

function Sparkline({ data, kind, tall }: { data: number[]; kind: "ok" | "warn" | "hot"; tall?: boolean }) {
  const h = tall ? 56 : 22;
  if (data.length < 2) return <svg className={`spark ${kind}`} viewBox={`0 0 100 ${h}`} preserveAspectRatio="none" />;
  const top = Math.max(100, ...data, 1);
  const step = 100 / (HISTORY - 1);
  const pts = data.map((v, i) => `${(i + (HISTORY - data.length)) * step},${h - (v / top) * (h - 2) - 1}`).join(" ");
  const first = (HISTORY - data.length) * step;
  return (
    <svg className={`spark ${kind}`} viewBox={`0 0 100 ${h}`} preserveAspectRatio="none">
      <polygon className="spark-fill" points={`${first},${h} ${pts} 100,${h}`} />
      <polyline className="spark-line" points={pts} />
    </svg>
  );
}
function Bar({ pct, kind, label }: { pct: number; kind: "ok" | "warn" | "hot"; label: string }) {
  return (
    <div className="cbar" title={`${pct.toFixed(1)}%`}>
      <div className={`cbar-fill ${kind}`} style={{ width: `${Math.min(100, Math.max(0, pct))}%` }} />
      <span className="cbar-label">{label}</span>
    </div>
  );
}

// ─── panel: list ⟷ full-page detail ──────────────────────────────────

export function ContainersPanel({ hosts, onNavigate, initialHostId }: { hosts: HostOption[]; onNavigate?: NavFn; initialHostId?: string }) {
  // Deep-link: `#containers?c=<name>&src=<local|hostId>` restores the open
  // container and its source. `src` is captured once on mount so a late hosts[]
  // load can still resolve a host source even after the sync effect rewrites it.
  const [source, setSource] = useState<ContainerSource>(() => sourceFromKey(hashParam("src"), hosts) ?? defaultSource(initialHostId, hosts));
  const pendingSrcRef = useRef<string | null>(typeof window !== "undefined" ? hashParam("src") : null);
  // When the source-reset effect should KEEP the open container across a source
  // change — set only while applying a deep-link host source (the source switch
  // is part of restoring the link, not a user navigating away).
  const keepOpenRef = useRef(false);
  // Cross-section link: arriving from a Host card switches to that host. A
  // deep-link `src=` host is reconciled here too, once the host list arrives.
  useEffect(() => {
    const deepHost = pendingSrcRef.current;
    if (deepHost && deepHost !== "local" && hosts.some((h) => h.id === deepHost)) {
      setSource((cur) => {
        if (cur.kind === "host" && cur.hostId === deepHost) return cur; // already there
        keepOpenRef.current = true; // changing source → preserve the deep-linked container
        return { kind: "host", hostId: deepHost };
      });
      pendingSrcRef.current = null;
    } else if (initialHostId && hosts.some((h) => h.id === initialHostId)) {
      setSource((cur) => (cur.kind === "host" && cur.hostId === initialHostId ? cur : { kind: "host", hostId: initialHostId }));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialHostId, hosts]);
  const [items, setItems] = useState<ManagedContainer[] | null>(null);
  const [stats, setStats] = useState<Record<string, ContainerStats>>({});
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [open, setOpen] = useState<{ name: string; tab?: Tab } | null>(() => {
    const c = hashParam("c");
    return c ? { name: c } : null;
  });
  const [confirmAction, setConfirmAction] = useState<{ name: string; action: ContainerAction } | null>(null);
  const [queryText, setQueryText] = useState("");
  const [filter, setFilter] = useState<StateFilter>("all");
  const [sort, setSort] = useState<SortKey>("state");
  const [df, setDf] = useState<SystemDf | null>(null);
  const [alertsOpen, setAlertsOpen] = useState(false);
  const histRef = useRef<Record<string, { cpu: number[]; mem: number[] }>>({});
  const loadingRef = useRef(false);
  loadingRef.current = loading;

  const load = useCallback(async () => {
    setLoading(true); setErr(null);
    try { setItems((await api.containers(source)).containers); }
    catch (e) { setErr(e instanceof Error ? e.message : String(e)); setItems(null); }
    finally { setLoading(false); }
  }, [source]);

  // Reset + reload whenever the source changes. The first run (mount) must NOT
  // clear `open`, or it would discard a container restored from the deep-link.
  const firstLoadRef = useRef(true);
  useEffect(() => {
    setItems(null); setStats({}); setDf(null); histRef.current = {};
    if (firstLoadRef.current) firstLoadRef.current = false;
    else if (keepOpenRef.current) keepOpenRef.current = false; // deep-link source restore — keep open
    else setOpen(null);
    void load();
  }, [load]);

  // Mirror the open container + source into the hash so the view is shareable
  // (#containers?c=…&src=…). Skips the first run: on a deep-link load the inbound
  // hash is already correct, and on a fresh nav we let App push the bare
  // `#containers` section entry (preserving Back/Forward). Once we've written a
  // deep-link, closing the container rewrites the bare section to drop `c`.
  const syncedRef = useRef(false);
  const lastWriteRef = useRef("");
  useEffect(() => {
    if (!syncedRef.current) {
      syncedRef.current = true;
      lastWriteRef.current = (hashParam("c") || hashParam("src")) ? window.location.hash.replace(/^#\/?/, "") : "";
      return;
    }
    const params: Record<string, string | undefined> = {};
    if (open?.name) params.c = open.name;
    if (source.kind === "host") params.src = source.hostId;
    const hasParams = !!(params.c || params.src);
    if (!hasParams && !lastWriteRef.current) return; // nothing of ours to manage
    const desired = buildHash("containers", params);
    if (window.location.hash.replace(/^#\/?/, "") !== desired) replaceHash(desired);
    lastWriteRef.current = hasParams ? desired : "";
  }, [open, source]);

  // Disk usage (`docker system df`) is heavy — fetch it ONCE per source, deferred,
  // so the container list + first stats win the SSH pool first (server caches it).
  useEffect(() => {
    const t = window.setTimeout(() => { api.systemDf(source).then(setDf).catch(() => setDf(null)); }, 600);
    return () => window.clearTimeout(t);
  }, [source]);

  useEffect(() => {
    if (open || !items || items.length === 0) return;  // pause while a page is open
    let alive = true;
    async function pull() {
      if (document.hidden) return;
      try {
        // One batched call (one SSH connection for the whole set) instead of N.
        const next = (await api.containersStats(source)).stats;
        if (!alive) return;
        for (const [name, s] of Object.entries(next)) {
          const h = histRef.current[name] ?? { cpu: [], mem: [] };
          h.cpu = [...h.cpu, s.cpu_pct ?? 0].slice(-HISTORY);
          h.mem = [...h.mem, s.mem_pct ?? 0].slice(-HISTORY);
          histRef.current[name] = h;
        }
        setStats(next);
      } catch { /* transient — keep last values */ }
    }
    void pull();
    const t = window.setInterval(pull, 4000);
    return () => { alive = false; window.clearInterval(t); };
  }, [items, source, open]);

  async function runAct(name: string, action: ContainerAction) {
    setBusy(`${name}:${action}`); setErr(null);
    try { await api.containerAction(source, name, action); await load(); }
    catch (e) { setErr(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(null); }
  }
  // Disruptive actions (stop/restart) require an explicit confirmation — a
  // misclick on a production engine container would drop live inference.
  // start is non-destructive and runs immediately.
  function act(name: string, action: ContainerAction) {
    if (action === "start") { void runAct(name, action); return; }
    setConfirmAction({ name, action });
  }

  const view = useMemo(() => {
    let v = items || [];
    const q = queryText.trim().toLowerCase();
    if (q) v = v.filter((c) => c.name.toLowerCase().includes(q) || c.image.toLowerCase().includes(q));
    if (filter !== "all") v = v.filter((c) => (filter === "running") === (stateClass(c.state) === "online"));
    return [...v].sort((a, b) => {
      if (sort === "name") return a.name.localeCompare(b.name);
      if (sort === "cpu") return (stats[b.name]?.cpu_pct ?? 0) - (stats[a.name]?.cpu_pct ?? 0);
      if (sort === "mem") return (stats[b.name]?.mem_usage ?? 0) - (stats[a.name]?.mem_usage ?? 0);
      const sa = stateClass(a.state) === "online" ? 0 : 1, sb = stateClass(b.state) === "online" ? 0 : 1;
      return sa - sb || a.name.localeCompare(b.name);
    });
  }, [items, queryText, filter, sort, stats]);

  // Confirmation overlay for disruptive actions — rendered in both the open
  // (ContainerPage) and list views so a stop/restart triggered from either
  // surface is gated the same way.
  const confirmModal = confirmAction && (
    <ConfirmActionModal
      name={confirmAction.name}
      action={confirmAction.action}
      busy={busy === `${confirmAction.name}:${confirmAction.action}`}
      onConfirm={() => { const a = confirmAction; setConfirmAction(null); void runAct(a.name, a.action); }}
      onCancel={() => setConfirmAction(null)}
    />
  );

  if (open) {
    return (
      <>
        <ContainerPage source={source} name={open.name} initialTab={open.tab} busy={busy}
          onBack={() => { setOpen(null); void load(); }}
          onAct={(a) => act(open.name, a)} onNavigate={onNavigate} />
        {confirmModal}
      </>
    );
  }

  const running = (items || []).filter((c) => stateClass(c.state) === "online");
  const stopped = (items || []).length - running.length;
  const sumCpu = running.reduce((n, c) => n + (stats[c.name]?.cpu_pct ?? 0), 0);
  const sumMem = running.reduce((n, c) => n + (stats[c.name]?.mem_usage ?? 0), 0);

  return (
    <div className="containers">
      <div className="containers-bar">
        <label className="containers-source">
          <Server size={14} />
          <select value={source.kind === "local" ? "__local__" : source.hostId}
            onChange={(e) => { const v = e.target.value; setSource(v === "__local__" ? { kind: "local" } : { kind: "host", hostId: v }); }}>
            <option value="__local__">This daemon · docker socket</option>
            {hosts.map((h) => <option key={h.id} value={h.id}>{h.label} · SSH</option>)}
          </select>
        </label>
        <div className="containers-search"><Search size={13} /><input value={queryText} onChange={(e) => setQueryText(e.target.value)} placeholder="filter name / image…" /></div>
        <div className="containers-chips">
          {(["all", "running", "stopped"] as StateFilter[]).map((f) => <button key={f} className={filter === f ? "active" : ""} onClick={() => setFilter(f)}>{f}</button>)}
        </div>
        <label className="containers-sort"><ArrowDownUp size={13} />
          <select value={sort} onChange={(e) => setSort(e.target.value as SortKey)}>
            <option value="state">state</option><option value="name">name</option><option value="cpu">cpu</option><option value="mem">memory</option>
          </select>
        </label>
        <button className="ghost-button" onClick={() => setAlertsOpen(true)} title="Engine health alerts → Telegram"><Bell size={14} /> Alerts</button>
        <button className="ghost-button" onClick={() => void load()} disabled={loading}>
          {loading ? <Loader2 size={14} className="spin" /> : <RefreshCw size={14} />} Refresh
        </button>
      </div>
      {alertsOpen && <AlertsModal onClose={() => setAlertsOpen(false)} />}
      {confirmModal}

      {items !== null && (
        <div className="containers-summary">
          <span className="csum ok"><b>{running.length}</b> running</span>
          <span className="csum"><b>{stopped}</b> stopped</span>
          <span className="csum"><Cpu size={12} /> <b>{sumCpu.toFixed(0)}%</b> CPU</span>
          <span className="csum"><MemoryStick size={12} /> <b>{fmtBytes(sumMem)}</b></span>
          {df && <span className="csum" title={df.types.map((t) => `${t.type}: ${fmtBytes(t.size)} (${fmtBytes(t.reclaimable)} reclaimable)`).join("\n")}><Database size={12} /> <b>{fmtBytes(df.total_size)}</b> disk</span>}
          <span className="containers-auto">live · 4s</span>
        </div>
      )}

      {err && <div className="containers-err"><AlertTriangle size={14} /> {err}</div>}
      {loading && items === null && <div className="containers-loading"><Loader2 size={18} className="spin" /> Querying containers…</div>}
      {items !== null && view.length === 0 && !err && (
        <div className="containers-empty"><Boxes size={22} /><strong>{items.length === 0 ? "No managed containers" : "Nothing matches the filter"}</strong>
          <span>Only vLLM/engine containers (<code>vllm*</code> / <code>sndr-daemon</code> or label <code>sndr.managed=true</code>).</span></div>
      )}

      <div className="containers-grid">
        {view.map((c) => (
          <ContainerCard key={c.id || c.name} c={c} stats={stats[c.name]} history={histRef.current[c.name]}
            busy={busy} onAct={act} onOpen={(tab) => setOpen({ name: c.name, tab })} />
        ))}
      </div>
    </div>
  );
}

function ContainerCard({ c, stats, history, busy, onAct, onOpen }: {
  c: ManagedContainer; stats?: ContainerStats; history?: { cpu: number[]; mem: number[] };
  busy: string | null; onAct: (n: string, a: ContainerAction) => void; onOpen: (tab?: Tab) => void;
}) {
  const st = stateClass(c.state);
  const online = st === "online";
  const cpu = stats?.cpu_pct ?? 0, memPct = stats?.mem_pct ?? 0;
  const ports = (c.ports || "").split(",").map((p) => p.trim()).filter(Boolean);
  const [menu, setMenu] = useState(false);
  const acting = busy?.startsWith(`${c.name}:`) ?? false;

  return (
    <div className={`ccard ${st}`}>
      <span className={`ccard-edge ${st}`} />
      <div className="ccard-top">
        <span className={`container-dot ${st}`} />
        <strong className="ccard-name" title={c.name} onClick={() => onOpen()}>{c.name}</strong>
        <span className={`container-badge ${st}`}>{c.state || "—"}</span>
        <div className="ccard-menu-wrap">
          <button className="ccard-kebab" onClick={() => setMenu((v) => !v)} title="More"><MoreVertical size={15} /></button>
          {menu && (
            <>
              <div className="ccard-menu-back" onClick={() => setMenu(false)} />
              <div className="ccard-menu" onClick={() => setMenu(false)}>
                <button onClick={() => onOpen("config")}><Settings size={13} /> Config</button>
                <button onClick={() => onOpen("stats")}><Cpu size={13} /> Stats</button>
                <button onClick={() => onOpen("processes")}><Activity size={13} /> Processes</button>
                <button onClick={() => onOpen("files")}><Folder size={13} /> Files</button>
                <button onClick={() => onOpen("changes")}><GitCompare size={13} /> Changes</button>
                <button onClick={() => onOpen("exec")}><TerminalSquare size={13} /> Exec</button>
              </div>
            </>
          )}
        </div>
      </div>
      <code className="ccard-image" title={`${c.image}${c.id ? ` · ${c.id}` : ""}`}>{c.image}{c.id ? <span className="ccard-id"> · {c.id.slice(0, 12)}</span> : null}</code>
      {(c.labels?.["sndr.preset"] || c.networks) && (
        <div className="ccard-tags">
          {c.labels?.["sndr.preset"] && <span className="ccard-tag preset" title="Source preset"><Database size={10} /> {c.labels["sndr.preset"]}</span>}
          {c.networks && <span className="ccard-tag net" title="Networks"><Network size={10} /> {c.networks}</span>}
        </div>
      )}

      <div className="ccard-metrics">
        <div className="ccard-metric">
          <div className="ccard-metric-h"><span>CPU</span><b className={online ? "" : "muted"}>{online ? `${cpu.toFixed(0)}%` : "—"}</b></div>
          <Sparkline data={online ? history?.cpu ?? [] : []} kind={pctClass(cpu)} />
        </div>
        <div className="ccard-metric">
          <div className="ccard-metric-h"><span>MEM</span><b className={online ? "" : "muted"}>{online ? `${memPct.toFixed(0)}%` : "—"}</b></div>
          <Sparkline data={online ? history?.mem ?? [] : []} kind={pctClass(memPct)} />
        </div>
      </div>

      <div className="ccard-strip">
        <span title="status"><Clock size={11} /> {compactStatus(c.status) || (online ? "up" : "stopped")}</span>
        <span title="memory">{online ? `${fmtBytes(stats?.mem_usage)}${stats?.mem_limit ? ` / ${fmtBytes(stats?.mem_limit)}` : ""}` : "—"}</span>
        {online && stats?.pids ? <span title="processes"><Activity size={11} /> {stats.pids}</span> : null}
      </div>

      <div className="ccard-ports">
        {ports.length ? ports.slice(0, 3).map((p) => <span key={p} className="port-chip" title={p}>{p}</span>) : <span className="ccard-noports">no published ports</span>}
        {ports.length > 3 && <span className="port-chip more" title={ports.slice(3).join("\n")}>+{ports.length - 3}</span>}
      </div>

      <div className="ccard-foot">
        {online ? (
          <>
            <button className="ccard-act" disabled={acting} onClick={() => onAct(c.name, "restart")} title="Restart">{busy === `${c.name}:restart` ? <Loader2 size={13} className="spin" /> : <RotateCw size={13} />}</button>
            <button className="ccard-act" disabled={acting} onClick={() => onAct(c.name, "stop")} title="Stop">{busy === `${c.name}:stop` ? <Loader2 size={13} className="spin" /> : <Square size={13} />}</button>
          </>
        ) : (
          <button className="ccard-act primary" disabled={acting} onClick={() => onAct(c.name, "start")} title="Start">{busy === `${c.name}:start` ? <Loader2 size={13} className="spin" /> : <Play size={13} />} Start</button>
        )}
        <button className="ccard-act" onClick={() => onOpen("logs")} title="Logs"><FileText size={13} /></button>
        <button className="ccard-open" onClick={() => onOpen()}>Open <ChevronRight size={13} /></button>
      </div>
    </div>
  );
}

// ─── full-page container view ─────────────────────────────────────────

type Tab = "overview" | "config" | "processes" | "files" | "changes" | "logs" | "stats" | "exec";
const TABS: { id: Tab; label: string; icon: React.ReactNode }[] = [
  { id: "overview", label: "Overview", icon: <Box size={15} /> },
  { id: "config", label: "Config", icon: <Settings size={15} /> },
  { id: "processes", label: "Processes", icon: <Activity size={15} /> },
  { id: "files", label: "Files", icon: <Folder size={15} /> },
  { id: "changes", label: "Changes", icon: <GitCompare size={15} /> },
  { id: "logs", label: "Logs", icon: <FileText size={15} /> },
  { id: "stats", label: "Stats", icon: <Cpu size={15} /> },
  { id: "exec", label: "Exec", icon: <TerminalSquare size={15} /> },
];

function ContainerPage({ source, name, busy, onBack, onAct, initialTab, onNavigate }: {
  source: ContainerSource; name: string; busy: string | null;
  onBack: () => void; onAct: (a: ContainerAction) => void; initialTab?: Tab; onNavigate?: NavFn;
}) {
  const [tab, setTab] = useState<Tab>(initialTab ?? "overview");
  const [inspect, setInspect] = useState<Inspect | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [showUpdate, setShowUpdate] = useState(false);

  const reloadInspect = useCallback(() => {
    api.containerInspect(source, name).then(setInspect).catch((e) => setErr(e instanceof Error ? e.message : String(e)));
  }, [source, name]);
  useEffect(() => { reloadInspect(); }, [reloadInspect]);

  const state = inspect?.State ?? {};
  const online = !!state.Running;
  const health = state.Health?.Status as string | undefined;
  const image = inspect?.Config?.Image ?? "";

  return (
    <div className="cpage">
      <div className="cpage-head">
        <button className="ghost-button" onClick={onBack}><ArrowLeft size={15} /> Back</button>
        <span className={`container-dot ${online ? "online" : "offline"}`} />
        <strong className="cpage-name">{name}</strong>
        <span className={`container-badge ${online ? "online" : "offline"}`}>{state.Status || "—"}</span>
        {health && <span className={`health-badge ${health}`}><Heart size={11} /> {health}</span>}
        <div className="cpage-acts">
          {online ? (
            <>
              <button disabled={busy !== null} onClick={() => onAct("restart")}><RotateCw size={14} /> Restart</button>
              <button disabled={busy !== null} onClick={() => onAct("stop")}><Square size={14} /> Stop</button>
            </>
          ) : (
            <button disabled={busy !== null} onClick={() => onAct("start")}><Play size={14} /> Start</button>
          )}
          <button title="Copy a shareable link to this container" onClick={() => {
            void navigator.clipboard?.writeText(window.location.href).then(
              () => toast(`Link to ${name} copied`, "success"),
              () => toast("Could not copy link", "error"),
            );
          }}><Copy size={14} /> Copy link</button>
          <button className="primary-button" onClick={() => setShowUpdate(true)}><Wrench size={14} /> Update</button>
        </div>
      </div>
      <code className="cpage-sub">{image}{inspect?.Id ? ` · ${String(inspect.Id).slice(0, 12)}` : ""}</code>

      {err && <div className="containers-err"><AlertTriangle size={13} /> {err}</div>}

      <div className="cpage-body">
        <nav className="cpage-rail">
          {TABS.map((t) => (
            <button key={t.id} className={tab === t.id ? "active" : ""} onClick={() => setTab(t.id)}>{t.icon} {t.label}</button>
          ))}
        </nav>
        <div className="cpage-content">
          {tab === "overview" && <OverviewTab source={source} name={name} inspect={inspect} online={online} onNavigate={onNavigate} />}
          {tab === "config" && <ConfigTab source={source} name={name} inspect={inspect} onChanged={reloadInspect} />}
          {tab === "processes" && <ProcessesTab source={source} name={name} online={online} />}
          {tab === "files" && <FilesTab source={source} name={name} />}
          {tab === "changes" && <ChangesTab source={source} name={name} />}
          {tab === "logs" && <LogsTab source={source} name={name} />}
          {tab === "stats" && <StatsTab source={source} name={name} online={online} />}
          {tab === "exec" && <ExecTab source={source} name={name} />}
        </div>
      </div>

      {showUpdate && <UpdatePanel source={source} name={name} onClose={() => { setShowUpdate(false); reloadInspect(); }} />}
    </div>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return <div className="kv-row"><span className="kv-key">{label}</span><span className="kv-val">{children}</span></div>;
}

function useLiveStats(source: ContainerSource, name: string, online: boolean, periodMs: number) {
  const [s, setS] = useState<ContainerStats | null>(null);
  const [hist, setHist] = useState<{ cpu: number[]; mem: number[] }>({ cpu: [], mem: [] });
  useEffect(() => {
    if (!online) return;
    let alive = true;
    async function pull() {
      if (document.hidden) return;
      try {
        const st = (await api.containerStats(source, name)).stats;
        if (!alive) return;
        setS(st);
        setHist((h) => ({ cpu: [...h.cpu, st.cpu_pct ?? 0].slice(-HISTORY), mem: [...h.mem, st.mem_pct ?? 0].slice(-HISTORY) }));
      } catch { /* transient */ }
    }
    void pull();
    const t = window.setInterval(pull, periodMs);
    return () => { alive = false; window.clearInterval(t); };
  }, [source, name, online, periodMs]);
  return { s, hist };
}

function SourceCard({ source, name, onNavigate }: { source: ContainerSource; name: string; onNavigate?: NavFn }) {
  const [rep, setRep] = useState<SourceReport | null>(null);
  const [engine, setEngine] = useState<{ reachable: boolean; port: number | null } | null>(null);
  useEffect(() => { api.containerSource(source, name).then(setRep).catch(() => setRep(null)); }, [source, name]);
  // Engine readiness — is the vLLM API inside actually serving (not just the
  // container running)? The live half of the proof, alongside config drift.
  useEffect(() => {
    let alive = true;
    api.containerEngine(source, name).then((e) => { if (alive) setEngine(e); }).catch(() => { if (alive) setEngine(null); });
    return () => { alive = false; };
  }, [source, name]);
  if (!rep) return null;
  return (
    <section className="src-card">
      <div className="src-head">
        <Database size={14} />
        {engine && (
          engine.reachable
            ? <span className="src-engine ok" title={`engine /health on :${engine.port}`}><span className="live-dot on" /> engine serving{engine.port ? ` :${engine.port}` : ""}</span>
            : <span className="src-engine bad" title="container is up but the engine isn't answering /health"><AlertTriangle size={11} /> engine not responding</span>
        )}
        {rep.preset_id ? (
          <>
            <span className="src-label">Source config</span>
            <strong>{rep.preset_title || rep.preset_id}</strong>
            <span className="src-by" title={`linked by ${rep.linked_by}`}>via {rep.linked_by}</span>
            {rep.drift_count > 0
              ? <span className="src-drift bad"><AlertTriangle size={11} /> {rep.drift_count} drift</span>
              : <span className="src-drift ok"><ShieldCheck size={11} /> in sync</span>}
            {onNavigate && <button className="ghost-button" onClick={() => onNavigate("configs")}>Open in Configs <ChevronRight size={12} /></button>}
          </>
        ) : (
          <span className="src-label">No linked preset — launched outside the GUI (no <code>sndr.preset</code> label / name match)</span>
        )}
      </div>
      {rep.drift_count > 0 && (
        <div className="src-drift-list">
          <div className="src-drift-title">Runtime differs from the config that defines it:</div>
          {rep.drift.slice(0, 12).map((d, i) => (
            <div key={i} className={`drift-row ${d.kind}`}>
              <code className="drift-field">{d.field}</code>
              <span className="drift-exp" title="config">{d.expected}</span>
              <span className="drift-arrow">→</span>
              <span className="drift-act" title="running">{d.actual ?? "(unset)"}</span>
            </div>
          ))}
          {rep.drift.length > 12 && <div className="drift-more">+{rep.drift.length - 12} more</div>}
        </div>
      )}
      {rep.patch_sync && (rep.patch_sync.missing.length > 0 || rep.patch_sync.extra.length > 0) && (
        <div className="src-patch-sync">
          {rep.patch_sync.missing.length > 0 && (
            <span className="sps bad" title={rep.patch_sync.missing.join("\n")}><AlertTriangle size={11} /> {rep.patch_sync.missing.length} config patches NOT live</span>
          )}
          {rep.patch_sync.extra.length > 0 && (
            <span className="sps warn" title={rep.patch_sync.extra.join("\n")}>{rep.patch_sync.extra.length} live patches not in config</span>
          )}
          {rep.patch_sync.in_sync.length > 0 && <span className="sps ok">{rep.patch_sync.in_sync.length} in sync</span>}
        </div>
      )}
      {rep.live_patch_count > 0 && (
        <div className="src-patches">
          <div className="src-patches-head">
            <Wrench size={12} /> <strong>{rep.live_patch_count}</strong> Genesis patches live in this engine
            {onNavigate && <button className="link-btn" onClick={() => onNavigate("patches")}>open Patches</button>}
          </div>
          <div className="src-patches-chips">
            {rep.live_patches.slice(0, 16).map((p) => <code key={p.flag} className="patch-chip" title={`${p.flag}=${p.value}`}>{p.flag.replace(/^GENESIS_ENABLE_/, "")}</code>)}
            {rep.live_patches.length > 16 && <span className="drift-more">+{rep.live_patches.length - 16}</span>}
          </div>
        </div>
      )}
    </section>
  );
}

function OverviewTab({ source, name, inspect, online, onNavigate }: { source: ContainerSource; name: string; inspect: Inspect | null; online: boolean; onNavigate?: NavFn }) {
  const { s, hist } = useLiveStats(source, name, online, 3000);
  if (!inspect) return <Loading />;
  const cfg = inspect.Config ?? {}, state = inspect.State ?? {}, net = inspect.NetworkSettings ?? {};
  const cmd = [...(cfg.Entrypoint ?? []), ...(cfg.Cmd ?? [])].join(" ") || "—";
  const networks = Object.keys(net.Networks ?? {});
  const ip = net.IPAddress || (networks.length ? net.Networks[networks[0]]?.IPAddress : "") || "—";
  const cpu = s?.cpu_pct ?? 0, memPct = s?.mem_pct ?? 0;
  return (
    <div className="ov">
      <SourceCard source={source} name={name} onNavigate={onNavigate} />
      {online && (
        <div className="ov-live">
          <div className="ov-metric"><div className="ov-metric-h"><Cpu size={13} /> CPU <b>{cpu.toFixed(1)}%</b></div><Sparkline data={hist.cpu} kind={pctClass(cpu)} tall /></div>
          <div className="ov-metric"><div className="ov-metric-h"><MemoryStick size={13} /> Memory <b>{fmtBytes(s?.mem_usage)}{s?.mem_limit ? ` / ${fmtBytes(s?.mem_limit)}` : ""}</b></div><Sparkline data={hist.mem} kind={pctClass(memPct)} tall /></div>
        </div>
      )}
      <div className="kv">
        <Row label="Image">{cfg.Image || inspect.Image}</Row>
        <Row label="Command"><code className="kv-cmd">{cmd}</code></Row>
        <Row label="Created">{inspect.Created ? new Date(inspect.Created).toLocaleString() : "—"}</Row>
        <Row label="Started">{state.StartedAt && !String(state.StartedAt).startsWith("0001") ? new Date(state.StartedAt).toLocaleString() : "—"}</Row>
        <Row label="Restart count">{inspect.RestartCount ?? 0}</Row>
        <Row label="Health">{state.Health?.Status ?? "—"}</Row>
        <Row label="IP / network">{ip}{networks.length ? ` · ${networks.join(", ")}` : ""}</Row>
        {s && <Row label="Network I/O">↓ {fmtBytes(s.net_rx)} / ↑ {fmtBytes(s.net_tx)}</Row>}
        {s && <Row label="Block I/O">↓ {fmtBytes(s.blk_read)} / ↑ {fmtBytes(s.blk_write)}</Row>}
        {s?.pids ? <Row label="Processes">{s.pids}</Row> : null}
      </div>
    </div>
  );
}

function gpuOf(host: any): string {
  const reqs = host?.DeviceRequests ?? [];
  for (const r of reqs) {
    const caps = (r?.Capabilities ?? []).flat();
    if (r?.Driver === "nvidia" || caps.includes("gpu")) return r?.Count === -1 ? "all" : String(r?.Count ?? "all");
  }
  return "—";
}
const SECRET_RE = /(KEY|TOKEN|SECRET|PASS|PASSWORD|CREDENTIAL)/i;

function EditableSettings({ source, name, inspect, onChanged }: { source: ContainerSource; name: string; inspect: Inspect; onChanged: () => void }) {
  const host = inspect.HostConfig ?? {};
  const [rp, setRp] = useState<string>(host.RestartPolicy?.Name || "no");
  const [cpus, setCpus] = useState<string>(host.NanoCpus ? (host.NanoCpus / 1e9).toString() : "");
  const [memGiB, setMemGiB] = useState<string>(host.Memory ? (host.Memory / 1024 ** 3).toFixed(1) : "");
  const [nets, setNets] = useState<DockerNetwork[]>([]);
  const [attach, setAttach] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const connected = Object.keys(inspect.NetworkSettings?.Networks ?? {});

  useEffect(() => { api.systemNetworks(source).then((r) => setNets(r.networks)).catch(() => setNets([])); }, [source]);

  async function saveSettings() {
    setBusy(true); setMsg(null);
    try {
      await api.containerSettings(source, name, {
        restart_policy: rp,
        cpus: cpus ? Number(cpus) : null,
        memory: memGiB ? Math.round(Number(memGiB) * 1024 ** 3) : null,
      });
      setMsg({ ok: true, text: "Settings updated (live)." }); onChanged();
    } catch (e) { setMsg({ ok: false, text: e instanceof Error ? e.message : String(e) }); }
    finally { setBusy(false); }
  }
  async function net(network: string, action: "connect" | "disconnect") {
    setBusy(true); setMsg(null);
    try { await api.containerNetwork(source, name, network, action); setMsg({ ok: true, text: `${action}ed ${network}.` }); setAttach(""); onChanged(); }
    catch (e) { setMsg({ ok: false, text: e instanceof Error ? e.message : String(e) }); }
    finally { setBusy(false); }
  }
  const attachable = nets.filter((n) => !connected.includes(n.name));

  return (
    <section className="cfg-edit">
      <h4><Wrench size={13} /> Live settings <span>docker update — no recreate</span></h4>
      <div className="cfg-edit-grid">
        <label><span>Restart policy</span>
          <select value={rp} onChange={(e) => setRp(e.target.value)}>
            {["no", "always", "unless-stopped", "on-failure"].map((p) => <option key={p} value={p}>{p}</option>)}
          </select>
        </label>
        <label><span>CPUs</span><input value={cpus} onChange={(e) => setCpus(e.target.value)} placeholder="unlimited" inputMode="decimal" /></label>
        <label><span>Memory (GiB)</span><input value={memGiB} onChange={(e) => setMemGiB(e.target.value)} placeholder="unlimited" inputMode="decimal" /></label>
        <button className="primary-button" disabled={busy} onClick={() => void saveSettings()}>{busy ? <Loader2 size={13} className="spin" /> : <Settings size={13} />} Apply</button>
      </div>
      <div className="cfg-nets">
        <span className="cfg-nets-label"><Network size={12} /> Networks</span>
        {connected.length ? connected.map((n) => (
          <span key={n} className="net-chip">{n}<button title="Disconnect" disabled={busy} onClick={() => void net(n, "disconnect")}><X size={11} /></button></span>
        )) : <span className="cfg-nets-empty">none</span>}
        {attachable.length > 0 && (
          <span className="cfg-nets-attach">
            <select value={attach} onChange={(e) => setAttach(e.target.value)}>
              <option value="">attach…</option>
              {attachable.map((n) => <option key={n.name} value={n.name}>{n.name}</option>)}
            </select>
            {attach && <button className="ghost-button" disabled={busy} onClick={() => void net(attach, "connect")}>Attach</button>}
          </span>
        )}
      </div>
      {msg && <div className={msg.ok ? "upd-done" : "containers-err"}>{!msg.ok && <AlertTriangle size={12} />} {msg.text}</div>}
      <p className="upd-hint">Live edits require apply enabled (SNDR_ENABLE_APPLY=1). cpus/memory/restart apply without recreating; env/ports/image changes need a rebuild.</p>
    </section>
  );
}

function ConfigTab({ source, name, inspect, onChanged }: { source: ContainerSource; name: string; inspect: Inspect | null; onChanged: () => void }) {
  if (!inspect) return <Loading />;
  const cfg = inspect.Config ?? {}, host = inspect.HostConfig ?? {};
  const env: string[] = cfg.Env ?? [], mounts: any[] = inspect.Mounts ?? [];
  const labels: Record<string, string> = cfg.Labels ?? {};
  const ports = Object.keys(inspect.NetworkSettings?.Ports ?? cfg.ExposedPorts ?? {});
  return (
    <div className="cfg">
      <EditableSettings source={source} name={name} inspect={inspect} onChanged={onChanged} />
      <section><h4><Settings size={13} /> Runtime</h4><div className="kv">
        <Row label="Image">{cfg.Image}</Row>
        <Row label="Entrypoint"><code className="kv-cmd">{(cfg.Entrypoint ?? []).join(" ") || "—"}</code></Row>
        <Row label="Command"><code className="kv-cmd">{(cfg.Cmd ?? []).join(" ") || "—"}</code></Row>
        <Row label="Working dir">{cfg.WorkingDir || "—"}</Row>
        <Row label="Network mode">{host.NetworkMode || "—"}</Row>
      </div></section>
      <section><h4><Cpu size={13} /> Resources</h4><div className="kv">
        <Row label="GPUs">{gpuOf(host)}</Row>
        <Row label="Privileged">{host.Privileged ? "yes" : "no"}</Row>
        <Row label="Ports">{ports.join(", ") || "—"}</Row>
      </div></section>
      <section><h4><Layers size={13} /> Environment <span>({env.length})</span></h4><div className="inspect-mono">
        {env.length === 0 ? <em>none</em> : env.map((e) => { const i = e.indexOf("="); const k = i >= 0 ? e.slice(0, i) : e; const v = i >= 0 ? e.slice(i + 1) : "";
          return <div key={e}><span className="env-k">{k}</span>=<span className="env-v">{SECRET_RE.test(k) ? "••••••••" : v}</span></div>; })}
      </div></section>
      <section><h4><HardDrive size={13} /> Mounts <span>({mounts.length})</span></h4><div className="inspect-mono">
        {mounts.length === 0 ? <em>none</em> : mounts.map((m, i) => <div key={i}><span className="env-k">{m.Source}</span> → {m.Destination} <span className="env-v">{m.RW ? "rw" : "ro"}{m.Type ? ` (${m.Type})` : ""}</span></div>)}
      </div></section>
      <section><h4><Boxes size={13} /> Labels <span>({Object.keys(labels).length})</span></h4><div className="inspect-mono">
        {Object.keys(labels).length === 0 ? <em>none</em> : Object.entries(labels).map(([k, v]) => <div key={k}><span className="env-k">{k}</span>=<span className="env-v">{v}</span></div>)}
      </div></section>
    </div>
  );
}

function ProcessesTab({ source, name, online }: { source: ContainerSource; name: string; online: boolean }) {
  const [data, setData] = useState<{ titles: string[]; processes: string[][] } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const load = useCallback(() => {
    api.containerTop(source, name).then((d) => setData({ titles: d.titles, processes: d.processes })).catch((e) => setErr(e instanceof Error ? e.message : String(e)));
  }, [source, name]);
  useEffect(() => { if (online) load(); }, [load, online]);
  if (!online) return <NotRunning />;
  if (err) return <ErrBox msg={err} />;
  if (!data) return <Loading />;
  return (
    <div className="proc">
      <div className="proc-bar"><span>{data.processes.length} processes</span><button className="ghost-button" onClick={load}><RefreshCw size={13} /></button></div>
      <table className="ptable"><thead><tr>{data.titles.map((t) => <th key={t}>{t}</th>)}</tr></thead>
        <tbody>{data.processes.map((row, i) => <tr key={i}>{row.map((cell, j) => <td key={j} className={j === data.titles.length - 1 ? "ptable-cmd" : ""}>{cell}</td>)}</tr>)}</tbody>
      </table>
    </div>
  );
}

function ChangesTab({ source, name }: { source: ContainerSource; name: string }) {
  const [data, setData] = useState<{ kind: string; path: string }[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const load = useCallback(() => { api.containerChanges(source, name).then((d) => setData(d.changes)).catch((e) => setErr(e instanceof Error ? e.message : String(e))); }, [source, name]);
  useEffect(() => { load(); }, [load]);
  if (err) return <ErrBox msg={err} />;
  if (!data) return <Loading />;
  const mark = { added: "A", modified: "C", deleted: "D" } as Record<string, string>;
  return (
    <div className="diff">
      <div className="proc-bar"><span>{data.length} changed paths vs image</span><button className="ghost-button" onClick={load}><RefreshCw size={13} /></button></div>
      {data.length === 0 ? <div className="containers-empty"><GitCompare size={20} /><strong>No changes</strong><span>Container filesystem matches its image.</span></div> : (
        <div className="inspect-mono diff-list">{data.map((c, i) => <div key={i} className={`diff-${c.kind}`}><span className="diff-mark">{mark[c.kind]}</span> {c.path}</div>)}</div>
      )}
    </div>
  );
}

const _CODE_EXT = new Set(["py", "js", "ts", "tsx", "jsx", "json", "yaml", "yml", "toml", "sh", "bash",
  "conf", "cfg", "ini", "env", "xml", "html", "css", "sql", "go", "rs", "c", "cpp", "h", "lua", "rb"]);
const _ARCH_EXT = new Set(["tar", "gz", "tgz", "zip", "xz", "bz2", "7z", "whl", "bin", "safetensors", "pt", "gguf"]);
function fileGlyph(e: FsEntry): { Icon: typeof FileText; cls: string } {
  if (e.is_dir) return { Icon: Folder, cls: "dir" };
  if (e.is_link) return { Icon: FileIcon, cls: "link" };
  const ext = (e.name.split(".").pop() || "").toLowerCase();
  if (_CODE_EXT.has(ext)) return { Icon: FileCode, cls: "code" };
  if (_ARCH_EXT.has(ext)) return { Icon: FileArchive, cls: "arch" };
  if (["log", "out", "err"].includes(ext)) return { Icon: FileText, cls: "log" };
  return { Icon: FileText, cls: "" };
}
const _DOWNLOAD_MAX = 5_000_000;

function FilesTab({ source, name }: { source: ContainerSource; name: string }) {
  const [path, setPath] = useState("/");
  const [entries, setEntries] = useState<FsEntry[] | null>(null);
  const [open, setOpen] = useState<{ entry: FsEntry; content: string; truncated: boolean } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);

  const join = (p: string, n: string) => (p === "/" ? "" : p) + "/" + n;

  const load = useCallback(async (p: string) => {
    setLoading(true); setErr(null); setOpen(null);
    try { setEntries((await api.containerFs(source, name, p)).entries); setPath(p); }
    catch (e) { setErr(e instanceof Error ? e.message : String(e)); }
    finally { setLoading(false); }
  }, [source, name]);
  useEffect(() => { void load("/"); }, [load]);

  async function openFile(e: FsEntry) {
    setErr(null); setBusy(true);
    try { const r = await api.containerFile(source, name, join(path, e.name)); setOpen({ entry: e, content: r.content, truncated: r.truncated }); }
    catch (ex) { setErr(ex instanceof Error ? ex.message : String(ex)); }
    finally { setBusy(false); }
  }
  async function download(fullPath: string, fname: string) {
    setBusy(true); setErr(null);
    try {
      const r = await api.containerFile(source, name, fullPath, _DOWNLOAD_MAX);
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob([r.content], { type: "application/octet-stream" }));
      a.download = fname || "file"; a.click(); URL.revokeObjectURL(a.href);
    } catch (ex) { setErr(ex instanceof Error ? ex.message : String(ex)); }
    finally { setBusy(false); }
  }

  const parts = path.split("/").filter(Boolean);
  const parent = "/" + parts.slice(0, -1).join("/");

  return (
    <div className={`files ${open ? "has-view" : ""}`}>
      <div className="files-bar">
        <button className="files-nav" title="Root" onClick={() => void load("/")}><Home size={13} /></button>
        <button className="files-nav" title="Up" disabled={path === "/"} onClick={() => void load(parent)}><ArrowUp size={13} /></button>
        <div className="files-crumb">
          <button onClick={() => void load("/")}>/</button>
          {parts.map((p, i) => <span key={i}><ChevronRight size={11} /><button onClick={() => void load("/" + parts.slice(0, i + 1).join("/"))}>{p}</button></span>)}
        </div>
        <span className="files-count">{entries ? `${entries.length} items` : ""}</span>
        <button className="files-nav" title="Refresh" onClick={() => void load(path)} disabled={loading}>{loading ? <Loader2 size={13} className="spin" /> : <RefreshCw size={13} />}</button>
      </div>
      {err && <ErrBox msg={err} />}
      <div className="files-split">
        <div className="files-list">
          {(entries ?? []).map((e) => {
            const g = fileGlyph(e);
            return (
              <div key={e.name} className={`file-row ${g.cls} ${open?.entry.name === e.name ? "active" : ""}`}
                onClick={() => e.is_dir ? void load(join(path, e.name)) : void openFile(e)}>
                <g.Icon size={14} className="file-glyph" />
                <span className="file-name">{e.name}{e.is_link && e.link_target ? <span className="file-link"> → {e.link_target}</span> : null}</span>
                <span className="file-size">{e.is_dir ? "" : fmtBytes(e.size)}</span>
                <span className="file-mtime">{e.mtime}</span>
                <span className="file-perms">{e.perms}</span>
                {!e.is_dir && (
                  <button className="file-dl" title="Download" onClick={(ev) => { ev.stopPropagation(); void download(join(path, e.name), e.name); }}><Download size={13} /></button>
                )}
              </div>
            );
          })}
          {entries && entries.length === 0 && !loading && <div className="file-empty"><Folder size={18} /> empty directory</div>}
          {loading && !entries && <Loading />}
        </div>
        {open && (
          <div className="files-view">
            <div className="files-view-head">
              {(() => { const g = fileGlyph(open.entry); return <g.Icon size={14} className={`file-glyph ${g.cls}`} />; })()}
              <code title={join(path, open.entry.name)}>{open.entry.name}</code>
              <span className="files-view-size">{fmtBytes(open.entry.size)}</span>
              {open.truncated && <span className="files-trunc" title="Preview is capped; download for the full file">preview</span>}
              <button className="ghost-button" title="Copy" disabled={busy} onClick={() => navigator.clipboard?.writeText(open.content)}><Copy size={13} /></button>
              <button className="ghost-button" title="Download" disabled={busy} onClick={() => void download(join(path, open.entry.name), open.entry.name)}>{busy ? <Loader2 size={13} className="spin" /> : <Download size={13} />}</button>
              <button className="ghost-button" title="Close" onClick={() => setOpen(null)}><X size={14} /></button>
            </div>
            <pre className="container-logs wrap files-content">{open.content || "(empty file)"}</pre>
            {open.truncated && <div className="files-view-foot"><AlertTriangle size={12} /> Showing the first 64 KB — <button className="link-btn" onClick={() => void download(join(path, open.entry.name), open.entry.name)}>download the full file</button>.</div>}
          </div>
        )}
      </div>
    </div>
  );
}

function StatsTab({ source, name, online }: { source: ContainerSource; name: string; online: boolean }) {
  const { s, hist } = useLiveStats(source, name, online, 2000);
  if (!online) return <NotRunning />;
  if (!s) return <Loading label="Sampling…" />;
  const cpu = s.cpu_pct ?? 0, memPct = s.mem_pct ?? 0;
  return (
    <div className="stats-tab">
      <div className="stats-big">
        <div className="stats-metric"><div className="stats-metric-head"><Cpu size={13} /> CPU <b>{cpu.toFixed(1)}%</b></div><Sparkline data={hist.cpu} kind={pctClass(cpu)} tall /></div>
        <div className="stats-metric"><div className="stats-metric-head"><MemoryStick size={13} /> Memory <b>{fmtBytes(s.mem_usage)}{s.mem_limit ? ` / ${fmtBytes(s.mem_limit)}` : ""}</b> ({memPct.toFixed(1)}%)</div><Sparkline data={hist.mem} kind={pctClass(memPct)} tall /></div>
      </div>
      <div className="kv">
        <Row label="Network RX / TX">{fmtBytes(s.net_rx)} / {fmtBytes(s.net_tx)}</Row>
        <Row label="Block read / write">{fmtBytes(s.blk_read)} / {fmtBytes(s.blk_write)}</Row>
        <Row label="Processes">{s.pids ?? 0}</Row>
      </div>
    </div>
  );
}

function LogsTab({ source, name }: { source: ContainerSource; name: string }) {
  const [logs, setLogs] = useState("");
  const [tail, setTail] = useState(500);
  const [q, setQ] = useState("");
  const [wrap, setWrap] = useState(true);
  const [live, setLive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const preRef = useRef<HTMLPreElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const load = useCallback(async () => {
    setLoading(true); setErr(null);
    try { setLogs((await api.containerLogs(source, name, tail)).logs); }
    catch (e) { setErr(e instanceof Error ? e.message : String(e)); }
    finally { setLoading(false); }
  }, [source, name, tail]);

  // Snapshot mode (when not live).
  useEffect(() => { if (!live) void load(); }, [load, live]);

  // Live mode: stream over fetch, append (cap buffer), auto-scroll.
  useEffect(() => {
    if (!live) { abortRef.current?.abort(); abortRef.current = null; return; }
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setLogs(""); setErr(null);
    void api.containerLogStream(source, name, tail,
      { onLine: (t) => setLogs((prev) => (prev + t).slice(-400000)),
        onError: (m) => setErr(m) },
      ctrl.signal);
    return () => ctrl.abort();
  }, [live, source, name, tail]);

  useEffect(() => { if (preRef.current) preRef.current.scrollTop = preRef.current.scrollHeight; }, [logs]);

  const shown = q ? logs.split("\n").filter((l) => l.toLowerCase().includes(q.toLowerCase())).join("\n") : logs;
  function download() {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([logs], { type: "text/plain" }));
    a.download = `${name}.log`; a.click(); URL.revokeObjectURL(a.href);
  }
  return (
    <div className="logs-tab">
      <div className="logs-bar">
        <button className={`ghost-button ${live ? "live-on" : ""}`} onClick={() => setLive(!live)} title="Follow logs live">
          <span className={`live-dot ${live ? "on" : ""}`} /> {live ? "Live" : "Follow"}
        </button>
        <div className="containers-search"><Search size={12} /><input value={q} onChange={(e) => setQ(e.target.value)} placeholder="grep logs…" /></div>
        <select value={tail} onChange={(e) => setTail(Number(e.target.value))}>{[200, 500, 1000, 2000].map((n) => <option key={n} value={n}>{n} lines</option>)}</select>
        <button className={`ghost-button ${wrap ? "on" : ""}`} onClick={() => setWrap(!wrap)}>wrap</button>
        {!live && <button className="ghost-button" onClick={() => void load()} disabled={loading}>{loading ? <Loader2 size={13} className="spin" /> : <RefreshCw size={13} />}</button>}
        <button className="ghost-button" onClick={download}><DownloadCloud size={13} /></button>
      </div>
      {err && <ErrBox msg={err} />}
      <pre ref={preRef} className={`container-logs ansi ${wrap ? "wrap" : ""}`}
        dangerouslySetInnerHTML={{ __html: ansiToHtml(shown) || (loading ? "" : "(no output)") }} />
    </div>
  );
}

function ExecTab({ source, name }: { source: ContainerSource; name: string }) {
  const [cmd, setCmd] = useState("");
  const [hist, setHist] = useState<Array<{ cmd: string; rc: number; out: string }>>([]);
  const [running, setRunning] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  async function run() {
    const argv = splitArgv(cmd.trim());
    if (argv.length === 0) return;
    setRunning(true); setErr(null);
    try {
      const r = await api.containerExec(source, name, argv);
      setHist((h) => [...h, { cmd, rc: r.exit_code, out: (r.stdout || "") + (r.stderr ? `\n${r.stderr}` : "") }]);
      setCmd("");
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)); }
    finally { setRunning(false); }
  }
  return (
    <div className="exec-tab">
      <div className="container-exec-warn"><AlertTriangle size={12} /> Runs inside the container. Requires <code>SNDR_ENABLE_EXEC=1</code>.</div>
      <div className="exec-scroll">
        {hist.map((h, i) => (
          <div key={i} className="exec-entry">
            <div className="exec-cmd"><code>$ {h.cmd}</code> <span className={h.rc === 0 ? "exec-rc ok" : "exec-rc bad"}>exit {h.rc}</span></div>
            {h.out && <pre className="container-logs wrap">{h.out}</pre>}
          </div>
        ))}
      </div>
      {err && <ErrBox msg={err} />}
      <div className="container-exec-input">
        <code>$</code>
        <input value={cmd} onChange={(e) => setCmd(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && !running) void run(); }} placeholder="python3 -c 'print(1)'" autoFocus />
        <button className="primary-button" disabled={running || !cmd.trim()} onClick={() => void run()}>{running ? <Loader2 size={13} className="spin" /> : <Play size={13} />} Run</button>
      </div>
    </div>
  );
}

function UpdatePanel({ source, name, onClose }: { source: ContainerSource; name: string; onClose: () => void }) {
  const [plan, setPlan] = useState<ContainerUpdatePlan | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState<string | null>(null);
  const [scan, setScan] = useState<ImageScan | null>(null);
  const [scanning, setScanning] = useState(false);
  useEffect(() => { api.containerUpdatePlan(source, name).then(setPlan).catch((e) => setErr(e instanceof Error ? e.message : String(e))); }, [source, name]);

  async function runScan() {
    setScanning(true); setScan(null); setErr(null);
    try { setScan(await api.containerScan(source, name)); }
    catch (e) { setErr(e instanceof Error ? e.message : String(e)); }
    finally { setScanning(false); }
  }

  async function guardedUpdate() {
    setBusy(true); setErr(null); setDone(null);
    try { const r = await api.containerPull(source, name, true); setDone(`Pulled ${r.image}${r.restarted ? " and restarted" : ""}.`); }
    catch (e) { setErr(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }

  return (
    <div className="container-drawer-backdrop" onClick={onClose}>
      <div className="container-modal upd-modal" onClick={(e) => e.stopPropagation()}>
        <div className="container-modal-head"><Wrench size={16} /><strong>Update {name}</strong><div className="container-modal-acts"><button className="ghost-button" onClick={onClose}><X size={15} /></button></div></div>
        <div className="container-modal-body">
          {err && <ErrBox msg={err} />}
          {!plan ? <Loading /> : plan.is_engine ? (
            <div className="upd">
              <p className="upd-note"><AlertTriangle size={13} /> This is a vLLM engine. {plan.policy} Run these on the GPU host:</p>
              <div className="upd-cmds">
                {plan.commands.map((c, i) => (
                  <div key={i} className="upd-cmd"><code>{c}</code>{!c.trim().startsWith("#") && <button className="ghost-button" onClick={() => navigator.clipboard?.writeText(c)}><Copy size={12} /></button>}</div>
                ))}
              </div>
              <button className="ghost-button" onClick={() => navigator.clipboard?.writeText(plan.commands.filter((c) => !c.trim().startsWith("#")).join("\n"))}><Copy size={13} /> Copy all</button>
              <div className="upd-pins"><span>Patcher-supported pins:</span> {plan.supported_pins.map((p) => <code key={p} className={p === plan.canonical_pin ? "pin canonical" : "pin"}>{p}</code>)}</div>
            </div>
          ) : (
            <div className="upd">
              <p className="upd-note">Pull the latest <code>{plan.image}</code> and restart the container.</p>
              {done && <div className="upd-done">{done}</div>}
              <button className="primary-button" disabled={busy} onClick={guardedUpdate}>{busy ? <Loader2 size={13} className="spin" /> : <DownloadCloud size={13} />} Pull image + restart</button>
              <p className="upd-hint">Requires the daemon to run with apply enabled (SNDR_ENABLE_APPLY=1).</p>
            </div>
          )}

          <div className="upd-scan">
            <div className="upd-scan-head"><ShieldCheck size={14} /> <strong>Image vulnerability scan</strong> <span>safe-pull · Grype/Trivy</span>
              <button className="ghost-button" onClick={() => void runScan()} disabled={scanning}>{scanning ? <Loader2 size={13} className="spin" /> : <ShieldAlert size={13} />} Scan</button>
            </div>
            {scanning && <p className="upd-hint">Scanning {plan?.image || "image"} — this pulls a CVE database, may take a minute…</p>}
            {scan && !scan.available && <p className="upd-hint"><AlertTriangle size={12} /> {scan.reason}</p>}
            {scan && scan.available && scan.counts && (
              <div className="scan-result">
                <span className="scan-scanner">{scan.scanner}</span>
                {scan.total === 0 ? <span className="sev clean"><ShieldCheck size={12} /> no known CVEs</span> : (
                  <>
                    {scan.counts.critical > 0 && <span className="sev critical">{scan.counts.critical} critical</span>}
                    {scan.counts.high > 0 && <span className="sev high">{scan.counts.high} high</span>}
                    {scan.counts.medium > 0 && <span className="sev medium">{scan.counts.medium} medium</span>}
                    {scan.counts.low > 0 && <span className="sev low">{scan.counts.low} low</span>}
                    <span className="sev total">{scan.total} total</span>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// Confirmation gate for disruptive container actions (stop / restart). Cancel
// is the autofocused default and Esc cancels, so the destructive path always
// takes a deliberate second action.
function ConfirmActionModal({ name, action, busy, onConfirm, onCancel }: {
  name: string; action: ContainerAction; busy: boolean;
  onConfirm: () => void; onCancel: () => void;
}) {
  const dialogRef = useRef<HTMLDivElement>(null);
  useDialogFocus(dialogRef);
  useEscapeKey(onCancel);
  const verb = action === "stop" ? "Stop" : "Restart";
  return (
    <div className="container-drawer-backdrop" onClick={onCancel}>
      <div ref={dialogRef} className="container-modal confirm-modal" role="dialog" aria-modal="true" aria-label={`${verb} ${name}`} onClick={(e) => e.stopPropagation()}>
        <div className="container-modal-head">
          <AlertTriangle size={16} /><strong>{verb} {name}?</strong>
          <div className="container-modal-acts"><button className="ghost-button" onClick={onCancel} aria-label="Cancel"><X size={15} /></button></div>
        </div>
        <div className="container-modal-body">
          <p className="confirm-msg">
            {action === "stop"
              ? "This stops the container. If it serves a live engine, in-flight inference is dropped until it is started again."
              : "This restarts the container. The engine is briefly unavailable and any in-flight inference is interrupted."}
          </p>
          <div className="confirm-actions">
            {/* eslint-disable-next-line jsx-a11y/no-autofocus */}
            <button className="ghost-button" onClick={onCancel} disabled={busy} autoFocus>Cancel</button>
            <button className={`primary-button ${action === "stop" ? "danger" : ""}`} onClick={onConfirm} disabled={busy}>
              {busy ? <Loader2 size={13} className="spin" /> : action === "stop" ? <Square size={13} /> : <RotateCw size={13} />} {verb}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function AlertsModal({ onClose }: { onClose: () => void }) {
  const [cfg, setCfg] = useState<AlertConfig | null>(null);
  const [chatId, setChatId] = useState("");
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);

  const reload = useCallback(() => {
    api.alertsConfig().then((c) => { setCfg(c); setChatId(c.chat_id); }).catch((e) => setMsg({ ok: false, text: e instanceof Error ? e.message : String(e) }));
  }, []);
  useEffect(() => { reload(); }, [reload]);

  async function save(enabled?: boolean) {
    setBusy(true); setMsg(null);
    try {
      const next = await api.alertsSetConfig({
        enabled: enabled ?? cfg?.enabled, chat_id: chatId,
        ...(token ? { bot_token: token } : {}),
      });
      setCfg(next); setToken("");
      setMsg({ ok: true, text: "Saved." });
    } catch (e) { setMsg({ ok: false, text: e instanceof Error ? e.message : String(e) }); }
    finally { setBusy(false); }
  }
  async function test() {
    setBusy(true); setMsg(null);
    try { const r = await api.alertsTest(); setMsg({ ok: r.ok, text: r.ok ? "Test sent — check Telegram." : (r.error || "Send failed") }); }
    catch (e) { setMsg({ ok: false, text: e instanceof Error ? e.message : String(e) }); }
    finally { setBusy(false); }
  }

  return (
    <div className="container-drawer-backdrop" onClick={onClose}>
      <div className="container-modal upd-modal" onClick={(e) => e.stopPropagation()}>
        <div className="container-modal-head"><Bell size={16} /><strong>Engine health alerts</strong><span>Telegram</span>
          <div className="container-modal-acts"><button className="ghost-button" onClick={onClose}><X size={15} /></button></div>
        </div>
        <div className="container-modal-body">
          {!cfg ? <Loading /> : (
            <div className="alerts-cfg">
              <p className="upd-note">Get a push when a managed engine container goes <b>DOWN</b> (crash / OOM / stop) or recovers. The daemon watches over the docker socket.</p>
              <label className="alerts-row"><span>Enabled</span>
                <button className={`toggle ${cfg.enabled ? "on" : ""}`} disabled={busy} onClick={() => void save(!cfg.enabled)}><span className="toggle-knob" /></button>
              </label>
              <label className="alerts-row"><span>Chat ID</span>
                <input value={chatId} onChange={(e) => setChatId(e.target.value)} placeholder="e.g. 123456789" />
              </label>
              <label className="alerts-row"><span>Bot token</span>
                <input type="password" value={token} onChange={(e) => setToken(e.target.value)} placeholder={cfg.has_token ? "•••••• (stored — leave blank to keep)" : "123456:ABC-DEF…"} />
              </label>
              <div className="alerts-status">
                <span className={`sev ${cfg.has_token ? "clean" : "low"}`}>{cfg.has_token ? "token stored" : "no token"}</span>
                <span className={`sev ${cfg.configured ? "clean" : "low"}`}>{cfg.configured ? "configured" : "incomplete"}</span>
              </div>
              <div className="alerts-actions">
                <button className="primary-button" disabled={busy} onClick={() => void save()}>{busy ? <Loader2 size={13} className="spin" /> : <Settings size={13} />} Save</button>
                <button className="ghost-button" disabled={busy || !cfg.configured} onClick={() => void test()}><Send size={13} /> Send test</button>
              </div>
              {msg && <div className={msg.ok ? "upd-done" : "containers-err"}>{!msg.ok && <AlertTriangle size={13} />} {msg.text}</div>}
              <p className="upd-hint">Saving requires the daemon to run with apply enabled (SNDR_ENABLE_APPLY=1). Token is stored encrypted; env <code>SNDR_TELEGRAM_BOT_TOKEN</code>/<code>SNDR_TELEGRAM_CHAT_ID</code> also work.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── small shared bits ────────────────────────────────────────────────
function Loading({ label = "Loading…" }: { label?: string }) { return <div className="containers-loading"><Loader2 size={16} className="spin" /> {label}</div>; }
function NotRunning() { return <div className="containers-empty"><Activity size={20} /><strong>Container not running</strong><span>Start it to see live data.</span></div>; }
function ErrBox({ msg }: { msg: string }) { return <div className="containers-err"><AlertTriangle size={13} /> {msg}</div>; }
