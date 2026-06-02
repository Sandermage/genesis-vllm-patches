import { useEffect, useRef, useState } from "react";
import { AlertTriangle, Box, ChevronRight, Cpu, Link2, Loader2, RefreshCw, Server, ShieldCheck } from "lucide-react";
import { api, type FleetHost } from "./api";

type Status = "online" | "partial" | "offline";

function statusOf(h: FleetHost): Status {
  if (!h.ssh_ok && (h.error || h.engines.length === 0)) return "offline";
  if (h.engines.some((e) => e.reachable)) return "online";
  return "partial";  // SSH reachable but no engine answering
}
const STATUS_LABEL: Record<Status, string> = { online: "online", partial: "ssh only", offline: "offline" };

// Fleet overview — every registered engine host at a glance (the hybrid model's
// width layer). One concurrent SSH+probe sweep; drill into a host's card for
// detail. Read-only: nothing here mutates a server.
export function FleetPanel({ onOpenHost }: { onOpenHost: (id: string) => void }) {
  const [hosts, setHosts] = useState<FleetHost[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const loadingRef = useRef(false);
  loadingRef.current = loading;

  async function load() {
    setLoading(true); setErr(null);
    try { setHosts((await api.fleetOverview()).hosts); }
    catch (e) { setErr(e instanceof Error ? e.message : String(e)); }
    finally { setLoading(false); }
  }
  useEffect(() => { void load(); }, []);
  // Auto-refresh the sweep every 60s (skip while one is in flight or tab hidden,
  // so we don't pile up SSH connections to the fleet).
  useEffect(() => {
    const t = window.setInterval(() => { if (!loadingRef.current && !document.hidden) void load(); }, 60000);
    return () => window.clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const online = (hosts || []).filter((h) => statusOf(h) === "online").length;
  const totalGpus = (hosts || []).reduce((n, h) => n + h.gpu_count, 0);
  const totalPatches = (hosts || []).reduce((n, h) => n + h.active_patches, 0);

  return (
    <div className="fleet">
      <div className="fleet-bar">
        <div className="fleet-stats">
          <span className="fleet-stat"><b>{hosts ? hosts.length : "—"}</b> servers</span>
          <span className="fleet-stat ok"><b>{online}</b> online</span>
          <span className="fleet-stat"><b>{totalGpus}</b> GPUs</span>
          <span className="fleet-stat"><b>{totalPatches}</b> live patches</span>
        </div>
        <span className="fleet-auto">auto · 60s</span>
        <button className="ghost-button" onClick={() => void load()} disabled={loading}>
          {loading ? <Loader2 size={14} className="spin" /> : <RefreshCw size={14} />} Refresh fleet
        </button>
      </div>
      {err && <div className="fleet-err"><AlertTriangle size={14} /> {err}</div>}

      {hosts !== null && hosts.length === 0 && (
        <div className="fleet-empty"><Server size={22} /><strong>No engine hosts yet</strong>
          <span>Add a GPU server in <b>Hosts</b> — it shows up here with its live state.</span></div>
      )}

      {loading && hosts === null && <div className="fleet-loading"><Loader2 size={18} className="spin" /> Sweeping the fleet over SSH…</div>}

      <div className="fleet-overview-grid">
        {(hosts || []).map((h) => {
          const st = statusOf(h);
          return (
            <button key={h.id} className={`fleet-server ${st}`} onClick={() => onOpenHost(h.id)} title="Open this host's card">
              <div className="fleet-server-head">
                <span className={`fleet-server-dot ${st}`} />
                <strong>{h.label}</strong>
                {h.role && <span className="fleet-server-role">{h.role}</span>}
                <span className="fleet-server-status">{STATUS_LABEL[st]}</span>
                <ChevronRight size={15} className="fleet-server-go" />
              </div>
              <code className="fleet-server-host">{h.host}</code>

              {h.error && <div className="fleet-server-err"><AlertTriangle size={12} /> {h.error}</div>}

              {h.models.length > 0 && (
                <div className="fleet-server-models">{h.models.map((m) => <span key={m} className="fleet-server-model" title={m}><Box size={10} />{m.split("/").pop()}</span>)}</div>
              )}

              <div className="fleet-server-meta">
                {h.vllm_version && <span title="vLLM build">vLLM {h.vllm_version}</span>}
                {h.gpu_count > 0 && <span><Cpu size={11} /> {h.gpu_count}× {h.arch || "GPU"}</span>}
                {h.interconnect && <span><Link2 size={11} /> {h.interconnect}</span>}
                {h.active_patches > 0 && <span className="fleet-server-patches"><ShieldCheck size={11} /> {h.active_patches} patches</span>}
                {h.engines.length > 0 && <span>{h.engines.length} container{h.engines.length > 1 ? "s" : ""}</span>}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
