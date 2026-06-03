import { useEffect, useRef, useState } from "react";
import { AlertTriangle, Box, ChevronRight, Cpu, Heart, Link2, Loader2, RefreshCw, Server, ShieldCheck } from "lucide-react";
import { api, type FleetHost } from "./api";
import { SkeletonCards } from "./Skeleton";

type Status = "online" | "partial" | "offline";

function statusOf(h: FleetHost): Status {
  if (!h.ssh_ok && (h.error || h.engines.length === 0)) return "offline";
  if (h.engines.some((e) => e.reachable)) return "online";
  return "partial";  // SSH reachable but no engine answering
}
const STATUS_LABEL: Record<Status, string> = { online: "online", partial: "ssh only", offline: "offline" };

const mib = (v: string | null | undefined) => parseInt(v || "0", 10) || 0;
const gpb = (m: number) => Math.round(m / 1024); // MiB -> GiB
const pct = (v: string | null | undefined) => Math.max(0, Math.min(100, parseInt(v || "0", 10) || 0));
const shortGpu = (name: string) => name.replace(/^NVIDIA\s+/i, "").replace(/\s+(GPU|Graphics)$/i, "");
const shortVer = (v: string) => v.replace(/^(\d+\.\d+\.\d+).*/, "$1");

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

      {loading && hosts === null && <SkeletonCards count={4} />}

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

              {/* GPU block — model, total VRAM, interconnect + per-GPU utilisation bars */}
              {h.gpus.length > 0 && (() => {
                const totalVram = h.gpus.reduce((n, g) => n + mib(g.memory_total_mib), 0);
                return (
                  <div className="fleet-gpu">
                    <div className="fleet-gpu-head">
                      <Cpu size={12} /> {h.gpus.length}× {shortGpu(h.gpus[0].name)}
                      {totalVram > 0 && <span className="fleet-gpu-vram">{gpb(totalVram)} GB</span>}
                      {h.interconnect && <span className="fleet-gpu-ic"><Link2 size={10} /> {h.interconnect}</span>}
                    </div>
                    <div className="fleet-gpu-bars">
                      {h.gpus.map((g, i) => {
                        const u = pct(g.utilization);
                        return (
                          <div key={i} className="fleet-gpu-bar" title={`GPU ${i} · ${shortGpu(g.name)} · ${gpb(mib(g.memory_total_mib))}GB · ${u}% util`}>
                            <div className="fleet-gpu-fill" style={{ width: `${Math.max(u, 2)}%` }} />
                            <span>{u}%</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              })()}

              {/* Engines — per-container reachability, port, version and patch count */}
              {h.engines.length > 0 && (
                <div className="fleet-eng">
                  <div className="fleet-eng-head"><Box size={11} /> {h.engines.length} container{h.engines.length > 1 ? "s" : ""}</div>
                  {h.engines.slice(0, 4).map((e, i) => (
                    <div key={i} className="fleet-eng-row" title={`${e.container ?? "container"}${e.port ? " · :" + e.port : ""} · ${e.reachable ? "reachable" : "not reachable"}`}>
                      <span className={`fleet-eng-dot ${e.reachable ? "up" : "down"}`} />
                      <code className="fleet-eng-name">{e.container ?? "—"}</code>
                      {e.port && <span className="fleet-eng-port">:{e.port}</span>}
                      {e.reachable && e.version && <span className="fleet-eng-ver">{shortVer(e.version)}</span>}
                      {e.patches > 0 && <span className="fleet-eng-patches"><ShieldCheck size={9} /> {e.patches}</span>}
                    </div>
                  ))}
                </div>
              )}

              {h.models.length > 0 && (
                <div className="fleet-server-models">{h.models.map((m) => <span key={m} className="fleet-server-model" title={m}><Box size={10} />{m.split("/").pop()}</span>)}</div>
              )}

              <div className="fleet-server-meta">
                {h.vllm_version && <span title="vLLM build"><Heart size={11} /> vLLM {shortVer(h.vllm_version)}</span>}
                {h.active_patches > 0 && <span className="fleet-server-patches" title="active Genesis patches"><ShieldCheck size={11} /> {h.active_patches} patches</span>}
                <span className="fleet-server-open"><ChevronRight size={11} /> open host card</span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
