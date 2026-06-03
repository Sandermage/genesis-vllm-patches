import { useEffect, useRef, useState } from "react";
import { AlertTriangle, Bell, Check, ChevronRight, Cpu, HardDrive, Server } from "lucide-react";
import { api, type Alert, type AlertsSnapshot } from "./api";

const CAT_ICON: Record<string, React.ReactNode> = {
  gpu: <Cpu size={14} />,
  disk: <HardDrive size={14} />,
  host: <Server size={14} />,
};

function rel(sinceSec: number): string {
  const s = Math.max(0, Math.round(Date.now() / 1000 - sinceSec));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  return `${Math.floor(s / 3600)}h`;
}

function AlertRow({ a, resolved }: { a: Alert; resolved?: boolean }) {
  return (
    <div className={`alert-row ${a.level}${resolved ? " resolved" : ""}`}>
      <span className="alert-cat">{CAT_ICON[a.category] ?? <AlertTriangle size={14} />}</span>
      <div className="alert-body">
        <strong className="alert-title">{a.title}</strong>
        <span className="alert-detail">{a.detail}</span>
      </div>
      <span className="alert-age">{resolved ? "cleared" : rel(a.first_seen)}</span>
    </div>
  );
}

// Header bell — polls the hardware alert store and surfaces a badge + dropdown.
export function AlertsBell({ onOpenHardware }: { onOpenHardware?: () => void }) {
  const [snap, setSnap] = useState<AlertsSnapshot | null>(null);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let alive = true;
    const load = async () => {
      try { const s = await api.alerts(); if (alive) setSnap(s); } catch { /* daemon may be offline */ }
    };
    void load();
    const t = window.setInterval(() => { if (!document.hidden) void load(); }, 10000);
    return () => { alive = false; window.clearInterval(t); };
  }, []);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  const crit = snap?.counts.critical ?? 0;
  const warn = snap?.counts.warn ?? 0;
  const total = crit + warn;
  const tone = crit > 0 ? "critical" : warn > 0 ? "warn" : "ok";
  const active = snap?.active ?? [];
  const recent = snap?.recent ?? [];

  return (
    <div className="alerts-bell" ref={ref}>
      <button className={`tool-button alerts-trigger ${tone}`} onClick={() => setOpen((o) => !o)}
        title={total > 0 ? `${total} active alert${total === 1 ? "" : "s"}` : "No active alerts"}>
        <Bell size={16} />
        {total > 0 && <span className={`alerts-badge ${tone}`}>{total}</span>}
      </button>
      {open && (
        <div className="alerts-pop">
          <div className="alerts-pop-head">
            <strong>Alerts</strong>
            {onOpenHardware && (
              <button className="alerts-link" onClick={() => { setOpen(false); onOpenHardware(); }}>
                Hardware <ChevronRight size={13} />
              </button>
            )}
          </div>
          {active.length === 0 && (
            <div className="alerts-clear"><Check size={18} /><span>All clear — no active alerts.</span></div>
          )}
          {active.length > 0 && (
            <div className="alerts-list">
              {active.map((a) => <AlertRow key={a.key} a={a} />)}
            </div>
          )}
          {recent.length > 0 && (
            <>
              <div className="alerts-sub">Recently cleared</div>
              <div className="alerts-list muted">
                {recent.slice(0, 5).map((a) => <AlertRow key={`${a.key}:${a.resolved_at}`} a={a} resolved />)}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
