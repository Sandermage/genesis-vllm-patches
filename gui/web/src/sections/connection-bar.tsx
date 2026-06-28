// SPDX-License-Identifier: Apache-2.0
// Connection bar — the top-bar server switcher (re-point the GUI's Product API
// at any host's daemon, health-pinged) + the small connection map.
import { useEffect, useMemo, useState } from "react";
import { Check, ChevronDown, Database, Link2, Monitor, PackageCheck, PlugZap, Plus, Server } from "lucide-react";
import { type HostProfile, normalizeBaseUrl, hostLabel } from "../api";
import { type RuntimeMode } from "../nav";
import { tr } from "../i18n";

type ConnTarget = { id: string; label: string; baseUrl: string; isLocal: boolean; engineHost?: boolean };

export function ServerSwitcher({
  apiBase,
  connectionTone,
  onSwitch,
  hostProfiles,
  onManageHosts,
  onAddRemoteHost,
  onOpenHost
}: {
  apiBase: string;
  connectionTone: "success" | "warning" | "danger";
  onSwitch: (baseUrl: string) => void;
  hostProfiles: HostProfile[];
  onManageHosts: () => void;
  // Explicit "Add a remote host" action — opens the remote-SSH install wizard.
  // This is the ONLY entry to the SSH fleet flow; it is never the first-run
  // default (a beginner lands on Choose & Launch against the local daemon).
  onAddRemoteHost: () => void;
  onOpenHost: (hostId: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [health, setHealth] = useState<Record<string, "ok" | "down" | "checking">>({});
  const normBase = normalizeBaseUrl(apiBase);

  const targets = useMemo<ConnTarget[]>(() => {
    const localUrl = normalizeBaseUrl(typeof window !== "undefined" ? window.location.origin : "http://127.0.0.1:8765");
    const list: ConnTarget[] = [{ id: "__local__", label: tr("This host (local daemon)"), baseUrl: localUrl, isLocal: true }];
    for (const h of hostProfiles) {
      const url = normalizeBaseUrl(`http://${h.host}:${h.port || 8765}`);
      if (!list.some((t) => t.baseUrl === url)) list.push({ id: h.id, label: h.label, baseUrl: url, isLocal: false, engineHost: h.transport === "ssh" || !!h.ssh_user });
    }
    if (!list.some((t) => t.baseUrl === normBase)) list.push({ id: "__current__", label: hostLabel(normBase), baseUrl: normBase, isLocal: false });
    return list;
  }, [hostProfiles, normBase]);

  const active = targets.find((t) => t.baseUrl === normBase);

  async function ping(t: ConnTarget) {
    setHealth((h) => ({ ...h, [t.id]: "checking" }));
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), 3000);
    try {
      // /api/v1/health is auth-exempt — sending Authorization would add a CORS
      // preflight that can make a reachable daemon look down (matches App.tsx).
      const res = await fetch(`${t.baseUrl}/api/v1/health`, { signal: controller.signal });
      setHealth((h) => ({ ...h, [t.id]: res.ok ? "ok" : "down" }));
    } catch {
      setHealth((h) => ({ ...h, [t.id]: "down" }));
    } finally {
      window.clearTimeout(timer);
    }
  }

  // eslint-disable-next-line react-hooks/exhaustive-deps -- ping the targets only when the dropdown opens
  useEffect(() => { if (open) targets.forEach((t) => void ping(t)); }, [open]);
  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => { if (!(e.target as HTMLElement).closest(".server-switcher")) setOpen(false); };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  const dot = (id: string) => <span className={`srv-dot ${health[id] === "ok" ? "ok" : health[id] === "down" ? "down" : "checking"}`} />;

  return (
    <div className="server-switcher">
      <button className={`server-current tone-${connectionTone}`} onClick={() => setOpen((v) => !v)} title={`${tr("Connected daemon:")} ${normBase}`}>
        <Server size={15} />
        <span className="server-current-label">{active ? active.label : hostLabel(normBase)}</span>
        <ChevronDown size={14} />
      </button>
      {open && (
        <div className="server-menu">
          <div className="server-menu-head">{tr("Daemon connection · from the host registry")}</div>
          <div className="server-list">
            {targets.map((t) => {
              // An engine host (SSH box, no daemon) can't be a daemon target —
              // selecting it opens its card instead of failing a daemon switch.
              // Only when the health ping CONFIRMS it's down (not while still
              // "checking") — otherwise a reachable node daemon would be misrouted.
              const isEngine = !!t.engineHost && health[t.id] === "down" && !t.isLocal && t.id !== "__current__";
              return (
                <div className={`server-item ${t.baseUrl === normBase ? "active" : ""} ${isEngine ? "engine" : ""}`} key={t.id}>
                  <button className="server-pick" onClick={() => { setOpen(false); if (isEngine) onOpenHost(t.id); else onSwitch(t.baseUrl); }}
                    title={isEngine ? tr("Engine host — open its card to see runtime state (Discover / Chat / Terminal)") : t.baseUrl}>
                    {isEngine ? <Server size={13} className="server-engine-ic" /> : dot(t.id)}
                    <span className="server-item-label">{t.label}</span>
                    <span className="server-item-url">{isEngine ? tr("engine host →") : hostLabel(t.baseUrl)}</span>
                    {t.baseUrl === normBase && <Check size={14} className="server-active-check" />}
                  </button>
                </div>
              );
            })}
          </div>
          <button className="server-add" onClick={() => { setOpen(false); onAddRemoteHost(); }}><Plus size={14} /> {tr("Add a remote host")}</button>
          <button className="server-add" onClick={() => { setOpen(false); onManageHosts(); }}><Server size={14} /> {tr("Manage hosts")}</button>
          <div className="server-menu-hint">{tr("Daemons serve patches/presets/configs. A GPU box runs the")} <b>{tr("engine")}</b> {tr("— pick it to open its card and see what's running (models, GPUs, live patches).")}</div>
        </div>
      )}
    </div>
  );
}

export function ConnectionMap({
  runtimeMode,
  runtimeTarget,
  selectedPreset,
  patchCount,
  apiBase
}: {
  runtimeMode: RuntimeMode;
  runtimeTarget: string;
  selectedPreset: string;
  patchCount: number;
  apiBase: string;
}) {
  const nodes = [
    { icon: <Monitor size={18} />, label: tr("GUI Shell"), detail: runtimeMode === "remote" ? tr("remote desktop") : tr("local web") },
    { icon: <PlugZap size={18} />, label: tr("Product API"), detail: apiBase.replace(/^https?:\/\//, "") },
    { icon: <Database size={18} />, label: tr("V2 Catalog"), detail: selectedPreset },
    { icon: <PackageCheck size={18} />, label: tr("Patch Registry"), detail: `${patchCount || "-"} ${tr("entries")}` },
    { icon: <Server size={18} />, label: tr("Runtime Target"), detail: runtimeTarget },
    { icon: <Link2 size={18} />, label: tr("OpenAI API"), detail: tr("client endpoint") }
  ];
  return (
    <section className="connection-map" aria-label={tr("Control plane connection map")}>
      {nodes.map((node, index) => (
        <div className="connection-node" key={node.label}>
          <div className="node-icon">{node.icon}</div>
          <strong>{node.label}</strong>
          <span>{node.detail}</span>
          {index < nodes.length - 1 && <i className="node-link" />}
        </div>
      ))}
    </section>
  );
}

