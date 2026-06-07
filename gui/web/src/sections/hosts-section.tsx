// SPDX-License-Identifier: Apache-2.0
// Hosts section — the Hosts tab: fleet overview with live probes (FleetHostCard),
// this-host inventory, host-profile CRUD (HostFormModal + table), runtime-target
// matrix and an SSH terminal modal. Extracted from App.tsx (modularization) with
// no behavior change.
 
import { lazy, Suspense, useEffect, useState } from "react";
import { Cpu, Database, Network, PackageCheck, Pencil, Server, SlidersHorizontal, Trash2 } from "lucide-react";
import {
  api, type HostProfile, type EnvironmentReport, type ProductOverview, type ProductCapability,
  type HostInventory, type FleetHost, type ReliabilitySnapshot
} from "../api";
import { type RuntimeMode } from "../nav";
import { InfoRows } from "../components/primitives";
import { ModuleCard, ModuleGrid } from "../components/layout";
import { CodeBlock, CopyButton } from "../components/code-block";
import { TabbedSection } from "../components/tabbed-section";
import { ConfirmDialog } from "../components/dialogs";
import { CapabilityTable } from "../components/capability-table";
import { toast } from "../components/toast";
import { DependencyStackPanel, HostInventoryPanel } from "./environment";
import { FleetHostCard, roleTone, tunnelCommand } from "./fleet-host-card";
import { HostFormModal } from "./host-form-modal";
import { ProjectCatalogPanel } from "./project-catalog";
import { ThisHostCard } from "./this-host-card";

const TerminalModal = lazy(() => import("../Terminal").then((m) => ({ default: m.TerminalModal })));

export function HostsSection({
  hostProfiles,
  environment,
  overview,
  runtimeTargets,
  apiBase,
  runtimeMode,
  onHostsRefresh,
  onChatWithHost,
  onAddServer,
  focusHostId,
  onFocusConsumed,
  onSetupNode,
  onContainers,
  onHardware,
  applyEnabled
}: {
  hostProfiles: HostProfile[];
  environment: EnvironmentReport | null;
  overview: ProductOverview | null;
  runtimeTargets: ProductCapability[];
  apiBase: string;
  runtimeMode: RuntimeMode;
  onHostsRefresh: () => void;
  onChatWithHost: (profile: HostProfile) => void;
  onAddServer: (profile: HostProfile) => Promise<boolean>;
  focusHostId: string | null;
  onFocusConsumed: () => void;
  onSetupNode: (id: string) => void;
  onContainers: (id: string) => void;
  onHardware: (id: string) => void;
  applyEnabled?: boolean;
}) {
  const [inventory, setInventory] = useState<HostInventory | null>(null);
  const [modal, setModal] = useState<{ profile: HostProfile | null } | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<{ id: string; label: string } | null>(null);
  const askDelete = (id: string) => setConfirmDelete({ id, label: hostProfiles.find((h) => h.id === id)?.label ?? id });
  const [terminalHost, setTerminalHost] = useState<HostProfile | null>(null);
  const [reliability, setReliability] = useState<ReliabilitySnapshot>({});
  const [fleetById, setFleetById] = useState<Record<string, FleetHost>>({});
  useEffect(() => {
    let cancelled = false;
    api.hostInventory().then((data) => { if (!cancelled) setInventory(data); }).catch(() => {});
    const loadRel = () => api.hostsReliability().then((r) => { if (!cancelled) setReliability(r); }).catch(() => {});
    // Live fleet sweep (GPU util / containers / patches) to enrich each card.
    const loadFleet = () => api.fleetOverview().then((r) => {
      if (!cancelled) setFleetById(Object.fromEntries(r.hosts.map((h) => [h.id, h])));
    }).catch(() => {});
    void loadRel();
    void loadFleet();
    const tf = window.setInterval(() => { if (!document.hidden) void loadFleet(); }, 60000);
    const t = window.setInterval(() => { if (!document.hidden) void loadRel(); }, 8000);
    return () => { cancelled = true; window.clearInterval(t); window.clearInterval(tf); };
  }, []);
  async function remove(id: string) {
    try { await api.hostDelete(id); onHostsRefresh(); toast(`Host removed: ${id}`, "success"); } catch { toast("Failed to remove host", "error"); }
  }
  return (
    <>
      <TabbedSection
        id="hosts"
        tabs={[
          {
            id: "fleet",
            label: "Fleet",
            icon: <Server size={15} />,
            render: () => (
              <ModuleGrid>
                <ModuleCard title="Host fleet" icon={<Server size={18} />} desc="The daemon host plus every saved target, with a live per-host engine probe." wide>
                  <div className="fleet-toolbar">
                    <span className="muted">{hostProfiles.length} saved host{hostProfiles.length === 1 ? "" : "s"}</span>
                    <button className="primary-action" onClick={() => setModal({ profile: null })}><Server size={15} /> Add host</button>
                  </div>
                  {(() => {
                    const fl = Object.values(fleetById);
                    const online = fl.filter((h) => h.engines.some((e) => e.reachable)).length;
                    const gpus = fl.reduce((n, h) => n + (h.gpu_count || 0), 0);
                    const livePatches = fl.reduce((n, h) => n + (h.active_patches || 0), 0);
                    return (
                      <div className="fleet-kpis">
                        <span className="fleet-kpi"><strong>{hostProfiles.length}</strong> servers</span>
                        <span className="fleet-kpi ok"><strong>{online}</strong> online</span>
                        <span className="fleet-kpi"><strong>{gpus}</strong> GPUs</span>
                        <span className="fleet-kpi"><strong>{livePatches}</strong> live patches</span>
                      </div>
                    );
                  })()}
                  <div className="fleet-grid">
                    <ThisHostCard inventory={inventory} environment={environment} apiBase={apiBase} />
                    {hostProfiles.map((profile) => (
                      <FleetHostCard key={profile.id} profile={profile} onEdit={(p) => setModal({ profile: p })} onDelete={askDelete} onChat={onChatWithHost} onAddServer={onAddServer} onRefresh={onHostsRefresh} onTerminal={setTerminalHost} focused={focusHostId === profile.id} onFocusConsumed={onFocusConsumed} onSetupNode={onSetupNode} onContainers={onContainers} onHardware={onHardware} reliability={reliability[profile.id] ?? null} fleet={fleetById[profile.id] ?? null} applyEnabled={applyEnabled} />
                    ))}
                  </div>
                  {hostProfiles.length === 0 && <p className="muted">No remote hosts yet — add your GPU box to probe its engine from here.</p>}
                </ModuleCard>
              </ModuleGrid>
            )
          },
          {
            id: "inventory",
            label: "Inventory",
            icon: <Cpu size={15} />,
            render: () => (
              <ModuleGrid className="stretch-row">
                <ModuleCard title="Daemon host inventory" icon={<Cpu size={18} />} desc="Live OS / Python / Docker / GPU / vLLM snapshot of the host serving this UI." wide>
                  <HostInventoryPanel inventory={inventory} environment={environment} />
                </ModuleCard>
                <ModuleCard title="Dependency stack" icon={<PackageCheck size={18} />} desc="Python libraries and runtime tools detected on the daemon host.">
                  <DependencyStackPanel env={environment} />
                </ModuleCard>
                <ModuleCard title="Project & catalog" icon={<Database size={18} />} desc="Catalog coverage and project parameters this daemon serves.">
                  <ProjectCatalogPanel overview={overview} environment={environment} />
                </ModuleCard>
                <ModuleCard title="Runtime target matrix" icon={<Network size={18} />} desc="Which runtime backends can be rendered or controlled on this host." wide>
                  <CapabilityTable rows={runtimeTargets} />
                </ModuleCard>
              </ModuleGrid>
            )
          },
          {
            id: "profiles",
            label: "Profiles",
            icon: <SlidersHorizontal size={15} />,
            render: () => (
              <ModuleGrid>
                <ModuleCard title="Saved host profiles" icon={<Server size={18} />} desc="Operator-local registry of hosts. Edit role, hardware, ports and tags — execution stays manual." wide>
                  <div className="fleet-toolbar">
                    <span className="muted">{hostProfiles.length} profile{hostProfiles.length === 1 ? "" : "s"}</span>
                    <button className="primary-action" onClick={() => setModal({ profile: null })}><Server size={15} /> Add host</button>
                  </div>
                  <HostProfileTable profiles={hostProfiles} onEdit={(p) => setModal({ profile: p })} onDelete={askDelete} />
                </ModuleCard>
              </ModuleGrid>
            )
          },
          {
            id: "access",
            label: "Access",
            icon: <Network size={15} />,
            render: () => (
              <ModuleGrid>
                <ModuleCard title="Current connection" icon={<Network size={18} />} desc="How this UI reaches the Product API.">
                  <InfoRows rows={[
                    ["API base", apiBase],
                    ["Mode", runtimeMode === "remote" ? "Remote (SSH tunnel)" : "Local server"],
                    ["Engine", `${environment?.engine_name ?? "vLLM"} ${environment?.engine_version ?? (environment?.engine_installed ? "" : "(not installed)")}`.trim()],
                    ["Saved profiles", String(hostProfiles.length)]
                  ]} />
                </ModuleCard>
                <ModuleCard title="Remote access (SSH tunnel)" icon={<Server size={18} />} desc="Keep the daemon on 127.0.0.1; forward a loopback port from your laptop." wide>
                  <CodeBlock lines={[
                    "# on your laptop — forward the daemon's loopback port",
                    "ssh -L 8765:127.0.0.1:8765 user@gpu-host",
                    "# then open the UI at http://127.0.0.1:8765"
                  ]} />
                </ModuleCard>
              </ModuleGrid>
            )
          }
        ]}
      />
      {modal && <HostFormModal initial={modal.profile} onClose={() => setModal(null)} onSaved={onHostsRefresh} />}
      {confirmDelete && (
        <ConfirmDialog
          title="Remove host profile?"
          message={<>This deletes the saved profile <strong>{confirmDelete.label}</strong> (connection details, SSH target, stored key reference). The remote host is not touched, but the profile must be re-added to manage it from here.</>}
          confirmLabel="Remove host"
          danger
          onConfirm={() => { const id = confirmDelete.id; setConfirmDelete(null); void remove(id); }}
          onCancel={() => setConfirmDelete(null)}
        />
      )}
      {terminalHost && <Suspense fallback={null}><TerminalModal host={terminalHost} onClose={() => setTerminalHost(null)} /></Suspense>}
    </>
  );
}

// Dense table view of saved profiles for the Profiles tab.
function HostProfileTable({
  profiles,
  onEdit,
  onDelete
}: {
  profiles: HostProfile[];
  onEdit: (profile: HostProfile) => void;
  onDelete: (id: string) => void;
}) {
  if (profiles.length === 0) return <p className="muted">No saved profiles yet — add one above.</p>;
  return (
    <table className="module-table host-table">
      <thead>
        <tr><th>Label</th><th>Host</th><th>Role</th><th>Hardware</th><th>Ports</th><th>Tags</th><th></th></tr>
      </thead>
      <tbody>
        {profiles.map((profile) => (
          <tr key={profile.id}>
            <td><strong>{profile.label}</strong></td>
            <td>{profile.host}<br /><small className="muted">{profile.transport}{profile.ssh_target ? ` · ${profile.ssh_target}` : ""}</small></td>
            <td>{profile.role ? <span className={`fleet-role tone-${roleTone(profile.role)}`}>{profile.role}</span> : <span className="muted">—</span>}</td>
            <td>{profile.hardware || "—"}{profile.gpus ? ` · ${profile.gpus} GPU` : ""}</td>
            <td><small>gui {profile.port}<br />engine {profile.engine_port}</small></td>
            <td>{profile.tags.length ? <div className="fleet-tags">{profile.tags.map((t) => <span key={t} className="fleet-tag">{t}</span>)}</div> : <span className="muted">—</span>}</td>
            <td className="host-table-actions">
              <CopyButton value={tunnelCommand(profile)} label="tunnel command" />
              <button className="icon-only" onClick={() => onEdit(profile)} aria-label={`Edit ${profile.label}`}><Pencil size={14} /></button>
              <button className="icon-only danger" onClick={() => onDelete(profile.id)} aria-label={`Delete ${profile.label}`}><Trash2 size={14} /></button>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// CodeBlock + CopyButton extracted to ./components/code-block.
// Editable text field (YAML / config) with copy + a fullscreen-edit expand.

