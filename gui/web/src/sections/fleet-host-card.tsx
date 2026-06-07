// SPDX-License-Identifier: Apache-2.0
// Fleet host card — one card per saved host: identity, role, live engine probe,
// SSH checks, discovery, one-click node setup, and live GPU/engine telemetry.
// Extracted from App.tsx (modularization) with no behavior change. roleTone and
// tunnelCommand are exported because the Hosts table reuses them.
import { useEffect, useRef, useState } from "react";
import {
  Server, Cpu, Link2, Boxes, ShieldCheck, KeyRound, MessageSquare, Loader2, PlugZap,
  Activity, Network, SquareTerminal, Sparkles, Rocket, CheckCircle2, AlertTriangle,
  CircleAlert, Box, Pencil, Trash2, Lock, Copy
} from "lucide-react";
import {
  api, type HostProfile, type HostReliability, type FleetHost, type HostProbe,
  type SshCheckResult, type HostDiscovery, type HostSndrState, type NodeSetupResult
} from "../api";
import { toast } from "../components/toast";
import { CapChip } from "../components/primitives";
import { CopyButton } from "../components/code-block";

export function roleTone(role: string): string {
  if (role === "production") return "danger";
  if (role === "staging") return "warn";
  if (role === "dev" || role === "experiment") return "info";
  return "muted";
}

export function tunnelCommand(profile: HostProfile): string {
  return profile.transport === "ssh" && profile.ssh_target
    ? `ssh -L ${profile.port}:127.0.0.1:${profile.port} ${profile.ssh_target}`
    : `# local — open http://127.0.0.1:${profile.port}`;
}

// Actionable guidance when the daemon is read-only (apply gated off). Shows the
// exact restart command with a copy button, instead of a bare "apply is disabled".
function ApplyDisabledNote({ what }: { what: string }) {
  // Canonical entrypoint: `sndr.cli` is the top-level package — importable
  // without the `vllm` namespace, unlike the `vllm.sndr_core.cli` shim (which
  // raises ModuleNotFoundError on nodes where vllm isn't installed). Run it
  // from the SNDR install directory, or after `pip install vllm-sndr-core[gui-api]`.
  const cmd = "python3 -m sndr.cli gui-api --enable-apply";
  const [copied, setCopied] = useState(false);
  return (
    <div className="apply-gate">
      <Lock size={14} />
      <div className="apply-gate-body">
        <strong>{what} needs apply — the daemon is read-only.</strong>
        <span>Restart it with apply enabled (or set <code>SNDR_ENABLE_APPLY=1</code>) — run from the SNDR install dir, or after <code>pip install vllm-sndr-core[gui-api]</code>:</span>
        <div className="apply-gate-cmdrow">
          <code className="apply-gate-cmd">{cmd}</code>
          <button className="apply-gate-copy" onClick={() => { navigator.clipboard?.writeText(cmd); setCopied(true); window.setTimeout(() => setCopied(false), 1500); }}>
            {copied ? <CheckCircle2 size={12} /> : <Copy size={12} />} {copied ? "copied" : "copy"}
          </button>
        </div>
      </div>
    </div>
  );
}

// Bar sparkline of reachability samples (1 = reachable, 0 = down).
function RelSpark({ samples }: { samples: number[] }) {
  if (!samples.length) return null;
  return (
    <svg className="fleet-rel-svg" viewBox={`0 0 ${samples.length} 1`} preserveAspectRatio="none" aria-hidden="true">
      {samples.map((s, i) => (
        <rect key={i} x={i + 0.12} y={s ? 0 : 0.55} width={0.76} height={s ? 1 : 0.45} className={s ? "ok" : "down"} />
      ))}
    </svg>
  );
}

export function FleetHostCard({
  profile,
  onEdit,
  onDelete,
  onChat,
  onAddServer,
  onRefresh,
  onTerminal,
  focused,
  onFocusConsumed,
  onContainers,
  onHardware,
  reliability,
  fleet,
  applyEnabled
}: {
  profile: HostProfile;
  onEdit: (profile: HostProfile) => void;
  onDelete: (id: string) => void;
  onChat: (profile: HostProfile) => void;
  onAddServer: (profile: HostProfile) => Promise<boolean>;
  onRefresh: () => void;
  onTerminal: (profile: HostProfile) => void;
  focused?: boolean;
  onFocusConsumed?: () => void;
  onSetupNode?: (id: string) => void;
  onContainers?: (id: string) => void;
  onHardware?: (id: string) => void;
  reliability?: HostReliability | null;
  fleet?: FleetHost | null;
  applyEnabled?: boolean;
}) {
  const [probe, setProbe] = useState<HostProbe | null>(null);
  const [busy, setBusy] = useState(false);
  const [checkedAt, setCheckedAt] = useState<string | null>(null);
  const [ssh, setSsh] = useState<SshCheckResult | null>(null);
  const [sshBusy, setSshBusy] = useState(false);
  const [keyBusy, setKeyBusy] = useState(false);
  const [disco, setDisco] = useState<HostDiscovery | null>(null);
  const [discoBusy, setDiscoBusy] = useState(false);
  const [sndr, setSndr] = useState<HostSndrState | null>(null);
  // One-click "Set up as node": install the SNDR daemon on this host over SSH.
  const [nodeForm, setNodeForm] = useState(false);
  const [nodePw, setNodePw] = useState("");
  const [nodeBusy, setNodeBusy] = useState(false);
  const [nodeResult, setNodeResult] = useState<NodeSetupResult | null>(null);
  async function installNode() {
    if (nodePw.length < 4 || nodeBusy) return;
    setNodeBusy(true); setNodeResult(null);
    try {
      const r = await api.installNode(profile.id, nodePw, profile.engine_port || 8102);
      setNodeResult(r);
      if (r.ok) onRefresh();  // refresh so the switcher re-probes and sees the new daemon
    } catch (e) {
      setNodeResult({ ok: false, applied: false, steps: [], error: e instanceof Error ? e.message : String(e) });
    } finally { setNodeBusy(false); }
  }
  // null = unknown, false = probed and no daemon (engine box), true = daemon found.
  const [daemonOk, setDaemonOk] = useState<boolean | null>(null);
  const [connecting, setConnecting] = useState(false);
  const cardRef = useRef<HTMLElement | null>(null);
  const isSsh = profile.transport === "ssh" || !!profile.ssh_user;
  // Opened from the connection switcher → scroll into view and auto-discover so
  // the operator immediately sees what's running on this host (the runtime view).
  useEffect(() => {
    if (!focused) return;
    cardRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
    if (isSsh && !disco && !discoBusy) void discover();
    onFocusConsumed?.();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focused]);
  async function discover() {
    setDiscoBusy(true);
    try {
      const d = await api.discoverHost(profile.id);
      setDisco(d);
      // Light Path B: also read this host's own sndr_core identity (patcher
      // version / vLLM build / config + patch-registry size) from its container.
      api.sndrState(profile.id).then((s) => setSndr(s.ok ? s : null)).catch(() => {});
      if (d.engine_port_set) { toast(`Discovered engine → port ${d.engine_port_set} set on ${profile.label}`, "success"); onRefresh(); }
      else if (!d.engines.length) toast(d.error || "Nothing discovered on host", "info");
      else toast(`Found ${d.engines.length} engine(s), ${d.gpus.length} GPU(s)`, "success");
    } catch (err) {
      toast(err instanceof Error ? err.message : "Discovery failed", "error");
    } finally { setDiscoBusy(false); }
  }
  async function applyEnginePort(port: number) {
    try { await api.hostUpsert({ ...profile, engine_port: port }); toast(`Engine port → ${port}`, "success"); onRefresh(); } catch { toast("Failed to set port", "error"); }
  }
  async function fetchKey() {
    setKeyBusy(true);
    try {
      const r = await api.fetchApiKey(profile.id);
      if (r.found) { toast(`API key fetched from ${r.source} → ${r.key_masked}`, "success"); onRefresh(); }
      else toast(r.error || "No API key found on host", "info");
    } catch (err) {
      toast(err instanceof Error ? err.message : "Fetch failed", "error");
    } finally { setKeyBusy(false); }
  }
  async function checkSsh() {
    setSshBusy(true);
    try {
      setSsh(await api.sshCheck({ host: profile.host, host_id: profile.id, user: profile.ssh_user, auth_method: profile.ssh_auth, key_path: profile.ssh_key_path, ssh_port: profile.ssh_port }));
    } catch (err) {
      setSsh({ available: true, ssh_ok: false, sftp_ok: false, latency_ms: null, banner: null, uname: null, error: err instanceof Error ? err.message : String(err) });
    } finally {
      setSshBusy(false);
    }
  }
  async function check() {
    setBusy(true);
    try {
      setProbe(await api.hostProbe(profile.host, profile.engine_port, undefined, profile.id));
    } catch (err) {
      setProbe({ reachable: false, host: profile.host, port: profile.engine_port, base_url: "", version: null, models: [], latency_ms: null, error: err instanceof Error ? err.message : String(err) });
    } finally {
      setBusy(false);
      setCheckedAt(new Date().toLocaleTimeString());
    }
  }
  const statusLabel = !probe ? "not probed" : probe.reachable ? "engine up" : "unreachable";
  const statusTone = !probe ? "muted" : probe.reachable ? "ok" : "danger";
  return (
    <article className={`fleet-card ${focused ? "focused" : ""}`} ref={cardRef}>
      <header className="fleet-card-head">
        <div className="fleet-card-id">
          <Server size={16} />
          <strong>{profile.label}</strong>
          {profile.role && <span className={`fleet-role tone-${roleTone(profile.role)}`}>{profile.role}</span>}
        </div>
        <span className={`fleet-status ${statusTone}`}><span className="fleet-dot" />{statusLabel}</span>
      </header>
      <dl className="fleet-meta">
        <div><dt>Host</dt><dd>{profile.host}</dd></div>
        <div><dt>Transport</dt><dd>{profile.transport}{profile.ssh_target ? ` · ${profile.ssh_target}` : ""}</dd></div>
        <div><dt>Hardware</dt><dd>{profile.hardware || "—"}{profile.gpus ? ` · ${profile.gpus} GPU` : ""}</dd></div>
        <div><dt>Ports</dt><dd>gui {profile.port} · engine {profile.engine_port}</dd></div>
        {probe?.reachable && <div><dt>vLLM</dt><dd>{probe.version ?? "running"}{probe.latency_ms != null ? ` · ${probe.latency_ms} ms` : ""}</dd></div>}
        {probe?.reachable && <div><dt>Served</dt><dd>{probe.models.length ? probe.models.join(", ") : "no models loaded"}</dd></div>}
        {profile.notes && <div><dt>Notes</dt><dd>{profile.notes}</dd></div>}
        {probe && !probe.reachable && probe.error && <div><dt>Probe</dt><dd className="fleet-err">{probe.error}</dd></div>}
      </dl>
      {fleet && (fleet.gpus.length > 0 || fleet.engines.length > 0) && (() => {
        const totalVram = fleet.gpus.reduce((n, g) => n + (parseInt(g.memory_total_mib || "0", 10) || 0), 0);
        const sg = (name: string) => name.replace(/^NVIDIA\s+/i, "").replace(/\s+(GPU|Graphics)$/i, "");
        const sv = (v: string) => v.replace(/^(\d+\.\d+\.\d+).*/, "$1");
        return (
          <div className="fleet-live">
            {fleet.gpus.length > 0 && (
              <div className="fleet-live-sect">
                <div className="fleet-live-t"><Cpu size={12} /> {fleet.gpus.length}× {sg(fleet.gpus[0].name)}
                  {totalVram > 0 && <span className="fleet-live-vram">{Math.round(totalVram / 1024)} GB</span>}
                  {fleet.interconnect && <span className="fleet-live-ic"><Link2 size={10} /> {fleet.interconnect}</span>}
                </div>
                <div className="fleet-live-bars">
                  {fleet.gpus.map((g, i) => {
                    const u = Math.max(0, Math.min(100, parseInt(g.utilization || "0", 10) || 0));
                    return <div key={i} className="fleet-live-bar" title={`GPU ${i} · ${sg(g.name)} · ${Math.round((parseInt(g.memory_total_mib || "0", 10) || 0) / 1024)} GB · ${u}% util`}>
                      <div className="fleet-live-fill" style={{ width: `${Math.max(u, 2)}%` }} /><span>{u}%</span></div>;
                  })}
                </div>
              </div>
            )}
            {fleet.engines.length > 0 && (
              <div className="fleet-live-sect">
                <div className="fleet-live-t"><Boxes size={11} /> {fleet.engines.length} container{fleet.engines.length > 1 ? "s" : ""}
                  {fleet.active_patches > 0 && <span className="fleet-live-patches"><ShieldCheck size={10} /> {fleet.active_patches} patches</span>}
                  {fleet.vllm_version && <span className="fleet-live-ver">vLLM {sv(fleet.vllm_version)}</span>}
                </div>
                {fleet.engines.slice(0, 4).map((e, i) => (
                  <div key={i} className="fleet-live-eng" title={`${e.container ?? "container"}${e.port ? " · :" + e.port : ""} · ${e.reachable ? "reachable" : "unreachable"}`}>
                    <span className={`fleet-live-dot ${e.reachable ? "up" : "down"}`} />
                    <code className="fleet-live-cname">{e.container ?? "—"}</code>
                    {e.port && <span className="fleet-live-port">:{e.port}</span>}
                    {e.reachable && e.version && <span className="fleet-live-evr">{sv(e.version)}</span>}
                    {e.patches > 0 && <span className="fleet-live-ep"><ShieldCheck size={9} /> {e.patches}</span>}
                    {e.models[0] && <span className="fleet-live-emodel" title={e.models.join(", ")}>{e.models[0].split("/").pop()}</span>}
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })()}
      {reliability && reliability.checks > 1 && (
        <div className={`fleet-rel ${reliability.state}`} title={`${reliability.checks} reachability checks · breaker ${reliability.state}`}>
          <span className="fleet-rel-up">{reliability.uptime_pct}% up</span>
          <RelSpark samples={reliability.samples} />
          {reliability.state === "open" && <span className="fleet-rel-state">cooling down</span>}
          {reliability.state === "half_open" && <span className="fleet-rel-state">recovering</span>}
        </div>
      )}
      {isSsh && ssh && (
        <dl className="fleet-meta fleet-ssh-meta">
          <div><dt>SSH</dt><dd className={ssh.ssh_ok ? "fleet-ok" : "fleet-err"}>{ssh.ssh_ok ? `auth ok${ssh.latency_ms != null ? ` · ${ssh.latency_ms} ms` : ""}` : (ssh.error || "failed")}</dd></div>
          {ssh.ssh_ok && <div><dt>SFTP</dt><dd className={ssh.sftp_ok ? "fleet-ok" : "fleet-err"}>{ssh.sftp_ok ? "available" : "unavailable"}</dd></div>}
          {ssh.uname && <div><dt>Remote</dt><dd>{ssh.uname}</dd></div>}
          {!ssh.available && <div><dt>SSH</dt><dd className="fleet-err">paramiko not installed (gui-remote extra)</dd></div>}
        </dl>
      )}
      <div className="fleet-caps">
        <CapChip on={!!probe?.reachable} label="engine" />
        <CapChip on={profile.transport === "ssh"} label="ssh tunnel" />
        {isSsh && <CapChip on={!!ssh?.ssh_ok} label={`ssh ${profile.ssh_auth}`} />}
        {isSsh && ssh?.ssh_ok && <CapChip on={!!ssh?.sftp_ok} label="sftp" />}
        {profile.has_ssh_password && <span className="cap-chip neutral"><KeyRound size={11} />pw stored</span>}
        {profile.gpus > 0 && <span className="cap-chip neutral"><Server size={11} />{profile.gpus} GPU</span>}
        {checkedAt && <span className="fleet-checked">checked {checkedAt}</span>}
      </div>
      {profile.tags.length > 0 && (
        <div className="fleet-tags">{profile.tags.map((tag) => <span key={tag} className="fleet-tag">{tag}</span>)}</div>
      )}
      <code className="fleet-tunnel-line">{tunnelCommand(profile)}</code>
      <div className="fleet-connect">
        <button className="primary-action" onClick={() => onChat(profile)} title={`Open the chat against ${profile.host}:${profile.engine_port}`}>
          <MessageSquare size={14} /> Chat with engine
        </button>
        <button
          className="ghost-button"
          disabled={connecting}
          onClick={async () => { setConnecting(true); try { setDaemonOk(await onAddServer(profile)); } finally { setConnecting(false); } }}
          title={daemonOk === false
            ? "No SNDR daemon reachable here — check the daemon port (8765), or set it up as a node below."
            : `Connect the GUI to the SNDR daemon at http://${profile.host}:${profile.port || 8765}`}>
          {connecting ? <Loader2 size={14} className="spin" /> : <PlugZap size={14} />} {connecting ? "Connecting…" : "Connect daemon"}
        </button>
        {onContainers && (
          <button className="ghost-button" onClick={() => onContainers(profile.id)}
            title={`Manage the containers on ${profile.host} in the Containers section`}>
            <Boxes size={14} /> Containers
          </button>
        )}
        {onHardware && (
          <button className="ghost-button" onClick={() => onHardware(profile.id)}
            title={`View live GPU & hardware telemetry for ${profile.host}`}>
            <Cpu size={14} /> GPU
          </button>
        )}
        {isSsh && (
          <button
            className={`ghost-button ${nodeForm ? "active" : ""}`}
            onClick={() => setNodeForm((v) => !v)}
            title="One-click: install (or reinstall) the SNDR management daemon on this host over SSH, so the GUI can switch to its native view.">
            <Boxes size={14} /> {nodeForm ? "Hide setup" : daemonOk === true ? "Reinstall node" : "Set up as node →"}
          </button>
        )}
      </div>

      {nodeForm && (
        <div className="node-setup">
          <div className="node-setup-head"><Boxes size={13} /> Install SNDR daemon on this node — one click</div>
          <p className="node-setup-desc">Ships the daemon code over SSH, runs it as a sidecar of the engine (LAN-bound, auth on). Then switch the GUI's top menu to this node for native management of its catalog / patches / configs.</p>
          <div className="node-setup-row">
            <label className="param-field"><span>Admin password</span>
              <input type="password" value={nodePw} onChange={(e) => setNodePw(e.target.value)} placeholder="min 4 chars — login as 'root'" autoComplete="off" spellCheck={false} />
            </label>
            <label className="param-field"><span>Engine port</span>
              <input type="number" value={profile.engine_port || 8102} readOnly title="The node's vLLM engine port (from the card)" />
            </label>
          </div>
          {applyEnabled === false && <ApplyDisabledNote what="Installing a node over SSH" />}
          <div className="node-setup-actions">
            <button className="primary-action danger" onClick={() => void installNode()} disabled={nodePw.length < 4 || nodeBusy || applyEnabled === false}>
              {nodeBusy ? <Loader2 size={14} className="spin" /> : <Rocket size={14} />} Install node over SSH
            </button>
            <button className="ghost-button" onClick={() => setNodeForm(false)} disabled={nodeBusy}>Cancel</button>
          </div>
          {nodeResult && (
            <div className={`node-setup-result ${nodeResult.ok ? "ok" : "fail"}`}>
              <div className="node-setup-result-head">
                {nodeResult.ok ? <CheckCircle2 size={14} /> : <AlertTriangle size={14} />}
                <strong>{nodeResult.ok ? `Node ready on :${nodeResult.port} — switch to it from the top menu (login: root)` : "Setup failed"}</strong>
                {nodeResult.error && !/apply is disabled/i.test(nodeResult.error) && <span className="node-setup-err">{nodeResult.error}</span>}
              </div>
              {nodeResult.error && /apply is disabled/i.test(nodeResult.error) && <ApplyDisabledNote what="Installing a node over SSH" />}
              <ol className="node-setup-steps">
                {nodeResult.steps.map((s, i) => (
                  <li key={i} className={s.rc === 0 ? "ok" : "fail"}>
                    <code>{s.rc === 0 ? "✓" : "✗"} {s.cmd}</code>
                    {s.output && <pre>{s.output}</pre>}
                  </li>
                ))}
              </ol>
            </div>
          )}
        </div>
      )}

      <div className="fleet-checks">
        <button className="ghost-button" onClick={() => void check()} disabled={busy}>
          <Activity size={14} /> {busy ? "Probing…" : "Probe engine"}
        </button>
        {isSsh && (
          <button className="ghost-button" onClick={() => void checkSsh()} disabled={sshBusy} title="Test SSH auth + SFTP">
            <Network size={14} /> {sshBusy ? "Checking…" : "SSH check"}
          </button>
        )}
      </div>
      {isSsh && (
        <div className="fleet-checks">
          <button className="ghost-button fleet-discover" onClick={() => void discover()} disabled={discoBusy} title="SSH in and auto-find vLLM containers, ports, models and GPUs — sets the engine port for you">
            {discoBusy ? <Loader2 size={14} className="spin" /> : <Sparkles size={14} />} {discoBusy ? "Discovering…" : "Discover"}
          </button>
          <button className="ghost-button" onClick={() => void fetchKey()} disabled={keyBusy} title="Read the engine's VLLM_API_KEY off the host over SSH and store it on this profile">
            <KeyRound size={14} /> {keyBusy ? "Fetching…" : "Fetch key"}
          </button>
          <button className="ghost-button" onClick={() => onTerminal(profile)} title="Open an SSH terminal to this host (requires the daemon started with SNDR_ENABLE_APPLY=1)">
            <SquareTerminal size={14} /> Terminal
          </button>
        </div>
      )}
      {sndr && sndr.ok && (
        <div className="fleet-sndr">
          <span className="fleet-sndr-lbl"><Boxes size={12} /> sndr_core on host</span>
          <span className="fleet-sndr-chip" title="Genesis patcher (sndr_core) version">patcher {sndr.sndr_version || "?"}</span>
          <span className="fleet-sndr-chip">vLLM {sndr.vllm_version || "?"}</span>
          {sndr.configs != null && <span className="fleet-sndr-chip">{sndr.configs} configs</span>}
          {sndr.patches != null && <span className="fleet-sndr-chip">{sndr.patches} patches in registry</span>}
        </div>
      )}
      {disco && (
        <div className="fleet-disco">
          {disco.engines.length > 0 ? (
            <>
              <span className="fleet-disco-label"><Sparkles size={12} /> Discovered on host</span>
              {disco.engines.map((e) => (
                <div key={e.container} className="fleet-engine-block">
                  <button className={`fleet-engine ${e.host_port === profile.engine_port ? "active" : ""}`} onClick={() => e.host_port && void applyEnginePort(e.host_port)} title={e.host_port === profile.engine_port ? "Active engine port" : "Use this port"}>
                    <span className={`fleet-engine-dot ${e.reachable ? "ok" : "down"}`} />
                    <code className="fleet-engine-port">:{e.host_port ?? "?"}</code>
                    <span className="fleet-engine-name">{e.container}</span>
                    <span className="fleet-engine-meta">{e.version ? `v${e.version}` : e.status}</span>
                  </button>
                  {e.models && e.models.length > 0 && (
                    <div className="fleet-models">{e.models.map((m) => <span key={m} className="fleet-model" title={m}><Box size={10} />{m.split("/").pop()}</span>)}</div>
                  )}
                  {e.genesis_flags && e.genesis_flags.length > 0 && (
                    <div className="fleet-patches">
                      <span className="fleet-patches-lbl"><ShieldCheck size={11} /> {e.genesis_flags.length} active patches</span>
                      {e.genesis_flags.slice(0, 16).map((f) => <span key={f} className="fleet-patch">{f.replace("GENESIS_ENABLE_", "")}</span>)}
                      {e.genesis_flags.length > 16 && <span className="fleet-patch more">+{e.genesis_flags.length - 16}</span>}
                    </div>
                  )}
                </div>
              ))}
            </>
          ) : <span className="fleet-disco-label muted">{disco.error || "No vLLM containers found"}</span>}
          {disco.gpus.length > 0 && <div className="fleet-gpus">{disco.gpus.map((g, i) => <span key={i} className="fleet-gpu"><Cpu size={11} />{g.name} · {Math.round(Number(g.memory_total_mib) / 1024)}GB{g.arch ? ` · ${g.arch}` : ""}{g.utilization != null ? ` · ${g.utilization}%` : ""}</span>)}</div>}
          {disco.interconnect && <span className="fleet-interconnect"><Link2 size={11} /> {disco.interconnect.has_nvlink ? "NVLink" : disco.interconnect.worst_link} — {disco.interconnect.note}</span>}
          {disco.arch_advice && disco.arch_advice.recommendations.length > 0 && (
            <div className="fleet-advice">
              <span className="fleet-disco-label"><ShieldCheck size={12} /> Arch-aware flags ({disco.arch_advice.arch})</span>
              {disco.arch_advice.recommendations.map((rec, i) => (
                <span key={i} className={`fleet-rec ${rec.level}`}>{rec.level === "ok" ? <CheckCircle2 size={11} /> : <CircleAlert size={11} />} {rec.text}</span>
              ))}
            </div>
          )}
        </div>
      )}
      <footer className="fleet-card-actions">
        <CopyButton value={tunnelCommand(profile)} label="tunnel command" />
        <span className="fleet-actions-spacer" />
        <button className="icon-only" onClick={() => onEdit(profile)} aria-label={`Edit ${profile.label}`}><Pencil size={14} /></button>
        <button className="icon-only danger" onClick={() => onDelete(profile.id)} aria-label={`Delete ${profile.label}`}><Trash2 size={14} /></button>
      </footer>
    </article>
  );
}
