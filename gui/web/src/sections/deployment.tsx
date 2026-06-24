// SPDX-License-Identifier: Apache-2.0
// Deployment console — pick a target + preset, review the resolved launch
// parameters, host readiness and dependency plan, then render the exact
// deployment artifact (compose / quadlet / k8s / systemd / bare-metal / proxmox)
// with copy + download. Read-only against the daemon: nothing is executed on the
// host.
import { useEffect, useState, type ReactNode } from "react";
import {
  Rocket, Box, Package, Layers, Settings, Cpu, Server, Database, ShieldCheck,
  CheckCircle2, AlertCircle, CircleAlert, HardDrive, FileText, Terminal, Download
} from "lucide-react";
import { api, type PresetListResult, type DeployTargetsResult, type DeploymentPlan, type DeployTarget } from "../api";
import { tr } from "../i18n";
import { ModuleGrid, ModuleCard } from "../components/layout";
import { InfoRows, RailCheck } from "../components/primitives";
import { CodeBlock, CopyButton } from "../components/code-block";
import { fmtParam } from "../lib/format";
import { SkeletonLines } from "../Skeleton";

const DEPLOY_TARGET_ICONS: Record<string, ReactNode> = {
  compose: <Box size={16} />,
  quadlet: <Package size={16} />,
  kubernetes: <Layers size={16} />,
  systemd: <Settings size={16} />,
  bare_metal: <Cpu size={16} />,
  proxmox: <Server size={16} />,
  proxmox_vm: <HardDrive size={16} />,
  sndr_daemon: <Terminal size={16} />
};

function downloadText(filename: string, content: string) {
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(url);
}

export function DeploymentConsole({
  presets,
  selectedPreset,
  onSelectPreset
}: {
  presets: PresetListResult | null;
  selectedPreset: string;
  onSelectPreset: (id: string) => void;
}) {
  const [meta, setMeta] = useState<DeployTargetsResult | null>(null);
  const [target, setTarget] = useState<string>("compose");
  const [plan, setPlan] = useState<DeploymentPlan | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [paths, setPaths] = useState<Record<string, string>>({});
  const [imagePin, setImagePin] = useState("");
  const [withDaemon, setWithDaemon] = useState(false);

  useEffect(() => {
    let cancelled = false;
    api.deployTargets()
      .then((data) => { if (!cancelled) setMeta(data); })
      .catch(() => { /* targets card simply stays empty */ });
    return () => { cancelled = true; };
  }, []);

  // Reset operator path overrides whenever the preset changes.
  useEffect(() => { setPaths({}); }, [selectedPreset]);

  useEffect(() => {
    if (!selectedPreset) return;
    let cancelled = false;
    const handle = window.setTimeout(() => {
      setLoading(true);
      api.deployPlan({
        preset_id: selectedPreset,
        target,
        host_paths: Object.keys(paths).length ? paths : undefined,
        image_override: imagePin.trim() || undefined,
        with_daemon: withDaemon || undefined
      })
        .then((result) => { if (!cancelled) { setPlan(result); setError(null); } })
        .catch((err) => { if (!cancelled) setError(err instanceof Error ? err.message : String(err)); })
        .finally(() => { if (!cancelled) setLoading(false); });
    }, 300);
    return () => { cancelled = true; window.clearTimeout(handle); };
  }, [selectedPreset, target, paths, imagePin, withDaemon]);

  const targets = meta?.targets ?? [];
  const host = meta?.host ?? null;
  const activeTarget: DeployTarget | undefined = targets.find((entry) => entry.id === target);
  const params = plan?.parameters;
  const deps = plan?.dependencies;
  const presetList = presets?.presets ?? [];

  const depTone = deps && (deps.n_blockers ?? 0) > 0 ? "blocked" : deps && (deps.n_warnings ?? 0) > 0 ? "warning" : "pass";
  const dockerOk = host?.docker.installed && host.docker.daemon_running;
  const gpuOk = host?.nvidia.installed && host.nvidia.n_gpus > 0;

  return (
    <ModuleGrid>
      <ModuleCard
        title={tr("Deployment target")}
        icon={<Rocket size={18} />}
        desc={tr("Choose how the pinned vLLM stack lands on the host. Each target renders a ready-to-apply artifact.")}
        wide
      >
        <div className="deploy-targets">
          {targets.map((entry) => (
            <button
              key={entry.id}
              type="button"
              className={`deploy-target ${entry.id === target ? "active" : ""}`}
              onClick={() => setTarget(entry.id)}
            >
              <span className="deploy-target-head">
                {DEPLOY_TARGET_ICONS[entry.id] ?? <Box size={16} />}
                <strong>{entry.label}</strong>
                {entry.needs && <em className="deploy-need">{tr("needs")} {entry.needs}</em>}
              </span>
              <small>{entry.summary}</small>
            </button>
          ))}
          {targets.length === 0 && <p className="muted">{tr("Loading deployment targets…")}</p>}
        </div>
      </ModuleCard>

      <ModuleCard
        title={tr("Preset & launch parameters")}
        icon={<Database size={18} />}
        desc={tr("The resolved engine command for this preset — tensor parallelism, KV-cache dtype, context window and Genesis pin.")}
      >
        <label className="deploy-field">
          <span>{tr("Preset")}</span>
          <select value={selectedPreset} onChange={(event) => onSelectPreset(event.target.value)}>
            {presetList.length === 0 && <option value={selectedPreset}>{selectedPreset}</option>}
            {presetList.map((preset) => (
              <option key={preset.id} value={preset.id}>{preset.id}</option>
            ))}
          </select>
        </label>
        <label className="deploy-field">
          <span>{tr("Engine pin override")}</span>
          <input type="text" value={imagePin} onChange={(e) => setImagePin(e.target.value)} placeholder={tr("blank = preset default image")} spellCheck={false} autoComplete="off" />
        </label>
        <label className="deploy-toggle">
          <input type="checkbox" checked={withDaemon} onChange={(e) => setWithDaemon(e.target.checked)} />
          <span>{tr("Also provision the SNDR management daemon (sidecar)")}</span>
        </label>
        {params ? (
          <>
          <InfoRows
            rows={[
              [tr("Tensor parallel"), `${fmtParam(params.tensor_parallel)} GPU`],
              [tr("KV-cache dtype"), fmtParam(params.kv_cache_dtype)],
              [tr("Max context"), fmtParam(params.max_model_len)],
              [tr("Max sequences"), fmtParam(params.max_num_seqs)],
              [tr("GPU mem util"), params.gpu_memory_utilization != null ? `${Math.round(params.gpu_memory_utilization * 100)}%` : "—"],
              [tr("Min VRAM / GPU"), params.min_vram_per_gpu_mib != null ? `${Math.round(params.min_vram_per_gpu_mib / 1024)} GiB` : "—"],
              [tr("Genesis flags"), fmtParam(params.genesis_env_count)],
              [tr("Pin"), fmtParam(params.genesis_pin)],
              [tr("Container"), fmtParam(params.container_name)],
              [tr("Host port"), fmtParam(params.host_port)],
              [tr("Image"), fmtParam(params.image)]
            ]}
          />
          {Array.isArray(params.argv) && params.argv.length > 0 && (
            <div className="deploy-argv">
              <div className="deploy-argv-head">
                <span className="deploy-argv-label"><Terminal size={12} /> {tr("Resolved vLLM argv")}</span>
                <CopyButton value={params.argv.join(" ")} label={tr("Copy")} />
              </div>
              <div className="deploy-argv-tokens">
                {params.argv.map((token, i) => <code key={i} className="deploy-argv-token">{token}</code>)}
              </div>
            </div>
          )}
          </>
        ) : (
          loading ? <SkeletonLines count={5} /> : <p className="muted">{tr("Select a preset to resolve launch parameters.")}</p>
        )}
      </ModuleCard>

      <ModuleCard
        title={tr("Host readiness")}
        icon={<ShieldCheck size={18} />}
        desc={tr("Live inventory of the daemon host and the dependency plan for this preset.")}
      >
        <div className="setup-glance">
          <RailCheck label="OS" value={host ? `${host.os.distro || host.os.system} ${host.os.arch}` : "…"} status="pass" />
          <RailCheck label="Docker" value={host ? (dockerOk ? `${tr("running")} ${host.docker.server_version ?? ""}`.trim() : host.docker.installed ? tr("stopped") : tr("missing")) : "…"} status={dockerOk ? "pass" : "warning"} />
          <RailCheck label="GPU" value={host ? (gpuOk ? `${host.nvidia.n_gpus}× ${host.nvidia.gpu_names[0] ?? "GPU"}` : tr("none")) : "…"} status={gpuOk ? "pass" : "warning"} />
          <RailCheck label="vLLM" value={host ? (host.vllm.installed ? host.vllm.version ?? tr("installed") : tr("not installed")) : "…"} status={host?.vllm.installed ? "pass" : "warning"} />
        </div>
        {deps && (
          <div className="deploy-deps">
            <div className={`deploy-deps-head tone-${depTone}`}>
              {depTone === "pass" ? <CheckCircle2 size={15} /> : depTone === "blocked" ? <AlertCircle size={15} /> : <CircleAlert size={15} />}
              <strong>
                {deps.is_ready
                  ? tr("Host is ready for this preset")
                  : `${deps.n_blockers} ${deps.n_blockers === 1 ? tr("blocker") : tr("blockers")} · ${deps.n_warnings} ${deps.n_warnings === 1 ? tr("warning") : tr("warnings")}`}
              </strong>
            </div>
            {(deps.items ?? []).map((item, index) => (
              <div className={`deploy-dep-item sev-${item.severity}`} key={`${item.scope}-${index}`}>
                <div className="deploy-dep-row">
                  <span className={`sev-dot sev-${item.severity === "blocker" ? "danger" : item.severity === "warning" ? "warn" : "info"}`} />
                  <strong>{item.target}</strong>
                  <em>{item.action}</em>
                </div>
                <small>{item.reason}</small>
                {item.suggested_command && <CodeBlock lines={[item.suggested_command]} />}
              </div>
            ))}
            {(deps.items ?? []).length === 0 && <p className="muted">{tr("No host changes required for this preset.")}</p>}
          </div>
        )}
      </ModuleCard>

      {plan && (plan.mount_vars?.length ?? 0) > 0 && (
        <ModuleCard
          title={tr("Storage & mount paths")}
          icon={<HardDrive size={18} />}
          desc={tr("Map the preset's container mounts to real host paths. Edits re-render the artifact below.")}
          wide
        >
          <div className="deploy-mounts">
            {(plan.mount_vars ?? []).map((mount) => (
              <label className="deploy-field" key={mount.name}>
                <span>{mount.name} <em className="deploy-mount-target">→ {mount.container}</em></span>
                <input
                  type="text"
                  value={paths[mount.name] !== undefined ? paths[mount.name] : mount.value}
                  spellCheck={false}
                  onChange={(event) => setPaths((prev) => ({ ...prev, [mount.name]: event.target.value }))}
                />
              </label>
            ))}
          </div>
        </ModuleCard>
      )}

      <ModuleCard
        title={activeTarget ? `${activeTarget.label} — ${activeTarget.filename}` : tr("Generated artifact")}
        icon={<FileText size={18} />}
        desc={tr("The exact file to drop on the host, plus the operator commands to apply it.")}
        wide
      >
        {error && <div className="inline-error"><AlertCircle size={15} /> {error}</div>}
        {plan?.artifact ? (
          <>
            <div className="deploy-artifact-bar">
              <span className="deploy-artifact-name"><Terminal size={14} /> {plan.artifact.filename}</span>
              <div className="deploy-artifact-actions">
                <CopyButton value={plan.artifact.content} label={plan.artifact.filename} />
                <button className="ghost-button" onClick={() => downloadText(plan.artifact.filename, plan.artifact.content)}>
                  <Download size={14} /> {tr("Download")}
                </button>
              </div>
            </div>
            <div className="deploy-artifact"><CodeBlock lines={plan.artifact.content.split("\n")} title={plan.artifact.filename} /></div>
            <div className="deploy-cmd-head">
              <h4 className="deploy-cmd-title">{tr("Apply commands")}</h4>
              {(() => {
                // One copyable, fail-fast shell script: write the artifact via a
                // heredoc, then run the apply commands — no manual file shuffling.
                const eof = "SNDR_EOF";
                const script = [
                  "#!/usr/bin/env bash", "set -euo pipefail", "",
                  `# ${activeTarget?.label ?? "deploy"} — generated by SNDR Control Center`,
                  `cat > ${plan.artifact.filename} <<'${eof}'`,
                  plan.artifact.content.replace(/\n$/, ""),
                  eof, "",
                  ...(plan.commands ?? []),
                ].join("\n");
                return (
                  <div className="deploy-cmd-actions">
                    <CopyButton value={script} label={tr("apply script")} />
                    <button className="ghost-button" onClick={() => downloadText(`apply-${activeTarget?.id ?? "deploy"}.sh`, script)}><Download size={13} /> {tr("Script")}</button>
                  </div>
                );
              })()}
            </div>
            <CodeBlock lines={plan.commands ?? []} />
          </>
        ) : (
          loading ? <SkeletonLines count={5} /> : <p className="muted">{tr("Select a preset and target to render the deployment artifact.")}</p>
        )}
      </ModuleCard>
    </ModuleGrid>
  );
}
