// SPDX-License-Identifier: Apache-2.0
// First-run setup wizard: a guided, read-only path (detect → mode → preset →
// validate → launch) that never mutates the host. Extracted from App.tsx
// (modularization).
//
// Enterprise hardening over the inline original (markup classes unchanged, so
// visuals are identical):
//   * the progress track is a real role="progressbar" with aria-valuenow/min/max
//     + aria-valuetext, so assistive tech announces "N of M steps ready";
//   * the active step carries aria-current="step".
import { useState } from "react";
import { CheckCircle2, AlertCircle, CircleAlert, Circle, Database, Rocket, ShieldCheck, Network } from "lucide-react";
import { type EnvironmentReport, type ProductOverview, type DoctorReport } from "../api";
import { type GateStatus, InfoRows, RailCheck, StatusBadge, DoctorStat } from "../components/primitives";
import { type SectionId, type RuntimeMode } from "../nav";
import { CodeBlock } from "../components/code-block";
import { EnvironmentPanel } from "./environment";

type WizardStatus = "done" | "active" | "todo" | "warning" | "blocked";

export function SetupWizard({
  environment,
  overview,
  doctorReport,
  gateCounts,
  selectedPreset,
  runtimeMode,
  apiBase,
  onSection
}: {
  environment: EnvironmentReport | null;
  overview: ProductOverview | null;
  doctorReport: DoctorReport | null;
  gateCounts: Record<GateStatus, number>;
  selectedPreset: string;
  runtimeMode: RuntimeMode;
  apiBase: string;
  onSection: (section: SectionId) => void;
}) {
  const env = environment;
  const dockerTool = env?.tools.find((tool) => tool.name === "docker");
  const nvidiaTool = env?.tools.find((tool) => tool.name === "nvidia-smi");
  const doctorBlocked = doctorReport?.summary.blocked ?? 0;
  const doctorWarn = doctorReport?.summary.warning ?? 0;
  const host = runtimeMode === "remote" ? "gpu-build-01" : "127.0.0.1";

  const steps: Array<{ key: string; title: string; hint: string; status: WizardStatus }> = [
    { key: "detect", title: "Detect host", hint: "Engine, Python and runtime tools", status: env ? "done" : "active" },
    { key: "mode", title: "Connection mode", hint: "Local server or remote SSH tunnel", status: "done" },
    { key: "preset", title: "Choose a preset", hint: "Pick a workload-matched config", status: selectedPreset ? "done" : "todo" },
    { key: "validate", title: "Validate", hint: "Run diagnostics, clear blockers", status: doctorReport ? (doctorBlocked > 0 ? "blocked" : doctorWarn > 0 ? "warning" : "done") : "todo" },
    { key: "launch", title: "Plan & launch", hint: "Compose a launch plan", status: gateCounts.blocked > 0 ? "blocked" : selectedPreset ? "done" : "todo" }
  ];
  const [active, setActive] = useState(0);
  const done = steps.filter((step) => step.status === "done").length;
  const cur = steps[active];

  const tone = (status: WizardStatus) =>
    status === "done" ? "ok" : status === "blocked" ? "danger" : status === "warning" ? "warn" : status === "active" ? "accent" : "muted";
  const icon = (status: WizardStatus) =>
    status === "done" ? <CheckCircle2 size={15} /> : status === "blocked" ? <AlertCircle size={15} /> : status === "warning" ? <CircleAlert size={15} /> : <Circle size={15} />;

  return (
    <section className="setup-wizard">
      <div className="setup-progress">
        <div>
          <strong>First-run setup</strong>
          <span>Guided path — read-only, no host mutation</span>
        </div>
        <div className="setup-progress-meta">{done}/{steps.length} ready</div>
        <div
          className="setup-progress-track"
          role="progressbar"
          aria-valuenow={done}
          aria-valuemin={0}
          aria-valuemax={steps.length}
          aria-valuetext={`${done} of ${steps.length} steps ready`}
        >
          <span style={{ width: `${(done / steps.length) * 100}%` }} />
        </div>
      </div>

      <div className="setup-glance">
        <RailCheck label="Engine" value={env?.engine_version ? `vLLM ${env.engine_version}` : "not installed"} status={env?.engine_installed ? "pass" : "warning"} />
        <RailCheck label="Docker" value={dockerTool?.present ? "available" : "missing"} status={dockerTool?.present ? "pass" : "warning"} />
        <RailCheck label="GPU (nvidia-smi)" value={nvidiaTool?.present ? "available" : "missing"} status={nvidiaTool?.present ? "pass" : "warning"} />
        <RailCheck label="Doctor" value={`${doctorBlocked} blocked · ${doctorWarn} warn`} status={doctorBlocked > 0 ? "warning" : "pass"} />
        <RailCheck label="Preset" value={selectedPreset || "none"} status={selectedPreset ? "pass" : "warning"} />
        <RailCheck label="Gates" value={`${gateCounts.pass}/${gateCounts.pass + gateCounts.warning + gateCounts.blocked}`} status={gateCounts.blocked > 0 ? "warning" : "pass"} />
      </div>

      <div className="setup-grid">
        <aside className="setup-steps">
          {steps.map((step, index) => (
            <button
              key={step.key}
              className={`setup-step tone-${tone(step.status)} ${index === active ? "active" : ""}`}
              aria-current={index === active ? "step" : undefined}
              onClick={() => setActive(index)}
            >
              <span className="setup-step-icon">{icon(step.status)}</span>
              <div>
                <strong>{step.title}</strong>
                <small>{step.hint}</small>
              </div>
              <span className="setup-step-num">{index + 1}</span>
            </button>
          ))}
        </aside>

        <section className="setup-content">
          <header className="setup-content-head">
            <h3>{cur.title}</h3>
            <StatusBadge status={cur.status === "done" ? "available" : cur.status === "blocked" ? "missing" : cur.status === "warning" ? "partial" : "deferred"} />
          </header>

          {cur.key === "detect" && (
            <>
              <EnvironmentPanel env={env} />
              <CodeBlock lines={["python -m vllm.sndr_core.cli gui-api --host 127.0.0.1 --port 8765", "python -m vllm.sndr_core.cli doctor --all"]} />
            </>
          )}
          {cur.key === "mode" && (
            <>
              <InfoRows rows={[
                ["Active mode", runtimeMode === "remote" ? "Remote desktop (SSH tunnel)" : "Local server"],
                ["API base", apiBase],
                ["Bind", "127.0.0.1 (localhost only)"],
                ["Writes", "Disabled — dry-run apply jobs only"]
              ]} />
              <CodeBlock lines={runtimeMode === "remote"
                ? [`ssh -L 8765:127.0.0.1:8765 user@${host}`, "# then open http://127.0.0.1:8765 locally"]
                : ["python -m vllm.sndr_core.cli gui-api --host 127.0.0.1 --port 8765"]} />
            </>
          )}
          {cur.key === "preset" && (
            <>
              <InfoRows rows={[
                ["Selected preset", selectedPreset || "none"],
                ["Catalog presets", overview?.catalog.presets_count ?? "-"],
                ["Models", overview?.catalog.models_count ?? "-"],
                ["Profiles", overview?.catalog.profiles_count ?? "-"]
              ]} />
              <div className="setup-actions">
                <button className="ghost-button" onClick={() => onSection("presets")}><Database size={15} /> Browse presets</button>
                <button className="primary-action" onClick={() => onSection("launch-plan")}><Rocket size={15} /> Recommend by workload</button>
              </div>
            </>
          )}
          {cur.key === "validate" && (
            <>
              <div className="doctor-stat-row">
                <DoctorStat tone="ok" value={doctorReport?.summary.ok ?? 0} label="Healthy" />
                <DoctorStat tone="warn" value={doctorWarn} label="Warnings" />
                <DoctorStat tone="danger" value={doctorBlocked} label="Blocked" />
                <DoctorStat tone="info" value={doctorReport?.findings.length ?? 0} label="Checks" />
              </div>
              <div className="setup-actions">
                <button className="primary-action" onClick={() => onSection("doctor")}><ShieldCheck size={15} /> Open Doctor</button>
              </div>
            </>
          )}
          {cur.key === "launch" && (
            <>
              <InfoRows rows={[
                ["Gates passing", gateCounts.pass],
                ["Warnings", gateCounts.warning],
                ["Blocked", gateCounts.blocked],
                ["Preset", selectedPreset || "none"]
              ]} />
              <div className="setup-actions">
                <button className="primary-action" onClick={() => onSection("launch-plan")}><Rocket size={15} /> Open Launch Plan</button>
                <button className="ghost-button" onClick={() => onSection("services")}><Network size={15} /> Service lifecycle</button>
              </div>
            </>
          )}

          <div className="setup-nav">
            <button className="ghost-button" disabled={active === 0} onClick={() => setActive((value) => Math.max(0, value - 1))}>Back</button>
            <span className="setup-nav-detect">
              {cur.key === "detect" && <>Engine {env?.engine_version ? `vLLM ${env.engine_version}` : "not installed"} · Docker {dockerTool?.present ? "✓" : "—"} · GPU {nvidiaTool?.present ? "✓" : "—"}</>}
            </span>
            <button className="primary-action" disabled={active === steps.length - 1} onClick={() => setActive((value) => Math.min(steps.length - 1, value + 1))}>Next step</button>
          </div>
        </section>
      </div>
    </section>
  );
}
