// SPDX-License-Identifier: Apache-2.0
// Service lifecycle planner — pick an action (status/logs/start/restart/stop) +
// runtime target, review the resolved plan + gates, then run it as a dry-run or
// (when the daemon enables apply, with explicit confirm for mutating actions)
// execute it locally or over SSH. Extracted from App.tsx (modularization).
//
// Enterprise hardening over the inline original (markup classes unchanged):
//   * the action segmented control is a role="group" whose buttons expose
//     aria-pressed, so assistive tech announces which action is selected.
import { useEffect, useState } from "react";
import { Activity, AlertCircle, CircleAlert, RefreshCw, Play } from "lucide-react";
import { api, type ServiceActionPlan, type Job, type EngineStatus } from "../api";
import { type GateStatus, StatusPill } from "../components/primitives";
import { CopyButton } from "../components/code-block";
import { GateRow } from "./gate-row";
import { JobResultBlock } from "./jobs";

const SERVICE_RUNTIME_TARGETS: Array<{ id: string; label: string }> = [
  { id: "docker_compose", label: "Docker Compose" },
  { id: "docker", label: "Docker" },
  { id: "systemd", label: "systemd" },
  { id: "podman", label: "Podman" },
  { id: "quadlet", label: "Quadlet" },
  { id: "kubernetes", label: "Kubernetes" }
];

const SERVICE_ACTIONS = ["status", "logs", "start", "restart", "stop"];

export function ServiceLifecyclePlanner({
  selectedPreset,
  runtimeTarget,
  host
}: {
  selectedPreset: string;
  runtimeTarget: string;
  host: string;
}) {
  const [action, setAction] = useState("status");
  const [target, setTarget] = useState(runtimeTarget);
  const [plan, setPlan] = useState<ServiceActionPlan | null>(null);
  const [state, setState] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [applying, setApplying] = useState(false);
  const [applyEnabled, setApplyEnabled] = useState(false);
  const [sshTarget, setSshTarget] = useState("");
  const [confirm, setConfirm] = useState(false);
  const [engine, setEngine] = useState<EngineStatus | null>(null);
  const [engineChecking, setEngineChecking] = useState(false);

  // Re-seed the target when the active preset's default runtime changes.
  useEffect(() => { setTarget(runtimeTarget); }, [runtimeTarget]);

  async function checkEngine() {
    setEngineChecking(true);
    try {
      setEngine(await api.engineStatus(host));
    } catch {
      setEngine(null);
    } finally {
      setEngineChecking(false);
    }
  }

  useEffect(() => {
    let cancelled = false;
    api.authStatus().then((s) => { if (!cancelled) setApplyEnabled(s.apply_enabled); }).catch(() => {});
    void checkEngine();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [host]);

  const mutating = plan?.mutating ?? false;
  const transport = sshTarget.trim() ? "ssh" : "local";
  // Execution is allowed when the daemon enables apply; mutating actions also
  // need an explicit confirm. Otherwise the action is recorded as a dry-run.
  const willExecute = applyEnabled;
  const blockedMutation = applyEnabled && mutating && !confirm;

  async function runApply() {
    setApplying(true);
    setError(null);
    try {
      const result = await api.serviceApply({
        preset_id: selectedPreset,
        action,
        runtime_target: target,
        host,
        transport,
        ssh_target: sshTarget.trim(),
        confirm
      });
      setJob(result);
      // After a mutating local action, re-probe the engine so the operator
      // immediately sees whether it actually came up / went down.
      if (mutating && transport === "local") {
        window.setTimeout(() => void checkEngine(), 1500);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setApplying(false);
    }
  }

  useEffect(() => {
    if (!selectedPreset) return;
    setJob(null);
    setConfirm(false);
    let cancelled = false;
    setState("loading");
    setError(null);
    api.servicePlan({ preset_id: selectedPreset, action, runtime_target: target, host })
      .then((result) => {
        if (cancelled) return;
        setPlan(result);
        setState("ready");
      })
      .catch((err) => {
        if (cancelled) return;
        setPlan(null);
        setState("error");
        setError(err instanceof Error ? err.message : String(err));
      });
    return () => { cancelled = true; };
  }, [selectedPreset, action, target, host]);

  const engineTone = engine?.reachable ? "ok" : "danger";
  const engineLabel = engineChecking
    ? "checking…"
    : engine?.reachable
      ? `engine up${engine.version ? ` · v${engine.version}` : ""}`
      : "engine down";

  return (
    <div className="service-planner">
      <div className="service-toolbar">
        <div className="settings-segmented service-actions" role="group" aria-label="Service action">
          {SERVICE_ACTIONS.map((item) => (
            <button key={item} className={action === item ? "active" : ""} aria-pressed={action === item} onClick={() => setAction(item)}>{item}</button>
          ))}
        </div>
        <label className="service-runtime">
          <span>Runtime</span>
          <select value={target} onChange={(event) => setTarget(event.target.value)}>
            {SERVICE_RUNTIME_TARGETS.map((rt) => <option key={rt.id} value={rt.id}>{rt.label}</option>)}
          </select>
        </label>
        <span className="service-target">{host}</span>
      </div>

      <div className="service-engine-strip">
        <span className={`fleet-status ${engineTone}`}><span className="fleet-dot" />{engineLabel}</span>
        {engine?.reachable && engine.models.length > 0 && <span className="service-engine-models">serving {engine.models.join(", ")}</span>}
        {engine && !engine.reachable && engine.error && <span className="service-engine-err">{engine.error}</span>}
        <button className="ghost-button" onClick={() => void checkEngine()} disabled={engineChecking}>
          <Activity size={14} /> {engineChecking ? "Checking…" : "Re-check engine"}
        </button>
      </div>

      {error && <div className="config-plan-error"><AlertCircle size={15} /><span>{error}</span></div>}

      {plan && (
        <>
          <div className="service-plan-head">
            <div>
              <strong>{plan.container_name}</strong>
              <span>{plan.plan_id}</span>
            </div>
            <StatusPill tone={plan.mutating ? "warning" : "success"}>
              {plan.mutating ? "mutating" : "read-only"}
            </StatusPill>
          </div>

          <div className="service-steps">
            {plan.steps.map((step) => (
              <div className="service-step" key={step.order}>
                <span className="service-step-num">{step.order}</span>
                <div>
                  <small>{step.title}</small>
                  <code>{step.command}</code>
                </div>
                <CopyButton value={step.command} label={step.title} />
              </div>
            ))}
          </div>

          {plan.side_effects.length > 0 && (
            <div className="service-effects">
              <strong>Side effects</strong>
              {plan.side_effects.map((effect, index) => (
                <p key={index}><CircleAlert size={13} /> {effect}</p>
              ))}
            </div>
          )}

          <div className="gates-list">
            {plan.gates.map((gate) => (
              <GateRow
                key={gate.id}
                gate={{ id: gate.id, label: gate.title, detail: gate.detail, status: gate.status as GateStatus, action: "Inspect" }}
              />
            ))}
          </div>

          <div className="service-exec-row">
            <label className="service-ssh">
              <span>SSH target (empty = local)</span>
              <input
                value={sshTarget}
                onChange={(event) => setSshTarget(event.target.value)}
                placeholder="user@gpu-host"
              />
            </label>
            {willExecute && mutating && (
              <label className="service-confirm">
                <input type="checkbox" checked={confirm} onChange={(event) => setConfirm(event.target.checked)} />
                <span>Confirm — execute this mutating action on {sshTarget.trim() || "localhost"}</span>
              </label>
            )}
          </div>

          <div className="service-footer">
            <div className="service-rollback"><RefreshCw size={13} /> {plan.rollback}</div>
            <button
              className="primary-action"
              onClick={() => void runApply()}
              disabled={applying || blockedMutation}
            >
              <Play size={15} />{" "}
              {applying
                ? willExecute ? "Executing…" : "Recording…"
                : willExecute
                  ? (mutating ? `Execute ${action} (${transport})` : `Run ${action} (${transport})`)
                  : "Apply (dry-run)"}
            </button>
          </div>
          <p className="service-reason">
            {willExecute
              ? blockedMutation
                ? "Tick confirm to execute this mutating action."
                : `Apply enabled — ${mutating ? "mutating" : "read-only"} action runs over ${transport}.`
              : plan.action_reason}
          </p>

          {job && <JobResultBlock job={job} showNote />}
        </>
      )}
      {state === "loading" && !plan && <p className="muted">Planning…</p>}
    </div>
  );
}
