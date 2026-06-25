// SPDX-License-Identifier: Apache-2.0
// Dedicated launch surface — the hero "set it and launch" screen. Consolidates
// what-will-run, the resolved runtime parameters, preflight readiness and the
// prominent Launch control. Pure presentational (no own state).
import { CircleAlert, CheckCircle2, Rocket, SlidersHorizontal, ListChecks, Play } from "lucide-react";
import { type LaunchPlanEndpoint, type Job } from "../api";
import { type RuntimeMode, type Gate } from "../nav";
import { type GateStatus, InfoRows } from "../components/primitives";
import { CopyButton, CodeBlock } from "../components/code-block";
import { asNumber, asText } from "../lib/coerce";
import { formatTokens } from "../lib/format";
import { JobResultBlock } from "./jobs";
import { tr } from "../i18n";

function LaunchParam({ label, value }: { label: string; value: string }) {
  return (
    <div className="launch-param">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export function LaunchPanel({
  selectedPreset,
  model,
  hardware,
  profile,
  host,
  composed,
  planSummary,
  card,
  patchPolicy,
  runtimeTitle,
  runtimeMode,
  endpoints,
  gates,
  gateCounts,
  applyEnabled,
  actionReason,
  launchConfirm,
  setLaunchConfirm,
  launchBusy,
  launchSshTarget,
  launchJob,
  onLaunch,
  onConfigure,
  onViewGates
}: {
  selectedPreset: string;
  model: string;
  hardware: string;
  profile: string;
  host: string;
  composed: Record<string, unknown>;
  planSummary: Record<string, unknown>;
  card: Record<string, unknown>;
  patchPolicy: string;
  runtimeTitle: string;
  runtimeMode: RuntimeMode;
  endpoints?: LaunchPlanEndpoint[];
  gates: Gate[];
  gateCounts: Record<GateStatus, number>;
  applyEnabled: boolean;
  actionReason?: string;
  launchConfirm: boolean;
  setLaunchConfirm: (value: boolean) => void;
  launchBusy: boolean;
  launchSshTarget: string;
  launchJob: Job | null;
  onLaunch: () => void;
  onConfigure: () => void;
  onViewGates: () => void;
}) {
  const ssh = launchSshTarget.trim();
  const blockers = gates.filter((gate) => gate.status === "blocked");
  const warnings = gateCounts.warning ?? 0;
  const blocked = blockers.length > 0;
  const totalGates =
    (gateCounts.pass ?? 0) + (gateCounts.warning ?? 0) + (gateCounts.blocked ?? 0) + (gateCounts.planned ?? 0);
  const primaryEndpoint =
    endpoints?.find((endpoint) => /openai|api/i.test(endpoint.label))?.url ??
    endpoints?.[0]?.url ??
    `http://${host}:8000/v1`;
  const readinessTone = blocked ? "blocked" : warnings ? "warn" : "ready";
  const readinessText = blocked ? tr("Launch blocked") : warnings ? tr("Ready — with warnings") : tr("Ready to launch");
  const command = [
    `sndr launch apply --preset ${selectedPreset || "<preset>"}`,
    ssh ? `  --ssh ${ssh}` : "  # local execution",
    "  --confirm"
  ];
  return (
    <section className="launch-panel">
      <div className="launch-hero">
        <div className="launch-hero-id">
          <span className="launch-hero-kicker">{tr("Step 3 · Review & Launch")}</span>
          <h2>{selectedPreset || tr("No preset selected")}</h2>
          <p>{model} · {hardware}</p>
        </div>
        <div className={`launch-readiness ${readinessTone}`}>
          {blocked ? <CircleAlert size={18} /> : <CheckCircle2 size={18} />}
          <div>
            <strong>{readinessText}</strong>
            <small>{gateCounts.pass ?? 0}/{totalGates} {tr("gates passing")}</small>
          </div>
        </div>
      </div>

      <div className="launch-grid">
        <div className="launch-main">
          <section className="launch-card">
            <h3><Rocket size={16} /> {tr("What will run")}</h3>
            <InfoRows
              rows={[
                [tr("Preset"), selectedPreset || "-"],
                [tr("Model"), model],
                [tr("Hardware"), hardware],
                [tr("Profile"), profile],
                [tr("Runtime"), runtimeTitle],
                [tr("Host"), host],
                [tr("Transport"), ssh ? `SSH · ${ssh}` : tr("Local execution")],
                [tr("Mode"), runtimeMode === "remote" ? tr("Remote desktop") : tr("Local server")]
              ]}
            />
            <label className="endpoint-field launch-endpoint">
              <span>{tr("Serving endpoint (after launch)")}</span>
              <div>
                <input value={primaryEndpoint} readOnly />
                <CopyButton value={primaryEndpoint} label={tr("endpoint")} />
              </div>
            </label>
          </section>

          <section className="launch-card">
            <div className="launch-card-head">
              <h3><SlidersHorizontal size={16} /> {tr("Runtime parameters")}</h3>
              <button className="ghost-button" onClick={onConfigure}><SlidersHorizontal size={14} /> {tr("Adjust")}</button>
            </div>
            <div className="launch-params">
              <LaunchParam label={tr("Max context")} value={formatTokens(asNumber(planSummary.context) || asNumber(composed.max_model_len))} />
              <LaunchParam label={tr("Max sequences")} value={String(asNumber(planSummary.max_num_seqs) || asNumber(composed.max_num_seqs) || "-")} />
              <LaunchParam label={tr("GPU mem util")} value={asText(composed.gpu_memory_utilization, "-")} />
              <LaunchParam label={tr("KV cache")} value={asText(composed.kv_cache_dtype, "-")} />
              <LaunchParam label={tr("Spec decode")} value={`${asText(composed.spec_decode_method, "-")}/K=${asText(composed.spec_decode_K, "-")}`} />
              <LaunchParam label={tr("Enabled patches")} value={String(asNumber(planSummary.enabled_patches_count) || asNumber(composed.enabled_patches_count) || "-")} />
              <LaunchParam label={tr("Patch policy")} value={patchPolicy} />
              <LaunchParam label={tr("Fallback")} value={asText(planSummary.fallback_preset, asText(card.fallback_preset, tr("none")))} />
              <LaunchParam label={tr("Evidence")} value={asText(planSummary.evidence_visibility, "-")} />
              <LaunchParam label={tr("Primary metric")} value={planSummary.primary_metric_value != null ? `${asText(planSummary.primary_metric_kind, "")} ${asNumber(planSummary.primary_metric_value)}`.trim() : "-"} />
              <LaunchParam label={tr("Patch coverage")} value={asNumber(planSummary.patch_coverage_total) ? `${asNumber(planSummary.patch_coverage_mapped)}/${asNumber(planSummary.patch_coverage_total)}` : "-"} />
            </div>
          </section>
        </div>

        <aside className="launch-rail">
          <section className="launch-card">
            <div className="launch-card-head">
              <h3><ListChecks size={16} /> {tr("Readiness")}</h3>
              <button className="ghost-button" onClick={onViewGates}>{tr("All gates")}</button>
            </div>
            <div className="readiness-counts" role="group" aria-label={tr("Gate readiness summary")}>
              <span className="rc ok">{gateCounts.pass ?? 0} {tr("pass")}</span>
              <span className="rc warn">{warnings} {tr("warn")}</span>
              <span className="rc bad">{blockers.length} {tr("blocked")}</span>
            </div>
            {blocked ? (
              <ul className="blocker-list">
                {blockers.map((gate) => (
                  <li key={gate.id}><CircleAlert size={13} /> {gate.label}</li>
                ))}
              </ul>
            ) : (
              <p className="muted">{tr("No blockers — preflight clear.")}</p>
            )}
          </section>

          <section className="launch-card launch-action">
            <h3><Play size={16} /> {tr("Launch")}</h3>
            {applyEnabled ? (
              <>
                <label className="service-confirm launch-confirm">
                  <input type="checkbox" checked={launchConfirm} onChange={(event) => setLaunchConfirm(event.target.checked)} />
                  <span>{tr("Confirm — start")} <strong>{selectedPreset}</strong> {tr("now")}</span>
                </label>
                <button className="launch-go" disabled={launchBusy || !launchConfirm} onClick={onLaunch}>
                  <Play size={17} />
                  {launchBusy ? tr("Launching…") : tr("Launch model")}
                </button>
                <p className="muted">
                  {launchConfirm
                    ? tr("Starts the runtime for this preset over the selected transport.")
                    : tr("Tick confirm — this is a mutating action.")}
                </p>
              </>
            ) : (
              <>
                <button className="launch-go disabled" disabled>
                  <Play size={17} /> {tr("Launch (read-only)")}
                </button>
                <p className="muted">{actionReason ?? tr("Read-only daemon. Start it with --enable-apply to launch from the GUI.")}</p>
              </>
            )}
            <details className="launch-cmd">
              <summary>{tr("Equivalent CLI command")}</summary>
              <CodeBlock lines={command} />
            </details>
            {launchJob && <JobResultBlock job={launchJob} />}
          </section>
        </aside>
      </div>
    </section>
  );
}
