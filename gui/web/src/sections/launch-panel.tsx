// SPDX-License-Identifier: Apache-2.0
// Dedicated launch surface — the hero "set it and launch" screen. Consolidates
// what-will-run, the resolved runtime parameters, preflight readiness and the
// prominent Launch control. Pure presentational (no own state). Extracted from
// App.tsx (modularization).
//
// Enterprise touch over the inline original (classes unchanged): the readiness
// pass/warn/blocked tally is grouped under role="group" + aria-label so screen
// readers announce it as one labelled summary.
import { CircleAlert, CheckCircle2, Rocket, SlidersHorizontal, ListChecks, Play } from "lucide-react";
import { type LaunchPlanEndpoint, type Job } from "../api";
import { type RuntimeMode, type Gate } from "../nav";
import { type GateStatus, InfoRows } from "../components/primitives";
import { CopyButton, CodeBlock } from "../components/code-block";
import { asNumber, asText } from "../lib/coerce";
import { formatTokens } from "../lib/format";
import { JobResultBlock } from "./jobs";

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
  const readinessText = blocked ? "Launch blocked" : warnings ? "Ready — with warnings" : "Ready to launch";
  const command = [
    `sndr launch apply --preset ${selectedPreset || "<preset>"}`,
    ssh ? `  --ssh ${ssh}` : "  # local execution",
    "  --confirm"
  ];
  return (
    <section className="launch-panel">
      <div className="launch-hero">
        <div className="launch-hero-id">
          <span className="launch-hero-kicker">Step 3 · Review &amp; Launch</span>
          <h2>{selectedPreset || "No preset selected"}</h2>
          <p>{model} · {hardware}</p>
        </div>
        <div className={`launch-readiness ${readinessTone}`}>
          {blocked ? <CircleAlert size={18} /> : <CheckCircle2 size={18} />}
          <div>
            <strong>{readinessText}</strong>
            <small>{gateCounts.pass ?? 0}/{totalGates} gates passing</small>
          </div>
        </div>
      </div>

      <div className="launch-grid">
        <div className="launch-main">
          <section className="launch-card">
            <h3><Rocket size={16} /> What will run</h3>
            <InfoRows
              rows={[
                ["Preset", selectedPreset || "-"],
                ["Model", model],
                ["Hardware", hardware],
                ["Profile", profile],
                ["Runtime", runtimeTitle],
                ["Host", host],
                ["Transport", ssh ? `SSH · ${ssh}` : "Local execution"],
                ["Mode", runtimeMode === "remote" ? "Remote desktop" : "Local server"]
              ]}
            />
            <label className="endpoint-field launch-endpoint">
              <span>Serving endpoint (after launch)</span>
              <div>
                <input value={primaryEndpoint} readOnly />
                <CopyButton value={primaryEndpoint} label="endpoint" />
              </div>
            </label>
          </section>

          <section className="launch-card">
            <div className="launch-card-head">
              <h3><SlidersHorizontal size={16} /> Runtime parameters</h3>
              <button className="ghost-button" onClick={onConfigure}><SlidersHorizontal size={14} /> Adjust</button>
            </div>
            <div className="launch-params">
              <LaunchParam label="Max context" value={formatTokens(asNumber(planSummary.context) || asNumber(composed.max_model_len))} />
              <LaunchParam label="Max sequences" value={String(asNumber(planSummary.max_num_seqs) || asNumber(composed.max_num_seqs) || "-")} />
              <LaunchParam label="GPU mem util" value={asText(composed.gpu_memory_utilization, "-")} />
              <LaunchParam label="KV cache" value={asText(composed.kv_cache_dtype, "-")} />
              <LaunchParam label="Spec decode" value={`${asText(composed.spec_decode_method, "-")}/K=${asText(composed.spec_decode_K, "-")}`} />
              <LaunchParam label="Enabled patches" value={String(asNumber(planSummary.enabled_patches_count) || asNumber(composed.enabled_patches_count) || "-")} />
              <LaunchParam label="Patch policy" value={patchPolicy} />
              <LaunchParam label="Fallback" value={asText(planSummary.fallback_preset, asText(card.fallback_preset, "none"))} />
            </div>
          </section>
        </div>

        <aside className="launch-rail">
          <section className="launch-card">
            <div className="launch-card-head">
              <h3><ListChecks size={16} /> Readiness</h3>
              <button className="ghost-button" onClick={onViewGates}>All gates</button>
            </div>
            <div className="readiness-counts" role="group" aria-label="Gate readiness summary">
              <span className="rc ok">{gateCounts.pass ?? 0} pass</span>
              <span className="rc warn">{warnings} warn</span>
              <span className="rc bad">{blockers.length} blocked</span>
            </div>
            {blocked ? (
              <ul className="blocker-list">
                {blockers.map((gate) => (
                  <li key={gate.id}><CircleAlert size={13} /> {gate.label}</li>
                ))}
              </ul>
            ) : (
              <p className="muted">No blockers — preflight clear.</p>
            )}
          </section>

          <section className="launch-card launch-action">
            <h3><Play size={16} /> Launch</h3>
            {applyEnabled ? (
              <>
                <label className="service-confirm launch-confirm">
                  <input type="checkbox" checked={launchConfirm} onChange={(event) => setLaunchConfirm(event.target.checked)} />
                  <span>Confirm — start <strong>{selectedPreset}</strong> now</span>
                </label>
                <button className="launch-go" disabled={launchBusy || !launchConfirm} onClick={onLaunch}>
                  <Play size={17} />
                  {launchBusy ? "Launching…" : "Launch model"}
                </button>
                <p className="muted">
                  {launchConfirm
                    ? "Starts the runtime for this preset over the selected transport."
                    : "Tick confirm — this is a mutating action."}
                </p>
              </>
            ) : (
              <>
                <button className="launch-go disabled" disabled>
                  <Play size={17} /> Launch (read-only)
                </button>
                <p className="muted">{actionReason ?? "Read-only daemon. Start it with --enable-apply to launch from the GUI."}</p>
              </>
            )}
            <details className="launch-cmd">
              <summary>Equivalent CLI command</summary>
              <CodeBlock lines={command} />
            </details>
            {launchJob && <JobResultBlock job={launchJob} />}
          </section>
        </aside>
      </div>
    </section>
  );
}
