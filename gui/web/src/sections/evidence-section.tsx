// SPDX-License-Identifier: Apache-2.0
// The Evidence section: release-gate proof status across the patch catalog, plus
// the selected preset's evidence references with visibility/type coverage and a
// queue-a-collection job. Extracted from section-workspace.tsx.
import { FileText, ShieldCheck, SquareTerminal } from "lucide-react";
import { tr } from "../i18n";
import { asRecord, asText } from "../lib/coerce";
import { api, type ProofStatusReport } from "../api";
import { TabbedSection } from "../components/tabbed-section";
import { ModuleCard, ModuleGrid } from "../components/layout";
import { CodeBlock } from "../components/code-block";
import { PercentBar } from "../components/charts";
import { CompactList } from "../components/primitives";
import { EvidenceRows } from "./bench";
import { ProofStatusPanel } from "./proof";
import { QueueJobButton } from "./jobs";

export function EvidenceSection({
  card,
  proofStatus,
  selectedPreset,
  onMonitorJob
}: {
  card: Record<string, unknown>;
  proofStatus: ProofStatusReport | null;
  selectedPreset: string;
  onMonitorJob: (id: string) => void;
}) {
          const refs = (Array.isArray(card.evidence_refs) ? card.evidence_refs : []).map(asRecord);
        const byVisibility = refs.reduce<Record<string, number>>((acc, ref) => {
          const key = asText(ref.visibility, "unknown");
          acc[key] = (acc[key] ?? 0) + 1;
          return acc;
        }, {});
        const byType = refs.reduce<Record<string, number>>((acc, ref) => {
          const key = asText(ref.type, "evidence");
          acc[key] = (acc[key] ?? 0) + 1;
          return acc;
        }, {});
        return (
        <TabbedSection
          id="evidence"
          tabs={[
            {
              id: "proof",
              label: tr("Proof status"),
              icon: <ShieldCheck size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={tr("Proof artifact status")} icon={<ShieldCheck size={18} />} desc={tr("Release-gate evidence across the whole patch catalog — every patch bucketed by the strongest proof it carries (measured baseline → bench attached → static-only → failed → dead), with family / tier / lifecycle breakdowns.")} wide>
                    <ProofStatusPanel report={proofStatus} />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "collect",
              label: tr("Collect & coverage"),
              icon: <SquareTerminal size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={`${tr("Preset evidence")} · ${selectedPreset}`} icon={<FileText size={18} />} desc={tr("Evidence references the selected preset card exposes, and how they break down by visibility and type.")} wide>
                    <EvidenceRows card={card} />
                    {refs.length > 0 && (
                      <div className="evidence-coverage" style={{ marginTop: "var(--sp-3)" }}>
                        <PercentBar
                          value={byVisibility.public ?? 0}
                          max={refs.length}
                          label={tr("public refs")}
                          caption={`${byVisibility.public ?? 0} ${tr("of")} ${refs.length} ${refs.length === 1 ? tr("reference public") : tr("references public")}`}
                          tone={byVisibility.public ? "ok" : "warn"}
                        />
                        <CompactList
                          rows={[
                            ...Object.entries(byType).map(([k, v]) => [k, String(v)] as [string, string]),
                            ...Object.entries(byVisibility).map(([k, v]) => [`${tr("visibility")}: ${k}`, String(v)] as [string, string])
                          ]}
                        />
                      </div>
                    )}
                  </ModuleCard>
                  <ModuleCard title={tr("Collect & attach evidence")} icon={<SquareTerminal size={18} />} desc={tr("Queue a dry-run evidence-collection job for this preset, or copy the exact CLI to run on the rig where the engine lives.")} wide>
                    <QueueJobButton
                      label={`${tr("Queue evidence")} (${selectedPreset})`}
                      run={() => api.evidenceAttach({ preset_id: selectedPreset })}
                      onMonitor={onMonitorJob}
                    />
                    <CodeBlock lines={[`sndr evidence collect --preset ${selectedPreset}`, "sndr patches bench-attach <PATCH_ID> bench.json --baseline baseline.json", "sndr patches release-check --mode require-bench"]} />
                  </ModuleCard>
                </ModuleGrid>
              )
            }
          ]}
        />
        );
}
