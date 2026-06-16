// SPDX-License-Identifier: Apache-2.0
// The Reports section: generate a redacted snapshot bundle (preset/gates/patches/
// proof) for hand-off, with a live capture summary, plus a session activity log.
// Extracted from section-workspace.tsx.
import { Clock3, FileText, Table2 } from "lucide-react";
import { tr } from "../i18n";
import { targetTitle } from "../lib/format";
import { asText } from "../lib/coerce";
import type { ProductCapability, ProofStatusReport } from "../api";
import type { GateStatus } from "../components/primitives";
import { TabbedSection } from "../components/tabbed-section";
import { ModuleCard, ModuleGrid } from "../components/layout";
import { CompactList, InfoRows } from "../components/primitives";
import { ReportGenerator } from "./api-explorer";
import { EventLog } from "./operational-console";

export function ReportsSection({
  card, selectedPreset, runtimeTargets, runtimeTarget, patchPolicy, gateCounts, proofStatus, events
}: {
  card: Record<string, unknown>;
  selectedPreset: string;
  runtimeTargets: ProductCapability[];
  runtimeTarget: string;
  patchPolicy: string;
  gateCounts: Record<GateStatus, number>;
  proofStatus: ProofStatusReport | null;
  events: Array<[string, string, string]>;
}) {
  return (
            <TabbedSection
          id="reports"
          tabs={[
            {
              id: "generate",
              label: tr("Generate"),
              icon: <Table2 size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={tr("Generate a report")} icon={<Table2 size={18} />} desc={tr("Capture a redacted snapshot bundle (preset, gates, patches, proof) into the operator-local reports dir — a shareable hand-off / sign-off artifact.")} wide>
                    <ReportGenerator selectedPreset={selectedPreset} />
                  </ModuleCard>
                  <ModuleCard title={tr("What this snapshot captures")} icon={<FileText size={18} />} desc={tr("The live state baked into a report generated right now.")} wide>
                    <InfoRows
                      rows={[
                        [tr("Preset"), selectedPreset],
                        [tr("Runtime target"), targetTitle(runtimeTargets, runtimeTarget)],
                        [tr("Patch policy"), patchPolicy],
                        [tr("Readiness gates"), `${gateCounts.pass} ${tr("ok")} / ${gateCounts.warning} ${tr("warn")} / ${gateCounts.blocked} ${tr("blocked")}`],
                        [tr("Proof artifacts"), proofStatus?.available ? String(proofStatus.total) : tr("unavailable")],
                        [tr("Evidence visibility"), asText(card.evidence_visibility, "-")]
                      ]}
                    />
                    <CompactList
                      rows={[
                        ["HTML", tr("Shareable operator review page")],
                        ["PDF", tr("Archival / sign-off document")],
                        ["JSON", tr("Machine-readable snapshot")],
                        ["Markdown", tr("Inline notes and runbooks")]
                      ]}
                    />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "activity",
              label: tr("Activity log"),
              icon: <Clock3 size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={tr("Recent activity")} icon={<Clock3 size={18} />} desc={tr("A live feed of control-center actions this session — what was selected, planned, queued or applied.")} wide>
                    <EventLog events={events} />
                  </ModuleCard>
                </ModuleGrid>
              )
            }
          ]}
        />
  );
}
