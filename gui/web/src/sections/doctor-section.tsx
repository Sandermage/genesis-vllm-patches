// SPDX-License-Identifier: Apache-2.0
// The Doctor section: a top-of-panel plain-English verdict line, the aggregated
// diagnostics summary + findings + host caveats, the launch-readiness gates
// list, and patch registry coverage. Extracted from section-workspace.tsx.
import { Activity, AlertTriangle, CheckCircle2, CircleAlert, PackageCheck, ShieldCheck, Stethoscope, TriangleAlert } from "lucide-react";
import { tr } from "../i18n";
import type { DoctorReport, PatchDoctorReport } from "../api";
import type { Gate, SectionId } from "../nav";
import { ModuleCard, ModuleGrid } from "../components/layout";
import { TabbedSection } from "../components/tabbed-section";
import { DoctorFindings, DoctorSummary } from "./doctor";
import { CaveatsPanel } from "./diagnostics";
import { GateRow } from "./gate-row";
import { DoctorCoveragePanel } from "./patch-doctor";

// One plain-English verdict line at the top of the Doctor panel, distilled from
// doctorReport.summary — what a non-operator needs in one glance before the full
// diagnostics. blocked > warning > all-clear, in that priority.
function DoctorVerdict({ report }: { report: DoctorReport | null }) {
  if (!report) {
    return (
      <div className="doctor-verdict info">
        <Activity size={18} /> <strong>{tr("Running diagnostics…")}</strong>
      </div>
    );
  }
  const blocked = report.summary.blocked ?? 0;
  const warning = report.summary.warning ?? 0;
  if (blocked > 0) {
    return (
      <div className="doctor-verdict blocked">
        <CircleAlert size={18} />
        <strong>{blocked} {blocked === 1 ? tr("blocker — fix before launch") : tr("blockers — fix before launch")}</strong>
      </div>
    );
  }
  if (warning > 0) {
    return (
      <div className="doctor-verdict warn">
        <TriangleAlert size={18} />
        <strong>{warning} {warning === 1 ? tr("warning") : tr("warnings")}</strong>
      </div>
    );
  }
  return (
    <div className="doctor-verdict ok">
      <CheckCircle2 size={18} /> <strong>{tr("Serving correctly ✓")}</strong>
    </div>
  );
}

export function DoctorSection({
  doctorReport, gates, patchDoctor, onSection, simpleMode = false
}: {
  doctorReport: DoctorReport | null;
  gates: Gate[];
  patchDoctor: PatchDoctorReport | null;
  onSection: (section: SectionId) => void;
  // In Simple mode the full diagnostics collapse under the verdict line.
  simpleMode?: boolean;
}) {
  return (
    <>
      <DoctorVerdict report={doctorReport} />
      {simpleMode ? (
        <details className="doctor-details-collapse">
          <summary>{tr("Show full diagnostics")}</summary>
          {renderDoctorTabs(doctorReport, gates, patchDoctor, onSection)}
        </details>
      ) : (
        renderDoctorTabs(doctorReport, gates, patchDoctor, onSection)
      )}
    </>
  );
}

function renderDoctorTabs(
  doctorReport: DoctorReport | null,
  gates: Gate[],
  patchDoctor: PatchDoctorReport | null,
  onSection: (section: SectionId) => void
) {
  return (
            <TabbedSection
          id="doctor"
          tabs={[
            {
              id: "diagnostics",
              label: tr("Diagnostics"),
              icon: <Stethoscope size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={tr("Diagnostics Summary")} icon={<Activity size={18} />} desc={tr("Aggregated environment, runtime, catalog, patch and proof health.")} wide>
                    <DoctorSummary report={doctorReport} />
                  </ModuleCard>
                  <ModuleCard title={tr("Findings")} icon={<Stethoscope size={18} />} desc={tr("Grouped by category — expand a row for evidence, action and CLI.")} wide>
                    <DoctorFindings report={doctorReport} />
                  </ModuleCard>
                  <ModuleCard title={tr("Host caveats")} icon={<AlertTriangle size={18} />} desc={tr("Known host-condition issues (kernel, virtualization, GPU, pin) evaluated live against this host — triggered caveats first.")} wide>
                    <CaveatsPanel />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "gates",
              label: tr("Readiness gates"),
              icon: <ShieldCheck size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={tr("Launch Readiness Gates")} icon={<ShieldCheck size={18} />} desc={tr("Per-gate blockers for the selected preset launch.")} wide>
                    <div className="gates-list">
                      {gates.map((gate) => (
                        <GateRow gate={gate} key={gate.id} onNavigate={onSection} />
                      ))}
                    </div>
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "coverage",
              label: tr("Coverage"),
              icon: <PackageCheck size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={tr("Registry Coverage")} icon={<PackageCheck size={18} />} desc={tr("Patch apply-module coverage and validation.")} wide>
                    <DoctorCoveragePanel report={patchDoctor} />
                  </ModuleCard>
                </ModuleGrid>
              )
            }
          ]}
        />
  );
}
