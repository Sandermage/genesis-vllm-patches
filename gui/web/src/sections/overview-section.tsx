// SPDX-License-Identifier: Apache-2.0
// The Overview section: a tabbed system map — KPI hero + status snapshot cards
// (Summary), runtime environment/targets (Environment), and workload/family
// coverage (Coverage). Derives its small catalog rollups (bench-proven count,
// family/workload counts, patch rows) from props. Extracted from
// section-workspace.tsx.
import {
  Activity, Box, Cpu, Database, FileText, GitBranch, Layers3, Monitor,
  Rocket, Route, Server, ShieldCheck, Wrench
} from "lucide-react";
import { tr } from "../i18n";
import { asNumber, asRecord } from "../lib/coerce";
import { targetTitle } from "../lib/format";
import type {
  ProductOverview, PresetListResult, PatchListResult, HostProfile, DoctorReport,
  EnvironmentReport, ProductCapability
} from "../api";
import type { Gate, RuntimeMode, SectionId } from "../nav";
import type { ViewportTier } from "../hooks/useViewport";
import type { GuiSettings } from "../settings";
import { TabbedSection } from "../components/tabbed-section";
import { ModuleCard, ModuleGrid } from "../components/layout";
import { OvKpi, PercentBar } from "../components/charts";
import { CompactList, InfoRows, KpiGrid, type GateStatus } from "../components/primitives";
import { CapabilityTable } from "../components/capability-table";
import { ConnectionMap } from "./connection-bar";
import { EnvironmentPanel } from "./environment";

export function OverviewSection({
  overview,
  presets,
  patches,
  hostProfiles,
  runtimeMode,
  doctorReport,
  environment,
  viewport,
  settings,
  runtimeTarget,
  selectedPreset,
  apiBase,
  gateCounts,
  gates,
  runtimeTargets,
  patchPolicy,
  featureRows,
  onSection
}: {
  overview: ProductOverview | null;
  presets: PresetListResult | null;
  patches: PatchListResult | null;
  hostProfiles: HostProfile[];
  runtimeMode: RuntimeMode;
  doctorReport: DoctorReport | null;
  environment: EnvironmentReport | null;
  viewport: ViewportTier;
  settings: GuiSettings;
  runtimeTarget: string;
  selectedPreset: string;
  apiBase: string;
  gateCounts: Record<GateStatus, number>;
  gates: Gate[];
  runtimeTargets: ProductCapability[];
  patchPolicy: string;
  featureRows: ProductCapability[];
  onSection: (section: SectionId) => void;
}) {
  const benchProven = (presets?.presets ?? []).filter(
    (p) => asNumber(asRecord(p.card?.primary_metric).value) > 0
  ).length;
  const familyCounts = overview?.catalog.family_counts ?? {};
  const workloadCounts = overview?.catalog.workload_counts ?? {};
  const patchRows = patches?.patches ?? [];
  return (
            <TabbedSection
          id="overview"
          tabs={[
            {
              id: "summary",
              label: tr("Summary"),
              icon: <Activity size={15} />,
              render: () => (
                <>
                <div className="ov-hero">
                  <OvKpi icon={<Database size={15} />} label={tr("Presets")} value={overview?.catalog.presets_count ?? "—"} sub={`${benchProven} ${tr("bench-proven")}`} onClick={() => onSection("presets")} />
                  <OvKpi icon={<Box size={15} />} label={tr("Models")} value={overview?.catalog.models_count ?? "—"} sub={`${Object.keys(familyCounts).length} ${tr("families")}`} onClick={() => onSection("models")} />
                  <OvKpi icon={<Wrench size={15} />} label={tr("Patches")} value={patchRows.length || patches?.total || "—"} sub={`${patchRows.filter((p) => p.default_on).length} ${tr("default-on")}`} onClick={() => onSection("patches")} />
                  <OvKpi icon={<Server size={15} />} label={tr("Hosts")} value={hostProfiles.length} sub={runtimeMode === "remote" ? tr("remote + local") : tr("local fleet")} onClick={() => onSection("hosts")} />
                  <OvKpi icon={<ShieldCheck size={15} />} label={tr("Doctor")} value={doctorReport ? (doctorReport.findings.length ? `${doctorReport.findings.length} ${tr("findings")}` : tr("clean")) : "—"} sub={doctorReport && doctorReport.findings.length ? `${doctorReport.findings.filter((f) => f.severity === "blocked").length} ${tr("blocked")} · ${doctorReport.findings.filter((f) => f.severity === "warning").length} ${tr("warn")}` : undefined} tone={doctorReport?.findings?.some((f) => f.severity === "blocked") ? "warn" : "ok"} onClick={() => onSection("doctor")} />
                  <OvKpi icon={<Rocket size={15} />} label={tr("Engine")} value={environment?.engine_installed ? tr("ready") : "—"} sub={environment?.engine_installed ? `${environment.engine_name ?? "vLLM"} ${environment.engine_version ?? ""}`.trim() : tr("not installed")} tone={environment?.engine_installed ? "ok" : undefined} onClick={() => onSection("services")} />
                  {viewport === "ultra" && (
                    <>
                      <OvKpi icon={<GitBranch size={15} />} label={tr("Profiles")} value={overview?.catalog.profiles_count ?? "—"} sub={tr("runtime recipes")} onClick={() => onSection("configs")} />
                      <OvKpi icon={<Cpu size={15} />} label={tr("Hardware")} value={overview?.catalog.hardware_count ?? "—"} sub={tr("defined targets")} />
                      <OvKpi icon={<FileText size={15} />} label={tr("Preset cards")} value={overview?.catalog.preset_cards_count ?? "—"} sub={`${overview?.catalog.unannotated_presets_count ?? 0} ${tr("unannotated")}`} onClick={() => onSection("presets")} />
                    </>
                  )}
                </div>
                {settings.showConnectionMap && (
                  <ModuleGrid>
                    <ModuleCard title={tr("Control Plane Connections")} icon={<Route size={18} />} wide>
                      <ConnectionMap
                        runtimeMode={runtimeMode}
                        runtimeTarget={runtimeTarget}
                        selectedPreset={selectedPreset}
                        patchCount={patchRows.length}
                        apiBase={apiBase}
                      />
                    </ModuleCard>
                  </ModuleGrid>
                )}
                {/* Status cards live in their own grid (no full-width sibling, which
                    would defeat auto-fit's track collapsing and strand whitespace). */}
                <ModuleGrid className="ov-status-grid">
                  <ModuleCard title={tr("Platform Snapshot")} icon={<Monitor size={18} />}>
                    <InfoRows
                      rows={[
                        [tr("Brand"), overview?.capabilities.platform.public_brand ?? "-"],
                        [tr("Package"), overview?.capabilities.platform.package_name ?? "-"],
                        [tr("Version"), overview?.capabilities.platform.sndr_core_version ?? "-"],
                        [tr("OS"), `${overview?.capabilities.platform.os_name ?? "-"} / ${overview?.capabilities.platform.machine ?? "-"}`],
                        [tr("Python"), overview?.capabilities.platform.python_version ?? "-"]
                      ]}
                    />
                  </ModuleCard>
                  <ModuleCard title={tr("Catalog Health")} icon={<Database size={18} />} desc={tr("Annotation coverage and load integrity — not raw counts (those are above).")}>
                    <PercentBar
                      value={overview?.catalog.preset_cards_count ?? 0}
                      max={overview?.catalog.presets_count || 1}
                      label={tr("card coverage")}
                      caption={`${overview?.catalog.preset_cards_count ?? 0}/${overview?.catalog.presets_count ?? 0} ${tr("presets annotated")}`}
                      tone={(overview?.catalog.preset_load_error_count ?? 0) > 0 ? "warn" : "ok"}
                    />
                    <KpiGrid
                      rows={[
                        [tr("Bench-proven"), benchProven],
                        [tr("Families"), Object.keys(familyCounts).length],
                        [tr("Unannotated"), overview?.catalog.unannotated_presets_count ?? 0],
                        [tr("Load errors"), overview?.catalog.preset_load_error_count ?? 0]
                      ]}
                    />
                  </ModuleCard>
                  <ModuleCard title={tr("Launch Readiness")} icon={<ShieldCheck size={18} />} desc={tr("Gate verdict for the selected preset launch.")}>
                    <PercentBar
                      value={gateCounts.pass}
                      max={gates.length || 1}
                      label={tr("gates passing")}
                      caption={`${gateCounts.pass} ${tr("ok")} · ${gateCounts.warning} ${tr("warn")} · ${gateCounts.blocked} ${tr("blocked")}`}
                      tone={gateCounts.blocked > 0 ? "warn" : "ok"}
                    />
                    <InfoRows
                      rows={[
                        [tr("Selected preset"), selectedPreset],
                        [tr("Runtime target"), targetTitle(runtimeTargets, runtimeTarget)],
                        [tr("Mode"), runtimeMode === "remote" ? tr("Remote (SSH tunnel)") : tr("Local server")],
                        [tr("Patch policy"), patchPolicy]
                      ]}
                    />
                  </ModuleCard>
                  <ModuleCard title={tr("Engine & API")} icon={<Cpu size={18} />} desc={tr("The inference engine and the API surface it exposes (core/OS live in Platform Snapshot).")}>
                    <InfoRows
                      rows={[
                        [tr("Engine"), `${environment?.engine_name ?? "vLLM"} ${environment?.engine_version ?? ""}`.trim() || "vLLM"],
                        [tr("Installed"), environment?.engine_installed ? tr("yes") : tr("not installed")],
                        [tr("Runtime targets"), `${runtimeTargets.length} ${tr("available")}`],
                        [tr("Capabilities"), `${featureRows.length} ${tr("features")}`],
                        [tr("OpenAI API"), environment?.engine_installed ? tr("ready") : tr("engine off")]
                      ]}
                    />
                  </ModuleCard>
                </ModuleGrid>
                </>
              )
            },
            {
              id: "environment",
              label: tr("Environment"),
              icon: <Cpu size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={tr("Runtime Environment")} icon={<Cpu size={18} />} desc={tr("Project version, engine version and the installed dependency stack.")} wide>
                    <EnvironmentPanel env={environment} />
                  </ModuleCard>
                  <ModuleCard title={tr("Runtime Targets")} icon={<Server size={18} />} wide>
                    <CapabilityTable rows={runtimeTargets} />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "coverage",
              label: tr("Coverage"),
              icon: <Layers3 size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={tr("Workload Coverage")} icon={<Layers3 size={18} />}>
                    <CompactList rows={Object.entries(workloadCounts).map(([key, value]) => [key, String(value)])} />
                  </ModuleCard>
                  <ModuleCard title={tr("Family Coverage")} icon={<Box size={18} />}>
                    <CompactList rows={Object.entries(familyCounts).map(([key, value]) => [key, String(value)])} />
                  </ModuleCard>
                </ModuleGrid>
              )
            }
          ]}
        />
  );
}
