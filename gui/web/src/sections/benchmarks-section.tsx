// SPDX-License-Identifier: Apache-2.0
// The Benchmarks section: baseline metric + capability status, live engine
// micro-bench (A/B), proof coverage, and a queue-a-run plan. Extracted from
// section-workspace.tsx.
import { Activity, BarChart3, Play, ShieldCheck, SquareTerminal, TimerReset } from "lucide-react";
import { tr } from "../i18n";
import { asNumber, asRecord } from "../lib/coerce";
import { api, type ProductCapability, type PresetRecord, type ProofStatusReport } from "../api";
import { TabbedSection } from "../components/tabbed-section";
import { ModuleCard, ModuleGrid } from "../components/layout";
import { CodeBlock } from "../components/code-block";
import { CapabilityTable } from "../components/capability-table";
import { WorkflowSteps } from "../components/shell-bits";
import { EngineStatusCard, EngineBenchPanel } from "../lazy-panels";
import { BenchmarkBaselinePanel } from "./bench";
import { ProofStatusPanel } from "./proof";
import { QueueJobButton } from "./jobs";

export function BenchmarksSection({
  card,
  composed,
  selectedPresetRecord,
  selectedPreset,
  featureRows,
  proofStatus,
  onMonitorJob
}: {
  card: Record<string, unknown>;
  composed: Record<string, unknown>;
  selectedPresetRecord: PresetRecord | null;
  selectedPreset: string;
  featureRows: ProductCapability[];
  proofStatus: ProofStatusReport | null;
  onMonitorJob: (id: string) => void;
}) {
  return (
            <TabbedSection
          id="benchmarks"
          tabs={[
            {
              id: "baseline",
              label: tr("Baseline"),
              icon: <BarChart3 size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={tr("Benchmark Baseline")} icon={<BarChart3 size={18} />} desc={tr("Reference metric and the resolved runtime it was measured on.")} wide>
                    <BenchmarkBaselinePanel card={card} composed={composed} record={selectedPresetRecord} selectedPreset={selectedPreset} />
                  </ModuleCard>
                  <ModuleCard title={tr("Capability Status")} icon={<Activity size={18} />} wide>
                    <CapabilityTable rows={featureRows.filter((feature) => feature.id === "benchmark_runs")} />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "live",
              label: tr("Live bench"),
              icon: <TimerReset size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={tr("Live Engine")} icon={<Activity size={18} />} desc={tr("The bench drives the running engine — start the runtime first.")}>
                    <EngineStatusCard />
                  </ModuleCard>
                  <ModuleCard title={tr("Live Benchmark + A/B")} icon={<TimerReset size={18} />} desc={tr("Run a real micro-benchmark against the engine; run twice for an A/B delta.")} wide>
                    <EngineBenchPanel referenceTps={asNumber(asRecord(card.primary_metric).value) || null} />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "coverage",
              label: tr("Coverage"),
              icon: <ShieldCheck size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={tr("Benchmark Coverage")} icon={<ShieldCheck size={18} />} wide>
                    <ProofStatusPanel report={proofStatus} />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "run",
              label: tr("Run plan"),
              icon: <Play size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={tr("Run Plan")} icon={<Play size={18} />}>
                    <WorkflowSteps rows={[["1", tr("Warmup"), tr("Stabilize cache and CUDA graph path")], ["2", tr("Load test"), tr("Measure TTFT/TPS/acceptance")], ["3", tr("Proof"), tr("Attach immutable evidence refs")]]} />
                  </ModuleCard>
                  <ModuleCard title={tr("Run Commands")} icon={<SquareTerminal size={18} />} desc={tr("Queue a benchmark as a job, or copy the commands to run on the rig.")}>
                    <QueueJobButton
                      label={`${tr("Queue bench")} (${selectedPreset})`}
                      run={() => api.benchRun({ preset_id: selectedPreset, profile: "quick", ctx: "8k" })}
                      onMonitor={onMonitorJob}
                    />
                    <CodeBlock lines={[`sndr bench run --preset ${selectedPreset} --quick`, `sndr bench run --preset ${selectedPreset} --ctx 8k`, "sndr evidence attach-bench --release-check"]} />
                  </ModuleCard>
                </ModuleGrid>
              )
            }
          ]}
        />
  );
}
