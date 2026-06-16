// SPDX-License-Identifier: Apache-2.0
// The Services section: lifecycle planner (start/stop/restart/status/logs across
// runtime targets), live engine status + metrics + dependencies, and the
// lifecycle capability contract. Extracted from section-workspace.tsx.
import { Activity, Cpu, Gauge, Network, ShieldCheck } from "lucide-react";
import { tr } from "../i18n";
import { runtimeHost } from "../lib/overview-presenters";
import type { ProductCapability, EnvironmentReport } from "../api";
import type { RuntimeMode } from "../nav";
import type { GuiSettings } from "../settings";
import { TabbedSection } from "../components/tabbed-section";
import { ModuleCard, ModuleGrid } from "../components/layout";
import { CapabilityTable } from "../components/capability-table";
import { ServiceLifecyclePlanner, EngineStatusCard, EngineMetricsPanel } from "../lazy-panels";
import { EnvironmentPanel } from "./environment";

export function ServicesSection({
  selectedPreset, runtimeTarget, runtimeMode, settings, environment, featureRows
}: {
  selectedPreset: string;
  runtimeTarget: string;
  runtimeMode: RuntimeMode;
  settings: GuiSettings;
  environment: EnvironmentReport | null;
  featureRows: ProductCapability[];
}) {
  return (
            <TabbedSection
          id="services"
          tabs={[
            {
              id: "lifecycle",
              label: tr("Lifecycle"),
              icon: <Network size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={tr("Lifecycle Planner")} icon={<Network size={18} />} desc={tr("Plan start/stop/restart/status/logs across any runtime target, with live engine reachability and post-action verification.")} wide>
                    <ServiceLifecyclePlanner
                      selectedPreset={selectedPreset}
                      runtimeTarget={runtimeTarget}
                      host={runtimeHost(runtimeMode, settings.remoteHost)}
                    />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "engine",
              label: tr("Engine"),
              icon: <Cpu size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={tr("Live Engine")} icon={<Activity size={18} />} desc={tr("Reachability, loaded model and version of the running vLLM OpenAI server.")}>
                    <EngineStatusCard />
                  </ModuleCard>
                  <ModuleCard title={tr("Live Metrics")} icon={<Gauge size={18} />} desc={tr("Prometheus KPIs from the running engine — queue, KV cache, throughput, TTFT/TPOT, spec-decode.")}>
                    <EngineMetricsPanel />
                  </ModuleCard>
                  <ModuleCard title={tr("Engine & Dependencies")} icon={<Cpu size={18} />} desc={tr("Installed engine and library versions on the daemon host — what would serve.")} wide>
                    <EnvironmentPanel env={environment} />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "contracts",
              label: tr("Contracts"),
              icon: <ShieldCheck size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={tr("Lifecycle Surface")} icon={<ShieldCheck size={18} />} desc={tr("Which lifecycle capabilities the Product API exposes today.")} wide>
                    <CapabilityTable rows={featureRows.filter((feature) => ["service_lifecycle", "web_daemon", "desktop_remote"].includes(feature.id))} />
                  </ModuleCard>
                </ModuleGrid>
              )
            }
          ]}
        />
  );
}
