// SPDX-License-Identifier: Apache-2.0
// The section renderer. Given the active sectionId plus a snapshot of Product
// API state (passed as props), it renders that section's workspace: the heading
// and the per-section tabbed panels. Extracted verbatim from App.tsx so the app
// shell keeps only state/routing/composition and renders <SectionWorkspace/>
// inside one Suspense boundary. Pure presentation — no App-local coupling.
import { Suspense, useMemo } from "react";
import { Boxes, Command, Cpu, Gauge, GitCompare, LayoutGrid, Link2, MessageSquare, Rocket, Route, Server, SlidersHorizontal, Sparkles } from "lucide-react";
import { tr } from "../i18n";
import type { ChatTarget } from "../Engine";
import type { ViewportTier } from "../hooks/useViewport";
import type { GuiSettings } from "../settings";
import type { Gate, RuntimeMode, SectionId } from "../nav";
import { AuthUser, BundleSpec, DiffUpstreamReport, DoctorReport, EnvironmentReport, HostProfile, PatchDoctorReport, PatchListResult, PresetExplainResult, PresetListResult, PresetRecord, ProductCapability, ProductOverview, ProofStatusReport, UserPresetList, V2ConfigCatalog, V2ConfigPreview } from "../api";
import { runtimeHost } from "../lib/overview-presenters";
import { sectionSpec } from "../lib/section-spec";
import { AdvancedSection } from "./advanced-section";
import { PresetsSection } from "./presets-section";
import { OverviewSection } from "./overview-section";
import { ClientsSection } from "./clients-section";
import { PatchesSection } from "./patches-section";
import { BenchmarksSection } from "./benchmarks-section";
import { EvidenceSection } from "./evidence-section";
import { SetupSection } from "./setup-section";
import { ServicesSection } from "./services-section";
import { DoctorSection } from "./doctor-section";
import { ReportsSection } from "./reports-section";
import { ModuleCard, ModuleGrid } from "../components/layout";
import { type GateStatus } from "../components/primitives";
import { TabbedSection } from "../components/tabbed-section";
import { toast } from "../components/toast";
import { FleetPanel } from "../Fleet";
import { SkeletonCards } from "../Skeleton";
import { ChatConsole, ConfigsSection, HostsSection, KvCalcPanel, BaselinePanel, CopilotPanel, ContainersPanel, VirtualizationPanel, HardwarePanel, RoutingPanel, FlagsPanel } from "../lazy-panels";
import { ModelsWorkbench } from "./models-workbench";
import { OperationsConsole } from "./operations";

export function SectionWorkspace({
  sectionId,
  viewport,
  overview,
  presets,
  filteredPresets,
  selectedPreset,
  selectedPresetRecord,
  explain,
  runtimeMode,
  runtimeTarget,
  patchPolicy,
  runtimeTargets,
  featureRows,
  patches,
  patchDoctor,
  configCatalog,
  configPreview,
  bundles,
  diffUpstream,
  proofStatus,
  userPresets,
  doctorReport,
  environment,
  hostProfiles,
  gates,
  gateCounts,
  events,
  cliLines,
  apiBase,
  settings,
  onSection,
  onPreset,
  onCommand,
  onSettings,
  onConfigPreview,
  onUserPresetsRefresh,
  onHostsRefresh,
  onMonitorJob,
  authUser,
  onAuthRefresh,
  chatTarget,
  onChatWithHost,
  onAddServer,
  focusHostId,
  onFocusConsumed,
  onFocusHost,
  installIntent,
  onSetupNode,
  onContainers,
  onHardware,
  applyEnabled
}: {
  sectionId: SectionId;
  viewport: ViewportTier;
  overview: ProductOverview | null;
  presets: PresetListResult | null;
  filteredPresets: PresetRecord[];
  selectedPreset: string;
  selectedPresetRecord: PresetRecord | null;
  explain: PresetExplainResult | null;
  runtimeMode: RuntimeMode;
  runtimeTarget: string;
  patchPolicy: string;
  runtimeTargets: ProductCapability[];
  featureRows: ProductCapability[];
  patches: PatchListResult | null;
  patchDoctor: PatchDoctorReport | null;
  configCatalog: V2ConfigCatalog | null;
  configPreview: V2ConfigPreview | null;
  bundles: BundleSpec[];
  diffUpstream: DiffUpstreamReport | null;
  proofStatus: ProofStatusReport | null;
  userPresets: UserPresetList | null;
  doctorReport: DoctorReport | null;
  environment: EnvironmentReport | null;
  hostProfiles: HostProfile[];
  gates: Gate[];
  gateCounts: Record<GateStatus, number>;
  events: Array<[string, string, string]>;
  cliLines: string[];
  apiBase: string;
  settings: GuiSettings;
  onSection: (section: SectionId) => void;
  onPreset: (id: string) => void;
  onCommand: () => void;
  onSettings: (patch: Partial<GuiSettings>) => void;
  onConfigPreview: (preview: V2ConfigPreview) => void;
  onUserPresetsRefresh: () => void;
  onHostsRefresh: () => void;
  onMonitorJob: (id: string) => void;
  authUser: AuthUser | null;
  onAuthRefresh: () => void;
  chatTarget: ChatTarget | null;
  onChatWithHost: (profile: HostProfile) => void;
  onAddServer: (profile: HostProfile) => Promise<boolean>;
  focusHostId: string | null;
  onFocusConsumed: () => void;
  onFocusHost: (id: string) => void;
  installIntent: { hostId?: string; target?: string } | null;
  onSetupNode: (id: string) => void;
  onContainers: (id: string) => void;
  onHardware: (id: string) => void;
  applyEnabled?: boolean;
}) {
  const spec = sectionSpec(sectionId);
  // Stable host option list — recompute only when profiles change, so the lazy
  // Containers/Hardware panels don't see a fresh array prop on every render.
  const hostOptions = useMemo(() => hostProfiles.map((h) => ({ id: h.id, label: h.label })), [hostProfiles]);
  const card = (explain?.card ?? selectedPresetRecord?.card ?? {}) as Record<string, unknown>;
  const composed = (explain?.composed ?? {}) as Record<string, unknown>;
  const patchRows = patches?.patches ?? [];

  return (
    <section className={`section-workspace section-${sectionId}`}>
      <header className="section-heading">
        <div>
          <span>{spec.kicker}</span>
          <h1>{spec.title}</h1>
          <p>{spec.description}</p>
        </div>
        <div className="section-actions">
          {sectionId === "presets" && selectedPreset && (
            <button
              className="tool-button"
              title={`${tr("Copy a shareable link to")} ${selectedPreset}`}
              onClick={() => {
                const url = window.location.href;
                void navigator.clipboard?.writeText(url).then(
                  () => toast(`${tr("Link to")} ${selectedPreset} ${tr("copied")}`, "success"),
                  () => toast(tr("Could not copy link"), "error")
                );
              }}
            >
              <Link2 size={16} />
              {tr("Copy Link")}
            </button>
          )}
          <button className="tool-button" onClick={() => onSection("launch-plan")}>
            <Rocket size={16} />
            {tr("Launch Plan")}
          </button>
          <button className="tool-button" onClick={onCommand} title={tr("Open the command palette (⌘K)")}>
            <Command size={16} />
            {tr("Quick Action")}
          </button>
        </div>
      </header>

      {sectionId === "overview" && (
        <OverviewSection
          overview={overview}
          presets={presets}
          patches={patches}
          hostProfiles={hostProfiles}
          runtimeMode={runtimeMode}
          doctorReport={doctorReport}
          environment={environment}
          viewport={viewport}
          settings={settings}
          runtimeTarget={runtimeTarget}
          selectedPreset={selectedPreset}
          apiBase={apiBase}
          gateCounts={gateCounts}
          gates={gates}
          runtimeTargets={runtimeTargets}
          patchPolicy={patchPolicy}
          featureRows={featureRows}
          onSection={onSection}
        />
      )}

      {sectionId === "setup" && (
        <SetupSection
          environment={environment}
          overview={overview}
          doctorReport={doctorReport}
          gateCounts={gateCounts}
          selectedPreset={selectedPreset}
          runtimeMode={runtimeMode}
          apiBase={apiBase}
          installIntent={installIntent}
          presets={presets}
          onSection={onSection}
          onPreset={onPreset}
        />
      )}

      {sectionId === "fleet" && (
        <ModuleGrid>
          <ModuleCard title={tr("Fleet overview")} icon={<LayoutGrid size={18} />} desc={tr("Every registered GPU/engine host at a glance — a single concurrent SSH sweep shows status, running model, vLLM version, GPUs and live patch count per server. Click a server to drill into its card.")} wide>
            <FleetPanel onOpenHost={(id) => { onFocusHost(id); onSection("hosts"); }} />
          </ModuleCard>
        </ModuleGrid>
      )}

      {sectionId === "containers" && (
        <ModuleGrid>
          <ModuleCard title={tr("Containers")} icon={<Boxes size={18} />} desc={tr("Manage the vLLM/engine containers on a server — list, live CPU/memory, logs, start/stop/restart, and (when SNDR_ENABLE_EXEC is on) exec inside. Pick the local daemon's host (docker socket) or a registered host (over SSH). Scoped to engine containers only.")} wide>
            <Suspense fallback={<SkeletonCards count={6} />}>
              <ContainersPanel hosts={hostOptions} onNavigate={(section) => onSection(section as SectionId)} initialHostId={focusHostId ?? undefined} />
            </Suspense>
          </ModuleCard>
        </ModuleGrid>
      )}

      {(sectionId === "virtualization" || sectionId === "kubernetes") && (
        <ModuleGrid>
          <ModuleCard title={tr("Virtualization")} icon={<Server size={18} />} desc={tr("One control plane over compute — Proxmox VE hosts & guests (VMs/LXC) and Kubernetes (nodes, pods, events, KubeVirt VMs, deploy) — each linked back to the SNDR preset it runs. Read-only; degrades to a connect/not-installed card per source.")} wide>
            <Suspense fallback={<SkeletonCards count={4} />}>
              <VirtualizationPanel />
            </Suspense>
          </ModuleCard>
        </ModuleGrid>
      )}

      {sectionId === "hardware" && (
        <ModuleGrid>
          <ModuleCard title={tr("GPU & Hardware")} icon={<Cpu size={18} />} desc={tr("Live per-GPU telemetry over nvidia-smi — utilisation, VRAM, temperature, power vs limits, clocks, fan, PCIe link, pstate and ECC — plus host CPU/RAM. Pick the local daemon host or a registered host (over SSH).")} wide>
            <Suspense fallback={<SkeletonCards count={2} />}>
              <HardwarePanel hosts={hostOptions} initialHostId={focusHostId ?? undefined} />
            </Suspense>
          </ModuleCard>
        </ModuleGrid>
      )}

      {sectionId === "routing" && (
        <ModuleGrid>
          <ModuleCard title={tr("Workload routing")} icon={<Route size={18} />} desc={tr("The deterministic spec-decode router — the same brain the gateway uses. Per bench-validated profile: which workloads are allowed/denied and their measured TPS delta. Classify a request shape by its response_format / tool_choice / workload_class signals and see which profile it resolves to.")} wide>
            <Suspense fallback={<SkeletonCards count={2} />}>
              <RoutingPanel />
            </Suspense>
          </ModuleCard>
        </ModuleGrid>
      )}

      {sectionId === "flags" && (
        <ModuleGrid>
          <ModuleCard title={tr("Env-flag matrix")} icon={<SlidersHorizontal size={18} />} desc={tr("Every GENESIS_ENABLE_* flag in the registry with its effective default — searchable, filterable by family. Name a running engine container to overlay its live ON/OFF state and flag drift (missing = default-on but off on the engine; extra = on beyond the default).")} wide>
            <Suspense fallback={<SkeletonCards count={2} />}>
              <FlagsPanel />
            </Suspense>
          </ModuleCard>
        </ModuleGrid>
      )}

      {sectionId === "hosts" && (
        <HostsSection
          hostProfiles={hostProfiles}
          environment={environment}
          overview={overview}
          runtimeTargets={runtimeTargets}
          apiBase={apiBase}
          runtimeMode={runtimeMode}
          onHostsRefresh={onHostsRefresh}
          onChatWithHost={onChatWithHost}
          onAddServer={onAddServer}
          focusHostId={focusHostId}
          onFocusConsumed={onFocusConsumed}
          onSetupNode={onSetupNode}
          onContainers={onContainers}
          onHardware={onHardware}
          applyEnabled={applyEnabled}
        />
      )}

      {sectionId === "models" && (
        <ModelsWorkbench
          catalog={configCatalog}
          presets={presets?.presets ?? []}
          selectedPreset={selectedPreset}
          selectedModel={selectedPresetRecord?.model ?? null}
          composed={composed}
          card={card}
          patchCount={patchRows.length}
          runtimeTarget={runtimeTarget}
          patchPolicy={patchPolicy}
          onPreset={onPreset}
        />
      )}

      {sectionId === "configs" && (
        <ConfigsSection
          catalog={configCatalog}
          preview={configPreview}
          selectedPreset={selectedPreset}
          userPresets={userPresets}
          onPreview={onConfigPreview}
          onUserPresetsRefresh={onUserPresetsRefresh}
        />
      )}

      {sectionId === "presets" && (
        <PresetsSection
          overview={overview}
          presets={presets}
          filteredPresets={filteredPresets}
          selectedPreset={selectedPreset}
          selectedPresetRecord={selectedPresetRecord}
          explain={explain}
          runtimeTargets={runtimeTargets}
          runtimeTarget={runtimeTarget}
          patchPolicy={patchPolicy}
          card={card}
          composed={composed}
          onPreset={onPreset}
          onSection={onSection}
        />
      )}

      {sectionId === "services" && (
        <ServicesSection
          selectedPreset={selectedPreset}
          runtimeTarget={runtimeTarget}
          runtimeMode={runtimeMode}
          settings={settings}
          environment={environment}
          featureRows={featureRows}
        />
      )}

      {sectionId === "doctor" && (
        <DoctorSection
          doctorReport={doctorReport}
          gates={gates}
          patchDoctor={patchDoctor}
          onSection={onSection}
        />
      )}

      {sectionId === "patches" && (
        <PatchesSection
          patches={patches}
          configCatalog={configCatalog}
          bundles={bundles}
          diffUpstream={diffUpstream}
          selectedPreset={selectedPreset}
          patchPolicy={patchPolicy}
          composed={composed}
        />
      )}

      {sectionId === "benchmarks" && (
        <BenchmarksSection
          card={card}
          composed={composed}
          selectedPresetRecord={selectedPresetRecord}
          selectedPreset={selectedPreset}
          featureRows={featureRows}
          proofStatus={proofStatus}
          onMonitorJob={onMonitorJob}
        />
      )}

      {sectionId === "evidence" && (
        <EvidenceSection
          card={card}
          proofStatus={proofStatus}
          selectedPreset={selectedPreset}
          onMonitorJob={onMonitorJob}
        />
      )}

      {sectionId === "clients" && (
        <ClientsSection
          runtimeMode={runtimeMode}
          settings={settings}
          composed={composed}
          selectedPreset={selectedPreset}
          runtimeTargets={runtimeTargets}
          runtimeTarget={runtimeTarget}
        />
      )}

      {sectionId === "planner" && (
        <ModuleGrid>
          <ModuleCard title={tr("KV-cache / VRAM fit calculator")} icon={<Gauge size={18} />} desc={tr("GQA, MoE and tensor-parallel aware. Slide context to see weights / KV / overhead vs the per-GPU budget, the max context per KV dtype, and the VRAM curve.")} wide>
            <KvCalcPanel />
          </ModuleCard>
          <ModuleCard title={tr("Quality-baseline regression diff")} icon={<GitCompare size={18} />} desc={tr("Save a trusted bench/eval result, then diff a new run against it — direction-aware regression flags + a CI exit code.")} wide>
            <BaselinePanel />
          </ModuleCard>
        </ModuleGrid>
      )}

      {(sectionId === "chat" || sectionId === "copilot") && (
        <TabbedSection
          id="chat"
          tabs={[
            {
              id: "model",
              label: tr("Model chat"),
              icon: <MessageSquare size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={tr("Local model chat")} icon={<MessageSquare size={18} />} desc={tr("Multi-turn streaming conversation with a running vLLM model. Set the engine host/port, pick the model, tune the system prompt and sampling — a direct line to the inference server.")} wide>
                    <ChatConsole defaultHost={runtimeHost(runtimeMode, settings.remoteHost)} target={chatTarget} />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "copilot",
              label: tr("Ops Copilot"),
              icon: <Sparkles size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={tr("Ops Copilot")} icon={<Sparkles size={18} />} desc={tr("An assistant that can read this control plane. Ask it about your presets, patches, doctor findings, hosts or VRAM fit in plain language — it calls read-only Product API tools, answers from the real live data, and proposes changes (with the exact section/CLI) for you to review and apply. It never mutates anything itself.")} wide>
                    <CopilotPanel onNavigate={(section) => onSection(section as SectionId)} />
                  </ModuleCard>
                </ModuleGrid>
              )
            }
          ]}
        />
      )}

      {sectionId === "reports" && (
        <ReportsSection
          card={card}
          selectedPreset={selectedPreset}
          runtimeTargets={runtimeTargets}
          runtimeTarget={runtimeTarget}
          patchPolicy={patchPolicy}
          gateCounts={gateCounts}
          proofStatus={proofStatus}
          events={events}
        />
      )}

      {sectionId === "operations" && (
        <OperationsConsole onMonitor={onMonitorJob} />
      )}

      {sectionId === "advanced" && (
        <AdvancedSection
          apiBase={apiBase}
          settings={settings}
          environment={environment}
          authUser={authUser}
          featureRows={featureRows}
          patchDoctor={patchDoctor}
          selectedPreset={selectedPreset}
          runtimeTarget={runtimeTarget}
          patchPolicy={patchPolicy}
          composed={composed}
          cliLines={cliLines}
          events={events}
          onMonitorJob={onMonitorJob}
          onSettings={onSettings}
          onAuthRefresh={onAuthRefresh}
        />
      )}
    </section>
  );
}
