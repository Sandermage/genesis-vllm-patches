// SPDX-License-Identifier: Apache-2.0
// The section renderer. Given the active sectionId plus a snapshot of Product
// API state (passed as props), it renders that section's workspace: the heading
// and the per-section tabbed panels. Extracted verbatim from App.tsx so the app
// shell keeps only state/routing/composition and renders <SectionWorkspace/>
// inside one Suspense boundary. Pure presentation — no App-local coupling.
import { Suspense, useEffect, useMemo, useState } from "react";
import {
  Activity, AlertTriangle, BadgeCheck, BarChart3, Bell, Box, Boxes, Clock3, Code2,
  Command, Cpu, Database, DownloadCloud, FileText, Gauge, GitBranch, GitCompare,
  KeyRound, Layers3, LayoutGrid, Link2, ListChecks, MessageSquare, Monitor, Network,
  PackageCheck, Palette, Play, Rocket, Route, Server, Settings, ShieldCheck,
  SlidersHorizontal, Sparkles, SquareTerminal, Stethoscope, Table2, Terminal,
  TimerReset, Wrench
} from "lucide-react";
import { tr } from "../i18n";
import type { ChatTarget } from "../Engine";
import type { ViewportTier } from "../hooks/useViewport";
import type { GuiSettings } from "../settings";
import type { Gate, RuntimeMode, SectionId } from "../nav";
import {
  api, AuthUser, BundleSpec, DiffUpstreamReport, DoctorReport, EnvironmentReport,
  HostProfile, PatchDoctorReport, PatchListResult, PresetExplainResult,
  PresetListResult, PresetRecord, ProductCapability, ProductOverview,
  ProofStatusReport, UserPresetList, V2ConfigCatalog, V2ConfigPreview
} from "../api";
import { asNumber, asRecord, asText, countRecord } from "../lib/coerce";
import { targetTitle } from "../lib/format";
import { runtimeHost } from "../lib/overview-presenters";
import { sectionSpec } from "../lib/section-spec";
import { CapabilityTable } from "../components/capability-table";
import { BarList, OvKpi, PercentBar } from "../components/charts";
import { CodeBlock } from "../components/code-block";
import { ModuleCard, ModuleGrid } from "../components/layout";
import { CompactList, InfoRows, KpiGrid, type GateStatus } from "../components/primitives";
import { CodeTabs, TabIntro, WorkflowSteps } from "../components/shell-bits";
import { TabbedSection } from "../components/tabbed-section";
import { toast } from "../components/toast";
import { SecurityPanel, UserAdminPanel } from "../Auth";
import { FleetPanel } from "../Fleet";
import { SkeletonCards } from "../Skeleton";
import { UpdatesPanel } from "../Updates";
import {
  ChatConsole, EngineBenchPanel, EngineMetricsPanel, EnginePlayground, EngineStatusCard,
  ConfigsSection, HostsSection, DeploymentConsole, ServiceLifecyclePlanner,
  PatchInventoryControl, ApiTokenManager, NotificationSettings, AppearanceSettings,
  ApiTokenField, KvCalcPanel, BaselinePanel, InstallWizard, CopilotPanel,
  ContainersPanel, VirtualizationPanel, HardwarePanel, RoutingPanel, FlagsPanel, LicensePanel
} from "../lazy-panels";
import { EndpointExplorer, ReportGenerator } from "./api-explorer";
import { AuditLogPanel } from "./audit-log";
import { BenchmarkBaselinePanel, EvidenceRows } from "./bench";
import { ConfigDraftEditor } from "./config-draft-editor";
import { ConnectionMap } from "./connection-bar";
import { CaveatsPanel, ConfigKeysPanel, TracesPanel } from "./diagnostics";
import { DoctorFindings, DoctorSummary } from "./doctor";
import { EnvironmentPanel } from "./environment";
import { GateRow } from "./gate-row";
import { QueueJobButton } from "./jobs";
import { LayerEditor } from "./layer-editor";
import { ModelsWorkbench } from "./models-workbench";
import { EventLog } from "./operational-console";
import { OperationsConsole } from "./operations";
import { AdminSurfaceMatrix, DoctorCoveragePanel } from "./patch-doctor";
import { PatchLifecycleGraph, PatchModelSupport, PatchRegistryInsight, PatchSummaryPanel } from "./patch-overview";
import { PresetCatalogTable } from "./preset-catalog";
import { PresetPolicyGraph } from "./preset-insight";
import { PresetQuickPanel } from "./preset-quick";
import { PresetRecommendPanel } from "./preset-recommend";
import { PresetSelectedView, PresetSummaryStrip } from "./preset-views";
import { ProofStatusPanel } from "./proof";
import { EndpointRows } from "./rail-cards";
import { BundlesPanel, UpstreamDiffPanel } from "./registry";
import { SetupWizard } from "./setup-wizard";

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
  // Controlled tab for the Presets section so catalog action buttons can jump
  // straight to Selected / Edit / Policy.
  const [presetTab, setPresetTab] = useState("catalog");
  // Setup tabs are controlled so "Set up as node" can jump to the Install tab,
  // while Deploy / Guided stay clickable (the bug was a controlled tab with no
  // change handler — it got stuck).
  // Setup flows in a logical order: orient (Guided) → install the daemon on a
  // server (Install) → render deploy artifacts for a model (Deploy). Default to
  // the guided entry point; jump to Install when a node-setup intent arrives.
  const [setupTab, setSetupTab] = useState("guided");
  useEffect(() => { if (installIntent) setSetupTab("install"); }, [installIntent]);
  const card = (explain?.card ?? selectedPresetRecord?.card ?? {}) as Record<string, unknown>;
  const composed = (explain?.composed ?? {}) as Record<string, unknown>;
  const familyCounts = overview?.catalog.family_counts ?? {};
  const workloadCounts = overview?.catalog.workload_counts ?? {};
  const patchRows = patches?.patches ?? [];
  const patchSummary = patches?.summary ?? null;
  // Presets that carry a measured primary metric — a catalog-health signal that
  // isn't shown anywhere else (the hero shows raw counts, not bench coverage).
  const benchProven = (presets?.presets ?? []).filter(
    (p) => asNumber(asRecord(p.card?.primary_metric).value) > 0
  ).length;

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
      )}

      {sectionId === "setup" && (
        <TabbedSection
          id="setup"
          activeTab={setupTab}
          onTabChange={setSetupTab}
          tabs={[
            // 1) Orient: where you are + what to do next (read-only, safe).
            {
              id: "guided",
              label: `1 · ${tr("Guided setup")}`,
              icon: <ShieldCheck size={15} />,
              render: () => (
                <>
                  <TabIntro icon={<ShieldCheck size={16} />} title={tr("Start here — guided setup")}
                    text={tr("A read-only checklist of where you stand: environment, engine, dependencies and launch gates, with a clear next step. Nothing is changed here — it just tells you what to do.")} />
                  <SetupWizard
                    environment={environment}
                    overview={overview}
                    doctorReport={doctorReport}
                    gateCounts={gateCounts}
                    selectedPreset={selectedPreset}
                    runtimeMode={runtimeMode}
                    apiBase={apiBase}
                    onSection={onSection}
                  />
                </>
              )
            },
            // 2) Install the SNDR daemon / engine onto a GPU server over SSH.
            {
              id: "install",
              label: `2 · ${tr("Install onto host")}`,
              icon: <Server size={15} />,
              render: () => (
                <>
                  <TabIntro icon={<Server size={16} />} title={tr("Install onto a GPU host (over SSH)")}
                    text={tr("Pick a registered host and a preset, preview the exact install plan, then apply it over SSH — it ships the daemon/engine onto that server so the GUI can manage it. Gated: review the plan before it runs.")} />
                  <InstallWizard initial={installIntent || undefined} />
                </>
              )
            },
            // 3) Render deploy artifacts (compose/systemd/run) for a chosen model.
            {
              id: "deploy",
              label: `3 · ${tr("Deploy a model")}`,
              icon: <Rocket size={15} />,
              render: () => (
                <>
                  <TabIntro icon={<Rocket size={16} />} title={tr("Deploy a model preset")}
                    text={tr("Turn a preset into ready-to-run artifacts — docker-compose, systemd unit or a docker run line, with the right image, GPUs, ports and patch env baked in. Copy them to the host, or use Install above to push over SSH.")} />
                  <DeploymentConsole
                    presets={presets}
                    selectedPreset={selectedPreset}
                    onSelectPreset={onPreset}
                  />
                </>
              )
            }
          ]}
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
        <TabbedSection
          id="presets"
          activeTab={presetTab}
          onTabChange={setPresetTab}
          tabs={[
            {
              id: "catalog",
              label: tr("Catalog"),
              icon: <Database size={15} />,
              render: () => (
                <div className="preset-catalog-view">
                  {presets?.load_errors && presets.load_errors.length > 0 && (
                    <div className="preset-load-errors">
                      <AlertTriangle size={15} />
                      <div>
                        <strong>{presets.load_errors.length} {presets.load_errors.length > 1 ? tr("presets failed to load") : tr("preset failed to load")}</strong>
                        {presets.load_errors.slice(0, 6).map((e, i) => (
                          <div key={i} className="preset-load-error"><code>{e.preset ?? e.id ?? e.file ?? "?"}</code> — {e.error ?? e.message ?? JSON.stringify(e)}</div>
                        ))}
                        {presets.load_errors.length > 6 && <div className="muted">+{presets.load_errors.length - 6} {tr("more")}</div>}
                      </div>
                    </div>
                  )}
                  <PresetSummaryStrip presets={filteredPresets} selectedPreset={selectedPreset} />
                  <div className="preset-catalog-split">
                    <ModuleCard
                      title={tr("Preset Catalog")}
                      icon={<Database size={18} />}
                      desc={tr("Pick a preset to inspect its runtime, edit a local copy, or launch it.")}
                    >
                      <PresetCatalogTable
                        presets={filteredPresets}
                        selectedPreset={selectedPreset}
                        onPreset={onPreset}
                        onEdit={(id) => { onPreset(id); setPresetTab("edit"); }}
                      />
                    </ModuleCard>
                    <PresetQuickPanel
                      selectedPreset={selectedPreset}
                      record={selectedPresetRecord}
                      card={card}
                      composed={composed}
                      onOpenCard={() => setPresetTab("selected")}
                      onEdit={() => setPresetTab("edit")}
                      onPolicy={() => setPresetTab("recommend")}
                      onLaunch={() => onSection("launch-plan")}
                    />
                  </div>
                </div>
              )
            },
            {
              id: "recommend",
              label: tr("Recommend & analytics"),
              icon: <Rocket size={15} />,
              render: () => {
                const allPresets = presets?.presets ?? [];
                const annotated = allPresets.filter((p) => p.has_card).length;
                const benchProven = allPresets.filter((p) => asNumber(asRecord(p.card?.primary_metric).value) > 0).length;
                const statusDist = countRecord(allPresets.map((p) => asText(p.card?.status, p.has_card ? "annotated" : "unannotated")));
                const visibilityDist = countRecord(allPresets.filter((p) => p.has_card).map((p) => asText(p.card?.evidence_visibility, "unknown")));
                const fallbacks = allPresets.filter((p) => p.card?.fallback_preset);
                const bar = (counts: Record<string, number>): Array<[string, number, string]> => {
                  const max = Math.max(1, ...Object.values(counts));
                  return Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([k, v]) => [k, Math.round((v / max) * 100), String(v)]);
                };
                const kpis: Array<[string, number]> = [
                  [tr("Presets"), allPresets.length], [tr("Annotated"), annotated], [tr("Bench-proven"), benchProven],
                  [tr("Fallbacks"), fallbacks.length], [tr("Families"), Object.keys(familyCounts).length], [tr("Workloads"), Object.keys(workloadCounts).length]
                ];
                return (
                <>
                  <ModuleGrid>
                    <ModuleCard title={tr("Recommend a preset")} icon={<Rocket size={18} />} desc={tr("Rank presets for a workload + rig + concurrency target, then inspect the winner.")} wide>
                      <PresetRecommendPanel
                        hardwareOptions={Array.from(new Set((presets?.presets ?? []).map((preset) => preset.hardware))).filter(Boolean)}
                        workloadCounts={overview?.catalog.workload_counts ?? {}}
                        onSelect={(id) => { onPreset(id); setPresetTab("selected"); }}
                      />
                    </ModuleCard>
                  </ModuleGrid>
                  <div className="preset-analytics-heading"><BarChart3 size={15} /> {tr("Catalog analytics")} <span>{tr("policy, coverage and annotation across")} {allPresets.length} {tr("presets")}</span></div>
                  <div className="preset-analytics-kpis">
                    {kpis.map(([label, value]) => (
                      <div className="preset-stat" key={label}><span className="preset-stat-value">{value}</span><span className="preset-stat-label">{label}</span></div>
                    ))}
                  </div>
                  <ModuleGrid className="preset-analytics-grid">
                    <ModuleCard title={tr("Workload Policy")} icon={<SlidersHorizontal size={18} />} desc={`${tr("Allow/deny for")} ${selectedPreset}.`}>
                      <PresetPolicyGraph card={card} />
                    </ModuleCard>
                    <ModuleCard title={tr("Status Distribution")} icon={<ShieldCheck size={18} />} desc={`${annotated} ${tr("annotated")} · ${Object.keys(statusDist).length} ${tr("statuses")}`}>
                      <BarList rows={bar(statusDist)} />
                    </ModuleCard>
                    <ModuleCard title={tr("Evidence Visibility")} icon={<FileText size={18} />} desc={`${Object.keys(visibilityDist).length} ${Object.keys(visibilityDist).length === 1 ? tr("visibility level") : tr("visibility levels")}`}>
                      {Object.keys(visibilityDist).length ? <BarList rows={bar(visibilityDist)} /> : <p className="muted">{tr("No annotated presets.")}</p>}
                    </ModuleCard>
                    <ModuleCard title={tr("Workload Coverage")} icon={<Layers3 size={18} />} desc={`${Object.keys(workloadCounts).length} ${tr("workload classes")}`}>
                      <BarList rows={bar(workloadCounts)} />
                    </ModuleCard>
                    <ModuleCard title={tr("Family Coverage")} icon={<Box size={18} />} desc={`${Object.keys(familyCounts).length} ${tr("routing families")}`}>
                      <BarList rows={bar(familyCounts)} />
                    </ModuleCard>
                    <ModuleCard title={tr("Fallback Chains")} icon={<GitBranch size={18} />} desc={`${fallbacks.length} ${tr("of")} ${allPresets.length} ${tr("presets")}`}>
                      {fallbacks.length ? (
                        <CompactList rows={fallbacks.map((p) => [p.id, `→ ${asText(p.card?.fallback_preset, "-")}`] as [string, string])} />
                      ) : (
                        <p className="muted">{tr("No fallback chains declared.")}</p>
                      )}
                    </ModuleCard>
                  </ModuleGrid>
                </>
                );
              }
            },
            {
              id: "selected",
              label: tr("Selected"),
              icon: <FileText size={15} />,
              render: () => (
                <PresetSelectedView
                  selectedPreset={selectedPreset}
                  record={selectedPresetRecord}
                  card={card}
                  composed={composed}
                  explain={explain}
                  runtimeTargets={runtimeTargets}
                  runtimeTarget={runtimeTarget}
                  patchPolicy={patchPolicy}
                  onEdit={() => setPresetTab("edit")}
                  onLaunch={() => onSection("launch-plan")}
                  onConfigs={() => onSection("configs")}
                />
              )
            },
            {
              id: "edit",
              label: tr("Edit"),
              icon: <SlidersHorizontal size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard
                    title={`${tr("Visual Editor")} — ${selectedPreset}`}
                    icon={<Wrench size={18} />}
                    desc={tr("Full preset editing: pointers, card metadata and any other fields the preset defines. Saves an operator-local copy.")}
                    wide
                  >
                    <LayerEditor kind="preset" layerId={selectedPreset} />
                  </ModuleCard>
                </ModuleGrid>
              )
            }
          ]}
        />
      )}

      {sectionId === "services" && (
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
      )}

      {sectionId === "doctor" && (
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
      )}

      {sectionId === "patches" && (
        <TabbedSection
          id="patches"
          tabs={[
            {
              id: "registry",
              label: tr("Registry"),
              icon: <PackageCheck size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={tr("Patch Registry Summary")} icon={<PackageCheck size={18} />} desc={`${patches?.total ?? patchRows.length} ${tr("runtime overlays across")} ${new Set(patchRows.map((p) => p.family)).size} ${tr("families")}.`} wide>
                    <PatchSummaryPanel summary={patchSummary} total={patches?.total ?? patchRows.length} selectedCount={asNumber(composed.enabled_patches_count)} />
                  </ModuleCard>
                  <ModuleCard title={tr("Lifecycle & Default Behavior")} icon={<BarChart3 size={18} />} wide>
                    <PatchLifecycleGraph summary={patchSummary} />
                  </ModuleCard>
                  <ModuleCard title={tr("Status, Families & Legend")} icon={<ListChecks size={18} />} desc={tr("Implementation maturity, subsystem coverage, and what each registry value means.")} wide>
                    <PatchRegistryInsight summary={patchSummary} patches={patchRows} />
                  </ModuleCard>
                  <ModuleCard title={tr("Supported Models")} icon={<Cpu size={18} />} desc={tr("Catalog models the patch family targets — per-patch applicability is in the Inventory tab.")} wide>
                    <PatchModelSupport models={configCatalog?.models ?? []} />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "inventory",
              label: tr("Inventory"),
              icon: <Table2 size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={tr("Patch Inventory Control")} icon={<Table2 size={18} />} wide>
                    <PatchInventoryControl patches={patchRows} />
                  </ModuleCard>
                  <ModuleCard title={tr("Patch Bundles")} icon={<Layers3 size={18} />} wide>
                    <BundlesPanel bundles={bundles} />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "flags",
              label: tr("Flags"),
              icon: <SlidersHorizontal size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={tr("Env-flag matrix")} icon={<SlidersHorizontal size={18} />} desc={tr("Every GENESIS_ENABLE_* flag with its effective default — searchable, filterable by family. Name a running engine container to overlay its live ON/OFF state and flag drift.")} wide>
                    <Suspense fallback={<SkeletonCards count={2} />}>
                      <FlagsPanel />
                    </Suspense>
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "upstream",
              label: tr("Upstream & policy"),
              icon: <GitBranch size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={tr("Upstream Diff")} icon={<GitBranch size={18} />} wide>
                    <UpstreamDiffPanel report={diffUpstream} />
                  </ModuleCard>
                  <ModuleCard title={tr("Policy Preview")} icon={<Code2 size={18} />} wide>
                    <CodeBlock lines={[`preset=${selectedPreset}`, `policy=${patchPolicy}`, "strict_image_digest=true", "dry_run=true"]} />
                  </ModuleCard>
                </ModuleGrid>
              )
            }
          ]}
        />
      )}

      {sectionId === "benchmarks" && (
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
      )}

      {sectionId === "evidence" && (() => {
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
      })()}

      {sectionId === "clients" && (() => {
        const clientHost = runtimeHost(runtimeMode, settings.remoteHost);
        const baseUrl = `http://${clientHost}:8000/v1`;
        const modelName = String(composed.served_model_name ?? selectedPreset);
        return (
        <ModuleGrid>
          <ModuleCard title={tr("Live Engine")} icon={<Activity size={18} />} desc={tr("Is the runtime up? Loaded model and version from the running server.")} wide>
            <EngineStatusCard />
          </ModuleCard>
          <ModuleCard title={tr("Playground")} icon={<MessageSquare size={18} />} desc={tr("Send a real prompt to the running engine — a one-click smoke test.")} wide>
            <EnginePlayground />
          </ModuleCard>
          <ModuleCard title={tr("Client Endpoints")} icon={<Link2 size={18} />} desc={tr("OpenAI-compatible API, health and metrics URLs for the selected runtime host.")} wide>
            <EndpointRows host={clientHost} />
          </ModuleCard>
          <ModuleCard title={tr("Quick Start")} icon={<Code2 size={18} />} desc={`${tr("Copy-paste clients for the served model")} "${modelName}".`} wide>
            <CodeTabs
              tabs={[
                {
                  id: "curl",
                  label: "cURL",
                  lines: [
                    `curl ${baseUrl}/chat/completions \\`,
                    `  -H "Content-Type: application/json" \\`,
                    `  -d '{`,
                    `    "model": "${modelName}",`,
                    `    "messages": [{"role": "user", "content": "Hello"}],`,
                    `    "max_tokens": 256`,
                    `  }'`
                  ]
                },
                {
                  id: "python",
                  label: "Python",
                  lines: [
                    "from openai import OpenAI",
                    "",
                    `client = OpenAI(base_url="${baseUrl}", api_key="not-needed")`,
                    "resp = client.chat.completions.create(",
                    `    model="${modelName}",`,
                    '    messages=[{"role": "user", "content": "Hello"}],',
                    "    max_tokens=256,",
                    ")",
                    "print(resp.choices[0].message.content)"
                  ]
                },
                {
                  id: "stream",
                  label: tr("Streaming"),
                  lines: [
                    "from openai import OpenAI",
                    "",
                    `client = OpenAI(base_url="${baseUrl}", api_key="not-needed")`,
                    "stream = client.chat.completions.create(",
                    `    model="${modelName}",`,
                    '    messages=[{"role": "user", "content": "Hello"}],',
                    "    stream=True,",
                    ")",
                    "for chunk in stream:",
                    "    delta = chunk.choices[0].delta.content or \"\"",
                    "    print(delta, end=\"\", flush=True)"
                  ]
                },
                {
                  id: "health",
                  label: tr("Health"),
                  lines: [
                    `curl http://${clientHost}:8000/health`,
                    `curl http://${clientHost}:8001/metrics | grep vllm:`,
                    `curl ${baseUrl}/models`
                  ]
                }
              ]}
            />
          </ModuleCard>
          <ModuleCard title={tr("Served Model")} icon={<Box size={18} />} desc={tr("What the OpenAI-compatible server exposes for this preset.")}>
            <InfoRows
              rows={[
                [tr("Model name"), modelName],
                [tr("Base URL"), baseUrl],
                [tr("Runtime target"), targetTitle(runtimeTargets, runtimeTarget)],
                [tr("Host mode"), runtimeMode === "remote" ? tr("Remote host") : tr("Local server")]
              ]}
            />
          </ModuleCard>
          <ModuleCard title={tr("Authentication")} icon={<KeyRound size={18} />} desc={tr("The Product API token; the inference server itself follows your vLLM launch flags.")}>
            <InfoRows
              rows={[
                ["GUI/Product API", tr("Open by default; set SNDR_GUI_TOKEN to require a bearer token")],
                [tr("Header"), "Authorization: Bearer <token> or X-SNDR-Token: <token>"],
                [tr("Inference API key"), tr('OpenAI clients accept any value (e.g. "not-needed") unless vLLM --api-key is set')]
              ]}
            />
          </ModuleCard>
          <ModuleCard title={tr("Client Modes")} icon={<Layers3 size={18} />} desc={tr("How operators reach this control plane.")}>
            <CompactList rows={[[tr("Web UI"), tr("Browser control center")], [tr("Desktop"), tr("Tauri remote shell")], ["API", tr("OpenAI-compatible endpoint")], ["CLI", tr("Operator mirror")]]} />
          </ModuleCard>
        </ModuleGrid>
        );
      })()}

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
      )}

      {sectionId === "operations" && (
        <OperationsConsole onMonitor={onMonitorJob} />
      )}

      {sectionId === "advanced" && (
        <TabbedSection
          id="advanced"
          tabs={[
            {
              id: "operations",
              label: tr("Operations"),
              icon: <Terminal size={15} />,
              render: () => <OperationsConsole onMonitor={onMonitorJob} />
            },
            {
              id: "config-keys",
              label: tr("Config keys"),
              icon: <SlidersHorizontal size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={tr("Config-key glossary")} icon={<SlidersHorizontal size={18} />} desc={tr("Every GENESIS_ENABLE_* flag, V1/V2 config key and policy key with provenance — searchable operator reference (mirrors `sndr config-keys`).")} wide>
                    <ConfigKeysPanel />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "traces",
              label: tr("Traces"),
              icon: <Activity size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={tr("Diagnostic trace catalog")} icon={<Activity size={18} />} desc={tr("Per-patch debug traces — the container path each lands at and the env var that enables it. Operator reference (mirrors `sndr trace list`).")} wide>
                    <TracesPanel />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "appearance",
              label: tr("Appearance"),
              icon: <Palette size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={tr("Appearance and Operator Mode")} icon={<Palette size={18} />} wide>
                    <AppearanceSettings settings={settings} onSettings={onSettings} />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "license",
              label: tr("License & modules"),
              icon: <BadgeCheck size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={tr("License & SNDR Engine")} icon={<BadgeCheck size={18} />} desc={tr("Active tier (community vs commercial SNDR Engine), the signed license token (subject / expiry / signature), whether the vllm.sndr_engine overlay is installed, and how many engine-tier patches it unlocks.")} wide>
                    <Suspense fallback={<SkeletonCards count={1} />}>
                      <LicensePanel />
                    </Suspense>
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "notifications",
              label: tr("Notifications"),
              icon: <Bell size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={tr("Alerts & notifications")} icon={<Bell size={18} />} desc={tr("Get a Telegram push when a managed engine container goes DOWN (crash / OOM / stop) or recovers. The daemon watches over the docker socket; gated behind apply.")} wide>
                    <NotificationSettings />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "api",
              label: tr("API & Schema"),
              icon: <Settings size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={tr("Daemon & Access")} icon={<Settings size={18} />} desc={tr("Daemon endpoint, OpenAPI and the optional access token for remote/tunnel use.")}>
                    <InfoRows rows={[
                      [tr("API Base"), apiBase],
                      ["OpenAPI", `${apiBase}/openapi.json`],
                      [tr("Mode"), tr("Read-only Product API")],
                      ["SNDR Core", environment?.sndr_core_version ?? "-"],
                      [tr("Frontend"), tr("Vite/React, served by daemon")]
                    ]} />
                    <ApiTokenField />
                  </ModuleCard>
                  <ModuleCard title={tr("API Tokens")} icon={<KeyRound size={18} />} desc={tr("Named, revocable Bearer tokens for programmatic / CI access (auth required). Plaintext shown once.")} wide>
                    <ApiTokenManager enabled={!!authUser} />
                  </ModuleCard>
                  <ModuleCard title={tr("Endpoint Explorer")} icon={<Code2 size={18} />} desc={tr("Send a live GET to any read-only Product API endpoint and inspect the JSON.")} wide>
                    <EndpointExplorer />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "audit",
              label: tr("Audit log"),
              icon: <FileText size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={tr("Audit log")} icon={<FileText size={18} />} desc={tr("Tamper-evident record of daemon events — auth, jobs, operations and system actions. Live.")} wide>
                    <AuditLogPanel />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "updates",
              label: tr("Updates"),
              icon: <DownloadCloud size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={tr("Updates")} icon={<DownloadCloud size={18} />} desc={tr("Pin-gated self-update for the GUI + sndr_core patcher. The vLLM pin only moves to a patcher-supported value; the server docker step stays manual. Apply is gated + confirmed.")} wide>
                    <UpdatesPanel />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "admin",
              label: tr("Admin"),
              icon: <KeyRound size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={tr("Admin Surface Matrix")} icon={<KeyRound size={18} />} desc={tr("Product API write/read surfaces and their status.")} wide>
                    <AdminSurfaceMatrix featureRows={featureRows} patchDoctor={patchDoctor} />
                  </ModuleCard>
                  <ModuleCard title={tr("Engine & Dependencies")} icon={<Cpu size={18} />} desc={tr("Versions and runtime tools on the daemon host.")}>
                    <EnvironmentPanel env={environment} />
                  </ModuleCard>
                  <ModuleCard title={tr("Feature Contracts")} icon={<ShieldCheck size={18} />} desc={tr("Capability inventory with live statuses.")}>
                    <CapabilityTable rows={featureRows} />
                  </ModuleCard>
                  {authUser && (
                    <ModuleCard title={tr("Account & Security")} icon={<ShieldCheck size={18} />} desc={tr("Your password and two-factor settings.")} wide>
                      <SecurityPanel user={authUser} onChanged={onAuthRefresh} />
                    </ModuleCard>
                  )}
                  {authUser?.role === "admin" && (
                    <ModuleCard title={tr("User Management")} icon={<KeyRound size={18} />} desc={tr("Create, list and remove accounts (admin).")} wide>
                      <UserAdminPanel currentUser={authUser} />
                    </ModuleCard>
                  )}
                </ModuleGrid>
              )
            },
            {
              id: "developer",
              label: tr("Developer"),
              icon: <SlidersHorizontal size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={tr("Config Draft and Diff")} icon={<SlidersHorizontal size={18} />} desc={`${tr("Local runtime draft for")} ${selectedPreset}.`} wide>
                    <ConfigDraftEditor
                      selectedPreset={selectedPreset}
                      composed={composed}
                      runtimeTarget={runtimeTarget}
                      patchPolicy={patchPolicy}
                    />
                  </ModuleCard>
                  <ModuleCard title={tr("CLI Mirror")} icon={<SquareTerminal size={18} />} desc={tr("Equivalent CLI for the current operator context.")}>
                    <CodeBlock lines={cliLines} />
                  </ModuleCard>
                  <ModuleCard title={tr("Live Events")} icon={<Activity size={18} />} desc={tr("Daemon event feed (jobs, lifecycle, reports).")}>
                    <EventLog events={events} />
                  </ModuleCard>
                </ModuleGrid>
              )
            }
          ]}
        />
      )}
    </section>
  );
}
