// SPDX-License-Identifier: Apache-2.0
// The section renderer. Given the active sectionId plus a snapshot of Product
// API state (passed as props), it renders that section's workspace: the heading
// and the per-section tabbed panels. Extracted verbatim from App.tsx so the app
// shell keeps only state/routing/composition and renders <SectionWorkspace/>
// inside one Suspense boundary. Pure presentation — no App-local coupling.
import { Suspense, useEffect, useMemo, useState } from "react";
import { Activity, AlertTriangle, BarChart3, Boxes, Clock3, Command, Cpu, FileText, Gauge, GitCompare, LayoutGrid, Link2, MessageSquare, Network, PackageCheck, Play, Rocket, Route, Server, ShieldCheck, SlidersHorizontal, Sparkles, SquareTerminal, Stethoscope, Table2, TimerReset } from "lucide-react";
import { tr } from "../i18n";
import type { ChatTarget } from "../Engine";
import type { ViewportTier } from "../hooks/useViewport";
import type { GuiSettings } from "../settings";
import type { Gate, RuntimeMode, SectionId } from "../nav";
import { api, AuthUser, BundleSpec, DiffUpstreamReport, DoctorReport, EnvironmentReport, HostProfile, PatchDoctorReport, PatchListResult, PresetExplainResult, PresetListResult, PresetRecord, ProductCapability, ProductOverview, ProofStatusReport, UserPresetList, V2ConfigCatalog, V2ConfigPreview } from "../api";
import { asNumber, asRecord, asText } from "../lib/coerce";
import { targetTitle } from "../lib/format";
import { runtimeHost } from "../lib/overview-presenters";
import { sectionSpec } from "../lib/section-spec";
import { AdvancedSection } from "./advanced-section";
import { PresetsSection } from "./presets-section";
import { OverviewSection } from "./overview-section";
import { ClientsSection } from "./clients-section";
import { PatchesSection } from "./patches-section";
import { CapabilityTable } from "../components/capability-table";
import { PercentBar } from "../components/charts";
import { CodeBlock } from "../components/code-block";
import { ModuleCard, ModuleGrid } from "../components/layout";
import { CompactList, InfoRows, type GateStatus } from "../components/primitives";
import { TabIntro, WorkflowSteps } from "../components/shell-bits";
import { TabbedSection } from "../components/tabbed-section";
import { toast } from "../components/toast";
import { FleetPanel } from "../Fleet";
import { SkeletonCards } from "../Skeleton";
import { ChatConsole, EngineBenchPanel, EngineMetricsPanel, EngineStatusCard, ConfigsSection, HostsSection, DeploymentConsole, ServiceLifecyclePlanner, KvCalcPanel, BaselinePanel, InstallWizard, CopilotPanel, ContainersPanel, VirtualizationPanel, HardwarePanel, RoutingPanel, FlagsPanel } from "../lazy-panels";
import { ReportGenerator } from "./api-explorer";
import { BenchmarkBaselinePanel, EvidenceRows } from "./bench";
import { CaveatsPanel } from "./diagnostics";
import { DoctorFindings, DoctorSummary } from "./doctor";
import { EnvironmentPanel } from "./environment";
import { GateRow } from "./gate-row";
import { QueueJobButton } from "./jobs";
import { ModelsWorkbench } from "./models-workbench";
import { EventLog } from "./operational-console";
import { OperationsConsole } from "./operations";
import { DoctorCoveragePanel } from "./patch-doctor";
import { ProofStatusPanel } from "./proof";
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
