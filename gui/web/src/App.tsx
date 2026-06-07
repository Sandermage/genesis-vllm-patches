import {
  Activity,
  AlertCircle,
  BarChart3,
  Bell,
  Box,
  ChevronRight,
  Clock3,
  Code2,
  Command,
  Cpu,
  Database,
  DownloadCloud,
  FileText,
  Gauge,
  GitBranch,
  GitCompare,
  HardDrive,
  Home,
  BadgeCheck,
  KeyRound,
  Layers3,
  Link2,
  LayoutGrid,
  Boxes,
  AlertTriangle,
  ListChecks,
  Monitor,
  MessageSquare,
  Network,
  PackageCheck,
  Palette,
  PanelLeft,
  Play,
  PlugZap,
  RefreshCw,
  Route,
  Rocket,
  Search,
  Server,
  Settings,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  SquareTerminal,
  Stethoscope,
  Table2,
  TimerReset,
  Terminal,
  Wrench
} from "lucide-react";
import { Component, Suspense, lazy, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { sectionFromHash, recordIdFromHash, buildHash, replaceHash } from "./route";
import { type SectionId, type RuntimeMode, type Gate } from "./nav";
import {
  type ConsoleTab, type AccentMode, type GuiSettings,
  nextTheme, themeLabel, themeIcon, VALID_THEMES
} from "./settings";
// useFetch now used only inside extracted sections.
import { asRecord, asText, asNumber, countRecord } from "./lib/coerce";
import { formatTokens, targetTitle } from "./lib/format";
// config-utils / runtime-draft / element-fields now used only inside extracted sections.
import { LayerEditor } from "./sections/layer-editor";
import { ConfigDraftEditor } from "./sections/config-draft-editor";
import { ModelsWorkbench } from "./sections/models-workbench";
// settings-panels lazy-loaded below.
// ConfigsSection lazy-loaded below.
import { CapabilityTable } from "./components/capability-table";
import { PlanChip, KeyValue, ArtifactPreview, type ArtifactTab } from "./components/display-bits";
import { Step, Metric, PanelHeader, TabIntro, CodeTabs } from "./components/shell-bits";
import { ServerSwitcher, ConnectionMap } from "./sections/connection-bar";
// HostsSection lazy-loaded below.
import { PresetSelectedView, PresetSummaryStrip } from "./sections/preset-views";
import { StatusBadge, StatusPill, InfoRows, CompactList, KpiGrid, type GateStatus } from "./components/primitives";
import { PercentBar, BarList, OvKpi } from "./components/charts";
import { CaveatsPanel, ConfigKeysPanel, TracesPanel } from "./sections/diagnostics";
import { DoctorSummary, DoctorFindings } from "./sections/doctor";
import { BundlesPanel, UpstreamDiffPanel } from "./sections/registry";
import { EnvironmentPanel } from "./sections/environment";
import { DoctorCoveragePanel, AdminSurfaceMatrix } from "./sections/patch-doctor";
import { BenchmarkBaselinePanel, EvidenceRows } from "./sections/bench";
import { AuditLogPanel } from "./sections/audit-log";
import { RuntimeEndpoint, BenchmarkCard, EvidenceCard, PatchMatrix, EndpointRows } from "./sections/rail-cards";
import { EndpointExplorer, ReportGenerator } from "./sections/api-explorer";
import { InfoDialog, ShortcutsModal } from "./components/dialogs";
import { RecommendationRow } from "./sections/recommendation-row";
import { ModuleGrid, ModuleCard } from "./components/layout";
import { toast, ToastHost } from "./components/toast";
import { OperationsConsole } from "./sections/operations";
// DeploymentConsole lazy-loaded below.
import { GateRow } from "./sections/gate-row";
import { SetupWizard } from "./sections/setup-wizard";
import { JobMonitorModal, QueueJobButton } from "./sections/jobs";
// ServiceLifecyclePlanner lazy-loaded below.
import { CommandPalette } from "./sections/command-palette";
import { EventLog, OperationalConsole } from "./sections/operational-console";
import { LaunchPanel } from "./sections/launch-panel";
import { PresetPolicyGraph } from "./sections/preset-insight";
import { PatchSummaryPanel, PatchLifecycleGraph, PatchRegistryInsight, PatchModelSupport } from "./sections/patch-overview";
// PatchInventoryControl lazy-loaded below.
import { PresetRecommendPanel } from "./sections/preset-recommend";
import { type RecommendForm, defaultRecommend, workloadChoices } from "./recommend";
import { TabbedSection } from "./components/tabbed-section";
import { PresetQuickPanel } from "./sections/preset-quick";
import { PresetCatalogTable } from "./sections/preset-catalog";
import { ProofStatusPanel } from "./sections/proof";
import { CodeBlock } from "./components/code-block";
// dialog helpers now used only inside extracted modals.
import { SkeletonCards } from "./Skeleton";
import {
  BundleSpec,
  DiffUpstreamReport,
  DoctorReport,
  EnvironmentReport,
  HostProfile,
  Job,
  BackendEvent,
  LaunchPlanResult,
  ProofStatusReport,
  UserPresetList,
  V2ConfigCatalog,
  V2ConfigPreview,
  PresetExplainResult,
  PatchDoctorReport,
  PatchListResult,
  PresetListResult,
  PresetRecord,
  PresetRecommendResult,
  ProductCapability,
  ProductOverview,
  AuthStatus,
  AuthUser,
  api,
  getApiToken,
  normalizeBaseUrl,
  hostLabel
} from "./api";
import { AccountMenu, LoginScreen, SecurityPanel, UserAdminPanel } from "./Auth";
// Engine/Planner/Copilot/Installer components only render inside SectionWorkspace
// (non-default sections), so they are code-split out of the initial bundle and
// fetched on first visit. One Suspense boundary around SectionWorkspace covers
// them all. `ChatTarget` is a type, so it stays a plain type import.
import type { ChatTarget } from "./Engine";
import { AlertsBell } from "./Alerts";
const ChatConsole = lazy(() => import("./Engine").then((m) => ({ default: m.ChatConsole })));
const EngineBenchPanel = lazy(() => import("./Engine").then((m) => ({ default: m.EngineBenchPanel })));
const EngineMetricsPanel = lazy(() => import("./Engine").then((m) => ({ default: m.EngineMetricsPanel })));
const EnginePlayground = lazy(() => import("./Engine").then((m) => ({ default: m.EnginePlayground })));
const EngineStatusCard = lazy(() => import("./Engine").then((m) => ({ default: m.EngineStatusCard })));

// Heavy tab-content sections — only rendered for their own section, so they are
// code-split out of the initial bundle and fetched on first visit. The Suspense
// boundary around SectionWorkspace covers them all. (configs-workbench alone is
// ~1000 lines; this is the bulk of the initial-bundle saving.)
const ConfigsSection = lazy(() => import("./sections/configs-workbench").then((m) => ({ default: m.ConfigsSection })));
const HostsSection = lazy(() => import("./sections/hosts-section").then((m) => ({ default: m.HostsSection })));
const DeploymentConsole = lazy(() => import("./sections/deployment").then((m) => ({ default: m.DeploymentConsole })));
const ServiceLifecyclePlanner = lazy(() => import("./sections/services").then((m) => ({ default: m.ServiceLifecyclePlanner })));
const PatchInventoryControl = lazy(() => import("./sections/patch-inventory").then((m) => ({ default: m.PatchInventoryControl })));
const ApiTokenManager = lazy(() => import("./sections/settings-panels").then((m) => ({ default: m.ApiTokenManager })));
const NotificationSettings = lazy(() => import("./sections/settings-panels").then((m) => ({ default: m.NotificationSettings })));
const AppearanceSettings = lazy(() => import("./sections/settings-panels").then((m) => ({ default: m.AppearanceSettings })));
const ApiTokenField = lazy(() => import("./sections/settings-panels").then((m) => ({ default: m.ApiTokenField })));
// ModelManagementPanel is lazy-loaded inside ./sections/models-workbench now.
// Lazy: the xterm-based terminal is heavy and rarely opened — keep it out of the
// initial bundle so the app loads fast; the chunk fetches only when a terminal opens.
// TerminalModal is lazy-loaded inside ./sections/hosts-section now.
import { UpdatesPanel } from "./Updates";
const KvCalcPanel = lazy(() => import("./Planner").then((m) => ({ default: m.KvCalcPanel })));
const BaselinePanel = lazy(() => import("./Planner").then((m) => ({ default: m.BaselinePanel })));
const InstallWizard = lazy(() => import("./Installer").then((m) => ({ default: m.InstallWizard })));
const CopilotPanel = lazy(() => import("./Copilot").then((m) => ({ default: m.CopilotPanel })));
import { FleetPanel } from "./Fleet";
// Lazy-loaded: the container management UI (~1.2k lines) only renders on the
// Containers section, so it is code-split out of the initial bundle.
const ContainersPanel = lazy(() => import("./Containers").then((m) => ({ default: m.ContainersPanel })));
// Lazy-loaded: the GPU/hardware telemetry dashboard only renders on its section.
const HardwarePanel = lazy(() => import("./Hardware").then((m) => ({ default: m.HardwarePanel })));
const RoutingPanel = lazy(() => import("./Routing").then((m) => ({ default: m.RoutingPanel })));
const FlagsPanel = lazy(() => import("./Flags").then((m) => ({ default: m.FlagsPanel })));
const LicensePanel = lazy(() => import("./License").then((m) => ({ default: m.LicensePanel })));

type LoadState = "idle" | "loading" | "ready" | "error";
// GateStatus is now owned by ./components/primitives (re-imported above) so the
// RailCheck primitive and the gate logic share one source of truth.
// RuntimeMode moved to ./nav.
// SectionId / RuntimeMode / Gate / GATE_TARGET now live in ./nav (imported above).
// ArtifactTab moved to ./components/display-bits.
// ConsoleTab / ThemeMode / DensityMode / AccentMode / DetailMode / GuiSettings +
// theme helpers (nextTheme/themeLabel/themeIcon/THEME_CYCLE/VALID_THEMES) now
// live in ./settings (imported above).

// RecommendForm / defaultRecommend / workloadChoices moved to ./recommend (imported above).

type NavItem = {
  id: SectionId;
  icon: ReactNode;
  label: string;
};

// Gate type moved to ./nav.

// RuntimeConfigDraft moved to ./lib/runtime-draft.


const GUI_SETTINGS_STORAGE_KEY = "sndr.gui.settings";
const AUTO_REFRESH_INTERVAL_MS = 20_000;

const defaultGuiSettings: GuiSettings = {
  theme: "light",
  density: "comfortable",
  accent: "teal",
  detailMode: "engineer",
  showConnectionMap: true,
  autoRefresh: false,
  sidebarCollapsed: false
};


// Sidebar grouped into a logical workflow: see → your servers → define what to
// run → deploy it → use it → prove it → tools. Each group renders under a small
// header so related sections sit together instead of in one long scatter.
type NavGroup = { label?: string; items: NavItem[] };
const navGroups: NavGroup[] = [
  { items: [
    { id: "overview", icon: <Home size={17} />, label: "Overview" },
  ] },
  { label: "Infrastructure", items: [
    { id: "hosts", icon: <LayoutGrid size={17} />, label: "Fleet" },
    { id: "containers", icon: <Boxes size={17} />, label: "Containers" },
    { id: "hardware", icon: <Cpu size={17} />, label: "Hardware" },
    { id: "setup", icon: <Settings size={17} />, label: "Setup" },
  ] },
  { label: "Models & Config", items: [
    { id: "models", icon: <Box size={17} />, label: "Models" },
    { id: "presets", icon: <Database size={17} />, label: "Presets" },
    { id: "configs", icon: <SlidersHorizontal size={17} />, label: "Configs" },
    { id: "planner", icon: <Gauge size={17} />, label: "Planner" },
  ] },
  { label: "Deploy", items: [
    { id: "launch-plan", icon: <Rocket size={17} />, label: "Launch Plan" },
    { id: "services", icon: <Network size={17} />, label: "Services" },
  ] },
  { label: "Engine", items: [
    { id: "chat", icon: <MessageSquare size={17} />, label: "Chat" },
    { id: "routing", icon: <Route size={17} />, label: "Routing" },
    { id: "clients", icon: <Link2 size={17} />, label: "Clients" },
  ] },
  { label: "Validate", items: [
    { id: "doctor", icon: <ShieldCheck size={17} />, label: "Doctor" },
    { id: "patches", icon: <Wrench size={17} />, label: "Patches" },
    { id: "benchmarks", icon: <BarChart3 size={17} />, label: "Benchmarks" },
    { id: "evidence", icon: <FileText size={17} />, label: "Evidence" },
    { id: "reports", icon: <Table2 size={17} />, label: "Reports" },
  ] },
  { label: "Tools", items: [
    { id: "copilot", icon: <Sparkles size={17} />, label: "Copilot" },
    { id: "advanced", icon: <SlidersHorizontal size={17} />, label: "Advanced" },
  ] },
];
// Flat list (command palette / lookups) — preserves the grouped order.
const navItems: NavItem[] = navGroups.flatMap((g) => g.items);

// Hash-routing helpers (sectionFromHash / recordIdFromHash / buildHash /
// replaceHash) live in ./route so ContainersPanel can share them without a
// circular import. sectionFromHash returns a plain string; callers cast to
// SectionId after the SECTION_IDS validation guarantees membership.

// useFetch + FetchState extracted to ./hooks/useFetch (modularization slice).

export default function App() {
  const [overview, setOverview] = useState<ProductOverview | null>(null);
  const [presets, setPresets] = useState<PresetListResult | null>(null);
  const [patches, setPatches] = useState<PatchListResult | null>(null);
  const [patchDoctor, setPatchDoctor] = useState<PatchDoctorReport | null>(null);
  const [configCatalog, setConfigCatalog] = useState<V2ConfigCatalog | null>(null);
  const [configPreview, setConfigPreview] = useState<V2ConfigPreview | null>(null);
  const [bundles, setBundles] = useState<BundleSpec[]>([]);
  const [diffUpstream, setDiffUpstream] = useState<DiffUpstreamReport | null>(null);
  const [proofStatus, setProofStatus] = useState<ProofStatusReport | null>(null);
  const [userPresets, setUserPresets] = useState<UserPresetList | null>(null);
  const [doctorReport, setDoctorReport] = useState<DoctorReport | null>(null);
  const [environment, setEnvironment] = useState<EnvironmentReport | null>(null);
  const [hostProfiles, setHostProfiles] = useState<HostProfile[]>([]);
  const [selectedPreset, setSelectedPreset] = useState<string>(() => recordIdFromHash() ?? "prod-35b-multiconc");
  const [explain, setExplain] = useState<PresetExplainResult | null>(null);
  const [launchPlan, setLaunchPlan] = useState<LaunchPlanResult | null>(null);
  const [recommend, setRecommend] = useState<PresetRecommendResult | null>(null);
  const [state, setState] = useState<LoadState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [apiBase, setApiBase] = useState(api.baseUrl);
  const [runtimeMode, setRuntimeMode] = useState<RuntimeMode>("remote");
  const [runtimeTarget, setRuntimeTarget] = useState("docker_compose");
  const [patchPolicy, setPatchPolicy] = useState("safe");
  const [applyEnabled, setApplyEnabled] = useState(false);
  const [authState, setAuthState] = useState<AuthStatus | null>(null);
  const [launchConfirm, setLaunchConfirm] = useState(false);
  const [launchJob, setLaunchJob] = useState<Job | null>(null);
  const [launchBusy, setLaunchBusy] = useState(false);
  const [launchSshTarget, setLaunchSshTarget] = useState("");
  const [launchTab, setLaunchTab] = useState("recommend");
  const [recommendForm, setRecommendForm] = useState<RecommendForm>(defaultRecommend);
  const [activeSection, setActiveSection] = useState<SectionId>(() => (sectionFromHash() as SectionId | null) ?? "launch-plan");
  // When a host is opened from the connection switcher, focus + auto-discover it.
  const [focusHostId, setFocusHostId] = useState<string | null>(null);
  // "Set up as node" from an engine card → prefill the installer (daemon target).
  const [installIntent, setInstallIntent] = useState<{ hostId?: string; target?: string } | null>(null);
  const [artifactTab, setArtifactTab] = useState<ArtifactTab>("compose");
  const [consoleTab, setConsoleTab] = useState<ConsoleTab>("jobs");
  const [dialog, setDialog] = useState<string | null>(null);
  const [monitorJobId, setMonitorJobId] = useState<string | null>(null);
  const [jobActionBusy, setJobActionBusy] = useState<"" | "bench" | "evidence">("");
  const [commandOpen, setCommandOpen] = useState(false);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  // Timestamp of the last bare `g` keypress — drives the `g <key>` two-stroke
  // navigation chords (g p → Presets, etc.) within a short window.
  const gChordRef = useRef(0);
  const [chatTarget, setChatTarget] = useState<ChatTarget | null>(null);
  const [settings, setSettings] = useState<GuiSettings>(() => loadGuiSettings());

  async function loadAll() {
    setState("loading");
    setError(null);
    // Kick off the auxiliary surfaces (hosts/env/bundles/…) in PARALLEL with the
    // critical path — they're independent, so overlapping them shaves the startup
    // wall-clock instead of running them after the dashboard is ready.
    void loadAux();
    try {
      const [overviewData, presetData, recommendationData, patchData, doctorData, configCatalogData] = await Promise.all([
        api.overview(),
        api.presets({}),
        api.recommendPresets(recommendForm),
        api.patches({}),
        api.patchDoctor(),
        api.v2ConfigCatalog()
      ]);
      setOverview(overviewData);
      setPresets(presetData);
      setRecommend(recommendationData);
      setPatches(patchData);
      setPatchDoctor(doctorData);
      setConfigCatalog(configCatalogData);

      // A deep-link (#presets?id=…) wins over the recommendation default so a
      // shared/bookmarked link lands on the exact preset it points at.
      const hashId = recordIdFromHash();
      const nextPreset =
        (hashId && presetData.presets.some((preset) => preset.id === hashId) ? hashId : null) ??
        recommendationData.results[0]?.id ??
        overviewData.catalog.default_presets[0] ??
        presetData.presets.find((preset) => preset.has_card)?.id ??
        presetData.presets[0]?.id ??
        selectedPreset;
      setSelectedPreset(nextPreset);
      const nextRecord =
        presetData.presets.find((preset) => preset.id === nextPreset) ?? null;
      const [nextExplain, nextLaunchPlan] = await Promise.all([
        api.explainPreset(nextPreset),
        api.launchPlan({
          preset_id: nextPreset,
          runtime_target: runtimeTarget,
          patch_policy: patchPolicy,
          host: runtimeHost(runtimeMode),
          mode: runtimeMode
        })
      ]);
      setExplain(nextExplain);
      setLaunchPlan(nextLaunchPlan);
      setConfigPreview(
        await api.v2ConfigPreview({
          model_id:
            nextLaunchPlan.summary.model ?? nextRecord?.model ?? "qwen3.6-35b-a3b-fp8",
          hardware_id:
            nextLaunchPlan.summary.hardware ?? nextRecord?.hardware ?? recommendForm.hardware,
          profile_id: nextLaunchPlan.summary.profile ?? undefined,
          runtime: "docker"
        })
      );
      setState("ready");
    } catch (err) {
      setState("error");
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function loadAux() {
    // Secondary, best-effort surfaces. A failure here must not blank the
    // primary dashboard, so it never flips the global error state.
    try {
      const [bundleData, diffData, proofData, userPresetData, doctorData, envData, hostData] = await Promise.all([
        api.bundles(),
        api.diffUpstream(),
        api.proofStatus(),
        api.userPresets(),
        api.doctor(),
        api.environment(),
        api.hosts()
      ]);
      setBundles(bundleData.bundles);
      setDiffUpstream(diffData);
      setProofStatus(proofData);
      setUserPresets(userPresetData);
      setDoctorReport(doctorData);
      setEnvironment(envData);
      setHostProfiles(hostData.hosts);
    } catch {
      // Auxiliary capability data is optional.
    }
  }

  async function refreshUserPresets() {
    try {
      setUserPresets(await api.userPresets());
    } catch {
      // Optional surface.
    }
  }

  async function refreshHosts() {
    try {
      setHostProfiles((await api.hosts()).hosts);
    } catch {
      // Optional surface.
    }
  }

  async function loadExplain(id: string) {
    setSelectedPreset(id);
    setError(null);
    try {
      const [nextExplain, nextLaunchPlan] = await Promise.all([
        api.explainPreset(id),
        api.launchPlan({
          preset_id: id,
          runtime_target: runtimeTarget,
          patch_policy: patchPolicy,
          host: runtimeHost(runtimeMode),
          mode: runtimeMode
        })
      ]);
      setExplain(nextExplain);
      setLaunchPlan(nextLaunchPlan);
      setConfigPreview(
        await api.v2ConfigPreview({
          model_id: nextLaunchPlan.summary.model,
          hardware_id: nextLaunchPlan.summary.hardware,
          profile_id: nextLaunchPlan.summary.profile ?? undefined,
          runtime: "docker"
        })
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function refreshLaunchPlan(
    id: string = selectedPreset,
    target: string = runtimeTarget,
    policy: string = patchPolicy,
    mode: RuntimeMode = runtimeMode
  ) {
    if (!id) return;
    setError(null);
    try {
      setLaunchPlan(
        await api.launchPlan({
          preset_id: id,
          runtime_target: target,
          patch_policy: policy,
          host: runtimeHost(mode),
          mode
        })
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  useEffect(() => {
    let cancelled = false;
    api.authStatus().then((s) => {
      if (cancelled) return;
      setAuthState(s);
      setApplyEnabled(s.apply_enabled);
    }).catch(() => {
      // Status endpoint unreachable — fall back to an open profile so the app
      // still attempts to load rather than hanging on a blank gate.
      if (!cancelled) setAuthState({
        auth_required: false, apply_enabled: false, backends: ["local"],
        oauth_providers: [], context: { in_container: false, system_user: "", pam_enabled: false }, user: null
      });
    });
    return () => { cancelled = true; };
  }, [apiBase]);

  const refreshAuth = () => {
    api.authStatus().then((s) => { setAuthState(s); setApplyEnabled(s.apply_enabled); }).catch(() => {});
  };
  const onAuthenticated = () => {
    refreshAuth();
  };

  async function runLaunchApply() {
    setLaunchBusy(true);
    setError(null);
    try {
      const sshTarget = launchSshTarget.trim();
      const job = await api.launchApply({
        preset_id: selectedPreset,
        runtime_target: runtimeTarget,
        host: runtimeMode === "remote" ? "gpu-build-01" : "127.0.0.1",
        transport: sshTarget ? "ssh" : "local",
        ssh_target: sshTarget,
        confirm: launchConfirm
      });
      setLaunchJob(job);
      // Open the live job monitor so the operator watches the runtime come up
      // (the modal polls until terminal — completes the Observe loop).
      setMonitorJobId(job.job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLaunchBusy(false);
    }
  }

  async function runBenchJob() {
    setJobActionBusy("bench");
    setError(null);
    try {
      const job = await api.benchRun({ preset_id: selectedPreset });
      setMonitorJobId(job.job_id);
      toast(`Benchmark queued for ${selectedPreset}`, "success");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      toast("Benchmark failed to queue", "error");
    } finally {
      setJobActionBusy("");
    }
  }

  async function runEvidenceJob() {
    setJobActionBusy("evidence");
    setError(null);
    try {
      const job = await api.evidenceAttach({ preset_id: selectedPreset });
      setMonitorJobId(job.job_id);
      toast(`Evidence attach queued for ${selectedPreset}`, "success");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      toast("Evidence attach failed to queue", "error");
    } finally {
      setJobActionBusy("");
    }
  }

  async function runRecommend(nextForm: RecommendForm = recommendForm) {
    setError(null);
    try {
      const nextRecommendation = await api.recommendPresets(nextForm);
      setRecommend(nextRecommendation);
      const first = nextRecommendation.results[0]?.id;
      if (first) {
        await loadExplain(first);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  // Re-point the whole GUI at a different daemon (saved-server switch).
  function switchServer(baseUrl: string) {
    const next = api.setBaseUrl(normalizeBaseUrl(baseUrl));
    setApiBase(next);
    setError(null);
    void loadAll();
  }

  // Probe for an actual SNDR daemon (Product API) at a base URL. A GPU box
  // usually runs the vLLM *engine* (8101/8102), not the daemon (8765) — so we
  // must never re-point the GUI at it blindly or the whole UI blanks out.
  async function probeDaemon(url: string): Promise<boolean> {
    try {
      const ctrl = new AbortController();
      const t = window.setTimeout(() => ctrl.abort(), 3000);
      // /api/v1/health is auth-exempt, so DON'T send an Authorization header: it
      // would turn the cross-origin probe into a non-simple request needing a CORS
      // preflight (OPTIONS), which a remote daemon may not answer — making a live
      // daemon look unreachable. A plain GET to the public health route is enough.
      const res = await fetch(`${url}/api/v1/health`, { signal: ctrl.signal });
      window.clearTimeout(t);
      return res.ok;
    } catch { return false; }
  }

  // Guarded switch used by the connection dropdown: the local daemon always
  // connects; any remote target is probed first so an engine-only host can't
  // strand the GUI in a blank, daemon-down state.
  async function switchServerGuarded(baseUrl: string) {
    const url = normalizeBaseUrl(baseUrl);
    if (url === normalizeBaseUrl(window.location.origin) || await probeDaemon(url)) {
      switchServer(url);
      return;
    }
    toast(`No SNDR daemon at ${hostLabel(url)}. If that host runs the vLLM engine, use its card's SSH / Discover / Terminal / Chat instead — not a daemon connection.`, "error");
  }

  // Self-heal a stale/bad daemon address. The GUI is served by a daemon at the
  // page origin (it's provably up — it served this page), so if the configured
  // apiBase points elsewhere and is unreachable, fall back to the local daemon
  // automatically instead of stranding the operator on the daemon-down screen.
  // (A common cause: an old localStorage value pointing at a GPU/engine host.)
  const autoRecoveredRef = useRef<string | null>(null);
  useEffect(() => {
    if (state !== "error" || typeof window === "undefined") return;
    const origin = normalizeBaseUrl(window.location.origin);
    const cur = normalizeBaseUrl(apiBase);
    if (cur === origin || autoRecoveredRef.current === cur) return;  // already local / already tried
    autoRecoveredRef.current = cur;
    void (async () => {
      if (await probeDaemon(origin)) {
        toast(`Daemon at ${hostLabel(cur)} is unreachable — switched back to the local daemon.`, "info");
        switchServer(origin);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state, apiBase]);

  // A host card can drive the rest of the app: open the chat against its engine,
  // or register its daemon in the server switcher.
  function chatWithHost(profile: HostProfile) {
    // Pass the hostId — the daemon resolves a key-protected engine's bearer
    // server-side from the encrypted secret (the raw key never reaches the browser).
    setChatTarget({ host: profile.host, port: profile.engine_port || 8000, hostId: profile.id, nonce: Date.now() });
    setActiveSection("chat");
    toast(`Chat → ${profile.label} (${profile.host}:${profile.engine_port || 8000})`, "info");
  }
  // Returns true if a real daemon was found and the GUI re-pointed at it; false
  // if the host has no daemon (an engine box). The caller (host card) reflects a
  // false result by disabling its button — no error toast, the label explains.
  async function addHostAsServer(profile: HostProfile): Promise<boolean> {
    const port = profile.port || 8765;
    const url = normalizeBaseUrl(`http://${profile.host}:${port}`);
    // Probe for an actual SNDR daemon before re-pointing the GUI — a GPU server
    // usually runs the engine, not the daemon, so switching would blank the UI.
    if (!(await probeDaemon(url))) {
      // Explicit feedback (was previously silent → looked like the button did
      // nothing): say WHERE we looked and the likely fixes.
      toast(
        `No SNDR daemon reachable at ${profile.host}:${port}. ` +
        `Check the daemon port (8765 by default, not the engine port ${profile.engine_port || 8000}), ` +
        `that the node daemon is running, and that it allows this origin — or "Set up as node" below.`,
        "error",
      );
      return false;
    }
    switchServer(url);  // the host is already in the registry — just connect
    toast(`Connected to ${profile.label} daemon (${url})`, "success");
    return true;
  }

  function selectWorkload(workload: string) {
    const nextForm = { ...recommendForm, workload };
    setRecommendForm(nextForm);
    void runRecommend(nextForm);
  }

  // Global keyboard shortcuts: ⌘K palette, `?` help overlay, and GitHub/Linear
  // -style `g <key>` navigation chords. All chords/single keys are suppressed
  // while typing in a field so they never eat real input.
  useEffect(() => {
    // section ← chord key (second stroke after `g`).
    const G_NAV: Record<string, SectionId> = {
      o: "overview", s: "setup", f: "fleet", h: "hosts", m: "models",
      c: "configs", p: "presets", n: "containers", d: "doctor",
      l: "launch-plan", b: "benchmarks",
    };
    const onKey = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setCommandOpen((open) => !open);
        return;
      }
      if (event.key === "Escape") {
        setCommandOpen(false);
        setShortcutsOpen(false);
        return;
      }
      const target = event.target as HTMLElement | null;
      const typing = !!target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable);
      if (typing || event.metaKey || event.ctrlKey || event.altKey) return;
      if (event.key === "?") {
        event.preventDefault();
        setShortcutsOpen((open) => !open);
        return;
      }
      const now = Date.now();
      if (event.key === "g") {
        gChordRef.current = now; // arm the chord
        return;
      }
      if (now - gChordRef.current < 1200) {
        const dest = G_NAV[event.key.toLowerCase()];
        if (dest) {
          event.preventDefault();
          setActiveSection(dest);
        }
        gChordRef.current = 0; // consume (success or miss)
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // URL deep-linking: keep the location hash in sync with the active route so
  // every view is bookmarkable/shareable and survives a page refresh.
  //   - On a SECTION change we write the canonical hash for the new section
  //     (with ?id= for the preset view) via location.hash, which pushes a
  //     history entry — so browser Back/Forward walks the section history.
  //   - While STAYING on the Presets view, a preset change is mirrored with
  //     replaceState so clicking through presets doesn't flood the Back stack.
  //   - Other sections own their own deep-link params (e.g. ContainersPanel
  //     manages #containers?c=…&src=…); we deliberately leave their params
  //     untouched here so we never clobber a container deep-link on load.
  // The hashchange listener below closes the loop.
  useEffect(() => {
    const currentSection = sectionFromHash();
    if (currentSection !== activeSection) {
      // Section changed → write the canonical hash for the new section (push).
      window.location.hash = buildHash(activeSection, activeSection === "presets" ? { id: selectedPreset } : undefined);
      return;
    }
    if (activeSection === "presets") {
      const desired = buildHash("presets", { id: selectedPreset });
      if (window.location.hash.replace(/^#\/?/, "") !== desired) replaceHash(desired);
    }
  }, [activeSection, selectedPreset]);

  // Browser Back/Forward (and manual hash edits) drive the active section and,
  // on the Presets view, the selected preset. Guarded against the writer above.
  useEffect(() => {
    const onHashChange = () => {
      const next = sectionFromHash();
      if (next) setActiveSection(next as SectionId);
      const rec = recordIdFromHash();
      if (rec && next === "presets" && rec !== selectedPreset) {
        void loadExplain(rec); // restores selection + explain payload together
      }
    };
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedPreset]);

  useEffect(() => {
    // Defer the initial data load until auth status is known. When auth is
    // required and there is no session, the login gate is shown and loading
    // waits — this avoids a burst of 401s before sign-in. After login,
    // authState.user changes and this effect re-runs.
    if (authState === null) return;
    if (authState.auth_required && !authState.user) return;
    void loadAll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [authState?.auth_required, authState?.user?.username]);

  useEffect(() => {
    window.localStorage.setItem(GUI_SETTINGS_STORAGE_KEY, JSON.stringify(settings));
    document.documentElement.dataset.theme = settings.theme;
    document.documentElement.dataset.density = settings.density;
    document.documentElement.dataset.accent = settings.accent;
  }, [settings]);

  useEffect(() => {
    if (state === "ready") {
      void refreshLaunchPlan();
    }
    // Refresh only when launch inputs change; refreshLaunchPlan reads the
    // latest state and would make the dependency list unstable.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runtimeMode, runtimeTarget, patchPolicy]);

  useEffect(() => {
    if (!settings.autoRefresh) return;
    const id = window.setInterval(() => {
      void loadAll();
    }, AUTO_REFRESH_INTERVAL_MS);
    return () => window.clearInterval(id);
    // Re-arm the interval only when the toggle changes; loadAll reads the
    // latest state through its closure.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settings.autoRefresh]);

  function updateSettings(patch: Partial<GuiSettings>) {
    setSettings((current) => ({ ...current, ...patch }));
  }

  const selectedPresetRecord = useMemo(
    () => presets?.presets.find((preset) => preset.id === selectedPreset) ?? null,
    [presets, selectedPreset]
  );

  const card = (explain?.card ?? selectedPresetRecord?.card ?? {}) as Record<string, unknown>;
  const composed = (explain?.composed ?? {}) as Record<string, unknown>;
  const metric = asRecord(card.primary_metric);
  const evidenceRefs = Array.isArray(card.evidence_refs) ? card.evidence_refs : [];
  const primaryMetricValue = asNumber(metric.value);
  const primaryMetricKind = asText(metric.kind, "Metric");
  const visibility = asText(card.evidence_visibility, "unknown");
  const planSummary = launchPlan?.summary ?? {};

  const filteredPresets = useMemo(() => {
    const rows = presets?.presets ?? [];
    const needle = query.trim().toLowerCase();
    if (!needle) return rows;
    return rows.filter((row) =>
      [row.id, row.model, row.hardware, row.profile, row.card?.title, row.card?.routing_family]
        .filter(Boolean)
        .some((value) => String(value).toLowerCase().includes(needle))
    );
  }, [presets, query]);

  const hardwareOptions = useMemo(() => {
    const values = new Set<string>();
    (presets?.presets ?? []).forEach((preset) => values.add(preset.hardware));
    values.add(defaultRecommend.hardware);
    return Array.from(values).sort();
  }, [presets]);

  const runtimeTargets = overview?.capabilities.runtime_targets ?? [];
  const featureRows = overview?.capabilities.features ?? [];
  const patchRows = patches?.patches ?? [];
  const patchSummary = patches?.summary ?? null;
  const gates = useMemo(
    () =>
      launchPlan?.gates.map((gate) => ({
        id: gate.id,
        label: gate.title,
        detail: gate.detail,
        status: gate.status,
        action: gate.action
      })) ??
      buildReadinessGates({
        overview,
        runtimeTarget,
        selectedPresetRecord,
        explain
      }),
    [launchPlan, overview, runtimeTarget, selectedPresetRecord, explain]
  );
  const gateCounts = countGates(gates);
  const authReady = authState !== null && (!authState.auth_required || Boolean(authState.user));
  const liveEvents = useLiveEvents(apiBase, authReady);
  const events = buildEvents({ state, error, selectedPreset, runtimeTarget, visibility, live: liveEvents });
  const cliLines = launchPlan?.cli_mirror ?? buildCliMirror({
    selectedPreset,
    runtimeTarget,
    patchPolicy,
    recommendForm,
    apiBase
  });
  const planId = launchPlan?.plan_id ?? `plan_${selectedPreset.replace(/[^a-zA-Z0-9]+/g, "_")}`;
  const endpointHost = runtimeMode === "remote" ? "gpu-build-01" : "127.0.0.1";
  const connectionTone: "success" | "warning" | "danger" =
    state === "error" ? "danger" : state === "loading" ? "warning" : "success";
  const connectionLabel =
    state === "error" ? "Disconnected" : state === "loading" ? "Connecting" : "Connected";
  const presetCount = overview?.catalog.presets_count ?? presets?.presets.length ?? 0;
  const engineReady = overview?.capabilities.platform.engine_installed ?? false;
  const shellClass = [
    "desktop-shell",
    `theme-${settings.theme}`,
    `density-${settings.density}`,
    `accent-${settings.accent}`,
    `detail-${settings.detailMode}`,
    settings.showConnectionMap ? "map-on" : "map-off"
  ].join(" ");

  // Hold the shell until auth status is known, so child panels don't fire a
  // burst of requests (and 401s) before we know whether a login is required.
  if (authState === null) {
    return (
      <main className={shellClass}>
        <div className="login-backdrop">
          <div className="auth-loading"><span className="auth-spinner" /> Connecting…</div>
        </div>
      </main>
    );
  }

  // Auth gate: when the daemon requires authentication and there is no active
  // session, the whole shell is replaced by the sign-in screen.
  if (authState?.auth_required && !authState.user) {
    return (
      <main className={shellClass}>
        <LoginScreen status={authState} onAuthenticated={onAuthenticated} />
      </main>
    );
  }

  return (
    <main className={shellClass}>
      <aside className={`sidebar${settings.sidebarCollapsed ? " collapsed" : ""}`}>
        <div className="brand-row">
          <div className="brand-mark">S</div>
          <div>
            <strong>SNDR</strong>
            <span>Control Center</span>
          </div>
          <small>v{overview?.capabilities.platform.sndr_core_version ?? environment?.sndr_core_version ?? "—"}</small>
          <button
            className="sidebar-toggle"
            title={settings.sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            aria-label={settings.sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            aria-pressed={settings.sidebarCollapsed}
            onClick={() => updateSettings({ sidebarCollapsed: !settings.sidebarCollapsed })}
          >
            <PanelLeft size={16} />
          </button>
        </div>

        <nav className="side-nav" aria-label="SNDR sections">
          {navGroups.map((group, gi) => (
            <div className="side-nav-group" key={group.label ?? `g${gi}`}>
              {group.label && <div className="side-nav-header">{group.label}</div>}
              {group.items.map((item) => (
                <button
                  className={activeSection === item.id ? "active" : ""}
                  key={item.id}
                  onClick={() => setActiveSection(item.id)}
                  title={settings.sidebarCollapsed ? item.label : undefined}
                  aria-label={item.label}
                >
                  {item.icon}
                  <span>{item.label}</span>
                </button>
              ))}
            </div>
          ))}
        </nav>

        <div className="daemon-card" title={`API Daemon · ${apiBase} · SNDR Core v${environment?.sndr_core_version ?? "—"} · ${runtimeMode === "remote" ? "Remote Desktop" : "Local Server"}`}>
          <div className="daemon-line">
            <span className="live-dot" />
            <strong>API Daemon</strong>
            <StatusBadge status={state === "error" ? "missing" : state === "loading" ? "partial" : "available"} />
          </div>
          <p>{apiBase}</p>
          <div className="daemon-meta">
            <span>SNDR Core v{environment?.sndr_core_version ?? "—"}</span>
            <span>Engine: {environment ? (environment.engine_version ? `vLLM ${environment.engine_version}` : "vLLM not installed") : "…"}</span>
            <span>Mode: {runtimeMode === "remote" ? "Remote Desktop" : "Local Server"} · Read-only</span>
          </div>
          <button className="ghost-button daemon-docs" title="Open API docs" onClick={() => window.open(`${apiBase.replace(/\/$/, "")}/docs`, "_blank", "noopener,noreferrer")}>
            <FileText size={14} /> <span className="daemon-docs-label">View API Docs</span>
          </button>
        </div>
      </aside>

      <section className="main-shell">
        <header className="topbar">
          <div className="topbar-left">
            <div className="mode-toggle" aria-label="GPU target" title="Where the GPU engine runs (launch/SSH/chat host hints). To change which daemon the GUI connects to, use the server switcher on the right.">
              <button
                className={runtimeMode === "local" ? "active" : ""}
                onClick={() => setRuntimeMode("local")}
                title="GPU target: this machine"
              >
                <Monitor size={16} />
                Local GPU
              </button>
              <button
                className={runtimeMode === "remote" ? "active" : ""}
                onClick={() => setRuntimeMode("remote")}
                title="GPU target: a remote node (via SSH)"
              >
                <ShieldCheck size={16} />
                Remote GPU
              </button>
            </div>
            <StatusPill tone={connectionTone}>{connectionLabel}</StatusPill>
            <span className="host-title">{endpointHost}</span>
            <span className="host-spec">{selectedPresetRecord?.hardware ?? recommendForm.hardware}</span>
            <span className="host-spec">{engineReady ? "vLLM engine ready" : "Engine not installed"}</span>
          </div>

          <div className="topbar-actions">
            <AlertsBell onOpenHardware={() => setActiveSection("hardware")} />
            <ServerSwitcher apiBase={apiBase} connectionTone={connectionTone} onSwitch={(url) => void switchServerGuarded(url)} hostProfiles={hostProfiles} onManageHosts={() => setActiveSection("hosts")} onOpenHost={(id) => { setFocusHostId(id); setActiveSection("hosts"); }} />
            <button className="tool-button" onClick={() => void loadAll()}>
              <RefreshCw size={16} />
              Sync Catalog
            </button>
            <button
              className="tool-button"
              title={`Theme: ${themeLabel(settings.theme)} — switch to ${themeLabel(nextTheme(settings.theme))}`}
              onClick={() => updateSettings({ theme: nextTheme(settings.theme) })}
            >
              {themeIcon(nextTheme(settings.theme))}
              {themeLabel(nextTheme(settings.theme))}
            </button>
            <button
              className="tool-button"
              onClick={() => setActiveSection("advanced")}
            >
              <Settings size={16} />
              Settings
            </button>
            <button
              className="tool-button"
              onClick={() => setCommandOpen(true)}
            >
              <Command size={16} />
              Command
            </button>
            {authState?.user && (
              <AccountMenu user={authState.user} onLoggedOut={() => { refreshAuth(); }} />
            )}
          </div>
        </header>

        {error && (
          <section className="alert">
            <AlertCircle size={18} />
            <span>{error}</span>
          </section>
        )}

        {state === "error" && (
          <section className="daemon-down">
            <PlugZap size={18} />
            <div className="daemon-down-text">
              <strong>Can't reach a SNDR daemon at {apiBase}</strong>
              <span>This is where patches, presets and the patcher version come from. A GPU server runs vLLM <em>engines</em>, not the daemon — manage it from <b>Hosts</b> (SSH check · Discover · Terminal · Chat). Point the GUI back at a running daemon to see project data.</span>
            </div>
            <button className="primary-action" onClick={() => switchServer(window.location.origin)}>
              <Home size={15} /> Use local daemon
            </button>
          </section>
        )}

        <SectionErrorBoundary section={activeSection}>
        {activeSection === "launch-plan" ? (
          <section className="section-workspace section-launch-plan">
        <header className="section-heading">
          <div>
            <span>Operator workbench</span>
            <h1>Launch Plan</h1>
            <p>Recommend a preset, compose the runtime, clear gates, then launch — plan-before-apply with a live job console.</p>
          </div>
          <div className="section-actions">
            <button className="tool-button" onClick={() => void refreshLaunchPlan()}>
              <RefreshCw size={16} /> Re-run Gates
            </button>
            <button className="tool-button" onClick={() => void loadAll()}>
              <RefreshCw size={16} /> Sync
            </button>
          </div>
        </header>
        <section className="process-strip" aria-label="Launch process">
          <Step
            number="1"
            title="Choose Preset"
            detail={selectedPreset || recommendForm.workload.replace(/_/g, " ")}
            state="done"
            active={launchTab === "recommend"}
            onClick={() => setLaunchTab("recommend")}
          />
          <Step
            number="2"
            title="Configure"
            detail={`${targetTitle(runtimeTargets, runtimeTarget)} · ${patchPolicy}`}
            state="done"
            active={launchTab === "compose"}
            onClick={() => setLaunchTab("compose")}
          />
          <Step
            number="3"
            title="Review & Launch"
            detail={gateCounts.blocked > 0 ? `${gateCounts.blocked} blocked` : applyEnabled ? "ready to launch" : "read-only"}
            state={gateCounts.blocked > 0 ? "warning" : "active"}
            active={launchTab === "launch"}
            onClick={() => setLaunchTab("launch")}
          />
          <Step
            number="4"
            title="Observe"
            detail={launchJob ? `job ${launchJob.status}` : `${evidenceRefs.length} evidence refs`}
            state={launchJob ? "done" : "idle"}
            active={launchTab === "console"}
            onClick={() => setLaunchTab("console")}
          />
        </section>

        <section className="metric-strip">
          <Metric icon={<Database size={18} />} label="Presets" value={overview?.catalog.presets_count ?? "-"} />
          <Metric icon={<Cpu size={18} />} label="Models" value={overview?.catalog.models_count ?? "-"} />
          <Metric icon={<GitBranch size={18} />} label="Profiles" value={overview?.catalog.profiles_count ?? "-"} />
          <Metric
            icon={<Sparkles size={18} />}
            label="Product API"
            value={state === "loading" ? "Loading" : state === "error" ? "Error" : "Ready"}
          />
          <Metric
            icon={<Gauge size={18} />}
            label={primaryMetricKind}
            value={primaryMetricValue > 0 ? primaryMetricValue : "Pending"}
          />
        </section>

        <TabbedSection
          id="launch-plan"
          activeTab={launchTab}
          onTabChange={setLaunchTab}
          tabs={[
            {
              id: "recommend",
              label: "1 · Choose",
              icon: <SlidersHorizontal size={15} />,
              render: () => (
          <section className="panel builder-panel">
            <PanelHeader
              label="A."
              title="Preset Recommendation Builder"
              action={`${recommend?.total_matches ?? 0} matches`}
              icon={<SlidersHorizontal size={18} />}
            />

            <div className="builder-section">
              <span className="section-index">1.</span>
              <div>
                <h3>Workload</h3>
                <div className="segmented-row">
                  {workloadChoices.map((workload) => (
                    <button
                      className={recommendForm.workload === workload.id ? "active" : ""}
                      key={workload.id}
                      onClick={() => selectWorkload(workload.id)}
                    >
                      {workload.label}
                      <small>{overview?.catalog.workload_counts[workload.id] ?? 0}</small>
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="builder-section two-column">
              <span className="section-index">2.</span>
              <label className="field">
                <span>Hardware Target</span>
                <select
                  value={recommendForm.hardware}
                  onChange={(event) =>
                    setRecommendForm({ ...recommendForm, hardware: event.target.value })
                  }
                >
                  {hardwareOptions.map((hardware) => (
                    <option value={hardware} key={hardware}>
                      {hardware}
                    </option>
                  ))}
                </select>
              </label>
              <div className="hardware-card">
                <HardDrive size={18} />
                <div>
                  <strong>Current host</strong>
                  <span>{runtimeMode === "remote" ? "Remote GPU node" : "Local workstation"}</span>
                </div>
              </div>
            </div>

            <div className="builder-section constraints-row">
              <span className="section-index">3.</span>
              <label className="field compact">
                <span>Concurrency</span>
                <input
                  type="number"
                  min={1}
                  max={32}
                  value={recommendForm.concurrency}
                  onChange={(event) =>
                    setRecommendForm({
                      ...recommendForm,
                      concurrency: Number(event.target.value)
                    })
                  }
                />
              </label>
              <label className="field compact">
                <span>Result Count</span>
                <input
                  type="number"
                  min={1}
                  max={12}
                  value={recommendForm.top}
                  onChange={(event) =>
                    setRecommendForm({ ...recommendForm, top: Number(event.target.value) })
                  }
                />
              </label>
              <div className="toggle-field">
                <span>Public evidence</span>
                <button
                  type="button"
                  className={recommendForm.preferPublic ? "toggle active" : "toggle"}
                  aria-label="Prefer public evidence"
                  aria-pressed={recommendForm.preferPublic}
                  onClick={() =>
                    setRecommendForm({ ...recommendForm, preferPublic: !recommendForm.preferPublic })
                  }
                />
              </div>
              <div className="toggle-field">
                <span>Safe patch policy</span>
                <button
                  type="button"
                  className={patchPolicy === "safe" ? "toggle active" : "toggle"}
                  aria-label="Safe patch policy"
                  aria-pressed={patchPolicy === "safe"}
                  onClick={() => setPatchPolicy(patchPolicy === "safe" ? "aggressive" : "safe")}
                />
              </div>
              <button className="primary-action" onClick={() => void runRecommend()}>
                <Search size={16} />
                Recalculate
              </button>
            </div>

            <div className="builder-section table-section">
              <span className="section-index">4.</span>
              <div className="recommend-table-wrap">
                <div className="table-toolbar">
                  <h3>Recommendation Results</h3>
                  <label className="search-box">
                    <Search size={15} />
                    <input
                      value={query}
                      onChange={(event) => setQuery(event.target.value)}
                      placeholder="Search preset, model, family"
                    />
                  </label>
                </div>
                <table className="recommend-table">
                  <thead>
                    <tr>
                      <th>Preset</th>
                      <th>Model Family</th>
                      <th>Mode</th>
                      <th>Status</th>
                      <th>Allowed Workloads</th>
                      <th>Evidence</th>
                      <th>Fallback</th>
                      <th>Risk</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(recommend?.results ?? []).map((row) => (
                      <RecommendationRow
                        key={row.id}
                        row={row}
                        active={row.id === selectedPreset}
                        onSelect={() => void loadExplain(row.id)}
                      />
                    ))}
                  </tbody>
                </table>
                {recommend?.results.length === 0 && (
                  <div className="empty-state">No recommendation results for this query.</div>
                )}
                <div className="table-footer">
                  <span>
                    Showing {(recommend?.results ?? []).length} of {recommend?.total_candidates ?? 0} candidates
                  </span>
                  <button className="ghost-button" onClick={() => setActiveSection("presets")}>View All Presets</button>
                </div>
              </div>
            </div>
          </section>

              )
            },
            {
              id: "compose",
              label: "2 · Configure",
              icon: <SlidersHorizontal size={15} />,
              render: () => (
          <section className="panel composer-panel">
            <PanelHeader
              label="B."
              title="Launch Plan Composer"
              action={`Plan ID: ${planId}`}
              icon={<Rocket size={18} />}
            />

            <div className="composer-grid">
              <section>
                <div className="plan-flow">
                  <PlanChip label="Selected Preset" value={selectedPreset} />
                  <ChevronRight size={18} />
                  <PlanChip label="Model" value={selectedPresetRecord?.model ?? "-"} />
                  <ChevronRight size={18} />
                  <PlanChip label="Hardware" value={selectedPresetRecord?.hardware ?? recommendForm.hardware} />
                  <ChevronRight size={18} />
                  <PlanChip label="Profile" value={selectedPresetRecord?.profile ?? "-"} />
                  <ChevronRight size={18} />
                  <PlanChip label="Baseline" value={primaryMetricValue > 0 ? `${primaryMetricValue.toLocaleString()} ${primaryMetricKind.replace(/^agg_/, "")}` : "pending"} />
                </div>

                <div className="option-block">
                  <h3>Runtime Target</h3>
                  <div className="runtime-grid">
                    {runtimeTargets.map((target) => (
                      <button
                        className={runtimeTarget === target.id ? "active" : ""}
                        key={target.id}
                        onClick={() => setRuntimeTarget(target.id)}
                      >
                        <span>{target.title}</span>
                        <StatusBadge status={target.status} />
                      </button>
                    ))}
                  </div>
                </div>

                <div className="option-block">
                  <h3>Patch Policy</h3>
                  <div className="policy-row">
                    {["compact", "safe", "minimal"].map((policy) => (
                      <button
                        className={patchPolicy === policy ? "active" : ""}
                        key={policy}
                        onClick={() => setPatchPolicy(policy)}
                      >
                        {policy}
                      </button>
                    ))}
                    <span className="policy-note">
                      Strict image digest and dry-run mode are enabled for GUI preview.
                    </span>
                  </div>
                </div>

                <div className="option-block">
                  <h3>Launch Target</h3>
                  <label className="field">
                    <span>SSH target — empty = local execution</span>
                    <input
                      value={launchSshTarget}
                      onChange={(event) => setLaunchSshTarget(event.target.value)}
                      placeholder="user@gpu-host"
                    />
                  </label>
                  <p className="policy-note">
                    {launchSshTarget.trim()
                      ? `Apply Launch will run over SSH on ${launchSshTarget.trim()} (when --enable-apply).`
                      : "Apply Launch runs locally (when --enable-apply). Set an SSH target to launch on a remote host."}
                  </p>
                </div>

                <ArtifactPreview
                  artifacts={launchPlan?.artifacts ?? []}
                  activeTab={artifactTab}
                  setActiveTab={setArtifactTab}
                />
              </section>

              <section className="plan-summary">
                <h3>Plan Summary</h3>
                <KeyValue label="Preset" value={selectedPreset} />
                <KeyValue label="Model" value={asText(planSummary.model, selectedPresetRecord?.model ?? "-")} />
                <KeyValue label="Hardware" value={selectedPresetRecord?.hardware ?? recommendForm.hardware} />
                <KeyValue label="Runtime" value={targetTitle(runtimeTargets, runtimeTarget)} />
                <KeyValue label="Mode" value={runtimeMode === "remote" ? "Remote SSH tunnel" : "Local web daemon"} />
                <KeyValue label="Context" value={formatTokens(asNumber(planSummary.context) || asNumber(composed.max_model_len))} />
                <KeyValue label="Sequences" value={String(asNumber(planSummary.max_num_seqs) || asNumber(composed.max_num_seqs) || "-")} />
                <KeyValue label="KV cache" value={asText(composed.kv_cache_dtype, "-")} />
                <KeyValue label="Spec decode" value={`${asText(composed.spec_decode_method, "-")} / K=${asText(composed.spec_decode_K, "-")}`} />
                <KeyValue label="Patches" value={asNumber(planSummary.enabled_patches_count) || asNumber(composed.enabled_patches_count) || "-"} />
                <KeyValue label="Patch policy" value={patchPolicy} />
                <KeyValue label="Fallback" value={asText(planSummary.fallback_preset, asText(card.fallback_preset, "-"))} />
                <KeyValue label="Plan ID" value={planId} />
                <button className="primary-action launch-continue" onClick={() => setLaunchTab("launch")}>
                  Review &amp; Launch
                  <ChevronRight size={16} />
                </button>
                <p className="policy-note">Step 3 confirms readiness and starts the runtime.</p>
              </section>
            </div>
          </section>

              )
            },
            {
              id: "launch",
              label: "3 · Launch",
              icon: <Rocket size={15} />,
              render: () => (
                <LaunchPanel
                  selectedPreset={selectedPreset}
                  model={asText(planSummary.model, selectedPresetRecord?.model ?? "-")}
                  hardware={selectedPresetRecord?.hardware ?? recommendForm.hardware}
                  profile={selectedPresetRecord?.profile ?? "-"}
                  host={endpointHost}
                  composed={composed}
                  planSummary={planSummary}
                  card={card}
                  patchPolicy={patchPolicy}
                  runtimeTitle={targetTitle(runtimeTargets, runtimeTarget)}
                  runtimeMode={runtimeMode}
                  endpoints={launchPlan?.endpoints}
                  gates={gates}
                  gateCounts={gateCounts}
                  applyEnabled={applyEnabled}
                  actionReason={launchPlan?.action_reason}
                  launchConfirm={launchConfirm}
                  setLaunchConfirm={setLaunchConfirm}
                  launchBusy={launchBusy}
                  launchSshTarget={launchSshTarget}
                  launchJob={launchJob}
                  onLaunch={() => void runLaunchApply()}
                  onConfigure={() => setLaunchTab("compose")}
                  onViewGates={() => setLaunchTab("gates")}
                />
              )
            },
            {
              id: "gates",
              label: "Gates",
              icon: <ListChecks size={15} />,
              render: () => (
          <section className="panel gates-panel">
            <PanelHeader
              label="C."
              title="Gates & Blockers"
              action={`${gateCounts.pass} ok / ${gateCounts.warning} warn / ${gateCounts.blocked} blocked`}
              icon={<ListChecks size={18} />}
            />
            <div className="gates-list">
              {gates.map((gate) => (
                <GateRow gate={gate} key={gate.id} onNavigate={setActiveSection} />
              ))}
            </div>
            <button
              className="wide-secondary"
              onClick={() => void refreshLaunchPlan()}
            >
              <RefreshCw size={15} />
              Re-run All Gates
            </button>
          </section>

              )
            },
            {
              id: "console",
              label: "Console",
              icon: <SquareTerminal size={15} />,
              render: () => (
          <section className="panel console-panel">
            <PanelHeader
              label="E."
              title="Job and Event Console"
              action="Read-only mirror"
              icon={<SquareTerminal size={18} />}
            />
            <OperationalConsole
              activeTab={consoleTab}
              setActiveTab={setConsoleTab}
              selectedPreset={selectedPreset}
              presetCount={presetCount}
              gates={gates}
              events={events}
              lines={cliLines}
              onMonitor={setMonitorJobId}
            />
          </section>
              )
            },
            {
              id: "endpoints",
              label: "Endpoints & Evidence",
              icon: <Link2 size={15} />,
              render: () => (
          <div className="lp-endpoints-grid">
            <RuntimeEndpoint
              host={endpointHost}
              endpoints={launchPlan?.endpoints}
            />
            <BenchmarkCard
              metricKind={primaryMetricKind}
              metricValue={primaryMetricValue}
              context={asNumber(composed.max_model_len)}
              visibility={visibility}
              busy={jobActionBusy === "bench"}
              onRun={() => void runBenchJob()}
            />
            <EvidenceCard
              visibility={visibility}
              evidenceCount={evidenceRefs.length}
              busy={jobActionBusy === "evidence"}
              onAttach={() => void runEvidenceJob()}
            />
            <PatchMatrix
              summary={patchSummary}
              registryTotal={patches?.total ?? patchRows.length}
              selectedCount={asNumber(composed.enabled_patches_count)}
              onExplain={() => setDialog("Patch policy matrix reflects the live registry: default-applied, marker-only, opt-in and blocked patches, plus the count enabled by the selected preset plan.")}
            />
          </div>
              )
            }
          ]}
        />
          </section>
        ) : (
          <Suspense fallback={<SkeletonCards count={6} />}>
          <SectionWorkspace
            sectionId={activeSection}
            overview={overview}
            presets={presets}
            filteredPresets={filteredPresets}
            selectedPreset={selectedPreset}
            selectedPresetRecord={selectedPresetRecord}
            explain={explain}
            runtimeMode={runtimeMode}
            runtimeTarget={runtimeTarget}
            patchPolicy={patchPolicy}
            runtimeTargets={runtimeTargets}
            featureRows={featureRows}
            patches={patches}
            patchDoctor={patchDoctor}
            configCatalog={configCatalog}
            configPreview={configPreview}
            bundles={bundles}
            diffUpstream={diffUpstream}
            proofStatus={proofStatus}
            userPresets={userPresets}
            doctorReport={doctorReport}
            environment={environment}
            hostProfiles={hostProfiles}
            gates={gates}
            gateCounts={gateCounts}
            events={events}
            cliLines={cliLines}
            apiBase={apiBase}
            settings={settings}
            onSection={setActiveSection}
            onPreset={(id) => void loadExplain(id)}
            onCommand={() => setCommandOpen(true)}
            onSettings={updateSettings}
            onConfigPreview={setConfigPreview}
            onUserPresetsRefresh={refreshUserPresets}
            onHostsRefresh={refreshHosts}
            onMonitorJob={setMonitorJobId}
            authUser={authState?.user ?? null}
            onAuthRefresh={refreshAuth}
            chatTarget={chatTarget}
            onChatWithHost={chatWithHost}
            onAddServer={addHostAsServer}
            focusHostId={focusHostId}
            onFocusConsumed={() => setFocusHostId(null)}
            onFocusHost={setFocusHostId}
            installIntent={installIntent}
            onSetupNode={(id) => { setInstallIntent({ hostId: id, target: "sndr_daemon" }); setActiveSection("setup"); }}
            onContainers={(id) => { setFocusHostId(id); setActiveSection("containers"); }}
            onHardware={(id) => { setFocusHostId(id); setActiveSection("hardware"); }}
            applyEnabled={applyEnabled}
          />
          </Suspense>
        )}
        </SectionErrorBoundary>

        <footer className="status-footer">
          <span>SNDR_HOME ~/.sndr</span>
          <span>API {apiBase}</span>
          <span className={`connected ${connectionTone}`}><span className="live-dot" /> {connectionLabel}</span>
          <span>Mode {runtimeMode === "remote" ? "Remote Desktop" : "Local Server"}</span>
        </footer>
      </section>
      {dialog && <InfoDialog message={dialog} onClose={() => setDialog(null)} />}
      {monitorJobId && <JobMonitorModal jobId={monitorJobId} onClose={() => setMonitorJobId(null)} />}
      <ToastHost />
      {commandOpen && (
        <CommandPalette
          onClose={() => setCommandOpen(false)}
          onSection={(section) => {
            setActiveSection(section);
            setCommandOpen(false);
          }}
          onRefresh={() => {
            void loadAll();
            setCommandOpen(false);
          }}
          onShortcuts={() => {
            setShortcutsOpen(true);
            setCommandOpen(false);
          }}
          settings={settings}
          onSettings={updateSettings}
          searchItems={[
            ...navItems.map((nav) => ({ icon: nav.icon, title: nav.label, detail: "Section", run: () => setActiveSection(nav.id) })),
            ...(presets?.presets ?? []).slice(0, 80).map((preset) => ({
              icon: <Database size={16} />, title: preset.id, detail: `Preset · ${preset.model} · ${preset.hardware}`,
              run: () => { void loadExplain(preset.id); setActiveSection("presets"); }
            })),
            ...(configCatalog?.models ?? []).map((m) => ({
              icon: <Boxes size={16} />, title: m.title || m.id, detail: `Model · ${m.summary || m.id}`,
              run: () => setActiveSection("configs")
            })),
            ...(configCatalog?.hardware ?? []).map((h) => ({
              icon: <Server size={16} />, title: h.title || h.id, detail: `Hardware · ${h.summary || h.id}`,
              run: () => setActiveSection("configs")
            })),
            ...(configCatalog?.profiles ?? []).map((p) => ({
              icon: <SlidersHorizontal size={16} />, title: p.title || p.id, detail: `Profile · ${p.summary || p.id}`,
              run: () => setActiveSection("configs")
            })),
            ...(patches?.patches ?? []).map((p) => ({
              icon: <PackageCheck size={16} />, title: p.patch_id, detail: `Patch · ${p.title}`,
              run: () => setActiveSection("patches")
            }))
          ]}
        />
      )}
      {shortcutsOpen && <ShortcutsModal onClose={() => setShortcutsOpen(false)} />}
    </main>
  );
}

// Keyboard-shortcut reference overlay (opened with `?`). Documents the global
// chords so power users can discover them without leaving the keyboard.
// ShortcutsModal extracted to ./components/dialogs.
class SectionErrorBoundary extends Component<{ section: string; children: ReactNode }, { error: Error | null }> {
  state: { error: Error | null } = { error: null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  componentDidUpdate(prev: { section: string }) {
    if (prev.section !== this.props.section && this.state.error) {
      this.setState({ error: null });
    }
  }
  render() {
    if (this.state.error) {
      return (
        <section className="section-workspace">
          <div className="error-boundary">
            <AlertCircle size={22} />
            <div>
              <strong>This panel hit a rendering error.</strong>
              <p>{this.state.error.message || "Unexpected error while rendering the panel."}</p>
              <button type="button" className="ghost-button" onClick={() => this.setState({ error: null })}>
                <RefreshCw size={14} /> Retry
              </button>
            </div>
          </div>
        </section>
      );
    }
    return this.props.children;
  }
}

// ConnectionMap extracted to ./sections/connection-bar.

// Settings tab panels (ApiTokenManager/NotificationSettings/AppearanceSettings + primitives) extracted to ./sections/settings-panels.
function SectionWorkspace({
  sectionId,
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
              title={`Copy a shareable link to ${selectedPreset}`}
              onClick={() => {
                const url = window.location.href;
                void navigator.clipboard?.writeText(url).then(
                  () => toast(`Link to ${selectedPreset} copied`, "success"),
                  () => toast("Could not copy link", "error")
                );
              }}
            >
              <Link2 size={16} />
              Copy Link
            </button>
          )}
          <button className="tool-button" onClick={() => onSection("launch-plan")}>
            <Rocket size={16} />
            Launch Plan
          </button>
          <button className="tool-button" onClick={onCommand} title="Open the command palette (⌘K)">
            <Command size={16} />
            Quick Action
          </button>
        </div>
      </header>

      {sectionId === "overview" && (
        <TabbedSection
          id="overview"
          tabs={[
            {
              id: "summary",
              label: "Summary",
              icon: <Activity size={15} />,
              render: () => (
                <>
                <div className="ov-hero">
                  <OvKpi icon={<Database size={15} />} label="Presets" value={overview?.catalog.presets_count ?? "—"} sub={`${(presets?.presets ?? []).filter((p) => asNumber(asRecord(p.card?.primary_metric).value) > 0).length} bench-proven`} onClick={() => onSection("presets")} />
                  <OvKpi icon={<Box size={15} />} label="Models" value={overview?.catalog.models_count ?? "—"} />
                  <OvKpi icon={<Wrench size={15} />} label="Patches" value={patchRows.length || patches?.total || "—"} sub={`${patchRows.filter((p) => p.default_on).length} default-on`} />
                  <OvKpi icon={<Server size={15} />} label="Hosts" value={hostProfiles.length} onClick={() => onSection("hosts")} />
                  <OvKpi icon={<ShieldCheck size={15} />} label="Doctor" value={doctorReport ? (doctorReport.findings.length ? `${doctorReport.findings.length} findings` : "clean") : "—"} tone={doctorReport?.findings?.some((f) => f.severity === "blocked") ? "warn" : "ok"} onClick={() => onSection("doctor")} />
                  <OvKpi icon={<Rocket size={15} />} label="Engine" value={environment?.engine_installed ? "ready" : "—"} tone={environment?.engine_installed ? "ok" : undefined} onClick={() => onSection("services")} />
                </div>
                <ModuleGrid>
                  {settings.showConnectionMap && (
                    <ModuleCard title="Control Plane Connections" icon={<Route size={18} />} wide>
                      <ConnectionMap
                        runtimeMode={runtimeMode}
                        runtimeTarget={runtimeTarget}
                        selectedPreset={selectedPreset}
                        patchCount={patchRows.length}
                        apiBase={apiBase}
                      />
                    </ModuleCard>
                  )}
                  <ModuleCard title="Platform Snapshot" icon={<Monitor size={18} />}>
                    <InfoRows
                      rows={[
                        ["Brand", overview?.capabilities.platform.public_brand ?? "-"],
                        ["Package", overview?.capabilities.platform.package_name ?? "-"],
                        ["Version", overview?.capabilities.platform.sndr_core_version ?? "-"],
                        ["OS", `${overview?.capabilities.platform.os_name ?? "-"} / ${overview?.capabilities.platform.machine ?? "-"}`],
                        ["Python", overview?.capabilities.platform.python_version ?? "-"]
                      ]}
                    />
                  </ModuleCard>
                  <ModuleCard title="Catalog Health" icon={<Database size={18} />}>
                    <KpiGrid
                      rows={[
                        ["Presets", overview?.catalog.presets_count ?? 0],
                        ["Cards", overview?.catalog.preset_cards_count ?? 0],
                        ["Models", overview?.catalog.models_count ?? 0],
                        ["Patches", patchRows.length || patches?.total || 0]
                      ]}
                    />
                  </ModuleCard>
                  <ModuleCard title="Launch Readiness" icon={<ShieldCheck size={18} />} desc="Gate verdict for the selected preset launch.">
                    <PercentBar
                      value={gateCounts.pass}
                      max={gates.length || 1}
                      label="gates passing"
                      caption={`${gateCounts.pass} ok · ${gateCounts.warning} warn · ${gateCounts.blocked} blocked`}
                      tone={gateCounts.blocked > 0 ? "warn" : "ok"}
                    />
                    <InfoRows
                      rows={[
                        ["Selected preset", selectedPreset],
                        ["Runtime target", targetTitle(runtimeTargets, runtimeTarget)],
                        ["Mode", runtimeMode === "remote" ? "Remote (SSH tunnel)" : "Local server"],
                        ["Patch policy", patchPolicy]
                      ]}
                    />
                  </ModuleCard>
                  <ModuleCard title="Engine & Versions" icon={<Cpu size={18} />} desc="What is installed in the API daemon's shell.">
                    <InfoRows
                      rows={[
                        ["SNDR Core", environment?.sndr_core_version ?? overview?.capabilities.platform.sndr_core_version ?? "-"],
                        ["Engine", `${environment?.engine_name ?? "vLLM"} ${environment?.engine_version ?? (environment?.engine_installed ? "" : "(not installed)")}`.trim()],
                        ["Engine installed", environment?.engine_installed ? "yes" : "no"],
                        ["Doctor findings", doctorReport ? `${doctorReport.findings.length} (${doctorReport.findings.filter((f) => f.severity === "blocked").length} blocked · ${doctorReport.findings.filter((f) => f.severity === "warning").length} warn)` : "-"],
                        ["Platform", `${overview?.capabilities.platform.os_name ?? "-"} / ${overview?.capabilities.platform.machine ?? "-"}`]
                      ]}
                    />
                  </ModuleCard>
                </ModuleGrid>
                </>
              )
            },
            {
              id: "environment",
              label: "Environment",
              icon: <Cpu size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Runtime Environment" icon={<Cpu size={18} />} desc="Project version, engine version and the installed dependency stack." wide>
                    <EnvironmentPanel env={environment} />
                  </ModuleCard>
                  <ModuleCard title="Runtime Targets" icon={<Server size={18} />} wide>
                    <CapabilityTable rows={runtimeTargets} />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "coverage",
              label: "Coverage",
              icon: <Layers3 size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Workload Coverage" icon={<Layers3 size={18} />}>
                    <CompactList rows={Object.entries(workloadCounts).map(([key, value]) => [key, String(value)])} />
                  </ModuleCard>
                  <ModuleCard title="Family Coverage" icon={<Box size={18} />}>
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
              label: "1 · Guided setup",
              icon: <ShieldCheck size={15} />,
              render: () => (
                <>
                  <TabIntro icon={<ShieldCheck size={16} />} title="Start here — guided setup"
                    text="A read-only checklist of where you stand: environment, engine, dependencies and launch gates, with a clear next step. Nothing is changed here — it just tells you what to do." />
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
              label: "2 · Install onto host",
              icon: <Server size={15} />,
              render: () => (
                <>
                  <TabIntro icon={<Server size={16} />} title="Install onto a GPU host (over SSH)"
                    text="Pick a registered host and a preset, preview the exact install plan, then apply it over SSH — it ships the daemon/engine onto that server so the GUI can manage it. Gated: review the plan before it runs." />
                  <InstallWizard initial={installIntent || undefined} />
                </>
              )
            },
            // 3) Render deploy artifacts (compose/systemd/run) for a chosen model.
            {
              id: "deploy",
              label: "3 · Deploy a model",
              icon: <Rocket size={15} />,
              render: () => (
                <>
                  <TabIntro icon={<Rocket size={16} />} title="Deploy a model preset"
                    text="Turn a preset into ready-to-run artifacts — docker-compose, systemd unit or a docker run line, with the right image, GPUs, ports and patch env baked in. Copy them to the host, or use Install above to push over SSH." />
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
          <ModuleCard title="Fleet overview" icon={<LayoutGrid size={18} />} desc="Every registered GPU/engine host at a glance — a single concurrent SSH sweep shows status, running model, vLLM version, GPUs and live patch count per server. Click a server to drill into its card." wide>
            <FleetPanel onOpenHost={(id) => { onFocusHost(id); onSection("hosts"); }} />
          </ModuleCard>
        </ModuleGrid>
      )}

      {sectionId === "containers" && (
        <ModuleGrid>
          <ModuleCard title="Containers" icon={<Boxes size={18} />} desc="Manage the vLLM/engine containers on a server — list, live CPU/memory, logs, start/stop/restart, and (when SNDR_ENABLE_EXEC is on) exec inside. Pick the local daemon's host (docker socket) or a registered host (over SSH). Scoped to engine containers only." wide>
            <Suspense fallback={<SkeletonCards count={6} />}>
              <ContainersPanel hosts={hostOptions} onNavigate={(section) => onSection(section as SectionId)} initialHostId={focusHostId ?? undefined} />
            </Suspense>
          </ModuleCard>
        </ModuleGrid>
      )}

      {sectionId === "hardware" && (
        <ModuleGrid>
          <ModuleCard title="GPU & Hardware" icon={<Cpu size={18} />} desc="Live per-GPU telemetry over nvidia-smi — utilisation, VRAM, temperature, power vs limits, clocks, fan, PCIe link, pstate and ECC — plus host CPU/RAM. Pick the local daemon host or a registered host (over SSH)." wide>
            <Suspense fallback={<SkeletonCards count={2} />}>
              <HardwarePanel hosts={hostOptions} initialHostId={focusHostId ?? undefined} />
            </Suspense>
          </ModuleCard>
        </ModuleGrid>
      )}

      {sectionId === "routing" && (
        <ModuleGrid>
          <ModuleCard title="Workload routing" icon={<Route size={18} />} desc="The deterministic spec-decode router — the same brain the gateway uses. Per bench-validated profile: which workloads are allowed/denied and their measured TPS delta. Classify a request shape by its response_format / tool_choice / workload_class signals and see which profile it resolves to." wide>
            <Suspense fallback={<SkeletonCards count={2} />}>
              <RoutingPanel />
            </Suspense>
          </ModuleCard>
        </ModuleGrid>
      )}

      {sectionId === "flags" && (
        <ModuleGrid>
          <ModuleCard title="Env-flag matrix" icon={<SlidersHorizontal size={18} />} desc="Every GENESIS_ENABLE_* flag in the registry with its effective default — searchable, filterable by family. Name a running engine container to overlay its live ON/OFF state and flag drift (missing = default-on but off on the engine; extra = on beyond the default)." wide>
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
              label: "Catalog",
              icon: <Database size={15} />,
              render: () => (
                <div className="preset-catalog-view">
                  {presets?.load_errors && presets.load_errors.length > 0 && (
                    <div className="preset-load-errors">
                      <AlertTriangle size={15} />
                      <div>
                        <strong>{presets.load_errors.length} preset{presets.load_errors.length > 1 ? "s" : ""} failed to load</strong>
                        {presets.load_errors.slice(0, 6).map((e, i) => (
                          <div key={i} className="preset-load-error"><code>{e.preset ?? e.id ?? e.file ?? "?"}</code> — {e.error ?? e.message ?? JSON.stringify(e)}</div>
                        ))}
                        {presets.load_errors.length > 6 && <div className="muted">+{presets.load_errors.length - 6} more</div>}
                      </div>
                    </div>
                  )}
                  <PresetSummaryStrip presets={filteredPresets} selectedPreset={selectedPreset} />
                  <div className="preset-catalog-split">
                    <ModuleCard
                      title="Preset Catalog"
                      icon={<Database size={18} />}
                      desc="Pick a preset to inspect its runtime, edit a local copy, or launch it."
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
                      onPolicy={() => setPresetTab("policy")}
                      onLaunch={() => onSection("launch-plan")}
                    />
                  </div>
                </div>
              )
            },
            {
              id: "recommend",
              label: "Recommend",
              icon: <Rocket size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Recommend a preset" icon={<Rocket size={18} />} desc="Rank presets for a workload + rig + concurrency target, then inspect the winner." wide>
                    <PresetRecommendPanel
                      hardwareOptions={Array.from(new Set((presets?.presets ?? []).map((preset) => preset.hardware))).filter(Boolean)}
                      workloadCounts={overview?.catalog.workload_counts ?? {}}
                      onSelect={(id) => { onPreset(id); setPresetTab("selected"); }}
                    />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "selected",
              label: "Selected",
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
              label: "Edit",
              icon: <SlidersHorizontal size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard
                    title={`Visual Editor — ${selectedPreset}`}
                    icon={<Wrench size={18} />}
                    desc="Full preset editing: pointers, card metadata and any other fields the preset defines. Saves an operator-local copy."
                    wide
                  >
                    <LayerEditor kind="preset" layerId={selectedPreset} />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "policy",
              label: "Policy & analytics",
              icon: <BarChart3 size={15} />,
              render: () => {
                const allPresets = presets?.presets ?? [];
                const statusDist = countRecord(allPresets.map((p) => asText(p.card?.status, p.has_card ? "annotated" : "unannotated")));
                const visibilityDist = countRecord(allPresets.filter((p) => p.has_card).map((p) => asText(p.card?.evidence_visibility, "unknown")));
                const fallbacks = allPresets.filter((p) => p.card?.fallback_preset);
                const bar = (counts: Record<string, number>): Array<[string, number, string]> => {
                  const max = Math.max(1, ...Object.values(counts));
                  return Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([k, v]) => [k, Math.round((v / max) * 100), String(v)]);
                };
                return (
                <ModuleGrid>
                  <ModuleCard title="Workload Policy" icon={<SlidersHorizontal size={18} />} desc={`Allow/deny rules for ${selectedPreset}.`} wide>
                    <PresetPolicyGraph card={card} presets={allPresets} />
                  </ModuleCard>
                  <ModuleCard title="Status Distribution" icon={<ShieldCheck size={18} />} desc="Production readiness across the catalog.">
                    <BarList rows={bar(statusDist)} />
                  </ModuleCard>
                  <ModuleCard title="Evidence Visibility" icon={<FileText size={18} />} desc="How evidence is published across annotated presets.">
                    {Object.keys(visibilityDist).length ? <BarList rows={bar(visibilityDist)} /> : <p className="muted">No annotated presets.</p>}
                  </ModuleCard>
                  <ModuleCard title="Workload Coverage" icon={<Layers3 size={18} />} desc="Presets allowing each workload (catalog-wide).">
                    <BarList rows={bar(workloadCounts)} />
                  </ModuleCard>
                  <ModuleCard title="Family Coverage" icon={<Box size={18} />} desc="Presets per routing family.">
                    <BarList rows={bar(familyCounts)} />
                  </ModuleCard>
                  <ModuleCard title="Fallback Chains" icon={<GitBranch size={18} />} desc="Presets that declare a fallback target.">
                    {fallbacks.length ? (
                      <CompactList rows={fallbacks.map((p) => [p.id, `→ ${asText(p.card?.fallback_preset, "-")}`] as [string, string])} />
                    ) : (
                      <p className="muted">No fallback chains declared.</p>
                    )}
                  </ModuleCard>
                </ModuleGrid>
                );
              }
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
              label: "Lifecycle",
              icon: <Network size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Lifecycle Planner" icon={<Network size={18} />} desc="Plan start/stop/restart/status/logs across any runtime target, with live engine reachability and post-action verification." wide>
                    <ServiceLifecyclePlanner
                      selectedPreset={selectedPreset}
                      runtimeTarget={runtimeTarget}
                      host={runtimeMode === "remote" ? "gpu-build-01" : "127.0.0.1"}
                    />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "engine",
              label: "Engine",
              icon: <Cpu size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Live Engine" icon={<Activity size={18} />} desc="Reachability, loaded model and version of the running vLLM OpenAI server.">
                    <EngineStatusCard />
                  </ModuleCard>
                  <ModuleCard title="Live Metrics" icon={<Gauge size={18} />} desc="Prometheus KPIs from the running engine — queue, KV cache, throughput, TTFT/TPOT, spec-decode.">
                    <EngineMetricsPanel />
                  </ModuleCard>
                  <ModuleCard title="Engine & Dependencies" icon={<Cpu size={18} />} desc="Installed engine and library versions on the daemon host — what would serve." wide>
                    <EnvironmentPanel env={environment} />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "contracts",
              label: "Contracts",
              icon: <ShieldCheck size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Lifecycle Surface" icon={<ShieldCheck size={18} />} desc="Which lifecycle capabilities the Product API exposes today." wide>
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
              label: "Diagnostics",
              icon: <Stethoscope size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Diagnostics Summary" icon={<Activity size={18} />} desc="Aggregated environment, runtime, catalog, patch and proof health." wide>
                    <DoctorSummary report={doctorReport} />
                  </ModuleCard>
                  <ModuleCard title="Findings" icon={<Stethoscope size={18} />} desc="Grouped by category — expand a row for evidence, action and CLI." wide>
                    <DoctorFindings report={doctorReport} />
                  </ModuleCard>
                  <ModuleCard title="Host caveats" icon={<AlertTriangle size={18} />} desc="Known host-condition issues (kernel, virtualization, GPU, pin) evaluated live against this host — triggered caveats first." wide>
                    <CaveatsPanel />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "gates",
              label: "Readiness gates",
              icon: <ShieldCheck size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Launch Readiness Gates" icon={<ShieldCheck size={18} />} desc="Per-gate blockers for the selected preset launch." wide>
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
              label: "Coverage",
              icon: <PackageCheck size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Registry Coverage" icon={<PackageCheck size={18} />} desc="Patch apply-module coverage and validation." wide>
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
              label: "Registry",
              icon: <PackageCheck size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Patch Registry Summary" icon={<PackageCheck size={18} />} desc={`${patches?.total ?? patchRows.length} runtime overlays across ${new Set(patchRows.map((p) => p.family)).size} families.`} wide>
                    <PatchSummaryPanel summary={patchSummary} total={patches?.total ?? patchRows.length} selectedCount={asNumber(composed.enabled_patches_count)} />
                  </ModuleCard>
                  <ModuleCard title="Lifecycle & Default Behavior" icon={<BarChart3 size={18} />} wide>
                    <PatchLifecycleGraph summary={patchSummary} />
                  </ModuleCard>
                  <ModuleCard title="Status, Families & Legend" icon={<ListChecks size={18} />} desc="Implementation maturity, subsystem coverage, and what each registry value means." wide>
                    <PatchRegistryInsight summary={patchSummary} patches={patchRows} />
                  </ModuleCard>
                  <ModuleCard title="Supported Models" icon={<Cpu size={18} />} desc="Catalog models the patch family targets — per-patch applicability is in the Inventory tab." wide>
                    <PatchModelSupport models={configCatalog?.models ?? []} />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "inventory",
              label: "Inventory",
              icon: <Table2 size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Patch Inventory Control" icon={<Table2 size={18} />} wide>
                    <PatchInventoryControl patches={patchRows} />
                  </ModuleCard>
                  <ModuleCard title="Patch Bundles" icon={<Layers3 size={18} />} wide>
                    <BundlesPanel bundles={bundles} />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "flags",
              label: "Flags",
              icon: <SlidersHorizontal size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Env-flag matrix" icon={<SlidersHorizontal size={18} />} desc="Every GENESIS_ENABLE_* flag with its effective default — searchable, filterable by family. Name a running engine container to overlay its live ON/OFF state and flag drift." wide>
                    <Suspense fallback={<SkeletonCards count={2} />}>
                      <FlagsPanel />
                    </Suspense>
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "upstream",
              label: "Upstream & policy",
              icon: <GitBranch size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Upstream Diff" icon={<GitBranch size={18} />} wide>
                    <UpstreamDiffPanel report={diffUpstream} />
                  </ModuleCard>
                  <ModuleCard title="Policy Preview" icon={<Code2 size={18} />} wide>
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
              label: "Baseline",
              icon: <BarChart3 size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Benchmark Baseline" icon={<BarChart3 size={18} />} desc="Reference metric and the resolved runtime it was measured on." wide>
                    <BenchmarkBaselinePanel card={card} composed={composed} record={selectedPresetRecord} selectedPreset={selectedPreset} />
                  </ModuleCard>
                  <ModuleCard title="Capability Status" icon={<Activity size={18} />} wide>
                    <CapabilityTable rows={featureRows.filter((feature) => feature.id === "benchmark_runs")} />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "live",
              label: "Live bench",
              icon: <TimerReset size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Live Engine" icon={<Activity size={18} />} desc="The bench drives the running engine — start the runtime first.">
                    <EngineStatusCard />
                  </ModuleCard>
                  <ModuleCard title="Live Benchmark + A/B" icon={<TimerReset size={18} />} desc="Run a real micro-benchmark against the engine; run twice for an A/B delta." wide>
                    <EngineBenchPanel referenceTps={asNumber(asRecord(card.primary_metric).value) || null} />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "coverage",
              label: "Coverage",
              icon: <ShieldCheck size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Benchmark Coverage" icon={<ShieldCheck size={18} />} wide>
                    <ProofStatusPanel report={proofStatus} />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "run",
              label: "Run plan",
              icon: <Play size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Run Plan" icon={<Play size={18} />}>
                    <WorkflowSteps rows={[["1", "Warmup", "Stabilize cache and CUDA graph path"], ["2", "Load test", "Measure TTFT/TPS/acceptance"], ["3", "Proof", "Attach immutable evidence refs"]]} />
                  </ModuleCard>
                  <ModuleCard title="Run Commands" icon={<SquareTerminal size={18} />} desc="Queue a benchmark as a job, or copy the commands to run on the rig.">
                    <QueueJobButton
                      label={`Queue bench (${selectedPreset})`}
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
              label: "Proof",
              icon: <ShieldCheck size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Evidence Report" icon={<FileText size={18} />} desc="Evidence references attached to the selected preset card." wide>
                    <EvidenceRows card={card} />
                  </ModuleCard>
                  <ModuleCard title="Proof Artifact Status" icon={<ShieldCheck size={18} />} desc="Proof artifacts across the catalog by lifecycle." wide>
                    <ProofStatusPanel report={proofStatus} />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "coverage",
              label: "Coverage",
              icon: <Activity size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Evidence Coverage" icon={<Activity size={18} />} desc="This preset's references grouped by visibility and type." wide>
                    {refs.length ? (
                      <div className="evidence-coverage">
                        <PercentBar
                          value={byVisibility.public ?? 0}
                          max={refs.length}
                          label="public refs"
                          caption={`${byVisibility.public ?? 0} of ${refs.length} reference${refs.length === 1 ? "" : "s"} public`}
                          tone={byVisibility.public ? "ok" : "warn"}
                        />
                        <CompactList
                          rows={[
                            ...Object.entries(byType).map(([k, v]) => [k, String(v)] as [string, string]),
                            ...Object.entries(byVisibility).map(([k, v]) => [`visibility: ${k}`, String(v)] as [string, string])
                          ]}
                        />
                      </div>
                    ) : (
                      <p className="muted">This preset exposes no evidence references yet.</p>
                    )}
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "bundle",
              label: "Bundle",
              icon: <SquareTerminal size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Report Bundle Commands" icon={<SquareTerminal size={18} />} desc="Queue evidence collection as a job, or copy the commands to run on the rig." wide>
                    <QueueJobButton
                      label={`Queue evidence (${selectedPreset})`}
                      run={() => api.evidenceAttach({ preset_id: selectedPreset })}
                      onMonitor={onMonitorJob}
                    />
                    <CodeBlock lines={[`sndr evidence collect --preset ${selectedPreset}`, "sndr report create --format html,pdf", "sndr proof attach --release-check"]} />
                  </ModuleCard>
                </ModuleGrid>
              )
            }
          ]}
        />
        );
      })()}

      {sectionId === "clients" && (() => {
        const clientHost = runtimeMode === "remote" ? "gpu-build-01" : "127.0.0.1";
        const baseUrl = `http://${clientHost}:8000/v1`;
        const modelName = String(composed.served_model_name ?? selectedPreset);
        return (
        <ModuleGrid>
          <ModuleCard title="Live Engine" icon={<Activity size={18} />} desc="Is the runtime up? Loaded model and version from the running server." wide>
            <EngineStatusCard />
          </ModuleCard>
          <ModuleCard title="Playground" icon={<MessageSquare size={18} />} desc="Send a real prompt to the running engine — a one-click smoke test." wide>
            <EnginePlayground />
          </ModuleCard>
          <ModuleCard title="Client Endpoints" icon={<Link2 size={18} />} desc="OpenAI-compatible API, health and metrics URLs for the selected runtime host." wide>
            <EndpointRows host={clientHost} />
          </ModuleCard>
          <ModuleCard title="Quick Start" icon={<Code2 size={18} />} desc={`Copy-paste clients for the served model "${modelName}".`} wide>
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
                  label: "Streaming",
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
                  label: "Health",
                  lines: [
                    `curl http://${clientHost}:8000/health`,
                    `curl http://${clientHost}:8001/metrics | grep vllm:`,
                    `curl ${baseUrl}/models`
                  ]
                }
              ]}
            />
          </ModuleCard>
          <ModuleCard title="Served Model" icon={<Box size={18} />} desc="What the OpenAI-compatible server exposes for this preset.">
            <InfoRows
              rows={[
                ["Model name", modelName],
                ["Base URL", baseUrl],
                ["Runtime target", targetTitle(runtimeTargets, runtimeTarget)],
                ["Host mode", runtimeMode === "remote" ? "Remote host" : "Local server"]
              ]}
            />
          </ModuleCard>
          <ModuleCard title="Authentication" icon={<KeyRound size={18} />} desc="The Product API token; the inference server itself follows your vLLM launch flags.">
            <InfoRows
              rows={[
                ["GUI/Product API", "Open by default; set SNDR_GUI_TOKEN to require a bearer token"],
                ["Header", "Authorization: Bearer <token> or X-SNDR-Token: <token>"],
                ["Inference API key", 'OpenAI clients accept any value (e.g. "not-needed") unless vLLM --api-key is set']
              ]}
            />
          </ModuleCard>
          <ModuleCard title="Client Modes" icon={<Layers3 size={18} />} desc="How operators reach this control plane.">
            <CompactList rows={[["Web UI", "Browser control center"], ["Desktop", "Tauri remote shell"], ["API", "OpenAI-compatible endpoint"], ["CLI", "Operator mirror"]]} />
          </ModuleCard>
        </ModuleGrid>
        );
      })()}

      {sectionId === "planner" && (
        <ModuleGrid>
          <ModuleCard title="KV-cache / VRAM fit calculator" icon={<Gauge size={18} />} desc="GQA, MoE and tensor-parallel aware. Slide context to see weights / KV / overhead vs the per-GPU budget, the max context per KV dtype, and the VRAM curve." wide>
            <KvCalcPanel />
          </ModuleCard>
          <ModuleCard title="Quality-baseline regression diff" icon={<GitCompare size={18} />} desc="Save a trusted bench/eval result, then diff a new run against it — direction-aware regression flags + a CI exit code." wide>
            <BaselinePanel />
          </ModuleCard>
        </ModuleGrid>
      )}

      {sectionId === "copilot" && (
        <ModuleGrid>
          <ModuleCard title="Ops Copilot" icon={<Sparkles size={18} />} desc="A read-only tool-calling assistant over the Product API. It answers from real catalog/doctor/preset/patch/capacity data and proposes changes for you to review & apply — it never mutates anything itself." wide>
            <CopilotPanel onNavigate={(section) => onSection(section as SectionId)} />
          </ModuleCard>
        </ModuleGrid>
      )}

      {sectionId === "chat" && (
        <ModuleGrid>
          <ModuleCard title="Local model chat" icon={<MessageSquare size={18} />} desc="Multi-turn streaming conversation with a running local vLLM model. Set the engine host/port, pick the model, tune the system prompt and sampling." wide>
            <ChatConsole defaultHost={runtimeMode === "remote" ? "192.168.1.10" : "127.0.0.1"} target={chatTarget} />
          </ModuleCard>
        </ModuleGrid>
      )}

      {sectionId === "reports" && (
        <TabbedSection
          id="reports"
          tabs={[
            {
              id: "templates",
              label: "Templates",
              icon: <Table2 size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Report Types" icon={<Table2 size={18} />} desc="Generate a redacted snapshot bundle into the operator-local reports dir." wide>
                    <ReportGenerator selectedPreset={selectedPreset} />
                  </ModuleCard>
                  <ModuleCard title="Export Formats" icon={<SquareTerminal size={18} />} desc="Formats the report CLI can render from a snapshot." wide>
                    <CompactList
                      rows={[
                        ["HTML", "Shareable operator review page"],
                        ["PDF", "Archival / sign-off document"],
                        ["JSON", "Machine-readable snapshot"],
                        ["Markdown", "Inline notes and runbooks"]
                      ]}
                    />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "snapshot",
              label: "Snapshot",
              icon: <FileText size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Current Snapshot" icon={<FileText size={18} />} desc="The live state a generated report would capture right now." wide>
                    <InfoRows
                      rows={[
                        ["Preset", selectedPreset],
                        ["Runtime target", targetTitle(runtimeTargets, runtimeTarget)],
                        ["Patch policy", patchPolicy],
                        ["Readiness gates", `${gateCounts.pass} ok / ${gateCounts.warning} warn / ${gateCounts.blocked} blocked`],
                        ["Proof artifacts", proofStatus?.available ? String(proofStatus.total) : "unavailable"],
                        ["Evidence visibility", asText(card.evidence_visibility, "-")]
                      ]}
                    />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "activity",
              label: "Activity",
              icon: <Clock3 size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Recent Activity" icon={<Clock3 size={18} />} desc="Session actions reflected in the control center." wide>
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
              label: "Operations",
              icon: <Terminal size={15} />,
              render: () => <OperationsConsole onMonitor={onMonitorJob} />
            },
            {
              id: "config-keys",
              label: "Config keys",
              icon: <SlidersHorizontal size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Config-key glossary" icon={<SlidersHorizontal size={18} />} desc="Every GENESIS_ENABLE_* flag, V1/V2 config key and policy key with provenance — searchable operator reference (mirrors `sndr config-keys`)." wide>
                    <ConfigKeysPanel />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "traces",
              label: "Traces",
              icon: <Activity size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Diagnostic trace catalog" icon={<Activity size={18} />} desc="Per-patch debug traces — the container path each lands at and the env var that enables it. Operator reference (mirrors `sndr trace list`)." wide>
                    <TracesPanel />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "appearance",
              label: "Appearance",
              icon: <Palette size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Appearance and Operator Mode" icon={<Palette size={18} />} wide>
                    <AppearanceSettings settings={settings} onSettings={onSettings} />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "license",
              label: "License & modules",
              icon: <BadgeCheck size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="License & SNDR Engine" icon={<BadgeCheck size={18} />} desc="Active tier (community vs commercial SNDR Engine), the signed license token (subject / expiry / signature), whether the vllm.sndr_engine overlay is installed, and how many engine-tier patches it unlocks." wide>
                    <Suspense fallback={<SkeletonCards count={1} />}>
                      <LicensePanel />
                    </Suspense>
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "notifications",
              label: "Notifications",
              icon: <Bell size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Alerts & notifications" icon={<Bell size={18} />} desc="Get a Telegram push when a managed engine container goes DOWN (crash / OOM / stop) or recovers. The daemon watches over the docker socket; gated behind apply." wide>
                    <NotificationSettings />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "api",
              label: "API & Schema",
              icon: <Settings size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Daemon & Access" icon={<Settings size={18} />} desc="Daemon endpoint, OpenAPI and the optional access token for remote/tunnel use.">
                    <InfoRows rows={[
                      ["API Base", apiBase],
                      ["OpenAPI", `${apiBase}/openapi.json`],
                      ["Mode", "Read-only Product API"],
                      ["SNDR Core", environment?.sndr_core_version ?? "-"],
                      ["Frontend", "Vite/React, served by daemon"]
                    ]} />
                    <ApiTokenField />
                  </ModuleCard>
                  <ModuleCard title="API Tokens" icon={<KeyRound size={18} />} desc="Named, revocable Bearer tokens for programmatic / CI access (auth required). Plaintext shown once." wide>
                    <ApiTokenManager enabled={!!authUser} />
                  </ModuleCard>
                  <ModuleCard title="Endpoint Explorer" icon={<Code2 size={18} />} desc="Send a live GET to any read-only Product API endpoint and inspect the JSON." wide>
                    <EndpointExplorer />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "audit",
              label: "Audit log",
              icon: <FileText size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Audit log" icon={<FileText size={18} />} desc="Tamper-evident record of daemon events — auth, jobs, operations and system actions. Live." wide>
                    <AuditLogPanel />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "updates",
              label: "Updates",
              icon: <DownloadCloud size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Updates" icon={<DownloadCloud size={18} />} desc="Pin-gated self-update for the GUI + sndr_core patcher. The vLLM pin only moves to a patcher-supported value; the server docker step stays manual. Apply is gated + confirmed." wide>
                    <UpdatesPanel />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "admin",
              label: "Admin",
              icon: <KeyRound size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Admin Surface Matrix" icon={<KeyRound size={18} />} desc="Product API write/read surfaces and their status." wide>
                    <AdminSurfaceMatrix featureRows={featureRows} patchDoctor={patchDoctor} />
                  </ModuleCard>
                  <ModuleCard title="Engine & Dependencies" icon={<Cpu size={18} />} desc="Versions and runtime tools on the daemon host.">
                    <EnvironmentPanel env={environment} />
                  </ModuleCard>
                  <ModuleCard title="Feature Contracts" icon={<ShieldCheck size={18} />} desc="Capability inventory with live statuses.">
                    <CapabilityTable rows={featureRows} />
                  </ModuleCard>
                  {authUser && (
                    <ModuleCard title="Account & Security" icon={<ShieldCheck size={18} />} desc="Your password and two-factor settings." wide>
                      <SecurityPanel user={authUser} onChanged={onAuthRefresh} />
                    </ModuleCard>
                  )}
                  {authUser?.role === "admin" && (
                    <ModuleCard title="User Management" icon={<KeyRound size={18} />} desc="Create, list and remove accounts (admin)." wide>
                      <UserAdminPanel currentUser={authUser} />
                    </ModuleCard>
                  )}
                </ModuleGrid>
              )
            },
            {
              id: "developer",
              label: "Developer",
              icon: <SlidersHorizontal size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Config Draft and Diff" icon={<SlidersHorizontal size={18} />} desc={`Local runtime draft for ${selectedPreset}.`} wide>
                    <ConfigDraftEditor
                      selectedPreset={selectedPreset}
                      composed={composed}
                      runtimeTarget={runtimeTarget}
                      patchPolicy={patchPolicy}
                    />
                  </ModuleCard>
                  <ModuleCard title="CLI Mirror" icon={<SquareTerminal size={18} />} desc="Equivalent CLI for the current operator context.">
                    <CodeBlock lines={cliLines} />
                  </ModuleCard>
                  <ModuleCard title="Live Events" icon={<Activity size={18} />} desc="Daemon event feed (jobs, lifecycle, reports).">
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

function sectionSpec(sectionId: SectionId) {
  const specs: Record<
    SectionId,
    { kicker: string; title: string; description: string }
  > = {
    overview: {
      kicker: "System map",
      title: "Overview",
      description: "One screen summary of Product API health, catalog coverage, runtime targets and workload readiness.",
    },
    setup: {
      kicker: "First-run workflow",
      title: "Setup",
      description: "Local server and remote desktop setup path with explicit daemon, tunnel and gate stages.",
    },
    fleet: {
      kicker: "Multi-server overview",
      title: "Fleet",
      description: "Every registered GPU/engine host at a glance — one concurrent SSH sweep shows status, running model, vLLM version, GPUs and live patch count per server.",
    },
    hosts: {
      kicker: "Runtime inventory",
      title: "Hosts",
      description: "Local and remote host inventory, transport state, runtime tools and SSH tunnel commands.",
    },
    models: {
      kicker: "Model catalog",
      title: "Models",
      description: "Model families, hardware envelopes and composed runtime details from the V2 registry.",
    },
    configs: {
      kicker: "V2 config editor",
      title: "Configs",
      description: "Graphical editor for V2 model, hardware, profile and preset composition with safe draft preview.",
    },
    presets: {
      kicker: "Preset catalog",
      title: "Presets",
      description: "Full preset table with cards, workload policy, evidence visibility and selected explain payload.",
    },
    planner: {
      kicker: "Capacity & regression",
      title: "Planner",
      description: "KV-cache / VRAM fit calculator (GQA, MoE and tensor-parallel aware, calibratable) and quality-baseline regression diff.",
    },
    copilot: {
      kicker: "Read-only assistant",
      title: "Ops Copilot",
      description: "Tool-calling assistant over the read-only Product API — answers from real catalog/doctor/preset/patch/capacity data and proposes changes you review & apply.",
    },
    "launch-plan": {
      kicker: "Operator workbench",
      title: "Launch Plan",
      description: "Recommendation builder, plan composer, readiness gates, artifacts and CLI mirror.",
    },
    services: {
      kicker: "Lifecycle",
      title: "Services",
      description: "Service lifecycle, rendered launch artifacts, status, logs and safe write API boundary.",
    },
    containers: {
      kicker: "Docker control",
      title: "Containers",
      description: "Manage the vLLM/engine containers on a server — live CPU/memory, logs, start/stop/restart, and gated exec — over the local docker socket or a registered host via SSH.",
    },
    hardware: {
      kicker: "GPU telemetry",
      title: "GPU & Hardware",
      description: "Live per-GPU utilisation, VRAM, temperature, power, clocks, fan, PCIe, pstate and ECC over nvidia-smi — for the daemon host or a registered host via SSH.",
    },
    routing: {
      kicker: "Spec-decode routing",
      title: "Workload routing",
      description: "Per bench-validated profile: which workloads are allowed/denied and their measured TPS delta — plus a classifier that predicts how a request's signals resolve to a profile. One source of truth, shared with the gateway.",
    },
    flags: {
      kicker: "Patch flags",
      title: "Env-flag matrix",
      description: "Every GENESIS_ENABLE_* flag with its default, searchable and filterable — overlay a running engine's live ON/OFF state and flag drift.",
    },
    doctor: {
      kicker: "Diagnostics",
      title: "Doctor",
      description: "Readiness gates, blockers, warnings and release-proof preflight diagnostics.",
    },
    patches: {
      kicker: "Patch control",
      title: "Patches",
      description: "Patch simulation, policy matrix, enabled patch count and safe/minimal/compact policy preview.",
    },
    benchmarks: {
      kicker: "Performance",
      title: "Benchmarks",
      description: "Benchmark baselines, expected TPS/TTFT context, run plan and evidence orchestration state.",
    },
    evidence: {
      kicker: "Proof bundle",
      title: "Evidence",
      description: "Evidence references, visibility, benchmark baseline status and future release report bundles.",
    },
    clients: {
      kicker: "Integrations",
      title: "Clients",
      description: "OpenAI-compatible endpoints, health/metrics URLs, client snippets and GUI/CLI integration modes.",
    },
    chat: {
      kicker: "Local model chat",
      title: "Chat",
      description: "Multi-turn streaming chat with any running local vLLM model — pick the engine host/port and model, tune the system prompt and sampling.",
    },
    reports: {
      kicker: "Operator reports",
      title: "Reports",
      description: "Launch, benchmark, patch and release-proof report types planned for GUI export.",
    },
    operations: {
      kicker: "Project workbench",
      title: "Operations",
      description: "Run sndr_core's canonical maintenance, audit and proof workflows as live-monitored jobs — the CLI surface, integrated.",
    },
    advanced: {
      kicker: "Developer surface",
      title: "Advanced",
      description: "API base, OpenAPI/schema, feature contracts, CLI mirror and future desktop settings.",
    }
  };
  return specs[sectionId];
}

// PatchMatrixViewer extracted to ./sections/patch-matrix.

// CatalogBadge type is now owned/exported by ./sections/catalog-cards (imported above).

// CatalogCard extracted to ./sections/catalog-cards.

// modelFamily/itemBadges/ModelSummaryStrip/ModelKeyFacts + ModelsWorkbench extracted to ./sections/models-workbench.

// Configs workbench (ConfigElementEditor/V2ConfigWorkbench/ConfigsSection/ConfigSelect/ConfigItemInspector/CompositionChain/ResolvedConfig + CodeEditorField + config-artifact builders) extracted to ./sections/configs-workbench.
// TabIntro extracted to ./components/shell-bits.

// Overview hero KPI tile — bold headline numbers, optionally click-through to a section.
// OvKpi extracted to ./components/charts.

// TabbedSection extracted to ./components/tabbed-section.

// ModuleCard extracted to ./components/layout.

// InfoRows extracted to ./components/primitives.

// KpiGrid extracted to ./components/primitives.

// RuntimeEnvelopePanel + PresetPolicyGraph extracted to ./sections/preset-insight.

// ConfigDraftEditor + Collapsible + DraftControl extracted to ./sections/config-draft-editor.

// CHART_PALETTE / DonutSegment / SegmentBar / PercentBar / segmentsFromCounts
// / BarList extracted to ./components/charts.

// CapabilityTable extracted to ./components/capability-table.

// CompactList extracted to ./components/primitives.

function WorkflowSteps({ rows }: { rows: Array<[string, string, string]> }) {
  return (
    <div className="workflow-steps">
      {rows.map(([number, title, detail]) => (
        <div key={number}>
          <span>{number}</span>
          <div>
            <strong>{title}</strong>
            <small>{detail}</small>
          </div>
        </div>
      ))}
    </div>
  );
}

// QueueJobButton extracted to ./sections/jobs.

// EndpointExplorer + ReportGenerator extracted to ./sections/api-explorer.

// HOST_ROLES moved to ./sections/host-form-modal.

// roleTone + tunnelCommand + FleetHostCard (+ ApplyDisabledNote / RelSpark) extracted to ./sections/fleet-host-card.

// The daemon host this UI talks to — live inventory, no probe needed.
// ThisHostCard extracted to ./sections/this-host-card.

// Detailed inventory grid for the Inventory tab.
// HostInventoryPanel + DependencyStackPanel extracted to ./sections/environment.

// Illustrated empty state — icon badge + title + guidance + optional recovery
// action. Used on primary content areas in place of bare muted "No X" text.
// EmptyState extracted to ./components/empty-state.

// Project & catalog snapshot — fills the row beside the dependency stack with
// the most useful project parameters: catalog counts, annotation coverage,
// capability readiness and the workload/lifecycle distribution.
// ProjectCatalogPanel extracted to ./sections/project-catalog.

// Add/edit modal for a host profile.
// HostFormModal (+ HOST_ROLES) extracted to ./sections/host-form-modal.

// Enterprise Hosts section: fleet overview with live probes, full daemon-host
// inventory, profile CRUD with a modal, runtime-target matrix and access.
// HostsSection + HostProfileTable extracted to ./sections/hosts-section.
// CodeTabs extracted to ./components/shell-bits.

// Enterprise recommend tab: surface the catalog's recommendation engine inside
// Presets. Pick a workload + rig + concurrency and get a ranked, scored list of
// presets — click to select and inspect. Connects browse → recommend → select.
// PresetRecommendPanel extracted to ./sections/preset-recommend.

// What this preset overrides relative to its declared fallback target.
// PresetFallbackDiff + PresetSelectedView + PresetSummaryStrip extracted to ./sections/preset-views.

// CommandPalette extracted to ./sections/command-palette.

function loadGuiSettings(): GuiSettings {
  try {
    const raw = window.localStorage.getItem(GUI_SETTINGS_STORAGE_KEY);
    if (!raw) return defaultGuiSettings;
    const parsed = JSON.parse(raw) as Partial<GuiSettings>;
    return {
      ...defaultGuiSettings,
      ...parsed,
      theme: parsed.theme && VALID_THEMES.has(parsed.theme) ? parsed.theme : "light",
      density: parsed.density === "compact" ? "compact" : "comfortable",
      accent: isAccent(parsed.accent) ? parsed.accent : defaultGuiSettings.accent,
      detailMode: parsed.detailMode === "operator" ? "operator" : "engineer",
      showConnectionMap:
        typeof parsed.showConnectionMap === "boolean"
          ? parsed.showConnectionMap
          : defaultGuiSettings.showConnectionMap,
      autoRefresh:
        typeof parsed.autoRefresh === "boolean"
          ? parsed.autoRefresh
          : defaultGuiSettings.autoRefresh,
      sidebarCollapsed:
        typeof parsed.sidebarCollapsed === "boolean"
          ? parsed.sidebarCollapsed
          : defaultGuiSettings.sidebarCollapsed
    };
  } catch {
    return defaultGuiSettings;
  }
}

function isAccent(value: unknown): value is AccentMode {
  return value === "teal" || value === "blue" || value === "emerald" || value === "amber";
}

// RecommendationRow extracted to ./sections/recommendation-row.

// CopyButton extracted to ./components/code-block.

// RuntimeEndpoint + BenchmarkCard + EvidenceCard + PatchMatrix extracted to ./sections/rail-cards.

// PlanChip + KeyValue + ArtifactPreview (+ ArtifactTab) extracted to ./components/display-bits.

// GATE_TARGET moved to ./nav.


// JobsTable + Progress + JobMonitorModal extracted to ./sections/jobs.

// EventLog + CliMirror + OperationalConsole extracted to ./sections/operational-console.

// RailStat + RailCheck extracted to ./components/primitives.

function buildReadinessGates({
  overview,
  runtimeTarget,
  selectedPresetRecord,
  explain
}: {
  overview: ProductOverview | null;
  runtimeTarget: string;
  selectedPresetRecord: PresetRecord | null;
  explain: PresetExplainResult | null;
}): Gate[] {
  const capabilities = overview?.capabilities;
  const featureRows = capabilities?.features ?? [];
  const target = capabilities?.runtime_targets.find((item) => item.id === runtimeTarget);
  const catalogErrors = overview?.catalog.preset_load_error_count ?? 0;
  const hasCard = Boolean(selectedPresetRecord?.has_card || explain?.card);
  const serviceLifecycle = featureRows.find((feature) => feature.id === "service_lifecycle");
  const benchmarkRuns = featureRows.find((feature) => feature.id === "benchmark_runs");

  return [
    {
      id: "catalog",
      label: "Catalog Snapshot",
      detail: catalogErrors === 0 ? "V2 registry loaded without errors" : `${catalogErrors} load errors`,
      status: catalogErrors === 0 ? "pass" : "blocked",
      action: "Re-run"
    },
    {
      id: "preset-card",
      label: "Preset Card",
      detail: hasCard ? "Operator card and explain payload available" : "Preset has no product card yet",
      status: hasCard ? "pass" : "warning",
      action: "Open"
    },
    {
      id: "runtime",
      label: "Runtime Target",
      detail: target?.detail ?? "Runtime target not selected",
      status: targetStatus(target),
      action: "Check"
    },
    {
      id: "engine",
      label: "Engine Installed",
      detail: capabilities?.platform.engine_installed ? "vLLM package detected" : "Engine package not installed in this shell",
      status: capabilities?.platform.engine_installed ? "pass" : "warning",
      action: "Doctor"
    },
    {
      id: "service-api",
      label: "Service Lifecycle API",
      detail: serviceLifecycle?.detail ?? "Plan/apply lifecycle API available (execution gated by --enable-apply)",
      status: serviceLifecycle?.status === "available" ? "pass" : "warning",
      action: "Plan"
    },
    {
      id: "evidence",
      label: "Evidence Orchestration",
      detail: benchmarkRuns?.detail ?? "Evidence/report jobs available; full GPU runs are a rig action",
      status: benchmarkRuns?.status === "available" ? "pass" : "warning",
      action: "Report"
    },
    {
      id: "release-proof",
      label: "Release Proof",
      detail: "Generate a proof/report bundle (Reports) before a production launch — recommended",
      status: "warning",
      action: "Generate proof"
    }
  ];
}

function targetStatus(target: ProductCapability | undefined): GateStatus {
  if (!target) return "blocked";
  if (target.status === "available") return "pass";
  if (target.status === "render_only" || target.status === "partial") return "warning";
  if (target.status === "deferred") return "planned";
  return "blocked";
}

function countGates(gates: Gate[]) {
  return gates.reduce(
    (acc, gate) => {
      acc[gate.status] += 1;
      return acc;
    },
    { pass: 0, warning: 0, blocked: 0, planned: 0 } as Record<GateStatus, number>
  );
}

function useLiveEvents(apiBase: string, enabled: boolean): BackendEvent[] {
  const [events, setEvents] = useState<BackendEvent[]>([]);
  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    const token = getApiToken();
    // Open mode: native SSE stream. EventSource cannot send an Authorization
    // header, so token-protected daemons fall back to authenticated polling.
    if (!token && typeof EventSource !== "undefined") {
      const source = new EventSource(`${apiBase}/api/v1/events`);
      source.addEventListener("snapshot", (event) => {
        try {
          const data = JSON.parse((event as MessageEvent).data);
          if (!cancelled) setEvents((data.events ?? []).slice(-100));
        } catch { /* ignore malformed frame */ }
      });
      source.addEventListener("event", (event) => {
        try {
          const item = JSON.parse((event as MessageEvent).data) as BackendEvent;
          if (!cancelled) setEvents((prev) => [...prev, item].slice(-100));
        } catch { /* ignore malformed frame */ }
      });
      source.onerror = () => { /* browser auto-reconnects */ };
      return () => { cancelled = true; source.close(); };
    }
    let cursor = 0;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const result = await api.eventsRecent(cursor);
        if (cancelled) return;
        if (result.events.length) {
          setEvents((prev) => [...prev, ...result.events].slice(-100));
          cursor = result.last_seq;
        }
      } catch { /* daemon may be briefly unreachable */ }
      if (!cancelled) timer = setTimeout(poll, 4000);
    };
    void poll();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [apiBase, enabled]);
  return events;
}

function buildEvents({
  state,
  error,
  selectedPreset,
  runtimeTarget,
  visibility,
  live = []
}: {
  state: LoadState;
  error: string | null;
  selectedPreset: string;
  runtimeTarget: string;
  visibility: string;
  live?: BackendEvent[];
}): Array<[string, string, string]> {
  const now = new Date();
  const stamp = (offset: number) =>
    new Date(now.getTime() - offset * 60_000).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit"
    });
  const rows: Array<[string, string, string]> = [
    [stamp(0), state === "error" ? "error" : "info", state === "loading" ? "Refreshing Product API snapshot..." : `Selected preset ${selectedPreset}`],
    [stamp(2), "info", `Runtime target set to ${runtimeTarget}`],
    [stamp(4), visibility === "public" ? "info" : "warn", `Evidence visibility: ${visibility}`],
    [stamp(6), "info", "Catalog and capability surfaces loaded through typed Product API"]
  ];
  if (error) rows.unshift([stamp(0), "error", error]);
  // Real backend events (dry-run jobs, lifecycle) take precedence at the top.
  const liveRows: Array<[string, string, string]> = live
    .slice(-12)
    .reverse()
    .map((event) => {
      const time = new Date(event.ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      const tone = event.kind === "error" ? "error" : event.kind === "job" ? "ok" : "info";
      return [time, tone, event.message];
    });
  return [...liveRows, ...rows];
}

function buildCliMirror({
  selectedPreset,
  runtimeTarget,
  patchPolicy,
  recommendForm,
  apiBase
}: {
  selectedPreset: string;
  runtimeTarget: string;
  patchPolicy: string;
  recommendForm: RecommendForm;
  apiBase: string;
}) {
  return [
    `$ sndr preset recommend --workload ${recommendForm.workload} --hardware ${recommendForm.hardware} --concurrency ${recommendForm.concurrency}${recommendForm.preferPublic ? " --prefer-public-evidence" : ""}`,
    `$ sndr preset explain ${selectedPreset}`,
    `$ sndr launch plan --preset ${selectedPreset} --runtime-target ${runtimeTarget} --patch-policy ${patchPolicy} --dry-run`,
    `$ sndr doctor --host current --all`,
    `$ curl ${apiBase}/api/v1/health`
  ];
}

// targetTitle moved to ./lib/format.

function runtimeHost(mode: RuntimeMode) {
  return mode === "remote" ? "gpu-build-01" : "127.0.0.1";
}

// countRecord extracted to ./lib/coerce.

// buildRuntimeDraft + buildDraftYaml extracted to ./lib/runtime-draft.


// ParamFields extracted to ./sections/config-draft-editor.

// DRAFT_FIELD_LABELS + runtimeDraftDiff extracted to ./lib/runtime-draft.


// asRecord / asText / asNumber / asStringArray extracted to ./lib/coerce.

// shortWorkload / formatTokens / formatVram extracted to ./lib/format.
