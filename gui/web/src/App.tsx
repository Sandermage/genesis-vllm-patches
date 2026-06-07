import {
  Activity,
  AlertCircle,
  BarChart3,
  Bell,
  Box,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Clock3,
  Code2,
  Command,
  Copy,
  Cpu,
  Database,
  Download,
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
  MemoryStick,
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
  Rows3,
  Search,
  Send,
  Server,
  Settings,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  SquareTerminal,
  Stethoscope,
  Table2,
  TimerReset,
  Trash2,
  Terminal,
  Layers,
  Pencil,
  Maximize2,
  Plus,
  Check,
  Loader2,
  X,
  Wrench
} from "lucide-react";
import { Component, Fragment, Suspense, lazy, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { sectionFromHash, recordIdFromHash, buildHash, replaceHash } from "./route";
import { type SectionId, type RuntimeMode, type Gate } from "./nav";
import {
  type ConsoleTab, type ThemeMode, type DensityMode, type AccentMode, type DetailMode, type GuiSettings,
  nextTheme, themeLabel, themeIcon, VALID_THEMES
} from "./settings";
import { useFetch } from "./hooks/useFetch";
import { asRecord, asText, asNumber, asStringArray, countRecord } from "./lib/coerce";
import { formatTokens, formatVram } from "./lib/format";
import { getIn, setIn, objToYaml } from "./lib/config-utils";
import { type RuntimeConfigDraft, buildRuntimeDraft, buildDraftYaml, runtimeDraftDiff } from "./lib/runtime-draft";
import { BoolField, SelectField } from "./components/form-fields";
import { type ElementKind, ELEMENT_FIELDS, discoverExtraFields, groupFields, ElementField } from "./sections/element-fields";
import { LayerEditor } from "./sections/layer-editor";
import { StatusBadge, StatusPill, InfoRows, CompactList, KpiGrid, RailCheck, type GateStatus } from "./components/primitives";
import { PercentBar, BarList, OvKpi } from "./components/charts";
import { CaveatsPanel, ConfigKeysPanel, TracesPanel } from "./sections/diagnostics";
import { DoctorSummary, DoctorFindings } from "./sections/doctor";
import { ConfigComparePanel, ConfigPlanPanel, ConfigApplyPanel } from "./sections/config";
import { BundlesPanel, UpstreamDiffPanel } from "./sections/registry";
import { UserPresetsPanel, ProfileDeltaPanel } from "./sections/presets";
import { HostInventoryPanel, DependencyStackPanel, EnvironmentPanel } from "./sections/environment";
import { DoctorCoveragePanel, AdminSurfaceMatrix } from "./sections/patch-doctor";
import { BenchmarkBaselinePanel, EvidenceRows } from "./sections/bench";
import { AuditLogPanel } from "./sections/audit-log";
import { RuntimeEndpoint, BenchmarkCard, EvidenceCard, PatchMatrix, EndpointRows } from "./sections/rail-cards";
import { EndpointExplorer, ReportGenerator } from "./sections/api-explorer";
import { ConfirmDialog, InfoDialog } from "./components/dialogs";
import { CatalogCard, ModelFitCard, ModelFitMatrix, KvEnvelopeCard, type CatalogBadge } from "./sections/catalog-cards";
import { RecommendationRow } from "./sections/recommendation-row";
import { ModuleGrid, ModuleCard } from "./components/layout";
import { toast, ToastHost } from "./components/toast";
import { OperationsConsole } from "./sections/operations";
import { DeploymentConsole } from "./sections/deployment";
import { GateRow } from "./sections/gate-row";
import { SetupWizard } from "./sections/setup-wizard";
import { JobMonitorModal, JobResultBlock } from "./sections/jobs";
import { ServiceLifecyclePlanner } from "./sections/services";
import { CommandPalette } from "./sections/command-palette";
import { EventLog, OperationalConsole } from "./sections/operational-console";
import { LaunchPanel } from "./sections/launch-panel";
import { RuntimeEnvelopePanel, PresetPolicyGraph } from "./sections/preset-insight";
import { PatchMatrixViewer } from "./sections/patch-matrix";
import { PatchSummaryPanel, PatchLifecycleGraph, PatchRegistryInsight, PatchModelSupport } from "./sections/patch-overview";
import { PatchInventoryControl } from "./sections/patch-inventory";
import { PresetRecommendPanel } from "./sections/preset-recommend";
import { type RecommendForm, defaultRecommend, workloadChoices } from "./recommend";
import { TabbedSection } from "./components/tabbed-section";
import { ProjectCatalogPanel } from "./sections/project-catalog";
import { PresetQuickPanel } from "./sections/preset-quick";
import { PresetCatalogTable } from "./sections/preset-catalog";
import { EmptyState } from "./components/empty-state";
import { HostFormModal } from "./sections/host-form-modal";
import { ThisHostCard } from "./sections/this-host-card";
import { FleetHostCard, roleTone, tunnelCommand } from "./sections/fleet-host-card";
import { ProofStatusPanel } from "./sections/proof";
import { CodeBlock, CopyButton } from "./components/code-block";
import { useDialogFocus, closeOnBackdrop } from "./dialog";
import { SkeletonLines, SkeletonCards } from "./Skeleton";
import {
  AlertConfig,
  BundleSpec,
  FleetHost,
  HostInventory,
  ReliabilitySnapshot,
  DiffUpstreamReport,
  DoctorReport,
  EnvironmentReport,
  HostProfile,
  Job,
  LaunchPlanArtifact,
  BackendEvent,
  LaunchPlanResult,
  ProofStatusReport,
  UserPresetList,
  V2ConfigApplyResult,
  V2ConfigCatalog,
  V2ConfigItem,
  V2ConfigPlan,
  V2ConfigPreview,
  PresetExplainResult,
  PatchDoctorReport,
  PatchListResult,
  PresetListResult,
  PresetRecord,
  PresetRecommendResult,
  ProductCapability,
  ProductOverview,
  V2LayerApplyResult,
  V2LayerDefinition,
  ApiTokenRecord,
  AuthStatus,
  AuthUser,
  api,
  getApiToken,
  setApiToken,
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
const ModelManagementPanel = lazy(() => import("./Engine").then((m) => ({ default: m.ModelManagementPanel })));
// Lazy: the xterm-based terminal is heavy and rarely opened — keep it out of the
// initial bundle so the app loads fast; the chunk fetches only when a terminal opens.
const TerminalModal = lazy(() => import("./Terminal").then((m) => ({ default: m.TerminalModal })));
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
type ArtifactTab = "compose" | "systemd" | "commands" | "env";
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
function ShortcutsModal({ onClose }: { onClose: () => void }) {
  const dialogRef = useRef<HTMLElement>(null);
  useDialogFocus(dialogRef);
  const groups: Array<{ title: string; rows: Array<[string[], string]> }> = [
    { title: "Global", rows: [
      [["⌘", "K"], "Open command palette"],
      [["?"], "Toggle this shortcuts help"],
      [["Esc"], "Close palette / dialog"],
    ] },
    { title: "Command palette", rows: [
      [["↑", "↓"], "Move between results"],
      [["↵"], "Run highlighted result"],
    ] },
    { title: "Go to (press g, then…)", rows: [
      [["g", "o"], "Overview"],
      [["g", "s"], "Setup"],
      [["g", "f"], "Fleet"],
      [["g", "h"], "Hosts"],
      [["g", "m"], "Models"],
      [["g", "c"], "Configs"],
      [["g", "p"], "Presets"],
      [["g", "n"], "Containers"],
      [["g", "d"], "Doctor"],
      [["g", "l"], "Launch Plan"],
      [["g", "b"], "Benchmarks"],
    ] },
  ];
  return (
    <div className="dialog-backdrop" role="presentation" onClick={closeOnBackdrop(onClose)}>
      <section ref={dialogRef} className="shortcuts-dialog" role="dialog" aria-modal="true" aria-label="Keyboard shortcuts">
        <div className="shortcuts-head">
          <Command size={16} />
          <strong>Keyboard shortcuts</strong>
          <button className="icon-only" onClick={onClose} aria-label="Close"><X size={15} /></button>
        </div>
        <div className="shortcuts-grid">
          {groups.map((group) => (
            <div className="shortcuts-group" key={group.title}>
              <h4>{group.title}</h4>
              {group.rows.map(([keys, label]) => (
                <div className="shortcuts-row" key={label}>
                  <span className="shortcuts-keys">
                    {keys.map((key, i) => <kbd key={i}>{key}</kbd>)}
                  </span>
                  <span className="shortcuts-label">{label}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

// LaunchParam + LaunchPanel extracted to ./sections/launch-panel.

function Step({
  number,
  title,
  detail,
  state,
  active = false,
  onClick
}: {
  number: string;
  title: string;
  detail: string;
  state: "done" | "active" | "warning" | "idle";
  active?: boolean;
  onClick?: () => void;
}) {
  const content = (
    <>
      <span>{number}</span>
      <div>
        <strong>{title}</strong>
        <small>{detail}</small>
      </div>
    </>
  );
  if (!onClick) {
    return <div className={`step ${state}`}>{content}</div>;
  }
  return (
    <button type="button" className={`step step-button ${state} ${active ? "current" : ""}`} onClick={onClick} aria-current={active}>
      {content}
    </button>
  );
}

function Metric({
  icon,
  label,
  value
}: {
  icon: ReactNode;
  label: string;
  value: string | number;
}) {
  return (
    <div className="metric">
      {icon}
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
    </div>
  );
}

function PanelHeader({
  label,
  title,
  action,
  icon
}: {
  label: string;
  title: string;
  action?: string;
  icon: ReactNode;
}) {
  return (
    <div className="panel-header">
      <div>
        {icon}
        <span>{label}</span>
        <h2>{title}</h2>
      </div>
      {action && <small>{action}</small>}
    </div>
  );
}

// StatusBadge / StatusPill extracted to ./components/primitives.

// Switch the standalone GUI between multiple daemon servers. The active server's
// baseUrl IS the api base — selecting one re-points every API call. Each server
// is health-pinged so you can see which are up before switching.
// Connection switcher — a reflection of the host registry (single source of
// truth). The targets are "This host" (the local daemon serving the GUI) plus
// every host profile: connecting points the GUI's Product API at that host's
// daemon. There is no separate server list — you create and manage connections
// as host cards in the Hosts section.
type ConnTarget = { id: string; label: string; baseUrl: string; isLocal: boolean; engineHost?: boolean };

function ServerSwitcher({
  apiBase,
  connectionTone,
  onSwitch,
  hostProfiles,
  onManageHosts,
  onOpenHost
}: {
  apiBase: string;
  connectionTone: "success" | "warning" | "danger";
  onSwitch: (baseUrl: string) => void;
  hostProfiles: HostProfile[];
  onManageHosts: () => void;
  onOpenHost: (hostId: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [health, setHealth] = useState<Record<string, "ok" | "down" | "checking">>({});
  const normBase = normalizeBaseUrl(apiBase);

  const targets = useMemo<ConnTarget[]>(() => {
    const localUrl = normalizeBaseUrl(typeof window !== "undefined" ? window.location.origin : "http://127.0.0.1:8765");
    const list: ConnTarget[] = [{ id: "__local__", label: "This host (local daemon)", baseUrl: localUrl, isLocal: true }];
    for (const h of hostProfiles) {
      const url = normalizeBaseUrl(`http://${h.host}:${h.port || 8765}`);
      if (!list.some((t) => t.baseUrl === url)) list.push({ id: h.id, label: h.label, baseUrl: url, isLocal: false, engineHost: h.transport === "ssh" || !!h.ssh_user });
    }
    if (!list.some((t) => t.baseUrl === normBase)) list.push({ id: "__current__", label: hostLabel(normBase), baseUrl: normBase, isLocal: false });
    return list;
  }, [hostProfiles, normBase]);

  const active = targets.find((t) => t.baseUrl === normBase);

  async function ping(t: ConnTarget) {
    setHealth((h) => ({ ...h, [t.id]: "checking" }));
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), 3000);
    try {
      const res = await fetch(`${t.baseUrl}/api/v1/health`, { signal: controller.signal, headers: { ...(getApiToken() ? { Authorization: `Bearer ${getApiToken()}` } : {}) } });
      setHealth((h) => ({ ...h, [t.id]: res.ok ? "ok" : "down" }));
    } catch {
      setHealth((h) => ({ ...h, [t.id]: "down" }));
    } finally {
      window.clearTimeout(timer);
    }
  }

  // eslint-disable-next-line react-hooks/exhaustive-deps -- ping the targets only when the dropdown opens
  useEffect(() => { if (open) targets.forEach((t) => void ping(t)); }, [open]);
  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => { if (!(e.target as HTMLElement).closest(".server-switcher")) setOpen(false); };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  const dot = (id: string) => <span className={`srv-dot ${health[id] === "ok" ? "ok" : health[id] === "down" ? "down" : "checking"}`} />;

  return (
    <div className="server-switcher">
      <button className={`server-current tone-${connectionTone}`} onClick={() => setOpen((v) => !v)} title={`Connected daemon: ${normBase}`}>
        <Server size={15} />
        <span className="server-current-label">{active ? active.label : hostLabel(normBase)}</span>
        <ChevronDown size={14} />
      </button>
      {open && (
        <div className="server-menu">
          <div className="server-menu-head">Daemon connection · from the host registry</div>
          <div className="server-list">
            {targets.map((t) => {
              // An engine host (SSH box, no daemon) can't be a daemon target —
              // selecting it opens its card instead of failing a daemon switch.
              // Only when the health ping CONFIRMS it's down (not while still
              // "checking") — otherwise a reachable node daemon would be misrouted.
              const isEngine = !!t.engineHost && health[t.id] === "down" && !t.isLocal && t.id !== "__current__";
              return (
                <div className={`server-item ${t.baseUrl === normBase ? "active" : ""} ${isEngine ? "engine" : ""}`} key={t.id}>
                  <button className="server-pick" onClick={() => { setOpen(false); if (isEngine) onOpenHost(t.id); else onSwitch(t.baseUrl); }}
                    title={isEngine ? "Engine host — open its card to see runtime state (Discover / Chat / Terminal)" : t.baseUrl}>
                    {isEngine ? <Server size={13} className="server-engine-ic" /> : dot(t.id)}
                    <span className="server-item-label">{t.label}</span>
                    <span className="server-item-url">{isEngine ? "engine host →" : hostLabel(t.baseUrl)}</span>
                    {t.baseUrl === normBase && <Check size={14} className="server-active-check" />}
                  </button>
                </div>
              );
            })}
          </div>
          <button className="server-add" onClick={() => { setOpen(false); onManageHosts(); }}><Plus size={14} /> Add / manage hosts</button>
          <div className="server-menu-hint">Daemons serve patches/presets/configs. A GPU box runs the <b>engine</b> — pick it to open its card and see what's running (models, GPUs, live patches).</div>
        </div>
      )}
    </div>
  );
}

// jobTone + JobResultBlock extracted to ./sections/jobs (shared executor-job card).

// Section-level error boundary: a render error in one panel shows an inline
// recoverable message instead of crashing the whole shell. Resets when the
// active section changes so navigating away clears a stuck panel.
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

function ConnectionMap({
  runtimeMode,
  runtimeTarget,
  selectedPreset,
  patchCount,
  apiBase
}: {
  runtimeMode: RuntimeMode;
  runtimeTarget: string;
  selectedPreset: string;
  patchCount: number;
  apiBase: string;
}) {
  const nodes = [
    { icon: <Monitor size={18} />, label: "GUI Shell", detail: runtimeMode === "remote" ? "remote desktop" : "local web" },
    { icon: <PlugZap size={18} />, label: "Product API", detail: apiBase.replace(/^https?:\/\//, "") },
    { icon: <Database size={18} />, label: "V2 Catalog", detail: selectedPreset },
    { icon: <PackageCheck size={18} />, label: "Patch Registry", detail: `${patchCount || "-"} entries` },
    { icon: <Server size={18} />, label: "Runtime Target", detail: runtimeTarget },
    { icon: <Link2 size={18} />, label: "OpenAI API", detail: "client endpoint" }
  ];
  return (
    <section className="connection-map" aria-label="Control plane connection map">
      {nodes.map((node, index) => (
        <div className="connection-node" key={node.label}>
          <div className="node-icon">{node.icon}</div>
          <strong>{node.label}</strong>
          <span>{node.detail}</span>
          {index < nodes.length - 1 && <i className="node-link" />}
        </div>
      ))}
    </section>
  );
}

// Managed personal-access tokens for programmatic / CI access to the Product API.
function ApiTokenManager({ enabled }: { enabled: boolean }) {
  const [tokens, setTokens] = useState<ApiTokenRecord[] | null>(null);
  const [unavailable, setUnavailable] = useState(!enabled);
  const [label, setLabel] = useState("");
  const [created, setCreated] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  function load() {
    api.apiTokens().then((result) => { setTokens(result.tokens); setUnavailable(false); }).catch(() => setUnavailable(true));
  }
  // Only probe the protected endpoint when authenticated — avoids a benign 401.
  useEffect(() => { if (enabled) load(); else setUnavailable(true); }, [enabled]);
  async function create() {
    setBusy(true);
    try {
      const result = await api.apiTokenCreate(label.trim() || "api-token");
      setCreated(result.token);
      setLabel("");
      load();
      toast("API token created", "success");
    } catch {
      toast("Failed to create token", "error");
    } finally { setBusy(false); }
  }
  async function revoke(id: string) {
    try { await api.apiTokenRevoke(id); load(); toast("Token revoked", "success"); } catch { toast("Failed to revoke token", "error"); }
  }
  const [confirmRevoke, setConfirmRevoke] = useState<{ id: string; label: string } | null>(null);
  const stamp = (ts: number) => new Date(ts * 1000).toLocaleDateString([], { month: "short", day: "2-digit", year: "numeric" });
  if (unavailable) {
    return <p className="muted">API token management requires authentication. Start the daemon with auth enabled (<code>SNDR_AUTH=on</code>) and sign in to mint revocable Bearer tokens.</p>;
  }
  return (
    <div className="token-manager">
      {created && (
        <div className="token-created">
          <div className="token-created-head"><KeyRound size={14} /> New token — copy it now, it won't be shown again</div>
          <div className="token-created-value"><code>{created}</code><CopyButton value={created} label="API token" /></div>
          <button className="ghost-button" onClick={() => setCreated(null)}>Dismiss</button>
        </div>
      )}
      <div className="token-create-row">
        <input value={label} onChange={(event) => setLabel(event.target.value)} placeholder="Token label (e.g. ci-readonly)" maxLength={64} />
        <button className="primary-action" onClick={() => void create()} disabled={busy}><KeyRound size={15} /> {busy ? "Creating…" : "Create token"}</button>
      </div>
      {tokens && tokens.length > 0 ? (
        <table className="module-table token-table">
          <thead><tr><th>Label</th><th>Prefix</th><th>Created</th><th>Last used</th><th></th></tr></thead>
          <tbody>
            {tokens.map((token) => (
              <tr key={token.id}>
                <td><strong>{token.label}</strong></td>
                <td><code>{token.prefix}…</code></td>
                <td className="muted">{stamp(token.created_at)}</td>
                <td className="muted">{token.last_used ? stamp(token.last_used) : "never"}</td>
                <td><button className="icon-only danger" onClick={() => setConfirmRevoke({ id: token.id, label: token.label })} aria-label={`Revoke ${token.label}`}><Trash2 size={14} /></button></td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : <p className="muted">No API tokens yet. Create one for programmatic / CI access — it authenticates as you via <code>Authorization: Bearer …</code>.</p>}
      {confirmRevoke && (
        <ConfirmDialog
          title="Revoke API token?"
          message={<>Revoking <strong>{confirmRevoke.label}</strong> immediately breaks any CI job or script that authenticates with it. This cannot be undone.</>}
          confirmLabel="Revoke"
          danger
          onConfirm={() => { const id = confirmRevoke.id; setConfirmRevoke(null); void revoke(id); }}
          onCancel={() => setConfirmRevoke(null)}
        />
      )}
    </div>
  );
}

function ApiTokenField() {
  const [value, setValue] = useState(getApiToken());
  const [saved, setSaved] = useState(false);
  return (
    <div className="token-field">
      <label className="param-field">
        <span>Access token — for remote/tunnel daemons started with SNDR_GUI_TOKEN</span>
        <input
          type="password"
          value={value}
          onChange={(event) => { setValue(event.target.value); setSaved(false); }}
          placeholder="leave empty for localhost (no auth)"
        />
      </label>
      <div className="config-actions">
        <span className="config-actions-note">{saved ? "Saved — sent as Authorization: Bearer" : "Stored in this browser only"}</span>
        <button className="ghost-button" onClick={() => { setApiToken(""); setValue(""); setSaved(true); }}>Clear</button>
        <button className="primary-action" onClick={() => { setApiToken(value); setSaved(true); }}>
          <KeyRound size={14} /> Save token
        </button>
      </div>
    </div>
  );
}

// Telegram alerts / notifications settings — the global home for the engine
// health-watch config (also reachable from the Containers panel's bell button).
function NotificationSettings() {
  const [cfg, setCfg] = useState<AlertConfig | null>(null);
  const [chatId, setChatId] = useState("");
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);

  function reload() {
    api.alertsConfig().then((c) => { setCfg(c); setChatId(c.chat_id); }).catch((e) => setMsg({ ok: false, text: e instanceof Error ? e.message : String(e) }));
  }
  useEffect(reload, []);

  async function save(enabled?: boolean) {
    setBusy(true); setMsg(null);
    try {
      const next = await api.alertsSetConfig({ enabled: enabled ?? cfg?.enabled, chat_id: chatId, ...(token ? { bot_token: token } : {}) });
      setCfg(next); setToken(""); setMsg({ ok: true, text: "Saved." });
    } catch (e) { setMsg({ ok: false, text: e instanceof Error ? e.message : String(e) }); }
    finally { setBusy(false); }
  }
  async function test() {
    setBusy(true); setMsg(null);
    try { const r = await api.alertsTest(); setMsg({ ok: r.ok, text: r.ok ? "Test sent — check Telegram." : (r.error || "Send failed") }); }
    catch (e) { setMsg({ ok: false, text: e instanceof Error ? e.message : String(e) }); }
    finally { setBusy(false); }
  }

  if (!cfg) return <SkeletonLines count={4} />;
  return (
    <div className="notif">
      <div className="notif-grid">
        <div className="notif-fields">
          <div className="notif-row"><span>Enabled</span>
            <button className={`toggle ${cfg.enabled ? "on" : ""}`} disabled={busy} onClick={() => void save(!cfg.enabled)} aria-pressed={cfg.enabled} aria-label="Enable alerts"><span className="toggle-knob" /></button>
          </div>
          <label className="notif-row"><span>Telegram chat ID</span>
            <input value={chatId} onChange={(e) => setChatId(e.target.value)} placeholder="e.g. 123456789" />
          </label>
          <label className="notif-row"><span>Bot token</span>
            <input type="password" value={token} onChange={(e) => setToken(e.target.value)} placeholder={cfg.has_token ? "•••••• (stored — leave blank to keep)" : "123456:ABC-DEF…"} />
          </label>
          <div className="notif-status">
            <span className={`sev ${cfg.has_token ? "clean" : "low"}`}>{cfg.has_token ? "token stored" : "no token"}</span>
            <span className={`sev ${cfg.configured ? "clean" : "low"}`}>{cfg.configured ? "configured" : "incomplete"}</span>
            <span className={`sev ${cfg.enabled ? "clean" : "low"}`}>{cfg.enabled ? "watching" : "off"}</span>
          </div>
          <div className="notif-actions">
            <button className="primary-button" disabled={busy} onClick={() => void save()}>{busy ? <Loader2 size={13} className="spin" /> : <Settings size={13} />} Save</button>
            <button className="ghost-button" disabled={busy || !cfg.configured} onClick={() => void test()}><Send size={13} /> Send test</button>
          </div>
          {msg && <div className={msg.ok ? "notif-ok" : "notif-err"}>{!msg.ok && <AlertTriangle size={13} />} {msg.text}</div>}
        </div>
        <div className="notif-help">
          <h4><Bell size={14} /> What fires</h4>
          <ul>
            <li><b>🔴 DOWN</b> — a managed engine container exits / OOM-kills / is stopped.</li>
            <li><b>🟢 Recovered</b> — it comes back to running.</li>
          </ul>
          <h4>Set up Telegram</h4>
          <ol>
            <li>Message <code>@BotFather</code> → <code>/newbot</code> → copy the <b>bot token</b>.</li>
            <li>Message your new bot once, then open <code>api.telegram.org/bot&lt;token&gt;/getUpdates</code> and copy your <b>chat ID</b>.</li>
            <li>Paste both above, enable, and <b>Send test</b>.</li>
          </ol>
          <p className="notif-note">Token is stored encrypted. Env <code>SNDR_TELEGRAM_BOT_TOKEN</code> / <code>SNDR_TELEGRAM_CHAT_ID</code> / <code>SNDR_ALERTS=1</code> also work for headless deploys. Saving requires the daemon to run with <code>SNDR_ENABLE_APPLY=1</code>.</p>
        </div>
      </div>
    </div>
  );
}

function AppearanceSettings({
  settings,
  onSettings
}: {
  settings: GuiSettings;
  onSettings: (patch: Partial<GuiSettings>) => void;
}) {
  return (
    <div className="settings-grid">
      <SettingGroup title="Theme" icon={<Palette size={16} />}>
        <SegmentedSetting
          value={settings.theme}
          options={[
            ["light", "Light"],
            ["dark", "Dark"],
            ["carbon", "Carbon"],
            ["lime", "Lime"]
          ]}
          onChange={(theme) => onSettings({ theme: theme as ThemeMode })}
        />
      </SettingGroup>
      <SettingGroup title="Density" icon={<Rows3 size={16} />}>
        <SegmentedSetting
          value={settings.density}
          options={[
            ["comfortable", "Comfortable"],
            ["compact", "Compact"]
          ]}
          onChange={(density) => onSettings({ density: density as DensityMode })}
        />
      </SettingGroup>
      <SettingGroup title="Accent" icon={<Sparkles size={16} />}>
        <SwatchSetting
          value={settings.accent}
          options={["teal", "blue", "emerald", "amber"]}
          onChange={(accent) => onSettings({ accent: accent as AccentMode })}
        />
      </SettingGroup>
      <SettingGroup title="Detail Mode" icon={<PanelLeft size={16} />}>
        <SegmentedSetting
          value={settings.detailMode}
          options={[
            ["operator", "Operator"],
            ["engineer", "Engineer"]
          ]}
          onChange={(detailMode) => onSettings({ detailMode: detailMode as DetailMode })}
        />
      </SettingGroup>
      <SettingGroup title="Visual Layers" icon={<Route size={16} />}>
        <ToggleSetting
          label="Connection map"
          active={settings.showConnectionMap}
          onClick={() => onSettings({ showConnectionMap: !settings.showConnectionMap })}
        />
        <ToggleSetting
          label="Auto refresh"
          active={settings.autoRefresh}
          onClick={() => onSettings({ autoRefresh: !settings.autoRefresh })}
        />
      </SettingGroup>
    </div>
  );
}

function SettingGroup({
  title,
  icon,
  children
}: {
  title: string;
  icon: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="setting-group">
      <h3>
        {icon}
        {title}
      </h3>
      {children}
    </section>
  );
}

function SegmentedSetting({
  value,
  options,
  onChange
}: {
  value: string;
  options: Array<[string, string]>;
  onChange: (value: string) => void;
}) {
  return (
    <div className="settings-segmented">
      {options.map(([id, label]) => (
        <button className={value === id ? "active" : ""} key={id} onClick={() => onChange(id)}>
          {label}
        </button>
      ))}
    </div>
  );
}

function SwatchSetting({
  value,
  options,
  onChange
}: {
  value: string;
  options: string[];
  onChange: (value: string) => void;
}) {
  return (
    <div className="swatch-row">
      {options.map((option) => (
        <button
          aria-label={`Use ${option} accent`}
          className={`swatch ${option} ${value === option ? "active" : ""}`}
          key={option}
          onClick={() => onChange(option)}
        />
      ))}
    </div>
  );
}

function ToggleSetting({
  label,
  active,
  onClick
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button className="settings-toggle" onClick={onClick}>
      <span>{label}</span>
      <i className={active ? "active" : ""} />
    </button>
  );
}

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

function modelFamily(id: string): string {
  if (id.startsWith("qwen")) return "Qwen 3.6";
  if (id.startsWith("gemma")) return "Gemma 4";
  const head = id.split("-")[0];
  return head.charAt(0).toUpperCase() + head.slice(1);
}

function itemBadges(item: V2ConfigItem): CatalogBadge[] {
  const f = item.fields ?? {};
  const out: CatalogBadge[] = [];
  const push = (label: unknown, tone: CatalogBadge["tone"] = "neutral") => {
    if (label === null || label === undefined || label === "") return;
    out.push({ label: String(label), tone });
  };
  if (item.kind === "model") {
    push(f.quantization ?? f.dtype, "accent");
    if (f.kv_cache_dtype && f.kv_cache_dtype !== "auto") push(`KV ${f.kv_cache_dtype}`);
    if (Number(f.patch_count) > 0) push(`${f.patch_count} patches`, "ok");
  } else if (item.kind === "hardware") {
    push(f.n_gpus ? `${f.n_gpus}× GPU` : null, "accent");
    if (f.max_model_len) push(`${Math.round(Number(f.max_model_len) / 1024)}k ctx`);
    if (f.runtime_default) push(String(f.runtime_default));
  } else if (item.kind === "profile") {
    push(item.parent_model, "accent");
    if (f.max_num_seqs) push(`seqs ${f.max_num_seqs}`);
    if (Number(f.enable_delta) > 0) push(`+${f.enable_delta} patch`, "ok");
  } else if (item.kind === "preset") {
    push(f.model ?? item.model, "accent");
    push(f.hardware ?? item.hardware);
  }
  return out.slice(0, 3);
}

// ModelFitCard + ModelFitMatrix extracted to ./sections/catalog-cards.

function ModelSummaryStrip({ models, activeId }: { models: V2ConfigItem[]; activeId: string }) {
  const fieldsOf = (m: V2ConfigItem) => (m.fields ?? {}) as Record<string, any>;
  const families = new Set(models.map((m) => modelFamily(m.id))).size;
  const moe = models.filter((m) => String(fieldsOf(m).attention_arch ?? "").includes("moe")).length;
  const toolReady = models.filter((m) => fieldsOf(m).tool_call_parser).length;
  const vrams = models.map((m) => Number(fieldsOf(m).min_total_vram_mib) || 0).filter(Boolean);
  const vramRange = vrams.length ? `${Math.round(Math.min(...vrams) / 1024)}–${Math.round(Math.max(...vrams) / 1024)} GB` : "—";
  const tiles: Array<{ label: string; value: string }> = [
    { label: "Models", value: String(models.length) },
    { label: "Families", value: String(families) },
    { label: "MoE", value: String(moe) },
    { label: "Dense", value: String(models.length - moe) },
    { label: "Tool-ready", value: String(toolReady) },
    { label: "Min VRAM", value: vramRange }
  ];
  return (
    <div className="preset-summary-strip model-summary-strip">
      {tiles.map((tile) => (
        <div className="preset-stat" key={tile.label}>
          <span className="preset-stat-value">{tile.value}</span>
          <span className="preset-stat-label">{tile.label}</span>
        </div>
      ))}
      <div className="preset-stat"><span className="preset-stat-value">{activeId || "—"}</span><span className="preset-stat-label">Selected</span></div>
    </div>
  );
}

function ModelKeyFacts({ fields, def }: { fields: Record<string, any>; def: Record<string, any> }) {
  const caps = (def.capabilities ?? {}) as Record<string, any>;
  const reqs = (def.requires ?? {}) as Record<string, any>;
  const vram = Number(fields.min_total_vram_mib ?? reqs.min_total_vram_mib) || 0;
  const facts: Array<[string, string]> = [
    ["Arch", String(fields.attention_arch ?? caps.attention_arch ?? "—")],
    ["Quant", String(def.quantization ?? fields.quantization ?? "baked/none")],
    ["Dtype", String(def.dtype ?? fields.dtype ?? "—")],
    ["KV cache", String(caps.kv_cache_dtype ?? fields.kv_cache_dtype ?? "—")],
    ["Min VRAM", vram ? `${Math.round(vram / 1024)} GB` : "—"],
    ["Min GPUs", String(fields.min_gpu_count ?? reqs.min_gpu_count ?? "—")],
    ["Patches", String(fields.patch_count ?? Object.keys(def.patches ?? {}).length ?? "—")]
  ];
  return (
    <div className="model-keyfacts">
      {facts.map(([label, value]) => (
        <div className="model-keyfact" key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
        </div>
      ))}
    </div>
  );
}

// KvEnvelopeCard extracted to ./sections/catalog-cards.

function ModelsWorkbench({
  catalog,
  presets,
  selectedPreset,
  selectedModel,
  composed,
  card,
  patchCount,
  runtimeTarget,
  patchPolicy,
  onPreset
}: {
  catalog: V2ConfigCatalog | null;
  presets: PresetRecord[];
  selectedPreset: string;
  selectedModel: string | null;
  composed: Record<string, unknown>;
  card: Record<string, unknown>;
  patchCount: number;
  runtimeTarget: string;
  patchPolicy: string;
  onPreset: (id: string) => void;
}) {
  const models = catalog?.models ?? [];
  const [filter, setFilter] = useState("");
  const [picked, setPicked] = useState<string | null>(null);
  const activeId = picked ?? selectedModel ?? models[0]?.id ?? "";
  const active = models.find((model) => model.id === activeId) ?? null;
  const visible = models.filter((model) => {
    const needle = filter.trim().toLowerCase();
    if (!needle) return true;
    return [model.id, model.title, model.summary]
      .filter(Boolean)
      .some((value) => String(value).toLowerCase().includes(needle));
  });
  const groupedModels = useMemo(() => {
    const map = new Map<string, V2ConfigItem[]>();
    visible.forEach((model) => {
      const family = modelFamily(model.id);
      if (!map.has(family)) map.set(family, []);
      map.get(family)!.push(model);
    });
    return Array.from(map.entries());
  }, [visible]);
  const modelPresets = presets.filter((preset) => preset.model === activeId);
  const { data: cacheReport } = useFetch((signal) => api.modelsCache(signal), []);
  // Real transformer dims (layers / KV heads / head_dim / params / quant) — used
  // for KV+VRAM math; surfaced here so the catalog isn't just curated metadata.
  // calcModels keys by family ("qwen3.6-27b-int4"), the catalog by full variant;
  // match on the "<family>-<size>" base (e.g. qwen3.6-27b, gemma-4-31b).
  const { data: archMeta } = useFetch(() => api.calcModels(), []);
  const archEntry = (() => {
    if (!archMeta || !activeId) return null;
    if (archMeta.models[activeId]) return { key: activeId, arch: archMeta.models[activeId] };
    const base = (id: string) => (id.toLowerCase().match(/^(.+?-\d+b)/)?.[1] ?? id.toLowerCase());
    const t = base(activeId);
    for (const [k, v] of Object.entries(archMeta.models)) if (base(k) === t) return { key: k, arch: v };
    return null;
  })();
  const arch = archEntry?.arch ?? null;
  const archKey = archEntry?.key ?? null;
  const { data: fullDef, state: defState } = useFetch(
    (signal) => api.v2Layer("model", activeId, signal),
    [activeId],
    { enabled: Boolean(activeId) }
  );

  const def = (fullDef?.definition ?? {}) as Record<string, any>;
  const caps = (def.capabilities ?? {}) as Record<string, any>;
  const reqs = (def.requires ?? {}) as Record<string, any>;
  const vers = (def.versions ?? {}) as Record<string, any>;
  const spec = (caps.spec_decode ?? {}) as Record<string, any>;
  const patchMatrix = (def.patches ?? {}) as Record<string, string>;
  const attribution = (def.patches_attribution ?? {}) as Record<string, any>;
  const notes: string[] = Array.isArray(def.notes) ? def.notes : [];
  // Representative rig for the fit envelope: prefer a hardware a preset actually
  // uses with this model, else the model's stated minimums, else a 2×24GB A5000.
  const envHw = (() => {
    const hw = catalog?.hardware.find((h) => h.id === modelPresets[0]?.hardware);
    if (hw) return { tp: Number(hw.fields?.n_gpus) || 2, vram: Number(hw.fields?.min_vram_per_gpu_mib) || 24564, label: hw.id };
    return { tp: Number(reqs.min_gpu_count) || 2, vram: 24564, label: "2×24GB (default)" };
  })();
  const dval = (value: unknown): string => {
    if (Array.isArray(value)) return value.length ? value.map(String).join(", ") : "-";
    if (value === null || value === undefined || value === "") return "-";
    if (typeof value === "boolean") return value ? "yes" : "no";
    return String(value);
  };

  return (
    <div className="models-view">
      <ModelSummaryStrip models={models} activeId={activeId} />
    <section className="models-workbench">
      <section className="model-list-panel">
        <div className="config-panel-title">
          <Box size={16} />
          <strong>Model Catalog</strong>
          <span>{models.length}</span>
        </div>
        <label className="search-box">
          <Search size={15} />
          <input value={filter} onChange={(event) => setFilter(event.target.value)} placeholder="Search model" />
        </label>
        <div className="catalog-list">
          {groupedModels.map(([family, items]) => (
            <div className="catalog-group" key={family}>
              <div className="catalog-group-head">
                <span>{family}</span>
                <small>{items.length}</small>
              </div>
              {items.map((model) => (
                <CatalogCard
                  key={model.id}
                  icon={<Box size={16} />}
                  id={model.id}
                  title={[model.fields?.attention_arch, model.fields?.dtype].filter(Boolean).map(String).join(" · ") || model.title}
                  badges={itemBadges(model)}
                  active={model.id === activeId}
                  onClick={() => setPicked(model.id)}
                />
              ))}
            </div>
          ))}
          {visible.length === 0 && (
            <EmptyState
              icon={<Boxes size={22} />}
              title="No models match"
              message={filter ? <>Nothing in the catalog matches “{filter}”.</> : "The model catalog is empty."}
              action={filter ? { label: "Clear search", icon: <X size={14} />, onClick: () => setFilter("") } : undefined}
            />
          )}
        </div>
        <div className="catalog-foot">
          {visible.length} of {models.length} models · {groupedModels.length} famil{groupedModels.length === 1 ? "y" : "ies"}
        </div>
      </section>

      <section className="model-detail-tabbed">
        <div className="model-detail-head">
          <Cpu size={18} />
          <div>
            <strong>{def.title ?? active?.title ?? activeId ?? "No model"}</strong>
            <span>{def.model_path ?? active?.summary ?? "model definition"}</span>
          </div>
          <StatusBadge status={defState === "loading" ? "partial" : active ? "available" : "missing"} />
        </div>
        {active && <ModelKeyFacts fields={(active.fields ?? {}) as Record<string, any>} def={def} />}
        <TabbedSection
          id={`model-${activeId}`}
          tabs={[
            {
              id: "overview",
              label: "Overview",
              icon: <Box size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Identity" icon={<Box size={18} />} desc="Serving name, precision, provenance and trust settings.">
                    <InfoRows
                      rows={[
                        ["Model id", activeId || "-"],
                        ["Served name", dval(def.served_model_name)],
                        ["Maintainer", dval(def.maintainer)],
                        ["Quantization", dval(def.quantization)],
                        ["Dtype", dval(def.dtype)],
                        ["Trust remote code", dval(def.trust_remote_code)],
                        ["License", dval(def.license)],
                        ["Validated", dval(def.last_validated)]
                      ]}
                    />
                  </ModuleCard>
                  <ModuleCard title="Architecture" icon={<Cpu size={18} />} desc="Real transformer dimensions — the basis for KV-cache and VRAM sizing.">
                    {arch ? (() => {
                      // KV bytes/token = 2 (K+V) × layers × kv_heads × head_dim × elem_bytes.
                      const kvTok = (elem: number) => 2 * arch.num_layers * arch.num_kv_heads * arch.head_dim * elem;
                      const mbPerK = (elem: number) => (kvTok(elem) * 1024) / (1024 * 1024);
                      const gbAt = (ctx: number, elem: number) => (kvTok(elem) * ctx) / (1024 ** 3);
                      return (
                        <>
                          <InfoRows rows={[
                            ["Parameters", `${arch.params_b}B${arch.is_moe && arch.active_params_b ? ` · ${arch.active_params_b}B active (MoE)` : ""}`],
                            ["Type", arch.is_moe ? "Mixture-of-Experts" : "Dense"],
                            ["Layers", String(arch.num_layers)],
                            ["KV heads", String(arch.num_kv_heads)],
                            ["Head dim", String(arch.head_dim)],
                            ["Weight precision", `${arch.weight_bits}-bit`]
                          ]} />
                          <div className="kv-footprint">
                            <div className="kv-footprint-head"><MemoryStick size={13} /> KV-cache footprint (1 request)</div>
                            <div className="kv-footprint-rows">
                              <div><span>fp8</span><b>{mbPerK(1).toFixed(1)} MB</b> / 1K · <b>{gbAt(32768, 1).toFixed(1)} GB</b> @ 32K</div>
                              <div><span>fp16</span><b>{mbPerK(2).toFixed(1)} MB</b> / 1K · <b>{gbAt(32768, 2).toFixed(1)} GB</b> @ 32K</div>
                            </div>
                          </div>
                        </>
                      );
                    })() : <p className="muted">No architecture metadata for this model id ({activeId || "—"}).</p>}
                  </ModuleCard>
                  <ModuleCard title="Fit envelope" icon={<Gauge size={18} />} desc="Does it fit? Context × concurrency headroom on a representative rig." wide>
                    <KvEnvelopeCard modelKey={archKey} tp={envHw.tp} vram={envHw.vram} rigLabel={envHw.label} />
                  </ModuleCard>
                  <ModuleCard
                    title="Local Cache"
                    icon={<HardDrive size={18} />}
                    desc={`Checkpoint presence on the API daemon host${cacheReport ? ` (${cacheReport.host})` : ""}.`}
                  >
                    {(() => {
                      const entry = cacheReport?.models.find((m) => m.model_id === activeId);
                      if (!cacheReport) return <p className="muted">Checking cache…</p>;
                      if (!entry) return <p className="muted">No cache entry for this model.</p>;
                      return (
                        <>
                          <InfoRows
                            rows={[
                              ["Checkpoint path", entry.model_path || "-"],
                              ["Present on host", entry.present ? "yes" : "no"],
                              ["Size", entry.size_mib != null ? formatVram(entry.size_mib) : entry.present ? "unknown" : "-"]
                            ]}
                          />
                          <p className="fit-note">
                            {entry.present
                              ? "Checkpoint directory is present on the daemon host."
                              : "Absent on the daemon host — expected when controlling a remote GPU host from a laptop."}
                          </p>
                        </>
                      );
                    })()}
                  </ModuleCard>
                  <ModuleCard title={`Presets using ${activeId || "model"}`} icon={<Database size={18} />} desc="Catalog presets that reference this model — click to load.">
                    {modelPresets.length ? (
                      <div className="model-preset-chips">
                        {modelPresets.map((preset) => (
                          <button key={preset.id} className={preset.id === selectedPreset ? "active" : ""} onClick={() => onPreset(preset.id)}>
                            <strong>{preset.id}</strong>
                            <small>{preset.profile ?? "no profile"}</small>
                          </button>
                        ))}
                      </div>
                    ) : (
                      <p className="muted">No presets reference this model yet.</p>
                    )}
                  </ModuleCard>
                  <ModuleCard title="Notes" icon={<FileText size={18} />} desc="Maintainer notes and migration history.">
                    {notes.length ? (
                      <ul className="model-notes">{notes.map((note, index) => (<li key={index}>{note}</li>))}</ul>
                    ) : (
                      <p className="muted">No maintainer notes.</p>
                    )}
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "capabilities",
              label: "Capabilities",
              icon: <Layers3 size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Capabilities" icon={<Layers3 size={18} />} desc="Attention arch, parsers and speculative decode.">
                    <InfoRows
                      rows={[
                        ["Attention arch", dval(caps.attention_arch)],
                        ["Tool parser", dval(caps.tool_call_parser)],
                        ["Reasoning parser", dval(caps.reasoning_parser)],
                        ["Auto tool choice", dval(caps.enable_auto_tool_choice)],
                        ["KV cache dtype", dval(caps.kv_cache_dtype)],
                        ["Spec decode", `${dval(spec.method)} / K=${dval(spec.num_speculative_tokens)}`]
                      ]}
                    />
                  </ModuleCard>
                  <ModuleCard title="Version Pins" icon={<GitBranch size={18} />} desc="Genesis and vLLM pins this model was validated on.">
                    <InfoRows
                      rows={[
                        ["Genesis pin min", dval(vers.genesis_pin_min)],
                        ["vLLM pin required", dval(vers.vllm_pin_required)],
                        ["Reference metrics", dval(vers.reference_metrics_ref)],
                        ["Pin hold", dval(vers.pin_hold)]
                      ]}
                    />
                  </ModuleCard>
                  <ModuleCard title="Generation & Serving" icon={<SlidersHorizontal size={18} />} desc="Chat template, tool/reasoning parsing and sampling defaults." wide>
                    <InfoRows
                      rows={[
                        ["Served name", dval(def.served_model_name)],
                        ["Chat template", def.chat_template ? "custom template" : "model default"],
                        ["Tool parser", dval(caps.tool_call_parser)],
                        ["Reasoning parser", dval(caps.reasoning_parser)],
                        ["Auto tool choice", dval(caps.enable_auto_tool_choice)]
                      ]}
                    />
                    {(() => {
                      const gen = asRecord(def.override_generation_config);
                      const keys = Object.keys(gen);
                      return keys.length ? (
                        <>
                          <h5 className="model-subhead">Sampling overrides</h5>
                          <InfoRows rows={keys.map((key) => [key.replace(/_/g, " "), dval(gen[key])] as [string, string])} />
                        </>
                      ) : (
                        <p className="muted model-gen-none">No sampling overrides — uses the model's generation defaults.</p>
                      );
                    })()}
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "fit",
              label: "Hardware fit",
              icon: <Gauge size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Requirements" icon={<ShieldCheck size={18} />} desc="Minimum GPUs, VRAM and CUDA capability.">
                    <InfoRows
                      rows={[
                        ["Min GPUs", dval(reqs.min_gpu_count)],
                        ["Min VRAM", formatVram(reqs.min_total_vram_mib)],
                        ["Min CUDA cap", Array.isArray(reqs.min_cuda_capability) ? reqs.min_cuda_capability.join(".") : "-"],
                        ["Arch blocklist", dval(reqs.rig_arch_blocklist)]
                      ]}
                    />
                  </ModuleCard>
                  <ModuleCard title="Hardware Fit" icon={<Gauge size={18} />} desc="Check this model against a rig: GPU count, CUDA capability and VRAM context." wide>
                    <ModelFitCard
                      modelId={activeId}
                      hardwareOptions={(catalog?.hardware ?? []).map((item) => item.id)}
                      defaultHardware={modelPresets[0]?.hardware ?? ""}
                    />
                  </ModuleCard>
                  <ModuleCard title="Fit Matrix" icon={<Table2 size={18} />} desc="Where this model can run across every catalogued rig — fits, blockers and VRAM headroom." wide>
                    <ModelFitMatrix modelId={activeId} hardwareIds={(catalog?.hardware ?? []).map((item) => item.id)} />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "patches",
              label: "Patch matrix",
              icon: <Wrench size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Patch Matrix" icon={<Wrench size={18} />} desc="Canonical env-flag overrides shipped with this model." wide>
                    <PatchMatrixViewer patches={patchMatrix} attribution={attribution} loading={defState === "loading"} />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "runtime",
              label: "Runtime",
              icon: <SlidersHorizontal size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Runtime Envelope" icon={<SlidersHorizontal size={18} />} desc={`Composed runtime for ${selectedPreset}.`} wide>
                    <RuntimeEnvelopePanel card={card} composed={composed} patchCount={patchCount} />
                  </ModuleCard>
                  <ModuleCard title="Config Draft" icon={<Code2 size={18} />} desc="Local runtime draft (diff + YAML preview)." wide>
                    <ConfigDraftEditor
                      selectedPreset={selectedPreset}
                      composed={composed}
                      runtimeTarget={runtimeTarget}
                      patchPolicy={patchPolicy}
                    />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "cache",
              label: "Cache & download",
              icon: <Download size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Checkpoint Cache" icon={<Download size={18} />} desc="Model weights present on the daemon host. Queue a pull for absent checkpoints." wide>
                    <ModelManagementPanel />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "edit",
              label: "Edit",
              icon: <PackageCheck size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={`Visual Editor — ${activeId}`} icon={<Wrench size={18} />} desc="Full model definition editing (adaptive). Saves an operator-local copy." wide>
                    <LayerEditor kind="model" layerId={activeId} />
                  </ModuleCard>
                </ModuleGrid>
              )
            }
          ]}
        />
      </section>
    </section>
    </div>
  );
}

// ElementKind/FieldSpec/ELEMENT_FIELDS + discoverExtraFields/groupFields/fieldWarning/ElementField extracted to ./sections/element-fields.
// LayerEditor extracted to ./sections/layer-editor.

function ConfigElementEditor({ catalog }: { catalog: V2ConfigCatalog | null }) {
  const itemsByKind: Record<ElementKind, V2ConfigItem[]> = {
    model: catalog?.models ?? [],
    hardware: catalog?.hardware ?? [],
    profile: catalog?.profiles ?? [],
    preset: catalog?.presets ?? []
  };
  const [kind, setKind] = useState<ElementKind>("model");
  const [itemId, setItemId] = useState("");
  const [filter, setFilter] = useState("");
  const [edited, setEdited] = useState<Record<string, any> | null>(null);
  const [source, setSource] = useState("");
  const [state, setState] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const [applyResult, setApplyResult] = useState<V2LayerApplyResult | null>(null);
  const [applying, setApplying] = useState(false);

  const items = itemsByKind[kind];
  const activeId = itemId || items[0]?.id || "";
  const visible = items.filter((item) => {
    const needle = filter.trim().toLowerCase();
    return !needle || item.id.toLowerCase().includes(needle) || (item.title ?? "").toLowerCase().includes(needle);
  });

  useEffect(() => {
    if (!activeId) {
      setEdited(null);
      return;
    }
    const controller = new AbortController();
    setState("loading");
    setError(null);
    setApplyResult(null);
    api.v2Layer(kind, activeId, controller.signal)
      .then((layer) => {
        if (controller.signal.aborted) return;
        setEdited(layer.definition as Record<string, any>);
        setSource(layer.source);
        setState("ready");
      })
      .catch((err) => {
        if (controller.signal.aborted) return;
        setEdited(null);
        setState("error");
        setError(err instanceof Error ? err.message : String(err));
      });
    return () => controller.abort();
  }, [kind, activeId]);

  // Adaptive: curated fields first, then any extra scalar leaves the loaded
  // definition actually contains (so new/unknown fields still render + edit).
  const fields = useMemo(() => {
    const curated = ELEMENT_FIELDS[kind];
    if (!edited) return curated;
    const known = new Set(curated.map((spec) => spec.path));
    return [...curated, ...discoverExtraFields(edited, known)];
  }, [kind, edited]);
  const yaml = edited ? objToYaml(edited) : ["# Select an element to edit"];

  async function runSave() {
    if (!edited) return;
    setApplying(true);
    try {
      const result = await api.v2LayerApply({ kind, layer_id: activeId, yaml_text: yaml.join("\n") + "\n" });
      setApplyResult(result);
    } catch (err) {
      setApplyResult({
        kind, layer_id: activeId, target_path: "", action: "create", written: false,
        bytes_written: 0, status: "blocked", message: err instanceof Error ? err.message : String(err), blocked_reasons: []
      });
    } finally {
      setApplying(false);
    }
  }

  return (
    <section className="element-editor">
      <aside className="element-rail">
        <div className="config-panel-title">
          <Layers3 size={16} />
          <strong>Element</strong>
        </div>
        <p className="config-panel-desc">Choose a config layer and item to inspect and edit.</p>
        <div className="element-kind-tiles">
          {([
            { id: "model", icon: <Box size={15} /> },
            { id: "hardware", icon: <Cpu size={15} /> },
            { id: "profile", icon: <SlidersHorizontal size={15} /> },
            { id: "preset", icon: <Database size={15} /> }
          ] as Array<{ id: ElementKind; icon: ReactNode }>).map((option) => (
            <button
              key={option.id}
              type="button"
              className={`element-kind-tile${kind === option.id ? " active" : ""}`}
              onClick={() => { setKind(option.id); setItemId(""); }}
            >
              {option.icon}
              <span>{option.id}</span>
              <small>{itemsByKind[option.id].length}</small>
            </button>
          ))}
        </div>
        <label className="search-box">
          <Search size={15} />
          <input value={filter} onChange={(event) => setFilter(event.target.value)} placeholder={`Search ${kind}`} />
        </label>
        <div className="catalog-list">
          {visible.map((item) => (
            <CatalogCard
              key={item.id}
              icon={
                item.kind === "hardware" ? <Cpu size={16} />
                : item.kind === "profile" ? <SlidersHorizontal size={16} />
                : item.kind === "preset" ? <Database size={16} />
                : <Box size={16} />
              }
              id={item.id}
              title={item.title && item.title !== item.id ? item.title : undefined}
              badges={itemBadges(item)}
              active={item.id === activeId}
              onClick={() => setItemId(item.id)}
            />
          ))}
          {visible.length === 0 && (
            <EmptyState
              icon={<Layers size={22} />}
              title={`No ${kind} matches`}
              message={filter ? <>Nothing in {kind} matches “{filter}”.</> : `No ${kind} elements in the catalog.`}
              action={filter ? { label: "Clear search", icon: <X size={14} />, onClick: () => setFilter("") } : undefined}
            />
          )}
        </div>
      </aside>

      <section className="element-form-panel">
        <div className="config-panel-title">
          <SlidersHorizontal size={16} />
          <strong>{activeId || "No selection"}</strong>
          <StatusBadge status={state === "loading" ? "partial" : state === "error" ? "missing" : "available"} />
        </div>
        <p className="element-source">{source}</p>
        {error && <div className="config-plan-error"><AlertCircle size={15} /><span>{error}</span></div>}
        {edited ? (
          <div className="element-groups">
            {groupFields(fields).map(([group, groupFieldsList]) => (
              <div className="element-group" key={group}>
                {group && <div className="element-group-head">{group}</div>}
                <div className="element-fields">
                  {groupFieldsList.map((spec) => (
                    <ElementField
                      key={spec.path}
                      spec={spec}
                      value={getIn(edited, spec.path)}
                      onChange={(value) => setEdited((current) => (current ? setIn(current, spec.path, value) : current))}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <SkeletonLines count={6} />
        )}
        {applyResult && (
          <div className={`element-apply ${applyResult.status}`}>
            <span className="finding-icon">
              {applyResult.status === "applied" ? <CheckCircle2 size={14} /> : <AlertCircle size={14} />}
            </span>
            <div>
              <strong>{applyResult.status === "applied" ? "Saved to user dir" : applyResult.status}</strong>
              <small>{applyResult.target_path || applyResult.message}</small>
              {applyResult.blocked_reasons.length > 0 && <small>{applyResult.blocked_reasons.join("; ")}</small>}
            </div>
          </div>
        )}
        <div className="config-actions">
          <span className="config-actions-note">Edits update the YAML draft on the right</span>
          <button className="ghost-button" onClick={() => { setItemId(activeId); setState("idle"); }}>
            <RefreshCw size={14} /> Reload
          </button>
          <button className="primary-action" onClick={() => void runSave()} disabled={!edited || applying}>
            <PackageCheck size={14} /> {applying ? "Saving…" : "Save to user dir"}
          </button>
        </div>
      </section>

      <section className="element-yaml-panel">
        <div className="config-panel-title">
          <Code2 size={16} />
          <strong>{kind}.yaml</strong>
          <span>{yaml.length} lines</span>
        </div>
        <CodeBlock lines={yaml} />
        <p className="element-note">
          Full definition with your edits. Copy to export, or use the Compose tab → Apply Plan to persist an
          operator-local preset. Direct model/hardware/profile writes need a layer-apply API.
        </p>
      </section>
    </section>
  );
}

function ConfigsSection(props: {
  catalog: V2ConfigCatalog | null;
  preview: V2ConfigPreview | null;
  selectedPreset: string;
  userPresets: UserPresetList | null;
  onPreview: (preview: V2ConfigPreview) => void;
  onUserPresetsRefresh: () => void;
}) {
  const [tab, setTab] = useState<"compose" | "edit">("compose");
  return (
    <div className="configs-section">
      <div className="configs-top-tabs">
        <button className={tab === "compose" ? "active" : ""} onClick={() => setTab("compose")}>
          <Layers3 size={15} /> Compose
        </button>
        <button className={tab === "edit" ? "active" : ""} onClick={() => setTab("edit")}>
          <SlidersHorizontal size={15} /> Edit element
        </button>
      </div>
      {tab === "compose" ? <V2ConfigWorkbench {...props} /> : <ConfigElementEditor catalog={props.catalog} />}
    </div>
  );
}

// Visual composition chain — a preset = model + hardware + profile, resolved to
// the runtime config. Makes the layering explicit instead of three side-by-side
// cards the operator has to mentally stitch together.
function CompositionChain({ model, hardware, profile, composed }: {
  model: V2ConfigItem | null; hardware: V2ConfigItem | null; profile: V2ConfigItem | null; composed: Record<string, any>;
}) {
  const layers = [
    { kind: "Model", icon: <Box size={13} />, item: model, optional: false,
      fact: model ? [model.fields?.quantization || model.fields?.dtype, model.fields?.attention_arch].filter(Boolean).join(" · ") : "" },
    { kind: "Hardware", icon: <Server size={13} />, item: hardware, optional: false,
      fact: hardware ? `${hardware.fields?.n_gpus ?? "?"}× GPU${hardware.fields?.min_vram_per_gpu_mib ? ` · ${Math.round(hardware.fields.min_vram_per_gpu_mib / 1024)}GB` : ""}` : "" },
    { kind: "Profile", icon: <SlidersHorizontal size={13} />, item: profile, optional: true,
      fact: profile ? String(profile.fields?.role || profile.status || "override") : "" }
  ];
  const resultFacts = [
    composed?.max_model_len && `${(Number(composed.max_model_len) / 1024).toFixed(0)}K ctx`,
    composed?.max_num_seqs && `${composed.max_num_seqs} seq`,
    composed?.kv_cache_dtype && `KV ${composed.kv_cache_dtype}`
  ].filter(Boolean) as string[];
  return (
    <div className="comp-chain">
      {layers.map((l) => (
        <Fragment key={l.kind}>
          <div className={`comp-layer ${l.item ? "" : l.optional ? "none" : "empty"}`}>
            <div className="comp-layer-head">{l.icon} {l.kind}{l.optional && !l.item ? " (none)" : ""}</div>
            <code className="comp-layer-id">{l.item?.id ?? (l.optional ? "—" : "pick one")}</code>
            {l.fact && <div className="comp-layer-fact">{l.fact}</div>}
          </div>
          <ChevronRight className="comp-arrow" size={16} />
        </Fragment>
      ))}
      <div className="comp-layer result">
        <div className="comp-layer-head"><Rocket size={13} /> Composed</div>
        <code className="comp-layer-id">runtime config</code>
        <div className="comp-layer-fact">{resultFacts.length ? resultFacts.join(" · ") : "preview to resolve"}</div>
      </div>
    </div>
  );
}

// The full resolved runtime config (every scalar the layers compose to) — the
// composer previously surfaced only a handful of key fields.
function ResolvedConfig({ composed }: { composed: Record<string, any> }) {
  const entries = Object.entries(composed || {})
    .filter(([, v]) => v !== null && v !== undefined && v !== "" && typeof v !== "object")
    .sort((a, b) => a[0].localeCompare(b[0]));
  const envCount = Object.keys((composed?.genesis_env as Record<string, unknown>) ?? {}).length;
  if (!entries.length) return null;
  return (
    <details className="resolved-config" open>
      <summary>Resolved runtime config — {entries.length} parameters{envCount ? ` · ${envCount} patch flags` : ""}</summary>
      <div className="resolved-grid">
        {entries.map(([k, v]) => (
          <div key={k} className="resolved-row"><code>{k}</code><span>{typeof v === "boolean" ? (v ? "yes" : "no") : String(v)}</span></div>
        ))}
      </div>
    </details>
  );
}

function V2ConfigWorkbench({
  catalog,
  preview,
  selectedPreset,
  userPresets,
  onPreview,
  onUserPresetsRefresh
}: {
  catalog: V2ConfigCatalog | null;
  preview: V2ConfigPreview | null;
  selectedPreset: string;
  userPresets: UserPresetList | null;
  onPreview: (preview: V2ConfigPreview) => void;
  onUserPresetsRefresh: () => void;
}) {
  const initialPreset = catalog?.presets.find((preset) => preset.id === selectedPreset) ?? catalog?.presets[0];
  const [modelId, setModelId] = useState(initialPreset?.model ?? "qwen3.6-35b-a3b-fp8");
  const [hardwareId, setHardwareId] = useState(initialPreset?.hardware ?? "a5000-2x-24gbvram-16cpu-128gbram");
  const [profileId, setProfileId] = useState(initialPreset?.profile ?? "");
  const [runtime, setRuntime] = useState(initialPreset?.runtime ?? "docker");
  const [filter, setFilter] = useState("");
  const [draftPresetId, setDraftPresetId] = useState(`gui-draft-${selectedPreset}`);
  const [configPlan, setConfigPlan] = useState<V2ConfigPlan | null>(null);
  const [planState, setPlanState] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [planError, setPlanError] = useState<string | null>(null);
  const [applyResult, setApplyResult] = useState<V2ConfigApplyResult | null>(null);
  const [applyState, setApplyState] = useState<"idle" | "loading" | "done">("idle");
  const [applyError, setApplyError] = useState<string | null>(null);
  const [profileDef, setProfileDef] = useState<V2LayerDefinition | null>(null);
  const [artifactKind, setArtifactKind] = useState<"compose" | "run" | "systemd" | "env">("compose");
  // composed is derived from preview, so depending on `preview` covers it; the
  // `?? {}` is inlined to avoid an unstable new-object dependency each render.
  const baseDraft = useMemo(
    () => buildRuntimeDraft(preview?.composed ?? {}, runtime, "safe"),
    [preview, runtime]
  );
  const [draft, setDraft] = useState<RuntimeConfigDraft>(baseDraft);
  const [yamlText, setYamlText] = useState<string | null>(null);
  useEffect(() => {
    setDraft(baseDraft);
    setYamlText(null);
  }, [baseDraft]);
  const setParam = (patch: Partial<RuntimeConfigDraft>) => setDraft((current) => ({ ...current, ...patch }));

  useEffect(() => {
    if (!profileId) {
      setProfileDef(null);
      return;
    }
    let cancelled = false;
    api.v2Layer("profile", profileId)
      .then((detail) => {
        if (!cancelled) setProfileDef(detail);
      })
      .catch(() => {
        if (!cancelled) setProfileDef(null);
      });
    return () => {
      cancelled = true;
    };
  }, [profileId]);

  useEffect(() => {
    if (!initialPreset) return;
    setModelId(initialPreset.model ?? modelId);
    setHardwareId(initialPreset.hardware ?? hardwareId);
    setProfileId(initialPreset.profile ?? "");
    setRuntime(initialPreset.runtime ?? "docker");
    setDraftPresetId(`gui-draft-${initialPreset.id}`);
    setConfigPlan(null);
    setPlanState("idle");
    setPlanError(null);
    setApplyResult(null);
    setApplyState("idle");
    setApplyError(null);
    // Only resync when catalog/selected preset changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialPreset?.id]);

  const compatibleProfiles = (catalog?.profiles ?? []).filter(
    (profile) => !profile.parent_model || profile.parent_model === modelId
  );
  const visiblePresets = (catalog?.presets ?? []).filter((preset) => {
    const needle = filter.trim().toLowerCase();
    if (!needle) return true;
    return [preset.id, preset.title, preset.model, preset.hardware, preset.profile]
      .filter(Boolean)
      .some((value) => String(value).toLowerCase().includes(needle));
  });
  const selectedModel = catalog?.models.find((item) => item.id === modelId) ?? null;
  const selectedHardware = catalog?.hardware.find((item) => item.id === hardwareId) ?? null;
  const selectedProfile = catalog?.profiles.find((item) => item.id === profileId) ?? null;
  const configCtx: ConfigContext = { presetId: draftPresetId, modelId, hardwareId, profileId, runtime, draft };
  const configCli = buildConfigCli(configCtx);
  const configArtifacts = buildConfigArtifacts(configCtx);
  const configDiffs = runtimeDraftDiff(baseDraft, draft);
  const configYaml = yamlText ?? buildFullConfigYaml(configCtx).join("\n");

  async function runPreview(next?: {
    model?: string | null;
    hardware?: string | null;
    profile?: string | null;
    runtime?: string | null;
  }) {
    const model = next?.model ?? modelId;
    const hardware = next?.hardware ?? hardwareId;
    const profile = next?.profile ?? profileId;
    const runtimeValue = next?.runtime ?? runtime;
    if (!model || !hardware) return;
    setConfigPlan(null);
    setPlanState("idle");
    setPlanError(null);
    onPreview(
      await api.v2ConfigPreview({
        model_id: model,
        hardware_id: hardware,
        profile_id: profile || undefined,
        runtime: runtimeValue || undefined
      })
    );
  }

  function applyPresetItem(preset: V2ConfigItem) {
    const nextModel = preset.model ?? modelId;
    const nextHardware = preset.hardware ?? hardwareId;
    const nextProfile = preset.profile ?? "";
    const nextRuntime = preset.runtime ?? "docker";
    setModelId(nextModel);
    setHardwareId(nextHardware);
    setProfileId(nextProfile);
    setRuntime(nextRuntime);
    setDraftPresetId(`gui-draft-${preset.id}`);
    void runPreview({
      model: nextModel,
      hardware: nextHardware,
      profile: nextProfile,
      runtime: nextRuntime
    });
  }

  async function runConfigPlan() {
    setPlanState("loading");
    setPlanError(null);
    try {
      const nextPlan = await api.v2ConfigPlan({
        preset_id: draftPresetId,
        model_id: modelId,
        hardware_id: hardwareId,
        profile_id: profileId || undefined,
        runtime: runtime || undefined
      });
      setConfigPlan(nextPlan);
      setPlanState("ready");
      setApplyResult(null);
      setApplyState("idle");
      setApplyError(null);
    } catch (err) {
      setPlanState("error");
      setPlanError(err instanceof Error ? err.message : String(err));
    }
  }

  async function runConfigApply() {
    if (!configPlan) return;
    setApplyState("loading");
    setApplyError(null);
    try {
      const result = await api.v2ConfigApply({
        preset_id: draftPresetId,
        model_id: modelId,
        hardware_id: hardwareId,
        profile_id: profileId || undefined,
        runtime: runtime || undefined,
        expected_plan_id: configPlan.plan_id
      });
      setApplyResult(result);
      setApplyState("done");
      if (result.status === "applied") {
        onUserPresetsRefresh();
      }
    } catch (err) {
      setApplyState("done");
      setApplyError(err instanceof Error ? err.message : String(err));
    }
  }

  return (
    <section className="config-workbench">
      <div className="config-control-bar">
        <div className="config-control-selects">
          <ConfigSelect
            label="Model"
            icon={<Box size={16} />}
            value={modelId}
            items={catalog?.models ?? []}
            onChange={(value) => {
              setModelId(value);
              const firstProfile = catalog?.profiles.find((profile) => profile.parent_model === value)?.id ?? "";
              setProfileId(firstProfile);
              void runPreview({ model: value, profile: firstProfile });
            }}
          />
          <ConfigSelect
            label="Hardware"
            icon={<HardDrive size={16} />}
            value={hardwareId}
            items={catalog?.hardware ?? []}
            onChange={(value) => { setHardwareId(value); void runPreview({ hardware: value }); }}
          />
          <ConfigSelect
            label="Profile"
            icon={<Gauge size={16} />}
            value={profileId}
            items={compatibleProfiles}
            allowEmpty
            onChange={(value) => { setProfileId(value); void runPreview({ profile: value }); }}
          />
          <label className="config-select-card">
            <span><Server size={16} /> Runtime</span>
            <select value={runtime} onChange={(event) => { setRuntime(event.target.value); void runPreview({ runtime: event.target.value }); }}>
              {["docker", "podman", "kubernetes", "systemd", "bare"].map((item) => (<option key={item} value={item}>{item}</option>))}
            </select>
            <small>Container / orchestration target</small>
          </label>
          <label className="config-select-card">
            <span><PackageCheck size={16} /> Draft id</span>
            <input
              value={draftPresetId}
              onChange={(event) => { setDraftPresetId(event.target.value); setConfigPlan(null); setPlanState("idle"); setPlanError(null); }}
            />
            <small>Operator-local config name</small>
          </label>
        </div>
        <div className="config-control-actions">
          <button className="ghost-button" onClick={() => void runPreview()}><RefreshCw size={15} /> Preview</button>
          <button className="ghost-button" onClick={() => void runConfigPlan()} disabled={planState === "loading"}>
            <ListChecks size={15} /> {planState === "loading" ? "Planning" : "Plan"}
          </button>
          <button
            className={configPlan?.valid ? "primary-action" : "disabled-launch"}
            disabled={!configPlan?.valid || applyState === "loading"}
            title={configPlan?.valid ? "Write this draft to your operator-local config dir" : "Run Plan first"}
            onClick={() => void runConfigApply()}
          >
            <PackageCheck size={15} /> {applyState === "loading" ? "Applying" : "Apply Plan"}
          </button>
        </div>
      </div>

      <div className="config-status-strip">
        <RailCheck label="Compatible" value={preview?.compatible ? "yes" : "no"} status={preview?.compatible ? "pass" : "warning"} />
        <RailCheck label="Status" value={preview?.status ?? "—"} status={preview?.status ? "pass" : "warning"} />
        <RailCheck label="Context" value={formatTokens(asNumber(preview?.composed.max_model_len))} status="pass" />
        <RailCheck label="Sequences" value={String(asNumber(preview?.composed.max_num_seqs) || "—")} status="pass" />
        <RailCheck label="KV cache" value={asText(preview?.composed.kv_cache_dtype, "—")} status="pass" />
        <RailCheck label="Spec decode" value={asText(preview?.composed.spec_decode_method, "—")} status="pass" />
        <RailCheck label="Patches" value={String(asNumber(preview?.composed.enabled_patches_count) || "—")} status="pass" />
      </div>

      <TabbedSection
        id={`config-${modelId}-${hardwareId}-${profileId}`}
        tabs={[
          {
            id: "layers", label: "Layers", icon: <SlidersHorizontal size={15} />,
            render: () => (
              <ModuleGrid>
                <ModuleCard title="Layer Inspector" icon={<SlidersHorizontal size={18} />} desc="A preset is composed from three layers — model + hardware + profile — resolved to one runtime config." wide>
                  <CompositionChain model={selectedModel} hardware={selectedHardware} profile={selectedProfile} composed={preview?.composed ?? {}} />
                  {selectedProfile?.parent_model && modelId && selectedProfile.parent_model !== modelId && (
                    <div className="comp-conflict">
                      <AlertTriangle size={13} /> Profile <code>{selectedProfile.id}</code> targets model <code>{selectedProfile.parent_model}</code>, but the composer has <code>{modelId}</code> — sizing/patches from this profile may not apply cleanly.
                    </div>
                  )}
                  <ResolvedConfig composed={preview?.composed ?? {}} />
                  <div className="config-layers-row">
                    <ConfigItemInspector title="Model" item={selectedModel} />
                    <ConfigItemInspector title="Hardware" item={selectedHardware} />
                    <ConfigItemInspector title="Profile" item={selectedProfile} />
                  </div>
                  {profileDef && <ProfileDeltaPanel def={profileDef.definition} />}
                </ModuleCard>
                <ModuleCard title="Preset Templates" icon={<Database size={18} />} desc="Load a builtin preset's layer stack into the composer.">
                  <label className="search-box">
                    <Search size={15} />
                    <input value={filter} onChange={(event) => setFilter(event.target.value)} placeholder="Search preset/model/profile" />
                  </label>
                  <div className="preset-template-list">
                    {visiblePresets.map((preset) => (
                      <button
                        className={preset.model === modelId && preset.hardware === hardwareId && preset.profile === profileId ? "active" : ""}
                        key={preset.id}
                        onClick={() => applyPresetItem(preset)}
                      >
                        <strong>{preset.id}</strong>
                        <span>{preset.model}</span>
                        <small>{preset.profile ?? "no profile"} / {preset.status || "unannotated"}</small>
                      </button>
                    ))}
                  </div>
                </ModuleCard>
                <ModuleCard title="User Presets" icon={<PackageCheck size={18} />} desc="Operator-local presets written by Apply Plan.">
                  <UserPresetsPanel presets={userPresets} />
                </ModuleCard>
              </ModuleGrid>
            )
          },
          {
            id: "fit", label: "Fit", icon: <Gauge size={15} />,
            render: () => (
              <ModuleGrid>
                <ModuleCard title="Compatibility & Fit" icon={<Gauge size={18} />} desc={`${modelId} × ${hardwareId}`} wide>
                  <ModelFitCard
                    modelId={modelId}
                    hardwareOptions={(catalog?.hardware ?? []).map((item) => item.id)}
                    defaultHardware={hardwareId}
                  />
                </ModuleCard>
                <ModuleCard title="Compose Messages" icon={<FileText size={18} />} desc="Notes emitted while composing this configuration.">
                  {(preview?.messages ?? []).length
                    ? <CompactList rows={(preview?.messages ?? []).map((message, index) => [`Message ${index + 1}`, message])} />
                    : <p className="muted">Composition produced no messages.</p>}
                </ModuleCard>
              </ModuleGrid>
            )
          },
          {
            id: "tune", label: "Tune", icon: <SlidersHorizontal size={15} />,
            render: () => (
              <ModuleGrid>
                <ModuleCard title="Runtime Parameters" icon={<SlidersHorizontal size={18} />} desc={`Local draft — diff vs composed baseline (${configDiffs.length} changes).`} wide>
                  <div className="param-editor-controls">
                    <ParamFields baseDraft={baseDraft} draft={draft} set={setParam} />
                  </div>
                </ModuleCard>
              </ModuleGrid>
            )
          },
          {
            id: "yaml", label: "YAML", icon: <Code2 size={15} />,
            render: () => (
              <ModuleGrid>
                <ModuleCard title="Draft YAML" icon={<Code2 size={18} />} desc="Editable preview — persist via Apply Plan." wide>
                  <div className="yaml-editor">
                    <CodeEditorField value={configYaml} onChange={setYamlText} label="config YAML" />
                    <div className="config-actions">
                      <span className="config-actions-note">Editable preview — persist via Apply Plan</span>
                      <button className="ghost-button" onClick={() => setYamlText(null)}>Reset to generated</button>
                    </div>
                  </div>
                </ModuleCard>
              </ModuleGrid>
            )
          },
          {
            id: "cli", label: "CLI", icon: <SquareTerminal size={15} />,
            render: () => (
              <ModuleGrid>
                <ModuleCard title="CLI Mirror" icon={<SquareTerminal size={18} />} desc="Equivalent sndr CLI to reproduce this composition." wide>
                  <CodeBlock lines={configCli} />
                </ModuleCard>
              </ModuleGrid>
            )
          },
          {
            id: "diff", label: "Diff", icon: <GitBranch size={15} />,
            render: () => (
              <ModuleGrid>
                <ModuleCard title="Draft Diff" icon={<GitBranch size={18} />} desc="Draft vs composed baseline + planned file diff." wide>
                  <div className="diff-panel tall">
                    <strong>Draft vs composed baseline ({configDiffs.length})</strong>
                    {configDiffs.length ? configDiffs.map((line, index) => <span key={index}>{line}</span>) : <span>No parameter changes</span>}
                    {configPlan && configPlan.diff_lines.length > 0 && (
                      <>
                        <strong className="diff-section">Planned file diff</strong>
                        {configPlan.diff_lines.map((line, index) => (
                          <span key={`plan-${index}`} className={line.startsWith("+") ? "add" : line.startsWith("-") ? "del" : ""}>{line}</span>
                        ))}
                      </>
                    )}
                  </div>
                </ModuleCard>
              </ModuleGrid>
            )
          },
          {
            id: "artifacts", label: "Artifacts", icon: <Layers3 size={15} />,
            render: () => (
              <ModuleGrid>
                <ModuleCard title="Generated Artifacts" icon={<Layers3 size={18} />} desc="Compose / docker run / systemd / env rendered from this draft." wide>
                  <div className="artifact-mode">
                    <div className="artifact-tabs">
                      {(["compose", "run", "systemd", "env"] as const).map((kind) => (
                        <button key={kind} className={artifactKind === kind ? "active" : ""} onClick={() => setArtifactKind(kind)}>
                          {kind === "run" ? "docker run" : kind}
                        </button>
                      ))}
                    </div>
                    <CodeBlock lines={configArtifacts[artifactKind]} />
                  </div>
                </ModuleCard>
              </ModuleGrid>
            )
          },
          {
            id: "compare", label: "Compare", icon: <GitBranch size={15} />,
            render: () => (
              <ModuleGrid>
                <ModuleCard title="Compare presets" icon={<GitBranch size={18} />} desc="Diff two presets' composed runtime configuration — context, concurrency, KV, patches and more." wide>
                  <ConfigComparePanel presets={catalog?.presets ?? []} />
                </ModuleCard>
              </ModuleGrid>
            )
          },
          {
            id: "plan", label: "Plan & Apply", icon: <PackageCheck size={15} />,
            render: () => (
              <ModuleGrid>
                <ModuleCard title="Plan & Apply" icon={<PackageCheck size={18} />} desc="Write-safe plan, then Apply Plan to persist an operator-local preset." wide>
                  {planError && <div className="config-plan-error"><AlertCircle size={15} /><span>{planError}</span></div>}
                  {applyError && <div className="config-plan-error"><AlertCircle size={15} /><span>{applyError}</span></div>}
                  {configPlan ? <ConfigPlanPanel plan={configPlan} /> : <p className="muted">Run “Plan” to preview the write (diff + target path), then “Apply Plan”.</p>}
                  {applyResult && <ConfigApplyPanel result={applyResult} />}
                </ModuleCard>
              </ModuleGrid>
            )
          }
        ]}
      />
    </section>
  );
}

// Side-by-side comparison of two presets' composed runtime configs. Uses the
// preset explain endpoint (composed dict) and highlights every differing field
// — the "show me prod vs staging" view operators ask for.
// ConfigComparePanel / ConfigPlanPanel / ConfigApplyPanel extracted to ./sections/config.

// UserPresetsPanel + ProfileDeltaPanel extracted to ./sections/presets.

function ConfigSelect({
  label,
  icon,
  value,
  items,
  allowEmpty = false,
  onChange
}: {
  label: string;
  icon: ReactNode;
  value: string;
  items: V2ConfigItem[];
  allowEmpty?: boolean;
  onChange: (value: string) => void;
}) {
  const selected = items.find((item) => item.id === value) ?? null;
  return (
    <label className="config-select-card">
      <span>{icon} {label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {allowEmpty && <option value="">No profile</option>}
        {items.map((item) => (
          <option key={item.id} value={item.id}>{item.id}</option>
        ))}
      </select>
      <small>{selected?.title ?? "No layer selected"}</small>
    </label>
  );
}

const _INSPECTOR_ROWS: Record<string, Array<[string, string]>> = {
  Model: [
    ["served_model_name", "Served"], ["quantization", "Quant"], ["dtype", "Dtype"],
    ["attention_arch", "Attention"], ["kv_cache_dtype", "KV cache"],
    ["min_gpu_count", "Min GPUs"], ["min_total_vram_mib", "Min VRAM"], ["patch_count", "Patches"]
  ],
  Hardware: [
    ["n_gpus", "GPUs"], ["min_vram_per_gpu_mib", "VRAM/GPU"], ["max_model_len", "Max ctx"],
    ["max_num_seqs", "Max seqs"], ["gpu_memory_utilization", "GPU util"], ["runtime_default", "Runtime"]
  ],
  Profile: [
    ["max_model_len", "Max ctx"], ["max_num_seqs", "Max seqs"],
    ["gpu_memory_utilization", "GPU util"], ["enable_delta", "Patch +"], ["disable_delta", "Patch −"]
  ]
};

function ConfigItemInspector({ title, item }: { title: string; item: V2ConfigItem | null }) {
  const icon = title === "Hardware" ? <Cpu size={16} /> : title === "Profile" ? <SlidersHorizontal size={16} /> : <Box size={16} />;
  if (!item) {
    return (
      <div className="config-item-inspector empty">
        <span className="catalog-card-ico">{icon}</span>
        <div><strong>{title}</strong><small>No selection</small></div>
      </div>
    );
  }
  const f = item.fields ?? {};
  const fmt = (key: string, value: any): string => {
    if (value === null || value === undefined || value === "") return "-";
    if (key.includes("vram")) return formatVram(Number(value));
    if (key === "max_model_len") return formatTokens(Number(value));
    if (Array.isArray(value)) return value.length ? value.map(String).join(", ") : "-";
    return String(value);
  };
  const rows = (_INSPECTOR_ROWS[title] ?? [])
    .filter(([key]) => f[key] !== undefined && f[key] !== null && f[key] !== "")
    .map(([key, label]) => [label, fmt(key, f[key])] as [string, string]);
  return (
    <div className="config-item-inspector card">
      <div className="config-inspector-head">
        <span className="catalog-card-ico">{icon}</span>
        <div>
          <strong>{title}: {item.id}</strong>
          {item.title && item.title !== item.id && <small>{item.title}</small>}
        </div>
      </div>
      {itemBadges(item).length > 0 && (
        <span className="catalog-badges">
          {itemBadges(item).map((badge, index) => (
            <span key={index} className={`catalog-badge tone-${badge.tone ?? "neutral"}`}>{badge.label}</span>
          ))}
        </span>
      )}
      {rows.length > 0 && <InfoRows rows={rows} />}
      <small className="config-inspector-src">{item.source}</small>
    </div>
  );
}

// ModuleGrid extracted to ./components/layout.

// Short explanatory banner at the top of a tab — what it does + when to use it.
function TabIntro({ icon, title, text }: { icon: ReactNode; title: string; text: string }) {
  return (
    <div className="tab-intro">
      <span className="tab-intro-icon">{icon}</span>
      <div className="tab-intro-body">
        <strong>{title}</strong>
        <span>{text}</span>
      </div>
    </div>
  );
}

// Overview hero KPI tile — bold headline numbers, optionally click-through to a section.
// OvKpi extracted to ./components/charts.

// TabbedSection extracted to ./components/tabbed-section.

// ModuleCard extracted to ./components/layout.

// InfoRows extracted to ./components/primitives.

// KpiGrid extracted to ./components/primitives.

// RuntimeEnvelopePanel + PresetPolicyGraph extracted to ./sections/preset-insight.

function ConfigDraftEditor({
  selectedPreset,
  composed,
  runtimeTarget,
  patchPolicy
}: {
  selectedPreset: string;
  composed: Record<string, unknown>;
  runtimeTarget: string;
  patchPolicy: string;
}) {
  const baseDraft = useMemo(
    () => buildRuntimeDraft(composed, runtimeTarget, patchPolicy),
    [composed, runtimeTarget, patchPolicy]
  );
  const [draft, setDraft] = useState<RuntimeConfigDraft>(baseDraft);

  useEffect(() => {
    setDraft(baseDraft);
  }, [baseDraft, selectedPreset]);

  const diffs = runtimeDraftDiff(baseDraft, draft);
  const previewLines = buildDraftYaml(selectedPreset, draft);
  const set = (patch: Partial<RuntimeConfigDraft>) => setDraft((current) => ({ ...current, ...patch }));

  return (
    <div className="param-editor">
      <div className="param-editor-controls">
        <ParamFields baseDraft={baseDraft} draft={draft} set={set} />
      </div>

      <div className="param-editor-preview">
        <div className="diff-panel">
          <strong>Pending changes ({diffs.length})</strong>
          {diffs.length ? diffs.map((diff, index) => <span key={index}>{diff}</span>) : <span>No changes vs composed baseline</span>}
        </div>
        <CodeBlock lines={previewLines} />
        <div className="config-actions">
          <span className="config-actions-note">Persist via Configs → Apply Plan</span>
          <button className="ghost-button" onClick={() => setDraft(baseDraft)}>Reset</button>
          <button className="ghost-button" onClick={() => void navigator.clipboard?.writeText(previewLines.join("\n"))}>
            <Copy size={14} /> Copy YAML
          </button>
        </div>
      </div>
    </div>
  );
}

function Collapsible({
  title,
  subtitle,
  right,
  defaultOpen = true,
  children
}: {
  title: string;
  subtitle?: string;
  right?: number;
  defaultOpen?: boolean;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className={`collapsible ${open ? "open" : ""}`}>
      <button className="collapsible-head" onClick={() => setOpen((value) => !value)} aria-expanded={open}>
        <ChevronRight className="coll-caret" size={15} />
        <strong>{title}</strong>
        {subtitle && <span>{subtitle}</span>}
        {right ? <em className="coll-badge">{right} changed</em> : null}
      </button>
      {open && <div className="collapsible-body">{children}</div>}
    </section>
  );
}

// BoolField + SelectField extracted to ./components/form-fields.
function DraftControl({
  label,
  value,
  min,
  max,
  step,
  suffix = "",
  onChange
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  suffix?: string;
  onChange: (value: number) => void;
}) {
  return (
    <label className="config-field">
      <span>{label}</span>
      <div className="range-row">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(event) => onChange(Number(event.target.value))}
        />
        <input
          type="number"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(event) => onChange(Number(event.target.value))}
        />
        {suffix && <em>{suffix}</em>}
      </div>
    </label>
  );
}

// CHART_PALETTE / DonutSegment / SegmentBar / PercentBar / segmentsFromCounts
// / BarList extracted to ./components/charts.

function CapabilityTable({ rows }: { rows: ProductCapability[] }) {
  return (
    <table className="module-table">
      <thead>
        <tr>
          <th>Capability</th>
          <th>Status</th>
          <th>Required</th>
          <th>Detail</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.id}>
            <td>
              <strong>{row.title}</strong>
              <small>{row.id}</small>
            </td>
            <td><StatusBadge status={row.status} /></td>
            <td>{row.required_tools.length ? row.required_tools.join(", ") : "built-in"}</td>
            <td>{row.detail}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

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

function QueueJobButton({ label, run, onMonitor }: { label: string; run: () => Promise<Job>; onMonitor?: (id: string) => void }) {
  const [job, setJob] = useState<Job | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  return (
    <div className="queue-job">
      <button
        className="primary-action"
        disabled={busy}
        onClick={async () => {
          setBusy(true);
          setError(null);
          try {
            const result = await run();
            setJob(result);
            // Consistent across the GUI: open the live job monitor when available.
            onMonitor?.(result.job_id);
          } catch (err) {
            setError(err instanceof Error ? err.message : String(err));
          } finally {
            setBusy(false);
          }
        }}
      >
        <Play size={15} /> {busy ? "Queuing…" : label}
      </button>
      {error && <div className="config-plan-error"><AlertCircle size={14} /><span>{error}</span></div>}
      {job && !onMonitor && <JobResultBlock job={job} />}
      {job && onMonitor && (
        <button className="ghost-button queue-job-reopen" onClick={() => onMonitor(job.job_id)}>
          <Activity size={14} /> View job {job.job_id}
        </button>
      )}
    </div>
  );
}

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
function HostsSection({
  hostProfiles,
  environment,
  overview,
  runtimeTargets,
  apiBase,
  runtimeMode,
  onHostsRefresh,
  onChatWithHost,
  onAddServer,
  focusHostId,
  onFocusConsumed,
  onSetupNode,
  onContainers,
  onHardware,
  applyEnabled
}: {
  hostProfiles: HostProfile[];
  environment: EnvironmentReport | null;
  overview: ProductOverview | null;
  runtimeTargets: ProductCapability[];
  apiBase: string;
  runtimeMode: RuntimeMode;
  onHostsRefresh: () => void;
  onChatWithHost: (profile: HostProfile) => void;
  onAddServer: (profile: HostProfile) => Promise<boolean>;
  focusHostId: string | null;
  onFocusConsumed: () => void;
  onSetupNode: (id: string) => void;
  onContainers: (id: string) => void;
  onHardware: (id: string) => void;
  applyEnabled?: boolean;
}) {
  const [inventory, setInventory] = useState<HostInventory | null>(null);
  const [modal, setModal] = useState<{ profile: HostProfile | null } | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<{ id: string; label: string } | null>(null);
  const askDelete = (id: string) => setConfirmDelete({ id, label: hostProfiles.find((h) => h.id === id)?.label ?? id });
  const [terminalHost, setTerminalHost] = useState<HostProfile | null>(null);
  const [reliability, setReliability] = useState<ReliabilitySnapshot>({});
  const [fleetById, setFleetById] = useState<Record<string, FleetHost>>({});
  useEffect(() => {
    let cancelled = false;
    api.hostInventory().then((data) => { if (!cancelled) setInventory(data); }).catch(() => {});
    const loadRel = () => api.hostsReliability().then((r) => { if (!cancelled) setReliability(r); }).catch(() => {});
    // Live fleet sweep (GPU util / containers / patches) to enrich each card.
    const loadFleet = () => api.fleetOverview().then((r) => {
      if (!cancelled) setFleetById(Object.fromEntries(r.hosts.map((h) => [h.id, h])));
    }).catch(() => {});
    void loadRel();
    void loadFleet();
    const tf = window.setInterval(() => { if (!document.hidden) void loadFleet(); }, 60000);
    const t = window.setInterval(() => { if (!document.hidden) void loadRel(); }, 8000);
    return () => { cancelled = true; window.clearInterval(t); window.clearInterval(tf); };
  }, []);
  async function remove(id: string) {
    try { await api.hostDelete(id); onHostsRefresh(); toast(`Host removed: ${id}`, "success"); } catch { toast("Failed to remove host", "error"); }
  }
  return (
    <>
      <TabbedSection
        id="hosts"
        tabs={[
          {
            id: "fleet",
            label: "Fleet",
            icon: <Server size={15} />,
            render: () => (
              <ModuleGrid>
                <ModuleCard title="Host fleet" icon={<Server size={18} />} desc="The daemon host plus every saved target, with a live per-host engine probe." wide>
                  <div className="fleet-toolbar">
                    <span className="muted">{hostProfiles.length} saved host{hostProfiles.length === 1 ? "" : "s"}</span>
                    <button className="primary-action" onClick={() => setModal({ profile: null })}><Server size={15} /> Add host</button>
                  </div>
                  {(() => {
                    const fl = Object.values(fleetById);
                    const online = fl.filter((h) => h.engines.some((e) => e.reachable)).length;
                    const gpus = fl.reduce((n, h) => n + (h.gpu_count || 0), 0);
                    const livePatches = fl.reduce((n, h) => n + (h.active_patches || 0), 0);
                    return (
                      <div className="fleet-kpis">
                        <span className="fleet-kpi"><strong>{hostProfiles.length}</strong> servers</span>
                        <span className="fleet-kpi ok"><strong>{online}</strong> online</span>
                        <span className="fleet-kpi"><strong>{gpus}</strong> GPUs</span>
                        <span className="fleet-kpi"><strong>{livePatches}</strong> live patches</span>
                      </div>
                    );
                  })()}
                  <div className="fleet-grid">
                    <ThisHostCard inventory={inventory} environment={environment} apiBase={apiBase} />
                    {hostProfiles.map((profile) => (
                      <FleetHostCard key={profile.id} profile={profile} onEdit={(p) => setModal({ profile: p })} onDelete={askDelete} onChat={onChatWithHost} onAddServer={onAddServer} onRefresh={onHostsRefresh} onTerminal={setTerminalHost} focused={focusHostId === profile.id} onFocusConsumed={onFocusConsumed} onSetupNode={onSetupNode} onContainers={onContainers} onHardware={onHardware} reliability={reliability[profile.id] ?? null} fleet={fleetById[profile.id] ?? null} applyEnabled={applyEnabled} />
                    ))}
                  </div>
                  {hostProfiles.length === 0 && <p className="muted">No remote hosts yet — add your GPU box to probe its engine from here.</p>}
                </ModuleCard>
              </ModuleGrid>
            )
          },
          {
            id: "inventory",
            label: "Inventory",
            icon: <Cpu size={15} />,
            render: () => (
              <ModuleGrid className="stretch-row">
                <ModuleCard title="Daemon host inventory" icon={<Cpu size={18} />} desc="Live OS / Python / Docker / GPU / vLLM snapshot of the host serving this UI." wide>
                  <HostInventoryPanel inventory={inventory} environment={environment} />
                </ModuleCard>
                <ModuleCard title="Dependency stack" icon={<PackageCheck size={18} />} desc="Python libraries and runtime tools detected on the daemon host.">
                  <DependencyStackPanel env={environment} />
                </ModuleCard>
                <ModuleCard title="Project & catalog" icon={<Database size={18} />} desc="Catalog coverage and project parameters this daemon serves.">
                  <ProjectCatalogPanel overview={overview} environment={environment} />
                </ModuleCard>
                <ModuleCard title="Runtime target matrix" icon={<Network size={18} />} desc="Which runtime backends can be rendered or controlled on this host." wide>
                  <CapabilityTable rows={runtimeTargets} />
                </ModuleCard>
              </ModuleGrid>
            )
          },
          {
            id: "profiles",
            label: "Profiles",
            icon: <SlidersHorizontal size={15} />,
            render: () => (
              <ModuleGrid>
                <ModuleCard title="Saved host profiles" icon={<Server size={18} />} desc="Operator-local registry of hosts. Edit role, hardware, ports and tags — execution stays manual." wide>
                  <div className="fleet-toolbar">
                    <span className="muted">{hostProfiles.length} profile{hostProfiles.length === 1 ? "" : "s"}</span>
                    <button className="primary-action" onClick={() => setModal({ profile: null })}><Server size={15} /> Add host</button>
                  </div>
                  <HostProfileTable profiles={hostProfiles} onEdit={(p) => setModal({ profile: p })} onDelete={askDelete} />
                </ModuleCard>
              </ModuleGrid>
            )
          },
          {
            id: "access",
            label: "Access",
            icon: <Network size={15} />,
            render: () => (
              <ModuleGrid>
                <ModuleCard title="Current connection" icon={<Network size={18} />} desc="How this UI reaches the Product API.">
                  <InfoRows rows={[
                    ["API base", apiBase],
                    ["Mode", runtimeMode === "remote" ? "Remote (SSH tunnel)" : "Local server"],
                    ["Engine", `${environment?.engine_name ?? "vLLM"} ${environment?.engine_version ?? (environment?.engine_installed ? "" : "(not installed)")}`.trim()],
                    ["Saved profiles", String(hostProfiles.length)]
                  ]} />
                </ModuleCard>
                <ModuleCard title="Remote access (SSH tunnel)" icon={<Server size={18} />} desc="Keep the daemon on 127.0.0.1; forward a loopback port from your laptop." wide>
                  <CodeBlock lines={[
                    "# on your laptop — forward the daemon's loopback port",
                    "ssh -L 8765:127.0.0.1:8765 user@gpu-host",
                    "# then open the UI at http://127.0.0.1:8765"
                  ]} />
                </ModuleCard>
              </ModuleGrid>
            )
          }
        ]}
      />
      {modal && <HostFormModal initial={modal.profile} onClose={() => setModal(null)} onSaved={onHostsRefresh} />}
      {confirmDelete && (
        <ConfirmDialog
          title="Remove host profile?"
          message={<>This deletes the saved profile <strong>{confirmDelete.label}</strong> (connection details, SSH target, stored key reference). The remote host is not touched, but the profile must be re-added to manage it from here.</>}
          confirmLabel="Remove host"
          danger
          onConfirm={() => { const id = confirmDelete.id; setConfirmDelete(null); void remove(id); }}
          onCancel={() => setConfirmDelete(null)}
        />
      )}
      {terminalHost && <Suspense fallback={null}><TerminalModal host={terminalHost} onClose={() => setTerminalHost(null)} /></Suspense>}
    </>
  );
}

// Dense table view of saved profiles for the Profiles tab.
function HostProfileTable({
  profiles,
  onEdit,
  onDelete
}: {
  profiles: HostProfile[];
  onEdit: (profile: HostProfile) => void;
  onDelete: (id: string) => void;
}) {
  if (profiles.length === 0) return <p className="muted">No saved profiles yet — add one above.</p>;
  return (
    <table className="module-table host-table">
      <thead>
        <tr><th>Label</th><th>Host</th><th>Role</th><th>Hardware</th><th>Ports</th><th>Tags</th><th></th></tr>
      </thead>
      <tbody>
        {profiles.map((profile) => (
          <tr key={profile.id}>
            <td><strong>{profile.label}</strong></td>
            <td>{profile.host}<br /><small className="muted">{profile.transport}{profile.ssh_target ? ` · ${profile.ssh_target}` : ""}</small></td>
            <td>{profile.role ? <span className={`fleet-role tone-${roleTone(profile.role)}`}>{profile.role}</span> : <span className="muted">—</span>}</td>
            <td>{profile.hardware || "—"}{profile.gpus ? ` · ${profile.gpus} GPU` : ""}</td>
            <td><small>gui {profile.port}<br />engine {profile.engine_port}</small></td>
            <td>{profile.tags.length ? <div className="fleet-tags">{profile.tags.map((t) => <span key={t} className="fleet-tag">{t}</span>)}</div> : <span className="muted">—</span>}</td>
            <td className="host-table-actions">
              <CopyButton value={tunnelCommand(profile)} label="tunnel command" />
              <button className="icon-only" onClick={() => onEdit(profile)} aria-label={`Edit ${profile.label}`}><Pencil size={14} /></button>
              <button className="icon-only danger" onClick={() => onDelete(profile.id)} aria-label={`Delete ${profile.label}`}><Trash2 size={14} /></button>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// CodeBlock + CopyButton extracted to ./components/code-block.
// Editable text field (YAML / config) with copy + a fullscreen-edit expand.
function CodeEditorField({ value, onChange, label }: { value: string; onChange: (next: string) => void; label?: string }) {
  const [expanded, setExpanded] = useState(false);
  const dialogRef = useRef<HTMLElement>(null);
  useDialogFocus(dialogRef, expanded);
  useEffect(() => {
    if (!expanded) return;
    const onKey = (event: KeyboardEvent) => { if (event.key === "Escape") setExpanded(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expanded]);
  return (
    <div className="code-wrap code-editor-field">
      <div className="code-actions">
        <button className="icon-only" title="Expand editor" aria-label="Expand editor to fullscreen" onClick={() => setExpanded(true)}><Maximize2 size={13} /></button>
        <CopyButton value={value} label={label ?? "text"} />
      </div>
      <textarea className="yaml-area" value={value} spellCheck={false} onChange={(event) => onChange(event.target.value)} />
      {expanded && (
        <div className="dialog-backdrop" role="presentation" onClick={closeOnBackdrop(() => setExpanded(false))}>
          <section ref={dialogRef} className="code-expand" role="dialog" aria-modal="true">
            <header className="code-expand-head">
              <Code2 size={15} />
              <strong>{label ?? "Editor"}</strong>
              <span className="muted">{value.split("\n").length} lines</span>
              <CopyButton value={value} label={label ?? "text"} />
              <button className="icon-only" onClick={() => setExpanded(false)} aria-label="Close"><X size={16} /></button>
            </header>
            <textarea className="yaml-area code-expand-editor" value={value} spellCheck={false} autoFocus onChange={(event) => onChange(event.target.value)} />
          </section>
        </div>
      )}
    </div>
  );
}

function CodeTabs({ tabs }: { tabs: Array<{ id: string; label: string; lines: string[] }> }) {
  const [active, setActive] = useState(tabs[0]?.id ?? "");
  const current = tabs.find((tab) => tab.id === active) ?? tabs[0];
  return (
    <div className="code-tabs">
      <div className="code-tabs-bar">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            type="button"
            className={tab.id === current?.id ? "active" : ""}
            onClick={() => setActive(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      {current && <CodeBlock lines={current.lines} />}
    </div>
  );
}

// Enterprise recommend tab: surface the catalog's recommendation engine inside
// Presets. Pick a workload + rig + concurrency and get a ranked, scored list of
// presets — click to select and inspect. Connects browse → recommend → select.
// PresetRecommendPanel extracted to ./sections/preset-recommend.

// What this preset overrides relative to its declared fallback target.
function PresetFallbackDiff({ explain }: { explain: PresetExplainResult | null }) {
  const fallback = (explain?.fallback_diff ?? null) as { fallback_preset?: string; diffs?: string[] } | null;
  if (!fallback || !fallback.fallback_preset) {
    return <p className="muted">No fallback declared for this preset.</p>;
  }
  const diffs = Array.isArray(fallback.diffs) ? fallback.diffs : [];
  return (
    <div className="fallback-diff">
      <div className="fallback-diff-head"><GitBranch size={14} /> overrides vs <strong>{fallback.fallback_preset}</strong></div>
      {diffs.length > 0 ? (
        <ul className="fallback-diff-list">
          {diffs.map((line, index) => <li key={index}>{line}</li>)}
        </ul>
      ) : <p className="muted">Matches its fallback exactly — no overrides.</p>}
    </div>
  );
}

// Enterprise "selected preset" detail view: a hero with identity + quick
// actions, a row of runtime metric tiles, then equal-height detail cards
// (identity, evidence, workload rules, fallback) and the editable draft.
function PresetSelectedView({
  selectedPreset,
  record,
  card,
  composed,
  explain,
  runtimeTargets,
  runtimeTarget,
  patchPolicy,
  onEdit,
  onLaunch,
  onConfigs
}: {
  selectedPreset: string;
  record: PresetRecord | null;
  card: Record<string, unknown>;
  composed: Record<string, unknown>;
  explain: PresetExplainResult | null;
  runtimeTargets: ProductCapability[];
  runtimeTarget: string;
  patchPolicy: string;
  onEdit: () => void;
  onLaunch: () => void;
  onConfigs: () => void;
}) {
  const cm = { ...composed, ...(explain?.composed ?? {}) } as Record<string, unknown>;
  // asText only accepts strings; composed numerics (max_num_seqs, spec K, patch
  // count) need a numeric-aware formatter or they fall back to a dash.
  const val = (value: unknown) =>
    typeof value === "number" ? String(value) : typeof value === "string" && value.trim() ? value : "—";
  const status = asText(card.status, record?.has_card ? "available" : "missing");
  const util = asNumber(cm.gpu_memory_utilization);
  const specMethod = asText(cm.spec_decode_method, "");
  const allow = asStringArray(card.workload_allow);
  const deny = asStringArray(card.workload_deny);
  const metrics: Array<{ label: string; value: string }> = [
    { label: "Max context", value: formatTokens(asNumber(cm.max_model_len)) },
    { label: "Max sequences", value: val(cm.max_num_seqs) },
    { label: "GPU mem util", value: util > 0 ? `${Math.round(util * 100)}%` : "—" },
    { label: "KV cache", value: val(cm.kv_cache_dtype) },
    { label: "Spec decode", value: specMethod ? `${specMethod} · K=${val(cm.spec_decode_K)}` : "—" },
    { label: "Enabled patches", value: val(cm.enabled_patches_count) }
  ];
  return (
    <div className="preset-selected">
      <header className="preset-hero">
        <div className="preset-hero-main">
          <div className="preset-hero-id">
            <span className="preset-hero-icon"><Database size={20} /></span>
            <div className="preset-hero-text">
              <h2>{asText(card.title, "Unannotated preset")}</h2>
              <code>{selectedPreset}</code>
            </div>
            <StatusBadge status={status} />
          </div>
          <div className="preset-hero-chips">
            <span className="hero-chip"><Box size={12} />{record?.model ?? "—"}</span>
            <span className="hero-chip"><Server size={12} />{record?.hardware ?? "—"}</span>
            {record?.profile && <span className="hero-chip"><SlidersHorizontal size={12} />{record.profile}</span>}
            <span className="hero-chip"><Network size={12} />{targetTitle(runtimeTargets, runtimeTarget)}</span>
            <span className="hero-chip"><Wrench size={12} />{patchPolicy} policy</span>
          </div>
        </div>
        <div className="preset-hero-actions">
          <button className="primary-action" onClick={onLaunch}><Rocket size={15} /> Launch</button>
          <button className="ghost-button" onClick={onEdit}><Wrench size={15} /> Edit</button>
          <button className="ghost-button" onClick={onConfigs}><SlidersHorizontal size={15} /> Open in Configs</button>
        </div>
      </header>

      <div className="preset-metrics">
        {metrics.map((metric) => (
          <div className="preset-metric" key={metric.label}>
            <strong>{metric.value}</strong>
            <span>{metric.label}</span>
          </div>
        ))}
      </div>

      <ModuleGrid className="stretch-row">
        <ModuleCard title="Identity" icon={<FileText size={18} />} desc="What composes this preset.">
          <InfoRows
            rows={[
              ["Model", record?.model ?? asText(cm.model, "—")],
              ["Hardware", record?.hardware ?? asText(cm.hardware, "—")],
              ["Profile", record?.profile ?? asText(cm.profile, "—")],
              ["Mode", asText(card.mode, "—")],
              ["Runtime target", targetTitle(runtimeTargets, runtimeTarget)],
              ["Fallback", asText(card.fallback_preset, "none")],
              ["Evidence", asText(card.evidence_visibility, "—")]
            ]}
          />
        </ModuleCard>
        <ModuleCard title="Workload Rules" icon={<SlidersHorizontal size={18} />} desc="Where this preset is allowed or denied.">
          <div className="wl-rules">
            {allow.length > 0 && (
              <div className="wl-group">
                <span className="wl-label ok">Allow</span>
                <div className="fleet-caps">{allow.map((item) => <span className="cap-chip on" key={item}>{item.replace(/_/g, " ")}</span>)}</div>
              </div>
            )}
            {deny.length > 0 && (
              <div className="wl-group">
                <span className="wl-label danger">Deny</span>
                <div className="fleet-caps">{deny.map((item) => <span className="cap-chip off" key={item}>{item.replace(/_/g, " ")}</span>)}</div>
              </div>
            )}
            {allow.length === 0 && deny.length === 0 && <p className="muted">No workload restrictions — usable for any workload.</p>}
          </div>
        </ModuleCard>
        <ModuleCard title="Evidence References" icon={<ShieldCheck size={18} />} desc="Proof refs attached to this preset.">
          <EvidenceRows card={card} />
        </ModuleCard>
        <ModuleCard title="Overrides vs Fallback" icon={<GitBranch size={18} />} desc="What this preset changes relative to its fallback.">
          <PresetFallbackDiff explain={explain} />
        </ModuleCard>
      </ModuleGrid>

      <ModuleGrid>
        <ModuleCard title="Editable Runtime Draft" icon={<Code2 size={18} />} desc="Tune a local copy and render artifacts — operator-safe." wide>
          <ConfigDraftEditor
            selectedPreset={selectedPreset}
            composed={composed}
            runtimeTarget={runtimeTarget}
            patchPolicy={patchPolicy}
          />
        </ModuleCard>
      </ModuleGrid>
    </div>
  );
}

// OperationsConsole (+ OP_GROUP_ICON) extracted to ./sections/operations.
// OperationsConsole moved to ./sections/operations (see import).

function PresetSummaryStrip({ presets, selectedPreset }: { presets: PresetRecord[]; selectedPreset: string }) {
  const annotated = presets.filter((preset) => preset.has_card).length;
  const missing = presets.length - annotated;
  const models = new Set(presets.map((preset) => preset.model)).size;
  const hardware = new Set(presets.map((preset) => preset.hardware)).size;
  const tiles: Array<{ label: string; value: string; tone?: string }> = [
    { label: "Presets", value: String(presets.length) },
    { label: "Annotated", value: String(annotated), tone: "ok" },
    { label: "Missing card", value: String(missing), tone: missing ? "warn" : "ok" },
    { label: "Models", value: String(models) },
    { label: "Hardware", value: String(hardware) },
    { label: "Selected", value: selectedPreset || "—" }
  ];
  return (
    <div className="preset-summary-strip">
      {tiles.map((tile) => (
        <div className={`preset-stat ${tile.tone ?? ""}`} key={tile.label}>
          <span className="preset-stat-value">{tile.value}</span>
          <span className="preset-stat-label">{tile.label}</span>
        </div>
      ))}
    </div>
  );
}

// PresetQuickPanel extracted to ./sections/preset-quick.

// Benchmark-baseline chip for the preset catalog: surfaces the measured
// reference metric (primary_metric) at the list level, so bench-proven presets
// are distinguishable from pending ones at a glance. value 0 / missing = pending.
// PresetBaselineCell + PresetCatalogTable extracted to ./sections/preset-catalog.

// PatchSummaryPanel + PatchLifecycleGraph + PatchRegistryInsight + PatchModelSupport extracted to ./sections/patch-overview.

// PatchInventoryControl + PatchFamilyGroup extracted to ./sections/patch-inventory.

// formatAppliesTo extracted to ./lib/format.

// PatchExplainPanel (+ lifecycle/default explanation helpers) extracted to ./sections/patch-explain.

// CaveatsPanel / ConfigKeysPanel / TracesPanel extracted to ./sections/diagnostics.
// DoctorStat extracted to ./components/primitives.
// SEVERITY_META + DoctorSummary / DoctorFindings (+ DoctorCategory / SeverityDot /
// DoctorFindingRow) extracted to ./sections/doctor.

// WizardStatus + SetupWizard extracted to ./sections/setup-wizard.

// fmtParam extracted to ./lib/format.
// DeploymentConsole (+ DEPLOY_TARGET_ICONS + downloadText) extracted to ./sections/deployment.



// EnvironmentPanel extracted to ./sections/environment.

// SERVICE_RUNTIME_TARGETS + ServiceLifecyclePlanner extracted to ./sections/services.

// DoctorCoveragePanel + AdminSurfaceMatrix extracted to ./sections/patch-doctor.
// (BundlesPanel + UpstreamDiffPanel previously extracted to ./sections/registry;
//  ProofStatusPanel + drill-down to ./sections/proof.)

// BenchmarkBaselinePanel + EvidenceRows extracted to ./sections/bench.

// EndpointRows extracted to ./sections/rail-cards.

// Reusable confirmation dialog for destructive/irreversible actions. Focus is
// trapped, Cancel is the autofocused default, Esc/backdrop cancel, and the
// confirm button can be styled as danger. Keeps destructive paths deliberate.
// ConfirmDialog + InfoDialog extracted to ./components/dialogs.

// toast + ToastHost (+ ToastTone) extracted to ./components/toast.

// ── Audit log (surfaces the daemon's recorded events: auth, jobs, system) ──
// AuditLogPanel extracted to ./sections/audit-log.

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

function PlanChip({ label, value }: { label: string; value: string }) {
  return (
    <div className="plan-chip">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ArtifactPreview({
  artifacts,
  activeTab,
  setActiveTab
}: {
  artifacts: LaunchPlanArtifact[];
  activeTab: ArtifactTab;
  setActiveTab: (tab: ArtifactTab) => void;
}) {
  const artifactByKind = new Map(
    artifacts.map((artifact) => [artifact.kind, artifact])
  );
  const activeArtifact = artifactByKind.get(activeTab);
  const tabs: Array<{ id: ArtifactTab; label: string }> = [
    { id: "compose", label: "Compose" },
    { id: "systemd", label: "systemd Unit" },
    { id: "commands", label: "CLI Commands" },
    { id: "env", label: "Environment Diff" }
  ];
  const fallback = [
    "# Waiting for launch plan Product API",
    "# Backend endpoint: /api/v1/launch/plan",
    "# Generated artifacts are intentionally not composed in React."
  ].join("\n");
  const title = activeArtifact?.title ?? "Product API Artifact";
  const content = activeArtifact?.content ?? fallback;

  return (
    <section className="artifact-preview">
      <div className="artifact-tabs">
        {tabs.map((tab) => (
          <button
            className={activeTab === tab.id ? "active" : ""}
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <strong className="artifact-title">{title}</strong>
      <CodeBlock lines={content.split("\n")} title={title} />
    </section>
  );
}

function KeyValue({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="key-value">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

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

function targetTitle(targets: ProductCapability[], id: string) {
  return targets.find((target) => target.id === id)?.title ?? id;
}

function runtimeHost(mode: RuntimeMode) {
  return mode === "remote" ? "gpu-build-01" : "127.0.0.1";
}

// countRecord extracted to ./lib/coerce.

// buildRuntimeDraft + buildDraftYaml extracted to ./lib/runtime-draft.

type ConfigContext = {
  presetId: string;
  modelId: string;
  hardwareId: string;
  profileId: string;
  runtime: string;
  draft: RuntimeConfigDraft;
};

function buildFullConfigYaml(ctx: ConfigContext): string[] {
  const { presetId, modelId, hardwareId, profileId, runtime, draft } = ctx;
  return [
    "schema_version: 2",
    "kind: preset",
    `id: ${presetId}`,
    `model: ${modelId}`,
    `hardware: ${hardwareId}`,
    ...(profileId ? [`profile: ${profileId}`] : []),
    `runtime: ${runtime}`,
    "card:",
    "  title: GUI draft preset",
    "  status: experimental",
    "  audience: operator",
    `patch_policy: ${draft.patch_policy}`,
    "sizing_override:",
    `  max_model_len: ${draft.max_model_len}`,
    `  max_num_seqs: ${draft.max_num_seqs}`,
    `  max_num_batched_tokens: ${draft.max_num_batched_tokens}`,
    `  gpu_memory_utilization: ${draft.gpu_memory_utilization.toFixed(2)}`,
    `  enable_chunked_prefill: ${draft.enable_chunked_prefill}`,
    `  enforce_eager: ${draft.enforce_eager}`,
    `  disable_custom_all_reduce: ${draft.disable_custom_all_reduce}`,
    "capabilities:",
    `  kv_cache_dtype: ${draft.kv_cache_dtype}`,
    "  spec_decode:",
    `    method: ${draft.spec_decode_method || "none"}`,
    `    num_speculative_tokens: ${draft.spec_decode_K}`
  ];
}

function buildConfigCli(ctx: ConfigContext): string[] {
  const { presetId, modelId, hardwareId, profileId, runtime, draft } = ctx;
  const profileFlag = profileId ? ` --profile ${profileId}` : "";
  return [
    "# Compose and inspect the layered config",
    `sndr config compose --model ${modelId} --hardware ${hardwareId}${profileFlag} --runtime ${runtime}`,
    `sndr config show ${presetId} --format yaml`,
    "",
    "# Plan a write-safe operator-local draft (~/.sndr/model_configs)",
    `sndr config plan --preset ${presetId} \\`,
    `  --model ${modelId} --hardware ${hardwareId}${profileFlag} --runtime ${runtime}`,
    "sndr config apply --plan <plan_id>",
    "",
    "# Dry-run a launch from the draft",
    `sndr launch plan --preset ${presetId} --runtime-target ${runtime} --patch-policy ${draft.patch_policy} --dry-run`
  ];
}

function buildConfigArtifacts(ctx: ConfigContext): Record<"compose" | "run" | "systemd" | "env", string[]> {
  const { presetId, modelId, runtime, draft } = ctx;
  const container = `vllm-${presetId}`;
  const env = [
    `SNDR_PRESET=${presetId}`,
    `SNDR_MODEL=${modelId}`,
    `SNDR_RUNTIME=${runtime}`,
    `SNDR_PATCH_POLICY=${draft.patch_policy}`,
    `VLLM_MAX_MODEL_LEN=${draft.max_model_len}`,
    `VLLM_MAX_NUM_SEQS=${draft.max_num_seqs}`,
    `VLLM_GPU_MEMORY_UTILIZATION=${draft.gpu_memory_utilization.toFixed(2)}`,
    `VLLM_KV_CACHE_DTYPE=${draft.kv_cache_dtype}`
  ];
  return {
    env,
    run: [
      "docker run --rm --gpus all \\",
      `  --name ${container} \\`,
      "  -p 8000:8000 -p 8001:8001 \\",
      ...env.map((entry) => `  -e ${entry} \\`),
      "  ghcr.io/sndr/vllm-runtime:catalog \\",
      `  --served-model-name ${presetId} \\`,
      `  --max-model-len ${draft.max_model_len} \\`,
      `  --max-num-seqs ${draft.max_num_seqs} \\`,
      `  --gpu-memory-utilization ${draft.gpu_memory_utilization.toFixed(2)} \\`,
      `  --kv-cache-dtype ${draft.kv_cache_dtype}`
    ],
    compose: [
      'version: "3.8"',
      "services:",
      "  sndr-vllm:",
      "    image: ghcr.io/sndr/vllm-runtime:catalog",
      `    container_name: ${container}`,
      '    ports: ["8000:8000", "8001:8001"]',
      "    environment:",
      ...env.map((entry) => {
        const [key, ...rest] = entry.split("=");
        return `      ${key}: ${rest.join("=")}`;
      }),
      "    deploy:",
      "      resources:",
      "        reservations:",
      "          devices: [{ capabilities: [gpu] }]"
    ],
    systemd: [
      "[Unit]",
      `Description=SNDR vLLM runtime (${presetId})`,
      "After=network-online.target",
      "",
      "[Service]",
      ...env.map((entry) => `Environment=${entry}`),
      `ExecStart=/usr/bin/docker run --rm --gpus all --name ${container} \\`,
      "  -p 8000:8000 ghcr.io/sndr/vllm-runtime:catalog",
      "Restart=on-failure",
      "RestartSec=5s",
      "",
      "[Install]",
      "WantedBy=multi-user.target"
    ]
  };
}

function ParamFields({
  baseDraft,
  draft,
  set
}: {
  baseDraft: RuntimeConfigDraft;
  draft: RuntimeConfigDraft;
  set: (patch: Partial<RuntimeConfigDraft>) => void;
}) {
  const changed = (...keys: Array<keyof RuntimeConfigDraft>) =>
    keys.filter((key) => baseDraft[key] !== draft[key]).length;
  return (
    <>
      <Collapsible
        title="Sizing & memory"
        subtitle="context, batching, GPU memory"
        right={changed("max_model_len", "max_num_seqs", "max_num_batched_tokens", "gpu_memory_utilization", "enable_chunked_prefill", "enforce_eager", "disable_custom_all_reduce") || undefined}
      >
        <div className="param-grid">
          <DraftControl label="Max context" value={draft.max_model_len} min={4096} max={1048576} step={4096} onChange={(value) => set({ max_model_len: value })} />
          <DraftControl label="Max sequences" value={draft.max_num_seqs} min={1} max={256} step={1} onChange={(value) => set({ max_num_seqs: value })} />
          <DraftControl label="Max batched tokens" value={draft.max_num_batched_tokens} min={512} max={32768} step={512} onChange={(value) => set({ max_num_batched_tokens: value })} />
          <DraftControl label="GPU memory" value={Math.round(draft.gpu_memory_utilization * 100)} min={40} max={98} step={1} suffix="%" onChange={(value) => set({ gpu_memory_utilization: value / 100 })} />
        </div>
        <div className="param-toggles">
          <BoolField label="Chunked prefill" value={draft.enable_chunked_prefill} onChange={(value) => set({ enable_chunked_prefill: value })} />
          <BoolField label="Enforce eager" value={draft.enforce_eager} onChange={(value) => set({ enforce_eager: value })} />
          <BoolField label="Disable custom all-reduce" value={draft.disable_custom_all_reduce} onChange={(value) => set({ disable_custom_all_reduce: value })} />
        </div>
      </Collapsible>

      <Collapsible title="Speculative decode" subtitle="draft method and depth" right={changed("spec_decode_method", "spec_decode_K") || undefined}>
        <div className="param-grid">
          <SelectField label="Method" value={draft.spec_decode_method} options={["none", "mtp", "ngram", "eagle"]} onChange={(value) => set({ spec_decode_method: value })} />
          <DraftControl label="Num spec tokens (K)" value={draft.spec_decode_K} min={0} max={8} step={1} onChange={(value) => set({ spec_decode_K: value })} />
        </div>
      </Collapsible>

      <Collapsible title="KV cache & runtime" subtitle="cache dtype, target, patch policy" right={changed("kv_cache_dtype", "runtime_target", "patch_policy") || undefined}>
        <div className="param-grid">
          <SelectField label="KV cache dtype" value={draft.kv_cache_dtype} options={["auto", "fp8", "turboquant_k8v4", "turboquant_k8v8", "int8"]} onChange={(value) => set({ kv_cache_dtype: value })} />
          <SelectField label="Runtime target" value={draft.runtime_target} options={["docker", "docker_compose", "podman", "kubernetes", "systemd", "bare-metal"]} onChange={(value) => set({ runtime_target: value })} />
        </div>
        <div className="param-field">
          <span>Patch policy</span>
          <div className="settings-segmented">
            {["compact", "safe", "minimal"].map((policy) => (
              <button key={policy} className={draft.patch_policy === policy ? "active" : ""} onClick={() => set({ patch_policy: policy })}>{policy}</button>
            ))}
          </div>
        </div>
      </Collapsible>
    </>
  );
}

// DRAFT_FIELD_LABELS + runtimeDraftDiff extracted to ./lib/runtime-draft.


// asRecord / asText / asNumber / asStringArray extracted to ./lib/coerce.

// shortWorkload / formatTokens / formatVram extracted to ./lib/format.
