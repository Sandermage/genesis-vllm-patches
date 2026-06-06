import {
  Activity,
  AlertCircle,
  BarChart3,
  Bell,
  Box,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Circle,
  CircleAlert,
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
  Moon,
  MemoryStick,
  Monitor,
  MessageSquare,
  MousePointerClick,
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
  Sun,
  Table2,
  TimerReset,
  Trash2,
  Terminal,
  Layers,
  Package,
  Pencil,
  Leaf,
  Maximize2,
  Plus,
  Check,
  Loader2,
  Lock,
  X,
  Wrench
} from "lucide-react";
import { Component, Fragment, Suspense, lazy, useEffect, useMemo, useRef, useState, type ReactNode, type KeyboardEvent as ReactKeyboardEvent } from "react";
import { sectionFromHash, recordIdFromHash, buildHash, replaceHash } from "./route";
import { useFetch } from "./hooks/useFetch";
import { asRecord, asText, asNumber, asStringArray, countRecord } from "./lib/coerce";
import { formatAppliesTo, fmtParam, shortWorkload, formatTokens, formatVram, totalVramGiB } from "./lib/format";
import { StatusBadge, StatusPill, InfoRows, CompactList, DoctorStat, KpiGrid, CapChip, RailCheck, type GateStatus } from "./components/primitives";
import { SegmentBar, PercentBar, BarList, OvKpi, segmentsFromCounts } from "./components/charts";
import { CaveatsPanel, ConfigKeysPanel, TracesPanel } from "./sections/diagnostics";
import { DoctorSummary, DoctorFindings } from "./sections/doctor";
import { ConfigComparePanel, ConfigPlanPanel, ConfigApplyPanel } from "./sections/config";
import { BundlesPanel, UpstreamDiffPanel } from "./sections/registry";
import { UserPresetsPanel, ProfileDeltaPanel } from "./sections/presets";
import { HostInventoryPanel, DependencyStackPanel, EnvironmentPanel } from "./sections/environment";
import { DoctorCoveragePanel, AdminSurfaceMatrix } from "./sections/patch-doctor";
import { BenchmarkBaselinePanel, EvidenceRows } from "./sections/bench";
import { AuditLogPanel } from "./sections/audit-log";
import { RuntimeEndpoint, BenchmarkCard, EvidenceCard, PatchMatrix } from "./sections/rail-cards";
import { EndpointExplorer, ReportGenerator } from "./sections/api-explorer";
import { ConfirmDialog, InfoDialog } from "./components/dialogs";
import { ProofStatusPanel } from "./sections/proof";
import { CodeBlock, CopyButton } from "./components/code-block";
import { useDialogFocus, useEscapeKey, closeOnBackdrop } from "./dialog";
import { Skeleton, SkeletonMetrics, SkeletonLines, SkeletonCards, SkeletonTable } from "./Skeleton";
import {
  AlertConfig,
  BundleSpec,
  DeployTarget,
  DeployTargetsResult,
  DeploymentPlan,
  EngineStatus,
  FleetHost,
  HostInventory,
  HostReliability,
  ReliabilitySnapshot,
  HostProbe,
  HostDiscovery,
  HostSndrState,
  NodeSetupResult,
  SshCheckResult,
  OperationsResult,
  DiffUpstreamReport,
  DoctorReport,
  EnvironmentReport,
  HostProfile,
  Job,
  LaunchPlanArtifact,
  LaunchPlanEndpoint,
  BackendEvent,
  FitCheck,
  LaunchPlanResult,
  MemoryFitReport,
  ProofStatusReport,
  UserPresetList,
  V2ConfigApplyResult,
  V2ConfigCatalog,
  V2ConfigItem,
  V2ConfigPlan,
  V2ConfigPreview,
  PresetExplainResult,
  PatchDoctorReport,
  PatchExplainResult,
  PatchListResult,
  PatchRow,
  PresetListResult,
  PresetRecord,
  PresetRecommendation,
  PresetRecommendResult,
  ProductCapability,
  ProductOverview,
  ServiceActionPlan,
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
type RuntimeMode = "local" | "remote";
// GateStatus is now owned by ./components/primitives (re-imported above) so the
// RailCheck primitive and the gate logic share one source of truth.
type SectionId =
  | "overview"
  | "setup"
  | "fleet"
  | "hosts"
  | "hardware"
  | "models"
  | "configs"
  | "presets"
  | "planner"
  | "copilot"
  | "launch-plan"
  | "services"
  | "containers"
  | "routing"
  | "doctor"
  | "patches"
  | "flags"
  | "benchmarks"
  | "evidence"
  | "clients"
  | "chat"
  | "reports"
  | "operations"
  | "advanced";
type ArtifactTab = "compose" | "systemd" | "commands" | "env";
type ConsoleTab = "jobs" | "events" | "logs" | "cli";
type ThemeMode = "light" | "dark" | "carbon" | "lime";
type DensityMode = "comfortable" | "compact";
type AccentMode = "teal" | "blue" | "emerald" | "amber";
type DetailMode = "operator" | "engineer";

type RecommendForm = {
  workload: string;
  hardware: string;
  concurrency: number;
  top: number;
  preferPublic: boolean;
};

type NavItem = {
  id: SectionId;
  icon: ReactNode;
  label: string;
};

type Gate = {
  id: string;
  label: string;
  detail: string;
  status: GateStatus;
  action: string;
};

type GuiSettings = {
  theme: ThemeMode;
  density: DensityMode;
  accent: AccentMode;
  detailMode: DetailMode;
  showConnectionMap: boolean;
  autoRefresh: boolean;
  sidebarCollapsed: boolean;
};

type RuntimeConfigDraft = {
  max_model_len: number;
  max_num_seqs: number;
  max_num_batched_tokens: number;
  gpu_memory_utilization: number;
  enable_chunked_prefill: boolean;
  enforce_eager: boolean;
  disable_custom_all_reduce: boolean;
  kv_cache_dtype: string;
  spec_decode_method: string;
  spec_decode_K: number;
  runtime_target: string;
  patch_policy: string;
};

const defaultRecommend: RecommendForm = {
  workload: "free_chat",
  hardware: "a5000-2x-24gbvram-16cpu-128gbram",
  concurrency: 8,
  top: 5,
  preferPublic: true
};

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

const THEME_CYCLE: ThemeMode[] = ["light", "dark", "carbon", "lime"];
function nextTheme(current: ThemeMode): ThemeMode {
  const index = THEME_CYCLE.indexOf(current);
  return THEME_CYCLE[(index + 1) % THEME_CYCLE.length] ?? "dark";
}
function themeLabel(theme: ThemeMode): string {
  return theme === "light" ? "Light" : theme === "dark" ? "Dark" : theme === "carbon" ? "Carbon" : "Lime";
}
function themeIcon(theme: ThemeMode): ReactNode {
  return theme === "light" ? <Sun size={16} /> : theme === "carbon" ? <Sparkles size={16} /> : theme === "lime" ? <Leaf size={16} /> : <Moon size={16} />;
}
const _VALID_THEMES = new Set<ThemeMode>(["light", "dark", "carbon", "lime"]);

const workloadChoices = [
  { id: "free_chat", label: "Free chat" },
  { id: "code_gen", label: "Code gen" },
  { id: "tool_call.short", label: "Tool calls" },
  { id: "structured_json.short", label: "Structured JSON" },
  { id: "summarization", label: "Summarization" },
  { id: "long_context_qa", label: "Long context" }
];

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

function LaunchParam({ label, value }: { label: string; value: string }) {
  return (
    <div className="launch-param">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

// Dedicated launch surface — the hero "set it and launch" screen. Consolidates
// what-will-run, the resolved runtime parameters, preflight readiness and the
// prominent Launch control (previously a small button buried in the composer).
function LaunchPanel({
  selectedPreset,
  model,
  hardware,
  profile,
  host,
  composed,
  planSummary,
  card,
  patchPolicy,
  runtimeTitle,
  runtimeMode,
  endpoints,
  gates,
  gateCounts,
  applyEnabled,
  actionReason,
  launchConfirm,
  setLaunchConfirm,
  launchBusy,
  launchSshTarget,
  launchJob,
  onLaunch,
  onConfigure,
  onViewGates
}: {
  selectedPreset: string;
  model: string;
  hardware: string;
  profile: string;
  host: string;
  composed: Record<string, unknown>;
  planSummary: Record<string, unknown>;
  card: Record<string, unknown>;
  patchPolicy: string;
  runtimeTitle: string;
  runtimeMode: RuntimeMode;
  endpoints?: LaunchPlanEndpoint[];
  gates: Gate[];
  gateCounts: Record<GateStatus, number>;
  applyEnabled: boolean;
  actionReason?: string;
  launchConfirm: boolean;
  setLaunchConfirm: (value: boolean) => void;
  launchBusy: boolean;
  launchSshTarget: string;
  launchJob: Job | null;
  onLaunch: () => void;
  onConfigure: () => void;
  onViewGates: () => void;
}) {
  const ssh = launchSshTarget.trim();
  const blockers = gates.filter((gate) => gate.status === "blocked");
  const warnings = gateCounts.warning ?? 0;
  const blocked = blockers.length > 0;
  const totalGates =
    (gateCounts.pass ?? 0) + (gateCounts.warning ?? 0) + (gateCounts.blocked ?? 0) + (gateCounts.planned ?? 0);
  const primaryEndpoint =
    endpoints?.find((endpoint) => /openai|api/i.test(endpoint.label))?.url ??
    endpoints?.[0]?.url ??
    `http://${host}:8000/v1`;
  const readinessTone = blocked ? "blocked" : warnings ? "warn" : "ready";
  const readinessText = blocked ? "Launch blocked" : warnings ? "Ready — with warnings" : "Ready to launch";
  const command = [
    `sndr launch apply --preset ${selectedPreset || "<preset>"}`,
    ssh ? `  --ssh ${ssh}` : "  # local execution",
    "  --confirm"
  ];
  return (
    <section className="launch-panel">
      <div className="launch-hero">
        <div className="launch-hero-id">
          <span className="launch-hero-kicker">Step 3 · Review &amp; Launch</span>
          <h2>{selectedPreset || "No preset selected"}</h2>
          <p>{model} · {hardware}</p>
        </div>
        <div className={`launch-readiness ${readinessTone}`}>
          {blocked ? <CircleAlert size={18} /> : <CheckCircle2 size={18} />}
          <div>
            <strong>{readinessText}</strong>
            <small>{gateCounts.pass ?? 0}/{totalGates} gates passing</small>
          </div>
        </div>
      </div>

      <div className="launch-grid">
        <div className="launch-main">
          <section className="launch-card">
            <h3><Rocket size={16} /> What will run</h3>
            <InfoRows
              rows={[
                ["Preset", selectedPreset || "-"],
                ["Model", model],
                ["Hardware", hardware],
                ["Profile", profile],
                ["Runtime", runtimeTitle],
                ["Host", host],
                ["Transport", ssh ? `SSH · ${ssh}` : "Local execution"],
                ["Mode", runtimeMode === "remote" ? "Remote desktop" : "Local server"]
              ]}
            />
            <label className="endpoint-field launch-endpoint">
              <span>Serving endpoint (after launch)</span>
              <div>
                <input value={primaryEndpoint} readOnly />
                <CopyButton value={primaryEndpoint} label="endpoint" />
              </div>
            </label>
          </section>

          <section className="launch-card">
            <div className="launch-card-head">
              <h3><SlidersHorizontal size={16} /> Runtime parameters</h3>
              <button className="ghost-button" onClick={onConfigure}><SlidersHorizontal size={14} /> Adjust</button>
            </div>
            <div className="launch-params">
              <LaunchParam label="Max context" value={formatTokens(asNumber(planSummary.context) || asNumber(composed.max_model_len))} />
              <LaunchParam label="Max sequences" value={String(asNumber(planSummary.max_num_seqs) || asNumber(composed.max_num_seqs) || "-")} />
              <LaunchParam label="GPU mem util" value={asText(composed.gpu_memory_utilization, "-")} />
              <LaunchParam label="KV cache" value={asText(composed.kv_cache_dtype, "-")} />
              <LaunchParam label="Spec decode" value={`${asText(composed.spec_decode_method, "-")}/K=${asText(composed.spec_decode_K, "-")}`} />
              <LaunchParam label="Enabled patches" value={String(asNumber(planSummary.enabled_patches_count) || asNumber(composed.enabled_patches_count) || "-")} />
              <LaunchParam label="Patch policy" value={patchPolicy} />
              <LaunchParam label="Fallback" value={asText(planSummary.fallback_preset, asText(card.fallback_preset, "none"))} />
            </div>
          </section>
        </div>

        <aside className="launch-rail">
          <section className="launch-card">
            <div className="launch-card-head">
              <h3><ListChecks size={16} /> Readiness</h3>
              <button className="ghost-button" onClick={onViewGates}>All gates</button>
            </div>
            <div className="readiness-counts">
              <span className="rc ok">{gateCounts.pass ?? 0} pass</span>
              <span className="rc warn">{warnings} warn</span>
              <span className="rc bad">{blockers.length} blocked</span>
            </div>
            {blocked ? (
              <ul className="blocker-list">
                {blockers.map((gate) => (
                  <li key={gate.id}><CircleAlert size={13} /> {gate.label}</li>
                ))}
              </ul>
            ) : (
              <p className="muted">No blockers — preflight clear.</p>
            )}
          </section>

          <section className="launch-card launch-action">
            <h3><Play size={16} /> Launch</h3>
            {applyEnabled ? (
              <>
                <label className="service-confirm launch-confirm">
                  <input type="checkbox" checked={launchConfirm} onChange={(event) => setLaunchConfirm(event.target.checked)} />
                  <span>Confirm — start <strong>{selectedPreset}</strong> now</span>
                </label>
                <button className="launch-go" disabled={launchBusy || !launchConfirm} onClick={onLaunch}>
                  <Play size={17} />
                  {launchBusy ? "Launching…" : "Launch model"}
                </button>
                <p className="muted">
                  {launchConfirm
                    ? "Starts the runtime for this preset over the selected transport."
                    : "Tick confirm — this is a mutating action."}
                </p>
              </>
            ) : (
              <>
                <button className="launch-go disabled" disabled>
                  <Play size={17} /> Launch (read-only)
                </button>
                <p className="muted">{actionReason ?? "Read-only daemon. Start it with --enable-apply to launch from the GUI."}</p>
              </>
            )}
            <details className="launch-cmd">
              <summary>Equivalent CLI command</summary>
              <CodeBlock lines={command} />
            </details>
            {launchJob && <JobResultBlock job={launchJob} />}
          </section>
        </aside>
      </div>
    </section>
  );
}

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

function jobTone(job: Job): "neutral" | "success" | "danger" {
  return job.dry_run ? "neutral" : job.status === "succeeded" ? "success" : "danger";
}

// Shared executor-job result card. Used by the Launch Plan, Configs apply queue
// and the Services lifecycle planner so the three apply paths render identically.
function JobResultBlock({ job, showNote = false }: { job: Job; showNote?: boolean }) {
  return (
    <div className="service-job">
      <div className="service-job-head">
        <div>
          <strong>{job.job_id}</strong>
          <span>{job.kind}</span>
        </div>
        <StatusPill tone={jobTone(job)}>
          {job.dry_run ? "dry-run recorded" : `executed: ${job.status}`}
        </StatusPill>
      </div>
      <CodeBlock lines={job.log} />
      {showNote && job.note && <p className="service-reason">{job.note}</p>}
    </div>
  );
}

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

function PatchMatrixViewer({
  patches,
  attribution,
  loading
}: {
  patches: Record<string, string>;
  attribution: Record<string, any>;
  loading: boolean;
}) {
  const [needle, setNeedle] = useState("");
  const entries = Object.entries(patches);
  const enabled = entries.filter(([, value]) => value === "1" || value === "true").length;
  const visible = entries.filter(([flag]) =>
    !needle.trim() || flag.toLowerCase().includes(needle.trim().toLowerCase())
  );
  const attributionRows = Object.entries(attribution);

  if (loading && entries.length === 0) {
    return <div className="skel-grid cards" role="status" aria-label="Loading patch matrix…"><Skeleton variant="card" count={6} /></div>;
  }
  if (entries.length === 0) {
    return <p className="muted">This model defines no canonical patch overrides.</p>;
  }

  return (
    <div className="patch-matrix">
      <div className="patch-matrix-toolbar">
        <PercentBar
          value={enabled}
          max={entries.length}
          label="flags enabled"
          caption={`${enabled} of ${entries.length} env flags on`}
        />
        <label className="search-box">
          <Search size={15} />
          <input value={needle} onChange={(event) => setNeedle(event.target.value)} placeholder="Filter env flag" />
        </label>
      </div>
      <div className="patch-matrix-scroll">
        <table className="module-table compact">
          <thead>
            <tr>
              <th>Env flag</th>
              <th>State</th>
            </tr>
          </thead>
          <tbody>
            {visible.map(([flag, value]) => {
              const on = value === "1" || value === "true";
              return (
                <tr key={flag}>
                  <td><code>{flag.replace(/^GENESIS_ENABLE_/, "")}</code></td>
                  <td><StatusBadge status={on ? "applied" : "blocked"} /></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {attributionRows.length > 0 && (
        <div className="patch-attribution">
          <strong>Load-bearing attribution</strong>
          {attributionRows.map(([patchId, meta]) => (
            <p key={patchId}>
              <em>{patchId}</em>
              <span>{String(meta?.role ?? "documented")}</span>
            </p>
          ))}
        </div>
      )}
    </div>
  );
}

type CatalogBadge = { label: string; tone?: "neutral" | "accent" | "ok" | "warn" };

function CatalogCard({
  icon,
  id,
  title,
  badges = [],
  active,
  onClick
}: {
  icon: ReactNode;
  id: string;
  title?: string;
  badges?: CatalogBadge[];
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button type="button" className={`catalog-card${active ? " active" : ""}`} onClick={onClick}>
      <span className="catalog-card-ico">{icon}</span>
      <span className="catalog-card-body">
        <strong>{id}</strong>
        {title && <small>{title}</small>}
        {badges.length > 0 && (
          <span className="catalog-badges">
            {badges.map((badge, index) => (
              <span key={`${badge.label}-${index}`} className={`catalog-badge tone-${badge.tone ?? "neutral"}`}>
                {badge.label}
              </span>
            ))}
          </span>
        )}
      </span>
    </button>
  );
}

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

function ModelFitCard({
  modelId,
  hardwareOptions,
  defaultHardware
}: {
  modelId: string;
  hardwareOptions: string[];
  defaultHardware: string;
}) {
  const initial =
    hardwareOptions.find((id) => id === defaultHardware) ?? hardwareOptions[0] ?? "";
  const [hardware, setHardware] = useState(initial);

  // Re-pin the hardware select when the default changes (e.g. a preset loads).
  useEffect(() => {
    if (hardwareOptions.length && !hardwareOptions.includes(hardware)) {
      setHardware(initial);
    }
  }, [hardwareOptions, initial, hardware]);

  const { data: report, state, reload } = useFetch(
    (signal) => api.memoryFit({ model_id: modelId, hardware_id: hardware }, signal),
    [modelId, hardware],
    { enabled: Boolean(modelId && hardware) }
  );

  const sevIcon = (severity: FitCheck["severity"]) =>
    severity === "ok" ? (
      <CheckCircle2 size={15} className="fit-ico ok" />
    ) : (
      <CircleAlert size={15} className={`fit-ico ${severity}`} />
    );

  const vram = report?.vram;
  return (
    <div className="model-fit">
      <div className="model-fit-bar">
        <label className="model-fit-pick">
          <span>Target hardware</span>
          <select value={hardware} onChange={(event) => setHardware(event.target.value)}>
            {hardwareOptions.map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
          </select>
        </label>
        {state === "ready" && report && (
          <span className={`fit-verdict ${report.compatible ? "ok" : "blocked"}`}>
            {report.compatible ? (
              <>
                <CheckCircle2 size={15} /> Compatible
              </>
            ) : (
              <>
                <CircleAlert size={15} /> Blocked
              </>
            )}
          </span>
        )}
      </div>

      {state === "loading" && <p className="muted">Checking fit…</p>}
      {state === "error" && (
        <p className="muted fit-error">
          Could not build a fit report for this pairing.
          <button type="button" className="link-button" onClick={reload}>
            <RefreshCw size={13} /> Retry
          </button>
        </p>
      )}

      {state === "ready" && report && (
        <>
          <ul className="fit-checks">
            {report.checks.map((check) => (
              <li key={check.id} className={`fit-check ${check.severity}`}>
                {sevIcon(check.severity)}
                <div>
                  <strong>{check.title}</strong>
                  <small>{check.detail}</small>
                </div>
              </li>
            ))}
          </ul>
          {vram && (
            <PercentBar
              value={Math.min(vram.model_min_mib, vram.rig_floor_mib)}
              max={Math.max(vram.model_min_mib, vram.rig_floor_mib, 1)}
              label="VRAM (informational)"
              caption={`model needs ${formatVram(vram.model_min_mib)} · rig floor ${formatVram(
                vram.rig_floor_mib
              )} (${vram.n_gpus}×${formatVram(vram.vram_per_gpu_mib)} × ${Math.round(
                vram.gpu_memory_utilization * 100
              )}%) · KV ${vram.kv_cache_dtype}`}
              tone={vram.headroom_mib >= 0 ? "ok" : "warn"}
            />
          )}
          <p className="fit-note">
            Verdict uses the deterministic requirements (GPU count, CUDA capability, blocklist).
            VRAM is shown for context — the rig floor is a conservative match threshold, not the
            card's real capacity, so it never flips the verdict.
          </p>
        </>
      )}
    </div>
  );
}

// Enterprise fit matrix: probe the model against EVERY catalogued rig so an
// operator sees at a glance where it can run and where it is blocked.
function ModelFitMatrix({ modelId, hardwareIds }: { modelId: string; hardwareIds: string[] }) {
  const [rows, setRows] = useState<MemoryFitReport[] | null>(null);
  const [loading, setLoading] = useState(false);
  const key = hardwareIds.join(",");
  useEffect(() => {
    if (!modelId || hardwareIds.length === 0) { setRows([]); return; }
    let cancelled = false;
    setLoading(true);
    Promise.all(hardwareIds.map((hw) => api.memoryFit({ model_id: modelId, hardware_id: hw }).catch(() => null)))
      .then((reports) => { if (!cancelled) setRows(reports.filter((report): report is MemoryFitReport => Boolean(report))); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelId, key]);

  if (loading && !rows) return <p className="muted">Checking {hardwareIds.length} rigs…</p>;
  if (!rows || rows.length === 0) return <p className="muted">No hardware in the catalog to match against.</p>;
  const fits = rows.filter((report) => report.compatible).length;
  return (
    <div className="fit-matrix">
      <div className="fit-matrix-summary">
        <span className="fleet-status ok"><span className="fleet-dot" />fits on {fits}</span>
        <span className="fleet-status danger"><span className="fleet-dot" />blocked on {rows.length - fits}</span>
      </div>
      <div className="patch-table-scroll">
        <table className="module-table fit-matrix-table">
          <thead><tr><th>Hardware</th><th>Verdict</th><th>GPUs</th><th>VRAM headroom</th><th>KV</th><th>Blockers</th></tr></thead>
          <tbody>
            {rows.map((report) => {
              const blockers = report.checks.filter((check) => !check.ok).map((check) => check.title);
              return (
                <tr key={report.hardware_id}>
                  <td><strong>{report.hardware_title || report.hardware_id}</strong></td>
                  <td><span className={`fit-pill ${report.compatible ? "ok" : "blocked"}`}>{report.compatible ? "fits" : "blocked"}</span></td>
                  <td>{report.vram.n_gpus}× {formatVram(report.vram.vram_per_gpu_mib)}</td>
                  <td className={report.vram.headroom_mib >= 0 ? "fit-pos" : "fit-neg"}>{report.vram.headroom_mib >= 0 ? "+" : "−"}{Math.abs(report.vram.headroom_mib / 1024).toFixed(1)} GB</td>
                  <td>{report.vram.kv_cache_dtype}</td>
                  <td>{blockers.length ? blockers.join(", ") : <span className="muted">—</span>}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="fit-note">Verdict uses deterministic requirements (GPU count, CUDA capability, arch blocklist). VRAM headroom is informational context against a conservative rig floor.</p>
    </div>
  );
}

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

// Interactive fit envelope for a model on a representative rig — context ×
// concurrency, cell colour = headroom. Reuses the KV calculator backend so the
// catalog answers "what can this model actually run?" in place.
function KvEnvelopeCard({ modelKey, tp, vram, rigLabel }: { modelKey: string | null; tp: number; vram: number; rigLabel: string }) {
  const [env, setEnv] = useState<{ contexts: number[]; concurrencies: number[]; grid: Array<Array<{ context: number; headroom_mib: number; fits: boolean }>> } | null>(null);
  const [err, setErr] = useState(false);
  useEffect(() => {
    if (!modelKey) return;
    setEnv(null); setErr(false);
    let alive = true;
    api.calcKv({ model_id: modelKey, context: 32768, concurrency: 1, tp, gpu_count: tp, gpu_vram_mib: vram, util: 0.9, kv_dtype: "fp8" })
      .then((r) => { if (alive) setEnv(r.envelope); }).catch(() => { if (alive) setErr(true); });
    return () => { alive = false; };
  }, [modelKey, tp, vram]);
  if (!modelKey) return <p className="muted">No KV sizing metadata for this model id.</p>;
  if (err) return <p className="muted">Couldn't compute the fit envelope.</p>;
  if (!env) return <p className="muted">Computing fit envelope…</p>;
  const fmtC = (c: number) => (c >= 1000 ? `${Math.round(c / 1000)}K` : String(c));
  const color = (h: number) => (h < 0 ? "over" : h < 2048 ? "tight" : h < 6144 ? "ok" : "good");
  return (
    <>
      <p className="fit-note">On <code>{rigLabel}</code> · {tp}× {Math.round(vram / 1024)}GB · fp8 KV · 90% util.</p>
      <div className="heatmap">
        <div className="heatmap-grid" style={{ gridTemplateColumns: `40px repeat(${env.contexts.length}, 1fr)` }}>
          <span className="heatmap-corner" />
          {env.contexts.map((c) => <span key={c} className="heatmap-xlabel">{fmtC(c)}</span>)}
          {[...env.grid].reverse().map((row, ri) => {
            const k = [...env.concurrencies].reverse()[ri];
            return (
              <Fragment key={k}>
                <span className="heatmap-ylabel">{k}×</span>
                {row.map((cell) => <div key={cell.context} className={`heatmap-cell ${color(cell.headroom_mib)}`} title={`${fmtC(cell.context)} ctx · ${k} conc → ${cell.fits ? "fits" : "over budget"}`} />)}
              </Fragment>
            );
          })}
        </div>
        <div className="heatmap-legend"><span className="good" /> roomy<span className="ok" /> fits<span className="tight" /> tight<span className="over" /> over</div>
      </div>
    </>
  );
}

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

type ElementKind = "model" | "hardware" | "profile" | "preset";
type FieldSpec = {
  path: string;
  label: string;
  type: "text" | "number" | "select" | "bool";
  options?: string[];
  group?: string;
  hint?: string;
};

const ELEMENT_FIELDS: Record<ElementKind, FieldSpec[]> = {
  model: [
    { path: "title", label: "Title", type: "text", group: "Identity" },
    { path: "served_model_name", label: "Served name", type: "text", group: "Identity" },
    { path: "model_path", label: "Model path", type: "text", group: "Identity", hint: "Container/host checkpoint path" },
    { path: "maintainer", label: "Maintainer", type: "text", group: "Identity" },
    { path: "license", label: "License", type: "text", group: "Identity" },
    { path: "last_validated", label: "Last validated", type: "text", group: "Identity" },
    { path: "dtype", label: "Dtype", type: "select", options: ["float16", "bfloat16", "float32"], group: "Precision" },
    { path: "quantization", label: "Quantization", type: "text", group: "Precision" },
    { path: "trust_remote_code", label: "Trust remote code", type: "bool", group: "Precision" },
    { path: "capabilities.attention_arch", label: "Attention arch", type: "select", options: ["dense", "hybrid_gdn_moe", "hybrid_mamba", "moe", "gemma4_dense", "gemma4_moe"], group: "Capabilities" },
    { path: "capabilities.kv_cache_dtype", label: "KV cache dtype", type: "select", options: ["auto", "fp8", "turboquant_k8v4", "turboquant_k8v8", "int8"], group: "Capabilities" },
    { path: "capabilities.tool_call_parser", label: "Tool parser", type: "text", group: "Capabilities" },
    { path: "capabilities.reasoning_parser", label: "Reasoning parser", type: "text", group: "Capabilities" },
    { path: "capabilities.enable_auto_tool_choice", label: "Auto tool choice", type: "bool", group: "Capabilities" },
    { path: "capabilities.spec_decode.method", label: "Spec method", type: "select", options: ["mtp", "ngram", "eagle"], group: "Speculative decode" },
    { path: "capabilities.spec_decode.num_speculative_tokens", label: "Spec K", type: "number", group: "Speculative decode" },
    { path: "requires.min_gpu_count", label: "Min GPUs", type: "number", group: "Requirements" },
    { path: "requires.min_total_vram_mib", label: "Min VRAM (MiB)", type: "number", group: "Requirements" },
    { path: "versions.genesis_pin_min", label: "Genesis pin (min)", type: "text", group: "Version pins" },
    { path: "versions.vllm_pin_required", label: "vLLM pin required", type: "text", group: "Version pins" },
    { path: "versions.reference_metrics_ref", label: "Reference metrics ref", type: "text", group: "Version pins" }
  ],
  hardware: [
    { path: "title", label: "Title", type: "text", group: "Identity" },
    { path: "maintainer", label: "Maintainer", type: "text", group: "Identity" },
    { path: "hardware.n_gpus", label: "GPU count", type: "number", group: "GPU" },
    { path: "hardware.min_vram_per_gpu_mib", label: "Min VRAM/GPU (MiB)", type: "number", group: "GPU" },
    { path: "hardware.cuda_capability_min", label: "CUDA cap min", type: "text", group: "GPU", hint: "e.g. 8.6 (Ampere)" },
    { path: "sizing.max_model_len", label: "Max context", type: "number", group: "Sizing" },
    { path: "sizing.max_num_seqs", label: "Max sequences", type: "number", group: "Sizing" },
    { path: "sizing.max_num_batched_tokens", label: "Max batched tokens", type: "number", group: "Sizing" },
    { path: "sizing.gpu_memory_utilization", label: "GPU mem util", type: "number", group: "Sizing" },
    { path: "sizing.enable_chunked_prefill", label: "Chunked prefill", type: "bool", group: "Sizing" },
    { path: "sizing.enforce_eager", label: "Enforce eager", type: "bool", group: "Sizing" },
    { path: "sizing.disable_custom_all_reduce", label: "Disable custom all-reduce", type: "bool", group: "Sizing" },
    { path: "runtime.default", label: "Default runtime", type: "select", options: ["docker", "podman", "bare-metal"], group: "Runtime" }
  ],
  profile: [
    { path: "parent_model", label: "Parent model", type: "text", group: "Identity" },
    { path: "status", label: "Status", type: "select", options: ["experimental", "validated", "promoted"], group: "Identity" },
    { path: "role", label: "Role", type: "select", options: ["default", "structured", "gateway", "bench", "dev", "qa", "diagnostic"], group: "Identity" },
    { path: "created", label: "Created", type: "text", group: "Identity" },
    { path: "sizing_override.max_model_len", label: "Max context", type: "number", group: "Sizing override", hint: "Leave empty to inherit hardware" },
    { path: "sizing_override.max_num_seqs", label: "Max sequences", type: "number", group: "Sizing override" },
    { path: "sizing_override.max_num_batched_tokens", label: "Max batched tokens", type: "number", group: "Sizing override" },
    { path: "sizing_override.gpu_memory_utilization", label: "GPU mem util", type: "number", group: "Sizing override" },
    { path: "sizing_override.enforce_eager", label: "Enforce eager", type: "bool", group: "Sizing override" },
    { path: "versions_override.vllm_pin_required", label: "vLLM pin required", type: "text", group: "Version override" },
    { path: "versions_override.genesis_pin", label: "Genesis pin", type: "text", group: "Version override" },
    { path: "promotion.promote_to", label: "Promote to", type: "text", group: "Promotion" },
    { path: "promotion.notes", label: "Promotion notes", type: "text", group: "Promotion" }
  ],
  preset: [
    { path: "model", label: "Model", type: "text", group: "Composition" },
    { path: "hardware", label: "Hardware", type: "text", group: "Composition" },
    { path: "profile", label: "Profile", type: "text", group: "Composition" },
    { path: "runtime", label: "Runtime", type: "select", options: ["docker", "podman", "kubernetes", "systemd", "bare-metal"], group: "Composition" },
    { path: "card.title", label: "Card title", type: "text", group: "Card" },
    { path: "card.summary", label: "Summary", type: "text", group: "Card" },
    { path: "card.status", label: "Card status", type: "select", options: ["experimental", "production_candidate", "production"], group: "Card" },
    { path: "card.mode", label: "Mode", type: "text", group: "Card" },
    { path: "card.audience", label: "Audience", type: "select", options: ["operator", "developer", "internal"], group: "Card" },
    { path: "card.evidence_visibility", label: "Evidence visibility", type: "select", options: ["public", "private", "mixed"], group: "Card" },
    { path: "card.fallback_preset", label: "Fallback preset", type: "text", group: "Card" }
  ]
};

// Top-level keys excluded from auto-discovery: structural, noisy, or shown
// elsewhere (patch matrix), or arrays/dicts that need bespoke editors.
const _AUTO_EXCLUDE = new Set([
  "patches", "patches_attribution", "notes", "schema_version", "kind", "id",
  "patches_delta", "system_env"
]);

function _isScalar(v: any): boolean {
  return v === null || ["string", "number", "boolean"].includes(typeof v);
}

// Adaptive discovery: walk the loaded definition and surface every scalar leaf
// that the curated schema does not already cover, so the editor reflects
// whatever fields a given model/hardware/profile/preset actually contains.
function discoverExtraFields(obj: any, known: Set<string>): FieldSpec[] {
  const out: FieldSpec[] = [];
  const walk = (node: any, prefix: string): void => {
    if (!node || typeof node !== "object" || Array.isArray(node)) return;
    for (const [key, value] of Object.entries(node)) {
      const path = prefix ? `${prefix}.${key}` : key;
      if (!prefix && _AUTO_EXCLUDE.has(key)) continue;
      if (_isScalar(value)) {
        if (!known.has(path)) {
          const type = typeof value === "boolean" ? "bool" : typeof value === "number" ? "number" : "text";
          out.push({ path, label: key, type, group: prefix ? `More · ${prefix}` : "More" });
        }
      } else if (value && typeof value === "object" && !Array.isArray(value)) {
        walk(value, path);
      }
      // arrays / arrays-of-objects are left to the YAML panel (bespoke shape)
    }
  };
  walk(obj, "");
  return out;
}

function groupFields(fields: FieldSpec[]): Array<[string, FieldSpec[]]> {
  const order: string[] = [];
  const byGroup = new Map<string, FieldSpec[]>();
  for (const spec of fields) {
    const group = spec.group ?? "";
    if (!byGroup.has(group)) { byGroup.set(group, []); order.push(group); }
    byGroup.get(group)!.push(spec);
  }
  return order.map((group) => [group, byGroup.get(group)!]);
}

function getIn(obj: any, path: string): any {
  return path.split(".").reduce((current, key) => (current == null ? undefined : current[key]), obj);
}

function setIn(obj: any, path: string, value: any): any {
  const keys = path.split(".");
  const clone = Array.isArray(obj) ? [...obj] : { ...obj };
  let cursor = clone;
  for (let i = 0; i < keys.length - 1; i += 1) {
    const key = keys[i];
    const next = cursor[key];
    cursor[key] = next && typeof next === "object" ? (Array.isArray(next) ? [...next] : { ...next }) : {};
    cursor = cursor[key];
  }
  cursor[keys[keys.length - 1]] = value;
  return clone;
}

function yamlScalar(value: any): string {
  if (value === null || value === undefined) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") return String(value);
  const text = String(value);
  return /[:#{}[\],&*?|<>=!%@`]/.test(text) || text.trim() !== text ? JSON.stringify(text) : text;
}

function objToYaml(obj: any, indent = 0): string[] {
  const pad = "  ".repeat(indent);
  const lines: string[] = [];
  const entries: Array<[string, any]> = Array.isArray(obj)
    ? obj.map((value, index) => [String(index), value])
    : Object.entries(obj ?? {});
  for (const [key, value] of entries) {
    const label = `${pad}${key}:`;
    if (value === null || value === undefined) {
      lines.push(`${label} null`);
    } else if (Array.isArray(value)) {
      if (value.length === 0) { lines.push(`${label} []`); continue; }
      lines.push(label);
      value.forEach((item) => {
        if (item && typeof item === "object") {
          const sub = objToYaml(item, indent + 2);
          lines.push(`${pad}  -`);
          lines.push(...sub);
        } else {
          lines.push(`${pad}  - ${yamlScalar(item)}`);
        }
      });
    } else if (typeof value === "object") {
      if (Object.keys(value).length === 0) { lines.push(`${label} {}`); continue; }
      lines.push(label);
      lines.push(...objToYaml(value, indent + 1));
    } else {
      lines.push(`${label} ${yamlScalar(value)}`);
    }
  }
  return lines;
}

function TextField({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) {
  return (
    <label className="param-field">
      <span>{label}</span>
      <input value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

function NumberField({ label, value, onChange }: { label: string; value: number; onChange: (value: number) => void }) {
  return (
    <label className="param-field">
      <span>{label}</span>
      <input type="number" value={value} onChange={(event) => onChange(Number(event.target.value))} />
    </label>
  );
}

// Live sanity-check for the most error-prone numeric config fields — catches a
// bad value (e.g. gpu_memory_utilization 1.5) before it's saved/applied.
function fieldWarning(spec: FieldSpec, value: any): string | null {
  if (value == null || value === "" || spec.type !== "number") return null;
  const n = Number(value);
  if (!Number.isFinite(n)) return "not a number";
  const leaf = spec.path.split(".").pop();
  switch (leaf) {
    case "gpu_memory_utilization": return n > 0 && n <= 1 ? null : "expected 0 < util ≤ 1";
    case "max_num_seqs": return Number.isInteger(n) && n >= 1 && n <= 4096 ? null : "expected 1–4096";
    case "max_num_batched_tokens": return n >= 256 ? null : "expected ≥ 256";
    case "max_model_len": return n >= 256 ? null : "expected ≥ 256";
    case "num_speculative_tokens": return n >= 0 && n <= 16 ? null : "expected 0–16";
    case "n_gpus":
    case "min_gpu_count": return Number.isInteger(n) && n >= 1 && n <= 8 ? null : "expected 1–8";
    default: return n < 0 ? "must be ≥ 0" : null;
  }
}

function ElementField({ spec, value, onChange }: { spec: FieldSpec; value: any; onChange: (value: any) => void }) {
  const warn = fieldWarning(spec, value);
  const field = (() => {
    if (spec.type === "bool") {
      return <BoolField label={spec.label} value={Boolean(value)} onChange={onChange} />;
    }
    if (spec.type === "number") {
      return <NumberField label={spec.label} value={typeof value === "number" ? value : Number(value) || 0} onChange={onChange} />;
    }
    if (spec.type === "select") {
      const current = value == null ? "" : String(value);
      const options = spec.options ?? [];
      const merged = current && !options.includes(current) ? [current, ...options] : options;
      return <SelectField label={spec.label} value={current} options={merged} onChange={onChange} />;
    }
    return <TextField label={spec.label} value={value == null ? "" : String(value)} onChange={onChange} />;
  })();
  if (!spec.hint && !warn) return field;
  return (
    <div className={`element-field-hinted${warn ? " invalid" : ""}`}>
      {field}
      {warn
        ? <small className="element-field-warn"><AlertTriangle size={11} /> {warn}</small>
        : spec.hint ? <small className="element-field-hint">{spec.hint}</small> : null}
    </div>
  );
}

function LayerEditor({ kind, layerId }: { kind: ElementKind; layerId: string }) {
  const [edited, setEdited] = useState<Record<string, any> | null>(null);
  const [source, setSource] = useState("");
  const [state, setState] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const [applyResult, setApplyResult] = useState<V2LayerApplyResult | null>(null);
  const [applying, setApplying] = useState(false);

  useEffect(() => {
    if (!layerId) { setEdited(null); return; }
    const controller = new AbortController();
    setState("loading");
    setError(null);
    setApplyResult(null);
    api.v2Layer(kind, layerId, controller.signal)
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
  }, [kind, layerId]);

  const fields = useMemo(() => {
    const curated = ELEMENT_FIELDS[kind];
    if (!edited) return curated;
    const known = new Set(curated.map((spec) => spec.path));
    return [...curated, ...discoverExtraFields(edited, known)];
  }, [kind, edited]);
  const yaml = edited ? objToYaml(edited) : [`# Loading ${kind}…`];

  async function runSave() {
    if (!edited) return;
    setApplying(true);
    try {
      const result = await api.v2LayerApply({ kind, layer_id: layerId, yaml_text: yaml.join("\n") + "\n" });
      setApplyResult(result);
    } catch (err) {
      setApplyResult({
        kind, layer_id: layerId, target_path: "", action: "create", written: false,
        bytes_written: 0, status: "blocked", message: err instanceof Error ? err.message : String(err), blocked_reasons: []
      });
    } finally {
      setApplying(false);
    }
  }

  return (
    <div className="preset-editor">
      <p className="element-source">{source || (state === "loading" ? "loading…" : "")}</p>
      {error && <div className="config-plan-error"><AlertCircle size={15} /><span>{error}</span></div>}
      {edited ? (
        <div className="preset-editor-cols">
          <div className="element-groups">
            {groupFields(fields).map(([group, list]) => (
              <div className="element-group" key={group}>
                {group && <div className="element-group-head">{group}</div>}
                <div className="element-fields">
                  {list.map((spec) => (
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
          <div className="preset-editor-yaml">
            <div className="config-panel-title"><Code2 size={16} /><strong>{kind}.yaml</strong><span>{yaml.length} lines</span></div>
            <CodeBlock lines={yaml} />
          </div>
        </div>
      ) : (
        <p className="muted">Select a {kind} to edit.</p>
      )}
      {applyResult && (
        <div className={`element-apply ${applyResult.status}`}>
          <span className="finding-icon">
            {applyResult.status === "applied" ? <CheckCircle2 size={14} /> : <AlertCircle size={14} />}
          </span>
          <div>
            <strong>{applyResult.status === "applied" ? "Saved to user dir" : applyResult.status}</strong>
            <small>{applyResult.target_path || applyResult.message}</small>
          </div>
        </div>
      )}
      <div className="config-actions">
        <span className="config-actions-note">Edits write an operator-local {kind} copy (never the builtin)</span>
        <button className="primary-action" onClick={() => void runSave()} disabled={!edited || applying}>
          <PackageCheck size={14} /> {applying ? "Saving…" : "Save to user dir"}
        </button>
      </div>
    </div>
  );
}

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

function ModuleGrid({ children, className }: { children: ReactNode; className?: string }) {
  return <section className={`module-grid${className ? ` ${className}` : ""}`}>{children}</section>;
}

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

function TabbedSection({
  id,
  tabs,
  activeTab,
  onTabChange
}: {
  id: string;
  tabs: Array<{ id: string; label: string; icon?: ReactNode; render: () => ReactNode }>;
  // Optional controlled mode: when provided, the parent owns the active tab so
  // in-panel buttons (e.g. "Edit preset") can switch tabs programmatically.
  activeTab?: string;
  onTabChange?: (id: string) => void;
}) {
  const [internal, setInternal] = useState(tabs[0]?.id ?? "");
  const active = activeTab ?? internal;
  const setActive = (next: string) => {
    if (onTabChange) onTabChange(next);
    else setInternal(next);
  };
  // Reset to the first tab when the section changes (component is keyed per section).
  useEffect(() => {
    if (activeTab === undefined) setInternal(tabs[0]?.id ?? "");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);
  const current = tabs.find((tab) => tab.id === active) ?? tabs[0];
  const activeIndex = Math.max(0, tabs.findIndex((tab) => tab.id === current?.id));
  // WCAG APG tabs pattern: roving tabindex + arrow/Home/End keyboard navigation.
  const onTabKey = (event: { key: string; preventDefault: () => void }) => {
    const keys = ["ArrowRight", "ArrowLeft", "Home", "End"];
    if (!keys.includes(event.key) || tabs.length === 0) return;
    event.preventDefault();
    let next = activeIndex;
    if (event.key === "ArrowRight") next = (activeIndex + 1) % tabs.length;
    else if (event.key === "ArrowLeft") next = (activeIndex - 1 + tabs.length) % tabs.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = tabs.length - 1;
    setActive(tabs[next].id);
  };
  return (
    <div className="section-tabs-wrap">
      {/* eslint-disable-next-line jsx-a11y/interactive-supports-focus -- ARIA tablist uses roving tabindex on the tabs; the container itself is not focusable by design */}
      <div className="section-tabs" role="tablist" onKeyDown={onTabKey}>
        {tabs.map((tab) => (
          <button
            key={tab.id}
            type="button"
            role="tab"
            id={`${id}-tab-${tab.id}`}
            aria-selected={tab.id === current?.id}
            aria-controls={`${id}-panel-${tab.id}`}
            tabIndex={tab.id === current?.id ? 0 : -1}
            className={tab.id === current?.id ? "active" : ""}
            onClick={() => setActive(tab.id)}
          >
            {tab.icon}
            <span>{tab.label}</span>
          </button>
        ))}
      </div>
      <div
        className="section-tab-body"
        role="tabpanel"
        id={`${id}-panel-${current?.id}`}
        aria-labelledby={`${id}-tab-${current?.id}`}
      >
        {current?.render()}
      </div>
    </div>
  );
}

function ModuleCard({
  title,
  icon,
  desc,
  children,
  wide = false
}: {
  title: string;
  icon: ReactNode;
  desc?: string;
  children: ReactNode;
  wide?: boolean;
}) {
  return (
    <section className={`module-card ${wide ? "wide" : ""}`}>
      <div className="module-card-title">
        <span className="module-card-icon">{icon}</span>
        <div className="module-card-heading">
          <h2>{title}</h2>
          {desc && <p>{desc}</p>}
        </div>
      </div>
      {children}
    </section>
  );
}

// InfoRows extracted to ./components/primitives.

// KpiGrid extracted to ./components/primitives.

function RuntimeEnvelopePanel({
  card,
  composed,
  patchCount
}: {
  card: Record<string, unknown>;
  composed: Record<string, unknown>;
  patchCount: number;
}) {
  const metric = asRecord(card.primary_metric);
  const context = asNumber(composed.max_model_len);
  const sequences = asNumber(composed.max_num_seqs);
  const patches = asNumber(composed.enabled_patches_count);
  return (
    <div className="runtime-envelope">
      <BarList
        rows={[
          ["Context", Math.min(100, Math.round(context / 4096)), formatTokens(context)],
          ["Concurrency", Math.min(100, sequences * 10), String(sequences || "-")],
          ["Enabled patches", patchCount ? Math.round((patches / patchCount) * 100) : 0, String(patches || 0)],
          ["Metric", Math.min(100, Math.round(asNumber(metric.value) / 8)), String(asNumber(metric.value) || "pending")]
        ]}
      />
      <InfoRows
        rows={[
          ["KV Cache", asText(composed.kv_cache_dtype, "-")],
          ["Spec Decode", asText(composed.spec_decode_method, "-")],
          ["Spec K", String(asNumber(composed.spec_decode_K) || "-")],
          ["Evidence", asText(card.evidence_visibility, "unknown")]
        ]}
      />
    </div>
  );
}

function PresetPolicyGraph({
  card,
  presets
}: {
  card: Record<string, unknown>;
  presets: PresetRecord[];
}) {
  const allow = asStringArray(card.workload_allow);
  const deny = asStringArray(card.workload_deny);
  const statuses = countRecord(
    presets
      .filter((preset) => preset.has_card)
      .map((preset) => asText(preset.card?.status, "unknown"))
  );
  const maxStatus = Math.max(1, ...Object.values(statuses));
  return (
    <div className="policy-graph">
      <div className="policy-pill-grid">
        {allow.map((item) => <span className="policy-pill allow" key={`allow-${item}`}>{item}</span>)}
        {deny.map((item) => <span className="policy-pill deny" key={`deny-${item}`}>{item}</span>)}
      </div>
      <BarList
        rows={Object.entries(statuses).map(([status, value]) => [
          status,
          Math.round((value / maxStatus) * 100),
          String(value)
        ])}
      />
    </div>
  );
}

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

function BoolField({ label, value, onChange }: { label: string; value: boolean; onChange: (value: boolean) => void }) {
  return (
    <button className="bool-field" onClick={() => onChange(!value)} aria-pressed={value}>
      <span>{label}</span>
      <i className={value ? "active" : ""} />
    </button>
  );
}

function SelectField({
  label,
  value,
  options,
  onChange
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
}) {
  return (
    <label className="param-field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => (
          <option key={option} value={option}>{option}</option>
        ))}
      </select>
    </label>
  );
}

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

const HOST_ROLES = ["production", "staging", "dev", "experiment"] as const;

function roleTone(role: string): string {
  if (role === "production") return "danger";
  if (role === "staging") return "warn";
  if (role === "dev" || role === "experiment") return "info";
  return "muted";
}

function tunnelCommand(profile: HostProfile): string {
  return profile.transport === "ssh" && profile.ssh_target
    ? `ssh -L ${profile.port}:127.0.0.1:${profile.port} ${profile.ssh_target}`
    : `# local — open http://127.0.0.1:${profile.port}`;
}

// One fleet card per saved host: identity, role, hardware, tags + a live
// engine reachability probe against host:engine_port.
// CapChip extracted to ./components/primitives; totalVramGiB to ./lib/format.

// Actionable guidance when the daemon is read-only (apply gated off). Shows the
// exact restart command with a copy button, instead of a bare "apply is disabled".
function ApplyDisabledNote({ what }: { what: string }) {
  const cmd = "python3 -m vllm.sndr_core.cli gui-api --enable-apply";
  const [copied, setCopied] = useState(false);
  return (
    <div className="apply-gate">
      <Lock size={14} />
      <div className="apply-gate-body">
        <strong>{what} needs apply — the daemon is read-only.</strong>
        <span>Restart it with apply enabled (or set <code>SNDR_ENABLE_APPLY=1</code>):</span>
        <div className="apply-gate-cmdrow">
          <code className="apply-gate-cmd">{cmd}</code>
          <button className="apply-gate-copy" onClick={() => { navigator.clipboard?.writeText(cmd); setCopied(true); window.setTimeout(() => setCopied(false), 1500); }}>
            {copied ? <CheckCircle2 size={12} /> : <Copy size={12} />} {copied ? "copied" : "copy"}
          </button>
        </div>
      </div>
    </div>
  );
}

// Bar sparkline of reachability samples (1 = reachable, 0 = down).
function RelSpark({ samples }: { samples: number[] }) {
  if (!samples.length) return null;
  return (
    <svg className="fleet-rel-svg" viewBox={`0 0 ${samples.length} 1`} preserveAspectRatio="none" aria-hidden="true">
      {samples.map((s, i) => (
        <rect key={i} x={i + 0.12} y={s ? 0 : 0.55} width={0.76} height={s ? 1 : 0.45} className={s ? "ok" : "down"} />
      ))}
    </svg>
  );
}

function FleetHostCard({
  profile,
  onEdit,
  onDelete,
  onChat,
  onAddServer,
  onRefresh,
  onTerminal,
  focused,
  onFocusConsumed,
  onContainers,
  onHardware,
  reliability,
  fleet,
  applyEnabled
}: {
  profile: HostProfile;
  onEdit: (profile: HostProfile) => void;
  onDelete: (id: string) => void;
  onChat: (profile: HostProfile) => void;
  onAddServer: (profile: HostProfile) => Promise<boolean>;
  onRefresh: () => void;
  onTerminal: (profile: HostProfile) => void;
  focused?: boolean;
  onFocusConsumed?: () => void;
  onSetupNode?: (id: string) => void;
  onContainers?: (id: string) => void;
  onHardware?: (id: string) => void;
  reliability?: HostReliability | null;
  fleet?: FleetHost | null;
  applyEnabled?: boolean;
}) {
  const [probe, setProbe] = useState<HostProbe | null>(null);
  const [busy, setBusy] = useState(false);
  const [checkedAt, setCheckedAt] = useState<string | null>(null);
  const [ssh, setSsh] = useState<SshCheckResult | null>(null);
  const [sshBusy, setSshBusy] = useState(false);
  const [keyBusy, setKeyBusy] = useState(false);
  const [disco, setDisco] = useState<HostDiscovery | null>(null);
  const [discoBusy, setDiscoBusy] = useState(false);
  const [sndr, setSndr] = useState<HostSndrState | null>(null);
  // One-click "Set up as node": install the SNDR daemon on this host over SSH.
  const [nodeForm, setNodeForm] = useState(false);
  const [nodePw, setNodePw] = useState("");
  const [nodeBusy, setNodeBusy] = useState(false);
  const [nodeResult, setNodeResult] = useState<NodeSetupResult | null>(null);
  async function installNode() {
    if (nodePw.length < 4 || nodeBusy) return;
    setNodeBusy(true); setNodeResult(null);
    try {
      const r = await api.installNode(profile.id, nodePw, profile.engine_port || 8102);
      setNodeResult(r);
      if (r.ok) onRefresh();  // refresh so the switcher re-probes and sees the new daemon
    } catch (e) {
      setNodeResult({ ok: false, applied: false, steps: [], error: e instanceof Error ? e.message : String(e) });
    } finally { setNodeBusy(false); }
  }
  // null = unknown, false = probed and no daemon (engine box), true = daemon found.
  const [daemonOk, setDaemonOk] = useState<boolean | null>(null);
  const [connecting, setConnecting] = useState(false);
  const cardRef = useRef<HTMLElement | null>(null);
  const isSsh = profile.transport === "ssh" || !!profile.ssh_user;
  // Opened from the connection switcher → scroll into view and auto-discover so
  // the operator immediately sees what's running on this host (the runtime view).
  useEffect(() => {
    if (!focused) return;
    cardRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
    if (isSsh && !disco && !discoBusy) void discover();
    onFocusConsumed?.();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focused]);
  async function discover() {
    setDiscoBusy(true);
    try {
      const d = await api.discoverHost(profile.id);
      setDisco(d);
      // Light Path B: also read this host's own sndr_core identity (patcher
      // version / vLLM build / config + patch-registry size) from its container.
      api.sndrState(profile.id).then((s) => setSndr(s.ok ? s : null)).catch(() => {});
      if (d.engine_port_set) { toast(`Discovered engine → port ${d.engine_port_set} set on ${profile.label}`, "success"); onRefresh(); }
      else if (!d.engines.length) toast(d.error || "Nothing discovered on host", "info");
      else toast(`Found ${d.engines.length} engine(s), ${d.gpus.length} GPU(s)`, "success");
    } catch (err) {
      toast(err instanceof Error ? err.message : "Discovery failed", "error");
    } finally { setDiscoBusy(false); }
  }
  async function applyEnginePort(port: number) {
    try { await api.hostUpsert({ ...profile, engine_port: port }); toast(`Engine port → ${port}`, "success"); onRefresh(); } catch { toast("Failed to set port", "error"); }
  }
  async function fetchKey() {
    setKeyBusy(true);
    try {
      const r = await api.fetchApiKey(profile.id);
      if (r.found) { toast(`API key fetched from ${r.source} → ${r.key_masked}`, "success"); onRefresh(); }
      else toast(r.error || "No API key found on host", "info");
    } catch (err) {
      toast(err instanceof Error ? err.message : "Fetch failed", "error");
    } finally { setKeyBusy(false); }
  }
  async function checkSsh() {
    setSshBusy(true);
    try {
      setSsh(await api.sshCheck({ host: profile.host, host_id: profile.id, user: profile.ssh_user, auth_method: profile.ssh_auth, key_path: profile.ssh_key_path, ssh_port: profile.ssh_port }));
    } catch (err) {
      setSsh({ available: true, ssh_ok: false, sftp_ok: false, latency_ms: null, banner: null, uname: null, error: err instanceof Error ? err.message : String(err) });
    } finally {
      setSshBusy(false);
    }
  }
  async function check() {
    setBusy(true);
    try {
      setProbe(await api.hostProbe(profile.host, profile.engine_port, undefined, profile.id));
    } catch (err) {
      setProbe({ reachable: false, host: profile.host, port: profile.engine_port, base_url: "", version: null, models: [], latency_ms: null, error: err instanceof Error ? err.message : String(err) });
    } finally {
      setBusy(false);
      setCheckedAt(new Date().toLocaleTimeString());
    }
  }
  const statusLabel = !probe ? "not probed" : probe.reachable ? "engine up" : "unreachable";
  const statusTone = !probe ? "muted" : probe.reachable ? "ok" : "danger";
  return (
    <article className={`fleet-card ${focused ? "focused" : ""}`} ref={cardRef}>
      <header className="fleet-card-head">
        <div className="fleet-card-id">
          <Server size={16} />
          <strong>{profile.label}</strong>
          {profile.role && <span className={`fleet-role tone-${roleTone(profile.role)}`}>{profile.role}</span>}
        </div>
        <span className={`fleet-status ${statusTone}`}><span className="fleet-dot" />{statusLabel}</span>
      </header>
      <dl className="fleet-meta">
        <div><dt>Host</dt><dd>{profile.host}</dd></div>
        <div><dt>Transport</dt><dd>{profile.transport}{profile.ssh_target ? ` · ${profile.ssh_target}` : ""}</dd></div>
        <div><dt>Hardware</dt><dd>{profile.hardware || "—"}{profile.gpus ? ` · ${profile.gpus} GPU` : ""}</dd></div>
        <div><dt>Ports</dt><dd>gui {profile.port} · engine {profile.engine_port}</dd></div>
        {probe?.reachable && <div><dt>vLLM</dt><dd>{probe.version ?? "running"}{probe.latency_ms != null ? ` · ${probe.latency_ms} ms` : ""}</dd></div>}
        {probe?.reachable && <div><dt>Served</dt><dd>{probe.models.length ? probe.models.join(", ") : "no models loaded"}</dd></div>}
        {profile.notes && <div><dt>Notes</dt><dd>{profile.notes}</dd></div>}
        {probe && !probe.reachable && probe.error && <div><dt>Probe</dt><dd className="fleet-err">{probe.error}</dd></div>}
      </dl>
      {fleet && (fleet.gpus.length > 0 || fleet.engines.length > 0) && (() => {
        const totalVram = fleet.gpus.reduce((n, g) => n + (parseInt(g.memory_total_mib || "0", 10) || 0), 0);
        const sg = (name: string) => name.replace(/^NVIDIA\s+/i, "").replace(/\s+(GPU|Graphics)$/i, "");
        const sv = (v: string) => v.replace(/^(\d+\.\d+\.\d+).*/, "$1");
        return (
          <div className="fleet-live">
            {fleet.gpus.length > 0 && (
              <div className="fleet-live-sect">
                <div className="fleet-live-t"><Cpu size={12} /> {fleet.gpus.length}× {sg(fleet.gpus[0].name)}
                  {totalVram > 0 && <span className="fleet-live-vram">{Math.round(totalVram / 1024)} GB</span>}
                  {fleet.interconnect && <span className="fleet-live-ic"><Link2 size={10} /> {fleet.interconnect}</span>}
                </div>
                <div className="fleet-live-bars">
                  {fleet.gpus.map((g, i) => {
                    const u = Math.max(0, Math.min(100, parseInt(g.utilization || "0", 10) || 0));
                    return <div key={i} className="fleet-live-bar" title={`GPU ${i} · ${sg(g.name)} · ${Math.round((parseInt(g.memory_total_mib || "0", 10) || 0) / 1024)} GB · ${u}% util`}>
                      <div className="fleet-live-fill" style={{ width: `${Math.max(u, 2)}%` }} /><span>{u}%</span></div>;
                  })}
                </div>
              </div>
            )}
            {fleet.engines.length > 0 && (
              <div className="fleet-live-sect">
                <div className="fleet-live-t"><Boxes size={11} /> {fleet.engines.length} container{fleet.engines.length > 1 ? "s" : ""}
                  {fleet.active_patches > 0 && <span className="fleet-live-patches"><ShieldCheck size={10} /> {fleet.active_patches} patches</span>}
                  {fleet.vllm_version && <span className="fleet-live-ver">vLLM {sv(fleet.vllm_version)}</span>}
                </div>
                {fleet.engines.slice(0, 4).map((e, i) => (
                  <div key={i} className="fleet-live-eng" title={`${e.container ?? "container"}${e.port ? " · :" + e.port : ""} · ${e.reachable ? "reachable" : "unreachable"}`}>
                    <span className={`fleet-live-dot ${e.reachable ? "up" : "down"}`} />
                    <code className="fleet-live-cname">{e.container ?? "—"}</code>
                    {e.port && <span className="fleet-live-port">:{e.port}</span>}
                    {e.reachable && e.version && <span className="fleet-live-evr">{sv(e.version)}</span>}
                    {e.patches > 0 && <span className="fleet-live-ep"><ShieldCheck size={9} /> {e.patches}</span>}
                    {e.models[0] && <span className="fleet-live-emodel" title={e.models.join(", ")}>{e.models[0].split("/").pop()}</span>}
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })()}
      {reliability && reliability.checks > 1 && (
        <div className={`fleet-rel ${reliability.state}`} title={`${reliability.checks} reachability checks · breaker ${reliability.state}`}>
          <span className="fleet-rel-up">{reliability.uptime_pct}% up</span>
          <RelSpark samples={reliability.samples} />
          {reliability.state === "open" && <span className="fleet-rel-state">cooling down</span>}
          {reliability.state === "half_open" && <span className="fleet-rel-state">recovering</span>}
        </div>
      )}
      {isSsh && ssh && (
        <dl className="fleet-meta fleet-ssh-meta">
          <div><dt>SSH</dt><dd className={ssh.ssh_ok ? "fleet-ok" : "fleet-err"}>{ssh.ssh_ok ? `auth ok${ssh.latency_ms != null ? ` · ${ssh.latency_ms} ms` : ""}` : (ssh.error || "failed")}</dd></div>
          {ssh.ssh_ok && <div><dt>SFTP</dt><dd className={ssh.sftp_ok ? "fleet-ok" : "fleet-err"}>{ssh.sftp_ok ? "available" : "unavailable"}</dd></div>}
          {ssh.uname && <div><dt>Remote</dt><dd>{ssh.uname}</dd></div>}
          {!ssh.available && <div><dt>SSH</dt><dd className="fleet-err">paramiko not installed (gui-remote extra)</dd></div>}
        </dl>
      )}
      <div className="fleet-caps">
        <CapChip on={!!probe?.reachable} label="engine" />
        <CapChip on={profile.transport === "ssh"} label="ssh tunnel" />
        {isSsh && <CapChip on={!!ssh?.ssh_ok} label={`ssh ${profile.ssh_auth}`} />}
        {isSsh && ssh?.ssh_ok && <CapChip on={!!ssh?.sftp_ok} label="sftp" />}
        {profile.has_ssh_password && <span className="cap-chip neutral"><KeyRound size={11} />pw stored</span>}
        {profile.gpus > 0 && <span className="cap-chip neutral"><Server size={11} />{profile.gpus} GPU</span>}
        {checkedAt && <span className="fleet-checked">checked {checkedAt}</span>}
      </div>
      {profile.tags.length > 0 && (
        <div className="fleet-tags">{profile.tags.map((tag) => <span key={tag} className="fleet-tag">{tag}</span>)}</div>
      )}
      <code className="fleet-tunnel-line">{tunnelCommand(profile)}</code>
      <div className="fleet-connect">
        <button className="primary-action" onClick={() => onChat(profile)} title={`Open the chat against ${profile.host}:${profile.engine_port}`}>
          <MessageSquare size={14} /> Chat with engine
        </button>
        <button
          className="ghost-button"
          disabled={connecting}
          onClick={async () => { setConnecting(true); try { setDaemonOk(await onAddServer(profile)); } finally { setConnecting(false); } }}
          title={daemonOk === false
            ? "No SNDR daemon reachable here — check the daemon port (8765), or set it up as a node below."
            : `Connect the GUI to the SNDR daemon at http://${profile.host}:${profile.port || 8765}`}>
          {connecting ? <Loader2 size={14} className="spin" /> : <PlugZap size={14} />} {connecting ? "Connecting…" : "Connect daemon"}
        </button>
        {onContainers && (
          <button className="ghost-button" onClick={() => onContainers(profile.id)}
            title={`Manage the containers on ${profile.host} in the Containers section`}>
            <Boxes size={14} /> Containers
          </button>
        )}
        {onHardware && (
          <button className="ghost-button" onClick={() => onHardware(profile.id)}
            title={`View live GPU & hardware telemetry for ${profile.host}`}>
            <Cpu size={14} /> GPU
          </button>
        )}
        {isSsh && (
          <button
            className={`ghost-button ${nodeForm ? "active" : ""}`}
            onClick={() => setNodeForm((v) => !v)}
            title="One-click: install (or reinstall) the SNDR management daemon on this host over SSH, so the GUI can switch to its native view.">
            <Boxes size={14} /> {nodeForm ? "Hide setup" : daemonOk === true ? "Reinstall node" : "Set up as node →"}
          </button>
        )}
      </div>

      {nodeForm && (
        <div className="node-setup">
          <div className="node-setup-head"><Boxes size={13} /> Install SNDR daemon on this node — one click</div>
          <p className="node-setup-desc">Ships the daemon code over SSH, runs it as a sidecar of the engine (LAN-bound, auth on). Then switch the GUI's top menu to this node for native management of its catalog / patches / configs.</p>
          <div className="node-setup-row">
            <label className="param-field"><span>Admin password</span>
              <input type="password" value={nodePw} onChange={(e) => setNodePw(e.target.value)} placeholder="min 4 chars — login as 'root'" autoComplete="off" spellCheck={false} />
            </label>
            <label className="param-field"><span>Engine port</span>
              <input type="number" value={profile.engine_port || 8102} readOnly title="The node's vLLM engine port (from the card)" />
            </label>
          </div>
          {applyEnabled === false && <ApplyDisabledNote what="Installing a node over SSH" />}
          <div className="node-setup-actions">
            <button className="primary-action danger" onClick={() => void installNode()} disabled={nodePw.length < 4 || nodeBusy || applyEnabled === false}>
              {nodeBusy ? <Loader2 size={14} className="spin" /> : <Rocket size={14} />} Install node over SSH
            </button>
            <button className="ghost-button" onClick={() => setNodeForm(false)} disabled={nodeBusy}>Cancel</button>
          </div>
          {nodeResult && (
            <div className={`node-setup-result ${nodeResult.ok ? "ok" : "fail"}`}>
              <div className="node-setup-result-head">
                {nodeResult.ok ? <CheckCircle2 size={14} /> : <AlertTriangle size={14} />}
                <strong>{nodeResult.ok ? `Node ready on :${nodeResult.port} — switch to it from the top menu (login: root)` : "Setup failed"}</strong>
                {nodeResult.error && !/apply is disabled/i.test(nodeResult.error) && <span className="node-setup-err">{nodeResult.error}</span>}
              </div>
              {nodeResult.error && /apply is disabled/i.test(nodeResult.error) && <ApplyDisabledNote what="Installing a node over SSH" />}
              <ol className="node-setup-steps">
                {nodeResult.steps.map((s, i) => (
                  <li key={i} className={s.rc === 0 ? "ok" : "fail"}>
                    <code>{s.rc === 0 ? "✓" : "✗"} {s.cmd}</code>
                    {s.output && <pre>{s.output}</pre>}
                  </li>
                ))}
              </ol>
            </div>
          )}
        </div>
      )}

      <div className="fleet-checks">
        <button className="ghost-button" onClick={() => void check()} disabled={busy}>
          <Activity size={14} /> {busy ? "Probing…" : "Probe engine"}
        </button>
        {isSsh && (
          <button className="ghost-button" onClick={() => void checkSsh()} disabled={sshBusy} title="Test SSH auth + SFTP">
            <Network size={14} /> {sshBusy ? "Checking…" : "SSH check"}
          </button>
        )}
      </div>
      {isSsh && (
        <div className="fleet-checks">
          <button className="ghost-button fleet-discover" onClick={() => void discover()} disabled={discoBusy} title="SSH in and auto-find vLLM containers, ports, models and GPUs — sets the engine port for you">
            {discoBusy ? <Loader2 size={14} className="spin" /> : <Sparkles size={14} />} {discoBusy ? "Discovering…" : "Discover"}
          </button>
          <button className="ghost-button" onClick={() => void fetchKey()} disabled={keyBusy} title="Read the engine's VLLM_API_KEY off the host over SSH and store it on this profile">
            <KeyRound size={14} /> {keyBusy ? "Fetching…" : "Fetch key"}
          </button>
          <button className="ghost-button" onClick={() => onTerminal(profile)} title="Open an SSH terminal to this host (requires the daemon started with SNDR_ENABLE_APPLY=1)">
            <SquareTerminal size={14} /> Terminal
          </button>
        </div>
      )}
      {sndr && sndr.ok && (
        <div className="fleet-sndr">
          <span className="fleet-sndr-lbl"><Boxes size={12} /> sndr_core on host</span>
          <span className="fleet-sndr-chip" title="Genesis patcher (sndr_core) version">patcher {sndr.sndr_version || "?"}</span>
          <span className="fleet-sndr-chip">vLLM {sndr.vllm_version || "?"}</span>
          {sndr.configs != null && <span className="fleet-sndr-chip">{sndr.configs} configs</span>}
          {sndr.patches != null && <span className="fleet-sndr-chip">{sndr.patches} patches in registry</span>}
        </div>
      )}
      {disco && (
        <div className="fleet-disco">
          {disco.engines.length > 0 ? (
            <>
              <span className="fleet-disco-label"><Sparkles size={12} /> Discovered on host</span>
              {disco.engines.map((e) => (
                <div key={e.container} className="fleet-engine-block">
                  <button className={`fleet-engine ${e.host_port === profile.engine_port ? "active" : ""}`} onClick={() => e.host_port && void applyEnginePort(e.host_port)} title={e.host_port === profile.engine_port ? "Active engine port" : "Use this port"}>
                    <span className={`fleet-engine-dot ${e.reachable ? "ok" : "down"}`} />
                    <code className="fleet-engine-port">:{e.host_port ?? "?"}</code>
                    <span className="fleet-engine-name">{e.container}</span>
                    <span className="fleet-engine-meta">{e.version ? `v${e.version}` : e.status}</span>
                  </button>
                  {e.models && e.models.length > 0 && (
                    <div className="fleet-models">{e.models.map((m) => <span key={m} className="fleet-model" title={m}><Box size={10} />{m.split("/").pop()}</span>)}</div>
                  )}
                  {e.genesis_flags && e.genesis_flags.length > 0 && (
                    <div className="fleet-patches">
                      <span className="fleet-patches-lbl"><ShieldCheck size={11} /> {e.genesis_flags.length} active patches</span>
                      {e.genesis_flags.slice(0, 16).map((f) => <span key={f} className="fleet-patch">{f.replace("GENESIS_ENABLE_", "")}</span>)}
                      {e.genesis_flags.length > 16 && <span className="fleet-patch more">+{e.genesis_flags.length - 16}</span>}
                    </div>
                  )}
                </div>
              ))}
            </>
          ) : <span className="fleet-disco-label muted">{disco.error || "No vLLM containers found"}</span>}
          {disco.gpus.length > 0 && <div className="fleet-gpus">{disco.gpus.map((g, i) => <span key={i} className="fleet-gpu"><Cpu size={11} />{g.name} · {Math.round(Number(g.memory_total_mib) / 1024)}GB{g.arch ? ` · ${g.arch}` : ""}{g.utilization != null ? ` · ${g.utilization}%` : ""}</span>)}</div>}
          {disco.interconnect && <span className="fleet-interconnect"><Link2 size={11} /> {disco.interconnect.has_nvlink ? "NVLink" : disco.interconnect.worst_link} — {disco.interconnect.note}</span>}
          {disco.arch_advice && disco.arch_advice.recommendations.length > 0 && (
            <div className="fleet-advice">
              <span className="fleet-disco-label"><ShieldCheck size={12} /> Arch-aware flags ({disco.arch_advice.arch})</span>
              {disco.arch_advice.recommendations.map((rec, i) => (
                <span key={i} className={`fleet-rec ${rec.level}`}>{rec.level === "ok" ? <CheckCircle2 size={11} /> : <CircleAlert size={11} />} {rec.text}</span>
              ))}
            </div>
          )}
        </div>
      )}
      <footer className="fleet-card-actions">
        <CopyButton value={tunnelCommand(profile)} label="tunnel command" />
        <span className="fleet-actions-spacer" />
        <button className="icon-only" onClick={() => onEdit(profile)} aria-label={`Edit ${profile.label}`}><Pencil size={14} /></button>
        <button className="icon-only danger" onClick={() => onDelete(profile.id)} aria-label={`Delete ${profile.label}`}><Trash2 size={14} /></button>
      </footer>
    </article>
  );
}

// The daemon host this UI talks to — live inventory, no probe needed.
function ThisHostCard({ inventory, environment, apiBase }: { inventory: HostInventory | null; environment: EnvironmentReport | null; apiBase: string }) {
  const gpuOk = !!(inventory?.nvidia.installed && inventory.nvidia.n_gpus > 0);
  const dockerOk = !!(inventory?.docker.installed && inventory.docker.daemon_running);
  const nvRuntime = !!inventory?.docker.nvidia_runtime_present;
  const vllmOk = !!inventory?.vllm.installed;
  const vram = inventory?.nvidia.gpu_total_vram_mib ?? [];
  const totalVram = totalVramGiB(vram);
  const perGpu = vram.length ? Math.round((vram[0] ?? 0) / 1024) : 0;
  return (
    <article className="fleet-card this-host">
      <header className="fleet-card-head">
        <div className="fleet-card-id">
          <HardDrive size={16} />
          <strong>This host <em className="host-tag">daemon</em></strong>
        </div>
        <span className="fleet-status ok"><span className="fleet-dot" />connected</span>
      </header>
      <dl className="fleet-meta">
        <div><dt>API base</dt><dd>{apiBase}</dd></div>
        <div><dt>OS</dt><dd>{inventory ? `${inventory.os.distro || inventory.os.system} ${inventory.os.arch}` : "…"}</dd></div>
        <div><dt>Python</dt><dd>{inventory ? `${inventory.python.version} · ${inventory.python.venv_active ? "venv" : "system"}` : "…"}</dd></div>
        <div><dt>Docker</dt><dd>{inventory ? (dockerOk ? `running ${inventory.docker.server_version ?? ""}`.trim() : inventory.docker.installed ? "stopped" : "missing") : "…"}</dd></div>
        <div><dt>GPU</dt><dd>{inventory ? (gpuOk ? `${inventory.nvidia.n_gpus}× ${inventory.nvidia.gpu_names[0] ?? "GPU"}` : "none detected") : "…"}</dd></div>
        {gpuOk && <div><dt>Driver / CUDA</dt><dd>{inventory?.nvidia.driver_version ?? "—"}{inventory?.nvidia.cuda_version ? ` · CUDA ${inventory.nvidia.cuda_version}` : ""}</dd></div>}
        {gpuOk && totalVram > 0 && <div><dt>VRAM</dt><dd>{totalVram} GiB{vram.length > 1 ? ` · ${perGpu} GiB/GPU` : ""}</dd></div>}
        <div><dt>vLLM</dt><dd>{inventory ? (vllmOk ? inventory.vllm.version ?? "installed" : "not installed") : "…"}</dd></div>
        <div><dt>SNDR Core</dt><dd>{environment ? `v${environment.sndr_core_version}` : "…"}</dd></div>
      </dl>
      <div className="fleet-caps">
        <CapChip on={gpuOk} label="GPU" />
        <CapChip on={dockerOk} label="Docker" />
        <CapChip on={nvRuntime} label="NVIDIA runtime" />
        <CapChip on={vllmOk} label="Engine" />
      </div>
    </article>
  );
}

// Detailed inventory grid for the Inventory tab.
// HostInventoryPanel + DependencyStackPanel extracted to ./sections/environment.

// Illustrated empty state — icon badge + title + guidance + optional recovery
// action. Used on primary content areas in place of bare muted "No X" text.
function EmptyState({ icon, title, message, action }: {
  icon?: ReactNode;
  title: string;
  message?: ReactNode;
  action?: { label: string; onClick: () => void; icon?: ReactNode };
}) {
  return (
    <div className="empty-state" role="status">
      {icon && <span className="empty-state-icon">{icon}</span>}
      <strong>{title}</strong>
      {message && <span className="empty-state-msg">{message}</span>}
      {action && (
        <button className="ghost-button" onClick={action.onClick}>
          {action.icon}{action.label}
        </button>
      )}
    </div>
  );
}

// Project & catalog snapshot — fills the row beside the dependency stack with
// the most useful project parameters: catalog counts, annotation coverage,
// capability readiness and the workload/lifecycle distribution.
function ProjectCatalogPanel({ overview, environment }: { overview: ProductOverview | null; environment: EnvironmentReport | null }) {
  if (!overview) return <SkeletonMetrics count={4} />;
  const catalog = overview.catalog;
  const features = overview.capabilities.features ?? [];
  const capsReady = features.filter((feature) => feature.status === "available").length;
  const tiles: Array<[string, number]> = [
    ["Presets", catalog.presets_count],
    ["Models", catalog.models_count],
    ["Profiles", catalog.profiles_count],
    ["Hardware", catalog.hardware_count]
  ];
  const annotated = catalog.presets_count ? Math.round((catalog.preset_cards_count / catalog.presets_count) * 100) : 0;
  const lifecycle = Object.entries(catalog.status_counts || {});
  const workloads = Object.entries(catalog.workload_counts || {}).slice(0, 6);
  const families = Object.entries(catalog.family_counts || {}).slice(0, 8);
  return (
    <div className="project-catalog">
      <div className="catalog-tiles">
        {tiles.map(([label, value]) => (
          <div className="catalog-tile" key={label}>
            <strong>{value}</strong>
            <span>{label}</span>
          </div>
        ))}
      </div>
      <InfoRows rows={[
        ["Engine target", `${environment?.engine_name ?? "vLLM"} ${environment?.engine_version ?? "(not installed)"}`.trim()],
        ["Annotated presets", `${catalog.preset_cards_count}/${catalog.presets_count} · ${annotated}%`],
        ["Capabilities ready", `${capsReady}/${features.length}`],
        ["Load errors", String(catalog.preset_load_error_count)]
      ]} />
      {lifecycle.length > 0 && (
        <div className="catalog-dist">
          <span className="catalog-dist-label">Lifecycle</span>
          <div className="fleet-caps">
            {lifecycle.map(([key, value]) => <span className="cap-chip neutral" key={key}>{key} · {value}</span>)}
          </div>
        </div>
      )}
      {workloads.length > 0 && (
        <div className="catalog-dist">
          <span className="catalog-dist-label">Workloads</span>
          <div className="fleet-caps">
            {workloads.map(([key, value]) => <span className="cap-chip neutral" key={key}>{key.replace(/_/g, " ")} · {value}</span>)}
          </div>
        </div>
      )}
      {families.length > 0 && (
        <div className="catalog-dist">
          <span className="catalog-dist-label">Patch families</span>
          <div className="fleet-caps">
            {families.map(([key, value]) => <span className="cap-chip neutral" key={key}>{key} · {value}</span>)}
          </div>
        </div>
      )}
    </div>
  );
}

// Add/edit modal for a host profile.
function HostFormModal({
  initial,
  onClose,
  onSaved
}: {
  initial: HostProfile | null;
  onClose: () => void;
  onSaved: () => void;
}) {
  const blank = { label: "", host: "", ssh_target: "", port: 8765, engine_port: 8000, api_key: "", ssh_user: "", ssh_auth: "agent", ssh_key_path: "", ssh_port: 22, ssh_password: "", role: "", hardware: "", gpus: 0, notes: "", tags: "" };
  const [form, setForm] = useState(initial
    ? { label: initial.label, host: initial.host, ssh_target: initial.ssh_target, port: initial.port, engine_port: initial.engine_port, api_key: "", ssh_user: initial.ssh_user, ssh_auth: initial.ssh_auth || "agent", ssh_key_path: initial.ssh_key_path, ssh_port: initial.ssh_port || 22, ssh_password: "", role: initial.role, hardware: initial.hardware, gpus: initial.gpus, notes: initial.notes, tags: initial.tags.join(", ") }
    : blank);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const dialogRef = useRef<HTMLElement>(null);
  useDialogFocus(dialogRef);
  useEscapeKey(onClose);
  const set = (patch: Partial<typeof form>) => setForm((prev) => ({ ...prev, ...patch }));
  async function save() {
    setBusy(true);
    setError(null);
    try {
      const saved = await api.hostUpsert({
        ...(initial ? { id: initial.id } : {}),
        label: form.label,
        host: form.host,
        ssh_target: form.ssh_target,
        transport: form.ssh_target || form.ssh_user ? "ssh" : "local",
        port: Number(form.port),
        engine_port: Number(form.engine_port),
        // Only send the engine key when the operator actually typed one — a
        // blank field means "keep the stored key" (it's never pre-filled, since
        // the daemon never returns it), so editing a host can't silently wipe it.
        ...(form.api_key ? { api_key: form.api_key } : {}),
        ssh_user: form.ssh_user,
        ssh_auth: form.ssh_auth,
        ssh_key_path: form.ssh_key_path,
        ssh_port: Number(form.ssh_port) || 22,
        role: form.role,
        hardware: form.hardware,
        gpus: Number(form.gpus),
        notes: form.notes,
        tags: form.tags.split(",").map((tag) => tag.trim()).filter(Boolean)
      });
      // A typed SSH password is persisted (encrypted) via the check endpoint,
      // never through the plaintext profile.
      if (form.ssh_auth === "password" && form.ssh_password) {
        try { await api.sshCheck({ host: saved.host, host_id: saved.id, user: form.ssh_user, auth_method: "password", password: form.ssh_password, ssh_port: Number(form.ssh_port) || 22 }); } catch { /* surfaced on the card's SSH check */ }
      }
      onSaved();
      onClose();
      toast(initial ? `Host updated: ${form.label}` : `Host added: ${form.label}`, "success");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      toast("Failed to save host", "error");
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="dialog-backdrop" role="presentation" onClick={closeOnBackdrop(onClose)}>
      <section ref={dialogRef} className="host-modal" role="dialog" aria-modal="true">
        <div className="module-card-title">
          <Server size={18} />
          <h2>{initial ? `Edit ${initial.label}` : "Add host profile"}</h2>
        </div>
        <div className="host-form-grid">
          <label className="param-field"><span>Label</span><input value={form.label} onChange={(e) => set({ label: e.target.value })} placeholder="Prod A5000" /></label>
          <label className="param-field"><span>Host / IP</span><input value={form.host} onChange={(e) => set({ host: e.target.value })} placeholder="192.168.1.10" /></label>
          <label className="param-field"><span>SSH target</span><input value={form.ssh_target} onChange={(e) => set({ ssh_target: e.target.value })} placeholder="user@192.168.1.10" /></label>
          <label className="param-field"><span>Role</span>
            <select value={form.role} onChange={(e) => set({ role: e.target.value })}>
              <option value="">— none —</option>
              {HOST_ROLES.map((role) => <option key={role} value={role}>{role}</option>)}
            </select>
          </label>
          <label className="param-field"><span>GUI port</span><input type="number" value={form.port} onChange={(e) => set({ port: Number(e.target.value) })} /></label>
          <label className="param-field"><span>Engine port</span><input type="number" value={form.engine_port} onChange={(e) => set({ engine_port: Number(e.target.value) })} /></label>
          <label className="param-field"><span>Hardware</span><input value={form.hardware} onChange={(e) => set({ hardware: e.target.value })} placeholder="2× A5000 24GB" /></label>
          <label className="param-field"><span>GPUs</span><input type="number" value={form.gpus} onChange={(e) => set({ gpus: Number(e.target.value) })} /></label>
        </div>
        <label className="param-field"><span>Engine API key{initial?.has_api_key ? " (stored — leave blank to keep)" : " (optional)"}</span><input type="password" value={form.api_key} onChange={(e) => set({ api_key: e.target.value })} placeholder={initial?.has_api_key ? "•••••• stored — type to replace" : "if the engine needs one — e.g. genesis-local"} autoComplete="off" spellCheck={false} /></label>
        <div className="host-form-grid host-ssh-grid">
          <label className="param-field"><span>SSH user</span><input value={form.ssh_user} onChange={(e) => set({ ssh_user: e.target.value })} placeholder="sander" /></label>
          <label className="param-field"><span>SSH port</span><input type="number" value={form.ssh_port} onChange={(e) => set({ ssh_port: Number(e.target.value) || 22 })} /></label>
          <label className="param-field"><span>SSH auth</span>
            <select value={form.ssh_auth} onChange={(e) => set({ ssh_auth: e.target.value })}>
              <option value="agent">ssh-agent / default keys</option>
              <option value="key">private key file</option>
              <option value="password">password</option>
            </select>
          </label>
          {form.ssh_auth === "key" && <label className="param-field"><span>Private key path</span><input value={form.ssh_key_path} onChange={(e) => set({ ssh_key_path: e.target.value })} placeholder="~/.ssh/id_ed25519" spellCheck={false} /></label>}
          {form.ssh_auth === "password" && <label className="param-field"><span>SSH password {initial ? "(stored, blank = keep)" : ""}</span><input type="password" value={form.ssh_password} onChange={(e) => set({ ssh_password: e.target.value })} placeholder="encrypted at rest" autoComplete="off" spellCheck={false} /></label>}
        </div>
        <label className="param-field"><span>Tags (comma-separated)</span><input value={form.tags} onChange={(e) => set({ tags: e.target.value })} placeholder="27b, tq-k8v4" /></label>
        <label className="param-field"><span>Notes</span><input value={form.notes} onChange={(e) => set({ notes: e.target.value })} placeholder="MTP K=3 / Wave 8" /></label>
        {error && <div className="inline-error"><AlertCircle size={15} /> {error}</div>}
        <div className="host-modal-actions">
          <button className="ghost-button" onClick={onClose}>Cancel</button>
          <button className="primary-action" onClick={() => void save()} disabled={busy || !(form.label || form.host)}>
            <Server size={15} /> {busy ? "Saving…" : initial ? "Save changes" : "Add profile"}
          </button>
        </div>
      </section>
    </div>
  );
}

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
function PresetRecommendPanel({
  hardwareOptions,
  workloadCounts,
  onSelect
}: {
  hardwareOptions: string[];
  workloadCounts: Record<string, number>;
  onSelect: (id: string) => void;
}) {
  const [workload, setWorkload] = useState("free_chat");
  const [hardware, setHardware] = useState(hardwareOptions[0] ?? defaultRecommend.hardware);
  const [concurrency, setConcurrency] = useState(8);
  const [top, setTop] = useState(5);
  const [result, setResult] = useState<PresetRecommendResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setLoading(true);
    setError(null);
    try {
      setResult(await api.recommendPresets({ workload, hardware, concurrency, top }));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }
  // eslint-disable-next-line react-hooks/exhaustive-deps -- one-shot load on mount
  useEffect(() => { void run(); }, []);

  return (
    <div className="preset-recommend">
      <div className="rec-workloads">
        {workloadChoices.map((choice) => (
          <button
            key={choice.id}
            className={workload === choice.id ? "active" : ""}
            onClick={() => setWorkload(choice.id)}
          >
            {choice.label}<small>{workloadCounts[choice.id] ?? 0}</small>
          </button>
        ))}
      </div>
      <div className="rec-controls">
        <label className="param-field"><span>Target hardware</span>
          <select value={hardware} onChange={(event) => setHardware(event.target.value)}>
            {hardwareOptions.map((id) => <option key={id} value={id}>{id}</option>)}
          </select>
        </label>
        <label className="param-field"><span>Concurrency</span>
          <input type="number" min={1} value={concurrency} onChange={(event) => setConcurrency(Number(event.target.value))} />
        </label>
        <label className="param-field"><span>Top N</span>
          <input type="number" min={1} max={20} value={top} onChange={(event) => setTop(Number(event.target.value))} />
        </label>
        <button className="primary-action" onClick={() => void run()} disabled={loading}>
          <Rocket size={15} /> {loading ? "Ranking…" : "Recommend"}
        </button>
      </div>
      {error && <div className="config-plan-error"><AlertCircle size={15} /><span>{error}</span></div>}
      {result && (
        <>
          <div className="rec-summary">
            <span className={result.total_matches > 0 ? "fleet-status ok" : "fleet-status danger"}>
              <span className="fleet-dot" />{result.total_matches} of {result.total_candidates} candidates match
            </span>
          </div>
          {result.results.length > 0 ? (
            <div className="patch-table-scroll">
              <table className="module-table rec-table">
                <thead><tr><th>#</th><th>Preset</th><th>Model</th><th>Hardware</th><th>Profile</th><th>Baseline</th><th></th></tr></thead>
                <tbody>
                  {result.results.map((rec) => (
                    <tr key={rec.id}>
                      <td><span className="rec-rank">{rec.rank}</span></td>
                      <td><strong>{rec.id}</strong>{asText(rec.card?.status, "") && <small className="rec-status">{asText(rec.card?.status, "")}</small>}</td>
                      <td>{rec.model}</td>
                      <td>{rec.hardware}</td>
                      <td>{rec.profile ?? "—"}</td>
                      <td><PresetBaselineCell card={rec.card} /></td>
                      <td><button className="ghost-button" onClick={() => onSelect(rec.id)}><FileText size={13} /> Inspect</button></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <EmptyState
              icon={<Database size={22} />}
              title="No presets match"
              message="No preset fits this workload on the selected hardware. Try a different rig or workload above, or relax the concurrency target."
            />
          )}
        </>
      )}
    </div>
  );
}

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

const _OP_GROUP_ICON: Record<string, ReactNode> = {
  "Diagnostics": <Stethoscope size={18} />,
  "Registry audits": <ShieldCheck size={18} />,
  "Config & catalog": <Database size={18} />,
  "Proof & release": <PackageCheck size={18} />
};

// Project Operations console — surfaces sndr_core's canonical CLI maintenance
// workflows as one-click, live-monitored jobs. Commands are server-defined;
// the client only sends an operation id.
function OperationsConsole({ onMonitor }: { onMonitor: (id: string) => void }) {
  const [data, setData] = useState<OperationsResult | null>(null);
  const [busy, setBusy] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api.operations()
      .then((result) => { if (!cancelled) setData(result); })
      .catch((err) => { if (!cancelled) setError(err instanceof Error ? err.message : String(err)); });
    return () => { cancelled = true; };
  }, []);

  async function run(opId: string) {
    setBusy(opId);
    setError(null);
    try {
      const job = await api.operationRun(opId);
      onMonitor(job.job_id);
      toast(`Operation started: ${opId}`, "success");
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      toast(`Operation failed: ${msg}`, "error");
    } finally {
      setBusy("");
    }
  }

  if (!data) return <ModuleGrid><ModuleCard title="Operations" icon={<Terminal size={18} />} wide>{error ? <p className="muted">{error}</p> : <SkeletonCards count={4} />}</ModuleCard></ModuleGrid>;

  const groups: string[] = [];
  data.operations.forEach((op) => { if (!groups.includes(op.group)) groups.push(op.group); });
  const applyOn = data.apply_enabled;

  return (
    <>
      <div className={`ops-banner ${applyOn ? "live" : "readonly"}`}>
        {applyOn ? <Activity size={16} /> : <ShieldCheck size={16} />}
        <div>
          <strong>{applyOn ? "Apply enabled — operations run live on this host" : "Read-only daemon — operations return a dry-run"}</strong>
          <span>{applyOn
            ? "Each run executes the sndr_core CLI as a background job; watch it live in the monitor."
            : "Commands are mirrored so you can copy them. Start the daemon with --enable-apply to run them here."}</span>
        </div>
      </div>
      {error && <div className="inline-error"><AlertCircle size={15} /> {error}</div>}
      <ModuleGrid className="stretch-row">
        {groups.map((group) => (
          <ModuleCard key={group} title={group} icon={_OP_GROUP_ICON[group] ?? <Terminal size={18} />} desc={`${data.operations.filter((op) => op.group === group).length} operations`}>
            <div className="ops-list">
              {data.operations.filter((op) => op.group === group).map((op) => (
                <div className="ops-row" key={op.id}>
                  <div className="ops-row-text">
                    <strong>{op.label}</strong>
                    <small>{op.description}</small>
                    <code>{op.command.replace(/^\S*python\S*\s+-m\s+/, "")}</code>
                  </div>
                  <div className="ops-row-action">
                    <span className="ops-est">{op.estimate}</span>
                    <button className="primary-action" onClick={() => void run(op.id)} disabled={busy === op.id}>
                      <Play size={14} /> {busy === op.id ? "Starting…" : "Run"}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </ModuleCard>
        ))}
      </ModuleGrid>
    </>
  );
}

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

function PresetQuickPanel({
  selectedPreset,
  record,
  card,
  composed,
  onOpenCard,
  onEdit,
  onPolicy,
  onLaunch
}: {
  selectedPreset: string;
  record: PresetRecord | null;
  card: Record<string, unknown>;
  composed: Record<string, unknown>;
  onOpenCard: () => void;
  onEdit: () => void;
  onPolicy: () => void;
  onLaunch: () => void;
}) {
  if (!selectedPreset) {
    return (
      <section className="preset-quick empty">
        <div className="preset-quick-empty">
          <MousePointerClick size={26} />
          <strong>Select a preset</strong>
          <p>Click any row in the catalog to see its runtime, evidence and editing actions here.</p>
        </div>
      </section>
    );
  }
  const status = record?.has_card ? asText(card.status, "available") : "missing";
  const workloads = asStringArray(card.workload_allow);
  return (
    <section className="preset-quick">
      <div className="preset-quick-head">
        <div>
          <span className="preset-quick-kicker">Selected preset</span>
          <strong>{selectedPreset}</strong>
        </div>
        <StatusBadge status={status} />
      </div>
      <p className="preset-quick-title">{asText(card.title, "Unannotated preset — no card metadata yet.")}</p>
      <InfoRows
        rows={[
          ["Model", asText(record?.model ?? composed.model, "-")],
          ["Hardware", asText(record?.hardware ?? composed.hardware, "-")],
          ["Profile", asText(record?.profile ?? composed.profile, "-")],
          ["Mode", asText(card.mode, "-")],
          ["Max context", formatTokens(asNumber(composed.max_model_len))],
          ["KV cache", asText(composed.kv_cache_dtype, "-")],
          ["Spec decode", `${asText(composed.spec_decode_method, "-")} / K=${asText(composed.spec_decode_K, "-")}`],
          ["Patches", asText(composed.enabled_patches_count, "-")],
          ["Fallback", asText(card.fallback_preset, "none")]
        ]}
      />
      {workloads.length > 0 && (
        <div className="preset-quick-workloads">
          <span className="preset-quick-kicker">Allowed workloads</span>
          <div className="chip-row">
            {workloads.map((item) => <span className="chip" key={item}>{item}</span>)}
          </div>
        </div>
      )}
      <div className="preset-quick-actions">
        <button className="primary-button" onClick={onEdit}>
          <Wrench size={15} /> Edit preset
        </button>
        <button className="ghost-button" onClick={onOpenCard}>
          <FileText size={14} /> Full card
        </button>
        <button className="ghost-button" onClick={onPolicy}>
          <BarChart3 size={14} /> Policy
        </button>
        <button className="ghost-button" onClick={onLaunch}>
          <Rocket size={14} /> Launch
        </button>
      </div>
    </section>
  );
}

// Benchmark-baseline chip for the preset catalog: surfaces the measured
// reference metric (primary_metric) at the list level, so bench-proven presets
// are distinguishable from pending ones at a glance. value 0 / missing = pending.
function PresetBaselineCell({ card }: { card: Record<string, any> }) {
  const m = asRecord(card?.primary_metric);
  const value = asNumber(m.value);
  if (!m.kind && !value) return <span className="muted">—</span>;
  const tip = [asText(m.source, ""), asText(m.measured_at, "")].filter(Boolean).join(" · ");
  if (value > 0) {
    const kind = asText(m.kind, "TPS").replace(/^agg_/, "");
    return <span className="bench-chip ok" title={tip || undefined}>{value.toLocaleString()} {kind}</span>;
  }
  return <span className="bench-chip pending" title={tip || undefined}>pending</span>;
}

function PresetCatalogTable({
  presets,
  selectedPreset,
  onPreset,
  onEdit
}: {
  presets: PresetRecord[];
  selectedPreset: string;
  onPreset: (id: string) => void;
  onEdit?: (id: string) => void;
}) {
  const [statusFilter, setStatusFilter] = useState("all");
  const [benchFilter, setBenchFilter] = useState<"all" | "proven" | "pending">("all");
  const [sortKey, setSortKey] = useState<"id" | "model" | "status" | "baseline">("id");
  const [sortDir, setSortDir] = useState<1 | -1>(1);

  const statusOf = (preset: PresetRecord) =>
    preset.has_card ? asText(preset.card?.status, "available") : "missing";
  // Measured reference throughput (0 = pending) — used for the Baseline sort.
  const baselineOf = (preset: PresetRecord) => asNumber(asRecord(preset.card?.primary_metric).value);
  const statuses = Array.from(new Set(presets.map(statusOf))).sort();
  const counts = presets.reduce<Record<string, number>>((acc, preset) => {
    const key = statusOf(preset);
    acc[key] = (acc[key] ?? 0) + 1;
    return acc;
  }, {});

  const provenCount = presets.filter((preset) => baselineOf(preset) > 0).length;
  const rows = presets
    .filter((preset) => statusFilter === "all" || statusOf(preset) === statusFilter)
    .filter((preset) => benchFilter === "all" || (benchFilter === "proven") === (baselineOf(preset) > 0))
    .sort((a, b) => {
      if (sortKey === "baseline") return (baselineOf(a) - baselineOf(b)) * sortDir;
      const va = sortKey === "status" ? statusOf(a) : String(a[sortKey] ?? "");
      const vb = sortKey === "status" ? statusOf(b) : String(b[sortKey] ?? "");
      return va.localeCompare(vb) * sortDir;
    });

  const toggleSort = (key: "id" | "model" | "status" | "baseline") => {
    if (sortKey === key) setSortDir((dir) => (dir === 1 ? -1 : 1));
    // Baseline defaults to descending — operators want the fastest presets first.
    else { setSortKey(key); setSortDir(key === "baseline" ? -1 : 1); }
  };
  const caret = (key: string) => (sortKey === key ? (sortDir === 1 ? " ↑" : " ↓") : "");

  return (
    <div className="preset-catalog">
      <div className="filter-chips">
        <button className={statusFilter === "all" ? "active" : ""} onClick={() => setStatusFilter("all")}>
          All <em>{presets.length}</em>
        </button>
        {statuses.map((status) => (
          <button key={status} className={statusFilter === status ? "active" : ""} onClick={() => setStatusFilter(status)}>
            {status.replace(/_/g, " ")} <em>{counts[status]}</em>
          </button>
        ))}
        <span className="filter-chips-sep" aria-hidden="true" />
        <span className="filter-chips-label">Baseline</span>
        <button className={benchFilter === "all" ? "active" : ""} onClick={() => setBenchFilter("all")}>any <em>{presets.length}</em></button>
        <button className={benchFilter === "proven" ? "active" : ""} onClick={() => setBenchFilter("proven")}>bench-proven <em>{provenCount}</em></button>
        <button className={benchFilter === "pending" ? "active" : ""} onClick={() => setBenchFilter("pending")}>pending <em>{presets.length - provenCount}</em></button>
      </div>
      <div className="catalog-scroll">
        <table className="module-table">
          <thead>
            <tr>
              <th className="sortable" onClick={() => toggleSort("id")}>Preset{caret("id")}</th>
              <th className="sortable" onClick={() => toggleSort("model")}>Model{caret("model")}</th>
              <th>Hardware</th>
              <th>Profile</th>
              <th className="sortable" onClick={() => toggleSort("status")}>Card{caret("status")}</th>
              <th className="sortable" onClick={() => toggleSort("baseline")}>Baseline{caret("baseline")}</th>
              <th aria-label="Actions" />
            </tr>
          </thead>
          <tbody>
            {rows.map((preset) => (
              <tr
                className={`preset-catalog-row ${preset.id === selectedPreset ? "selected-row" : ""}`}
                key={preset.id}
                onClick={() => onPreset(preset.id)}
              >
                <td>
                  <button className="link-button" onClick={(event) => { event.stopPropagation(); onPreset(preset.id); }}>
                    {preset.id === selectedPreset && <ChevronRight size={13} className="row-active-caret" />}
                    {preset.id}
                  </button>
                </td>
                <td>{preset.model}</td>
                <td>{preset.hardware}</td>
                <td>{preset.profile ?? "-"}</td>
                <td><StatusBadge status={statusOf(preset)} /></td>
                <td><PresetBaselineCell card={preset.card} /></td>
                <td className="preset-row-actions">
                  {onEdit && (
                    <button
                      className="icon-button"
                      title={`Edit ${preset.id}`}
                      aria-label={`Edit ${preset.id}`}
                      onClick={(event) => { event.stopPropagation(); onEdit(preset.id); }}
                    >
                      <Wrench size={14} />
                    </button>
                  )}
                </td>
              </tr>
            ))}
            {rows.length === 0 && (() => {
              const filtered = statusFilter !== "all" || benchFilter !== "all";
              return (
              <tr><td colSpan={7}>
                <EmptyState
                  icon={<Database size={22} />}
                  title="No presets for this filter"
                  message={filtered ? "No presets match the active filters." : "The preset catalog is empty."}
                  action={filtered ? { label: "Clear filters", icon: <X size={14} />, onClick: () => { setStatusFilter("all"); setBenchFilter("all"); } } : undefined}
                />
              </td></tr>
              );
            })()}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function PatchSummaryPanel({
  summary,
  total,
  selectedCount
}: {
  summary: PatchListResult["summary"] | null;
  total: number;
  selectedCount: number;
}) {
  const lifecycleRows = Object.entries(summary?.lifecycle_counts ?? {});
  const productionRows = Object.entries(summary?.production_default_counts ?? {});
  return (
    <div className="patch-summary-grid">
      <KpiGrid
        rows={[
          ["Registry", total],
          ["Selected Plan", selectedCount || "-"],
          ["Stable", summary?.lifecycle_counts.stable ?? 0],
          ["Default Applied", summary?.production_default_counts.applied ?? 0]
        ]}
      />
      <CompactList rows={lifecycleRows.map(([key, value]) => [`lifecycle:${key}`, String(value)])} />
      <CompactList rows={productionRows.map(([key, value]) => [`default:${key}`, String(value)])} />
    </div>
  );
}

function PatchLifecycleGraph({
  summary
}: {
  summary: PatchListResult["summary"] | null;
}) {
  const lifecycle = summary?.lifecycle_counts ?? {};
  const production = summary?.production_default_counts ?? {};
  const lifecycleTotal = Object.values(lifecycle).reduce((a, b) => a + b, 0);
  const productionTotal = Object.values(production).reduce((a, b) => a + b, 0);
  const lifecycleColors: Record<string, string> = {
    stable: "var(--ok)", experimental: "var(--warn)", research: "var(--info)",
    retired: "var(--danger)", qa: "var(--accent)"
  };
  const defaultColors: Record<string, string> = {
    applied: "var(--ok)", marker: "var(--warn)", "opt-in": "var(--info)", blocked: "var(--danger)"
  };
  return (
    <div className="patch-graph-grid">
      <section>
        <strong>Lifecycle distribution</strong>
        <SegmentBar
          segments={segmentsFromCounts(lifecycle, lifecycleColors)}
          total={lifecycleTotal}
          totalLabel="patches"
        />
      </section>
      <section>
        <strong>Production default behavior</strong>
        <SegmentBar
          segments={segmentsFromCounts(production, defaultColors)}
          total={productionTotal}
          totalLabel="patches"
        />
      </section>
    </div>
  );
}

const _IMPL_MEANING: Record<string, string> = {
  full: "complete overlay — observable ON/OFF difference",
  partial: "some anchors wired; not yet fully effective",
  marker_only: "registry marker, no runtime code",
  placeholder: "reserved id, implementation pending",
  experimental: "wired but unproven; needs evidence",
  retired: "superseded/removed, kept for audit"
};

function _patchBar(counts: Record<string, number>, limit = 99): Array<[string, number, string]> {
  const max = Math.max(1, ...Object.values(counts));
  return Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([k, v]) => [k.replace(/_/g, " "), Math.round((v / max) * 100), String(v)] as [string, number, string]);
}

/** Tab-1 insight: status + family distributions + a plain-language legend. */
function PatchRegistryInsight({
  summary,
  patches
}: {
  summary: PatchListResult["summary"] | null;
  patches: PatchRow[];
}) {
  const implCounts = summary?.implementation_status_counts ?? {};
  const familyCounts = countRecord(patches.map((patch) => patch.family || "uncategorized"));
  return (
    <div className="patch-insight">
      <div className="patch-insight-grid">
        <div>
          <h5>Implementation status</h5>
          <BarList rows={_patchBar(implCounts)} />
        </div>
        <div>
          <h5>Families <em>{Object.keys(familyCounts).length}</em></h5>
          <BarList rows={_patchBar(familyCounts, 12)} />
        </div>
      </div>
      <div className="patch-legend">
        <h5>What the values mean</h5>
        <dl>
          <div><dt>Lifecycle</dt><dd><b>stable</b> safe default · <b>experimental</b> needs evidence · <b>research</b> idea-only · <b>legacy</b> older but kept · <b>retired</b> audit-only · <b>coordinator</b> orchestrates others.</dd></div>
          <div><dt>Production default</dt><dd><b>applied</b> on with real code · <b>marker</b> on but no effect · <b>opt-in</b> off by default · <b>blocked</b> not production-safe.</dd></div>
          <div><dt>Implementation</dt><dd>{Object.entries(_IMPL_MEANING).map(([k, v]) => `${k}: ${v}`).join(" · ")}.</dd></div>
        </dl>
      </div>
    </div>
  );
}

/** Tab-1 supported models — the catalog the patch family targets. */
function PatchModelSupport({ models }: { models: Array<{ id: string; title?: string }> }) {
  return (
    <div className="patch-models">
      <p className="muted">
        Patches target the catalog models below. Each patch declares its own applicability — model family,
        TurboQuant, vLLM version range — shown per-patch in the Inventory tab under <strong>Supported models</strong>.
      </p>
      <div className="chip-row">
        {models.length ? models.map((model) => (
          <span className="chip" key={model.id} title={model.title ?? model.id}>{model.id}</span>
        )) : <span className="muted">No models in the catalog.</span>}
      </div>
    </div>
  );
}

function PatchInventoryControl({ patches }: { patches: PatchRow[] }) {
  const [needle, setNeedle] = useState("");
  const [lifecycle, setLifecycle] = useState("all");
  const [productionDefault, setProductionDefault] = useState("all");
  const [groupByFamily, setGroupByFamily] = useState(true);
  const [selectedPatchId, setSelectedPatchId] = useState<string>("");
  const [patchExplain, setPatchExplain] = useState<PatchExplainResult | null>(null);
  const [patchExplainState, setPatchExplainState] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [patchExplainError, setPatchExplainError] = useState<string | null>(null);
  const [overrides, setOverrides] = useState<Record<string, { state: string; env_flag: string }>>({});
  useEffect(() => { api.patchOverrides().then((o) => setOverrides(o.overrides)).catch(() => {}); }, []);
  const lifecycleOptions = Array.from(new Set(patches.map((patch) => patch.lifecycle))).sort();
  const defaultOptions = Array.from(new Set(patches.map((patch) => patch.production_default))).sort();
  const visibleRows = patches.filter((patch) => {
    const haystack = [
      patch.patch_id,
      patch.title,
      patch.family,
      patch.tier,
      patch.lifecycle,
      patch.production_default,
      patch.env_flag,
      patch.apply_module
    ].join(" ").toLowerCase();
    return (
      (!needle.trim() || haystack.includes(needle.trim().toLowerCase())) &&
      (lifecycle === "all" || patch.lifecycle === lifecycle) &&
      (productionDefault === "all" || patch.production_default === productionDefault)
    );
  });
  const selectedPatch = visibleRows.find((patch) => patch.patch_id === selectedPatchId) ?? visibleRows[0] ?? null;
  // Set of every registered patch id — lets the explain panel turn
  // requires/conflicts into clickable links to navigable patches.
  const patchIdSet = useMemo(() => new Set(patches.map((p) => p.patch_id)), [patches]);
  const familyGroups = useMemo(() => {
    const map = new Map<string, PatchRow[]>();
    visibleRows.forEach((patch) => {
      const family = patch.family || "other";
      map.set(family, [...(map.get(family) ?? []), patch]);
    });
    return Array.from(map.entries()).sort((a, b) => b[1].length - a[1].length);
    // visibleRows is recomputed per render; intentional for ~230 rows.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [needle, lifecycle, productionDefault, patches]);

  useEffect(() => {
    if (!selectedPatch?.patch_id) {
      setPatchExplain(null);
      setPatchExplainState("idle");
      return;
    }
    let cancelled = false;
    setPatchExplainState("loading");
    setPatchExplainError(null);
    api.patchExplain(selectedPatch.patch_id)
      .then((detail) => {
        if (cancelled) return;
        setPatchExplain(detail);
        setPatchExplainState("ready");
      })
      .catch((err) => {
        if (cancelled) return;
        setPatchExplain(null);
        setPatchExplainState("error");
        setPatchExplainError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [selectedPatch?.patch_id]);

  return (
    <div className="patch-control">
      <div className="patch-control-toolbar">
        <label className="search-box">
          <Search size={15} />
          <input
            value={needle}
            onChange={(event) => setNeedle(event.target.value)}
            placeholder="Filter patch id, family, env flag"
          />
        </label>
        <select value={lifecycle} onChange={(event) => setLifecycle(event.target.value)}>
          <option value="all">All lifecycles</option>
          {lifecycleOptions.map((item) => <option key={item} value={item}>{item}</option>)}
        </select>
        <select value={productionDefault} onChange={(event) => setProductionDefault(event.target.value)}>
          <option value="all">All defaults</option>
          {defaultOptions.map((item) => <option key={item} value={item}>{item}</option>)}
        </select>
        <div className="settings-segmented">
          <button className={groupByFamily ? "active" : ""} onClick={() => setGroupByFamily(true)}>By family</button>
          <button className={!groupByFamily ? "active" : ""} onClick={() => setGroupByFamily(false)}>Flat list</button>
        </div>
        <span>{visibleRows.length} matched · {familyGroups.length} families</span>
      </div>
      <div className="patch-control-grid">
        <div className="patch-table-scroll">
          {groupByFamily ? (
            <div className="patch-family-groups">
              {familyGroups.map(([family, rows]) => (
                <PatchFamilyGroup
                  key={family}
                  family={family}
                  rows={rows}
                  selectedId={selectedPatch?.patch_id ?? ""}
                  onSelect={setSelectedPatchId}
                />
              ))}
              {familyGroups.length === 0 && (() => {
                const filtered = needle.trim() !== "" || lifecycle !== "all" || productionDefault !== "all";
                return (
                  <EmptyState
                    icon={<PackageCheck size={22} />}
                    title="No patches match"
                    message={filtered ? "No patches in the registry match the active filters." : "The patch registry is empty."}
                    action={filtered ? { label: "Clear filters", icon: <X size={14} />, onClick: () => { setNeedle(""); setLifecycle("all"); setProductionDefault("all"); } } : undefined}
                  />
                );
              })()}
            </div>
          ) : (
            <table className="module-table patch-table patch-table--flat">
              <colgroup>
                <col style={{ width: "17%" }} />
                <col style={{ width: "12%" }} />
                <col style={{ width: "11%" }} />
                <col style={{ width: "12%" }} />
                <col style={{ width: "10%" }} />
                <col style={{ width: "38%" }} />
              </colgroup>
              <thead>
                <tr>
                  <th>Patch</th>
                  <th>Lifecycle</th>
                  <th>Default</th>
                  <th>Family</th>
                  <th>Upstream</th>
                  <th>Title</th>
                </tr>
              </thead>
              <tbody>
                {visibleRows.map((patch) => (
                  <tr
                    className={`patch-flat-row ${selectedPatch?.patch_id === patch.patch_id ? "selected-row" : ""}`}
                    key={patch.patch_id}
                  >
                    <td>
                      <button className="link-button" onClick={() => setSelectedPatchId(patch.patch_id)}>
                        <strong>{patch.patch_id}</strong>
                        <small>{patch.env_flag || patch.tier}</small>
                      </button>
                    </td>
                    <td><StatusBadge status={patch.lifecycle} /></td>
                    <td>{patch.production_default}</td>
                    <td>{patch.family || "-"}</td>
                    <td>{patch.upstream_pr ? `#${patch.upstream_pr}` : "-"}</td>
                    <td>{patch.title}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
        <PatchExplainPanel
          patch={selectedPatch}
          detail={patchExplain}
          state={patchExplainState}
          error={patchExplainError}
          allPatchIds={patchIdSet}
          onSelectPatch={(id) => setSelectedPatchId(id)}
          override={overrides[selectedPatch?.patch_id ?? ""]?.state ?? "default"}
          overrideCount={Object.keys(overrides).length}
          onSetOverride={async (state) => {
            if (!selectedPatch) return;
            try {
              const res = await api.setPatchOverride(selectedPatch.patch_id, state, selectedPatch.env_flag);
              setOverrides(res.overrides);
            } catch { /* surfaced inline */ }
          }}
        />
      </div>
    </div>
  );
}

function PatchFamilyGroup({
  family,
  rows,
  selectedId,
  onSelect
}: {
  family: string;
  rows: PatchRow[];
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const applied = rows.filter((row) => row.production_default === "applied").length;
  const hasSelected = rows.some((row) => row.patch_id === selectedId);
  // Auto-open the group when a patch inside it becomes selected, but only on the
  // transition — so a manual collapse still wins while the selection stays here.
  useEffect(() => {
    if (hasSelected) setOpen(true);
  }, [hasSelected]);
  const expanded = open;
  return (
    <section className={`patch-family ${expanded ? "open" : ""}`}>
      <button className="patch-family-head" onClick={() => setOpen((value) => !value)} aria-expanded={expanded}>
        <ChevronRight className="coll-caret" size={14} />
        <strong>{family}</strong>
        <span className="patch-family-count">{rows.length}</span>
        <em>{applied} applied</em>
      </button>
      {expanded && (
        <div className="patch-family-rows">
          {rows.map((patch) => (
            <button
              key={patch.patch_id}
              className={`patch-row-mini ${patch.patch_id === selectedId ? "active" : ""}`}
              onClick={() => onSelect(patch.patch_id)}
            >
              <strong>{patch.patch_id}</strong>
              <StatusBadge status={patch.lifecycle} />
              <span className="patch-row-default">{patch.production_default}</span>
              <small>{patch.title}</small>
            </button>
          ))}
        </div>
      )}
    </section>
  );
}

// formatAppliesTo extracted to ./lib/format.

function PatchExplainPanel({
  patch,
  detail,
  state,
  error,
  override,
  overrideCount,
  onSetOverride,
  allPatchIds,
  onSelectPatch
}: {
  patch: PatchRow | null;
  detail: PatchExplainResult | null;
  state: "idle" | "loading" | "ready" | "error";
  error: string | null;
  override: string;
  overrideCount: number;
  onSetOverride: (state: string) => void;
  allPatchIds: Set<string>;
  onSelectPatch: (id: string) => void;
}) {
  if (!patch) {
    return (
      <aside className="patch-explain">
        <strong>No patch selected</strong>
        <p>Use the search and filters to select a patch from the registry.</p>
      </aside>
    );
  }
  const spec = detail?.spec ?? {};
  const meta = detail?.meta ?? {};
  const liveDecision = detail?.live_decision;
  const description = asText(meta.experimental_note ?? spec.experimental_note, "");
  const appliesRows = formatAppliesTo(spec.applies_to ?? meta.applies_to);
  const requires = asStringArray(spec.requires_patches ?? meta.requires_patches);
  const conflicts = asStringArray(spec.conflicts_with ?? meta.conflicts_with);
  const relatedPrs = asStringArray(spec.related_upstream_prs ?? meta.related_upstream_prs);
  const credit = asText(meta.credit ?? spec.credit, "");
  const canForce = Boolean(patch.env_flag);

  return (
    <aside className="patch-explain">
      <div className="patch-explain-head">
        <Wrench size={17} />
        <div>
          <strong>{patch.patch_id}</strong>
          <span>{patch.title || "Registry patch"}</span>
        </div>
        <StatusBadge status={patch.lifecycle} />
      </div>

      {/* Enablement override — operator forces on/off, written into the launch env. */}
      <div className="patch-override">
        <div className="patch-override-head">
          <strong>Enablement override</strong>
          {overrideCount > 0 && <span className="chip">{overrideCount} active</span>}
        </div>
        <div className="override-toggle">
          {[
            { id: "default", label: "Registry default" },
            { id: "on", label: "Force on" },
            { id: "off", label: "Force off" }
          ].map((opt) => (
            <button
              key={opt.id}
              className={override === opt.id ? "active" : ""}
              disabled={!canForce && opt.id !== "default"}
              onClick={() => onSetOverride(opt.id)}
            >
              {opt.label}
            </button>
          ))}
        </div>
        <p className="muted">
          {canForce
            ? <>Writes <code>{patch.env_flag}={override === "off" ? "0" : "1"}</code> into the launch env (operator-local, reflected in the Launch Plan).</>
            : "This patch has no env flag — enablement is not operator-controllable."}
        </p>
      </div>

      {description && (
        <div className="explain-note">
          <strong>What it does</strong>
          <p>{description}</p>
        </div>
      )}

      {appliesRows.length > 0 && (
        <div className="patch-applies">
          <strong>Supported models / applicability</strong>
          <InfoRows rows={appliesRows} />
        </div>
      )}
      {appliesRows.length === 0 && state === "ready" && (
        <p className="muted patch-applies-none">Applies to all catalog models (no model-specific constraints).</p>
      )}

      {(requires.length > 0 || conflicts.length > 0) && (
        <div className="patch-deps">
          {requires.length > 0 && (
            <div>
              <span className="patch-dep-label">Requires</span>
              <div className="chip-row">
                {requires.map((r) => allPatchIds.has(r)
                  ? <button type="button" className="chip chip-link" key={r} onClick={() => onSelectPatch(r)} title={`Open ${r} in the registry`}>{r} →</button>
                  : <span className="chip chip-unknown" key={r} title="Not present in the current registry view">{r}</span>)}
              </div>
            </div>
          )}
          {conflicts.length > 0 && (
            <div>
              <span className="patch-dep-label danger">Conflicts with</span>
              <div className="chip-row">
                {conflicts.map((c) => allPatchIds.has(c)
                  ? <button type="button" className="chip danger chip-link" key={c} onClick={() => onSelectPatch(c)} title={`Open ${c} in the registry`}>{c} →</button>
                  : <span className="chip danger chip-unknown" key={c} title="Not present in the current registry view">{c}</span>)}
              </div>
            </div>
          )}
        </div>
      )}

      <InfoRows
        rows={[
          ["Production default", patch.production_default],
          ["Tier", patch.tier],
          ["Family", patch.family || "-"],
          ["Implementation", asText(spec.implementation_status, patch.implementation_status)],
          ["Env flag", patch.env_flag || "-"],
          ["Apply module", patch.apply_module || "-"],
          ["Upstream PR", patch.upstream_pr ? `#${patch.upstream_pr}` : "-"],
          ["Related PRs", relatedPrs.length ? relatedPrs.map((p) => `#${p}`).join(", ") : "-"],
          ["Category", asText(spec.category, "-")],
          ["Source", asText(spec.source, "-")],
          ["Credit", credit || "-"]
        ]}
      />

      <div className={`patch-live-state ${state}`}>
        <Activity size={15} />
        <span>
          {state === "loading"
            ? "Loading Product API explain payload"
            : state === "error"
              ? `Explain API error: ${error ?? "unknown"}`
              : liveDecision
                ? `Live decision: ${liveDecision[0] ? "apply" : "skip"} / ${liveDecision[1]}`
                : `Live decision unavailable${detail?.live_decision_error ? `: ${detail.live_decision_error}` : ""}`}
        </span>
      </div>

      <div className="explain-note">
        <strong>Lifecycle — {patch.lifecycle}</strong>
        <p>{patchLifecycleExplanation(patch.lifecycle)}</p>
      </div>
      <div className="explain-note">
        <strong>Default behavior — {patch.production_default}</strong>
        <p>{patchDefaultExplanation(patch.production_default)}</p>
      </div>
    </aside>
  );
}

// CaveatsPanel / ConfigKeysPanel / TracesPanel extracted to ./sections/diagnostics.
// DoctorStat extracted to ./components/primitives.
// SEVERITY_META + DoctorSummary / DoctorFindings (+ DoctorCategory / SeverityDot /
// DoctorFindingRow) extracted to ./sections/doctor.

type WizardStatus = "done" | "active" | "todo" | "warning" | "blocked";

const DEPLOY_TARGET_ICONS: Record<string, ReactNode> = {
  compose: <Box size={16} />,
  quadlet: <Package size={16} />,
  kubernetes: <Layers size={16} />,
  systemd: <Settings size={16} />,
  bare_metal: <Cpu size={16} />,
  proxmox: <Server size={16} />
};

function downloadText(filename: string, content: string) {
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(url);
}

// fmtParam extracted to ./lib/format.

// Enterprise deployment console — pick a target + preset, review the resolved
// fine launch parameters, host readiness and dependency plan, then render the
// exact deployment artifact (compose / quadlet / k8s / systemd / bare-metal /
// proxmox) with copy + download. Read-only against the daemon: nothing is
// executed on the host.
function DeploymentConsole({
  presets,
  selectedPreset,
  onSelectPreset
}: {
  presets: PresetListResult | null;
  selectedPreset: string;
  onSelectPreset: (id: string) => void;
}) {
  const [meta, setMeta] = useState<DeployTargetsResult | null>(null);
  const [target, setTarget] = useState<string>("compose");
  const [plan, setPlan] = useState<DeploymentPlan | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [paths, setPaths] = useState<Record<string, string>>({});

  useEffect(() => {
    let cancelled = false;
    api.deployTargets()
      .then((data) => { if (!cancelled) setMeta(data); })
      .catch(() => { /* targets card simply stays empty */ });
    return () => { cancelled = true; };
  }, []);

  // Reset operator path overrides whenever the preset changes.
  useEffect(() => { setPaths({}); }, [selectedPreset]);

  useEffect(() => {
    if (!selectedPreset) return;
    let cancelled = false;
    const handle = window.setTimeout(() => {
      setLoading(true);
      api.deployPlan({
        preset_id: selectedPreset,
        target,
        host_paths: Object.keys(paths).length ? paths : undefined
      })
        .then((result) => { if (!cancelled) { setPlan(result); setError(null); } })
        .catch((err) => { if (!cancelled) setError(err instanceof Error ? err.message : String(err)); })
        .finally(() => { if (!cancelled) setLoading(false); });
    }, 300);
    return () => { cancelled = true; window.clearTimeout(handle); };
  }, [selectedPreset, target, paths]);

  const targets = meta?.targets ?? [];
  const host = meta?.host ?? null;
  const activeTarget: DeployTarget | undefined = targets.find((entry) => entry.id === target);
  const params = plan?.parameters;
  const deps = plan?.dependencies;
  const presetList = presets?.presets ?? [];

  const depTone = deps && deps.n_blockers > 0 ? "blocked" : deps && deps.n_warnings > 0 ? "warning" : "pass";
  const dockerOk = host?.docker.installed && host.docker.daemon_running;
  const gpuOk = host?.nvidia.installed && host.nvidia.n_gpus > 0;

  return (
    <ModuleGrid>
      <ModuleCard
        title="Deployment target"
        icon={<Rocket size={18} />}
        desc="Choose how the pinned vLLM stack lands on the host. Each target renders a ready-to-apply artifact."
        wide
      >
        <div className="deploy-targets">
          {targets.map((entry) => (
            <button
              key={entry.id}
              type="button"
              className={`deploy-target ${entry.id === target ? "active" : ""}`}
              onClick={() => setTarget(entry.id)}
            >
              <span className="deploy-target-head">
                {DEPLOY_TARGET_ICONS[entry.id] ?? <Box size={16} />}
                <strong>{entry.label}</strong>
                {entry.needs && <em className="deploy-need">needs {entry.needs}</em>}
              </span>
              <small>{entry.summary}</small>
            </button>
          ))}
          {targets.length === 0 && <p className="muted">Loading deployment targets…</p>}
        </div>
      </ModuleCard>

      <ModuleCard
        title="Preset & launch parameters"
        icon={<Database size={18} />}
        desc="The resolved engine command for this preset — tensor parallelism, KV-cache dtype, context window and Genesis pin."
      >
        <label className="deploy-field">
          <span>Preset</span>
          <select value={selectedPreset} onChange={(event) => onSelectPreset(event.target.value)}>
            {presetList.length === 0 && <option value={selectedPreset}>{selectedPreset}</option>}
            {presetList.map((preset) => (
              <option key={preset.id} value={preset.id}>{preset.id}</option>
            ))}
          </select>
        </label>
        {params ? (
          <InfoRows
            rows={[
              ["Tensor parallel", `${fmtParam(params.tensor_parallel)} GPU`],
              ["KV-cache dtype", fmtParam(params.kv_cache_dtype)],
              ["Max context", fmtParam(params.max_model_len)],
              ["Max sequences", fmtParam(params.max_num_seqs)],
              ["GPU mem util", params.gpu_memory_utilization != null ? `${Math.round(params.gpu_memory_utilization * 100)}%` : "—"],
              ["Min VRAM / GPU", params.min_vram_per_gpu_mib != null ? `${Math.round(params.min_vram_per_gpu_mib / 1024)} GiB` : "—"],
              ["Genesis flags", fmtParam(params.genesis_env_count)],
              ["Pin", fmtParam(params.genesis_pin)],
              ["Container", fmtParam(params.container_name)],
              ["Host port", fmtParam(params.host_port)],
              ["Image", fmtParam(params.image)]
            ]}
          />
        ) : (
          loading ? <SkeletonLines count={5} /> : <p className="muted">Select a preset to resolve launch parameters.</p>
        )}
      </ModuleCard>

      <ModuleCard
        title="Host readiness"
        icon={<ShieldCheck size={18} />}
        desc="Live inventory of the daemon host and the dependency plan for this preset."
      >
        <div className="setup-glance">
          <RailCheck label="OS" value={host ? `${host.os.distro || host.os.system} ${host.os.arch}` : "…"} status="pass" />
          <RailCheck label="Docker" value={host ? (dockerOk ? `running ${host.docker.server_version ?? ""}`.trim() : host.docker.installed ? "stopped" : "missing") : "…"} status={dockerOk ? "pass" : "warning"} />
          <RailCheck label="GPU" value={host ? (gpuOk ? `${host.nvidia.n_gpus}× ${host.nvidia.gpu_names[0] ?? "GPU"}` : "none") : "…"} status={gpuOk ? "pass" : "warning"} />
          <RailCheck label="vLLM" value={host ? (host.vllm.installed ? host.vllm.version ?? "installed" : "not installed") : "…"} status={host?.vllm.installed ? "pass" : "warning"} />
        </div>
        {deps && (
          <div className="deploy-deps">
            <div className={`deploy-deps-head tone-${depTone}`}>
              {depTone === "pass" ? <CheckCircle2 size={15} /> : depTone === "blocked" ? <AlertCircle size={15} /> : <CircleAlert size={15} />}
              <strong>
                {deps.is_ready
                  ? "Host is ready for this preset"
                  : `${deps.n_blockers} blocker${deps.n_blockers === 1 ? "" : "s"} · ${deps.n_warnings} warning${deps.n_warnings === 1 ? "" : "s"}`}
              </strong>
            </div>
            {deps.items.map((item, index) => (
              <div className={`deploy-dep-item sev-${item.severity}`} key={`${item.scope}-${index}`}>
                <div className="deploy-dep-row">
                  <span className={`sev-dot sev-${item.severity === "blocker" ? "danger" : item.severity === "warning" ? "warn" : "info"}`} />
                  <strong>{item.target}</strong>
                  <em>{item.action}</em>
                </div>
                <small>{item.reason}</small>
                {item.suggested_command && <CodeBlock lines={[item.suggested_command]} />}
              </div>
            ))}
            {deps.items.length === 0 && <p className="muted">No host changes required for this preset.</p>}
          </div>
        )}
      </ModuleCard>

      {plan && plan.mount_vars.length > 0 && (
        <ModuleCard
          title="Storage & mount paths"
          icon={<HardDrive size={18} />}
          desc="Map the preset's container mounts to real host paths. Edits re-render the artifact below."
          wide
        >
          <div className="deploy-mounts">
            {plan.mount_vars.map((mount) => (
              <label className="deploy-field" key={mount.name}>
                <span>{mount.name} <em className="deploy-mount-target">→ {mount.container}</em></span>
                <input
                  type="text"
                  value={paths[mount.name] !== undefined ? paths[mount.name] : mount.value}
                  spellCheck={false}
                  onChange={(event) => setPaths((prev) => ({ ...prev, [mount.name]: event.target.value }))}
                />
              </label>
            ))}
          </div>
        </ModuleCard>
      )}

      <ModuleCard
        title={activeTarget ? `${activeTarget.label} — ${activeTarget.filename}` : "Generated artifact"}
        icon={<FileText size={18} />}
        desc="The exact file to drop on the host, plus the operator commands to apply it."
        wide
      >
        {error && <div className="inline-error"><AlertCircle size={15} /> {error}</div>}
        {plan ? (
          <>
            <div className="deploy-artifact-bar">
              <span className="deploy-artifact-name"><Terminal size={14} /> {plan.artifact.filename}</span>
              <div className="deploy-artifact-actions">
                <CopyButton value={plan.artifact.content} label={plan.artifact.filename} />
                <button className="ghost-button" onClick={() => downloadText(plan.artifact.filename, plan.artifact.content)}>
                  <Download size={14} /> Download
                </button>
              </div>
            </div>
            <div className="deploy-artifact"><CodeBlock lines={plan.artifact.content.split("\n")} title={plan.artifact.filename} /></div>
            <div className="deploy-cmd-head">
              <h4 className="deploy-cmd-title">Apply commands</h4>
              {(() => {
                // One copyable, fail-fast shell script: write the artifact via a
                // heredoc, then run the apply commands — no manual file shuffling.
                const eof = "SNDR_EOF";
                const script = [
                  "#!/usr/bin/env bash", "set -euo pipefail", "",
                  `# ${activeTarget?.label ?? "deploy"} — generated by SNDR Control Center`,
                  `cat > ${plan.artifact.filename} <<'${eof}'`,
                  plan.artifact.content.replace(/\n$/, ""),
                  eof, "",
                  ...plan.commands,
                ].join("\n");
                return (
                  <div className="deploy-cmd-actions">
                    <CopyButton value={script} label="apply script" />
                    <button className="ghost-button" onClick={() => downloadText(`apply-${activeTarget?.id ?? "deploy"}.sh`, script)}><Download size={13} /> Script</button>
                  </div>
                );
              })()}
            </div>
            <CodeBlock lines={plan.commands} />
          </>
        ) : (
          loading ? <SkeletonLines count={5} /> : <p className="muted">Select a preset and target to render the deployment artifact.</p>
        )}
      </ModuleCard>
    </ModuleGrid>
  );
}

function SetupWizard({
  environment,
  overview,
  doctorReport,
  gateCounts,
  selectedPreset,
  runtimeMode,
  apiBase,
  onSection
}: {
  environment: EnvironmentReport | null;
  overview: ProductOverview | null;
  doctorReport: DoctorReport | null;
  gateCounts: Record<GateStatus, number>;
  selectedPreset: string;
  runtimeMode: RuntimeMode;
  apiBase: string;
  onSection: (section: SectionId) => void;
}) {
  const env = environment;
  const dockerTool = env?.tools.find((tool) => tool.name === "docker");
  const nvidiaTool = env?.tools.find((tool) => tool.name === "nvidia-smi");
  const doctorBlocked = doctorReport?.summary.blocked ?? 0;
  const doctorWarn = doctorReport?.summary.warning ?? 0;
  const host = runtimeMode === "remote" ? "gpu-build-01" : "127.0.0.1";

  const steps: Array<{ key: string; title: string; hint: string; status: WizardStatus }> = [
    { key: "detect", title: "Detect host", hint: "Engine, Python and runtime tools", status: env ? "done" : "active" },
    { key: "mode", title: "Connection mode", hint: "Local server or remote SSH tunnel", status: "done" },
    { key: "preset", title: "Choose a preset", hint: "Pick a workload-matched config", status: selectedPreset ? "done" : "todo" },
    { key: "validate", title: "Validate", hint: "Run diagnostics, clear blockers", status: doctorReport ? (doctorBlocked > 0 ? "blocked" : doctorWarn > 0 ? "warning" : "done") : "todo" },
    { key: "launch", title: "Plan & launch", hint: "Compose a launch plan", status: gateCounts.blocked > 0 ? "blocked" : selectedPreset ? "done" : "todo" }
  ];
  const [active, setActive] = useState(0);
  const done = steps.filter((step) => step.status === "done").length;
  const cur = steps[active];

  const tone = (status: WizardStatus) =>
    status === "done" ? "ok" : status === "blocked" ? "danger" : status === "warning" ? "warn" : status === "active" ? "accent" : "muted";
  const icon = (status: WizardStatus) =>
    status === "done" ? <CheckCircle2 size={15} /> : status === "blocked" ? <AlertCircle size={15} /> : status === "warning" ? <CircleAlert size={15} /> : <Circle size={15} />;

  return (
    <section className="setup-wizard">
      <div className="setup-progress">
        <div>
          <strong>First-run setup</strong>
          <span>Guided path — read-only, no host mutation</span>
        </div>
        <div className="setup-progress-meta">{done}/{steps.length} ready</div>
        <div className="setup-progress-track"><span style={{ width: `${(done / steps.length) * 100}%` }} /></div>
      </div>

      <div className="setup-glance">
        <RailCheck label="Engine" value={env?.engine_version ? `vLLM ${env.engine_version}` : "not installed"} status={env?.engine_installed ? "pass" : "warning"} />
        <RailCheck label="Docker" value={dockerTool?.present ? "available" : "missing"} status={dockerTool?.present ? "pass" : "warning"} />
        <RailCheck label="GPU (nvidia-smi)" value={nvidiaTool?.present ? "available" : "missing"} status={nvidiaTool?.present ? "pass" : "warning"} />
        <RailCheck label="Doctor" value={`${doctorBlocked} blocked · ${doctorWarn} warn`} status={doctorBlocked > 0 ? "warning" : "pass"} />
        <RailCheck label="Preset" value={selectedPreset || "none"} status={selectedPreset ? "pass" : "warning"} />
        <RailCheck label="Gates" value={`${gateCounts.pass}/${gateCounts.pass + gateCounts.warning + gateCounts.blocked}`} status={gateCounts.blocked > 0 ? "warning" : "pass"} />
      </div>

      <div className="setup-grid">
        <aside className="setup-steps">
          {steps.map((step, index) => (
            <button
              key={step.key}
              className={`setup-step tone-${tone(step.status)} ${index === active ? "active" : ""}`}
              onClick={() => setActive(index)}
            >
              <span className="setup-step-icon">{icon(step.status)}</span>
              <div>
                <strong>{step.title}</strong>
                <small>{step.hint}</small>
              </div>
              <span className="setup-step-num">{index + 1}</span>
            </button>
          ))}
        </aside>

        <section className="setup-content">
          <header className="setup-content-head">
            <h3>{cur.title}</h3>
            <StatusBadge status={cur.status === "done" ? "available" : cur.status === "blocked" ? "missing" : cur.status === "warning" ? "partial" : "deferred"} />
          </header>

          {cur.key === "detect" && (
            <>
              <EnvironmentPanel env={env} />
              <CodeBlock lines={["python -m vllm.sndr_core.cli gui-api --host 127.0.0.1 --port 8765", "python -m vllm.sndr_core.cli doctor --all"]} />
            </>
          )}
          {cur.key === "mode" && (
            <>
              <InfoRows rows={[
                ["Active mode", runtimeMode === "remote" ? "Remote desktop (SSH tunnel)" : "Local server"],
                ["API base", apiBase],
                ["Bind", "127.0.0.1 (localhost only)"],
                ["Writes", "Disabled — dry-run apply jobs only"]
              ]} />
              <CodeBlock lines={runtimeMode === "remote"
                ? [`ssh -L 8765:127.0.0.1:8765 user@${host}`, "# then open http://127.0.0.1:8765 locally"]
                : ["python -m vllm.sndr_core.cli gui-api --host 127.0.0.1 --port 8765"]} />
            </>
          )}
          {cur.key === "preset" && (
            <>
              <InfoRows rows={[
                ["Selected preset", selectedPreset || "none"],
                ["Catalog presets", overview?.catalog.presets_count ?? "-"],
                ["Models", overview?.catalog.models_count ?? "-"],
                ["Profiles", overview?.catalog.profiles_count ?? "-"]
              ]} />
              <div className="setup-actions">
                <button className="ghost-button" onClick={() => onSection("presets")}><Database size={15} /> Browse presets</button>
                <button className="primary-action" onClick={() => onSection("launch-plan")}><Rocket size={15} /> Recommend by workload</button>
              </div>
            </>
          )}
          {cur.key === "validate" && (
            <>
              <div className="doctor-stat-row">
                <DoctorStat tone="ok" value={doctorReport?.summary.ok ?? 0} label="Healthy" />
                <DoctorStat tone="warn" value={doctorWarn} label="Warnings" />
                <DoctorStat tone="danger" value={doctorBlocked} label="Blocked" />
                <DoctorStat tone="info" value={doctorReport?.findings.length ?? 0} label="Checks" />
              </div>
              <div className="setup-actions">
                <button className="primary-action" onClick={() => onSection("doctor")}><ShieldCheck size={15} /> Open Doctor</button>
              </div>
            </>
          )}
          {cur.key === "launch" && (
            <>
              <InfoRows rows={[
                ["Gates passing", gateCounts.pass],
                ["Warnings", gateCounts.warning],
                ["Blocked", gateCounts.blocked],
                ["Preset", selectedPreset || "none"]
              ]} />
              <div className="setup-actions">
                <button className="primary-action" onClick={() => onSection("launch-plan")}><Rocket size={15} /> Open Launch Plan</button>
                <button className="ghost-button" onClick={() => onSection("services")}><Network size={15} /> Service lifecycle</button>
              </div>
            </>
          )}

          <div className="setup-nav">
            <button className="ghost-button" disabled={active === 0} onClick={() => setActive((value) => Math.max(0, value - 1))}>Back</button>
            <span className="setup-nav-detect">
              {cur.key === "detect" && <>Engine {env?.engine_version ? `vLLM ${env.engine_version}` : "not installed"} · Docker {dockerTool?.present ? "✓" : "—"} · GPU {nvidiaTool?.present ? "✓" : "—"}</>}
            </span>
            <button className="primary-action" disabled={active === steps.length - 1} onClick={() => setActive((value) => Math.min(steps.length - 1, value + 1))}>Next step</button>
          </div>
        </section>
      </div>
    </section>
  );
}

// EnvironmentPanel extracted to ./sections/environment.

const SERVICE_RUNTIME_TARGETS: Array<{ id: string; label: string }> = [
  { id: "docker_compose", label: "Docker Compose" },
  { id: "docker", label: "Docker" },
  { id: "systemd", label: "systemd" },
  { id: "podman", label: "Podman" },
  { id: "quadlet", label: "Quadlet" },
  { id: "kubernetes", label: "Kubernetes" }
];

function ServiceLifecyclePlanner({
  selectedPreset,
  runtimeTarget,
  host
}: {
  selectedPreset: string;
  runtimeTarget: string;
  host: string;
}) {
  const [action, setAction] = useState("status");
  const [target, setTarget] = useState(runtimeTarget);
  const [plan, setPlan] = useState<ServiceActionPlan | null>(null);
  const [state, setState] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [applying, setApplying] = useState(false);
  const [applyEnabled, setApplyEnabled] = useState(false);
  const [sshTarget, setSshTarget] = useState("");
  const [confirm, setConfirm] = useState(false);
  const [engine, setEngine] = useState<EngineStatus | null>(null);
  const [engineChecking, setEngineChecking] = useState(false);

  // Re-seed the target when the active preset's default runtime changes.
  useEffect(() => { setTarget(runtimeTarget); }, [runtimeTarget]);

  async function checkEngine() {
    setEngineChecking(true);
    try {
      setEngine(await api.engineStatus(host));
    } catch {
      setEngine(null);
    } finally {
      setEngineChecking(false);
    }
  }

  useEffect(() => {
    let cancelled = false;
    api.authStatus().then((s) => { if (!cancelled) setApplyEnabled(s.apply_enabled); }).catch(() => {});
    void checkEngine();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [host]);

  const mutating = plan?.mutating ?? false;
  const transport = sshTarget.trim() ? "ssh" : "local";
  // Execution is allowed when the daemon enables apply; mutating actions also
  // need an explicit confirm. Otherwise the action is recorded as a dry-run.
  const willExecute = applyEnabled;
  const blockedMutation = applyEnabled && mutating && !confirm;

  async function runApply() {
    setApplying(true);
    setError(null);
    try {
      const result = await api.serviceApply({
        preset_id: selectedPreset,
        action,
        runtime_target: target,
        host,
        transport,
        ssh_target: sshTarget.trim(),
        confirm
      });
      setJob(result);
      // After a mutating local action, re-probe the engine so the operator
      // immediately sees whether it actually came up / went down.
      if (mutating && transport === "local") {
        window.setTimeout(() => void checkEngine(), 1500);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setApplying(false);
    }
  }

  useEffect(() => {
    if (!selectedPreset) return;
    setJob(null);
    setConfirm(false);
    let cancelled = false;
    setState("loading");
    setError(null);
    api.servicePlan({ preset_id: selectedPreset, action, runtime_target: target, host })
      .then((result) => {
        if (cancelled) return;
        setPlan(result);
        setState("ready");
      })
      .catch((err) => {
        if (cancelled) return;
        setPlan(null);
        setState("error");
        setError(err instanceof Error ? err.message : String(err));
      });
    return () => { cancelled = true; };
  }, [selectedPreset, action, target, host]);

  const engineTone = engine?.reachable ? "ok" : "danger";
  const engineLabel = engineChecking
    ? "checking…"
    : engine?.reachable
      ? `engine up${engine.version ? ` · v${engine.version}` : ""}`
      : "engine down";

  return (
    <div className="service-planner">
      <div className="service-toolbar">
        <div className="settings-segmented service-actions">
          {["status", "logs", "start", "restart", "stop"].map((item) => (
            <button key={item} className={action === item ? "active" : ""} onClick={() => setAction(item)}>{item}</button>
          ))}
        </div>
        <label className="service-runtime">
          <span>Runtime</span>
          <select value={target} onChange={(event) => setTarget(event.target.value)}>
            {SERVICE_RUNTIME_TARGETS.map((rt) => <option key={rt.id} value={rt.id}>{rt.label}</option>)}
          </select>
        </label>
        <span className="service-target">{host}</span>
      </div>

      <div className="service-engine-strip">
        <span className={`fleet-status ${engineTone}`}><span className="fleet-dot" />{engineLabel}</span>
        {engine?.reachable && engine.models.length > 0 && <span className="service-engine-models">serving {engine.models.join(", ")}</span>}
        {engine && !engine.reachable && engine.error && <span className="service-engine-err">{engine.error}</span>}
        <button className="ghost-button" onClick={() => void checkEngine()} disabled={engineChecking}>
          <Activity size={14} /> {engineChecking ? "Checking…" : "Re-check engine"}
        </button>
      </div>

      {error && <div className="config-plan-error"><AlertCircle size={15} /><span>{error}</span></div>}

      {plan && (
        <>
          <div className="service-plan-head">
            <div>
              <strong>{plan.container_name}</strong>
              <span>{plan.plan_id}</span>
            </div>
            <StatusPill tone={plan.mutating ? "warning" : "success"}>
              {plan.mutating ? "mutating" : "read-only"}
            </StatusPill>
          </div>

          <div className="service-steps">
            {plan.steps.map((step) => (
              <div className="service-step" key={step.order}>
                <span className="service-step-num">{step.order}</span>
                <div>
                  <small>{step.title}</small>
                  <code>{step.command}</code>
                </div>
                <CopyButton value={step.command} label={step.title} />
              </div>
            ))}
          </div>

          {plan.side_effects.length > 0 && (
            <div className="service-effects">
              <strong>Side effects</strong>
              {plan.side_effects.map((effect, index) => (
                <p key={index}><CircleAlert size={13} /> {effect}</p>
              ))}
            </div>
          )}

          <div className="gates-list">
            {plan.gates.map((gate) => (
              <GateRow
                key={gate.id}
                gate={{ id: gate.id, label: gate.title, detail: gate.detail, status: gate.status as GateStatus, action: "Inspect" }}
              />
            ))}
          </div>

          <div className="service-exec-row">
            <label className="service-ssh">
              <span>SSH target (empty = local)</span>
              <input
                value={sshTarget}
                onChange={(event) => setSshTarget(event.target.value)}
                placeholder="user@gpu-host"
              />
            </label>
            {willExecute && mutating && (
              <label className="service-confirm">
                <input type="checkbox" checked={confirm} onChange={(event) => setConfirm(event.target.checked)} />
                <span>Confirm — execute this mutating action on {sshTarget.trim() || "localhost"}</span>
              </label>
            )}
          </div>

          <div className="service-footer">
            <div className="service-rollback"><RefreshCw size={13} /> {plan.rollback}</div>
            <button
              className="primary-action"
              onClick={() => void runApply()}
              disabled={applying || blockedMutation}
            >
              <Play size={15} />{" "}
              {applying
                ? willExecute ? "Executing…" : "Recording…"
                : willExecute
                  ? (mutating ? `Execute ${action} (${transport})` : `Run ${action} (${transport})`)
                  : "Apply (dry-run)"}
            </button>
          </div>
          <p className="service-reason">
            {willExecute
              ? blockedMutation
                ? "Tick confirm to execute this mutating action."
                : `Apply enabled — ${mutating ? "mutating" : "read-only"} action runs over ${transport}.`
              : plan.action_reason}
          </p>

          {job && <JobResultBlock job={job} showNote />}
        </>
      )}
      {state === "loading" && !plan && <p className="muted">Planning…</p>}
    </div>
  );
}

// DoctorCoveragePanel + AdminSurfaceMatrix extracted to ./sections/patch-doctor.
// (BundlesPanel + UpstreamDiffPanel previously extracted to ./sections/registry;
//  ProofStatusPanel + drill-down to ./sections/proof.)

// BenchmarkBaselinePanel + EvidenceRows extracted to ./sections/bench.

function EndpointRows({ host }: { host: string }) {
  const rows: Array<[string, string]> = [
    ["OpenAI API", `http://${host}:8000/v1`],
    ["Health", `http://${host}:8000/health`],
    ["Metrics", `http://${host}:8001/metrics`],
    ["Docs", `http://${host}:8000/docs`]
  ];
  return (
    <div className="endpoint-rows">
      {rows.map(([label, value]) => (
        <label className="endpoint-field" key={label}>
          <span>{label}</span>
          <div>
            <input value={value} readOnly />
            <CopyButton value={value} label={label} />
          </div>
        </label>
      ))}
    </div>
  );
}

// Reusable confirmation dialog for destructive/irreversible actions. Focus is
// trapped, Cancel is the autofocused default, Esc/backdrop cancel, and the
// confirm button can be styled as danger. Keeps destructive paths deliberate.
// ConfirmDialog + InfoDialog extracted to ./components/dialogs.

// ── Toast notifications (PegaProx-style transient feedback) ───────────────
type ToastTone = "info" | "success" | "error";
function toast(message: string, tone: ToastTone = "info") {
  window.dispatchEvent(new CustomEvent("sndr-toast", { detail: { message, tone, id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}` } }));
}
function ToastHost() {
  const [items, setItems] = useState<Array<{ id: string; message: string; tone: ToastTone }>>([]);
  useEffect(() => {
    const onToast = (event: Event) => {
      const detail = (event as CustomEvent).detail as { id: string; message: string; tone: ToastTone };
      setItems((prev) => [...prev.slice(-3), detail]);
      // Errors linger long enough to actually read a failure; transient
      // success/info notices clear quickly.
      const ttl = detail.tone === "error" ? 8000 : 4200;
      window.setTimeout(() => setItems((prev) => prev.filter((item) => item.id !== detail.id)), ttl);
    };
    window.addEventListener("sndr-toast", onToast);
    return () => window.removeEventListener("sndr-toast", onToast);
  }, []);
  if (items.length === 0) return null;
  return (
    <div className="toast-host" role="region" aria-label="Notifications">
      {items.map((item) => (
        // Errors announce assertively (role=alert); success/info politely
        // (role=status). The per-toast role carries the right aria-live, so the
        // container itself stays a plain labelled region.
        <div key={item.id} className={`toast toast-${item.tone}`} role={item.tone === "error" ? "alert" : "status"} aria-atomic="true">
          {item.tone === "success" ? <CheckCircle2 size={15} /> : item.tone === "error" ? <AlertCircle size={15} /> : <Activity size={15} />}
          <span>{item.message}</span>
          <button className="icon-only" onClick={() => setItems((prev) => prev.filter((x) => x.id !== item.id))} aria-label="Dismiss"><X size={13} /></button>
        </div>
      ))}
    </div>
  );
}

// ── Audit log (surfaces the daemon's recorded events: auth, jobs, system) ──
// AuditLogPanel extracted to ./sections/audit-log.

function CommandPalette({
  onClose,
  onSection,
  onRefresh,
  onShortcuts,
  settings,
  onSettings,
  searchItems
}: {
  onClose: () => void;
  onSection: (section: SectionId) => void;
  onRefresh: () => void;
  onShortcuts: () => void;
  settings: GuiSettings;
  onSettings: (patch: Partial<GuiSettings>) => void;
  searchItems: Array<{ icon: ReactNode; title: string; detail: string; keep?: boolean; run: () => void }>;
}) {
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const listRef = useRef<HTMLDivElement>(null);
  const dialogRef = useRef<HTMLElement>(null);
  useDialogFocus(dialogRef);
  const commands: Array<{ icon: ReactNode; title: string; detail: string; keep?: boolean; run: () => void }> = [
    { icon: <RefreshCw size={16} />, title: "Sync Catalog", detail: "Refresh overview, presets, patch registry and doctor state", run: onRefresh },
    { icon: <Rocket size={16} />, title: "Open Launch Plan", detail: "Recommendation builder and launch composer", run: () => onSection("launch-plan") },
    { icon: <Terminal size={16} />, title: "Open Operations", detail: "Run project maintenance and diagnostic workflows", run: () => onSection("operations") },
    { icon: <ShieldCheck size={16} />, title: "Run Doctor View", detail: "Readiness gates and registry doctor panel", run: () => onSection("doctor") },
    { icon: <PackageCheck size={16} />, title: "Patch Matrix", detail: "Patch lifecycle, default policy and registry coverage", run: () => onSection("patches") },
    { icon: themeIcon(nextTheme(settings.theme)), title: "Toggle Theme", detail: `Cycle themes (next: ${themeLabel(nextTheme(settings.theme))})`, keep: true, run: () => onSettings({ theme: nextTheme(settings.theme) }) },
    { icon: <Rows3 size={16} />, title: "Toggle Density", detail: "Switch comfortable/compact density", keep: true, run: () => onSettings({ density: settings.density === "compact" ? "comfortable" : "compact" }) },
    { icon: <Command size={16} />, title: "Keyboard Shortcuts", detail: "Show all shortcuts and navigation chords (?)", run: onShortcuts },
    { icon: <Settings size={16} />, title: "Settings", detail: "Appearance, API, schema and admin", run: () => onSection("advanced") }
  ];
  const all = [...commands, ...searchItems];
  const q = query.trim().toLowerCase();
  const shown = (q ? all.filter((item) => `${item.title} ${item.detail}`.toLowerCase().includes(q)) : commands).slice(0, 40);
  // Keep the highlighted index in range as the filtered set shrinks/grows.
  const activeIndex = Math.min(active, Math.max(0, shown.length - 1));
  // Reset the highlight to the top whenever the query changes.
  useEffect(() => { setActive(0); }, [q]);
  // Scroll the highlighted row into view as it moves past the visible window.
  useEffect(() => {
    const row = listRef.current?.children[activeIndex] as HTMLElement | undefined;
    row?.scrollIntoView({ block: "nearest" });
  }, [activeIndex]);
  const runAt = (index: number) => {
    const item = shown[index];
    if (!item) return;
    item.run();
    if (!item.keep) onClose();
  };
  const onKeyDown = (event: ReactKeyboardEvent<HTMLInputElement>) => {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActive((i) => Math.min(i + 1, shown.length - 1));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setActive((i) => Math.max(i - 1, 0));
    } else if (event.key === "Enter") {
      event.preventDefault();
      runAt(activeIndex);
    }
  };
  return (
    <div className="dialog-backdrop" role="presentation" onClick={closeOnBackdrop(onClose)}>
      <section ref={dialogRef} className="command-dialog" role="dialog" aria-modal="true">
        <div className="command-search">
          <Search size={16} />
          <input autoFocus value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search commands, sections, presets, models, configs, patches…" spellCheck={false}
            role="combobox" aria-expanded aria-controls="command-list" aria-activedescendant={shown[activeIndex] ? `command-item-${activeIndex}` : undefined}
            onKeyDown={onKeyDown} />
          <kbd>esc</kbd>
        </div>
        <div className="command-list" id="command-list" role="listbox" ref={listRef}>
          {shown.map((item, index) => (
            <button
              key={`${item.title}-${index}`}
              id={`command-item-${index}`}
              role="option"
              aria-selected={index === activeIndex}
              className={index === activeIndex ? "active" : ""}
              onMouseMove={() => setActive(index)}
              onClick={() => runAt(index)}
            >
              {item.icon}
              <span>
                <strong>{item.title}</strong>
                <small>{item.detail}</small>
              </span>
              <ChevronRight size={16} />
            </button>
          ))}
          {shown.length === 0 && <p className="muted command-empty">No matches for “{query}”.</p>}
        </div>
      </section>
    </div>
  );
}

function loadGuiSettings(): GuiSettings {
  try {
    const raw = window.localStorage.getItem(GUI_SETTINGS_STORAGE_KEY);
    if (!raw) return defaultGuiSettings;
    const parsed = JSON.parse(raw) as Partial<GuiSettings>;
    return {
      ...defaultGuiSettings,
      ...parsed,
      theme: parsed.theme && _VALID_THEMES.has(parsed.theme) ? parsed.theme : "light",
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

function RecommendationRow({
  row,
  active,
  onSelect
}: {
  row: PresetRecommendation;
  active: boolean;
  onSelect: () => void;
}) {
  const card = row.card ?? {};
  const family = asText(card.routing_family, row.model);
  const allowed = asStringArray(card.workload_allow);
  const fallback = asText(card.fallback_preset, "-");
  const visibility = asText(card.evidence_visibility, "unknown");
  const risk = visibility === "public" ? "Low" : visibility === "private" ? "Medium" : "Unknown";

  return (
    <tr className={active ? "active" : ""}>
      <td>
        <button className="preset-select" onClick={onSelect}>
          <span className="radio-dot">{active ? <CheckCircle2 size={15} /> : <Circle size={15} />}</span>
          <strong>{row.id}</strong>
        </button>
      </td>
      <td>{family}</td>
      <td>{asText(card.mode, row.profile ?? "-")}</td>
      <td>
        <StatusBadge status={asText(card.status, "missing")} />
      </td>
      <td>
        <div className="workload-icons">
          {allowed.slice(0, 4).map((item) => (
            <span key={item}>{shortWorkload(item)}</span>
          ))}
        </div>
      </td>
      <td>
        <span className={`visibility ${visibility}`}>{visibility}</span>
      </td>
      <td>{fallback}</td>
      <td>
        <span className={`risk ${risk.toLowerCase()}`}>{risk}</span>
      </td>
    </tr>
  );
}

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

const _GATE_TARGET: Record<string, { section: SectionId; label: string }> = {
  catalog: { section: "configs", label: "Open Configs" },
  "preset-card": { section: "presets", label: "Open Presets" },
  preset_card: { section: "presets", label: "Open Presets" },
  runtime: { section: "hosts", label: "Open Hosts" },
  engine: { section: "doctor", label: "Open Doctor" },
  engine_package: { section: "doctor", label: "Open Doctor" },
  patch_doctor: { section: "patches", label: "Open Patch Doctor" },
  "service-api": { section: "services", label: "Open Services" },
  service_lifecycle: { section: "services", label: "Open Services" },
  evidence: { section: "evidence", label: "Open Evidence" },
  evidence_orchestration: { section: "evidence", label: "Open Evidence" },
  "release-proof": { section: "reports", label: "Open Reports" },
  release_proof: { section: "reports", label: "Open Reports" }
};

function GateRow({ gate, onNavigate }: { gate: Gate; onNavigate?: (section: SectionId) => void }) {
  const [open, setOpen] = useState(false);
  const target = _GATE_TARGET[gate.id];
  return (
    <div className={`gate-row ${gate.status} ${open ? "open" : ""}`}>
      <button className="gate-main" onClick={() => setOpen((value) => !value)} aria-expanded={open}>
        <span className="gate-icon">
          {gate.status === "pass" && <CheckCircle2 size={16} />}
          {gate.status === "warning" && <CircleAlert size={16} />}
          {gate.status === "blocked" && <AlertCircle size={16} />}
          {gate.status === "planned" && <Clock3 size={16} />}
        </span>
        <div>
          <strong>{gate.label}</strong>
          <small>{gate.detail}</small>
        </div>
        <span className="gate-status">{gate.status}</span>
        <ChevronRight className="gate-caret" size={16} />
      </button>
      {open && (
        <div className="gate-detail">
          <p>{gate.detail}</p>
          <div className="gate-detail-actions">
            <span className="gate-action-hint"><Wrench size={13} /> {gate.action}</span>
            {target && onNavigate && (
              <button className="ghost-button" onClick={() => onNavigate(target.section)}>
                <ChevronRight size={14} /> {target.label}
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function JobsTable({ onMonitor }: { onMonitor?: (id: string) => void }) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [openId, setOpenId] = useState<string | null>(null);
  const [state, setState] = useState<"loading" | "ready">("loading");

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const result = await api.jobs();
        if (!cancelled) { setJobs(result.jobs); setState("ready"); }
      } catch {
        if (!cancelled) setState("ready");
      }
    };
    void load();
    const timer = setInterval(load, 5000);
    return () => { cancelled = true; clearInterval(timer); };
  }, []);

  if (state === "loading") return <SkeletonTable rows={5} cols={6} />;
  if (jobs.length === 0) {
    return (
      <p className="muted">
        No jobs yet. Run <strong>Apply Launch</strong>, a service action, or queue a
        bench/evidence job — real dry-run and executed jobs appear here.
      </p>
    );
  }
  const cls = (job: Job) => job.dry_run ? "queued" : job.status === "succeeded" ? "done" : job.status === "failed" ? "failed" : "running";
  const stamp = (ts: number) => new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  return (
    <section className="jobs-block">
      <table className="jobs-table">
        <thead>
          <tr><th>Job</th><th>Kind</th><th>Status</th><th>Steps</th><th>Time</th><th></th></tr>
        </thead>
        <tbody>
          {jobs.map((job) => (
            <Fragment key={job.job_id}>
              <tr className={openId === job.job_id ? "active job-row" : "job-row"} onClick={() => setOpenId(openId === job.job_id ? null : job.job_id)}>
                <td><code>{job.job_id}</code></td>
                <td>{job.kind}</td>
                <td><span className={`job-status ${cls(job)}`}>{job.dry_run ? "dry-run" : job.status}</span></td>
                <td>{job.steps.length}</td>
                <td>{stamp(job.created_at)}</td>
                <td className="job-row-actions">
                  {onMonitor && (
                    <button className="icon-only" title="Live monitor" aria-label={`Monitor ${job.job_id}`} onClick={(event) => { event.stopPropagation(); onMonitor(job.job_id); }}>
                      <Activity size={14} />
                    </button>
                  )}
                  <ChevronRight size={14} className={openId === job.job_id ? "job-caret open" : "job-caret"} />
                </td>
              </tr>
              {openId === job.job_id && (
                <tr className="job-detail-row">
                  <td colSpan={6}>
                    {job.note && <p className="fit-note">{job.note}</p>}
                    <CodeBlock lines={job.log} />
                  </td>
                </tr>
              )}
            </Fragment>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function Progress({ value }: { value: number }) {
  return (
    <span className="progress-track">
      <span style={{ width: `${Math.max(0, Math.min(100, value))}%` }} />
    </span>
  );
}

const _JOB_TERMINAL = new Set(["succeeded", "failed", "done", "error", "cancelled"]);

// Live job monitor modal — polls a job until it reaches a terminal state,
// streaming status, steps and log. Shared by launch / benchmark / evidence.
function JobMonitorModal({ jobId, onClose }: { jobId: string; onClose: () => void }) {
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const dialogRef = useRef<HTMLElement>(null);
  useDialogFocus(dialogRef);
  useEscapeKey(onClose);

  useEffect(() => {
    let cancelled = false;
    let timer = 0;
    const poll = async () => {
      try {
        const next = await api.job(jobId);
        if (cancelled) return;
        setJob(next);
        const terminal = next.dry_run || _JOB_TERMINAL.has(next.status);
        if (!terminal) timer = window.setTimeout(poll, 1500);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      }
    };
    void poll();
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [jobId]);

  const running = !!job && !job.dry_run && !_JOB_TERMINAL.has(job.status);
  const tone = !job ? "running" : job.dry_run ? "queued" : job.status === "succeeded" || job.status === "done" ? "done" : job.status === "failed" || job.status === "error" ? "failed" : "running";
  const statusText = !job ? "polling…" : job.dry_run ? "dry-run (recorded)" : job.status;
  const progress = typeof job?.progress === "number" ? job.progress : running ? -1 : 100;

  return (
    <div className="dialog-backdrop" role="presentation" onClick={closeOnBackdrop(onClose)}>
      <section ref={dialogRef} className="job-monitor" role="dialog" aria-modal="true">
        <header className="job-monitor-head">
          <div className="job-monitor-title">
            <Activity size={18} className={running ? "spin" : ""} />
            <div>
              <h2>{job?.title ?? "Job"}</h2>
              <code>{jobId}{job ? ` · ${job.kind}` : ""}</code>
            </div>
          </div>
          <span className={`job-status ${tone}`}>{statusText}</span>
          <button className="icon-only" onClick={onClose} aria-label="Close"><X size={16} /></button>
        </header>

        {progress >= 0 ? <Progress value={progress} /> : <div className="job-monitor-indeterminate"><span /></div>}
        {error && <div className="inline-error"><AlertCircle size={15} /> {error}</div>}

        {job && job.steps.length > 0 && (
          <div className="job-monitor-steps">
            {job.steps.map((step) => (
              <div className={`job-step ${step.status}`} key={step.order}>
                <span className="job-step-icon">
                  {step.status === "succeeded" || step.status === "done" ? <CheckCircle2 size={14} /> : step.status === "failed" ? <AlertCircle size={14} /> : step.status === "running" ? <Activity size={14} className="spin" /> : <Circle size={14} />}
                </span>
                <strong>{step.title}</strong>
                <code>{step.command}</code>
              </div>
            ))}
          </div>
        )}

        {job && job.note && <p className="fit-note">{job.note}</p>}

        <div className="job-monitor-log">
          <div className="job-monitor-log-head"><Terminal size={13} /> Log{running ? " · live" : ""}<CopyButton value={(job?.log ?? []).join("\n")} label="job log" /></div>
          <pre className="code-block">{(job?.log ?? ["(waiting for output…)"]).join("\n")}</pre>
        </div>

        <div className="job-monitor-foot">
          <span className="muted">{running ? "Polling every 1.5s until the job finishes." : job?.dry_run ? "Dry-run — start the daemon with --enable-apply to execute." : "Job finished."}</span>
          <button className="primary-action" onClick={onClose}>Close</button>
        </div>
      </section>
    </div>
  );
}

function EventLog({ events }: { events: Array<[string, string, string]> }) {
  return (
    <section className="event-log">
      <div className="event-list">
        {events.map(([time, tone, text], index) => (
          <div className="event-row" key={`${index}-${time}`}>
            <span>{time}</span>
            <em className={tone}>{tone}</em>
            <p>{text}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

function CliMirror({ lines }: { lines: string[] }) {
  return (
    <section className="cli-mirror">
      <CodeBlock lines={lines} title="CLI mirror" />
    </section>
  );
}

function OperationalConsole({
  activeTab,
  setActiveTab,
  selectedPreset,
  presetCount,
  gates,
  events,
  lines,
  onMonitor
}: {
  activeTab: ConsoleTab;
  setActiveTab: (tab: ConsoleTab) => void;
  selectedPreset: string;
  presetCount: number;
  gates: Gate[];
  events: Array<[string, string, string]>;
  lines: string[];
  onMonitor?: (id: string) => void;
}) {
  const blockedGates = gates.filter((gate) => gate.status === "blocked");
  const warnGates = gates.filter((gate) => gate.status === "warning");
  const logLines = [
    `[catalog] registry loaded ${presetCount} presets`,
    `[planner] dry-run launch plan ready for ${selectedPreset}`,
    `[doctor] ${gates.filter((gate) => gate.status === "pass").length}/${gates.length} readiness gates passing`,
    ...warnGates.map((gate) => `[gate] warning: ${gate.label} — ${gate.detail}`),
    ...blockedGates.map((gate) => `[gate] blocked: ${gate.label} — ${gate.detail}`)
  ];

  return (
    <section className="operational-console">
      <div className="console-tabs unified">
        <button className={activeTab === "jobs" ? "active" : ""} onClick={() => setActiveTab("jobs")}>Jobs</button>
        <button className={activeTab === "events" ? "active" : ""} onClick={() => setActiveTab("events")}>Events</button>
        <button className={activeTab === "logs" ? "active" : ""} onClick={() => setActiveTab("logs")}>Logs</button>
        <button className={activeTab === "cli" ? "active" : ""} onClick={() => setActiveTab("cli")}>CLI Mirror</button>
      </div>
      {activeTab === "jobs" && <JobsTable onMonitor={onMonitor} />}
      {activeTab === "events" && <EventLog events={events} />}
      {activeTab === "logs" && (
        <section className="cli-mirror">
          <CodeBlock lines={logLines} title="Logs" />
        </section>
      )}
      {activeTab === "cli" && <CliMirror lines={lines} />}
    </section>
  );
}

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

function buildRuntimeDraft(
  composed: Record<string, unknown>,
  runtimeTarget: string,
  patchPolicy: string
): RuntimeConfigDraft {
  return {
    max_model_len: asNumber(composed.max_model_len) || 32768,
    max_num_seqs: asNumber(composed.max_num_seqs) || 1,
    max_num_batched_tokens: asNumber(composed.max_num_batched_tokens) || 4096,
    gpu_memory_utilization: asNumber(composed.gpu_memory_utilization) || 0.9,
    enable_chunked_prefill:
      typeof composed.enable_chunked_prefill === "boolean" ? composed.enable_chunked_prefill : true,
    enforce_eager: typeof composed.enforce_eager === "boolean" ? composed.enforce_eager : false,
    disable_custom_all_reduce:
      typeof composed.disable_custom_all_reduce === "boolean" ? composed.disable_custom_all_reduce : true,
    kv_cache_dtype: asText(composed.kv_cache_dtype, "auto"),
    spec_decode_method: asText(composed.spec_decode_method, "none"),
    spec_decode_K: asNumber(composed.spec_decode_K),
    runtime_target: runtimeTarget,
    patch_policy: patchPolicy
  };
}

function buildDraftYaml(presetId: string, d: RuntimeConfigDraft): string[] {
  return [
    `# Draft runtime overlay for ${presetId}`,
    `runtime: ${d.runtime_target}`,
    `patch_policy: ${d.patch_policy}`,
    `sizing_override:`,
    `  max_model_len: ${d.max_model_len}`,
    `  max_num_seqs: ${d.max_num_seqs}`,
    `  max_num_batched_tokens: ${d.max_num_batched_tokens}`,
    `  gpu_memory_utilization: ${d.gpu_memory_utilization.toFixed(2)}`,
    `  enable_chunked_prefill: ${d.enable_chunked_prefill}`,
    `  enforce_eager: ${d.enforce_eager}`,
    `  disable_custom_all_reduce: ${d.disable_custom_all_reduce}`,
    `capabilities:`,
    `  kv_cache_dtype: ${d.kv_cache_dtype}`,
    `  spec_decode:`,
    `    method: ${d.spec_decode_method || "none"}`,
    `    num_speculative_tokens: ${d.spec_decode_K}`
  ];
}

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

const DRAFT_FIELD_LABELS: Record<string, string> = {
  max_model_len: "Max context",
  max_num_seqs: "Max sequences",
  max_num_batched_tokens: "Max batched tokens",
  gpu_memory_utilization: "GPU memory util",
  enable_chunked_prefill: "Chunked prefill",
  enforce_eager: "Enforce eager",
  disable_custom_all_reduce: "Disable custom all-reduce",
  kv_cache_dtype: "KV cache dtype",
  spec_decode_method: "Spec method",
  spec_decode_K: "Spec K",
  runtime_target: "Runtime target",
  patch_policy: "Patch policy"
};

function runtimeDraftDiff(base: RuntimeConfigDraft, draft: RuntimeConfigDraft) {
  const rows: string[] = [];
  (Object.keys(base) as Array<keyof RuntimeConfigDraft>).forEach((key) => {
    if (base[key] !== draft[key]) {
      rows.push(`${DRAFT_FIELD_LABELS[key] ?? key}: ${base[key]} → ${draft[key]}`);
    }
  });
  return rows;
}

function patchLifecycleExplanation(lifecycle: string) {
  const explanations: Record<string, string> = {
    stable: "Stable patches are expected to be safe in normal production profiles and should appear in release reports.",
    experimental: "Experimental patches need explicit evidence before they can be treated as a safe default.",
    research: "Research patches document an idea or investigation path and should stay out of automatic launch plans.",
    retired: "Retired patches remain visible for audit history but should not be proposed for new runtime plans.",
    qa: "QA patches are validation or test-oriented entries; expose them for diagnostics, not routine launch."
  };
  return explanations[lifecycle] ?? "Lifecycle is defined by the registry and should be reviewed before enabling this patch.";
}

function patchDefaultExplanation(value: string) {
  const explanations: Record<string, string> = {
    applied: "Default-on with a real apply module. The GUI can include it in launch summaries and patch proof.",
    marker: "Default-on marker without runtime effect. The GUI must label it clearly so operators do not assume code changed.",
    "opt-in": "Disabled by default. It should require explicit operator selection and a fresh plan before Apply is available.",
    blocked: "Blocked for production use because implementation or lifecycle state is not safe enough for automatic enablement."
  };
  return explanations[value] ?? "Default behavior is registry-defined and should be treated conservatively.";
}

// asRecord / asText / asNumber / asStringArray extracted to ./lib/coerce.

// shortWorkload / formatTokens / formatVram extracted to ./lib/format.
