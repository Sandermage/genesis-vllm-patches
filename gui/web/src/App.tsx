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
  Languages,
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
import { type SectionId, type RuntimeMode, type Gate, NAV_SECTIONS } from "./nav";
import {
  type ConsoleTab, type GuiSettings,
  nextTheme, themeLabel, themeIcon, DEFAULT_REMOTE_HOST,
  GUI_SETTINGS_STORAGE_KEY, loadGuiSettings
} from "./settings";
import { asRecord, asText, asNumber, countRecord } from "./lib/coerce";
import { formatTokens, targetTitle } from "./lib/format";
import { lsSet } from "./lib/safe-storage";
import { buildReadinessGates, countGates } from "./lib/readiness-gates";
import { useLiveEvents } from "./hooks/useLiveEvents";
import { LayerEditor } from "./sections/layer-editor";
import { ConfigDraftEditor } from "./sections/config-draft-editor";
import { ModelsWorkbench } from "./sections/models-workbench";
import { CapabilityTable } from "./components/capability-table";
import { PlanChip, KeyValue, ArtifactPreview, type ArtifactTab } from "./components/display-bits";
import { Step, Metric, PanelHeader, TabIntro, CodeTabs } from "./components/shell-bits";
import { ServerSwitcher, ConnectionMap } from "./sections/connection-bar";
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
import { GateRow } from "./sections/gate-row";
import { SetupWizard } from "./sections/setup-wizard";
import { JobMonitorModal, QueueJobButton } from "./sections/jobs";
import { CommandPalette } from "./sections/command-palette";
import { EventLog, OperationalConsole } from "./sections/operational-console";
import { LaunchPanel } from "./sections/launch-panel";
import { PresetPolicyGraph } from "./sections/preset-insight";
import { PatchSummaryPanel, PatchLifecycleGraph, PatchRegistryInsight, PatchModelSupport } from "./sections/patch-overview";
import { PresetRecommendPanel } from "./sections/preset-recommend";
import { type RecommendForm, defaultRecommend, workloadChoices } from "./recommend";
import { TabbedSection } from "./components/tabbed-section";
import { PresetQuickPanel } from "./sections/preset-quick";
import { PresetCatalogTable } from "./sections/preset-catalog";
import { ProofStatusPanel } from "./sections/proof";
import { CodeBlock } from "./components/code-block";
import { SkeletonCards } from "./Skeleton";
import { useViewport, type ViewportTier } from "./hooks/useViewport";
import { useLang, t, tr } from "./i18n";
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
import { UpdatesPanel } from "./Updates";
const KvCalcPanel = lazy(() => import("./Planner").then((m) => ({ default: m.KvCalcPanel })));
const BaselinePanel = lazy(() => import("./Planner").then((m) => ({ default: m.BaselinePanel })));
const InstallWizard = lazy(() => import("./Installer").then((m) => ({ default: m.InstallWizard })));
const CopilotPanel = lazy(() => import("./Copilot").then((m) => ({ default: m.CopilotPanel })));
import { FleetPanel } from "./Fleet";
// Lazy-loaded: the container management UI (~1.2k lines) only renders on the
// Containers section, so it is code-split out of the initial bundle.
const ContainersPanel = lazy(() => import("./Containers").then((m) => ({ default: m.ContainersPanel })));
const VirtualizationPanel = lazy(() => import("./sections/virtualization").then((m) => ({ default: m.VirtualizationPanel })));

// EN/RU language switch — flips the new bilingual surfaces (Virtualization, nav)
// live and persists the choice. Self-contained so it needs no App state.
function LangToggle() {
  const [lang, set] = useLang();
  return (
    <button className="tool-button" title={lang === "en" ? "Язык: English → Русский" : "Language: Русский → English"}
      onClick={() => set(lang === "en" ? "ru" : "en")}>
      <Languages size={16} /> {lang === "en" ? "EN" : "RU"}
    </button>
  );
}
// Lazy-loaded: the GPU/hardware telemetry dashboard only renders on its section.
const HardwarePanel = lazy(() => import("./Hardware").then((m) => ({ default: m.HardwarePanel })));
const RoutingPanel = lazy(() => import("./Routing").then((m) => ({ default: m.RoutingPanel })));
const FlagsPanel = lazy(() => import("./Flags").then((m) => ({ default: m.FlagsPanel })));
const LicensePanel = lazy(() => import("./License").then((m) => ({ default: m.LicensePanel })));

type LoadState = "idle" | "loading" | "ready" | "error";

type NavItem = {
  id: SectionId;
  icon: ReactNode;
  label: string;
};


const AUTO_REFRESH_INTERVAL_MS = 20_000;


// Sidebar grouped into a logical workflow: see → your servers → define what to
// run → deploy it → use it → prove it → tools. Each group renders under a small
// header so related sections sit together instead of in one long scatter.
type NavGroup = { label?: string; items: NavItem[] };
const navGroups: NavGroup[] = (() => {
  // Derived from the single-source NAV_SECTIONS registry (nav.ts): group by the
  // group label, preserving first-appearance order. "" = the ungrouped lead item.
  const order: string[] = [];
  const byGroup = new Map<string, NavItem[]>();
  for (const s of NAV_SECTIONS) {
    if (!byGroup.has(s.group)) { byGroup.set(s.group, []); order.push(s.group); }
    const Icon = s.icon;
    byGroup.get(s.group)?.push({ id: s.id, icon: <Icon size={17} />, label: tr(s.label) });
  }
  return order.map((g) => ({ label: g ? tr(g) : undefined, items: byGroup.get(g) ?? [] }));
})();
// Flat list (command palette / lookups) — preserves the grouped order.
const navItems: NavItem[] = navGroups.flatMap((g) => g.items);

// Hash-routing helpers (sectionFromHash / recordIdFromHash / buildHash /
// replaceHash) live in ./route so ContainersPanel can share them without a
// circular import. sectionFromHash returns a plain string; callers cast to
// SectionId after the SECTION_IDS validation guarantees membership.

export default function App() {
  const [navLang] = useLang();
  const viewport = useViewport();
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
  // No hard-coded default preset id — a stale one (e.g. an unshipped
  // "prod-35b-multiconc") would 404 on a deep link. loadAll() seeds the real
  // first catalog/recommended preset once the catalog loads.
  const [selectedPreset, setSelectedPreset] = useState<string>(() => recordIdFromHash() ?? "");
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
          host: runtimeHost(runtimeMode, settings.remoteHost),
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
          host: runtimeHost(runtimeMode, settings.remoteHost),
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
          host: runtimeHost(mode, settings.remoteHost),
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
        host: runtimeHost(runtimeMode, settings.remoteHost),
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
      toast(`${tr("Benchmark queued for")} ${selectedPreset}`, "success");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      toast(tr("Benchmark failed to queue"), "error");
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
      toast(`${tr("Evidence attach queued for")} ${selectedPreset}`, "success");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      toast(tr("Evidence attach failed to queue"), "error");
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
    toast(`${tr("No SNDR daemon at")} ${hostLabel(url)}. ${tr("If that host runs the vLLM engine, use its card's SSH / Discover / Terminal / Chat instead — not a daemon connection.")}`, "error");
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
        toast(`${tr("Daemon at")} ${hostLabel(cur)} ${tr("is unreachable — switched back to the local daemon.")}`, "info");
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
    toast(`${tr("Chat")} → ${profile.label} (${profile.host}:${profile.engine_port || 8000})`, "info");
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
        `${tr("No SNDR daemon reachable at")} ${profile.host}:${port}. ` +
        `${tr("Check the daemon port (8765 by default, not the engine port")} ${profile.engine_port || 8000}), ` +
        `${tr("that the node daemon is running, and that it allows this origin — or \"Set up as node\" below.")}`,
        "error",
      );
      return false;
    }
    switchServer(url);  // the host is already in the registry — just connect
    toast(`${tr("Connected to")} ${profile.label} ${tr("daemon")} (${url})`, "success");
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
    lsSet(GUI_SETTINGS_STORAGE_KEY, JSON.stringify(settings));
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
      if (!document.hidden) void loadAll();
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
  const endpointHost = runtimeHost(runtimeMode, settings.remoteHost);
  const connectionTone: "success" | "warning" | "danger" =
    state === "error" ? "danger" : state === "loading" ? "warning" : "success";
  const connectionLabel =
    state === "error" ? tr("Disconnected") : state === "loading" ? tr("Connecting") : tr("Connected");
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
      <main className={shellClass} data-viewport={viewport.tier}>
        <div className="login-backdrop">
          <div className="auth-loading"><span className="auth-spinner" /> {tr("Connecting…")}</div>
        </div>
      </main>
    );
  }

  // Auth gate: when the daemon requires authentication and there is no active
  // session, the whole shell is replaced by the sign-in screen.
  if (authState?.auth_required && !authState.user) {
    return (
      <main className={shellClass} data-viewport={viewport.tier}>
        <LoginScreen status={authState} onAuthenticated={onAuthenticated} />
      </main>
    );
  }

  return (
    <main className={shellClass} data-viewport={viewport.tier}>
      <aside className={`sidebar${settings.sidebarCollapsed ? " collapsed" : ""}`}>
        <div className="brand-row">
          <div className="brand-mark">S</div>
          <div>
            <strong>SNDR</strong>
            <span>{tr("Control Center")}</span>
          </div>
          <small>v{overview?.capabilities.platform.sndr_core_version ?? environment?.sndr_core_version ?? "—"}</small>
          <button
            className="sidebar-toggle"
            title={settings.sidebarCollapsed ? tr("Expand sidebar") : tr("Collapse sidebar")}
            aria-label={settings.sidebarCollapsed ? tr("Expand sidebar") : tr("Collapse sidebar")}
            aria-pressed={settings.sidebarCollapsed}
            onClick={() => updateSettings({ sidebarCollapsed: !settings.sidebarCollapsed })}
          >
            <PanelLeft size={16} />
          </button>
        </div>

        <nav className="side-nav" aria-label={tr("SNDR sections")}>
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
                  <span>{t(navLang, `nav.${item.id}`, item.label)}</span>
                </button>
              ))}
            </div>
          ))}
        </nav>

        <div className="daemon-card" title={`${tr("API Daemon")} · ${apiBase} · SNDR Core v${environment?.sndr_core_version ?? "—"} · ${runtimeMode === "remote" ? tr("Remote Desktop") : tr("Local Server")}`}>
          <div className="daemon-line">
            <span className="live-dot" />
            <strong>{tr("API Daemon")}</strong>
            <StatusBadge status={state === "error" ? "missing" : state === "loading" ? "partial" : "available"} />
          </div>
          <p>{apiBase}</p>
          <div className="daemon-meta">
            <span>SNDR Core v{environment?.sndr_core_version ?? "—"}</span>
            <span>{tr("Engine")}: {environment ? (environment.engine_version ? `vLLM ${environment.engine_version}` : tr("vLLM not installed")) : "…"}</span>
            <span>{tr("Mode")}: {runtimeMode === "remote" ? tr("Remote Desktop") : tr("Local Server")} · {tr("Read-only")}</span>
          </div>
          <button className="ghost-button daemon-docs" title={tr("Open API docs")} onClick={() => window.open(`${apiBase.replace(/\/$/, "")}/docs`, "_blank", "noopener,noreferrer")}>
            <FileText size={14} /> <span className="daemon-docs-label">{tr("View API Docs")}</span>
          </button>
        </div>
      </aside>

      <section className="main-shell">
        <header className="topbar">
          <div className="topbar-left">
            <div className="mode-toggle" aria-label={tr("GPU target")} title={tr("Where the GPU engine runs (launch/SSH/chat host hints). To change which daemon the GUI connects to, use the server switcher on the right.")}>
              <button
                className={runtimeMode === "local" ? "active" : ""}
                onClick={() => setRuntimeMode("local")}
                title={tr("GPU target: this machine")}
              >
                <Monitor size={16} />
                {tr("Local GPU")}
              </button>
              <button
                className={runtimeMode === "remote" ? "active" : ""}
                onClick={() => setRuntimeMode("remote")}
                title={tr("GPU target: a remote node (via SSH)")}
              >
                <ShieldCheck size={16} />
                {tr("Remote GPU")}
              </button>
            </div>
            <StatusPill tone={connectionTone}>{connectionLabel}</StatusPill>
            {runtimeMode === "remote" ? (
              <input
                className="host-title host-title-input"
                value={settings.remoteHost}
                onChange={(e) => updateSettings({ remoteHost: e.target.value })}
                onBlur={(e) => updateSettings({ remoteHost: e.target.value.trim() || DEFAULT_REMOTE_HOST })}
                spellCheck={false}
                aria-label={tr("Remote engine host")}
                title={tr("Engine host for Remote GPU mode — set the address of your GPU node (e.g. 192.168.1.10)")}
                placeholder={DEFAULT_REMOTE_HOST}
              />
            ) : (
              <span className="host-title">{endpointHost}</span>
            )}
            <span className="host-spec">{selectedPresetRecord?.hardware ?? recommendForm.hardware}</span>
            <span className="host-spec">{engineReady ? tr("vLLM engine ready") : tr("Engine not installed")}</span>
          </div>

          <div className="topbar-actions">
            <AlertsBell onOpenHardware={() => setActiveSection("hardware")} />
            <ServerSwitcher apiBase={apiBase} connectionTone={connectionTone} onSwitch={(url) => void switchServerGuarded(url)} hostProfiles={hostProfiles} onManageHosts={() => setActiveSection("hosts")} onOpenHost={(id) => { setFocusHostId(id); setActiveSection("hosts"); }} />
            <button className="tool-button" onClick={() => void loadAll()}>
              <RefreshCw size={16} />
              {tr("Sync Catalog")}
            </button>
            <LangToggle />
            <button
              className="tool-button"
              title={`${tr("Theme")}: ${themeLabel(settings.theme)} — ${tr("switch to")} ${themeLabel(nextTheme(settings.theme))}`}
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
              {tr("Settings")}
            </button>
            <button
              className="tool-button"
              onClick={() => setCommandOpen(true)}
            >
              <Command size={16} />
              {tr("Command")}
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
              <strong>{tr("Can't reach a SNDR daemon at")} {apiBase}</strong>
              <span>{tr("This is where patches, presets and the patcher version come from. A GPU server runs vLLM")} <em>{tr("engines")}</em>{tr(", not the daemon — manage it from")} <b>{tr("Hosts")}</b> {tr("(SSH check · Discover · Terminal · Chat). Point the GUI back at a running daemon to see project data.")}</span>
            </div>
            <button className="primary-action" onClick={() => switchServer(window.location.origin)}>
              <Home size={15} /> {tr("Use local daemon")}
            </button>
          </section>
        )}

        <SectionErrorBoundary section={activeSection}>
        {activeSection === "launch-plan" ? (
          <section className="section-workspace section-launch-plan">
        <header className="section-heading">
          <div>
            <span>{tr("Operator workbench")}</span>
            <h1>{tr("Launch Plan")}</h1>
            <p>{tr("Recommend a preset, compose the runtime, clear gates, then launch — plan-before-apply with a live job console.")}</p>
          </div>
          <div className="section-actions">
            <button className="tool-button" onClick={() => void refreshLaunchPlan()}>
              <RefreshCw size={16} /> {tr("Re-run Gates")}
            </button>
            <button className="tool-button" onClick={() => void loadAll()}>
              <RefreshCw size={16} /> {tr("Sync")}
            </button>
          </div>
        </header>
        <section className="process-strip" aria-label={tr("Launch process")}>
          <Step
            number="1"
            title={tr("Choose Preset")}
            detail={selectedPreset || recommendForm.workload.replace(/_/g, " ")}
            state="done"
            active={launchTab === "recommend"}
            onClick={() => setLaunchTab("recommend")}
          />
          <Step
            number="2"
            title={tr("Configure")}
            detail={`${targetTitle(runtimeTargets, runtimeTarget)} · ${patchPolicy}`}
            state="done"
            active={launchTab === "compose"}
            onClick={() => setLaunchTab("compose")}
          />
          <Step
            number="3"
            title={tr("Review & Launch")}
            detail={gateCounts.blocked > 0 ? `${gateCounts.blocked} ${tr("blocked")}` : applyEnabled ? tr("ready to launch") : tr("read-only")}
            state={gateCounts.blocked > 0 ? "warning" : "active"}
            active={launchTab === "launch"}
            onClick={() => setLaunchTab("launch")}
          />
          <Step
            number="4"
            title={tr("Observe")}
            detail={launchJob ? `${tr("job")} ${launchJob.status}` : `${evidenceRefs.length} ${tr("evidence refs")}`}
            state={launchJob ? "done" : "idle"}
            active={launchTab === "console"}
            onClick={() => setLaunchTab("console")}
          />
        </section>

        <section className="metric-strip">
          <Metric icon={<Database size={18} />} label={tr("Presets")} value={overview?.catalog.presets_count ?? "-"} />
          <Metric icon={<Cpu size={18} />} label={tr("Models")} value={overview?.catalog.models_count ?? "-"} />
          <Metric icon={<GitBranch size={18} />} label={tr("Profiles")} value={overview?.catalog.profiles_count ?? "-"} />
          <Metric
            icon={<Sparkles size={18} />}
            label={tr("Product API")}
            value={state === "loading" ? tr("Loading") : state === "error" ? tr("Error") : tr("Ready")}
          />
          <Metric
            icon={<Gauge size={18} />}
            label={primaryMetricKind}
            value={primaryMetricValue > 0 ? primaryMetricValue : tr("Pending")}
          />
        </section>

        <TabbedSection
          id="launch-plan"
          activeTab={launchTab}
          onTabChange={setLaunchTab}
          tabs={[
            {
              id: "recommend",
              label: `1 · ${tr("Choose")}`,
              icon: <SlidersHorizontal size={15} />,
              render: () => (
          <section className="panel builder-panel">
            <PanelHeader
              label="A."
              title={tr("Preset Recommendation Builder")}
              action={`${recommend?.total_matches ?? 0} ${tr("matches")}`}
              icon={<SlidersHorizontal size={18} />}
            />

            <div className="builder-section">
              <span className="section-index">1.</span>
              <div>
                <h3>{tr("Workload")}</h3>
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
                <span>{tr("Hardware Target")}</span>
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
                  <strong>{tr("Current host")}</strong>
                  <span>{runtimeMode === "remote" ? tr("Remote GPU node") : tr("Local workstation")}</span>
                </div>
              </div>
            </div>

            <div className="builder-section constraints-row">
              <span className="section-index">3.</span>
              <label className="field compact">
                <span>{tr("Concurrency")}</span>
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
                <span>{tr("Result Count")}</span>
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
                <span>{tr("Public evidence")}</span>
                <button
                  type="button"
                  className={recommendForm.preferPublic ? "toggle active" : "toggle"}
                  aria-label={tr("Prefer public evidence")}
                  aria-pressed={recommendForm.preferPublic}
                  onClick={() =>
                    setRecommendForm({ ...recommendForm, preferPublic: !recommendForm.preferPublic })
                  }
                />
              </div>
              <div className="toggle-field">
                <span>{tr("Safe patch policy")}</span>
                <button
                  type="button"
                  className={patchPolicy === "safe" ? "toggle active" : "toggle"}
                  aria-label={tr("Safe patch policy")}
                  aria-pressed={patchPolicy === "safe"}
                  onClick={() => setPatchPolicy(patchPolicy === "safe" ? "aggressive" : "safe")}
                />
              </div>
              <button className="primary-action" onClick={() => void runRecommend()}>
                <Search size={16} />
                {tr("Recalculate")}
              </button>
            </div>

            <div className="builder-section table-section">
              <span className="section-index">4.</span>
              <div className="recommend-table-wrap">
                <div className="table-toolbar">
                  <h3>{tr("Recommendation Results")}</h3>
                  <label className="search-box">
                    <Search size={15} />
                    <input
                      aria-label={tr("Search presets")}
                      value={query}
                      onChange={(event) => setQuery(event.target.value)}
                      placeholder={tr("Search preset, model, family")}
                    />
                  </label>
                </div>
                <table className="recommend-table">
                  <thead>
                    <tr>
                      <th>{tr("Preset")}</th>
                      <th>{tr("Model Family")}</th>
                      <th>{tr("Mode")}</th>
                      <th>{tr("Status")}</th>
                      <th>{tr("Allowed Workloads")}</th>
                      <th>{tr("Evidence")}</th>
                      <th>{tr("Fallback")}</th>
                      <th>{tr("Risk")}</th>
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
                  <div className="empty-state">{tr("No recommendation results for this query.")}</div>
                )}
                <div className="table-footer">
                  <span>
                    {tr("Showing")} {(recommend?.results ?? []).length} {tr("of")} {recommend?.total_candidates ?? 0} {tr("candidates")}
                  </span>
                  <button className="ghost-button" onClick={() => setActiveSection("presets")}>{tr("View All Presets")}</button>
                </div>
              </div>
            </div>
          </section>

              )
            },
            {
              id: "compose",
              label: `2 · ${tr("Configure")}`,
              icon: <SlidersHorizontal size={15} />,
              render: () => (
          <section className="panel composer-panel">
            <PanelHeader
              label="B."
              title={tr("Launch Plan Composer")}
              action={`${tr("Plan ID")}: ${planId}`}
              icon={<Rocket size={18} />}
            />

            <div className="composer-grid">
              <section>
                <div className="plan-flow">
                  <PlanChip label={tr("Selected Preset")} value={selectedPreset} />
                  <ChevronRight size={18} />
                  <PlanChip label={tr("Model")} value={selectedPresetRecord?.model ?? "-"} />
                  <ChevronRight size={18} />
                  <PlanChip label={tr("Hardware")} value={selectedPresetRecord?.hardware ?? recommendForm.hardware} />
                  <ChevronRight size={18} />
                  <PlanChip label={tr("Profile")} value={selectedPresetRecord?.profile ?? "-"} />
                  <ChevronRight size={18} />
                  <PlanChip label={tr("Baseline")} value={primaryMetricValue > 0 ? `${primaryMetricValue.toLocaleString()} ${primaryMetricKind.replace(/^agg_/, "")}` : tr("pending")} />
                </div>

                <div className="option-block">
                  <h3>{tr("Runtime Target")}</h3>
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
                  <h3>{tr("Patch Policy")}</h3>
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
                      {tr("Strict image digest and dry-run mode are enabled for GUI preview.")}
                    </span>
                  </div>
                </div>

                <div className="option-block">
                  <h3>{tr("Launch Target")}</h3>
                  <label className="field">
                    <span>{tr("SSH target — empty = local execution")}</span>
                    <input
                      value={launchSshTarget}
                      onChange={(event) => setLaunchSshTarget(event.target.value)}
                      placeholder="user@gpu-host"
                    />
                  </label>
                  <p className="policy-note">
                    {launchSshTarget.trim()
                      ? `${tr("Apply Launch will run over SSH on")} ${launchSshTarget.trim()} ${tr("(when --enable-apply).")}`
                      : tr("Apply Launch runs locally (when --enable-apply). Set an SSH target to launch on a remote host.")}
                  </p>
                </div>

                <ArtifactPreview
                  artifacts={launchPlan?.artifacts ?? []}
                  activeTab={artifactTab}
                  setActiveTab={setArtifactTab}
                />
              </section>

              <section className="plan-summary">
                <h3>{tr("Plan Summary")}</h3>
                <KeyValue label={tr("Preset")} value={selectedPreset} />
                <KeyValue label={tr("Model")} value={asText(planSummary.model, selectedPresetRecord?.model ?? "-")} />
                <KeyValue label={tr("Hardware")} value={selectedPresetRecord?.hardware ?? recommendForm.hardware} />
                <KeyValue label={tr("Runtime")} value={targetTitle(runtimeTargets, runtimeTarget)} />
                <KeyValue label={tr("Mode")} value={runtimeMode === "remote" ? tr("Remote SSH tunnel") : tr("Local web daemon")} />
                <KeyValue label={tr("Context")} value={formatTokens(asNumber(planSummary.context) || asNumber(composed.max_model_len))} />
                <KeyValue label={tr("Sequences")} value={String(asNumber(planSummary.max_num_seqs) || asNumber(composed.max_num_seqs) || "-")} />
                <KeyValue label={tr("KV cache")} value={asText(composed.kv_cache_dtype, "-")} />
                <KeyValue label={tr("Spec decode")} value={`${asText(composed.spec_decode_method, "-")} / K=${asText(composed.spec_decode_K, "-")}`} />
                <KeyValue label={tr("Patches")} value={asNumber(planSummary.enabled_patches_count) || asNumber(composed.enabled_patches_count) || "-"} />
                <KeyValue label={tr("Patch policy")} value={patchPolicy} />
                <KeyValue label={tr("Fallback")} value={asText(planSummary.fallback_preset, asText(card.fallback_preset, "-"))} />
                <KeyValue label={tr("Plan ID")} value={planId} />
                <button className="primary-action launch-continue" onClick={() => setLaunchTab("launch")}>
                  {tr("Review & Launch")}
                  <ChevronRight size={16} />
                </button>
                <p className="policy-note">{tr("Step 3 confirms readiness and starts the runtime.")}</p>
              </section>
            </div>
          </section>

              )
            },
            {
              id: "launch",
              label: `3 · ${tr("Launch")}`,
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
              label: tr("Gates"),
              icon: <ListChecks size={15} />,
              render: () => (
          <section className="panel gates-panel">
            <PanelHeader
              label="C."
              title={tr("Gates & Blockers")}
              action={`${gateCounts.pass} ${tr("ok")} / ${gateCounts.warning} ${tr("warn")} / ${gateCounts.blocked} ${tr("blocked")}`}
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
              {tr("Re-run All Gates")}
            </button>
          </section>

              )
            },
            {
              id: "console",
              label: tr("Console"),
              icon: <SquareTerminal size={15} />,
              render: () => (
          <section className="panel console-panel">
            <PanelHeader
              label="E."
              title={tr("Job and Event Console")}
              action={tr("Read-only mirror")}
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
              label: tr("Endpoints & Evidence"),
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
              onExplain={() => setDialog(tr("Patch policy matrix reflects the live registry: default-applied, marker-only, opt-in and blocked patches, plus the count enabled by the selected preset plan."))}
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
            viewport={viewport.tier}
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
          <span>{tr("Mode")} {runtimeMode === "remote" ? tr("Remote Desktop") : tr("Local Server")}</span>
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
            ...navItems.map((nav) => ({ icon: nav.icon, title: nav.label, detail: tr("Section"), run: () => setActiveSection(nav.id) })),
            ...(presets?.presets ?? []).slice(0, 80).map((preset) => ({
              icon: <Database size={16} />, title: preset.id, detail: `${tr("Preset")} · ${preset.model} · ${preset.hardware}`,
              run: () => { void loadExplain(preset.id); setActiveSection("presets"); }
            })),
            ...(configCatalog?.models ?? []).map((m) => ({
              icon: <Boxes size={16} />, title: m.title || m.id, detail: `${tr("Model")} · ${m.summary || m.id}`,
              run: () => setActiveSection("configs")
            })),
            ...(configCatalog?.hardware ?? []).map((h) => ({
              icon: <Server size={16} />, title: h.title || h.id, detail: `${tr("Hardware")} · ${h.summary || h.id}`,
              run: () => setActiveSection("configs")
            })),
            ...(configCatalog?.profiles ?? []).map((p) => ({
              icon: <SlidersHorizontal size={16} />, title: p.title || p.id, detail: `${tr("Profile")} · ${p.summary || p.id}`,
              run: () => setActiveSection("configs")
            })),
            ...(patches?.patches ?? []).map((p) => ({
              icon: <PackageCheck size={16} />, title: p.patch_id, detail: `${tr("Patch")} · ${p.title}`,
              run: () => setActiveSection("patches")
            }))
          ]}
        />
      )}
      {shortcutsOpen && <ShortcutsModal onClose={() => setShortcutsOpen(false)} />}
    </main>
  );
}

class SectionErrorBoundary extends Component<{ section: string; children: ReactNode }, { error: Error | null }> {
  override state: { error: Error | null } = { error: null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  override componentDidUpdate(prev: { section: string }) {
    if (prev.section !== this.props.section && this.state.error) {
      this.setState({ error: null });
    }
  }
  override render() {
    if (this.state.error) {
      return (
        <section className="section-workspace">
          <div className="error-boundary">
            <AlertCircle size={22} />
            <div>
              <strong>{tr("This panel hit a rendering error.")}</strong>
              <p>{this.state.error.message || tr("Unexpected error while rendering the panel.")}</p>
              <button type="button" className="ghost-button" onClick={() => this.setState({ error: null })}>
                <RefreshCw size={14} /> {tr("Retry")}
              </button>
            </div>
          </div>
        </section>
      );
    }
    return this.props.children;
  }
}

function SectionWorkspace({
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

function sectionSpec(sectionId: SectionId) {
  const specs: Record<
    SectionId,
    { kicker: string; title: string; description: string }
  > = {
    overview: {
      kicker: tr("System map"),
      title: tr("Overview"),
      description: tr("One screen summary of Product API health, catalog coverage, runtime targets and workload readiness."),
    },
    setup: {
      kicker: tr("First-run workflow"),
      title: tr("Setup"),
      description: tr("Local server and remote desktop setup path with explicit daemon, tunnel and gate stages."),
    },
    fleet: {
      kicker: tr("Multi-server overview"),
      title: tr("Fleet"),
      description: tr("Every registered GPU/engine host at a glance — one concurrent SSH sweep shows status, running model, vLLM version, GPUs and live patch count per server."),
    },
    hosts: {
      kicker: tr("Runtime inventory"),
      title: tr("Hosts"),
      description: tr("Local and remote host inventory, transport state, runtime tools and SSH tunnel commands."),
    },
    models: {
      kicker: tr("Model catalog"),
      title: tr("Models"),
      description: tr("Model families, hardware envelopes and composed runtime details from the V2 registry."),
    },
    configs: {
      kicker: tr("V2 config editor"),
      title: tr("Configs"),
      description: tr("Graphical editor for V2 model, hardware, profile and preset composition with safe draft preview."),
    },
    presets: {
      kicker: tr("Preset catalog"),
      title: tr("Presets"),
      description: tr("Full preset table with cards, workload policy, evidence visibility and selected explain payload."),
    },
    planner: {
      kicker: tr("Capacity & regression"),
      title: tr("Planner"),
      description: tr("KV-cache / VRAM fit calculator (GQA, MoE and tensor-parallel aware, calibratable) and quality-baseline regression diff."),
    },
    copilot: {
      kicker: tr("Read-only assistant"),
      title: tr("Ops Copilot"),
      description: tr("Tool-calling assistant over the read-only Product API — answers from real catalog/doctor/preset/patch/capacity data and proposes changes you review & apply."),
    },
    "launch-plan": {
      kicker: tr("Operator workbench"),
      title: tr("Launch Plan"),
      description: tr("Recommendation builder, plan composer, readiness gates, artifacts and CLI mirror."),
    },
    services: {
      kicker: tr("Lifecycle"),
      title: tr("Services"),
      description: tr("Service lifecycle, rendered launch artifacts, status, logs and safe write API boundary."),
    },
    containers: {
      kicker: tr("Docker control"),
      title: tr("Containers"),
      description: tr("Manage the vLLM/engine containers on a server — live CPU/memory, logs, start/stop/restart, and gated exec — over the local docker socket or a registered host via SSH."),
    },
    kubernetes: {
      kicker: tr("Cluster"),
      title: tr("Kubernetes"),
      description: tr("Read-only Kubernetes view — cluster status and nodes with GPU capacity/allocatable/requested, conditions, taints and labels. Honours your kubeconfig + RBAC."),
    },
    virtualization: {
      kicker: tr("Compute"),
      title: tr("Virtualization"),
      description: tr("Proxmox VE hosts & guests, KubeVirt VMs and Kubernetes nodes in one pane — each linked back to the SNDR preset it runs."),
    },
    hardware: {
      kicker: tr("GPU telemetry"),
      title: tr("GPU & Hardware"),
      description: tr("Live per-GPU utilisation, VRAM, temperature, power, clocks, fan, PCIe, pstate and ECC over nvidia-smi — for the daemon host or a registered host via SSH."),
    },
    routing: {
      kicker: tr("Spec-decode routing"),
      title: tr("Workload routing"),
      description: tr("Per bench-validated profile: which workloads are allowed/denied and their measured TPS delta — plus a classifier that predicts how a request's signals resolve to a profile. One source of truth, shared with the gateway."),
    },
    flags: {
      kicker: tr("Patch flags"),
      title: tr("Env-flag matrix"),
      description: tr("Every GENESIS_ENABLE_* flag with its default, searchable and filterable — overlay a running engine's live ON/OFF state and flag drift."),
    },
    doctor: {
      kicker: tr("Diagnostics"),
      title: tr("Doctor"),
      description: tr("Readiness gates, blockers, warnings and release-proof preflight diagnostics."),
    },
    patches: {
      kicker: tr("Patch control"),
      title: tr("Patches"),
      description: tr("Patch simulation, policy matrix, enabled patch count and safe/minimal/compact policy preview."),
    },
    benchmarks: {
      kicker: tr("Performance"),
      title: tr("Benchmarks"),
      description: tr("Benchmark baselines, expected TPS/TTFT context, run plan and evidence orchestration state."),
    },
    evidence: {
      kicker: tr("Proof bundle"),
      title: tr("Evidence"),
      description: tr("Evidence references, visibility, benchmark baseline status and future release report bundles."),
    },
    clients: {
      kicker: tr("Integrations"),
      title: tr("Clients"),
      description: tr("OpenAI-compatible endpoints, health/metrics URLs, client snippets and GUI/CLI integration modes."),
    },
    chat: {
      kicker: tr("Local model chat"),
      title: tr("Chat"),
      description: tr("Multi-turn streaming chat with any running local vLLM model — pick the engine host/port and model, tune the system prompt and sampling."),
    },
    reports: {
      kicker: tr("Operator reports"),
      title: tr("Reports"),
      description: tr("Launch, benchmark, patch and release-proof report types planned for GUI export."),
    },
    operations: {
      kicker: tr("Project workbench"),
      title: tr("Operations"),
      description: tr("Run sndr_core's canonical maintenance, audit and proof workflows as live-monitored jobs — the CLI surface, integrated."),
    },
    advanced: {
      kicker: tr("Developer surface"),
      title: tr("Advanced"),
      description: tr("API base, OpenAPI/schema, feature contracts, CLI mirror and future desktop settings."),
    }
  };
  return specs[sectionId];
}

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
    [stamp(0), state === "error" ? "error" : "info", state === "loading" ? tr("Refreshing Product API snapshot...") : `${tr("Selected preset")} ${selectedPreset}`],
    [stamp(2), "info", `${tr("Runtime target set to")} ${runtimeTarget}`],
    [stamp(4), visibility === "public" ? "info" : "warn", `${tr("Evidence visibility")}: ${visibility}`],
    [stamp(6), "info", tr("Catalog and capability surfaces loaded through typed Product API")]
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

function runtimeHost(mode: RuntimeMode, remoteHost: string = DEFAULT_REMOTE_HOST) {
  return mode === "remote" ? (remoteHost.trim() || DEFAULT_REMOTE_HOST) : "127.0.0.1";
}
