import { AlertCircle, ChevronRight, Command, Cpu, Database, FileText, Gauge, GitBranch, HardDrive, Home, Link2, Languages, Boxes, ListChecks, Monitor, PackageCheck, PanelLeft, PlugZap, RefreshCw, Rocket, Search, Server, Settings, ShieldCheck, SlidersHorizontal, Sparkles, SquareTerminal } from "lucide-react";
import { Component, Suspense, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { sectionFromHash, recordIdFromHash, buildHash, replaceHash } from "./route";
import { type SectionId, type RuntimeMode, NAV_SECTIONS } from "./nav";
import { type ConsoleTab, type GuiSettings, nextTheme, themeLabel, themeIcon, DEFAULT_REMOTE_HOST, GUI_SETTINGS_STORAGE_KEY, loadGuiSettings } from "./settings";
import { asRecord, asText, asNumber } from "./lib/coerce";
import { formatTokens, targetTitle } from "./lib/format";
import { lsSet } from "./lib/safe-storage";
import { buildReadinessGates, countGates } from "./lib/readiness-gates";
import { useLiveEvents } from "./hooks/useLiveEvents";
import { LiveModelChip } from "./components/live-model";
import { SectionWorkspace } from "./sections/section-workspace";
import { buildEvents, buildCliMirror, runtimeHost } from "./lib/overview-presenters";
import { type FetchState } from "./hooks/useApiQuery";
import { PlanChip, KeyValue, ArtifactPreview, type ArtifactTab } from "./components/display-bits";
import { Step, Metric, PanelHeader } from "./components/shell-bits";
import { ServerSwitcher } from "./sections/connection-bar";
import { StatusBadge, StatusPill } from "./components/primitives";
import { RuntimeEndpoint, BenchmarkCard, EvidenceCard, PatchMatrix } from "./sections/rail-cards";
import { InfoDialog, ShortcutsModal } from "./components/dialogs";
import { RecommendationRow } from "./sections/recommendation-row";
import { toast, ToastHost } from "./components/toast";
import { GateRow } from "./sections/gate-row";
import { JobMonitorModal } from "./sections/jobs";
import { CommandPalette } from "./sections/command-palette";
import { OperationalConsole } from "./sections/operational-console";
import { LaunchPanel } from "./sections/launch-panel";
import { type RecommendForm, defaultRecommend, workloadChoices } from "./recommend";
import { TabbedSection } from "./components/tabbed-section";
import { SkeletonCards } from "./Skeleton";
import { useViewport } from "./hooks/useViewport";
import { useLang, t, tr } from "./i18n";
import { BundleSpec, DiffUpstreamReport, DoctorReport, EnvironmentReport, HostProfile, Job, LaunchPlanResult, ProofStatusReport, UserPresetList, V2ConfigCatalog, V2ConfigPreview, PresetExplainResult, PatchDoctorReport, PatchListResult, PresetListResult, PresetRecommendResult, ProductOverview, AuthStatus, api, normalizeBaseUrl, hostLabel } from "./api";
import { AccountMenu, LoginScreen } from "./Auth";
// Heavy section panels are lazy()-loaded from ./lazy-panels and rendered inside
// SectionWorkspace under one Suspense boundary. `ChatTarget` is a type consumed
// by the chat target selector in the shell, so it stays a plain type import.
import type { ChatTarget } from "./Engine";
import { AlertsBell } from "./Alerts";

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
  const [state, setState] = useState<FetchState>("idle");
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
                title={tr("Engine host for Remote GPU mode — set the address of your GPU node (e.g. 192.0.2.10)")}
                placeholder={DEFAULT_REMOTE_HOST}
              />
            ) : (
              <span className="host-title">{endpointHost}</span>
            )}
            <span className="host-spec">{selectedPresetRecord?.hardware ?? recommendForm.hardware}</span>
            <span className="host-spec">{engineReady ? tr("vLLM engine ready") : tr("Engine not installed")}</span>
          </div>

          <div className="topbar-actions">
            <LiveModelChip onOpen={() => setActiveSection("clients")} />
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
