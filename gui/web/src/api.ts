import { lsGet, lsSet, lsRemove } from "./lib/safe-storage";

export type CapabilityStatus =
  | "available"
  | "partial"
  | "render_only"
  | "deferred"
  | "missing";

export type PlatformSnapshot = {
  public_brand: string;
  package_name: string;
  sndr_core_version: string;
  os_name: string;
  machine: string;
  python_version: string;
  engine_installed: boolean;
};

export type ProductCapability = {
  id: string;
  title: string;
  kind: string;
  status: CapabilityStatus;
  detail: string;
  required_tools: string[];
  present_tools: string[];
  module?: string | null;
};

export type ProductCapabilities = {
  platform: PlatformSnapshot;
  runtime_targets: ProductCapability[];
  features: ProductCapability[];
  warnings: string[];
};

export type CatalogSummary = {
  models_count: number;
  hardware_count: number;
  profiles_count: number;
  presets_count: number;
  preset_cards_count: number;
  unannotated_presets_count: number;
  preset_load_error_count: number;
  status_counts: Record<string, number>;
  workload_counts: Record<string, number>;
  family_counts: Record<string, number>;
  default_presets: string[];
  preset_load_errors: string[];
};

export type ProductOverview = {
  capabilities: ProductCapabilities;
  catalog: CatalogSummary;
};

export type PresetRecord = {
  id: string;
  model: string;
  hardware: string;
  profile: string | null;
  runtime: string | null;
  has_card: boolean;
  card: Record<string, any>;
};

export type PresetListResult = {
  filters: Record<string, string | null>;
  matched: number;
  total: number;
  presets: PresetRecord[];
  load_errors: Array<Record<string, string>>;
};

export type PresetRecommendation = {
  id: string;
  rank: number;
  model: string;
  hardware: string;
  profile: string | null;
  runtime: string | null;
  card: Record<string, any>;
};

export type PresetRecommendResult = {
  query: Record<string, any>;
  results: PresetRecommendation[];
  total_matches: number;
  total_candidates: number;
};

export type PresetExplainResult = {
  id: string;
  card: Record<string, any>;
  composed: Record<string, any>;
  fallback_diff: Record<string, any> | null;
};

export type LaunchPlanGate = {
  id: string;
  title: string;
  status: "pass" | "warning" | "blocked" | "planned";
  detail: string;
  action: string;
};

export type LaunchPlanEndpoint = {
  label: string;
  url: string;
};

export type LaunchPlanArtifact = {
  kind: "compose" | "systemd" | "commands" | "env";
  title: string;
  content: string;
};

export type LaunchPlanResult = {
  plan_id: string;
  preset_id: string;
  runtime_target: string;
  patch_policy: string;
  mode: string;
  host: string;
  actionable: boolean;
  action_reason: string;
  summary: Record<string, any>;
  gates: LaunchPlanGate[];
  endpoints: LaunchPlanEndpoint[];
  artifacts: LaunchPlanArtifact[];
  cli_mirror: string[];
  events: Array<Record<string, string>>;
};

export type V2ConfigItem = {
  id: string;
  kind: "model" | "hardware" | "profile" | "preset" | string;
  title: string;
  source: string;
  summary: string;
  status: string;
  parent_model?: string | null;
  model?: string | null;
  hardware?: string | null;
  profile?: string | null;
  runtime?: string | null;
  fields: Record<string, any>;
};

export type V2ConfigCatalog = {
  models: V2ConfigItem[];
  hardware: V2ConfigItem[];
  profiles: V2ConfigItem[];
  presets: V2ConfigItem[];
};

export type V2ConfigPreview = {
  selection: Record<string, string | null>;
  compatible: boolean;
  status: string;
  messages: string[];
  composed: Record<string, any>;
  draft_yaml: string;
};

export type V2ConfigPlan = {
  plan_id: string;
  preset_id: string;
  selection: Record<string, string | null>;
  target_path: string;
  backup_path?: string | null;
  action: string;
  read_only: boolean;
  apply_enabled: boolean;
  valid: boolean;
  blocked_reasons: string[];
  warnings: string[];
  steps: string[];
  diff_lines: string[];
  draft_yaml: string;
};

export type HostProfile = {
  id: string;
  label: string;
  host: string;
  transport: string;
  ssh_target: string;
  port: number;
  notes: string;
  role: string;
  hardware: string;
  gpus: number;
  engine_port: number;
  api_key?: string;        // write-only: sent when setting a key; never returned
  has_api_key?: boolean;   // read: whether an engine key is stored (value is server-side)
  ssh_user: string;
  ssh_auth: string;        // "agent" | "key" | "password"
  ssh_key_path: string;
  ssh_port: number;
  has_ssh_password?: boolean;
  gpu_vram_mib?: number;
  gpu_name?: string;
  gpu_arch?: string;
  interconnect?: string;
  tags: string[];
};

export type DiscoveredEngine = {
  container: string;
  host_port: number | null;
  image: string;
  status: string;
  ports: string;
  reachable?: boolean;
  version?: string | null;
  models?: string[];
  genesis_flags?: string[];  // active GENESIS_ENABLE_*=1 on the running container
};
export type DiscoveredGpu = { name: string; memory_total_mib: string; utilization: string | null; compute_cap?: number | null; arch?: string };
export type ArchAdvice = { arch: string; compute_cap: number | null; fp8_kv_native: boolean; fp8_weights_native: boolean; fp4_weights: boolean; flash_attention: number; recommendations: Array<{ level: string; text: string }> };
export type Interconnect = { has_nvlink: boolean; worst_link: string; note: string };
export type HostDiscovery = {
  available: boolean;
  docker: boolean;
  engines: DiscoveredEngine[];
  gpus: DiscoveredGpu[];
  error: string | null;
  engine_port_set?: number | null;
  arch_advice?: ArchAdvice;
  interconnect?: Interconnect;
};

export type HostModelConfig = {
  ok: boolean; error: string | null; model_path?: string; container?: string;
  num_layers?: number; num_attention_heads?: number; num_kv_heads?: number; head_dim?: number; hidden_size?: number;
  is_moe?: boolean; num_experts?: number; max_context?: number; sliding_window?: number; global_layers?: number | null;
  weights_bytes?: number | null; quant_method?: string; model_type?: string;
};

// A host's own sndr_core management identity (read from its container over SSH).
export type HostSndrState = {
  ok: boolean; container: string | null; vllm_version: string | null;
  sndr_version: string | null; configs: number | null; patches: number | null; error: string | null;
};

export type SshCheckResult = {
  available: boolean;
  ssh_ok: boolean;
  sftp_ok: boolean;
  latency_ms: number | null;
  banner: string | null;
  uname: string | null;
  error: string | null;
  forgot?: boolean;
};

export type HostProbe = {
  reachable: boolean;
  host: string;
  port: number;
  base_url: string;
  version: string | null;
  models: string[];
  latency_ms: number | null;
  error: string | null;
};

export type JobStep = { order: number; title: string; command: string; status: string; log: string };

export type Job = {
  job_id: string;
  kind: string;
  title: string;
  status: string;
  dry_run: boolean;
  created_at: number;
  summary: Record<string, any>;
  steps: JobStep[];
  log: string[];
  cli_mirror: string[];
  note: string;
  progress?: number | null;
};

export type DependencyInfo = { name: string; version: string | null; present: boolean; kind: string };

export type EnvironmentReport = {
  brand: string;
  package_name: string;
  sndr_core_version: string;
  engine_name: string;
  engine_version: string | null;
  engine_installed: boolean;
  python_version: string;
  os_name: string;
  machine: string;
  dependencies: DependencyInfo[];
  tools: DependencyInfo[];
  // Self-locating launch context — a restart command that works on the node the
  // daemon actually runs on (vs a static guess that fails from the wrong dir).
  python_executable?: string;
  install_root?: string | null;
  sndr_importable_globally?: boolean;
  restart_command?: string;
};

export type DeployTarget = {
  id: string;
  label: string;
  filename: string;
  kind: "yaml" | "ini" | "bash";
  needs: string;
  summary: string;
};

export type HostInventory = {
  os: { system: string; release: string; distro: string; arch: string };
  python: { binary_path: string; version: string; implementation: string; venv_active: boolean; pip_present: boolean; pip_version: string | null };
  docker: { installed: boolean; binary_path: string | null; version: string | null; daemon_running: boolean; server_version: string | null; nvidia_runtime_present: boolean; notes: string };
  nvidia: { installed: boolean; driver_version: string | null; cuda_version: string | null; n_gpus: number; gpu_names: string[]; gpu_total_vram_mib: number[]; notes: string };
  vllm: { installed: boolean; version: string | null; location: string | null };
};

export type DeployTargetsResult = { targets: DeployTarget[]; host: HostInventory };

// Rich live GPU + hardware telemetry (nvidia-smi), local daemon host or remote.
export type GpuInfo = {
  name: string | null; uuid: string | null; serial: string | null; driver_version: string | null; vbios_version: string | null;
  mem_used: number | null; mem_total: number | null; mem_free: number | null;
  gpu_util: number | null; mem_util: number | null;
  temp_gpu: number | null; temp_mem: string | null;
  power: number | null; power_default_limit: number | null; power_max_limit: number | null; power_min_limit: number | null;
  fan_speed: number | null;
  pcie_gen: number | null; pcie_gen_max: number | null; pcie_width: number | null; pcie_width_max: number | null;
  clock_gpu: number | null; clock_gpu_max: number | null; clock_mem: number | null; clock_mem_max: number | null; clock_sm: number | null;
  compute_mode: string | null; pstate: string | null;
  ecc_corrected: string | null; ecc_uncorrected: string | null;
};
export type NetInterface = { name: string; rx_bytes: number; tx_bytes: number };
export type DiskInfo = { mount: string; total_gb: number | null; used_gb: number | null; free_gb: number | null; used_pct?: number };
export type HardwareSystem = {
  hostname: string | null; cpu: string | null; cpu_count: number | null;
  ram_total_gb?: number; ram_available_gb?: number; ram_used_gb?: number;
  primary_ip?: string | null; net?: NetInterface[]; disk?: DiskInfo | null; platform?: string | null;
};
export type HardwareTelemetry = { gpus: GpuInfo[]; system: HardwareSystem; error: string | null };

export type K8sStatus = {
  available: boolean; error: string | null;
  version?: string | null; platform?: string | null;
  node_count?: number; nodes_ready?: number; gpu_node_count?: number; namespace_count?: number;
};
export type K8sTaint = { key: string | null; value: string | null; effect: string | null };
export type K8sNode = {
  name: string | null; ready: boolean; schedulable: boolean; roles: string[];
  kubelet_version: string | null; os_image: string | null;
  cpu_capacity: string | null; mem_capacity: string | null;
  gpu_capacity: number | null; gpu_allocatable: number | null; gpu_requested?: number; gpu_free?: number | null;
  pressures: string[]; taints: K8sTaint[]; gpu_labels: Record<string, string>; label_count: number;
};
export type K8sNodesResult = { available: boolean; error: string | null; nodes: K8sNode[] };
export type K8sPod = {
  name: string | null; namespace: string | null; node: string | null; phase: string | null;
  ready: string; ready_ok: boolean; restarts: number; gpu_request: number; reason: string | null; images: (string | null)[];
  // SNDR identity (present when the pod was rendered by `sndr k8s`).
  sndr_managed: boolean; sndr_preset: string | null; sndr_patch_count: number | null;
  sndr_pin: string | null; sndr_patches: string[];
};
export type K8sPodsResult = { available: boolean; error: string | null; pods: K8sPod[] };
export type K8sEvent = {
  type: string | null; reason: string | null; message: string | null;
  object: string | null; namespace: string | null; count: number | null;
};
export type K8sEventsResult = { available: boolean; error: string | null; events: K8sEvent[] };
export type KubeVirtVM = {
  name: string | null; namespace: string | null; kind: "kubevirt"; phase: string | null; running: boolean;
  node: string | null; cpu_cores: number | null; memory: string | null; ip: string | null;
  gpu_count: number; ready: boolean; sndr_preset: string | null;
};
export type KubeVirtResult = { available: boolean; installed?: boolean; error: string | null; vms: KubeVirtVM[] };
export type ProxmoxStatus = {
  available: boolean; configured?: boolean; error: string | null; host?: string;
  node_count?: number; nodes_online?: number; vm_count?: number; vm_running?: number;
  lxc_count?: number; lxc_running?: number; sndr_managed?: number;
};
export type ProxmoxNode = {
  name: string | null; status: string | null; online: boolean;
  cpu_pct: number | null; cpu_cores: number | null;
  mem_used: number | null; mem_total: number | null; mem_pct: number | null;
  disk_used: number | null; disk_total: number | null; disk_pct: number | null;
  uptime: number | null; level: string;
};
export type ProxmoxGuest = {
  vmid: number | null; name: string | null; kind: "vm" | "lxc" | string; status: string | null; running: boolean;
  node: string | null; cpu_pct: number | null; cpu_cores: number | null;
  mem_used: number | null; mem_total: number | null; mem_pct: number | null; disk_total: number | null;
  disk_used: number | null; net_in: number | null; net_out: number | null; disk_read: number | null; disk_write: number | null;
  uptime: number | null; tags: string[]; sndr_preset: string | null; template: boolean;
};
export type ProxmoxDisk = { id: string; volume: string; size: string | null; storage: string | null };
export type ProxmoxNet = { id: string; model: string | null; mac: string | null; bridge: string | null; ip: string | null; name: string | null };
export type ProxmoxDevice = { address: string; name: string; kind: "gpu" | "audio" | "usb" | "net" | "storage" | "pci" | string };
export type ProxmoxGuestDetail = {
  available: boolean; error: string | null; vmid: number; kind: string; node: string;
  cores: number | null; sockets: number | null; cpu_type: string | null;
  memory_mb: number | null; swap_mb: number | null; balloon: number | null;
  bios: string | null; machine: string | null; ostype: string | null;
  onboot: boolean; boot_order: string | null; agent_enabled: boolean;
  qmpstatus: string | null; ha_managed: boolean | null; unprivileged: boolean | null;
  features: string | null; description: string | null; tags: string[];
  devices: ProxmoxDevice[]; disks: ProxmoxDisk[]; networks: ProxmoxNet[]; agent_ips: string[];
};
export type ProxmoxNodeDetail = {
  available: boolean; error: string | null; node: string;
  cpu_model: string | null; cpu_cores: number | null; cpu_threads: number | null;
  cpu_sockets: number | null; cpu_mhz: string | null; cpu_vendor: string | null;
  kernel: string | null; pve_version: string | null; loadavg: string[];
  swap_total: number | null; swap_used: number | null;
  rootfs_total: number | null; rootfs_used: number | null;
  gpus: string[]; uptime: number | null;
};
export type ProxmoxNodesResult = { available: boolean; error: string | null; nodes: ProxmoxNode[] };
export type ProxmoxGuestsResult = { available: boolean; error: string | null; guests: ProxmoxGuest[] };
export type AlertLevel = "critical" | "warn" | "info" | "ok";
export type Alert = {
  key: string; level: AlertLevel; category: string; title: string; detail: string; host: string;
  first_seen: number; last_seen: number; resolved_at?: number;
};
export type AlertsSnapshot = { active: Alert[]; recent: Alert[]; counts: { critical: number; warn: number; info: number } };
export type HostReliability = {
  uptime_pct: number; checks: number; samples: number[];
  state: "closed" | "open" | "half_open"; consecutive_fails: number; last_ok: number | null;
};
export type ReliabilitySnapshot = Record<string, HostReliability | null>;
export type RoutingArtifact = {
  profile: string; model_id: string; decision: string; k: number | null;
  allowed_workloads: string[]; denied_workloads: string[]; workload_classes: string[];
  delta_tps_per_class: Record<string, number>;
  profile_tps_per_class: Record<string, number>;
  baseline_tps_per_class: Record<string, number>;
  profile_delta_global: number | null; acceptance_mean: number | null;
  vram_free_mib_min: number | null; vllm_pin: string; notes: string;
};
export type RoutingArtifacts = { available: boolean; reason?: string; artifacts: RoutingArtifact[] };
export type RoutingActive = { available: boolean; reason?: string; ok?: boolean; error?: string; profile?: string | null; source?: string; artifact?: RoutingArtifact | null; candidates?: string[] };
export type RoutingSignals = { response_format?: unknown; tool_choice?: unknown; workload_class?: string };
export type LicenseEngine = { installed: boolean; module: string | null; version: string | null };
export type LicenseInfo = { subject: string | null; expires: string | null; signature_valid: boolean | null; path: string | null };
export type LicenseStatus = {
  available: boolean; reason?: string;
  core?: string; tier?: string; engine?: LicenseEngine; license?: LicenseInfo;
  eligible?: boolean; status?: string;
  premium_patches_enabled?: number; engine_tier_patches?: number;
};
export type FlagRow = {
  env_flag: string; patch_id: string | null; title: string | null; family: string | null;
  tier: string | null; lifecycle: string | null; default_on: boolean;
  live_on?: boolean; drift?: "in_sync" | "missing" | "extra";
};
export type FlagMatrix = {
  flags: FlagRow[];
  counts: { total: number; default_on: number; default_off: number; missing: number; extra: number };
  has_live: boolean;
};
export type RoutingClassify = {
  available: boolean; reason?: string;
  profile?: string; signal?: string; workload_class?: string | null;
  accepted?: boolean; expected_delta_tps?: number | null; active_profile?: string | null;
};

export type DeployParameters = {
  image: string;
  container_name: string;
  host_port: number | null;
  tensor_parallel: number;
  min_vram_per_gpu_mib: number | null;
  model_path: string | null;
  max_model_len: number | null;
  max_num_seqs: number | null;
  gpu_memory_utilization: number | null;
  kv_cache_dtype: string | null;
  genesis_pin: string | null;
  genesis_env_count: number;
  lifecycle: string | null;
  maintainer: string | null;
  argv: string[];
};

export type DeployPlanItem = {
  scope: string;
  action: string;
  target: string;
  severity: "blocker" | "warning" | "info";
  reason: string;
  suggested_command: string;
};

export type DeployDependencies = {
  config_key: string;
  items: DeployPlanItem[];
  notes: string[];
  is_ready: boolean;
  n_blockers: number;
  n_warnings: number;
};

export type DeployMountVar = { name: string; value: string; container: string };

export type DeploymentPlan = {
  preset_id: string;
  preset_label: string;
  target: string;
  target_label: string;
  artifact: { kind: "yaml" | "ini" | "bash"; filename: string; content: string };
  parameters: DeployParameters;
  mount_vars: DeployMountVar[];
  dependencies: DeployDependencies;
  commands: string[];
};

export type ApiTokenRecord = {
  id: string;
  label: string;
  prefix: string;
  created_at: number;
  last_used: number | null;
  created_by: string;
};

export type InstallTarget = { id: string; label: string; filename: string; kind: string; needs: string; summary: string };
export type InstallTargets = { targets: InstallTarget[]; hosts: HostProfile[]; apply_enabled: boolean };
export type InstallStep = { order: number; kind: string; title: string; danger: boolean; cmd?: string; file?: string };
export type InstallPlan = {
  host: { label: string; host: string }; preset_id: string; target: string; target_label: string;
  artifact: { kind: string; filename: string; content: string };
  parameters?: Record<string, unknown>; dependencies?: Record<string, unknown>; image_override?: string | null; with_daemon?: boolean;
  steps: InstallStep[]; danger_count: number; provisions_infra: boolean;
  dry_run: boolean; can_apply: boolean; notes: string;
};
export type InstallApplyStep = { cmd: string; rc: number; output: string };
export type InstallApplyResult = {
  ok: boolean; applied: boolean; target: string; target_label?: string;
  artifact?: string; steps: InstallApplyStep[]; error: string | null;
};
export type NodeSetupResult = { ok: boolean; applied: boolean; port?: number; steps: InstallApplyStep[]; error: string | null };

export type ModelArch = { name: string; num_layers: number; num_kv_heads: number; head_dim: number; params_b: number; weight_bits: number; is_moe: boolean; active_params_b: number | null };
export type CalcModels = { models: Record<string, ModelArch>; kv_dtypes: Record<string, number> };
export type KvEstimate = { model: string; weights_per_gpu_mib: number; kv_per_gpu_mib: number; kv_total_mib: number; overhead_mib: number; total_per_gpu_mib: number; budget_per_gpu_mib: number; headroom_mib: number; fits: boolean; max_context: number; kv_bytes_per_token: number; tp: number; concurrency: number; context: number };
export type KvCurvePoint = { context: number; weights_mib: number; kv_mib: number; overhead_mib: number; total_mib: number; fits: boolean };
export type EnvelopeCell = { context: number; concurrency: number; fits: boolean; headroom_mib: number; total_per_gpu_mib: number };
export type KvRec = { kv_dtype: string; kv_bytes: number; fits: boolean; headroom_mib: number; max_context: number; total_per_gpu_mib: number; recommended: boolean };
export type KvCalcResult = {
  arch: ModelArch; kv_dtype: string; overhead_mib: number;
  rig: { tp: number; gpu_count: number; gpu_vram_mib: number; util: number };
  result: KvEstimate; by_dtype: Record<string, number>; by_tp: Record<string, number>; curve: KvCurvePoint[];
  envelope: { contexts: number[]; concurrencies: number[]; grid: EnvelopeCell[][] };
  recommendation: KvRec[];
  arch_advice: ArchAdvice | null;
};

// ── Fleet overview (all engine hosts at a glance) ───────────────────────────
export type FleetEngine = { container: string | null; port: number | null; reachable: boolean; version: string | null; models: string[]; patches: number };
export type FleetGpu = { name: string; memory_total_mib: string; arch: string | null; utilization: string | null };
export type FleetHost = {
  id: string; label: string; host: string; role: string;
  ssh_ok: boolean; engines: FleetEngine[]; gpus: FleetGpu[]; gpu_count: number;
  arch: string; interconnect: string | null; active_patches: number;
  models: string[]; vllm_version: string | null; error: string | null;
};

// ── Ops copilot (read-only tool-calling assistant) ──────────────────────────
export type CopilotTool = { name: string; description: string; category: "read" | "plan" | string };
export type CopilotStep = { tool: string; args: Record<string, unknown>; ok: boolean };
export type CopilotProposedAction = { kind: string; label: string; section: string; params?: Record<string, unknown> };
export type CopilotResult = {
  reply: string; steps: CopilotStep[]; proposed_actions: CopilotProposedAction[];
  usage: Record<string, number>; stopped: "final" | "max_steps" | string;
};
export type CopilotChatOpts = { host?: string; port?: number; host_id?: string; model?: string; max_steps?: number };

export type BaselineRec = { id: string; label: string; saved_at: number; scenarios: string[] };
export type BaselineTrend = {
  metric: string; scenario: string | null;
  points: Array<{ saved_at: number; label: string; value: number }>;
  lower_is_better: boolean; metrics_available: string[];
};
export type BaselineDiff = {
  threshold_pct: number; regressed: number; improved: number; has_regression: boolean; exit_code: number; verdict: string;
  scenarios: Array<{ name: string; status: string; metrics: Array<{ metric: string; current: number; baseline: number; delta: number; pct: number; lower_is_better: boolean; regression: boolean; improvement: boolean }> }>;
};

export type UpdateStatus = {
  sndr_core_version: string;
  supported_pins: string[];
  canonical_pin: string | null;
  git: { is_repo: boolean; branch?: string; commit?: string; dirty?: boolean; remote?: string };
  gui_build: { published?: boolean; built_at?: number; bundle?: string | null };
  apply_enabled: boolean;
};

export type UpdateCheck = {
  is_repo: boolean;
  branch?: string;
  local_commit?: string;
  remote_commit?: string | null;
  update_available: boolean;
  error?: string | null;
};

export type UpdateStep = { order: number; title: string; kind: string; cmd: string; pin?: string | null };
export type UpdatePlan = {
  valid: boolean;
  blocked_reasons: string[];
  pin_gate: { ok: boolean; target_pin: string | null; reason: string | null };
  target_pin: string | null;
  current_version: string;
  steps: UpdateStep[];
};

export type UpdateApplyResult = {
  applied: boolean;
  status: string;
  message?: string;
  results?: Array<{ order: number; title: string; command: string; exit_code: number; status: string; stdout: string; stderr: string }>;
  manual_steps?: UpdateStep[];
  plan?: UpdatePlan;
};

export type RagDoc = {
  id: string;
  kind: string;
  title: string;
  ref: string;
  snippet: string;
  score: number;
};

export type RagResult = {
  query: string;
  matched: number;
  docs: RagDoc[];
};

export type RagPreview = {
  ok: boolean;
  error: string | null;
  path?: string;
  files: number;
  chunks: number;
  sample: string[];
};

export type Operation = {
  id: string;
  label: string;
  group: string;
  description: string;
  command: string;
  mutating: boolean;
  estimate: string;
};

export type OperationsResult = { operations: Operation[]; apply_enabled: boolean };

export type Caveat = {
  id: string; severity: string; title: string; message: string;
  docs_url: string | null; triggered: boolean | null;
};
export type CaveatsResult = {
  caveats: Caveat[]; total: number; triggered_count: number;
  host_facts_available: boolean; facts_error: string | null;
};
export type ConfigKeyMeta = { source: string } & Record<string, string>;
export type ConfigKeysResult = {
  keys: Record<string, ConfigKeyMeta>; total: number;
  by_source: Record<string, number>;
};
export type TraceSpec = {
  id: string; container_path: string; patch_id: string;
  enable_env: string | null; category: string; description: string;
};
export type TraceCatalog = {
  traces: TraceSpec[]; categories: string[];
  by_category: Record<string, number>; total: number;
};
export type FleetDeployHostResult = {
  host_id: string; label?: string; host?: string; ok: boolean;
  error: string | null; mutating_steps?: number; plan: unknown;
};
export type FleetDeployPlan = {
  preset_id: string; target: string; results: FleetDeployHostResult[];
  rollup: { hosts: number; ready: number; errors: number; mutating_steps_total: number; apply_enabled: boolean };
};

export type ServiceStep = { order: number; title: string; command: string };

export type ServiceActionPlan = {
  plan_id: string;
  preset_id: string;
  action: string;
  runtime_target: string;
  host: string;
  container_name: string;
  mutating: boolean;
  actionable: boolean;
  action_reason: string;
  steps: ServiceStep[];
  side_effects: string[];
  gates: Array<{ id: string; title: string; status: string; detail: string }>;
  cli_mirror: string[];
  rollback: string;
};

export type DoctorFinding = {
  category: string;
  id: string;
  title: string;
  severity: "ok" | "info" | "warning" | "blocked";
  detail: string;
  evidence: string;
  action: string;
  cli: string;
};

export type DoctorReport = {
  findings: DoctorFinding[];
  summary: Record<string, number>;
  categories: string[];
  generated_for: string;
  warnings: string[];
};

export type V2LayerApplyResult = {
  kind: string;
  layer_id: string;
  target_path: string;
  backup_path?: string | null;
  action: string;
  written: boolean;
  bytes_written: number;
  status: "applied" | "blocked" | "conflict" | string;
  message: string;
  blocked_reasons: string[];
};

export type V2LayerDefinition = {
  kind: string;
  id: string;
  source: string;
  definition: Record<string, any>;
};

export type V2ConfigApplyResult = {
  plan_id: string;
  preset_id: string;
  target_path: string;
  backup_path?: string | null;
  action: string;
  written: boolean;
  bytes_written: number;
  status: "applied" | "blocked" | "conflict" | string;
  message: string;
  blocked_reasons: string[];
};

export type UserPreset = {
  id: string;
  path: string;
  model?: string | null;
  hardware?: string | null;
  profile?: string | null;
  runtime?: string | null;
  size_bytes: number;
};

export type UserPresetList = {
  count: number;
  presets: UserPreset[];
};

export type BundleSpec = {
  name: string;
  umbrella_flag: string;
  tier: string;
  description: string;
  module_path: string;
  has_apply?: boolean | null;
  import_error?: string | null;
};

export type DiffUpstreamReport = {
  merged_upstream: Array<Record<string, any>>;
  has_upstream_pr: Array<Record<string, any>>;
};

export type ProofStatusReport = {
  available: boolean;
  reason?: string;
  total: number;
  counts: Record<string, number>;
  patches: Array<Record<string, any>>;
};

export type ReportBundleResult = {
  bundle_id: string;
  report_type: string;
  preset_id: string;
  bundle_dir: string;
  files: string[];
  redacted: boolean;
  created_at: number;
  summary: string;
  note: string;
};

export type AuthUser = {
  username: string;
  role: string;
  source: string;
  email: string | null;
  totp_enabled: boolean;
  has_password: boolean;
  recovery_codes_remaining: number;
  disabled: boolean;
  created_at: number;
  last_login: number | null;
};

export type AuthStatus = {
  auth_required: boolean;
  apply_enabled: boolean;
  backends: string[];
  oauth_providers: string[];
  context: { in_container: boolean; system_user: string; pam_enabled: boolean };
  user: AuthUser | null;
};

export type LoginResponse = {
  ok: boolean;
  needs_2fa?: boolean;
  username?: string;
  token?: string;
  user?: AuthUser;
};

export type EngineStatus = {
  reachable: boolean;
  host: string;
  base_url: string;
  metrics_url: string;
  version: string | null;
  models: string[];
  error: string | null;
};

export type EngineMetricSample = {
  ts: number;
  throughput: number | null;
  kv_cache: number | null;
  running: number | null;
  waiting: number | null;
};

export type EngineMetrics = {
  reachable: boolean;
  metrics_url: string;
  error: string | null;
  kpis: Record<string, number>;
  metric_families?: number;
  history?: EngineMetricSample[];
};

export type EngineChatResult = {
  reply: string;
  model: string;
  usage: Record<string, number>;
  finish_reason: string | null;
  latency_ms: number;
};

export type EngineBenchResult = {
  ok: boolean;
  params: { num_requests: number; concurrency: number; max_tokens: number; temperature: number; model: string };
  metrics: {
    throughput_tok_s: number;
    ttft_avg_ms: number | null;
    ttft_p50_ms: number | null;
    ttft_p90_ms: number | null;
    tpot_avg_ms: number | null;
    cv_pct: number;
    total_tokens: number;
    requests_ok: number;
    requests_failed: number;
    wall_clock_s: number;
  };
  methodology: string;
};

export type BackendEvent = {
  seq: number;
  ts: number;
  kind: string;
  message: string;
  detail: Record<string, any>;
};

export type ModelCacheEntry = {
  model_id: string;
  model_path: string;
  present: boolean;
  size_mib: number | null;
};

export type HubModel = {
  id: string;
  downloads: number | null;
  likes: number | null;
  pipeline_tag: string | null;
  gated: boolean;
  tags: string[];
};

export type ModelCacheReport = {
  host: string;
  total: number;
  present_count: number;
  models: ModelCacheEntry[];
};

export type FitCheck = {
  id: string;
  title: string;
  ok: boolean;
  severity: "ok" | "info" | "warning" | "blocked";
  detail: string;
};

export type MemoryFitReport = {
  model_id: string;
  hardware_id: string;
  model_title: string;
  hardware_title: string;
  compatible: boolean;
  checks: FitCheck[];
  vram: {
    model_min_mib: number;
    rig_floor_mib: number;
    headroom_mib: number;
    n_gpus: number;
    vram_per_gpu_mib: number;
    gpu_memory_utilization: number;
    kv_cache_dtype: string;
  };
  notes: string[];
};

export type PatchRow = {
  patch_id: string;
  tier: string;
  lifecycle: string;
  family: string;
  default_on: boolean;
  production_default: string;
  implementation_status: string;
  env_flag: string;
  upstream_pr: number | string | null;
  title: string;
  apply_module: string;
};

export type PatchListResult = {
  filters: Record<string, string | boolean | null>;
  matched: number;
  total: number;
  patches: PatchRow[];
  summary: {
    tier_counts: Record<string, number>;
    lifecycle_counts: Record<string, number>;
    production_default_counts: Record<string, number>;
    implementation_status_counts: Record<string, number>;
  };
};

export type PatchIssue = { severity: string; patch_id: string; message: string };
export type PatchDoctorReport = {
  registry_size: number;
  issues: PatchIssue[];
  coverage: {
    total: number;
    mapped: number;
    unmapped: string[];
    intentionally_unmapped: string[];
  };
};

export type PatchExplainResult = {
  patch_id: string;
  meta: Record<string, any>;
  spec: Record<string, any>;
  live_decision?: [boolean, string] | null;
  live_decision_error?: string | null;
};

// When the UI is served by the daemon itself (production / `sndr gui-api`),
// the API lives at the same origin — use it so there is no CORS hop and no
// hardcoded port. In Vite dev (5173/5174) the API is on a separate port, so
// fall back to the canonical daemon address.
const _VITE_DEV_PORTS = new Set(["5173", "5174"]);
const _SAME_ORIGIN =
  typeof window !== "undefined" && !_VITE_DEV_PORTS.has(window.location.port)
    ? window.location.origin
    : "";
const DEFAULT_API_BASE =
  import.meta.env.VITE_SNDR_API_BASE?.replace(/\/$/, "") ??
  (_SAME_ORIGIN || "http://127.0.0.1:8765");
const API_BASE_STORAGE_KEY = "sndr.gui.apiBase";
const API_TOKEN_STORAGE_KEY = "sndr.gui.token";

export function getApiBase() {
  return (
    lsGet(API_BASE_STORAGE_KEY)?.replace(/\/$/, "") ??
    DEFAULT_API_BASE
  );
}

export function setApiBase(value: string) {
  const next = value.trim().replace(/\/$/, "");
  if (!next) {
    lsRemove(API_BASE_STORAGE_KEY);
    return DEFAULT_API_BASE;
  }
  lsSet(API_BASE_STORAGE_KEY, next);
  return next;
}

export function getApiToken() {
  return lsGet(API_TOKEN_STORAGE_KEY) ?? "";
}

export function setApiToken(value: string) {
  const next = value.trim();
  if (!next) lsRemove(API_TOKEN_STORAGE_KEY);
  else lsSet(API_TOKEN_STORAGE_KEY, next);
  return next;
}

// Connection helpers. The daemon connection targets are derived from the host
// registry (single source of truth) — there is no separate saved-server list.
export function normalizeBaseUrl(value: string): string {
  return (value || "").trim().replace(/\/+$/, "");
}

export function hostLabel(baseUrl: string): string {
  try { return new URL(baseUrl).host || baseUrl; } catch { return baseUrl; }
}

function authHeaders(): Record<string, string> {
  const token = getApiToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

function sameOriginApi(): boolean {
  const base = getApiBase();
  if (!base) return true; // empty base == same origin
  try {
    return new URL(base, window.location.href).origin === window.location.origin;
  } catch {
    return false;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${getApiBase()}${path}`, {
    ...init,
    // Same-origin (daemon-served / production): send the httpOnly session
    // cookie so OAuth logins work. Cross-origin (Vite dev): omit credentials
    // to satisfy CORS and rely on the bearer token instead.
    credentials: sameOriginApi() ? "include" : "omit",
    headers: { ...authHeaders(), ...(init?.headers ?? {}) }
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      detail = payload.detail ?? detail;
    } catch {
      // Keep the HTTP status fallback.
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

async function postJson<T>(path: string, payload: Record<string, any>): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

function query(params: Record<string, string | number | boolean | undefined | null>) {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      search.set(key, String(value));
    }
  });
  const value = search.toString();
  return value ? `?${value}` : "";
}

// ─── Container management ────────────────────────────────────────────
export interface ManagedContainer {
  name: string;
  id: string;
  image: string;
  state: string;
  status: string;
  ports: string;
  created: string;
  labels: Record<string, string>;
  networks?: string;
}
export interface ContainerStats {
  cpu_pct: number;
  mem_usage: number;
  mem_limit: number;
  mem_pct: number;
  net_rx?: number;
  net_tx?: number;
  blk_read?: number;
  blk_write?: number;
  pids?: number;
}
export interface ContainerExecResult {
  ok: boolean;
  container: string;
  exit_code: number;
  stdout: string;
  stderr: string;
}
export type ContainerAction = "start" | "stop" | "restart";
export interface ContainerTop { titles: string[]; processes: string[][]; }
export interface ContainerChange { kind: "added" | "modified" | "deleted"; path: string; }
export interface FsEntry {
  name: string; is_dir: boolean; is_link: boolean; link_target?: string | null;
  perms: string; owner: string; group: string; size: number; mtime: string;
}
export type UpdateMode = "manual" | "semi" | "auto";
export interface ContainerUpdatePlan {
  container: string; image: string; is_engine: boolean;
  supported_pins: string[]; canonical_pin: string | null;
  guarded_update: boolean; policy: string; commands: string[];
  mode: UpdateMode; is_critical: boolean; modes: UpdateMode[];
  update_available?: boolean; running_image_id?: string; latest_image_id?: string;
  has_previous?: boolean;
}
export interface ImageScan {
  available: boolean; image: string; reason?: string; scanner?: string;
  counts?: { critical: number; high: number; medium: number; low: number; negligible: number; unknown: number };
  total?: number; error?: string;
}
export interface DiskType { type: string; total_count: number; active: number; size: number; reclaimable: number; }
export interface SystemDf { types: DiskType[]; total_size: number; }
export interface AlertConfig { enabled: boolean; chat_id: string; has_token: boolean; configured: boolean; channel: string; }
export interface DockerNetwork { name: string; driver: string; scope: string; }
export interface DriftItem { field: string; expected: string; actual: string | null; kind: "image" | "missing" | "changed"; }
export interface SourceReport {
  container: string; preset_id: string | null; linked_by: "label" | "name" | null;
  preset_title: string | null; drift: DriftItem[]; drift_count: number;
  live_patches: { flag: string; value: string }[]; live_patch_count: number;
  patch_sync?: { in_sync: string[]; missing: string[]; extra: string[] };
  // Identity stamped on the container by the launcher (sndr.* labels).
  served_model?: string | null; pin?: string | null; role?: string | null;
}
export interface ContainerSettings { cpus?: number | null; memory?: number | null; restart_policy?: string | null; }

// A container source: the local daemon's host (socket) or a registered host (SSH).
export type ContainerSource = { kind: "local" } | { kind: "host"; hostId: string };
function containerBase(src: ContainerSource): string {
  return src.kind === "host"
    ? `/api/v1/hosts/${encodeURIComponent(src.hostId)}/containers`
    : "/api/v1/containers";
}

export const api = {
  get baseUrl() {
    return getApiBase();
  },
  setBaseUrl: setApiBase,
  overview: () => request<ProductOverview>("/api/v1/overview"),
  capabilities: () => request<ProductCapabilities>("/api/v1/capabilities"),
  presets: (params: {
    family?: string;
    workload?: string;
    hardware?: string;
    mode?: string;
    status?: string;
  }) => request<PresetListResult>(`/api/v1/presets${query(params)}`),
  recommendPresets: (params: {
    workload: string;
    hardware?: string;
    concurrency?: number;
    top?: number;
  }) =>
    request<PresetRecommendResult>(
      `/api/v1/presets/recommend${query(params)}`
    ),
  explainPreset: (id: string) =>
    request<PresetExplainResult>(`/api/v1/presets/${id}/explain`),
  v2ConfigCatalog: () =>
    request<V2ConfigCatalog>("/api/v1/configs/v2/catalog"),
  v2ConfigPreview: (params: {
    model_id: string;
    hardware_id: string;
    profile_id?: string;
    runtime?: string;
  }) => request<V2ConfigPreview>(`/api/v1/configs/v2/preview${query(params)}`),
  v2ConfigPlan: (payload: {
    preset_id?: string;
    model_id: string;
    hardware_id: string;
    profile_id?: string;
    runtime?: string;
  }) => postJson<V2ConfigPlan>("/api/v1/configs/v2/plan", payload),
  v2ConfigApply: async (payload: {
    preset_id?: string;
    model_id: string;
    hardware_id: string;
    profile_id?: string;
    runtime?: string;
    expected_plan_id?: string;
  }): Promise<V2ConfigApplyResult> => {
    // "blocked" (422) and "conflict" (409) are meaningful business outcomes,
    // not transport errors: the server returns the full result as `detail`.
    // We surface it instead of throwing so the GUI can render the state.
    const response = await fetch(`${getApiBase()}/api/v1/configs/v2/apply`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify(payload)
    });
    const json = await response.json().catch(() => ({}));
    if (response.ok) return json as V2ConfigApplyResult;
    if (json && typeof json.detail === "object" && json.detail) {
      return json.detail as V2ConfigApplyResult;
    }
    throw new Error(json?.detail ?? `${response.status} ${response.statusText}`);
  },
  userPresets: () => request<UserPresetList>("/api/v1/configs/v2/user-presets"),
  v2Layer: (kind: string, id: string, signal?: AbortSignal) =>
    request<V2LayerDefinition>(
      `/api/v1/configs/v2/layer/${encodeURIComponent(kind)}/${encodeURIComponent(id)}`,
      { signal }
    ),
  v2LayerApply: async (payload: { kind: string; layer_id: string; yaml_text: string }): Promise<V2LayerApplyResult> => {
    const response = await fetch(`${getApiBase()}/api/v1/configs/v2/layer/apply`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify(payload)
    });
    const json = await response.json().catch(() => ({}));
    if (response.ok) return json as V2LayerApplyResult;
    if (json && typeof json.detail === "object" && json.detail) return json.detail as V2LayerApplyResult;
    throw new Error(json?.detail ?? `${response.status} ${response.statusText}`);
  },
  bundles: () => request<{ bundles: BundleSpec[] }>("/api/v1/patches/bundles"),
  diffUpstream: () => request<DiffUpstreamReport>("/api/v1/patches/diff-upstream"),
  proofStatus: () => request<ProofStatusReport>("/api/v1/proof/status"),
  launchPlan: (params: {
    preset_id: string;
    runtime_target?: string;
    patch_policy?: string;
    host?: string;
    mode?: string;
  }) => request<LaunchPlanResult>(`/api/v1/launch/plan${query(params)}`),
  patches: (params: {
    tier?: string;
    lifecycle?: string;
    family?: string;
    default_on?: boolean;
    has_upstream?: boolean;
  }) => request<PatchListResult>(`/api/v1/patches${query(params)}`),
  patchExplain: (patchId: string) =>
    request<PatchExplainResult>(`/api/v1/patches/${encodeURIComponent(patchId)}/explain`),
  patchOverrides: () => request<{ overrides: Record<string, { state: string; env_flag: string }> }>("/api/v1/patches/overrides"),
  setPatchOverride: (patch_id: string, state: string, env_flag: string) =>
    postJson<{ ok: boolean; overrides: Record<string, { state: string; env_flag: string }> }>("/api/v1/patches/overrides", { patch_id, state, env_flag }),
  patchDoctor: () => request<PatchDoctorReport>("/api/v1/patches/doctor"),
  doctor: () => request<DoctorReport>("/api/v1/doctor"),
  memoryFit: (params: { model_id: string; hardware_id: string }, signal?: AbortSignal) =>
    request<MemoryFitReport>(`/api/v1/memory/fit${query(params)}`, { signal }),
  modelsCache: (signal?: AbortSignal) => request<ModelCacheReport>("/api/v1/models/cache", { signal }),
  eventsRecent: (since_seq = 0) =>
    request<{ events: BackendEvent[]; last_seq: number }>(`/api/v1/events/recent${query({ since_seq })}`),
  reportBundle: (payload: { report_type: string; preset_id?: string; redact?: boolean }) =>
    postJson<ReportBundleResult>("/api/v1/reports/bundle", payload),
  environment: () => request<EnvironmentReport>("/api/v1/environment"),
  operations: () => request<OperationsResult>("/api/v1/operations"),
  operationRun: (operation: string) => postJson<Job>("/api/v1/operations/run", { operation }),
  caveats: () => request<CaveatsResult>("/api/v1/caveats"),
  configKeys: () => request<ConfigKeysResult>("/api/v1/config-keys"),
  traces: () => request<TraceCatalog>("/api/v1/traces"),
  deployTargets: () => request<DeployTargetsResult>("/api/v1/deploy/targets"),
  deployPlan: (payload: { preset_id: string; target: string; host_paths?: Record<string, string> }) =>
    postJson<DeploymentPlan>("/api/v1/deploy/plan", payload),
  servicePlan: (params: { preset_id: string; action: string; runtime_target?: string; host?: string }) =>
    request<ServiceActionPlan>(`/api/v1/services/plan${query(params)}`),
  serviceApply: (payload: {
    preset_id: string;
    action: string;
    runtime_target?: string;
    host?: string;
    transport?: string;
    ssh_target?: string;
    confirm?: boolean;
  }) => postJson<Job>("/api/v1/services/apply", payload),
  authStatus: () => request<AuthStatus>("/api/v1/auth/status"),
  login: (username: string, password: string) =>
    postJson<LoginResponse>("/api/v1/auth/login", { username, password }),
  login2fa: (username: string, code: string) =>
    postJson<LoginResponse>("/api/v1/auth/login/2fa", { username, code }),
  logout: () => postJson<{ ok: boolean }>("/api/v1/auth/logout", {}),
  me: () => request<AuthUser>("/api/v1/auth/me"),
  changePassword: (current: string, next: string) =>
    postJson<{ ok: boolean }>("/api/v1/auth/password", { current, new: next }),
  listUsers: () => request<{ users: AuthUser[] }>("/api/v1/auth/users"),
  createUser: (username: string, password: string, role: string) =>
    postJson<AuthUser>("/api/v1/auth/users", { username, password, role }),
  deleteUser: (username: string) =>
    request<{ ok: boolean }>(`/api/v1/auth/users/${encodeURIComponent(username)}`, { method: "DELETE" }),
  apiTokens: () => request<{ tokens: ApiTokenRecord[] }>("/api/v1/auth/tokens"),
  apiTokenCreate: (label: string) => postJson<{ token: string; record: ApiTokenRecord }>("/api/v1/auth/tokens", { label }),
  apiTokenRevoke: (id: string) => request<{ revoked: boolean }>(`/api/v1/auth/tokens/${encodeURIComponent(id)}`, { method: "DELETE" }),
  enroll2fa: () => postJson<{ secret: string; otpauth_uri: string }>("/api/v1/auth/2fa/enroll", {}),
  activate2fa: (code: string) =>
    postJson<{ ok: boolean; totp_enabled: boolean; recovery_codes: string[] }>("/api/v1/auth/2fa/activate", { code }),
  regenerateRecovery: () => postJson<{ ok: boolean; recovery_codes: string[] }>("/api/v1/auth/2fa/recovery", {}),
  disable2fa: () => postJson<{ ok: boolean; totp_enabled: boolean }>("/api/v1/auth/2fa/disable", {}),
  revokeSessions: () => postJson<{ ok: boolean }>("/api/v1/auth/sessions/revoke", {}),
  oauthLoginUrl: (provider: string) => `${getApiBase()}/api/v1/auth/oauth/${provider}/login`,
  engineStatus: (host?: string, port?: number, apiKey?: string, hostId?: string) =>
    request<EngineStatus>(`/api/v1/engine/status${query({ host, port, host_id: hostId })}`, apiKey ? { headers: { "X-Engine-Api-Key": apiKey } } : undefined),
  engineMetrics: (host?: string, port?: number) => request<EngineMetrics>(`/api/v1/engine/metrics${query({ host, port })}`),
  chatRetrieve: (queryText: string, k = 5, sources?: { project?: boolean; vaults?: string[] }) =>
    postJson<RagResult>("/api/v1/chat/retrieve", { query: queryText, k, project: sources?.project ?? true, vaults: sources?.vaults ?? [] }),
  ragPreview: (path: string) => postJson<RagPreview>("/api/v1/chat/rag/preview", { path }),
  engineChat: (payload: {
    messages: Array<{ role: string; content: string }>;
    model?: string;
    max_tokens?: number;
    temperature?: number;
    host?: string;
    port?: number;
    host_id?: string;
  }) => postJson<EngineChatResult>("/api/v1/engine/chat", payload),
  engineBench: (params: {
    num_requests?: number;
    concurrency?: number;
    max_tokens?: number;
    temperature?: number;
    prompt?: string;
    model?: string;
    host?: string;
  }) => postJson<EngineBenchResult>("/api/v1/engine/bench", params),
  modelsDownload: (model_id: string) => postJson<Job>("/api/v1/models/download", { model_id }),
  hubSearch: (q: string, limit = 20) => request<{ results: HubModel[] }>(`/api/v1/models/hub/search${query({ query: q, limit })}`),
  downloadRepo: (repo_id: string) => postJson<Job>("/api/v1/models/download", { repo_id }),
  engineChatStream: async (
    payload: { messages: Array<{ role: string; content: string }>; model?: string; max_tokens?: number; temperature?: number; top_p?: number; presence_penalty?: number; frequency_penalty?: number; stop?: string[]; host?: string; port?: number; apiKey?: string; hostId?: string; chat_template_kwargs?: { enable_thinking?: boolean } },
    handlers: { onDelta: (text: string) => void; onDone: (meta: { tokens?: number; latency_ms?: number; ttft_ms?: number }) => void; onError: (msg: string) => void },
    signal?: AbortSignal
  ) => {
    const { apiKey, hostId, ...rest } = payload;
    const body = { ...rest, ...(hostId ? { host_id: hostId } : {}) };
    const response = await fetch(`${getApiBase()}/api/v1/engine/chat/stream`, {
      method: "POST",
      credentials: sameOriginApi() ? "include" : "omit",
      headers: { "Content-Type": "application/json", ...authHeaders(), ...(apiKey ? { "X-Engine-Api-Key": apiKey } : {}) },
      body: JSON.stringify(body),
      signal
    });
    if (!response.ok || !response.body) {
      handlers.onError(`Stream failed (${response.status})`);
      return;
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let nl: number;
      while ((nl = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, nl).trim();
        buffer = buffer.slice(nl + 1);
        if (!line) continue;
        try {
          const obj = JSON.parse(line);
          if (obj.error) handlers.onError(obj.error);
          else if (obj.done) handlers.onDone(obj);
          else if (obj.delta) handlers.onDelta(obj.delta);
        } catch {
          // ignore a partial/garbled frame
        }
      }
    }
  },
  raw: (path: string) => request<unknown>(path),
  launchApply: (payload: {
    preset_id: string;
    runtime_target?: string;
    host?: string;
    transport?: string;
    ssh_target?: string;
    confirm?: boolean;
  }) => postJson<Job>("/api/v1/launch/apply", payload),
  benchRun: (payload: { preset_id: string; profile?: string; ctx?: string }) =>
    postJson<Job>("/api/v1/bench/run", payload),
  evidenceAttach: (payload: { preset_id: string }) =>
    postJson<Job>("/api/v1/evidence/attach", payload),
  jobs: () => request<{ jobs: Job[] }>("/api/v1/jobs"),
  job: (id: string) => request<Job>(`/api/v1/jobs/${encodeURIComponent(id)}`),
  hosts: () => request<{ hosts: HostProfile[] }>("/api/v1/hosts"),
  hostUpsert: (payload: Partial<HostProfile>) => postJson<HostProfile>("/api/v1/hosts", payload),
  hostDelete: (id: string) =>
    request<{ deleted: boolean }>(`/api/v1/hosts/${encodeURIComponent(id)}`, { method: "DELETE" }),
  hostProbe: (host: string, port: number, apiKey?: string, hostId?: string) =>
    request<HostProbe>(`/api/v1/hosts/probe${query({ host, port, host_id: hostId })}`, apiKey ? { headers: { "X-Engine-Api-Key": apiKey } } : undefined),
  sshCheck: (payload: {
    host: string; host_id?: string; user?: string; auth_method?: string;
    key_path?: string; password?: string; ssh_port?: number; forget_password?: boolean;
  }) => postJson<SshCheckResult>("/api/v1/hosts/ssh-check", payload),
  fetchApiKey: (host_id: string, container?: string) =>
    postJson<{ available: boolean; found: boolean; source: string | null; key_masked?: string; error: string | null }>("/api/v1/hosts/fetch-api-key", { host_id, container }),
  discoverHost: (host_id: string) => postJson<HostDiscovery>("/api/v1/hosts/discover", { host_id }),
  modelConfig: (host_id: string, container?: string) => postJson<HostModelConfig>("/api/v1/hosts/model-config", { host_id, container }),
  sndrState: (host_id: string, container?: string) => postJson<HostSndrState>("/api/v1/hosts/sndr-state", { host_id, container }),
  installTargets: () => request<InstallTargets>("/api/v1/install/targets"),
  installPlan: (host_id: string, preset_id: string, target: string, image_override?: string, with_daemon?: boolean) => postJson<InstallPlan>("/api/v1/install/plan", { host_id, preset_id, target, image_override: image_override || undefined, with_daemon: with_daemon || undefined }),
  installApply: (host_id: string, preset_id: string, target: string, image_override?: string, with_daemon?: boolean) => postJson<InstallApplyResult>("/api/v1/install/apply", { host_id, preset_id, target, confirm: true, image_override: image_override || undefined, with_daemon: with_daemon || undefined }),
  installNode: (host_id: string, admin_password: string, engine_port?: number) =>
    postJson<NodeSetupResult>("/api/v1/install/node", { host_id, admin_password, engine_port, confirm: true }),
  fleetDeployPlan: (payload: { preset_id: string; target: string; host_ids: string[]; with_daemon?: boolean; image_override?: string }) =>
    postJson<FleetDeployPlan>("/api/v1/fleet/deploy-plan", payload),
  calcModels: () => request<CalcModels>("/api/v1/calc/models"),
  calcKv: (payload: Record<string, unknown>) => postJson<KvCalcResult>("/api/v1/calc/kv", payload),
  fleetOverview: () => request<{ hosts: FleetHost[] }>("/api/v1/fleet/overview"),
  copilotTools: () => request<{ tools: CopilotTool[] }>("/api/v1/copilot/tools"),
  copilotChat: (messages: Array<{ role: string; content: string }>, opts?: CopilotChatOpts) =>
    postJson<CopilotResult>("/api/v1/copilot/chat", { messages, ...(opts || {}) }),
  baselines: () => request<{ baselines: BaselineRec[] }>("/api/v1/baselines"),
  baselineTrend: (metric?: string, scenario?: string) => request<BaselineTrend>(`/api/v1/baselines/trend${query({ metric, scenario })}`),
  baselineSave: (result: unknown, label?: string) => postJson<{ id: string; label: string; saved_at: number }>("/api/v1/baselines", { result, label }),
  baselineDelete: (id: string) => request<{ deleted: boolean }>(`/api/v1/baselines/${encodeURIComponent(id)}`, { method: "DELETE" }),
  baselineDiff: (current: unknown, baseline_id: string, threshold_pct = 5) => postJson<BaselineDiff>("/api/v1/baselines/diff", { current, baseline_id, threshold_pct }),
  updateStatus: () => request<UpdateStatus>("/api/v1/update/status"),
  updateCheck: () => request<UpdateCheck>("/api/v1/update/check"),
  updatePlan: (target_pin?: string) => postJson<UpdatePlan>("/api/v1/update/plan", { target_pin }),
  updateApply: (confirm: boolean, target_pin?: string) => postJson<UpdateApplyResult>("/api/v1/update/apply", { confirm, target_pin }),
  hostInventory: () => request<HostInventory>("/api/v1/host/inventory"),
  hostGpu: () => request<HardwareTelemetry>("/api/v1/host/gpu"),
  hostGpuRemote: (hostId: string) => request<HardwareTelemetry>(`/api/v1/hosts/${encodeURIComponent(hostId)}/gpu`),
  k8sStatus: () => request<K8sStatus>("/api/v1/k8s/status"),
  k8sNodes: () => request<K8sNodesResult>("/api/v1/k8s/nodes"),
  k8sPods: () => request<K8sPodsResult>("/api/v1/k8s/pods"),
  k8sEvents: (warningsOnly = false) => request<K8sEventsResult>(`/api/v1/k8s/events${warningsOnly ? "?warnings_only=true" : ""}`),
  k8sKubevirt: () => request<KubeVirtResult>("/api/v1/k8s/kubevirt"),
  proxmoxStatus: () => request<ProxmoxStatus>("/api/v1/proxmox/status"),
  proxmoxNodes: () => request<ProxmoxNodesResult>("/api/v1/proxmox/nodes"),
  proxmoxGuests: () => request<ProxmoxGuestsResult>("/api/v1/proxmox/guests"),
  proxmoxGuestDetail: (node: string, kind: string, vmid: number) =>
    request<ProxmoxGuestDetail>(`/api/v1/proxmox/guests/${encodeURIComponent(node)}/${kind}/${vmid}`),
  proxmoxNodeDetail: (node: string) =>
    request<ProxmoxNodeDetail>(`/api/v1/proxmox/nodes/${encodeURIComponent(node)}`),
  alerts: () => request<AlertsSnapshot>("/api/v1/alerts"),
  hostsReliability: () => request<ReliabilitySnapshot>("/api/v1/hosts/reliability"),
  flagsMatrix: (container?: string) => request<FlagMatrix>(`/api/v1/flags/matrix${query({ container })}`),
  license: () => request<LicenseStatus>("/api/v1/license"),
  routingArtifacts: () => request<RoutingArtifacts>("/api/v1/routing/artifacts"),
  routingActive: () => request<RoutingActive>("/api/v1/routing/active"),
  routingSetActive: (profile: string | null) => postJson<RoutingActive>("/api/v1/routing/active", { profile }),
  routingClassify: (signals: RoutingSignals, profile?: string) => postJson<RoutingClassify>("/api/v1/routing/classify", { signals, profile }),

  // Container management — one shape over both transports (local socket / host SSH).
  containers: (src: ContainerSource) =>
    request<{ containers: ManagedContainer[]; source: string }>(containerBase(src)),
  containerInspect: (src: ContainerSource, name: string) =>
    request<Record<string, unknown>>(`${containerBase(src)}/${encodeURIComponent(name)}`),
  containerLogs: (src: ContainerSource, name: string, tail = 200) =>
    request<{ container: string; logs: string }>(`${containerBase(src)}/${encodeURIComponent(name)}/logs${query({ tail })}`),
  containerStats: (src: ContainerSource, name: string) =>
    request<{ container: string; stats: ContainerStats }>(`${containerBase(src)}/${encodeURIComponent(name)}/stats`),
  // Batched: stats for ALL managed containers in one request (one SSH connection).
  containersStats: (src: ContainerSource) =>
    request<{ stats: Record<string, ContainerStats> }>(`${containerBase(src)}/stats`),
  containerAction: (src: ContainerSource, name: string, action: ContainerAction) =>
    postJson<{ ok: boolean; action: string; container: string }>(
      `${containerBase(src)}/${encodeURIComponent(name)}/action`, { action, confirm: true }),
  containerExec: (src: ContainerSource, name: string, argv: string[]) =>
    postJson<ContainerExecResult>(
      `${containerBase(src)}/${encodeURIComponent(name)}/exec`, { argv, confirm: true }),
  containerTop: (src: ContainerSource, name: string) =>
    request<ContainerTop & { container: string }>(`${containerBase(src)}/${encodeURIComponent(name)}/top`),
  containerChanges: (src: ContainerSource, name: string) =>
    request<{ container: string; changes: ContainerChange[] }>(`${containerBase(src)}/${encodeURIComponent(name)}/changes`),
  containerFs: (src: ContainerSource, name: string, path: string) =>
    request<{ path: string; entries: FsEntry[] }>(`${containerBase(src)}/${encodeURIComponent(name)}/fs${query({ path })}`),
  containerFile: (src: ContainerSource, name: string, path: string, maxBytes?: number) =>
    request<{ path: string; content: string; truncated: boolean }>(`${containerBase(src)}/${encodeURIComponent(name)}/file${query({ path, max_bytes: maxBytes })}`),
  containerUpdatePlan: (src: ContainerSource, name: string) =>
    request<ContainerUpdatePlan>(`${containerBase(src)}/${encodeURIComponent(name)}/update-plan`),
  // Project versions running INSIDE a container (SNDR Core + vLLM + config/patch
  // counts), introspected via a probe. Same shape as the Hosts fleet view.
  containerSndrState: (src: ContainerSource, name: string) =>
    request<HostSndrState>(`${containerBase(src)}/${encodeURIComponent(name)}/sndr-state`),
  containerPull: (src: ContainerSource, name: string, restart = false) =>
    postJson<{ ok: boolean; container: string; image: string; output: string; restarted?: boolean }>(
      `${containerBase(src)}/${encodeURIComponent(name)}/pull`, { confirm: true, restart }),
  // Choose the update mode (manual / semi / auto). Rejected (ok:false) for
  // critical containers asking for auto.
  containerSetUpdateMode: (src: ContainerSource, name: string, mode: UpdateMode) =>
    postJson<{ ok: boolean; mode: UpdateMode; error: string | null }>(
      `${containerBase(src)}/${encodeURIComponent(name)}/update-mode`, { mode }),
  // Recreate (stop+rm+create+start) so a new — or rolled-back — image actually
  // takes effect. Guarded server-side: refused for the management daemon + engines.
  containerRecreate: (src: ContainerSource, name: string, rollback = false) =>
    postJson<{ ok: boolean; container: string; recreated?: boolean; image?: string; rolled_back?: boolean; previous_image_id?: string }>(
      `${containerBase(src)}/${encodeURIComponent(name)}/recreate`, { confirm: true, rollback }),
  containerScan: (src: ContainerSource, name: string) =>
    request<ImageScan>(`${containerBase(src)}/${encodeURIComponent(name)}/scan`),
  containerSource: (src: ContainerSource, name: string) =>
    request<SourceReport>(`${containerBase(src)}/${encodeURIComponent(name)}/source`),
  containerEngine: (src: ContainerSource, name: string) =>
    request<{ reachable: boolean; port: number | null; status_code: number | null; reason?: string }>(`${containerBase(src)}/${encodeURIComponent(name)}/engine`),
  systemDf: (src: ContainerSource) =>
    request<SystemDf>(src.kind === "host" ? `/api/v1/hosts/${encodeURIComponent(src.hostId)}/system/df` : "/api/v1/system/df"),
  containerSettings: (src: ContainerSource, name: string, s: ContainerSettings) =>
    postJson<{ ok: boolean; container: string; updated: boolean }>(
      `${containerBase(src)}/${encodeURIComponent(name)}/settings`, { ...s, confirm: true }),
  containerNetwork: (src: ContainerSource, name: string, network: string, action: "connect" | "disconnect") =>
    postJson<{ ok: boolean; network: string; action: string }>(
      `${containerBase(src)}/${encodeURIComponent(name)}/network`, { network, action, confirm: true }),
  systemNetworks: (src: ContainerSource) =>
    request<{ networks: DockerNetwork[] }>(src.kind === "host" ? `/api/v1/hosts/${encodeURIComponent(src.hostId)}/system/networks` : "/api/v1/system/networks"),
  alertsConfig: () => request<AlertConfig>("/api/v1/alerts/config"),
  alertsSetConfig: (cfg: { enabled?: boolean; chat_id?: string; bot_token?: string }) => postJson<AlertConfig>("/api/v1/alerts/config", cfg),
  alertsTest: () => postJson<{ ok: boolean; error?: string }>("/api/v1/alerts/test", {}),
  // Live log stream (ND-JSON over fetch — bearer auth via header, no token in URL).
  containerLogStream: async (
    src: ContainerSource, name: string, tail: number,
    handlers: { onLine: (text: string) => void; onError: (msg: string) => void },
    signal?: AbortSignal
  ) => {
    const url = `${getApiBase()}${containerBase(src)}/${encodeURIComponent(name)}/logs/stream${query({ tail })}`;
    const response = await fetch(url, {
      credentials: sameOriginApi() ? "include" : "omit",
      headers: { ...authHeaders() },
      signal
    });
    if (!response.ok || !response.body) { handlers.onError(`Stream failed (${response.status})`); return; }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let nl: number;
      while ((nl = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, nl).trim();
        buffer = buffer.slice(nl + 1);
        if (!line) continue;
        try { const o = JSON.parse(line); if (typeof o.line === "string") handlers.onLine(o.line); } catch { /* skip */ }
      }
    }
  }
};
