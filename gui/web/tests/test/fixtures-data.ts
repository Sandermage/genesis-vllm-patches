// SPDX-License-Identifier: Apache-2.0
// Single source of truth for test API fixtures, shared by the jsdom shell test
// (tests/test/api-fixtures.ts, method-keyed) and the hermetic Playwright E2E
// (e2e/fixtures.ts, URL-keyed). PURE DATA — no vitest/playwright imports — so
// both runtimes can import it. Shapes mirror the api.ts types with empty content
// so every panel reaches its empty state instead of crashing on a missing field.

const CAPABILITIES = {
  platform: {
    public_brand: "SNDR Control Center", package_name: "sndr",
    sndr_core_version: "0.0.0-test", os_name: "linux", machine: "x86_64",
    python_version: "3.12.0", engine_installed: false,
  },
  runtime_targets: [], features: [], warnings: [],
};

const CATALOG = {
  models_count: 0, hardware_count: 0, profiles_count: 0, presets_count: 0,
  preset_cards_count: 0, unannotated_presets_count: 0, preset_load_error_count: 0,
  status_counts: {}, workload_counts: {}, family_counts: {},
  default_presets: [], preset_load_errors: [],
};

const HOST_INVENTORY = {
  os: { system: "Linux", release: "6.0", distro: "ubuntu", arch: "x86_64" },
  python: { binary_path: "/usr/bin/python3", version: "3.12.0", implementation: "CPython", venv_active: false, pip_present: true, pip_version: "24.0" },
  docker: { installed: false, binary_path: null, version: null, daemon_running: false, server_version: null, nvidia_runtime_present: false, notes: "" },
  nvidia: { installed: false, driver_version: null, cuda_version: null, n_gpus: 0, gpu_names: [], gpu_total_vram_mib: [], notes: "" },
  vllm: { installed: false, version: null, location: null },
};

const ENVIRONMENT = {
  brand: "SNDR", package_name: "sndr", sndr_core_version: "0.0.0-test",
  engine_name: "vLLM", engine_version: null, engine_installed: false,
  python_version: "3.12.0", os_name: "linux", machine: "x86_64",
  dependencies: [], tools: [],
};

/**
 * Canonical fixtures keyed by api method name. The jsdom mock keys off these
 * directly; the E2E maps URLs to them via URL_TABLE below.
 */
export const RESPONSES: Record<string, unknown> = {
  overview: { capabilities: CAPABILITIES, catalog: CATALOG },
  capabilities: CAPABILITIES,
  recommendPresets: { query: {}, results: [], total_matches: 0, total_candidates: 0 },
  presets: { filters: {}, matched: 0, total: 0, presets: [], load_errors: [] },
  explainPreset: { id: "", card: {}, composed: {}, fallback_diff: null },
  patchDoctor: { registry_size: 0, issues: [], coverage: { total: 0, mapped: 0, unmapped: [], intentionally_unmapped: [] } },
  bundles: { bundles: [] },
  diffUpstream: { generated_at: 0, patches: [], summary: {} },
  patchOverrides: { overrides: {} },
  patches: { filters: {}, matched: 0, total: 0, patches: [], summary: { tier_counts: {}, lifecycle_counts: {}, production_default_counts: {}, implementation_status_counts: {} } },
  v2ConfigCatalog: { models: [], hardware: [], profiles: [], presets: [] },
  v2ConfigPreview: { selection: {}, compatible: true, status: "ok", messages: [], composed: {}, draft_yaml: "" },
  userPresets: { presets: [] },
  launchPlan: { plan_id: "", preset_id: "", runtime_target: "docker", patch_policy: "", mode: "single", host: "", actionable: false, action_reason: "", summary: {}, gates: [], endpoints: [], artifacts: [], cli_mirror: [], events: [] },
  proofStatus: { generated_at: 0, sources: [], summary: {} },
  hostInventory: HOST_INVENTORY,
  doctor: { findings: [], summary: {}, categories: [], generated_for: "test", warnings: [] },
  environment: ENVIRONMENT,
  hosts: { hosts: [] },
  eventsRecent: { events: [], latest_seq: 0 },
  modelsCache: { host: "local", total: 0, present_count: 0, models: [] },
  memoryFit: { model_id: "", hardware_id: "", model_title: "", hardware_title: "", compatible: true, checks: [], vram: { model_min_mib: 0, rig_floor_mib: 0, headroom_mib: 0, n_gpus: 0 } },
  calcModels: { models: {}, kv_dtypes: {} },
  calcKv: {
    arch: { name: "", num_layers: 0, num_kv_heads: 0, head_dim: 0, params_b: 0, weight_bits: 16, is_moe: false, active_params_b: null },
    kv_dtype: "fp16", overhead_mib: 0,
    rig: { tp: 1, gpu_count: 1, gpu_vram_mib: 0, util: 0.9 },
    result: { model: "", weights_per_gpu_mib: 0, kv_per_gpu_mib: 0, kv_total_mib: 0, overhead_mib: 0, total_per_gpu_mib: 0, budget_per_gpu_mib: 0, headroom_mib: 0, fits: true, max_context: 0, kv_bytes_per_token: 0, tp: 1, concurrency: 1, context: 0 },
    by_dtype: {}, by_tp: {}, curve: [],
    envelope: { contexts: [], concurrencies: [], grid: [] },
    recommendation: [], arch_advice: null,
  },
  copilotTools: { tools: [] },
  servicePlan: { plan_id: "", preset_id: "", action: "status", runtime_target: "docker", host: "", container_name: "", mutating: false, actionable: false, action_reason: "", steps: [], side_effects: [], gates: [], cli_mirror: [], rollback: "" },
  containersStats: { stats: {} },
  systemDf: { types: [], total_size: 0 },
  containers: { containers: [], source: "local" },
  baselineTrend: { points: [], metric: "", scenario: "" },
  baselines: { baselines: [] },
  operations: { operations: [], apply_enabled: false },
  caveats: { caveats: [], total: 0, triggered_count: 0, host_facts_available: false, facts_error: null },
  configKeys: { keys: {}, total: 0, by_source: {} },
  traces: { traces: [], categories: [], by_category: {}, total: 0 },
  deployTargets: { targets: [], host: HOST_INVENTORY },
  jobs: { jobs: [] },
  authStatus: { auth_required: false, apply_enabled: false, backends: [], oauth_providers: [], context: { in_container: false, system_user: "test", pam_enabled: false }, user: null },
  alerts: { active: [], recent: [], counts: { critical: 0, warn: 0, info: 0 } },
};

/**
 * URL-suffix → method-key, for the E2E catch-all matcher (first whose suffix the
 * pathname includes wins). Ordered most-specific first so e.g. /containers/stats
 * resolves before /containers and /patches/doctor before /patches.
 */
export const URL_TABLE: Array<[string, string]> = [
  ["/api/v1/overview", "overview"],
  ["/api/v1/capabilities", "capabilities"],
  ["/api/v1/presets/recommend", "recommendPresets"],
  ["/api/v1/presets", "presets"],
  ["/explain", "explainPreset"],
  ["/api/v1/patches/doctor", "patchDoctor"],
  ["/api/v1/patches/bundles", "bundles"],
  ["/api/v1/patches/diff-upstream", "diffUpstream"],
  ["/api/v1/patches/overrides", "patchOverrides"],
  ["/api/v1/patches", "patches"],
  ["/api/v1/configs/v2/catalog", "v2ConfigCatalog"],
  ["/api/v1/configs/v2/preview", "v2ConfigPreview"],
  ["/api/v1/configs/v2/user-presets", "userPresets"],
  ["/api/v1/launch/plan", "launchPlan"],
  ["/api/v1/proof/status", "proofStatus"],
  ["/api/v1/host/inventory", "hostInventory"],
  ["/api/v1/doctor", "doctor"],
  ["/api/v1/environment", "environment"],
  ["/api/v1/hosts", "hosts"],
  ["/api/v1/events/recent", "eventsRecent"],
  ["/api/v1/models/cache", "modelsCache"],
  ["/api/v1/memory/fit", "memoryFit"],
  ["/api/v1/calc/models", "calcModels"],
  ["/api/v1/calc/kv", "calcKv"],
  ["/api/v1/copilot/tools", "copilotTools"],
  ["/api/v1/services/plan", "servicePlan"],
  ["/api/v1/containers/stats", "containersStats"],
  ["/api/v1/system/df", "systemDf"],
  ["/api/v1/containers", "containers"],
  ["/api/v1/baselines/trend", "baselineTrend"],
  ["/api/v1/baselines", "baselines"],
  ["/api/v1/operations", "operations"],
  ["/api/v1/caveats", "caveats"],
  ["/api/v1/config-keys", "configKeys"],
  ["/api/v1/traces", "traces"],
  ["/api/v1/deploy/targets", "deployTargets"],
  ["/api/v1/jobs", "jobs"],
  ["/api/v1/auth/status", "authStatus"],
  ["/api/v1/alerts", "alerts"],
];
