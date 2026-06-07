// SPDX-License-Identifier: Apache-2.0
// Minimal valid-shaped API fixtures for jsdom shell tests. Mirrors the boot
// path in App.tsx so a full <App /> render reaches the "ready" dashboard with
// empty data — exercising the orchestration + SectionWorkspace routing under
// coverage. Network methods not listed fall back to a resolved {} via the Proxy
// in makeApiMock (sufficient for sections that load lazily on navigation).
import { vi } from "vitest";

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

/** Boot-path responses keyed by api method name. */
export const API_FIXTURES: Record<string, unknown> = {
  overview: { capabilities: CAPABILITIES, catalog: CATALOG },
  capabilities: CAPABILITIES,
  presets: { filters: {}, matched: 0, total: 0, presets: [], load_errors: [] },
  recommendPresets: { query: {}, results: [], total_matches: 0, total_candidates: 0 },
  patches: { filters: {}, matched: 0, total: 0, patches: [], summary: { tier_counts: {}, lifecycle_counts: {}, production_default_counts: {}, implementation_status_counts: {} } },
  patchDoctor: { registry_size: 0, issues: [], coverage: { total: 0, mapped: 0, unmapped: [], intentionally_unmapped: [] } },
  v2ConfigCatalog: { models: [], hardware: [], profiles: [], presets: [] },
  explainPreset: { id: "", card: {}, composed: {}, fallback_diff: null },
  launchPlan: { plan_id: "", preset_id: "", runtime_target: "docker", patch_policy: "", mode: "single", host: "", actionable: false, action_reason: "", summary: {}, gates: [], endpoints: [], artifacts: [], cli_mirror: [], events: [] },
  v2ConfigPreview: { selection: {}, compatible: true, status: "ok", messages: [], composed: {}, draft_yaml: "" },
  bundles: { bundles: [] },
  diffUpstream: { generated_at: 0, patches: [], summary: {} },
  proofStatus: { generated_at: 0, sources: [], summary: {} },
  userPresets: { presets: [] },
  doctor: { generated_at: 0, checks: [], summary: {}, counts: { critical: 0, warn: 0, info: 0 } },
  environment: { generated_at: 0, sections: [], summary: {}, sndr_core_version: "0.0.0-test", engine_version: null },
  hosts: { hosts: [] },
  authStatus: { auth_required: false, apply_enabled: false, backends: [], oauth_providers: [], context: { in_container: false, system_user: "test", pam_enabled: false }, user: null },
  eventsRecent: { events: [], latest_seq: 0 },
  alerts: { active: [], recent: [], counts: { critical: 0, warn: 0, info: 0 } },
  jobs: { jobs: [] },
};

/**
 * Build a mock `api` from the real one: the data-fetching methods named in
 * API_FIXTURES resolve to their fixture; every other member (synchronous URL
 * builders like oauthLoginUrl, helpers, and network methods only hit on user
 * action) keeps its real implementation. This preserves the sync-vs-async
 * contract — overriding everything with a Promise breaks `{api.oauthLoginUrl()}`
 * style direct renders.
 */
export function makeApiMock<T extends object>(realApi: T, overrides: Record<string, unknown> = {}): T {
  const cache = new Map<string, ReturnType<typeof vi.fn>>();
  // A Proxy delegates lazily: it never spreads realApi, so its getters (e.g.
  // the `baseUrl` accessor that reads localStorage) fire only on real access —
  // after the test's browser stubs are in place.
  return new Proxy(realApi, {
    get(target, prop, receiver) {
      if (typeof prop === "string") {
        if (prop in overrides) return overrides[prop];
        if (prop in API_FIXTURES) {
          if (!cache.has(prop)) cache.set(prop, vi.fn().mockResolvedValue(API_FIXTURES[prop]));
          return cache.get(prop);
        }
      }
      return Reflect.get(target, prop, receiver);
    },
  });
}
