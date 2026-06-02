// Compile-time API contract guard.
//
// Every route the hand-written client (`src/api.ts`) calls is listed here and
// asserted against the OpenAPI-generated `paths` type in `schema.gen.ts`. If
// the daemon renames or drops a route, regenerating the schema makes this file
// fail `tsc` — catching frontend/backend drift at build time instead of at
// runtime. Regenerate with `npm run gen:api`; check drift with `npm run check:api`.
import type { paths } from "./schema.gen";

// `satisfies` keeps the literal types while forcing each entry to be a real
// key of the generated `paths` interface.
export const CLIENT_ROUTES = [
  "/api/v1/health",
  "/api/v1/auth/status",
  "/api/v1/overview",
  "/api/v1/capabilities",
  "/api/v1/catalog/summary",
  "/api/v1/presets",
  "/api/v1/presets/recommend",
  "/api/v1/presets/{preset_id}",
  "/api/v1/presets/{preset_id}/explain",
  "/api/v1/configs/v2/catalog",
  "/api/v1/configs/v2/preview",
  "/api/v1/configs/v2/plan",
  "/api/v1/configs/v2/apply",
  "/api/v1/configs/v2/user-presets",
  "/api/v1/configs/v2/layer/apply",
  "/api/v1/configs/v2/layer/{kind}/{layer_id}",
  "/api/v1/launch/plan",
  "/api/v1/patches",
  "/api/v1/patches/{patch_id}/explain",
  "/api/v1/patches/doctor",
  "/api/v1/patches/bundles",
  "/api/v1/patches/diff-upstream",
  "/api/v1/proof/status",
  "/api/v1/operations",
  "/api/v1/operations/run",
  "/api/v1/auth/tokens",
  "/api/v1/auth/tokens/{token_id}",
  "/api/v1/doctor",
  "/api/v1/environment",
  "/api/v1/deploy/targets",
  "/api/v1/deploy/plan",
  "/api/v1/memory/fit",
  "/api/v1/models/cache",
  "/api/v1/services/plan",
  "/api/v1/services/apply",
  "/api/v1/jobs",
  "/api/v1/jobs/{job_id}",
  "/api/v1/hosts",
  "/api/v1/hosts/{host_id}",
  "/api/v1/hosts/probe",
  "/api/v1/host/inventory",
  "/api/v1/events/recent",
  "/api/v1/reports/bundle"
] as const satisfies readonly (keyof paths)[];

export type ClientRoute = (typeof CLIENT_ROUTES)[number];
