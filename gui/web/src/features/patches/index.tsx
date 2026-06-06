// SPDX-License-Identifier: Apache-2.0
/**
 * Patches feature module — public exports.
 *
 * Surface:
 *   PatchesView     — Carbon DataTable view (inventory + filters + table).
 *   listPatches     — API client for /api/v1/patches with filters.
 *   getPatchInventory — aggregate counts.
 *   getPatchDetail  — full detail for one patch.
 *   Types           — PatchSummary, PatchDetail, PatchInventoryReport.
 */
export { PatchesView, default } from './PatchesView';
export {
  getPatchDetail,
  getPatchInventory,
  listPatches,
} from './api';
export type {
  PatchDetail,
  PatchInventoryReport,
  PatchLifecycle,
  PatchSummary,
  PatchTier,
} from './api';
