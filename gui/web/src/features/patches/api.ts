// SPDX-License-Identifier: Apache-2.0
/**
 * Patches API client — typed wrappers over /api/v1/patches.
 *
 * Each function returns just the ``data`` slice of the envelope; the
 * request-id / meta block is handled by ``apiClient`` so callers can stay
 * focused on the resource itself.
 */
import { apiClient } from '@/api/client';

export type PatchLifecycle = 'experimental' | 'active' | 'deprecated' | 'retired';
export type PatchTier = 'community' | 'engine';

export interface PatchSummary {
  id: string;
  title: string;
  family: string;
  tier: PatchTier;
  lifecycle: PatchLifecycle;
  default_on: boolean;
  enabled_now: boolean;
}

export interface PatchDetail extends PatchSummary {
  description: string;
  apply_module: string | null;
  upstream_pr: string | null;
  vllm_version_range: string | null;
  conflicts_with: string[];
  superseded_by: string[];
  applies_to: Record<string, unknown>;
}

export interface PatchInventoryReport {
  total: number;
  active: number;
  retired: number;
  enabled_now: number;
  by_family: Record<string, number>;
  by_lifecycle: Record<string, number>;
  by_tier: Record<string, number>;
}

export interface ListPatchesFilters {
  family?: string;
  tier?: PatchTier;
  lifecycle?: PatchLifecycle;
  enabledOnly?: boolean;
}

function buildQuery(filters: ListPatchesFilters): string {
  const params = new URLSearchParams();
  if (filters.family) params.set('family', filters.family);
  if (filters.tier) params.set('tier', filters.tier);
  if (filters.lifecycle) params.set('lifecycle', filters.lifecycle);
  if (filters.enabledOnly) params.set('enabled_only', 'true');
  const qs = params.toString();
  return qs ? `?${qs}` : '';
}

export async function listPatches(
  filters: ListPatchesFilters = {},
): Promise<PatchSummary[]> {
  const response = await apiClient.get<PatchSummary[]>(
    `/api/v1/patches${buildQuery(filters)}`,
  );
  return response.data;
}

export async function getPatchInventory(): Promise<PatchInventoryReport> {
  const response = await apiClient.get<PatchInventoryReport>(
    '/api/v1/patches/inventory',
  );
  return response.data;
}

export async function getPatchDetail(patchId: string): Promise<PatchDetail> {
  const response = await apiClient.get<PatchDetail>(
    `/api/v1/patches/${encodeURIComponent(patchId)}`,
  );
  return response.data;
}
