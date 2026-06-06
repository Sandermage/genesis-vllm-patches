// SPDX-License-Identifier: Apache-2.0
/** Pins API — typed wrappers over /api/v1/engines/{engine}/pins. */
import { apiClient } from '@/api/client';

export type PinStatus = 'current' | 'previous' | 'staging' | 'deprecated' | 'retired';

export interface PinSummary {
  pin: string;
  status: PinStatus;
  full_version: string;
  upstream_sha: string | null;
  generated_at: string | null;
  has_manifest: boolean;
  has_drift: boolean;
  bench_tps_last: number | null;
}

export interface PinManifestSummary {
  pin: string;
  file_count: number;
  anchor_count: number;
  patch_count: number;
}

export async function listPins(engine: string): Promise<PinSummary[]> {
  const response = await apiClient.get<PinSummary[]>(`/api/v1/engines/${engine}/pins`);
  return response.data;
}

export async function getPinManifest(engine: string, pin: string): Promise<PinManifestSummary> {
  const response = await apiClient.get<PinManifestSummary>(`/api/v1/engines/${engine}/pins/${pin}`);
  return response.data;
}
