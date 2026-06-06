// SPDX-License-Identifier: Apache-2.0
/** Drift API — typed wrappers over the drift detection endpoints. */
import { apiClient } from '@/api/client';

export type DriftSeverity = 'ok' | 'benign' | 'drift' | 'blocked';

export interface AnchorDrift {
  anchor_name: string;
  severity: DriftSeverity;
  expected_md5: string | null;
  actual_md5: string | null;
  notes: string | null;
}

export interface FileDrift {
  file_path: string;
  severity: DriftSeverity;
  anchors: AnchorDrift[];
  affected_patches: string[];
}

export interface DriftSummary {
  engine: string;
  pin: string;
  checked_at: string;
  overall_severity: DriftSeverity;
  files_ok: number;
  files_benign: number;
  files_drift: number;
  files_blocked: number;
  affected_patches: string[];
}

export async function getDriftSummary(engine: string, pin: string): Promise<DriftSummary> {
  const response = await apiClient.get<DriftSummary>(`/api/v1/engines/${engine}/pins/${pin}/drift`);
  return response.data;
}
