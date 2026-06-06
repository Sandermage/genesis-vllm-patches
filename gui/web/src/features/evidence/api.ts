// SPDX-License-Identifier: Apache-2.0
import { apiClient } from '@/api/client';

export type EvidenceStatus = 'ok' | 'warning' | 'fail' | 'skipped';

export interface EvidenceGate {
  id: string;
  name: string;
  status: EvidenceStatus;
  summary: string | null;
  last_run: string | null;
}

export interface EvidenceReport {
  gates_total: number;
  gates_ok: number;
  gates_warning: number;
  gates_fail: number;
  gates_skipped: number;
  gates: EvidenceGate[];
}

export async function getEvidenceReport(): Promise<EvidenceReport> {
  return (await apiClient.get<EvidenceReport>('/api/v1/evidence')).data;
}
