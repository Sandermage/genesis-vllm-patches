// SPDX-License-Identifier: Apache-2.0
import { apiClient } from '@/api/client';

export type BenchOutcome = 'pending' | 'running' | 'success' | 'regression' | 'failed';

export interface BenchSummary {
  id: string;
  timestamp: string;
  model: string;
  pin: string;
  wall_tps: number;
  decode_tpot_ms: number;
  ttft_ms: number;
  accept_rate: number | null;
  cv: number;
  n: number;
  outcome: BenchOutcome;
  delta_tps_vs_baseline: number | null;
}

export async function listBenchRuns(model?: string): Promise<BenchSummary[]> {
  const qs = model ? `?model=${encodeURIComponent(model)}` : '';
  return (await apiClient.get<BenchSummary[]>(`/api/v1/bench/runs${qs}`)).data;
}
