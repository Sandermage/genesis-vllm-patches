// SPDX-License-Identifier: Apache-2.0
import { apiClient } from '@/api/client';

export type JobState = 'queued' | 'running' | 'succeeded' | 'failed' | 'canceled';

export interface JobSummary {
  id: string;
  kind: string;
  state: JobState;
  progress_pct: number;
  started_at: string | null;
  finished_at: string | null;
  operator: string | null;
  summary: string | null;
}

export async function listJobs(state?: JobState): Promise<JobSummary[]> {
  const qs = state ? `?state=${encodeURIComponent(state)}` : '';
  return (await apiClient.get<JobSummary[]>(`/api/v1/jobs${qs}`)).data;
}

export async function getJob(id: string): Promise<JobSummary> {
  return (await apiClient.get<JobSummary>(`/api/v1/jobs/${encodeURIComponent(id)}`)).data;
}
