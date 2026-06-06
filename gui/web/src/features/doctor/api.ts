// SPDX-License-Identifier: Apache-2.0
import { apiClient } from '@/api/client';

export type DoctorSeverity = 'info' | 'warning' | 'error' | 'critical';

export interface DoctorFinding {
  category: string;
  severity: DoctorSeverity;
  title: string;
  detail: string | null;
  remediation: string | null;
}

export interface DoctorReport {
  checked_at: string;
  ok: boolean;
  findings: DoctorFinding[];
  counts: Record<string, number>;
}

export async function runDoctor(): Promise<DoctorReport> {
  return (await apiClient.get<DoctorReport>('/api/v1/doctor')).data;
}
