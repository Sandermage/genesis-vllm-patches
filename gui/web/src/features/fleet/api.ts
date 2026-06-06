// SPDX-License-Identifier: Apache-2.0
import { apiClient } from '@/api/client';

export interface FleetReport {
  total_hosts: number;
  online: number;
  degraded: number;
  offline: number;
  unknown: number;
  total_gpus: number;
  total_vram_gib: number;
}

export async function getFleetReport(): Promise<FleetReport> {
  return (await apiClient.get<FleetReport>('/api/v1/fleet')).data;
}
