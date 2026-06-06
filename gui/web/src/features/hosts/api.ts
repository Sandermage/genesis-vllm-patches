// SPDX-License-Identifier: Apache-2.0
import { apiClient } from '@/api/client';

export interface GpuInfo {
  index: number;
  name: string;
  sm_capability: string | null;
  vram_total_mib: number;
  vram_used_mib: number;
  utilization_pct: number;
  temperature_c: number | null;
  power_draw_w: number | null;
}

export interface HostHardware {
  cpu_model: string;
  cpu_cores: number;
  ram_total_gib: number;
  ram_available_gib: number;
  gpus: GpuInfo[];
}

export interface HostSoftware {
  os_id: string;
  os_version: string;
  kernel: string;
  docker_version: string | null;
  nvidia_driver: string | null;
  cuda_version: string | null;
}

export type HostStatus = 'online' | 'degraded' | 'offline' | 'unknown';

export interface HostSummary {
  hostname: string;
  status: HostStatus;
  last_seen_at: string;
  sndr_version: string | null;
  sndr_install_root: string | null;
  active_engine: string | null;
  active_engine_pin: string | null;
  hardware: HostHardware | null;
  software: HostSoftware | null;
  notes: string | null;
}

export interface FleetReport {
  total_hosts: number;
  online: number;
  degraded: number;
  offline: number;
  unknown: number;
  total_gpus: number;
  total_vram_gib: number;
}

export async function listHosts(): Promise<HostSummary[]> {
  return (await apiClient.get<HostSummary[]>('/api/v1/hosts')).data;
}

export async function getLocalHost(): Promise<HostSummary> {
  return (await apiClient.get<HostSummary>('/api/v1/hosts/local')).data;
}

export async function getHost(hostname: string): Promise<HostSummary> {
  return (await apiClient.get<HostSummary>(`/api/v1/hosts/${encodeURIComponent(hostname)}`)).data;
}

export async function getFleetReport(): Promise<FleetReport> {
  return (await apiClient.get<FleetReport>('/api/v1/fleet')).data;
}
