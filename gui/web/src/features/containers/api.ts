// SPDX-License-Identifier: Apache-2.0
import { apiClient } from '@/api/client';

export type ContainerState = 'running' | 'paused' | 'exited' | 'restarting' | 'created' | 'dead' | 'unknown';

export interface ContainerPort {
  container_port: number;
  host_port: number | null;
  protocol: 'tcp' | 'udp';
}

export interface ContainerSummary {
  name: string;
  container_id: string;
  image: string;
  image_digest: string | null;
  state: ContainerState;
  status: string;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  served_model_name: string | null;
  engine: string | null;
  engine_pin: string | null;
  ports: ContainerPort[];
}

export interface ContainerDetail extends ContainerSummary {
  cmd: string[];
  env: Record<string, string>;
  mounts: Record<string, string>;
  labels: Record<string, string>;
  sndr_apply_summary: Record<string, number>;
}

export interface ContainerInventoryReport {
  total: number;
  by_state: Record<string, number>;
  by_engine: Record<string, number>;
}

export async function listContainers(engine?: string): Promise<ContainerSummary[]> {
  const qs = engine ? `?engine=${encodeURIComponent(engine)}` : '';
  return (await apiClient.get<ContainerSummary[]>(`/api/v1/containers${qs}`)).data;
}

export async function getContainerInventory(): Promise<ContainerInventoryReport> {
  return (await apiClient.get<ContainerInventoryReport>('/api/v1/containers/inventory')).data;
}

export async function getContainer(name: string): Promise<ContainerDetail> {
  return (await apiClient.get<ContainerDetail>(`/api/v1/containers/${encodeURIComponent(name)}`)).data;
}
