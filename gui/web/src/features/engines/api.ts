// SPDX-License-Identifier: Apache-2.0
/** Engine API — typed wrappers over the sndr REST API. */
import { apiClient, type Envelope } from '@/api/client';

export interface EngineSummary {
  name: string;
  display_name: string;
  active: boolean;
  version: string | null;
  pin: string | null;
  container_count: number;
  notes: string[];
}

export interface EngineDetail extends EngineSummary {
  supported_pins: string[];
  patch_count_community: number;
  patch_count_engine: number;
  install_root: string | null;
  capabilities: Record<string, boolean>;
}

export async function listEngines(): Promise<EngineSummary[]> {
  const response = await apiClient.get<EngineSummary[]>('/api/v1/engines');
  return response.data;
}

export async function getEngine(name: string): Promise<EngineDetail> {
  const response = await apiClient.get<EngineDetail>(`/api/v1/engines/${name}`);
  return response.data;
}
