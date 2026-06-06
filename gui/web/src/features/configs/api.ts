// SPDX-License-Identifier: Apache-2.0
import { apiClient } from '@/api/client';

export interface ModelSummary {
  id: string;
  title: string | null;
  served_model_name: string | null;
  family: string | null;
  quant_format: string | null;
  kv_cache_dtype: string | null;
  spec_method: string | null;
  parameter_count: string | null;
}

export interface HardwareSummary {
  id: string;
  title: string | null;
  gpu: string | null;
  gpu_count: number | null;
  vram_per_gpu_gib: number | null;
  cpu_cores: number | null;
  ram_gib: number | null;
}

export interface ProfileSummary {
  id: string;
  title: string | null;
  parent_model: string | null;
  role: string | null;
}

export interface PresetSummary {
  id: string;
  title: string | null;
  composed_key: string | null;
  parent_model: string | null;
}

export interface ConfigCatalog {
  models: ModelSummary[];
  hardware: HardwareSummary[];
  profiles: ProfileSummary[];
  presets: PresetSummary[];
}

export async function getConfigCatalog(): Promise<ConfigCatalog> {
  return (await apiClient.get<ConfigCatalog>('/api/v1/configs')).data;
}
