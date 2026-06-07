// SPDX-License-Identifier: Apache-2.0
// Runtime config draft model + pure helpers: build a draft from a composed
// config, render its YAML overlay, and diff a draft against its baseline.
// Extracted from App.tsx (modularization) with no behavior change.
import { asNumber, asText } from "./coerce";

export type RuntimeConfigDraft = {
  max_model_len: number;
  max_num_seqs: number;
  max_num_batched_tokens: number;
  gpu_memory_utilization: number;
  enable_chunked_prefill: boolean;
  enforce_eager: boolean;
  disable_custom_all_reduce: boolean;
  kv_cache_dtype: string;
  spec_decode_method: string;
  spec_decode_K: number;
  runtime_target: string;
  patch_policy: string;
};

export const DRAFT_FIELD_LABELS: Record<string, string> = {
  max_model_len: "Max context",
  max_num_seqs: "Max sequences",
  max_num_batched_tokens: "Max batched tokens",
  gpu_memory_utilization: "GPU memory util",
  enable_chunked_prefill: "Chunked prefill",
  enforce_eager: "Enforce eager",
  disable_custom_all_reduce: "Disable custom all-reduce",
  kv_cache_dtype: "KV cache dtype",
  spec_decode_method: "Spec method",
  spec_decode_K: "Spec K",
  runtime_target: "Runtime target",
  patch_policy: "Patch policy"
};

/** Derive an editable runtime draft from a composed config + target/policy. */
export function buildRuntimeDraft(
  composed: Record<string, unknown>,
  runtimeTarget: string,
  patchPolicy: string
): RuntimeConfigDraft {
  return {
    max_model_len: asNumber(composed.max_model_len) || 32768,
    max_num_seqs: asNumber(composed.max_num_seqs) || 1,
    max_num_batched_tokens: asNumber(composed.max_num_batched_tokens) || 4096,
    gpu_memory_utilization: asNumber(composed.gpu_memory_utilization) || 0.9,
    enable_chunked_prefill:
      typeof composed.enable_chunked_prefill === "boolean" ? composed.enable_chunked_prefill : true,
    enforce_eager: typeof composed.enforce_eager === "boolean" ? composed.enforce_eager : false,
    disable_custom_all_reduce:
      typeof composed.disable_custom_all_reduce === "boolean" ? composed.disable_custom_all_reduce : true,
    kv_cache_dtype: asText(composed.kv_cache_dtype, "auto"),
    spec_decode_method: asText(composed.spec_decode_method, "none"),
    spec_decode_K: asNumber(composed.spec_decode_K),
    runtime_target: runtimeTarget,
    patch_policy: patchPolicy
  };
}

/** Render a draft as the YAML runtime-overlay lines shown in the editor preview. */
export function buildDraftYaml(presetId: string, d: RuntimeConfigDraft): string[] {
  return [
    `# Draft runtime overlay for ${presetId}`,
    `runtime: ${d.runtime_target}`,
    `patch_policy: ${d.patch_policy}`,
    `sizing_override:`,
    `  max_model_len: ${d.max_model_len}`,
    `  max_num_seqs: ${d.max_num_seqs}`,
    `  max_num_batched_tokens: ${d.max_num_batched_tokens}`,
    `  gpu_memory_utilization: ${d.gpu_memory_utilization.toFixed(2)}`,
    `  enable_chunked_prefill: ${d.enable_chunked_prefill}`,
    `  enforce_eager: ${d.enforce_eager}`,
    `  disable_custom_all_reduce: ${d.disable_custom_all_reduce}`,
    `capabilities:`,
    `  kv_cache_dtype: ${d.kv_cache_dtype}`,
    `  spec_decode:`,
    `    method: ${d.spec_decode_method || "none"}`,
    `    num_speculative_tokens: ${d.spec_decode_K}`
  ];
}

/** Human-readable "label: base → draft" rows for every changed field. */
export function runtimeDraftDiff(base: RuntimeConfigDraft, draft: RuntimeConfigDraft): string[] {
  const rows: string[] = [];
  (Object.keys(base) as Array<keyof RuntimeConfigDraft>).forEach((key) => {
    if (base[key] !== draft[key]) {
      rows.push(`${DRAFT_FIELD_LABELS[key] ?? key}: ${base[key]} → ${draft[key]}`);
    }
  });
  return rows;
}
