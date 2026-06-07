// SPDX-License-Identifier: Apache-2.0
// Element field schema + editor: the curated per-kind field specs, adaptive
// discovery of extra scalar fields, grouping, per-field validation and the
// ElementField input dispatcher. Extracted from App.tsx (modularization) with
// no behavior change. Shared by LayerEditor and ConfigElementEditor.
 
import { AlertTriangle } from "lucide-react";
import { TextField, NumberField, BoolField, SelectField } from "../components/form-fields";

export type ElementKind = "model" | "hardware" | "profile" | "preset";
export type FieldSpec = {
  path: string;
  label: string;
  type: "text" | "number" | "select" | "bool";
  options?: string[];
  group?: string;
  hint?: string;
};

export const ELEMENT_FIELDS: Record<ElementKind, FieldSpec[]> = {
  model: [
    { path: "title", label: "Title", type: "text", group: "Identity" },
    { path: "served_model_name", label: "Served name", type: "text", group: "Identity" },
    { path: "model_path", label: "Model path", type: "text", group: "Identity", hint: "Container/host checkpoint path" },
    { path: "maintainer", label: "Maintainer", type: "text", group: "Identity" },
    { path: "license", label: "License", type: "text", group: "Identity" },
    { path: "last_validated", label: "Last validated", type: "text", group: "Identity" },
    { path: "dtype", label: "Dtype", type: "select", options: ["float16", "bfloat16", "float32"], group: "Precision" },
    { path: "quantization", label: "Quantization", type: "text", group: "Precision" },
    { path: "trust_remote_code", label: "Trust remote code", type: "bool", group: "Precision" },
    { path: "capabilities.attention_arch", label: "Attention arch", type: "select", options: ["dense", "hybrid_gdn_moe", "hybrid_mamba", "moe", "gemma4_dense", "gemma4_moe"], group: "Capabilities" },
    { path: "capabilities.kv_cache_dtype", label: "KV cache dtype", type: "select", options: ["auto", "fp8", "turboquant_k8v4", "turboquant_k8v8", "int8"], group: "Capabilities" },
    { path: "capabilities.tool_call_parser", label: "Tool parser", type: "text", group: "Capabilities" },
    { path: "capabilities.reasoning_parser", label: "Reasoning parser", type: "text", group: "Capabilities" },
    { path: "capabilities.enable_auto_tool_choice", label: "Auto tool choice", type: "bool", group: "Capabilities" },
    { path: "capabilities.spec_decode.method", label: "Spec method", type: "select", options: ["mtp", "ngram", "eagle"], group: "Speculative decode" },
    { path: "capabilities.spec_decode.num_speculative_tokens", label: "Spec K", type: "number", group: "Speculative decode" },
    { path: "requires.min_gpu_count", label: "Min GPUs", type: "number", group: "Requirements" },
    { path: "requires.min_total_vram_mib", label: "Min VRAM (MiB)", type: "number", group: "Requirements" },
    { path: "versions.genesis_pin_min", label: "Genesis pin (min)", type: "text", group: "Version pins" },
    { path: "versions.vllm_pin_required", label: "vLLM pin required", type: "text", group: "Version pins" },
    { path: "versions.reference_metrics_ref", label: "Reference metrics ref", type: "text", group: "Version pins" }
  ],
  hardware: [
    { path: "title", label: "Title", type: "text", group: "Identity" },
    { path: "maintainer", label: "Maintainer", type: "text", group: "Identity" },
    { path: "hardware.n_gpus", label: "GPU count", type: "number", group: "GPU" },
    { path: "hardware.min_vram_per_gpu_mib", label: "Min VRAM/GPU (MiB)", type: "number", group: "GPU" },
    { path: "hardware.cuda_capability_min", label: "CUDA cap min", type: "text", group: "GPU", hint: "e.g. 8.6 (Ampere)" },
    { path: "sizing.max_model_len", label: "Max context", type: "number", group: "Sizing" },
    { path: "sizing.max_num_seqs", label: "Max sequences", type: "number", group: "Sizing" },
    { path: "sizing.max_num_batched_tokens", label: "Max batched tokens", type: "number", group: "Sizing" },
    { path: "sizing.gpu_memory_utilization", label: "GPU mem util", type: "number", group: "Sizing" },
    { path: "sizing.enable_chunked_prefill", label: "Chunked prefill", type: "bool", group: "Sizing" },
    { path: "sizing.enforce_eager", label: "Enforce eager", type: "bool", group: "Sizing" },
    { path: "sizing.disable_custom_all_reduce", label: "Disable custom all-reduce", type: "bool", group: "Sizing" },
    { path: "runtime.default", label: "Default runtime", type: "select", options: ["docker", "podman", "bare-metal"], group: "Runtime" }
  ],
  profile: [
    { path: "parent_model", label: "Parent model", type: "text", group: "Identity" },
    { path: "status", label: "Status", type: "select", options: ["experimental", "validated", "promoted"], group: "Identity" },
    { path: "role", label: "Role", type: "select", options: ["default", "structured", "gateway", "bench", "dev", "qa", "diagnostic"], group: "Identity" },
    { path: "created", label: "Created", type: "text", group: "Identity" },
    { path: "sizing_override.max_model_len", label: "Max context", type: "number", group: "Sizing override", hint: "Leave empty to inherit hardware" },
    { path: "sizing_override.max_num_seqs", label: "Max sequences", type: "number", group: "Sizing override" },
    { path: "sizing_override.max_num_batched_tokens", label: "Max batched tokens", type: "number", group: "Sizing override" },
    { path: "sizing_override.gpu_memory_utilization", label: "GPU mem util", type: "number", group: "Sizing override" },
    { path: "sizing_override.enforce_eager", label: "Enforce eager", type: "bool", group: "Sizing override" },
    { path: "versions_override.vllm_pin_required", label: "vLLM pin required", type: "text", group: "Version override" },
    { path: "versions_override.genesis_pin", label: "Genesis pin", type: "text", group: "Version override" },
    { path: "promotion.promote_to", label: "Promote to", type: "text", group: "Promotion" },
    { path: "promotion.notes", label: "Promotion notes", type: "text", group: "Promotion" }
  ],
  preset: [
    { path: "model", label: "Model", type: "text", group: "Composition" },
    { path: "hardware", label: "Hardware", type: "text", group: "Composition" },
    { path: "profile", label: "Profile", type: "text", group: "Composition" },
    { path: "runtime", label: "Runtime", type: "select", options: ["docker", "podman", "kubernetes", "systemd", "bare-metal"], group: "Composition" },
    { path: "card.title", label: "Card title", type: "text", group: "Card" },
    { path: "card.summary", label: "Summary", type: "text", group: "Card" },
    { path: "card.status", label: "Card status", type: "select", options: ["experimental", "production_candidate", "production"], group: "Card" },
    { path: "card.mode", label: "Mode", type: "text", group: "Card" },
    { path: "card.audience", label: "Audience", type: "select", options: ["operator", "developer", "internal"], group: "Card" },
    { path: "card.evidence_visibility", label: "Evidence visibility", type: "select", options: ["public", "private", "mixed"], group: "Card" },
    { path: "card.fallback_preset", label: "Fallback preset", type: "text", group: "Card" }
  ]
};

// Top-level keys excluded from auto-discovery: structural, noisy, or shown
// elsewhere (patch matrix), or arrays/dicts that need bespoke editors.
const _AUTO_EXCLUDE = new Set([
  "patches", "patches_attribution", "notes", "schema_version", "kind", "id",
  "patches_delta", "system_env"
]);

function _isScalar(v: any): boolean {
  return v === null || ["string", "number", "boolean"].includes(typeof v);
}

// Adaptive discovery: walk the loaded definition and surface every scalar leaf
// that the curated schema does not already cover, so the editor reflects
// whatever fields a given model/hardware/profile/preset actually contains.
export function discoverExtraFields(obj: any, known: Set<string>): FieldSpec[] {
  const out: FieldSpec[] = [];
  const walk = (node: any, prefix: string): void => {
    if (!node || typeof node !== "object" || Array.isArray(node)) return;
    for (const [key, value] of Object.entries(node)) {
      const path = prefix ? `${prefix}.${key}` : key;
      if (!prefix && _AUTO_EXCLUDE.has(key)) continue;
      if (_isScalar(value)) {
        if (!known.has(path)) {
          const type = typeof value === "boolean" ? "bool" : typeof value === "number" ? "number" : "text";
          out.push({ path, label: key, type, group: prefix ? `More · ${prefix}` : "More" });
        }
      } else if (value && typeof value === "object" && !Array.isArray(value)) {
        walk(value, path);
      }
      // arrays / arrays-of-objects are left to the YAML panel (bespoke shape)
    }
  };
  walk(obj, "");
  return out;
}

export function groupFields(fields: FieldSpec[]): Array<[string, FieldSpec[]]> {
  const order: string[] = [];
  const byGroup = new Map<string, FieldSpec[]>();
  for (const spec of fields) {
    const group = spec.group ?? "";
    if (!byGroup.has(group)) { byGroup.set(group, []); order.push(group); }
    byGroup.get(group)!.push(spec);
  }
  return order.map((group) => [group, byGroup.get(group)!]);
}

// Live sanity-check for the most error-prone numeric config fields — catches a
// bad value (e.g. gpu_memory_utilization 1.5) before it's saved/applied.
function fieldWarning(spec: FieldSpec, value: any): string | null {
  if (value == null || value === "" || spec.type !== "number") return null;
  const n = Number(value);
  if (!Number.isFinite(n)) return "not a number";
  const leaf = spec.path.split(".").pop();
  switch (leaf) {
    case "gpu_memory_utilization": return n > 0 && n <= 1 ? null : "expected 0 < util ≤ 1";
    case "max_num_seqs": return Number.isInteger(n) && n >= 1 && n <= 4096 ? null : "expected 1–4096";
    case "max_num_batched_tokens": return n >= 256 ? null : "expected ≥ 256";
    case "max_model_len": return n >= 256 ? null : "expected ≥ 256";
    case "num_speculative_tokens": return n >= 0 && n <= 16 ? null : "expected 0–16";
    case "n_gpus":
    case "min_gpu_count": return Number.isInteger(n) && n >= 1 && n <= 8 ? null : "expected 1–8";
    default: return n < 0 ? "must be ≥ 0" : null;
  }
}

export function ElementField({ spec, value, onChange }: { spec: FieldSpec; value: any; onChange: (value: any) => void }) {
  const warn = fieldWarning(spec, value);
  const field = (() => {
    if (spec.type === "bool") {
      return <BoolField label={spec.label} value={Boolean(value)} onChange={onChange} />;
    }
    if (spec.type === "number") {
      return <NumberField label={spec.label} value={typeof value === "number" ? value : Number(value) || 0} onChange={onChange} />;
    }
    if (spec.type === "select") {
      const current = value == null ? "" : String(value);
      const options = spec.options ?? [];
      const merged = current && !options.includes(current) ? [current, ...options] : options;
      return <SelectField label={spec.label} value={current} options={merged} onChange={onChange} />;
    }
    return <TextField label={spec.label} value={value == null ? "" : String(value)} onChange={onChange} />;
  })();
  if (!spec.hint && !warn) return field;
  return (
    <div className={`element-field-hinted${warn ? " invalid" : ""}`}>
      {field}
      {warn
        ? <small className="element-field-warn"><AlertTriangle size={11} /> {warn}</small>
        : spec.hint ? <small className="element-field-hint">{spec.hint}</small> : null}
    </div>
  );
}

