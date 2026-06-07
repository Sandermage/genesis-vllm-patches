// SPDX-License-Identifier: Apache-2.0
import { describe, it, expect } from "vitest";
import { buildRuntimeDraft, buildDraftYaml, runtimeDraftDiff } from "./runtime-draft";

describe("buildRuntimeDraft", () => {
  it("derives a draft from a composed config, applying defaults", () => {
    const d = buildRuntimeDraft({ max_model_len: 8192, kv_cache_dtype: "fp8" }, "docker", "production");
    expect(d.max_model_len).toBe(8192);
    expect(d.kv_cache_dtype).toBe("fp8");
    expect(d.runtime_target).toBe("docker");
    expect(d.patch_policy).toBe("production");
    // defaults fill the gaps
    expect(d.gpu_memory_utilization).toBe(0.9);
    expect(d.enable_chunked_prefill).toBe(true);
  });
});

describe("buildDraftYaml", () => {
  it("renders the runtime overlay header + sizing", () => {
    const d = buildRuntimeDraft({ max_model_len: 8192 }, "docker", "prod");
    const yaml = buildDraftYaml("p27b", d);
    expect(yaml[0]).toBe("# Draft runtime overlay for p27b");
    expect(yaml).toContain("  max_model_len: 8192");
    expect(yaml).toContain("  gpu_memory_utilization: 0.90");
  });
});

describe("runtimeDraftDiff", () => {
  it("lists only the changed fields with friendly labels", () => {
    const base = buildRuntimeDraft({ max_model_len: 8192 }, "docker", "prod");
    const draft = { ...base, max_num_seqs: 16, enforce_eager: true };
    const rows = runtimeDraftDiff(base, draft);
    expect(rows).toContain("Max sequences: 1 → 16");
    expect(rows.some((r) => r.startsWith("Enforce eager:"))).toBe(true);
    expect(rows.some((r) => r.startsWith("Max context:"))).toBe(false); // unchanged
  });
});
