// SPDX-License-Identifier: Apache-2.0
import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { RuntimeEnvelopePanel, PresetPolicyGraph } from "./preset-insight";

afterEach(cleanup);

describe("RuntimeEnvelopePanel", () => {
  it("renders the envelope bars + KV/spec rows", () => {
    render(
      <RuntimeEnvelopePanel
        card={{ primary_metric: { value: 120 }, evidence_visibility: "public" }}
        composed={{ max_model_len: 8192, max_num_seqs: 4, kv_cache_dtype: "fp8", spec_decode_method: "ngram", spec_decode_K: 3, enabled_patches_count: 10 }}
        patchCount={40}
      />
    );
    expect(screen.getByText("Context")).toBeTruthy();
    expect(screen.getByText("8K")).toBeTruthy();
    expect(screen.getByText("fp8")).toBeTruthy();
    expect(screen.getByText("public")).toBeTruthy();
  });
});

describe("PresetPolicyGraph", () => {
  it("renders allow/deny pills + status distribution", () => {
    render(
      <PresetPolicyGraph
        card={{ workload_allow: ["free_chat"], workload_deny: ["tool_call"] }}
        presets={[
          { has_card: true, card: { status: "available" } },
          { has_card: true, card: { status: "available" } },
          { has_card: true, card: { status: "missing" } },
        ] as never}
      />
    );
    expect(screen.getByText("free_chat")).toBeTruthy();
    expect(screen.getByText("tool_call")).toBeTruthy();
    expect(screen.getByText("available")).toBeTruthy();
  });
});
