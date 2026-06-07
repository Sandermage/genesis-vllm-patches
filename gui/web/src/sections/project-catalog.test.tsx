// SPDX-License-Identifier: Apache-2.0
import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { ProjectCatalogPanel } from "./project-catalog";

afterEach(cleanup);

const overview = {
  catalog: {
    presets_count: 11, models_count: 4, profiles_count: 6, hardware_count: 3,
    preset_cards_count: 9, preset_load_error_count: 0,
    status_counts: { available: 8, missing: 3 },
    workload_counts: { free_chat: 5, code_gen: 4 },
    family_counts: { attention: 12, moe: 7 },
  },
  capabilities: { features: [{ status: "available" }, { status: "deferred" }] },
} as never;

describe("ProjectCatalogPanel", () => {
  it("renders a skeleton when overview is null", () => {
    const { container } = render(<ProjectCatalogPanel overview={null} environment={null} />);
    expect(container.querySelector("[class*=skeleton]")).not.toBeNull();
  });

  it("renders count tiles + annotation row + labelled chip distributions", () => {
    render(<ProjectCatalogPanel overview={overview} environment={{ engine_name: "vLLM", engine_version: "0.20.2" } as never} />);
    expect(screen.getByText("Presets")).toBeTruthy();
    expect(screen.getByText(/9\/11 · 82%/)).toBeTruthy();
    expect(screen.getByRole("group", { name: "Workload distribution" })).toBeTruthy();
    expect(screen.getByRole("group", { name: "Patch family distribution" })).toBeTruthy();
  });
});
