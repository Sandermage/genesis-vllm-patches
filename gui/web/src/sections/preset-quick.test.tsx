// SPDX-License-Identifier: Apache-2.0
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
import { PresetQuickPanel } from "./preset-quick";

afterEach(cleanup);

const cbs = { onOpenCard: vi.fn(), onEdit: vi.fn(), onPolicy: vi.fn(), onLaunch: vi.fn() };

describe("PresetQuickPanel", () => {
  it("renders the select-a-preset empty-state when none selected", () => {
    render(<PresetQuickPanel selectedPreset="" record={null} card={{}} composed={{}} {...cbs} />);
    expect(screen.getByText("Select a preset")).toBeTruthy();
  });

  it("renders the runtime summary + workload group + actions", () => {
    render(
      <PresetQuickPanel
        selectedPreset="a5000-2x-27b"
        record={{ has_card: true, model: "qwen3.6-27b", hardware: "2× A5000", profile: "tq" } as never}
        card={{ status: "available", title: "27B TQ", workload_allow: ["free_chat"], mode: "balanced" }}
        composed={{ max_model_len: 8192, kv_cache_dtype: "fp8" }}
        {...cbs}
      />
    );
    expect(screen.getByText("a5000-2x-27b")).toBeTruthy();
    expect(screen.getByText("8K")).toBeTruthy();
    expect(screen.getByRole("group", { name: "Allowed workloads" })).toBeTruthy();
  });

  it("fires the action callbacks", () => {
    const onEdit = vi.fn();
    const onLaunch = vi.fn();
    render(<PresetQuickPanel selectedPreset="p" record={null} card={{}} composed={{}} {...cbs} onEdit={onEdit} onLaunch={onLaunch} />);
    fireEvent.click(screen.getByText("Edit preset"));
    fireEvent.click(screen.getByText("Launch"));
    expect(onEdit).toHaveBeenCalled();
    expect(onLaunch).toHaveBeenCalled();
  });
});
