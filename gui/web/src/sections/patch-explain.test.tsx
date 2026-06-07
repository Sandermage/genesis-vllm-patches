// SPDX-License-Identifier: Apache-2.0
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
import { PatchExplainPanel } from "./patch-explain";

afterEach(cleanup);

const patch = {
  patch_id: "P94", title: "zero-alloc prepare", lifecycle: "stable", production_default: "applied",
  tier: "community", family: "spec_decode", implementation_status: "full", env_flag: "GENESIS_ENABLE_P94",
  apply_module: "sndr.p94", upstream_pr: 41043,
} as never;

const detail = {
  spec: { applies_to: { is_turboquant: "true" }, requires_patches: ["P70"], conflicts_with: [] },
  meta: {}, live_decision: [true, "enabled by default"],
} as never;

function renderPanel(extra: Record<string, unknown> = {}) {
  return render(
    <PatchExplainPanel
      patch={patch} detail={detail} state="ready" error={null} override="default" overrideCount={0}
      onSetOverride={vi.fn()} allPatchIds={new Set(["P70", "P94"])} onSelectPatch={vi.fn()}
      {...extra}
    />
  );
}

describe("PatchExplainPanel", () => {
  it("renders an empty-state when no patch selected", () => {
    render(<PatchExplainPanel patch={null} detail={null} state="idle" error={null} override="default" overrideCount={0} onSetOverride={vi.fn()} allPatchIds={new Set()} onSelectPatch={vi.fn()} />);
    expect(screen.getByText("No patch selected")).toBeTruthy();
  });

  it("renders the override group with aria-pressed + live decision", () => {
    renderPanel();
    expect(screen.getByRole("group", { name: "Enablement override" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Registry default" }).getAttribute("aria-pressed")).toBe("true");
    expect(screen.getByText(/Live decision: apply/)).toBeTruthy();
  });

  it("fires onSetOverride when force-on is clicked", () => {
    const onSetOverride = vi.fn();
    renderPanel({ onSetOverride });
    fireEvent.click(screen.getByRole("button", { name: "Force on" }));
    expect(onSetOverride).toHaveBeenCalledWith("on");
  });

  it("navigates to a required patch via its chip link", () => {
    const onSelectPatch = vi.fn();
    renderPanel({ onSelectPatch });
    fireEvent.click(screen.getByText("P70 →"));
    expect(onSelectPatch).toHaveBeenCalledWith("P70");
  });
});
