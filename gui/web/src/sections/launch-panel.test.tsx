// SPDX-License-Identifier: Apache-2.0
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
import { LaunchPanel } from "./launch-panel";

afterEach(cleanup);

const base = {
  selectedPreset: "a5000-2x-27b", model: "qwen3.6-27b", hardware: "2× A5000", profile: "tq",
  host: "127.0.0.1", composed: { max_model_len: 8192, kv_cache_dtype: "fp8" }, planSummary: {}, card: {},
  patchPolicy: "production", runtimeTitle: "vLLM", runtimeMode: "local" as const, endpoints: undefined,
  gateCounts: { pass: 4, warning: 1, blocked: 0, planned: 0 },
  actionReason: undefined, launchConfirm: false, launchBusy: false, launchSshTarget: "", launchJob: null,
  setLaunchConfirm: vi.fn(), onLaunch: vi.fn(), onConfigure: vi.fn(), onViewGates: vi.fn(),
};

describe("LaunchPanel", () => {
  it("shows the ready verdict + grouped readiness counts when no blockers", () => {
    render(<LaunchPanel {...base} gates={[{ id: "g", label: "engine", detail: "", status: "pass", action: "" }]} applyEnabled />);
    expect(screen.getByText("Ready — with warnings")).toBeTruthy();
    expect(screen.getByRole("group", { name: "Gate readiness summary" })).toBeTruthy();
    expect(screen.getByText(/4\/5 gates passing/)).toBeTruthy();
  });

  it("lists blockers and reflects the blocked verdict", () => {
    render(
      <LaunchPanel
        {...base}
        gateCounts={{ pass: 2, warning: 0, blocked: 1, planned: 0 }}
        gates={[{ id: "b", label: "GPU missing", detail: "no gpu", status: "blocked", action: "" }]}
        applyEnabled
      />
    );
    expect(screen.getByText("Launch blocked")).toBeTruthy();
    expect(screen.getByText("GPU missing")).toBeTruthy();
  });

  it("requires confirm before the launch button fires", () => {
    const onLaunch = vi.fn();
    const setLaunchConfirm = vi.fn();
    const { rerender } = render(
      <LaunchPanel {...base} gates={[]} applyEnabled onLaunch={onLaunch} setLaunchConfirm={setLaunchConfirm} launchConfirm={false} />
    );
    const go = screen.getByText("Launch model").closest("button")!;
    expect(go.disabled).toBe(true);
    fireEvent.click(screen.getByRole("checkbox"));
    expect(setLaunchConfirm).toHaveBeenCalledWith(true);
    rerender(<LaunchPanel {...base} gates={[]} applyEnabled onLaunch={onLaunch} setLaunchConfirm={setLaunchConfirm} launchConfirm={true} />);
    fireEvent.click(screen.getByText("Launch model"));
    expect(onLaunch).toHaveBeenCalled();
  });

  it("disables launch in read-only mode", () => {
    render(<LaunchPanel {...base} gates={[]} applyEnabled={false} />);
    expect(screen.getByText("Launch (read-only)").closest("button")!.disabled).toBe(true);
  });
});
