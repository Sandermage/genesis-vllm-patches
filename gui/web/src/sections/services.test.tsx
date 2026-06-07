// SPDX-License-Identifier: Apache-2.0
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";

vi.mock("../api", () => ({
  api: { engineStatus: vi.fn(), authStatus: vi.fn(), servicePlan: vi.fn(), serviceApply: vi.fn() },
}));
import { api } from "../api";
import { ServiceLifecyclePlanner } from "./services";

afterEach(cleanup);

const plan = {
  container_name: "vllm-pn95", plan_id: "svc-1", mutating: false, action_reason: "read-only status probe",
  steps: [{ order: 1, title: "status", command: "docker ps", status: "ok" }],
  side_effects: [], gates: [], rollback: "no rollback needed",
};

function mock(applyEnabled = false) {
  (api.engineStatus as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({ reachable: true, version: "0.20.2", models: ["qwen"] });
  (api.authStatus as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({ apply_enabled: applyEnabled });
  (api.servicePlan as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(plan);
}

describe("ServiceLifecyclePlanner", () => {
  it("renders the resolved plan + engine status with aria-pressed action toggles", async () => {
    mock();
    render(<ServiceLifecyclePlanner selectedPreset="p27b" runtimeTarget="docker_compose" host="127.0.0.1" />);
    await waitFor(() => expect(screen.getByText("vllm-pn95")).toBeTruthy());
    // status is the default action and is pressed.
    const statusBtn = screen.getByRole("button", { name: "status" });
    expect(statusBtn.getAttribute("aria-pressed")).toBe("true");
    expect(screen.getByText(/engine up/)).toBeTruthy();
  });

  it("re-plans when a different action is selected", async () => {
    mock();
    render(<ServiceLifecyclePlanner selectedPreset="p27b" runtimeTarget="docker_compose" host="127.0.0.1" />);
    await waitFor(() => expect(screen.getByText("vllm-pn95")).toBeTruthy());
    fireEvent.click(screen.getByRole("button", { name: "restart" }));
    await waitFor(() =>
      expect((api.servicePlan as unknown as ReturnType<typeof vi.fn>).mock.calls.some((c) => c[0].action === "restart")).toBe(true)
    );
  });
});
