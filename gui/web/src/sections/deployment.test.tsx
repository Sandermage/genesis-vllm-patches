// SPDX-License-Identifier: Apache-2.0
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, waitFor } from "@testing-library/react";

vi.mock("../api", () => ({
  api: { deployTargets: vi.fn(), deployPlan: vi.fn() },
}));
import { api } from "../api";
import { DeploymentConsole } from "./deployment";

afterEach(cleanup);

const targets = {
  targets: [
    { id: "compose", label: "Docker Compose", filename: "compose.yaml", summary: "single host", needs: null },
  ],
  host: {
    os: { distro: "Ubuntu", system: "Linux", arch: "x86_64" },
    docker: { installed: true, daemon_running: true, server_version: "27.0" },
    nvidia: { installed: true, n_gpus: 2, gpu_names: ["A5000", "A5000"] },
    vllm: { installed: true, version: "0.20.2" },
  },
};

const plan = {
  parameters: { tensor_parallel: 2, kv_cache_dtype: "fp8", max_model_len: 8192, genesis_pin: "dev338" },
  dependencies: { is_ready: true, n_blockers: 0, n_warnings: 0, items: [] },
  mount_vars: [],
  artifact: { filename: "compose.yaml", content: "services:\n  vllm: {}" },
  commands: ["docker compose up -d"],
};

describe("DeploymentConsole", () => {
  it("renders deployment targets + host readiness once metadata loads", async () => {
    (api.deployTargets as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(targets);
    (api.deployPlan as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(plan);
    render(<DeploymentConsole presets={null} selectedPreset="" onSelectPreset={vi.fn()} />);
    await waitFor(() => expect(screen.getByText("Docker Compose")).toBeTruthy());
    expect(screen.getByText("Host readiness")).toBeTruthy();
  });

  it("debounce-renders the resolved plan artifact for a selected preset", async () => {
    (api.deployTargets as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(targets);
    (api.deployPlan as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(plan);
    render(<DeploymentConsole presets={{ presets: [{ id: "p27b" }] } as never} selectedPreset="p27b" onSelectPreset={vi.fn()} />);
    await waitFor(() => expect(screen.getAllByText("compose.yaml").length).toBeGreaterThan(0), { timeout: 2000 });
    expect(screen.getByText("Apply commands")).toBeTruthy();
  });
});
