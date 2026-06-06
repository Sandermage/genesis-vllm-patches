// SPDX-License-Identifier: Apache-2.0
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
import { SetupWizard } from "./setup-wizard";

afterEach(cleanup);

const env = {
  engine_version: "0.20.2", engine_installed: true, engine_name: "vLLM",
  sndr_core_version: "12.1", python_version: "3.11.6", os_name: "Linux", machine: "x86_64",
  dependencies: [{ name: "vllm", present: true, version: "0.20.2" }],
  tools: [{ name: "docker", present: true }, { name: "nvidia-smi", present: true }],
};
const doctor = { summary: { ok: 5, warning: 1, blocked: 0 }, findings: [{}, {}] };
const gateCounts = { pass: 4, warning: 1, blocked: 0, planned: 0 };

function renderWizard(extra: Record<string, unknown> = {}) {
  return render(
    <SetupWizard
      environment={env as never}
      overview={{ catalog: { presets_count: 11, models_count: 4, profiles_count: 6 } } as never}
      doctorReport={doctor as never}
      gateCounts={gateCounts as never}
      selectedPreset="a5000-2x-27b"
      runtimeMode="local"
      apiBase="http://127.0.0.1:8765"
      onSection={vi.fn()}
      {...extra}
    />
  );
}

describe("SetupWizard", () => {
  it("exposes a real progressbar reflecting ready steps", () => {
    renderWizard();
    const bar = screen.getByRole("progressbar");
    // detect+mode+preset+validate done (blocked=0, warn=1 -> warning not done),
    // launch done (selectedPreset set, no blockers) -> 4 of 5.
    expect(bar.getAttribute("aria-valuemax")).toBe("5");
    expect(bar.getAttribute("aria-valuenow")).toBe("4");
    expect(bar.getAttribute("aria-valuetext")).toContain("of 5 steps ready");
  });

  it("marks the active step with aria-current and switches on click", () => {
    renderWizard();
    const detectStep = screen.getByRole("button", { name: /Detect host/ });
    expect(detectStep.getAttribute("aria-current")).toBe("step");
    fireEvent.click(screen.getByRole("button", { name: /Connection mode/ }));
    expect(screen.getByRole("button", { name: /Connection mode/ }).getAttribute("aria-current")).toBe("step");
    // Mode step surfaces the API base.
    expect(screen.getByText("http://127.0.0.1:8765")).toBeTruthy();
  });

  it("navigates to a section from a step action", () => {
    const onSection = vi.fn();
    renderWizard({ onSection });
    fireEvent.click(screen.getByRole("button", { name: /Choose a preset/ }));
    fireEvent.click(screen.getByText("Browse presets"));
    expect(onSection).toHaveBeenCalledWith("presets");
  });
});
