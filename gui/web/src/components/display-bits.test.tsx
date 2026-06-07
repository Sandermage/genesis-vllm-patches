// SPDX-License-Identifier: Apache-2.0
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
import { PlanChip, KeyValue, ArtifactPreview } from "./display-bits";

afterEach(cleanup);

describe("PlanChip + KeyValue", () => {
  it("render label/value pairs", () => {
    render(<><PlanChip label="Runtime" value="docker" /><KeyValue label="GPUs" value={2} /></>);
    expect(screen.getByText("Runtime")).toBeTruthy();
    expect(screen.getByText("docker")).toBeTruthy();
    expect(screen.getByText("GPUs")).toBeTruthy();
    expect(screen.getByText("2")).toBeTruthy();
  });
});

describe("ArtifactPreview", () => {
  it("renders a WCAG tablist + the active artifact content", () => {
    render(
      <ArtifactPreview
        artifacts={[{ kind: "compose", title: "compose.yaml", content: "services: {}" }] as never}
        activeTab="compose"
        setActiveTab={vi.fn()}
      />
    );
    expect(screen.getAllByRole("tab")).toHaveLength(4);
    expect(screen.getByRole("tab", { name: "Compose" }).getAttribute("aria-selected")).toBe("true");
    expect(screen.getByText("compose.yaml")).toBeTruthy();
  });

  it("switches tab + falls back when an artifact is missing", () => {
    const setActiveTab = vi.fn();
    render(<ArtifactPreview artifacts={[]} activeTab="env" setActiveTab={setActiveTab} />);
    expect(screen.getByText(/Waiting for launch plan/)).toBeTruthy();
    fireEvent.click(screen.getByRole("tab", { name: "CLI Commands" }));
    expect(setActiveTab).toHaveBeenCalledWith("commands");
  });
});
