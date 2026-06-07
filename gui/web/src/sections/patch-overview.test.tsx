// SPDX-License-Identifier: Apache-2.0
import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { PatchSummaryPanel, PatchLifecycleGraph, PatchRegistryInsight, PatchModelSupport } from "./patch-overview";

afterEach(cleanup);

const summary = {
  lifecycle_counts: { stable: 80, experimental: 20, retired: 4 },
  production_default_counts: { applied: 60, "opt-in": 30, blocked: 2 },
  implementation_status_counts: { full: 70, partial: 10 },
} as never;

describe("PatchSummaryPanel", () => {
  it("renders KPI totals + lifecycle/default rows", () => {
    render(<PatchSummaryPanel summary={summary} total={134} selectedCount={12} />);
    expect(screen.getByText("134")).toBeTruthy();
    expect(screen.getByText("lifecycle:stable")).toBeTruthy();
    expect(screen.getByText("default:applied")).toBeTruthy();
  });
});

describe("PatchLifecycleGraph", () => {
  it("renders both distribution sections", () => {
    render(<PatchLifecycleGraph summary={summary} />);
    expect(screen.getByText("Lifecycle distribution")).toBeTruthy();
    expect(screen.getByText("Production default behavior")).toBeTruthy();
  });
});

describe("PatchRegistryInsight", () => {
  it("renders implementation + family bars with the legend", () => {
    render(<PatchRegistryInsight summary={summary} patches={[{ family: "attention" }, { family: "moe" }] as never} />);
    expect(screen.getByText("Implementation status")).toBeTruthy();
    expect(screen.getByText("What the values mean")).toBeTruthy();
  });
});

describe("PatchModelSupport", () => {
  it("renders model chips", () => {
    render(<PatchModelSupport models={[{ id: "qwen3.6-27b" }]} />);
    expect(screen.getByText("qwen3.6-27b")).toBeTruthy();
  });
  it("renders empty-state when no models", () => {
    render(<PatchModelSupport models={[]} />);
    expect(screen.getByText("No models in the catalog.")).toBeTruthy();
  });
});
