// SPDX-License-Identifier: Apache-2.0
import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { BundlesPanel, UpstreamDiffPanel } from "./registry";

afterEach(cleanup);

describe("BundlesPanel", () => {
  it("renders an empty-state message when no bundles", () => {
    render(<BundlesPanel bundles={[]} />);
    expect(screen.getByText(/No multi-patch bundles/)).toBeTruthy();
  });

  it("renders a row per bundle with its umbrella flag", () => {
    render(
      <BundlesPanel
        bundles={[{ name: "moe-stack", tier: "community", umbrella_flag: "GENESIS_ENABLE_MOE", description: "MoE bundle" }] as never}
      />
    );
    expect(screen.getByText("moe-stack")).toBeTruthy();
    expect(screen.getByText("GENESIS_ENABLE_MOE")).toBeTruthy();
    expect(screen.getByText("MoE bundle")).toBeTruthy();
  });
});

describe("UpstreamDiffPanel", () => {
  it("renders a skeleton when report is null", () => {
    const { container } = render(<UpstreamDiffPanel report={null} />);
    expect(container.querySelector("[class*=skeleton]")).not.toBeNull();
  });

  it("renders the empty upstream-PR state", () => {
    render(<UpstreamDiffPanel report={{ has_upstream_pr: [], merged_upstream: [] } as never} />);
    expect(screen.getByText(/No patches currently track/)).toBeTruthy();
  });

  it("renders active upstream-PR rows", () => {
    render(
      <UpstreamDiffPanel
        report={{
          has_upstream_pr: [{ patch_id: "P94", title: "zero-alloc", upstream_pr: 41043, lifecycle: "active" }],
          merged_upstream: [{ patch_id: "PN9" }],
        } as never}
      />
    );
    expect(screen.getByText("P94")).toBeTruthy();
    expect(screen.getByText("#41043")).toBeTruthy();
  });
});
