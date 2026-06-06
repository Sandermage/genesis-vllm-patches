// SPDX-License-Identifier: Apache-2.0
import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
import { DoctorCoveragePanel, AdminSurfaceMatrix } from "./patch-doctor";

afterEach(cleanup);

const report = {
  registry_size: 134,
  coverage: { total: 134, mapped: 130, unmapped: ["P71", "PN26b"], intentionally_unmapped: ["PX"] },
  issues: [
    { patch_id: "P1", severity: "ERROR", message: "missing anchor" },
    { patch_id: "P2", severity: "WARNING", message: "stale range" },
    { patch_id: "P3", severity: "INFO", message: "note" },
  ],
};

describe("DoctorCoveragePanel", () => {
  it("renders coverage rows + the unmapped chips", () => {
    render(<DoctorCoveragePanel report={report as never} />);
    expect(screen.getByText(/130 of 134 patches/)).toBeTruthy();
    expect(screen.getByText("P71")).toBeTruthy();
    expect(screen.getByText("missing anchor")).toBeTruthy();
  });

  it("filters validation issues by severity chip", () => {
    render(<DoctorCoveragePanel report={report as never} />);
    // Click the ERROR chip → only the ERROR issue's message remains.
    fireEvent.click(screen.getByText(/error 1/));
    expect(screen.getByText("missing anchor")).toBeTruthy();
    expect(screen.queryByText("stale range")).toBeNull();
  });
});

describe("AdminSurfaceMatrix", () => {
  it("renders the surface rows with patch-doctor counts", () => {
    render(<AdminSurfaceMatrix featureRows={[] as never} patchDoctor={report as never} />);
    expect(screen.getByText("Patch Inventory")).toBeTruthy();
    expect(screen.getByText("134 registry entries")).toBeTruthy();
    expect(screen.getByText("3 validation issues")).toBeTruthy();
  });
});
