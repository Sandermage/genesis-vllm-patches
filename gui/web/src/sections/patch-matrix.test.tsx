// SPDX-License-Identifier: Apache-2.0
import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
import { PatchMatrixViewer } from "./patch-matrix";

afterEach(cleanup);

const patches = { GENESIS_ENABLE_P94: "1", GENESIS_ENABLE_P71: "0", GENESIS_ENABLE_PN95: "1" };

describe("PatchMatrixViewer", () => {
  it("renders a skeleton while loading with no entries", () => {
    render(<PatchMatrixViewer patches={{}} attribution={{}} loading />);
    expect(screen.getByRole("status", { name: "Loading patch matrix…" })).toBeTruthy();
  });

  it("renders an empty-state when the model defines no overrides", () => {
    render(<PatchMatrixViewer patches={{}} attribution={{}} loading={false} />);
    expect(screen.getByText(/no canonical patch overrides/)).toBeTruthy();
  });

  it("renders the flag table with enabled count + filters via the labelled input", () => {
    render(<PatchMatrixViewer patches={patches} attribution={{ "attr-X1": { role: "load-bearing" } }} loading={false} />);
    expect(screen.getByText(/2 of 3 env flags on/)).toBeTruthy();
    expect(screen.getByText("P94")).toBeTruthy();
    const filter = screen.getByLabelText("Filter env flags");
    fireEvent.change(filter, { target: { value: "P71" } });
    expect(screen.getByText("P71")).toBeTruthy();
    expect(screen.queryByText("P94")).toBeNull();
  });
});
