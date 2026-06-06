// SPDX-License-Identifier: Apache-2.0
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
import { GateRow } from "./gate-row";

afterEach(cleanup);

const gate = { id: "patch_doctor", label: "Patch doctor", detail: "3 issues", status: "warning" as const, action: "Run sndr patches doctor" };

describe("GateRow", () => {
  it("renders collapsed with the status and a disclosure trigger", () => {
    render(<GateRow gate={gate} />);
    expect(screen.getByText("Patch doctor")).toBeTruthy();
    const trigger = screen.getByRole("button", { name: /Patch doctor/ });
    expect(trigger.getAttribute("aria-expanded")).toBe("false");
  });

  it("wires aria-controls to the detail region id when expanded (WCAG disclosure)", () => {
    render(<GateRow gate={gate} />);
    const trigger = screen.getByRole("button", { name: /Patch doctor/ });
    fireEvent.click(trigger);
    expect(trigger.getAttribute("aria-expanded")).toBe("true");
    const controls = trigger.getAttribute("aria-controls");
    expect(controls).toBeTruthy();
    expect(document.getElementById(controls!)).not.toBeNull();
  });

  it("navigates to the resolving section via the gate target", () => {
    const onNavigate = vi.fn();
    render(<GateRow gate={gate} onNavigate={onNavigate} />);
    fireEvent.click(screen.getByRole("button", { name: /Patch doctor/ }));
    fireEvent.click(screen.getByText("Open Patch Doctor"));
    expect(onNavigate).toHaveBeenCalledWith("patches");
  });
});
