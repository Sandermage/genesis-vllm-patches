// SPDX-License-Identifier: Apache-2.0
import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor, act } from "@testing-library/react";
import { ToastHost, toast } from "./toast";

afterEach(cleanup);

describe("toast + ToastHost", () => {
  it("renders a dispatched toast and dismisses it on click", async () => {
    render(<ToastHost />);
    act(() => toast("saved ok", "success"));
    await waitFor(() => expect(screen.getByText("saved ok")).toBeTruthy());
    fireEvent.click(screen.getByLabelText("Dismiss"));
    expect(screen.queryByText("saved ok")).toBeNull();
  });

  it("marks error toasts with role=alert", async () => {
    render(<ToastHost />);
    act(() => toast("it broke", "error"));
    await waitFor(() => expect(screen.getByRole("alert")).toBeTruthy());
    expect(screen.getByText("it broke")).toBeTruthy();
  });
});
