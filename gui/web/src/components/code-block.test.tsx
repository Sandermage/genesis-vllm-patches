// SPDX-License-Identifier: Apache-2.0
import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";

// ../dialog is a thin focus/escape helper; stub it so the components render
// in isolation without real focus-trap side effects.
vi.mock("../dialog", () => ({
  useDialogFocus: () => {},
  closeOnBackdrop: (fn: () => void) => () => fn(),
}));

import { CopyButton, CodeBlock } from "./code-block";

afterEach(cleanup);

describe("CopyButton", () => {
  beforeEach(() => {
    Object.assign(navigator, { clipboard: { writeText: vi.fn().mockResolvedValue(undefined) } });
  });
  it("copies the value on click", async () => {
    render(<CopyButton value="hello" label="thing" />);
    const btn = screen.getByLabelText("Copy thing");
    fireEvent.click(btn);
    await waitFor(() => expect(navigator.clipboard.writeText).toHaveBeenCalledWith("hello"));
  });
});

describe("CodeBlock", () => {
  it("renders each line in a <pre>", () => {
    const { container } = render(<CodeBlock lines={["line a", "line b"]} />);
    const pre = container.querySelector("pre.code-block");
    expect(pre).not.toBeNull();
    expect(pre!.textContent).toContain("line a");
    expect(pre!.textContent).toContain("line b");
  });
  it("opens a fullscreen dialog on Expand", () => {
    render(<CodeBlock lines={["x"]} title="My Output" />);
    fireEvent.click(screen.getByLabelText("Expand to fullscreen"));
    expect(screen.getByRole("dialog")).toBeTruthy();
    expect(screen.getByText("My Output")).toBeTruthy();
  });
});
