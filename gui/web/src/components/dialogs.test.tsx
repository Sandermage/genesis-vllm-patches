// SPDX-License-Identifier: Apache-2.0
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";

vi.mock("../dialog", () => ({
  useDialogFocus: () => {},
  useEscapeKey: () => {},
  closeOnBackdrop: (fn: () => void) => () => fn(),
}));

import { ConfirmDialog, InfoDialog } from "./dialogs";

afterEach(cleanup);

describe("ConfirmDialog", () => {
  it("renders title/message and fires confirm + cancel", () => {
    const onConfirm = vi.fn();
    const onCancel = vi.fn();
    render(<ConfirmDialog title="Delete pin" message="Are you sure?" confirmLabel="Delete" danger onConfirm={onConfirm} onCancel={onCancel} />);
    expect(screen.getByText("Delete pin")).toBeTruthy();
    expect(screen.getByText("Are you sure?")).toBeTruthy();
    fireEvent.click(screen.getByText("Delete"));
    expect(onConfirm).toHaveBeenCalled();
    fireEvent.click(screen.getByText("Cancel"));
    expect(onCancel).toHaveBeenCalled();
  });
});

describe("InfoDialog", () => {
  it("renders the preview message and closes", () => {
    const onClose = vi.fn();
    render(<InfoDialog message="dry-run output" onClose={onClose} />);
    expect(screen.getByText("dry-run output")).toBeTruthy();
    fireEvent.click(screen.getByText("Close"));
    expect(onClose).toHaveBeenCalled();
  });
});
