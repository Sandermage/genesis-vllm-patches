// SPDX-License-Identifier: Apache-2.0
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";

vi.mock("../api", () => ({
  api: { raw: vi.fn(), reportBundle: vi.fn() },
}));
import { api } from "../api";
import { EndpointExplorer, ReportGenerator } from "./api-explorer";

afterEach(cleanup);

describe("EndpointExplorer", () => {
  it("sends the selected endpoint and renders the JSON result + timing", async () => {
    (api.raw as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({ ok: true, value: 42 });
    render(<EndpointExplorer />);
    fireEvent.click(screen.getByText("Send"));
    await waitFor(() => expect(screen.getByText(/200 OK/)).toBeTruthy());
    expect(api.raw).toHaveBeenCalled();
  });

  it("surfaces an error result on a failed request", async () => {
    (api.raw as unknown as ReturnType<typeof vi.fn>).mockRejectedValue(new Error("boom"));
    render(<EndpointExplorer />);
    fireEvent.click(screen.getByText("Send"));
    await waitFor(() => expect(screen.getByText(/error ·/)).toBeTruthy());
  });
});

describe("ReportGenerator", () => {
  it("generates a redacted bundle and renders its id + files", async () => {
    (api.reportBundle as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      bundle_id: "bundle-7", redacted: true, report_type: "catalog",
      files: ["a.json", "b.json"], bundle_dir: "/sndr/bundles/7", note: "done",
    });
    render(<ReportGenerator selectedPreset="p27b" />);
    fireEvent.click(screen.getAllByText("Generate")[0]);
    await waitFor(() => expect(screen.getByText("bundle-7")).toBeTruthy());
    expect(screen.getByText("redacted")).toBeTruthy();
    expect(screen.getByText("a.json, b.json")).toBeTruthy();
  });
});
