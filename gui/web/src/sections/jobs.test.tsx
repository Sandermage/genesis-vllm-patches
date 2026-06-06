// SPDX-License-Identifier: Apache-2.0
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";

vi.mock("../api", () => ({ api: { jobs: vi.fn(), job: vi.fn() } }));
vi.mock("../dialog", () => ({
  useDialogFocus: () => {},
  useEscapeKey: () => {},
  closeOnBackdrop: (fn: () => void) => () => fn(),
}));
import { api } from "../api";
import { JobsTable, Progress, JobMonitorModal } from "./jobs";

afterEach(cleanup);

const job = {
  job_id: "job-7", kind: "launch", status: "succeeded", dry_run: false, created_at: 1_700_000_000,
  steps: [{ order: 1, title: "render", command: "sndr render", status: "succeeded" }],
  log: ["line a"], note: "done", title: "Launch", progress: 100,
};

describe("Progress", () => {
  it("renders a clamped progressbar with aria values", () => {
    render(<Progress value={140} />);
    const bar = screen.getByRole("progressbar");
    expect(bar.getAttribute("aria-valuenow")).toBe("100");
    expect(bar.getAttribute("aria-valuemax")).toBe("100");
  });
});

describe("JobsTable", () => {
  it("renders job rows with scope=col headers after load", async () => {
    (api.jobs as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({ jobs: [job] });
    const { container } = render(<JobsTable onMonitor={vi.fn()} />);
    await waitFor(() => expect(screen.getByText("job-7")).toBeTruthy());
    expect(container.querySelectorAll('th[scope="col"]').length).toBe(6);
  });

  it("fires onMonitor without toggling the row open", async () => {
    (api.jobs as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({ jobs: [job] });
    const onMonitor = vi.fn();
    render(<JobsTable onMonitor={onMonitor} />);
    await waitFor(() => expect(screen.getByText("job-7")).toBeTruthy());
    fireEvent.click(screen.getByLabelText("Monitor job-7"));
    expect(onMonitor).toHaveBeenCalledWith("job-7");
  });
});

describe("JobMonitorModal", () => {
  it("polls the job and renders its terminal status + steps", async () => {
    (api.job as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(job);
    render(<JobMonitorModal jobId="job-7" onClose={vi.fn()} />);
    await waitFor(() => expect(screen.getByText("succeeded")).toBeTruthy());
    expect(screen.getByText("render")).toBeTruthy();
  });
});
