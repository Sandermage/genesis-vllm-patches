// SPDX-License-Identifier: Apache-2.0
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";
import { Box } from "lucide-react";

vi.mock("../api", () => ({
  api: { memoryFit: vi.fn(), calcKv: vi.fn() },
}));
import { api } from "../api";
import { CatalogCard, ModelFitCard, ModelFitMatrix, KvEnvelopeCard } from "./catalog-cards";

afterEach(cleanup);

const fit = {
  compatible: true,
  hardware_id: "a5000-2x",
  hardware_title: "2× A5000",
  checks: [{ id: "gpu", ok: true, severity: "ok", title: "GPU count", detail: "2 of 2" }],
  vram: { model_min_mib: 20000, rig_floor_mib: 24000, n_gpus: 2, vram_per_gpu_mib: 24564, gpu_memory_utilization: 0.9, kv_cache_dtype: "fp8", headroom_mib: 4096 },
};

describe("CatalogCard", () => {
  it("renders id/title/badges and fires onClick", () => {
    const onClick = vi.fn();
    render(<CatalogCard icon={<Box />} id="qwen3.6-27b" title="Qwen" badges={[{ label: "int4", tone: "accent" }]} active onClick={onClick} />);
    expect(screen.getByText("qwen3.6-27b")).toBeTruthy();
    expect(screen.getByText("int4")).toBeTruthy();
    fireEvent.click(screen.getByText("qwen3.6-27b"));
    expect(onClick).toHaveBeenCalled();
  });
});

describe("ModelFitCard", () => {
  it("loads a fit report and shows the compatibility verdict", async () => {
    (api.memoryFit as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(fit);
    render(<ModelFitCard modelId="qwen3.6-27b" hardwareOptions={["a5000-2x"]} defaultHardware="a5000-2x" />);
    await waitFor(() => expect(screen.getByText("Compatible")).toBeTruthy());
    expect(screen.getByText("GPU count")).toBeTruthy();
  });
});

describe("ModelFitMatrix", () => {
  it("probes every rig and summarizes fits/blocked", async () => {
    (api.memoryFit as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(fit);
    render(<ModelFitMatrix modelId="qwen3.6-27b" hardwareIds={["a5000-2x"]} />);
    await waitFor(() => expect(screen.getByText(/fits on 1/)).toBeTruthy());
    expect(screen.getByText("fits")).toBeTruthy();
  });
});

describe("KvEnvelopeCard", () => {
  it("renders an empty hint when no model key", () => {
    render(<KvEnvelopeCard modelKey={null} tp={2} vram={24564} rigLabel="2× A5000" />);
    expect(screen.getByText(/No KV sizing metadata/)).toBeTruthy();
  });

  it("renders the heatmap once the envelope resolves", async () => {
    (api.calcKv as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      envelope: { contexts: [4096], concurrencies: [1], grid: [[{ context: 4096, headroom_mib: 8000, fits: true }]] },
    });
    const { container } = render(<KvEnvelopeCard modelKey="qwen3.6-27b" tp={2} vram={24564} rigLabel="2× A5000" />);
    await waitFor(() => expect(container.querySelector(".heatmap")).not.toBeNull());
  });
});
