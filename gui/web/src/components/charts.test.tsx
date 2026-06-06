// SPDX-License-Identifier: Apache-2.0
import { describe, it, expect, afterEach, vi } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
import { SegmentBar, PercentBar, BarList, OvKpi, segmentsFromCounts } from "./charts";

afterEach(cleanup);

describe("segmentsFromCounts", () => {
  it("drops zeros and sorts by value descending", () => {
    expect(segmentsFromCounts({ a: 1, b: 0, c: 5 })).toEqual([
      { label: "c", value: 5, color: undefined },
      { label: "a", value: 1, color: undefined },
    ]);
  });
  it("applies a color map when given", () => {
    expect(segmentsFromCounts({ a: 2 }, { a: "#fff" })).toEqual([
      { label: "a", value: 2, color: "#fff" },
    ]);
  });
});

describe("SegmentBar", () => {
  it("renders a legend entry per segment with percentages", () => {
    const { container } = render(
      <SegmentBar segments={[{ label: "ok", value: 3 }, { label: "bad", value: 1 }]} total={4} totalLabel="checks" />,
    );
    expect(container.querySelectorAll(".segment-legend > div").length).toBe(2);
    expect(screen.getByText("75%")).toBeTruthy();
    expect(screen.getByText("25%")).toBeTruthy();
  });
  it("does not divide by zero when total + values are 0", () => {
    expect(() => render(<SegmentBar segments={[]} total={0} totalLabel="x" />)).not.toThrow();
  });
});

describe("PercentBar", () => {
  it("computes a clamped percentage", () => {
    render(<PercentBar value={50} max={200} label="util" />);
    expect(screen.getByText("25")).toBeTruthy();
  });
  it("max=0 → 0% (no divide-by-zero)", () => {
    render(<PercentBar value={5} max={0} label="x" />);
    expect(screen.getByText("0")).toBeTruthy();
  });
});

describe("BarList", () => {
  it("renders one row per entry", () => {
    const { container } = render(<BarList rows={[["a", 50, "5"], ["b", 100, "10"]]} />);
    expect(container.querySelectorAll(".bar-list > div").length).toBe(2);
  });
});

describe("OvKpi", () => {
  it("renders a div when not clickable", () => {
    const { container } = render(<OvKpi icon={<i />} label="Servers" value={3} />);
    expect(container.querySelector("div.ov-kpi")).not.toBeNull();
    expect(screen.getByText("Servers")).toBeTruthy();
  });
  it("renders a clickable button and fires onClick", () => {
    const onClick = vi.fn();
    const { container } = render(<OvKpi icon={<i />} label="GPUs" value={2} onClick={onClick} />);
    const btn = container.querySelector("button.ov-kpi.clickable");
    expect(btn).not.toBeNull();
    fireEvent.click(btn!);
    expect(onClick).toHaveBeenCalledOnce();
  });
});
