// SPDX-License-Identifier: Apache-2.0
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
import { ElementField, discoverExtraFields, groupFields, ELEMENT_FIELDS } from "./element-fields";

afterEach(cleanup);

describe("ELEMENT_FIELDS", () => {
  it("defines curated specs for every element kind", () => {
    for (const kind of ["model", "hardware", "profile", "preset"] as const) {
      expect(Array.isArray(ELEMENT_FIELDS[kind])).toBe(true);
      expect(ELEMENT_FIELDS[kind].length).toBeGreaterThan(0);
    }
  });
});

describe("discoverExtraFields", () => {
  it("surfaces uncovered scalar leaves the curated schema misses", () => {
    const extra = discoverExtraFields({ a: 1, nested: { b: "x" }, arr: [1] }, new Set(["a"]));
    const paths = extra.map((f) => f.path);
    expect(paths).toContain("nested.b");
    expect(paths).not.toContain("a"); // already known
    expect(paths).not.toContain("arr"); // arrays left to YAML
  });
});

describe("groupFields", () => {
  it("groups specs by their group, preserving first-seen order", () => {
    const grouped = groupFields([
      { path: "x", label: "X", type: "text", group: "G1" },
      { path: "y", label: "Y", type: "text", group: "G2" },
      { path: "z", label: "Z", type: "text", group: "G1" },
    ]);
    expect(grouped.map(([g]) => g)).toEqual(["G1", "G2"]);
    expect(grouped[0][1].map((s) => s.path)).toEqual(["x", "z"]);
  });
});

describe("ElementField", () => {
  it("renders a bool toggle for a bool spec and emits the toggle", () => {
    const onChange = vi.fn();
    render(<ElementField spec={{ path: "p", label: "Eager", type: "bool" }} value={false} onChange={onChange} />);
    fireEvent.click(screen.getByRole("button", { name: /Eager/ }));
    expect(onChange).toHaveBeenCalledWith(true);
  });

  it("shows a validation warning for an out-of-range numeric field", () => {
    render(<ElementField spec={{ path: "gpu_memory_utilization", label: "Util", type: "number" }} value={1.5} onChange={vi.fn()} />);
    expect(screen.getByText(/expected 0 < util/)).toBeTruthy();
  });
});
