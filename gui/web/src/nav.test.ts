// SPDX-License-Identifier: Apache-2.0
import { describe, it, expect } from "vitest";
import { GATE_TARGET } from "./nav";

describe("GATE_TARGET", () => {
  it("routes a known gate id to its resolving section", () => {
    expect(GATE_TARGET.patch_doctor).toEqual({ section: "patches", label: "Open Patch Doctor" });
  });

  it("keeps dash and underscore spellings of the same gate in sync", () => {
    const pairs: Array<[string, string]> = [
      ["preset-card", "preset_card"],
      ["service-api", "service_lifecycle"],
      ["release-proof", "release_proof"],
    ];
    for (const [dash, underscore] of pairs) {
      expect(GATE_TARGET[dash]).toBeDefined();
      expect(GATE_TARGET[underscore]).toBeDefined();
      expect(GATE_TARGET[dash].section).toBe(GATE_TARGET[underscore].section);
    }
  });
});
