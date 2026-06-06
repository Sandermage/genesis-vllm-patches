// SPDX-License-Identifier: Apache-2.0
import { describe, it, expect } from "vitest";
import { asRecord, asText, asNumber, asStringArray, countRecord } from "./coerce";

describe("asRecord", () => {
  it("passes through a plain object", () => {
    const o = { a: 1 };
    expect(asRecord(o)).toBe(o);
  });
  it("rejects arrays, null, primitives → {}", () => {
    expect(asRecord([1, 2])).toEqual({});
    expect(asRecord(null)).toEqual({});
    expect(asRecord(undefined)).toEqual({});
    expect(asRecord("x")).toEqual({});
    expect(asRecord(5)).toEqual({});
  });
});

describe("asText", () => {
  it("returns a non-empty string as-is", () => {
    expect(asText("hi", "fb")).toBe("hi");
  });
  it("falls back on empty/whitespace/non-string", () => {
    expect(asText("", "fb")).toBe("fb");
    expect(asText("   ", "fb")).toBe("fb");
    expect(asText(null, "fb")).toBe("fb");
    expect(asText(42, "fb")).toBe("fb");
  });
});

describe("asNumber", () => {
  it("returns finite numbers, else 0", () => {
    expect(asNumber(3.5)).toBe(3.5);
    expect(asNumber(0)).toBe(0);
    expect(asNumber(NaN)).toBe(0);
    expect(asNumber(Infinity)).toBe(0);
    expect(asNumber("5")).toBe(0);
    expect(asNumber(null)).toBe(0);
  });
});

describe("asStringArray", () => {
  it("stringifies array elements", () => {
    expect(asStringArray([1, "a", true])).toEqual(["1", "a", "true"]);
  });
  it("non-arrays → []", () => {
    expect(asStringArray("ab")).toEqual([]);
    expect(asStringArray(null)).toEqual([]);
    expect(asStringArray({ 0: "x" })).toEqual([]);
  });
});

describe("countRecord", () => {
  it("builds a frequency map", () => {
    expect(countRecord(["a", "b", "a", "a", "b"])).toEqual({ a: 3, b: 2 });
  });
  it("empty list → {}", () => {
    expect(countRecord([])).toEqual({});
  });
});
