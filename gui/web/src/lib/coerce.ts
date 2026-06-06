// SPDX-License-Identifier: Apache-2.0
// Pure value-coercion helpers shared across panels. Defensive readers for the
// loosely-typed (Record<string, unknown>) payloads the API surfaces. Extracted
// from App.tsx (modularization) with no behavior change.

/** Narrow to a plain object, else `{}`. Arrays are NOT objects here. */
export function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

/** A non-empty (trimmed) string, else `fallback`. */
export function asText(value: unknown, fallback: string): string {
  return typeof value === "string" && value.trim() ? value : fallback;
}

/** A finite number, else `0`. */
export function asNumber(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

/** An array stringified element-wise, else `[]`. */
export function asStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.map((item) => String(item)) : [];
}

/** Frequency map of a string list: value → count. */
export function countRecord(values: string[]): Record<string, number> {
  return values.reduce<Record<string, number>>((acc, value) => {
    acc[value] = (acc[value] ?? 0) + 1;
    return acc;
  }, {});
}
