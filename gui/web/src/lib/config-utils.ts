// SPDX-License-Identifier: Apache-2.0
// Generic, dependency-free data utilities shared by the config editors: deep
// path get/set on plain objects, and a small YAML serializer. Extracted from
// App.tsx (modularization) with no behavior change. Kept type-light (any) to
// match the dynamic config payloads these operate on.
 

/** Read a dotted path out of a nested object; undefined if any segment is missing. */
export function getIn(obj: any, path: string): any {
  return path.split(".").reduce((current, key) => (current == null ? undefined : current[key]), obj);
}

/** Immutably write `value` at a dotted path, cloning each touched node on the way. */
export function setIn(obj: any, path: string, value: any): any {
  const keys = path.split(".");
  const clone = Array.isArray(obj) ? [...obj] : { ...obj };
  let cursor = clone;
  for (let i = 0; i < keys.length - 1; i += 1) {
    const key = keys[i];
    const next = cursor[key];
    cursor[key] = next && typeof next === "object" ? (Array.isArray(next) ? [...next] : { ...next }) : {};
    cursor = cursor[key];
  }
  cursor[keys[keys.length - 1]] = value;
  return clone;
}

/** Render a single YAML scalar, quoting when the value needs it. */
export function yamlScalar(value: any): string {
  if (value === null || value === undefined) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") return String(value);
  const text = String(value);
  return /[:#{}[\],&*?|<>=!%@`]/.test(text) || text.trim() !== text ? JSON.stringify(text) : text;
}

/** Serialize a plain object/array tree into YAML lines (2-space indent). */
export function objToYaml(obj: any, indent = 0): string[] {
  const pad = "  ".repeat(indent);
  const lines: string[] = [];
  const entries: Array<[string, any]> = Array.isArray(obj)
    ? obj.map((value, index) => [String(index), value])
    : Object.entries(obj ?? {});
  for (const [key, value] of entries) {
    const label = `${pad}${key}:`;
    if (value === null || value === undefined) {
      lines.push(`${label} null`);
    } else if (Array.isArray(value)) {
      if (value.length === 0) { lines.push(`${label} []`); continue; }
      lines.push(label);
      value.forEach((item) => {
        if (item && typeof item === "object") {
          const sub = objToYaml(item, indent + 2);
          lines.push(`${pad}  -`);
          lines.push(...sub);
        } else {
          lines.push(`${pad}  - ${yamlScalar(item)}`);
        }
      });
    } else if (typeof value === "object") {
      if (Object.keys(value).length === 0) { lines.push(`${label} {}`); continue; }
      lines.push(label);
      lines.push(...objToYaml(value, indent + 1));
    } else {
      lines.push(`${label} ${yamlScalar(value)}`);
    }
  }
  return lines;
}
