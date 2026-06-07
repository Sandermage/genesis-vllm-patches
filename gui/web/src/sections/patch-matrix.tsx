// SPDX-License-Identifier: Apache-2.0
// Patch matrix viewer: the per-model env-flag override table with a filter,
// an enabled-count bar and load-bearing attribution. Extracted from App.tsx
// (modularization).
//
// Enterprise touch over the inline original (classes unchanged): the filter
// input gets an aria-label (a placeholder is not a reliable accessible name).
import { useState } from "react";
import { Search } from "lucide-react";
import { Skeleton } from "../Skeleton";
import { PercentBar } from "../components/charts";
import { StatusBadge } from "../components/primitives";

export function PatchMatrixViewer({
  patches,
  attribution,
  loading
}: {
  patches: Record<string, string>;
  attribution: Record<string, any>;
  loading: boolean;
}) {
  const [needle, setNeedle] = useState("");
  const entries = Object.entries(patches);
  const enabled = entries.filter(([, value]) => value === "1" || value === "true").length;
  const visible = entries.filter(([flag]) =>
    !needle.trim() || flag.toLowerCase().includes(needle.trim().toLowerCase())
  );
  const attributionRows = Object.entries(attribution);

  if (loading && entries.length === 0) {
    return <div className="skel-grid cards" role="status" aria-label="Loading patch matrix…"><Skeleton variant="card" count={6} /></div>;
  }
  if (entries.length === 0) {
    return <p className="muted">This model defines no canonical patch overrides.</p>;
  }

  return (
    <div className="patch-matrix">
      <div className="patch-matrix-toolbar">
        <PercentBar
          value={enabled}
          max={entries.length}
          label="flags enabled"
          caption={`${enabled} of ${entries.length} env flags on`}
        />
        <label className="search-box">
          <Search size={15} />
          <input value={needle} onChange={(event) => setNeedle(event.target.value)} placeholder="Filter env flag" aria-label="Filter env flags" />
        </label>
      </div>
      <div className="patch-matrix-scroll">
        <table className="module-table compact">
          <thead>
            <tr>
              <th scope="col">Env flag</th>
              <th scope="col">State</th>
            </tr>
          </thead>
          <tbody>
            {visible.map(([flag, value]) => {
              const on = value === "1" || value === "true";
              return (
                <tr key={flag}>
                  <td><code>{flag.replace(/^GENESIS_ENABLE_/, "")}</code></td>
                  <td><StatusBadge status={on ? "applied" : "blocked"} /></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {attributionRows.length > 0 && (
        <div className="patch-attribution">
          <strong>Load-bearing attribution</strong>
          {attributionRows.map(([patchId, meta]) => (
            <p key={patchId}>
              <em>{patchId}</em>
              <span>{String(meta?.role ?? "documented")}</span>
            </p>
          ))}
        </div>
      )}
    </div>
  );
}
