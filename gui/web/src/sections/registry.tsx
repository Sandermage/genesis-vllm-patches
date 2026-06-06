// SPDX-License-Identifier: Apache-2.0
// Registry-facing section panels: multi-patch bundles + upstream-PR diff.
// Extracted from App.tsx (modularization) with no behavior change.
import { type BundleSpec, type DiffUpstreamReport } from "../api";
import { StatusBadge, KpiGrid } from "../components/primitives";
import { SkeletonLines } from "../Skeleton";

export function BundlesPanel({ bundles }: { bundles: BundleSpec[] }) {
  if (!bundles.length) {
    return <p className="muted">No multi-patch bundles reported by the registry.</p>;
  }
  return (
    <table className="module-table">
      <thead>
        <tr>
          <th>Bundle</th>
          <th>Tier</th>
          <th>Umbrella flag</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
        {bundles.map((bundle) => (
          <tr key={bundle.name}>
            <td><strong>{bundle.name}</strong></td>
            <td><StatusBadge status={bundle.tier} /></td>
            <td><code>{bundle.umbrella_flag}</code></td>
            <td>{bundle.description}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function UpstreamDiffPanel({ report }: { report: DiffUpstreamReport | null }) {
  if (!report) {
    return <SkeletonLines count={5} />;
  }
  const active = report.has_upstream_pr;
  return (
    <div className="runtime-envelope">
      <KpiGrid
        rows={[
          ["Active upstream PRs", active.length],
          ["Merged upstream", report.merged_upstream.length]
        ]}
      />
      {active.length === 0 ? (
        <p className="muted">No patches currently track an open upstream PR.</p>
      ) : (
        <div className="patch-table-scroll">
          <table className="module-table patch-table">
            <thead>
              <tr>
                <th>Patch</th>
                <th>Upstream PR</th>
                <th>Lifecycle</th>
              </tr>
            </thead>
            <tbody>
              {active.map((row, index) => (
                <tr key={`${String(row.patch_id)}-${index}`}>
                  <td>
                    <strong>{String(row.patch_id)}</strong>
                    <small>{String(row.title ?? "")}</small>
                  </td>
                  <td>{row.upstream_pr ? `#${row.upstream_pr}` : "-"}</td>
                  <td><StatusBadge status={String(row.lifecycle ?? "unknown")} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
