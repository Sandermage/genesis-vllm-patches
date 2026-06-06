// SPDX-License-Identifier: Apache-2.0
// Patch-doctor section panels: apply-module coverage + validation drill-down,
// and the admin API surface matrix. Extracted from App.tsx (modularization)
// with no behavior change.
import { useState } from "react";
import { type PatchDoctorReport, type ProductCapability } from "../api";
import { StatusBadge, InfoRows } from "../components/primitives";
import { PercentBar } from "../components/charts";

export function DoctorCoveragePanel({ report }: { report: PatchDoctorReport | null }) {
  const coverage = report?.coverage;
  const total = coverage?.total ?? 0;
  const mapped = coverage?.mapped ?? 0;
  const issues = report?.issues ?? [];
  const [sev, setSev] = useState<"" | "ERROR" | "WARNING" | "INFO">("");
  const order: Record<string, number> = { ERROR: 0, WARNING: 1, INFO: 2 };
  const counts = issues.reduce<Record<string, number>>((a, i) => { a[i.severity] = (a[i.severity] ?? 0) + 1; return a; }, {});
  const shown = [...issues]
    .filter((i) => !sev || i.severity === sev)
    .sort((a, b) => (order[a.severity] ?? 9) - (order[b.severity] ?? 9));
  const unmapped = coverage?.unmapped ?? [];
  return (
    <div className="doctor-coverage">
      <PercentBar value={mapped} max={total} label="apply modules mapped" caption={`${mapped} of ${total} patches`} tone="accent" />
      <InfoRows
        rows={[
          ["Registry Size", report?.registry_size ?? "-"],
          ["Validation Issues", issues.length],
          ["Mapped", coverage?.mapped ?? "-"],
          ["Intentionally Unmapped", coverage?.intentionally_unmapped.length ?? "-"],
          ["Unmapped", unmapped.length]
        ]}
      />

      {issues.length > 0 && (
        <div className="audit-drill">
          <div className="audit-drill-bar">
            <strong>Validation issues</strong>
            {(["ERROR", "WARNING", "INFO"] as const).map((s) => (counts[s] ? (
              <button key={s} className={`audit-sevchip ${s.toLowerCase()} ${sev === s ? "active" : ""}`}
                onClick={() => setSev(sev === s ? "" : s)}>{s.toLowerCase()} {counts[s]}</button>
            ) : null))}
            {sev && <button className="audit-clear" onClick={() => setSev("")}>clear</button>}
          </div>
          <div className="audit-list">
            {shown.slice(0, 200).map((i, idx) => (
              <div key={`${i.patch_id}-${idx}`} className={`audit-issue ${i.severity.toLowerCase()}`}>
                <span className={`audit-sev ${i.severity.toLowerCase()}`}>{i.severity}</span>
                <span className="audit-pid">{i.patch_id}</span>
                <span className="audit-msg">{i.message}</span>
              </div>
            ))}
            {shown.length > 200 && <div className="audit-more">+{shown.length - 200} more…</div>}
          </div>
        </div>
      )}

      {unmapped.length > 0 && (
        <div className="audit-drill">
          <div className="audit-drill-bar"><strong>Unmapped patches</strong> <span className="muted">({unmapped.length} — no apply_module)</span></div>
          <div className="audit-chips">{unmapped.map((p) => <span key={p} className="audit-chip">{p}</span>)}</div>
        </div>
      )}
    </div>
  );
}

export function AdminSurfaceMatrix({
  featureRows,
  patchDoctor
}: {
  featureRows: ProductCapability[];
  patchDoctor: PatchDoctorReport | null;
}) {
  const rows: Array<[string, string, string, string]> = [
    ["Catalog", "GET", "Ready", "models, hardware, profiles, presets"],
    ["Preset Workbench", "GET", "Ready", "list, explain, recommend"],
    ["Patch Inventory", "GET", "Ready", `${patchDoctor?.registry_size ?? "-"} registry entries`],
    ["Patch Doctor", "GET", "Ready", `${patchDoctor?.issues.length ?? "-"} validation issues`],
    ["Service Lifecycle", "POST", "Ready", "plan/apply start-stop (gated by --enable-apply + confirm)"],
    ["Launch Apply", "POST", "Ready", "launch a preset (gated + confirm)"],
    ["Jobs and Events", "GET/SSE", "Ready", "dry-run/executed jobs + /events stream"],
    ["Reports", "POST", "Ready", "redacted bundle generation to $SNDR_HOME"],
    ["Bench / Evidence", "POST", "Ready", "queue dry-run jobs (full runs on rig)"],
    ["Remote Host Profiles", "GET/POST/DELETE", "Ready", "operator-local profiles + SSH tunnel command"]
  ];
  const featureStatus = new Map(featureRows.map((feature) => [feature.id, feature.status]));
  return (
    <table className="module-table">
      <thead>
        <tr>
          <th>Surface</th>
          <th>Transport</th>
          <th>Status</th>
          <th>Contract</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([surface, transport, status, contract]) => (
          <tr key={surface}>
            <td><strong>{surface}</strong></td>
            <td>{transport}</td>
            <td><StatusBadge status={status === "Ready" ? "available" : featureStatus.get("service_lifecycle") ?? "deferred"} /></td>
            <td>{contract}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
