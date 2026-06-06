// SPDX-License-Identifier: Apache-2.0
// Small pure presentational primitives shared across panels. Props-only, no
// closure or data dependencies. Extracted from App.tsx (modularization) with
// no behavior change.
import { type ReactNode } from "react";

export function StatusBadge({ status }: { status: string }) {
  return <span className={`status-badge ${status}`}>{status.replace(/_/g, " ")}</span>;
}

export function StatusPill({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "success" | "warning" | "danger" | "neutral";
}) {
  return <span className={`status-pill ${tone}`}>{children}</span>;
}

export function InfoRows({ rows }: { rows: Array<[string, string | number]> }) {
  return (
    <div className="info-rows">
      {rows.map(([label, value]) => (
        <div key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
        </div>
      ))}
    </div>
  );
}

export function CompactList({ rows }: { rows: Array<[string, string]> }) {
  return (
    <div className="compact-list">
      {rows.length ? rows.map(([label, value], index) => (
        <div key={`${label}-${value}-${index}`}>
          <span>{label}</span>
          <strong>{value}</strong>
        </div>
      )) : <p className="muted">No rows for this view.</p>}
    </div>
  );
}

export function DoctorStat({ tone, value, label }: { tone: string; value: number; label: string }) {
  return (
    <div className={`doctor-stat tone-${tone}`}>
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  );
}

export function KpiGrid({ rows }: { rows: Array<[string, string | number]> }) {
  return (
    <div className="kpi-grid">
      {rows.map(([label, value]) => (
        <div key={label}>
          <strong>{value}</strong>
          <span>{label}</span>
        </div>
      ))}
    </div>
  );
}
