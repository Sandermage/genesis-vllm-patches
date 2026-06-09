// SPDX-License-Identifier: Apache-2.0
// Capability table — a labelled grid of Product API capabilities (status +
// required tools + detail). Shared across several tabs. Extracted from App.tsx
// (modularization).
//
// Enterprise touch over the inline original (classes unchanged): headers carry
// scope=col for screen-reader table navigation.
import { type ProductCapability } from "../api";
import { tr } from "../i18n";
import { StatusBadge } from "./primitives";

export function CapabilityTable({ rows }: { rows: ProductCapability[] }) {
  return (
    <table className="module-table">
      <thead>
        <tr>
          <th scope="col">{tr("Capability")}</th>
          <th scope="col">{tr("Status")}</th>
          <th scope="col">{tr("Required")}</th>
          <th scope="col">{tr("Detail")}</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.id}>
            <td>
              <strong>{row.title}</strong>
              <small>{row.id}</small>
            </td>
            <td><StatusBadge status={row.status} /></td>
            <td>{row.required_tools.length ? row.required_tools.join(", ") : tr("built-in")}</td>
            <td>{row.detail}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
