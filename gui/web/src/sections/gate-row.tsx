// SPDX-License-Identifier: Apache-2.0
// Readiness-gate row: a collapsible disclosure showing a gate's status, detail
// and a one-click jump to the section that resolves it. Extracted from App.tsx
// (modularization).
//
// Enterprise hardening over the original inline version:
//   * proper WCAG disclosure wiring — the trigger's aria-controls points at the
//     detail region's id (useId), so assistive tech announces the relationship;
//   * the decorative status glyph is aria-hidden (status is also text);
//   * the four-way status→icon branch is a typed lookup instead of a chain.
// Markup classes are unchanged, so the visual style is identical.
import { useId, useState, type ReactNode } from "react";
import { CheckCircle2, CircleAlert, AlertCircle, Clock3, ChevronRight, Wrench } from "lucide-react";
import { type Gate, type SectionId, GATE_TARGET } from "../nav";
import { type GateStatus } from "../components/primitives";

const GATE_ICON: Record<GateStatus, ReactNode> = {
  pass: <CheckCircle2 size={16} />,
  warning: <CircleAlert size={16} />,
  blocked: <AlertCircle size={16} />,
  planned: <Clock3 size={16} />
};

export function GateRow({ gate, onNavigate }: { gate: Gate; onNavigate?: (section: SectionId) => void }) {
  const [open, setOpen] = useState(false);
  const target = GATE_TARGET[gate.id];
  const detailId = useId();
  return (
    <div className={`gate-row ${gate.status} ${open ? "open" : ""}`}>
      <button
        className="gate-main"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        aria-controls={detailId}
      >
        <span className="gate-icon" aria-hidden="true">{GATE_ICON[gate.status]}</span>
        <div>
          <strong>{gate.label}</strong>
          <small>{gate.detail}</small>
        </div>
        <span className="gate-status">{gate.status}</span>
        <ChevronRight className="gate-caret" size={16} />
      </button>
      {open && (
        <div className="gate-detail" id={detailId}>
          <p>{gate.detail}</p>
          <div className="gate-detail-actions">
            <span className="gate-action-hint"><Wrench size={13} /> {gate.action}</span>
            {target && onNavigate && (
              <button className="ghost-button" onClick={() => onNavigate(target.section)}>
                <ChevronRight size={14} /> {target.label}
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
