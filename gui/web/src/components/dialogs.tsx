// SPDX-License-Identifier: Apache-2.0
// Shared modal dialogs: a confirm/cancel prompt and a one-button info preview.
// Both trap focus and close on Escape/backdrop. Extracted from App.tsx
// (modularization) with no behavior change.
import { useRef, type ReactNode } from "react";
import { AlertTriangle, Command, X } from "lucide-react";
import { tr } from "../i18n";
import { useDialogFocus, useEscapeKey, closeOnBackdrop } from "../dialog";

export function ConfirmDialog({ title, message, confirmLabel, danger, onConfirm, onCancel }: {
  title: string;
  message: ReactNode;
  confirmLabel?: string;
  danger?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const dialogRef = useRef<HTMLElement>(null);
  useDialogFocus(dialogRef);
  useEscapeKey(onCancel);
  return (
    <div className="dialog-backdrop" role="presentation" onClick={closeOnBackdrop(onCancel)}>
      <section ref={dialogRef} className="info-dialog confirm-dialog" role="dialog" aria-modal="true" aria-label={title}>
        <div className="module-card-title">
          <AlertTriangle size={18} />
          <h2>{title}</h2>
        </div>
        <p>{message}</p>
        <div className="confirm-actions">
          <button className="ghost-button" onClick={onCancel} autoFocus>{tr("Cancel")}</button>
          <button className={`primary-action${danger ? " danger" : ""}`} onClick={onConfirm}>{confirmLabel ? tr(confirmLabel) : tr("Confirm")}</button>
        </div>
      </section>
    </div>
  );
}

export function InfoDialog({ message, onClose }: { message: string; onClose: () => void }) {
  const dialogRef = useRef<HTMLElement>(null);
  useDialogFocus(dialogRef);
  useEscapeKey(onClose);
  return (
    <div className="dialog-backdrop" role="presentation" onClick={closeOnBackdrop(onClose)}>
      <section ref={dialogRef} className="info-dialog" role="dialog" aria-modal="true" aria-label={tr("GUI Action Preview")}>
        <div className="module-card-title">
          <Command size={18} />
          <h2>{tr("GUI Action Preview")}</h2>
        </div>
        <p>{message}</p>
        <button className="primary-action" onClick={onClose}>{tr("Close")}</button>
      </section>
    </div>
  );
}

// Keyboard shortcuts reference modal.
export function ShortcutsModal({ onClose }: { onClose: () => void }) {
  const dialogRef = useRef<HTMLElement>(null);
  useDialogFocus(dialogRef);
  const groups: Array<{ title: string; rows: Array<[string[], string]> }> = [
    { title: tr("Global"), rows: [
      [["⌘", "K"], tr("Open command palette")],
      [["?"], tr("Toggle this shortcuts help")],
      [["Esc"], tr("Close palette / dialog")],
    ] },
    { title: tr("Command palette"), rows: [
      [["↑", "↓"], tr("Move between results")],
      [["↵"], tr("Run highlighted result")],
    ] },
    { title: tr("Go to (press g, then…)"), rows: [
      [["g", "o"], tr("Overview")],
      [["g", "s"], tr("Setup")],
      [["g", "f"], tr("Fleet")],
      [["g", "h"], tr("Hosts")],
      [["g", "m"], tr("Models")],
      [["g", "c"], tr("Configs")],
      [["g", "p"], tr("Presets")],
      [["g", "n"], tr("Containers")],
      [["g", "d"], tr("Doctor")],
      [["g", "l"], tr("Launch Plan")],
      [["g", "b"], tr("Benchmarks")],
    ] },
  ];
  return (
    <div className="dialog-backdrop" role="presentation" onClick={closeOnBackdrop(onClose)}>
      <section ref={dialogRef} className="shortcuts-dialog" role="dialog" aria-modal="true" aria-label={tr("Keyboard shortcuts")}>
        <div className="shortcuts-head">
          <Command size={16} />
          <strong>{tr("Keyboard shortcuts")}</strong>
          <button className="icon-only" onClick={onClose} aria-label={tr("Close")}><X size={15} /></button>
        </div>
        <div className="shortcuts-grid">
          {groups.map((group) => (
            <div className="shortcuts-group" key={group.title}>
              <h4>{group.title}</h4>
              {group.rows.map(([keys, label]) => (
                <div className="shortcuts-row" key={label}>
                  <span className="shortcuts-keys">
                    {keys.map((key, i) => <kbd key={i}>{key}</kbd>)}
                  </span>
                  <span className="shortcuts-label">{label}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
