// SPDX-License-Identifier: Apache-2.0
// Reusable code-display + copy primitives. Extracted from App.tsx
// (modularization) with no behavior change.
import { useEffect, useRef, useState } from "react";
import { CheckCircle2, Copy, Maximize2, Terminal, X } from "lucide-react";
import { useDialogFocus, closeOnBackdrop } from "../dialog";

export function CopyButton({ value, label }: { value: string; label: string }) {
  const [done, setDone] = useState(false);
  async function copy() {
    try {
      await navigator.clipboard.writeText(value);
    } catch {
      // Clipboard API can be blocked; fall back to a transient confirmation.
    }
    setDone(true);
    window.setTimeout(() => setDone(false), 1200);
  }
  return (
    <button
      className={`icon-only ${done ? "done" : ""}`}
      onClick={() => void copy()}
      aria-label={`Copy ${label}`}
      title={`Copy ${label}`}
    >
      {done ? <CheckCircle2 size={14} /> : <Copy size={14} />}
    </button>
  );
}

export function CodeBlock({ lines, title }: { lines: string[]; title?: string }) {
  const [expanded, setExpanded] = useState(false);
  const dialogRef = useRef<HTMLElement>(null);
  useDialogFocus(dialogRef, expanded);
  useEffect(() => {
    if (!expanded) return;
    const onKey = (event: KeyboardEvent) => { if (event.key === "Escape") setExpanded(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expanded]);
  const body = lines.map((line, index) => <span key={index}>{line || " "}</span>);
  const joined = lines.join("\n");
  return (
    <>
      <div className="code-wrap">
        <div className="code-actions">
          <button className="icon-only" title="Expand" aria-label="Expand to fullscreen" onClick={() => setExpanded(true)}><Maximize2 size={13} /></button>
          <CopyButton value={joined} label="code block" />
        </div>
        <pre className="code-block">{body}</pre>
      </div>
      {expanded && (
        <div className="dialog-backdrop" role="presentation" onClick={closeOnBackdrop(() => setExpanded(false))}>
          <section ref={dialogRef} className="code-expand" role="dialog" aria-modal="true">
            <header className="code-expand-head">
              <Terminal size={15} />
              <strong>{title ?? "Output"}</strong>
              <span className="muted">{lines.length} lines</span>
              <CopyButton value={joined} label="code block" />
              <button className="icon-only" onClick={() => setExpanded(false)} aria-label="Close"><X size={16} /></button>
            </header>
            <pre className="code-block code-expand-pre">{body}</pre>
          </section>
        </div>
      )}
    </>
  );
}
