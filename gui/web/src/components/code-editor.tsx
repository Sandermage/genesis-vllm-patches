// SPDX-License-Identifier: Apache-2.0
// Lean, dependency-free syntax-highlighted editor for YAML/config text. The
// classic overlay pattern: a real <textarea> (the source of truth for input,
// caret and selection) sits transparently over a highlighted <pre> mirror with
// identical metrics; the mirror's scroll is JS-synced to the textarea. So
// editing is always correct regardless of highlighting, and we add zero deps
// (no Monaco/CodeMirror) — in keeping with the lean bundle. Plus Tab→2-spaces
// and a light "tabs in YAML" validation hint.
import { useMemo, useRef, type ReactNode } from "react";
import { tr } from "../i18n";

type Token = { text: string; cls: string };

// Lossless single-line YAML tokenizer: the emitted segments always concatenate
// back to the exact input line, so the mirror never drifts from the textarea.
function tokenizeLine(line: string): Token[] {
  const tokens: Token[] = [];
  const indent = (line.match(/^\s*/)?.[0]) ?? "";
  if (indent) tokens.push({ text: indent, cls: "" });
  let rest = line.slice(indent.length);
  if (!rest) return tokens;
  if (rest.startsWith("#")) { tokens.push({ text: rest, cls: "tok-comment" }); return tokens; }

  if (rest.startsWith("- ")) { tokens.push({ text: "- ", cls: "tok-punct" }); rest = rest.slice(2); }
  else if (rest === "-") { tokens.push({ text: "-", cls: "tok-punct" }); return tokens; }

  const keyMatch = rest.match(/^([^\s:#][^:#]*?):(?=\s|$)/);
  if (keyMatch) {
    const key = keyMatch[1] ?? "";
    tokens.push({ text: key, cls: "tok-key" });
    tokens.push({ text: ":", cls: "tok-punct" });
    rest = rest.slice(key.length + 1);
  }

  let comment = "";
  const ci = rest.indexOf(" #");
  if (ci >= 0) { comment = rest.slice(ci); rest = rest.slice(0, ci); }
  if (rest) {
    const v = rest.trim();
    let cls = "";
    if (/^(['"]).*\1$/.test(v)) cls = "tok-string";
    else if (/^-?\d+(\.\d+)?$/.test(v)) cls = "tok-num";
    else if (/^(true|false|null|yes|no|~)$/i.test(v)) cls = "tok-bool";
    tokens.push({ text: rest, cls });
  }
  if (comment) tokens.push({ text: comment, cls: "tok-comment" });
  return tokens;
}

function highlight(value: string): ReactNode {
  const out: ReactNode[] = [];
  value.split("\n").forEach((line, i) => {
    if (i > 0) out.push("\n");
    tokenizeLine(line).forEach((t, j) =>
      out.push(t.cls ? <span key={`${i}-${j}`} className={t.cls}>{t.text}</span> : t.text)
    );
  });
  // Trailing newline so the mirror keeps the textarea's height on an empty last line.
  out.push("\n");
  return out;
}

export function CodeEditor({
  value,
  onChange,
  expanded,
  autoFocus,
  ariaLabel,
}: {
  value: string;
  onChange: (next: string) => void;
  expanded?: boolean;
  autoFocus?: boolean;
  ariaLabel?: string;
}) {
  const preRef = useRef<HTMLPreElement>(null);
  // Memoize the per-keystroke hot path: re-tokenizing the whole document and
  // re-scanning for tab lines on every render is O(document) per keypress.
  const highlighted = useMemo(() => highlight(value), [value]);
  const tabLines = useMemo(
    () => value.split("\n").filter((l) => /^\t| \t/.test(l)).length,
    [value],
  );

  const onScroll = (e: React.UIEvent<HTMLTextAreaElement>) => {
    if (preRef.current) {
      preRef.current.scrollTop = e.currentTarget.scrollTop;
      preRef.current.scrollLeft = e.currentTarget.scrollLeft;
    }
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Tab" && !e.shiftKey && !e.metaKey && !e.ctrlKey) {
      e.preventDefault();
      const ta = e.currentTarget;
      const start = ta.selectionStart;
      const end = ta.selectionEnd;
      onChange(value.slice(0, start) + "  " + value.slice(end));
      // Controlled re-render resets the caret to the end; restore it next frame,
      // and re-sync the highlight mirror (selection changes don't fire onScroll, so
      // a Tab that auto-scrolls the textarea would otherwise desync the overlay).
      requestAnimationFrame(() => {
        ta.selectionStart = ta.selectionEnd = start + 2;
        if (preRef.current) { preRef.current.scrollTop = ta.scrollTop; preRef.current.scrollLeft = ta.scrollLeft; }
      });
    }
  };

  return (
    <div className={`code-editor${expanded ? " ce-expanded" : ""}`}>
      <pre ref={preRef} className="code-editor-hl" aria-hidden="true">{highlighted}</pre>
      <textarea
        className="code-editor-ta"
        value={value}
        spellCheck={false}
        autoFocus={autoFocus}
        aria-label={ariaLabel}
        onScroll={onScroll}
        onKeyDown={onKeyDown}
        onChange={(e) => onChange(e.target.value)}
      />
      {tabLines > 0 && (
        <div className="code-editor-hint" role="status">{tabLines} {tr("line(s) use tabs — YAML needs spaces")}</div>
      )}
    </div>
  );
}
