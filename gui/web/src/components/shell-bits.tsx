// SPDX-License-Identifier: Apache-2.0
// Small shared shell presentational helpers: workflow step, metric tile, panel
// header, tab intro banner and a code-tabs viewer. Extracted from App.tsx
// (modularization) with no behavior change.
import { useState, type ReactNode } from "react";
import { tr } from "../i18n";
import { CodeBlock } from "./code-block";

export function Step({
  number,
  title,
  detail,
  state,
  active = false,
  onClick
}: {
  number: string;
  title: string;
  detail: string;
  state: "done" | "active" | "warning" | "idle";
  active?: boolean;
  onClick?: () => void;
}) {
  const content = (
    <>
      <span>{number}</span>
      <div>
        <strong>{title}</strong>
        <small>{detail}</small>
      </div>
    </>
  );
  if (!onClick) {
    return <div className={`step ${state}`}>{content}</div>;
  }
  return (
    <button type="button" className={`step step-button ${state} ${active ? "current" : ""}`} onClick={onClick} aria-current={active}>
      {content}
    </button>
  );
}

export function Metric({
  icon,
  label,
  value
}: {
  icon: ReactNode;
  label: string;
  value: string | number;
}) {
  return (
    <div className="metric">
      {icon}
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
    </div>
  );
}

export function PanelHeader({
  label,
  title,
  action,
  icon
}: {
  label: string;
  title: string;
  action?: string;
  icon: ReactNode;
}) {
  return (
    <div className="panel-header">
      <div>
        {icon}
        <span>{label}</span>
        <h2>{title}</h2>
      </div>
      {action && <small>{action}</small>}
    </div>
  );
}

// Short explanatory banner at the top of a tab — what it does + when to use it.
export function TabIntro({ icon, title, text }: { icon: ReactNode; title: string; text: string }) {
  return (
    <div className="tab-intro">
      <span className="tab-intro-icon">{icon}</span>
      <div className="tab-intro-body">
        <strong>{title}</strong>
        <span>{text}</span>
      </div>
    </div>
  );
}

// Enterprise touch over the inline original: the tab strip is a WCAG tablist
// (role=tablist/tab + aria-selected) instead of plain buttons.
export function CodeTabs({ tabs }: { tabs: Array<{ id: string; label: string; lines: string[] }> }) {
  const [active, setActive] = useState(tabs[0]?.id ?? "");
  const current = tabs.find((tab) => tab.id === active) ?? tabs[0];
  return (
    <div className="code-tabs">
      <div className="code-tabs-bar" role="tablist" aria-label={tr("Code view")}>
        {tabs.map((tab) => (
          <button
            key={tab.id}
            type="button"
            role="tab"
            aria-selected={tab.id === current?.id}
            className={tab.id === current?.id ? "active" : ""}
            onClick={() => setActive(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      {current && <CodeBlock lines={current.lines} />}
    </div>
  );
}
