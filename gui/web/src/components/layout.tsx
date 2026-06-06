// SPDX-License-Identifier: Apache-2.0
// Core layout primitives shared across the GUI: a responsive module grid and a
// titled module card. Extracted from App.tsx (modularization) with no behavior
// change.
import { type ReactNode } from "react";

export function ModuleGrid({ children, className }: { children: ReactNode; className?: string }) {
  return <section className={`module-grid${className ? ` ${className}` : ""}`}>{children}</section>;
}

export function ModuleCard({
  title,
  icon,
  desc,
  children,
  wide = false
}: {
  title: string;
  icon: ReactNode;
  desc?: string;
  children: ReactNode;
  wide?: boolean;
}) {
  return (
    <section className={`module-card ${wide ? "wide" : ""}`}>
      <div className="module-card-title">
        <span className="module-card-icon">{icon}</span>
        <div className="module-card-heading">
          <h2>{title}</h2>
          {desc && <p>{desc}</p>}
        </div>
      </div>
      {children}
    </section>
  );
}
