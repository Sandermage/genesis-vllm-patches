// SPDX-License-Identifier: Apache-2.0
// Small shared presentational helpers: a labelled plan chip, a key/value row,
// and the launch-artifact tabbed preview. Extracted from App.tsx (modularization)
// with no behavior change.
import { type LaunchPlanArtifact } from "../api";
import { tr } from "../i18n";
import { CodeBlock } from "./code-block";

/** Which launch-plan artifact tab is shown in ArtifactPreview. */
export type ArtifactTab = "compose" | "systemd" | "commands" | "env";

export function PlanChip({ label, value }: { label: string; value: string }) {
  return (
    <div className="plan-chip">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export function KeyValue({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="key-value">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export function ArtifactPreview({
  artifacts,
  activeTab,
  setActiveTab
}: {
  artifacts: LaunchPlanArtifact[];
  activeTab: ArtifactTab;
  setActiveTab: (tab: ArtifactTab) => void;
}) {
  const artifactByKind = new Map(
    artifacts.map((artifact) => [artifact.kind, artifact])
  );
  const activeArtifact = artifactByKind.get(activeTab);
  const tabs: Array<{ id: ArtifactTab; label: string }> = [
    { id: "compose", label: tr("Compose") },
    { id: "systemd", label: tr("systemd Unit") },
    { id: "commands", label: tr("CLI Commands") },
    { id: "env", label: tr("Environment Diff") }
  ];
  const fallback = [
    "# Waiting for launch plan Product API",
    "# Backend endpoint: /api/v1/launch/plan",
    "# Generated artifacts are intentionally not composed in React."
  ].join("\n");
  const title = activeArtifact?.title ?? tr("Product API Artifact");
  const content = activeArtifact?.content ?? fallback;

  return (
    <section className="artifact-preview">
      <div className="artifact-tabs" role="tablist" aria-label={tr("Launch artifact")}>
        {tabs.map((tab) => (
          <button
            role="tab"
            aria-selected={activeTab === tab.id}
            className={activeTab === tab.id ? "active" : ""}
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <strong className="artifact-title">{title}</strong>
      <CodeBlock lines={content.split("\n")} title={title} />
    </section>
  );
}
