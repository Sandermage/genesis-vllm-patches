// SPDX-License-Identifier: Apache-2.0
// Project catalog snapshot — count tiles, annotation/capability rows and the
// lifecycle/workload/family chip distributions. Extracted from App.tsx
// (modularization).
//
// Enterprise touch over the inline original (classes unchanged): each chip
// distribution is a role="group" with an aria-label so assistive tech announces
// what the chips count.
import { type ProductOverview, type EnvironmentReport } from "../api";
import { SkeletonMetrics } from "../Skeleton";
import { InfoRows } from "../components/primitives";

export function ProjectCatalogPanel({ overview, environment }: { overview: ProductOverview | null; environment: EnvironmentReport | null }) {
  if (!overview) return <SkeletonMetrics count={4} />;
  const catalog = overview.catalog;
  const features = overview.capabilities.features ?? [];
  const capsReady = features.filter((feature) => feature.status === "available").length;
  const tiles: Array<[string, number]> = [
    ["Presets", catalog.presets_count],
    ["Models", catalog.models_count],
    ["Profiles", catalog.profiles_count],
    ["Hardware", catalog.hardware_count]
  ];
  const annotated = catalog.presets_count ? Math.round((catalog.preset_cards_count / catalog.presets_count) * 100) : 0;
  const lifecycle = Object.entries(catalog.status_counts || {});
  const workloads = Object.entries(catalog.workload_counts || {}).slice(0, 6);
  const families = Object.entries(catalog.family_counts || {}).slice(0, 8);
  return (
    <div className="project-catalog">
      <div className="catalog-tiles">
        {tiles.map(([label, value]) => (
          <div className="catalog-tile" key={label}>
            <strong>{value}</strong>
            <span>{label}</span>
          </div>
        ))}
      </div>
      <InfoRows rows={[
        ["Engine target", `${environment?.engine_name ?? "vLLM"} ${environment?.engine_version ?? "(not installed)"}`.trim()],
        ["Annotated presets", `${catalog.preset_cards_count}/${catalog.presets_count} · ${annotated}%`],
        ["Capabilities ready", `${capsReady}/${features.length}`],
        ["Load errors", String(catalog.preset_load_error_count)]
      ]} />
      {lifecycle.length > 0 && (
        <div className="catalog-dist">
          <span className="catalog-dist-label">Lifecycle</span>
          <div className="fleet-caps" role="group" aria-label="Preset lifecycle distribution">
            {lifecycle.map(([key, value]) => <span className="cap-chip neutral" key={key}>{key} · {value}</span>)}
          </div>
        </div>
      )}
      {workloads.length > 0 && (
        <div className="catalog-dist">
          <span className="catalog-dist-label">Workloads</span>
          <div className="fleet-caps" role="group" aria-label="Workload distribution">
            {workloads.map(([key, value]) => <span className="cap-chip neutral" key={key}>{key.replace(/_/g, " ")} · {value}</span>)}
          </div>
        </div>
      )}
      {families.length > 0 && (
        <div className="catalog-dist">
          <span className="catalog-dist-label">Patch families</span>
          <div className="fleet-caps" role="group" aria-label="Patch family distribution">
            {families.map(([key, value]) => <span className="cap-chip neutral" key={key}>{key} · {value}</span>)}
          </div>
        </div>
      )}
    </div>
  );
}
