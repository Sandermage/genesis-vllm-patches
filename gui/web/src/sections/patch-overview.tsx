// SPDX-License-Identifier: Apache-2.0
// Patch overview panels (the patches-tab header strip): KPI summary, lifecycle /
// production-default distributions, registry insight with a plain-language
// legend, and the supported-models chips. Extracted from App.tsx
// (modularization) with no behavior change.
import { type PatchListResult, type PatchRow } from "../api";
import { countRecord } from "../lib/coerce";
import { SegmentBar, BarList, segmentsFromCounts } from "../components/charts";
import { KpiGrid, CompactList } from "../components/primitives";

export function PatchSummaryPanel({
  summary,
  total,
  selectedCount
}: {
  summary: PatchListResult["summary"] | null;
  total: number;
  selectedCount: number;
}) {
  const lifecycleRows = Object.entries(summary?.lifecycle_counts ?? {});
  const productionRows = Object.entries(summary?.production_default_counts ?? {});
  return (
    <div className="patch-summary-grid">
      <KpiGrid
        rows={[
          ["Registry", total],
          ["Selected Plan", selectedCount || "-"],
          ["Stable", summary?.lifecycle_counts.stable ?? 0],
          ["Default Applied", summary?.production_default_counts.applied ?? 0]
        ]}
      />
      <CompactList rows={lifecycleRows.map(([key, value]) => [`lifecycle:${key}`, String(value)])} />
      <CompactList rows={productionRows.map(([key, value]) => [`default:${key}`, String(value)])} />
    </div>
  );
}

export function PatchLifecycleGraph({
  summary
}: {
  summary: PatchListResult["summary"] | null;
}) {
  const lifecycle = summary?.lifecycle_counts ?? {};
  const production = summary?.production_default_counts ?? {};
  const lifecycleTotal = Object.values(lifecycle).reduce((a, b) => a + b, 0);
  const productionTotal = Object.values(production).reduce((a, b) => a + b, 0);
  const lifecycleColors: Record<string, string> = {
    stable: "var(--ok)", experimental: "var(--warn)", research: "var(--info)",
    retired: "var(--danger)", qa: "var(--accent)"
  };
  const defaultColors: Record<string, string> = {
    applied: "var(--ok)", marker: "var(--warn)", "opt-in": "var(--info)", blocked: "var(--danger)"
  };
  return (
    <div className="patch-graph-grid">
      <section>
        <strong>Lifecycle distribution</strong>
        <SegmentBar
          segments={segmentsFromCounts(lifecycle, lifecycleColors)}
          total={lifecycleTotal}
          totalLabel="patches"
        />
      </section>
      <section>
        <strong>Production default behavior</strong>
        <SegmentBar
          segments={segmentsFromCounts(production, defaultColors)}
          total={productionTotal}
          totalLabel="patches"
        />
      </section>
    </div>
  );
}

const IMPL_MEANING: Record<string, string> = {
  full: "complete overlay — observable ON/OFF difference",
  partial: "some anchors wired; not yet fully effective",
  marker_only: "registry marker, no runtime code",
  placeholder: "reserved id, implementation pending",
  experimental: "wired but unproven; needs evidence",
  retired: "superseded/removed, kept for audit"
};

function patchBar(counts: Record<string, number>, limit = 99): Array<[string, number, string]> {
  const max = Math.max(1, ...Object.values(counts));
  return Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([k, v]) => [k.replace(/_/g, " "), Math.round((v / max) * 100), String(v)] as [string, number, string]);
}

/** Tab-1 insight: status + family distributions + a plain-language legend. */
export function PatchRegistryInsight({
  summary,
  patches
}: {
  summary: PatchListResult["summary"] | null;
  patches: PatchRow[];
}) {
  const implCounts = summary?.implementation_status_counts ?? {};
  const familyCounts = countRecord(patches.map((patch) => patch.family || "uncategorized"));
  return (
    <div className="patch-insight">
      <div className="patch-insight-grid">
        <div>
          <h5>Implementation status</h5>
          <BarList rows={patchBar(implCounts)} />
        </div>
        <div>
          <h5>Families <em>{Object.keys(familyCounts).length}</em></h5>
          <BarList rows={patchBar(familyCounts, 12)} />
        </div>
      </div>
      <div className="patch-legend">
        <h5>What the values mean</h5>
        <dl>
          <div><dt>Lifecycle</dt><dd><b>stable</b> safe default · <b>experimental</b> needs evidence · <b>research</b> idea-only · <b>legacy</b> older but kept · <b>retired</b> audit-only · <b>coordinator</b> orchestrates others.</dd></div>
          <div><dt>Production default</dt><dd><b>applied</b> on with real code · <b>marker</b> on but no effect · <b>opt-in</b> off by default · <b>blocked</b> not production-safe.</dd></div>
          <div><dt>Implementation</dt><dd>{Object.entries(IMPL_MEANING).map(([k, v]) => `${k}: ${v}`).join(" · ")}.</dd></div>
        </dl>
      </div>
    </div>
  );
}

/** Tab-1 supported models — the catalog the patch family targets. */
export function PatchModelSupport({ models }: { models: Array<{ id: string; title?: string }> }) {
  return (
    <div className="patch-models">
      <p className="muted">
        Patches target the catalog models below. Each patch declares its own applicability — model family,
        TurboQuant, vLLM version range — shown per-patch in the Inventory tab under <strong>Supported models</strong>.
      </p>
      <div className="chip-row">
        {models.length ? models.map((model) => (
          <span className="chip" key={model.id} title={model.title ?? model.id}>{model.id}</span>
        )) : <span className="muted">No models in the catalog.</span>}
      </div>
    </div>
  );
}
