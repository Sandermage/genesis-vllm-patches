// SPDX-License-Identifier: Apache-2.0
// The Patches section: a tabbed view over the patch registry (summary/lifecycle/
// insight/model support), the inventory + bundles, the env-flag matrix, and the
// upstream diff + policy preview. Derives patch rows/summary from the patches
// prop. Extracted from section-workspace.tsx.
import { Suspense } from "react";
import {
  AlertTriangle, Anchor, BarChart3, Code2, Cpu, GitBranch, Layers3, ListChecks, PackageCheck,
  SlidersHorizontal, Table2
} from "lucide-react";
import { tr } from "../i18n";
import { asNumber } from "../lib/coerce";
import type { PatchListResult, V2ConfigCatalog, BundleSpec, DiffUpstreamReport } from "../api";
import { TabbedSection } from "../components/tabbed-section";
import { ModuleCard, ModuleGrid } from "../components/layout";
import { CodeBlock } from "../components/code-block";
import { PatchInventoryControl, FlagsPanel } from "../lazy-panels";
import { SkeletonCards } from "../Skeleton";
import { PatchLifecycleGraph, PatchModelSupport, PatchRegistryInsight, PatchSummaryPanel } from "./patch-overview";
import { BundlesPanel, UpstreamDiffPanel } from "./registry";
import { AnchorManifestPanel } from "./anchor-manifest";
import { RetireImpactPanel } from "./retire-impact";

export function PatchesSection({
  patches,
  configCatalog,
  bundles,
  diffUpstream,
  selectedPreset,
  patchPolicy,
  composed
}: {
  patches: PatchListResult | null;
  configCatalog: V2ConfigCatalog | null;
  bundles: BundleSpec[];
  diffUpstream: DiffUpstreamReport | null;
  selectedPreset: string;
  patchPolicy: string;
  composed: Record<string, unknown>;
}) {
  const patchRows = patches?.patches ?? [];
  const patchSummary = patches?.summary ?? null;
  return (
            <TabbedSection
          id="patches"
          tabs={[
            {
              id: "registry",
              label: tr("Registry"),
              icon: <PackageCheck size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={tr("Patch Registry Summary")} icon={<PackageCheck size={18} />} desc={`${patches?.total ?? patchRows.length} ${tr("runtime overlays across")} ${new Set(patchRows.map((p) => p.family)).size} ${tr("families")}.`} wide>
                    <PatchSummaryPanel summary={patchSummary} total={patches?.total ?? patchRows.length} selectedCount={asNumber(composed.enabled_patches_count)} />
                  </ModuleCard>
                  <ModuleCard title={tr("Lifecycle & Default Behavior")} icon={<BarChart3 size={18} />} wide>
                    <PatchLifecycleGraph summary={patchSummary} />
                  </ModuleCard>
                  <ModuleCard title={tr("Status, Families & Legend")} icon={<ListChecks size={18} />} desc={tr("Implementation maturity, subsystem coverage, and what each registry value means.")} wide>
                    <PatchRegistryInsight summary={patchSummary} patches={patchRows} />
                  </ModuleCard>
                  <ModuleCard title={tr("Supported Models")} icon={<Cpu size={18} />} desc={tr("Catalog models the patch family targets — per-patch applicability is in the Inventory tab.")} wide>
                    <PatchModelSupport models={configCatalog?.models ?? []} />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "inventory",
              label: tr("Inventory"),
              icon: <Table2 size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={tr("Patch Inventory Control")} icon={<Table2 size={18} />} wide>
                    <PatchInventoryControl patches={patchRows} />
                  </ModuleCard>
                  <ModuleCard title={tr("Patch Bundles")} icon={<Layers3 size={18} />} wide>
                    <BundlesPanel bundles={bundles} />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "flags",
              label: tr("Flags"),
              icon: <SlidersHorizontal size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={tr("Env-flag matrix")} icon={<SlidersHorizontal size={18} />} desc={tr("Every GENESIS_ENABLE_* flag with its effective default — searchable, filterable by family. Name a running engine container to overlay its live ON/OFF state and flag drift.")} wide>
                    <Suspense fallback={<SkeletonCards count={2} />}>
                      <FlagsPanel />
                    </Suspense>
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "anchors",
              label: tr("Anchors"),
              icon: <Anchor size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={tr("Anchor Source-of-Truth")} icon={<Anchor size={18} />} desc={tr("Per-pin anchor manifests that drive the live patcher (fast anchoring + md5-drift fallback). Shows each pin's anchor counts, which is active for the running engine, and live drift vs the installed source.")} wide>
                    <AnchorManifestPanel />
                  </ModuleCard>
                  <ModuleCard title={tr("Retire-Impact")} icon={<AlertTriangle size={18} />} desc={tr("Which active dependents a retired patch would break. HIGH = a perf-bearing dependent whose anchor targets the retired patch's bytes (the silent perf-regression class the pin-bump preflight gate catches); MEDIUM = a registry edge only.")} wide>
                    <RetireImpactPanel />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "upstream",
              label: tr("Upstream & policy"),
              icon: <GitBranch size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={tr("Upstream Diff")} icon={<GitBranch size={18} />} wide>
                    <UpstreamDiffPanel report={diffUpstream} />
                  </ModuleCard>
                  <ModuleCard title={tr("Policy Preview")} icon={<Code2 size={18} />} wide>
                    <CodeBlock lines={[`preset=${selectedPreset}`, `policy=${patchPolicy}`, "strict_image_digest=true", "dry_run=true"]} />
                  </ModuleCard>
                </ModuleGrid>
              )
            }
          ]}
        />
  );
}
