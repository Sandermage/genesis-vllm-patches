// SPDX-License-Identifier: Apache-2.0
// Launch-readiness gate logic for the overview workspace. Pure functions that
// project a ProductOverview (+ selected preset / explain payload) into the gate
// rows rendered by the readiness panel, plus small status/count helpers. Kept
// free of React so they stay unit-testable and reusable across sections.
import { tr } from "../i18n";
import type { Gate } from "../nav";
import type { GateStatus } from "../components/primitives";
import type {
  ProductOverview,
  PresetRecord,
  PresetExplainResult,
  ProductCapability
} from "../api";

export function buildReadinessGates({
  overview,
  runtimeTarget,
  selectedPresetRecord,
  explain
}: {
  overview: ProductOverview | null;
  runtimeTarget: string;
  selectedPresetRecord: PresetRecord | null;
  explain: PresetExplainResult | null;
}): Gate[] {
  const capabilities = overview?.capabilities;
  const featureRows = capabilities?.features ?? [];
  const target = capabilities?.runtime_targets.find((item) => item.id === runtimeTarget);
  const catalogErrors = overview?.catalog.preset_load_error_count ?? 0;
  const hasCard = Boolean(selectedPresetRecord?.has_card || explain?.card);
  const serviceLifecycle = featureRows.find((feature) => feature.id === "service_lifecycle");
  const benchmarkRuns = featureRows.find((feature) => feature.id === "benchmark_runs");

  return [
    {
      id: "catalog",
      label: tr("Catalog Snapshot"),
      detail: catalogErrors === 0 ? tr("V2 registry loaded without errors") : `${catalogErrors} ${tr("load errors")}`,
      status: catalogErrors === 0 ? "pass" : "blocked",
      action: tr("Re-run")
    },
    {
      id: "preset-card",
      label: tr("Preset Card"),
      detail: hasCard ? tr("Operator card and explain payload available") : tr("Preset has no product card yet"),
      status: hasCard ? "pass" : "warning",
      action: tr("Open")
    },
    {
      id: "runtime",
      label: tr("Runtime Target"),
      detail: target?.detail ?? tr("Runtime target not selected"),
      status: targetStatus(target),
      action: tr("Check")
    },
    {
      id: "engine",
      label: tr("Engine Installed"),
      detail: capabilities?.platform.engine_installed ? tr("vLLM package detected") : tr("Engine package not installed in this shell"),
      status: capabilities?.platform.engine_installed ? "pass" : "warning",
      action: tr("Doctor")
    },
    {
      id: "service-api",
      label: tr("Service Lifecycle API"),
      detail: serviceLifecycle?.detail ?? tr("Plan/apply lifecycle API available (execution gated by --enable-apply)"),
      status: serviceLifecycle?.status === "available" ? "pass" : "warning",
      action: tr("Plan")
    },
    {
      id: "evidence",
      label: tr("Evidence Orchestration"),
      detail: benchmarkRuns?.detail ?? tr("Evidence/report jobs available; full GPU runs are a rig action"),
      status: benchmarkRuns?.status === "available" ? "pass" : "warning",
      action: tr("Report")
    },
    {
      id: "release-proof",
      label: tr("Release Proof"),
      detail: tr("Generate a proof/report bundle (Reports) before a production launch — recommended"),
      status: "warning",
      action: tr("Generate proof")
    }
  ];
}

export function targetStatus(target: ProductCapability | undefined): GateStatus {
  if (!target) return "blocked";
  if (target.status === "available") return "pass";
  if (target.status === "render_only" || target.status === "partial") return "warning";
  if (target.status === "deferred") return "planned";
  return "blocked";
}

export function countGates(gates: Gate[]) {
  return gates.reduce(
    (acc, gate) => {
      acc[gate.status] += 1;
      return acc;
    },
    { pass: 0, warning: 0, blocked: 0, planned: 0 } as Record<GateStatus, number>
  );
}
