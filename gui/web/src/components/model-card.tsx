// SPDX-License-Identifier: Apache-2.0
// Fused model/preset card — one scannable tile per preset for the consumer
// funnel (Choose & Launch). Collapses what used to take a table row + a fit
// matrix + a card panel into a single glanceable unit:
//   · a status badge (Production / Experimental / Single-card escape-hatch /
//     Blocked-on-rig) derived from card.status + evidence_visibility AND the
//     LIVE preflight verdict against the current rig;
//   · the measured throughput (card.primary_metric, "Pending" if unproven);
//   · the hardware requirement (min-GPUs / min-VRAM / SM from card.hardware_fit);
//   · a fit pill for the current rig (reuses the preflight verdict);
//   · a one-line "why this" (workload + drafter).
import { type ReactNode } from "react";
import { CheckCircle2, CircleAlert, Cpu, Gauge, HelpCircle, LifeBuoy } from "lucide-react";
import { tr } from "../i18n";
import { type PresetRecord, type PreflightFitReport } from "../api";
import { asNumber, asText } from "../lib/coerce";

// The four consumer-facing statuses, in escalating "can I run this?" order.
export type ModelCardStatus = "production" | "experimental" | "escape-hatch" | "blocked";

type StatusMeta = { tone: "ok" | "warn" | "info" | "muted"; label: string };

const STATUS_META: Record<ModelCardStatus, StatusMeta> = {
  production: { tone: "ok", label: tr("Production") },
  experimental: { tone: "warn", label: tr("Experimental") },
  "escape-hatch": { tone: "info", label: tr("Single-card escape hatch") },
  blocked: { tone: "muted", label: tr("Blocked on your rig") }
};

/** Derive the consumer status from the card's declared status + the LIVE
 *  preflight verdict. A 2× preset on a 1-card rig that FAILs only on gpu_count
 *  is the escape-hatch case (there is a llama.cpp fallback); a FAIL on
 *  VRAM/SM with no fallback is plainly "blocked on your rig". When the rig
 *  clears it (or no preflight ran), we fall back to the card's own maturity.
 *
 *  Maturity mapping (card.status — production_candidate | experimental | qa |
 *  example | …). Only "production"/"production_candidate" earns the green
 *  Production badge; "experimental" and the non-production maturities (qa,
 *  example, bench_pending, anything unrecognised) get the honest amber
 *  Experimental badge rather than being silently dressed up as Production. */
export function deriveStatus(
  card: Record<string, unknown>,
  preflight: PreflightFitReport | null
): ModelCardStatus {
  const declared = asText(card.status, "");
  const fromCard: ModelCardStatus = declared.startsWith("production")
    ? "production"
    : "experimental";

  if (preflight && !preflight.can_run) {
    const failures = preflight.checks.filter((c) => c.status === "fail");
    const onlyGpuCount =
      failures.length > 0 && failures.every((c) => c.dimension === "gpu_count");
    const hasFallback = Boolean(asText(card.fallback_preset, ""));
    // gpu_count-only fail + a declared fallback → the escape-hatch path exists.
    if (onlyGpuCount && hasFallback) return "escape-hatch";
    return "blocked";
  }
  return fromCard;
}

function StatusChip({ status }: { status: ModelCardStatus }) {
  const meta = STATUS_META[status];
  const icon: ReactNode =
    status === "production" ? <CheckCircle2 size={13} />
      : status === "escape-hatch" ? <LifeBuoy size={13} />
        : status === "blocked" ? <CircleAlert size={13} />
          : <CircleAlert size={13} />;
  return <span className={`mc-status tone-${meta.tone}`}>{icon} {meta.label}</span>;
}

/** Human "needs N× M GB, sm_X.Y" string from card.hardware_fit (the typed
 *  envelope `sndr preflight` projects). Falls back gracefully when partial. */
function hardwareReq(fit: Record<string, unknown>): string {
  const gpus = asNumber(fit.requires_min_gpu_count) || asNumber(fit.tensor_parallel);
  const vram = asNumber(fit.requires_min_vram_gb);
  const cc = Array.isArray(fit.requires_min_cuda_capability)
    ? (fit.requires_min_cuda_capability as number[])
    : null;
  const parts: string[] = [];
  if (gpus) parts.push(`${gpus}× ${vram ? `${vram}GB` : tr("GPU")}`);
  else if (vram) parts.push(`${vram}GB/GPU`);
  if (cc && cc.length >= 2) parts.push(`sm_${cc[0]}.${cc[1]}+`);
  return parts.length ? parts.join(" · ") : tr("no declared requirement");
}

/** One-line "why pick this" from the card's allowed workloads + drafter. */
function whyThis(card: Record<string, unknown>): string {
  const allow = Array.isArray(card.workload_allow) ? (card.workload_allow as string[]) : [];
  const workloads = allow.slice(0, 3).map((w) => w.replace(/[._]/g, " ")).join(", ");
  const drafter = asText(card.drafter ?? card.drafter_model, "");
  const bits: string[] = [];
  if (workloads) bits.push(`${tr("Best for")} ${workloads}`);
  if (drafter) bits.push(`${tr("drafter")} ${drafter}`);
  if (!bits.length) {
    const summary = asText(card.summary, "").split("\n")[0]?.trim();
    return summary || tr("Composed runtime preset.");
  }
  return bits.join(" · ");
}

export function ModelCard({
  preset,
  preflight,
  active,
  onSelect
}: {
  preset: PresetRecord;
  // Live fit against the current/modeled rig (null = not yet checked).
  preflight: PreflightFitReport | null;
  active: boolean;
  onSelect: () => void;
}) {
  const card = (preset.card ?? {}) as Record<string, unknown>;
  const status = deriveStatus(card, preflight);
  const fit = (card.hardware_fit ?? {}) as Record<string, unknown>;
  const metric = (card.primary_metric ?? {}) as Record<string, unknown>;
  const tps = asNumber(metric.value);
  const metricKind = asText(metric.kind, "agg_TPS").replace(/^agg_/, "");
  const title = asText(card.title, preset.id);

  // Fit pill reflects the live preflight verdict (the same can_run the CLI
  // returns); absent a preflight it stays neutral ("not checked").
  const fitTone: "ok" | "warn" | "blocked" | "neutral" =
    !preflight ? "neutral"
      : preflight.can_run
        ? (preflight.verdict.includes("warning") ? "warn" : "ok")
        : "blocked";
  const fitLabel =
    !preflight ? tr("fit not checked")
      : preflight.can_run
        ? (fitTone === "warn" ? tr("fits with care") : tr("fits your rig"))
        : tr("won't fit");

  return (
    <button
      type="button"
      className={`model-card${active ? " active" : ""} mc-${status}`}
      onClick={onSelect}
      aria-pressed={active}
    >
      <div className="mc-head">
        <strong className="mc-id">{preset.id}</strong>
        <StatusChip status={status} />
      </div>
      <div className="mc-title">{title}</div>
      <div className="mc-meta">
        <span className="mc-metric" title={tr("Measured throughput from the preset's primary metric")}>
          <Gauge size={13} /> {tps > 0 ? `${tps.toLocaleString()} ${metricKind}` : tr("TPS pending")}
        </span>
        <span className="mc-hw" title={tr("Hardware the preset declares it needs")}>
          <Cpu size={13} /> {hardwareReq(fit)}
        </span>
        <span className={`fit-pill ${fitTone}`}>{fitLabel}</span>
      </div>
      <div className="mc-why" title={tr("Why pick this preset")}>
        <HelpCircle size={12} /> {whyThis(card)}
      </div>
    </button>
  );
}
