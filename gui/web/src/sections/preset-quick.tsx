// SPDX-License-Identifier: Apache-2.0
// Selected-preset quick panel: status + runtime summary + allowed workloads and
// the edit/card/policy/launch actions. Extracted from App.tsx (modularization).
//
// Enterprise touch over the inline original (classes unchanged): the allowed-
// workload chips are grouped under role="group" + aria-label.
import { MousePointerClick, Wrench, FileText, BarChart3, Rocket } from "lucide-react";
import { type PresetRecord } from "../api";
import { asText, asStringArray, asNumber } from "../lib/coerce";
import { formatTokens } from "../lib/format";
import { StatusBadge, InfoRows } from "../components/primitives";

export function PresetQuickPanel({
  selectedPreset,
  record,
  card,
  composed,
  onOpenCard,
  onEdit,
  onPolicy,
  onLaunch
}: {
  selectedPreset: string;
  record: PresetRecord | null;
  card: Record<string, unknown>;
  composed: Record<string, unknown>;
  onOpenCard: () => void;
  onEdit: () => void;
  onPolicy: () => void;
  onLaunch: () => void;
}) {
  if (!selectedPreset) {
    return (
      <section className="preset-quick empty">
        <div className="preset-quick-empty">
          <MousePointerClick size={26} />
          <strong>Select a preset</strong>
          <p>Click any row in the catalog to see its runtime, evidence and editing actions here.</p>
        </div>
      </section>
    );
  }
  const status = record?.has_card ? asText(card.status, "available") : "missing";
  const workloads = asStringArray(card.workload_allow);
  return (
    <section className="preset-quick">
      <div className="preset-quick-head">
        <div>
          <span className="preset-quick-kicker">Selected preset</span>
          <strong>{selectedPreset}</strong>
        </div>
        <StatusBadge status={status} />
      </div>
      <p className="preset-quick-title">{asText(card.title, "Unannotated preset — no card metadata yet.")}</p>
      <InfoRows
        rows={[
          ["Model", asText(record?.model ?? composed.model, "-")],
          ["Hardware", asText(record?.hardware ?? composed.hardware, "-")],
          ["Profile", asText(record?.profile ?? composed.profile, "-")],
          ["Mode", asText(card.mode, "-")],
          ["Max context", formatTokens(asNumber(composed.max_model_len))],
          ["KV cache", asText(composed.kv_cache_dtype, "-")],
          ["Spec decode", `${asText(composed.spec_decode_method, "-")} / K=${asText(composed.spec_decode_K, "-")}`],
          ["Patches", asText(composed.enabled_patches_count, "-")],
          ["Fallback", asText(card.fallback_preset, "none")]
        ]}
      />
      {workloads.length > 0 && (
        <div className="preset-quick-workloads">
          <span className="preset-quick-kicker">Allowed workloads</span>
          <div className="chip-row" role="group" aria-label="Allowed workloads">
            {workloads.map((item) => <span className="chip" key={item}>{item}</span>)}
          </div>
        </div>
      )}
      <div className="preset-quick-actions">
        <button className="primary-button" onClick={onEdit}>
          <Wrench size={15} /> Edit preset
        </button>
        <button className="ghost-button" onClick={onOpenCard}>
          <FileText size={14} /> Full card
        </button>
        <button className="ghost-button" onClick={onPolicy}>
          <BarChart3 size={14} /> Policy
        </button>
        <button className="ghost-button" onClick={onLaunch}>
          <Rocket size={14} /> Launch
        </button>
      </div>
    </section>
  );
}
