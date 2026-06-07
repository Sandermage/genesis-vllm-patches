// SPDX-License-Identifier: Apache-2.0
// Preset insight panels: the runtime envelope (context/concurrency/patches/
// metric bars + KV/spec rows) and the workload policy graph (allow/deny pills +
// per-status distribution). Extracted from App.tsx (modularization) with no
// behavior change.
import { type PresetRecord } from "../api";
import { asRecord, asNumber, asText, asStringArray, countRecord } from "../lib/coerce";
import { formatTokens } from "../lib/format";
import { BarList } from "../components/charts";
import { InfoRows } from "../components/primitives";

export function RuntimeEnvelopePanel({
  card,
  composed,
  patchCount
}: {
  card: Record<string, unknown>;
  composed: Record<string, unknown>;
  patchCount: number;
}) {
  const metric = asRecord(card.primary_metric);
  const context = asNumber(composed.max_model_len);
  const sequences = asNumber(composed.max_num_seqs);
  const patches = asNumber(composed.enabled_patches_count);
  return (
    <div className="runtime-envelope">
      <BarList
        rows={[
          ["Context", Math.min(100, Math.round(context / 4096)), formatTokens(context)],
          ["Concurrency", Math.min(100, sequences * 10), String(sequences || "-")],
          ["Enabled patches", patchCount ? Math.round((patches / patchCount) * 100) : 0, String(patches || 0)],
          ["Metric", Math.min(100, Math.round(asNumber(metric.value) / 8)), String(asNumber(metric.value) || "pending")]
        ]}
      />
      <InfoRows
        rows={[
          ["KV Cache", asText(composed.kv_cache_dtype, "-")],
          ["Spec Decode", asText(composed.spec_decode_method, "-")],
          ["Spec K", String(asNumber(composed.spec_decode_K) || "-")],
          ["Evidence", asText(card.evidence_visibility, "unknown")]
        ]}
      />
    </div>
  );
}

export function PresetPolicyGraph({
  card,
  presets
}: {
  card: Record<string, unknown>;
  presets: PresetRecord[];
}) {
  const allow = asStringArray(card.workload_allow);
  const deny = asStringArray(card.workload_deny);
  const statuses = countRecord(
    presets
      .filter((preset) => preset.has_card)
      .map((preset) => asText(preset.card?.status, "unknown"))
  );
  const maxStatus = Math.max(1, ...Object.values(statuses));
  return (
    <div className="policy-graph">
      <div className="policy-pill-grid">
        {allow.map((item) => <span className="policy-pill allow" key={`allow-${item}`}>{item}</span>)}
        {deny.map((item) => <span className="policy-pill deny" key={`deny-${item}`}>{item}</span>)}
      </div>
      <BarList
        rows={Object.entries(statuses).map(([status, value]) => [
          status,
          Math.round((value / maxStatus) * 100),
          String(value)
        ])}
      />
    </div>
  );
}
