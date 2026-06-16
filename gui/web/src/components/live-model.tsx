// SPDX-License-Identifier: Apache-2.0
// Presentation of the live running-model detail (served model bridged to the
// SNDR catalog). Two self-fetching surfaces share one deduped query:
//   • LiveModelChip   — compact top-bar indicator (auto-discovered daemon engine)
//   • LiveModelInline — rich capability/preset line for the Chat & Clients panels
import { Cpu } from "lucide-react";
import { tr } from "../i18n";
import { useEngineModel } from "../hooks/useEngineModel";
import { firstModel, fmtCtx } from "../lib/live-model";

type Target = { host?: string; port?: number; apiKey?: string; hostId?: string };

/** Compact "what's running" chip for the top bar. Auto-discovers the daemon's
 *  configured engine when no target is given; renders nothing when nothing is
 *  served (keeps the bar clean). */
export function LiveModelChip({ onOpen, ...target }: Target & { onOpen?: () => void }) {
  const { data } = useEngineModel(target.host, target.port, target.apiKey, target.hostId);
  const m = firstModel(data);
  if (!m) {
    // Always visible (even with no engine) so the indicator is discoverable;
    // muted "no engine" state points the operator at the Clients screen.
    return (
      <button type="button" className="live-model-chip live-model-chip-off"
        title={tr("No running model detected — open Clients to point Host/Port at a vLLM engine")} onClick={onOpen}>
        <span className="live-dot" />
        <Cpu size={13} />
        <span className="live-model-name">{tr("No engine")}</span>
      </button>
    );
  }
  const ctx = fmtCtx(m.max_model_len);
  const preset = m.catalog?.presets?.[0]?.id;
  const title = [
    `${tr("Running model")}: ${m.id}`,
    ctx ? `${tr("Context")}: ${ctx}` : null,
    m.catalog ? `${tr("Catalog model")}: ${m.catalog.model_id}` : tr("Off-catalog model"),
    preset ? `${tr("Preset")}: ${preset}` : null,
    data?.version ? `vLLM ${data.version}` : null,
  ].filter(Boolean).join(" · ");
  return (
    <button type="button" className="live-model-chip" title={title} onClick={onOpen}>
      <span className="live-dot live-on" />
      <Cpu size={13} />
      <span className="live-model-name">{m.catalog?.model_id ?? m.id}</span>
      {ctx && <span className="live-model-ctx">{ctx}</span>}
    </button>
  );
}

/** Rich one-line detail: served model + context + capabilities + matched preset.
 *  Renders nothing when the target engine has no running model. */
export function LiveModelInline(target: Target) {
  const { data } = useEngineModel(target.host, target.port, target.apiKey, target.hostId);
  const m = firstModel(data);
  if (!m) return null;
  const c = m.catalog;
  const ctx = fmtCtx(m.max_model_len);
  return (
    <div className="live-model-inline">
      <span className="live-dot live-on" />
      <Cpu size={13} />
      <strong>{m.id}</strong>
      {ctx && <span className="live-model-tag">{tr("ctx")} {ctx}</span>}
      {c ? (
        <>
          {c.match_kind !== "id" && c.model_id !== m.id && <span className="live-model-tag">{c.model_id}</span>}
          {c.capabilities.tool_call_parser && <span className="live-model-tag">{tr("tools")}: {c.capabilities.tool_call_parser}</span>}
          {c.capabilities.reasoning_parser && <span className="live-model-tag">{tr("reasoning")}: {c.capabilities.reasoning_parser}</span>}
          {c.capabilities.spec_decode && <span className="live-model-tag">{tr("spec-decode")}</span>}
          {c.presets[0] && <span className="live-model-tag">{tr("preset")}: {c.presets[0].id}</span>}
        </>
      ) : (
        <span className="live-model-tag muted">{tr("off-catalog")}</span>
      )}
    </div>
  );
}
