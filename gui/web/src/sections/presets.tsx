// SPDX-License-Identifier: Apache-2.0
// Preset section panels: operator-local user presets + profile-delta inspector.
import { type UserPresetList } from "../api";
import { tr } from "../i18n";

export function UserPresetsPanel({ presets }: { presets: UserPresetList | null }) {
  const rows = presets?.presets ?? [];
  return (
    <div className="config-item-inspector">
      <strong>{tr("User presets")} ({presets?.count ?? 0})</strong>
      <span>{tr("operator-local config dir")}</span>
      {rows.length === 0 ? (
        <p className="muted">{tr("No operator-local presets yet. Apply a draft to create one.")}</p>
      ) : (
        rows.map((preset) => (
          <p key={preset.id}>
            <em>{preset.id}</em>
            <code>{preset.model ?? "?"}{preset.profile ? ` / ${preset.profile}` : ""}</code>
          </p>
        ))
      )}
    </div>
  );
}

export function ProfileDeltaPanel({ def }: { def: Record<string, any> }) {
  const delta = (def.patches_delta ?? {}) as Record<string, any>;
  const enable = (delta.enable ?? {}) as Record<string, string>;
  const disable = Array.isArray(delta.disable) ? delta.disable : [];
  const override = (delta.override ?? {}) as Record<string, string>;
  const sizing = def.sizing_override as Record<string, any> | null;
  const op = def.override_policy as Record<string, any> | null;
  const promo = def.promotion as Record<string, any> | null;
  const validation = def.validation as Record<string, any> | null;
  const routing = def.routing as Record<string, any> | null;
  return (
    <div className="config-item-inspector delta">
      <strong>{tr("Profile delta:")} {def.id}</strong>
      <span>{String(def.status ?? "experimental")} · {tr("role")} {String(def.role ?? "default")}</span>
      <p><em>{tr("enable")}</em><code>{Object.keys(enable).length}</code></p>
      <p><em>{tr("disable")}</em><code>{disable.length}</code></p>
      <p><em>{tr("override")}</em><code>{Object.keys(override).length}</code></p>
      <p><em>{tr("sizing override")}</em><code>{sizing ? tr("yes") : tr("no")}</code></p>
      {op && <p title={String(op.reason ?? "")}><em>{tr("override policy")}</em><code>{String(op.override_class ?? "—")}{op.expires_at ? ` · ${tr("expires")} ${String(op.expires_at)}` : ""}</code></p>}
      {promo && <p><em>{tr("promotion")}</em><code>{promo.promote_to ? `→ ${String(promo.promote_to)}` : tr("none")}{Array.isArray(promo.validation_required) ? ` · ${promo.validation_required.length} ${tr("gates")}` : ""}</code></p>}
      {validation && (validation.artifact_id || validation.config_hash) && <p><em>{tr("validation")}</em><code>{String(validation.artifact_id ?? validation.config_hash)}</code></p>}
      {routing && <p><em>{tr("routing")}</em><code>{String(routing.routing_family ?? routing.family ?? "—")}</code></p>}
      {disable.length > 0 && (
        <p><em>{tr("disabled")}</em><code>{disable.map(String).join(", ")}</code></p>
      )}
    </div>
  );
}
