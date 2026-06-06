// SPDX-License-Identifier: Apache-2.0
// Preset section panels: operator-local user presets + profile-delta inspector.
// Extracted from App.tsx (modularization) with no behavior change.
import { type UserPresetList } from "../api";

export function UserPresetsPanel({ presets }: { presets: UserPresetList | null }) {
  const rows = presets?.presets ?? [];
  return (
    <div className="config-item-inspector">
      <strong>User presets ({presets?.count ?? 0})</strong>
      <span>operator-local config dir</span>
      {rows.length === 0 ? (
        <p className="muted">No operator-local presets yet. Apply a draft to create one.</p>
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
  return (
    <div className="config-item-inspector delta">
      <strong>Profile delta: {def.id}</strong>
      <span>{String(def.status ?? "experimental")} · role {String(def.role ?? "default")}</span>
      <p><em>enable</em><code>{Object.keys(enable).length}</code></p>
      <p><em>disable</em><code>{disable.length}</code></p>
      <p><em>override</em><code>{Object.keys(override).length}</code></p>
      <p><em>sizing override</em><code>{sizing ? "yes" : "no"}</code></p>
      {disable.length > 0 && (
        <p><em>disabled</em><code>{disable.map(String).join(", ")}</code></p>
      )}
    </div>
  );
}
