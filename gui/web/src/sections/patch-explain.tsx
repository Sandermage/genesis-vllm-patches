// SPDX-License-Identifier: Apache-2.0
// Patch explain panel — the registry drill-down aside: enablement override,
// applicability, requires/conflicts graph, full metadata and the live decision
// from the Product API. Extracted from App.tsx (modularization).
//
// Enterprise touch over the inline original (classes unchanged): the
// default/force-on/force-off override is a role="group" whose buttons expose
// aria-pressed, so assistive tech announces the chosen override.
import { Wrench, Activity } from "lucide-react";
import { type PatchRow, type PatchExplainResult } from "../api";
import { asText, asStringArray } from "../lib/coerce";
import { formatAppliesTo } from "../lib/format";
import { StatusBadge, InfoRows } from "../components/primitives";

function patchLifecycleExplanation(lifecycle: string) {
  const explanations: Record<string, string> = {
    stable: "Stable patches are expected to be safe in normal production profiles and should appear in release reports.",
    experimental: "Experimental patches need explicit evidence before they can be treated as a safe default.",
    research: "Research patches document an idea or investigation path and should stay out of automatic launch plans.",
    retired: "Retired patches remain visible for audit history but should not be proposed for new runtime plans.",
    qa: "QA patches are validation or test-oriented entries; expose them for diagnostics, not routine launch."
  };
  return explanations[lifecycle] ?? "Lifecycle is defined by the registry and should be reviewed before enabling this patch.";
}

function patchDefaultExplanation(value: string) {
  const explanations: Record<string, string> = {
    applied: "Default-on with a real apply module. The GUI can include it in launch summaries and patch proof.",
    marker: "Default-on marker without runtime effect. The GUI must label it clearly so operators do not assume code changed.",
    "opt-in": "Disabled by default. It should require explicit operator selection and a fresh plan before Apply is available.",
    blocked: "Blocked for production use because implementation or lifecycle state is not safe enough for automatic enablement."
  };
  return explanations[value] ?? "Default behavior is registry-defined and should be treated conservatively.";
}

const OVERRIDE_OPTIONS = [
  { id: "default", label: "Registry default" },
  { id: "on", label: "Force on" },
  { id: "off", label: "Force off" }
];

export function PatchExplainPanel({
  patch,
  detail,
  state,
  error,
  override,
  overrideCount,
  onSetOverride,
  allPatchIds,
  onSelectPatch
}: {
  patch: PatchRow | null;
  detail: PatchExplainResult | null;
  state: "idle" | "loading" | "ready" | "error";
  error: string | null;
  override: string;
  overrideCount: number;
  onSetOverride: (state: string) => void;
  allPatchIds: Set<string>;
  onSelectPatch: (id: string) => void;
}) {
  if (!patch) {
    return (
      <aside className="patch-explain">
        <strong>No patch selected</strong>
        <p>Use the search and filters to select a patch from the registry.</p>
      </aside>
    );
  }
  const spec = detail?.spec ?? {};
  const meta = detail?.meta ?? {};
  const liveDecision = detail?.live_decision;
  const description = asText(meta.experimental_note ?? spec.experimental_note, "");
  const appliesRows = formatAppliesTo(spec.applies_to ?? meta.applies_to);
  const requires = asStringArray(spec.requires_patches ?? meta.requires_patches);
  const conflicts = asStringArray(spec.conflicts_with ?? meta.conflicts_with);
  const relatedPrs = asStringArray(spec.related_upstream_prs ?? meta.related_upstream_prs);
  const credit = asText(meta.credit ?? spec.credit, "");
  const canForce = Boolean(patch.env_flag);

  return (
    <aside className="patch-explain">
      <div className="patch-explain-head">
        <Wrench size={17} />
        <div>
          <strong>{patch.patch_id}</strong>
          <span>{patch.title || "Registry patch"}</span>
        </div>
        <StatusBadge status={patch.lifecycle} />
      </div>

      {/* Enablement override — operator forces on/off, written into the launch env. */}
      <div className="patch-override">
        <div className="patch-override-head">
          <strong>Enablement override</strong>
          {overrideCount > 0 && <span className="chip">{overrideCount} active</span>}
        </div>
        <div className="override-toggle" role="group" aria-label="Enablement override">
          {OVERRIDE_OPTIONS.map((opt) => (
            <button
              key={opt.id}
              className={override === opt.id ? "active" : ""}
              aria-pressed={override === opt.id}
              disabled={!canForce && opt.id !== "default"}
              onClick={() => onSetOverride(opt.id)}
            >
              {opt.label}
            </button>
          ))}
        </div>
        <p className="muted">
          {canForce
            ? <>Writes <code>{patch.env_flag}={override === "off" ? "0" : "1"}</code> into the launch env (operator-local, reflected in the Launch Plan).</>
            : "This patch has no env flag — enablement is not operator-controllable."}
        </p>
      </div>

      {description && (
        <div className="explain-note">
          <strong>What it does</strong>
          <p>{description}</p>
        </div>
      )}

      {appliesRows.length > 0 && (
        <div className="patch-applies">
          <strong>Supported models / applicability</strong>
          <InfoRows rows={appliesRows} />
        </div>
      )}
      {appliesRows.length === 0 && state === "ready" && (
        <p className="muted patch-applies-none">Applies to all catalog models (no model-specific constraints).</p>
      )}

      {(requires.length > 0 || conflicts.length > 0) && (
        <div className="patch-deps">
          {requires.length > 0 && (
            <div>
              <span className="patch-dep-label">Requires</span>
              <div className="chip-row">
                {requires.map((r) => allPatchIds.has(r)
                  ? <button type="button" className="chip chip-link" key={r} onClick={() => onSelectPatch(r)} title={`Open ${r} in the registry`}>{r} →</button>
                  : <span className="chip chip-unknown" key={r} title="Not present in the current registry view">{r}</span>)}
              </div>
            </div>
          )}
          {conflicts.length > 0 && (
            <div>
              <span className="patch-dep-label danger">Conflicts with</span>
              <div className="chip-row">
                {conflicts.map((c) => allPatchIds.has(c)
                  ? <button type="button" className="chip danger chip-link" key={c} onClick={() => onSelectPatch(c)} title={`Open ${c} in the registry`}>{c} →</button>
                  : <span className="chip danger chip-unknown" key={c} title="Not present in the current registry view">{c}</span>)}
              </div>
            </div>
          )}
        </div>
      )}

      <InfoRows
        rows={[
          ["Production default", patch.production_default],
          ["Tier", patch.tier],
          ["Family", patch.family || "-"],
          ["Implementation", asText(spec.implementation_status, patch.implementation_status)],
          ["Env flag", patch.env_flag || "-"],
          ["Apply module", patch.apply_module || "-"],
          ["Upstream PR", patch.upstream_pr ? `#${patch.upstream_pr}` : "-"],
          ["Related PRs", relatedPrs.length ? relatedPrs.map((p) => `#${p}`).join(", ") : "-"],
          ["Category", asText(spec.category, "-")],
          ["Source", asText(spec.source, "-")],
          ["Credit", credit || "-"]
        ]}
      />

      <div className={`patch-live-state ${state}`}>
        <Activity size={15} />
        <span>
          {state === "loading"
            ? "Loading Product API explain payload"
            : state === "error"
              ? `Explain API error: ${error ?? "unknown"}`
              : liveDecision
                ? `Live decision: ${liveDecision[0] ? "apply" : "skip"} / ${liveDecision[1]}`
                : `Live decision unavailable${detail?.live_decision_error ? `: ${detail.live_decision_error}` : ""}`}
        </span>
      </div>

      <div className="explain-note">
        <strong>Lifecycle — {patch.lifecycle}</strong>
        <p>{patchLifecycleExplanation(patch.lifecycle)}</p>
      </div>
      <div className="explain-note">
        <strong>Default behavior — {patch.production_default}</strong>
        <p>{patchDefaultExplanation(patch.production_default)}</p>
      </div>
    </aside>
  );
}
