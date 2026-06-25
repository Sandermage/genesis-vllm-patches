// SPDX-License-Identifier: Apache-2.0
// Patch explain panel — the registry drill-down aside: enablement override,
// applicability, requires/conflicts graph, full metadata and the live decision
// from the Product API.
import { Wrench, Activity } from "lucide-react";
import { type PatchRow, type PatchExplainResult } from "../api";
import { asText, asStringArray } from "../lib/coerce";
import { formatAppliesTo } from "../lib/format";
import { StatusBadge, InfoRows } from "../components/primitives";
import { tr } from "../i18n";

function patchLifecycleExplanation(lifecycle: string) {
  const explanations: Record<string, string> = {
    stable: tr("Stable patches are expected to be safe in normal production profiles and should appear in release reports."),
    experimental: tr("Experimental patches need explicit evidence before they can be treated as a safe default."),
    research: tr("Research patches document an idea or investigation path and should stay out of automatic launch plans."),
    retired: tr("Retired patches remain visible for audit history but should not be proposed for new runtime plans."),
    qa: tr("QA patches are validation or test-oriented entries; expose them for diagnostics, not routine launch.")
  };
  return explanations[lifecycle] ?? tr("Lifecycle is defined by the registry and should be reviewed before enabling this patch.");
}

function patchDefaultExplanation(value: string) {
  const explanations: Record<string, string> = {
    applied: tr("Default-on with a real apply module. The GUI can include it in launch summaries and patch proof."),
    marker: tr("Default-on marker without runtime effect. The GUI must label it clearly so operators do not assume code changed."),
    "opt-in": tr("Disabled by default. It should require explicit operator selection and a fresh plan before Apply is available."),
    blocked: tr("Blocked for production use because implementation or lifecycle state is not safe enough for automatic enablement.")
  };
  return explanations[value] ?? tr("Default behavior is registry-defined and should be treated conservatively.");
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
        <strong>{tr("No patch selected")}</strong>
        <p>{tr("Use the search and filters to select a patch from the registry.")}</p>
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
  const composesWith = asStringArray(spec.composes_with ?? meta.composes_with);
  const supersededBy = asText(meta.superseded_by ?? spec.superseded_by, "");
  const prRelationship = asText(meta.upstream_pr_relationship ?? spec.upstream_pr_relationship, "").replace(/_/g, " ");
  const vvrRaw = meta.vllm_version_range ?? spec.vllm_version_range;
  const pinGate = Array.isArray(vvrRaw) ? vvrRaw.filter(Boolean).map(String).join(", ") : asText(vvrRaw, "");
  const canForce = Boolean(patch.env_flag);

  return (
    <aside className="patch-explain">
      <div className="patch-explain-head">
        <Wrench size={17} />
        <div>
          <strong>{patch.patch_id}</strong>
          <span>{patch.title || tr("Registry patch")}</span>
        </div>
        <StatusBadge status={patch.lifecycle} />
      </div>

      <div className="patch-override">
        <div className="patch-override-head">
          <strong>{tr("Enablement override")}</strong>
          {overrideCount > 0 && <span className="chip">{overrideCount} {tr("active")}</span>}
        </div>
        <div className="override-toggle" role="group" aria-label={tr("Enablement override")}>
          {OVERRIDE_OPTIONS.map((opt) => (
            <button
              key={opt.id}
              className={override === opt.id ? "active" : ""}
              aria-pressed={override === opt.id}
              disabled={!canForce && opt.id !== "default"}
              onClick={() => onSetOverride(opt.id)}
            >
              {tr(opt.label)}
            </button>
          ))}
        </div>
        <p className="muted">
          {canForce
            ? <>{tr("Writes")} <code>{patch.env_flag}={override === "off" ? "0" : "1"}</code> {tr("into the launch env (operator-local, reflected in the Launch Plan).")}</>
            : tr("This patch has no env flag — enablement is not operator-controllable.")}
        </p>
      </div>

      {description && (
        <div className="explain-note">
          <strong>{tr("What it does")}</strong>
          <p>{description}</p>
        </div>
      )}

      {appliesRows.length > 0 && (
        <div className="patch-applies">
          <strong>{tr("Supported models / applicability")}</strong>
          <InfoRows rows={appliesRows} />
        </div>
      )}
      {appliesRows.length === 0 && state === "ready" && (
        <p className="muted patch-applies-none">{tr("Applies to all catalog models (no model-specific constraints).")}</p>
      )}

      {(requires.length > 0 || conflicts.length > 0 || composesWith.length > 0) && (
        <div className="patch-deps">
          {requires.length > 0 && (
            <div>
              <span className="patch-dep-label">{tr("Requires")}</span>
              <div className="chip-row">
                {requires.map((r) => allPatchIds.has(r)
                  ? <button type="button" className="chip chip-link" key={r} onClick={() => onSelectPatch(r)} title={`${tr("Open")} ${r} ${tr("in the registry")}`}>{r} →</button>
                  : <span className="chip chip-unknown" key={r} title={tr("Not present in the current registry view")}>{r}</span>)}
              </div>
            </div>
          )}
          {conflicts.length > 0 && (
            <div>
              <span className="patch-dep-label danger">{tr("Conflicts with")}</span>
              <div className="chip-row">
                {conflicts.map((c) => allPatchIds.has(c)
                  ? <button type="button" className="chip danger chip-link" key={c} onClick={() => onSelectPatch(c)} title={`${tr("Open")} ${c} ${tr("in the registry")}`}>{c} →</button>
                  : <span className="chip danger chip-unknown" key={c} title={tr("Not present in the current registry view")}>{c}</span>)}
              </div>
            </div>
          )}
          {composesWith.length > 0 && (
            <div>
              <span className="patch-dep-label">{tr("Composes with")}</span>
              <div className="chip-row">
                {composesWith.map((c) => allPatchIds.has(c)
                  ? <button type="button" className="chip chip-link" key={c} onClick={() => onSelectPatch(c)} title={`${tr("Open")} ${c} ${tr("in the registry")}`}>{c} →</button>
                  : <span className="chip chip-unknown" key={c} title={tr("Not present in the current registry view")}>{c}</span>)}
              </div>
            </div>
          )}
        </div>
      )}

      <InfoRows
        rows={[
          [tr("Production default"), patch.production_default],
          [tr("Tier"), patch.tier],
          [tr("Family"), patch.family || "-"],
          [tr("Implementation"), asText(spec.implementation_status, patch.implementation_status)],
          [tr("Env flag"), patch.env_flag || "-"],
          [tr("Apply module"), patch.apply_module || "-"],
          [tr("Upstream PR"), patch.upstream_pr ? `#${patch.upstream_pr}${prRelationship ? ` · ${prRelationship}` : ""}` : (prRelationship || "-")],
          [tr("Related PRs"), relatedPrs.length ? relatedPrs.map((p) => `#${p}`).join(", ") : "-"],
          [tr("Pin gate"), pinGate || "-"],
          [tr("Category"), asText(spec.category, "-")],
          [tr("Source"), asText(spec.source, "-")],
          [tr("Credit"), credit || "-"]
        ]}
      />

      <div className={`patch-live-state ${state}`}>
        <Activity size={15} />
        <span>
          {state === "loading"
            ? tr("Loading Product API explain payload")
            : state === "error"
              ? `${tr("Explain API error:")} ${error ?? tr("unknown")}`
              : liveDecision
                ? `${tr("Live decision:")} ${liveDecision[0] ? tr("apply") : tr("skip")} / ${liveDecision[1]}`
                : `${tr("Live decision unavailable")}${detail?.live_decision_error ? `: ${detail.live_decision_error}` : ""}`}
        </span>
      </div>

      {supersededBy && (
        <div className="explain-note">
          <strong>{tr("Superseded by")}</strong>
          <p>{supersededBy}</p>
        </div>
      )}
      <div className="explain-note">
        <strong>{tr("Lifecycle")} — {tr(patch.lifecycle)}</strong>
        <p>{patchLifecycleExplanation(patch.lifecycle)}</p>
      </div>
      <div className="explain-note">
        <strong>{tr("Default behavior")} — {tr(patch.production_default)}</strong>
        <p>{patchDefaultExplanation(patch.production_default)}</p>
      </div>
    </aside>
  );
}
