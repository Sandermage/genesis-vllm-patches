// SPDX-License-Identifier: Apache-2.0
// Retire-impact monitor — surfaces the anchor-SoT retire-impact detector: which
// active dependents a retired patch would break. HIGH = a perf-bearing dependent
// whose anchor targets the retired patch's emitted bytes (the dev301-class silent
// perf regression the pin-bump preflight gate exists to catch); MEDIUM = a
// registry edge only. Read-only; self-fetches /api/v1/patches/retire-impact.
import { AlertTriangle, CircleAlert, CircleCheck, Loader2, RefreshCw } from "lucide-react";
import { tr } from "../i18n";
import { useApiQuery } from "../hooks/useApiQuery";
import { api, type RetireImpactEdge } from "../api";

function EdgeRow({ e }: { e: RetireImpactEdge }) {
  const high = e.severity === "HIGH";
  return (
    <div className={`retire-edge ${high ? "high" : "medium"}`} title={e.detail ?? ""}>
      <span className={`retire-sev ${high ? "high" : "medium"}`}>{e.severity}</span>
      <code className="retire-pair">{e.retired} → {e.dependent}</code>
      <span className="retire-meta">{e.dependent_lifecycle ?? "—"}{e.dependent_default_on ? ` · ${tr("default-on")}` : ""}</span>
      <span className="retire-via">{tr("via")} {(e.via ?? []).join(", ")}</span>
    </div>
  );
}

export function RetireImpactPanel() {
  const { data, state, error, reload } = useApiQuery(
    ["retire-impact"],
    (signal) => api.retireImpact(signal),
    { staleTime: 30_000 },
  );

  if (state === "loading" || state === "idle") {
    return <div className="muted"><Loader2 size={14} /> {tr("Loading retire-impact…")}</div>;
  }
  if (state === "error") {
    return <div className="chat-advisory"><CircleAlert size={13} /> <span>{error ?? tr("Failed to load retire-impact.")}</span></div>;
  }

  const edges = data?.edges ?? [];
  const high = data?.high_count ?? 0;
  const medium = data?.medium_count ?? 0;

  return (
    <div className="retire-impact">
      <div className="retire-impact-head">
        <span className={high > 0 ? "retire-summary danger" : "retire-summary ok"}>
          {high > 0
            ? <><AlertTriangle size={13} /> {high} {tr("HIGH")}</>
            : <><CircleCheck size={13} /> {tr("no HIGH breaks")}</>}
          {medium > 0 ? <span className="muted"> · {medium} {tr("MEDIUM")}</span> : null}
        </span>
        <button className="ghost-button" onClick={reload}><RefreshCw size={12} /> {tr("Refresh")}</button>
      </div>
      {data?.error && <div className="muted">{tr("Detector unavailable off-engine")} ({data.error})</div>}
      {edges.length === 0
        ? <div className="muted">{tr("No retirement breaks an active dependent — clean.")}</div>
        : <div className="retire-edges">{edges.map((e, i) => <EdgeRow key={`${e.retired}-${e.dependent}-${i}`} e={e} />)}</div>}
    </div>
  );
}
