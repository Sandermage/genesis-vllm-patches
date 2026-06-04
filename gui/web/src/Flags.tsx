import { memo, useEffect, useMemo, useState } from "react";
import { Box, RefreshCw, Search, ToggleLeft, ToggleRight } from "lucide-react";
import { api, type FlagMatrix, type FlagRow } from "./api";

const DRIFT_LABEL: Record<string, string> = { missing: "missing", extra: "extra", in_sync: "in sync" };

export function FlagsPanel() {
  const [data, setData] = useState<FlagMatrix | null>(null);
  const [loading, setLoading] = useState(false);
  const [q, setQ] = useState("");
  const [family, setFamily] = useState("");
  const [def, setDef] = useState<"" | "on" | "off">("");
  const [container, setContainer] = useState("");

  const load = (c?: string) => {
    setLoading(true);
    api.flagsMatrix(c || undefined).then(setData).catch(() => setData(null)).finally(() => setLoading(false));
  };
  useEffect(() => { load(); }, []);

  const families = useMemo(() => Array.from(new Set((data?.flags ?? []).map((f) => f.family).filter(Boolean))).sort() as string[], [data]);
  const rows = useMemo(() => {
    const needle = q.trim().toLowerCase();
    return (data?.flags ?? []).filter((f) =>
      (!needle || f.env_flag.toLowerCase().includes(needle) || (f.patch_id ?? "").toLowerCase().includes(needle)) &&
      (!family || f.family === family) &&
      (!def || (def === "on" ? f.default_on : !f.default_on)));
  }, [data, q, family, def]);

  const c = data?.counts;
  return (
    <div className="fm">
      <div className="fm-bar">
        <label className="fm-search"><Search size={14} />
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="filter by flag or patch id…" spellCheck={false} />
        </label>
        <select value={family} onChange={(e) => setFamily(e.target.value)}>
          <option value="">all families</option>
          {families.map((f) => <option key={f} value={f}>{f}</option>)}
        </select>
        <select value={def} onChange={(e) => setDef(e.target.value as "" | "on" | "off")}>
          <option value="">any default</option>
          <option value="on">default ON</option>
          <option value="off">default OFF</option>
        </select>
        <label className="fm-live"><Box size={13} />
          <input value={container} onChange={(e) => setContainer(e.target.value)} placeholder="container for live state…" spellCheck={false}
            onKeyDown={(e) => { if (e.key === "Enter") load(container); }} />
        </label>
        <button className="ghost-button" onClick={() => load(container)} disabled={loading}>
          {loading ? <RefreshCw size={14} className="spin" /> : <RefreshCw size={14} />} {container ? "Overlay live" : "Reload"}
        </button>
      </div>

      {c && (
        <div className="fm-counts">
          <span className="fm-kpi"><strong>{c.total}</strong> flags</span>
          <span className="fm-kpi ok"><ToggleRight size={13} /> {c.default_on} default ON</span>
          <span className="fm-kpi"><ToggleLeft size={13} /> {c.default_off} default OFF</span>
          {data?.has_live && <span className="fm-kpi warn">{c.missing} missing</span>}
          {data?.has_live && <span className="fm-kpi extra">{c.extra} extra</span>}
          <span className="fm-kpi muted">{rows.length} shown</span>
        </div>
      )}

      <div className="fm-table">
        <div className="fm-head">
          <span>Flag</span><span>Patch</span><span>Family</span><span>Default</span>
          {data?.has_live && <span>Live</span>}
        </div>
        {rows.map((f) => <FlagRowView key={f.env_flag} f={f} hasLive={data?.has_live ?? false} />)}
        {rows.length === 0 && <div className="fm-empty">No flags match the filters.</div>}
      </div>
    </div>
  );
}

const FlagRowView = memo(function FlagRowView({ f, hasLive }: { f: FlagRow; hasLive: boolean }) {
  return (
    <div className={`fm-row${f.drift && f.drift !== "in_sync" ? ` drift-${f.drift}` : ""}`}>
      <span className="fm-flag" title={f.title ?? ""}>{f.env_flag.replace(/^GENESIS_ENABLE_/, "")}</span>
      <span className="fm-patch">{f.patch_id}</span>
      <span className="fm-fam">{f.family}{f.tier ? ` · ${f.tier}` : ""}</span>
      <span className={`fm-default ${f.default_on ? "on" : "off"}`}>{f.default_on ? "ON" : "OFF"}</span>
      {hasLive && (
        <span className="fm-livecell">
          {f.live_on === undefined ? "—" : <span className={`fm-livebadge ${f.live_on ? "on" : "off"}`}>{f.live_on ? "ON" : "OFF"}</span>}
          {f.drift && f.drift !== "in_sync" && <em className={`fm-drift ${f.drift}`}>{DRIFT_LABEL[f.drift]}</em>}
        </span>
      )}
    </div>
  );
});
