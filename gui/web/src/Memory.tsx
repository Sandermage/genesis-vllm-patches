// SPDX-License-Identifier: Apache-2.0
// Persistent neural-graph memory panel — the operator-facing view of the
// memory engine (/api/v1/memory/*). Iteration 1c-i: data-wired (search /
// recall / remember / neighbors / stats / rebuild-links). The Obsidian-like
// Sigma.js force-graph is layered on in 1c-ii.
import { useCallback, useEffect, useState } from "react";
import { Brain, Search, Plus, RefreshCw, Network, Gauge, Share2, List } from "lucide-react";
import { api, type MemGraph, type MemHit, type MemNode, type MemNeighbor, type MemStats } from "./api";
import { MemoryGraph } from "./MemoryGraph";
import { tr } from "./i18n";

// Single homelab owner for now; the proxy/session will supply this later.
const OWNER = 1;

export function MemoryPanel() {
  const [stats, setStats] = useState<MemStats | null>(null);
  const [q, setQ] = useState("");
  const [hits, setHits] = useState<MemHit[]>([]);
  const [brain, setBrain] = useState(true); // recall (graph expand) vs pure search
  const [selected, setSelected] = useState<MemNode | null>(null);
  const [neighbors, setNeighbors] = useState<MemNeighbor[]>([]);
  const [remember, setRemember] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null); // transient success/info (not an error)
  const [view, setView] = useState<"list" | "graph">("list");
  const [graph, setGraph] = useState<MemGraph | null>(null);

  const loadStats = useCallback(() => {
    api.memoryStats(OWNER).then(setStats).catch((e) => setErr(String(e)));
  }, []);

  useEffect(loadStats, [loadStats]);

  const loadGraph = useCallback(() => {
    api.memoryGraph(OWNER, 300).then(setGraph).catch((e) => setErr(String(e)));
  }, []);

  useEffect(() => { if (view === "graph") loadGraph(); }, [view, loadGraph]);

  const runSearch = useCallback(async () => {
    if (!q.trim()) { setHits([]); return; }
    setBusy(true); setErr(null); setNotice(null);
    try {
      const r = brain
        ? await api.memoryRecall(q, OWNER, { limit: 25, expand_depth: 2, reinforce: false })
        : await api.memorySearch(q, OWNER, 25);
      setHits(r);
    } catch (e) { setErr(String(e)); }
    finally { setBusy(false); }
  }, [q, brain]);

  const openNode = useCallback(async (id: number) => {
    setErr(null);
    try {
      const [node, nb] = await Promise.all([api.memoryNode(id, OWNER), api.memoryNeighbors(id, OWNER)]);
      setSelected(node); setNeighbors(nb);
    } catch (e) { setErr(String(e)); }
  }, []);

  const doRemember = useCallback(async () => {
    if (!remember.trim()) return;
    setBusy(true); setErr(null); setNotice(null);
    try {
      await api.memoryRemember(remember, OWNER);
      setRemember("");
      loadStats();
      if (view === "graph") loadGraph();
    } catch (e) { setErr(String(e)); }
    finally { setBusy(false); }
  }, [remember, loadStats, loadGraph, view]);

  const rebuildLinks = useCallback(async () => {
    setBusy(true); setErr(null); setNotice(null);
    try {
      // Consolidate = auto-link + detect communities (clouds) + rank importance.
      const r = await api.memoryConsolidate(OWNER, { tau: 0.8, k: 10 });
      // Success feedback goes to `notice` (not `err`) so it isn't styled as an error.
      setNotice(`+${r.linked} ${tr("links")}, ${r.communities} ${tr("clouds")}`);
      loadStats();
      if (view === "graph") loadGraph();
    } catch (e) { setErr(String(e)); }
    finally { setBusy(false); }
  }, [loadStats, loadGraph, view]);

  return (
    <div className="mem-panel" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div className="mem-stats" style={{ display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
        <span style={{ display: "flex", alignItems: "center", gap: 6 }}><Brain size={15} /> <b>{tr("Memory")}</b></span>
        <span style={{ display: "flex", alignItems: "center", gap: 6 }}><Gauge size={13} /> {tr("Nodes")}: <b>{stats?.nodes ?? "—"}</b></span>
        <span style={{ display: "flex", alignItems: "center", gap: 6 }}><Network size={13} /> {tr("Edges")}: <b>{stats?.edges ?? "—"}</b></span>
        <button className="btn btn-ghost" onClick={rebuildLinks} disabled={busy} style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <RefreshCw size={13} /> {tr("Rebuild links")}
        </button>
        <div style={{ marginLeft: "auto", display: "flex", gap: 4 }}>
          <button className="btn btn-ghost" aria-pressed={view === "list"} onClick={() => setView("list")} title={tr("List")} style={{ display: "flex", alignItems: "center", gap: 4, opacity: view === "list" ? 1 : 0.6 }}>
            <List size={13} /> {tr("List")}
          </button>
          <button className="btn btn-ghost" aria-pressed={view === "graph"} onClick={() => setView("graph")} title={tr("Graph")} style={{ display: "flex", alignItems: "center", gap: 4, opacity: view === "graph" ? 1 : 0.6 }}>
            <Share2 size={13} /> {tr("Graph")}
          </button>
        </div>
      </div>

      <div className="mem-remember" style={{ display: "flex", gap: 8 }}>
        <input
          value={remember}
          onChange={(e) => setRemember(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") doRemember(); }}
          placeholder={tr("Remember a fact, note, or idea…")}
          style={{ flex: 1 }}
        />
        <button className="btn" onClick={doRemember} disabled={busy || !remember.trim()} style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <Plus size={13} /> {tr("Remember")}
        </button>
      </div>

      <div className="mem-search" style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") runSearch(); }}
          placeholder={tr("Search memory…")}
          style={{ flex: 1 }}
        />
        <label style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 12 }} title={tr("Spread activation across the graph (brain recall) vs pure vector search")}>
          <input type="checkbox" checked={brain} onChange={(e) => setBrain(e.target.checked)} /> {tr("Brain recall")}
        </label>
        <button className="btn" onClick={runSearch} disabled={busy} style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <Search size={13} /> {tr("Search")}
        </button>
      </div>

      {err && <div className="mem-err" style={{ fontSize: 12, color: "var(--danger, #f97362)" }}>{err}</div>}
      {notice && <div className="mem-notice" style={{ fontSize: 12, color: "var(--success, #5fd07d)" }}>{notice}</div>}

      {view === "graph" && (
        graph
          ? <MemoryGraph graph={graph} onSelect={openNode} />
          : <div style={{ opacity: 0.5, fontSize: 13, padding: 24 }}>{tr("Loading graph…")}</div>
      )}

      {view === "list" && (
      <div style={{ display: "flex", gap: 16, alignItems: "flex-start" }}>
        <ul className="mem-hits" style={{ flex: 1, listStyle: "none", padding: 0, margin: 0, display: "flex", flexDirection: "column", gap: 6 }}>
          {hits.map((h) => (
            <li key={h.id}>
              <button
                type="button"
                onClick={() => openNode(h.id)}
                style={{ width: "100%", textAlign: "left", font: "inherit", color: "inherit", background: "none", cursor: "pointer", padding: "6px 10px", border: "1px solid var(--border, #2a2a2a)", borderRadius: 6, display: "flex", justifyContent: "space-between", gap: 8 }}
              >
                <span>{h.content}</span>
                <span style={{ opacity: 0.6, fontVariantNumeric: "tabular-nums" }}>{h.score.toFixed(3)}</span>
              </button>
            </li>
          ))}
          {!hits.length && !busy && <li style={{ opacity: 0.5, fontSize: 13 }}>{tr("No results yet — remember something, then search.")}</li>}
        </ul>

        {selected && (
          <div className="mem-detail" style={{ flex: 1, border: "1px solid var(--border, #2a2a2a)", borderRadius: 6, padding: 12 }}>
            <div style={{ fontSize: 12, opacity: 0.6 }}>#{selected.id} · {selected.kind} · {tr("accessed")} {selected.access_count}×</div>
            <p style={{ margin: "6px 0 12px" }}>{selected.content}</p>
            <div style={{ fontSize: 12, opacity: 0.7, marginBottom: 4 }}>{tr("Connections")} ({neighbors.length})</div>
            <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "flex", flexDirection: "column", gap: 4 }}>
              {neighbors.map((n) => (
                <li key={`${n.id}-${n.rel}`}>
                  <button
                    type="button"
                    onClick={() => openNode(n.id)}
                    style={{ width: "100%", textAlign: "left", font: "inherit", color: "inherit", background: "none", border: "none", padding: 0, cursor: "pointer", fontSize: 13, display: "flex", justifyContent: "space-between", gap: 8 }}
                  >
                    <span>→ #{n.id} <span style={{ opacity: 0.6 }}>{n.rel}</span></span>
                    <span style={{ opacity: 0.6, fontVariantNumeric: "tabular-nums" }}>{n.weight.toFixed(2)}</span>
                  </button>
                </li>
              ))}
              {!neighbors.length && <li style={{ opacity: 0.5, fontSize: 12 }}>{tr("No connections yet — try Rebuild links.")}</li>}
            </ul>
          </div>
        )}
      </div>
      )}
    </div>
  );
}
