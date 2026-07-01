// SPDX-License-Identifier: Apache-2.0
// Persistent neural-graph memory panel — the operator-facing view of the
// memory engine (/api/v1/memory/*). Iteration 1c-i: data-wired (search /
// recall / remember / neighbors / stats / rebuild-links). The Obsidian-like
// Sigma.js force-graph is layered on in 1c-ii.
import { useCallback, useEffect, useState } from "react";
import { Brain, Search, Plus, RefreshCw, Network, Gauge, Share2, List, Trash2, Download, Upload, X, Sparkles } from "lucide-react";
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
  const [detailBusy, setDetailBusy] = useState(false); // loading a node's detail + neighbors
  const [view, setView] = useState<"list" | "graph">("list");
  const [graph, setGraph] = useState<MemGraph | null>(null);
  const [limit, setLimit] = useState(25);      // max recall/search hits
  const [depth, setDepth] = useState(2);        // brain-recall graph-expand depth
  const [showImport, setShowImport] = useState(false);
  const [importPath, setImportPath] = useState("");

  const loadStats = useCallback(() => {
    api.memoryStats(OWNER).then(setStats).catch((e) => setErr(String(e)));
  }, []);

  useEffect(loadStats, [loadStats]);

  const loadGraph = useCallback(() => {
    api.memoryGraph(OWNER, 300).then(setGraph).catch((e) => setErr(String(e)));
  }, []);

  useEffect(() => { if (view === "graph") loadGraph(); }, [view, loadGraph]);

  const runSearch = useCallback(async () => {
    // Guard against Enter-key re-entry: the buttons are disabled while busy,
    // but the input onKeyDown paths aren't — a held/double Enter used to fire
    // duplicate requests (and the older response could land last).
    if (busy) return;
    if (!q.trim()) { setHits([]); return; }
    setBusy(true); setErr(null); setNotice(null);
    try {
      const r = brain
        ? await api.memoryRecall(q, OWNER, { limit, expand_depth: depth, reinforce: false })
        : await api.memorySearch(q, OWNER, limit);
      setHits(r);
    } catch (e) { setErr(String(e)); }
    finally { setBusy(false); }
  }, [busy, q, brain, limit, depth]);

  const openNode = useCallback(async (id: number) => {
    setErr(null); setNotice(null); setDetailBusy(true);
    try {
      const [node, nb] = await Promise.all([api.memoryNode(id, OWNER), api.memoryNeighbors(id, OWNER)]);
      setSelected(node); setNeighbors(nb);
    } catch (e) { setErr(String(e)); }
    finally { setDetailBusy(false); }
  }, []);

  const doRemember = useCallback(async () => {
    if (busy) return; // Enter-key path bypasses the disabled button — no duplicate nodes
    if (!remember.trim()) return;
    setBusy(true); setErr(null); setNotice(null);
    try {
      await api.memoryRemember(remember, OWNER);
      setRemember("");
      loadStats();
      if (view === "graph") loadGraph();
    } catch (e) { setErr(String(e)); }
    finally { setBusy(false); }
  }, [busy, remember, loadStats, loadGraph, view]);

  const doForget = useCallback(async (id: number) => {
    if (!window.confirm(tr("Forget this memory? This permanently deletes the node and its connections."))) return;
    setBusy(true); setErr(null); setNotice(null);
    try {
      await api.memoryDelete(id, OWNER);
      setSelected(null); setNeighbors([]);
      setHits((h) => h.filter((x) => x.id !== id));
      setNotice(tr("Forgotten."));
      loadStats();
      if (view === "graph") loadGraph();
    } catch (e) { setErr(String(e)); }
    finally { setBusy(false); }
  }, [loadStats, loadGraph, view]);

  const doExport = useCallback(async () => {
    setBusy(true); setErr(null); setNotice(null);
    try {
      // Backup the owner's whole graph (nodes + edges) as JSON — a portable
      // snapshot of the persistent memory DB.
      const g = await api.memoryGraph(OWNER, 100000);
      const blob = new Blob([JSON.stringify(g, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url; link.download = `sndr-memory-owner-${OWNER}.json`; link.click();
      URL.revokeObjectURL(url);
      setNotice(`${tr("Exported")} ${g.nodes.length} ${tr("nodes")}, ${g.edges.length} ${tr("edges")}.`);
    } catch (e) { setErr(String(e)); }
    finally { setBusy(false); }
  }, []);

  const doImport = useCallback(async () => {
    if (busy) return; // Enter-key path bypasses the disabled button — no duplicate imports
    setBusy(true); setErr(null); setNotice(null);
    try {
      const r = await api.memoryImportObsidian(importPath.trim() || ".", OWNER);  // empty → whole vault
      setShowImport(false); setImportPath("");
      setNotice(`${tr("Imported")} ${r.notes} ${tr("notes")}, ${r.links} ${tr("links")}${r.missing ? ` · ${r.missing} ${tr("unresolved")}` : ""}.`);
      loadStats();
      if (view === "graph") loadGraph();
    } catch (e) { setErr(String(e)); }
    finally { setBusy(false); }
  }, [busy, importPath, loadStats, loadGraph, view]);

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
        <span style={{ display: "flex", alignItems: "center", gap: 6 }}><Share2 size={13} /> {tr("Communities")}: <b>{stats?.communities ?? "—"}</b></span>
        <button className="btn btn-ghost" onClick={rebuildLinks} disabled={busy} style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <RefreshCw size={13} /> {tr("Rebuild links")}
        </button>
        <button className="btn btn-ghost" onClick={doExport} disabled={busy} title={tr("Download the memory graph as JSON (backup)")} style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <Download size={13} /> {tr("Export")}
        </button>
        <button className="btn btn-ghost" onClick={() => { setErr(null); setNotice(null); setShowImport(true); }} disabled={busy} title={tr("Import an Obsidian vault (notes → memories, [[wikilinks]] → edges)")} style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <Upload size={13} /> {tr("Import")}
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

      <div className="mem-remember" style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <input
          value={remember}
          onChange={(e) => setRemember(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") doRemember(); }}
          placeholder={tr("Remember a fact, note, or idea…")}
          style={{ flex: "1 1 220px", minWidth: 0 }}
        />
        <button className="btn" onClick={doRemember} disabled={busy || !remember.trim()} style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <Plus size={13} /> {tr("Remember")}
        </button>
      </div>

      <div className="mem-search" style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") runSearch(); }}
          placeholder={tr("Search memory…")}
          style={{ flex: "1 1 200px", minWidth: 0 }}
        />
        <label style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 12 }} title={tr("Spread activation across the graph (brain recall) vs pure vector search")}>
          <input type="checkbox" checked={brain} onChange={(e) => setBrain(e.target.checked)} /> {tr("Brain recall")}
        </label>
        <label style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 12 }} title={tr("Max results")}>
          {tr("Limit")} <input type="number" min={1} max={200} value={limit} onChange={(e) => setLimit(Math.min(200, Math.max(1, Number(e.target.value) || 25)))} style={{ width: 56 }} />
        </label>
        {brain && (
          <label style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 12 }} title={tr("Graph-expansion hops for brain recall")}>
            {tr("Depth")} <input type="number" min={0} max={5} value={depth} onChange={(e) => setDepth(Math.min(5, Math.max(0, Number(e.target.value) || 0)))} style={{ width: 48 }} />
          </label>
        )}
        <button className="btn" onClick={runSearch} disabled={busy} style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <Search size={13} /> {tr("Search")}
        </button>
      </div>

      {err && <div role="alert" style={{ fontSize: 12.5, color: "var(--danger)", background: "var(--danger-soft)", padding: "7px 11px", borderRadius: "var(--r-sm)" }}>{err}</div>}
      {notice && <div style={{ fontSize: 12.5, color: "var(--ok)", background: "var(--ok-soft)", padding: "7px 11px", borderRadius: "var(--r-sm)" }}>{notice}</div>}

      {view === "graph" && (
        graph
          ? <MemoryGraph graph={graph} onSelect={openNode} />
          : <div style={{ opacity: 0.5, fontSize: 13, padding: 24 }}>{tr("Loading graph…")}</div>
      )}

      {view === "list" && (
      <div style={{ display: "flex", gap: 16, alignItems: "flex-start", flexWrap: "wrap" }}>
        <ul className="mem-hits" style={{ flex: "1 1 300px", minWidth: 0, listStyle: "none", padding: 0, margin: 0, display: "flex", flexDirection: "column", gap: 6 }}>
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

        {detailBusy && !selected && (
          <div style={{ flex: 1, opacity: 0.6, fontSize: 13, padding: 12 }}>{tr("Loading…")}</div>
        )}
        {selected && (
          <div className="mem-detail" style={{ flex: "1 1 300px", minWidth: 0, border: "1px solid var(--border)", borderRadius: "var(--r-sm)", padding: 12, background: "var(--surface)", opacity: detailBusy ? 0.55 : 1, transition: "opacity 0.15s" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div style={{ fontSize: 12, opacity: 0.6, flex: 1 }}>#{selected.id} · {selected.kind} · {tr("accessed")} {selected.access_count}×</div>
              <button className="btn btn-ghost" onClick={() => doForget(selected.id)} disabled={busy} title={tr("Forget this memory")} style={{ display: "flex", alignItems: "center", gap: 4, color: "var(--danger)" }}>
                <Trash2 size={13} /> {tr("Forget")}
              </button>
            </div>
            <p style={{ margin: "6px 0 10px" }}>{selected.content}</p>
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 12 }}>
              <span className="chip" title={tr("Importance (centrality) in [0,1]")} style={{ fontSize: 11, padding: "2px 8px", borderRadius: "var(--r-pill)", background: "var(--accent-soft)", color: "var(--accent-text)" }}>
                <Sparkles size={11} style={{ verticalAlign: "-1px" }} /> {tr("importance")} {selected.importance.toFixed(2)}
              </span>
              <span className="chip" title={tr("Reinforcement strength = 1 + ln(1 + accesses)")} style={{ fontSize: 11, padding: "2px 8px", borderRadius: "var(--r-pill)", background: "var(--surface-2)", color: "var(--text-muted)" }}>
                {tr("strength")} {selected.strength.toFixed(2)}
              </span>
              {selected.community_id !== null && (
                <span className="chip" title={tr("Community (cloud) id")} style={{ fontSize: 11, padding: "2px 8px", borderRadius: "var(--r-pill)", background: "var(--surface-2)", color: "var(--text-muted)" }}>
                  {tr("cloud")} #{selected.community_id}
                </span>
              )}
            </div>
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

      {showImport && (
        <div className="dialog-backdrop" role="presentation" onClick={(e) => { if (e.target === e.currentTarget) setShowImport(false); }}>
          <section className="info-dialog" role="dialog" aria-modal="true" aria-label={tr("Import Obsidian vault")} style={{ maxWidth: 520 }}>
            <div className="module-card-title" style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <Upload size={16} /> <h2 style={{ flex: 1, margin: 0 }}>{tr("Import Obsidian vault")}</h2>
              <button className="icon-only" aria-label={tr("Close")} onClick={() => setShowImport(false)}><X size={15} /></button>
            </div>
            <p style={{ fontSize: 13, color: "var(--text-muted)", margin: "8px 0" }}>
              {tr("Notes become memories, [[wikilinks]] become edges, #tags are kept. The path is confined to the server's vault root.")}
            </p>
            <input
              value={importPath}
              onChange={(e) => setImportPath(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") doImport(); }}
              placeholder={tr("subfolder, or empty for the whole vault")}
              autoFocus
              style={{ width: "100%", marginBottom: 12 }}
            />
            <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
              <button className="btn btn-ghost" onClick={() => setShowImport(false)}>{tr("Cancel")}</button>
              <button className="primary-action" onClick={doImport} disabled={busy} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <Upload size={13} /> {tr("Import")}
              </button>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
