// SPDX-License-Identifier: Apache-2.0
// Obsidian-like force-directed graph of the memory: nodes auto-form connections
// and cluster into colored "clouds" (Leiden community_id). Uses graphology +
// ForceAtlas2 to compute the layout (pure JS, no WebGL needed at this scale),
// rendered as SVG so it stays fully typed/testable. For very large graphs,
// swap the SVG renderer for Sigma.js (same graphology model).
import { useMemo, useState } from "react";
import Graph from "graphology";
import forceAtlas2 from "graphology-layout-forceatlas2";
import type { MemGraph } from "./api";

// Stable palette for community "clouds"; null community -> neutral.
const PALETTE = [
  "#4f9cf9", "#f97362", "#5fd07d", "#c77dff", "#f7b955",
  "#46c8c8", "#e86fb3", "#9aa0ff", "#8bd450", "#ff8fa3",
];
const NEUTRAL = "#8a8f98";

function colorFor(community: number | null): string {
  if (community == null) return NEUTRAL;
  const idx = ((community % PALETTE.length) + PALETTE.length) % PALETTE.length;
  return PALETTE[idx] ?? NEUTRAL;
}

type Pos = { x: number; y: number };

export function MemoryGraph({ graph, onSelect }: { graph: MemGraph; onSelect: (id: number) => void }) {
  const [hover, setHover] = useState<number | null>(null);

  const layout = useMemo(() => {
    const g = new Graph();
    const n = graph.nodes.length;
    graph.nodes.forEach((node, i) => {
      // Seed on a circle so ForceAtlas2 starts from distinct positions.
      const a = (2 * Math.PI * i) / Math.max(1, n);
      g.addNode(String(node.id), { x: Math.cos(a), y: Math.sin(a), size: node.id });
    });
    for (const e of graph.edges) {
      const s = String(e.src), d = String(e.dst);
      if (g.hasNode(s) && g.hasNode(d) && !g.hasEdge(s, d)) {
        g.addEdge(s, d, { weight: e.weight });
      }
    }
    if (n > 1) {
      forceAtlas2.assign(g, { iterations: 300, settings: forceAtlas2.inferSettings(g) });
    }
    const pos = new Map<number, Pos>();
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    graph.nodes.forEach((node) => {
      const x = g.getNodeAttribute(String(node.id), "x") as number;
      const y = g.getNodeAttribute(String(node.id), "y") as number;
      pos.set(node.id, { x, y });
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
    });
    return { pos, minX, minY, maxX, maxY };
  }, [graph]);

  if (!graph.nodes.length) {
    return <div style={{ opacity: 0.5, fontSize: 13, padding: 24 }}>No nodes yet — remember something, then Rebuild links.</div>;
  }

  const PAD = 40;
  const w = Math.max(1, layout.maxX - layout.minX);
  const h = Math.max(1, layout.maxY - layout.minY);
  const VW = 800, VH = 520;
  const sx = (VW - 2 * PAD) / w;
  const sy = (VH - 2 * PAD) / h;
  const sc = Math.min(sx, sy);
  const px = (x: number) => PAD + (x - layout.minX) * sc;
  const py = (y: number) => PAD + (y - layout.minY) * sc;
  const radius = (imp: number, acc: number) => 5 + Math.min(10, imp * 4 + Math.log1p(acc) * 1.5);

  return (
    <svg viewBox={`0 0 ${VW} ${VH}`} style={{ width: "100%", height: 520, border: "1px solid var(--border, #2a2a2a)", borderRadius: 8, background: "var(--panel, #16181d)" }}>
      {graph.edges.map((e, i) => {
        const a = layout.pos.get(e.src), b = layout.pos.get(e.dst);
        if (!a || !b) return null;
        const active = hover === e.src || hover === e.dst;
        return (
          <line
            key={i}
            x1={px(a.x)} y1={py(a.y)} x2={px(b.x)} y2={py(b.y)}
            stroke={active ? "#aab" : "#3a3f47"}
            strokeWidth={Math.max(0.5, e.weight * 2)}
            strokeOpacity={active ? 0.9 : 0.45}
          />
        );
      })}
      {graph.nodes.map((node) => {
        const p = layout.pos.get(node.id);
        if (!p) return null;
        const r = radius(node.importance, node.access_count);
        return (
          <g key={node.id} transform={`translate(${px(p.x)},${py(p.y)})`} style={{ cursor: "pointer" }}
             onMouseEnter={() => setHover(node.id)} onMouseLeave={() => setHover(null)}
             onClick={() => onSelect(node.id)}>
            <circle r={r} fill={colorFor(node.community_id)} stroke={hover === node.id ? "#fff" : "rgba(0,0,0,0.3)"} strokeWidth={hover === node.id ? 2 : 1} />
            <title>{node.content}</title>
            {hover === node.id && (
              <text x={r + 4} y={4} fontSize={12} fill="#e8eaed" style={{ pointerEvents: "none" }}>
                {node.content.length > 48 ? node.content.slice(0, 47) + "…" : node.content}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}
