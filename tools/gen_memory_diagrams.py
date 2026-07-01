# SPDX-License-Identifier: Apache-2.0
"""Generate the memory documentation SVGs (committed, reproducible).

Emits two static SVGs (GitHub renders them inline, scalable, no binary blobs):
  * docs/assets/memory-graph-clouds.svg — the neural-graph: nodes clustered into
    colored communities ("clouds") with intra/inter-cluster edges — what the GUI
    force-graph shows.
  * docs/assets/memory-decay-curve.svg — Ebbinghaus retention computed from the
    real formula R = exp(-age / (S·strength)), one curve per reinforcement level,
    visualizing the spacing effect (recall strengthens → slower decay).

Pure stdlib, deterministic (fixed seed). Run: python3 tools/gen_memory_diagrams.py
"""
from __future__ import annotations

import math
import random
from pathlib import Path

_OUT = Path(__file__).resolve().parent.parent / "docs" / "assets"
_BG = "#15171c"
_PALETTE = ["#4f9cf9", "#f97362", "#5fd07d", "#c77dff", "#f7b955", "#46c8c8"]
_FONT = "font-family='ui-sans-serif,Segoe UI,Helvetica,Arial,sans-serif'"


def _graph_svg() -> str:
    rng = random.Random(20260701)
    w, h = 900, 540
    # community centers + node counts (importance-varied sizes)
    clusters = [
        (190, 200, 11, "alpha"), (470, 150, 13, "beta"), (720, 250, 10, "gamma"),
        (300, 410, 9, "delta"), (620, 430, 8, "epsilon"),
    ]
    nodes: list[tuple[float, float, float, int]] = []  # x,y,r,community
    cluster_nodes: list[list[int]] = []
    for ci, (cx, cy, n, _name) in enumerate(clusters):
        idxs = []
        for _ in range(n):
            ang = rng.uniform(0, 2 * math.pi)
            rad = rng.uniform(8, 78)
            x = cx + rad * math.cos(ang)
            y = cy + rad * math.sin(ang)
            r = rng.choice([4, 4, 5, 5, 6, 7, 9])  # importance-varied
            idxs.append(len(nodes))
            nodes.append((x, y, r, ci))
        cluster_nodes.append(idxs)

    edges: list[tuple[int, int, bool]] = []  # a,b,intra
    for idxs in cluster_nodes:  # dense intra-cluster
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                if rng.random() < 0.28:
                    edges.append((idxs[i], idxs[j], True))
    for _ in range(7):  # a few inter-cluster bridges
        ca, cb = rng.sample(range(len(clusters)), 2)
        edges.append((rng.choice(cluster_nodes[ca]), rng.choice(cluster_nodes[cb]), False))

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' "
        f"width='{w}' height='{h}' role='img' aria-label='Memory neural graph'>",
        f"<rect width='{w}' height='{h}' rx='14' fill='{_BG}'/>",
        f"<text x='28' y='40' fill='#e8eaed' font-size='22' font-weight='700' {_FONT}>"
        "Neural-graph memory — knowledge clusters into clouds</text>",
        f"<text x='28' y='62' fill='#9aa0a6' font-size='13' {_FONT}>"
        "nodes = memories (size = importance) · edges = similar_to / co_access "
        "(width = strength) · colors = communities</text>",
    ]
    for a, b, intra in edges:  # edges under nodes
        x1, y1, _r1, _c1 = nodes[a]
        x2, y2, _r2, _c2 = nodes[b]
        col = "#3a3f47" if intra else "#5b6470"
        wdt = 1.1 if intra else 1.8
        dash = "" if intra else " stroke-dasharray='4 3'"
        parts.append(
            f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' "
            f"stroke='{col}' stroke-width='{wdt}' stroke-opacity='0.55'{dash}/>"
        )
    for x, y, r, c in nodes:
        col = _PALETTE[c % len(_PALETTE)]
        parts.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{r}' fill='{col}' "
            f"stroke='#0d0f12' stroke-width='1.2'/>"
        )
    # legend
    lx, ly = 28, h - 28
    for i, (_cx, _cy, _n, name) in enumerate(clusters):
        cx = lx + i * 150
        parts.append(f"<circle cx='{cx}' cy='{ly}' r='6' fill='{_PALETTE[i % len(_PALETTE)]}'/>")
        parts.append(
            f"<text x='{cx + 12}' y='{ly + 4}' fill='#c7ccd1' font-size='12' {_FONT}>"
            f"cloud {name}</text>"
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _decay_svg() -> str:
    w, h = 760, 440
    m = {"l": 64, "r": 24, "t": 64, "b": 56}
    pw, ph = w - m["l"] - m["r"], h - m["t"] - m["b"]
    days_max = 14.0
    s_day = 1.0  # EBBINGHAUS_S = 1 day, in these units

    def px(t: float) -> float:
        return m["l"] + (t / days_max) * pw

    def py(r: float) -> float:
        return m["t"] + (1 - r) * ph

    curves = [  # (strength, recalls-label, color)
        (1.0, "never recalled (strength 1.0)", _PALETTE[1]),
        (1.0 + math.log1p(1), "recalled 1× (1.69)", _PALETTE[0]),
        (1.0 + math.log1p(5), "recalled 5× (2.79)", _PALETTE[2]),
    ]
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' "
        f"width='{w}' height='{h}' role='img' aria-label='Ebbinghaus retention'>",
        f"<rect width='{w}' height='{h}' rx='14' fill='{_BG}'/>",
        f"<text x='28' y='38' fill='#e8eaed' font-size='20' font-weight='700' {_FONT}>"
        "Ebbinghaus retention — retrieval slows decay (spacing effect)</text>",
    ]
    # axes + gridlines
    for gx in range(0, 15, 2):
        x = px(gx)
        parts.append(f"<line x1='{x:.1f}' y1='{m['t']}' x2='{x:.1f}' y2='{m['t'] + ph}' stroke='#262a31' stroke-width='1'/>")
        parts.append(f"<text x='{x:.1f}' y='{m['t'] + ph + 20}' fill='#9aa0a6' font-size='11' text-anchor='middle' {_FONT}>{gx}d</text>")
    for gy in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = py(gy)
        parts.append(f"<line x1='{m['l']}' y1='{y:.1f}' x2='{m['l'] + pw}' y2='{y:.1f}' stroke='#262a31' stroke-width='1'/>")
        parts.append(f"<text x='{m['l'] - 10}' y='{y + 4:.1f}' fill='#9aa0a6' font-size='11' text-anchor='end' {_FONT}>{gy:.2f}</text>")
    parts.append(f"<text x='{m['l'] + pw / 2:.0f}' y='{h - 12}' fill='#c7ccd1' font-size='12' text-anchor='middle' {_FONT}>time since last access</text>")
    for si, (strength, label, color) in enumerate(curves):
        pts = []
        t = 0.0
        while t <= days_max + 1e-9:
            r = math.exp(-t / (s_day * strength))
            pts.append(f"{px(t):.1f},{py(r):.1f}")
            t += 0.25
        parts.append(f"<polyline points='{' '.join(pts)}' fill='none' stroke='{color}' stroke-width='2.4'/>")
        ly = m["t"] + 16 + si * 20
        parts.append(f"<line x1='{m['l'] + pw - 220}' y1='{ly}' x2='{m['l'] + pw - 198}' y2='{ly}' stroke='{color}' stroke-width='3'/>")
        parts.append(f"<text x='{m['l'] + pw - 192}' y='{ly + 4}' fill='#c7ccd1' font-size='12' {_FONT}>{label}</text>")
    parts.append("</svg>")
    return "\n".join(parts)


# ── flowchart primitives (static SVG — GitHub renders these instantly, unlike
#    client-rendered Mermaid, which spins/fails on larger graphs) ────────────
_FG = "#e8eaed"
_MUTED = "#9aa0a6"
_STROKE = "#3a3f47"
_LINE = "#5b6470"
_PANEL = "#1c1f27"


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _svg_open(w: int, h: int, title: str) -> str:
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' "
        f"width='{w}' height='{h}' role='img' aria-label='{_esc(title)}'>"
        "<defs>"
        f"<marker id='ah' viewBox='0 0 10 10' refX='8.5' refY='5' markerWidth='7' "
        f"markerHeight='7' orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' fill='{_LINE}'/></marker>"
        "<marker id='ahb' viewBox='0 0 10 10' refX='8.5' refY='5' markerWidth='7' "
        "markerHeight='7' orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' fill='#4f9cf9'/></marker>"
        "</defs>"
        f"<rect width='{w}' height='{h}' rx='14' fill='{_BG}'/>"
    )


def _title(x: float, y: float, text: str) -> str:
    return (f"<text x='{x}' y='{y}' fill='{_FG}' font-size='21' font-weight='700' "
            f"{_FONT}>{_esc(text)}</text>")


def _subtitle(x: float, y: float, text: str) -> str:
    return (f"<text x='{x}' y='{y}' fill='{_MUTED}' font-size='12.5' {_FONT}>{_esc(text)}</text>")


def _label(x: float, y: float, text: str, *, color: str = _MUTED, fs: float = 11,
           anchor: str = "middle") -> str:
    return (f"<text x='{x:.1f}' y='{y:.1f}' fill='{color}' font-size='{fs}' "
            f"text-anchor='{anchor}' {_FONT}>{_esc(text)}</text>")


def _box(x: float, y: float, w: float, h: float, lines: list[str], *,
         accent: str | None = None, fill: str = _PANEL, fs: float = 13, rx: float = 9) -> str:
    cx = x + w / 2
    n = len(lines)
    lh = fs + 4
    top = y + h / 2 - (n - 1) * lh / 2 + fs * 0.34
    stroke = accent or _STROKE
    sw = 1.7 if accent else 1.3
    parts = [f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='{rx}' fill='{fill}' "
             f"stroke='{stroke}' stroke-width='{sw}'/>"]
    for i, ln in enumerate(lines):
        weight = "700" if i == 0 else "400"
        color = _FG if i == 0 else _MUTED
        size = fs if i == 0 else fs - 2
        parts.append(
            f"<text x='{cx:.1f}' y='{top + i * lh:.1f}' fill='{color}' font-size='{size}' "
            f"font-weight='{weight}' text-anchor='middle' {_FONT}>{_esc(ln)}</text>"
        )
    return "".join(parts)


def _edge(x1: float, y1: float, x2: float, y2: float, *, color: str = _LINE,
          dashed: bool = False, width: float = 1.6, marker: str = "ah") -> str:
    dash = " stroke-dasharray='5 4'" if dashed else ""
    return (f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' "
            f"stroke='{color}' stroke-width='{width}'{dash} marker-end='url(#{marker})'/>")


def _architecture_svg() -> str:
    w, h = 960, 470
    p = [_svg_open(w, h, "genesis-memory architecture")]
    p.append(_title(28, 42, "Architecture — one CPU container, memory for any model"))
    p.append(_subtitle(28, 64, "a client / model → the gateway recalls + injects memory, forwards "
                               "upstream, captures the reply · REST + GUI + engine + Postgres in one image"))
    # external request path
    p.append(_box(28, 94, 120, 52, ["client / app"]))
    p.append(_box(250, 88, 220, 64, ["memory gateway", "/v1/chat/completions"], accent="#4f9cf9"))
    p.append(_box(556, 82, 380, 40, ["CLIProxyAPI → Claude · Gemini · GPT · …"]))
    p.append(_box(556, 128, 380, 40, ["vLLM engine (the 35B)"]))
    p.append(_edge(148, 120, 250, 120, color="#4f9cf9", marker="ahb"))
    p.append(_label(199, 112, "X-Memory-Upstream"))
    p.append(_edge(470, 110, 556, 102, color=_LINE))
    p.append(_edge(470, 134, 556, 148, color=_LINE))
    p.append(_label(513, 74, "forward"))
    # container panel
    p.append("<rect x='28' y='182' width='908' height='262' rx='14' fill='#191c22' "
             "stroke='#2b303a' stroke-width='1.4'/>")
    p.append(_label(46, 205, "genesis-memory container · CPU · one image · :8811",
                    color=_MUTED, fs=12, anchor="start"))
    p.append(_box(96, 224, 210, 46, ["/api/v1/memory/* · REST"]))
    p.append(_box(660, 224, 232, 46, ["GUI graph panel · Sigma.js"]))
    p.append(_box(392, 312, 176, 58, ["MemoryEngine"], accent="#c77dff"))
    p.append(_box(636, 312, 256, 58, ["maintenance loop", "consolidate + prune"]))
    p.append(_box(96, 388, 210, 44, ["Embedder · Model2Vec / Hash"]))
    p.append(_box(360, 388, 244, 44, ["Postgres + pgvector · nodes · edges"], accent="#5fd07d"))
    # gateway drives the engine (recall/inject/capture), and the engine fans out
    p.append(_edge(360, 152, 468, 312, color="#4f9cf9", marker="ahb"))
    p.append(_label(360, 250, "recall + inject", color="#7bb0f7"))
    p.append(_label(360, 264, "capture", color="#7bb0f7"))
    p.append(_edge(201, 270, 432, 312))
    p.append(_edge(776, 270, 520, 312, dashed=True))
    p.append(_label(662, 292, "same-origin"))
    p.append(_edge(636, 341, 572, 341))
    p.append(_edge(430, 370, 205, 388))
    p.append(_edge(480, 370, 482, 388))
    p.append("</svg>")
    return "\n".join(p)


def _brain_svg() -> str:
    w, h = 980, 600
    p = [_svg_open(w, h, "memory brain mechanics")]
    p.append(_title(28, 42, "Brain mechanics — deterministic, no LLM on the write path"))
    lanes = [(20, 292, "write", "#4f9cf9"), (332, 316, "recall", "#c77dff"),
             (668, 292, "consolidate · nightly", "#5fd07d")]
    for lx, lw, name, col in lanes:
        p.append(f"<rect x='{lx}' y='62' width='{lw}' height='520' rx='12' fill='#181b21' "
                 f"stroke='#262b34' stroke-width='1.2'/>")
        p.append(f"<text x='{lx + 16}' y='90' fill='{col}' font-size='13' font-weight='700' "
                 f"{_FONT}>{_esc(name)}</text>")
    # write lane
    p.append(_box(46, 112, 240, 44, ["remember(text)"], accent="#4f9cf9"))
    p.append(_edge(166, 156, 166, 190))
    p.append(_box(46, 190, 240, 44, ["node + embedding"]))
    p.append(_edge(166, 234, 166, 268))
    p.append(_box(46, 268, 240, 54, ["stored", "owner-scoped · vector + text"]))
    p.append(_label(166, 356, "dedup: same text → same node", color=_MUTED, fs=11))
    # recall lane (a vertical chain, then a fan-out)
    rx = 352
    p.append(_box(rx, 112, 276, 42, ["recall(query)"], accent="#c77dff"))
    p.append(_edge(490, 154, 490, 172))
    p.append(_box(rx, 172, 276, 38, ["vector ANN seeds"]))
    p.append(_edge(490, 210, 490, 226))
    p.append(_box(rx, 226, 276, 54, ["spreading activation", "act × weight × β · ≤3 hops · cycle-safe"]))
    p.append(_edge(490, 280, 490, 296))
    p.append(_box(rx, 296, 276, 54, ["× Ebbinghaus retention", "exp(−age / (S · strength · (1+importance)))"]))
    p.append(_edge(490, 350, 490, 366))
    p.append(_box(rx, 366, 276, 38, ["top-N results"]))
    p.append(_edge(490, 404, 420, 424))
    p.append(_edge(490, 404, 560, 424))
    p.append(_box(352, 424, 136, 58, ["touch", "strength ↑ (spacing)"]))
    p.append(_box(492, 424, 136, 58, ["Hebbian wire", "w ← min(1,(1−λ)w+η)"]))
    # consolidate lane (vertical chain)
    cx = 688
    p.append(_box(cx, 112, 252, 44, ["consolidate"], accent="#5fd07d"))
    p.append(_edge(814, 156, 814, 176))
    p.append(_box(cx, 176, 252, 40, ["kNN → similar_to edges"]))
    p.append(_edge(814, 216, 814, 232))
    p.append(_box(cx, 232, 252, 50, ["communities → clouds", "label propagation"]))
    p.append(_edge(814, 282, 814, 298))
    p.append(_box(cx, 298, 252, 40, ["importance = f(degree, access)"]))
    p.append(_edge(814, 338, 814, 354))
    p.append(_box(cx, 354, 252, 40, ["prune to cap · leak-bound"]))
    p.append("</svg>")
    return "\n".join(p)


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    figures = {
        "memory-graph-clouds.svg": _graph_svg,
        "memory-decay-curve.svg": _decay_svg,
        "memory-architecture.svg": _architecture_svg,
        "memory-brain-mechanics.svg": _brain_svg,
    }
    for name, fn in figures.items():
        (_OUT / name).write_text(fn(), encoding="utf-8")
        print("wrote", _OUT / name)


if __name__ == "__main__":
    main()
