#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the Open Graph / social-preview card (assets/og-card.png).

Run from repo root:  python3 assets/charts/_og_card.py

Produces a 1280x640 PNG used as the GitHub social preview (Settings ->
General -> Social preview) so repo links unfurl with a branded card on
Reddit / Hacker News / X / Slack. The headline figures come from the
reference 2x RTX A5000 sweep documented in docs/BENCHMARKS.md; keep them in
sync with the README headline table.

Dark, colorblind-safe palette matching assets/charts/_generate.py.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)

BG = "#0d1117"
FG = "#c9d1d9"
MUTED = "#8b949e"
ACCENT = "#3fb950"  # genesis green
BLUE = "#58a6ff"

# (value, label) headline stats — keep in sync with the README table.
STATS = [
    ("239.7", "tok/s · 35B FP8"),
    ("+53%", "vs stock vLLM"),
    ("256K", "context, verified"),
]


def main() -> str:
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)  # -> 1280x640 px
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title + tagline
    ax.text(0.06, 0.80, "SNDR Core Engine", color=FG, fontsize=58,
            fontweight="bold", va="center", family="sans-serif")
    ax.text(0.06, 0.665, "Genesis vLLM Patches", color=ACCENT, fontsize=30,
            fontweight="bold", va="center", family="sans-serif")
    ax.text(0.06, 0.575,
            "Qwen3.6 + Gemma4 on consumer NVIDIA — "
            "TurboQuant KV · MTP · hybrid GDN",
            color=MUTED, fontsize=17, va="center", family="sans-serif")

    # Stat blocks
    xs = [0.085, 0.395, 0.705]
    for x, (val, lab) in zip(xs, STATS):
        ax.text(x, 0.34, val, color=BLUE, fontsize=46, fontweight="bold",
                va="center", ha="left", family="sans-serif")
        ax.text(x, 0.215, lab, color=FG, fontsize=16, va="center", ha="left",
                family="sans-serif")

    # Divider + footer
    ax.plot([0.06, 0.94], [0.46, 0.46], color="#21262d", lw=2)
    ax.text(0.06, 0.09, "github.com/Sandermage/sndr_core_engine",
            color=MUTED, fontsize=18, va="center", family="monospace")
    ax.text(0.94, 0.09, "Apache-2.0 · 2x RTX A5000 ref rig",
            color=MUTED, fontsize=14, va="center", ha="right",
            family="sans-serif")

    out = os.path.abspath(os.path.join(OUT, "og-card.png"))
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
