# SPDX-License-Identifier: Apache-2.0
"""Pure, textual-free render helpers for the ``sndr tui`` cockpit.

These functions turn the read-only payloads from :mod:`sndr.cli.tui.data` (an
engine snapshot, a resolved rig, a fit-ranked catalog) into the small strings
and row tuples the Textual app paints. They import NO textual — so they unit-test
without the optional ``[tui]`` extra installed, and the app (:mod:`sndr.cli.tui.app`)
imports them rather than re-deriving the same wording.

Every helper is defensive in the same spirit as the data facade: a down engine,
a GPU-less box, or a partial candidate yields calm placeholder text, never a
crash. The helpers stay pure (render-only); all side effects live in the app.
"""
from __future__ import annotations

from typing import Any

# Glyphs — identical to the rest of the CLI (sndr/cli/_messages.py) so the whole
# surface renders one symbol set.
GLYPH_FIT = "✓"
GLYPH_NOFIT = "✗"

HELP_TEXT = """\
sndr cockpit — keys

  ↑/↓      move the catalog cursor
  Enter    serve the selected preset   (pull if needed + launch)
  k        stop the selected preset's engine
  d        doctor — full system diagnostic
  c        chat with the running engine
  r        refresh engine + catalog
  /        filter the catalog
  ?        this help
  q        quit

serve / stop ask to confirm first; doctor / chat drop to the terminal and
return to the cockpit on exit.
The catalog shows ✓ for presets that fit this rig, ✗ for those that don't.
No engine running yet?  select a ✓ row and press Enter — or exit and run: sndr run
"""


def render_engine(snap: dict[str, Any]) -> str:
    """Render an engine_snapshot into the ENGINE pane text.

    A down engine renders a calm "no engine" hint, never an error dump.
    """
    status = snap.get("status") or {}
    metrics = snap.get("metrics") or {}
    if not status.get("reachable"):
        return (
            "[b]engine[/b]  ✗ down\n"
            "  no engine reachable yet.\n"
            "  exit and run:  [b]sndr run[/b]"
        )
    kpis = metrics.get("kpis") or {}
    models = status.get("models") or []
    model = models[0] if models else "—"
    version = status.get("version") or "?"

    def _kpi(key: str, fmt: str = "{}") -> str:
        val = kpis.get(key)
        return fmt.format(val) if val is not None else "—"

    return (
        f"[b]engine[/b]  ● serving   v{version}\n"
        f"  model   {model}\n"
        f"  tok/s   {_kpi('generation_toks_per_s', '{:.0f}')}"
        f"   kv {_kpi('kv_cache_usage', '{:.0%}')}\n"
        f"  TTFT    {_kpi('ttft_avg_s', '{:.2f}s')}"
        f"   run {_kpi('requests_running')}  wait {_kpi('requests_waiting')}"
    )


def render_gpu(rig: Any | None) -> str:
    """Render the resolved rig into the GPU/RIG pane text."""
    if rig is None:
        return "[b]rig[/b]\n  detecting…"
    gpus = getattr(rig, "gpu_count", 0) or 0
    if gpus == 0:
        return (
            "[b]rig[/b]\n"
            "  ✗ no GPU detected\n"
            "  plan against a card with:  --fake-gpus 'RTX A5000:24564:8.6'"
        )
    vram = getattr(rig, "min_vram_gb", None)
    cap = getattr(rig, "min_compute_cap", None)
    cap_s = f"sm_{cap[0]}.{cap[1]}" if cap else "?"
    vram_s = f"{vram} GB" if vram else "? GB"
    return (
        f"[b]rig[/b]  {getattr(rig, 'source', 'rig')}\n"
        f"  {gpus}× GPU   {vram_s}/card   {cap_s}"
    )


def render_memory(snap: dict[str, Any] | None) -> str:
    """Render a memory_snapshot into the MEMORY pane text.

    A down/absent daemon renders a calm hint (never a raw error dump); a
    reachable daemon shows the owner's node/edge counts. Defensive: a ``None``
    or partial payload yields placeholder text, never a crash.
    """
    if not snap or not snap.get("reachable"):
        return (
            "[b]memory[/b]  ✗ offline\n"
            "  no memory daemon reachable.\n"
            "  start it:  [b]sndr up[/b]"
        )
    stats = snap.get("stats") or {}
    nodes = stats.get("nodes", 0)
    edges = stats.get("edges", 0)
    return f"[b]memory[/b]  ● {nodes} nodes  {edges} edges"


def catalog_rows(catalog: Any) -> list[tuple[str, str, str, str]]:
    """Project a wizard ``Catalog`` into the DataTable rows the cockpit paints.

    Each row is ``(fit_glyph, preset_id, status, metric)`` — ``✓`` when the
    preset can run on the rig, ``✗`` when it can't. Reuses the catalog's own
    ``menu(show_all=True)`` selector (falling back to ``candidates``) so the TUI
    shows the exact rows the wizard ranks, in the same order. A ``None`` catalog
    or a partial candidate yields an empty list / placeholder cells rather than
    raising — the app must paint on any box.
    """
    if catalog is None:
        return []
    rows = (
        catalog.menu(show_all=True)
        if hasattr(catalog, "menu")
        else list(getattr(catalog, "candidates", []))
    )
    out: list[tuple[str, str, str, str]] = []
    for c in rows:
        can = bool(getattr(c, "can_run", False))
        out.append(
            (
                GLYPH_FIT if can else GLYPH_NOFIT,
                getattr(c, "preset_id", "?"),
                getattr(c, "status", "?"),
                getattr(c, "metric_label", None) or "—",
            )
        )
    return out


__all__ = [
    "render_engine",
    "render_gpu",
    "render_memory",
    "catalog_rows",
    "HELP_TEXT",
    "GLYPH_FIT",
    "GLYPH_NOFIT",
]
