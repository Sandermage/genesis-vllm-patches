# SPDX-License-Identifier: Apache-2.0
"""GUI ↔ backend contract gate.

The GUI's `api.ts` hand-writes the daemon route paths it calls. This test pins
those paths against the backend's actual routes so a renamed/removed backend
route (or a typo'd GUI path) fails CI instead of 404-ing at runtime — the
structural-compatibility guard for the two halves of the project.

The v11→v12 migration split routes across two factories — the legacy monolith
(the full Control Center) and the modular ``server`` (persistent memory + the
migrated routes). ``unified.create_app`` COMPOSES them into one superset daemon
(legacy + memory), so a single deployment serves every GUI path. This gate pins
the GUI's paths against that unified daemon — the app the full-Control-Center
deployment runs — so a renamed backend route or typo'd GUI path fails CI instead
of 404-ing. A second test pins the ``/api/v1/memory/*`` routes to the modular
``server`` factory (what the ``genesis-memory`` container runs), so memory drift
is caught at its real source too.

It extracts every ``/api/...`` literal from api.ts, reduces each to its static
prefix (dropping ``${…}`` path params and ``${query(…)}`` suffixes), and checks a
backend route has that prefix segment-wise (``{param}`` segments are wildcards).
Skips cleanly if the GUI source isn't present (e.g. a backend-only checkout)."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

# create_app() instantiates the FastAPI application, so the contract gate can
# only run where the gui-api extra is installed. On the dep-free release CI
# (which installs no fastapi) skip cleanly instead of erroring at create_app().
pytest.importorskip("fastapi")


def _api_ts() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        cand = parent / "gui" / "web" / "src" / "api.ts"
        if cand.is_file():
            return cand
    return None


def _gui_paths(text: str) -> set[str]:
    """Static path prefix of every /api/ literal (before the first ${ or ?).

    Matches the literal between an opening quote/backtick and the SAME closing
    delimiter (backreference), so it never runs past the string into trailing code."""
    out: set[str] = set()
    for m in re.finditer(r"""(["'`])(/api/(?:(?!\1).)*)\1""", text):
        raw = m.group(2)
        prefix = re.split(r"\$\{|\?", raw)[0].rstrip("/")
        if prefix and prefix != "/api":
            out.add(prefix)
    return out


def _seg_prefix_match(prefix: str, route: str) -> bool:
    ps = [s for s in prefix.strip("/").split("/") if s]
    rs = [s for s in route.strip("/").split("/") if s]
    if len(ps) > len(rs):
        return False
    return all(p == r or r.startswith("{") for p, r in zip(ps, rs, strict=False))


def _api_routes(create_app) -> list[str]:
    app = create_app()
    return [
        r.path for r in app.routes
        if isinstance(getattr(r, "path", None), str) and r.path.startswith("/api/")
    ]


def test_unified_daemon_serves_every_gui_api_path():
    api_ts = _api_ts()
    if api_ts is None:
        pytest.skip("gui/web/src/api.ts not present (backend-only checkout)")
    from sndr.product_api.unified import create_app as unified_create_app

    # The unified daemon composes legacy (full Control Center) + memory, so it
    # alone must serve every path the GUI calls.
    routes = _api_routes(unified_create_app)
    assert routes, "unified app exposed no /api/ routes"
    gui_paths = _gui_paths(api_ts.read_text(encoding="utf-8"))
    assert len(gui_paths) > 30, f"suspiciously few GUI paths parsed ({len(gui_paths)}) — parser drift?"
    missing = sorted(p for p in gui_paths if not any(_seg_prefix_match(p, r) for r in routes))
    assert not missing, (
        f"GUI api.ts calls {len(missing)} path(s) with no matching backend route "
        f"in the unified daemon (drift — add the route or fix the GUI):\n  " + "\n  ".join(missing))


def test_unified_daemon_keeps_the_spa_catch_all_last():
    """Mounting memory onto the composed app must not shadow the SPA, nor let the
    SPA "/" catch-all shadow the API routes — the SPA mount stays LAST."""
    from sndr.product_api.unified import create_app as unified_create_app

    routes = unified_create_app().router.routes
    api_after_spa = False
    seen_spa = False
    for r in routes:
        if getattr(r, "name", None) in ("ui", "carbon-ui"):
            seen_spa = True
        elif seen_spa and getattr(r, "path", "").startswith("/api/"):
            api_after_spa = True
    assert not api_after_spa, "an /api/ route is registered AFTER the SPA catch-all (it would be shadowed)"


def test_memory_routes_served_by_the_modular_daemon():
    """The Memory panel's routes must live in the modular ``server`` factory —
    the daemon the genesis-memory container runs. Pins memory drift to its real
    source (the union test above would otherwise mask a memory route moving)."""
    api_ts = _api_ts()
    if api_ts is None:
        pytest.skip("gui/web/src/api.ts not present (backend-only checkout)")
    from sndr.product_api.server import create_app as modular_create_app

    routes = _api_routes(modular_create_app)
    mem_gui_paths = sorted(
        p for p in _gui_paths(api_ts.read_text(encoding="utf-8"))
        if p.startswith("/api/v1/memory/") and not p.endswith("/fit")  # /fit is the VRAM estimator (legacy)
    )
    assert mem_gui_paths, "no /api/v1/memory/* paths parsed from the GUI — parser drift?"
    missing = sorted(p for p in mem_gui_paths if not any(_seg_prefix_match(p, r) for r in routes))
    assert not missing, (
        "GUI memory path(s) not served by the modular server daemon "
        "(the genesis-memory container would 404):\n  " + "\n  ".join(missing))
