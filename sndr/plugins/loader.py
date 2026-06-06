# SPDX-License-Identifier: Apache-2.0
"""Entry-point-based plugin loader for sndr-platform.

External wheels publish patches via setuptools entry points and are loaded
here at boot. The community ``sndr`` wheel never imports these wheels
directly — discovery is via ``importlib.metadata.entry_points``.

Entry point group convention::

    sndr.engines.<engine>.patches
        Each entry returns a PatchSpec or a function that returns one.

Example ``pyproject.toml`` for a hypothetical ``sndr-engine`` wheel::

    [project.entry-points."sndr.engines.vllm.patches"]
    p67 = "sndr_engine.vllm.patches.p67:patch"
    pn21 = "sndr_engine.vllm.patches.pn21:patch"

Engineering principles applied:
  - Explicit over implicit: only registered entry points load
  - Fail safe: a broken plugin logs a warning, never aborts boot
  - Observable: every plugin discovery emits a structured log
"""
from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import Any

log = logging.getLogger("sndr.plugins.loader")


def discover_engine_patches(engine: str) -> list[Any]:
    """Discover and load patches registered by external wheels for one engine.

    Args:
        engine: Engine identifier ("vllm" or "sglang").

    Returns:
        List of patch objects (PatchSpec or whatever the entry point returns).
        Plugins that fail to load are skipped with a warning log; this function
        NEVER raises.
    """
    group = f"sndr.engines.{engine}.patches"
    loaded: list[Any] = []

    try:
        eps = entry_points(group=group)
    except Exception as e:  # noqa: BLE001
        log.warning(
            "plugins.discovery.failed",
            extra={"group": group, "error": str(e)},
        )
        return []

    for ep in eps:
        try:
            obj = ep.load()
            patch = obj() if callable(obj) and not hasattr(obj, "id") else obj
            loaded.append(patch)
            log.info(
                "plugins.patch.loaded",
                extra={
                    "group": group,
                    "name": ep.name,
                    "module": ep.value,
                },
            )
        except Exception as e:  # noqa: BLE001
            log.warning(
                "plugins.patch.load_failed",
                extra={
                    "group": group,
                    "name": ep.name,
                    "module": ep.value,
                    "error": str(e),
                },
                exc_info=True,
            )

    log.info(
        "plugins.discovery.complete",
        extra={"engine": engine, "loaded_count": len(loaded)},
    )
    return loaded


def get_plugin_info() -> dict[str, list[dict[str, str]]]:
    """Return a snapshot of every entry point in the sndr.* groups.

    Useful for the ``sndr engines info`` CLI command and the ``/api/v1/plugins``
    REST endpoint to surface which external wheels are installed.
    """
    info: dict[str, list[dict[str, str]]] = {}
    for engine in ("vllm", "sglang"):
        group = f"sndr.engines.{engine}.patches"
        try:
            eps = entry_points(group=group)
        except Exception:  # noqa: BLE001
            eps = []
        info[engine] = [
            {"name": ep.name, "module": ep.value}
            for ep in eps
        ]
    return info


__all__ = ["discover_engine_patches", "get_plugin_info"]
