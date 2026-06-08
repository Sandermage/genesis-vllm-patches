# SPDX-License-Identifier: Apache-2.0
"""Production-subset resolution for proof-attached bench gating.

The full PATCH_REGISTRY ships ~169 entries spanning every lifecycle
from ``stable`` down to ``research``. Requiring per-patch bench proof
for every one of them is intractable — most experimental opt-in
patches are never enabled in shipped presets and don't carry a real
production cost.

The production subset is the practical scope for hardened-release
bench gating: the union of

  1. every patch enabled (``GENESIS_ENABLE_*=1``) by any V2 preset
     whose alias matches ``PRODUCTION_PRESET_PATTERN`` (default
     ``prod-*``);
  2. every patch flagged ``default_on=True`` in the registry — those
     ship enabled even without an explicit preset opt-in.

The two sources together give the actual production-eligible code path
for an operator who runs ``sndr launch <prod-preset>`` with default
flags. Hardened-release policy attaches bench proof to this subset
only; everything outside stays under the cheaper ``require-static``
gate.

Author: Sandermage (Sander) Barzov Aleksandr.
"""
from __future__ import annotations

import fnmatch
import logging
from typing import Optional

log = logging.getLogger("genesis.proof.production_subset")


# Globs over V2 preset alias names. Any alias matching one of these is
# treated as a production-eligible preset; its enabled patches feed
# into the subset.
PRODUCTION_PRESET_PATTERN: tuple[str, ...] = (
    "prod-*",
)


def _enabled_env_keys_for_preset(alias: str) -> set[str]:
    """Resolve one V2 preset alias and return the set of
    ``GENESIS_ENABLE_*`` env keys it activates (value=='1'/'true').

    Returns an empty set if the alias cannot be resolved — production
    subset computation is best-effort, never raises.
    """
    try:
        from sndr.model_configs import registry_v2
    except ImportError:
        return set()
    try:
        composed = registry_v2.load_alias(alias)
    except Exception as e:  # noqa: BLE001 — best-effort
        log.debug("preset %s did not resolve: %s", alias, e)
        return set()
    env = composed.genesis_env or {}
    return {
        k for k, v in env.items()
        if isinstance(k, str) and k.startswith("GENESIS_ENABLE_")
        and str(v) in ("1", "true", "True")
    }


def _list_production_preset_aliases() -> list[str]:
    """Return every V2 preset alias whose name matches the production
    glob list. Deterministic order (sorted)."""
    try:
        from sndr.model_configs import registry_v2
    except ImportError:
        return []
    try:
        alias_dir = registry_v2._alias_dir()  # noqa: SLF001
    except Exception:  # noqa: BLE001
        return []
    if not alias_dir.is_dir():
        return []
    all_aliases = sorted(p.stem for p in alias_dir.glob("*.yaml"))
    out: list[str] = []
    for alias in all_aliases:
        for pattern in PRODUCTION_PRESET_PATTERN:
            if fnmatch.fnmatch(alias, pattern):
                out.append(alias)
                break
    return out


def get_production_subset(
    registry: Optional[dict] = None,
) -> frozenset[str]:
    """Return the patch_id frozenset that hardened-release gating
    treats as production-eligible.

    Composition:
      * Union of all ``GENESIS_ENABLE_*`` env keys flipped on by any
        preset matching :data:`PRODUCTION_PRESET_PATTERN`, mapped back
        to patch_ids via the registry's ``env_flag`` field.
      * Plus every registry entry where ``default_on=True``.

    The result is computed at call time and is cheap (~1 ms on
    169 entries) — no caching, callers are free to memoize.
    """
    if registry is None:
        from sndr.dispatcher.registry import PATCH_REGISTRY
        registry = PATCH_REGISTRY

    env_to_pid: dict[str, str] = {}
    default_on: set[str] = set()
    for pid, meta in registry.items():
        if not isinstance(meta, dict):
            continue
        env_flag = meta.get("env_flag")
        if isinstance(env_flag, str) and env_flag:
            env_to_pid[env_flag] = pid
        if meta.get("default_on") is True:
            default_on.add(pid)

    enabled_env_keys: set[str] = set()
    for alias in _list_production_preset_aliases():
        enabled_env_keys |= _enabled_env_keys_for_preset(alias)

    subset = set(default_on)
    for env_key in enabled_env_keys:
        pid = env_to_pid.get(env_key)
        if pid:
            subset.add(pid)
    return frozenset(subset)


def production_subset_breakdown(
    registry: Optional[dict] = None,
) -> dict:
    """Diagnostic helper — return the subset as a structured dict so
    operators can inspect the source of each inclusion."""
    if registry is None:
        from sndr.dispatcher.registry import PATCH_REGISTRY
        registry = PATCH_REGISTRY

    aliases = _list_production_preset_aliases()
    per_preset: dict[str, list[str]] = {}
    env_to_pid: dict[str, str] = {
        meta["env_flag"]: pid
        for pid, meta in registry.items()
        if isinstance(meta, dict) and isinstance(meta.get("env_flag"), str)
    }

    for alias in aliases:
        keys = _enabled_env_keys_for_preset(alias)
        per_preset[alias] = sorted(env_to_pid[k] for k in keys if k in env_to_pid)

    default_on = sorted(
        pid for pid, meta in registry.items()
        if isinstance(meta, dict) and meta.get("default_on") is True
    )
    subset = get_production_subset(registry)
    return {
        "presets_matched": aliases,
        "per_preset": per_preset,
        "default_on": default_on,
        "subset": sorted(subset),
        "subset_size": len(subset),
        "total_registry": len(registry),
    }


__all__ = [
    "PRODUCTION_PRESET_PATTERN",
    "get_production_subset",
    "production_subset_breakdown",
]
