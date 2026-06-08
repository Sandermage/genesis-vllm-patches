# SPDX-License-Identifier: Apache-2.0
"""Phase 5 community SDK — discovery layer.

Two complementary discovery paths:

  • filesystem — walk `plugins/community/**/manifest.yaml` (the repo-local
    tree where contributors land patches as part of the source layout).
  • entry-points — read `vllm.community_patches` setuptools entry-point
    group. Each entry should resolve to a `PatchManifest` instance, a
    `dict` (will be dataclass-materialized), or a `Path` pointing at a
    manifest YAML on disk.

`discover_all()` merges both, deduplicating by `(namespace, id)` pair —
filesystem wins on conflict (lets operators override an installed package
patch with a local clone in `plugins/community/`).
"""
from __future__ import annotations

import logging
from pathlib import Path

from sndr.model_configs.schema import SchemaError
from sndr.model_configs.schema_v2 import PatchManifest

from .manifest import (
    DEFAULT_PLUGINS_DIR,
    list_manifest_paths,
    load_manifest,
)


__all__ = [
    "discover_filesystem",
    "discover_entry_points",
    "discover_all",
    "DiscoveryError",
]


log = logging.getLogger("genesis.community.discovery")


class DiscoveryError(Exception):
    """Raised when a discovery source produces an unrecoverable error.

    Per-manifest validation errors are surfaced as `ValidationIssue`
    objects in the validator layer, NOT as exceptions here. Discovery
    only raises when the discovery mechanism itself is broken (e.g. an
    entry-point that imports a non-existent module).
    """


def discover_filesystem(
    root: Path = DEFAULT_PLUGINS_DIR,
) -> list[tuple[Path, PatchManifest]]:
    """Walk `plugins/community/` and return (path, manifest) tuples for
    every loadable manifest. Manifests that fail to load are LOGGED and
    SKIPPED — callers can re-run `validate_directory()` to see the
    structured errors.
    """
    out: list[tuple[Path, PatchManifest]] = []
    for path in list_manifest_paths(root):
        try:
            manifest = load_manifest(path)
        except SchemaError as e:
            log.warning("filesystem discovery: %s", e)
            continue
        out.append((path, manifest))
    return out


def discover_entry_points() -> list[tuple[str, PatchManifest]]:
    """Read the `vllm.community_patches` entry-point group.

    Each entry must resolve to one of:
      - a `PatchManifest` instance (returned as-is after `.validate()`);
      - a `dict` (materialized into `PatchManifest` via dataclass mapping);
      - a `Path` (loaded via `load_manifest()`).

    Returns (entry_point_name, manifest) tuples. Bad entries are LOGGED
    and skipped — never propagate import failures up to the caller, so
    one broken package doesn't disable the whole community SDK.
    """
    out: list[tuple[str, PatchManifest]] = []
    try:
        from importlib.metadata import entry_points
    except ImportError:
        log.debug("importlib.metadata.entry_points unavailable — skipping")
        return out

    try:
        eps = entry_points(group="vllm.community_patches")
    except TypeError:
        # Python <3.10 entry_points() returns a dict-like object.
        try:
            eps = entry_points().get("vllm.community_patches", [])
        except Exception as e:
            raise DiscoveryError(
                f"could not enumerate vllm.community_patches entry-points: {e}"
            ) from e

    for ep in eps:
        try:
            obj = ep.load()
        except Exception as e:
            log.warning("entry-point %s failed to load: %s: %s",
                        ep.name, type(e).__name__, e)
            continue

        manifest: PatchManifest | None = None
        if isinstance(obj, PatchManifest):
            try:
                obj.validate()
                manifest = obj
            except SchemaError as e:
                log.warning("entry-point %s: invalid manifest: %s", ep.name, e)
                continue
        elif isinstance(obj, dict):
            from sndr.model_configs.registry_v2 import _dataclass_from_dict
            try:
                manifest = _dataclass_from_dict(PatchManifest, obj)
                manifest.validate()
            except (SchemaError, Exception) as e:
                log.warning("entry-point %s: dict→PatchManifest failed: %s",
                            ep.name, e)
                continue
        elif isinstance(obj, Path):
            try:
                manifest = load_manifest(obj)
            except SchemaError as e:
                log.warning("entry-point %s: load_manifest(%s) failed: %s",
                            ep.name, obj, e)
                continue
        else:
            log.warning(
                "entry-point %s returned %s; expected PatchManifest, dict, or Path",
                ep.name, type(obj).__name__,
            )
            continue

        out.append((ep.name, manifest))
    return out


def discover_all(
    root: Path = DEFAULT_PLUGINS_DIR,
) -> list[PatchManifest]:
    """Merge filesystem + entry-point discovery.

    Deduplication: `(namespace, id)` pair is the unique key. Filesystem
    entries win over entry-point entries on conflict — the assumption is
    that a developer who explicitly cloned a patch into
    `plugins/community/` wants their local copy to override any installed
    package version.
    """
    seen: dict[tuple[str, str], PatchManifest] = {}

    # Entry-points first, so filesystem can overwrite later.
    for _name, manifest in discover_entry_points():
        key = (manifest.namespace, manifest.id)
        seen[key] = manifest

    for _path, manifest in discover_filesystem(root):
        key = (manifest.namespace, manifest.id)
        seen[key] = manifest

    # Stable ordering: by namespace then id.
    return [seen[k] for k in sorted(seen.keys())]
