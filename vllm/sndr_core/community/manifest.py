# SPDX-License-Identifier: Apache-2.0
"""Phase 5 community SDK — manifest loader + path enumeration.

Thin wrapper over `registry_v2.load_patch_manifest()`. Adds:

  • `load_manifest(path)` — same shape but raises uniformly with a path
    prefix in the error so operator sees which manifest is broken.
  • `list_manifest_paths(root)` — walk a `plugins/community/` tree and
    return every `manifest.yaml`, skipping `_template` directories
    (per §6.6 no-stub gate — templates are draft examples, not registry
    entries).
"""
from __future__ import annotations

from pathlib import Path

from vllm.sndr_core.model_configs.registry_v2 import load_patch_manifest
from vllm.sndr_core.model_configs.schema import SchemaError
from vllm.sndr_core.model_configs.schema_v2 import PatchManifest


__all__ = [
    "load_manifest",
    "list_manifest_paths",
    "REPO_ROOT",
    "DEFAULT_PLUGINS_DIR",
]


# Repo root is the project's top-level directory. The plugins/community/
# tree is tracked there; templates live under _template/ which validator
# rejects from release lists (no-stub gate §6.6).
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLUGINS_DIR = REPO_ROOT / "plugins" / "community"


def load_manifest(path: Path) -> PatchManifest:
    """Load a manifest from disk and run its dataclass-level `.validate()`.

    Raises SchemaError with the manifest path included in the message —
    operator sees which file failed without scrolling stack traces.
    """
    try:
        return load_patch_manifest(path)
    except SchemaError as e:
        raise SchemaError(f"{path}: {e}") from None
    except Exception as e:
        raise SchemaError(f"{path}: {type(e).__name__}: {e}") from None


def list_manifest_paths(root: Path = DEFAULT_PLUGINS_DIR) -> list[Path]:
    """Walk a community plugins root and return every `manifest.yaml`.

    Exclusion rules (§6.6 no-stub + §6.10 public/private boundary):

      • Any directory whose name starts with `_` (e.g. `_template`,
        `_private`) is treated as documentation-only and skipped.
      • Files starting with `_` (e.g. `_draft.yaml`) are skipped.
      • Empty roots return an empty list, not an error — fresh installs
        often have `plugins/community/.gitkeep` only.
    """
    if not root.is_dir():
        return []

    paths: list[Path] = []
    for path in sorted(root.rglob("manifest.yaml")):
        # Skip any path that traverses an underscore-prefixed directory.
        # Relative-to root lets us check each segment without including
        # the project root.
        rel_parts = path.relative_to(root).parts
        if any(p.startswith("_") for p in rel_parts):
            continue
        paths.append(path)
    return paths
