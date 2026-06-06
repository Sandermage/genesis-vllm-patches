# SPDX-License-Identifier: Apache-2.0
"""Runtime overlay for `PATCH_REGISTRY.apply_module` values.

Closes the Entry 12 metadata gap (127/136 patches were missing the
`apply_module` field) without editing the 2000+-line `registry.py`
source — preserves comments and per-entry lifecycle annotations.

Source of truth: legacy `@register_patch` decorator's wrapped function.
See `scripts/discover_apply_modules.py` for how the mapping is
derived. Re-run that script after any Stage 6 migration (when patch
apply hooks move out of `_per_patch_dispatch.py` into per-family
`integrations/<family>/` modules); the discovered modules will update
automatically and this overlay can be regenerated.

Invariant: this overlay ONLY adds the `apply_module` key when MISSING.
Existing values declared in `registry.py` are never overwritten.
"""
from __future__ import annotations

from typing import Any


# 127 patches that today live in the monolithic legacy dispatcher.
# When Stage 6 migrates a patch out, update its entry here (or
# re-generate via `python3 scripts/discover_apply_modules.py --emit-py
# vllm/sndr_core/dispatcher/_apply_module_overlay.py`).
APPLY_MODULE_OVERLAY: dict[str, str] = {
    pid: "vllm.sndr_core.apply._per_patch_dispatch"
    for pid in (
        "P1", "P100", "P101", "P103", "P107", "P12",
        "P14", "P15", "P15B", "P17", "P18b", "P20",
        "P22", "P23", "P24", "P26", "P27", "P28",
        "P29", "P3", "P31", "P32", "P34", "P36",
        "P37", "P38", "P38B", "P39a", "P4", "P40",
        "P44", "P46", "P5", "P58", "P59", "P5b",
        "P6", "P60", "P60b", "P61", "P61b", "P61c",
        "P62", "P63", "P64", "P65", "P66", "P67",
        "P67b", "P67c", "P68", "P7", "P70", "P71",
        "P72", "P74", "P75", "P77", "P78", "P79b",
        "P79c", "P79d", "P7b", "P8", "P81", "P82",
        "P83", "P84", "P85", "P86", "P87", "P91",
        "P94", "P95", "P98", "P99", "PN11", "PN12",
        "PN13", "PN14", "PN16", "PN17", "PN19", "PN21",
        "PN22", "PN23", "PN24", "PN25", "PN26", "PN26b",
        "PN27", "PN28", "PN29", "PN30", "PN31", "PN32",
        "PN33", "PN34", "PN35", "PN38", "PN40", "PN50",
        "PN51", "PN52", "PN54", "PN55", "PN56", "PN57",
        "PN58", "PN59", "PN61", "PN62", "PN65", "PN66",
        "PN67", "PN70", "PN72", "PN77", "PN78", "PN79",
        "PN8", "PN80", "PN82", "PN9", "PN90", "PN95",
        "PN96",
    )
}


def _has_integration_tree_module(patch_id: str) -> bool:
    """Return True iff `vllm/sndr_core/integrations/<family>/<file>.py`
    exists for this patch. The auto-discovery in
    `dispatcher.spec._build_apply_module_map` will resolve that path —
    we MUST NOT shadow it with the generic legacy module.

    Cheap filesystem-only check (no module import). Walks every
    `integrations/*/` directory looking for filenames that start with
    the patch_id lowercased (matches the established convention:
    `pn82_mamba_cudagraph_prefill_zero.py`, `p67_tq_multi_query_kernel.py`).
    """
    from pathlib import Path
    integrations_dir = Path(__file__).resolve().parent.parent / "integrations"
    if not integrations_dir.is_dir():
        return False
    prefix = patch_id.lower() + "_"
    exact = patch_id.lower() + ".py"
    for family_dir in integrations_dir.iterdir():
        if not family_dir.is_dir():
            continue
        for entry in family_dir.iterdir():
            if not entry.is_file() or entry.suffix != ".py":
                continue
            name = entry.name.lower()
            if name == exact or name.startswith(prefix):
                return True
    return False


def apply_overlay(registry: dict[str, dict[str, Any]]) -> int:
    """Merge `APPLY_MODULE_OVERLAY` into `registry` in-place.

    Returns the number of entries actually patched. The overlay NEVER
    overwrites:

      1. an explicit `apply_module` value already declared in `registry.py`
         (e.g. PN95's `vllm.sndr_core.integrations.kv_cache.pn95_tier_aware_cache`)
      2. patches that have an on-disk integration-tree module under
         `integrations/<family>/<patch_id>_*.py` (Stage 6 migrated patches
         like PN82 → `integrations/worker/pn82_mamba_cudagraph_prefill_zero.py`).
         The PatchSpec layer in `dispatcher.spec` prefers explicit
         registry values over derived ones, so writing the legacy path
         here would silently downgrade Stage 6 migrations to the
         monolith.

    Unknown patch ids in the overlay (i.e. in overlay but not in
    registry) are silently ignored so a stale overlay doesn't break
    import after a patch is retired.
    """
    patched = 0
    for patch_id, module_path in APPLY_MODULE_OVERLAY.items():
        entry = registry.get(patch_id)
        if entry is None:
            continue
        if "apply_module" in entry and entry["apply_module"]:
            continue
        if _has_integration_tree_module(patch_id):
            # Let spec.py's _build_apply_module_map fill this in.
            continue
        entry["apply_module"] = module_path
        patched += 1
    return patched
