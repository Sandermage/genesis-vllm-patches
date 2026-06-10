# SPDX-License-Identifier: Apache-2.0
"""CI-wide regression guard — every PATCH_REGISTRY entry's `family`
must resolve to a known category via
`sndr.dispatcher.spec.infer_category`.

Why this matters
----------------

`infer_category(family)` powers operator-facing surfaces:
  - `sndr family status <category>`
  - PATCHES_AUTO.md grouping
  - Family-contract test discovery
  - Dispatcher search-by-category

When a registry entry uses a family value not in `_FAMILY_TO_CATEGORY`,
`infer_category` returns "uncategorized" + a warning is emitted. The
patch becomes invisible to category-grouped surfaces.

v11.3.0 BUG #12 discovered: 29 patches used 4 families ('gemma4',
'observability', 'streaming', 'offload') not in the map. Master
plan §1.1 P0.2 fix added 'streaming' and 'offload' but the others
were missed. This commit adds all four to the map; this test pins
the invariant.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Status: v11.3.0+ regression guard.
"""
from __future__ import annotations


def test_no_family_resolves_to_uncategorized():
    """Every registry entry's `family` must resolve to a non-
    "uncategorized" category via `_FAMILY_TO_CATEGORY` (exact match
    or root-prefix fallback)."""
    from sndr.dispatcher.registry import PATCH_REGISTRY
    from sndr.dispatcher.spec import infer_category
    offenders: dict[str, list[str]] = {}
    for pid, meta in PATCH_REGISTRY.items():
        if not isinstance(meta, dict):
            continue
        family = meta.get("family")
        if not family:
            continue
        cat = infer_category(family)
        if cat == "uncategorized":
            offenders.setdefault(family, []).append(pid)
    if offenders:
        lines = [
            f"  family={fam!r}: {len(pids)} patches "
            f"(sample: {pids[:3]})"
            for fam, pids in sorted(offenders.items())
        ]
        raise AssertionError(
            f"{sum(len(v) for v in offenders.values())} patches use "
            f"a `family` value not in `_FAMILY_TO_CATEGORY` "
            f"(`vllm/sndr_core/dispatcher/spec.py`). `infer_category` "
            f"returns 'uncategorized' → patches invisible to "
            f"category-grouped surfaces.\n\n"
            f"Either:\n"
            f"  (a) add `\"<family>\": \"<category>\"` to "
            f"`_FAMILY_TO_CATEGORY` (preferred), or\n"
            f"  (b) rename the family to an existing entry.\n\n"
            f"Offenders:\n" + "\n".join(lines)
        )


def test_v11_3_0_family_map_includes_phase_2_2_relocations():
    """Pin the four families added in BUG #12 fix:
    - gemma4 (Phase 2.2 → integrations/model_compat/gemma4/)
    - observability (PN289 process_info, etc.)
    - streaming (PN200-203 streaming KV)
    - offload (P104/P105/PN102)
    """
    from sndr.dispatcher.spec import _FAMILY_TO_CATEGORY
    required = {"gemma4", "observability", "streaming", "offload"}
    missing = sorted(required - set(_FAMILY_TO_CATEGORY.keys()))
    assert not missing, (
        f"_FAMILY_TO_CATEGORY missing v11.3.0 BUG #12 families: "
        f"{missing}. If you intentionally removed one, you must also "
        f"sweep PATCH_REGISTRY for entries with those families and "
        f"rename. Otherwise re-add the entry."
    )
