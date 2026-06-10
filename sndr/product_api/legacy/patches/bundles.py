# SPDX-License-Identifier: Apache-2.0
"""Pure-data query layer for ``sndr patches bundles`` (M.6.1).

Bundles are atomic multi-patch orchestrators. The catalog mirrors
``tests/bundles/test_stage7_bundles_smoke.py::BUNDLES``; the
``test_bundles_catalog_matches_test_smoke`` drift detector compares the
two sources to catch divergent additions.
"""
from __future__ import annotations

from typing import Optional

from .types import BundleSpec


# Canonical bundle catalog. Tuple-of-tuples (not list-of-BundleSpec)
# so consumers — including the smoke-test drift detector — keep the
# stable subscriptable shape ``{(b[0], b[1], b[2]) for b in CATALOG}``.
BUNDLES_CATALOG: tuple[tuple[str, str, str, str], ...] = (
    (
        "tool_parsing_qwen3coder",
        "BUNDLE_TOOL_PARSING_QWEN3CODER",
        "community",
        "P15 + P61c + P64(×2) + PN56 — Qwen3-coder tool-parser fixes.",
    ),
    (
        "reasoning_qwen3",
        "BUNDLE_REASONING_QWEN3",
        "community",
        "P12 + P27 + P59 + P61 + P61b + PN51 — Qwen3 reasoning parser.",
    ),
    (
        "attention_gdn_spec",
        "BUNDLE_ATTENTION_GDN_SPEC",
        "community",
        "P60 + P60b — GDN spec-decode pipeline atomic apply.",
    ),
    (
        "attention_tq_multi_query",
        "BUNDLE_ATTENTION_TQ_MULTI_QUERY",
        "community",
        "P67 + P67b — TQ multi-query kernel + spec verify routing.",
    ),
    (
        "spec_decode_async_cleanup",
        "BUNDLE_SPEC_DECODE_ASYNC_CLEANUP",
        "community",
        "P79b + P79c + P79d — async cleanup of spec-decode artifacts.",
    ),
)


def list_bundles() -> list[BundleSpec]:
    """Return all bundles as ``BundleSpec`` records.

    ``has_apply`` is left ``None`` here — probing every bundle's module
    on list is wasteful. :func:`explain_bundle` performs the import
    lazily for the operator-asked entry.
    """
    return [
        BundleSpec(
            name=name,
            umbrella_flag=flag,
            tier=tier,
            description=desc,
            module_path=f"sndr.bundles.{name}",
            has_apply=None,
        )
        for name, flag, tier, desc in BUNDLES_CATALOG
    ]


def explain_bundle(name: str) -> Optional[BundleSpec]:
    """Probe a single bundle module to fill in ``has_apply``.

    Returns ``None`` if no bundle with the given name exists.
    Mirrors the import-probe pattern previously in
    ``cli.patches._run_bundles_explain``: failure is recorded in
    ``import_error`` rather than raised.
    """
    match = next((b for b in BUNDLES_CATALOG if b[0] == name), None)
    if match is None:
        return None
    bname, bflag, btier, bdesc = match
    module_path = f"sndr.bundles.{bname}"
    has_apply: Optional[bool] = None
    import_error: Optional[str] = None
    try:
        mod = __import__(module_path, fromlist=["apply"])
        has_apply = callable(getattr(mod, "apply", None))
    except Exception as e:
        has_apply = False
        import_error = f"{type(e).__name__}: {e}"
    return BundleSpec(
        name=bname,
        umbrella_flag=bflag,
        tier=btier,
        description=bdesc,
        module_path=module_path,
        has_apply=has_apply,
        import_error=import_error,
    )
