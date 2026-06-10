"""Positive-assertion guards against the post-rename stale-ref class.

Background — 2026-05-12 audit found that the v10 `patches/`→`integrations/`
and `paths/`→`locations/` renames left silent stale references in
`compat/categories.py::_WIRING_DIR`, `compat/self_test._check_wiring_imports`,
`compat/cache_parity_audit._read_patch_source`, and
`locations/project_paths::wiring_dir`. Each one degraded to `None` /
`0 modules` instead of crashing, so v11 shipped on top of the bugs.

These tests assert positive resolution (`is not None`, `> N`) so any
future rename that misses a downstream pointer fails loud immediately.

Single source of truth for the canonical wiring location is
`sndr.engines.vllm.locations.project_paths.wiring_dir()` — these tests
indirectly validate it via the four downstream helpers that depend on it.
"""

from __future__ import annotations

import pytest


# A handful of patch_ids covering different naming conventions so a
# regex error that breaks one form is still caught by another.
SENTINEL_PATCH_IDS = ["P67", "PN14", "PN65", "P38", "P75"]


class TestWiringDirResolves:
    """`wiring_dir()` MUST resolve to a real on-disk directory."""

    def test_wiring_dir_not_none(self):
        from sndr.engines.vllm.locations.project_paths import wiring_dir
        assert wiring_dir() is not None, (
            "wiring_dir() returned None — canonical patch directory "
            "is not findable. This usually means the post-rename "
            "fallback chain in project_paths.py is stale."
        )

    def test_wiring_dir_is_real_directory(self):
        from sndr.engines.vllm.locations.project_paths import wiring_dir
        wd = wiring_dir()
        assert wd is not None and wd.is_dir(), (
            f"wiring_dir() returned {wd!r} but it's not a directory."
        )

    def test_wiring_dir_contains_patch_files(self):
        """At least 100 patch files of the canonical naming convention.

        Lower-bound only — exact count is `_check_wiring_imports`'s job.
        Catches the case where `wiring_dir()` resolves to an empty stale
        directory (e.g. `sndr_core/wiring/`) instead of the active one.
        """
        from sndr.engines.vllm.locations.project_paths import wiring_dir
        wd = wiring_dir()
        canonical = list(wd.rglob("p[0-9]*.py")) + list(wd.rglob("pn[0-9]*.py"))
        legacy = list(wd.rglob("patch_*.py"))
        total = len(canonical) + len(legacy)
        assert total >= 100, (
            f"wiring_dir() = {wd} contains only {total} patch files "
            f"(canonical={len(canonical)}, legacy={len(legacy)}). "
            f"Expected ≥100. Probably resolved to a stale empty dir."
        )


class TestModuleForResolves:
    """`compat.categories.module_for(patch_id)` MUST resolve every
    sentinel patch_id. Catches a stale `_WIRING_DIR` constant."""

    @pytest.mark.parametrize("patch_id", SENTINEL_PATCH_IDS)
    def test_module_for_returns_dotted_path(self, patch_id: str):
        from sndr.compat.categories import module_for
        mod = module_for(patch_id)
        assert mod is not None, (
            f"module_for({patch_id!r}) returned None — compat/categories.py "
            f"can't find the wiring module. Probably `_WIRING_DIR` "
            f"points to a stale directory."
        )
        assert mod.startswith("sndr.engines.vllm.patches."), (
            f"module_for({patch_id!r}) = {mod!r} doesn't look like a "
            f"canonical sndr.engines.vllm.patches.* dotted path."
        )


class TestSelfTestFindsModules:
    """`compat.self_test._check_wiring_imports` MUST validate > 100
    modules. Catches the silent `0 modules pass` regression."""

    def test_check_wiring_imports_finds_many_modules(self):
        from sndr.compat.self_test import _check_wiring_imports
        status, msg = _check_wiring_imports()
        assert status in ("pass", "warn"), (
            f"_check_wiring_imports returned status={status!r}, msg={msg!r}"
        )
        # Parse the count out of "N wiring modules import cleanly"
        # or "N/M wiring modules imported; ..."
        import re
        m = re.search(r"(\d+)\s+wiring modules", msg)
        assert m is not None, (
            f"could not parse module count from msg={msg!r}"
        )
        count = int(m.group(1))
        assert count >= 100, (
            f"_check_wiring_imports validated only {count} modules — "
            f"expected ≥100. Probably scanning the wrong directory or "
            f"using the wrong filename glob."
        )


class TestRegistryEnvFlagDisambiguation:
    """When two integrations files share the same numeric stem
    (e.g. PN26 + PN26b both backed by `pn26_*_kernel.py` files in
    the same dir), `module_for` MUST use the registry's `env_flag`
    suffix to pick the right one. Catches the regression where
    sorted-first wins and PN26 silently routes to PN26b's source.
    """

    def test_pn26_routes_to_tq_unified(self):
        """PN26's env_flag is GENESIS_ENABLE_PN26_TQ_UNIFIED — the
        file `pn26_tq_unified_perf.py` should win over `pn26_sparse_v_kernel.py`."""
        from sndr.compat.categories import module_for
        mod = module_for("PN26")
        assert mod is not None and "tq_unified" in mod, (
            f"module_for('PN26') = {mod!r} — expected `tq_unified`. "
            f"Disambiguation pass in compat/categories.py is broken."
        )

    def test_pn26b_routes_to_sparse_v(self):
        """PN26b's env_flag is GENESIS_ENABLE_PN26_SPARSE_V — the
        file `pn26_sparse_v_kernel.py` should win.

        Note: PN26b's actual on-disk filename has no `b` suffix
        (legacy naming), so it's recovered via parent-pid (PN26)
        candidate-pool propagation."""
        from sndr.compat.categories import module_for
        mod = module_for("PN26b")
        assert mod is not None and "sparse_v" in mod, (
            f"module_for('PN26b') = {mod!r} — expected `sparse_v`. "
            f"Letter-suffix variant routing in compat/categories.py is broken."
        )


class TestCacheParityReadsSources:
    """`compat.cache_parity_audit._read_patch_source` MUST return source
    text for every sentinel patch_id. Catches a stale `wiring_root`."""

    @pytest.mark.parametrize("patch_id", SENTINEL_PATCH_IDS)
    def test_read_patch_source_returns_nontrivial_text(self, patch_id: str):
        from sndr.compat.cache_parity_audit import _read_patch_source
        src = _read_patch_source(patch_id)
        assert src is not None, (
            f"_read_patch_source({patch_id!r}) returned None — can't "
            f"find the source file."
        )
        assert len(src) >= 500, (
            f"_read_patch_source({patch_id!r}) returned only "
            f"{len(src)} chars — looks like a stub, not a real patch."
        )
