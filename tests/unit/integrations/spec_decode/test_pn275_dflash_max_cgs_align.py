# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for PN275 — DFlash drafter VllmConfig max_cgs
alignment (dev371 compat).

These tests exercise the wrapper closure built by `_build_wrapper`
against synthetic stand-ins for `vllm.config.utils.replace` and the
VllmConfig dataclass. No torch / CUDA / pydantic dependency required.

Covered behavior:

  * `should_apply()` is False without the env, True when env truthy.
  * The wrapper only intervenes when the source's class name is
    'VllmConfig'.
  * When caller supplies `compilation_config` in kwargs, wrapper
    defers to the original verbatim.
  * When source's compilation_config has consistent sizes/max,
    wrapper delegates without modifying kwargs.
  * When source has desynchronized max/sizes, wrapper builds an
    aligned compilation_config and injects it into kwargs.
  * Marker `_genesis_dflash_align_wrapped` prevents double-wrap on
    a second apply() call.
  * `is_applied()` / `revert()` round-trip on a fake module.
  * Registry entry shape matches the documented invariants.
"""
from __future__ import annotations

import os
import types
import unittest.mock as mock

import pytest


# ─── Module import (no torch dep) ───────────────────────────────────────


def _import_patch():
    """Import the patch module fresh. Test isolation: each call returns
    the module's current state (env-gated behavior changes with the
    surrounding monkeypatch)."""
    from sndr.engines.vllm.patches.spec_decode import (
        pn275_dflash_max_cgs_align as p,
    )
    return p


# ─── should_apply ───────────────────────────────────────────────────────


class TestShouldApply:
    def test_no_env_returns_false(self, monkeypatch):
        monkeypatch.delenv(
            "GENESIS_ENABLE_PN275_DFLASH_MAX_CGS_ALIGN", raising=False
        )
        assert _import_patch().should_apply() is False

    @pytest.mark.parametrize("v", ["1", "true", "yes", "on", "True", "YES"])
    def test_truthy_env_returns_true(self, monkeypatch, v):
        monkeypatch.setenv(
            "GENESIS_ENABLE_PN275_DFLASH_MAX_CGS_ALIGN", v
        )
        assert _import_patch().should_apply() is True

    @pytest.mark.parametrize("v", ["0", "", "false", "off", "no", "False"])
    def test_falsy_env_returns_false(self, monkeypatch, v):
        monkeypatch.setenv(
            "GENESIS_ENABLE_PN275_DFLASH_MAX_CGS_ALIGN", v
        )
        assert _import_patch().should_apply() is False


# ─── Wrapper closure behavior ───────────────────────────────────────────


class _FakeVllmConfig:
    """Stand-in. `type(x).__name__ == 'VllmConfig'` because of the
    class rename below at the bottom of the module."""

    def __init__(self, compilation_config):
        self.compilation_config = compilation_config


# Rename the class so it matches the wrapper's string-match check.
_FakeVllmConfig.__name__ = "VllmConfig"


class _FakeCompilationConfig:
    def __init__(self, max_cudagraph_capture_size, cudagraph_capture_sizes):
        self.max_cudagraph_capture_size = max_cudagraph_capture_size
        self.cudagraph_capture_sizes = cudagraph_capture_sizes


class _NonVllmConfig:
    def __init__(self):
        self.compilation_config = None


class TestWrapperBehavior:
    def _make_recorder(self):
        """Return (recorder, fake_replace). fake_replace mimics
        upstream `utils.replace` enough to be a wrap target — it
        records the call (instance, kwargs) and returns a sentinel."""
        recorded = []

        def fake_replace(instance, /, **kwargs):
            recorded.append({
                "instance": instance,
                "kwargs": dict(kwargs),
            })
            # When called with a CompilationConfig to align, return a
            # new CompilationConfig with the override applied. This
            # mimics `cls(**dataclass_dict)` shape for the recursive
            # alignment call inside the wrapper.
            if isinstance(instance, _FakeCompilationConfig):
                merged_max = kwargs.get(
                    "max_cudagraph_capture_size",
                    instance.max_cudagraph_capture_size,
                )
                merged_sizes = kwargs.get(
                    "cudagraph_capture_sizes",
                    instance.cudagraph_capture_sizes,
                )
                return _FakeCompilationConfig(
                    merged_max, merged_sizes,
                )
            return "ORIG_RESULT"
        return recorded, fake_replace

    def test_non_vllm_config_passes_through_untouched(self):
        recorded, fake_replace = self._make_recorder()
        wrapper = _import_patch()._build_wrapper(fake_replace)

        non_vllm = _NonVllmConfig()
        result = wrapper(non_vllm, attention_config="ATTN")
        assert result == "ORIG_RESULT"
        # Exactly one call to original; kwargs untouched
        assert len(recorded) == 1
        assert recorded[0]["instance"] is non_vllm
        assert recorded[0]["kwargs"] == {"attention_config": "ATTN"}

    def test_caller_supplied_compilation_config_kwarg_is_respected(self):
        recorded, fake_replace = self._make_recorder()
        wrapper = _import_patch()._build_wrapper(fake_replace)

        cc = _FakeCompilationConfig(
            max_cudagraph_capture_size=8,
            cudagraph_capture_sizes=[1, 2, 4, 6],
        )
        vc = _FakeVllmConfig(cc)
        operator_supplied_cc = _FakeCompilationConfig(99, [1])
        wrapper(vc, compilation_config=operator_supplied_cc)
        # Wrapper must NOT replace the operator's explicit kwarg
        assert len(recorded) == 1
        assert recorded[0]["kwargs"]["compilation_config"] is (
            operator_supplied_cc
        )

    def test_consistent_source_still_clears_sizes_for_p95_safety(self):
        """Even when source's max == max(sizes), wrapper MUST clear
        cudagraph_capture_sizes before delegating. Reason: P95
        text-patches `_set_cudagraph_sizes` to FORCE max=8 mid-
        rebuild, which then desyncs against the carried sizes and
        triggers the dev371 cross-validator. The only way to route
        the validator into its warning-only branch is to have
        `cudagraph_capture_sizes is None` at rebuild time. See M4
        retry receipts 2026-05-21 — first via inner-replace, then
        via object.__setattr__ direct-mutation after pydantic
        rejected None for the typed list[int] field."""
        recorded, fake_replace = self._make_recorder()
        wrapper = _import_patch()._build_wrapper(fake_replace)

        cc = _FakeCompilationConfig(
            max_cudagraph_capture_size=8,
            cudagraph_capture_sizes=[1, 2, 4, 8],
        )
        vc = _FakeVllmConfig(cc)
        wrapper(vc, attention_config="ATTN")
        # Exactly one outer replace call. The wrapper mutates source's
        # cc via object.__setattr__ before delegating; no inner
        # replace recorded.
        assert len(recorded) == 1
        # The source's compilation_config now has sizes = None
        assert cc.cudagraph_capture_sizes is None
        # Outer call sees the mutated source via dataclass_instance
        assert recorded[0]["instance"] is vc
        # Operator's attention_config kwarg preserved
        assert recorded[0]["kwargs"]["attention_config"] == "ATTN"
        # And compilation_config NOT injected as kwarg (mutation
        # propagates via the source reference)
        assert "compilation_config" not in recorded[0]["kwargs"]

    def test_empty_sizes_passes_through(self):
        recorded, fake_replace = self._make_recorder()
        wrapper = _import_patch()._build_wrapper(fake_replace)

        cc = _FakeCompilationConfig(
            max_cudagraph_capture_size=8,
            cudagraph_capture_sizes=[],   # empty
        )
        vc = _FakeVllmConfig(cc)
        wrapper(vc, attention_config="ATTN")
        assert len(recorded) == 1
        assert "compilation_config" not in recorded[0]["kwargs"]

    def test_max_none_passes_through(self):
        recorded, fake_replace = self._make_recorder()
        wrapper = _import_patch()._build_wrapper(fake_replace)

        cc = _FakeCompilationConfig(
            max_cudagraph_capture_size=None,
            cudagraph_capture_sizes=[1, 2, 4, 6],
        )
        vc = _FakeVllmConfig(cc)
        wrapper(vc, attention_config="ATTN")
        assert len(recorded) == 1
        assert "compilation_config" not in recorded[0]["kwargs"]

    def test_desync_source_clears_sizes(self):
        """Canonical defect case: source has max=8 vs sizes=[..., 6].
        Wrapper directly mutates `source.compilation_config.cudagraph_capture_sizes`
        to None via `object.__setattr__` (bypassing the pydantic
        typed-field validator that rejects None for `list[int]`).
        After this mutation, vllm's `_set_cudagraph_sizes` reaches
        the rebuild with sizes=None on the source → auto-recomputes
        them → validator at line 1703-1715 sees the just-recomputed
        local list but `self.compilation_config.cudagraph_capture_sizes
        is None` (the mutated source) → routes through the
        warning-only branch instead of raising."""
        recorded, fake_replace = self._make_recorder()
        wrapper = _import_patch()._build_wrapper(fake_replace)

        cc = _FakeCompilationConfig(
            max_cudagraph_capture_size=8,
            cudagraph_capture_sizes=[1, 2, 4, 6],
        )
        vc = _FakeVllmConfig(cc)
        wrapper(vc, attention_config="ATTN")
        # Exactly one outer replace call; mutation propagates via
        # the shared source reference.
        assert len(recorded) == 1
        # The source's compilation_config now has sizes = None
        assert cc.cudagraph_capture_sizes is None
        # Outer call sees the mutated source
        assert recorded[0]["instance"] is vc
        # Operator's attention_config kwarg preserved
        assert recorded[0]["kwargs"]["attention_config"] == "ATTN"
        # And compilation_config NOT injected as kwarg
        assert "compilation_config" not in recorded[0]["kwargs"]

    def test_setattr_failure_falls_through_to_original(self):
        """If the source's compilation_config rejects the
        `object.__setattr__` (e.g. slot-only class where the slot
        doesn't exist for the target field), the wrapper must NOT
        propagate the exception — fall through to the original outer
        call so pydantic can surface the real error.
        Defense-in-depth."""
        recorded = []

        def fake_replace(instance, /, **kwargs):
            recorded.append({"instance": instance, "kwargs": dict(kwargs)})
            return "ORIG_RESULT"

        wrapper = _import_patch()._build_wrapper(fake_replace)

        # Slot-only class — object.__setattr__ on a non-existent slot
        # raises AttributeError. Set initial slots via class definition.
        class _SlottedCC:
            __slots__ = (
                "max_cudagraph_capture_size",
                # NOTE: `cudagraph_capture_sizes` is intentionally
                # NOT in __slots__ — object.__setattr__ on it raises.
            )

            def __init__(self):
                self.max_cudagraph_capture_size = 8

            # Make getattr return the read-only sizes (the wrapper
            # reads cudagraph_capture_sizes via getattr).
            @property
            def cudagraph_capture_sizes(self):
                return [1, 2, 4, 6]

        slotted_cc = _SlottedCC()
        vc = _FakeVllmConfig(slotted_cc)
        result = wrapper(vc, attention_config="ATTN")

        assert result == "ORIG_RESULT"
        # Outer call still happened (defense-in-depth fallthrough)
        assert len(recorded) == 1


# ─── apply() / is_applied() / revert() against a fake module ───────────


class TestApplyAgainstFakeModule:
    """Exercise apply() / is_applied() / revert() by monkeypatching
    `importlib.import_module` to return a synthetic stand-in for
    `vllm.config.utils`. CPU-only — does not require vllm install."""

    def _make_fake_utils(self, replace_fn):
        m = types.ModuleType("vllm.config.utils")
        m.replace = replace_fn
        return m

    def test_apply_skips_when_env_unset(self, monkeypatch):
        monkeypatch.delenv(
            "GENESIS_ENABLE_PN275_DFLASH_MAX_CGS_ALIGN", raising=False,
        )
        p = _import_patch()
        status, reason = p.apply()
        assert status == "skipped"
        assert "GENESIS_ENABLE_PN275_DFLASH_MAX_CGS_ALIGN" in reason

    def test_apply_wraps_and_marks(self, monkeypatch):
        monkeypatch.setenv(
            "GENESIS_ENABLE_PN275_DFLASH_MAX_CGS_ALIGN", "1",
        )
        p = _import_patch()

        def original(instance, /, **kw):
            return "OK"
        fake_mod = self._make_fake_utils(original)

        with mock.patch(
            "importlib.import_module", return_value=fake_mod,
        ):
            status, reason = p.apply()
        assert status == "applied"
        # The marker MUST be present on the new fake_mod.replace
        assert getattr(fake_mod.replace, p._WRAPPED_ATTR, False) is True
        # Original is stashed for revert
        assert getattr(fake_mod.replace, p._ORIGINAL_ATTR) is original

    def test_apply_idempotent(self, monkeypatch):
        monkeypatch.setenv(
            "GENESIS_ENABLE_PN275_DFLASH_MAX_CGS_ALIGN", "1",
        )
        p = _import_patch()

        def original(instance, /, **kw):
            return "OK"
        fake_mod = self._make_fake_utils(original)

        with mock.patch(
            "importlib.import_module", return_value=fake_mod,
        ):
            p.apply()
            first_wrap = fake_mod.replace
            status, reason = p.apply()
            second_wrap = fake_mod.replace
        # Second apply must NOT double-wrap
        assert status == "applied"
        assert "idempotent" in reason
        assert first_wrap is second_wrap

    def test_revert_restores_original(self, monkeypatch):
        monkeypatch.setenv(
            "GENESIS_ENABLE_PN275_DFLASH_MAX_CGS_ALIGN", "1",
        )
        p = _import_patch()

        def original(instance, /, **kw):
            return "OK"
        fake_mod = self._make_fake_utils(original)

        with mock.patch(
            "importlib.import_module", return_value=fake_mod,
        ):
            p.apply()
            assert fake_mod.replace is not original
            ok = p.revert()
        assert ok is True
        assert fake_mod.replace is original


# ─── Registry contract ─────────────────────────────────────────────────


class TestRegistryEntry:
    def test_pn275_registered(self):
        from sndr.dispatcher import PATCH_REGISTRY
        assert "PN275" in PATCH_REGISTRY

    def test_pn275_metadata_shape(self):
        from sndr.dispatcher import PATCH_REGISTRY
        meta = PATCH_REGISTRY["PN275"]
        assert meta["env_flag"] == "GENESIS_ENABLE_PN275_DFLASH_MAX_CGS_ALIGN"
        assert meta["default_on"] is False
        assert meta["family"] == "spec_decode"
        assert meta["lifecycle"] == "experimental"
        assert meta["implementation_status"] == "experimental"
        assert meta["apply_module"] == (
            "sndr.engines.vllm.patches.spec_decode."
            "pn275_dflash_max_cgs_align"
        )
        # dev371-only — pin gate documents the dev338-incompatible
        # cross-validator that this patch addresses
        assert meta.get("vllm_version_range") == ">=0.20.2rc1.dev371"

    def test_pn275_dispatcher_hook(self):
        """The @register_patch hook must exist in the apply layer so
        sndr patches apply / boot dispatch can reach the patch."""
        from sndr.apply import apply_all
        assert hasattr(
            apply_all, "apply_patch_pn275_dflash_max_cgs_align"
        )


# ─── M2c: self-install at module-import (P103 pattern, spawn-safe) ─────


class TestSelfInstallHelper:
    """The helper that the text-patched self-install block calls at
    module-import time. Each spawn worker re-executes the appended
    block on every `import vllm.config.utils`, so the helper IS the
    install mechanism that survives the spawn boundary.

    The setattr-only apply() path (M2) only reaches APIServer +
    EngineCore — worker processes spawn fresh Python interpreters
    and never run our dispatcher. Confirmed empirically in the
    P2_DFLASH_M4_Q27_SMOKE_FAIL_2026-05-21 receipt.
    """

    def _make_fake_globals(self, replace_fn):
        """Synthesize the kind of module-globals dict that the
        text-patched `vllm/config/utils.py` would pass to the helper
        at module-import time."""
        return {"replace": replace_fn}

    def test_helper_no_env_returns_false(self, monkeypatch):
        monkeypatch.delenv(
            "GENESIS_ENABLE_PN275_DFLASH_MAX_CGS_ALIGN", raising=False,
        )
        p = _import_patch()
        fake_globals = self._make_fake_globals(lambda *a, **kw: None)
        result = p._genesis_pn275_install_at_import(fake_globals)
        assert result is False
        # globals must be untouched
        assert getattr(
            fake_globals["replace"], p._WRAPPED_ATTR, False,
        ) is False

    def test_helper_installs_wrap_when_env_set(self, monkeypatch):
        monkeypatch.setenv(
            "GENESIS_ENABLE_PN275_DFLASH_MAX_CGS_ALIGN", "1",
        )
        p = _import_patch()

        def orig(inst, /, **kw):
            return "ORIG"
        fake_globals = self._make_fake_globals(orig)
        result = p._genesis_pn275_install_at_import(fake_globals)
        assert result is True
        new_fn = fake_globals["replace"]
        assert new_fn is not orig
        assert getattr(new_fn, p._WRAPPED_ATTR, False) is True

    def test_helper_returns_false_when_no_replace_symbol(
        self, monkeypatch,
    ):
        """If the globals dict somehow lacks `replace` (interface
        drift), helper must NOT raise — return False and keep
        `vllm/config/utils.py` importable."""
        monkeypatch.setenv(
            "GENESIS_ENABLE_PN275_DFLASH_MAX_CGS_ALIGN", "1",
        )
        p = _import_patch()
        result = p._genesis_pn275_install_at_import({})
        assert result is False

    def test_helper_idempotent_when_already_wrapped(self, monkeypatch):
        monkeypatch.setenv(
            "GENESIS_ENABLE_PN275_DFLASH_MAX_CGS_ALIGN", "1",
        )
        p = _import_patch()

        def orig(inst, /, **kw):
            return "ORIG"
        fake_globals = self._make_fake_globals(orig)
        p._genesis_pn275_install_at_import(fake_globals)
        first_wrap = fake_globals["replace"]
        # Second call must NOT double-wrap
        result = p._genesis_pn275_install_at_import(fake_globals)
        assert result is True
        assert fake_globals["replace"] is first_wrap

    def test_helper_never_raises(self, monkeypatch):
        """Helper's try/except must swallow any exception so the
        text-patched module always imports cleanly."""
        monkeypatch.setenv(
            "GENESIS_ENABLE_PN275_DFLASH_MAX_CGS_ALIGN", "1",
        )
        p = _import_patch()

        # Pass an instance that doesn't behave like a normal mapping
        class _BrokenGlobals:
            def get(self, *a, **kw):
                raise RuntimeError("simulated globals break")

        # Should NOT raise — just return False
        result = p._genesis_pn275_install_at_import(_BrokenGlobals())
        assert result is False


class TestSelfInstallTextPatch:
    """The text-patched block appended to `vllm/config/utils.py` must
    contain the env guard AND the helper call. The anchor must match
    the exact upstream `replace()` function definition at the
    documented dev371 SHA. Drift markers must be specific."""

    def test_block_has_env_guard(self):
        p = _import_patch()
        block = p._PN275_SELF_INSTALL_BLOCK
        assert "GENESIS_ENABLE_PN275_DFLASH_MAX_CGS_ALIGN" in block, (
            "self-install block must check the env flag before "
            "wrapping — operators who don't opt in must see zero "
            "behavior change to vllm.config.utils.replace"
        )

    def test_block_calls_helper(self):
        p = _import_patch()
        block = p._PN275_SELF_INSTALL_BLOCK
        assert "_genesis_pn275_install_at_import" in block, (
            "self-install block must import + call the helper"
        )
        # Must pass globals() so the helper operates on the right namespace
        assert "globals()" in block

    def test_block_wrapped_in_try_except(self):
        """If sndr_core isn't on sys.path (partial install, test env),
        vllm/config/utils.py must still import cleanly."""
        p = _import_patch()
        block = p._PN275_SELF_INSTALL_BLOCK
        assert "try:" in block
        assert "except" in block

    def test_anchor_matches_dev371_upstream_signature(self):
        """The anchor IS the dev371 `replace()` function definition.
        If upstream renames/reformats `replace`, this assertion fails
        BEFORE the text-patch attempts to land on a drifted file."""
        p = _import_patch()
        anchor = p._PN275_SELF_INSTALL_ANCHOR
        # Function signature line
        assert (
            "def replace(dataclass_instance: ConfigT, /, **kwargs)"
            in anchor
        )
        # Final return is the splice point
        assert "return cls(**dataclass_dict)" in anchor

    def test_drift_marker_specific(self):
        """Drift markers must reference the patch identifier so two
        sibling Genesis patches on the same file don't collide."""
        p = _import_patch()
        block = p._PN275_SELF_INSTALL_BLOCK
        # Must mention PN275 by name in the appended comment block
        assert "PN275" in block

    def test_make_self_install_text_patcher_against_temp_tree(
        self, tmp_path, monkeypatch,
    ):
        """When given a temp vllm tree that contains the expected
        anchor file, `_make_self_install_text_patcher()` must return
        a TextPatcher object whose drift markers reference PN275."""
        p = _import_patch()
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        # Synthesize a minimal vllm/config/utils.py containing the
        # anchor exactly (so TextPatcher.find_anchor() can match).
        (cfg_dir / "utils.py").write_text(
            "# minimal stand-in\n"
            "from typing import TypeVar\n"
            "ConfigT = TypeVar('ConfigT')\n"
            "ConfigType = TypeVar('ConfigType')\n"
            "def is_init_field(cls, name): return True\n\n"
            + p._PN275_SELF_INSTALL_ANCHOR
        )
        import sndr.engines.vllm.detection.guards as guards
        import sndr.engines.vllm.detection.guards as _canon_guards
        orig = guards.vllm_install_root
        _orig_canon = _canon_guards.vllm_install_root
        # v12.x: the patch imports vllm_install_root from the shim for its
        # own gate, but resolve_vllm_file (canonical) calls the CANONICAL
        # vllm_install_root internally — mock both so the temp tree is seen.
        guards.vllm_install_root = _canon_guards.vllm_install_root = lambda: tmp_path
        try:
            patcher = p._make_self_install_text_patcher()
        finally:
            guards.vllm_install_root = orig
            _canon_guards.vllm_install_root = _orig_canon

        assert patcher is not None
        for m in patcher.upstream_drift_markers:
            assert "PN275" in m or "self-install" in m, (
                f"drift marker {m!r} too generic — risk of collision "
                f"with sibling Genesis patches on the same file"
            )

    def test_make_self_install_returns_none_when_vllm_tree_missing(
        self, monkeypatch,
    ):
        """On hosts without a resolvable vllm tree (torch-less CI,
        partial install), the TextPatcher factory must return None
        cleanly — apply() then falls back to the setattr-wrap step."""
        p = _import_patch()
        import sndr.engines.vllm.detection.guards as guards
        orig_root = guards.vllm_install_root
        orig_resolve = (
            __import__(
                "sndr.engines.vllm.detection.guards", fromlist=["resolve_vllm_file"],
            ).resolve_vllm_file
        )

        def fake_resolve(_):
            return None

        monkeypatch.setattr(
            "sndr.engines.vllm.detection.guards.resolve_vllm_file", fake_resolve,
        )
        try:
            patcher = p._make_self_install_text_patcher()
        finally:
            pass
        assert patcher is None


class TestValidatorWaiverTextPatch:
    """M2f — env-gated waiver of the dev371 cross-validator at
    vllm/config/vllm.py:1709-1715. The setattr+text-patch on
    utils.replace closes the replace-mediated entry point but NOT
    the direct `vllm_config.__post_init__()` call that EngineCore
    makes during `_perform_handshakes`. The validator is the single
    chokepoint that catches both paths, so this surgical text-patch
    converts the `raise ValueError(...)` to a `logger.warning(...)`
    behind the same opt-in env. Env-off behavior is byte-identical
    to upstream (the original raise is preserved verbatim in the
    else branch of the inserted env check).
    """

    def test_anchor_matches_dev371_validator_block_verbatim(self):
        """The anchor must match the dev371 upstream validator block
        byte-for-byte (16-space indent, exact f-string formatting,
        exact error message text). If upstream reformats this block,
        this test fails BEFORE the text-patch tries to land."""
        p = _import_patch()
        anchor = p._PN275_VALIDATOR_WAIVER_ANCHOR
        # Verify against the downloaded upstream source if available.
        import os
        upstream_path = "/tmp/dflash_investigation/vllm_config.py"
        if os.path.exists(upstream_path):
            upstream = open(upstream_path).read()
            assert anchor in upstream, (
                "anchor does NOT match upstream vllm/config/vllm.py "
                "verbatim — text-patch would fail to land"
            )
        # Structural sanity: anchor includes both the outer if and
        # the raise body
        assert "cudagraph_capture_sizes is not None:" in anchor
        assert "raise ValueError(" in anchor
        assert "customized max_cudagraph_capture_size" in anchor
        assert "cudagraph_capture_sizes(={valid_max_size})" in anchor

    def test_replacement_env_off_path_preserves_original_raise(self):
        """When env is unset/falsy, the inserted code falls through
        to the original raise — byte-identical to upstream behavior.
        This is the discipline requirement: env-off must be a no-op
        relative to the upstream validator."""
        p = _import_patch()
        replacement = p._PN275_VALIDATOR_WAIVER_REPLACEMENT
        # The else branch contains the exact original raise body
        assert "else:" in replacement
        # Original error message preserved verbatim in else branch
        assert "raise ValueError(" in replacement
        assert "customized max_cudagraph_capture_size" in replacement
        assert "should be consistent with the max value of" in replacement

    def test_replacement_env_on_path_uses_logger_warning(self):
        """When env is truthy, raise downgraded to logger.warning.
        Must use the existing `logger` symbol (vllm/config/vllm.py
        has its own logger; we don't import a new one)."""
        p = _import_patch()
        replacement = p._PN275_VALIDATOR_WAIVER_REPLACEMENT
        assert "logger.warning(" in replacement
        # Warning message identifies PN275 for log audit
        assert "[Genesis PN275]" in replacement

    def test_replacement_env_check_uses_pn275_env_flag(self):
        """The inserted env check must read the canonical PN275
        env flag (no aliases, no broad operator escape hatches)."""
        p = _import_patch()
        replacement = p._PN275_VALIDATOR_WAIVER_REPLACEMENT
        assert "GENESIS_ENABLE_PN275_DFLASH_MAX_CGS_ALIGN" in replacement
        # Standard truthy set (matches the rest of the patch's parsing)
        assert '"1", "true", "yes", "on"' in replacement

    def test_replacement_does_not_use_broad_except(self):
        """Discipline: M2f task spec forbids broad except wrapping
        around the validator. The env check is `os.environ.get()`
        which is non-throwing; no try/except needed and none allowed."""
        p = _import_patch()
        replacement = p._PN275_VALIDATOR_WAIVER_REPLACEMENT
        # No try/except around the validator logic itself
        assert "try:" not in replacement, (
            "validator waiver must NOT wrap the env check in try/except "
            "— operator's M2f discipline"
        )
        assert "except Exception" not in replacement
        assert "except:" not in replacement

    def test_validator_waiver_marker_is_pn275_specific(self):
        p = _import_patch()
        marker = p._GENESIS_PN275_VALIDATOR_WAIVER_MARKER
        assert "PN275" in marker
        # Distinct from the self-install marker so re-application
        # on the same file (config/utils.py vs config/vllm.py) doesn't
        # accidentally short-circuit
        assert marker != p._GENESIS_PN275_SELF_INSTALL_MARKER

    def test_make_validator_waiver_text_patcher_against_temp_tree(
        self, tmp_path, monkeypatch,
    ):
        """When given a temp vllm tree that contains the expected
        anchor in vllm/config/vllm.py, `_make_validator_waiver_text_patcher()`
        must return a TextPatcher with PN275-specific drift markers."""
        p = _import_patch()
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        # Synthesize a minimal vllm/config/vllm.py wrapping the anchor
        # in plausible surrounding lines so TextPatcher.find_anchor()
        # has full match context.
        (cfg_dir / "vllm.py").write_text(
            "# placeholder\n"
            "import logging\n"
            "logger = logging.getLogger(__name__)\n"
            "\n"
            "class VllmConfig:\n"
            "    def _set_cudagraph_sizes(self):\n"
            "        valid_max_size = 6\n"
            "        if True:\n"
            + p._PN275_VALIDATOR_WAIVER_ANCHOR
            + "\n            return\n"
        )
        import sndr.engines.vllm.detection.guards as guards
        import sndr.engines.vllm.detection.guards as _canon_guards
        orig = guards.vllm_install_root
        _orig_canon = _canon_guards.vllm_install_root
        # v12.x: the patch imports vllm_install_root from the shim for its
        # own gate, but resolve_vllm_file (canonical) calls the CANONICAL
        # vllm_install_root internally — mock both so the temp tree is seen.
        guards.vllm_install_root = _canon_guards.vllm_install_root = lambda: tmp_path
        try:
            patcher = p._make_validator_waiver_text_patcher()
        finally:
            guards.vllm_install_root = orig
            _canon_guards.vllm_install_root = _orig_canon

        assert patcher is not None
        for m in patcher.upstream_drift_markers:
            assert "PN275" in m

    def test_validator_waiver_returns_none_when_vllm_tree_missing(
        self, monkeypatch,
    ):
        """Symmetric to the self-install factory: missing vllm tree
        → None, no crash."""
        p = _import_patch()

        def fake_resolve(_):
            return None

        monkeypatch.setattr(
            "sndr.engines.vllm.detection.guards.resolve_vllm_file", fake_resolve,
        )
        patcher = p._make_validator_waiver_text_patcher()
        assert patcher is None


class TestSpawnSimulation:
    """End-to-end simulation of a freshly spawned worker process
    importing the (text-patched) `vllm/config/utils.py`. We simulate
    what would happen when the appended self-install block runs at
    module-import time in a fresh interpreter — independent of any
    parent-process state."""

    def test_spawn_worker_simulated_install(self, monkeypatch):
        """Simulate: worker spawns → re-imports vllm.config.utils →
        appended block runs → calls _genesis_pn275_install_at_import
        on the fresh module's globals → wrapper installs in worker's
        own module namespace. The wrap depends on NO parent state."""
        monkeypatch.setenv(
            "GENESIS_ENABLE_PN275_DFLASH_MAX_CGS_ALIGN", "1",
        )
        p = _import_patch()

        # Synthetic fresh worker's `vllm.config.utils` module globals
        def upstream_replace(instance, /, **kw):
            # Mimic upstream behavior: rebuild from dict
            return {"rebuilt": True, "kwargs": kw}

        worker_globals = {"replace": upstream_replace}

        # Worker imports config/utils.py → appended block fires →
        # calls the helper. We invoke it directly to simulate.
        result = p._genesis_pn275_install_at_import(worker_globals)
        assert result is True
        assert worker_globals["replace"] is not upstream_replace

        # The installed wrap must have the marker
        assert getattr(
            worker_globals["replace"], p._WRAPPED_ATTR, False,
        ) is True
        # Original is stashed for revert
        assert getattr(
            worker_globals["replace"], p._ORIGINAL_ATTR,
        ) is upstream_replace

    def test_spawn_worker_no_env_no_install(self, monkeypatch):
        """If the env isn't set in the spawned worker, the appended
        block short-circuits — no wrap, original `replace` stays."""
        monkeypatch.delenv(
            "GENESIS_ENABLE_PN275_DFLASH_MAX_CGS_ALIGN", raising=False,
        )
        p = _import_patch()

        def upstream_replace(instance, /, **kw):
            return None
        worker_globals = {"replace": upstream_replace}
        result = p._genesis_pn275_install_at_import(worker_globals)
        assert result is False
        # `replace` unchanged
        assert worker_globals["replace"] is upstream_replace
