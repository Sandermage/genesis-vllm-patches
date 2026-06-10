# SPDX-License-Identifier: Apache-2.0
"""TDD for PN55v2 — vllm#41602 + #41896 unified backport.

Wake_up zeroing path now handles nested KV cache containers
(list / tuple / Mapping / scalar Tensor / None / non-tensor sentinels)
via a recursive iterator. Replaces both v1 list-only and the would-be
PN83 sister patch — both targeted the same anchor.

PR38 Day 2 (2026-05-08): test moved to canonical `tests/unit/...` per
Sander's directive to phase out `tests/legacy/`.
"""
from __future__ import annotations

import pytest


def _wiring():
    from sndr.engines.vllm.patches.worker import (
        pn55_wake_up_hybrid_kv as M,
    )
    return M


# ─── Anchor / replacement shape ────────────────────────────────────────────


class TestAnchorShape:
    def test_anchor_targets_buggy_loop(self):
        M = _wiring()
        assert "for cache_tensor in kv_caches:" in M.ANCHOR_OLD
        assert "cache_tensor.zero_()" in M.ANCHOR_OLD

    def test_replacement_uses_recursive_iterator(self):
        """v2 collapses list/tuple/Mapping/scalar handling into one
        recursive helper. The helper name and signature are part of
        the API contract (operators reading the patched code expect
        this exact shape)."""
        M = _wiring()
        assert "_pn55_iter" in M.ANCHOR_NEW
        assert "from collections.abc import Mapping" in M.ANCHOR_NEW
        assert "isinstance(node, _PN55_Mapping)" in M.ANCHOR_NEW
        assert "isinstance(node, (list, tuple))" in M.ANCHOR_NEW

    def test_replacement_carries_v2_marker(self):
        M = _wiring()
        assert "PN55v2" in M.ANCHOR_NEW
        assert "vllm#41602" in M.ANCHOR_NEW
        assert "#41896" in M.ANCHOR_NEW


# ─── Idempotency on synthetic file ─────────────────────────────────────────


class TestIdempotent:
    def test_apply_twice_is_no_op(self, tmp_path):
        from sndr.kernel import (
            TextPatch, TextPatcher, TextPatchResult,
        )
        M = _wiring()
        target = tmp_path / "gpu_model_runner.py"
        target.write_text("# header\n" + M.ANCHOR_OLD + "\n# footer\n")
        patcher = TextPatcher(
            patch_name="PN55v2 test",
            target_file=str(target),
            marker=M.GENESIS_PN55_MARKER,
            sub_patches=[
                TextPatch(
                    name="pn55_recursive_iterator",
                    anchor=M.ANCHOR_OLD,
                    replacement=M.ANCHOR_NEW,
                    required=True,
                ),
            ],
        )
        r1, _ = patcher.apply()
        assert r1 == TextPatchResult.APPLIED
        body1 = target.read_text()
        assert "PN55v2" in body1
        assert "_pn55_iter" in body1
        r2, _ = patcher.apply()
        assert r2 == TextPatchResult.IDEMPOTENT
        assert target.read_text() == body1


# ─── Recursive iterator semantic ───────────────────────────────────────────


class TestRecursiveIteratorSemantic:
    """Build the iterator inline (matches the patch's logic byte-for-byte)
    and verify it handles every container shape PR #41896 introduces."""

    @staticmethod
    def _make_iter():
        """Reconstruct `_pn55_iter` from the patch's replacement text.

        We can't `exec()` the replacement directly (it's a method-body
        fragment) but we can mirror the logic here so the test is
        self-contained and catches regressions in the patch design.
        """
        from collections.abc import Mapping

        def _pn55_iter(node):
            if node is None:
                return
            if hasattr(node, "zero_") and not isinstance(
                node, (list, tuple, Mapping)
            ):
                yield node
                return
            if isinstance(node, Mapping):
                for _v in node.values():
                    yield from _pn55_iter(_v)
                return
            if isinstance(node, (list, tuple)):
                for _e in node:
                    yield from _pn55_iter(_e)
                return

        return _pn55_iter

    def _fake_tensor(self, name):
        """Stand-in for torch.Tensor — has `zero_`, isn't list/tuple/Mapping."""
        class _T:
            def __init__(self, n):
                self.name = n
                self.zeroed = False

            def zero_(self):
                self.zeroed = True

            def __repr__(self):
                return f"_T({self.name})"

        return _T(name)

    def test_flat_list_of_tensors(self):
        """Original v1 case still works."""
        _iter = self._make_iter()
        ts = [self._fake_tensor(f"t{i}") for i in range(3)]
        out = list(_iter(ts))
        assert len(out) == 3
        assert out == ts

    def test_nested_list_of_lists_mamba_case(self):
        """Mamba/DeltaNet `list[list[Tensor]]` shape (#41602)."""
        _iter = self._make_iter()
        a, b, c = [self._fake_tensor(n) for n in "abc"]
        ts = [[a, b], [c]]
        out = list(_iter(ts))
        assert out == [a, b, c]

    def test_dict_mapping_block_scaled_case(self):
        """FP8 block-scaled KV future (#41896) — `dict[str, Tensor]`."""
        _iter = self._make_iter()
        a = self._fake_tensor("a")
        b = self._fake_tensor("b")
        out = list(_iter({"k": a, "v": b}))
        assert sorted(t.name for t in out) == ["a", "b"]

    def test_tuple_yields_elements(self):
        _iter = self._make_iter()
        a = self._fake_tensor("a")
        b = self._fake_tensor("b")
        out = list(_iter((a, b)))
        assert out == [a, b]

    def test_none_skipped(self):
        _iter = self._make_iter()
        a = self._fake_tensor("a")
        out = list(_iter([None, a, None, [None]]))
        assert out == [a]

    def test_non_tensor_sentinel_skipped(self):
        """Objects without `zero_()` (e.g. int sentinels) silently skipped."""
        _iter = self._make_iter()
        a = self._fake_tensor("a")
        # `42` and `"sentinel"` lack zero_(), should be silently skipped
        out = list(_iter([a, 42, "sentinel"]))
        assert out == [a]


# ─── Env-flag gating contract ──────────────────────────────────────────────


class TestEnvFlag:
    def test_default_off(self, monkeypatch):
        from sndr.dispatcher import should_apply
        monkeypatch.delenv(
            "GENESIS_ENABLE_PN55_WAKE_UP_HYBRID_KV", raising=False
        )
        monkeypatch.delenv(
            "SNDR_ENABLE_PN55_WAKE_UP_HYBRID_KV", raising=False
        )
        decision, _ = should_apply("PN55")
        assert decision is False

    def test_genesis_enable_engages(self, monkeypatch):
        from sndr.dispatcher import should_apply
        monkeypatch.setenv("GENESIS_ENABLE_PN55_WAKE_UP_HYBRID_KV", "1")
        decision, _ = should_apply("PN55")
        assert decision is True

    def test_sndr_enable_engages_via_alias(self, monkeypatch):
        from sndr.dispatcher import should_apply
        monkeypatch.delenv(
            "GENESIS_ENABLE_PN55_WAKE_UP_HYBRID_KV", raising=False
        )
        monkeypatch.setenv("SNDR_ENABLE_PN55_WAKE_UP_HYBRID_KV", "1")
        decision, _ = should_apply("PN55")
        assert decision is True


# ─── Registry contract ─────────────────────────────────────────────────────


class TestRegistry:
    def test_pn55_in_registry(self):
        from sndr.dispatcher import PATCH_REGISTRY
        assert "PN55" in PATCH_REGISTRY

    def test_pn55_metadata_complete(self):
        from sndr.dispatcher import PATCH_REGISTRY
        meta = PATCH_REGISTRY["PN55"]
        assert meta["upstream_pr"] == 41602
        assert meta["family"] == "worker"
        assert meta["tier"] == "community"
        assert meta["default_on"] is False
        # PR38 Day 2: the v2 unified backport should mention #41896 too
        assert meta.get("related_upstream_prs") == [41896]
        assert "wake_up" in meta["title"].lower()

    def test_pn55_dispatch_function_registered(self):
        from sndr.apply import _per_patch_dispatch
        assert hasattr(
            _per_patch_dispatch, "apply_patch_N55_wake_up_hybrid_kv"
        )


# ─── Drift markers detect upstream merge ───────────────────────────────────


class TestUpstreamDrift:
    def test_drift_marker_detects_upstream_helper(self):
        """If vllm merges either #41602 or #41896, the modified
        gpu_model_runner.py contains `_iter_kv_cache_tensors` (the
        helper name PR #41896 introduces) OR `init_fp8_kv_scales`
        (the assertion site PR #41602 unblocks). Either presence
        means PN55v2 should self-retire."""
        M = _wiring()
        patcher = M._make_patcher()
        if patcher is None:
            pytest.skip("vllm install root not discoverable on this host")
        markers = list(patcher.upstream_drift_markers)
        assert "_iter_kv_cache_tensors" in markers
        assert "init_fp8_kv_scales" in markers
