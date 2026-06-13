# SPDX-License-Identifier: Apache-2.0
"""TDD for PN55v2 — vllm#41602 + #41896 unified backport.

Wake_up zeroing path handles nested KV cache containers
(list / tuple / Mapping / scalar Tensor / None) via a recursive
iterator. None sentinels are skipped silently; non-tensor leaves are
warn-skipped (upstream #44778 raises TypeError instead — deliberate
divergence, see the wiring docstring).

PR38 Day 2 (2026-05-08): test moved to canonical `tests/unit/...` per
Sander's directive to phase out `tests/legacy/`.

2026-06-11 hygiene pass (roadmap chunk-5 Theme C, #44778 review): the
hand-mirrored `_make_iter` copy of the iterator was REPLACED with the
exec-patched-text technique adopted from #44778's regression test
(`tests/v1/worker/test_gpu_model_runner_fp8_wake_up.py` runs the real
patched method CPU-only): apply the real TextPatcher to a pinned
fixture, exec the patched source, and assert behavior on the resulting
class. A mirrored copy can drift from ANCHOR_NEW without failing; the
exec'd text cannot.
"""
from __future__ import annotations

import logging

import pytest


def _wiring():
    from sndr.engines.vllm.patches.worker import (
        pn55_wake_up_hybrid_kv as M,
    )
    return M


# Synthetic mirror of vllm/v1/worker/gpu_model_runner.py on pin
# 0.22.1rc1.dev259+g303916e93 — just enough context for PN55's anchor:
# the module-level logger (`logger = init_logger(__name__)` in the real
# file) and the buggy flat zero loop inside init_fp8_kv_scales
# (pristine lines 947-950, anchor count==1, byte-verified 2026-06-11).
PIN_GPU_MODEL_RUNNER = (
    "import logging\n"
    "\n"
    "logger = logging.getLogger(__name__)\n"
    "\n"
    "\n"
    "class GPUModelRunner:\n"
    "    def init_fp8_kv_scales(self) -> None:\n"
    "        kv_caches = getattr(self, \"kv_caches\", [])\n"
    "        for cache_tensor in kv_caches:\n"
    "            if cache_tensor is not None:\n"
    "                cache_tensor.zero_()\n"
)


class _FakeTensor:
    """Stand-in for torch.Tensor — has `zero_`, isn't list/tuple/Mapping."""

    def __init__(self, name: str = "t"):
        self.name = name
        self.zeroed = False

    def zero_(self):
        self.zeroed = True

    def __repr__(self):
        return f"_FakeTensor({self.name})"


# ─── Helpers (exec-patched-text technique, per vllm#44778's test) ──────────


def _install_fixture(tmp_path, monkeypatch):
    """Write the pinned fixture and route the wiring at it.

    PN55's registry gate is owned by the registry agent — force the
    dispatcher gate open for apply-semantics tests (same seam as the
    PN371 test)."""
    M = _wiring()
    target = tmp_path / "gpu_model_runner.py"
    target.write_text(PIN_GPU_MODEL_RUNNER, encoding="utf-8")
    monkeypatch.setattr(M, "resolve_vllm_file", lambda rel: str(target))
    monkeypatch.setattr(M, "vllm_install_root", lambda: str(tmp_path))

    import sndr.dispatcher as dispatcher

    monkeypatch.setattr(
        dispatcher, "should_apply", lambda pid: (True, "test override")
    )
    monkeypatch.setattr(dispatcher, "log_decision", lambda *a, **k: None)
    return target


def _exec_runner_class(source: str, filename: str):
    """Exec a (patched or pristine) fixture text → its GPUModelRunner."""
    ns: dict = {"__name__": "gpu_model_runner_synthetic"}
    exec(compile(source, filename, "exec"), ns)
    return ns["GPUModelRunner"]


def _load_patched_runner(tmp_path, monkeypatch):
    """apply() the real patch against the pinned fixture, then exec the
    PATCHED source text — not a hand-mirrored copy of the iterator."""
    M = _wiring()
    target = _install_fixture(tmp_path, monkeypatch)
    status, reason = M.apply()
    assert status == "applied", reason
    return _exec_runner_class(
        target.read_text(encoding="utf-8"), "gpu_model_runner_patched.py"
    )


# ─── Pinned fixture fidelity ───────────────────────────────────────────────


class TestPinnedFixture:
    def test_anchor_matches_pinned_fixture_bytes(self):
        """ANCHOR_OLD must be a byte-exact substring of the pinned
        upstream text (the fixture mirrors pristine 947-950 on pin
        0.22.1rc1.dev259) — if ANCHOR_OLD drifts, this catches it."""
        M = _wiring()
        assert M.ANCHOR_OLD in PIN_GPU_MODEL_RUNNER

    def test_unpatched_fixture_reproduces_upstream_bug(self):
        """The pre-fix flat loop AttributeErrors on the nested
        list-of-lists shape (#41602 Mamba / #44778 PP>1 + fp8 KV) —
        proves the fixture actually exercises the bug class."""
        runner_cls = _exec_runner_class(
            PIN_GPU_MODEL_RUNNER, "gpu_model_runner_pristine.py"
        )
        runner = runner_cls()
        runner.kv_caches = [[_FakeTensor("a")], [_FakeTensor("b")]]
        with pytest.raises(AttributeError):
            runner.init_fp8_kv_scales()


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

    def test_replacement_warns_on_non_tensor_leaf(self):
        """2026-06-11 hygiene: non-tensor leaves are no longer silently
        dropped — the walker emits a log.warning so an unexpected cache
        layout surfaces in docker logs instead of vanishing."""
        M = _wiring()
        assert "logger.warning" in M.ANCHOR_NEW


# ─── Apply semantics on the pinned fixture ─────────────────────────────────


class TestApplySemantics:
    def test_apply_patches_fixture_and_plants_marker(
        self, tmp_path, monkeypatch
    ):
        M = _wiring()
        target = _install_fixture(tmp_path, monkeypatch)
        status, reason = M.apply()
        assert status == "applied", reason
        body = target.read_text(encoding="utf-8")
        assert M.GENESIS_PN55_MARKER in body
        assert "_pn55_iter" in body

    def test_apply_twice_is_idempotent(self, tmp_path, monkeypatch):
        M = _wiring()
        target = _install_fixture(tmp_path, monkeypatch)
        status1, _ = M.apply()
        assert status1 == "applied"
        body1 = target.read_text(encoding="utf-8")
        status2, reason2 = M.apply()
        assert status2 == "applied"
        assert "idempotent" in reason2
        assert target.read_text(encoding="utf-8") == body1

    def test_v1_marker_blocks_in_place_upgrade(self, tmp_path, monkeypatch):
        """A file patched by the old v1 list-only replacement must NOT
        be re-patched in place — apply() reports the diagnostic skip."""
        M = _wiring()
        target = _install_fixture(tmp_path, monkeypatch)
        target.write_text(
            f"# {M.GENESIS_PN55_V1_MARKER}\n" + PIN_GPU_MODEL_RUNNER,
            encoding="utf-8",
        )
        status, reason = M.apply()
        assert status == "skipped"
        assert "v1" in reason.lower()


# ─── Behavior of the exec'd PATCHED text (replaces the mirrored copy) ──────


class TestExecPatchedBehavior:
    """Exec the actually-patched fixture source and drive
    init_fp8_kv_scales through every container shape #41602/#41896/#44778
    cover — higher fidelity than the previous hand-mirrored iterator
    (which could drift from ANCHOR_NEW without failing)."""

    def test_flat_list_of_tensors(self, tmp_path, monkeypatch):
        """Original v1 case still works."""
        runner_cls = _load_patched_runner(tmp_path, monkeypatch)
        runner = runner_cls()
        ts = [_FakeTensor(f"t{i}") for i in range(3)]
        runner.kv_caches = ts
        runner.init_fp8_kv_scales()
        assert all(t.zeroed for t in ts)

    def test_nested_list_of_lists_mamba_pp_case(self, tmp_path, monkeypatch):
        """Mamba/DeltaNet `list[list[Tensor]]` (#41602) — also the exact
        PP>1 + fp8 KV shape #44778 reports live on Qwen3.6-27B."""
        runner_cls = _load_patched_runner(tmp_path, monkeypatch)
        runner = runner_cls()
        a, b, c = (_FakeTensor(n) for n in "abc")
        runner.kv_caches = [[a, b], [c]]
        runner.init_fp8_kv_scales()
        assert a.zeroed and b.zeroed and c.zeroed

    def test_dict_mapping_block_scaled_case(self, tmp_path, monkeypatch):
        """FP8 block-scaled KV future (#41896) — `dict[str, Tensor]`."""
        runner_cls = _load_patched_runner(tmp_path, monkeypatch)
        runner = runner_cls()
        a, b = _FakeTensor("a"), _FakeTensor("b")
        runner.kv_caches = [{"k": a, "v": b}]
        runner.init_fp8_kv_scales()
        assert a.zeroed and b.zeroed

    def test_tuple_descends_to_elements(self, tmp_path, monkeypatch):
        runner_cls = _load_patched_runner(tmp_path, monkeypatch)
        runner = runner_cls()
        a, b = _FakeTensor("a"), _FakeTensor("b")
        runner.kv_caches = (a, b)
        runner.init_fp8_kv_scales()
        assert a.zeroed and b.zeroed

    def test_mixed_containers_resolve_correctly(self, tmp_path, monkeypatch):
        runner_cls = _load_patched_runner(tmp_path, monkeypatch)
        runner = runner_cls()
        a, b, c = (_FakeTensor(n) for n in "abc")
        runner.kv_caches = [a, [b], (c,)]
        runner.init_fp8_kv_scales()
        assert a.zeroed and b.zeroed and c.zeroed

    def test_none_skipped_silently(self, tmp_path, monkeypatch, caplog):
        """None sentinels are an EXPECTED layout — no warning noise."""
        runner_cls = _load_patched_runner(tmp_path, monkeypatch)
        runner = runner_cls()
        a = _FakeTensor("a")
        runner.kv_caches = [None, a, None, [None]]
        with caplog.at_level(logging.WARNING):
            runner.init_fp8_kv_scales()
        assert a.zeroed
        assert not [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and "PN55" in r.getMessage()
        ]

    def test_non_tensor_leaf_warn_skipped(self, tmp_path, monkeypatch, caplog):
        """2026-06-11 hygiene: objects without `zero_()` are skipped WITH
        a log.warning naming the offending type (upstream #44778 raises
        TypeError here; PN55 deliberately stays non-fatal — a wedged
        wake_up on PROD is worse than one stale cache entry)."""
        runner_cls = _load_patched_runner(tmp_path, monkeypatch)
        runner = runner_cls()
        a = _FakeTensor("a")
        runner.kv_caches = [a, 42]
        with caplog.at_level(logging.WARNING):
            runner.init_fp8_kv_scales()
        assert a.zeroed
        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "PN55" in r.getMessage()
        ]
        assert len(warnings) == 1
        assert "int" in warnings[0].getMessage()


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
        # PR38 Day 2: the v2 unified backport should mention #41896 too.
        # 2026-06-11 hygiene (roadmap chunk-5 Theme C): #44778 is the
        # downstream backport of #41896 — functionally duplicate of
        # PN55v2, tracked so the upstream audit watches its merge state.
        # RED until the wave-2 registry delta lands (registry.py is
        # owned by the registry agent — see the task's returned delta).
        assert meta.get("related_upstream_prs") == [41896, 44778]
        assert "wake_up" in meta["title"].lower()

    def test_pn55_dispatch_function_registered(self):
        from sndr.apply import _per_patch_dispatch
        assert hasattr(
            _per_patch_dispatch, "apply_patch_N55_wake_up_hybrid_kv"
        )


# ─── Drift markers detect upstream merge ───────────────────────────────────


class TestUpstreamDrift:
    def test_drift_marker_detects_upstream_helper(
        self, tmp_path, monkeypatch
    ):
        """If vllm merges #41602 / #41896 / #44778, the modified
        gpu_model_runner.py contains `_iter_kv_cache_tensors` (the
        helper name both #41896 and its #44778 backport introduce).
        That presence means PN55v2 should self-retire.

        2026-06-11 narrowing (preflight residual triage §2):
        `init_fp8_kv_scales` was REMOVED from the markers — the name
        now exists natively in every pin via merged vllm#28783
        (FP8 KV + sleep(level=2) fix, MERGED 2025-11-30), while
        #41602/#41896 are both still OPEN (gh-verified 2026-06-11).
        Keeping it caused a false "upstream merged" self-retire on
        every post-#28783 pin."""
        M = _wiring()
        _install_fixture(tmp_path, monkeypatch)
        patcher = M._make_patcher()
        assert patcher is not None
        markers = list(patcher.upstream_drift_markers)
        assert "_iter_kv_cache_tensors" in markers
        assert "init_fp8_kv_scales" not in markers, (
            "PN55: 'init_fp8_kv_scales' name-collides with merged "
            "vllm#28783 — false self-retire on current pins"
        )

    def test_drift_markers_not_in_own_replacement(
        self, tmp_path, monkeypatch
    ):
        """PN369 false-skip class (tools/lint_drift_markers.py): a
        non-[Genesis marker that is a substring of our own replacement
        text reads back as 'upstream merged' on the next boot. The
        2026-06-11 warn-skip addition must NOT mention the upstream
        helper name inside the replacement."""
        M = _wiring()
        _install_fixture(tmp_path, monkeypatch)
        patcher = M._make_patcher()
        assert patcher is not None
        for dm in patcher.upstream_drift_markers:
            if dm.startswith("[Genesis"):
                continue  # defended convention — self-marker exempt
            for sp in patcher.sub_patches:
                assert dm not in (sp.replacement or ""), (
                    f"drift marker {dm!r} collides with own replacement "
                    f"text of sub-patch {sp.name!r}"
                )
