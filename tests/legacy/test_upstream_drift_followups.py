# SPDX-License-Identifier: Apache-2.0
"""Tests pinning upstream-PR drift detection for active backports.

These regressions ensure the PATCH_REGISTRY-tracked upstream PRs that
matter to Genesis are wired correctly into each patch's drift markers
or auto-retire probes. If an upstream PR merges and we don't notice,
we either ship duplicate code or block on stale anchors. These tests
catch that class of drift at CI time.

Coverage:
- P3 drift markers include both FP16 (Genesis form) and FP32 (PR #39988
  form) cast staircases.
- P82 drift markers include the PR #40819 block-verify symbols
  (`use_block_verify`, `verify_method`, `SpecVerifyMethod`).
- P5 has an auto-retire probe that detects PR #39931 merge by looking
  for `TQFullAttentionSpec` + `_get_full_attention_layer_indices` in
  the upstream `vllm.model_executor.layers.quantization.turboquant.config`
  module.
"""
from __future__ import annotations

import sys
import types

import pytest


class TestP3DriftBothCastForms:
    def test_p3_drift_markers_cover_fp16_and_fp32(self):
        from sndr.engines.vllm.patches.attention.turboquant.p3_tq_bf16_cast import (
            UPSTREAM_DRIFT_MARKERS,
            _NEW,
        )
        markers = "\n".join(UPSTREAM_DRIFT_MARKERS)
        # The Genesis-original FP16 staircase form is what P3 itself emits
        # (see _NEW), so it must NOT be a drift marker — otherwise the detector
        # matches our own applied patch text and falsely fires "upstream
        # merged -> skip P3" (self-collision / PN369 false-skip class,
        # remediated 2026-06-11). Pin both halves of that invariant.
        assert "tl.float16).to(tl.float8e4b15)" in _NEW, (
            "P3 replacement must emit the FP16 staircase form"
        )
        assert "tl.float16).to(tl.float8e4b15)" not in markers, (
            "P3 must NOT drift-detect its own emitted FP16 staircase form — "
            "that would self-collide and false-skip P3"
        )
        assert "tl.float32).to(tl.float8e4b15)" in markers, (
            "P3 must drift-detect PR #39988 FP32 staircase form so the "
            "detector fires correctly when upstream merges with FP32"
        )
        assert "PR #39988" in markers


class TestP82DriftCovers40819:
    def test_p82_drift_markers_cover_block_verify(self):
        # P82's drift markers live inline in _make_patcher; read them
        # via the patcher object.
        from sndr.engines.vllm.patches.spec_decode.p82_sglang_acceptance_threshold import (
            _make_patcher,
        )
        # The patcher is constructed with a threshold value; pass any
        # valid float since we only need the markers list.
        patcher = _make_patcher(0.5)
        if patcher is None:
            pytest.skip("rejection_sampler.py not present in this env "
                        "— marker structure still in source but patcher "
                        "needs the file")

        markers_str = "\n".join(patcher.upstream_drift_markers)
        assert "use_block_verify" in markers_str, (
            "P82 must drift-detect PR #40819's `use_block_verify` flag "
            "so we know when the canonical SGLang block-verify rule "
            "lands in upstream rejection_sampler.py"
        )
        assert "verify_method" in markers_str
        assert "SpecVerifyMethod" in markers_str, (
            "PR #40819 adds SpecVerifyMethod to vllm/config/speculative.py "
            "— P82 must watch for that symbol too"
        )

    def test_p82_drift_markers_source_includes_40819_context(self):
        """The marker block must include a comment explaining why we
        watch these strings — operator + future contributor context."""
        import inspect
        from sndr.engines.vllm.patches.spec_decode import p82_sglang_acceptance_threshold as patch_82_sglang_acceptance_threshold
        src = inspect.getsource(patch_82_sglang_acceptance_threshold)
        # Comment explaining why these markers exist
        assert "#40819" in src or "PR #40819" in src
        assert "complementary" in src.lower() or "block-verify" in src


class TestP5AutoRetireProbe:
    """Genesis P5 should auto-skip when PR #39931 (JartX) is detected
    in the upstream vllm install.

    Probe-1 fix 2026-06-11 (preflight residual triage par.6): #39931
    (MERGED 2026-05-05) defines `TQFullAttentionSpec` in
    `vllm.v1.kv_cache_interface` (pristine :327 on pin
    0.22.1rc1.dev259) — NOT in turboquant/config.py, where only
    `_get_full_attention_layer_indices` lives (pristine :235). The old
    probe hasattr-ed turboquant.config for BOTH symbols, so the
    auto-skip never fired. These tests fake each symbol at its REAL
    home.
    """

    @staticmethod
    def _fake_kv_iface(monkeypatch, with_spec: bool):
        fake = types.ModuleType("vllm.v1.kv_cache_interface")
        if with_spec:
            fake.TQFullAttentionSpec = type("TQFullAttentionSpec", (), {})
        monkeypatch.setitem(
            sys.modules, "vllm.v1.kv_cache_interface", fake,
        )

    def test_p5_skips_when_pr39931_symbols_present(self, monkeypatch):
        """Inject the PR #39931 symbols at their REAL homes. P5 must
        SKIP (defer to upstream) and explain why."""
        # TQFullAttentionSpec lives in vllm.v1.kv_cache_interface
        self._fake_kv_iface(monkeypatch, with_spec=True)
        # _get_full_attention_layer_indices lives in turboquant.config
        fake_tq_cfg = types.ModuleType(
            "vllm.model_executor.layers.quantization.turboquant.config"
        )
        fake_tq_cfg._get_full_attention_layer_indices = lambda *a, **kw: []
        monkeypatch.setitem(
            sys.modules,
            "vllm.model_executor.layers.quantization.turboquant.config",
            fake_tq_cfg,
        )

        # Also stub out the early-exit checks so we hit the probe
        from sndr.engines.vllm.patches.kv_cache import p5_page_size as patch_5_page_size
        # GENESIS_DISABLE_P5 must NOT be set; nor the keep-override
        monkeypatch.delenv("GENESIS_DISABLE_P5", raising=False)
        monkeypatch.delenv("GENESIS_DISABLE_P5_AUTORETIRE", raising=False)

        status, reason = patch_5_page_size.apply()
        assert status == "skipped", (
            f"P5 must skip when PR #39931 symbols are present; got "
            f"{status} ({reason})"
        )
        assert "#39931" in reason, (
            "skip reason must reference PR #39931 so operators can find "
            "the institutional reasoning"
        )
        assert "TQFullAttentionSpec" in reason, (
            "skip reason must name the probed symbol"
        )

    def test_p5_does_not_defer_on_old_buggy_location(self, monkeypatch):
        """Regression pin for the original Probe-1 bug: the symbols
        present ONLY in turboquant.config (the old probed location —
        where TQFullAttentionSpec never actually lives) must NOT
        trigger the defer."""
        self._fake_kv_iface(monkeypatch, with_spec=False)
        fake_tq_cfg = types.ModuleType(
            "vllm.model_executor.layers.quantization.turboquant.config"
        )
        fake_tq_cfg.TQFullAttentionSpec = type("TQFullAttentionSpec", (), {})
        fake_tq_cfg._get_full_attention_layer_indices = lambda *a, **kw: []
        monkeypatch.setitem(
            sys.modules,
            "vllm.model_executor.layers.quantization.turboquant.config",
            fake_tq_cfg,
        )
        from sndr.engines.vllm.patches.kv_cache import p5_page_size as patch_5_page_size
        monkeypatch.delenv("GENESIS_DISABLE_P5", raising=False)
        monkeypatch.delenv("GENESIS_DISABLE_P5_AUTORETIRE", raising=False)

        _status, reason = patch_5_page_size.apply()
        assert "#39931" not in reason, (
            "TQFullAttentionSpec at the OLD (wrong) location must not "
            "trigger the #39931 defer — Probe 1 keys on "
            "vllm.v1.kv_cache_interface"
        )

    def test_p5_keep_override_forces_apply_path(self, monkeypatch):
        """GENESIS_DISABLE_P5_AUTORETIRE=1 (documented in the skip
        message, previously never read) must force the KEEP path even
        when both #39931 probes hit."""
        self._fake_kv_iface(monkeypatch, with_spec=True)
        fake_tq_cfg = types.ModuleType(
            "vllm.model_executor.layers.quantization.turboquant.config"
        )
        fake_tq_cfg._get_full_attention_layer_indices = lambda *a, **kw: []
        monkeypatch.setitem(
            sys.modules,
            "vllm.model_executor.layers.quantization.turboquant.config",
            fake_tq_cfg,
        )
        from sndr.engines.vllm.patches.kv_cache import p5_page_size as patch_5_page_size
        monkeypatch.delenv("GENESIS_DISABLE_P5", raising=False)
        monkeypatch.setenv("GENESIS_DISABLE_P5_AUTORETIRE", "1")

        _status, reason = patch_5_page_size.apply()
        assert "#39931" not in reason, (
            "operator-forced KEEP must not return the #39931 defer "
            "reason (P5 may still skip later for platform reasons)"
        )

    def test_p5_proceeds_when_pr39931_symbols_absent(self, monkeypatch):
        """Without the PR #39931 symbols, P5 must NOT auto-skip on the
        retire probe (it may skip for other reasons, but not this
        one)."""
        # Build a fake module WITHOUT the PR #39931 symbols
        fake_tq_cfg = types.ModuleType(
            "vllm.model_executor.layers.quantization.turboquant.config"
        )
        # Note: deliberately DO NOT set TQFullAttentionSpec or the helper
        monkeypatch.setitem(
            sys.modules,
            "vllm.model_executor.layers.quantization.turboquant.config",
            fake_tq_cfg,
        )

        from sndr.engines.vllm.patches.kv_cache import p5_page_size as patch_5_page_size
        monkeypatch.delenv("GENESIS_DISABLE_P5", raising=False)

        _status, reason = patch_5_page_size.apply()
        # Should NOT mention the auto-retire reason text
        assert "#39931" not in reason, (
            "When #39931 symbols absent, P5 must not return the "
            "auto-retire reason"
        )

    def test_p5_probe_failure_is_non_fatal(self, monkeypatch):
        """If the probe itself raises (e.g. import infrastructure
        broken), P5 must NOT fail — fall through to normal apply."""
        from sndr.engines.vllm.patches.kv_cache import p5_page_size as patch_5_page_size

        # Force the probe import to raise
        import importlib
        original_im = importlib.import_module

        def _raising(name, *a, **kw):
            if name == "vllm.model_executor.layers.quantization.turboquant.config":
                raise RuntimeError("simulated infra failure")
            return original_im(name, *a, **kw)

        monkeypatch.setattr(importlib, "import_module", _raising)
        monkeypatch.delenv("GENESIS_DISABLE_P5", raising=False)

        # Should not raise, should reach a skip-or-applied status
        status, reason = patch_5_page_size.apply()
        assert status in ("applied", "skipped", "failed")
        # The probe failure must NOT show up as the reason — we should
        # have fallen through to normal apply path.
        assert "simulated infra failure" not in reason
