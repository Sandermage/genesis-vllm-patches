# SPDX-License-Identifier: Apache-2.0
"""TDD for PN77 FP8 lm_head compressor (Phase E MVP).

Quality gate: cosine_sim(weight_bf16, decompress(compress(weight_bf16))) ≥ 0.999
on synthetic weights matching real model dynamic range.

Real-weight validation deferred — requires loaded checkpoint, runs in
integration suite (not unit tests).
"""
from __future__ import annotations

import pytest
import torch

from sndr.engines.vllm.kernels_legacy import lm_head_fp8_compressor as lhc


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.flatten().to(torch.float32)
    bf = b.flatten().to(torch.float32)
    num = torch.dot(af, bf)
    denom = af.norm() * bf.norm() + 1e-12
    return float(num / denom)


# ─── Per-channel scale computation ─────────────────────────────────────


class TestComputeScale:
    def test_returns_per_row_scale(self):
        w = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=torch.bfloat16)
        scale = lhc.compute_per_channel_scale(w)
        assert scale.shape == (2,)
        # row 0 max = 3.0, row 1 max = 30.0; scale = max / 448
        assert abs(float(scale[0]) - (3.0 / 448.0)) < 1e-3
        assert abs(float(scale[1]) - (30.0 / 448.0)) < 1e-3

    def test_clamps_to_floor_for_zero_row(self):
        w = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=torch.bfloat16)
        scale = lhc.compute_per_channel_scale(w)
        # zero-row scale clamped to 1e-12
        assert float(scale[0]) >= lhc._SCALE_FLOOR / 2

    def test_rejects_non_2d_input(self):
        w = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="2D"):
            lhc.compute_per_channel_scale(w)

    def test_rejects_non_positive_fp8_max(self):
        w = torch.randn(4, 4, dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="fp8_max"):
            lhc.compute_per_channel_scale(w, fp8_max=0)
        with pytest.raises(ValueError, match="fp8_max"):
            lhc.compute_per_channel_scale(w, fp8_max=-1)


# ─── Compression / decompression roundtrip ────────────────────────────


class TestCompressDecompress:
    def test_compress_returns_fp8_dtype(self):
        w = torch.randn(8, 16, dtype=torch.bfloat16)
        w_fp8, scale = lhc.compress(w)
        assert w_fp8.dtype == torch.float8_e4m3fn
        assert scale.dtype == torch.float32
        assert w_fp8.shape == w.shape
        assert scale.shape == (8,)

    def test_decompress_returns_target_dtype(self):
        w = torch.randn(8, 16, dtype=torch.bfloat16)
        w_fp8, scale = lhc.compress(w)
        w_back = lhc.decompress(w_fp8, scale, output_dtype=torch.bfloat16)
        assert w_back.dtype == torch.bfloat16
        assert w_back.shape == w.shape

    def test_decompress_default_dtype_bf16(self):
        w = torch.randn(4, 4, dtype=torch.bfloat16)
        w_fp8, scale = lhc.compress(w)
        w_back = lhc.decompress(w_fp8, scale)
        assert w_back.dtype == torch.bfloat16

    def test_does_not_mutate_input(self):
        w = torch.randn(8, 16, dtype=torch.bfloat16)
        w_orig = w.clone()
        _ = lhc.compress(w)
        assert torch.equal(w, w_orig)


# ─── Quality gate: cosine_sim ≥ 0.999 ─────────────────────────────────


class TestQualityGate:
    def _make_realistic_weights(self, vocab=4096, hidden=2048, seed=42):
        """Synthetic weights matching real lm_head dynamic range.

        Real lm_head weights have:
        - vocab rows with HIGHLY varying norms (zipfian token frequency)
        - per-row values ~Gaussian with std ~0.02-0.05
        - some rare tokens may have near-zero rows
        """
        torch.manual_seed(seed)
        # Normal init * per-row magnitude (zipfian-ish)
        w = torch.randn(vocab, hidden, dtype=torch.float32) * 0.02
        # Scale rows by zipfian-ish factor
        row_factor = 1.0 / torch.arange(1, vocab + 1, dtype=torch.float32).pow(0.3)
        w = w * row_factor.unsqueeze(1)
        return w.to(torch.bfloat16)

    def test_cosine_sim_above_999_on_realistic_weights(self):
        w = self._make_realistic_weights()
        w_fp8, scale = lhc.compress(w)
        w_back = lhc.decompress(w_fp8, scale)
        cs = _cosine_sim(w, w_back)
        assert cs >= 0.999, f"cosine_sim {cs} < 0.999 quality gate"

    def test_cosine_sim_above_999_with_extreme_dynamic_range(self):
        """Some rows are tiny (near zero), some are large — common in lm_head."""
        torch.manual_seed(7)
        vocab, hidden = 2048, 1024
        w = torch.randn(vocab, hidden, dtype=torch.float32) * 0.02
        # Force first 100 rows to have huge scale
        w[:100] *= 50.0
        # Force last 100 rows to be near-zero
        w[-100:] *= 0.001
        w = w.to(torch.bfloat16)

        w_fp8, scale = lhc.compress(w)
        w_back = lhc.decompress(w_fp8, scale)
        cs = _cosine_sim(w, w_back)
        assert cs >= 0.999, f"cosine_sim {cs} < 0.999 on extreme range"

    def test_zero_rows_preserved_within_tolerance(self):
        """All-zero rows should decompress to all-zeros (or very close)."""
        w = torch.zeros(8, 16, dtype=torch.bfloat16)
        w[2:6] = torch.randn(4, 16, dtype=torch.bfloat16) * 0.01
        w_fp8, scale = lhc.compress(w)
        w_back = lhc.decompress(w_fp8, scale)
        # Zero rows should remain near-zero
        for i in [0, 1, 6, 7]:
            assert w_back[i].abs().max() < 1e-5, (
                f"zero row {i} drifted to {w_back[i].abs().max()}"
            )

    def test_logit_distance_bounded_on_synthetic_input(self):
        """Synthetic forward: x @ W.T → cast roundtrip W → x @ W'.T,
        bound the relative error in logits.

        NOTE: pure-random Gaussian input is the WORST-CASE scenario; real
        lm_head sees structured hidden states from prior layers with much
        lower per-position variance, so the production drift is expected
        to be 2-3× lower than this synthetic test.

        Threshold 5% is for the synthetic worst-case; integration test on
        real workload uses cosine_sim ≥ 0.999 + tool-call 10/10 + 1K-token
        prefix-cache fixture as the production quality gate (per upstream
        PR #35696 design discussion).
        """
        torch.manual_seed(0)
        vocab, hidden = 2048, 512
        w = torch.randn(vocab, hidden, dtype=torch.bfloat16) * 0.02
        x = torch.randn(8, hidden, dtype=torch.bfloat16) * 0.1

        w_fp8, scale = lhc.compress(w)
        w_back = lhc.decompress(w_fp8, scale)

        # Logits with original weight
        logits_orig = (x.float() @ w.float().T)
        # Logits with roundtripped weight
        logits_back = (x.float() @ w_back.float().T)

        rel_err = (logits_orig - logits_back).abs().mean() / (
            logits_orig.abs().mean() + 1e-8
        )
        # Synthetic worst-case bound; real workload expected 2-3× lower
        assert float(rel_err) < 0.05, (
            f"logit relative error {float(rel_err):.4f} > 5% on synthetic "
            f"worst-case input (alarming — investigate quantization)"
        )


# ─── Savings estimator ────────────────────────────────────────────────


class TestSavingsEstimator:
    def test_qwen3_27b_savings(self):
        """Qwen3.6-27B: vocab=248320, hidden=5120, BF16 — expect ~1.18 GiB total."""
        savings = lhc.estimate_savings_bytes(
            vocab_size=248320, hidden_size=5120, src_dtype=torch.bfloat16,
        )
        # 248320 * 5120 * 2 (bf16) = 2_541_158_400
        # 248320 * 5120 * 1 (fp8) = 1_270_579_200
        # 248320 * 4 (scale) = 993_280
        # savings = 2_541_158_400 - 1_270_579_200 - 993_280 = 1_269_585_920 ≈ 1211 MiB
        savings_mib = savings / (1024 * 1024)
        assert 1200 < savings_mib < 1220, (
            f"expected ~1211 MiB savings, got {savings_mib:.1f} MiB"
        )

    def test_qwen3_35b_savings(self):
        """Qwen3.6-35B: vocab=248320, hidden=2048, BF16 — expect ~485 MiB total."""
        savings = lhc.estimate_savings_bytes(
            vocab_size=248320, hidden_size=2048, src_dtype=torch.bfloat16,
        )
        savings_mib = savings / (1024 * 1024)
        assert 480 < savings_mib < 490, (
            f"expected ~485 MiB savings, got {savings_mib:.1f} MiB"
        )

    def test_savings_positive_for_bf16_input(self):
        savings = lhc.estimate_savings_bytes(1024, 512, torch.bfloat16)
        assert savings > 0

    def test_savings_negative_or_small_for_fp16_input(self):
        """fp16 → fp8 still saves but less prominently."""
        savings = lhc.estimate_savings_bytes(1024, 512, torch.float16)
        # fp16 = 2 bytes, same saving structure
        assert savings > 0


# ─── Edge cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_row(self):
        w = torch.randn(1, 16, dtype=torch.bfloat16)
        w_fp8, scale = lhc.compress(w)
        w_back = lhc.decompress(w_fp8, scale)
        assert w_back.shape == w.shape

    def test_decompress_rejects_shape_mismatch(self):
        w_fp8 = torch.zeros(8, 16, dtype=torch.float8_e4m3fn)
        bad_scale = torch.zeros(7, dtype=torch.float32)
        with pytest.raises(ValueError, match="scale shape"):
            lhc.decompress(w_fp8, bad_scale)

    def test_decompress_rejects_non_2d_weight(self):
        bad_w = torch.zeros(16, dtype=torch.float8_e4m3fn)
        scale = torch.zeros(16, dtype=torch.float32)
        with pytest.raises(ValueError, match="2D"):
            lhc.decompress(bad_w, scale)


# ─── Top-level integration entry: maybe_compress_lm_head_to_fp8 ───────


class _FakeLayer(torch.nn.Module):
    """Mock lm_head — single Parameter for easy testing."""
    def __init__(self, vocab=64, hidden=32, dtype=torch.bfloat16):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(vocab, hidden, dtype=dtype) * 0.02
        )


class _FakeModel(torch.nn.Module):
    """Mock model with lm_head + optional embed_tokens (for tied-weights test)."""
    def __init__(self, vocab=64, hidden=32, tied=False, dtype=torch.bfloat16):
        super().__init__()
        self.lm_head = _FakeLayer(vocab, hidden, dtype)
        if tied:
            # Tied: lm_head and embed_tokens share storage
            self.model = torch.nn.Module()
            self.model.embed_tokens = _FakeLayer(vocab, hidden, dtype)
            # Force same data_ptr by sharing the Parameter object
            self.model.embed_tokens.weight = self.lm_head.weight
        else:
            self.model = torch.nn.Module()
            self.model.embed_tokens = _FakeLayer(vocab, hidden, dtype)


class TestMaybeCompressLmHead:
    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        monkeypatch.delenv(lhc._PN77_ENV, raising=False)
        yield

    def test_env_off_returns_skipped(self):
        m = _FakeModel()
        status, reason = lhc.maybe_compress_lm_head_to_fp8(m)
        assert status == "skipped"
        assert "opt-in" in reason
        assert m.lm_head.weight.dtype == torch.bfloat16  # unchanged

    def test_env_on_compresses(self, monkeypatch):
        monkeypatch.setenv(lhc._PN77_ENV, "1")
        m = _FakeModel(vocab=128, hidden=64)
        orig_shape = m.lm_head.weight.shape
        status, reason = lhc.maybe_compress_lm_head_to_fp8(m)
        assert status == "applied", reason
        assert m.lm_head.weight.dtype == torch.float8_e4m3fn
        assert m.lm_head.weight.shape == orig_shape
        assert hasattr(m.lm_head, lhc._PN77_SCALE_ATTR)
        assert getattr(m.lm_head, lhc._PN77_MARKER) is True

    def test_idempotent_second_call(self, monkeypatch):
        monkeypatch.setenv(lhc._PN77_ENV, "1")
        m = _FakeModel()
        status1, _ = lhc.maybe_compress_lm_head_to_fp8(m)
        assert status1 == "applied"
        status2, reason2 = lhc.maybe_compress_lm_head_to_fp8(m)
        assert status2 == "skipped"
        assert "marker" in reason2.lower()

    def test_skip_tied_embeddings(self, monkeypatch):
        monkeypatch.setenv(lhc._PN77_ENV, "1")
        m = _FakeModel(tied=True)
        status, reason = lhc.maybe_compress_lm_head_to_fp8(m)
        assert status == "skipped"
        assert "tied" in reason.lower()
        # lm_head NOT modified
        assert m.lm_head.weight.dtype == torch.bfloat16

    def test_skip_already_fp8(self, monkeypatch):
        monkeypatch.setenv(lhc._PN77_ENV, "1")
        m = _FakeModel()
        # Mock pre-compressed FP8 lm_head
        m.lm_head.weight = torch.nn.Parameter(
            torch.zeros(64, 32, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        status, reason = lhc.maybe_compress_lm_head_to_fp8(m)
        assert status == "skipped"
        assert "fp8" in reason.lower() or "already" in reason.lower()

    def test_skip_unsupported_dtype(self, monkeypatch):
        monkeypatch.setenv(lhc._PN77_ENV, "1")
        m = _FakeModel(dtype=torch.float32)  # fp32 not supported
        status, reason = lhc.maybe_compress_lm_head_to_fp8(m)
        assert status == "skipped"

    def test_skip_no_lm_head(self, monkeypatch):
        monkeypatch.setenv(lhc._PN77_ENV, "1")
        m = torch.nn.Module()  # no lm_head attr
        status, reason = lhc.maybe_compress_lm_head_to_fp8(m)
        assert status == "skipped"
        assert "lm_head" in reason

    def test_failed_status_on_internal_error(self, monkeypatch):
        """If compress() raises, return ('failed', reason) — never raise."""
        monkeypatch.setenv(lhc._PN77_ENV, "1")

        def _broken_compress(*a, **kw):
            raise RuntimeError("synthetic failure")

        monkeypatch.setattr(lhc, "compress", _broken_compress)
        m = _FakeModel()
        status, reason = lhc.maybe_compress_lm_head_to_fp8(m)
        assert status == "failed"
        assert "synthetic" in reason

    def test_preserves_weight_loader_attribute(self, monkeypatch):
        """CRITICAL: vllm's set_weight_attrs attaches `weight_loader`,
        `input_dim`, `output_dim` etc to the Parameter. After PN77 replace,
        the new Parameter MUST inherit these — otherwise vllm's loader
        falls back to default_weight_loader which doesn't TP-shard,
        causing AssertionError on any subsequent load."""
        monkeypatch.setenv(lhc._PN77_ENV, "1")
        m = _FakeModel(vocab=64, hidden=32)
        # Simulate vllm's set_weight_attrs: attach callback + dims to weight
        def _fake_loader(p, w):
            pass
        m.lm_head.weight.weight_loader = _fake_loader
        m.lm_head.weight.input_dim = 1
        m.lm_head.weight.output_dim = 0

        status, _ = lhc.maybe_compress_lm_head_to_fp8(m)
        assert status == "applied"
        # Attrs preserved on new Parameter
        assert hasattr(m.lm_head.weight, "weight_loader")
        assert m.lm_head.weight.weight_loader is _fake_loader
        assert m.lm_head.weight.input_dim == 1
        assert m.lm_head.weight.output_dim == 0


# ─── _find_embed_tokens path resolver ─────────────────────────────────


class TestFindEmbedTokens:
    def test_finds_via_model_embed_tokens(self):
        m = _FakeModel()
        result = lhc._find_embed_tokens(m)
        assert result is m.model.embed_tokens

    def test_returns_none_when_missing(self):
        m = torch.nn.Module()
        result = lhc._find_embed_tokens(m)
        assert result is None

    def test_finds_top_level_embed_tokens(self):
        m = torch.nn.Module()
        m.embed_tokens = _FakeLayer()
        result = lhc._find_embed_tokens(m)
        assert result is m.embed_tokens
