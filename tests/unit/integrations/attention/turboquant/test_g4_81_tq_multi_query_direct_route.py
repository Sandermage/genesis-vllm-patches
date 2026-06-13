# SPDX-License-Identifier: Apache-2.0
"""G4_81 — TQ multi-query DIRECT decode routing (vllm#45144 blueprint).

Torch-less unit tests. The patch monkey-patches
``TurboQuantAttentionImpl.forward`` to intercept uniform K+1 spec-verify
batches and route them through the TQ grouped/split-KV decode kernel
directly (synthetic per-token expansion), bypassing the per-request
``_prefill_attention`` continuation path that rejects/degrades them.

Tests exercise the pure helpers against fakes (no vllm / no torch):
batch classification predicate (both prefill-classified and
decode-classified spec-verify shapes), launcher capability inspection,
route refusal logic, synthetic seq-len reference math, and the
apply()/revert() rebind mechanics against an injected fake vllm module
tree. Tensor-level synth-arg equivalence is covered by a torch-gated
test that skips when torch is unavailable (CI collection stays
torch-less).
"""
from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest

MODULE = (
    "sndr.engines.vllm.patches.attention.turboquant."
    "g4_81_tq_multi_query_direct_route"
)

ENV_FLAG = "GENESIS_ENABLE_G4_81_TQ_MQ_DIRECT_ROUTE"


@pytest.fixture
def mod():
    m = importlib.import_module(MODULE)
    # Reset module-level apply state between tests.
    yield m
    try:
        m.revert()
    except Exception:
        pass


# ─── Module hygiene ───────────────────────────────────────────────────


class TestModuleHygiene:
    def test_importable_torch_less(self, mod):
        assert mod is not None

    def test_no_top_level_torch_import(self, mod):
        import re
        from pathlib import Path

        src = Path(mod.__file__).read_text()
        assert not re.findall(
            r"^(?:import torch|from torch)", src, flags=re.M
        ), "top-level torch import breaks torch-less collection"

    def test_marker_constant_exists(self, mod):
        assert isinstance(mod.GENESIS_G4_81_MARKER, str)
        assert "G4_81" in mod.GENESIS_G4_81_MARKER

    def test_public_api(self, mod):
        assert callable(mod.apply)
        assert callable(mod.is_applied)
        assert callable(mod.revert)

    def test_env_flag_documented_in_source(self, mod):
        from pathlib import Path

        src = Path(mod.__file__).read_text()
        assert ENV_FLAG in src

    def test_build_for_drafting_checklist_note(self, mod):
        """Task requirement: build()/build_for_drafting() coverage must be
        an explicit item of the patch checklist (vllm#45144 lesson —
        their drafter bug came from build_for_drafting() missing the
        scale-allocation fix that build() had)."""
        from pathlib import Path

        src = Path(mod.__file__).read_text()
        assert "build_for_drafting" in src


# ─── Classification predicate (pure ints) ─────────────────────────────


def _classify(mod, **kw):
    defaults = dict(
        num_actual_tokens=0,
        max_query_len=0,
        max_seq_len=0,
        num_decodes=0,
        num_decode_tokens=0,
        is_prefill=False,
        query_start_loc_len=0,
    )
    defaults.update(kw)
    return mod._classify_spec_verify_batch(**defaults)


class TestClassifyPrefillShape:
    """Shape (a): today's builder — supports_spec_as_decode=False,
    uniform K+1 verify batches arrive PREFILL-classified."""

    def test_uniform_k3_verify_batch_routes(self, mod):
        # B=2 requests, K+1=4 (MTP K=3), each with prior cache.
        assert _classify(
            mod,
            num_actual_tokens=8,
            max_query_len=4,
            max_seq_len=100,
            is_prefill=True,
            query_start_loc_len=3,
        ) == (2, 4)

    def test_single_request_k1_2(self, mod):
        assert _classify(
            mod,
            num_actual_tokens=2,
            max_query_len=2,
            max_seq_len=50,
            is_prefill=True,
            query_start_loc_len=2,
        ) == (1, 2)

    def test_rejects_plain_prefill_first_chunk(self, mod):
        # max_seq_len == max_query_len → no prior cache → flash path is
        # better AND nothing is broken — do not route.
        assert _classify(
            mod,
            num_actual_tokens=4,
            max_query_len=4,
            max_seq_len=4,
            is_prefill=True,
            query_start_loc_len=2,
        ) is None

    def test_rejects_non_uniform_total(self, mod):
        # 7 tokens cannot be B * 4.
        assert _classify(
            mod,
            num_actual_tokens=7,
            max_query_len=4,
            max_seq_len=100,
            is_prefill=True,
            query_start_loc_len=3,
        ) is None

    def test_rejects_qsl_len_mismatch(self, mod):
        # Arithmetic uniformity proof needs qsl_len == B + 1.
        assert _classify(
            mod,
            num_actual_tokens=8,
            max_query_len=4,
            max_seq_len=100,
            is_prefill=True,
            query_start_loc_len=4,
        ) is None

    def test_rejects_max_query_len_1(self, mod):
        assert _classify(
            mod,
            num_actual_tokens=2,
            max_query_len=1,
            max_seq_len=100,
            is_prefill=False,
            query_start_loc_len=3,
        ) is None

    def test_rejects_k1_above_16(self, mod):
        assert _classify(
            mod,
            num_actual_tokens=34,
            max_query_len=17,
            max_seq_len=100,
            is_prefill=True,
            query_start_loc_len=3,
        ) is None

    def test_rejects_zero_tokens(self, mod):
        assert _classify(
            mod,
            num_actual_tokens=0,
            max_query_len=4,
            max_seq_len=100,
            is_prefill=True,
            query_start_loc_len=1,
        ) is None

    def test_rejects_prefill_shape_with_decodes_present(self, mod):
        # Mixed batch (1-token decodes + prefills) under today's builder
        # must NOT be routed (decode portion is not uniform K+1).
        assert _classify(
            mod,
            num_actual_tokens=9,
            max_query_len=4,
            max_seq_len=100,
            num_decodes=1,
            num_decode_tokens=1,
            is_prefill=True,
            query_start_loc_len=4,
        ) is None


class TestClassifyDecodeShape:
    """Shape (b): spec-as-decode builder generation (vllm#45144 class) —
    multi-query verify batches arrive DECODE-classified. Coverage here
    is what keeps build()/build_for_drafting() metadata handled if/when
    supports_spec_as_decode flips on a later pin."""

    def test_uniform_decode_classified_routes(self, mod):
        assert _classify(
            mod,
            num_actual_tokens=8,
            max_query_len=4,
            max_seq_len=100,
            num_decodes=2,
            num_decode_tokens=8,
            is_prefill=True,
            query_start_loc_len=3,
        ) == (2, 4)

    def test_rejects_mixed_decode_plus_prefill(self, mod):
        # num_decode_tokens < num_actual_tokens → prefill portion exists.
        assert _classify(
            mod,
            num_actual_tokens=12,
            max_query_len=4,
            max_seq_len=100,
            num_decodes=2,
            num_decode_tokens=8,
            is_prefill=True,
            query_start_loc_len=4,
        ) is None

    def test_rejects_non_uniform_decode_portion(self, mod):
        # 6 tokens over 2 decodes with max_query_len=4 → lens are (4, 2)
        # → not uniform → unsafe for synthetic expansion.
        assert _classify(
            mod,
            num_actual_tokens=6,
            max_query_len=4,
            max_seq_len=100,
            num_decodes=2,
            num_decode_tokens=6,
            is_prefill=True,
            query_start_loc_len=3,
        ) is None

    def test_rejects_single_token_decode_batch(self, mod):
        assert _classify(
            mod,
            num_actual_tokens=2,
            max_query_len=1,
            max_seq_len=100,
            num_decodes=2,
            num_decode_tokens=2,
            is_prefill=False,
            query_start_loc_len=3,
        ) is None


# ─── Launcher capability inspection + route refusal ───────────────────


def _launcher_with_sliding(
    query, kv_cache, block_table, seq_lens, Pi, centroids, scale,
    mse_bits, key_packed_size, value_quant_bits, key_fp8=False,
    norm_correction=False, PiT=None, mid_o_buf=None, output_buf=None,
    lse_buf=None, buf_holder=None, max_num_kv_splits=32,
    sliding_window=None, mm_prefix_range=None,
):
    """Overlay-shaped launcher (pr42637 triton_turboquant_decode.py:718)."""


def _launcher_pristine(
    query, kv_cache, block_table, seq_lens, Pi, centroids, scale,
    mse_bits, key_packed_size, value_quant_bits, key_fp8=False,
    norm_correction=False, PiT=None, mid_o_buf=None, output_buf=None,
    lse_buf=None, buf_holder=None, max_num_kv_splits=32,
):
    """Pristine-pin-shaped launcher (triton_turboquant_decode.py:486 on
    0.22.1rc1.dev259+g303916e93 — no sliding_window / mm_prefix_range)."""


class TestLauncherCapabilities:
    def test_overlay_launcher_caps(self, mod):
        caps = mod._launcher_params(_launcher_with_sliding)
        assert "sliding_window" in caps
        assert "mm_prefix_range" in caps
        assert "buf_holder" in caps

    def test_pristine_launcher_caps(self, mod):
        caps = mod._launcher_params(_launcher_pristine)
        assert "sliding_window" not in caps
        assert "mm_prefix_range" not in caps
        assert "buf_holder" in caps

    def test_uninspectable_launcher_gives_empty_caps(self, mod):
        class _Weird:
            __signature__ = property()  # raises on inspect

            def __call__(self):
                pass

        caps = mod._launcher_params(_Weird())
        assert caps == frozenset()


class TestRouteRefusal:
    def test_full_attn_layer_pristine_launcher_ok(self, mod):
        caps = mod._launcher_params(_launcher_pristine)
        assert mod._route_refusal(caps, None, False) is None

    def test_sliding_layer_needs_sliding_capable_launcher(self, mod):
        caps = mod._launcher_params(_launcher_pristine)
        reason = mod._route_refusal(caps, 1024, False)
        assert reason is not None
        assert "sliding" in reason

    def test_sliding_layer_overlay_launcher_ok(self, mod):
        caps = mod._launcher_params(_launcher_with_sliding)
        assert mod._route_refusal(caps, 1024, False) is None

    def test_mm_prefix_needs_capable_launcher(self, mod):
        caps = mod._launcher_params(_launcher_pristine)
        reason = mod._route_refusal(caps, None, True)
        assert reason is not None
        assert "mm_prefix" in reason

    def test_mm_prefix_overlay_launcher_ok(self, mod):
        caps = mod._launcher_params(_launcher_with_sliding)
        assert mod._route_refusal(caps, None, True) is None


# ─── Synthetic seq-len reference math ─────────────────────────────────


class TestSynthSeqLenReference:
    def test_reference_formula(self, mod):
        # Request with total seq_len=100, K1=4: virtual rows attend to
        # 97, 98, 99, 100 positions (token j is the LAST token of its
        # synthetic sequence — kernel sets query_pos = seq_len - 1).
        assert [
            mod._synth_seq_len_ref(100, 4, j) for j in range(4)
        ] == [97, 98, 99, 100]

    def test_last_row_sees_full_sequence(self, mod):
        # Final verify token must attend to the entire sequence.
        assert mod._synth_seq_len_ref(64, 3, 2) == 64

    def test_first_chunk_prompt_still_causal(self, mod):
        # seq_len == K1 (no prior cache): rows see 1..K1 — causal.
        assert [
            mod._synth_seq_len_ref(4, 4, j) for j in range(4)
        ] == [1, 2, 3, 4]


# ─── apply()/revert() rebind mechanics (fake vllm tree) ───────────────


class _FakeImpl:
    """Stand-in for TurboQuantAttentionImpl."""

    def forward(
        self, layer, query, key, value, kv_cache, attn_metadata,
        output=None, output_scale=None, output_block_scale=None,
    ):
        return ("original", attn_metadata)


_FAKE_IMPL_PRISTINE_FORWARD = _FakeImpl.forward


def _reset_patch_state():
    """Force-reset module-level apply state + fake impl binding."""
    m = importlib.import_module(MODULE)
    m._APPLIED = False
    m._ORIGINAL_FORWARD = None
    _FakeImpl.forward = _FAKE_IMPL_PRISTINE_FORWARD


@pytest.fixture
def fake_vllm(monkeypatch):
    """Inject a minimal fake vllm.v1.attention.backends.turboquant_attn
    + ops module tree into sys.modules."""
    saved = {k: v for k, v in sys.modules.items() if k.startswith("vllm")}
    for k in list(sys.modules):
        if k.startswith("vllm"):
            del sys.modules[k]
    _reset_patch_state()

    names = [
        "vllm",
        "vllm.v1",
        "vllm.v1.attention",
        "vllm.v1.attention.backends",
        "vllm.v1.attention.ops",
    ]
    for name in names:
        sys.modules[name] = types.ModuleType(name)

    backend_mod = types.ModuleType("vllm.v1.attention.backends.turboquant_attn")
    backend_mod.TurboQuantAttentionImpl = _FakeImpl
    sys.modules["vllm.v1.attention.backends.turboquant_attn"] = backend_mod

    ops_mod = types.ModuleType(
        "vllm.v1.attention.ops.triton_turboquant_decode"
    )
    ops_mod.triton_turboquant_decode_attention = _launcher_pristine
    sys.modules["vllm.v1.attention.ops.triton_turboquant_decode"] = ops_mod

    yield backend_mod

    _reset_patch_state()
    for k in list(sys.modules):
        if k.startswith("vllm"):
            del sys.modules[k]
    sys.modules.update(saved)


class TestApplyRevert:
    def test_skipped_when_env_unset(self, mod, monkeypatch, fake_vllm):
        monkeypatch.delenv(ENV_FLAG, raising=False)
        status, reason = mod.apply()
        assert status == "skipped"
        assert ENV_FLAG in reason

    def test_applies_and_stamps_marker(self, mod, monkeypatch, fake_vllm):
        monkeypatch.setenv(ENV_FLAG, "1")
        status, reason = mod.apply()
        assert status == "applied", reason
        assert getattr(_FakeImpl.forward, "_genesis_g4_81_wrapped", False)
        assert mod.is_applied()

    def test_apply_idempotent(self, mod, monkeypatch, fake_vllm):
        monkeypatch.setenv(ENV_FLAG, "1")
        assert mod.apply()[0] == "applied"
        first = _FakeImpl.forward
        assert mod.apply()[0] == "applied"
        assert _FakeImpl.forward is first, "double-wrap detected"

    def test_revert_restores_original(self, mod, monkeypatch, fake_vllm):
        monkeypatch.setenv(ENV_FLAG, "1")
        original = _FakeImpl.forward
        assert mod.apply()[0] == "applied"
        assert _FakeImpl.forward is not original
        assert mod.revert() is True
        assert _FakeImpl.forward is original
        assert not mod.is_applied()

    def test_skipped_when_module_missing(self, mod, monkeypatch):
        monkeypatch.setenv(ENV_FLAG, "1")
        _reset_patch_state()
        saved = {k: v for k, v in sys.modules.items() if k.startswith("vllm")}
        for k in list(sys.modules):
            if k.startswith("vllm"):
                del sys.modules[k]
        try:
            status, reason = mod.apply()
            assert status == "skipped"
            assert "importable" in reason or "not " in reason
        finally:
            sys.modules.update(saved)


class TestWrapperFallThrough:
    """Wrapped forward must delegate to original for every non-routable
    batch shape (the safety contract: any wrapper failure = today's
    behavior, never a half-routed batch)."""

    def _applied_impl(self, mod, monkeypatch, fake_vllm):
        monkeypatch.setenv(ENV_FLAG, "1")
        status, reason = mod.apply()
        assert status == "applied", reason
        return _FakeImpl()

    def test_metadata_none_delegates(self, mod, monkeypatch, fake_vllm):
        impl = self._applied_impl(mod, monkeypatch, fake_vllm)
        out = impl.forward(None, None, None, None, None, None)
        assert out == ("original", None)

    def test_plain_decode_metadata_delegates(
        self, mod, monkeypatch, fake_vllm
    ):
        impl = self._applied_impl(mod, monkeypatch, fake_vllm)
        meta = SimpleNamespace(
            num_actual_tokens=2,
            max_query_len=1,
            max_seq_len=100,
            num_decodes=2,
            num_decode_tokens=2,
            is_prefill=False,
            query_start_loc=None,
            seq_lens=None,
            block_table=None,
        )
        out = impl.forward(None, None, None, None, None, meta)
        assert out == ("original", meta)

    def test_routable_shape_with_missing_tensors_delegates(
        self, mod, monkeypatch, fake_vllm
    ):
        """Routable predicate ints BUT seq_lens/block_table missing →
        must fall through to original, not raise."""
        impl = self._applied_impl(mod, monkeypatch, fake_vllm)

        class _FakeQSL:
            shape = (3,)

        meta = SimpleNamespace(
            num_actual_tokens=8,
            max_query_len=4,
            max_seq_len=100,
            num_decodes=0,
            num_decode_tokens=0,
            is_prefill=True,
            query_start_loc=_FakeQSL(),
            seq_lens=None,
            block_table=None,
        )

        class _FakeQuery:
            shape = (8, 1024)
            ndim = 2

        out = impl.forward(None, _FakeQuery(), None, None, None, meta)
        assert out == ("original", meta)


# ─── Torch-gated tensor equivalence ───────────────────────────────────


class TestSynthArgsTensor:
    def test_synth_args_match_reference(self, mod):
        torch = pytest.importorskip("torch")
        B, K1 = 3, 4
        seq_lens = torch.tensor([100, 64, 9], dtype=torch.int32)
        block_table = torch.arange(3 * 5, dtype=torch.int32).reshape(3, 5)
        synth_sl, synth_bt = mod._build_synth_args(
            seq_lens, block_table, B, K1
        )
        assert synth_sl.shape == (B * K1,)
        assert synth_bt.shape == (B * K1, 5)
        for i in range(B):
            for j in range(K1):
                assert int(synth_sl[i * K1 + j]) == mod._synth_seq_len_ref(
                    int(seq_lens[i]), K1, j
                )
                assert torch.equal(synth_bt[i * K1 + j], block_table[i])
        assert synth_bt.is_contiguous()
