# SPDX-License-Identifier: Apache-2.0
"""G4_60E reconciliation tests — fold of OPEN vllm#45207 + vllm#45181.

Upstream context (both PRs OPEN, NOT in pin 0.22.1rc1.dev259+g303916e93;
verified against /private/tmp/candidate_pin_current/vllm):

  * vllm#45207 "[Bugfix] Pad Mamba page size instead of scaling block_size
    in unify_kv_cache_spec_page_size" — MambaSpec page size is determined
    by its state shapes and does NOT scale with block_size. The pin's
    code scales block_size and trips a bare AssertionError (#43626 class)
    the moment a layer with a larger page (e.g. a bf16 dense drafter)
    joins a hybrid GDN model's specs.

  * vllm#45181 "[Spec Decode] Support mixed KV page sizes for DFlash" —
    (a) generic AttentionSpec padding fallback in
    unify_kv_cache_spec_page_size for non-divisible attention pages
    (e.g. 192-dim target KV heads vs 128-dim drafter heads = 3:2);
    (b) `_reshape_attention_kv_cache` stride hardening: num_blocks dim
    detection (the pin's inline code assumes physical dim 0 is the block
    index) + explicit K/V-dim stride so the V half of a padded page is
    not read from the padding tail.

Genesis G4_60E (PR #42637 cherry-pick) already owns the same function
with the same page_size_padded technique for TQ specs only — it inherits
the #43626 MambaSpec crash itself. This reconciliation folds both PRs
into one pass: TQ-native branch kept (Patches 1/3/4 untouched), then
MambaSpec padding (#45207), then divisible block-size scaling, then
generic AttentionSpec padding fallback (#45181), else an actionable
NotImplementedError.

Contract pinned here (TDD, written before the implementation):
  1. unify ladder: TQ pad > Mamba pad > divisible block-scale >
     AttentionSpec pad > actionable NotImplementedError.
  2. Genesis-unique 3-way unification: TQ k8v4 + Mamba (GDN) + bf16
     drafter pages unify with block sizes preserved on TQ and Mamba.
  3. Pure stride math for the #45181 reshape hardening is testable
     without torch (`_padded_attention_view_strides`).
  4. apply() additionally wraps the modular-runner
     `_reshape_kv_cache` and the legacy-runner
     `GPUModelRunner._reshape_kv_cache_tensors` with padded-view
     post-correction, and injects the upstream-parity helper name
     `_reshape_attention_kv_cache` (merged-form detection hook).
  5. Worker modules absent -> patches 5/6 skip gracefully, kv_cache_utils
     patches still install.
  6. revert() restores every wrapped symbol.

All tests run torch-less via sys.modules stubs (established pattern:
test_pn282_spec_decode_acceptance_metric.py, test_pn96b_gemma4_gate.py).

References:
  * https://github.com/vllm-project/vllm/pull/45207
  * https://github.com/vllm-project/vllm/pull/45181
  * https://github.com/vllm-project/vllm/pull/42637 (original cherry-pick)
  * docs/superpowers/journal/2026-06-11-pr-sweep-50-roadmap.md (chunk 1,
    Theme A)

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass, field

import pytest

_MOD_PATH = (
    "sndr.engines.vllm.patches.attention.turboquant.g4_60e_kv_cache_utils"
)


def _reload_module():
    """Fresh patch-module instance so the _APPLIED latch never leaks."""
    if _MOD_PATH in sys.modules:
        del sys.modules[_MOD_PATH]
    return importlib.import_module(_MOD_PATH)


# ─── Fake KV-cache spec classes (pin g303916e93 semantics) ─────────────
# Mirror vllm/v1/kv_cache_interface.py: page_size_bytes returns
# page_size_padded when set (with the >= real assertion), else the real
# page size. dtype is replaced by dtype_size since torch is unavailable;
# the patched unify never reads spec.dtype.


@dataclass(frozen=True, kw_only=True)
class FakeKVCacheSpec:
    block_size: int


@dataclass(frozen=True, kw_only=True)
class FakeAttentionSpec(FakeKVCacheSpec):
    num_kv_heads: int = 1
    head_size: int = 64
    dtype_size: int = 2
    page_size_padded: int | None = None

    @property
    def storage_block_size(self) -> int:
        return self.block_size

    @property
    def real_page_size_bytes(self) -> int:
        return (
            2
            * self.block_size
            * self.num_kv_heads
            * self.head_size
            * self.dtype_size
        )

    @property
    def page_size_bytes(self) -> int:
        if self.page_size_padded is not None:
            assert self.page_size_padded >= self.real_page_size_bytes
            return self.page_size_padded
        return self.real_page_size_bytes


@dataclass(frozen=True, kw_only=True)
class FakeFullAttentionSpec(FakeAttentionSpec):
    pass


@dataclass(frozen=True, kw_only=True)
class FakeSlidingWindowSpec(FakeAttentionSpec):
    sliding_window: int = 1024


@dataclass(frozen=True, kw_only=True)
class FakeTQFullAttentionSpec(FakeFullAttentionSpec):
    tq_slot_size: int = 0

    @property
    def real_page_size_bytes(self) -> int:
        if self.tq_slot_size > 0:
            return self.block_size * self.num_kv_heads * self.tq_slot_size
        return super().real_page_size_bytes


@dataclass(frozen=True, kw_only=True)
class FakeTQSlidingWindowSpec(FakeSlidingWindowSpec):
    tq_slot_size: int = 0

    @property
    def real_page_size_bytes(self) -> int:
        if self.tq_slot_size > 0:
            return self.block_size * self.num_kv_heads * self.tq_slot_size
        return super().real_page_size_bytes


@dataclass(frozen=True, kw_only=True)
class FakeMambaSpec(FakeKVCacheSpec):
    shapes: tuple = ()
    dtype_size: int = 2
    page_size_padded: int | None = None

    @property
    def real_page_size_bytes(self) -> int:
        total = 0
        for shape in self.shapes:
            numel = 1
            for dim in shape:
                numel *= dim
            total += numel * self.dtype_size
        return total

    @property
    def page_size_bytes(self) -> int:
        if self.page_size_padded is not None:
            assert self.page_size_padded >= self.real_page_size_bytes
            return self.page_size_padded
        return self.real_page_size_bytes


@dataclass(frozen=True, kw_only=True)
class FakeWeirdSpec(FakeKVCacheSpec):
    """Neither AttentionSpec nor MambaSpec — must hit the raise branch."""

    raw_page_size: int = 0

    @property
    def page_size_bytes(self) -> int:
        return self.raw_page_size


# ─── Fake module graph installer ───────────────────────────────────────


@dataclass
class FakeAttentionGroup:
    backend: object
    layer_names: list
    kv_cache_spec: object
    kv_cache_group_id: int = 0


def _install_fake_vllm(monkeypatch, with_workers: bool = True):
    """Install fake vllm.v1 submodules into sys.modules.

    Returns a namespace with the stub modules so tests can poke at the
    patched symbols. monkeypatch.setitem auto-restores sys.modules.
    """
    ns = types.SimpleNamespace()

    fake_iface = types.ModuleType("vllm.v1.kv_cache_interface")
    fake_iface.AttentionSpec = FakeAttentionSpec
    fake_iface.FullAttentionSpec = FakeFullAttentionSpec
    fake_iface.SlidingWindowSpec = FakeSlidingWindowSpec
    fake_iface.TQFullAttentionSpec = FakeTQFullAttentionSpec
    fake_iface.TQSlidingWindowSpec = FakeTQSlidingWindowSpec
    fake_iface.MambaSpec = FakeMambaSpec
    ns.iface = fake_iface

    fake_kcu = types.ModuleType("vllm.v1.core.kv_cache_utils")

    def _orig_is_uniform(kv_cache_spec):
        return True

    def _orig_unify(kv_cache_spec):
        return kv_cache_spec

    def _orig_get_groups(vllm_config, kv_cache_spec):
        return ["original-groups"]

    def _orig_uniform_page_size(kv_cache_spec):
        return ["uniform-page-size-groups"]

    def _orig_attention_free(kv_cache_spec):
        return not kv_cache_spec

    fake_kcu.is_kv_cache_spec_uniform = _orig_is_uniform
    fake_kcu.unify_kv_cache_spec_page_size = _orig_unify
    fake_kcu.get_kv_cache_groups = _orig_get_groups
    fake_kcu._get_kv_cache_groups_uniform_page_size = _orig_uniform_page_size
    fake_kcu.is_kv_cache_type_attention_free = _orig_attention_free
    ns.kcu = fake_kcu
    ns.orig_unify = _orig_unify
    ns.orig_get_groups = _orig_get_groups

    fake_v1 = types.ModuleType("vllm.v1")
    fake_core = types.ModuleType("vllm.v1.core")
    fake_core.kv_cache_utils = fake_kcu
    fake_v1.core = fake_core
    fake_v1.kv_cache_interface = fake_iface

    monkeypatch.setitem(sys.modules, "vllm.v1", fake_v1)
    monkeypatch.setitem(sys.modules, "vllm.v1.core", fake_core)
    monkeypatch.setitem(sys.modules, "vllm.v1.core.kv_cache_utils", fake_kcu)
    monkeypatch.setitem(
        sys.modules, "vllm.v1.kv_cache_interface", fake_iface
    )

    if with_workers:
        fake_attn_utils = types.ModuleType("vllm.v1.worker.gpu.attn_utils")

        def _orig_reshape_kv_cache(
            attn_groups,
            kv_cache_raw_tensors,
            cache_dtype,
            kernel_block_sizes,
            shared_kv_cache_layers,
        ):
            return {"sentinel_layer": "original-view"}

        fake_attn_utils._reshape_kv_cache = _orig_reshape_kv_cache
        ns.attn_utils = fake_attn_utils
        ns.orig_reshape = _orig_reshape_kv_cache

        fake_gmr = types.ModuleType("vllm.v1.worker.gpu_model_runner")

        class FakeGPUModelRunner:
            runner_only_attn_layers: set = set()

            def _kv_cache_spec_attn_group_iterator(self):
                return iter(())

            def _reshape_kv_cache_tensors(
                self, kv_cache_raw_tensors, kernel_block_sizes
            ):
                return {"sentinel_layer": "original-legacy-view"}

        fake_gmr.GPUModelRunner = FakeGPUModelRunner
        ns.gmr = fake_gmr
        ns.orig_legacy_reshape = (
            FakeGPUModelRunner._reshape_kv_cache_tensors
        )

        fake_worker = types.ModuleType("vllm.v1.worker")
        fake_gpu = types.ModuleType("vllm.v1.worker.gpu")
        fake_gpu.attn_utils = fake_attn_utils
        fake_worker.gpu = fake_gpu
        fake_worker.gpu_model_runner = fake_gmr
        fake_v1.worker = fake_worker

        monkeypatch.setitem(sys.modules, "vllm.v1.worker", fake_worker)
        monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu", fake_gpu)
        monkeypatch.setitem(
            sys.modules, "vllm.v1.worker.gpu.attn_utils", fake_attn_utils
        )
        monkeypatch.setitem(
            sys.modules, "vllm.v1.worker.gpu_model_runner", fake_gmr
        )

    return ns


def _apply(monkeypatch, with_workers: bool = True):
    ns = _install_fake_vllm(monkeypatch, with_workers=with_workers)
    mod = _reload_module()
    monkeypatch.setenv(mod._ENV_ENABLE, "1")
    status, msg = mod.apply()
    assert status == "applied", msg
    return mod, ns


# ─── Marker / retirement-tracking surface ──────────────────────────────


def test_marker_records_both_upstream_pr_refs():
    mod = _reload_module()
    marker = mod.GENESIS_G4_60E_MARKER
    assert "42637" in marker
    assert "45207" in marker
    assert "45181" in marker


def test_no_top_level_torch_import():
    """Family contract invariant: torch-less collection safety."""
    mod = _reload_module()
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(mod))
    top_level_imports = [
        n
        for n in tree.body
        if isinstance(n, (ast.Import, ast.ImportFrom))
    ]
    for node in top_level_imports:
        names = (
            [a.name for a in node.names]
            if isinstance(node, ast.Import)
            else [node.module or ""]
        )
        assert not any(n.split(".")[0] == "torch" for n in names)


# ─── Unify ladder: #45207 MambaSpec branch ─────────────────────────────


def test_mamba_page_padded_not_block_scaled(monkeypatch):
    """#43626 regression class: divisible Mamba page must be PADDED.

    The pin scales block_size, which leaves MambaSpec.page_size_bytes
    unchanged (state-shape determined) and dies on a bare assert.
    """
    mod, ns = _apply(monkeypatch)
    unify = ns.kcu.unify_kv_cache_spec_page_size

    mamba = FakeMambaSpec(block_size=16, shapes=((128, 128),))  # 32768
    drafter = FakeFullAttentionSpec(
        block_size=16, num_kv_heads=8, head_size=128
    )  # 65536
    assert mamba.page_size_bytes == 32768
    assert drafter.page_size_bytes == 65536
    assert 65536 % 32768 == 0  # divisible — the old code block-scaled here

    unified = unify({"mamba_layer": mamba, "drafter_layer": drafter})
    assert unified["mamba_layer"].page_size_bytes == 65536
    assert unified["mamba_layer"].page_size_padded == 65536
    assert unified["mamba_layer"].block_size == mamba.block_size
    assert unified["drafter_layer"] == drafter


def test_already_padded_mamba_repadded_at_new_max(monkeypatch):
    mod, ns = _apply(monkeypatch)
    unify = ns.kcu.unify_kv_cache_spec_page_size

    mamba = FakeMambaSpec(
        block_size=16, shapes=((64, 64),), page_size_padded=16384
    )  # real 8192, platform-padded to 16384
    drafter = FakeFullAttentionSpec(
        block_size=16, num_kv_heads=8, head_size=128
    )  # 65536
    unified = unify({"mamba_layer": mamba, "drafter_layer": drafter})
    assert unified["mamba_layer"].page_size_bytes == 65536
    assert unified["mamba_layer"].page_size_padded == 65536


def test_non_divisible_mamba_page_padded(monkeypatch):
    """Padding has no divisibility constraint (only block-scaling does)."""
    mod, ns = _apply(monkeypatch)
    unify = ns.kcu.unify_kv_cache_spec_page_size

    mamba = FakeMambaSpec(block_size=16, shapes=((96, 128),))  # 24576
    drafter = FakeFullAttentionSpec(
        block_size=16, num_kv_heads=8, head_size=128
    )  # 65536
    assert 65536 % mamba.page_size_bytes != 0
    unified = unify({"mamba_layer": mamba, "drafter_layer": drafter})
    assert unified["mamba_layer"].page_size_bytes == 65536


# ─── Unify ladder: #45181 AttentionSpec padding fallback ───────────────


def test_non_divisible_attention_page_padded(monkeypatch):
    """DFlash-class case: 192-dim target vs 128-dim drafter heads (3:2)."""
    mod, ns = _apply(monkeypatch)
    unify = ns.kcu.unify_kv_cache_spec_page_size

    target = FakeFullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=192
    )  # 12288
    draft = FakeSlidingWindowSpec(
        block_size=16, num_kv_heads=1, head_size=128, sliding_window=1024
    )  # 8192
    assert target.page_size_bytes == 12288
    assert draft.page_size_bytes == 8192
    assert 12288 % 8192 != 0

    unified = unify({"target_attn": target, "draft_attn": draft})
    assert unified["target_attn"] == target
    unified_draft = unified["draft_attn"]
    assert unified_draft.block_size == draft.block_size
    assert unified_draft.real_page_size_bytes == draft.real_page_size_bytes
    assert unified_draft.page_size_padded == target.page_size_bytes
    assert unified_draft.page_size_bytes == target.page_size_bytes


def test_divisible_attention_page_still_block_scaled(monkeypatch):
    """Divisible attention pages keep the upstream block-scaling path."""
    mod, ns = _apply(monkeypatch)
    unify = ns.kcu.unify_kv_cache_spec_page_size

    small = FakeFullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=64
    )  # 8192
    big = FakeFullAttentionSpec(
        block_size=16, num_kv_heads=4, head_size=64
    )  # 16384
    unified = unify({"small": small, "big": big})
    assert unified["small"].block_size == 32
    assert unified["small"].page_size_padded is None
    assert unified["small"].page_size_bytes == 16384
    assert unified["big"] == big


def test_unpaddable_spec_raises_actionable_error(monkeypatch):
    """Non-attention, non-Mamba, non-divisible -> actionable error."""
    mod, ns = _apply(monkeypatch)
    unify = ns.kcu.unify_kv_cache_spec_page_size

    weird = FakeWeirdSpec(block_size=16, raw_page_size=24576)
    drafter = FakeFullAttentionSpec(
        block_size=16, num_kv_heads=8, head_size=128
    )  # 65536
    with pytest.raises(NotImplementedError) as exc:
        unify({"weird_layer": weird, "drafter_layer": drafter})
    # Actionable: names the offending layer and both page sizes.
    assert "weird_layer" in str(exc.value)
    assert "24576" in str(exc.value)
    assert "65536" in str(exc.value)


def test_uniform_page_sizes_returned_unchanged(monkeypatch):
    mod, ns = _apply(monkeypatch)
    unify = ns.kcu.unify_kv_cache_spec_page_size

    specs = {
        "a": FakeFullAttentionSpec(block_size=16, num_kv_heads=4, head_size=64),
        "b": FakeFullAttentionSpec(block_size=16, num_kv_heads=4, head_size=64),
    }
    assert unify(specs) == specs


# ─── Unify ladder: TQ-native branch kept (Genesis Patches 1/3/4) ───────


def test_tq_spec_padded_even_when_divisible(monkeypatch):
    """TQ specs must NEVER be block-scaled — slot layout is kernel-bound."""
    mod, ns = _apply(monkeypatch)
    unify = ns.kcu.unify_kv_cache_spec_page_size

    tq = FakeTQFullAttentionSpec(
        block_size=16, num_kv_heads=4, head_size=128, tq_slot_size=128
    )  # 16*4*128 = 8192
    drafter = FakeFullAttentionSpec(
        block_size=16, num_kv_heads=8, head_size=128
    )  # 65536, divisible by 8192
    assert 65536 % tq.page_size_bytes == 0
    unified = unify({"tq_layer": tq, "drafter_layer": drafter})
    assert unified["tq_layer"].block_size == tq.block_size
    assert unified["tq_layer"].page_size_padded == 65536
    assert unified["tq_layer"].page_size_bytes == 65536


def test_three_way_tq_mamba_bf16_drafter_unification(monkeypatch):
    """Genesis-unique 3-way: TQ k8v4 + Mamba (GDN) + bf16 drafter.

    Exact 35B/27B hybrid experiment shape: hybrid GDN main model with TQ
    k8v4 attention pages plus a dense bf16 drafter whose page dominates.
    Both the TQ page and the Mamba page must pad to the drafter page with
    their block sizes (caching granularity) untouched.
    """
    mod, ns = _apply(monkeypatch)
    unify = ns.kcu.unify_kv_cache_spec_page_size

    tq = FakeTQFullAttentionSpec(
        block_size=16, num_kv_heads=4, head_size=128, tq_slot_size=96
    )  # 16*4*96 = 6144 (k8v4-class slot)
    mamba = FakeMambaSpec(
        block_size=16, shapes=((128, 16), (4, 64, 64))
    )  # (2048 + 16384) * 2 = 36864
    drafter = FakeFullAttentionSpec(
        block_size=16, num_kv_heads=8, head_size=128
    )  # 65536 (bf16 dense drafter)

    assert tq.page_size_bytes == 6144
    assert mamba.page_size_bytes == 36864
    assert drafter.page_size_bytes == 65536
    assert 65536 % 6144 != 0  # TQ page non-divisible
    assert 65536 % 36864 != 0  # Mamba page non-divisible

    unified = unify(
        {"tq_attn": tq, "gdn_mamba": mamba, "drafter_attn": drafter}
    )
    page_sizes = {s.page_size_bytes for s in unified.values()}
    assert page_sizes == {65536}
    assert unified["tq_attn"].block_size == tq.block_size
    assert unified["tq_attn"].page_size_padded == 65536
    assert unified["gdn_mamba"].block_size == mamba.block_size
    assert unified["gdn_mamba"].page_size_padded == 65536
    assert unified["drafter_attn"] == drafter


# ─── #45181 reshape stride hardening — pure math, no torch ─────────────


def _strides_fn():
    mod = _reload_module()
    return mod._padded_attention_view_strides


def test_strides_standard_flash_attention_layout():
    """Upstream parity: (num_blocks, 2, block, heads, head) float32,
    real page 256B padded to 384B -> page stride 96 elems, K/V half 32."""
    fn = _strides_fn()
    shape, strides, inv_order = fn(
        unpermuted_kv_cache_shape=(3, 2, 16, 1, 2),
        kv_cache_stride_order=(0, 1, 2, 3, 4),
        num_blocks=3,
        page_stride=96,
    )
    assert shape == (3, 2, 16, 1, 2)
    assert strides == (96, 32, 2, 2, 1)
    assert inv_order == [0, 1, 2, 3, 4]


def test_strides_hnd_layout_with_num_blocks_heads_ambiguity():
    """HND stride order (0,1,3,2,4) with num_kv_heads == num_blocks == 3.

    Detection must prefer unpermuted dim 0 (upstream parity), not the
    heads dim that has the same extent.
    """
    fn = _strides_fn()
    shape, strides, inv_order = fn(
        unpermuted_kv_cache_shape=(3, 2, 16, 3, 2),
        kv_cache_stride_order=(0, 1, 3, 2, 4),
        num_blocks=3,
        page_stride=256,
    )
    assert shape == (3, 2, 3, 16, 2)
    assert strides == (256, 96, 32, 2, 1)
    assert inv_order == [0, 1, 3, 2, 4]


def test_strides_diff_kv_layout_does_not_infer_kv_dim():
    """No size-2 dim adjacent to num_blocks -> no K/V halving."""
    fn = _strides_fn()
    shape, strides, _ = fn(
        unpermuted_kv_cache_shape=(3, 16, 1, 4),
        kv_cache_stride_order=(0, 1, 2, 3),
        num_blocks=3,
        page_stride=96,
    )
    assert shape == (3, 16, 1, 4)
    assert strides == (96, 4, 4, 1)


def test_strides_per_token_scale_quantized_layout():
    """int8 per-token-head scales: page stride 384, K/V half 128."""
    fn = _strides_fn()
    shape, strides, _ = fn(
        unpermuted_kv_cache_shape=(3, 2, 16, 1, 8),
        kv_cache_stride_order=(0, 1, 2, 3, 4),
        num_blocks=3,
        page_stride=384,
    )
    assert strides == (384, 128, 8, 8, 1)


def test_strides_tq_packed_layout_no_kv_dim():
    """Genesis TQ backend layout (num_blocks, block, heads, slot) — packed
    K+V slots, no leading 2: page stride lands on dim 0, nothing halved."""
    fn = _strides_fn()
    shape, strides, _ = fn(
        unpermuted_kv_cache_shape=(4, 16, 4, 96),
        kv_cache_stride_order=(0, 1, 2, 3),
        num_blocks=4,
        page_stride=8192,
    )
    assert shape == (4, 16, 4, 96)
    assert strides == (8192, 384, 96, 1)


def test_strides_kv_first_layout_detects_num_blocks_at_dim_1():
    """(2, num_blocks, ...) layouts: the pin's inline code strides dim 0
    by the page — corrupting the K/V dim. Hardened version finds the
    block dim at position 1 (upstream #45181 parity)."""
    fn = _strides_fn()
    shape, strides, _ = fn(
        unpermuted_kv_cache_shape=(2, 5, 16, 1, 8),
        kv_cache_stride_order=(0, 1, 2, 3, 4),
        num_blocks=5,
        page_stride=384,
    )
    assert shape == (2, 5, 16, 1, 8)
    assert strides[1] == 384  # page stride on the true block dim
    assert strides[0] == 64  # K/V dim: half the unpadded page stride


def test_strides_num_blocks_absent_raises():
    fn = _strides_fn()
    with pytest.raises(ValueError):
        fn(
            unpermuted_kv_cache_shape=(7, 16, 1, 8),
            kv_cache_stride_order=(0, 1, 2, 3),
            num_blocks=3,
            page_stride=96,
        )


# ─── apply() wiring: patches 5/6 ───────────────────────────────────────


def test_apply_installs_all_six_patches(monkeypatch):
    mod, ns = _apply(monkeypatch)

    assert getattr(
        ns.kcu.is_kv_cache_spec_uniform, "_genesis_g4_60e_wrapped", False
    )
    assert getattr(
        ns.kcu.unify_kv_cache_spec_page_size,
        "_genesis_g4_60e_wrapped",
        False,
    )
    assert hasattr(ns.kcu, "_is_tq_native_mixed_kv_cache_spec")
    assert getattr(
        ns.kcu.get_kv_cache_groups, "_genesis_g4_60e_wrapped", False
    )
    # Patch 5: modular-runner reshape wrap + upstream-parity helper.
    assert getattr(
        ns.attn_utils._reshape_kv_cache, "_genesis_g4_60e_wrapped", False
    )
    assert hasattr(ns.attn_utils, "_reshape_attention_kv_cache")
    assert getattr(
        ns.attn_utils._reshape_attention_kv_cache,
        "_genesis_g4_60e_injected",
        False,
    )
    # Patch 6: legacy-runner reshape wrap.
    assert getattr(
        ns.gmr.GPUModelRunner._reshape_kv_cache_tensors,
        "_genesis_g4_60e_wrapped",
        False,
    )


def test_apply_graceful_without_worker_modules(monkeypatch):
    """kv_cache_utils patches install even when worker modules are
    missing (older pins / partial trees)."""
    mod, ns = _apply(monkeypatch, with_workers=False)
    assert getattr(
        ns.kcu.unify_kv_cache_spec_page_size,
        "_genesis_g4_60e_wrapped",
        False,
    )


def test_apply_skips_patch5_on_upstream_merged_form(monkeypatch):
    """If the pin already carries #45181's helper natively, the reshape
    wrap self-skips (retirement-tracking hook)."""
    ns = _install_fake_vllm(monkeypatch)

    def _native_helper(*args, **kwargs):  # upstream merged form
        raise AssertionError("must not be called in this test")

    ns.attn_utils._reshape_attention_kv_cache = _native_helper

    mod = _reload_module()
    monkeypatch.setenv(mod._ENV_ENABLE, "1")
    status, _ = mod.apply()
    assert status == "applied"
    # Reshape functions untouched; helper not overwritten.
    assert not getattr(
        ns.attn_utils._reshape_kv_cache, "_genesis_g4_60e_wrapped", False
    )
    assert ns.attn_utils._reshape_attention_kv_cache is _native_helper
    assert not getattr(
        ns.gmr.GPUModelRunner._reshape_kv_cache_tensors,
        "_genesis_g4_60e_wrapped",
        False,
    )


def test_wrapped_modular_reshape_passthrough_without_padding(monkeypatch):
    """No padded attention specs -> wrapper returns the original result."""
    mod, ns = _apply(monkeypatch)

    group = FakeAttentionGroup(
        backend=object(),
        layer_names=["layer_a"],
        kv_cache_spec=FakeFullAttentionSpec(
            block_size=16, num_kv_heads=4, head_size=64
        ),
        kv_cache_group_id=0,
    )
    result = ns.attn_utils._reshape_kv_cache(
        attn_groups=[group],
        kv_cache_raw_tensors={"layer_a": object()},
        cache_dtype="auto",
        kernel_block_sizes=[16],
        shared_kv_cache_layers={},
    )
    assert result == {"sentinel_layer": "original-view"}


def test_wrapped_legacy_reshape_passthrough_without_padding(monkeypatch):
    mod, ns = _apply(monkeypatch)

    runner = ns.gmr.GPUModelRunner()
    result = runner._reshape_kv_cache_tensors({"layer_a": object()}, [16])
    assert result == {"sentinel_layer": "original-legacy-view"}


def test_revert_restores_all_symbols(monkeypatch):
    mod, ns = _apply(monkeypatch)
    assert mod.revert() is True

    assert ns.kcu.unify_kv_cache_spec_page_size is ns.orig_unify
    assert ns.kcu.get_kv_cache_groups is ns.orig_get_groups
    assert not hasattr(ns.kcu, "_is_tq_native_mixed_kv_cache_spec")
    assert ns.attn_utils._reshape_kv_cache is ns.orig_reshape
    assert not hasattr(ns.attn_utils, "_reshape_attention_kv_cache")
    assert (
        ns.gmr.GPUModelRunner._reshape_kv_cache_tensors
        is ns.orig_legacy_reshape
    )


def test_apply_idempotent(monkeypatch):
    mod, ns = _apply(monkeypatch)
    status, msg = mod.apply()
    assert status == "applied"
    assert "already" in msg or "idempotent" in msg
