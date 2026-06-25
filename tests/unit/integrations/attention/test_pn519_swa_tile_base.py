# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PN519 — start the SWA/chunked KV-tile loop exactly at
``first_allowed_key`` (backport+improve OPEN vllm#46087, fixes vllm#44575).

The bug (LIVE on Gemma4 SWA sliding layers via the Triton unified-attention
path)
----------------------------------------------------------------------------
``compute_tile_loop_bounds`` (in
``vllm/v1/attention/ops/triton_attention_helpers.py``) starts the
sliding-window/chunked tile loop at the tile FLOOR::

    tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)

and the two consumer kernels (``triton_unified_attention.py`` and its
``_diffkv`` sibling) then index ``seq_offset = j * TILE_SIZE + offs_t``. So a
window of ``W`` keys spans ``ceil((r + W) / TILE_SIZE)`` tiles instead of the
minimal ``ceil(W / TILE_SIZE)``, where ``r = first_allowed_key % TILE_SIZE``:

  1. PERF — one redundant tile per SWA request whenever ``r != 0``: the
     boundary tile's pre-window keys are loaded then masked out.
  2. DETERMINISM — the residue ``r`` shifts which keys land in the boundary
     tile, perturbing the online-softmax reduction ORDER, so the output is
     not byte-identical across windows whose ``first_allowed_key`` differ
     only by a sub-tile residue.

Our Gemma4 (26B-A4B + 31B) interleaved-SWA layers run this exact kernel —
the 512-wide global heads route through ``triton_unified_attention.py`` on
Ampere (PN351 vendors a sibling tune to the same file), and the sliding
layers hit the SWA tile loop.

The fix (vllm#46087)
--------------------
``compute_tile_loop_bounds`` returns a 4th value ``tile_base`` (non-zero only
on the 2D-pointer SWA/chunked path; ``USE_TD`` and 3D-segmented paths keep it
0). Both consumers offset ``seq_offset = tile_base + j * TILE_SIZE + offs_t``
so iteration starts EXACTLY at ``first_allowed_key``. ``tile_end`` is
unchanged, so the iteration count NEVER grows (one fewer or equal tiles), and
the boundary residue no longer reorders the reduction → byte-identical output.

OUR version over the raw PR (iron rule #10)
-------------------------------------------
1. Atomic three-file apply: because ``compute_tile_loop_bounds`` now returns a
   4-tuple, BOTH consumers must be updated in lockstep or the unpack breaks.
   PN519 fails LOUDLY (not a half-apply) if any of the three files' anchors
   drift — the helper-only apply (3-tuple producer vs 3-tuple unpack) would be
   a silent ValueError at first decode.
2. Runtime-inert on Qwen3.6: the 35B FP8 / 27B INT4 PROD models run FlashInfer
   / FA2 (head_dim=128), never the Triton unified kernel — the patched code
   path is imported but never executed. default_on is therefore scoped to the
   Gemma4 SWA configs only.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sndr.engines.vllm.patches.attention import pn519_swa_tile_base as pn519

_FIXTURES = Path(__file__).parent / "fixtures"
_FIX_HELPER = _FIXTURES / "pn519_helper_live_region.txt"
_FIX_CONSUMER_A = _FIXTURES / "pn519_consumerA_live_region.txt"
_FIX_CONSUMER_B = _FIXTURES / "pn519_consumerB_diffkv_live_region.txt"

_ENV_FLAG = "GENESIS_ENABLE_PN519_SWA_TILE_BASE"


# ─────────────────────────────────────────────────────────────────────
# 1. ANCHOR-PRESENCE + UNIQUENESS on the frozen live-dev424 fixtures
# ─────────────────────────────────────────────────────────────────────


def test_helper_anchors_present_and_unique():
    live = _FIX_HELPER.read_text(encoding="utf-8")
    for anchor in (
        pn519.HELPER_SIG_OLD,
        pn519.HELPER_INIT_OLD,
        pn519.HELPER_TILE_OLD,
        pn519.HELPER_RETURN_OLD,
    ):
        assert anchor in live, anchor
        assert live.count(anchor) == 1, anchor


def test_consumer_a_anchors_present_and_unique():
    live = _FIX_CONSUMER_A.read_text(encoding="utf-8")
    assert live.count(pn519.CONSUMER_A_CALL_OLD) == 1
    assert live.count(pn519.CONSUMER_SEQ_OFFSET_OLD) == 1


def test_consumer_b_anchors_present_and_unique():
    live = _FIX_CONSUMER_B.read_text(encoding="utf-8")
    assert live.count(pn519.CONSUMER_B_CALL_OLD) == 1
    assert live.count(pn519.CONSUMER_SEQ_OFFSET_OLD) == 1


# ─────────────────────────────────────────────────────────────────────
# 2. THE BUG IS PRESENT IN PRISTINE dev424 (TDD RED repro)
# ─────────────────────────────────────────────────────────────────────


def test_pristine_helper_starts_at_tile_floor_not_first_key():
    """REPRO of vllm#44575: pristine helper computes the loop base as the tile
    FLOOR (`first_allowed_key // TILE_SIZE`) and has NO `tile_base` offset, so
    the boundary tile carries pre-window keys."""
    live = _FIX_HELPER.read_text(encoding="utf-8")
    assert "first_allowed_key // TILE_SIZE" in live
    assert "tile_base" not in live, "pristine dev424 has no tile_base (the bug)"
    # The helper returns a 3-tuple (no tile_base) in pristine.
    assert "return loop_lo, loop_hi, max_seq_prefix_len" in live
    assert "return loop_lo, loop_hi, max_seq_prefix_len, tile_base" not in live


def test_pristine_consumers_index_without_tile_base():
    """Pristine consumers index `j * TILE_SIZE` with no `tile_base` offset."""
    for fx in (_FIX_CONSUMER_A, _FIX_CONSUMER_B):
        live = fx.read_text(encoding="utf-8")
        assert "seq_offset = j * TILE_SIZE + offs_t" in live
        assert "tile_base + j * TILE_SIZE" not in live


# ─────────────────────────────────────────────────────────────────────
# 3. PURE-MODEL determinism contract (mirror of the kernel math)
# ─────────────────────────────────────────────────────────────────────
#
# The kernel runs on GPU only, so the byte-identical-across-residue property
# (the heart of the fix) is exercised here against a pure-python model of the
# tile-iteration set that the kernel actually walks. `tiles_walked` returns
# the multiset of absolute key positions the loop touches; the fixed version
# must (a) touch NO key < first_allowed_key, and (b) be residue-invariant in
# tile COUNT.


def test_pristine_model_walks_redundant_boundary_tile():
    """With the FLOOR base, a non-tile-aligned window touches keys BEFORE
    first_allowed_key (the redundant boundary tile)."""
    walked = pn519.model_tiles_walked(
        first_allowed_key=1000, last_allowed_key=1255, tile_size=16,
        use_tile_base=False,
    )
    # floor(1000/16)=62 -> first tile starts at key 992 < 1000 (redundant).
    assert min(walked) < 1000, "pristine touches pre-window keys"


def test_fixed_model_starts_exactly_at_first_allowed_key():
    """With tile_base, the loop starts EXACTLY at first_allowed_key — no
    pre-window keys are ever touched."""
    walked = pn519.model_tiles_walked(
        first_allowed_key=1000, last_allowed_key=1255, tile_size=16,
        use_tile_base=True,
    )
    assert min(walked) == 1000, "fixed loop starts at first_allowed_key"


def test_fixed_model_tile_count_residue_invariant():
    """The fixed loop's tile COUNT depends only on the window WIDTH, not on
    first_allowed_key's residue mod TILE_SIZE — the determinism property."""
    width = 256
    counts = set()
    for r in range(16):  # sweep every residue
        fak = 1000 + r
        n = pn519.model_tile_count(
            first_allowed_key=fak, last_allowed_key=fak + width - 1,
            tile_size=16, use_tile_base=True,
        )
        counts.add(n)
    assert len(counts) == 1, (
        f"fixed tile count must be residue-invariant, got {counts}"
    )


def test_pristine_model_tile_count_varies_with_residue():
    """The pristine (FLOOR) loop's tile count grows by one when the residue is
    non-zero — the redundant tile the fix removes."""
    width = 256
    counts = set()
    for r in range(16):
        fak = 1000 + r
        n = pn519.model_tile_count(
            first_allowed_key=fak, last_allowed_key=fak + width - 1,
            tile_size=16, use_tile_base=False,
        )
        counts.add(n)
    assert len(counts) > 1, "pristine tile count is residue-dependent (the bug)"


def test_fixed_never_walks_more_tiles_than_pristine():
    """The fix never INCREASES the iteration count (tile_end is unchanged)."""
    for r in range(16):
        fak = 1000 + r
        last = fak + 255
        n_fixed = pn519.model_tile_count(fak, last, 16, use_tile_base=True)
        n_floor = pn519.model_tile_count(fak, last, 16, use_tile_base=False)
        assert n_fixed <= n_floor


# ─────────────────────────────────────────────────────────────────────
# 4. APPLY -> APPLIED across all three synthetic files
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def env_pn519_on(monkeypatch):
    monkeypatch.setenv(_ENV_FLAG, "1")
    monkeypatch.setenv("SNDR_ENABLE_PN519_SWA_TILE_BASE", "1")
    yield


@pytest.fixture
def synthetic_tree(tmp_path, monkeypatch):
    """Write the three live regions to tmp files and point resolve_vllm_file
    at them."""
    helper = tmp_path / "triton_attention_helpers.py"
    consumer_a = tmp_path / "triton_unified_attention.py"
    consumer_b = tmp_path / "triton_unified_attention_diffkv.py"
    helper.write_text(_FIX_HELPER.read_text(encoding="utf-8"), encoding="utf-8")
    consumer_a.write_text(_FIX_CONSUMER_A.read_text(encoding="utf-8"), encoding="utf-8")
    consumer_b.write_text(_FIX_CONSUMER_B.read_text(encoding="utf-8"), encoding="utf-8")

    def fake_resolve(rel: str):
        if rel.endswith("triton_attention_helpers.py"):
            return str(helper)
        if rel.endswith("triton_unified_attention_diffkv.py"):
            return str(consumer_b)
        if rel.endswith("triton_unified_attention.py"):
            return str(consumer_a)
        return None

    monkeypatch.setattr(pn519, "resolve_vllm_file", fake_resolve)
    return {"helper": helper, "a": consumer_a, "b": consumer_b}


def test_apply_applied_all_three_files(env_pn519_on, synthetic_tree):
    status, reason = pn519.apply()
    assert status == "applied", reason
    helper = synthetic_tree["helper"].read_text(encoding="utf-8")
    a = synthetic_tree["a"].read_text(encoding="utf-8")
    b = synthetic_tree["b"].read_text(encoding="utf-8")
    # marker on each file
    assert pn519.GENESIS_PN519_MARKER in helper
    assert pn519.GENESIS_PN519_MARKER in a
    assert pn519.GENESIS_PN519_MARKER in b
    # helper now returns the 4-tuple and computes tile_base
    assert "return loop_lo, loop_hi, max_seq_prefix_len, tile_base" in helper
    assert "tile_base = tl.maximum(0, first_allowed_key) - tile_start * TILE_SIZE" in helper
    assert "USE_TD: tl.constexpr = False," in helper
    # both consumers unpack the 4-tuple and offset seq_offset
    assert a.count("loop_lo, loop_hi, max_seq_prefix_len, tile_base =") == 1
    assert b.count("loop_lo, loop_hi, max_seq_prefix_len, tile_base =") == 1
    assert "seq_offset = tile_base + j * TILE_SIZE + offs_t" in a
    assert "seq_offset = tile_base + j * TILE_SIZE + offs_t" in b
    # consumer A passes USE_TD through to the helper call
    assert "USE_TD,\n" in a


def test_apply_helper_parses_after_patch(env_pn519_on, synthetic_tree):
    """The patched helper region, wrapped into a parseable stub, is valid
    python (no truncated tuple / dangling comma)."""
    pn519.apply()
    helper = synthetic_tree["helper"].read_text(encoding="utf-8")
    # The helper region is a real def; ast-parse it directly.
    import ast

    # strip @triton.jit-less: the region begins at `def compute_tile_loop...`
    ast.parse(helper)


def test_apply_is_idempotent(env_pn519_on, synthetic_tree):
    s1, _ = pn519.apply()
    assert s1 == "applied"
    snapshot = {
        k: v.read_text(encoding="utf-8") for k, v in synthetic_tree.items()
    }
    s2, r2 = pn519.apply()
    assert s2 == "skipped", r2
    for k, v in synthetic_tree.items():
        assert v.read_text(encoding="utf-8") == snapshot[k]


def test_is_applied_reflects_state(env_pn519_on, synthetic_tree):
    assert pn519.is_applied() is False
    pn519.apply()
    assert pn519.is_applied() is True


def test_disabled_by_default(monkeypatch, synthetic_tree):
    monkeypatch.delenv(_ENV_FLAG, raising=False)
    monkeypatch.delenv("SNDR_ENABLE_PN519_SWA_TILE_BASE", raising=False)
    status, reason = pn519.apply()
    assert status == "skipped", reason
    assert pn519.GENESIS_PN519_MARKER not in synthetic_tree["helper"].read_text(
        encoding="utf-8"
    )


def test_half_apply_fails_loudly_if_consumer_drifts(env_pn519_on, tmp_path, monkeypatch):
    """If the helper anchors apply but a consumer's seq_offset anchor has
    drifted (so the unpack would break), PN519 must FAIL — not silently leave a
    3-tuple producer feeding a 3-tuple unpack."""
    helper = tmp_path / "triton_attention_helpers.py"
    consumer_a = tmp_path / "triton_unified_attention.py"
    consumer_b = tmp_path / "triton_unified_attention_diffkv.py"
    helper.write_text(_FIX_HELPER.read_text(encoding="utf-8"), encoding="utf-8")
    consumer_a.write_text(_FIX_CONSUMER_A.read_text(encoding="utf-8"), encoding="utf-8")
    # Drift consumer B: remove the seq_offset anchor entirely.
    drifted = _FIX_CONSUMER_B.read_text(encoding="utf-8").replace(
        "seq_offset = j * TILE_SIZE + offs_t",
        "seq_offset = j * TILE_SIZE  # DRIFTED",
    )
    consumer_b.write_text(drifted, encoding="utf-8")

    def fake_resolve(rel: str):
        if rel.endswith("triton_attention_helpers.py"):
            return str(helper)
        if rel.endswith("triton_unified_attention_diffkv.py"):
            return str(consumer_b)
        if rel.endswith("triton_unified_attention.py"):
            return str(consumer_a)
        return None

    monkeypatch.setattr(pn519, "resolve_vllm_file", fake_resolve)
    status, reason = pn519.apply()
    assert status == "failed", reason


# ─────────────────────────────────────────────────────────────────────
# 5. REGISTRY + env-flag contract
# ─────────────────────────────────────────────────────────────────────


def test_registry_pn519_contract():
    from sndr.dispatcher.registry import PATCH_REGISTRY

    assert "PN519" in PATCH_REGISTRY
    meta = PATCH_REGISTRY["PN519"]
    assert meta["env_flag"] == _ENV_FLAG
    assert meta["family"] == "attention"
    assert meta["upstream_pr"] == 46087
    assert meta["upstream_pr_relationship"] == "backport"
    assert meta["applies_to"]["vllm_version_range"] == (">=0.23.0", "<0.24.0")
    assert meta["default_on"] is False
    assert "PN351" in set(meta.get("composes_with", []))


def test_env_flag_registered():
    from sndr.env import Flags

    assert hasattr(Flags, "PN519_SWA_TILE_BASE")
    assert Flags.PN519_SWA_TILE_BASE == "PN519_SWA_TILE_BASE"
