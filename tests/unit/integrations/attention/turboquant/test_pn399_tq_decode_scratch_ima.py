# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PN399 — CONSOLIDATED single-owner TQ decode-scratch.

Genesis backport+improvement of OPEN vllm#46067, re-authored against live
dev148. PN399 OWNS the TQ decode-scratch lifecycle: it wraps PN118's live
decode output (fixed-buffer CG branch BEFORE PN118's try_get, demoting it
to the eager elif) AND removes the now-dead decode reservations that
upstream cannot (PN118 __init__ reserve box+call+method; PN353A
decode-scratch get_simultaneous) while KEEPING the PN353A
continuation-prefill K/V reservation byte-intact. PN118/PN353A SOURCE files
are NOT edited (PN399 anchors their applied output).

Coverage:
  1. ANCHOR-PRESENCE on a frozen live-dev148 fixture: all four TQ anchor_old
     strings present; the PR's literal `= 128` ABSENT, `= 64` +
     `MAX_CACHED_LEN` present (P101/P98/PN369 silent-no-op guard class).
     B' anchor spans the full PN118 reserve method; C2 anchor brackets the
     PN353A decode reserve up to the continuation-prefill keep boundary.
  2. APPLY -> APPLIED on synthetic files built from the anchor blocks: all
     five drift markers present, `elif is_workspace_manager_initialized()`
     present, PN118 try_get body preserved verbatim (no double-rewrite),
     PN118 __init__ reserve REMOVED, PN353A decode reserve REMOVED,
     PN353A continuation-prefill reserve INTACT, idempotency via marker.
  3. DEPENDENCY: a synthetic with NO PN118 box -> B' SKIPS; a synthetic with
     NO PN353A reserve block -> C2 SKIPS (required anchor missing ->
     SKIPPED, not FAILED), proving requires_patches:[PN118, PN353A] is real.
  4. SHUTDOWN sub-patch: import sorts before reset_workspace_manager import;
     call follows reset_workspace_manager().
  5. REGISTRY: PN399 index > PN118 and > PN353A; PN399.requires_patches
     contains "PN118"+"PN353A"; PN118.composes_with + PN353A.composes_with
     contain "PN399".
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sndr.engines.vllm.patches.attention.turboquant import (
    pn399_tq_decode_scratch_ima as pn399,
)

_FIXTURES = Path(__file__).parent / "fixtures"
_FIXTURE_LIVE = _FIXTURES / "pn399_live_anchor_regions.txt"
_FIXTURE_SYN_TQ = _FIXTURES / "pn399_tq_attn_synthetic.py.txt"
_FIXTURE_SYN_SD = _FIXTURES / "pn399_shutdown_synthetic.py.txt"

_ENV_FLAG = "GENESIS_ENABLE_PN399_TQ_DECODE_SCRATCH_IMA"

_DRIFT_MARKERS = [
    "_DECODE_SCRATCH",
    "_get_decode_scratch",
    "reset_tq_decode_scratch",
    "self.max_decode_cudagraph_batch",
    "[Genesis PN399 — backport of vllm#46067]",
]


# ─────────────────────────────────────────────────────────────────────
# 1. ANCHOR-PRESENCE on frozen live-dev148 fixture
# ─────────────────────────────────────────────────────────────────────


def test_all_anchor_old_present_in_frozen_live_regions():
    """Every TQ anchor_old (const, __init__ B', decode C, PN353A reserve C2)
    is a substring of the frozen live-dev148 regions — guards against silent
    anchor drift (the P101/P98/PN369 silent-no-op class)."""
    live = _FIXTURE_LIVE.read_text(encoding="utf-8")
    for name, anchor in (
        ("const", pn399.TQ_ANCHOR_CONST_OLD),
        ("init", pn399.TQ_ANCHOR_INIT_OLD),
        ("decode", pn399.TQ_ANCHOR_DECODE_OLD),
        ("pn353a_reserve", pn399.TQ_ANCHOR_PN353A_DECODE_RESERVE_OLD),
    ):
        assert anchor in live, f"PN399 {name} anchor_old not present in live"
        assert live.count(anchor) == 1, f"PN399 {name} anchor not unique"


def test_init_anchor_spans_full_pn118_reserve_for_removal():
    """The consolidated B' __init__ anchor must span the ENTIRE live PN118
    reserve box + call + method (so the removal deletes all of it), not just
    the box head — proves B2 (boot-overhead removal) actually targets the
    method body."""
    init_old = pn399.TQ_ANCHOR_INIT_OLD
    # The anchor includes the call AND the def ... method body ending in `)`.
    assert "self._reserve_decode_workspace(vllm_config)" in init_old
    assert "def _reserve_decode_workspace(self, vllm_config) -> None:" in init_old
    assert "manager.reserve(" in init_old
    # The replacement must NOT re-emit the reserve method/call (it is removed),
    # but MUST insert the new attr.
    init_new = pn399.TQ_ANCHOR_INIT_NEW
    assert "self.max_decode_cudagraph_batch" in init_new
    assert "def _reserve_decode_workspace" not in init_new
    assert "self._reserve_decode_workspace(vllm_config)" not in init_new
    assert "manager.reserve(" not in init_new


def test_pn353a_decode_reserve_removed_but_continuation_kept():
    """C2 removes ONLY the PN353A decode-scratch get_simultaneous reservation;
    the continuation-prefill K/V reservation must stay (its comment line is
    re-emitted unchanged in the replacement, and the decode shapes are gone)."""
    old = pn399.TQ_ANCHOR_PN353A_DECODE_RESERVE_OLD
    new = pn399.TQ_ANCHOR_PN353A_DECODE_RESERVE_NEW
    # The OLD anchor contains the decode-scratch reserve...
    assert "# Decode scratch — mirrors _decode_attention's get_simultaneous." in old
    assert "(max_num_reqs, num_heads, max_num_splits, head_size + 1)" in old
    # ...and ends ON the continuation-prefill comment line (the keep boundary).
    assert "# Continuation-prefill K/V dequant buffers — only when chunked" in old
    # The NEW replacement drops the decode reserve but KEEPS the continuation
    # comment line (so the continuation-prefill block survives byte-intact).
    assert "# Decode scratch — mirrors" not in new
    assert "(max_num_reqs, num_heads, max_num_splits, head_size + 1)" not in new
    assert "# Continuation-prefill K/V dequant buffers — only when chunked" in new


def test_shutdown_anchor_old_present_in_live():
    """The pristine shutdown.py anchor (import + call) is present verbatim."""
    sd = _FIXTURE_SYN_SD.read_text(encoding="utf-8")
    assert pn399.SHUTDOWN_ANCHOR_OLD in sd
    assert sd.count(pn399.SHUTDOWN_ANCHOR_OLD) == 1


def test_pr_literal_128_anchor_absent_64_present():
    """The PR keys off `_CONTINUATION_DECODE_THRESHOLD = 128`; live dev148
    is `= 64` (P101) with a sibling `_CONTINUATION_DECODE_MAX_CACHED_LEN`.
    Assert the `= 128` spelling is ABSENT and the live `= 64` +
    `MAX_CACHED_LEN` spelling is present — proves PN399 did not blindly
    port the PR literal (which would silently no-op)."""
    live = _FIXTURE_LIVE.read_text(encoding="utf-8")
    assert "_CONTINUATION_DECODE_THRESHOLD = 128" not in live
    assert "_CONTINUATION_DECODE_THRESHOLD = 64" in pn399.TQ_ANCHOR_CONST_OLD
    assert "_CONTINUATION_DECODE_MAX_CACHED_LEN = 32768" in pn399.TQ_ANCHOR_CONST_OLD
    assert "= 128" not in pn399.TQ_ANCHOR_CONST_OLD


# ─────────────────────────────────────────────────────────────────────
# Helpers / fixtures for apply tests
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def env_pn399_on(monkeypatch):
    monkeypatch.setenv(_ENV_FLAG, "1")
    yield


def _redirect_resolver(monkeypatch, tq_path: Path | None, sd_path: Path | None):
    """Point pn399.resolve_vllm_file at the synthetic tmp files."""
    def fake_resolve(rel: str):
        if "turboquant_attn" in rel:
            return str(tq_path) if tq_path is not None else None
        if "shutdown" in rel:
            return str(sd_path) if sd_path is not None else None
        return None
    monkeypatch.setattr(pn399, "resolve_vllm_file", fake_resolve)


@pytest.fixture
def synthetic_targets(tmp_path):
    tq = tmp_path / "turboquant_attn.py"
    sd = tmp_path / "shutdown.py"
    tq.write_text(_FIXTURE_SYN_TQ.read_text(encoding="utf-8"), encoding="utf-8")
    sd.write_text(_FIXTURE_SYN_SD.read_text(encoding="utf-8"), encoding="utf-8")
    return tq, sd


# ─────────────────────────────────────────────────────────────────────
# 2. APPLY -> APPLIED on synthetic files
# ─────────────────────────────────────────────────────────────────────


def test_apply_applied_all_drift_markers_present(
    env_pn399_on, monkeypatch, synthetic_targets
):
    tq, sd = synthetic_targets
    _redirect_resolver(monkeypatch, tq, sd)

    status, reason = pn399.apply()
    assert status == "applied", reason

    tq_text = tq.read_text(encoding="utf-8")
    sd_text = sd.read_text(encoding="utf-8")
    combined = tq_text + sd_text
    for marker in _DRIFT_MARKERS:
        assert marker in combined, f"drift marker {marker!r} absent post-apply"

    # The marker is prepended for idempotency on both files.
    assert pn399.GENESIS_PN399_MARKER in tq_text
    assert pn399.GENESIS_PN399_MARKER in sd_text


def test_apply_demotes_pn118_to_elif_and_preserves_body(
    env_pn399_on, monkeypatch, synthetic_targets
):
    """PN399 inserts the CG branch BEFORE PN118's if and converts it to
    elif. PN118's try_get_simultaneous body must be preserved verbatim
    (no double-rewrite)."""
    tq, sd = synthetic_targets
    _redirect_resolver(monkeypatch, tq, sd)

    status, reason = pn399.apply()
    assert status == "applied", reason

    tq_text = tq.read_text(encoding="utf-8")
    assert "        elif is_workspace_manager_initialized():\n" in tq_text
    # The CG branch precedes the (now) elif.
    cg_idx = tq_text.find("max_batch = self.max_decode_cudagraph_batch")
    elif_idx = tq_text.find("elif is_workspace_manager_initialized():")
    assert cg_idx != -1 and elif_idx != -1 and cg_idx < elif_idx

    # PN118 try_get body preserved verbatim (byte-exact slice).
    pn118_body = (
        "            manager = current_workspace_manager()\n"
        "            if hasattr(manager, 'try_get_simultaneous'):\n"
        "                bufs = manager.try_get_simultaneous(\n"
        "                    ((B, Hq, S, D + 1), torch.float32),\n"
        "                    ((B, Hq, D), query.dtype),\n"
        "                    ((B, Hq), torch.float32),\n"
        "                )\n"
    )
    assert pn118_body in tq_text


def test_apply_inserts_init_attr_and_removes_pn118_reserve(
    env_pn399_on, monkeypatch, synthetic_targets
):
    """Consolidated B': the new max_decode_cudagraph_batch attr is inserted
    after the kv_splits assignment AND the now-dead PN118 __init__
    _reserve_decode_workspace box + call + method are REMOVED (boot-overhead
    cut)."""
    tq, sd = synthetic_targets
    _redirect_resolver(monkeypatch, tq, sd)

    status, reason = pn399.apply()
    assert status == "applied", reason

    tq_text = tq.read_text(encoding="utf-8")
    # Attr inserted right after the kv_splits assignment.
    kv_idx = tq_text.find(
        "vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph"
    )
    attr_idx = tq_text.find("self.max_decode_cudagraph_batch")
    assert kv_idx != -1 and attr_idx != -1 and kv_idx < attr_idx

    # B2 removal: the PN118 reserve box, call, and method are all gone.
    assert "self._reserve_decode_workspace(vllm_config)" not in tq_text
    assert "def _reserve_decode_workspace(self, vllm_config) -> None:" not in tq_text
    assert "# Pre-reserve decode scratch buffers so lock_workspace() at" not in tq_text
    # The file must still parse after the removal.
    import ast

    ast.parse(tq_text)


def test_apply_removes_pn353a_decode_reserve_keeps_continuation(
    env_pn399_on, monkeypatch, synthetic_targets
):
    """Consolidated C2: the PN353A decode-scratch get_simultaneous reservation
    is REMOVED while the PN353A continuation-prefill K/V reservation stays
    BYTE-INTACT (proven distinct call sites)."""
    tq, sd = synthetic_targets
    _redirect_resolver(monkeypatch, tq, sd)

    status, reason = pn399.apply()
    assert status == "applied", reason

    tq_text = tq.read_text(encoding="utf-8")
    # C2 removal: decode-scratch reserve comment + decode shapes gone.
    assert (
        "# Decode scratch — mirrors _decode_attention's get_simultaneous."
        not in tq_text
    )
    assert "(max_num_reqs, num_heads, max_num_splits, head_size + 1)" not in tq_text
    # Continuation-prefill reservation INTACT (comment + f16x2 get_simultaneous).
    assert "# Continuation-prefill K/V dequant buffers — only when chunked" in tq_text
    assert "cache_buf_shape = (1, num_kv_heads, alloc_len, head_size)" in tq_text
    assert "(cache_buf_shape, _genesis_pn353a_torch.float16)," in tq_text
    assert tq_text.count("(cache_buf_shape, _genesis_pn353a_torch.float16),") == 2
    import ast

    ast.parse(tq_text)


def test_apply_requires_pn353a_reserve_block_skips_when_absent(
    env_pn399_on, monkeypatch, tmp_path
):
    """If the PN353A reserve block is absent (PN353A disabled / output not
    present), C2's required anchor is missing -> the TQ patcher SKIPS cleanly
    (required_anchor_missing), proving requires_patches:[PN353A] is real."""
    # Reuse the full synthetic but strip the PN353A reserve block entirely.
    syn = _FIXTURE_SYN_TQ.read_text(encoding="utf-8")
    cut = syn.split("\nclass TurboQuantMetadataBuilder:")[0] + "\n"
    tq = tmp_path / "turboquant_attn.py"
    sd = tmp_path / "shutdown.py"
    tq.write_text(cut, encoding="utf-8")
    sd.write_text(_FIXTURE_SYN_SD.read_text(encoding="utf-8"), encoding="utf-8")
    _redirect_resolver(monkeypatch, tq, sd)

    status, reason = pn399.apply()
    assert status == "skipped", reason
    assert (
        "required_anchor_missing" in reason
        or "pn399_pn353a_decode_reserve_remove" in reason
        or "skipped" in reason.lower()
    )
    # No partial write: marker absent.
    assert pn399.GENESIS_PN399_MARKER not in tq.read_text(encoding="utf-8")
    # TRANSACTION GUARD (deep-audit 2026-06-19): when the TQ patcher SKIPS,
    # shutdown.py must be left BYTE-UNTOUCHED. shutdown.py's sub-patch wires
    # `import reset_tq_decode_scratch` — a symbol DEFINED only by the (now
    # skipped) turboquant_attn.py sub-patches — so applying it alone would
    # raise ImportError on engine teardown. Before the guard the shutdown
    # patcher applied independently here (its anchor is pristine-present),
    # leaving the dangling import; now apply() short-circuits after the TQ
    # skip and never touches shutdown.py.
    assert sd.read_text(encoding="utf-8") == _FIXTURE_SYN_SD.read_text(
        encoding="utf-8"
    )


def test_apply_shutdown_reset_call_and_import_order(
    env_pn399_on, monkeypatch, synthetic_targets
):
    """reset_tq_decode_scratch import sorts before reset_workspace_manager
    import; the call follows reset_workspace_manager()."""
    tq, sd = synthetic_targets
    _redirect_resolver(monkeypatch, tq, sd)

    status, reason = pn399.apply()
    assert status == "applied", reason

    sd_text = sd.read_text(encoding="utf-8")
    imp_tq = sd_text.find(
        "from vllm.v1.attention.backends.turboquant_attn import "
        "reset_tq_decode_scratch"
    )
    imp_ws = sd_text.find(
        "from vllm.v1.worker.workspace import reset_workspace_manager"
    )
    call_ws = sd_text.find("    reset_workspace_manager()\n")
    call_tq = sd_text.find("    reset_tq_decode_scratch()")
    assert imp_tq != -1 and imp_ws != -1 and call_ws != -1 and call_tq != -1
    assert imp_tq < imp_ws, "import must sort before reset_workspace_manager"
    assert call_ws < call_tq, "call must follow reset_workspace_manager()"


def test_apply_is_idempotent(env_pn399_on, monkeypatch, synthetic_targets):
    tq, sd = synthetic_targets
    _redirect_resolver(monkeypatch, tq, sd)

    status1, _ = pn399.apply()
    assert status1 == "applied"
    first_tq = tq.read_text(encoding="utf-8")

    status2, reason2 = pn399.apply()
    # Second apply: both files already carry the marker -> idempotent path.
    assert status2 == "applied", reason2
    assert "idempotent" in reason2.lower()
    assert tq.read_text(encoding="utf-8") == first_tq  # byte-stable


def test_is_applied_reflects_state(
    env_pn399_on, monkeypatch, synthetic_targets
):
    tq, sd = synthetic_targets
    _redirect_resolver(monkeypatch, tq, sd)
    assert pn399.is_applied() is False
    pn399.apply()
    assert pn399.is_applied() is True


# ─────────────────────────────────────────────────────────────────────
# 3. ORDERING vs PN118 — requires_patches:[PN118] is real
# ─────────────────────────────────────────────────────────────────────


def test_sub_patch_b_skips_without_pn118_box(
    env_pn399_on, monkeypatch, tmp_path
):
    """A synthetic file with the const + decode-head anchors but NO PN118
    __init__ box makes sub-patch B's required anchor missing -> the whole
    TQ patcher returns SKIPPED (not FAILED), proving PN399 hard-depends on
    PN118 having inserted its __init__ box first."""
    # File has const anchor + a decode head WITHOUT the PN118 box (so both
    # B and C anchors that reference the PN118 box are absent).
    no_pn118 = (
        "import torch\n\n"
        "_CONTINUATION_DECODE_THRESHOLD = 64\n"
        "_CONTINUATION_DECODE_MAX_CACHED_LEN = 32768\n\n\n"
        "class TurboQuantImpl:\n"
        "    def __init__(self, vllm_config):\n"
        "        self.max_num_kv_splits = (\n"
        "            vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph\n"
        "        )\n\n"
        "    def _decode_attention(self, query):\n"
        "        mid_o_buf = output_buf = lse_buf = None\n"
        "        if is_workspace_manager_initialized():\n"
        "            pass\n"
    )
    tq = tmp_path / "turboquant_attn.py"
    sd = tmp_path / "shutdown.py"
    tq.write_text(no_pn118, encoding="utf-8")
    sd.write_text(_FIXTURE_SYN_SD.read_text(encoding="utf-8"), encoding="utf-8")
    _redirect_resolver(monkeypatch, tq, sd)

    status, reason = pn399.apply()
    # sub-patch B is required and its anchor (kv_splits + PN118 box) is
    # absent -> required_anchor_missing -> SKIPPED for the TQ file.
    assert status == "skipped", reason
    assert "required_anchor_missing" in reason or "skipped" in reason.lower()
    # The file must NOT have been written (no marker, anchors untouched).
    assert pn399.GENESIS_PN399_MARKER not in tq.read_text(encoding="utf-8")


def test_disabled_by_default(monkeypatch, synthetic_targets):
    """With the env flag UNSET, PN399 is OFF (default_on=False)."""
    monkeypatch.delenv(_ENV_FLAG, raising=False)
    monkeypatch.delenv("SNDR_ENABLE_PN399_TQ_DECODE_SCRATCH_IMA", raising=False)
    tq, sd = synthetic_targets
    _redirect_resolver(monkeypatch, tq, sd)
    status, reason = pn399.apply()
    assert status == "skipped"
    assert "disabled" in reason.lower()
    assert pn399.GENESIS_PN399_MARKER not in tq.read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────
# 5. REGISTRY contract
# ─────────────────────────────────────────────────────────────────────


def test_registry_pn399_after_pn118_and_dependency_wired():
    from sndr.dispatcher.registry import PATCH_REGISTRY

    assert "PN399" in PATCH_REGISTRY, "PN399 missing from registry"
    assert "PN118" in PATCH_REGISTRY

    keys = list(PATCH_REGISTRY.keys())
    assert keys.index("PN399") > keys.index("PN118"), (
        "PN399 must be placed AFTER PN118 so insertion-order apply runs "
        "PN118 first (sub-patches B/C anchor the PN118 box)"
    )

    # PN399 still declares PN353A in requires_patches and is ordered after
    # it (PN353A stays in the registry, only its lifecycle flipped to
    # retired on dev301). On a pre-dev301 pin PN399 anchors PN353A's
    # applied reserve block; on dev301 PN353A is upstream-native + retired
    # so PN399 (default_off) skips cleanly. Ordering invariant unchanged.
    assert keys.index("PN399") > keys.index("PN353A"), (
        "PN399 must be placed AFTER PN353A so its applied reserve block "
        "exists when PN399's C2 removal anchors it (pre-dev301 pins)"
    )

    pn399_meta = PATCH_REGISTRY["PN399"]
    assert "PN118" in pn399_meta["requires_patches"]
    assert "PN353A" in pn399_meta["requires_patches"]
    assert pn399_meta["default_on"] is False
    assert pn399_meta["lifecycle"] == "experimental"
    assert pn399_meta["applies_to"]["is_turboquant"] is True
    assert pn399_meta["applies_to"]["vllm_version_range"] == (">=0.21.0", "<0.24.0")
    assert pn399_meta["upstream_pr"] == 46067
    assert pn399_meta["category"] == "stability"
    assert pn399_meta["family"] == "attention.turboquant"

    pn118_meta = PATCH_REGISTRY["PN118"]
    assert "PN399" in pn118_meta.get("composes_with", [])
    # PN353A RETIRED on dev301 (vllm#44053 merged native — see PN353A
    # registry note). PN399 was decoupled from PN353A's composes_with: on
    # dev301 PN353A cannot apply (anchor matches 0x), so it no longer
    # supplies a live reserve block for PN399's C2 anchor. PN399 keeps
    # PN353A in requires_patches (default_off) and skips cleanly when the
    # dependency is absent — verified by the after-chain skip semantics.
    pn353a_meta = PATCH_REGISTRY["PN353A"]
    assert pn353a_meta["lifecycle"] == "retired"
    assert "PN399" not in pn353a_meta.get("composes_with", [])


def test_env_flag_registered():
    from sndr.env import Flags
    assert hasattr(Flags, "PN399_TQ_DECODE_SCRATCH_IMA")
    assert Flags.PN399_TQ_DECODE_SCRATCH_IMA == "PN399_TQ_DECODE_SCRATCH_IMA"
