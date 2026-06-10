# SPDX-License-Identifier: Apache-2.0
"""Build-time anchor invariant tests — P1.1 of patcher evolution plan
(2026-05-07). Catches anchor drift / false-fire / overlap bugs in CI
BEFORE they cause live boot crashes (as PN79 Sub-3 missing 3A/3B + 1E
crashed live on 2026-05-07).

Concept: each TextPatcher's anchors must satisfy 4 invariants against
pristine vllm source:
  1. Each sub_patch.anchor (OLD) appears in pristine ≥1 time AND ==1 time
  2. Each sub_patch.replacement (NEW) does NOT appear in pristine
     (false-fire guard for Layer 2 idempotency check)
  3. upstream_drift_markers do NOT appear in pristine
     (false-fire guard for Layer 3 drift detection)
  4. Anchors within same patcher are non-overlapping at distinct
     byte offsets (sequential apply safe; later anchors stay valid
     after earlier replace)

Coverage as of 2026-05-07: PN79 (17 anchors across 3 files). Extension
to other patchers postponed — PN79 was the trigger so it's first.

Pristine fixtures: tests/legacy/pristine_fixtures/*.py
(see pristine_fixtures/README.md for pin and update procedure).

Run:
    pytest tests/legacy/test_patcher_anchors_invariants.py -v
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

PRISTINE_DIR = Path(__file__).resolve().parents[3] / "tests" / "legacy" / "pristine_fixtures"


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _pristine_source(filename: str) -> str:
    """Read pristine fixture by filename (e.g. 'chunk.py'). Cached."""
    path = PRISTINE_DIR / filename
    if not path.is_file():
        pytest.skip(f"pristine fixture {filename} not found at {path}")
    return path.read_text(encoding="utf-8")


def _check_anchor_invariants(
    patch_name: str,
    pristine_filename: str,
    sub_patches,
    drift_markers=(),
):
    """The 4 invariants. Raises pytest.fail with descriptive message
    on any violation."""
    src = _pristine_source(pristine_filename)

    # Invariant 1+2: anchor unique, replacement absent
    offsets = []
    for sp_name, anchor_old, anchor_new in sub_patches:
        # 1. OLD found exactly once
        cnt_old = src.count(anchor_old)
        if cnt_old != 1:
            pytest.fail(
                f"{patch_name}.{sp_name}: anchor (OLD) found {cnt_old}x in "
                f"{pristine_filename} (expected 1). Either pristine drifted "
                f"upstream OR anchor pattern is non-unique."
            )
        # 2. NEW absent from pristine (would idempotency-false-fire)
        if anchor_new in src:
            pytest.fail(
                f"{patch_name}.{sp_name}: replacement (NEW) found in pristine "
                f"{pristine_filename} — would false-fire Layer 2 idempotency "
                f"check. Pick a different/more-specific NEW or add Genesis "
                f"marker comment to disambiguate."
            )
        offsets.append((src.find(anchor_old), len(anchor_old), sp_name))

    # 4. Anchors within patcher are non-overlapping byte ranges
    offsets.sort(key=lambda t: t[0])
    for i in range(1, len(offsets)):
        prev_start, prev_len, prev_name = offsets[i - 1]
        curr_start, _, curr_name = offsets[i]
        prev_end = prev_start + prev_len
        if prev_end > curr_start:
            pytest.fail(
                f"{patch_name}: anchors {prev_name!r} (bytes "
                f"{prev_start}-{prev_end}) and {curr_name!r} (starts at "
                f"{curr_start}) OVERLAP in {pristine_filename}. Sequential "
                f"apply will mutate {prev_name} bytes then attempt to find "
                f"{curr_name} which now points into modified region."
            )

    # 3. Drift markers absent from pristine
    for marker in drift_markers:
        if marker in src:
            pytest.fail(
                f"{patch_name}: drift marker {marker!r} found in pristine "
                f"{pristine_filename} — would false-fire Layer 3 "
                f"upstream_merged check, blocking patch on a fresh source. "
                f"Pick a more-specific marker that's unique to the post-patch "
                f"state."
            )


# ─────────────────────────────────────────────────────────────────────
# PN79 — 17 anchors across 3 files
# ─────────────────────────────────────────────────────────────────────


def _pn79_module():
    from sndr.engines.vllm.patches.attention.gdn import pn79_inplace_ssm_state as M
    return M


class TestPN79AnchorInvariants:
    """All 17 PN79 anchors against pristine vllm 0.20.2rc1.dev9+g01d4d1ad3."""

    def test_PN79_chunk_py_sub1_anchors(self):
        """Sub-1 chunk.py — 7 anchors (1A drop import, 1B fwd sig, 1C fwd
        internal call, 1D forward rewrite, 1E_SIG/1E_VAL/1E_APPLY_CALL
        high-level chunk_gated_delta_rule wrapper).
        """
        m = _pn79_module()
        _check_anchor_invariants(
            "PN79 Sub-1 chunk.py",
            "chunk.py",
            sub_patches=[
                ("1A", m.ANCHOR_1A_IMPORT_OLD, m.ANCHOR_1A_IMPORT_NEW),
                ("1B", m.ANCHOR_1B_FWD_SIG_OLD, m.ANCHOR_1B_FWD_SIG_NEW),
                ("1C", m.ANCHOR_1C_FWD_INTERNAL_OLD, m.ANCHOR_1C_FWD_INTERNAL_NEW),
                ("1D", m.ANCHOR_1D_FORWARD_OLD, m.ANCHOR_1D_FORWARD_NEW),
                ("1E_SIG", m.ANCHOR_1E_SIG_OLD, m.ANCHOR_1E_SIG_NEW),
                ("1E_VAL", m.ANCHOR_1E_VAL_OLD, m.ANCHOR_1E_VAL_NEW),
                ("1E_APPLY_CALL", m.ANCHOR_1E_APPLY_CALL_OLD,
                 m.ANCHOR_1E_APPLY_CALL_NEW),
            ],
            drift_markers=[
                "ssm_state_indices",
                "has_initial_state",
                "torch.accelerator.device_index",
            ],
        )

    def test_PN79_chunk_delta_h_py_sub2_anchors(self):
        """Sub-2 chunk_delta_h.py — 7 anchors (2A heuristics, 2B kernel
        sig, 2C kernel main flow, 2D kernel epilogue, 2E wrapper sig,
        2F wrapper body strides, 2G wrapper kernel call).
        """
        m = _pn79_module()
        _check_anchor_invariants(
            "PN79 Sub-2 chunk_delta_h.py",
            "chunk_delta_h.py",
            sub_patches=[
                ("2A", m.ANCHOR_2A_HEURISTICS_OLD, m.ANCHOR_2A_HEURISTICS_NEW),
                ("2B", m.ANCHOR_2B_KERNEL_SIG_OLD, m.ANCHOR_2B_KERNEL_SIG_NEW),
                ("2C", m.ANCHOR_2C_KERNEL_MAIN_OLD, m.ANCHOR_2C_KERNEL_MAIN_NEW),
                ("2D", m.ANCHOR_2D_KERNEL_EPILOGUE_OLD,
                 m.ANCHOR_2D_KERNEL_EPILOGUE_NEW),
                ("2E", m.ANCHOR_2E_WRAPPER_SIG_OLD, m.ANCHOR_2E_WRAPPER_SIG_NEW),
                ("2F", m.ANCHOR_2F_WRAPPER_BODY_OLD, m.ANCHOR_2F_WRAPPER_BODY_NEW),
                ("2G", m.ANCHOR_2G_WRAPPER_KERNEL_CALL_OLD,
                 m.ANCHOR_2G_WRAPPER_KERNEL_CALL_NEW),
            ],
            drift_markers=[
                "IS_CONTINUOUS_BATCHING",
                "HAS_INITIAL_STATE_MASK",
                "stride_init_state_token",
            ],
        )

    def test_PN79_gdn_linear_attn_py_sub3_anchors(self):
        """Sub-3 gdn_linear_attn.py — 3 anchors (3A forward_cuda fallback,
        3B forward_native passthrough, 3C _forward_core gather/scatter elim).

        Sub-3 has NO drift markers (the candidate marker
        `ssm_state_indices=non_spec_state_indices_tensor` already appears in
        pristine decode-path / spec-path branches at lines 1024 + 1106 — too
        generic to be a reliable drift signal).
        """
        m = _pn79_module()
        _check_anchor_invariants(
            "PN79 Sub-3 gdn_linear_attn.py",
            "gdn_linear_attn.py",
            sub_patches=[
                ("3A", m.ANCHOR_3A_FORWARD_CUDA_OLD,
                 m.ANCHOR_3A_FORWARD_CUDA_NEW),
                ("3B", m.ANCHOR_3B_FORWARD_NATIVE_OLD,
                 m.ANCHOR_3B_FORWARD_NATIVE_NEW),
                ("3C", m.ANCHOR_3C_GATHER_SCATTER_OLD,
                 m.ANCHOR_3C_GATHER_SCATTER_NEW),
            ],
            drift_markers=[],
        )

    def test_PN79_olmo_hybrid_py_sub4_anchor(self):
        """Sub-4 olmo_hybrid.py — 1 anchor (4A _forward_core gather/scatter
        elim). Same pattern as Sub-3C but call site uses free-function
        `chunk_gated_delta_rule()` instead of bound method, and pristine
        has no `assert` lines (only present in gdn_linear_attn.py)."""
        m = _pn79_module()
        _check_anchor_invariants(
            "PN79 Sub-4 olmo_hybrid.py",
            "olmo_hybrid.py",
            sub_patches=[
                ("4A", m.ANCHOR_4A_OLMO_FORWARD_CORE_OLD,
                 m.ANCHOR_4A_OLMO_FORWARD_CORE_NEW),
            ],
            drift_markers=[],
        )


# ─────────────────────────────────────────────────────────────────────
# Cross-patcher conflict: no two patches modify same byte range
# ─────────────────────────────────────────────────────────────────────


class TestCrossPatcherInvariants:
    """If two different Genesis patchers target the same file, their
    anchor regions must NOT overlap. Sequential apply order would
    invalidate later anchors otherwise.

    Currently checks only PN79 (single patcher per file, trivially
    no conflict). When more patchers register pristine_fixtures, this
    will become more meaningful.
    """

    def test_PN79_anchors_disjoint_within_each_file(self):
        """Sanity check: PN79 itself has 7 anchors in chunk.py and 7 in
        chunk_delta_h.py — they must all be at disjoint byte ranges."""
        # Already covered by Invariant 4 in TestPN79AnchorInvariants but
        # this provides an explicit summary view.
        m = _pn79_module()
        chunk = _pristine_source("chunk.py")
        chunk_delta_h = _pristine_source("chunk_delta_h.py")
        gdn = _pristine_source("gdn_linear_attn.py")

        chunk_anchors = [
            ("1A", m.ANCHOR_1A_IMPORT_OLD),
            ("1B", m.ANCHOR_1B_FWD_SIG_OLD),
            ("1C", m.ANCHOR_1C_FWD_INTERNAL_OLD),
            ("1D", m.ANCHOR_1D_FORWARD_OLD),
            ("1E_SIG", m.ANCHOR_1E_SIG_OLD),
            ("1E_VAL", m.ANCHOR_1E_VAL_OLD),
            ("1E_APPLY_CALL", m.ANCHOR_1E_APPLY_CALL_OLD),
        ]
        kernel_anchors = [
            ("2A", m.ANCHOR_2A_HEURISTICS_OLD),
            ("2B", m.ANCHOR_2B_KERNEL_SIG_OLD),
            ("2C", m.ANCHOR_2C_KERNEL_MAIN_OLD),
            ("2D", m.ANCHOR_2D_KERNEL_EPILOGUE_OLD),
            ("2E", m.ANCHOR_2E_WRAPPER_SIG_OLD),
            ("2F", m.ANCHOR_2F_WRAPPER_BODY_OLD),
            ("2G", m.ANCHOR_2G_WRAPPER_KERNEL_CALL_OLD),
        ]
        gdn_anchors = [
            ("3A", m.ANCHOR_3A_FORWARD_CUDA_OLD),
            ("3B", m.ANCHOR_3B_FORWARD_NATIVE_OLD),
            ("3C", m.ANCHOR_3C_GATHER_SCATTER_OLD),
        ]

        for src, anchors, fname in [
            (chunk, chunk_anchors, "chunk.py"),
            (chunk_delta_h, kernel_anchors, "chunk_delta_h.py"),
            (gdn, gdn_anchors, "gdn_linear_attn.py"),
        ]:
            offsets = sorted([(src.find(a), len(a), n) for n, a in anchors])
            for i in range(1, len(offsets)):
                p_start, p_len, p_name = offsets[i - 1]
                c_start, _, c_name = offsets[i]
                assert p_start + p_len <= c_start, (
                    f"{fname}: PN79 anchors {p_name} (bytes {p_start}-"
                    f"{p_start+p_len}) and {c_name} (starts {c_start}) overlap"
                )


# ─────────────────────────────────────────────────────────────────────
# Pristine fixture integrity (audit P1 finding K — vllm install absent locally)
# ─────────────────────────────────────────────────────────────────────


class TestPristineFixtureIntegrity:
    """Pristine fixtures must match their declared MD5 (README.md).
    If a fixture file gets accidentally modified (e.g., editor strips
    trailing whitespace), invariant tests pass but the fixture no
    longer represents real pristine source.
    """

    EXPECTED_MD5 = {
        # vllm pin: 0.20.2rc1.dev93+g51f22dcfd (validated 2026-05-07)
        "chunk.py": "0f66320b6b74a11d7b4f3e7ea223ecec",
        "chunk_delta_h.py": "e90e9a46606fadcf22cf5f8425f0f490",
        "gdn_linear_attn.py": "18dc6a9c0b1f615a338b468c11fcb71c",
        "olmo_hybrid.py": "63ab5a2d29b29b522693188a8da2e421",
    }

    @pytest.mark.parametrize("filename,expected", EXPECTED_MD5.items())
    def test_fixture_md5_matches_declared(self, filename, expected):
        import hashlib
        path = PRISTINE_DIR / filename
        if not path.is_file():
            pytest.skip(f"fixture {filename} not found")
        actual = hashlib.md5(path.read_bytes()).hexdigest()
        assert actual == expected, (
            f"{filename} MD5 {actual} != declared {expected}. "
            f"Fixture file was modified or replaced. "
            f"See pristine_fixtures/README.md for update procedure."
        )
