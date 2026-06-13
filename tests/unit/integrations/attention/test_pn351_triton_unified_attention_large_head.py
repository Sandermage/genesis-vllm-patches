# SPDX-License-Identifier: Apache-2.0
"""PN351 — Triton unified_attention head_dim>=512 tune (vendor of vllm#43257).

Contract pinned here (TDD, written before the dual-anchor implementation):

Pin-bump protection (the reason this batch-3 task exists), corrected
2026-06-13 against the REAL upstream-main launch shape after review:

  THREE launch-site shapes exist in the wild (verified against the live
  pin + raw upstream ``main`` + ``gh pr diff 45151``):

    (A) CURRENT pin g303916e93: tail ``USE_TD_QO=use_td_qo,\n    )`` — NO
        ``**launch_kwargs`` line. The only shape we run today.
    (B) Upstream ``main`` BEFORE #45151: main refactored the launch to
        splat a tuned-params dict, so the tail is
        ``USE_TD_QO=use_td_qo,\n        **launch_kwargs,\n    )``.
    (C) Upstream ``main`` AFTER #45151 (FP8 group-quant fused into the
        epilogue): the 7 new kwargs are spliced between ``USE_TD_QO`` and
        the existing ``**launch_kwargs,`` CONTEXT line, so the tail is
        ``USE_TD_QO=use_td_qo,\n<7 kwargs>\n        **launch_kwargs,\n    )``.

  The original single-form ``USE_TD_QO=use_td_qo,\n    )`` anchor matches
  ONLY shape (A); both future shapes (B, C) end with ``**launch_kwargs,``
  before the ``)`` and silently break it — PN351 would lose its -3-7%
  decode_TPOT win on Gemma 4 on the first main-descended pin bump.

The fix is a MULTI-ANCHOR launch sub-patch (P18B/PN32 chain convention —
required-at-least-one):
  1. Three launch variants, all ``required=False``:
       - current-pin variant: anchor ends ``USE_TD_QO=use_td_qo,\n    )``.
       - upstream-main variant: anchor ends
         ``USE_TD_QO=use_td_qo,\n        **launch_kwargs,\n    )``.
       - post-#45151 variant: anchor ends on the last of the 7 inserted
         kwargs (``USE_UE8M0=output_group_ue8m0,``) + ``**launch_kwargs,``.
     On any given pin EXACTLY ONE matches; the others soft-skip. In B and
     C our literal kwargs are inserted BEFORE ``**launch_kwargs,`` (main's
     splat is empty on head_dim>=512, so no num_warps conflict).
  2. The tile sub-patch stays ``required=True`` (its function is byte-
     identical on the current pin AND upstream main, so its anchor never
     breaks).
  3. apply() enforces "at least one launch variant applied" — a tile-only
     apply is incoherent (Sub-1 sets the tile, the launch sets the config;
     either alone is a no-op functional regression), so PN351 must FAIL
     loudly rather than report a half-applied success.

Behavior is UNCHANGED — all launch variants emit the same
``num_warps=8 if head_size >= 512 else 4`` / ``num_stages=2 if head_size
>= 512 else 3`` kwargs. Only the anchor robustness changes.

Drift-marker hygiene (lint_drift_markers.py self-collision contract):
  the drift markers must never be substrings of PN351's own emitted text.
  PN351's marker uses the prefix ``[Genesis PN351`` (defended convention,
  exempt) — verified here so the lint stays 0 across the bump.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

# Unit tests patch fresh tmp files; the Layer-0 file cache must never
# satisfy apply() from a previous run's state (same convention as PN378).
os.environ.setdefault("GENESIS_NO_PATCH_CACHE", "1")

from sndr.engines.vllm.patches.attention import (  # noqa: E402
    pn351_triton_unified_attention_large_head as m,
)

PIN_TREE = Path("/private/tmp/candidate_pin_current/vllm/v1/attention/ops")
PIN_FILE = PIN_TREE / "triton_unified_attention.py"


# ── Fixtures ─────────────────────────────────────────────────────────
# Byte-faithful copies of the anchor regions of
# triton_unified_attention.py. The _get_tile_size head is identical on
# the current pin AND on upstream main (byte-verified 2026-06-13), so a
# single fixture covers every launch shape. The launch tails differ by
# pin shape (see the module docstring): current pin (A), upstream main
# pre-#45151 (B), upstream main post-#45151 (C).

# The real _get_tile_size head — parameter order and the
# ``return 16 if element_size >= 2 else 32`` default copied byte-exact
# from /private/tmp/candidate_pin_current/vllm (matches upstream main).
PIN_TILE_REGION = (
    "def _get_tile_size(\n"
    "    head_size: int,\n"
    "    sliding_window: int,\n"
    "    element_size: int,\n"
    "    is_prefill: bool,\n"
    ") -> int:\n"
    '    """Select tile size with Gemma3-specific optimization."""\n'
    "    if _is_gemma3_attention(head_size, sliding_window):\n"
    "        # Gemma3: use 32 for decode (default is 16)\n"
    "        return 32\n"
    "\n"
    "    # Default behavior\n"
    "    if is_prefill:\n"
    "        return 32\n"
    "    # Note: tile size must be at least 32 for fp8 (element_size == 1).\n"
    "    return 16 if element_size >= 2 else 32\n"
    "\n"
    "\n"
)

# (A) CURRENT pin g303916e93 launch tail — the unique 3-kwarg sequence
# PN351 anchors on, followed directly by the closing `)` (NO launch_kwargs).
PIN_LAUNCH_TAIL = (
    "    kernel_unified_attention[grid](\n"
    "        output_ptr=out,\n"
    "        CHUNK_LOOKBACK=chunk_lookback,\n"
    "        CHUNK_SIZE=chunk_size,\n"
    "        USE_TD=use_td,\n"
    "        USE_TD_QO=use_td_qo,\n"
    "    )\n"
)

PIN_SOURCE = PIN_TILE_REGION + PIN_LAUNCH_TAIL

# (B) Upstream main BEFORE #45151 launch tail — main refactored the launch
# to splat ``**launch_kwargs`` (native launch_num_warps/stages machinery).
# The ``**launch_kwargs,`` CONTEXT line sits between USE_TD_QO and the `)`.
MAIN_LAUNCH_TAIL = (
    "    kernel_unified_attention[grid](\n"
    "        output_ptr=out,\n"
    "        CHUNK_LOOKBACK=chunk_lookback,\n"
    "        CHUNK_SIZE=chunk_size,\n"
    "        USE_TD=use_td,\n"
    "        USE_TD_QO=use_td_qo,\n"
    "        **launch_kwargs,\n"
    "    )\n"
)

MAIN_SOURCE = PIN_TILE_REGION + MAIN_LAUNCH_TAIL

# The 7 kwargs vllm#45151 inserts at the launch call site, byte-exact from
# `gh pr diff 45151` (2026-06-13). They land between `USE_TD_QO=use_td_qo,`
# and the existing ``**launch_kwargs,`` context line (NOT before a bare `)`).
PR45151_INSERTED_KWARGS = (
    "        USE_FP8_GROUP=use_fp8_group,\n"
    "        GROUP_SIZE=group_size,\n"
    "        out_group_scale_ptr=output_group_scale,\n"
    "        out_group_scale_stride_0="
    "(output_group_scale.stride(0) if use_fp8_group else 0),\n"
    "        out_group_scale_stride_1="
    "(output_group_scale.stride(1) if use_fp8_group else 1),\n"
    "        NUM_GROUPS_PER_HEAD=num_groups_per_head,\n"
    "        USE_UE8M0=output_group_ue8m0,\n"
)

# (C) Upstream main AFTER #45151 launch tail — the 7 kwargs are spliced
# between USE_TD_QO and the existing ``**launch_kwargs,`` context line.
# This is the REAL post-#45151 shape (descends from main, which already
# carries **launch_kwargs); it is NOT the pristine-pin tail + 7 kwargs.
POST45151_LAUNCH_TAIL = (
    "    kernel_unified_attention[grid](\n"
    "        output_ptr=out,\n"
    "        CHUNK_LOOKBACK=chunk_lookback,\n"
    "        CHUNK_SIZE=chunk_size,\n"
    "        USE_TD=use_td,\n"
    "        USE_TD_QO=use_td_qo,\n"
    + PR45151_INSERTED_KWARGS
    + "        **launch_kwargs,\n"
    + "    )\n"
)

POST45151_SOURCE = PIN_TILE_REGION + POST45151_LAUNCH_TAIL


# ── Helpers ──────────────────────────────────────────────────────────


def _install_fake(tmp_path, monkeypatch, source_text):
    target = tmp_path / "triton_unified_attention.py"
    target.write_text(source_text, encoding="utf-8")
    monkeypatch.setattr(m, "resolve_vllm_file", lambda rel: target)
    monkeypatch.delenv("GENESIS_DISABLE_PN351", raising=False)
    return target


# ── Patcher shape ────────────────────────────────────────────────────


class TestPatcherShape:
    def test_has_tile_and_three_launch_variants(self, tmp_path, monkeypatch):
        """One required tile sub + three required=False launch variants."""
        _install_fake(tmp_path, monkeypatch, PIN_SOURCE)
        patcher = m._make_patcher()
        assert patcher is not None
        by_name = {sp.name: sp for sp in patcher.sub_patches}
        # Tile sub-patch — required, anchor in the untouched _get_tile_size.
        assert "pn351_get_tile_size_large_head" in by_name
        assert by_name["pn351_get_tile_size_large_head"].required is True
        # Three launch variants — all required=False (at-least-one semantics).
        for name in (
            "pn351_kernel_launch_warps_stages",
            "pn351_kernel_launch_warps_stages_main",
            "pn351_kernel_launch_warps_stages_post45151",
        ):
            assert name in by_name, name
            assert by_name[name].required is False

    def test_main_anchor_shape_ends_on_launch_kwargs(self):
        """The upstream-main launch anchor must include the
        ``**launch_kwargs,`` splat line that sits between USE_TD_QO and the
        closing `)` on every main-descended pin."""
        anchor = m.PN351_LAUNCH_MAIN_OLD
        assert "        **launch_kwargs,\n" in anchor
        assert "USE_TD_QO=use_td_qo," in anchor

    def test_post45151_anchor_shape_ends_on_inserted_kwargs(self):
        """The post-#45151 launch anchor must contain the 7 kwargs #45151
        inserts AND the ``**launch_kwargs,`` context line that follows them
        (the real post-#45151 tail, descended from main)."""
        anchor = m.PN351_LAUNCH_POST45151_OLD
        assert "USE_UE8M0=output_group_ue8m0," in anchor
        assert "USE_FP8_GROUP=use_fp8_group," in anchor
        # Must still re-anchor on the stable USE_TD_QO line for uniqueness.
        assert "USE_TD_QO=use_td_qo," in anchor
        # And must include the **launch_kwargs context line (the bug the
        # original derivation dropped).
        assert "        **launch_kwargs,\n" in anchor

    def test_all_launch_variants_emit_identical_behavior(self):
        """Behavior unchanged: all three variants set the same warps/stages."""
        for repl in (
            m.PN351_LAUNCH_NEW,
            m.PN351_LAUNCH_MAIN_NEW,
            m.PN351_LAUNCH_POST45151_NEW,
        ):
            assert "num_warps=8 if head_size >= 512 else 4," in repl
            assert "num_stages=2 if head_size >= 512 else 3," in repl
        # The two main-shaped variants keep main's **launch_kwargs splat,
        # inserting our literals BEFORE it (so the splat applies last).
        for repl in (m.PN351_LAUNCH_MAIN_NEW, m.PN351_LAUNCH_POST45151_NEW):
            assert "        **launch_kwargs,\n" in repl
            warps_pos = repl.index("num_warps=8 if head_size >= 512 else 4,")
            kwargs_pos = repl.index("        **launch_kwargs,\n")
            assert warps_pos < kwargs_pos, "literals must precede the splat"

    def test_patcher_none_when_target_missing(self, monkeypatch):
        monkeypatch.setattr(m, "resolve_vllm_file", lambda rel: None)
        assert m._make_patcher() is None


# ── Apply on the pristine pin form ───────────────────────────────────


class TestApplyPristine:
    def test_apply_pristine_installs_tile_and_pristine_launch(
        self, tmp_path, monkeypatch
    ):
        target = _install_fake(tmp_path, monkeypatch, PIN_SOURCE)
        status, reason = m.apply()
        assert status == "applied", reason
        out = target.read_text(encoding="utf-8")
        # Tile fast-path installed.
        assert "if head_size >= 512 and is_prefill and element_size == 1:" in out
        assert "        return 64\n" in out
        # Launch kwargs installed exactly once.
        assert out.count("num_warps=8 if head_size >= 512 else 4,") == 1
        assert out.count("num_stages=2 if head_size >= 512 else 3,") == 1
        # File still compiles after the splice.
        compile(out, str(target), "exec")

    def test_second_apply_is_idempotent(self, tmp_path, monkeypatch):
        _install_fake(tmp_path, monkeypatch, PIN_SOURCE)
        first, fr = m.apply()
        assert first == "applied", fr
        second, sr = m.apply()
        assert second == "applied"
        assert "idempotent" in sr.lower()


# ── Apply on the upstream-main pin form (B) — pre-#45151 launch_kwargs ─


class TestApplyMain:
    def test_current_pin_anchor_no_longer_matches_main(self):
        """Proof the break is real: the current-pin single-form anchor must
        NOT match the upstream-main launch tail (it ends with
        ``**launch_kwargs,`` before the `)`), which is why Variant B
        exists."""
        assert m.PN351_LAUNCH_OLD not in MAIN_LAUNCH_TAIL

    def test_apply_main_uses_main_variant(self, tmp_path, monkeypatch):
        target = _install_fake(tmp_path, monkeypatch, MAIN_SOURCE)
        status, reason = m.apply()
        assert status == "applied", reason
        assert "upstream-main anchor variant" in reason
        out = target.read_text(encoding="utf-8")
        # The tune still lands via the upstream-main variant, exactly once.
        assert out.count("num_warps=8 if head_size >= 512 else 4,") == 1
        assert out.count("num_stages=2 if head_size >= 512 else 3,") == 1
        # main's **launch_kwargs splat is preserved (not clobbered) and our
        # literals sit before it.
        assert out.count("        **launch_kwargs,\n") == 1
        warps_pos = out.index("num_warps=8 if head_size >= 512 else 4,")
        kwargs_pos = out.index("        **launch_kwargs,\n")
        assert warps_pos < kwargs_pos
        compile(out, str(target), "exec")


# ── Apply on the post-#45151 pin form (C) — the protection contract ──


class TestApplyPost45151:
    def test_current_pin_anchor_no_longer_matches_post45151(self):
        """Proof the break is real: the current-pin single-form anchor must
        NOT match the post-#45151 launch tail (which carries the 7 kwargs
        AND the ``**launch_kwargs,`` context line). This is the exact
        BLOCKER the review caught — the post-#45151 tail descends from main,
        not from the pristine pin."""
        assert m.PN351_LAUNCH_OLD not in POST45151_LAUNCH_TAIL
        # The post-#45151 source must end with **launch_kwargs before `)`,
        # proving the fixture is the REAL main-descended shape.
        assert "        **launch_kwargs,\n    )\n" in POST45151_SOURCE

    def test_apply_post45151_uses_post45151_variant(
        self, tmp_path, monkeypatch
    ):
        target = _install_fake(tmp_path, monkeypatch, POST45151_SOURCE)
        status, reason = m.apply()
        assert status == "applied", reason
        assert "post-#45151 anchor variant" in reason
        out = target.read_text(encoding="utf-8")
        # The tune still lands via the post-#45151 variant.
        assert out.count("num_warps=8 if head_size >= 512 else 4,") == 1
        assert out.count("num_stages=2 if head_size >= 512 else 3,") == 1
        # #45151's own kwargs are preserved (not clobbered).
        assert "USE_UE8M0=output_group_ue8m0," in out
        assert "USE_FP8_GROUP=use_fp8_group," in out
        # main's **launch_kwargs splat is preserved too; our literals slot
        # after the 7 kwargs and before the splat.
        assert out.count("        **launch_kwargs,\n") == 1
        warps_pos = out.index("num_warps=8 if head_size >= 512 else 4,")
        kwargs_pos = out.index("        **launch_kwargs,\n")
        ue8m0_pos = out.index("USE_UE8M0=output_group_ue8m0,")
        assert ue8m0_pos < warps_pos < kwargs_pos
        compile(out, str(target), "exec")


# ── At-least-one launch variant enforcement ──────────────────────────


class TestAtLeastOneLaunchVariant:
    def test_fails_when_no_launch_variant_matches(self, tmp_path, monkeypatch):
        """If the tile anchor matches but BOTH launch variants miss (a
        future drift that breaks both shapes), apply() must FAIL — a
        tile-only apply is functionally incoherent."""
        # Tile region intact, launch tail corrupted so neither variant hits.
        broken = PIN_TILE_REGION + (
            "    kernel_unified_attention[grid](\n"
            "        output_ptr=out,\n"
            "        SOME_RENAMED_KWARG=value,\n"
            "    )\n"
        )
        target = _install_fake(tmp_path, monkeypatch, broken)
        status, reason = m.apply()
        assert status == "failed", reason
        # The file must not be half-patched (no marker, no tile change left
        # committed) — the patcher validates all anchors before writing.
        assert "num_warps=8 if head_size >= 512" not in target.read_text(
            encoding="utf-8"
        )


# ── Drift-marker hygiene (lint_drift_markers.py contract) ────────────


class TestDriftMarkerHygiene:
    def test_markers_not_substring_of_own_emitted_text(self):
        patcher = m._make_patcher_for_source(PIN_SOURCE, "/tmp/fake.py")
        marker_line = f"# [Genesis wiring marker: {patcher.marker}]\n"
        assert patcher.upstream_drift_markers, "drift markers must exist"
        for dm in patcher.upstream_drift_markers:
            if dm.startswith("[Genesis"):
                continue  # defended convention — exempt
            for sp in patcher.sub_patches:
                assert dm not in sp.replacement, (
                    f"drift marker {dm!r} collides with {sp.name} replacement"
                )
            assert dm not in marker_line


# ── Pristine pin invariants (opportunistic) ──────────────────────────


@pytest.mark.skipif(
    not PIN_FILE.is_file(),
    reason="pristine pin tree not present on this machine",
)
class TestAnchorsAgainstPristinePin:
    def test_current_pin_launch_anchor_unique(self):
        src = PIN_FILE.read_text(encoding="utf-8")
        # Variant A (current pin) matches exactly once.
        assert src.count(m.PN351_LAUNCH_OLD) == 1
        # The upstream-main and post-#45151 anchors must NOT match the
        # current pin (it has no **launch_kwargs and no #45151 kwargs), so
        # the three variants are mutually exclusive.
        assert src.count(m.PN351_LAUNCH_MAIN_OLD) == 0
        assert src.count(m.PN351_LAUNCH_POST45151_OLD) == 0

    def test_tile_anchor_unique(self):
        src = PIN_FILE.read_text(encoding="utf-8")
        assert src.count(m.PN351_TILE_OLD) == 1

    def test_replacements_absent_in_pristine(self):
        src = PIN_FILE.read_text(encoding="utf-8")
        assert m.PN351_LAUNCH_NEW not in src
        assert m.PN351_LAUNCH_MAIN_NEW not in src
        assert m.PN351_LAUNCH_POST45151_NEW not in src
        assert m.PN351_TILE_NEW not in src
