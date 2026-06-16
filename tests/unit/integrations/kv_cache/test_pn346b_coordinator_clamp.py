# SPDX-License-Identifier: Apache-2.0
"""PN346B — coordinator-half clamp (vendor of OPEN vllm#45614).

Contract pinned here (TDD, written before the implementation):

  1. PN346B targets v1/core/kv_cache_coordinator.py (the COORDINATOR
     half of #45614). It must NOT target single_type_kv_cache_manager.py
     — that is PN346's file (the MANAGER half of the SAME PR). The two
     halves compose and ship together.
  2. The fix replaces the naked
        curr_hit_length = _new_hit_length
     inside HybridKVCacheCoordinator.find_longest_cache_hit's
     fixed-point loop with the clamped
        curr_hit_length = min(curr_hit_length, _new_hit_length)
     so the hit length is monotonically non-increasing across the
     fixed-point iteration (matches the manager-half guard PN346).
  3. The anchor is PIN-AGNOSTIC: it resolves byte-identically and
     uniquely (count==1) on BOTH the live dev491 (g1033ffac2) container
     and the PROD pin (g626fa9bba). It deliberately excludes the line
     immediately above (`if drop_eagle_block:` on dev491 vs
     `if use_eagle:` on PROD) which diverges by pin.
  4. apply() on a pristine coordinator installs the clamp; the patched
     file still compiles.
  5. Second apply() is idempotent (marker short-circuit → "applied"
     idempotent / "skipped").
  6. apply() on #45614's merged form self-skips via drift markers
     without touching the file.
  7. PN346B is opt-out-only (default-ON), mirroring PN346: it honors
     GENESIS_DISABLE_PN346B and ignores GENESIS_ENABLE_PN346B (the
     missing half of a correctness fix already shipping default-ON; a
     half-fix is worse than none).
  8. Drift markers fire on the merged form and do not collide with the
     module's own Genesis marker line.

Upstream regression test reference (do NOT run here — lives in the vLLM
tree, not Genesis): test_hybrid_mamba_eagle_does_not_reuse_lookahead_state
in tests/v1/core/test_prefix_caching.py (carried by PR #45614).
"""
from __future__ import annotations

import os
from pathlib import Path

# Unit tests patch fresh tmp files; the Layer-0 file cache must never
# satisfy apply() from a previous run's state.
os.environ.setdefault("GENESIS_NO_PATCH_CACHE", "1")

from sndr.engines.vllm.patches.kv_cache import (  # noqa: E402
    pn346b_mamba_mtp_apc_coordinator_clamp as m,
)

# ── Fake coordinator targets ─────────────────────────────────────────
# dev491-form (g1033ffac2): byte-faithful copy of the anchor region in
# v1/core/kv_cache_coordinator.py (lines 659-670, 16-space body indent).
# The loop unpacks `(spec, group_ids, manager_cls, use_eagle)` and the
# line above the anchor is `if drop_eagle_block:`.
DEV491_COORDINATOR = (
    "# fake v1/core/kv_cache_coordinator.py (pin g1033ffac2 / dev491 form)\n"
    "\n"
    "\n"
    "class HybridKVCacheCoordinator:\n"
    "    def find_longest_cache_hit(self):\n"
    "        longest_hit_length = 0\n"
    "        curr_hit_length = 0\n"
    "        eagle_verified = set()\n"
    "        while True:\n"
    "            for idx, (spec, group_ids, manager_cls, use_eagle) in (\n"
    "                enumerate(self.attention_groups)\n"
    "            ):\n"
    "                drop_eagle_block = use_eagle and idx not in eagle_verified\n"
    "                hit_blocks = self._lookup(\n"
    "                )\n"
    "                _new_hit_length = len(hit_blocks[0]) * spec.block_size\n"
    "                if drop_eagle_block:\n"
    "                    eagle_verified.add(idx)\n"
    "                elif _new_hit_length < curr_hit_length:\n"
    "                    # length shrunk; invalidate previous eagle verifications\n"
    "                    eagle_verified.clear()\n"
    "                curr_hit_length = _new_hit_length\n"
    "                for group_id, blocks in zip(group_ids, hit_blocks):\n"
    "                    hit_blocks_by_group[group_id] = blocks\n"
    "\n"
    "                longest_hit_length = max(longest_hit_length, curr_hit_length)\n"
    "            break\n"
)

# PROD-form (g626fa9bba): same anchor region but the line above is
# `if use_eagle:` and the loop unpacks `(spec, group_ids, manager_cls)`.
PROD_COORDINATOR = (
    "# fake v1/core/kv_cache_coordinator.py (pin g626fa9bba / PROD form)\n"
    "\n"
    "\n"
    "class HybridKVCacheCoordinator:\n"
    "    def find_longest_cache_hit(self):\n"
    "        longest_hit_length = 0\n"
    "        curr_hit_length = 0\n"
    "        eagle_verified = set()\n"
    "        while True:\n"
    "            for idx, (spec, group_ids, manager_cls) in (\n"
    "                enumerate(self.attention_groups)\n"
    "            ):\n"
    "                use_eagle = (\n"
    "                    idx in self.eagle_attn_group_indices\n"
    "                    and idx not in eagle_verified\n"
    "                )\n"
    "                hit_blocks = self._lookup(\n"
    "                )\n"
    "                _new_hit_length = len(hit_blocks[0]) * spec.block_size\n"
    "                if use_eagle:\n"
    "                    eagle_verified.add(idx)\n"
    "                elif _new_hit_length < curr_hit_length:\n"
    "                    # length shrunk; invalidate previous eagle verifications\n"
    "                    eagle_verified.clear()\n"
    "                curr_hit_length = _new_hit_length\n"
    "                for group_id, blocks in zip(group_ids, hit_blocks):\n"
    "                    hit_blocks_by_group[group_id] = blocks\n"
    "            break\n"
)

# #45614 merged form — what the coordinator looks like AFTER the PR
# lands (naked assignment → clamped). PN346B must self-skip on this.
MERGED_COORDINATOR = DEV491_COORDINATOR.replace(
    "                curr_hit_length = _new_hit_length\n",
    "                curr_hit_length = min(curr_hit_length, _new_hit_length)\n",
).replace("(pin g1033ffac2 / dev491 form)", "(post-vllm#45614 merged form)")

PIN_TREE = Path("/private/tmp/candidate_pin_current/vllm/v1/core")


# ── Helpers ──────────────────────────────────────────────────────────


def _install_fake(tmp_path, monkeypatch, text):
    coord = tmp_path / "kv_cache_coordinator.py"
    coord.write_text(text, encoding="utf-8")

    def _resolve(rel):
        if rel.endswith("kv_cache_coordinator.py"):
            return str(coord)
        return None

    monkeypatch.setattr(m, "resolve_vllm_file", _resolve)
    return coord


# ── Anchor shape (pin-agnostic) ──────────────────────────────────────


class TestAnchor:
    def test_anchor_resolves_unique_on_dev491(self):
        assert DEV491_COORDINATOR.count(m.PN346B_ANCHOR_OLD) == 1

    def test_anchor_resolves_unique_on_prod(self):
        assert PROD_COORDINATOR.count(m.PN346B_ANCHOR_OLD) == 1

    def test_anchor_excludes_pin_divergent_if_line(self):
        # The `if drop_eagle_block:` / `if use_eagle:` line must NOT be
        # part of the anchor, or it would resolve on only one pin.
        assert "if drop_eagle_block:" not in m.PN346B_ANCHOR_OLD
        assert "if use_eagle:" not in m.PN346B_ANCHOR_OLD

    def test_anchor_carries_naked_assignment(self):
        assert "curr_hit_length = _new_hit_length\n" in m.PN346B_ANCHOR_OLD
        # And the replacement carries the clamp.
        assert (
            "curr_hit_length = min(curr_hit_length, _new_hit_length)\n"
            in m.PN346B_ANCHOR_NEW
        )

    def test_clamp_absent_in_pristine_fixtures(self):
        for src in (DEV491_COORDINATOR, PROD_COORDINATOR):
            assert "min(curr_hit_length" not in src


# ── Patcher / target shape ───────────────────────────────────────────


class TestTargetShape:
    def test_targets_coordinator_not_single_type_manager(self):
        src = Path(m.__file__).read_text(encoding="utf-8")
        assert (
            'resolve_vllm_file("v1/core/kv_cache_coordinator.py")' in src
        )
        # PN346B must NOT patch PN346's file.
        assert (
            'resolve_vllm_file("v1/core/single_type_kv_cache_manager.py")'
            not in src
        )

    def test_module_references_sibling_pn346(self):
        doc = m.__doc__ or ""
        assert "PN346" in doc
        assert "45614" in doc

    def test_marker_constant_distinct_from_pn346(self):
        # Must not reuse PN346's marker / anchor constant names so the
        # P85 dual-anchor test (which imports PN346_ANCHOR_OLD/NEW) is
        # unaffected.
        assert "PN346B" in m.GENESIS_PN346B_MARKER
        assert hasattr(m, "PN346B_ANCHOR_OLD")
        assert hasattr(m, "PN346B_ANCHOR_NEW")
        assert not hasattr(m, "PN346_ANCHOR_OLD")


# ── Apply semantics ──────────────────────────────────────────────────


class TestApply:
    def test_apply_dev491_installs_clamp(self, tmp_path, monkeypatch):
        coord = _install_fake(tmp_path, monkeypatch, DEV491_COORDINATOR)
        status, reason = m.apply()
        assert status == "applied", reason
        out = coord.read_text(encoding="utf-8")
        assert "curr_hit_length = min(curr_hit_length, _new_hit_length)" in out
        # naked assignment gone (only the clamped form remains).
        assert out.count("curr_hit_length = _new_hit_length\n") == 0
        assert m.GENESIS_PN346B_MARKER in out
        compile(out, str(coord), "exec")

    def test_apply_prod_installs_clamp(self, tmp_path, monkeypatch):
        coord = _install_fake(tmp_path, monkeypatch, PROD_COORDINATOR)
        status, reason = m.apply()
        assert status == "applied", reason
        out = coord.read_text(encoding="utf-8")
        assert "curr_hit_length = min(curr_hit_length, _new_hit_length)" in out
        assert m.GENESIS_PN346B_MARKER in out
        compile(out, str(coord), "exec")

    def test_is_applied_true_after_apply(self, tmp_path, monkeypatch):
        _install_fake(tmp_path, monkeypatch, DEV491_COORDINATOR)
        assert m.is_applied() is False
        status, reason = m.apply()
        assert status == "applied", reason
        assert m.is_applied() is True

    def test_second_apply_idempotent(self, tmp_path, monkeypatch):
        _install_fake(tmp_path, monkeypatch, DEV491_COORDINATOR)
        first_status, first_reason = m.apply()
        assert first_status == "applied", first_reason
        second_status, _second_reason = m.apply()
        assert second_status == "applied"  # IDEMPOTENT maps to "applied"

    def test_self_skips_on_45614_merged_form(self, tmp_path, monkeypatch):
        coord = _install_fake(tmp_path, monkeypatch, MERGED_COORDINATOR)
        status, _reason = m.apply()
        assert status in ("skipped", "applied")
        # On merged form the file must not gain a SECOND clamp / Genesis
        # marker (self-skip), it stays as the upstream merged text.
        out = coord.read_text(encoding="utf-8")
        assert out.count(
            "curr_hit_length = min(curr_hit_length, _new_hit_length)"
        ) == 1
        assert m.GENESIS_PN346B_MARKER not in out

    def test_opt_out_disables(self, tmp_path, monkeypatch):
        coord = _install_fake(tmp_path, monkeypatch, DEV491_COORDINATOR)
        monkeypatch.setenv("GENESIS_DISABLE_PN346B", "1")
        status, _reason = m.apply()
        assert status == "skipped"
        assert coord.read_text(encoding="utf-8") == DEV491_COORDINATOR

    def test_default_on_ignores_enable_flag(self, tmp_path, monkeypatch):
        # opt-out-only: GENESIS_ENABLE_PN346B is NOT consulted; the patch
        # applies regardless (mirrors PN346).
        coord = _install_fake(tmp_path, monkeypatch, DEV491_COORDINATOR)
        monkeypatch.delenv("GENESIS_ENABLE_PN346B", raising=False)
        monkeypatch.delenv("GENESIS_DISABLE_PN346B", raising=False)
        status, reason = m.apply()
        assert status == "applied", reason
        assert "min(curr_hit_length" in coord.read_text(encoding="utf-8")

    def test_apply_skips_when_target_missing(self, monkeypatch):
        monkeypatch.setattr(m, "resolve_vllm_file", lambda rel: None)
        status, _reason = m.apply()
        assert status == "skipped"


# ── Drift markers ────────────────────────────────────────────────────


class TestDriftMarkers:
    def test_markers_fire_on_merged_form(self, tmp_path, monkeypatch):
        _install_fake(tmp_path, monkeypatch, DEV491_COORDINATOR)
        patcher = m._make_patcher()
        assert patcher is not None
        assert patcher.upstream_drift_markers
        assert any(
            dm in MERGED_COORDINATOR
            for dm in patcher.upstream_drift_markers
            if not dm.startswith("[Genesis")
        )

    def test_markers_no_self_collision(self, tmp_path, monkeypatch):
        _install_fake(tmp_path, monkeypatch, DEV491_COORDINATOR)
        patcher = m._make_patcher()
        marker_line = f"# [Genesis wiring marker: {patcher.marker}]\n"
        for dm in patcher.upstream_drift_markers:
            if dm.startswith("[Genesis"):
                continue
            assert dm not in marker_line


# ── Pristine pin invariants (opportunistic) ──────────────────────────


import pytest  # noqa: E402


@pytest.mark.skipif(
    not (PIN_TREE / "kv_cache_coordinator.py").is_file(),
    reason="pristine pin tree not present on this machine",
)
class TestAnchorsAgainstPristinePin:
    def test_anchor_unique_and_clamp_absent(self):
        src = (PIN_TREE / "kv_cache_coordinator.py").read_text(
            encoding="utf-8"
        )
        assert src.count(m.PN346B_ANCHOR_OLD) == 1
        assert "min(curr_hit_length" not in src

    def test_pn346_sibling_file_untouched(self):
        sibling = PIN_TREE / "single_type_kv_cache_manager.py"
        if not sibling.is_file():
            pytest.skip("sibling file absent in this pin")
        sib_src = sibling.read_text(encoding="utf-8")
        assert m.PN346B_ANCHOR_OLD not in sib_src
