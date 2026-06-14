# SPDX-License-Identifier: Apache-2.0
"""Cross-patch anchor-overlap detection in tools/lint_drift_markers.py
(deep-audit 2026-06-14 #2).

Two separate patches whose REQUIRED anchors overlap the same byte span in one
upstream file are a latent half-patched-boot footgun (the P23_WIRE/PN368
marlin_moe.py case). These tests pin the pure detection logic without a vLLM
tree, plus the file-reading collector on a temp fixture.
"""
from __future__ import annotations

import importlib.util
import pathlib

_TOOL = pathlib.Path(__file__).resolve().parents[3] / "tools" / "lint_drift_markers.py"
_spec = importlib.util.spec_from_file_location("lint_drift_markers", _TOOL)
lint = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lint)


class TestLocateAnchorSpans:
    def test_locates_unique_anchor(self):
        content = "AAAA\nBBBB\nCCCC\n"
        located = lint.locate_anchor_spans(
            content, [{"patch_id": "P1", "label": "x", "anchor": "BBBB"}]
        )
        assert len(located) == 1
        assert located[0]["start"] == 5
        assert located[0]["end"] == 9

    def test_drops_absent_anchor(self):
        located = lint.locate_anchor_spans(
            "AAAA", [{"patch_id": "P1", "label": "x", "anchor": "ZZ"}]
        )
        assert located == []

    def test_drops_ambiguous_anchor(self):
        # "AA" occurs twice -> not positionable -> dropped.
        located = lint.locate_anchor_spans(
            "AA__AA", [{"patch_id": "P1", "label": "x", "anchor": "AA"}]
        )
        assert located == []


class TestFindDestructiveCollisions:
    def _p(self, pid, anchor, replacement, label="a"):
        return {"patch_id": pid, "label": label, "anchor": anchor,
                "replacement": replacement}

    FILE = "L0\nAAA\nBBB\nCCC\nDDD\nL5\n"

    def test_destructive_overlap_flagged(self):
        # P1 replaces "AAA\nBBB" (removing BBB); P2's anchor is "BBB\nCCC".
        # Applying P1 destroys P2's anchor -> collision.
        patches = [
            self._p("P1", "AAA\nBBB\n", "AAA\n"),
            self._p("P2", "BBB\nCCC\n", "BBB\nCCC2\n"),
        ]
        found = lint.find_destructive_collisions(self.FILE, patches, set())
        assert len(found) == 1
        assert {found[0]["patch_a"], found[0]["patch_b"]} == {"P1", "P2"}
        assert found[0]["a_breaks_b"] is True

    def test_chaining_overlap_not_flagged(self):
        # P1's replacement PRESERVES P2's anchor "BBB\nCCC" -> they compose.
        patches = [
            self._p("P1", "AAA\nBBB\n", "AAA2\nBBB\n"),
            self._p("P2", "BBB\nCCC\n", "BBB\nCCC2\n"),
        ]
        assert lint.find_destructive_collisions(self.FILE, patches, set()) == []

    def test_disjoint_not_flagged(self):
        patches = [
            self._p("P1", "AAA\n", "AAA2\n"),
            self._p("P2", "DDD\n", "DDD2\n"),
        ]
        assert lint.find_destructive_collisions(self.FILE, patches, set()) == []

    def test_same_patch_not_flagged(self):
        patches = [
            self._p("P1", "AAA\nBBB\n", "AAA\n"),
            self._p("P1", "BBB\nCCC\n", "BBB\nCCC2\n"),
        ]
        assert lint.find_destructive_collisions(self.FILE, patches, set()) == []

    def test_declared_conflict_excluded(self):
        patches = [
            self._p("P23_WIRE", "AAA\nBBB\n", "AAA\n"),
            self._p("PN368", "BBB\nCCC\n", "BBB\nCCC2\n"),
        ]
        declared = {frozenset(("P23_WIRE", "PN368"))}
        assert lint.find_destructive_collisions(self.FILE, patches, declared) == []
        assert len(lint.find_destructive_collisions(self.FILE, patches, set())) == 1

    def test_ambiguous_anchor_skipped(self):
        # "XX" occurs twice -> not positionable -> excluded.
        patches = [
            self._p("P1", "XX", "Y"),
            self._p("P2", "XX", "Z"),
        ]
        assert lint.find_destructive_collisions("XX__XX", patches, set()) == []


class TestCollector:
    class _FakeSub:
        def __init__(self, anchor, replacement, required=True):
            self.anchor = anchor
            self.replacement = replacement
            self.required = required
            self.name = "s"

    class _FakePatcher:
        def __init__(self, target, subs, name="fp"):
            self.target_file = target
            self.patch_name = name
            self.sub_patches = [
                TestCollector._FakeSub(a, r) for a, r in subs
            ]

    def test_collector_flags_undeclared_overlap_in_real_file(self, tmp_path):
        f = tmp_path / "marlin_moe.py"
        # Two anchors that overlap on the shared "use_fp32_reduce=True" block.
        f.write_text(
            "def w13():\n"
            "    size_n = w13_num_shards * N\n"
            "    use_atomic_add=False\n"
            "    use_fp32_reduce=True\n"
            "    is_zp_float=False\n"
            "    return 1\n"
        )
        a_p23 = ("    size_n = w13_num_shards * N\n"
                 "    use_atomic_add=False\n"
                 "    use_fp32_reduce=True\n")
        # P23's replacement rewrites use_fp32_reduce -> removes the shared
        # block that PN368's anchor needs.
        r_p23 = ("    size_n = w13_num_shards * N\n"
                 "    use_atomic_add=True\n"
                 "    use_fp32_reduce=False\n")
        a_pn368 = ("    use_atomic_add=False\n"
                   "    use_fp32_reduce=True\n"
                   "    is_zp_float=False\n")
        r_pn368 = ("    use_atomic_add=True\n"
                   "    use_fp32_reduce=True\n"
                   "    is_zp_float=False\n")
        entries = [
            ("mod.p23", "_make", ["P23_WIRE"],
             self._FakePatcher(str(f), [(a_p23, r_p23)])),
            ("mod.pn368", "_make", ["PN368"],
             self._FakePatcher(str(f), [(a_pn368, r_pn368)])),
        ]
        # Undeclared -> flagged.
        found = lint.collect_cross_patch_collisions(entries, declared=set())
        assert len(found) == 1
        assert {found[0]["patch_a"], found[0]["patch_b"]} == {"P23_WIRE", "PN368"}
        assert found[0]["file"] == str(f)
        # Declared -> clean.
        declared = {frozenset(("P23_WIRE", "PN368"))}
        assert lint.collect_cross_patch_collisions(entries, declared) == []


class TestDeclaredPairsFromRegistry:
    def test_p23_pn368_pair_now_declared(self):
        # The Batch-1 fix declared this pair; the lint must treat it as handled.
        declared = lint._declared_conflict_pairs()
        assert frozenset(("P23_WIRE", "PN368")) in declared
        assert frozenset(("PN390", "PN369")) in declared


class TestDecideExit:
    def test_cross_patch_collision_fails_exit(self):
        report = {"summary": {"patchers_checked": 5, "violations": 0,
                              "cross_patch_collisions": 1}}
        assert lint.decide_exit(report) == 1

    def test_clean_passes(self):
        report = {"summary": {"patchers_checked": 5, "violations": 0,
                              "cross_patch_collisions": 0}}
        assert lint.decide_exit(report) == 0
