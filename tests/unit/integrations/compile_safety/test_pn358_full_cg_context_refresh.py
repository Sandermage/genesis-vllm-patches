# SPDX-License-Identifier: Apache-2.0
"""PN358 — FULL cudagraph forward-context refresh (vendor of vllm#44868).

Upstream bug (#44868, OPEN): a FULL CUDA graph entry bakes in references
to the forward-context tensors that existed at capture time (attention
metadata, slot mappings, ubatch slices, DP metadata, additional kwargs).
If a later step's live forward context carries FRESH tensors (different
storage), the captured graph silently replays against stale metadata —
wrong continuations / degenerate spec-decode output, no crash.

Contract pinned here (TDD, written before the implementation):
  1. Patcher carries THREE required sub-patches: the CUDAGraphEntry
     field, the FULL-capture record hook, and the pre-replay refresh
     hook. All-required => atomic (Layer 5 validates before writing).
  2. apply() on pin-form (g303916e93) target splices all three hooks
     and the result still compiles.
  3. Second apply() is idempotent (marker short-circuit).
  4. apply() on #44868's merged form self-skips via drift markers
     (reason: upstream_merged) without modifying the file.
  5. Drift markers do not collide with PN358's own replacement texts
     or its Layer-6 marker line (lint_drift_markers contract).
  6. Runtime capture walks dataclass/Mapping/Sequence trees, records
     tensor leaves with cached data_ptr + path, prunes tensor-less
     branches, and returns None when nothing was captured (so the
     replay hook costs a single `is not None` check).
  7. Genesis improvement #1 (data_ptr-pruned copy): refresh copies ONLY
     leaves whose live tensor moved storage (data_ptr changed); leaves
     still aliasing the captured storage are skipped — upstream #44868
     unconditionally copy_()s every leaf every replay (its 1-3% TPOT
     cost on metadata-heavy FULL graphs).
  8. Genesis improvement #2 (GENESIS_PN358_MODE=detect): logs stale-
     metadata hazards (per-path warn-once + counters) WITHOUT mutating
     captured tensors — the audit tool for whether the Genesis overlay
     leaks fresh tensors into captured graphs.
  9. Robustness over upstream: shape-mismatched leaves are skipped with
     a warning instead of crashing the replay path (upstream copy_
     would raise); refresh never raises (self-disables on first
     internal error); cyclic structures terminate.
 10. Anchors unique + drift markers absent in the pristine pin tree
     (opportunistic — skipped when the pin tree is not present).
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from sndr.engines.vllm.patches.compile_safety import (
    pn358_full_cg_context_refresh as m,
)

# ── Fake targets ─────────────────────────────────────────────────────
# Pin-form (g303916e93): compilation/cuda_graph.py — the three anchor
# regions byte-verified against the pristine tree (4/12/8-space
# indents; anchor count == 1 each).

PIN_CUDA_GRAPH = (
    "# fake compilation/cuda_graph.py (pin g303916e93 form)\n"
    "import dataclasses\n"
    "from typing import Any, ClassVar\n"
    "\n"
    "\n"
    "@dataclasses.dataclass\n"
    "class CUDAGraphEntry:\n"
    "    batch_descriptor: Any = None\n"
    "    cudagraph: Any | None = None\n"
    "    output: Any | None = None\n"
    "\n"
    "    # for cudagraph debugging, track the input addresses\n"
    "    # during capture, and check if they are the same during replay\n"
    "    input_addresses: list[int] | None = None\n"
    "\n"
    "\n"
    "class CUDAGraphWrapper:\n"
    "    def __call__(self, *args: Any, **kwargs: Any) -> Any | None:\n"
    "        forward_context = get_forward_context()\n"
    "        batch_descriptor = forward_context.batch_descriptor\n"
    "        entry = self.concrete_cudagraph_entries[batch_descriptor]\n"
    "\n"
    "        if entry.cudagraph is None:\n"
    "            input_addresses = [\n"
    "                x.data_ptr() for x in args if isinstance(x, torch.Tensor)\n"
    "            ]\n"
    "            entry.input_addresses = input_addresses\n"
    "            cudagraph = torch.cuda.CUDAGraph()\n"
    "\n"
    "            entry.cudagraph = cudagraph\n"
    "            return output\n"
    "\n"
    "        # Sync offloader before replay - ensures any external dependencies\n"
    "        # from pre-capture prefetches are satisfied.\n"
    "        get_offloader().sync_prev_onload()\n"
    "        entry.cudagraph.replay()\n"
    "        return entry.output\n"
)

# #44868 merged form (what cuda_graph.py looks like AFTER upstream
# lands the fix) — PN358 must self-skip on it.

MERGED_CUDA_GRAPH = (
    "# fake compilation/cuda_graph.py (post-vllm#44868 merged form)\n"
    "import dataclasses\n"
    "from typing import Any, ClassVar\n"
    "\n"
    "\n"
    "@dataclasses.dataclass\n"
    "class CUDAGraphEntry:\n"
    "    batch_descriptor: Any = None\n"
    "    cudagraph: Any | None = None\n"
    "    output: Any | None = None\n"
    "    captured_forward_context_tensors: Any | None = None\n"
    "\n"
    "    # for cudagraph debugging, track the input addresses\n"
    "    # during capture, and check if they are the same during replay\n"
    "    input_addresses: list[int] | None = None\n"
    "\n"
    "\n"
    "def _capture_forward_context_tensors() -> dict[str, Any]:\n"
    "    forward_context = get_forward_context()\n"
    "    return {}\n"
    "\n"
    "\n"
    "def _refresh_captured_forward_context_tensors(captured: Any) -> None:\n"
    "    if not captured:\n"
    "        return\n"
    "\n"
    "\n"
    "class CUDAGraphWrapper:\n"
    "    def __call__(self, *args: Any, **kwargs: Any) -> Any | None:\n"
    "        forward_context = get_forward_context()\n"
    "        batch_descriptor = forward_context.batch_descriptor\n"
    "        entry = self.concrete_cudagraph_entries[batch_descriptor]\n"
    "\n"
    "        if entry.cudagraph is None:\n"
    "            entry.input_addresses = input_addresses\n"
    "            if self.runtime_mode == CUDAGraphMode.FULL:\n"
    "                entry.captured_forward_context_tensors = (\n"
    "                    _capture_forward_context_tensors()\n"
    "                )\n"
    "            cudagraph = torch.cuda.CUDAGraph()\n"
    "\n"
    "            entry.cudagraph = cudagraph\n"
    "            return output\n"
    "\n"
    "        # Sync offloader before replay - ensures any external dependencies\n"
    "        # from pre-capture prefetches are satisfied.\n"
    "        get_offloader().sync_prev_onload()\n"
    "        if self.runtime_mode == CUDAGraphMode.FULL:\n"
    "            _refresh_captured_forward_context_tensors(\n"
    "                entry.captured_forward_context_tensors\n"
    "            )\n"
    "        entry.cudagraph.replay()\n"
    "        return entry.output\n"
)

PIN_TREE = Path("/private/tmp/candidate_pin_current/vllm/compilation")


# ── Test doubles ─────────────────────────────────────────────────────


class FakeTensor:
    """Stand-in for torch.Tensor — data_ptr / copy_ / shape only."""

    def __init__(self, ptr: int, shape: tuple = (2,), name: str = "t"):
        self._ptr = ptr
        self.shape = tuple(shape)
        self.name = name
        self.copied_from: object = None

    def data_ptr(self) -> int:
        return self._ptr

    def copy_(self, src: FakeTensor) -> FakeTensor:
        self.copied_from = src
        return self

    def __repr__(self) -> str:
        return f"FakeTensor({self.name}, ptr={self._ptr})"


class RaisingTensor(FakeTensor):
    """copy_ raises — exercises the never-raise guard rail."""

    def copy_(self, src):
        raise RuntimeError("poisoned copy_")


@dataclasses.dataclass
class FakeMeta:
    """Mirror of an attention-metadata dataclass (tensor + scalar mix)."""

    seq_lens: object
    block_tables: object
    max_seqlen: int = 0


@dataclasses.dataclass(frozen=True)
class FakeBatchDescriptor:
    """Tensor-less frozen dataclass — must be pruned at capture."""

    num_tokens: int = 2
    uniform: bool = True


def _ctx(**kwargs) -> SimpleNamespace:
    base = {
        "attn_metadata": None,
        "slot_mapping": None,
        "batch_descriptor": None,
        "ubatch_slices": None,
        "dp_metadata": None,
        "additional_kwargs": None,
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    """Torch-less tensor seam + clean module state per test."""
    monkeypatch.setattr(m, "_is_tensor", lambda o: isinstance(o, FakeTensor))
    m.reset_pn358_state()
    yield
    m.reset_pn358_state()


def _install_fake(tmp_path, monkeypatch, text):
    target = tmp_path / "cuda_graph.py"
    target.write_text(text, encoding="utf-8")
    monkeypatch.setattr(m, "resolve_vllm_file", lambda rel: str(target))
    return target


# ── Patcher shape ────────────────────────────────────────────────────


class TestPatcherShape:
    def test_three_required_sub_patches(self, tmp_path, monkeypatch):
        _install_fake(tmp_path, monkeypatch, PIN_CUDA_GRAPH)
        patcher = m._make_patcher()
        assert patcher is not None
        names = [sp.name for sp in patcher.sub_patches]
        assert names == [
            "pn358_entry_captured_fc_field",
            "pn358_capture_record",
            "pn358_replay_refresh",
        ]
        # All three are load-bearing: a partial splice (field without
        # refresh, or refresh without field) is incoherent — required
        # on every sub makes Layer 5 atomic.
        assert all(sp.required for sp in patcher.sub_patches)

    def test_drift_markers_match_44868_merged_form(
        self, tmp_path, monkeypatch
    ):
        _install_fake(tmp_path, monkeypatch, PIN_CUDA_GRAPH)
        patcher = m._make_patcher()
        assert patcher.upstream_drift_markers
        for dm in patcher.upstream_drift_markers:
            assert dm in MERGED_CUDA_GRAPH

    def test_module_tracks_pr_44868(self):
        assert "44868" in (m.__doc__ or "")
        assert "44868" in m.GENESIS_PN358_MARKER

    def test_patcher_none_when_target_missing(self, monkeypatch):
        monkeypatch.setattr(m, "resolve_vllm_file", lambda rel: None)
        assert m._make_patcher() is None
        status, reason = m.apply()
        assert status == "skipped"
        assert "not resolvable" in reason


# ── Apply semantics ──────────────────────────────────────────────────


class TestApply:
    def test_apply_pin_form_installs_all_three_hooks(
        self, tmp_path, monkeypatch
    ):
        target = _install_fake(tmp_path, monkeypatch, PIN_CUDA_GRAPH)
        status, reason = m.apply()
        assert status == "applied", reason

        out = target.read_text(encoding="utf-8")
        # Entry field installed.
        assert "genesis_pn358_captured_fc: Any | None = None" in out
        # Capture hook installed, FULL-gated, before graph construction.
        assert "genesis_pn358_capture(" in out
        assert out.index("genesis_pn358_capture(") < out.index(
            "cudagraph = torch.cuda.CUDAGraph()"
        )
        # Refresh hook installed before replay.
        assert "genesis_pn358_refresh(" in out
        assert out.index("genesis_pn358_refresh(") < out.index(
            "entry.cudagraph.replay()"
        )
        # File still compiles after the splice.
        compile(out, str(target), "exec")

    def test_second_apply_is_idempotent(self, tmp_path, monkeypatch):
        _install_fake(tmp_path, monkeypatch, PIN_CUDA_GRAPH)
        first_status, _ = m.apply()
        assert first_status == "applied"
        second_status, second_reason = m.apply()
        assert second_status == "skipped"
        assert "already applied" in second_reason

    def test_self_skips_on_44868_merged_form(self, tmp_path, monkeypatch):
        target = _install_fake(tmp_path, monkeypatch, MERGED_CUDA_GRAPH)
        status, reason = m.apply()
        assert status == "skipped"
        assert "upstream_merged" in reason
        assert target.read_text(encoding="utf-8") == MERGED_CUDA_GRAPH


# ── Lint contract (tools/lint_drift_markers.py) ──────────────────────


class TestDriftMarkerSelfCollision:
    def test_markers_not_substring_of_own_replacements(
        self, tmp_path, monkeypatch
    ):
        _install_fake(tmp_path, monkeypatch, PIN_CUDA_GRAPH)
        patcher = m._make_patcher()
        marker_line = f"# [Genesis wiring marker: {patcher.marker}]\n"
        for dm in patcher.upstream_drift_markers:
            for sp in patcher.sub_patches:
                assert dm not in sp.replacement, (
                    f"drift marker {dm!r} collides with {sp.name} "
                    f"replacement — would false-fire Layer 3"
                )
            assert dm not in marker_line


# ── Runtime: capture ─────────────────────────────────────────────────


class TestCapture:
    def test_capture_records_leaves_and_prunes_tensorless_branches(self):
        t_seq = FakeTensor(0x100, (2,), "seq_lens")
        t_blk = FakeTensor(0x200, (2, 2), "block_tables")
        t_slot = FakeTensor(0x300, (2,), "slot")
        ctx = _ctx(
            attn_metadata={
                "layer.0": FakeMeta(seq_lens=t_seq, block_tables=t_blk)
            },
            slot_mapping={"layer.0": t_slot},
            batch_descriptor=FakeBatchDescriptor(),
        )
        captured = m.genesis_pn358_capture(ctx)
        assert captured is not None
        # Tensor-bearing fields present; tensor-less ones pruned.
        assert set(captured.keys()) == {"attn_metadata", "slot_mapping"}

        leaves = {leaf.path: leaf for leaf in m.iter_captured_leaves(captured)}
        assert set(leaves) == {
            "attn_metadata['layer.0'].seq_lens",
            "attn_metadata['layer.0'].block_tables",
            "slot_mapping['layer.0']",
        }
        leaf = leaves["attn_metadata['layer.0'].seq_lens"]
        assert leaf.tensor is t_seq
        assert leaf.data_ptr == 0x100
        assert leaf.shape == (2,)
        assert m.get_stats()["captured_leaves"] == 3

    def test_capture_returns_none_when_no_tensors(self):
        ctx = _ctx(batch_descriptor=FakeBatchDescriptor())
        assert m.genesis_pn358_capture(ctx) is None

    def test_capture_handles_list_of_dicts_and_shared_tensors(self):
        shared = FakeTensor(0x400, (2,), "shared")
        ctx = _ctx(attn_metadata=[{"a": shared}, {"b": shared}])
        captured = m.genesis_pn358_capture(ctx)
        assert captured is not None
        paths = {leaf.path for leaf in m.iter_captured_leaves(captured)}
        assert "attn_metadata[0]['a']" in paths

    def test_capture_survives_cycles(self):
        cyc: dict = {"t": FakeTensor(0x500)}
        cyc["self"] = cyc
        ctx = _ctx(additional_kwargs=cyc)
        captured = m.genesis_pn358_capture(ctx)  # must terminate
        assert captured is not None

    def test_capture_never_raises(self):
        class Bomb:
            def __getattr__(self, name):
                raise RuntimeError("boom")

        assert m.genesis_pn358_capture(Bomb()) is None
        assert m.get_stats()["errors"] == 1


# ── Runtime: refresh (default mode, data_ptr-pruned) ─────────────────


def _capture_one_layer(t_captured: FakeTensor):
    ctx = _ctx(
        attn_metadata={"layer.0": FakeMeta(seq_lens=t_captured,
                                           block_tables=None)},
    )
    captured = m.genesis_pn358_capture(ctx)
    assert captured is not None
    return captured


class TestRefreshMode:
    def test_same_data_ptr_is_pruned_no_copy(self):
        t_cap = FakeTensor(0x100, (2,), "cap")
        captured = _capture_one_layer(t_cap)
        # Live context: a NEW metadata object whose tensor still aliases
        # the captured storage — the graph already reads it; copy is a
        # wasted kernel launch (upstream's 1-3% TPOT cost).
        t_live = FakeTensor(0x100, (2,), "live-alias")
        live = _ctx(
            attn_metadata={"layer.0": FakeMeta(seq_lens=t_live,
                                               block_tables=None)},
        )
        m.genesis_pn358_refresh(live, captured)
        assert t_cap.copied_from is None
        stats = m.get_stats()
        assert stats["pruned"] == 1
        assert stats["refreshed"] == 0

    def test_moved_data_ptr_is_copied_into_captured(self):
        t_cap = FakeTensor(0x100, (2,), "cap")
        captured = _capture_one_layer(t_cap)
        t_live = FakeTensor(0x999, (2,), "fresh")
        live = _ctx(
            attn_metadata={"layer.0": FakeMeta(seq_lens=t_live,
                                               block_tables=None)},
        )
        m.genesis_pn358_refresh(live, captured)
        assert t_cap.copied_from is t_live
        assert m.get_stats()["refreshed"] == 1

    def test_shape_mismatch_skips_copy_instead_of_crashing(self, caplog):
        t_cap = FakeTensor(0x100, (2,), "cap")
        captured = _capture_one_layer(t_cap)
        t_live = FakeTensor(0x999, (4,), "wrong-shape")
        live = _ctx(
            attn_metadata={"layer.0": FakeMeta(seq_lens=t_live,
                                               block_tables=None)},
        )
        with caplog.at_level(logging.WARNING):
            m.genesis_pn358_refresh(live, captured)
        assert t_cap.copied_from is None
        assert m.get_stats()["shape_mismatch"] == 1
        assert any("PN358" in r.message for r in caplog.records)

    def test_missing_key_and_none_field_skip_silently(self):
        t_cap = FakeTensor(0x100, (2,), "cap")
        captured = _capture_one_layer(t_cap)
        live = _ctx(attn_metadata={})  # layer.0 vanished
        m.genesis_pn358_refresh(live, captured)  # no raise
        live2 = _ctx()  # attn_metadata is None
        m.genesis_pn358_refresh(live2, captured)  # no raise
        assert t_cap.copied_from is None

    def test_live_leaf_not_a_tensor_counts_structural_mismatch(self):
        t_cap = FakeTensor(0x100, (2,), "cap")
        captured = _capture_one_layer(t_cap)
        live = _ctx(
            attn_metadata={"layer.0": FakeMeta(seq_lens="not-a-tensor",
                                               block_tables=None)},
        )
        m.genesis_pn358_refresh(live, captured)
        assert t_cap.copied_from is None
        assert m.get_stats()["structural_mismatch"] == 1

    def test_refresh_never_raises_and_self_disables(self, caplog):
        t_cap = RaisingTensor(0x100, (2,), "poison")
        captured = _capture_one_layer(t_cap)
        t_live = FakeTensor(0x999, (2,), "fresh")
        live = _ctx(
            attn_metadata={"layer.0": FakeMeta(seq_lens=t_live,
                                               block_tables=None)},
        )
        with caplog.at_level(logging.WARNING):
            m.genesis_pn358_refresh(live, captured)  # swallowed
        assert m.get_stats()["errors"] == 1
        # Self-disabled: subsequent calls are no-ops (no log flood).
        t_cap2 = FakeTensor(0x100, (2,), "cap2")
        captured2 = _capture_one_layer(t_cap2)
        m.genesis_pn358_refresh(live, captured2)
        assert m.get_stats()["refreshed"] == 0

    def test_refresh_handles_cyclic_live_tree(self):
        t_cap = FakeTensor(0x100)
        cyc_cap: dict = {"t": t_cap}
        cyc_cap["self"] = cyc_cap
        ctx_cap = _ctx(additional_kwargs=cyc_cap)
        captured = m.genesis_pn358_capture(ctx_cap)

        t_live = FakeTensor(0x999)
        cyc_live: dict = {"t": t_live}
        cyc_live["self"] = cyc_live
        live = _ctx(additional_kwargs=cyc_live)
        m.genesis_pn358_refresh(live, captured)  # must terminate
        assert t_cap.copied_from is t_live


# ── Runtime: detect mode (GENESIS_PN358_MODE=detect) ─────────────────


class TestDetectMode:
    def _detect(self, monkeypatch):
        monkeypatch.setenv("GENESIS_PN358_MODE", "detect")
        m.reset_pn358_state()

    def test_detect_logs_hazard_without_mutating(self, monkeypatch, caplog):
        self._detect(monkeypatch)
        t_cap = FakeTensor(0x100, (2,), "cap")
        captured = _capture_one_layer(t_cap)
        t_live = FakeTensor(0x999, (2,), "fresh")
        live = _ctx(
            attn_metadata={"layer.0": FakeMeta(seq_lens=t_live,
                                               block_tables=None)},
        )
        with caplog.at_level(logging.WARNING):
            m.genesis_pn358_refresh(live, captured)
        # No mutation in detect mode — audit only.
        assert t_cap.copied_from is None
        stats = m.get_stats()
        assert stats["stale_detected"] == 1
        assert stats["refreshed"] == 0
        hazard_logs = [
            r for r in caplog.records
            if "PN358" in r.message and "stale" in r.message
        ]
        assert len(hazard_logs) == 1
        assert "attn_metadata['layer.0'].seq_lens" in hazard_logs[0].message

    def test_detect_warns_once_per_path_but_keeps_counting(
        self, monkeypatch, caplog
    ):
        self._detect(monkeypatch)
        t_cap = FakeTensor(0x100, (2,), "cap")
        captured = _capture_one_layer(t_cap)
        t_live = FakeTensor(0x999, (2,), "fresh")
        live = _ctx(
            attn_metadata={"layer.0": FakeMeta(seq_lens=t_live,
                                               block_tables=None)},
        )
        with caplog.at_level(logging.WARNING):
            m.genesis_pn358_refresh(live, captured)
            m.genesis_pn358_refresh(live, captured)
        assert m.get_stats()["stale_detected"] == 2
        hazard_logs = [
            r for r in caplog.records
            if "PN358" in r.message and "stale" in r.message
        ]
        assert len(hazard_logs) == 1  # warn-once per path

    def test_detect_same_ptr_is_quiet(self, monkeypatch, caplog):
        self._detect(monkeypatch)
        t_cap = FakeTensor(0x100, (2,), "cap")
        captured = _capture_one_layer(t_cap)
        live = _ctx(
            attn_metadata={"layer.0": FakeMeta(
                seq_lens=FakeTensor(0x100, (2,), "alias"),
                block_tables=None)},
        )
        with caplog.at_level(logging.WARNING):
            m.genesis_pn358_refresh(live, captured)
        assert m.get_stats()["stale_detected"] == 0
        assert not [r for r in caplog.records if "stale" in r.message]


# ── Mode resolution ──────────────────────────────────────────────────


class TestModeResolution:
    def test_default_is_refresh(self, monkeypatch):
        monkeypatch.delenv("GENESIS_PN358_MODE", raising=False)
        m.reset_pn358_state()
        assert m._resolve_mode() == m.MODE_REFRESH

    def test_detect_value(self, monkeypatch):
        monkeypatch.setenv("GENESIS_PN358_MODE", "DETECT")
        m.reset_pn358_state()
        assert m._resolve_mode() == m.MODE_DETECT

    def test_unknown_value_falls_back_to_refresh(self, monkeypatch, caplog):
        monkeypatch.setenv("GENESIS_PN358_MODE", "bogus")
        m.reset_pn358_state()
        with caplog.at_level(logging.WARNING):
            assert m._resolve_mode() == m.MODE_REFRESH
        assert any("GENESIS_PN358_MODE" in r.message for r in caplog.records)


# ── Pristine pin invariants (opportunistic) ──────────────────────────


@pytest.mark.skipif(
    not (PIN_TREE / "cuda_graph.py").is_file(),
    reason="pristine pin tree not present on this machine",
)
class TestAnchorsAgainstPristinePin:
    def test_anchors_unique_and_markers_absent(self):
        src = (PIN_TREE / "cuda_graph.py").read_text(encoding="utf-8")
        for old, new in [
            (m.PN358_ENTRY_OLD, m.PN358_ENTRY_NEW),
            (m.PN358_CAPTURE_OLD, m.PN358_CAPTURE_NEW),
            (m.PN358_REPLAY_OLD, m.PN358_REPLAY_NEW),
        ]:
            assert src.count(old) == 1
            assert new not in src
        for dm in m._DRIFT_MARKERS:
            assert dm not in src

    def test_fixture_anchors_mirror_pin_form(self):
        # The fake target used above must carry the exact anchor bytes
        # the real pin file carries (fixture drift trap).
        for old in [
            m.PN358_ENTRY_OLD,
            m.PN358_CAPTURE_OLD,
            m.PN358_REPLAY_OLD,
        ]:
            assert PIN_CUDA_GRAPH.count(old) == 1
