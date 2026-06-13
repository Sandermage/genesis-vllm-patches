# SPDX-License-Identifier: Apache-2.0
"""TDD for PN383 — vendor of OPEN PR vllm#44784 (KV-offload + MTP
cuMemcpyBatchAsync segfault gate) + two Genesis extensions.

Upstream #44784 (issue #44780): ``OffloadingConnectorScheduler``
schedules EAGLE/MTP draft-attention groups into store/load. The draft
group's trailing block is rewritten by the drafter every step (no
stable hash) and the mis-sized block math produces an out-of-bounds
GPU block index that segfaults inside ``cuMemcpyBatchAsync`` — blocking
native CPU KV offload on EVERY MTP config (ours included: Qwen3.6
MTP K=3). The PR's four scheduler.py hunks: (1) ``is_eagle_group``
field on ``GroupOffloadConfig``, (2) eagle-group detection in
``SchedulerOffloadConfig.from_spec``, (3) eagle handling in ``_lookup``
(query-one-extra-block + pop of the volatile tail), (4) trailing-block
exclusion in ``_build_store_jobs``.

Genesis extensions (roadmap chunk-4 Theme 5 — verified against the
pristine pin tree 0.22.1rc1.dev259+g303916e93):

  1. Qwen3.6-specific ``is_eagle_group`` flagging. VERIFIED on the
     pin: ``KVCacheGroupSpec.is_eagle_group`` is only ever set by
     ``_annotate_eagle_groups_deepseek_v4`` — for Qwen3.5/3.6 MTP it
     stays False everywhere, so upstream's fallback flags ALL groups
     as eagle (every group loses its trailing block from store/load:
     prefix-reuse hit-rate loss across the whole target model). PN383
     narrows the fallback: flag only groups containing a
     drafter-convention layer name (the Qwen3.5/3.6 MTP drafter loads
     under the ``mtp`` module prefix); the all-groups fallback stays
     as a loud last resort.
  2. Re-add the pre-DMA bounds check the upstream PR described for
     gpu_worker.py but dropped from its final diff: validate block ids
     against the per-tensor row counts BEFORE the descriptor pointers
     are computed, raising a clear RuntimeError instead of a silent
     CUDA segfault.

Dormant by design: default OFF and behavior-reachable only when a KV
offloading backend is configured (none on today's PROD).

These tests verify textually (byte-verbatim fixture regions generated
from the pin — see ``_pn383_fixture_regions.py``), BEHAVIORALLY (the
patched ``from_spec`` is exec'd with fake specs to prove the Qwen3.6
narrowing), and opportunistically against the real pristine tree at
/private/tmp/candidate_pin_current.
"""
from __future__ import annotations

import ast
import types
from pathlib import Path

import pytest

from tests.unit.integrations.offload._pn383_fixture_regions import (
    BUILD_STORE_REGION,
    FROM_SPEC_REGION,
    GROUP_CONFIG_REGION,
    LOOKUP_REGION,
    WORKER_REGION,
)

PIN_TREE = Path("/private/tmp/candidate_pin_current/vllm")
PIN_SCHEDULER = (
    PIN_TREE
    / "distributed"
    / "kv_transfer"
    / "kv_connector"
    / "v1"
    / "offloading"
    / "scheduler.py"
)
PIN_GPU_WORKER = PIN_TREE / "v1" / "kv_offload" / "cpu" / "gpu_worker.py"


def _pn383():
    from sndr.engines.vllm.patches.offload import (
        pn383_offload_mtp_eagle_gate as M,
    )
    return M


# ─────────────────────────────────────────────────────────────────────
# Portable fixtures — pristine-shaped (pin g303916e93)
# ─────────────────────────────────────────────────────────────────────


def _fake_pristine_scheduler() -> str:
    return (
        "# fake offloading scheduler.py - pristine-shaped regions"
        " (pin g303916e93)\n"
        "from __future__ import annotations\n"
        "\n"
        "from typing import NamedTuple\n"
        "\n"
        "\n"
        + GROUP_CONFIG_REGION
        + "\n"
        "\n"
        "class SchedulerOffloadConfig(NamedTuple):\n"
        "    kv_group_configs: tuple\n"
        "    block_size_factor: int\n"
        "    num_workers: int\n"
        "    offload_prompt_only: bool\n"
        "\n"
        + FROM_SPEC_REGION
        + "\n"
        "\n"
        "class OffloadingConnectorScheduler:\n"
        + LOOKUP_REGION
        + "\n"
        + BUILD_STORE_REGION
    )


def _fake_pristine_worker() -> str:
    return (
        "# fake gpu_worker.py - pristine-shaped region (pin g303916e93)\n"
        "from __future__ import annotations\n"
        "\n"
        "\n"
        "class SingleDirectionOffloadingHandler:\n"
        "    def transfer_async(self, job_id, transfer_spec):\n"
        "        src_blocks, dst_blocks = transfer_spec\n"
        "        num_src_blocks = len(src_blocks)\n"
        "        num_dst_blocks = len(dst_blocks)\n"
        "        group_sizes = []\n"
        "        block_indices = []\n"
        "        all_src = all_dst = all_sizes = None\n"
        "        num_copy_ops = 0\n"
        "        op_idx = 0\n"
        + WORKER_REGION
    )


class _FakeLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []

    def info(self, msg, *a, **k):
        self.infos.append(msg % a if a else msg)

    def warning(self, msg, *a, **k):
        self.warnings.append(msg % a if a else msg)

    def debug(self, *a, **k):
        pass


def _apply_scheduler(tmp_path, monkeypatch):
    monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", "1")
    from sndr.kernel import TextPatchResult

    M = _pn383()
    target = tmp_path / "scheduler.py"
    target.write_text(_fake_pristine_scheduler(), encoding="utf-8")
    monkeypatch.setattr(
        M, "resolve_vllm_file", lambda rel: str(target) if "scheduler" in rel else None
    )
    patcher = M._make_scheduler_patcher()
    assert patcher is not None
    result, failure = patcher.apply()
    assert result == TextPatchResult.APPLIED, failure
    out = target.read_text(encoding="utf-8")
    ast.parse(out)
    return patcher, out


def _exec_scheduler(patched_text):
    fake_logger = _FakeLogger()
    namespace = {
        "get_sliding_window_size_in_blocks": lambda kv_spec, size: None,
        "logger": fake_logger,
        "cdiv": lambda a, b: -(-a // b),
    }
    exec(compile(patched_text, "<pn383-scheduler-fixture>", "exec"), namespace)
    return namespace, fake_logger


def _fake_spec(layer_names_per_group, *, use_eagle=True, eagle_flags=None):
    n = len(layer_names_per_group)
    flags = eagle_flags or [False] * n
    groups = [
        types.SimpleNamespace(
            layer_names=names, kv_cache_spec=None, is_eagle_group=flag
        )
        for names, flag in zip(layer_names_per_group, flags)
    ]
    speculative_config = (
        types.SimpleNamespace(use_eagle=lambda: True) if use_eagle else None
    )
    return types.SimpleNamespace(
        gpu_block_size=[16] * n,
        block_size_factor=1,
        hash_block_size=16,
        kv_cache_config=types.SimpleNamespace(kv_cache_groups=groups),
        vllm_config=types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(world_size=2),
            speculative_config=speculative_config,
        ),
        offload_prompt_only=False,
    )


# ─────────────────────────────────────────────────────────────────────
# 1. Sub-patch inventory and anchors
# ─────────────────────────────────────────────────────────────────────


class TestAnchors:
    def test_scheduler_sub_patch_inventory(self):
        """The four upstream hunks map to ten text anchors: hunk 1 =
        config field, hunk 2 = detection + construction flag, hunk 3 =
        six _lookup edits, hunk 4 = store tail exclusion."""
        M = _pn383()
        subs = M.build_scheduler_sub_patches()
        assert len(subs) == 10
        assert all(sp.required for sp in subs)
        names = {sp.name for sp in subs}
        assert names == {
            "pn383_group_config_eagle_field",
            "pn383_from_spec_eagle_detection",
            "pn383_from_spec_group_flag",
            "pn383_lookup_eagle_verified_set",
            "pn383_lookup_eagle_unverified",
            "pn383_lookup_query_max",
            "pn383_lookup_required_window",
            "pn383_lookup_eagle_pop",
            "pn383_lookup_eagle_clear",
            "pn383_store_tail_exclusion",
        }

    def test_worker_sub_patch_inventory(self):
        M = _pn383()
        subs = M.build_worker_sub_patches()
        assert len(subs) == 1
        assert subs[0].required
        assert subs[0].name == "pn383_pre_dma_bounds_check"

    def test_scheduler_anchors_unique_in_fixture(self):
        M = _pn383()
        src = _fake_pristine_scheduler()
        for sp in M.build_scheduler_sub_patches():
            assert src.count(sp.anchor) == 1, sp.name

    def test_worker_anchor_unique_in_fixture(self):
        M = _pn383()
        src = _fake_pristine_worker()
        for sp in M.build_worker_sub_patches():
            assert src.count(sp.anchor) == 1, sp.name

    def test_replacements_do_not_resurrect_anchors(self):
        M = _pn383()
        subs = M.build_scheduler_sub_patches() + M.build_worker_sub_patches()
        for sp in subs:
            for other in subs:
                assert other.anchor not in sp.replacement, (
                    sp.name,
                    other.name,
                )


# ─────────────────────────────────────────────────────────────────────
# 2. End-to-end apply on the fixtures
# ─────────────────────────────────────────────────────────────────────


class TestEndToEndApply:
    def test_scheduler_applies_all_subs(self, tmp_path, monkeypatch):
        patcher, out = _apply_scheduler(tmp_path, monkeypatch)
        assert len(patcher.applied_sub_patches) == 10
        assert "is_eagle_group: bool = False" in out
        assert "_pn383_eagle_verified" in out

    def test_worker_applies(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", "1")
        from sndr.kernel import TextPatchResult

        M = _pn383()
        target = tmp_path / "gpu_worker.py"
        target.write_text(_fake_pristine_worker(), encoding="utf-8")
        monkeypatch.setattr(
            M,
            "resolve_vllm_file",
            lambda rel: str(target) if "gpu_worker" in rel else None,
        )
        patcher = M._make_worker_patcher()
        assert patcher is not None
        result, failure = patcher.apply()
        assert result == TextPatchResult.APPLIED, failure
        out = target.read_text(encoding="utf-8")
        ast.parse(out)
        assert "RuntimeError" in out

    def test_scheduler_idempotent_on_second_apply(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", "1")
        from sndr.kernel import TextPatchResult

        _apply_scheduler(tmp_path, monkeypatch)
        M = _pn383()
        result2, _ = M._make_scheduler_patcher().apply()
        assert result2 == TextPatchResult.IDEMPOTENT


# ─────────────────────────────────────────────────────────────────────
# 3. Behavioral contract — patched from_spec exec'd with fake specs
# ─────────────────────────────────────────────────────────────────────


class TestFromSpecBehavior:
    def _from_spec(self, tmp_path, monkeypatch, spec):
        _, out = _apply_scheduler(tmp_path, monkeypatch)
        namespace, fake_logger = _exec_scheduler(out)
        config = namespace["SchedulerOffloadConfig"].from_spec(spec)
        return config, fake_logger

    def test_qwen36_mtp_flags_only_drafter_group(self, tmp_path, monkeypatch):
        """THE Genesis extension: on a Qwen3.6-shaped hybrid (target
        full-attn + GDN groups, drafter under the 'mtp' prefix) only
        the drafter group is flagged — NOT all groups (upstream's
        fallback would flag everything: hit-rate loss)."""
        spec = _fake_spec(
            [
                ["model.layers.0.self_attn.attn", "model.layers.3.self_attn.attn"],
                ["model.layers.1.linear_attn", "model.layers.2.linear_attn"],
                ["mtp.layers.48.self_attn.attn"],
            ]
        )
        config, _ = self._from_spec(tmp_path, monkeypatch, spec)
        assert [g.is_eagle_group for g in config.kv_group_configs] == [
            False,
            False,
            True,
        ]

    def test_pin_annotated_groups_pass_through(self, tmp_path, monkeypatch):
        """Groups already annotated by the engine (the DeepSeek V4 path
        on this pin) are respected without the name heuristic."""
        spec = _fake_spec(
            [["model.layers.0.attn"], ["model.layers.61.attn"]],
            eagle_flags=[False, True],
        )
        config, _ = self._from_spec(tmp_path, monkeypatch, spec)
        assert [g.is_eagle_group for g in config.kv_group_configs] == [False, True]

    def test_all_groups_fallback_warns(self, tmp_path, monkeypatch):
        """No annotation + no drafter-convention names + use_eagle ->
        upstream's conservative all-groups fallback, but LOUD."""
        spec = _fake_spec(
            [["model.layers.0.attn"], ["model.layers.1.attn"]],
        )
        config, fake_logger = self._from_spec(tmp_path, monkeypatch, spec)
        assert [g.is_eagle_group for g in config.kv_group_configs] == [True, True]
        assert fake_logger.warnings

    def test_no_spec_decode_flags_nothing(self, tmp_path, monkeypatch):
        spec = _fake_spec(
            [["model.layers.0.attn"], ["mtp.layers.48.self_attn.attn"]],
            use_eagle=False,
        )
        config, fake_logger = self._from_spec(tmp_path, monkeypatch, spec)
        assert [g.is_eagle_group for g in config.kv_group_configs] == [False, False]
        assert not fake_logger.warnings


# ─────────────────────────────────────────────────────────────────────
# 4. Replacement contract — faithful #44784 semantics + extras
# ─────────────────────────────────────────────────────────────────────


class TestReplacementContract:
    def _subs(self):
        M = _pn383()
        return {sp.name: sp for sp in (
            M.build_scheduler_sub_patches() + M.build_worker_sub_patches()
        )}

    def test_lookup_query_max_inflates_only_sliding_window(self):
        repl = self._subs()["pn383_lookup_query_max"].replacement
        assert "_pn383_query_max" in repl
        assert "max_hit_size_tokens + offloaded_block_size" in repl
        assert "sliding_window_size_in_blocks is not None" in repl

    def test_lookup_pop_decrements_and_marks_verified(self):
        repl = self._subs()["pn383_lookup_eagle_pop"].replacement
        assert "num_hit_blocks -= 1" in repl
        assert "_pn383_eagle_verified.add(group_idx)" in repl

    def test_lookup_clear_gated_on_non_eagle_groups(self):
        repl = self._subs()["pn383_lookup_eagle_clear"].replacement
        assert "if not group_config.is_eagle_group:" in repl
        assert "_pn383_eagle_verified.clear()" in repl

    def test_lookup_required_window_plus_one(self):
        repl = self._subs()["pn383_lookup_required_window"].replacement
        assert "_pn383_required_window = sliding_window_size_in_blocks" in repl
        assert "_pn383_required_window += 1" in repl

    def test_store_tail_exclusion_clamped_at_zero(self):
        repl = self._subs()["pn383_store_tail_exclusion"].replacement
        assert "if group_config.is_eagle_group:" in repl
        assert "num_blocks = max(0, num_blocks - 1)" in repl

    def test_worker_bounds_check_raises_before_dma(self):
        repl = self._subs()["pn383_pre_dma_bounds_check"].replacement
        assert "raise RuntimeError(" in repl
        assert ".shape[0]" in repl
        # The check must come BEFORE the descriptor-pointer loop.
        assert repl.index("raise RuntimeError(") < repl.index(
            "for data_ref in group_data_refs:"
        )

    def test_detection_narrows_before_all_groups_fallback(self):
        """The mtp-name heuristic must run BEFORE the all-groups
        fallback (otherwise the extension is dead code)."""
        repl = self._subs()["pn383_from_spec_eagle_detection"].replacement
        assert repl.index('"mtp"') < repl.index("set(\n                range(")
        assert "use_eagle()" in repl


# ─────────────────────────────────────────────────────────────────────
# 5. Self-collision invariants (tools/lint_drift_markers.py contract)
# ─────────────────────────────────────────────────────────────────────


class TestSelfCollision:
    def test_drift_markers_disjoint_from_emitted_text(self):
        M = _pn383()
        marker_line = f"# [Genesis wiring marker: {M.GENESIS_PN383_MARKER}]\n"
        replacements = [
            sp.replacement
            for sp in (
                M.build_scheduler_sub_patches() + M.build_worker_sub_patches()
            )
        ]
        for dm in tuple(M._SCHEDULER_DRIFT_MARKERS) + tuple(
            M._WORKER_DRIFT_MARKERS
        ):
            if dm.startswith("[Genesis"):
                continue  # defended convention — exempt from the lint
            for repl in replacements:
                assert dm not in repl, (dm, repl[:80])
            assert dm not in marker_line

    def test_drift_markers_absent_from_pristine_fixtures(self):
        M = _pn383()
        sched_src = _fake_pristine_scheduler()
        worker_src = _fake_pristine_worker()
        for dm in M._SCHEDULER_DRIFT_MARKERS:
            assert dm not in sched_src
        for dm in M._WORKER_DRIFT_MARKERS:
            assert dm not in worker_src


# ─────────────────────────────────────────────────────────────────────
# 6. Module apply() contract — env gate
# ─────────────────────────────────────────────────────────────────────


class TestModuleApply:
    def test_skips_when_env_unset(self, monkeypatch):
        M = _pn383()
        monkeypatch.delenv(
            "GENESIS_ENABLE_PN383_OFFLOAD_MTP_EAGLE_GATE", raising=False
        )
        status, detail = M.apply()
        assert status == "skipped"
        assert "GENESIS_ENABLE_PN383_OFFLOAD_MTP_EAGLE_GATE" in detail

    def test_applies_both_files_when_enabled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GENESIS_ENABLE_PN383_OFFLOAD_MTP_EAGLE_GATE", "1")
        monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", "1")
        M = _pn383()
        sched = tmp_path / "scheduler.py"
        sched.write_text(_fake_pristine_scheduler(), encoding="utf-8")
        worker = tmp_path / "gpu_worker.py"
        worker.write_text(_fake_pristine_worker(), encoding="utf-8")
        monkeypatch.setattr(
            M, "resolve_vllm_file", lambda rel: str(tmp_path / Path(rel).name)
        )
        monkeypatch.setattr(M, "vllm_install_root", lambda: str(tmp_path))
        status, detail = M.apply()
        assert status == "applied", detail
        assert "44784" in detail
        ast.parse(sched.read_text(encoding="utf-8"))
        ast.parse(worker.read_text(encoding="utf-8"))
        assert M.is_applied()

    def test_skips_when_targets_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GENESIS_ENABLE_PN383_OFFLOAD_MTP_EAGLE_GATE", "1")
        M = _pn383()
        monkeypatch.setattr(M, "resolve_vllm_file", lambda rel: None)
        monkeypatch.setattr(M, "vllm_install_root", lambda: str(tmp_path))
        status, _ = M.apply()
        assert status == "skipped"

    def test_marker_tracks_upstream_pr(self):
        M = _pn383()
        assert "44784" in M.GENESIS_PN383_MARKER


# ─────────────────────────────────────────────────────────────────────
# 7. Against the real pristine pin (opportunistic)
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not (PIN_SCHEDULER.is_file() and PIN_GPU_WORKER.is_file()),
    reason="pristine pin tree not present on this machine",
)
class TestAgainstPristinePin:
    def test_fixture_regions_match_pin(self):
        sched_src = PIN_SCHEDULER.read_text(encoding="utf-8")
        for region in (
            GROUP_CONFIG_REGION,
            FROM_SPEC_REGION,
            LOOKUP_REGION,
            BUILD_STORE_REGION,
        ):
            assert sched_src.count(region) == 1
        worker_src = PIN_GPU_WORKER.read_text(encoding="utf-8")
        assert worker_src.count(WORKER_REGION) == 1

    def test_anchors_unique_and_markers_absent(self):
        M = _pn383()
        sched_src = PIN_SCHEDULER.read_text(encoding="utf-8")
        for sp in M.build_scheduler_sub_patches():
            assert sched_src.count(sp.anchor) == 1, sp.name
            assert sp.replacement not in sched_src, sp.name
        for dm in M._SCHEDULER_DRIFT_MARKERS:
            assert dm not in sched_src
        worker_src = PIN_GPU_WORKER.read_text(encoding="utf-8")
        for sp in M.build_worker_sub_patches():
            assert worker_src.count(sp.anchor) == 1, sp.name
            assert sp.replacement not in worker_src, sp.name
        for dm in M._WORKER_DRIFT_MARKERS:
            assert dm not in worker_src

    def test_full_file_apply_and_compile(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", "1")
        from sndr.kernel import TextPatchResult

        M = _pn383()
        for pin_file, make in (
            (PIN_SCHEDULER, "_make_scheduler_patcher"),
            (PIN_GPU_WORKER, "_make_worker_patcher"),
        ):
            target = tmp_path / pin_file.name
            target.write_text(pin_file.read_text(encoding="utf-8"), encoding="utf-8")
            monkeypatch.setattr(M, "resolve_vllm_file", lambda rel, t=target: str(t))
            patcher = getattr(M, make)()
            result, failure = patcher.apply()
            assert result == TextPatchResult.APPLIED, (pin_file.name, failure)
            ast.parse(target.read_text(encoding="utf-8"))
