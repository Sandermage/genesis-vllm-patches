# SPDX-License-Identifier: Apache-2.0
"""PN367 — CUDA graph memory estimate clamp (vendor of vllm#44745).

History: PN367 originally vendored OPEN vllm#45076 (Oxygen56). On
2026-06-10 the author CLOSED #45076 and consolidated into #44745
(same clamp + the 1 MiB first-capture floor + unit tests, still OPEN).
PN367 v2 tracks #44745's merged form for drift detection and vendors
the floor for full parity.

Contract pinned here (TDD, written before the v2 implementation):
  1. Runner patcher carries TWO sub-patches: the required decoder
     mem-sample clamp and the optional first-capture 1 MiB floor.
  2. apply() on pin-form (g303916e93) targets installs clamp + floor +
     the worker final non-negative guard.
  3. Second apply() is idempotent (marker short-circuit).
  4. apply() on #44745's merged form self-skips via drift markers
     (reason: upstream_merged) — the previous v1 markers could NEVER
     match the real merged code (#44745 writes
     ``mem_samples.append(max(delta, 0))``, not the v1 marker text).
  5. Drift markers do not collide with PN367's own replacement texts
     or its Layer-6 marker line (lint_drift_markers contract).
  6. Anchors are unique and drift markers absent in the pristine pin
     tree (opportunistic — skipped when the pin tree is not present).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sndr.engines.vllm.patches.compile_safety import (
    pn367_cudagraph_mem_estimate_clamp as m,
)

# ── Fake targets ─────────────────────────────────────────────────────
# Pin-form (g303916e93): decoder profiling loop body at 24-space
# indentation, unprotected append, bare first_capture assignment.

PIN_RUNNER = (
    "# fake gpu_model_runner.py (pin g303916e93 form)\n"
    "class GPUModelRunner:\n"
    "    def profile_cudagraph_memory(self) -> int:\n"
    "        for mode, descs in groups:\n"
    "            with ctx:\n"
    "                if profile_descs:\n"
    "                    mem_samples: list[int] = []\n"
    "\n"
    "                    for i, desc in enumerate(profile_descs):\n"
    "                        mem_before = torch.cuda.mem_get_info()[0]\n"
    "                        self._warmup_and_capture(desc)\n"
    "                        torch.accelerator.synchronize()\n"
    "                        free_after = torch.cuda.mem_get_info()[0]\n"
    "                        mem_samples.append(mem_before - free_after)\n"
    "\n"
    "                    first_capture = mem_samples[0]\n"
    "                    # Use at least 1 MiB per graph for driver overhead\n"
    "                    per_graph = max(\n"
    "                        mem_samples[1] if len(mem_samples) > 1 else 0, 1 << 20\n"
    "                    )\n"
)

PIN_WORKER = (
    "# fake gpu_worker.py (pin g303916e93 form)\n"
    "class Worker:\n"
    "    def determine_available_memory(self) -> int:\n"
    "        if mode != CUDAGraphMode.NONE:\n"
    "            if profiling:\n"
    "                cudagraph_memory_estimate = self.model_runner.profile_cudagraph_memory()\n"
    "\n"
    "        return 0\n"
)

# #44745 merged form (what the files look like AFTER upstream lands the
# consolidated fix) — PN367 must self-skip on these.

MERGED_RUNNER = (
    "# fake gpu_model_runner.py (post-vllm#44745 merged form)\n"
    "class GPUModelRunner:\n"
    "    def profile_cudagraph_memory(self) -> int:\n"
    "        for mode, descs in groups:\n"
    "            with ctx:\n"
    "                if profile_descs:\n"
    "                    mem_samples: list[int] = []\n"
    "\n"
    "                    for i, desc in enumerate(profile_descs):\n"
    "                        torch.accelerator.empty_cache()\n"
    "                        mem_before = torch.cuda.mem_get_info()[0]\n"
    "                        self._warmup_and_capture(desc)\n"
    "                        torch.accelerator.synchronize()\n"
    "                        free_after = torch.cuda.mem_get_info()[0]\n"
    "                        delta = mem_before - free_after\n"
    "                        mem_samples.append(max(delta, 0))\n"
    "\n"
    "                    # Use at least 1 MiB per graph for driver overhead\n"
    "                    first_capture = max(mem_samples[0], 1 << 20)\n"
    "                    per_graph = max(\n"
    "                        mem_samples[1] if len(mem_samples) > 1 else 0, 1 << 20\n"
    "                    )\n"
)

MERGED_WORKER = (
    "# fake gpu_worker.py (post-vllm#44745 merged form)\n"
    "class Worker:\n"
    "    def determine_available_memory(self) -> int:\n"
    "        if mode != CUDAGraphMode.NONE:\n"
    "            if profiling:\n"
    "                cudagraph_memory_estimate = self.model_runner.profile_cudagraph_memory()\n"
    "                cudagraph_memory_estimate = max(cudagraph_memory_estimate, 0)\n"
    "\n"
    "        return 0\n"
)

PIN_TREE = Path("/private/tmp/candidate_pin_current/vllm/v1/worker")


# ── Helpers ──────────────────────────────────────────────────────────


def _install_fakes(tmp_path, monkeypatch, runner_text, worker_text):
    runner = tmp_path / "gpu_model_runner.py"
    worker = tmp_path / "gpu_worker.py"
    runner.write_text(runner_text, encoding="utf-8")
    worker.write_text(worker_text, encoding="utf-8")

    def _resolve(relpath: str):
        name = Path(relpath).name
        return str(tmp_path / name)

    monkeypatch.setattr(m, "resolve_vllm_file", _resolve)
    return runner, worker


# ── Patcher shape ────────────────────────────────────────────────────


class TestPatcherShape:
    def test_runner_patcher_has_clamp_and_floor_subs(self, tmp_path, monkeypatch):
        _install_fakes(tmp_path, monkeypatch, PIN_RUNNER, PIN_WORKER)
        patcher = m._make_runner_patcher()
        assert patcher is not None
        names = [sp.name for sp in patcher.sub_patches]
        assert "pn367_decoder_mem_sample_clamp" in names
        assert "pn367_first_capture_floor" in names
        by_name = {sp.name: sp for sp in patcher.sub_patches}
        assert by_name["pn367_decoder_mem_sample_clamp"].required is True
        # The floor is completeness parity with #44745 — never abort on it.
        assert by_name["pn367_first_capture_floor"].required is False

    def test_drift_markers_match_44745_merged_form(self, tmp_path, monkeypatch):
        """v1 regression: the old marker text never occurred in either
        #45076's or #44745's actual code, so the patch could not
        self-skip post-merge. v2 markers must be substrings of the
        merged-form fixtures above."""
        _install_fakes(tmp_path, monkeypatch, PIN_RUNNER, PIN_WORKER)
        runner = m._make_runner_patcher()
        worker = m._make_worker_patcher()
        assert any(dm in MERGED_RUNNER for dm in runner.upstream_drift_markers)
        assert any(dm in MERGED_WORKER for dm in worker.upstream_drift_markers)

    def test_module_tracks_consolidated_pr_44745(self):
        assert "44745" in (m.__doc__ or "")
        assert "44745" in m.GENESIS_PN367_MARKER
        # Content changed vs v1 (floor sub added) — marker must be bumped.
        assert "v2" in m.GENESIS_PN367_MARKER

    def test_patcher_none_when_target_missing(self, monkeypatch):
        monkeypatch.setattr(m, "resolve_vllm_file", lambda rel: None)
        assert m._make_runner_patcher() is None
        assert m._make_worker_patcher() is None


# ── Apply semantics ──────────────────────────────────────────────────


class TestApply:
    def test_apply_pin_form_installs_clamp_floor_and_guard(
        self, tmp_path, monkeypatch
    ):
        runner, worker = _install_fakes(
            tmp_path, monkeypatch, PIN_RUNNER, PIN_WORKER
        )
        status, reason = m.apply()
        assert status == "applied", reason

        runner_out = runner.read_text(encoding="utf-8")
        worker_out = worker.read_text(encoding="utf-8")
        # Clamp installed, raw unprotected append gone.
        assert "max(_g_pn367_delta, 0)" in runner_out
        assert "mem_samples.append(mem_before - free_after)" not in runner_out
        # Floor installed (parity with #44745).
        assert "first_capture = max(mem_samples[0], 1 << 20)" in runner_out
        # Worker final guard installed.
        assert (
            "cudagraph_memory_estimate = max(cudagraph_memory_estimate, 0)"
            in worker_out
        )
        # Both files compile after the splice.
        compile(runner_out, str(runner), "exec")
        compile(worker_out, str(worker), "exec")

    def test_second_apply_is_idempotent(self, tmp_path, monkeypatch):
        _install_fakes(tmp_path, monkeypatch, PIN_RUNNER, PIN_WORKER)
        first_status, _ = m.apply()
        assert first_status == "applied"
        second_status, second_reason = m.apply()
        assert second_status == "skipped"
        assert "already applied" in second_reason

    def test_self_skips_on_44745_merged_form(self, tmp_path, monkeypatch):
        runner, worker = _install_fakes(
            tmp_path, monkeypatch, MERGED_RUNNER, MERGED_WORKER
        )
        status, reason = m.apply()
        assert status == "skipped"
        assert "upstream_merged" in reason
        # Self-skip must not modify the merged files.
        assert runner.read_text(encoding="utf-8") == MERGED_RUNNER
        assert worker.read_text(encoding="utf-8") == MERGED_WORKER


# ── Lint contract (tools/lint_drift_markers.py) ──────────────────────


class TestDriftMarkerSelfCollision:
    def test_markers_not_substring_of_own_replacements(
        self, tmp_path, monkeypatch
    ):
        _install_fakes(tmp_path, monkeypatch, PIN_RUNNER, PIN_WORKER)
        for patcher in (m._make_runner_patcher(), m._make_worker_patcher()):
            marker_line = f"# [Genesis wiring marker: {patcher.marker}]\n"
            for dm in patcher.upstream_drift_markers:
                for sp in patcher.sub_patches:
                    assert dm not in sp.replacement, (
                        f"drift marker {dm!r} collides with "
                        f"{sp.name} replacement — would false-fire Layer 3"
                    )
                assert dm not in marker_line


# ── Pristine pin invariants (opportunistic) ──────────────────────────


@pytest.mark.skipif(
    not (PIN_TREE / "gpu_model_runner.py").is_file(),
    reason="pristine pin tree not present on this machine",
)
class TestAnchorsAgainstPristinePin:
    def test_runner_anchors_unique_and_markers_absent(self):
        src = (PIN_TREE / "gpu_model_runner.py").read_text(encoding="utf-8")
        patcher_subs = [
            (m.PN367_RUNNER_OLD, m.PN367_RUNNER_NEW),
            (m.PN367_FLOOR_OLD, m.PN367_FLOOR_NEW),
        ]
        for old, new in patcher_subs:
            assert src.count(old) == 1
            assert new not in src
        for dm in m._RUNNER_DRIFT_MARKERS:
            assert dm not in src

    def test_worker_anchor_unique_and_markers_absent(self):
        src = (PIN_TREE / "gpu_worker.py").read_text(encoding="utf-8")
        assert src.count(m.PN367_WORKER_OLD) == 1
        assert m.PN367_WORKER_NEW not in src
        for dm in m._WORKER_DRIFT_MARKERS:
            assert dm not in src
