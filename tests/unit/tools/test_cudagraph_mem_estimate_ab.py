# SPDX-License-Identifier: Apache-2.0
"""Tests for ``tools/cudagraph_mem_estimate_ab.py`` (vllm#45197 harness).

TDD contract (written BEFORE the implementation), per the 2026-06-11
roadmap row for #45197: "Do NOT flip configs blindly (upstream
CHANGES_REQUESTED); MEASURE =1 vs =0 on 35B/27B first". The tool boots
NOTHING itself — it

  (a) emits the TWO launcher env permutations
      (VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0/1), and
  (b) in diff-report mode parses two boot logs for the estimate lines
      + free-VRAM lines and computes the recovered-KV / overestimate
      deltas (the rig stage runs the boots and feeds the logs back).

The synthetic logs below are built from the EXACT logger format strings
of pin 0.22.1rc1.dev259+g303916e93 (byte-verified 2026-06-11):
  - vllm/v1/worker/gpu_model_runner.py: "Profiling CUDA graph memory",
    "Estimated %s CUDA graph memory", "Estimated CUDA graph memory:
    %.2f GiB total", "Graph capturing finished in %.0f secs, took
    %.2f GiB"
  - vllm/v1/worker/gpu_worker.py: "Available KV cache memory",
    "Initial free memory", "Free memory after profiling", the
    enabled-INFO / disabled-WARNING advisory lines
  - vllm/v1/core/kv_cache_utils.py: "GPU KV cache size: %s tokens",
    "Maximum concurrency for %s tokens per request: %.2fx"
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "cudagraph_mem_estimate_ab.py"

FLAG = "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"


def _import_tool():
    spec = importlib.util.spec_from_file_location(
        "cudagraph_mem_estimate_ab", TOOL_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cudagraph_mem_estimate_ab"] = mod
    assert spec.loader is not None  # nosec
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tool():
    return _import_tool()


# ── Synthetic boot logs (pin format strings, TP=2 rank prefixes) ──────

_PREFIX = "(Worker_TP0 pid=412) "

LOG_ON = "\n".join([
    "INFO 06-11 10:01:02 [gpu_model_runner.py:6412] " + _PREFIX
    + "Profiling CUDA graph memory: FULL=18 (largest=512), "
      "PIECEWISE=3 (largest=2048)",
    "DEBUG 06-11 10:01:09 [gpu_model_runner.py:6467] " + _PREFIX
    + "Estimated FULL CUDA graph memory: 412.00 MiB first-capture + "
      "(18-1) × 24.50 MiB per-graph",
    "DEBUG 06-11 10:01:11 [gpu_model_runner.py:6467] " + _PREFIX
    + "Estimated PIECEWISE CUDA graph memory: 96.00 MiB first-capture + "
      "(3-1) × 12.25 MiB per-graph",
    "INFO 06-11 10:01:12 [gpu_model_runner.py:6517] " + _PREFIX
    + "Estimated CUDA graph memory: 0.83 GiB total",
    "DEBUG 06-11 10:01:13 [gpu_worker.py:469] " + _PREFIX
    + "Initial free memory: 23.45 GiB; Requested memory: 0.920000 "
      "(util), 21.57 GiB",
    "DEBUG 06-11 10:01:13 [gpu_worker.py:475] " + _PREFIX
    + "Free memory after profiling: 4.12 GiB (total), 2.24 GiB "
      "(within requested)",
    "INFO 06-11 10:01:13 [gpu_worker.py:481] " + _PREFIX
    + "Available KV cache memory: 7.41 GiB",
    # second rank repeats the info_once line with an identical value
    "INFO 06-11 10:01:13 [gpu_worker.py:481] (Worker_TP1 pid=413) "
    "Available KV cache memory: 7.41 GiB",
    "INFO 06-11 10:01:14 [gpu_worker.py:497] " + _PREFIX
    + "CUDA graph memory profiling is enabled (default since v0.21.0). "
      "The current --gpu-memory-utilization=0.9200 is equivalent to "
      "--gpu-memory-utilization=0.8854 without CUDA graph memory "
      "profiling. To maintain the same effective KV cache size as "
      "before, increase --gpu-memory-utilization to 0.9546. To "
      "disable, set VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0.",
    "INFO 06-11 10:01:15 [kv_cache_utils.py:1744] "
    "GPU KV cache size: 242,176 tokens",
    "INFO 06-11 10:01:15 [kv_cache_utils.py:1746] "
    "Maximum concurrency for 131,072 tokens per request: 1.85x",
    "INFO 06-11 10:02:25 [gpu_model_runner.py:6586] " + _PREFIX
    + "Graph capturing finished in 18 secs, took 0.52 GiB",
    "",
])

LOG_OFF = "\n".join([
    "INFO 06-11 10:11:02 [gpu_model_runner.py:6412] " + _PREFIX
    + "Profiling CUDA graph memory: FULL=18 (largest=512), "
      "PIECEWISE=3 (largest=2048)",
    "INFO 06-11 10:11:12 [gpu_model_runner.py:6517] " + _PREFIX
    + "Estimated CUDA graph memory: 0.83 GiB total",
    "DEBUG 06-11 10:11:13 [gpu_worker.py:469] " + _PREFIX
    + "Initial free memory: 23.45 GiB; Requested memory: 0.920000 "
      "(util), 21.57 GiB",
    "DEBUG 06-11 10:11:13 [gpu_worker.py:475] " + _PREFIX
    + "Free memory after profiling: 4.12 GiB (total), 2.24 GiB "
      "(within requested)",
    "INFO 06-11 10:11:13 [gpu_worker.py:481] " + _PREFIX
    + "Available KV cache memory: 8.24 GiB",
    "WARNING 06-11 10:11:14 [gpu_worker.py:509] " + _PREFIX
    + "CUDA graph memory profiling is disabled "
      "(VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0). Without it, CUDA "
      "graph memory is not accounted for during KV cache allocation, "
      "which may require lowering --gpu-memory-utilization to avoid "
      "OOM. Consider re-enabling it (the default as of v0.21.0) and "
      "increasing --gpu-memory-utilization from 0.9200 to 0.9546.",
    "INFO 06-11 10:11:15 [kv_cache_utils.py:1744] "
    "GPU KV cache size: 269,440 tokens",
    "INFO 06-11 10:11:15 [kv_cache_utils.py:1746] "
    "Maximum concurrency for 131,072 tokens per request: 2.06x",
    "INFO 06-11 10:12:25 [gpu_model_runner.py:6586] " + _PREFIX
    + "Graph capturing finished in 19 secs, took 0.51 GiB",
    "",
])


# ── emit-env: the two launcher permutations ───────────────────────────


class TestEmitEnv:
    def test_two_arms_with_opposite_flag_values(self, tool):
        arms = tool.launcher_env_arms()
        assert len(arms) == 2
        by_name = {arm["name"]: arm for arm in arms}
        assert by_name["estimate-on"]["env"][FLAG] == "1"
        assert by_name["estimate-off"]["env"][FLAG] == "0"

    def test_both_arms_force_debug_logging(self, tool):
        """The free-VRAM lines and the per-mode estimate breakdown are
        logger.debug — PROD launchers run VLLM_LOGGING_LEVEL=WARNING
        which would suppress EVERY line the diff-report parses."""
        for arm in tool.launcher_env_arms():
            assert arm["env"]["VLLM_LOGGING_LEVEL"] == "DEBUG"

    def test_env_format(self, tool):
        out = tool.render_env_arms("env")
        assert f"{FLAG}=1" in out
        assert f"{FLAG}=0" in out

    def test_docker_format(self, tool):
        out = tool.render_env_arms("docker")
        assert f"-e {FLAG}=1 \\" in out
        assert f"-e {FLAG}=0 \\" in out

    def test_shell_format(self, tool):
        out = tool.render_env_arms("shell")
        assert f"export {FLAG}=1" in out
        assert f"export {FLAG}=0" in out

    def test_out_dir_writes_one_file_per_arm(self, tool, tmp_path):
        rc = tool.main([
            "emit-env", "--format", "env", "--out-dir", str(tmp_path),
        ])
        assert rc == 0
        on = tmp_path / "cudagraph_ab_estimate-on.env"
        off = tmp_path / "cudagraph_ab_estimate-off.env"
        assert f"{FLAG}=1" in on.read_text()
        assert f"{FLAG}=0" in off.read_text()


# ── boot-log parsing ──────────────────────────────────────────────────


class TestParseBootLog:
    def test_estimate_total(self, tool):
        parsed = tool.parse_boot_log(LOG_ON)
        assert parsed["estimate_total_gib"] == pytest.approx(0.83)

    def test_per_mode_estimates(self, tool):
        parsed = tool.parse_boot_log(LOG_ON)
        modes = {m["mode"]: m for m in parsed["per_mode"]}
        assert modes["FULL"]["first_capture_mib"] == pytest.approx(412.0)
        assert modes["FULL"]["num_graphs"] == 18
        assert modes["FULL"]["per_graph_mib"] == pytest.approx(24.5)
        assert modes["PIECEWISE"]["per_graph_mib"] == pytest.approx(12.25)

    def test_available_kv_dedup_across_ranks(self, tool):
        parsed = tool.parse_boot_log(LOG_ON)
        assert parsed["available_kv_gib"] == pytest.approx(7.41)

    def test_kv_tokens_thousands_separators(self, tool):
        parsed = tool.parse_boot_log(LOG_ON)
        assert parsed["kv_tokens"] == 242176

    def test_max_concurrency(self, tool):
        parsed = tool.parse_boot_log(LOG_OFF)
        assert parsed["max_concurrency"] == pytest.approx(2.06)

    def test_actual_capture_gib(self, tool):
        parsed = tool.parse_boot_log(LOG_ON)
        assert parsed["actual_capture_gib"] == pytest.approx(0.52)

    def test_free_vram_lines(self, tool):
        parsed = tool.parse_boot_log(LOG_ON)
        assert parsed["initial_free_gib"] == pytest.approx(23.45)
        assert parsed["requested_gib"] == pytest.approx(21.57)
        assert parsed["free_after_profiling_gib"] == pytest.approx(4.12)

    def test_flag_state_detection(self, tool):
        assert tool.parse_boot_log(LOG_ON)["flag_state"] == "on"
        assert tool.parse_boot_log(LOG_OFF)["flag_state"] == "off"
        assert tool.parse_boot_log("nothing here")["flag_state"] is None

    def test_suggested_util(self, tool):
        assert tool.parse_boot_log(LOG_ON)["suggested_util"] == (
            pytest.approx(0.9546)
        )
        assert tool.parse_boot_log(LOG_OFF)["suggested_util"] == (
            pytest.approx(0.9546)
        )


# ── diff report ───────────────────────────────────────────────────────


class TestBuildReport:
    def _report(self, tool, threshold_mib=200.0):
        return tool.build_report(
            tool.parse_boot_log(LOG_ON),
            tool.parse_boot_log(LOG_OFF),
            threshold_mib=threshold_mib,
        )

    def test_recovered_kv(self, tool):
        report = self._report(tool)
        assert report["kv_gib_recovered"] == pytest.approx(0.83, abs=1e-6)
        assert report["kv_tokens_recovered"] == 269440 - 242176

    def test_overestimate_vs_actual_capture(self, tool):
        """THE #45197 number: estimate minus measured capture cost."""
        report = self._report(tool)
        assert report["overestimate_gib"] == pytest.approx(0.83 - 0.52)

    def test_verdict_confirmed_above_threshold(self, tool):
        # 0.31 GiB ≈ 317 MiB > 200 MiB → root-cause patch candidate
        report = self._report(tool, threshold_mib=200.0)
        assert report["verdict"] == "OVERESTIMATE_CONFIRMED"

    def test_verdict_within_tolerance_below_threshold(self, tool):
        report = self._report(tool, threshold_mib=400.0)
        assert report["verdict"] == "WITHIN_TOLERANCE"

    def test_verdict_insufficient_data(self, tool):
        report = tool.build_report(
            tool.parse_boot_log("empty"),
            tool.parse_boot_log(LOG_OFF),
            threshold_mib=200.0,
        )
        assert report["verdict"] == "INSUFFICIENT_DATA"


# ── CLI ───────────────────────────────────────────────────────────────


class TestCli:
    def test_diff_report_json(self, tool, tmp_path, capsys):
        on = tmp_path / "on.log"
        off = tmp_path / "off.log"
        on.write_text(LOG_ON)
        off.write_text(LOG_OFF)
        rc = tool.main([
            "diff-report", "--log-on", str(on), "--log-off", str(off),
            "--json",
        ])
        assert rc == 0
        report = json.loads(capsys.readouterr().out)
        assert report["verdict"] == "OVERESTIMATE_CONFIRMED"
        assert report["arms"]["on"]["available_kv_gib"] == (
            pytest.approx(7.41)
        )

    def test_swapped_logs_rejected(self, tool, tmp_path):
        """Operator passed the =0 boot log as --log-on: fail loudly
        instead of producing a sign-flipped report."""
        on = tmp_path / "on.log"
        off = tmp_path / "off.log"
        on.write_text(LOG_OFF)
        off.write_text(LOG_ON)
        rc = tool.main([
            "diff-report", "--log-on", str(on), "--log-off", str(off),
        ])
        assert rc == 2

    def test_missing_log_file_is_usage_error(self, tool, tmp_path):
        rc = tool.main([
            "diff-report",
            "--log-on", str(tmp_path / "absent.log"),
            "--log-off", str(tmp_path / "absent2.log"),
        ])
        assert rc == 2


# ── presets.py vs a5000 YAML inconsistency (documentation contract) ───


class TestInconsistencyFindingDocumented:
    """The roadmap mandates documenting the presets.py(=0) vs a5000
    YAML(=1) inconsistency in the tool docstring and reconciling the
    hardware YAML comment (measure-first — values must NOT flip)."""

    def test_docstring_names_both_sources(self, tool):
        doc = tool.__doc__ or ""
        assert "sndr/compat/presets.py" in doc
        assert "a5000-2x-24gbvram-16cpu-128gbram.yaml" in doc
        assert "45197" in doc

    def test_presets_value_not_flipped(self):
        src = (REPO_ROOT / "sndr" / "compat" / "presets.py").read_text()
        assert f'"{FLAG}": "0"' in src

    def test_a5000_yaml_value_not_flipped_and_reconciled(self):
        yaml_path = (
            REPO_ROOT / "sndr" / "model_configs" / "builtin" / "hardware"
            / "a5000-2x-24gbvram-16cpu-128gbram.yaml"
        )
        text = yaml_path.read_text()
        assert f"{FLAG}: '1'" in text
        line = next(
            ln for ln in text.splitlines() if ln.startswith(f"  {FLAG}:")
        )
        # The reconcile comment must point at the measure-first tool
        # and at the diverging preset.
        assert "cudagraph_mem_estimate_ab" in line
        assert "presets.py" in line
        assert "45197" in line
