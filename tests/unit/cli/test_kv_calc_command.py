# SPDX-License-Identifier: Apache-2.0
"""Integration tests for `sndr kv-calc <preset>` / `sndr fit` (v12 CLI command).

Drives the real command end-to-end against the live preset corpus using offline
rig sources (--fake-gpus / --card) so there's no nvidia-smi dependency. The
load-bearing assertion is that the 35B at its true A5000 operating point lands
on the TIGHT verdict the dev424 PN403 live engine telemetry shows.
"""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

import pytest

pytest.importorskip("pydantic")

from sndr.cli.main import main  # noqa: E402

_A5000_2X = "RTX A5000:24564:8.6;RTX A5000:24564:8.6"


def _run(argv) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(argv)
    return rc, buf.getvalue()


class TestKvCalcRegistered:
    def test_kv_calc_and_fit_in_registry(self):
        from sndr.cli.commands import COMMAND_REGISTRY
        from sndr.cli.main import build_parser
        build_parser()
        assert "kv-calc" in COMMAND_REGISTRY
        assert "fit" in COMMAND_REGISTRY


class TestKvCalcVerdicts:
    def test_35b_on_true_a5000_is_tight(self):
        """35B @280K on real 24564 MiB A5000s reproduces the PN403 TIGHT point."""
        rc, out = _run(["kv-calc", "prod-qwen3.6-35b-balanced",
                        "--fake-gpus", _A5000_2X, "--kv-breakdown"])
        assert rc == 0  # TIGHT still boots → exit 0
        assert "TIGHT" in out
        assert "Model weights" in out
        assert "Available for KV" in out

    def test_35b_on_tiny_budget_fails(self):
        rc, out = _run(["kv-calc", "prod-qwen3.6-35b-balanced",
                        "--card", "12"])
        assert rc == 1  # FAIL → exit 1
        assert "FAIL" in out

    def test_ctx_override_changes_point(self):
        rc, out = _run(["kv-calc", "prod-qwen3.6-35b-balanced",
                        "--fake-gpus", _A5000_2X, "--ctx", "32k"])
        assert rc == 0
        assert "ctx=32,768" in out

    def test_27b_flagged_provisional(self):
        rc, out = _run(["kv-calc", "prod-qwen3.6-27b-tq-k8v4",
                        "--fake-gpus", _A5000_2X])
        assert "PROVISIONAL" in out


class TestKvCalcSolve:
    def test_solve_max_ctx_returns_sane_number(self):
        rc, out = _run(["kv-calc", "prod-qwen3.6-27b-tq-k8v4",
                        "--fake-gpus", _A5000_2X, "--solve-max-ctx"])
        assert rc == 0
        assert "MAX CTX" in out
        # Parse the reported max ctx and assert it's a real, large number.
        line = [ln for ln in out.splitlines() if "MAX CTX" in ln][0]
        digits = line.split(":")[1].split("tokens")[0].replace(",", "").strip()
        assert int(digits) >= 100_000


class TestKvCalcJson:
    def test_json_shape(self):
        rc, out = _run(["--output", "json", "kv-calc",
                        "prod-qwen3.6-35b-balanced", "--fake-gpus", _A5000_2X])
        assert rc == 0
        data = json.loads(out)
        assert data["preset"] == "prod-qwen3.6-35b-balanced"
        assert data["verdict"] in ("PASS", "TIGHT", "FAIL")
        assert data["provisional"] is False
        per = data["per_card_gib"]
        assert {"weights", "kv_pool_requested", "activation", "total",
                "headroom", "available_for_kv"} <= set(per)
        # The 35B available-for-KV must reproduce the live ~0.69 GiB pool.
        assert 0.5 <= per["available_for_kv"] <= 0.9

    def test_fit_alias_json_equivalent(self):
        rc, out = _run(["--output", "json", "fit",
                        "prod-qwen3.6-35b-balanced", "--fake-gpus", _A5000_2X])
        assert rc == 0
        data = json.loads(out)
        assert data["verdict"] in ("PASS", "TIGHT", "FAIL")

    def test_unknown_preset_errors_cleanly(self):
        rc, out = _run(["--output", "json", "kv-calc", "no-such-xyz",
                        "--card", "24"])
        assert rc == 2
        assert "error" in json.loads(out)
