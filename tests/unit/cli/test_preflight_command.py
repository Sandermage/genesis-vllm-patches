# SPDX-License-Identifier: Apache-2.0
"""Integration tests for `sndr preflight <preset>` (v12 CLI command).

Drives the real command end-to-end against the live preset corpus using
offline rig sources (--fake-gpus / --rig) so there's no nvidia-smi dependency.
"""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

import pytest

# The CLI entrypoint imports the product-API schemas (engines), which pull in
# pydantic transitively. pydantic is not part of the minimal public-CI dep set
# (see .github/workflows/test.yml), so skip cleanly rather than abort
# collection — mirrors the fastapi importorskip guard in
# tests/unit/product_api/test_preflight_route.py.
pytest.importorskip("pydantic")

from sndr.cli.main import main  # noqa: E402


def _run(argv) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(argv)
    return rc, buf.getvalue()


class TestPreflightCommandRegistered:
    def test_command_in_registry(self):
        # COMMAND_REGISTRY is populated by build_parser()/build_subparsers();
        # build the top-level parser (the real registration path) then assert.
        from sndr.cli.commands import COMMAND_REGISTRY
        from sndr.cli.main import build_parser
        build_parser()
        assert "preflight" in COMMAND_REGISTRY

    def test_help_does_not_crash(self):
        rc, out = _run(["preflight", "prod-qwen3.6-35b-balanced",
                        "--fake-gpus", "RTX A5000:24564:8.6;RTX A5000:24564:8.6"])
        assert "preflight" in out


class TestPreflightVerdicts:
    def test_2x_preset_on_single_3090_cannot_run(self):
        rc, out = _run(["preflight", "prod-qwen3.6-35b-balanced",
                        "--fake-gpus", "RTX 3090:24576:8.6"])
        assert rc == 1
        assert "CANNOT RUN" in out
        assert "gpu_count" in out

    def test_2x_preset_on_2x_a5000_can_run(self):
        rc, out = _run(["preflight", "prod-qwen3.6-35b-balanced",
                        "--fake-gpus", "RTX A5000:24564:8.6;RTX A5000:24564:8.6"])
        # rc 0 = boots. The byte-level projection row adds a TIGHT→WARN at the
        # 35B's 280K operating point (the dev424 PN403 live-measured tight
        # margin: kv_max_concurrency 1.388 < requested 2), so the verdict is
        # RUNNABLE-with-warnings, not a clean CAN RUN. The envelope rows still
        # all PASS; the projection is what makes the fit honest.
        assert rc == 0
        assert "RUNNABLE" in out or "CAN RUN" in out
        assert "projected_vram" in out  # the new byte-level row is present

    def test_sub_floor_tp2_is_runnable_with_warnings(self):
        rc, out = _run(["preflight", "prod-gemma4-26b-default",
                        "--fake-gpus", "RTX 4060Ti:16380:8.9;RTX 4060Ti:16380:8.9"])
        assert rc == 0  # WARN does not block
        assert "WARN" in out

    def test_old_sm_cannot_run(self):
        rc, out = _run(["preflight", "prod-qwen3.6-35b-balanced",
                        "--fake-gpus", "Tesla T4:15360:7.5;Tesla T4:15360:7.5"])
        assert rc == 1
        assert "CANNOT RUN" in out

    def test_rig_flag_against_builtin_single_3090(self):
        rc, out = _run(["preflight", "prod-qwen3.6-27b-tq-k8v4",
                        "--rig", "single-3090-24gbvram"])
        assert rc == 1
        assert "single-3090-24gbvram" in out


class TestPreflightJson:
    def test_json_output_shape(self):
        rc, out = _run(["--output", "json", "preflight",
                        "prod-qwen3.6-35b-balanced",
                        "--fake-gpus", "RTX A5000:24564:8.6;RTX A5000:24564:8.6"])
        assert rc == 0
        data = json.loads(out)
        assert data["preset"] == "prod-qwen3.6-35b-balanced"
        assert data["can_run"] is True
        assert data["envelope_source"] == "card.hardware_fit"
        assert {"min_vram_gb", "min_gpu_count", "tensor_parallel",
                "min_cuda_capability", "engine_pin"} <= set(data["required"])
        dims = {c["dimension"] for c in data["checks"]}
        assert {"gpu_count", "vram", "cuda_capability", "engine_pin"} <= dims

    def test_unknown_preset_errors_cleanly(self):
        rc, out = _run(["--output", "json", "preflight",
                        "no-such-preset-xyz",
                        "--fake-gpus", "RTX 3090:24576:8.6"])
        assert rc == 2
        data = json.loads(out)
        assert "error" in data
