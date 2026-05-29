# SPDX-License-Identifier: Apache-2.0
"""Tests for ``tools/verify_stress.py``.

The tool is an HTTP client so we don't make live calls; instead:
    1. argparse contract — flags accepted, defaults sane
    2. ProbeResult dataclass shape
    3. _format_verdict pure function
    4. _build_niah_prompt approximate-length behavior
    5. _filler_paragraph + _random_secret invariants
    6. PROBES + QUICK_SET wiring
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "verify_stress.py"

if not pytest.importorskip("aiohttp", reason="aiohttp required by verify_stress"):
    pytest.skip("aiohttp missing", allow_module_level=True)


def _import_tool():
    spec = importlib.util.spec_from_file_location("verify_stress", TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["verify_stress"] = mod
    assert spec.loader is not None  # nosec
    spec.loader.exec_module(mod)
    return mod


def test_argparse_defaults() -> None:
    mod = _import_tool()
    saved = sys.argv
    try:
        sys.argv = ["verify_stress.py"]
        args = mod._parse_args()
    finally:
        sys.argv = saved
    assert args.url == "http://localhost:8000/v1"
    assert args.model == "qwen3.6-35b"
    assert args.quick is False
    assert args.no_large_niah is False
    assert args.continue_on_fail is False
    assert args.long_ctx_timeout > 0
    assert args.tool_prefill_timeout > 0
    assert args.ceiling_start_tokens < args.ceiling_max_tokens
    assert args.ceiling_step_tokens > 0


def test_argparse_quick_mode() -> None:
    mod = _import_tool()
    saved = sys.argv
    try:
        sys.argv = ["verify_stress.py", "--quick", "--continue-on-fail"]
        args = mod._parse_args()
    finally:
        sys.argv = saved
    assert args.quick is True
    assert args.continue_on_fail is True


def test_probes_inventory_shape() -> None:
    mod = _import_tool()
    assert len(mod.PROBES) == 8, "8-probe contract per club-3090 verify-stress"
    names = {n for n, _ in mod.PROBES}
    assert "probe1_niah_small" in names
    assert "probe2_tool_prefill_oom" in names
    assert "probe7_niah_large" in names
    assert "probe8_ceiling_ladder" in names


def test_quick_set_subset_of_probes() -> None:
    mod = _import_tool()
    all_probe_names = {n for n, _ in mod.PROBES}
    assert mod.QUICK_SET <= all_probe_names
    # Quick mode should be 3 probes (per docstring: "probes 1+3+4 only")
    assert len(mod.QUICK_SET) == 3


def test_probe_result_shape() -> None:
    mod = _import_tool()
    r = mod.ProbeResult(name="x", status="PASS", elapsed_s=1.5)
    assert r.name == "x"
    assert r.status == "PASS"
    assert r.elapsed_s == 1.5
    assert r.detail == ""
    assert r.extra == {}


def test_format_verdict_all_pass() -> None:
    mod = _import_tool()
    results = [
        mod.ProbeResult(name="a", status="PASS", elapsed_s=1.0),
        mod.ProbeResult(name="b", status="PASS", elapsed_s=2.0),
    ]
    verdict, summary = mod._format_verdict(results)
    assert verdict == "PASS"
    assert "2 probe" in summary


def test_format_verdict_fail() -> None:
    mod = _import_tool()
    results = [
        mod.ProbeResult(name="a", status="PASS", elapsed_s=1.0),
        mod.ProbeResult(name="b", status="FAIL", elapsed_s=2.0,
                        detail="HTTP 500: oom"),
    ]
    verdict, summary = mod._format_verdict(results)
    assert verdict == "FAIL"
    assert "b" in summary
    assert "oom" in summary or "500" in summary


def test_format_verdict_warn() -> None:
    mod = _import_tool()
    results = [
        mod.ProbeResult(name="a", status="WARN", elapsed_s=1.0, detail="..."),
    ]
    verdict, _ = mod._format_verdict(results)
    assert verdict == "WARN"


def test_random_secret_format() -> None:
    mod = _import_tool()
    for _ in range(20):
        token, phrase = mod._random_secret()
        # Token shape: color_animal_NN
        parts = token.split("_")
        assert len(parts) == 3
        assert len(parts[2]) == 2
        assert parts[2].isdigit()
        # Phrase contains the token
        assert token in phrase


def test_filler_paragraph_length_monotonic() -> None:
    mod = _import_tool()
    short = mod._filler_paragraph(50)
    long = mod._filler_paragraph(500)
    assert len(short.split()) == 50
    assert len(long.split()) == 500


def test_build_niah_prompt_grows_with_target() -> None:
    mod = _import_tool()
    short = mod._build_niah_prompt(1000, "the secret is foo")
    long = mod._build_niah_prompt(10000, "the secret is foo")
    assert len(short) < len(long)
    # Both should contain the planted secret
    assert "the secret is foo" in short
    assert "the secret is foo" in long
