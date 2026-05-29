# SPDX-License-Identifier: Apache-2.0
"""Tests for ``scripts/audit_yaml_cross_rig_deltas.py``.

Per UNIFIED_DEVELOPMENT_PLAN v1.1 §2.9 Phase A — script extracts measured
cross-rig delta annotations from PATCH_REGISTRY credit fields and suggests
inline YAML comment enhancements. Pure-function coverage for the regex
extractors + tracked-tree smoke.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_yaml_cross_rig_deltas.py"


def _import_audit_module():
    spec = importlib.util.spec_from_file_location(
        "audit_yaml_cross_rig_deltas", SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["audit_yaml_cross_rig_deltas"] = mod
    assert spec.loader is not None  # nosec
    spec.loader.exec_module(mod)
    return mod


# ───────────────────────── pure-function coverage ──────────────────


def test_pct_regex_matches_percentage_delta() -> None:
    mod = _import_audit_module()
    samples = [
        ("+15% TPS", ["15%"]),
        ("+15-30% TPS", ["15-30%"]),
        ("-9.27% TPS", ["9.27%"]),
        ("+0.5pp accept", ["0.5pp"]),
    ]
    for text, expected_any in samples:
        matches = [m.group(0).strip().lstrip("+-−–")
                   for m in mod._PCT_RE.finditer(text)]
        for exp in expected_any:
            # Accept either +N% or N% (lenient on leading sign)
            assert any(exp.rstrip("%") in mm or exp in mm
                       for mm in matches), (
                f"Expected {exp!r} in {matches!r} for input {text!r}"
            )


def test_vram_regex_matches_memory_delta() -> None:
    mod = _import_audit_module()
    samples = [
        "-142 MiB/GPU",
        "+200-400 MiB margin",
        "-1.7 GiB peak",
        "−300-500 MB",
    ]
    for text in samples:
        matches = list(mod._VRAM_RE.finditer(text))
        assert matches, f"Expected VRAM delta in {text!r}, got nothing"


def test_rig_regex_identifies_hardware() -> None:
    mod = _import_audit_module()
    samples = [
        ("+12% on A5000 FP8", "A5000"),
        ("+10.5% on 3090 INT4", "3090"),
        ("+6.6% on H100 nightly", "H100"),
    ]
    for text, expected_rig in samples:
        m = mod._RIG_RE.search(text)
        assert m is not None
        assert expected_rig.lower() in m.group(1).lower()


def test_neutral_regex_detects_neutral_marker() -> None:
    mod = _import_audit_module()
    assert mod._NEUTRAL_RE.search("empirically neutral on our shape")
    assert mod._NEUTRAL_RE.search("within CV bounds")
    assert mod._NEUTRAL_RE.search("no measurable difference")
    assert not mod._NEUTRAL_RE.search("+15% TPS measured")


def test_bench_regex_detects_evidence_markers() -> None:
    mod = _import_audit_module()
    assert mod._BENCH_RE.search("bench-validated 2026-05-29")
    assert mod._BENCH_RE.search("empirically confirmed")
    assert mod._BENCH_RE.search("verified on rig")
    assert not mod._BENCH_RE.search("theoretical improvement")


def test_extract_deltas_combines_signals() -> None:
    mod = _import_audit_module()
    info = mod.PatchInfo(
        patch_id="P_TEST",
        title="Test patch",
        credit=(
            "+12% TPS on A5000 FP8, +10.5% on 3090 INT4 — "
            "bench-validated 2026-05-29 across multiple rigs. "
            "Neutral on 27B (within CV)."
        ),
    )
    mod._extract_deltas(info)
    assert info.deltas, "Expected at least one delta extracted"
    assert info.rigs, "Expected rigs identified"
    assert "A5000" in info.rigs[0] or "3090" in info.rigs[0]
    assert info.has_bench_evidence
    assert info.has_neutral


def test_extract_deltas_empty_credit() -> None:
    mod = _import_audit_module()
    info = mod.PatchInfo(patch_id="P_EMPTY", title="", credit="")
    mod._extract_deltas(info)
    assert info.deltas == []
    assert info.rigs == []
    assert not info.has_neutral
    assert not info.has_bench_evidence


def test_suggest_annotation_with_deltas() -> None:
    mod = _import_audit_module()
    info = mod.PatchInfo(
        patch_id="P_X",
        title="x",
        credit="",
        deltas=["+15%", "+30%"],
        rigs=["A5000"],
        has_bench_evidence=True,
    )
    usage = mod.YamlPatchUsage(
        patch_id="P_X", env_var="GENESIS_ENABLE_P_X",
        enabled=True, line_no=1, current_comment="",
    )
    suggestion = mod._suggest_annotation(info, usage)
    assert "15%" in suggestion or "30%" in suggestion
    assert "A5000" in suggestion
    assert "verify before commit" in suggestion


def test_suggest_annotation_no_evidence() -> None:
    mod = _import_audit_module()
    info = mod.PatchInfo(patch_id="P_X", title="", credit="")
    usage = mod.YamlPatchUsage(
        patch_id="P_X", env_var="GENESIS_ENABLE_P_X",
        enabled=True, line_no=1, current_comment="",
    )
    suggestion = mod._suggest_annotation(info, usage)
    assert "no isolated bench data" in suggestion


def test_has_deltas_in_comment_detects_existing() -> None:
    mod = _import_audit_module()
    assert mod._has_deltas_in_comment("# +32% TPS on 35B PROD")
    assert mod._has_deltas_in_comment("# −142 MiB/GPU at boot")
    assert mod._has_deltas_in_comment("# neutral on 27B")
    assert not mod._has_deltas_in_comment("# vllm#40807 backport")
    assert not mod._has_deltas_in_comment("# Genesis-original")


# ───────────────────────── tracked-tree smoke ─────────────────────


def test_tracked_tree_loads_registry_credits() -> None:
    """Script can parse our real registry without crashing."""
    mod = _import_audit_module()
    credits = mod._load_registry_credits()
    # Sanity: registry has at least 100 patches (we have 231)
    assert len(credits) >= 100
    # Spot-check well-known patches
    assert "P67" in credits or "P67c" in credits
    assert "PN286" in credits
    assert "PN59" in credits


def test_cli_smoke_against_real_yaml() -> None:
    """End-to-end: script runs against tracked-tree YAMLs without error."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--yaml",
         "qwen3.6-35b-a3b-fp8"],
        cwd=REPO_ROOT,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, (
        f"Script failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    # Output should mention the YAML name and have a summary section
    assert "qwen3.6-35b-a3b-fp8.yaml" in result.stdout
    assert "Summary" in result.stdout


def test_json_output_shape() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--yaml",
         "qwen3.6-35b-a3b-fp8", "--json"],
        cwd=REPO_ROOT,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert isinstance(data, list)
    if data:
        row = data[0]
        expected_keys = {
            "yaml", "line", "patch_id", "env_var", "enabled",
            "current_comment", "suggested_comment",
            "extracted_deltas", "extracted_rigs",
            "has_neutral", "has_bench_evidence",
        }
        assert expected_keys <= set(row.keys())
