# SPDX-License-Identifier: Apache-2.0
"""Tests for ``scripts/audit_phase3_relocation.py`` — Phase 3 Bucket 6.

Verifies that the four architectural-invariant rules from §0.5 of
``sndr_private/planning/audits/RELOCATION_DESIGN_2026-05-21_RU.md``
are enforced correctly:

  R1 — Gemma whitelist (only true Gemma-owned files in gemma4/)
  R2 — canonical apply path (registry never points at a shim)
  R3 — TQ/config boundary (structured-profile envs are registered)
  R4 — no shell scripts / launcher subprocess calls in integrations/

Each test creates a synthetic violation by monkeypatching the
script's module-level path constants to point at a temporary
fixture tree. The audit is then re-run against the synthetic tree
and the expected violation is asserted in the output.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_phase3_relocation.py"


def _load_module():
    """Import the audit script as a module for in-process testing."""
    name = "_audit_phase3_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None, (
        f"could not load spec for {SCRIPT_PATH}"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─── Top-level smoke tests ──────────────────────────────────────────────


def test_script_runs_on_clean_tree_and_exits_zero():
    """Running the audit unchanged against the current repo must exit 0."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--rule", "all"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"audit returned {result.returncode} on clean tree.\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert "All selected rules clean" in result.stdout


def test_script_json_mode_emits_valid_json():
    """``--json`` produces parseable JSON with expected top-level keys."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--json"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert "_summary" in payload
    for rule in ("R1", "R2", "R3", "R4"):
        assert rule in payload, f"{rule} missing from JSON output"
        assert "status" in payload[rule]
        assert "violations" in payload[rule]
        assert "infos" in payload[rule]


def test_script_rule_selection_runs_only_that_rule():
    """``--rule R1`` skips R2/R3/R4."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--rule", "R1", "--json"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert "R1" in payload
    assert "R2" not in payload
    assert "R3" not in payload
    assert "R4" not in payload


# ─── R1: synthetic violation fixture ────────────────────────────────────


def test_r1_flags_stray_non_whitelisted_file(tmp_path, monkeypatch):
    """R1 is a directory-level rule post-Phase-2.5: `integrations/gemma4/`
    must NOT exist. Any reintroduction — even by a single stray file
    — is flagged because the directory's mere existence is the
    violation.

    Phase 4.A (2026-05-22): refreshed from the pre-Phase-2.5 fixture
    that exercised file-level allowlist semantics. R1 was tightened
    (audit_phase3_relocation.py:117-142) to forbid the directory
    itself; per-file whitelist is gone. The test's intent — "R1
    catches a stray addition" — survives, but the granularity shifted
    to the directory level.
    """
    mod = _load_module()

    fake_root = tmp_path / "integrations" / "gemma4"
    fake_root.mkdir(parents=True)
    stray = fake_root / "stray_new_tq_patch.py"
    stray.write_text(
        '"""A new TurboQuant patch someone accidentally dropped in gemma4/"""\n'
    )

    monkeypatch.setattr(mod, "GEMMA4_DIR", fake_root)

    issues = mod.check_r1_gemma_whitelist()
    assert any("integrations/gemma4/" in i for i in issues), (
        f"R1 should have flagged the forbidden directory; got: {issues}"
    )
    assert any("forbidden post-Phase-2.5" in i for i in issues), (
        f"R1 message should explain the policy; got: {issues}"
    )


def test_r1_flags_unknown_subdirectory(tmp_path, monkeypatch):
    mod = _load_module()
    fake_root = tmp_path / "integrations" / "gemma4"
    fake_root.mkdir(parents=True)
    (fake_root / "kernels").mkdir()                  # allowed
    (fake_root / "unknown_subdir").mkdir()           # violation
    monkeypatch.setattr(mod, "GEMMA4_DIR", fake_root)

    issues = mod.check_r1_gemma_whitelist()
    assert any("unknown_subdir" in i for i in issues)
    assert not any("kernels" in i for i in issues)


def test_r1_clean_tree_returns_empty(tmp_path, monkeypatch):
    """Clean tree under post-Phase-2.5 R1 means the gemma4 directory
    does NOT exist on disk. R1 must return zero violations.

    Phase 4.A (2026-05-22): refreshed from the pre-Phase-2.5 fixture
    that built an "allowed-but-clean" gemma4/ tree (whitelisted
    contents). Under the tightened R1, ANY existence of the directory
    is forbidden — so the clean-tree expectation is the absent-tree
    expectation. The fixture deliberately does NOT call mkdir.
    """
    mod = _load_module()
    fake_root = tmp_path / "integrations" / "gemma4"
    # Intentionally do not create fake_root — the absent-directory
    # state is what "clean" means under the post-Phase-2.5 R1.
    monkeypatch.setattr(mod, "GEMMA4_DIR", fake_root)
    assert mod.check_r1_gemma_whitelist() == []


# ─── R2: synthetic shim-as-target ───────────────────────────────────────


def test_r2_flags_apply_module_pointing_at_shim(tmp_path, monkeypatch):
    """If registry.apply_module resolves to a file whose docstring
    contains the shim sentinel, R2 must flag it."""
    mod = _load_module()

    # Build a fake integrations/ tree with one shim and one real module.
    fake_integrations = tmp_path / "vllm" / "sndr_core" / "integrations"
    (fake_integrations / "gemma4").mkdir(parents=True)
    (fake_integrations / "gemma4" / "g4_77_relocated.py").write_text(
        '"""Compatibility shim — G4_77 relocated.\n\n'
        'Real implementation: vllm.sndr_core.integrations.other.g4_77\n'
        '"""\n'
        'from vllm.sndr_core.integrations.other.g4_77 import *  # noqa: F401,F403\n'
    )
    (fake_integrations / "other").mkdir(parents=True)
    (fake_integrations / "other" / "g4_77.py").write_text(
        '"""real implementation"""\ndef apply(): return ("applied", "ok")\n'
    )

    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "INTEGRATIONS", fake_integrations)

    # Stub _load_registry to return one entry pointing at the SHIM.
    def _fake_registry_shim():
        return {
            "G4_77": {
                "apply_module": (
                    "vllm.sndr_core.integrations.gemma4.g4_77_relocated"
                ),
            },
        }

    monkeypatch.setattr(mod, "_load_registry", _fake_registry_shim)
    issues = mod.check_r2_canonical_apply_path()
    assert any("G4_77" in i and "shim" in i for i in issues), (
        f"R2 should have flagged the shim target; got: {issues}"
    )

    # Now flip the registry to point at the real module — clean.
    def _fake_registry_real():
        return {
            "G4_77": {
                "apply_module": "vllm.sndr_core.integrations.other.g4_77",
            },
        }

    monkeypatch.setattr(mod, "_load_registry", _fake_registry_real)
    assert mod.check_r2_canonical_apply_path() == []


# ─── R3: synthetic profile env not registered ───────────────────────────


def test_r3_flags_unknown_env_in_structured_profile(tmp_path, monkeypatch):
    """An env in the structured profile that is neither registered nor
    in PENDING_REGISTRATION must be flagged as a violation."""
    mod = _load_module()

    fake_profile = tmp_path / "gemma4-tq-mtp-structured-k4.yaml"
    fake_profile.write_text(
        "patches_delta:\n"
        "  enable:\n"
        "    GENESIS_ENABLE_KNOWN_REG: '1'\n"
        "    GENESIS_ENABLE_TOTALLY_UNKNOWN: '1'\n"
    )
    monkeypatch.setattr(mod, "STRUCTURED_PROFILE", fake_profile)

    def _fake_registered():
        return {"GENESIS_ENABLE_KNOWN_REG"}

    monkeypatch.setattr(mod, "_registered_env_flags", _fake_registered)
    errors, infos = mod.check_r3_config_boundary()
    assert any("GENESIS_ENABLE_TOTALLY_UNKNOWN" in e for e in errors)
    # The known env should not appear in either bucket.
    assert not any("GENESIS_ENABLE_KNOWN_REG" in e for e in errors)
    assert not any("GENESIS_ENABLE_KNOWN_REG" in i for i in infos)


def test_r3_pending_registration_demoted_to_info(tmp_path, monkeypatch):
    """An env in PENDING_REGISTRATION surfaces as info, not error.

    The live PENDING_REGISTRATION set is empty after the R3 cleanup
    (2026-05-21) closed all 6 originally-pending entries via real
    PATCH_REGISTRY additions. This test injects a synthetic
    PENDING_REGISTRATION entry so the waiver-demotion mechanism
    itself stays under test even when no live waivers exist.
    """
    mod = _load_module()

    synthetic_env = "GENESIS_ENABLE_SOME_FUTURE_PENDING_PATCH"
    fake_profile = tmp_path / "gemma4-tq-mtp-structured-k4.yaml"
    fake_profile.write_text(
        "patches_delta:\n"
        "  enable:\n"
        f"    {synthetic_env}: '1'\n"
    )
    monkeypatch.setattr(mod, "STRUCTURED_PROFILE", fake_profile)
    monkeypatch.setattr(mod, "_registered_env_flags", lambda: set())
    # Inject a temporary waiver so the demotion path is exercised.
    monkeypatch.setattr(mod, "PENDING_REGISTRATION", frozenset({synthetic_env}))

    errors, infos = mod.check_r3_config_boundary()
    assert errors == []
    assert any(synthetic_env in i for i in infos)


def test_r3_known_profile_envs_all_validated_against_current_registry():
    """Live check: every env in the real structured profile is either
    registered or in PENDING_REGISTRATION. No silent gaps allowed.

    This test will FAIL if someone adds a new env to the structured
    profile without also adding a registry entry or a tracked waiver.
    """
    mod = _load_module()
    errors, _ = mod.check_r3_config_boundary()
    assert errors == [], (
        f"Structured profile has env(s) that are neither registered "
        f"nor in PENDING_REGISTRATION:\n" + "\n".join(errors)
    )


# ─── R4: synthetic shell script + subprocess invocation ─────────────────


def test_r4_flags_shell_script_under_integrations(tmp_path, monkeypatch):
    mod = _load_module()
    fake_integrations = tmp_path / "vllm" / "sndr_core" / "integrations"
    (fake_integrations / "spec_decode" / "deploy").mkdir(parents=True)
    (fake_integrations / "spec_decode" / "deploy" / "start_thing.sh").write_text(
        "#!/bin/bash\necho hello\n"
    )

    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "INTEGRATIONS", fake_integrations)

    issues = mod.check_r4_no_launchers_under_integrations()
    assert any("start_thing.sh" in i for i in issues)


def test_r4_flags_subprocess_docker_run_call(tmp_path, monkeypatch):
    mod = _load_module()
    fake_integrations = tmp_path / "vllm" / "sndr_core" / "integrations"
    (fake_integrations / "x").mkdir(parents=True)
    (fake_integrations / "x" / "evil.py").write_text(
        'import subprocess\n'
        'subprocess.run(["docker", "run", "--rm", "ubuntu", "uname"])\n'
    )

    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "INTEGRATIONS", fake_integrations)

    issues = mod.check_r4_no_launchers_under_integrations()
    assert any("evil.py" in i and "subprocess" in i for i in issues)


def test_r4_does_not_flag_docstring_mentions(tmp_path, monkeypatch):
    """Files that merely document launcher syntax in their docstring
    or string literals must NOT trigger R4."""
    mod = _load_module()
    fake_integrations = tmp_path / "vllm" / "sndr_core" / "integrations"
    (fake_integrations / "x").mkdir(parents=True)
    (fake_integrations / "x" / "describes_launchers.py").write_text(
        '"""This patch survives `exec vllm serve` and `docker run` invocations."""\n'
        '\n'
        '_TEXT_PATCH_BLOCK = (\n'
        '    "# When vllm serve is invoked via docker run, this works.\\n"\n'
        ')\n'
    )

    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "INTEGRATIONS", fake_integrations)

    issues = mod.check_r4_no_launchers_under_integrations()
    assert issues == [], (
        f"R4 must NOT flag docstring/string mentions; got: {issues}"
    )


def test_r4_clean_tree_returns_empty(tmp_path, monkeypatch):
    mod = _load_module()
    fake_integrations = tmp_path / "vllm" / "sndr_core" / "integrations"
    (fake_integrations / "x").mkdir(parents=True)
    (fake_integrations / "x" / "ok.py").write_text(
        '"""Just a normal patch module."""\ndef apply(): return ("applied", "ok")\n'
    )
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "INTEGRATIONS", fake_integrations)
    assert mod.check_r4_no_launchers_under_integrations() == []


# ─── Exit-code matrix ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "rule_arg,want_clean",
    [
        ("R1", True),
        ("R2", True),
        ("R3", True),  # pending-registration is info, exit 0
        ("R4", True),
        ("all", True),
    ],
)
def test_exit_codes_on_clean_tree(rule_arg, want_clean):
    """Repo HEAD is post-Phase-3-bucket-5 clean — every rule must exit 0."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--rule", rule_arg],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    expected = 0 if want_clean else 1
    assert result.returncode == expected, (
        f"--rule {rule_arg} expected exit {expected}, got "
        f"{result.returncode}:\n{result.stdout}"
    )
