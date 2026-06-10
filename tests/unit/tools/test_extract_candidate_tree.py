# SPDX-License-Identifier: Apache-2.0
"""CLI-contract tests for ``tools/extract_candidate_tree.sh``.

The extraction path itself (docker create/cp/rm over SSH) is exercised
empirically by the pin-bump validation run (see
docs/PIN_BUMP_PLAYBOOK.md) — these tests pin the offline contract:
usage text, argument validation, exit codes, and the no-pull policy
being stated in the script (pin policy: never docker pull without an
explicit operator instruction).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "tools" / "extract_candidate_tree.sh"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True, text=True, timeout=30,
    )


def test_help_exits_zero_and_prints_usage() -> None:
    res = _run("--help")
    assert res.returncode == 0
    assert "usage" in (res.stdout + res.stderr).lower()
    assert "--image" in res.stdout + res.stderr


def test_missing_image_arg_exits_two() -> None:
    res = _run()
    assert res.returncode == 2
    assert "--image" in res.stderr


def test_unknown_flag_exits_two() -> None:
    res = _run("--image", "x", "--bogus-flag")
    assert res.returncode == 2


def test_staging_outside_tmp_refused() -> None:
    res = _run("--image", "vllm/vllm-openai:nightly-x",
               "--staging", "/home/sander/danger")
    assert res.returncode == 2
    assert "/tmp/" in res.stderr


def test_no_docker_pull_in_script() -> None:
    """Pin policy: the script must never pull. `docker pull` may only
    appear inside comments or echo'd operator hints, never as an
    executed command."""
    for line in SCRIPT.read_text().splitlines():
        code = line.split("#", 1)[0]
        if "docker pull" not in code:
            continue
        stripped = code.strip()
        assert stripped.startswith(("echo", "printf")), (
            f"executed docker pull found: {line}")
