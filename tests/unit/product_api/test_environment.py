# SPDX-License-Identifier: Apache-2.0
"""Tests for the runtime environment / version report."""
from __future__ import annotations

import importlib.util
from dataclasses import asdict

from sndr.product_api.legacy.environment import collect_environment_report


def test_environment_report_has_project_and_engine_fields():
    report = collect_environment_report()
    assert report.sndr_core_version
    assert report.engine_name == "vLLM"
    assert isinstance(report.engine_installed, bool)
    # engine_version is None when vllm dist metadata is absent (dev source tree).
    assert report.engine_version is None or isinstance(report.engine_version, str)
    assert report.python_version


def test_engine_installed_reflects_vllm_not_commercial_tier():
    """``engine_installed`` describes the vLLM RUNTIME (engine_name/engine_version),
    not the optional commercial ``vllm.sndr_engine`` tier. Regression guard: these
    were wired to the wrong package, so a stock vLLM install reported "no"."""
    report = collect_environment_report()
    vllm_importable = importlib.util.find_spec("vllm") is not None
    assert report.engine_installed == vllm_importable
    # The test suite runs against a real vLLM source tree, so this must be True.
    assert report.engine_installed is True


def test_environment_report_lists_dependencies_and_tools():
    report = collect_environment_report()
    names = {d.name for d in report.dependencies}
    assert {"vllm", "torch", "fastapi"} <= names
    for dep in report.dependencies:
        assert dep.present == (dep.version is not None)
    tool_names = {t.name for t in report.tools}
    assert {"docker", "nvidia-smi"} <= tool_names

    payload = asdict(report)
    assert isinstance(payload["dependencies"], (list, tuple))


def test_restart_command_prefers_console_script_when_on_path(monkeypatch):
    # A real pip install puts an `sndr` console script on PATH -> the restart
    # command resolves from any working directory, so use it bare.
    from sndr.product_api.legacy import environment as env
    monkeypatch.setattr(env.shutil, "which", lambda name: "/usr/local/bin/sndr" if name == "sndr" else None)
    _, _, _, cmd = env._daemon_launch_context()
    assert cmd == "sndr gui-api --enable-apply"


def test_restart_command_cds_into_install_root_for_source_checkout(monkeypatch):
    # No console script (no pip install): the daemon runs from a source tree, so
    # the command MUST cd into the directory that contains the importable sndr/
    # package — otherwise `python -m sndr.cli` fails with No module named 'sndr'
    # (the exact error an operator hits running it from their home directory).
    from sndr.product_api.legacy import environment as env
    monkeypatch.setattr(env.shutil, "which", lambda name: None)
    py, root, _, cmd = env._daemon_launch_context()
    assert root is not None and (env._Path(root) / "sndr").is_dir()
    assert cmd == f"cd '{root}' && {py} -m sndr.cli gui-api --enable-apply"
    assert "sndr.cli.legacy" not in cmd   # never the vllm-namespace shim path


def test_environment_report_exposes_restart_command(monkeypatch):
    from sndr.product_api.legacy import environment as env
    monkeypatch.setattr(env.shutil, "which", lambda name: None)
    report = collect_environment_report()
    assert "sndr.cli gui-api --enable-apply" in report.restart_command
    assert report.python_executable
    assert report.install_root and report.install_root in report.restart_command
