# SPDX-License-Identifier: Apache-2.0
"""Tests for the runtime environment / version report."""
from __future__ import annotations

from dataclasses import asdict

from vllm.sndr_core.product_api.environment import collect_environment_report


def test_environment_report_has_project_and_engine_fields():
    report = collect_environment_report()
    assert report.sndr_core_version
    assert report.engine_name == "vLLM"
    assert isinstance(report.engine_installed, bool)
    # engine_version is None when vllm dist metadata is absent (dev source tree).
    assert report.engine_version is None or isinstance(report.engine_version, str)
    assert report.python_version


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
