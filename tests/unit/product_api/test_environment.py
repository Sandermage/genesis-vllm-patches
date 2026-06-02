# SPDX-License-Identifier: Apache-2.0
"""Tests for the runtime environment / version report."""
from __future__ import annotations

import importlib.util
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
