# SPDX-License-Identifier: Apache-2.0
"""Tests for ``sndr gui-api`` CLI wrapper."""
from __future__ import annotations

import argparse
import subprocess
import sys

from vllm.sndr_core.cli.gui_api import run_gui_api


def test_gui_api_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "vllm.sndr_core.cli", "gui-api", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--host" in result.stdout
    assert "--port" in result.stdout
    assert "--log-level" in result.stdout


def test_cli_import_does_not_pull_fastapi_or_uvicorn():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
baseline = set(sys.modules)
import vllm.sndr_core.cli
after = set(sys.modules)
heavy = {m for m in after - baseline if m.split('.')[0] in {'fastapi', 'uvicorn'}}
if heavy:
    print(sorted(heavy))
    raise SystemExit(1)
print('OK')
""",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_run_gui_api_calls_lazy_server(monkeypatch):
    calls = []

    def fake_run_server(*, host: str, port: int, log_level: str, enable_apply: bool = False) -> None:
        calls.append((host, port, log_level))

    import vllm.sndr_core.product_api.http_app as http_app

    monkeypatch.setattr(http_app, "run_server", fake_run_server)
    rc = run_gui_api(
        argparse.Namespace(host="127.0.0.1", port=9876, log_level="warning")
    )

    assert rc == 0
    assert calls == [("127.0.0.1", 9876, "warning")]


def test_run_gui_api_keyboardinterrupt_returns_130(monkeypatch):
    def fake_run_server(*, host: str, port: int, log_level: str, enable_apply: bool = False) -> None:
        raise KeyboardInterrupt

    import vllm.sndr_core.product_api.http_app as http_app

    monkeypatch.setattr(http_app, "run_server", fake_run_server)
    rc = run_gui_api(
        argparse.Namespace(host="127.0.0.1", port=9876, log_level="warning")
    )

    assert rc == 130
