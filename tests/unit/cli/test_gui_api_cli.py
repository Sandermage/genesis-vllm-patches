# SPDX-License-Identifier: Apache-2.0
"""Tests for ``sndr gui-api`` CLI wrapper."""
from __future__ import annotations

import argparse
import subprocess
import sys

from sndr.cli.legacy.gui_api import run_gui_api


import pytest


@pytest.mark.parametrize("module", ["sndr.cli.legacy", "sndr.cli"])
def test_gui_api_help_exits_zero(module):
    """Both the legacy shim path and the modern ``sndr.cli`` package must
    actually dispatch ``cli_main`` for ``python -m``. A ``from ... import *``
    shim does NOT inherit the target's ``if __name__ == '__main__'`` block,
    so a regressed shim imports and exits 0 with empty stdout — this asserts
    the argparse help is really produced."""
    result = subprocess.run(
        [sys.executable, "-m", module, "gui-api", "--help"],
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
import sndr.cli
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

    import sndr.product_api.legacy.http_app as http_app

    monkeypatch.setattr(http_app, "run_server", fake_run_server)
    rc = run_gui_api(
        argparse.Namespace(host="127.0.0.1", port=9876, log_level="warning")
    )

    assert rc == 0
    assert calls == [("127.0.0.1", 9876, "warning")]


def test_run_gui_api_keyboardinterrupt_returns_130(monkeypatch):
    def fake_run_server(*, host: str, port: int, log_level: str, enable_apply: bool = False) -> None:
        raise KeyboardInterrupt

    import sndr.product_api.legacy.http_app as http_app

    monkeypatch.setattr(http_app, "run_server", fake_run_server)
    rc = run_gui_api(
        argparse.Namespace(host="127.0.0.1", port=9876, log_level="warning")
    )

    assert rc == 130
