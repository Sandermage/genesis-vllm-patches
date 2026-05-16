# SPDX-License-Identifier: Apache-2.0
"""`sndr model-config new --from-running` — controlled-error contract.

The flag is hidden from `--help` via argparse.SUPPRESS because the
docker-inspect captor it advertises is not implemented yet. The flag
still parses for forward-compat scripts, but any direct invocation
must return a clean operator-readable error (not a Python traceback)
pointing at the supported `--template` alternative.

These tests pin both contracts so the surface can't silently drift
back to either:

  - re-appearing in `--help`, or
  - crashing on `--from-running <container>`.
"""
from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout

import pytest


def _run_model_config(argv):
    """Invoke the model-config CLI in-process; capture (rc, stdout, stderr)."""
    from vllm.sndr_core.compat import model_config_cli

    out = io.StringIO()
    err = io.StringIO()
    rc = 0
    try:
        with redirect_stdout(out), redirect_stderr(err):
            rc = model_config_cli.main(argv)
    except SystemExit as e:
        rc = int(e.code) if isinstance(e.code, int) else 2
    return rc, out.getvalue(), err.getvalue()


class TestFromRunningHidden:
    def test_new_help_does_not_advertise_from_running(self):
        """`model-config new --help` must not mention --from-running
        until the docker-inspect captor lands."""
        rc, out, err = _run_model_config(["new", "--help"])
        text = out + err
        assert "--from-running" not in text, (
            "--from-running is advertised in --help; "
            "argparse.SUPPRESS regressed."
        )

    def test_new_top_level_help_does_not_advertise(self):
        rc, out, err = _run_model_config(["--help"])
        text = out + err
        assert "--from-running" not in text


class TestFromRunningCleanError:
    def test_direct_invocation_returns_clean_error(self):
        """Passing `--from-running <container>` still parses and must
        return a clean error pointing at --template."""
        rc, out, err = _run_model_config([
            "new", "test-key", "--from-running", "some-container",
        ])
        assert rc != 0, "operator-readable error must return non-zero"
        text = out + err
        assert "from-running" in text or "--template" in text, (
            f"error message must mention the unsupported flag + "
            f"recommend --template; got stdout={out!r} stderr={err!r}"
        )
        assert "not implemented" in text.lower(), (
            f"error must explain the captor is unimplemented; got: "
            f"stdout={out!r} stderr={err!r}"
        )
