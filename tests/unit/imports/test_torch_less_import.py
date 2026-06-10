# SPDX-License-Identifier: Apache-2.0
"""Torch-less import contract — F-001/F-002 regression guard.

Audit `sndr_structure_deep_audit_2026-05-07.md` flagged that a fresh
`import sndr` pulled `runtime.prealloc` → `torch` at module
top-level. CLI / schema validator / doctor / pre-commit all broke
before reaching their argparse blocks.

The fix made `runtime` (and its torch-using `prealloc` submodule) lazy
via `__getattr__`. This test guards against regression by simulating
the absence of torch with `sys.modules['torch'] = None` — the same
G-002 trick from pre-`_genesis`-removal era, kept as the regression
guard for the canonical SNDR Core package.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap

import pytest


_PROBE_SCRIPT = textwrap.dedent(
    """
    import sys
    # Block torch BEFORE any vllm import. `sys.modules[name] = None`
    # makes import machinery treat the name as missing — lookup raises
    # ModuleNotFoundError at the first `import torch` it encounters.
    sys.modules['torch'] = None
    {body}
    """
).strip()


def _run_probe(body: str) -> subprocess.CompletedProcess[str]:
    """Run a torch-less probe in a fresh subprocess.

    A fresh interpreter is required because the parent test process
    has torch imported (pytest itself may have loaded it). Subprocess
    isolates the module-cache state."""
    return subprocess.run(
        [sys.executable, "-c", _PROBE_SCRIPT.format(body=body)],
        capture_output=True, text=True,
    )


class TestTorchLessImport:
    """`import sndr` and its CLI / compat helpers must succeed
    in environments without torch installed."""

    def test_sndr_core_imports_without_torch(self):
        rc = _run_probe(
            "import sndr; "
            "print('OK', sndr.__version__)"
        )
        assert rc.returncode == 0, (
            f"sndr_core import failed without torch:\n"
            f"  stdout: {rc.stdout!r}\n"
            f"  stderr: {rc.stderr!r}"
        )
        assert "OK" in rc.stdout

    def test_sndr_core_cli_module_loads_without_torch(self):
        rc = _run_probe("import sndr.cli; print('OK')")
        assert rc.returncode == 0, rc.stderr
        assert "OK" in rc.stdout

    def test_compat_schema_validator_loads_without_torch(self):
        rc = _run_probe(
            "from sndr.compat import schema_validator; print('OK')"
        )
        assert rc.returncode == 0, rc.stderr
        assert "OK" in rc.stdout


class TestRuntimeLazyAttributes:
    """Lazy `__getattr__` must still expose all submodules to consumers
    that explicitly access them (e.g. `from sndr.runtime
    import prealloc`)."""

    @pytest.mark.parametrize("name", [
        "buffer_mode", "prealloc", "prealloc_budget", "pool_budget",
        "memory_metrics", "gpu_profile", "spec_meta", "interface_guard",
    ])
    def test_runtime_submodule_accessible(self, name):
        """In a torch-equipped test process, runtime submodules must
        still resolve via normal attribute access. (We're running with
        torch present here — the lazy contract just defers the load,
        it doesn't hide submodules.)"""
        runtime = importlib.import_module("sndr.runtime")
        sub = getattr(runtime, name)
        assert sub.__name__ == f"sndr.runtime.{name}"
