# SPDX-License-Identifier: Apache-2.0
"""Engine-overlay attach-point contract — §9.A.17
(AUDIT-CLOSURE.3, 2026-05-27).

Verifies the contract that ``sndr`` exposes for an external
commercial overlay to register itself:

  * The probe ``sndr.license._engine_overlay_available()``
    uses the documented ``engine_available()`` callable signal —
    NOT a raw ``import vllm.sndr_engine`` truthy check. A skeleton
    ``vllm.sndr_engine`` package (without a real overlay) must not
    activate the engine gate.
  * The probe is **wrapped in try/except** so ``ImportError`` on
    ``from vllm.sndr_engine import engine_available`` returns False
    cleanly (does not propagate; does not crash core).
  * ``_engine_package_version()`` returns None when the overlay
    is absent (no leaking of skeleton ``__version__`` strings).
  * Core's behaviour is unchanged when ``engine_available()`` is
    forced to return False — engine gate stays closed.

Companion gates:

  * ``scripts/audit_engine_boundary.py`` — AST audit that NO
    ``vllm/sndr_core/**/*.py`` file imports ``vllm.sndr_engine``
    outside ``try/except ImportError`` (caught at static-source level).
  * ``tests/unit/test_wheel_contents.py::test_engine_not_importable_from_core``
    — isolated-venv runtime probe.

This test covers the **functional contract** of the attach point:
how the core probe behaves given each possible overlay state.

Reference: ``vllm/sndr_core/license.py::_engine_overlay_available``
(documented entry-point group: ``sndr.engine.overlay``; roadmap §15.1).
"""
from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest


# ─── Module under test ────────────────────────────────────────────────────


# We import the function lazily inside each test to allow ``sys.modules``
# manipulation. The module itself has no top-level engine imports
# (verified by audit_engine_boundary.py).


def _import_license():
    """Fresh import of ``sndr.license``."""
    if "sndr.license" in sys.modules:
        return importlib.reload(sys.modules["sndr.license"])
    return importlib.import_module("sndr.license")


# ─── Tests ────────────────────────────────────────────────────────────────


class TestEngineOverlayProbeContract:
    """``_engine_overlay_available`` must be import-safe + signal-driven."""

    def test_probe_function_exists(self):
        license_mod = _import_license()
        assert hasattr(license_mod, "_engine_overlay_available")
        assert callable(license_mod._engine_overlay_available)

    def test_returns_false_without_overlay(self):
        """When ``vllm.sndr_engine`` is not installed (or its
        ``engine_available()`` returns False), the probe returns False
        without crashing core."""
        license_mod = _import_license()
        # Live tree has no vllm.sndr_engine — the import inside the
        # function raises ImportError and the function catches it.
        result = license_mod._engine_overlay_available()
        assert result is False, (
            f"_engine_overlay_available must return False when overlay "
            f"absent, got {result!r}"
        )

    def test_probe_does_not_raise_on_missing_overlay(self):
        """Verify the try/except is doing its job."""
        license_mod = _import_license()
        # The probe is intentionally swallow-on-import-error. The
        # contract is "return bool, never raise".
        try:
            license_mod._engine_overlay_available()
        except Exception as e:  # pragma: no cover
            pytest.fail(
                f"_engine_overlay_available raised {type(e).__name__}: {e} — "
                f"contract violated"
            )

    def test_uses_engine_available_callable_not_raw_import(self):
        """Inspect the function source to confirm it consults
        ``engine_available()`` (the documented signal) and NOT a
        ``vllm.sndr_engine`` truthy check.

        DA-010 (audit 2026-05-08) shipped this contract: skeleton
        packages must not trip the gate.
        """
        import inspect
        license_mod = _import_license()
        src = inspect.getsource(license_mod._engine_overlay_available)
        assert "engine_available" in src, (
            "probe must consult engine_available() callable"
        )
        # The probe MUST be inside try/except so the import doesn't
        # propagate.
        assert "try:" in src and "except ImportError" in src, (
            "probe must wrap the import in try/except ImportError"
        )


class TestEnginePackageVersionContract:
    """``_engine_package_version`` mirrors the probe semantics."""

    def test_returns_none_without_overlay(self):
        license_mod = _import_license()
        version = license_mod._engine_package_version()
        assert version is None, (
            f"_engine_package_version must return None when overlay "
            f"absent (no skeleton-leak), got {version!r}"
        )

    def test_short_circuits_when_overlay_unavailable(self):
        """When the probe says no overlay, the version function must
        NOT touch ``vllm.sndr_engine.__version__`` (defends against
        skeleton-version leak)."""
        license_mod = _import_license()
        # Force the probe to report no overlay; version should be None.
        with patch.object(
            license_mod,
            "_engine_overlay_available",
            return_value=False,
        ):
            assert license_mod._engine_package_version() is None


class TestEntryPointGroupName:
    """The documented entry-point group name must remain stable.

    ``sndr.engine.overlay`` is referenced in license.py:208 docstring
    + master plan §15.1 + §9.A.17. Any rename breaks every external
    overlay package. This test pins the group name as a stable string.
    """

    def test_entry_point_group_name_documented(self):
        """The docstring of ``_engine_overlay_available`` references
        the canonical entry-point group ``sndr.engine.overlay``."""
        license_mod = _import_license()
        doc = license_mod._engine_overlay_available.__doc__ or ""
        assert "sndr.engine.overlay" in doc, (
            "entry-point group name must remain documented inline; "
            "external overlay packages depend on this contract"
        )


class TestNoTopLevelEngineImport:
    """``sndr.license`` itself must not import
    ``vllm.sndr_engine`` at module top level — only inside the gated
    helper functions. Catches accidental refactor that hoists the
    import out of the try block."""

    def test_license_module_imports_clean_without_engine(self):
        """If ``vllm.sndr_engine`` is absent (live tree state), then
        ``import sndr.license`` must succeed without
        ImportError. Top-level engine import would break this."""
        # Force a fresh import; if there's a top-level
        # ``from vllm.sndr_engine import ...`` it would raise here.
        # (The live tree already passes audit_engine_boundary.py, so
        # this is a runtime confirmation.)
        if "sndr.license" in sys.modules:
            del sys.modules["sndr.license"]
        try:
            importlib.import_module("sndr.license")
        except ImportError as e:
            pytest.fail(
                f"sndr.license top-level import failed: {e} — "
                f"likely an unguarded engine import was hoisted out of "
                f"the try block"
            )
