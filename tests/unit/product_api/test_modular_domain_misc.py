# SPDX-License-Identifier: Apache-2.0
"""Coverage for the modular domain services touched by the integrity pass.

These services previously shipped with zero unit coverage (the gap that
let the jobs / container_count stubs ship). The tests below lock the
behaviour of the two services cleaned up here:

  - ``license_status``: the catch around the license probe was narrowed
    from a dead ``(AttributeError, Exception)`` to a single ``Exception``;
    a missing probe must still degrade to an honest 'unknown'.
  - ``pins_service``: a dead ``module_path`` local was removed; the
    manifest-backed pin listing must still resolve real ``manifest.yaml``
    pins and reject unknown engines.
"""
from __future__ import annotations

import sndr.product_api.domain.license_status as license_status
from sndr.product_api.domain.license_status import get_license_status
from sndr.product_api.domain.pins_service import list_pins
from sndr.product_api.schemas.licensing import LicenseStatus
from sndr.product_api.schemas.pins import PinSummary


# ── license_status ──────────────────────────────────────────────────────────


def test_get_license_status_returns_schema_object():
    status = get_license_status()
    assert isinstance(status, LicenseStatus)
    # status is always one of the documented enum-ish strings (never crashes).
    assert isinstance(status.status, str) and status.status


def test_license_probe_failure_degrades_to_unknown(monkeypatch):
    """A probe that raises must be caught and reported as an honest
    'unknown'. The catch was simplified from the dead
    ``(AttributeError, Exception)`` tuple to a single ``Exception`` (same
    behaviour, no redundant subtype); this pins the degradation path that
    previously had no coverage at all."""
    class _FakeLicenseMod:
        @staticmethod
        def check_engine_tier_eligible():
            raise RuntimeError("probe blew up")

    import importlib

    real_import_module = importlib.import_module

    def _fake_import(name, *a, **k):
        if name == "sndr.license":
            return _FakeLicenseMod
        return real_import_module(name, *a, **k)

    # license_status imports ``from sndr import license as license_mod`` —
    # patch that binding path via the sndr package attribute.
    import sndr
    monkeypatch.setattr(sndr, "license", _FakeLicenseMod, raising=False)

    status = get_license_status()
    assert status.status == "unknown"
    assert status.message == "License probe not available."


def test_license_module_absent_returns_unknown(monkeypatch):
    """If the license module is not importable at all, status is unknown."""
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *a, **k):
        if name == "sndr.license" or (name == "sndr" and a and "license" in (a[2] or ())):
            raise ImportError("no license module")
        return real_import(name, *a, **k)

    # The service does ``from sndr import license`` — block that import.
    monkeypatch.setattr(builtins, "__import__", _blocked)
    # Drop any cached module so the from-import re-runs.
    import sys
    monkeypatch.delitem(sys.modules, "sndr.license", raising=False)

    status = get_license_status()
    assert status.status == "unknown"


# ── pins_service ────────────────────────────────────────────────────────────


def test_list_pins_resolves_real_manifest_pins():
    """The manifest-backed listing still works after the dead-code cleanup."""
    pins = list_pins("vllm")
    assert all(isinstance(p, PinSummary) for p in pins)
    names = {p.pin for p in pins}
    # The two legacy manifest.yaml pins are committed under engines/vllm/pins.
    assert "0.21.1_626fa9bba" in names
    assert "0.22.1_da1daf40b" in names
    for p in pins:
        assert p.has_manifest is True


def test_list_pins_unknown_engine_returns_empty():
    assert list_pins("definitely-not-an-engine") == []
