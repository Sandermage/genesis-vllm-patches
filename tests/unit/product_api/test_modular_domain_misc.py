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

import pytest

# The modular domain schemas (sndr.product_api.schemas.*) are pydantic models;
# the light CI test leg installs no pydantic. Skip cleanly there instead of
# failing collection (matches the importorskip convention used across this dir).
pytest.importorskip("pydantic")

import sndr.product_api.domain.license_status as license_status  # noqa: E402
from sndr.product_api.domain.license_status import get_license_status  # noqa: E402
from sndr.product_api.domain.pins_service import list_pins  # noqa: E402
from sndr.product_api.schemas.licensing import LicenseStatus  # noqa: E402
from sndr.product_api.schemas.pins import PinSummary  # noqa: E402


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
    """The manifest-backed listing still works, and now ALSO surfaces the current
    pins that ship anchors.json/drift.rej.json instead of manifest.yaml."""
    pins = list_pins("vllm")
    assert all(isinstance(p, PinSummary) for p in pins)
    by_name = {p.pin: p for p in pins}
    # The two legacy manifest.yaml pins are committed under engines/vllm/pins.
    assert by_name["0.21.1_626fa9bba"].has_manifest is True
    assert by_name["0.22.1_da1daf40b"].has_manifest is True
    # The current anchors.json-backed pins are now listed too (no manifest).
    assert "0.23.1_3f5a1e173" in by_name
    assert by_name["0.23.1_3f5a1e173"].has_manifest is False


def test_list_pins_unknown_engine_returns_empty():
    assert list_pins("definitely-not-an-engine") == []


def test_list_pins_derives_real_status_and_drift(monkeypatch):
    """Status + drift are derived from real on-disk artifacts, not hardcoded to
    staging/False. The canonical vllm_pin_required pin is 'current' and surfaces
    its genuine anchor drift; the prior declared pin is 'previous'; an undeclared
    on-disk pin is 'staging'. Current pins ship anchors.json + drift.rej.json (no
    manifest.yaml) and must STILL be listed (a manifest-only listing hid them)."""
    import sndr.product_api.legacy.updater as updater
    monkeypatch.setattr(
        updater, "supported_pins",
        lambda *a, **k: ["0.23.1rc1.dev424+g3f5a1e173", "0.23.1rc1.dev148+gb4c80ec0f"],
    )
    pins = {p.pin: p for p in list_pins("vllm")}
    # current pins (anchors.json/drift.rej.json, no manifest.yaml) are now listed
    assert "0.23.1_3f5a1e173" in pins, "the current pin must be listed even without manifest.yaml"
    cur = pins["0.23.1_3f5a1e173"]
    assert cur.status == "current"
    assert cur.has_drift is True       # genuine_anchor_drift has entries
    assert cur.has_manifest is False   # ships anchors.json, not manifest.yaml
    prev = pins["0.23.1_b4c80ec0f"]
    assert prev.status == "previous"
    assert prev.has_drift is False
    assert pins["0.23.1_04c2a8dea"].status == "staging"
