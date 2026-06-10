# SPDX-License-Identifier: Apache-2.0
"""Tests for write-safe operator-local report bundle generation."""
from __future__ import annotations

import json
from pathlib import Path

from sndr.product_api.legacy.reports import generate_report_bundle


def test_generate_report_bundle_writes_operator_local(tmp_path, monkeypatch):
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    result = generate_report_bundle(report_type="catalog", preset_id="", redact=True)
    bundle_dir = Path(result.bundle_dir)
    # Bundle is written under the operator-local SNDR_HOME, never the repo/server.
    assert str(tmp_path) in str(bundle_dir)
    assert bundle_dir.is_dir()
    # Snapshot JSON + human summary are present.
    names = {f for f in result.files}
    assert "snapshot.json" in names
    assert "summary.md" in names
    assert (bundle_dir / "snapshot.json").is_file()
    snapshot = json.loads((bundle_dir / "snapshot.json").read_text())
    assert snapshot["report_type"] == "catalog"
    assert result.redacted is True


def test_report_bundle_redacts_home_path(tmp_path, monkeypatch):
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    result = generate_report_bundle(report_type="catalog", preset_id="", redact=True)
    text = (Path(result.bundle_dir) / "snapshot.json").read_text()
    # The operator's real home directory must not leak into a redacted bundle.
    assert str(Path.home()) not in text
