# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the read-only anchor-manifest status (GUI surface)."""
from __future__ import annotations

import json

from sndr.product_api.legacy.patches import anchor_status as ast


def _manifest(vllm_ver, *, nfiles=2, patches_per_file=1, anchors_per_patch=2):
    files = {}
    for fi in range(nfiles):
        patches = {}
        for pi in range(patches_per_file):
            anchors = {
                f"anc{ai}": {"anchor_md5": "a", "byte_length": 1, "byte_offset": 0, "replacement_md5": "r"}
                for ai in range(anchors_per_patch)
            }
            patches[f"PN{fi}_{pi}"] = {"anchors": anchors}
        files[f"f{fi}.py"] = {"md5_pristine": f"md5{fi}", "patches": patches}
    return {
        "manifest_version": 1, "generated_at": "2026-06-21T00:00:00Z",
        "generated_by": "test", "pins": {"vllm": vllm_ver, "genesis": "12.0.0"}, "files": files,
    }


def test_counts_files_patches_anchors():
    c = ast._counts(_manifest("v", nfiles=2, patches_per_file=2, anchors_per_patch=3))
    assert c == {"files": 2, "patches": 4, "anchors": 12}


def test_manifest_status_marks_active_and_counts(tmp_path, monkeypatch):
    pin_dir = tmp_path / "0.23.1_gX"
    pin_dir.mkdir()
    (pin_dir / "anchors.json").write_text(json.dumps(_manifest("0.23.1+gX")))
    monkeypatch.setattr(ast, "_pins_dir", lambda: tmp_path)
    monkeypatch.setattr(ast, "_running_vllm", lambda: "0.23.1+gX")

    out = ast.manifest_status(drift=False)
    assert out["available"] is True and out["manifest_count"] == 1
    assert out["running_vllm"] == "0.23.1+gX"
    e = out["manifests"][0]
    assert e["active"] is True and e["vllm"] == "0.23.1+gX" and e["genesis"] == "12.0.0"
    assert (e["files"], e["patches"], e["anchors"]) == (2, 2, 4)
    assert out["drift"]["checked"] is False  # drift=False -> skipped


def test_manifest_status_no_active_on_pin_mismatch(tmp_path, monkeypatch):
    pin_dir = tmp_path / "p"
    pin_dir.mkdir()
    (pin_dir / "anchors.json").write_text(json.dumps(_manifest("0.23.1+gX")))
    monkeypatch.setattr(ast, "_pins_dir", lambda: tmp_path)
    monkeypatch.setattr(ast, "_running_vllm", lambda: "0.99-other-pin")

    out = ast.manifest_status(drift=True)
    assert all(not e["active"] for e in out["manifests"])
    assert out["drift"]["checked"] is False
    assert "no manifest matches" in out["drift"]["reason"]


def test_manifest_status_handles_no_pins_dir(monkeypatch):
    monkeypatch.setattr(ast, "_pins_dir", lambda: None)
    monkeypatch.setattr(ast, "_running_vllm", lambda: None)
    out = ast.manifest_status()
    assert out["available"] is False and out["manifest_count"] == 0 and out["manifests"] == []


def test_check_drift_reports_mismatches(monkeypatch, tmp_path):
    """_check_drift reuses verify_manifest_against_source against the live source;
    here we point the loader at a fake source tree and assert the count."""
    import sndr.engines.vllm.wiring.anchor_manifest as am

    # a manifest whose md5 will NOT match the fake source -> drift
    m = _manifest("v")
    captured = {}

    def _fake_verify(manifest, loader):
        captured["loaded"] = loader("f0.py")
        return ["f0.py: md5 mismatch", "f1.py: PN1_0.anc0 anchor_md5 mismatch"]

    monkeypatch.setattr(am, "verify_manifest_against_source", _fake_verify)
    # make vllm import resolve to a dir we control
    import types, sys
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__file__ = str(tmp_path / "vllm" / "__init__.py")
    (tmp_path / "vllm").mkdir()
    (tmp_path / "vllm" / "f0.py").write_text("source-of-f0")
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    res = ast._check_drift(m, limit=10)
    assert res["checked"] is True and res["in_sync"] is False
    assert res["drift_count"] == 2 and len(res["details"]) == 2
    assert captured["loaded"] == "source-of-f0"
