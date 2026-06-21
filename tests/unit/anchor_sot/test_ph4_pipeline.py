"""Phase 4 — bump-pipeline pure-logic tests (compare_manifest + build_manifest._mk).

The end-to-end pipeline (discovery + bare-image pristine + classify) is rig-tested
via `make audit-pin`; these cover the host-side logic that runs without a rig.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
COMPARE = REPO / "scripts" / "anchor_sot" / "compare_manifest.py"


def _write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj))
    return str(p)


def _run_compare(a, b):
    return subprocess.run(
        [sys.executable, str(COMPARE), a, b],
        capture_output=True, text=True,
    )


def test_compare_match_ignores_volatile_metadata(tmp_path):
    base = {"manifest_version": 1, "pins": {"vllm": "x"},
            "files": {"a.py": {"md5_pristine": "abc"}}}
    a = _write(tmp_path, "a.json", {**base, "generated_at": "2026-01-01T00:00:00Z",
                                    "generated_by": "gen"})
    b = _write(tmp_path, "b.json", {**base, "generated_at": "2026-12-31T23:59:59Z",
                                    "generated_by": "gen"})
    r = _run_compare(a, b)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "MATCH" in r.stdout


def test_compare_drift_when_anchor_content_differs(tmp_path):
    a = _write(tmp_path, "a.json", {"manifest_version": 1,
               "files": {"a.py": {"md5_pristine": "abc"}}})
    b = _write(tmp_path, "b.json", {"manifest_version": 1,
               "files": {"a.py": {"md5_pristine": "DIFFERENT"}}})
    r = _run_compare(a, b)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "DRIFT" in r.stdout
    assert "changed: a.py" in r.stdout


def test_compare_drift_reports_added_and_removed_files(tmp_path):
    a = _write(tmp_path, "a.json", {"files": {"keep.py": {}, "gone.py": {}}})
    b = _write(tmp_path, "b.json", {"files": {"keep.py": {}, "new.py": {}}})
    r = _run_compare(a, b)
    assert r.returncode == 1
    assert "gone.py" in r.stdout      # only in committed
    assert "new.py" in r.stdout       # only in fresh


def test_build_manifest_mk_roundtrips_anchor_target():
    """_mk must rebuild a frozen AnchorTarget from a discover.py dict
    (tuples for version range + merged markers, None when absent)."""
    sys.path.insert(0, str(REPO / "scripts" / "anchor_sot"))
    from build_manifest import _mk

    t = _mk({
        "patch_id": "PNX", "sub": "sub1", "target_rel": "a/b.py",
        "anchor": "ANCHOR", "replacement": "REPL", "required": True,
        "vllm_version_range": ["0.23.0", "0.24.0"],
        "upstream_merged_markers": ["MARKER"],
    })
    assert t.patch_id == "PNX"
    assert t.vllm_version_range == ("0.23.0", "0.24.0")
    assert t.upstream_merged_markers == ("MARKER",)

    t2 = _mk({
        "patch_id": "PNY", "sub": "s", "target_rel": "c.py",
        "anchor": "A", "replacement": None, "required": False,
        "vllm_version_range": None,
    })
    assert t2.vllm_version_range is None
    assert t2.upstream_merged_markers == ()
    assert t2.replacement is None
