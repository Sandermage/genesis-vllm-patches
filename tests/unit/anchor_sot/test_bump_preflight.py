"""build_manifest dependency_breakage section + bump_preflight gate.

  * build_manifest writes a populated ``dependency_breakage`` section into
    drift.rej.json (run over the live registry, so PN353A->PN399 is present),
  * summarize_rej prints the ⚠ retire-broken dependents block,
  * bump_preflight exits NON-ZERO when a perf dependent breaks ok->skip across
    pins, and exits 0 on a clean bump.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ANCHOR_SOT = REPO / "scripts" / "anchor_sot"
BUILD = ANCHOR_SOT / "build_manifest.py"
SUMMARIZE = ANCHOR_SOT / "summarize_rej.py"
PREFLIGHT = ANCHOR_SOT / "bump_preflight.py"


# ─── build_manifest dependency_breakage section ────────────────────────────

def _target(pid, sub, rel, anchor, repl, lifecycle=None):
    return {
        "patch_id": pid, "sub": sub, "target_rel": rel,
        "anchor": anchor, "replacement": repl, "required": True,
        "vllm_version_range": None, "upstream_merged_markers": [],
        "lifecycle": lifecycle,
    }


def _run_build(tmp_path, targets, files, pin="0.23.1rc1.dev1+gdeadbeef0"):
    repo = tmp_path / "repo"
    (repo / "sndr" / "engines" / "vllm" / "pins").mkdir(parents=True)
    tjson = tmp_path / "targets.json"
    pjson = tmp_path / "pristine.json"
    tjson.write_text(json.dumps(
        {"pin": pin, "genesis_pin": "g", "targets": targets}))
    pjson.write_text(json.dumps({"pin": pin, "files": files}))
    r = subprocess.run(
        [sys.executable, str(BUILD), str(tjson), str(pjson), str(repo), pin, "g"],
        capture_output=True, text=True)
    pin_dir = repo / "sndr" / "engines" / "vllm" / "pins" / "0.23.1_deadbeef0"
    return r, pin_dir


def test_build_writes_dependency_breakage_section(tmp_path):
    targets = [_target("PA", "s1", "f.py", "ANCHOR_A", "R")]
    r, pin_dir = _run_build(tmp_path, targets, {"f.py": "x ANCHOR_A y"})
    assert r.returncode == 0, r.stdout + r.stderr
    rej = json.loads((pin_dir / "drift.rej.json").read_text())
    # section present + populated from the LIVE registry (PN353A->PN399)
    db = rej["dependency_breakage"]
    assert set(db) == {"high_count", "medium_count", "edges"}
    edge = next((e for e in db["edges"]
                 if e["retired"] == "PN353A" and e["dependent"] == "PN399"), None)
    assert edge is not None and edge["severity"] == "HIGH"
    # build stdout surfaces the HIGH WARN block
    assert "dependency_breakage: HIGH=" in r.stdout
    assert "PN353A->PN399" in r.stdout


def test_summarize_prints_retire_broken_block(tmp_path):
    targets = [_target("PA", "s1", "f.py", "ANCHOR_A", "R")]
    _, pin_dir = _run_build(tmp_path, targets, {"f.py": "x ANCHOR_A y"})
    r = subprocess.run([sys.executable, str(SUMMARIZE), str(pin_dir)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "retire-broken dependents" in r.stdout
    assert "PN353A -> PN399" in r.stdout
    assert "HIGH" in r.stdout


# ─── bump_preflight gate ───────────────────────────────────────────────────

def _pin_dir(base, name, anchors, rej):
    d = base / name
    d.mkdir(parents=True)
    (d / "anchors.json").write_text(json.dumps(anchors))
    (d / "drift.rej.json").write_text(json.dumps(rej))
    return d


def _live_breakage():
    from sndr.engines.vllm.retire_impact import detect_on_live_registry
    return detect_on_live_registry().to_dict()


def test_preflight_fails_when_perf_dependent_breaks(tmp_path):
    """PN353A ok->retired across pins, PN399 ok->anchor_drift -> exit 1."""
    sys.path.insert(0, str(REPO))
    old = _pin_dir(
        tmp_path, "old",
        {"pins": {"vllm": "dev148"}, "files": {"tq.py": {"patches": {
            "PN353A": {"anchors": {"s1": {"byte_offset": 1}}},
            "PN399": {"anchors": {
                "pn399_pn353a_decode_reserve_remove": {"byte_offset": 2}}},
        }}}},
        {"rejected": []})
    new = _pin_dir(
        tmp_path, "new",
        {"pins": {"vllm": "dev301"}, "files": {"tq.py": {"patches": {}}}},
        {"rejected": [
            {"key": "PN353A::s1", "status": "retired"},
            {"key": "PN399::pn399_pn353a_decode_reserve_remove",
             "status": "anchor_drift"},
        ], "dependency_breakage": _live_breakage()})
    r = subprocess.run([sys.executable, str(PREFLIGHT), str(old), str(new)],
                       capture_output=True, text=True)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "RESULT: FAIL" in r.stdout
    # (a) newly retired, (b) PN353A->PN399 HIGH, (c) PN399 perf-landmine, (d) A/B
    assert "PN353A" in r.stdout
    assert "PN353A -> PN399" in r.stdout or "PN399" in r.stdout
    assert "perf landmines" in r.stdout
    assert "iron-rule #9" in r.stdout
    assert "genesis_bench_suite.py" in r.stdout


def test_preflight_passes_on_clean_bump(tmp_path):
    """No newly-retired patch + no breakage -> exit 0."""
    anchors = {"pins": {"vllm": "dev148"}, "files": {"f.py": {"patches": {
        "PA": {"anchors": {"s1": {"byte_offset": 1}}}}}}}
    rej = {"rejected": [],
           "dependency_breakage": {"high_count": 0, "medium_count": 0,
                                   "edges": []}}
    old = _pin_dir(tmp_path, "old", anchors, rej)
    new_anchors = {"pins": {"vllm": "dev301"}, "files": {"f.py": {"patches": {
        "PA": {"anchors": {"s1": {"byte_offset": 9}}}}}}}
    new = _pin_dir(tmp_path, "new", new_anchors, rej)
    r = subprocess.run([sys.executable, str(PREFLIGHT), str(old), str(new)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "RESULT: PASS" in r.stdout


def test_preflight_usage_error_exit_2(tmp_path):
    r = subprocess.run([sys.executable, str(PREFLIGHT), str(tmp_path / "only")],
                       capture_output=True, text=True)
    assert r.returncode == 2
