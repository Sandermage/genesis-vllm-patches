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
    assert set(db) == {"high_count", "high_mitigated_count",
                       "high_unmitigated_count", "medium_count", "edges"}
    edge = next((e for e in db["edges"]
                 if e["retired"] == "PN353A" and e["dependent"] == "PN399"), None)
    assert edge is not None and edge["severity"] == "HIGH"
    # the live PN353A->PN399 HIGH is MITIGATED (PN399 native-form C2 sibling)
    assert edge["mitigated"] is True
    assert db["high_mitigated_count"] == 1 and db["high_unmitigated_count"] == 0
    # build stdout surfaces the HIGH line + the MITIGATED note (not the WARN)
    assert "dependency_breakage: HIGH=" in r.stdout
    assert "PN353A->PN399" in r.stdout
    assert "MITIGATED" in r.stdout


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


def test_preflight_passes_when_live_pn399_break_is_mitigated(tmp_path):
    """APPLY-STATE-AWARE (the dev148->dev424 reality): PN353A ok->retired, the
    PN399 PN353A-form sibling ok->anchor_drift across pins — BUT on the live
    registry PN353A->PN399 is now MITIGATED (PN399 carries the native-form C2
    sibling that applies on the new pin). The operator already handled it, so the
    gate PASSES (exit 0), still LISTING the edge as HIGH-MITIGATED.

    (Before the apply-state-aware refinement this scenario exit-1'd — the
    conservative false-fail this change removes. The genuine FAIL path is covered
    by ``test_preflight_fails_on_synthetic_high_anchor_break`` and
    ``test_preflight_fails_on_unmitigated_high_even_with_a_mitigated_sibling``.)"""
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
        {"pins": {"vllm": "dev424"}, "files": {"tq.py": {"patches": {
            "PN399": {"anchors": {
                "pn399_native_decode_reserve_remove": {"byte_offset": 3}}},
        }}}},
        {"rejected": [
            {"key": "PN353A::s1", "status": "retired"},
            {"key": "PN399::pn399_pn353a_decode_reserve_remove",
             "status": "anchor_drift"},
        ], "dependency_breakage": _live_breakage()})
    r = subprocess.run([sys.executable, str(PREFLIGHT), str(old), str(new)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "RESULT: PASS" in r.stdout
    # the live PN353A->PN399 edge is present, marked MITIGATED, not a fail.
    assert "PN353A" in r.stdout and "PN399" in r.stdout
    assert "MITIGATED" in r.stdout


def test_preflight_fails_on_synthetic_high_anchor_break(tmp_path):
    """Registry-independent: a synthetic HIGH anchor-break perf dependent in the
    new pin's dependency_breakage section makes preflight exit non-zero. Locks
    the gate contract to ``severity == HIGH`` (the anchor-break tier) regardless
    of the live registry's current edge set."""
    old = _pin_dir(
        tmp_path, "old",
        {"pins": {"vllm": "dev148"}, "files": {"f.py": {"patches": {
            "SYNDEP": {"anchors": {"a1": {"byte_offset": 1}}}}}}},
        {"rejected": []})
    new = _pin_dir(
        tmp_path, "new",
        {"pins": {"vllm": "dev301"}, "files": {"f.py": {"patches": {}}}},
        {"rejected": [{"key": "SYNRET::s1", "status": "retired"}],
         "dependency_breakage": {
             "high_count": 1, "medium_count": 0,
             "edges": [{
                 "retired": "SYNRET", "retired_reason": "retired",
                 "dependent": "SYNDEP", "severity": "HIGH",
                 "via": ["anchor_name", "anchor_text"],
                 "dependent_category": "kernel_perf",
                 "dependent_lifecycle": "experimental",
                 "dependent_default_on": True,
                 "detail": ("retiring SYNRET (retired) breaks dependent SYNDEP "
                            "(SYNDEP anchor 'syndep_synret_*') — SYNDEP will "
                            "skip/no-op (anchor targets the retired patch's "
                            "emitted bytes — physically no-ops (the PN399 "
                            "class))")}]}})
    r = subprocess.run([sys.executable, str(PREFLIGHT), str(old), str(new)],
                       capture_output=True, text=True)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "RESULT: FAIL" in r.stdout
    assert "SYNDEP" in r.stdout
    assert "physically no-ops" in r.stdout


def test_preflight_passes_on_mitigated_high(tmp_path):
    """APPLY-STATE-AWARE: a HIGH edge flagged ``mitigated`` (the dependent has a
    working fallback path independent of the retired id — the PN399 native-form
    sibling) must NOT fail the gate. It is still LISTED (as HIGH-MITIGATED) but the
    operator already handled it, so exit 0."""
    old = _pin_dir(
        tmp_path, "old",
        {"pins": {"vllm": "dev148"}, "files": {"f.py": {"patches": {
            "SYNDEP": {"anchors": {"a1": {"byte_offset": 1}}}}}}},
        {"rejected": []})
    new = _pin_dir(
        tmp_path, "new",
        {"pins": {"vllm": "dev424"}, "files": {"f.py": {"patches": {
            "SYNDEP": {"anchors": {"a_native": {"byte_offset": 5}}}}}}},
        {"rejected": [{"key": "SYNRET::s1", "status": "retired"}],
         "dependency_breakage": {
             "high_count": 1, "medium_count": 0,
             "edges": [{
                 "retired": "SYNRET", "retired_reason": "retired",
                 "dependent": "SYNDEP", "severity": "HIGH", "mitigated": True,
                 "via": ["anchor_name", "anchor_text"],
                 "dependent_category": "kernel_perf",
                 "dependent_lifecycle": "experimental",
                 "dependent_default_on": True,
                 "detail": ("retiring SYNRET (retired) breaks dependent SYNDEP "
                            "(SYNDEP anchor 'syndep_synret_*') — HIGH-MITIGATED: "
                            "SYNDEP has an alternative anchor sub-patch not "
                            "referencing SYNRET (working fallback path)")}]}})
    r = subprocess.run([sys.executable, str(PREFLIGHT), str(old), str(new)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "RESULT: PASS" in r.stdout
    # still listed (operator visibility), but marked mitigated, not a fail.
    assert "SYNDEP" in r.stdout
    assert "MITIGATED" in r.stdout


def test_preflight_fails_on_unmitigated_high_even_with_a_mitigated_sibling(tmp_path):
    """A genuinely-unmitigated HIGH (only path references the retired id, no
    fallback — the real PN399-incident class) STILL fails the gate, even when
    ANOTHER edge in the same report is mitigated. The mitigated edge is not enough
    to clear an unmitigated one."""
    old = _pin_dir(
        tmp_path, "old",
        {"pins": {"vllm": "dev148"}, "files": {"f.py": {"patches": {
            "MDEP": {"anchors": {"a1": {"byte_offset": 1}}},
            "UDEP": {"anchors": {"a2": {"byte_offset": 2}}}}}}},
        {"rejected": []})
    new = _pin_dir(
        tmp_path, "new",
        {"pins": {"vllm": "dev424"}, "files": {"f.py": {"patches": {}}}},
        {"rejected": [{"key": "MRET::s1", "status": "retired"},
                      {"key": "URET::s1", "status": "retired"}],
         "dependency_breakage": {
             "high_count": 2, "medium_count": 0,
             "edges": [
                 {"retired": "MRET", "retired_reason": "retired",
                  "dependent": "MDEP", "severity": "HIGH", "mitigated": True,
                  "via": ["anchor_name", "anchor_text"],
                  "dependent_category": "kernel_perf",
                  "dependent_lifecycle": "experimental",
                  "dependent_default_on": True,
                  "detail": "MDEP HIGH-MITIGATED (native-form fallback)"},
                 {"retired": "URET", "retired_reason": "retired",
                  "dependent": "UDEP", "severity": "HIGH", "mitigated": False,
                  "via": ["anchor_name", "anchor_text"],
                  "dependent_category": "kernel_perf",
                  "dependent_lifecycle": "experimental",
                  "dependent_default_on": True,
                  "detail": ("UDEP will skip/no-op — anchor targets the retired "
                             "patch's emitted bytes — physically no-ops (the "
                             "PN399 class)")}]}})
    r = subprocess.run([sys.executable, str(PREFLIGHT), str(old), str(new)],
                       capture_output=True, text=True)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "RESULT: FAIL" in r.stdout
    assert "UDEP" in r.stdout
    # the mitigated sibling is listed but does not trip the fail on its own.
    assert "MITIGATED" in r.stdout


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
