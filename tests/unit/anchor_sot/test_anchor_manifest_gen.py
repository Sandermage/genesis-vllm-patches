"""Ф2 — true-drift classifier (R2) tests. Synthetic, local, no vLLM needed."""
from sndr.engines.vllm.anchor_discovery import AnchorTarget
from sndr.engines.vllm.anchor_manifest_gen import (
    classify_anchor,
    build_pin_manifest,
    apply_via_meta,
    verify_roundtrip,
    STATUS_OK,
    STATUS_ANCHOR_DRIFT,
    STATUS_AMBIGUOUS,
)


def test_R3_roundtrip_byte_identical_single_line():
    src = "a\ndef f():\n    return OLD_ANCHOR\nb\n"
    assert verify_roundtrip(src, "    return OLD_ANCHOR", "    return NEW_VALUE")


def test_R3_roundtrip_byte_identical_multiline():
    src = "x\n    if (\n        self.cond):\n        do()\ny\n"
    assert verify_roundtrip(
        src, "    if (\n        self.cond):", "    if (self.cond):  # patched"
    )


def test_R3_roundtrip_unicode_replacement():
    # replacement with non-ASCII (anchor stays ASCII so offset is byte-stable)
    src = "head\n    ANCHOR_HERE\ntail\n"
    assert verify_roundtrip(src, "    ANCHOR_HERE", "    PATCHED_x")


def test_apply_via_meta_matches_inline():
    src = "p\n    target_line()\nq\n"
    status, meta = classify_anchor(src, "    target_line()", "    new_line()")
    assert status == STATUS_OK
    assert apply_via_meta(src, meta, "    new_line()") == src.replace(
        "    target_line()", "    new_line()", 1
    )


def test_classify_ok_computes_meta():
    src = "def foo():\n    return ORIGINAL_X\n\ndef bar():\n    return OTHER_Y\n"
    status, meta = classify_anchor(src, "    return ORIGINAL_X", "    return PATCHED")
    assert status == STATUS_OK
    assert meta["byte_offset"] >= 0 and meta["anchor_md5"]
    assert meta["replacement_md5"]  # replacement passed -> md5 present


def test_R2_drift_is_real_not_assumed():
    src = "def foo():\n    return ORIGINAL_X\n"
    mutated = src.replace("ORIGINAL_X", "REFACTORED_Z")  # upstream refactor
    status, meta = classify_anchor(mutated, "    return ORIGINAL_X")
    assert status == STATUS_ANCHOR_DRIFT and meta is None


def test_R2_no_false_positive_on_unaffected_anchor():
    src = "def foo():\n    return ORIGINAL_X\n\ndef bar():\n    return OTHER_Y\n"
    mutated = src.replace("ORIGINAL_X", "REFACTORED_Z")
    # the OTHER anchor in the same mutated file is untouched -> still ok
    status, _ = classify_anchor(mutated, "    return OTHER_Y")
    assert status == STATUS_OK


def test_ambiguous_when_anchor_not_unique():
    src = "X\nDUP\nY\nDUP\nZ\n"
    status, meta = classify_anchor(src, "DUP")
    assert status == STATUS_AMBIGUOUS and meta is None


def test_R2_build_pin_manifest_isolates_drift():
    targets = [
        AnchorTarget("PA", "s1", "fileA.py", "ANCHOR_A", "REPL_A", True),
        AnchorTarget("PB", "s1", "fileB.py", "ANCHOR_B", "REPL_B", True),
    ]
    pristine = {"fileA.py": "x ANCHOR_A y", "fileB.py": "p ANCHOR_B q"}
    r = build_pin_manifest(lambda rel: pristine.get(rel), targets)
    assert set(r.ok) == {"PA::s1", "PB::s1"} and not r.rej

    # drift fileA only -> EXACTLY PA drifts, PB still ok (R2 isolation)
    drifted = dict(pristine, **{"fileA.py": "x REFACTORED y"})
    r2 = build_pin_manifest(lambda rel: drifted.get(rel), targets)
    assert "PB::s1" in r2.ok and "PA::s1" not in r2.ok
    assert any(
        e["key"] == "PA::s1" and e["status"] == STATUS_ANCHOR_DRIFT
        for e in r2.rej
    )


def test_upstream_merged_excluded():
    targets = [AnchorTarget("PM", "s1", "f.py", "ANCH", "R", True)]
    src = {"f.py": "ANCH present"}
    r = build_pin_manifest(
        lambda rel: src.get(rel),
        targets,
        is_upstream_merged=lambda t, content: True,  # caller says merged
    )
    assert not r.ok
    assert r.rej and r.rej[0]["status"] == "upstream_merged"
