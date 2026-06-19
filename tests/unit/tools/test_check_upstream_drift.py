# SPDX-License-Identifier: Apache-2.0
"""TDD contract for the redesigned ``tools/check_upstream_drift.py``.

These tests were written BEFORE the redesign (per CLAUDE.md TDD rule) and
encode the acceptance criteria for the 2026-06-13 drift-tool rewrite:

  1. FALSE-POSITIVE gone — the 6 sampled patches (PN390, PN394, P67,
     PN353A, P28, P3) report clean/ok against the PRISTINE dev148 tree,
     NOT drift.
  2. PRISTINE-GUARD works — running against the PATCHED tree (carrying
     Genesis wiring markers) exits 2 with a clear "patched tree, not
     pristine" message instead of emitting false drifts.
  3. DISJOINTNESS INVARIANT — the set of patch IDs reported as genuine
     ``anchor_drift`` / ``import_drift`` against the PRISTINE tree is
     DISJOINT from the live applied-set fixture.
  4. FALSE-NEGATIVE caught — a fixture where the import-target parser
     class is absent from all candidate paths → ``import_drift``; a
     fixture corrupting an inline-builder patcher's anchor →
     ``anchor_drift``.

The two ground-truth trees are controller-prepared under /tmp:
  - PRISTINE (unpatched upstream @ b4c80ec0f): /tmp/vllm_pristine_b4c80ec0f
  - PATCHED  (deployed, Genesis markers in-place): /tmp/vllm_dev148_root

Tests that need those trees skip cleanly when /tmp is not populated (so
the suite stays green on CI runners that don't carry the fixtures), but
the local acceptance run exercises them fully.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "check_upstream_drift.py"
APPLIED_SET_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "dev148_applied_set.json"

PRISTINE_TREE = Path("/tmp/vllm_pristine_b4c80ec0f")
PATCHED_TREE = Path("/tmp/vllm_dev148_root")
PIN = "0.23.1rc1.dev148+gb4c80ec0f"

# Statuses the tool assigns per anchor result. The two "genuine drift"
# statuses are the ONLY ones that drive a non-zero exit.
DRIFT_STATUSES = {"anchor_drift", "import_drift"}


def _load_tool():
    """Import the tool module by path (it lives under tools/, not a package)."""
    spec = importlib.util.spec_from_file_location(
        "check_upstream_drift_under_test", TOOL_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _pristine_available() -> bool:
    return (PRISTINE_TREE / "vllm").is_dir()


def _patched_available() -> bool:
    return (PATCHED_TREE / "vllm").is_dir()


needs_pristine = pytest.mark.skipif(
    not _pristine_available(),
    reason="pristine dev148 tree not present under /tmp",
)
needs_patched = pytest.mark.skipif(
    not _patched_available(),
    reason="patched dev148 tree not present under /tmp",
)


# ─── Tool surface contract ───────────────────────────────────────────────


def test_tool_exposes_run_drift_check():
    """The redesign must expose an importable entry point returning a
    structured report (not just a CLI). This is what the disjointness /
    false-negative tests consume without re-parsing stdout."""
    mod = _load_tool()
    assert hasattr(mod, "run_drift_check"), (
        "redesigned tool must expose run_drift_check(tree_root, ...) -> report dict"
    )
    assert hasattr(mod, "main")


def test_genesis_marker_probe_constant_present():
    """The pristine-guard probes for the Genesis wiring-marker string."""
    mod = _load_tool()
    # The probe constant must contain the canonical wiring-marker prefix
    # that TextPatcher.apply() Layer 6 writes.
    probe = getattr(mod, "GENESIS_WIRING_MARKER_PROBE", None)
    assert probe is not None
    assert "Genesis wiring marker" in probe


# ─── Acceptance 2: pristine guard ────────────────────────────────────────


@needs_patched
def test_patched_tree_rejected_with_exit_2():
    """Running against the PATCHED tree must exit 2 (invocation error)
    with a clear 'patched tree, not pristine' message — NOT emit drift."""
    mod = _load_tool()
    rc = mod.main(["check_upstream_drift.py", str(PATCHED_TREE)])
    assert rc == 2, f"expected exit 2 on patched tree, got {rc}"


@needs_patched
def test_patched_tree_guard_message(capsys):
    mod = _load_tool()
    mod.main(["check_upstream_drift.py", str(PATCHED_TREE)])
    err = capsys.readouterr().err.lower()
    assert "pristine" in err or "patched" in err, (
        "guard must explain the tree is patched, not pristine"
    )


@needs_pristine
def test_pristine_tree_passes_guard():
    """The pristine tree must NOT be rejected by the guard (no markers)."""
    mod = _load_tool()
    report = mod.run_drift_check(PRISTINE_TREE)
    # A real report (not a guard rejection) carries the anchors section.
    assert "anchors" in report


# ─── Acceptance 1: false-positives gone ──────────────────────────────────


SAMPLED_FALSE_POSITIVES = ["PN390", "PN394", "P67", "PN353A", "P28", "P3"]


@needs_pristine
@pytest.mark.parametrize("patch_id", SAMPLED_FALSE_POSITIVES)
def test_sampled_false_positive_is_clean(patch_id):
    """Each sampled FP must report a non-drift status against PRISTINE."""
    mod = _load_tool()
    report = mod.run_drift_check(PRISTINE_TREE)
    anchors = report["anchors"]
    assert patch_id in anchors, f"{patch_id} missing from report"
    status = anchors[patch_id]["status"]
    assert status not in DRIFT_STATUSES, (
        f"{patch_id} falsely reported drift: status={status} "
        f"detail={anchors[patch_id]}"
    )


# ─── Acceptance 3: disjointness invariant ────────────────────────────────


def _applied_set() -> set[str]:
    data = json.loads(APPLIED_SET_FIXTURE.read_text())
    return set(data["applied"])


def test_applied_set_fixture_loads():
    s = _applied_set()
    assert len(s) >= 90
    assert "P67" in s
    assert "PN390" in s


@needs_pristine
def test_drift_disjoint_from_applied_set():
    """The structural encoding of the meta-finding: a patch that applies
    cleanly at runtime (in the applied-set) cannot have drifted anchors
    against the pristine tree it was authored for. The tool's genuine-drift
    set MUST be disjoint from the applied-set."""
    mod = _load_tool()
    report = mod.run_drift_check(PRISTINE_TREE)
    genuine_drift = {
        pid
        for pid, r in report["anchors"].items()
        if r["status"] in DRIFT_STATUSES
    }
    overlap = genuine_drift & _applied_set()
    assert not overlap, (
        "DISJOINTNESS VIOLATION — these applied-set patches were reported "
        f"as genuine drift against pristine: {sorted(overlap)}"
    )


# ─── Acceptance 4: false-negatives caught ────────────────────────────────


def test_import_wiring_drift_detected(tmp_path):
    """A pristine-style tree where the import-target parser class is absent
    from ALL candidate module paths → PN287-class import_drift."""
    mod = _load_tool()
    assert hasattr(mod, "check_import_wiring"), (
        "tool must expose check_import_wiring(spec_module, tree_root) for "
        "import-based monkey-patch wiring (PN287 class)"
    )
    # Build a fake tree with vllm/ but WITHOUT the qwen3coder/xml parser
    # modules — so every candidate path fails to resolve.
    (tmp_path / "vllm").mkdir()
    import importlib
    pn287 = importlib.import_module(
        "sndr.engines.vllm.patches.tool_parsing."
        "pn287_qwen3coder_args_validity_observer"
    )
    result = mod.check_import_wiring(pn287, tmp_path)
    assert result is not None
    assert result["status"] == "import_drift", (
        f"expected import_drift when parser class is unresolvable, got {result}"
    )


def test_pn347_exposes_make_patcher_for_drift_shim():
    """PN347 (inline-builder class) must expose _make_patcher_for_drift()
    so the static drift tool can build its otherwise-inline TextPatcher."""
    import importlib
    pn347 = importlib.import_module(
        "sndr.engines.vllm.patches.quantization.marlin."
        "pn347_marlin_fp8_nk_correctness"
    )
    assert hasattr(pn347, "_make_patcher_for_drift")


def test_inline_builder_anchor_drift_detected(tmp_path):
    """A clean inline-built patcher reports OK; corrupting its anchor flips
    it to anchor_drift. Uses a synthetic file so the OK→drift transition is
    caused unambiguously by the corruption (not a pre-existing absence)."""
    mod = _load_tool()
    from sndr.kernel import TextPatch, TextPatcher

    (tmp_path / "vllm").mkdir()
    target = tmp_path / "vllm" / "fake_kernel.py"
    target.write_text(
        "def f():\n    UNIQUE_ANCHOR_TOKEN = 1\n    return UNIQUE_ANCHOR_TOKEN\n"
    )

    def build(anchor: str) -> TextPatcher:
        return TextPatcher(
            patch_name="inline-builder fixture",
            target_file=str(target),
            marker="[Genesis FIXTURE marker]",
            sub_patches=[TextPatch(
                name="fixture_sub", anchor=anchor,
                replacement="REPLACED", required=True,
            )],
        )

    clean = mod.check_patcher_anchors(build("UNIQUE_ANCHOR_TOKEN = 1"), tmp_path)
    assert clean["status"] == "ok", clean

    drifted = mod.check_patcher_anchors(
        build("UNIQUE_ANCHOR_TOKEN = 1\n# CORRUPTED-NEVER-MATCHES"), tmp_path,
    )
    assert drifted["status"] == "anchor_drift", drifted


@needs_pristine
def test_pn347_inline_builder_reaches_anchor_scan():
    """End-to-end: the shim lets the tool BUILD PN347's inline patcher and
    SCAN its anchor against the pristine tree (no longer silently dropped)."""
    mod = _load_tool()
    import importlib
    pn347 = importlib.import_module(
        "sndr.engines.vllm.patches.quantization.marlin."
        "pn347_marlin_fp8_nk_correctness"
    )
    from sndr.engines.vllm.detection import guards
    orig = guards.vllm_install_root
    try:
        guards.vllm_install_root = lambda: str(PRISTINE_TREE / "vllm")
        patcher = pn347._make_patcher_for_drift()
        assert patcher is not None, "PN347 target should resolve in pristine tree"
        result = mod.check_patcher_anchors(patcher, PRISTINE_TREE)
        assert result["status"] is not None  # reachable & scanned, not dropped
    finally:
        guards.vllm_install_root = orig


# ─── Version-gate parity ─────────────────────────────────────────────────


@needs_pristine
def test_version_gated_patch_not_counted_as_drift():
    """PN287 (range <0.23.0) and PN347 (range <0.22.1rc1.dev491) are
    version-gated OUT at the dev148 pin — they must be classified
    version_gated_skip, NOT drift."""
    mod = _load_tool()
    report = mod.run_drift_check(PRISTINE_TREE, expect_pin=PIN)
    anchors = report["anchors"]
    for pid in ("PN287", "PN347"):
        if pid in anchors:
            assert anchors[pid]["status"] not in DRIFT_STATUSES, (
                f"{pid} is version-gated out at {PIN}; must not be drift "
                f"(got {anchors[pid]['status']})"
            )


# ─── expect-pin guard ────────────────────────────────────────────────────


def test_read_tree_pin_resolver():
    """read_tree_pin parses the concrete pin from a tree's _version.py."""
    mod = _load_tool()
    assert hasattr(mod, "read_tree_pin")
    if _patched_available():
        # Patched tree carries a concrete _version.py with the dev148 pin.
        assert mod.read_tree_pin(PATCHED_TREE) == PIN


@needs_patched
def test_expect_pin_mismatch_rejected():
    """--expect-pin with a mismatching version against a tree that DOES
    declare a concrete pin exits 2, so the gate can never silently
    drift-check the wrong pin/tree. The patched tree declares dev148; the
    pristine guard would reject it first, so we disable the pristine guard
    here to isolate the pin-mismatch path."""
    mod = _load_tool()
    with pytest.raises(mod.PristineGuardError):
        mod.run_drift_check(
            PATCHED_TREE,
            expect_pin="0.99.0-wrong-pin",
            enforce_pristine=False,
        )


@needs_pristine
def test_expect_pin_used_when_tree_has_no_version():
    """A pristine `git clone` may lack a generated _version.py. --expect-pin
    then doubles as the operator-asserted gating pin (no rejection)."""
    mod = _load_tool()
    # Pristine tree has no concrete _version.py — the operator asserts the
    # pin. The run must succeed and use that pin for version-gating.
    report = mod.run_drift_check(PRISTINE_TREE, expect_pin=PIN)
    assert report["tree_pin"] == PIN
