# SPDX-License-Identifier: Apache-2.0
"""Tests for ``tools/pin_preflight.py`` — the pin-bump preflight verdict engine.

TDD contract (written BEFORE the implementation): one test per verdict
class, plus a behavior-parity test pinning ``evaluate_layer5`` against
the real ``TextPatcher._apply_layer5_legacy`` (the engine mirrors its
semantics but NEVER calls ``.apply()``, which writes to disk).

Verdict classes under test:
    OK | DRIFT_ANCHOR | AMBIGUOUS_ANCHOR | DRIFT_FILE_MOVED |
    UPSTREAM_MERGED | SUB_UPSTREAM_MERGED | STALE_RESIDUE |
    UNBUILDABLE | IMPORT_FAIL | RUNTIME_BINDING | EXPECTED_ALTERNATE
Binding sub-verdicts:
    BINDING_OK | BINDING_FILE_MISSING | BINDING_SYMBOL_MISSING |
    BINDING_UNRESOLVED
Plus: SELF_COLLISION_RISK static lint, version-range gating,
upstream-marker pass, manifest staleness, exit-code semantics.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "pin_preflight.py"


def _import_tool():
    spec = importlib.util.spec_from_file_location("pin_preflight", TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pin_preflight"] = mod
    assert spec.loader is not None  # nosec
    spec.loader.exec_module(mod)
    return mod


def _write_module(tmp_path, name, source):
    """Materialize a synthetic wiring module importable by dotted path.

    Unique package per test — sys.modules caches package __path__, so
    reusing one name across per-test tmp_paths goes stale.
    """
    pkg = f"synthetic_wiring_{name}"
    d = tmp_path / pkg
    d.mkdir(exist_ok=True)
    (d / "__init__.py").touch()
    (d / f"{name}.py").write_text(textwrap.dedent(source))
    if str(tmp_path) not in sys.path:
        sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()
    return f"{pkg}.{name}"


class _EnvSpecStub:
    """Minimal PatchSpec stand-in carrying only env_flag (the field
    evaluate_module's v1.2 env-forced retry reads)."""

    def __init__(self, env_flag):
        self.env_flag = env_flag


@pytest.fixture(scope="module")
def pf():
    return _import_tool()


@pytest.fixture()
def kernel():
    from sndr.kernel.text_patch import TextPatch, TextPatcher
    return TextPatch, TextPatcher


@pytest.fixture()
def mini_tree(tmp_path: Path) -> Path:
    """Fixture mini candidate tree — looks like an extracted vllm package
    (root contains v1/, the startup validation contract)."""
    root = tmp_path / "vllm"
    (root / "v1" / "attention" / "backends").mkdir(parents=True)
    (root / "v1" / "sample").mkdir(parents=True)
    (root / "entrypoints").mkdir(parents=True)
    # File 1: clean target — anchor appears exactly once.
    (root / "v1" / "sample" / "sampler.py").write_text(
        "import torch\n"
        "def sample(logits):\n"
        "    probs = logits.softmax(dim=-1)\n"
        "    return probs.argmax(dim=-1)\n"
    )
    # File 2: ambiguous target — anchor appears twice.
    (root / "v1" / "attention" / "backends" / "attn.py").write_text(
        "x = compute_thing()\n"
        "y = compute_thing()\n"
    )
    # File 3: upstream-merged target — carries a drift marker string.
    (root / "entrypoints" / "serving.py").write_text(
        "def serve():\n"
        "    use_native_streaming_fix()\n"
    )
    # File 4: a moved copy of sampler.py content under a new subdir
    # (the gdn/-split class) for moved-to candidate detection.
    (root / "v1" / "sample" / "rejection").mkdir(parents=True)
    (root / "v1" / "sample" / "rejection" / "sampler.py").write_text(
        "def sample(logits):\n"
        "    probs = logits.softmax(dim=-1)\n"
    )
    return root


def _patcher(kernel, target: Path, *, marker="GENESIS_TEST_MARKER_V1",
             anchor="    probs = logits.softmax(dim=-1)\n",
             replacement="    probs = logits.softmax(dim=-1).clamp_(0)\n",
             required=True, drift_markers=(), sub_kwargs=None,
             extra_subs=()):
    TextPatch, TextPatcher = kernel
    sub_kwargs = sub_kwargs or {}
    subs = [TextPatch(name="sub1", anchor=anchor, replacement=replacement,
                      required=required, **sub_kwargs)]
    subs.extend(extra_subs)
    return TextPatcher(
        patch_name="TEST fixture patcher",
        target_file=str(target),
        marker=marker,
        sub_patches=subs,
        upstream_drift_markers=list(drift_markers),
    )


# ─── evaluate_layer5 parity vs TextPatcher._apply_layer5_legacy ───────────


class TestLayer5Parity:
    """The engine's pure mirror must agree with the real Layer 5 legacy
    scan on every outcome class. The real method is pure (string in,
    string/skip out) so we can call it directly on fixture content."""

    def _both(self, pf, kernel, subs, content):
        from sndr.kernel.text_patch import TextPatcher, TextPatchResult
        real = TextPatcher(
            patch_name="parity", target_file="/nonexistent",
            marker="PARITY_MARKER", sub_patches=list(subs),
        )._apply_layer5_legacy(content)
        mine = pf.evaluate_layer5(subs, content)
        return real, mine, TextPatchResult

    def test_success_parity(self, pf, kernel):
        TextPatch, _ = kernel
        subs = [TextPatch(name="a", anchor="alpha\n", replacement="ALPHA\n")]
        real, mine, _ = self._both(pf, kernel, subs, "alpha\nbeta\n")
        modified, applied = real
        assert mine.status == "success"
        assert mine.modified == modified
        assert mine.applied == applied == ["a"]

    def test_required_anchor_missing_parity(self, pf, kernel):
        TextPatch, _ = kernel
        subs = [TextPatch(name="a", anchor="gone\n", replacement="x\n",
                          required=True)]
        real, mine, TPR = self._both(pf, kernel, subs, "alpha\n")
        assert real[0] == TPR.SKIPPED and real[1].reason == "required_anchor_missing"
        assert mine.status == "skipped"
        assert mine.reason == "required_anchor_missing"

    def test_ambiguous_anchor_parity(self, pf, kernel):
        TextPatch, _ = kernel
        subs = [TextPatch(name="a", anchor="dup\n", replacement="x\n")]
        real, mine, TPR = self._both(pf, kernel, subs, "dup\ndup\n")
        assert real[0] == TPR.SKIPPED and real[1].reason == "ambiguous_anchor"
        assert mine.status == "skipped"
        assert mine.reason == "ambiguous_anchor"

    def test_all_optional_missing_parity(self, pf, kernel):
        TextPatch, _ = kernel
        subs = [TextPatch(name="a", anchor="gone\n", replacement="x\n",
                          required=False)]
        real, mine, TPR = self._both(pf, kernel, subs, "alpha\n")
        assert real[0] == TPR.SKIPPED
        assert real[1].reason == "no_applicable_sub_patches"
        assert mine.status == "skipped"
        assert mine.reason == "no_applicable_sub_patches"

    def test_sub_merged_skip_silently_parity(self, pf, kernel):
        TextPatch, _ = kernel
        subs = [
            TextPatch(name="merged", anchor="alpha\n", replacement="A\n",
                      upstream_merged_markers=["native_impl"]),
            TextPatch(name="alive", anchor="beta\n", replacement="B\n"),
        ]
        content = "alpha\nbeta\nnative_impl\n"
        real, mine, _ = self._both(pf, kernel, subs, content)
        modified, applied = real
        assert applied == ["alive"]
        assert mine.applied == ["alive"]
        assert mine.modified == modified
        assert mine.sub_merged == [("merged", "native_impl")]

    def test_sub_merged_abort_bundle_parity(self, pf, kernel):
        TextPatch, _ = kernel
        subs = [TextPatch(name="merged", anchor="alpha\n", replacement="A\n",
                          upstream_merged_markers=["native_impl"],
                          on_upstream_merge="abort_bundle")]
        real, mine, TPR = self._both(pf, kernel, subs, "alpha\nnative_impl\n")
        assert real[0] == TPR.SKIPPED
        assert real[1].reason == "sub_upstream_merged_abort_bundle"
        assert mine.status == "skipped"
        assert mine.reason == "sub_upstream_merged_abort_bundle"

    def test_sequential_replacement_parity(self, pf, kernel):
        """Anchors are scanned against the progressively-modified text —
        a replacement may create or destroy a later anchor."""
        TextPatch, _ = kernel
        subs = [
            TextPatch(name="first", anchor="one\n", replacement="two\ntwo\n"),
            TextPatch(name="second", anchor="two\n", replacement="X\n"),
        ]
        content = "one\n"
        real, mine, TPR = self._both(pf, kernel, subs, content)
        # After sub "first", "two\n" appears twice → ambiguous for "second".
        assert real[0] == TPR.SKIPPED and real[1].reason == "ambiguous_anchor"
        assert mine.reason == "ambiguous_anchor"


# ─── per-patcher verdicts (read-only) ─────────────────────────────────────


class TestPatcherVerdicts:
    def test_verdict_ok(self, pf, kernel, mini_tree):
        p = _patcher(kernel, mini_tree / "v1" / "sample" / "sampler.py")
        row = pf.evaluate_patcher(p, mini_tree)
        assert row["verdict"] == pf.OK
        assert row["applied_subs"] == ["sub1"]

    def test_verdict_drift_anchor(self, pf, kernel, mini_tree):
        p = _patcher(kernel, mini_tree / "v1" / "sample" / "sampler.py",
                     anchor="this anchor does not exist\n")
        row = pf.evaluate_patcher(p, mini_tree)
        assert row["verdict"] == pf.DRIFT_ANCHOR

    def test_verdict_ambiguous_anchor(self, pf, kernel, mini_tree):
        p = _patcher(kernel,
                     mini_tree / "v1" / "attention" / "backends" / "attn.py",
                     anchor="compute_thing()")
        row = pf.evaluate_patcher(p, mini_tree)
        assert row["verdict"] == pf.AMBIGUOUS_ANCHOR

    def test_verdict_stale_residue(self, pf, kernel, mini_tree):
        target = mini_tree / "v1" / "sample" / "sampler.py"
        target.write_text(target.read_text() + "# GENESIS_TEST_MARKER_V1\n")
        p = _patcher(kernel, target)
        row = pf.evaluate_patcher(p, mini_tree)
        assert row["verdict"] == pf.STALE_RESIDUE

    def test_verdict_upstream_merged(self, pf, kernel, mini_tree):
        p = _patcher(kernel, mini_tree / "entrypoints" / "serving.py",
                     anchor="def serve():\n",
                     drift_markers=["use_native_streaming_fix"])
        row = pf.evaluate_patcher(p, mini_tree)
        assert row["verdict"] == pf.UPSTREAM_MERGED
        assert "use_native_streaming_fix" in row["detail"]

    def test_verdict_sub_upstream_merged(self, pf, kernel, mini_tree):
        from sndr.kernel.text_patch import TextPatch
        target = mini_tree / "entrypoints" / "serving.py"
        p = _patcher(
            kernel, target,
            anchor="def serve():\n",
            replacement="def serve():  # patched\n",
            sub_kwargs={"upstream_merged_markers": ["use_native_streaming_fix"]},
            extra_subs=[TextPatch(name="alive",
                                  anchor="    use_native_streaming_fix()\n",
                                  replacement="    pass\n")],
        )
        row = pf.evaluate_patcher(p, mini_tree)
        assert row["verdict"] == pf.SUB_UPSTREAM_MERGED
        assert row["sub_merged"] == [["sub1", "use_native_streaming_fix"]]

    def test_verdict_drift_file_moved_with_candidates(self, pf, kernel, mini_tree):
        gone = mini_tree / "v1" / "sample" / "old_location.py"  # never created
        p = _patcher(kernel, gone)
        row = pf.evaluate_patcher(p, mini_tree)
        assert row["verdict"] == pf.DRIFT_FILE_MOVED
        # Anchor's most distinctive (longest) line exists in two other files.
        cands = row["moved_to_candidates"]
        assert any("v1/sample/sampler.py" in c for c in cands)
        assert any("v1/sample/rejection/sampler.py" in c for c in cands)
        assert len(cands) <= 3

    def test_moved_candidates_stem_rename(self, pf, mini_tree):
        """The gdn/-split class: gdn_linear_attn.py was split into
        gdn/{qwen,olmo,kimi}_gdn_linear_attn.py — basename changed but
        the old stem survives as a substring of the new filenames."""
        d = mini_tree / "model_executor" / "layers" / "mamba" / "gdn"
        d.mkdir(parents=True)
        (d / "qwen_gdn_linear_attn.py").write_text("# split out\n")
        (d / "olmo_gdn_linear_attn.py").write_text("# split out\n")
        # Decoy: short generic stem ("linear" ⊂ "gdn_linear_attn") must
        # NOT match — both stems need >= 8 chars for substring matching.
        (mini_tree / "model_executor" / "layers" / "linear.py").write_text(
            "# decoy\n")
        cands = pf.find_moved_candidates(
            mini_tree,
            missing_rel="model_executor/layers/mamba/gdn_linear_attn.py")
        assert any("qwen_gdn_linear_attn.py" in c for c in cands)
        assert any("olmo_gdn_linear_attn.py" in c for c in cands)
        assert not any(c.endswith("/linear.py") for c in cands)


# ─── SELF_COLLISION_RISK lint (the PN369 class) ───────────────────────────


class TestSelfCollisionLint:
    def test_marker_in_replacement_flagged(self, pf, kernel, mini_tree):
        p = _patcher(kernel, mini_tree / "v1" / "sample" / "sampler.py",
                     replacement="    probs = relaxed_topk(logits)\n",
                     drift_markers=["relaxed_topk"])
        findings = pf.self_collision_findings(p)
        assert len(findings) == 1
        assert findings[0]["marker"] == "relaxed_topk"
        assert findings[0]["collides_with"] == "replacement"
        assert findings[0]["defended"] is False

    def test_genesis_prefixed_marker_flagged_as_defended(self, pf, kernel, mini_tree):
        """'[Genesis PNxxx' self-reference markers are flagged but tagged
        defended=True — PN353A-style custom apply() skips them by
        convention; stock TextPatcher.apply() does NOT, so they still
        appear in the lint output."""
        p = _patcher(kernel, mini_tree / "v1" / "sample" / "sampler.py",
                     replacement="    # [Genesis PN353A backport]\n    pass\n",
                     drift_markers=["[Genesis PN353A"])
        findings = pf.self_collision_findings(p)
        assert len(findings) == 1
        assert findings[0]["defended"] is True

    def test_clean_patcher_no_findings(self, pf, kernel, mini_tree):
        p = _patcher(kernel, mini_tree / "v1" / "sample" / "sampler.py",
                     drift_markers=["some_upstream_symbol"])
        assert pf.self_collision_findings(p) == []

    def test_marker_field_collision(self, pf, kernel, mini_tree):
        p = _patcher(kernel, mini_tree / "v1" / "sample" / "sampler.py",
                     marker="genesis_pn369_relaxed_ok",
                     drift_markers=["genesis_pn369_relaxed_ok"])
        findings = pf.self_collision_findings(p)
        assert findings and findings[0]["collides_with"] == "marker"


# ─── UNBUILDABLE / IMPORT_FAIL / module evaluation ────────────────────────


class TestModuleEvaluation:
    def test_unfillable_params_detected(self, pf):
        def builder_with_required(backend, threshold=0.5):  # noqa: ARG001
            return None

        def builder_clean(threshold=0.5):  # noqa: ARG001
            return None

        assert pf.unfillable_params(builder_with_required) == ["backend"]
        assert pf.unfillable_params(builder_clean) == []

    def test_import_fail_row(self, pf, mini_tree):
        rows = pf.evaluate_module(
            "sndr.engines.vllm.patches.no_such_module_xyz",
            ["PX1"], mini_tree,
        )
        assert len(rows) == 1
        assert rows[0]["verdict"] == pf.IMPORT_FAIL

    def _write_module(self, tmp_path, name, source):
        return _write_module(tmp_path, name, source)

    def test_synthetic_module_ok_and_unbuildable(self, pf, mini_tree, tmp_path):
        modname = self._write_module(tmp_path, "ptest_ok", """
            from sndr.kernel.text_patch import TextPatch, TextPatcher

            TARGET = {target!r}

            def _make_patcher():
                return TextPatcher(
                    patch_name="SYN ok",
                    target_file=TARGET,
                    marker="SYN_MARKER",
                    sub_patches=[TextPatch(
                        name="s1",
                        anchor="    probs = logits.softmax(dim=-1)\\n",
                        replacement="    probs = x\\n",
                        required=True,
                    )],
                )

            def _make_param_patcher(backend):
                return None
        """.format(target=str(mini_tree / "v1" / "sample" / "sampler.py")))
        rows = pf.evaluate_module(modname, ["SYN1"], mini_tree)
        verdicts = sorted(r["verdict"] for r in rows)
        assert pf.OK in verdicts
        assert pf.UNBUILDABLE in verdicts

    def test_synthetic_module_builder_returns_none(self, pf, mini_tree, tmp_path):
        modname = self._write_module(tmp_path, "ptest_none", """
            from sndr.engines.vllm.detection.guards import resolve_vllm_file

            def _make_patcher():
                target = resolve_vllm_file("v1/sample/no_longer_here.py")
                if target is None:
                    return None
                raise AssertionError("unreachable in this fixture")
        """)
        rows = pf.evaluate_module(modname, ["SYN2"], mini_tree)
        assert rows[0]["verdict"] == pf.DRIFT_FILE_MOVED

    def test_runtime_binding_module(self, pf, mini_tree, tmp_path):
        (mini_tree / "v1" / "worker").mkdir(parents=True, exist_ok=True)
        (mini_tree / "v1" / "worker" / "gpu_input_batch.py").write_text(
            "class InputBatch:\n    pass\n"
        )
        modname = self._write_module(tmp_path, "ptest_binding", """
            import importlib

            def apply():
                from vllm.v1.worker.gpu_input_batch import InputBatch
                mod = importlib.import_module("vllm.v1.worker.missing_mod")
                return InputBatch, mod
        """)
        rows = pf.evaluate_module(modname, ["SYN3"], mini_tree)
        assert rows[0]["verdict"] == pf.RUNTIME_BINDING
        by_path = {b["module"]: b for b in rows[0]["bindings"]}
        assert by_path["vllm.v1.worker.gpu_input_batch"]["verdict"] == pf.BINDING_OK
        assert by_path["vllm.v1.worker.missing_mod"]["verdict"] == pf.BINDING_FILE_MISSING


# ─── binding extraction + checking ────────────────────────────────────────


class TestBindings:
    def test_extract_bindings_forms(self, pf):
        src = textwrap.dedent("""
            import importlib
            from vllm.v1.sample.sampler import Sampler, top_k
            import vllm.v1.core.sched_helper

            _CANDIDATE_MODULE_PATHS = [
                "vllm.model_executor.layers.mamba.gdn_linear_attn",
                "vllm.model_executor.layers.gdn_linear_attn",
            ]

            def apply():
                m = importlib.import_module("vllm.v1.attention.ops.helper")
                d = importlib.import_module(compute_name())  # dynamic
                return m, d
        """)
        bindings = pf.extract_bindings(src)
        pairs = {(b["module"], b["attr"]) for b in bindings if b["module"]}
        assert ("vllm.v1.sample.sampler", "Sampler") in pairs
        assert ("vllm.v1.sample.sampler", "top_k") in pairs
        assert ("vllm.v1.core.sched_helper", None) in pairs
        assert ("vllm.v1.attention.ops.helper", None) in pairs
        assert ("vllm.model_executor.layers.mamba.gdn_linear_attn", None) in pairs
        assert ("vllm.model_executor.layers.gdn_linear_attn", None) in pairs
        assert any(b["module"] is None for b in bindings)  # dynamic → unresolved

    def test_extract_bindings_skips_self_package(self, pf):
        src = "from vllm.sndr_core.env import Flags\nimport vllm._genesis.guards\n"
        assert pf.extract_bindings(src) == []

    def test_check_binding_verdicts(self, pf, mini_tree):
        (mini_tree / "v1" / "sample" / "sampler.py").write_text(
            "class Sampler:\n    pass\n\ndef top_k(x):\n    return x\n"
        )
        v, _ = pf.check_binding(mini_tree, "vllm.v1.sample.sampler", "Sampler")
        assert v == pf.BINDING_OK
        v, _ = pf.check_binding(mini_tree, "vllm.v1.sample.sampler", "nope_attr")
        assert v == pf.BINDING_SYMBOL_MISSING
        v, _ = pf.check_binding(mini_tree, "vllm.v1.sample.gone_mod", None)
        assert v == pf.BINDING_FILE_MISSING
        # Package __init__ resolution: vllm.v1.sample is a dir without
        # __init__.py in the fixture → missing; add one → OK.
        (mini_tree / "v1" / "sample" / "__init__.py").touch()
        v, _ = pf.check_binding(mini_tree, "vllm.v1.sample", None)
        assert v == pf.BINDING_OK
        # Submodule-as-attr: `from vllm.v1.sample import sampler`
        v, _ = pf.check_binding(mini_tree, "vllm.v1.sample", "sampler")
        assert v == pf.BINDING_OK


# ─── version-range gating ─────────────────────────────────────────────────


class TestVersionRange:
    def test_tuple_form_in_range(self, pf):
        ok, _ = pf.check_version_range(
            (">=0.20.0", "<0.23.0"), "0.22.1rc1.dev259+g303916e93")
        assert ok is True

    def test_tuple_form_out_of_range(self, pf):
        ok, _ = pf.check_version_range(
            (">=0.20.0", "<0.21.0"), "0.22.1rc1.dev259+g303916e93")
        assert ok is False

    def test_comma_string_form(self, pf):
        ok, _ = pf.check_version_range(
            "<0.20.2rc1.dev93", "0.22.1rc1.dev259+g303916e93")
        assert ok is False

    def test_unknown_candidate_version(self, pf):
        ok, _ = pf.check_version_range((">=0.20.0",), None)
        assert ok is None


# ─── upstream-marker pass / manifest staleness ────────────────────────────


class TestTreeWidePasses:
    def test_upstream_markers_injectable(self, pf, mini_tree):
        markers = {
            "PR_TEST_present": {
                "file": "entrypoints/serving.py",
                "marker": "use_native_streaming_fix",
            },
            "PR_TEST_known": {
                "file": "entrypoints/serving.py",
                "marker": "use_native_streaming_fix",
                "verified_in_main_2026_01_01": True,
            },
            "PR_TEST_absent": {
                "file": "entrypoints/serving.py",
                "marker": "definitely_not_in_fixture",
            },
            "PR_TEST_file_missing": {
                "file": "no/such/file.py",
                "marker": "whatever",
            },
        }
        results = {r["key"]: r for r in
                   pf.check_upstream_markers(mini_tree, markers=markers)}
        assert results["PR_TEST_present"]["newly_merged"] is True
        assert results["PR_TEST_known"]["newly_merged"] is False
        assert results["PR_TEST_known"]["currently_present"] is True
        assert results["PR_TEST_absent"]["currently_present"] is False
        assert results["PR_TEST_file_missing"]["currently_present"] is False

    def test_manifest_staleness(self, pf, tmp_path):
        mpath = tmp_path / "anchor_manifest.json"
        mpath.write_text(json.dumps(
            {"pins": {"vllm": "0.22.1rc1.dev259+g303916e93"}, "files": {}}))
        fresh = pf.manifest_staleness(
            "0.22.1rc1.dev259+g303916e93", manifest_path=mpath)
        assert fresh["stale"] is False
        stale = pf.manifest_staleness(
            "0.23.0rc1.dev1+gdeadbeef", manifest_path=mpath)
        assert stale["stale"] is True


# ─── spec scope filter ────────────────────────────────────────────────────


class _SpecStub:
    def __init__(self, lifecycle, implementation_status, apply_module="m"):
        self.lifecycle = lifecycle
        self.implementation_status = implementation_status
        self.apply_module = apply_module


class TestSpecScope:
    def test_retired_lifecycle_excluded_even_with_full_impl(self, pf):
        """Registry contradiction observed on the current pin: _archive
        entries carry lifecycle=retired WITH explicit
        implementation_status=full. They never apply at runtime, so the
        preflight must exclude them from the sweep."""
        assert pf.spec_in_scope(_SpecStub("retired", "full")) is False
        assert pf.spec_in_scope(_SpecStub("deprecated", "full")) is False

    def test_active_states_in_scope(self, pf):
        assert pf.spec_in_scope(_SpecStub("legacy", "full")) is True
        assert pf.spec_in_scope(_SpecStub("stable", "live")) is True
        assert pf.spec_in_scope(_SpecStub("experimental", "runtime_hook")) is True

    def test_non_apply_states_excluded(self, pf):
        assert pf.spec_in_scope(_SpecStub("stable", "metadata_only")) is False
        assert pf.spec_in_scope(_SpecStub("stable", "scaffold")) is False
        assert pf.spec_in_scope(
            _SpecStub("stable", "live", apply_module=None)) is False


# ─── summary / exit-code semantics ────────────────────────────────────────


class TestSummary:
    def test_actionable_logic(self, pf):
        rows = [
            {"verdict": pf.OK, "in_version_range": True},
            {"verdict": pf.RUNTIME_BINDING, "in_version_range": True,
             "binding_ok": True},
            # Out-of-range drift → dispatcher will skip this patch on the
            # candidate pin → NOT actionable.
            {"verdict": pf.DRIFT_ANCHOR, "in_version_range": False},
        ]
        assert pf.count_actionable(rows, markers=[]) == 0

    def test_in_range_drift_is_actionable(self, pf):
        rows = [{"verdict": pf.DRIFT_ANCHOR, "in_version_range": True}]
        assert pf.count_actionable(rows, markers=[]) == 1

    def test_binding_failure_is_actionable(self, pf):
        rows = [{"verdict": pf.RUNTIME_BINDING, "in_version_range": True,
                 "binding_ok": False}]
        assert pf.count_actionable(rows, markers=[]) == 1

    def test_newly_merged_marker_is_actionable(self, pf):
        markers = [{"key": "PR_X", "newly_merged": True}]
        assert pf.count_actionable([], markers=markers) == 1

    def test_main_rejects_bad_root(self, pf, tmp_path):
        rc = pf.main([str(tmp_path / "not_a_vllm_tree")])
        assert rc == 2


# ─── out-of-range enforcement semantics ───────────────────────────────────


class TestOutOfRangeDetail:
    """Decision rule 1 (dispatcher/decision.py should_apply): a truthy
    env flag on an opt-in patch OVERRIDES applies_to — including the
    version range. Only default_on=True patches are strictly gated.
    The report must say which is which, or the operator misreads
    "out of range" as "disabled" (live PROD counter-example: P67/P82
    applied on 0.22.1 despite <0.22.0 ranges, 2026-06-10).
    """

    REG = {
        "PX_STRICT": {"default_on": True, "env_flag": "GENESIS_ENABLE_PX",
                      "lifecycle": "experimental"},
        "PX_OPTIN": {"default_on": False, "env_flag": "GENESIS_ENABLE_PY",
                     "lifecycle": "experimental"},
    }

    def test_default_on_is_strictly_gated(self, pf):
        detail = pf.classify_out_of_range(["PX_STRICT"], registry=self.REG)
        assert detail[0]["patch_id"] == "PX_STRICT"
        assert detail[0]["enforcement"] == "STRICT_SKIP"
        assert detail[0]["default_on"] is True

    def test_opt_in_is_override_able(self, pf):
        detail = pf.classify_out_of_range(["PX_OPTIN"], registry=self.REG)
        assert detail[0]["enforcement"] == "ENV_OVERRIDE_POSSIBLE"
        assert detail[0]["env_flag"] == "GENESIS_ENABLE_PY"

    def test_unknown_pid_marked(self, pf):
        detail = pf.classify_out_of_range(["NOPE"], registry=self.REG)
        assert detail[0]["enforcement"] == "UNKNOWN_PATCH"


# ─── patch-chain awareness (P18B-on-PN119 class) ──────────────────────────


class TestChainedAnchor:
    """A patch whose anchors target ANOTHER patch's post-apply output
    (live example: P18B_TEXT tunes the GQA/MHA launcher blocks that
    PN119's replacement creates) must NOT read as upstream drift.
    Pristine evaluation of the dependent patch misses every anchor;
    the reclassify pass must flip it to CHAINED_ANCHOR naming the
    provider, and it must not count as actionable.
    """

    def _rows(self, pf, kernel, mini_tree):
        TextPatch, TextPatcher = kernel
        target = mini_tree / "v1" / "sample" / "sampler.py"
        provider = TextPatcher(
            patch_name="PROVIDER adds tuned block",
            target_file=str(target),
            marker="GENESIS_PROVIDER_MARKER",
            sub_patches=[TextPatch(
                name="prov_sub",
                anchor="    probs = logits.softmax(dim=-1)\n",
                replacement=("    probs = logits.softmax(dim=-1)\n"
                             "    tuned_block(num_warps=4)\n"),
            )],
        )
        dependent = TextPatcher(
            patch_name="DEPENDENT tunes provider output",
            target_file=str(target),
            marker="GENESIS_DEPENDENT_MARKER",
            sub_patches=[TextPatch(
                name="dep_sub",
                anchor="    tuned_block(num_warps=4)\n",
                replacement="    tuned_block(num_warps=8)\n",
            )],
        )
        r1 = pf.evaluate_patcher(provider, mini_tree)
        r1["patch_ids"] = ["PROV"]
        r2 = pf.evaluate_patcher(dependent, mini_tree)
        r2["patch_ids"] = ["DEP"]
        return [r1, r2]

    def test_dependent_reclassified(self, pf, kernel, mini_tree):
        rows = self._rows(pf, kernel, mini_tree)
        assert rows[1]["verdict"] == pf.DRIFT_ANCHOR  # pre-pass state
        n = pf.reclassify_chained(rows)
        assert n == 1
        assert rows[0]["verdict"] == pf.OK
        assert rows[1]["verdict"] == pf.CHAINED_ANCHOR
        assert rows[1]["chained_on"] == ["PROV"]

    def test_chained_not_actionable(self, pf, kernel, mini_tree):
        rows = self._rows(pf, kernel, mini_tree)
        pf.reclassify_chained(rows)
        assert pf.CHAINED_ANCHOR not in pf.ACTIONABLE_VERDICTS
        assert pf.count_actionable(rows, []) == 0

    def test_unprovided_anchor_stays_drift(self, pf, kernel, mini_tree):
        rows = self._rows(pf, kernel, mini_tree)
        # Break the chain: provider no longer emits what dependent needs.
        rows[0]["_replacement_blob"] = "something else entirely\n"
        pf.reclassify_chained(rows)
        assert rows[1]["verdict"] == pf.DRIFT_ANCHOR

    def test_private_keys_stripped(self, pf, kernel, mini_tree):
        rows = self._rows(pf, kernel, mini_tree)
        pf.reclassify_chained(rows)
        for r in rows:
            assert "_replacement_blob" not in r
            assert "_missing_anchors" not in r


# ─── md5-gated diff patches (PN119 class) ─────────────────────────────────


class TestMd5DiffPatch:
    """PN119-style patches (md5 gate + bundled unified diff, applied via
    `patch`) expose no TextPatcher builder and previously fell through
    to RUNTIME_BINDING — making their replacement content invisible to
    the chain pass (P18B's provider). Convention probed statically:
    module attrs ``*_PRE_PATCH_MD5`` + ``*_DIFF_PATH`` + ``_target_path``.
    """

    DIFF = (
        "--- a/x.py\n"
        "+++ b/x.py\n"
        "@@ -1,3 +1,5 @@\n"
        " line_one\n"
        "-line_two\n"
        "+line_two_replaced\n"
        "+line_two_added(num_warps=4)\n"
        " line_three\n"
    )

    def _fake_mod(self, tmp_path, target: Path, md5_value: str):
        import types
        diff_path = tmp_path / "fake.diff"
        diff_path.write_text(self.DIFF)
        mod = types.ModuleType("fake_pn119_like")
        mod.FAKE_PRE_PATCH_MD5 = md5_value
        mod.FAKE_DIFF_PATH = diff_path
        mod._target_path = lambda: target
        mod.GENESIS_FAKE_MARKER = "Genesis FAKE md5 diff marker"
        return mod

    def test_md5_match_is_ok_and_provides_blob(self, pf, tmp_path, mini_tree):
        import hashlib
        target = mini_tree / "v1" / "sample" / "sampler.py"
        md5 = hashlib.md5(target.read_bytes()).hexdigest()
        mod = self._fake_mod(tmp_path, target, md5)
        row = pf.evaluate_md5_diff_patch(mod, mini_tree)
        assert row is not None
        assert row["verdict"] == pf.OK
        # post-apply reconstruction: added AND context lines, minus removed
        blob = row["_replacement_blob"]
        assert "line_two_added(num_warps=4)" in blob
        assert "line_one\nline_two_replaced" in blob
        assert "line_two\n" not in blob

    def test_md5_mismatch_is_drift(self, pf, tmp_path, mini_tree):
        target = mini_tree / "v1" / "sample" / "sampler.py"
        mod = self._fake_mod(tmp_path, target, "0" * 32)
        row = pf.evaluate_md5_diff_patch(mod, mini_tree)
        assert row["verdict"] == pf.DRIFT_ANCHOR
        assert "md5" in row["detail"]

    def test_marker_in_pristine_is_stale(self, pf, tmp_path, mini_tree):
        import hashlib
        target = mini_tree / "v1" / "sample" / "sampler.py"
        target.write_text(
            target.read_text() + "# Genesis FAKE md5 diff marker\n")
        md5 = hashlib.md5(target.read_bytes()).hexdigest()
        mod = self._fake_mod(tmp_path, target, md5)
        row = pf.evaluate_md5_diff_patch(mod, mini_tree)
        assert row["verdict"] == pf.STALE_RESIDUE

    def test_non_md5_module_returns_none(self, pf, mini_tree):
        import types
        mod = types.ModuleType("plain_binding_module")
        assert pf.evaluate_md5_diff_patch(mod, mini_tree) is None


# ─── v1.2: env-forced builder retry (PN204/PN96 class) ────────────────────


class TestEnvForcedRetry:
    """v1.1 false-positive class (triage 2026-06-11 §4): builders that
    early-exit on an unset GENESIS_* env flag returned None and read as
    DRIFT_FILE_MOVED with a canned 'target not found' detail even though
    every target resolves. v1.2 retries the builder with the registry
    env_flag forced to "1" (restored afterwards) and evaluates the built
    patcher normally, tagging the row env_forced=True.
    """

    OK_BUILDER = """
        import os
        from sndr.kernel.text_patch import TextPatch, TextPatcher

        TARGET = {target!r}

        def _make_patcher():
            if os.environ.get("GENESIS_ENABLE_SYNENVF", "0") != "1":
                return None
            return TextPatcher(
                patch_name="SYN env-forced",
                target_file=TARGET,
                marker="SYN_ENVF_MARKER",
                sub_patches=[TextPatch(
                    name="s1",
                    anchor="    probs = logits.softmax(dim=-1)\\n",
                    replacement="    probs = x\\n",
                    required=True,
                )],
            )
    """

    def test_retry_builds_and_tags_row(self, pf, mini_tree, tmp_path,
                                       monkeypatch):
        monkeypatch.delenv("GENESIS_ENABLE_SYNENVF", raising=False)
        modname = _write_module(
            tmp_path, "ptest_envforce",
            self.OK_BUILDER.format(
                target=str(mini_tree / "v1" / "sample" / "sampler.py")))
        rows = pf.evaluate_module(
            modname, ["SYNENVF"], mini_tree,
            specs=[_EnvSpecStub("GENESIS_ENABLE_SYNENVF")])
        assert len(rows) == 1
        assert rows[0]["verdict"] == pf.OK
        assert rows[0]["env_forced"] is True
        # try/finally restore: the forced flag must not leak.
        assert "GENESIS_ENABLE_SYNENVF" not in os.environ

    def test_explicit_zero_flag_restored(self, pf, mini_tree, tmp_path,
                                         monkeypatch):
        """Operator-disabled flag ("0") still gets a health evaluation
        (preflight judges the patch, not the opt-in) and the original
        value is restored."""
        monkeypatch.setenv("GENESIS_ENABLE_SYNENVF", "0")
        modname = _write_module(
            tmp_path, "ptest_envforce0",
            self.OK_BUILDER.format(
                target=str(mini_tree / "v1" / "sample" / "sampler.py")))
        rows = pf.evaluate_module(
            modname, ["SYNENVF"], mini_tree,
            specs=[_EnvSpecStub("GENESIS_ENABLE_SYNENVF")])
        assert rows[0]["verdict"] == pf.OK
        assert rows[0]["env_forced"] is True
        assert os.environ["GENESIS_ENABLE_SYNENVF"] == "0"

    def test_no_specs_no_retry(self, pf, mini_tree, tmp_path, monkeypatch):
        """Without a spec env_flag there is nothing to force — the
        builder-None row keeps the v1.1 DRIFT_FILE_MOVED path."""
        monkeypatch.delenv("GENESIS_ENABLE_SYNENVF", raising=False)
        modname = _write_module(
            tmp_path, "ptest_envforce_nospec",
            self.OK_BUILDER.format(
                target=str(mini_tree / "v1" / "sample" / "sampler.py")))
        rows = pf.evaluate_module(modname, ["SYNENVF"], mini_tree)
        assert rows[0]["verdict"] == pf.DRIFT_FILE_MOVED
        assert "env_forced" not in rows[0]

    def test_retry_none_with_missing_target_stays_drift(
            self, pf, mini_tree, tmp_path, monkeypatch):
        """Retry exhausted AND a resolve target is actually absent →
        DRIFT_FILE_MOVED is the honest verdict (plan §4.2: keep it only
        when targets are actually missing)."""
        monkeypatch.delenv("GENESIS_ENABLE_SYNGONE", raising=False)
        modname = _write_module(tmp_path, "ptest_envforce_gone", """
            import os
            from sndr.engines.vllm.detection.guards import resolve_vllm_file

            def _make_patcher():
                if os.environ.get("GENESIS_ENABLE_SYNGONE", "0") != "1":
                    return None
                target = resolve_vllm_file("v1/sample/no_longer_here.py")
                if target is None:
                    return None
                raise AssertionError("unreachable in this fixture")
        """)
        rows = pf.evaluate_module(
            modname, ["SYNGONE"], mini_tree,
            specs=[_EnvSpecStub("GENESIS_ENABLE_SYNGONE")])
        assert rows[0]["verdict"] == pf.DRIFT_FILE_MOVED
        assert "v1/sample/no_longer_here.py" in rows[0]["detail"]
        assert "GENESIS_ENABLE_SYNGONE" not in os.environ


# ─── v1.2: ENV_GATED_ABSTAIN verdict ──────────────────────────────────────


class TestEnvGatedAbstain:
    """Builder still abstains with the flag forced AND every
    resolve_vllm_file literal exists under the candidate root → the
    abstain is gated on something beyond the registry env_flag (second
    env condition, runtime probe, ...). NOT file drift — non-actionable
    ENV_GATED_ABSTAIN, fail-noisy in the report but no operator gate.
    """

    def _rows(self, pf, mini_tree, tmp_path, monkeypatch):
        monkeypatch.delenv("GENESIS_ENABLE_SYNGATE", raising=False)
        monkeypatch.delenv("GENESIS_SYNGATE_EXTRA", raising=False)
        modname = _write_module(tmp_path, "ptest_envgated", """
            import os
            from sndr.engines.vllm.detection.guards import resolve_vllm_file

            def _make_patcher():
                if os.environ.get("GENESIS_ENABLE_SYNGATE", "0") != "1":
                    return None
                # Second gate the registry flag does not cover.
                if os.environ.get("GENESIS_SYNGATE_EXTRA", "0") != "1":
                    return None
                target = resolve_vllm_file("v1/sample/sampler.py")
                if target is None:
                    return None
                raise AssertionError("unreachable in this fixture")
        """)
        return pf.evaluate_module(
            modname, ["SYNGATE"], mini_tree,
            specs=[_EnvSpecStub("GENESIS_ENABLE_SYNGATE")])

    def test_verdict_env_gated_abstain(self, pf, mini_tree, tmp_path,
                                       monkeypatch):
        rows = self._rows(pf, mini_tree, tmp_path, monkeypatch)
        assert len(rows) == 1
        assert rows[0]["verdict"] == pf.ENV_GATED_ABSTAIN
        assert rows[0]["env_forced"] is True
        # The detail must name the forced flag and say targets resolve.
        assert "GENESIS_ENABLE_SYNGATE" in rows[0]["detail"]
        assert "v1/sample/sampler.py" in rows[0]["detail"]

    def test_env_gated_abstain_not_actionable(self, pf, mini_tree, tmp_path,
                                              monkeypatch):
        rows = self._rows(pf, mini_tree, tmp_path, monkeypatch)
        assert pf.ENV_GATED_ABSTAIN not in pf.ACTIONABLE_VERDICTS
        for r in rows:
            r["in_version_range"] = True
        assert pf.count_actionable(rows, markers=[]) == 0


# ─── v1.2: _read_<param> UNBUILDABLE resolution (P82 class) ───────────────


class TestReadParamFill:
    """P82 false positive: _make_patcher(threshold) has no default BY
    DESIGN — apply() fills it via the module's own _read_threshold().
    v1.2 honors that convention: a required param <name> is fillable
    when the module exposes a callable _read_<name>."""

    def test_required_param_filled_via_module_reader(self, pf, mini_tree,
                                                     tmp_path):
        modname = _write_module(tmp_path, "ptest_readparam", """
            from sndr.kernel.text_patch import TextPatch, TextPatcher

            TARGET = {target!r}

            def _read_threshold():
                return 0.25

            def _make_patcher(threshold, min_draft_pos=0):
                assert threshold == 0.25
                return TextPatcher(
                    patch_name="SYN read-param",
                    target_file=TARGET,
                    marker="SYN_READPARAM_MARKER",
                    sub_patches=[TextPatch(
                        name="s1",
                        anchor="    probs = logits.softmax(dim=-1)\\n",
                        replacement="    probs = x\\n",
                        required=True,
                    )],
                )
        """.format(target=str(mini_tree / "v1" / "sample" / "sampler.py")))
        rows = pf.evaluate_module(modname, ["SYNRP"], mini_tree)
        assert len(rows) == 1
        assert rows[0]["verdict"] == pf.OK
        assert rows[0]["params_filled_via"] == {"threshold": "_read_threshold"}

    def test_param_without_reader_stays_unbuildable(self, pf, mini_tree,
                                                    tmp_path):
        modname = _write_module(tmp_path, "ptest_noreader", """
            def _make_patcher(backend):
                return None
        """)
        rows = pf.evaluate_module(modname, ["SYNNR"], mini_tree)
        assert rows[0]["verdict"] == pf.UNBUILDABLE
        assert "backend" in rows[0]["detail"]

    def test_reader_raising_is_unbuildable(self, pf, mini_tree, tmp_path):
        modname = _write_module(tmp_path, "ptest_badreader", """
            def _read_threshold():
                raise RuntimeError("boom")

            def _make_patcher(threshold):
                return None
        """)
        rows = pf.evaluate_module(modname, ["SYNBR"], mini_tree)
        assert rows[0]["verdict"] == pf.UNBUILDABLE
        assert "_read_threshold" in rows[0]["detail"]


# ─── v1.2: resolved-vs-unresolved target detail ───────────────────────────


class TestResolveTargetDetail:
    """The v1.1 canned detail 'target not found under candidate root'
    lies when every resolve_vllm_file literal exists (PN204 fallback
    class). v1.2 splits the literals and reports both sides."""

    def test_partial_resolve_detail_names_both_sides(self, pf, mini_tree,
                                                     tmp_path):
        modname = _write_module(tmp_path, "ptest_partial", """
            from sndr.engines.vllm.detection.guards import resolve_vllm_file

            def _make_patcher():
                target = (
                    resolve_vllm_file("v1/sample/old_gone_name.py")
                    or resolve_vllm_file("v1/sample/sampler.py")
                )
                if target is None:
                    return None
                raise AssertionError("unreachable in this fixture")
        """)
        rows = pf.evaluate_module(modname, ["SYNPART"], mini_tree)
        assert rows[0]["verdict"] == pf.DRIFT_FILE_MOVED
        detail = rows[0]["detail"]
        assert "v1/sample/old_gone_name.py" in detail
        # The resolving fallback literal must be reported, not hidden.
        assert "v1/sample/sampler.py" in detail

    def test_resolve_target_status_split(self, pf, mini_tree, tmp_path):
        modname = _write_module(tmp_path, "ptest_status", """
            from sndr.engines.vllm.detection.guards import resolve_vllm_file

            def _probe():
                a = resolve_vllm_file("v1/sample/sampler.py")
                b = resolve_vllm_file("v1/sample/old_gone_name.py")
                return a or b
        """)
        mod = importlib.import_module(modname)
        resolved, unresolved = pf._resolve_target_status(mod, mini_tree)
        assert resolved == ["v1/sample/sampler.py"]
        assert unresolved == ["v1/sample/old_gone_name.py"]
        # Back-compat wrapper keeps the v1.1 contract (unresolved only).
        assert pf._unresolved_resolve_targets(mod, mini_tree) == [
            "v1/sample/old_gone_name.py"]


# ─── v1.2: KNOWN_OPTIONAL_RETIRED (P64 class) ─────────────────────────────


class TestKnownOptionalRetired:
    """P64's serving patcher carries two required=False subs whose
    anchors died when upstream refactored the helper away — the module
    documents them as retired-by-design (journal 2026-06-09). Zero
    matches is the documented steady state, not drift. Convention: a
    module attr ``*_RETIRED_SUBS`` lists the retired sub names; an
    all-optional zero-match patcher whose every sub is listed
    reclassifies to non-actionable KNOWN_OPTIONAL_RETIRED."""

    MODULE = """
        from sndr.kernel.text_patch import TextPatch, TextPatcher

        TARGET = {target!r}
        {retired_line}

        def _make_patcher():
            return TextPatcher(
                patch_name="SYN optional-retired",
                target_file=TARGET,
                marker="SYN_OPTRET_MARKER",
                sub_patches=[
                    TextPatch(name="opt_a", anchor="anchor gone a\\n",
                              replacement="x\\n", required=False),
                    TextPatch(name="opt_b", anchor="anchor gone b\\n",
                              replacement="y\\n", required=False),
                ],
            )
    """

    def _rows(self, pf, mini_tree, tmp_path, name, retired_line):
        modname = _write_module(
            tmp_path, name,
            self.MODULE.format(
                target=str(mini_tree / "v1" / "sample" / "sampler.py"),
                retired_line=retired_line))
        return pf.evaluate_module(modname, ["SYNOR"], mini_tree)

    def test_listed_subs_reclassified(self, pf, mini_tree, tmp_path):
        rows = self._rows(
            pf, mini_tree, tmp_path, "ptest_optret",
            'SYN_RETIRED_SUBS = ("opt_a", "opt_b")')
        assert len(rows) == 1
        assert rows[0]["verdict"] == pf.KNOWN_OPTIONAL_RETIRED
        assert "SYN_RETIRED_SUBS" in rows[0]["detail"]

    def test_unlisted_sub_stays_drift(self, pf, mini_tree, tmp_path):
        rows = self._rows(
            pf, mini_tree, tmp_path, "ptest_optret_partial",
            'SYN_RETIRED_SUBS = ("opt_a",)')  # opt_b NOT declared retired
        assert rows[0]["verdict"] == pf.DRIFT_ANCHOR

    def test_no_attr_stays_drift(self, pf, mini_tree, tmp_path):
        rows = self._rows(pf, mini_tree, tmp_path, "ptest_optret_none", "")
        assert rows[0]["verdict"] == pf.DRIFT_ANCHOR

    def test_known_optional_retired_not_actionable(self, pf, mini_tree,
                                                   tmp_path):
        rows = self._rows(
            pf, mini_tree, tmp_path, "ptest_optret_action",
            'SYN_RETIRED_SUBS = ("opt_a", "opt_b")')
        assert pf.KNOWN_OPTIONAL_RETIRED not in pf.ACTIONABLE_VERDICTS
        for r in rows:
            r["in_version_range"] = True
        assert pf.count_actionable(rows, markers=[]) == 0


# ─── v1.3: EXPECTED_ALTERNATE (P91B dual-factory class) ───────────────────


# Present in the mini_tree sampler.py fixture (the dev371-style spelling).
_ALT_ANCHOR_PRESENT = "    probs = logits.softmax(dim=-1)\n"
# Absent from the fixture (the dev338-style spelling of the same site).
_ALT_ANCHOR_ABSENT = "    probs = logits.softmax(dim=-1).float()\n"


class TestExpectedAlternate:
    """P91B inc.py carries two by-design factory variants of the SAME
    code site (dev338 spells ``self.group_size``, dev371 bare
    ``group_size``); exactly one anchor matches on any live pin. The
    non-matching factory previously read DRIFT_ANCHOR — an actionable
    false positive. Convention: a module attr ``*_ANCHOR_ALTERNATION``
    maps each sub name to the sub names of its alternate variants; a
    zero-match DRIFT_ANCHOR row whose every sub is declared AND whose
    declared alternate actually matched in a sibling row of the same
    module reclassifies to non-actionable EXPECTED_ALTERNATE. A miss
    on EVERY variant stays DRIFT_ANCHOR (fail-noisy, never silent)."""

    MODULE = """
        from sndr.kernel.text_patch import TextPatch, TextPatcher

        TARGET = {target!r}
        {alternation_line}

        def _make_dev338_patcher():
            return TextPatcher(
                patch_name="SYN alt dev338",
                target_file=TARGET,
                marker="SYN_ALT_MARKER_dev338",
                sub_patches=[TextPatch(
                    name="alt_dev338",
                    anchor={anchor_338!r},
                    replacement="    probs = a\\n",
                    required=True,
                )],
            )

        def _make_dev371_patcher():
            return TextPatcher(
                patch_name="SYN alt dev371",
                target_file=TARGET,
                marker="SYN_ALT_MARKER_dev371",
                sub_patches=[TextPatch(
                    name="alt_dev371",
                    anchor={anchor_371!r},
                    replacement="    probs = b\\n",
                    required=True,
                )],
            )
    """

    ALTERNATION_LINE = (
        'SYN_ANCHOR_ALTERNATION = {'
        '"alt_dev338": ("alt_dev371",), '
        '"alt_dev371": ("alt_dev338",)}'
    )

    def _rows(self, pf, mini_tree, tmp_path, name, *,
              alternation_line, anchor_338=_ALT_ANCHOR_ABSENT,
              anchor_371=_ALT_ANCHOR_PRESENT):
        modname = _write_module(
            tmp_path, name,
            self.MODULE.format(
                target=str(mini_tree / "v1" / "sample" / "sampler.py"),
                alternation_line=alternation_line,
                anchor_338=anchor_338,
                anchor_371=anchor_371))
        return {r["builder"]: r
                for r in pf.evaluate_module(modname, ["SYNALT"], mini_tree)}

    def test_other_variant_matched_reclassifies(self, pf, mini_tree,
                                                tmp_path):
        rows = self._rows(pf, mini_tree, tmp_path, "ptest_alt",
                          alternation_line=self.ALTERNATION_LINE)
        assert rows["_make_dev371_patcher"]["verdict"] == pf.OK
        row_338 = rows["_make_dev338_patcher"]
        assert row_338["verdict"] == pf.EXPECTED_ALTERNATE
        assert row_338["alternate_matched"] == ["alt_dev371"]
        # The detail must name the declaring attr and the matched sub.
        assert "SYN_ANCHOR_ALTERNATION" in row_338["detail"]
        assert "alt_dev371" in row_338["detail"]

    def test_expected_alternate_not_actionable(self, pf, mini_tree,
                                               tmp_path):
        rows = self._rows(pf, mini_tree, tmp_path, "ptest_alt_action",
                          alternation_line=self.ALTERNATION_LINE)
        assert pf.EXPECTED_ALTERNATE not in pf.ACTIONABLE_VERDICTS
        for r in rows.values():
            r["in_version_range"] = True
        assert pf.count_actionable(list(rows.values()), markers=[]) == 0

    def test_neither_variant_matched_stays_drift(self, pf, mini_tree,
                                                 tmp_path):
        """Both anchors miss → declared alternation does NOT excuse the
        rows; both stay DRIFT_ANCHOR (real drift on every variant)."""
        rows = self._rows(
            pf, mini_tree, tmp_path, "ptest_alt_bothmiss",
            alternation_line=self.ALTERNATION_LINE,
            anchor_338="anchor gone a\n", anchor_371="anchor gone b\n")
        assert rows["_make_dev338_patcher"]["verdict"] == pf.DRIFT_ANCHOR
        assert rows["_make_dev371_patcher"]["verdict"] == pf.DRIFT_ANCHOR

    def test_no_alternation_attr_stays_drift(self, pf, mini_tree, tmp_path):
        rows = self._rows(pf, mini_tree, tmp_path, "ptest_alt_noattr",
                          alternation_line="")
        assert rows["_make_dev338_patcher"]["verdict"] == pf.DRIFT_ANCHOR

    def test_undeclared_sub_stays_drift(self, pf, mini_tree, tmp_path):
        """The missing sub itself must be declared in the map — a map
        that only lists the OTHER variant does not cover it."""
        rows = self._rows(
            pf, mini_tree, tmp_path, "ptest_alt_partial",
            alternation_line=(
                'SYN_ANCHOR_ALTERNATION = {"alt_dev371": ("alt_dev338",)}'))
        assert rows["_make_dev338_patcher"]["verdict"] == pf.DRIFT_ANCHOR

    # ─── real P91B wiring ────────────────────────────────────────────

    P91B_MODULE = (
        "sndr.engines.vllm.patches.quantization."
        "p91b_autoround_row_group_cdiv_multi_scheme"
    )

    def test_p91b_module_declares_alternation(self, pf):
        import importlib as _il
        mod = _il.import_module(self.P91B_MODULE)
        alt = getattr(mod, "P91B_ANCHOR_ALTERNATION", None)
        assert isinstance(alt, dict), (
            "P91B must declare P91B_ANCHOR_ALTERNATION (dual-factory "
            "dev338/dev371 by design — docstring Option A)")
        a338 = "p91b_inc_dev338_floor_partition_to_cdiv"
        a371 = "p91b_inc_dev371_floor_partition_to_cdiv"
        assert a371 in alt[a338]
        assert a338 in alt[a371]

    def test_p91b_dual_factory_end_to_end(self, pf, tmp_path, monkeypatch):
        """Wire the REAL P91B module against a synthetic tree carrying
        the dev371 inc.py spelling (the current-pin state, pristine
        inc.py:538): the dev371 factory row is OK, the dev338 row is
        EXPECTED_ALTERNATE, the two compressed-tensors rows are OK."""
        import sndr.engines.vllm.detection.guards as guards

        root = tmp_path / "synthetic_vllm"
        quant = root / "model_executor" / "layers" / "quantization"
        schemes = quant / "compressed_tensors" / "schemes"
        schemes.mkdir(parents=True)
        (quant / "inc.py").write_text(
            "class INCLinearMethod:\n"
            "    def create_weights(self, input_size_per_partition,"
            " group_size):\n"
            "        scales_and_zp_size = input_size_per_partition"
            " // group_size\n"
            "        return scales_and_zp_size\n"
        )
        ct_body = (
            "def create_weights(input_size, group_size,"
            " partition_scales):\n"
            "    if not partition_scales:\n"
            "        scales_and_zp_size = input_size // group_size\n"
            "    return scales_and_zp_size\n"
        )
        (schemes / "compressed_tensors_wNa16.py").write_text(ct_body)
        (schemes / "compressed_tensors_w4a8_fp8.py").write_text(ct_body)

        monkeypatch.setattr(guards, "vllm_install_root",
                            lambda: str(root))
        rows = {r["builder"]: r
                for r in pf.evaluate_module(self.P91B_MODULE, ["P91B"],
                                            root)}
        assert rows["_make_inc_dev371_patcher"]["verdict"] == pf.OK
        assert rows["_make_inc_dev338_patcher"]["verdict"] == \
            pf.EXPECTED_ALTERNATE
        assert rows["_make_wna16_patcher"]["verdict"] == pf.OK
        assert rows["_make_w4a8_fp8_patcher"]["verdict"] == pf.OK
