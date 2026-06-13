# SPDX-License-Identifier: Apache-2.0
"""TDD for PN376 — FP8 modules_to_not_convert substring match (vllm#44628).

Upstream bug (vllm#21669): ``Fp8Config.get_quant_method`` calls
``is_layer_skipped`` with the default ``skip_with_substr=False`` (exact
match), so HuggingFace-style short ``modules_to_not_convert`` patterns
(e.g. ``"linear_attn.in_proj_qkv"``) never match the fully-qualified
runtime prefix (``"language_model.model.layers.0.linear_attn.
in_proj_qkv"``). The ignored layer silently loads as FP8 despite the
checkpoint lacking its ``weight_scale`` — gibberish output. #44628 opts
the FP8 family into substring match (AWQ-family parity, #26909/#27416)
and rewrites the quant_utils experts branch so the legacy MoE
parent-in-child containment convention survives substring mode.

Test strategy (PN373 convention):
  1. Anchor byte-verification against the pristine candidate tree
     (/private/tmp/candidate_pin_current) — count==1 per anchor.
  2. Pristine bug reproduction — exact-match short-pattern miss
     (#21669) AND substr-mode experts-containment loss (the regression
     the PR's review caught). If these start FAILING after a pin bump,
     upstream merged #44628 — PN376 is retire-eligible (iron-rule-#11
     deep-diff then).
  3. Patched semantics — the full 11-case matrix of the PR's
     tests/quantization/test_layer_skipped.py, replayed against OUR
     patched quant_utils source, plus a Genesis-pinned test for the
     upstream-accepted experts-filter delta visible to AWQ substr
     callers (Gemma-4 AWQ MoE).
  4. Text-level checks on the patched fp8/fbgemm/mxfp4/modelopt call
     sites (these files import torch — exec is off the table, ast-parse
     + count checks only).
  5. MultiFilePatchTransaction integration on a temp tree: APPLIED →
     IDEMPOTENT → upstream-merged SKIP → core-drift atomic withhold →
     parity-drift soft-continue.
  6. Drift-marker hygiene (tools/lint_drift_markers.py mirror).
"""
from __future__ import annotations

import ast
import os
from collections.abc import Mapping
from types import MappingProxyType

import pytest


_PRISTINE_QUANT_DIR = (
    "/private/tmp/candidate_pin_current/vllm/model_executor/layers/"
    "quantization"
)

PRISTINE_FILES = {
    "fp8": f"{_PRISTINE_QUANT_DIR}/fp8.py",
    "quant_utils": f"{_PRISTINE_QUANT_DIR}/utils/quant_utils.py",
    "fbgemm": f"{_PRISTINE_QUANT_DIR}/fbgemm_fp8.py",
    "mxfp4": f"{_PRISTINE_QUANT_DIR}/mxfp4.py",
    "modelopt": f"{_PRISTINE_QUANT_DIR}/modelopt.py",
}

requires_pristine = pytest.mark.skipif(
    not all(os.path.isfile(p) for p in PRISTINE_FILES.values()),
    reason="pristine candidate pin tree not extracted on this host",
)

# ─── #44628 post-image constants (gh pr diff 44628, fetched 2026-06-11) ──
#
# The drift markers must be substrings of THESE texts (so the patch
# self-skips when the fix lands upstream) while NOT being substrings of
# anything PN376 itself writes (self-collision lint).

UPSTREAM_44628_QUANT_UTILS_POST_COMMENT = (
    "        # Preserve the legacy MoE convention where a child expert "
    "listed in\n"
    "        # ``ignored_layers`` (e.g. ``model.layers.0.mlp.experts.0.w1``) "
    "skips\n"
    "        # its parent ``RoutedExperts`` prefix "
    "(``model.layers.0.mlp.experts``)\n"
    "        # via parent-in-child containment. When ``skip_with_substr`` "
    "is\n"
    "        # enabled, also honour the substring direction so HF-style "
    "short\n"
    "        # patterns still match.\n"
)

UPSTREAM_44628_MODELOPT_POST_COMMENT = (
    "        # First check matching with fused layer support; use substring "
    "match\n"
    '        # so HF-style short patterns (e.g. "linear_attn.in_proj_qkv") '
    "work\n"
    "        # against the fully-qualified runtime prefix. Aligns with AWQ "
    "family\n"
    "        # (#26909, #27416).\n"
)


def _pn376():
    from sndr.engines.vllm.patches.quantization import (
        pn376_fp8_ignore_substring as M,
    )
    return M


def _pristine_src(key: str) -> str:
    with open(PRISTINE_FILES[key]) as f:
        return f.read()


def _merged_quant_utils_branch(M) -> str:
    """The experts branch as #44628's post-image will read once merged
    (upstream comment wording, NOT ours)."""
    return (
        '    elif "experts" in prefix:\n'
        "        expert_ignore_layers = [\n"
        '            layer_name for layer_name in ignored_layers if '
        '"experts" in layer_name\n'
        "        ]\n"
        + UPSTREAM_44628_QUANT_UTILS_POST_COMMENT
        + "        return any(\n"
        "            prefix in layer_name or (skip_with_substr and "
        "layer_name in prefix)\n"
        "            for layer_name in expert_ignore_layers\n"
        "        )\n"
    )


def _patched_src(key: str) -> str:
    """Pristine source with PN376's sub-patches for that file applied."""
    M = _pn376()
    src = _pristine_src(key)
    for anchor, replacement in M.SUB_PATCH_TEXTS[key]:
        assert anchor in src, f"{key}: anchor missing"
        src = src.replace(anchor, replacement, 1)
    return src


def _extract_is_layer_skipped(src: str):
    """Exec ONLY the is_layer_skipped function out of quant_utils.py
    source (the full file imports torch — not importable torch-less)."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "is_layer_skipped":
            seg = ast.get_source_segment(src, node)
            assert seg is not None
            ns: dict = {
                "Mapping": Mapping,
                "MappingProxyType": MappingProxyType,
            }
            exec(compile(seg, "is_layer_skipped_under_test.py", "exec"), ns)
            return ns["is_layer_skipped"]
    raise AssertionError("is_layer_skipped not found in source")


# ─── 1. Anchor byte-verification (iron rule #11) ──────────────────────


@requires_pristine
class TestAnchorAgainstPristine:
    def test_every_anchor_count_exactly_one(self):
        M = _pn376()
        for key, pairs in M.SUB_PATCH_TEXTS.items():
            src = _pristine_src(key)
            for anchor, _ in pairs:
                assert src.count(anchor) == 1, (
                    f"{key}: anchor count != 1:\n{anchor!r}"
                )

    def test_every_replacement_absent_from_pristine(self):
        M = _pn376()
        for key, pairs in M.SUB_PATCH_TEXTS.items():
            src = _pristine_src(key)
            for _, replacement in pairs:
                assert replacement not in src, f"{key}: replacement in pristine"

    def test_drift_markers_absent_from_pristine(self):
        """Markers must fire only on the post-#44628 merged form."""
        M = _pn376()
        for marker in M._QUANT_UTILS_DRIFT_MARKERS:
            assert marker not in _pristine_src("quant_utils")
        for marker in M._MODELOPT_DRIFT_MARKERS:
            assert marker not in _pristine_src("modelopt")

    @pytest.mark.parametrize("key", sorted(PRISTINE_FILES))
    def test_patched_source_is_valid_python(self, key):
        ast.parse(_patched_src(key))

    def test_fp8_has_exactly_two_call_sites_in_pin(self):
        """The pin (0.22.1rc1.dev259+g303916e93) has TWO is_layer_skipped
        call sites in fp8.py (LinearBase + RoutedExperts) — one more than
        #44628's base. Both must be patched (roadmap: 'both call sites')."""
        M = _pn376()
        assert len(M.SUB_PATCH_TEXTS["fp8"]) == 2
        assert _pristine_src("fp8").count("skip_with_substr") == 0
        assert _patched_src("fp8").count("skip_with_substr=True") == 2

    @pytest.mark.parametrize("key", ["fbgemm", "mxfp4", "modelopt"])
    def test_parity_files_gain_exactly_one_substr_optin(self, key):
        assert _pristine_src(key).count("skip_with_substr") == 0
        assert _patched_src(key).count("skip_with_substr=True") == 1


# ─── 2. Pristine bug reproduction (#21669 + the review regression) ────


@requires_pristine
class TestPristineBugReproduction:
    """Documents the bugs PN376 fixes. If these start FAILING after a
    pin bump, upstream merged #44628 — PN376 is retire-eligible."""

    def test_exact_match_short_pattern_misses_on_pristine(self):
        """#21669: HF-style short pattern fails exact match — the layer
        is silently fp8-quantized and outputs gibberish."""
        fn = _extract_is_layer_skipped(_pristine_src("quant_utils"))
        assert fn(
            prefix="language_model.model.layers.0.linear_attn.in_proj_qkv",
            ignored_layers=["linear_attn.in_proj_qkv"],
        ) is False

    def test_substr_mode_loses_experts_containment_on_pristine(self):
        """The #44628-review regression PN376's quant_utils sub-patch
        preserves: pristine gates the experts branch on
        ``not skip_with_substr``, so substr-mode callers (AWQ family)
        lose the parent-in-child containment convention."""
        fn = _extract_is_layer_skipped(_pristine_src("quant_utils"))
        assert fn(
            prefix="model.layers.0.mlp.experts",
            ignored_layers=["model.layers.0.mlp.experts.0.w1"],
            skip_with_substr=True,
        ) is False

    def test_pristine_fp8_call_sites_use_exact_match(self):
        src = _pristine_src("fp8")
        assert "skip_with_substr" not in src


# ─── 3. Patched quant_utils semantics (PR's 11-case matrix + Genesis) ─


@requires_pristine
class TestPatchedQuantUtilsSemantics:
    @pytest.fixture()
    def fn(self):
        return _extract_is_layer_skipped(_patched_src("quant_utils"))

    # — exact-match path (legacy default, preserved) —

    def test_exact_match_full_path_hit(self, fn):
        assert fn(
            prefix="model.layers.0.lm_head",
            ignored_layers=["model.layers.0.lm_head"],
        ) is True

    def test_exact_match_short_pattern_still_misses(self, fn):
        """Legacy exact-match callers keep their semantics."""
        assert fn(
            prefix="language_model.model.layers.0.linear_attn.in_proj_qkv",
            ignored_layers=["linear_attn.in_proj_qkv"],
        ) is False

    # — substring-match path (the fix) —

    def test_substr_match_short_pattern_hits(self, fn):
        assert fn(
            prefix="language_model.model.layers.0.linear_attn.in_proj_qkv",
            ignored_layers=["linear_attn.in_proj_qkv"],
            skip_with_substr=True,
        ) is True

    def test_substr_match_full_path_still_hits(self, fn):
        """Backwards compatibility: a full path is a substring of itself."""
        assert fn(
            prefix="model.layers.0.self_attn.q_proj",
            ignored_layers=["model.layers.0.self_attn.q_proj"],
            skip_with_substr=True,
        ) is True

    def test_substr_match_unrelated_misses(self, fn):
        assert fn(
            prefix="model.layers.0.mlp.down_proj",
            ignored_layers=["linear_attn.in_proj_qkv"],
            skip_with_substr=True,
        ) is False

    # — fused mapping path —

    def test_fused_mapping_substr_hit(self, fn):
        fused = MappingProxyType({"qkv_proj": ["q_proj", "k_proj", "v_proj"]})
        assert fn(
            prefix="model.layers.0.self_attn.qkv_proj",
            ignored_layers=[
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
            ],
            fused_mapping=fused,
            skip_with_substr=True,
        ) is True

    def test_fused_mapping_partial_skip_raises(self, fn):
        fused = MappingProxyType({"qkv_proj": ["q_proj", "k_proj", "v_proj"]})
        with pytest.raises(ValueError, match="same precision"):
            fn(
                prefix="model.layers.0.self_attn.qkv_proj",
                ignored_layers=["self_attn.q_proj"],
                fused_mapping=fused,
                skip_with_substr=True,
            )

    # — experts path —

    def test_experts_path_exact_default(self, fn):
        assert fn(
            prefix="model.layers.0.block_sparse_moe.experts.0.w1",
            ignored_layers=["model.layers.0.block_sparse_moe.experts.0.w1"],
        ) is True

    def test_experts_path_substr(self, fn):
        assert fn(
            prefix="model.layers.0.block_sparse_moe.experts.0.w1",
            ignored_layers=["block_sparse_moe.experts.0"],
            skip_with_substr=True,
        ) is True

    def test_experts_parent_prefix_with_child_ignored_legacy(self, fn):
        """Legacy MoE convention: a child expert in ignored_layers skips
        the parent RoutedExperts prefix (exact mode)."""
        assert fn(
            prefix="model.layers.0.mlp.experts",
            ignored_layers=["model.layers.0.mlp.experts.0.w1"],
        ) is True

    def test_experts_parent_prefix_with_child_ignored_substr(self, fn):
        """THE preserved convention under substring mode — pristine pin
        returns False here for substr callers (see bug-repro class)."""
        assert fn(
            prefix="model.layers.0.mlp.experts",
            ignored_layers=["model.layers.0.mlp.experts.0.w1"],
            skip_with_substr=True,
        ) is True

    # — empty ignored_layers —

    def test_empty_ignored_layers_never_skip(self, fn):
        assert fn(prefix="model.layers.0.lm_head", ignored_layers=[]) is False
        assert fn(
            prefix="model.layers.0.lm_head",
            ignored_layers=[],
            skip_with_substr=True,
        ) is False

    # — Genesis-pinned upstream-accepted delta (AWQ-visible) —

    def test_experts_prefix_non_experts_pattern_filtered_substr(self, fn):
        """Upstream-accepted semantics change PN376 inherits: under
        substr mode an experts-containing prefix consults ONLY
        experts-containing ignore entries (the experts branch no longer
        falls through to the generic substr else-branch). Pristine pin
        returned True here via the else-branch. Visible to AWQ substr
        callers (Gemma-4 AWQ MoE) when PN376 is ON — pinned so any
        future deviation is loud."""
        assert fn(
            prefix="model.layers.0.mlp.experts.0.w1",
            ignored_layers=["mlp"],
            skip_with_substr=True,
        ) is False

    def test_experts_prefix_non_experts_pattern_pristine_was_true(self):
        """Companion pin: proves the delta exists vs pristine (else-branch
        substr match over ALL ignore entries)."""
        fn = _extract_is_layer_skipped(_pristine_src("quant_utils"))
        assert fn(
            prefix="model.layers.0.mlp.experts.0.w1",
            ignored_layers=["mlp"],
            skip_with_substr=True,
        ) is True


# ─── 4. MultiFilePatchTransaction integration on a temp tree ─────────


@requires_pristine
class TestTextPatcherIntegration:
    REL = {
        "fp8": "model_executor/layers/quantization/fp8.py",
        "quant_utils": "model_executor/layers/quantization/utils/quant_utils.py",
        "fbgemm": "model_executor/layers/quantization/fbgemm_fp8.py",
        "mxfp4": "model_executor/layers/quantization/mxfp4.py",
        "modelopt": "model_executor/layers/quantization/modelopt.py",
    }

    @pytest.fixture()
    def temp_tree(self, tmp_path, monkeypatch):
        """Writable copies of all five pristine targets inside a
        synthetic vllm root; guards.vllm_install_root redirected."""
        monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", "1")
        for key, rel in self.REL.items():
            dst = tmp_path / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(_pristine_src(key))
        import sndr.engines.vllm.detection.guards as guards
        monkeypatch.setattr(guards, "vllm_install_root", lambda: str(tmp_path))
        return tmp_path

    def _read(self, tree, key) -> str:
        return (tree / self.REL[key]).read_text()

    def test_apply_then_idempotent(self, temp_tree):
        M = _pn376()
        status, reason = M.apply()
        assert status == "applied", reason
        for key in self.REL:
            content = self._read(temp_tree, key)
            assert M.GENESIS_PN376_MARKER in content, f"{key}: no marker"
            ast.parse(content)
        assert self._read(temp_tree, "fp8").count("skip_with_substr=True") == 2
        qu = self._read(temp_tree, "quant_utils")
        assert (
            "prefix in layer_name or (skip_with_substr and layer_name in prefix)"
            in qu
        )
        assert 'elif "experts" in prefix and not skip_with_substr:' not in qu
        assert M.is_applied() is True
        status2, reason2 = M.apply()
        assert status2 == "skipped"
        assert "already applied" in reason2

    def test_upstream_merged_form_skips(self, temp_tree):
        """Simulate the post-#44628 tree (next pin bump): PN376 must
        self-skip via the quant_utils drift marker, leaving every file
        untouched."""
        M = _pn376()
        qu_anchor = M.SUB_PATCH_TEXTS["quant_utils"][0][0]
        merged = _pristine_src("quant_utils").replace(
            qu_anchor, _merged_quant_utils_branch(M), 1
        )
        assert merged != _pristine_src("quant_utils")
        (temp_tree / self.REL["quant_utils"]).write_text(merged)
        status, reason = M.apply()
        assert status == "skipped"
        assert "upstream_merged" in reason
        # Nothing else may have been written.
        for key in ("fp8", "fbgemm", "mxfp4", "modelopt"):
            assert self._read(temp_tree, key) == _pristine_src(key)

    def test_core_anchor_drift_withholds_atomically(self, temp_tree):
        """fp8.py anchor drift → the quant_utils sub-patch must NOT land
        either (enabling substr at the call sites without the experts
        branch — or vice versa — is the partial state the transaction
        exists to prevent)."""
        M = _pn376()
        fp8_path = temp_tree / self.REL["fp8"]
        fp8_path.write_text(
            self._read(temp_tree, "fp8").replace(
                "fused_mapping=self.packed_modules_mapping,",
                "fused_mapping=self.packed_modules_mapping,  # drifted",
            )
        )
        status, reason = M.apply()
        assert status == "skipped"
        assert self._read(temp_tree, "quant_utils") == _pristine_src(
            "quant_utils"
        )
        assert M.GENESIS_PN376_MARKER not in self._read(temp_tree, "fp8")

    def test_parity_anchor_drift_does_not_withhold_core(self, temp_tree):
        """modelopt parity drift is soft: the core FP8 fix still lands
        (we run no modelopt models; parity is upstream-parity only)."""
        M = _pn376()
        mo_path = temp_tree / self.REL["modelopt"]
        mo_path.write_text(
            self._read(temp_tree, "modelopt").replace(
                "# First check exact matching with fused layer support",
                "# First check exact matching (drifted)",
            )
        )
        status, reason = M.apply()
        assert status == "applied", reason
        assert "modelopt" in reason
        assert M.GENESIS_PN376_MARKER in self._read(temp_tree, "fp8")
        assert M.GENESIS_PN376_MARKER not in self._read(temp_tree, "modelopt")

    def test_missing_parity_file_does_not_withhold_core(self, temp_tree):
        M = _pn376()
        (temp_tree / self.REL["mxfp4"]).unlink()
        status, reason = M.apply()
        assert status == "applied", reason
        assert M.GENESIS_PN376_MARKER in self._read(temp_tree, "quant_utils")

    def test_missing_core_file_skips(self, temp_tree):
        M = _pn376()
        (temp_tree / self.REL["quant_utils"]).unlink()
        status, _reason = M.apply()
        assert status == "skipped"
        assert M.is_applied() is False
        assert self._read(temp_tree, "fp8") == _pristine_src("fp8")


# ─── 5. Drift-marker hygiene (lint_drift_markers.py mirror) ───────────


class TestDriftMarkerHygiene:
    def _all_markers(self, M):
        return list(M._QUANT_UTILS_DRIFT_MARKERS) + list(
            M._MODELOPT_DRIFT_MARKERS
        )

    def test_markers_not_in_own_replacements(self):
        """Self-collision class (PN369 false-skip): a marker the patch
        itself emits would read as 'upstream merged' on next boot."""
        M = _pn376()
        for marker in self._all_markers(M):
            for pairs in M.SUB_PATCH_TEXTS.values():
                for _, replacement in pairs:
                    assert marker not in replacement

    def test_markers_not_in_idempotency_marker_line(self):
        M = _pn376()
        marker_line = f"# [Genesis wiring marker: {M.GENESIS_PN376_MARKER}]\n"
        for marker in self._all_markers(M):
            assert marker not in marker_line

    def test_quant_utils_markers_fire_on_merged_form(self):
        M = _pn376()
        merged = _merged_quant_utils_branch(M)
        for marker in M._QUANT_UTILS_DRIFT_MARKERS:
            assert marker in merged

    def test_modelopt_markers_fire_on_merged_form(self):
        M = _pn376()
        for marker in M._MODELOPT_DRIFT_MARKERS:
            assert marker in UPSTREAM_44628_MODELOPT_POST_COMMENT

    def test_anchors_not_substrings_of_replacements(self):
        """A replacement containing its own anchor would double-patch on
        a marker-less re-apply."""
        M = _pn376()
        for pairs in M.SUB_PATCH_TEXTS.values():
            for anchor, replacement in pairs:
                assert anchor not in replacement

    def test_module_references_env_flag(self):
        import inspect
        M = _pn376()
        src = inspect.getsource(M)
        assert "GENESIS_ENABLE_PN376_FP8_IGNORE_SUBSTRING" in src

    def test_env_flag_registered_in_flags_class(self):
        from sndr.env import Flags
        assert Flags.PN376_FP8_IGNORE_SUBSTRING == "PN376_FP8_IGNORE_SUBSTRING"

    def test_validation_harness_note_present(self):
        """Roadmap mandate: per-layer quant-scheme log diff on 35B must
        be documented as the gate before any default_on."""
        M = _pn376()
        doc = (M.__doc__ or "").lower()
        assert "per-layer quant-scheme" in doc
        assert "default_on" in doc
