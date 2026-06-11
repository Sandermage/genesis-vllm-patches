# SPDX-License-Identifier: Apache-2.0
"""TDD for Patch 59 — Qwen3 reasoning embedded tool_call recovery.

Backport of vllm-project/vllm#39055 (ZenoAFfectionate, still OPEN —
re-verified via `gh pr view 39055` on 2026-06-11).

Re-anchor batch 2026-06-11 (preflight residual triage §1b):
  - IMPORT_OLD follows the pristine `Iterable, Sequence` import.
  - Wrap variants A/B (dead residue anchors) retired; replaced by
    variant C (chained on P27's post-apply output) and variant D
    (pristine, P27-absent deployments).
  - apply() gains a require-at-least-one gate: it must NOT report
    "applied" when neither core </think>-present wrap variant matched.

Validates:
  1. Anchors against a synthetic PRISTINE-shaped file (current pin
     layout, quoted from /private/tmp/candidate_pin_current/vllm/
     reasoning/qwen3_reasoning_parser.py lines 1-16, 53-61, 136-158)
  2. Variant C/D mutual exclusivity + the P27 chain invariant the
     preflight CHAINED_ANCHOR pass relies on
  3. End-to-end chain: P27 applied first, then P59 lands via variant C
  4. Require-at-least-one gate (apply() returns "failed" on wrap miss)
  5. Idempotency, upstream-drift markers, self-collision disjointness
  6. Opt-in env-flag gating

Behavioural validation happens via blue/green container reproducer test —
see Genesis_Doc/spec_decode_investigation/v7_12_session/.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

from pathlib import Path

import pytest


# Synthetic mirror of the PRISTINE current-pin parser. Every P59 anchor
# site is quoted byte-exactly from the pristine tree (verified count==1
# against /private/tmp/candidate_pin_current/vllm/reasoning/
# qwen3_reasoning_parser.py on 2026-06-11). Methods irrelevant to P59
# anchors are omitted; the file must stay valid Python (ast.parse test).
SYNTHETIC_PRISTINE_FILE = (
    "# SPDX-License-Identifier: Apache-2.0\n"
    "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"
    "\n"
    "from collections.abc import Iterable, Sequence\n"
    "from typing import TYPE_CHECKING\n"
    "\n"
    "from vllm.entrypoints.openai.engine.protocol import DeltaMessage\n"
    "from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser\n"
    "\n"
    "if TYPE_CHECKING:\n"
    "    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest\n"
    "    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest\n"
    "    from vllm.tokenizers import TokenizerLike\n"
    "\n"
    "\n"
    "class Qwen3ReasoningParser(BaseThinkingReasoningParser):\n"
    "    \"\"\"Reasoning parser for the Qwen3/Qwen3.5 model family.\"\"\"\n"
    "\n"
    "    @property\n"
    "    def start_token(self) -> str:\n"
    "        \"\"\"The token that starts reasoning content.\"\"\"\n"
    "        return \"<think>\"\n"
    "\n"
    "    @property\n"
    "    def end_token(self) -> str:\n"
    "        \"\"\"The token that ends reasoning content.\"\"\"\n"
    "        return \"</think>\"\n"
    "\n"
    "    def extract_reasoning(self, model_output, request):\n"
    "        # Strip <think> if present in the generated output.\n"
    "        model_output_parts = model_output.partition(self.start_token)\n"
    "        model_output = (\n"
    "            model_output_parts[2] if model_output_parts[1] else model_output_parts[0]\n"
    "        )\n"
    "\n"
    "        if self.end_token in model_output:\n"
    "            reasoning, _, content = model_output.partition(self.end_token)\n"
    "            return reasoning, content or None\n"
    "\n"
    "        if not self.thinking_enabled:\n"
    "            # Thinking explicitly disabled — treat everything as content.\n"
    "            return None, model_output\n"
    "\n"
    "        # No </think> — check for implicit reasoning end via <tool_call>.\n"
    "        tool_call_index = model_output.find(self._tool_call_tag)\n"
    "        if tool_call_index != -1:\n"
    "            reasoning = model_output[:tool_call_index]\n"
    "            content = model_output[tool_call_index:]\n"
    "            return reasoning or None, content or None\n"
    "        # Thinking enabled but no </think>: output was truncated.\n"
    "        # Everything generated so far is reasoning.\n"
    "        return model_output, None\n"
)


@pytest.fixture
def pristine_parser_file(tmp_path):
    p = tmp_path / "qwen3_reasoning_parser.py"
    p.write_text(SYNTHETIC_PRISTINE_FILE)
    return str(p)


def _p59_module():
    import sndr.engines.vllm.patches.reasoning.p59_qwen3_reasoning_tool_call_recovery as p59
    return p59


def _p27_module():
    import sndr.engines.vllm.patches.reasoning.p27_reasoning_before_think as p27
    return p27


def _make_p59_patcher(target_file: str, marker_suffix: str):
    """Build a P59 patcher with the module's real sub-patch layout but a
    test-local marker (so tmp files never collide with the prod marker)."""
    from sndr.kernel.text_patch import TextPatcher
    p59 = _p59_module()
    prod = p59._make_patcher_for_target(target_file)
    return TextPatcher(
        patch_name="P59 test",
        target_file=target_file,
        marker=f"P59_TEST_{marker_suffix}",
        sub_patches=prod.sub_patches,
        upstream_drift_markers=prod.upstream_drift_markers,
    )


def _apply_p27(target_file: str):
    """Apply P27's non-streaming sub-patches (real constants from the P27
    module) so the file mirrors the post-P27 boot state P59 chains on."""
    from sndr.kernel.text_patch import TextPatcher, TextPatch, TextPatchResult
    p27 = _p27_module()
    patcher = TextPatcher(
        patch_name="P27 test (chain provider)",
        target_file=target_file,
        marker="P27_TEST_CHAIN_PROVIDER",
        sub_patches=[
            TextPatch(
                name="p27_nonstream_capture",
                anchor=p27._OLD_NONSTREAM,
                replacement=p27._NEW_NONSTREAM,
                required=True,
            ),
            TextPatch(
                name="p27_nonstream_return_pr35687",
                anchor=p27._OLD_NONSTREAM_RETURN_PR35687,
                replacement=p27._NEW_NONSTREAM_RETURN_PR35687,
                required=True,
            ),
        ],
    )
    result, failure = patcher.apply()
    assert result == TextPatchResult.APPLIED, failure


class TestP59AnchorsAgainstPristine:
    def test_required_anchors_unique_in_pristine_shape(self):
        p59 = _p59_module()
        for name, anchor in [
            ("IMPORT", p59.IMPORT_OLD),
            ("REGEX", p59.REGEX_OLD),
            ("METHOD", p59.METHOD_OLD),
            ("RETURN_TRUNC", p59.RETURN_TRUNC_OLD),
            ("RETURN_THINK_PRISTINE (variant D)", p59.RETURN_THINK_PRISTINE_OLD),
        ]:
            count = SYNTHETIC_PRISTINE_FILE.count(anchor)
            assert count == 1, (
                f"{name} anchor must appear exactly once in the pristine "
                f"shape (got {count})"
            )

    def test_import_anchor_includes_iterable(self):
        # Plan §1b item (a): pristine line 4 is
        # `from collections.abc import Iterable, Sequence` on this pin.
        p59 = _p59_module()
        assert "Iterable, Sequence" in p59.IMPORT_OLD

    def test_variant_c_absent_from_pristine(self):
        # Variant C anchors on P27's post-apply output — it must NOT
        # match a pristine (P27-absent) file.
        p59 = _p59_module()
        assert SYNTHETIC_PRISTINE_FILE.count(p59.RETURN_THINK_P27_CHAIN_OLD) == 0


class TestP59P27ChainInvariants:
    """The preflight CHAINED_ANCHOR pass (tools/pin_preflight.py::
    reclassify_chained) flips a DRIFT_ANCHOR verdict only when every
    missing anchor is a substring of a same-target sibling's replacement
    blob. These tests pin that invariant for the P27 -> P59 chain."""

    def test_variant_c_anchor_is_p27_post_apply_text(self):
        p59 = _p59_module()
        p27 = _p27_module()
        assert p59.RETURN_THINK_P27_CHAIN_OLD == p27._NEW_NONSTREAM_RETURN_PR35687

    def test_variant_c_anchor_within_p27_replacement_blob(self):
        # Mirror of pin_preflight's `_replacement_blob` construction.
        # (P27's _make_patcher needs a resolvable vllm tree, so the blob
        # is built from the module constants — the same strings its
        # patcher passes as sub-patch replacements.)
        p59 = _p59_module()
        p27 = _p27_module()
        blob = "\n".join([
            p27._NEW_NONSTREAM,
            p27._NEW_NONSTREAM_RETURN_PR35687,
            p27._NEW_STREAM_START,
        ])
        assert p59.RETURN_THINK_P27_CHAIN_OLD in blob

    def test_chain_derivation_is_valid(self):
        p59 = _p59_module()
        assert p59._P27_CHAIN_DERIVATION_OK is True
        assert p59.RETURN_THINK_P27_CHAIN_NEW != p59.RETURN_THINK_P27_CHAIN_OLD
        assert (
            "self._split_embedded_tool_calls(reasoning, content or None)"
            in p59.RETURN_THINK_P27_CHAIN_NEW
        )
        # P27's BEFORE-THINK prepend must survive the P59 wrap.
        assert "_genesis_before_think" in p59.RETURN_THINK_P27_CHAIN_NEW

    def test_variants_c_and_d_mutually_exclusive(self):
        # D (pristine 3-line block) must not be a substring of C
        # (P27 inserts comment lines between partition and return), so
        # at most one variant can match any given file state.
        p59 = _p59_module()
        assert p59.RETURN_THINK_PRISTINE_OLD not in p59.RETURN_THINK_P27_CHAIN_OLD


class TestP59ApplicationPristine:
    def test_apply_on_pristine_uses_variant_d(self, pristine_parser_file):
        from sndr.kernel.text_patch import TextPatchResult
        patcher = _make_p59_patcher(pristine_parser_file, "PRISTINE")
        result, failure = patcher.apply()
        assert result == TextPatchResult.APPLIED, failure
        assert "p59_wrap_think_return_pristine" in patcher.applied_sub_patches
        assert "p59_wrap_think_return_p27_chain" not in patcher.applied_sub_patches
        modified = Path(pristine_parser_file).read_text()
        assert "import re  # [Genesis P59 vllm#39055]" in modified
        assert "_EMBEDDED_TOOL_CALL_RE = re.compile" in modified
        assert "self._split_embedded_tool_calls(reasoning, content or None)" in modified
        assert "self._split_embedded_tool_calls(model_output, None)" in modified

    def test_modified_pristine_file_parses_as_python(self, pristine_parser_file):
        import ast
        from sndr.kernel.text_patch import TextPatchResult
        patcher = _make_p59_patcher(pristine_parser_file, "PARSE")
        result, _ = patcher.apply()
        assert result == TextPatchResult.APPLIED
        ast.parse(Path(pristine_parser_file).read_text())  # raises if invalid


class TestP59ApplicationChainedOnP27:
    def test_apply_after_p27_uses_variant_c(self, pristine_parser_file):
        from sndr.kernel.text_patch import TextPatchResult
        _apply_p27(pristine_parser_file)
        patcher = _make_p59_patcher(pristine_parser_file, "CHAIN")
        result, failure = patcher.apply()
        assert result == TextPatchResult.APPLIED, failure
        assert "p59_wrap_think_return_p27_chain" in patcher.applied_sub_patches
        assert "p59_wrap_think_return_pristine" not in patcher.applied_sub_patches
        modified = Path(pristine_parser_file).read_text()
        # Both patches' effects must coexist at the wrapped return site.
        assert "_genesis_before_think" in modified
        assert "self._split_embedded_tool_calls(reasoning, content or None)" in modified

    def test_modified_chained_file_parses_as_python(self, pristine_parser_file):
        import ast
        from sndr.kernel.text_patch import TextPatchResult
        _apply_p27(pristine_parser_file)
        patcher = _make_p59_patcher(pristine_parser_file, "CHAINPARSE")
        result, _ = patcher.apply()
        assert result == TextPatchResult.APPLIED
        ast.parse(Path(pristine_parser_file).read_text())


class TestP59RequireAtLeastOneWrap:
    """Plan §1b item (d): the patch must NOT report applied when none of
    the core </think>-present wrap variants matched. Pre-fix behavior
    false-reported "applied" with only the dead residue variants A/B in
    the sub-patch list (helper injected as dead code)."""

    def _patch_runtime(self, monkeypatch, p59, target: str, tmp_path):
        import sndr.dispatcher as dispatcher
        monkeypatch.setattr(dispatcher, "should_apply", lambda pid: (True, "test"))
        monkeypatch.setattr(dispatcher, "log_decision", lambda *a, **k: None)
        monkeypatch.setattr(p59, "vllm_install_root", lambda: str(tmp_path))
        monkeypatch.setattr(p59, "resolve_vllm_file", lambda rel: target)

    def test_apply_reports_failed_when_no_core_wrap_matched(
        self, monkeypatch, tmp_path
    ):
        p59 = _p59_module()
        # Simulate wrap-site drift: the </think>-present return changed
        # shape so neither variant C nor D matches, while the required
        # import/regex/method anchors are still intact.
        drifted = SYNTHETIC_PRISTINE_FILE.replace(
            "            return reasoning, content or None\n",
            "            return reasoning, content if content else None\n",
        )
        assert drifted != SYNTHETIC_PRISTINE_FILE
        target = tmp_path / "qwen3_reasoning_parser.py"
        target.write_text(drifted)
        self._patch_runtime(monkeypatch, p59, str(target), tmp_path)
        status, reason = p59.apply()
        assert status == "failed"
        assert "wrap" in reason

    def test_apply_reports_applied_on_pristine_shape(self, monkeypatch, tmp_path):
        p59 = _p59_module()
        target = tmp_path / "qwen3_reasoning_parser.py"
        target.write_text(SYNTHETIC_PRISTINE_FILE)
        self._patch_runtime(monkeypatch, p59, str(target), tmp_path)
        status, reason = p59.apply()
        assert status == "applied", reason


class TestP59Idempotency:
    def test_second_apply_is_idempotent(self, pristine_parser_file):
        from sndr.kernel.text_patch import TextPatchResult
        patcher = _make_p59_patcher(pristine_parser_file, "IDEMP")
        r1, _ = patcher.apply()
        r2, _ = patcher.apply()
        assert r1 == TextPatchResult.APPLIED
        assert r2 == TextPatchResult.IDEMPOTENT


class TestP59UpstreamDriftDetection:
    def test_skip_when_upstream_typed_helper_present(self, tmp_path):
        # Upstream's #39055 helper carries TYPED signatures (verified via
        # `gh pr diff 39055` on 2026-06-11); our backport injects untyped
        # ones. A file containing the typed shape means #39055 merged.
        from sndr.kernel.text_patch import TextPatchResult
        post_fix = tmp_path / "post_fix_parser.py"
        post_fix.write_text(
            "# upstream merged #39055\n"
            "        def _collect_or_keep(match: re.Match[str]) -> str:\n"
            "            return ''\n"
        )
        patcher = _make_p59_patcher(str(post_fix), "DRIFT")
        result, failure = patcher.apply()
        assert result == TextPatchResult.SKIPPED
        assert failure.reason == "upstream_merged"

    def test_drift_markers_disjoint_from_emitted_text(self):
        # Self-collision rule (§6 of the 2026-06-11 triage plan): no
        # upstream drift marker may be a substring of text this patch
        # itself writes (replacements or the idempotency marker line).
        p59 = _p59_module()
        emitted = "\n".join([
            p59.IMPORT_NEW,
            p59.REGEX_NEW,
            p59.METHOD_NEW,
            p59.RETURN_THINK_P27_CHAIN_NEW,
            p59.RETURN_THINK_PRISTINE_NEW,
            p59.RETURN_TRUNC_NEW,
            f"# [Genesis wiring marker: {p59.GENESIS_P59_MARKER}]",
        ])
        for marker in p59.UPSTREAM_DRIFT_MARKERS:
            assert marker not in emitted, (
                f"drift marker {marker!r} collides with P59's own emitted "
                "text — would self-defeat (PN353A bug class)"
            )


class TestP59OptIn:
    def test_apply_skips_without_env_flag(self, monkeypatch):
        monkeypatch.delenv("GENESIS_ENABLE_P59_QWEN3_TOOL_RECOVERY", raising=False)
        p59 = _p59_module()
        status, reason = p59.apply()
        assert status == "skipped"
        assert "opt-in" in reason

    def test_env_flag_truthy_returns_true(self, monkeypatch):
        monkeypatch.setenv("GENESIS_ENABLE_P59_QWEN3_TOOL_RECOVERY", "1")
        p59 = _p59_module()
        assert p59._is_enabled() is True
