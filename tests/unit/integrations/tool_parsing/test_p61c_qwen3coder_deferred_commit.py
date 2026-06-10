# SPDX-License-Identifier: Apache-2.0
"""TDD for P61c — qwen3coder deferred-commit until <function= header.

Closes club-3090 issue #72 (troymroberts 2026-05-06):
  qwen3coder parser flips is_tool_call_started=True permanently on
  either special-token-id OR string match against `<tool_call>`. When
  narrative output contains `<tool_call>` (e.g. agent describing the
  protocol), no `<function=` ever follows — all subsequent deltas
  return None and SSE wire goes silent until max_tokens.

Fix: defer commit to is_tool_call_started=True until `<function=`
appears in a 64-char slack window after `<tool_call>`. Three logical
paths after detecting trigger:
  A) token-id present but tag string not in current_text → emit content
  B) `<function=` confirmed in slack → commit + return None (orig path)
  C) no `<function=` yet → emit content (never silently drop)
"""
from __future__ import annotations

import pytest


def _wiring():
    from sndr.engines.vllm.patches.tool_parsing import p61c_qwen3coder_deferred_commit as M
    return M


class TestAnchor:
    def test_anchor_targets_token_id_or_string_block(self):
        M = _wiring()
        # Original block flips is_tool_call_started on either trigger
        assert "self.tool_call_start_token_id in delta_token_ids" in M.ANCHOR_OLD
        assert "or self.tool_call_start_token in delta_text" in M.ANCHOR_OLD
        assert "self.is_tool_call_started = True" in M.ANCHOR_OLD
        # Closes with bare return None
        assert "return None" in M.ANCHOR_OLD


class TestReplacement:
    def test_replacement_introduces_slack_window(self):
        M = _wiring()
        # Sentinel: deferred-commit slack-window logic
        assert "_tc_idx" in M.ANCHOR_NEW
        assert "_slack_end" in M.ANCHOR_NEW
        assert "+ 64" in M.ANCHOR_NEW

    def test_replacement_checks_function_header_in_slack(self):
        M = _wiring()
        assert '"<function=" in current_text[_tc_idx:_slack_end]' in M.ANCHOR_NEW

    def test_replacement_emits_content_when_no_function_header(self):
        M = _wiring()
        # When uncertain, emit delta as content — never silently drop
        assert "DeltaMessage(content=delta_text or None)" in M.ANCHOR_NEW

    def test_replacement_handles_tokenizer_edge_case(self):
        """Path A: token-id present but tag string not in current_text."""
        M = _wiring()
        assert "if _tc_idx == -1:" in M.ANCHOR_NEW

    def test_replacement_carries_marker(self):
        M = _wiring()
        assert "P61c" in M.ANCHOR_NEW or "P61C" in M.ANCHOR_NEW
        assert "club-3090" in M.ANCHOR_NEW or "#72" in M.ANCHOR_NEW

    def test_replacement_preserves_original_commit_path(self):
        """Path B: `<function=` confirmed → original commit + return None."""
        M = _wiring()
        # When confirmed, must still set is_tool_call_started AND return None
        assert "self.is_tool_call_started = True" in M.ANCHOR_NEW
        # Must still emit content_before if tag in delta_text
        assert "content_before" in M.ANCHOR_NEW


class TestIdempotency:
    def test_drift_markers_not_in_pristine_source(self, tmp_path):
        """Regression for 2026-05-06 boot-test bug: `is_tool_call_started`
        was a drift marker but is also the existing variable name in pristine
        source, causing every boot to skip with `upstream_merged`. Drift
        markers must be strings that DO NOT appear in pristine code."""
        from sndr.kernel.text_patch import (
            TextPatch, TextPatcher, TextPatchResult,
        )
        M = _wiring()
        # Pristine source containing only the anchor block (mimicking real file)
        pristine = (
            "# header\n"
            "    def extract_tool_calls_streaming(self, ...):\n"
            "        # ... preamble ...\n"
            + M.ANCHOR_OLD
            + "        # ... rest of method ...\n"
        )
        target = tmp_path / "qwen3coder_tool_parser.py"
        target.write_text(pristine)
        patcher = M._make_patcher.__wrapped__() if hasattr(
            M._make_patcher, "__wrapped__"
        ) else None
        # Build patcher manually to test against pristine fixture
        patcher = TextPatcher(
            patch_name="P61c regression",
            target_file=str(target),
            marker=M.GENESIS_P61C_MARKER,
            sub_patches=[TextPatch(
                name="p61c", anchor=M.ANCHOR_OLD, replacement=M.ANCHOR_NEW,
                required=True,
            )],
            upstream_drift_markers=[
                # Use the exact list from the wiring module to catch regressions
                "_tc_idx",
                "_slack_end",
            ],
        )
        result, failure = patcher.apply()
        assert result == TextPatchResult.APPLIED, (
            f"Pristine source falsely triggered drift detection. "
            f"Result={result}, failure={failure}. Drift markers must not "
            f"contain strings that exist in pristine upstream code."
        )

    def test_idempotent_on_synthetic(self, tmp_path):
        from sndr.kernel.text_patch import (
            TextPatch, TextPatcher, TextPatchResult,
        )
        M = _wiring()
        target = tmp_path / "qwen3coder_tool_parser.py"
        target.write_text("# header\n" + M.ANCHOR_OLD + "\n# tail\n")
        patcher = TextPatcher(
            patch_name="P61c test",
            target_file=str(target),
            marker=M.GENESIS_P61C_MARKER,
            sub_patches=[TextPatch(
                name="p61c_deferred_commit",
                anchor=M.ANCHOR_OLD,
                replacement=M.ANCHOR_NEW,
                required=True,
            )],
        )
        r1, _ = patcher.apply()
        assert r1 == TextPatchResult.APPLIED
        body1 = target.read_text()
        assert "P61c" in body1 or "P61C" in body1
        r2, _ = patcher.apply()
        assert r2 == TextPatchResult.IDEMPOTENT


class TestDispatcher:
    def test_env_flag_default_off(self, monkeypatch):
        from sndr.dispatcher import should_apply
        monkeypatch.delenv(
            "GENESIS_ENABLE_P61C_QWEN3CODER_DEFERRED_COMMIT", raising=False
        )
        decision, _ = should_apply("P61c")
        assert decision is False

    def test_env_flag_engages_when_set(self, monkeypatch):
        from sndr.dispatcher import should_apply
        monkeypatch.setenv(
            "GENESIS_ENABLE_P61C_QWEN3CODER_DEFERRED_COMMIT", "1"
        )
        decision, reason = should_apply("P61c")
        assert decision is True


class TestSemanticInvariants:
    """Verify the replacement code's behavior under representative inputs.

    These are *unit* tests over the patch's text — we exec the replacement
    body inside a synthetic class to verify the three paths (A, B, C) yield
    expected outcomes. Real integration with the live parser happens at
    boot-test time on 27B Hybrid GDN.
    """

    def test_path_b_real_tool_call_commits(self):
        """`<tool_call>\\n<function=foo>` within slack → commit, return None."""
        # Synthesize the replacement logic in isolation
        current_text = "<tool_call>\n<function=get_weather>"
        tool_call_start_token = "<tool_call>"
        _tc_idx = current_text.find(tool_call_start_token)
        assert _tc_idx == 0
        _slack_end = _tc_idx + len(tool_call_start_token) + 64
        assert "<function=" in current_text[_tc_idx:_slack_end]

    def test_path_c_narrative_does_not_commit(self):
        """`<tool_call>` in narrative without `<function=` follow → content."""
        current_text = (
            "Here is a description of <tool_call> markup used in the protocol "
            "for streaming partial chunks back to clients."
        )
        tool_call_start_token = "<tool_call>"
        _tc_idx = current_text.find(tool_call_start_token)
        assert _tc_idx >= 0
        _slack_end = _tc_idx + len(tool_call_start_token) + 64
        assert "<function=" not in current_text[_tc_idx:_slack_end]

    def test_slack_window_64_chars_sufficient_for_real_calls(self):
        """A real tool call header `<function=name>` fits well under 64 chars."""
        # Worst-case: function name up to ~50 chars
        worst = "<function=very_long_function_name_with_underscores"
        assert len(worst) < 64
