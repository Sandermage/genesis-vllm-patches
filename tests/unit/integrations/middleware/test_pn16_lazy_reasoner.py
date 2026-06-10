# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PN16 — lazy-reasoner middleware policy.

Tests the decision logic in `sndr.engines.vllm.middleware.lazy_reasoner`
without touching the text-patch wiring (covered by the wiring's anchor
invariants in a separate test).

What we cover:
  - Pre-decision heuristic (variant 1): short prompt + no tools + no
    schema + no reasoning signals → disable thinking
  - Client override (variant 3): explicit True/False respected
  - Reasoning signal patterns: math, code fence, CoT keywords
  - Stats counters move correctly per branch
  - Master env gate (default OFF)
  - Defensive against pydantic-frozen request models (object.__setattr__
    fallback)
  - Edge cases: empty messages, content-parts list, missing fields
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from sndr.engines.vllm.middleware import lazy_reasoner as lr


# ─── Helpers to build fake-request objects ──────────────────────────────


def _make_request(*, messages=None, tools=None, chat_template_kwargs=None,
                  response_format=None, max_tokens=None) -> SimpleNamespace:
    """Mimic ChatCompletionRequest shape with mutable attributes."""
    return SimpleNamespace(
        messages=messages or [],
        tools=tools,
        chat_template_kwargs=chat_template_kwargs,
        response_format=response_format,
        max_tokens=max_tokens,
    )


def _user_msg(text: str) -> SimpleNamespace:
    return SimpleNamespace(role="user", content=text)


def _assistant_msg(text: str) -> SimpleNamespace:
    return SimpleNamespace(role="assistant", content=text)


@pytest.fixture(autouse=True)
def reset_stats():
    """Each test starts with clean counters."""
    lr.reset_stats()
    yield
    lr.reset_stats()


@pytest.fixture
def env_pn16_on(monkeypatch):
    monkeypatch.setenv("GENESIS_ENABLE_PN16_LAZY_REASONER", "1")
    yield


@pytest.fixture
def env_pn16_classifier_active(monkeypatch):
    """Enable PN16 + force the classifier to actually run (V7 cap > 0
    OR V1-legacy on). Wave 7 perf optimization 2026-05-09: the classifier
    is skipped when neither V7 nor V1-legacy is configured to act on its
    decision. Tests that need to observe classifier behaviour should use
    this fixture instead of bare ``env_pn16_on``."""
    monkeypatch.setenv("GENESIS_ENABLE_PN16_LAZY_REASONER", "1")
    monkeypatch.setenv("GENESIS_PN16_CLASSIFIER_MAX_TOKENS", "200")
    yield


@pytest.fixture
def env_pn16_off(monkeypatch):
    monkeypatch.delenv("GENESIS_ENABLE_PN16_LAZY_REASONER", raising=False)
    yield


# ─── Master gate ────────────────────────────────────────────────────────


class TestMasterGate:
    def test_default_off_no_mutation(self, env_pn16_off):
        req = _make_request(messages=[_user_msg("hi")])
        lr.apply_hook(None, req)
        # No env → no stats movement, no mutation
        assert lr.get_stats()["total_requests"] == 0
        assert req.chat_template_kwargs is None

    def test_env_on_increments_total(self, env_pn16_on):
        req = _make_request(messages=[_user_msg("hi")])
        lr.apply_hook(None, req)
        assert lr.get_stats()["total_requests"] == 1


# ─── Variant 3 — client explicit override ───────────────────────────────


class TestClientOverride:
    def test_explicit_thinking_on_respected(self, env_pn16_on):
        req = _make_request(
            messages=[_user_msg("hi")],
            chat_template_kwargs={"enable_thinking": True},
        )
        lr.apply_hook(None, req)
        assert req.chat_template_kwargs == {"enable_thinking": True}
        assert lr.get_stats()["respect_explicit_on"] == 1
        assert lr.get_stats()["disabled_by_heuristic"] == 0

    def test_explicit_thinking_off_respected(self, env_pn16_on):
        req = _make_request(
            messages=[_user_msg("hi")],
            chat_template_kwargs={"enable_thinking": False},
        )
        lr.apply_hook(None, req)
        assert req.chat_template_kwargs == {"enable_thinking": False}
        assert lr.get_stats()["respect_explicit_off"] == 1
        assert lr.get_stats()["disabled_by_heuristic"] == 0

    def test_no_override_runs_heuristic_no_v1_default(self, env_pn16_on):
        """v2: V1 is OPT-IN only — by default, classifier hits don't
        mutate chat_template_kwargs. This is the cache-safe default.

        Wave 7 perf optimization 2026-05-09: when neither V7 nor
        V1-legacy is configured, the classifier is skipped entirely
        (its decision would have no consumer). So
        ``left_on_by_heuristic`` stays at 0 in this default mode.
        """
        req = _make_request(messages=[_user_msg("hi")])
        lr.apply_hook(None, req)
        assert req.chat_template_kwargs is None
        assert lr.get_stats()["disabled_by_heuristic"] == 0
        # Optimization: classifier short-circuited, no stat increment
        assert lr.get_stats()["left_on_by_heuristic"] == 0


# ─── Variant 1 — pre-decision heuristic ─────────────────────────────────


class TestPreDecisionHeuristic:
    def test_short_trivial_prompt_no_default_mutation(self, env_pn16_on):
        """v2: classifier may identify short trivial prompt, but with
        neither V1-legacy nor V7 configured, no mutation happens."""
        req = _make_request(messages=[_user_msg("Hello!")])
        lr.apply_hook(None, req)
        # v2 default: NO mutation
        assert req.chat_template_kwargs is None
        assert req.max_tokens is None

    def test_long_prompt_keeps_thinking(
        self, env_pn16_classifier_active, monkeypatch,
    ):
        """Classifier runs (V7 active) and confirms long prompt keeps thinking."""
        monkeypatch.setenv("GENESIS_PN16_THRESHOLD_CHARS", "100")
        long_text = "x" * 200
        req = _make_request(messages=[_user_msg(long_text)])
        lr.apply_hook(None, req)
        # Long prompt — heuristic leaves thinking ALONE (no mutation)
        assert req.chat_template_kwargs is None
        assert lr.get_stats()["left_on_by_heuristic"] == 1

    def test_tools_attached_keeps_thinking(self, env_pn16_classifier_active):
        req = _make_request(
            messages=[_user_msg("hi")],
            tools=[{"type": "function", "function": {"name": "x"}}],
        )
        lr.apply_hook(None, req)
        assert req.chat_template_kwargs is None
        assert lr.get_stats()["left_on_by_heuristic"] == 1

    def test_json_schema_response_format_keeps_thinking(self, env_pn16_on):
        rf = SimpleNamespace(type="json_schema")
        req = _make_request(
            messages=[_user_msg("hi")],
            response_format=rf,
        )
        lr.apply_hook(None, req)
        assert req.chat_template_kwargs is None

    def test_math_keyword_keeps_thinking(self, env_pn16_classifier_active):
        req = _make_request(messages=[_user_msg("Calculate 7919 prime")])
        lr.apply_hook(None, req)
        assert req.chat_template_kwargs is None
        assert lr.get_stats()["left_on_by_heuristic"] == 1

    def test_code_fence_keeps_thinking(self, env_pn16_on):
        req = _make_request(messages=[_user_msg("fix ```py\nprint(1)\n```")])
        lr.apply_hook(None, req)
        assert req.chat_template_kwargs is None

    def test_arithmetic_keeps_thinking(self, env_pn16_on):
        req = _make_request(messages=[_user_msg("what is 2 + 2")])
        lr.apply_hook(None, req)
        assert req.chat_template_kwargs is None

    def test_step_by_step_keeps_thinking(self, env_pn16_on):
        req = _make_request(messages=[_user_msg("explain step by step")])
        lr.apply_hook(None, req)
        assert req.chat_template_kwargs is None


# ─── Reasoning-signal pattern coverage ──────────────────────────────────


class TestReasoningSignals:
    @pytest.mark.parametrize("text,expected", [
        ("Calculate 5+5", True),
        ("solve x^2 = 4", True),
        ("Is 7 a prime?", True),
        ("```python\nx=1\n```", True),
        ("class Foo:", True),
        ("step-by-step explanation", True),
        ("explain why this happens", True),
        ("Hi how are you", False),
        ("Tell me about cats", False),
        ("Hello world", False),
        ("$x^2 + y^2 = z^2$", True),  # latex
        ("derive the formula", True),
        ("optimize the algorithm", True),
    ])
    def test_signal_detection(self, text, expected):
        assert lr._has_reasoning_signal(text) == expected, (
            f"{text!r} expected signal={expected} but got {not expected}"
        )


# ─── Total chars + content-parts handling ───────────────────────────────


class TestContentExtraction:
    def test_string_content(self):
        req = _make_request(messages=[_user_msg("hello world")])
        assert lr._total_chars(req) == 11

    def test_content_parts_list(self):
        """Multipart content joined with '\n' — char count includes the
        separator (acceptable: 1 char per gap is negligible for threshold)."""
        msg = SimpleNamespace(role="user", content=[
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ])
        req = _make_request(messages=[msg])
        # join("\n") adds one separator between parts → +1 per gap
        expected = len("first") + len("\n") + len("second")
        assert lr._total_chars(req) == expected

    def test_empty_messages(self):
        req = _make_request(messages=[])
        assert lr._total_chars(req) == 0
        last = lr._last_user_text(req)
        assert last == ""

    def test_multiple_messages_summed(self):
        req = _make_request(messages=[
            _user_msg("aaa"),
            _assistant_msg("bbbbb"),
            _user_msg("ccccccc"),
        ])
        assert lr._total_chars(req) == 3 + 5 + 7

    def test_last_user_text_skips_assistant(self):
        req = _make_request(messages=[
            _user_msg("first user"),
            _assistant_msg("assistant reply"),
            _user_msg("second user"),
        ])
        assert lr._last_user_text(req) == "second user"

    def test_dict_message_shape(self):
        """Some clients pass dicts directly instead of pydantic models."""
        req = _make_request(messages=[
            {"role": "user", "content": "from dict"},
        ])
        assert lr._total_chars(req) == len("from dict")
        assert lr._last_user_text(req) == "from dict"


# ─── Threshold env override ─────────────────────────────────────────────


class TestThresholdConfig:
    def test_threshold_default(self, monkeypatch):
        monkeypatch.delenv("GENESIS_PN16_THRESHOLD_CHARS", raising=False)
        assert lr._threshold_chars() == 300

    def test_threshold_env_override(self, monkeypatch):
        monkeypatch.setenv("GENESIS_PN16_THRESHOLD_CHARS", "1000")
        assert lr._threshold_chars() == 1000

    def test_threshold_invalid_falls_back(self, monkeypatch):
        monkeypatch.setenv("GENESIS_PN16_THRESHOLD_CHARS", "not a number")
        assert lr._threshold_chars() == 300

    def test_short_under_custom_threshold_no_default_mutation(
        self, env_pn16_on, monkeypatch,
    ):
        """v2 default: classifier hits but no variant configured to act
        → no mutation."""
        monkeypatch.setenv("GENESIS_PN16_THRESHOLD_CHARS", "20")
        req = _make_request(messages=[_user_msg("Hi!")])
        lr.apply_hook(None, req)
        assert req.chat_template_kwargs is None
        assert req.max_tokens is None


# ─── Variant 5 — prompt-engineering soft cap ───────────────────────────


class TestVariant5SoftCap:
    def test_max_thinking_tokens_default_zero(self, monkeypatch):
        monkeypatch.delenv("GENESIS_PN16_MAX_THINKING_TOKENS", raising=False)
        assert lr._max_thinking_tokens() == 0

    def test_max_thinking_tokens_env_read(self, monkeypatch):
        monkeypatch.setenv("GENESIS_PN16_MAX_THINKING_TOKENS", "200")
        assert lr._max_thinking_tokens() == 200

    def test_no_cap_no_hint_injection(self, env_pn16_classifier_active, monkeypatch):
        """When cap=0 (default), no soft-cap hint is injected.

        Wave 7: classifier short-circuits unless V7 or V1-legacy is on,
        so use ``env_pn16_classifier_active`` to actually exercise the
        classifier branch."""
        monkeypatch.setenv("GENESIS_PN16_THRESHOLD_CHARS", "10")
        monkeypatch.delenv("GENESIS_PN16_MAX_THINKING_TOKENS", raising=False)
        long_text = "calculate the sum of digits"
        req = _make_request(messages=[_user_msg(long_text)])
        lr.apply_hook(None, req)
        # Heuristic kept thinking on, but cap=0 → no hint
        assert req.messages[0].content == long_text  # unchanged
        assert lr.get_stats()["soft_cap_hint_injected"] == 0
        assert lr.get_stats()["left_on_by_heuristic"] == 1

    def test_cap_set_hint_injected_into_last_user_msg(
        self, env_pn16_on, monkeypatch,
    ):
        """When cap > 0 AND thinking is allowed, hint is appended to
        last user message."""
        monkeypatch.setenv("GENESIS_PN16_THRESHOLD_CHARS", "10")
        monkeypatch.setenv("GENESIS_PN16_MAX_THINKING_TOKENS", "200")
        long_text = "calculate the sum of digits"
        req = _make_request(messages=[_user_msg(long_text)])
        lr.apply_hook(None, req)
        # Hint should be appended to user message content
        assert long_text in req.messages[0].content
        assert "Genesis hint" in req.messages[0].content
        assert "200" in req.messages[0].content  # token count interpolated
        assert lr.get_stats()["soft_cap_hint_injected"] == 1

    def test_cap_appends_to_LAST_user_message_only(
        self, env_pn16_on, monkeypatch,
    ):
        """If multiple user messages, hint goes to the LAST one."""
        monkeypatch.setenv("GENESIS_PN16_THRESHOLD_CHARS", "10")
        monkeypatch.setenv("GENESIS_PN16_MAX_THINKING_TOKENS", "150")
        req = _make_request(messages=[
            _user_msg("first user message about reasoning"),
            _assistant_msg("assistant reply"),
            _user_msg("second user message about derive"),
        ])
        lr.apply_hook(None, req)
        # First user msg unchanged
        assert "Genesis hint" not in req.messages[0].content
        # Last user msg has hint
        assert "Genesis hint" in req.messages[2].content
        assert "150" in req.messages[2].content

    def test_cap_with_no_user_message_skips_safely(
        self, env_pn16_on, monkeypatch, caplog,
    ):
        """If there is no user message, hint injection skips gracefully."""
        monkeypatch.setenv("GENESIS_PN16_THRESHOLD_CHARS", "10")
        monkeypatch.setenv("GENESIS_PN16_MAX_THINKING_TOKENS", "200")
        # Only assistant message — no user message
        req = _make_request(messages=[
            _assistant_msg("just an assistant message about derive math"),
        ])
        # No exception, no hint injected
        lr.apply_hook(None, req)
        assert lr.get_stats()["soft_cap_hint_injected"] == 0

    def test_cap_with_dict_message(self, env_pn16_on, monkeypatch):
        """Dict-shaped messages also accept hint injection."""
        monkeypatch.setenv("GENESIS_PN16_THRESHOLD_CHARS", "10")
        monkeypatch.setenv("GENESIS_PN16_MAX_THINKING_TOKENS", "200")
        req = _make_request(messages=[
            {"role": "user", "content": "calculate something here"},
        ])
        lr.apply_hook(None, req)
        assert "Genesis hint" in req.messages[0]["content"]
        assert lr.get_stats()["soft_cap_hint_injected"] == 1

    def test_cap_with_content_parts_list(self, env_pn16_on, monkeypatch):
        """Content-parts lists get a new text part appended."""
        monkeypatch.setenv("GENESIS_PN16_THRESHOLD_CHARS", "10")
        monkeypatch.setenv("GENESIS_PN16_MAX_THINKING_TOKENS", "200")
        msg = SimpleNamespace(role="user", content=[
            {"type": "text", "text": "calculate something here"},
        ])
        req = _make_request(messages=[msg])
        lr.apply_hook(None, req)
        # Original part preserved + hint added as new part
        assert len(msg.content) == 2
        assert msg.content[0]["text"] == "calculate something here"
        assert "Genesis hint" in msg.content[1]["text"]
        assert lr.get_stats()["soft_cap_hint_injected"] == 1

    def test_cap_injects_when_v1_legacy_off_and_short_prompt(
        self, env_pn16_on, monkeypatch,
    ):
        """v2: with V1-legacy OFF (default), short trivial prompts
        keep thinking ON, so V5 hint DOES inject when configured."""
        monkeypatch.delenv("GENESIS_PN16_THRESHOLD_CHARS", raising=False)
        monkeypatch.setenv("GENESIS_PN16_MAX_THINKING_TOKENS", "200")
        monkeypatch.delenv("GENESIS_PN16_V1_LEGACY", raising=False)
        req = _make_request(messages=[_user_msg("Hi!")])
        lr.apply_hook(None, req)
        # V1-legacy OFF → no template mutation
        assert req.chat_template_kwargs is None
        # V5 hint injected because thinking remains on
        assert "Genesis hint" in req.messages[0].content
        assert lr.get_stats()["soft_cap_hint_injected"] == 1

    def test_cap_does_NOT_inject_when_v1_legacy_disabled_thinking(
        self, env_pn16_on, monkeypatch,
    ):
        """When V1-legacy IS opted in and disables thinking, V5 must
        not also inject a hint (would be wasted tokens)."""
        monkeypatch.delenv("GENESIS_PN16_THRESHOLD_CHARS", raising=False)
        monkeypatch.setenv("GENESIS_PN16_MAX_THINKING_TOKENS", "200")
        monkeypatch.setenv("GENESIS_PN16_V1_LEGACY", "1")
        req = _make_request(messages=[_user_msg("Hi!")])
        lr.apply_hook(None, req)
        # V1-legacy fired
        assert req.chat_template_kwargs == {"enable_thinking": False}
        # V5 must NOT also fire
        assert "Genesis hint" not in req.messages[0].content
        assert lr.get_stats()["soft_cap_hint_injected"] == 0
        assert lr.get_stats()["disabled_by_heuristic"] == 1

    def test_cap_does_NOT_inject_when_explicit_client_choice(
        self, env_pn16_on, monkeypatch,
    ):
        """Variant 3 (explicit client) wins — no variant-5 hint either."""
        monkeypatch.setenv("GENESIS_PN16_THRESHOLD_CHARS", "10")
        monkeypatch.setenv("GENESIS_PN16_MAX_THINKING_TOKENS", "200")
        req = _make_request(
            messages=[_user_msg("calculate something here")],
            chat_template_kwargs={"enable_thinking": True},
        )
        lr.apply_hook(None, req)
        # User message unchanged
        assert "Genesis hint" not in req.messages[0].content
        assert lr.get_stats()["soft_cap_hint_injected"] == 0


# V4 LogitsProcessor strict-cap path was removed 2026-05-05 (upstream-blocked
# by vllm v1's spec-decode + custom-logitsprocs restriction). The previous
# `TestUpstreamBlockerWarning` class lived here and verified the one-shot
# warning was emitted; both the warning function and its tests are gone now.
# When variant 5 (soft cap) runs at cap > 0, it does so silently — the
# module docstring is the canonical documentation of why V4 is deferred.


# ─── Variant 7 — max_tokens hard cap (cache-safe replacement for V1) ────


class TestVariant7MaxTokensCap:
    def test_default_zero_no_cap(self, monkeypatch):
        monkeypatch.delenv("GENESIS_PN16_CLASSIFIER_MAX_TOKENS", raising=False)
        assert lr._classifier_max_tokens() == 0

    def test_env_read(self, monkeypatch):
        monkeypatch.setenv("GENESIS_PN16_CLASSIFIER_MAX_TOKENS", "200")
        assert lr._classifier_max_tokens() == 200

    def test_v7_caps_max_tokens_when_classifier_hits(
        self, env_pn16_on, monkeypatch,
    ):
        """v2 V7: short trivial prompt + V7 cap configured → request's
        max_tokens is clamped, but chat_template_kwargs UNTOUCHED."""
        monkeypatch.setenv("GENESIS_PN16_CLASSIFIER_MAX_TOKENS", "200")
        req = _make_request(messages=[_user_msg("hi")])
        lr.apply_hook(None, req)
        # V7 capped max_tokens
        assert req.max_tokens == 200
        # CACHE-SAFETY: chat_template_kwargs NOT mutated
        assert req.chat_template_kwargs is None
        assert lr.get_stats()["max_tokens_capped"] == 1

    def test_v7_does_not_lower_already_tighter_max_tokens(
        self, env_pn16_on, monkeypatch,
    ):
        """If client already requested fewer tokens than the cap, leave
        it alone (don't INCREASE)."""
        monkeypatch.setenv("GENESIS_PN16_CLASSIFIER_MAX_TOKENS", "500")
        req = _make_request(
            messages=[_user_msg("hi")],
            max_tokens=100,  # client wants only 100
        )
        lr.apply_hook(None, req)
        # client's tighter cap respected
        assert req.max_tokens == 100
        assert lr.get_stats()["max_tokens_capped"] == 0

    def test_v7_lowers_higher_max_tokens(
        self, env_pn16_on, monkeypatch,
    ):
        """If client requested more than cap, V7 lowers it."""
        monkeypatch.setenv("GENESIS_PN16_CLASSIFIER_MAX_TOKENS", "200")
        req = _make_request(
            messages=[_user_msg("hi")],
            max_tokens=4096,
        )
        lr.apply_hook(None, req)
        assert req.max_tokens == 200
        assert lr.get_stats()["max_tokens_capped"] == 1

    def test_v7_does_not_fire_on_long_prompts(
        self, env_pn16_on, monkeypatch,
    ):
        """Classifier rejects long prompts → V7 doesn't fire even if
        configured."""
        monkeypatch.setenv("GENESIS_PN16_CLASSIFIER_MAX_TOKENS", "200")
        monkeypatch.setenv("GENESIS_PN16_THRESHOLD_CHARS", "50")
        long_text = "x" * 200
        req = _make_request(
            messages=[_user_msg(long_text)],
            max_tokens=4096,
        )
        lr.apply_hook(None, req)
        # Long prompt → classifier said "keep thinking" → V7 skipped
        assert req.max_tokens == 4096
        assert lr.get_stats()["max_tokens_capped"] == 0

    def test_v7_skipped_for_tool_requests(
        self, env_pn16_on, monkeypatch,
    ):
        """V7 must NOT fire on tool-using requests (they need full
        token budget for tool args)."""
        monkeypatch.setenv("GENESIS_PN16_CLASSIFIER_MAX_TOKENS", "200")
        req = _make_request(
            messages=[_user_msg("hi")],
            tools=[{"type": "function", "function": {"name": "x"}}],
            max_tokens=4096,
        )
        lr.apply_hook(None, req)
        # Tools attached → classifier said "keep thinking" → V7 skipped
        assert req.max_tokens == 4096
        assert lr.get_stats()["max_tokens_capped"] == 0


# ─── Variant 8 — tool-presence think-budget system message (cache-safe) ─


class TestVariant8ToolBudget:
    def test_default_zero_no_op(self, monkeypatch):
        monkeypatch.delenv("GENESIS_PN16_TOOL_THINK_BUDGET", raising=False)
        assert lr._tool_think_budget() == 0

    def test_env_read(self, monkeypatch):
        monkeypatch.setenv("GENESIS_PN16_TOOL_THINK_BUDGET", "200")
        assert lr._tool_think_budget() == 200

    def test_v8_skipped_when_no_tools(self, env_pn16_on, monkeypatch):
        """V8 only fires for tool-attached requests."""
        monkeypatch.setenv("GENESIS_PN16_TOOL_THINK_BUDGET", "200")
        req = _make_request(messages=[_user_msg("hi")], tools=None)
        lr.apply_hook(None, req)
        # No system message inserted
        assert all(
            "[Genesis-PN16-V8]" not in str(getattr(m, "content", ""))
            for m in req.messages
        )
        assert lr.get_stats()["tool_budget_prepended"] == 0

    def test_v8_inserts_system_msg_when_tools_present_no_existing_system(
        self, env_pn16_on, monkeypatch,
    ):
        monkeypatch.setenv("GENESIS_PN16_TOOL_THINK_BUDGET", "200")
        req = _make_request(
            messages=[_user_msg("What's the weather in Paris?")],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )
        lr.apply_hook(None, req)
        # New system message inserted at position 0
        assert len(req.messages) == 2
        first = req.messages[0]
        first_role = (
            getattr(first, "role", None)
            if not isinstance(first, dict)
            else first.get("role")
        )
        first_content = (
            getattr(first, "content", None)
            if not isinstance(first, dict)
            else first.get("content")
        )
        assert first_role == "system"
        assert "[Genesis-PN16-V8]" in first_content
        assert "200" in first_content
        assert lr.get_stats()["tool_budget_prepended"] == 1

    def test_v8_appends_to_existing_system_msg(
        self, env_pn16_on, monkeypatch,
    ):
        monkeypatch.setenv("GENESIS_PN16_TOOL_THINK_BUDGET", "150")
        sys_msg = SimpleNamespace(role="system", content="You are helpful.")
        req = _make_request(
            messages=[sys_msg, _user_msg("Weather in NYC?")],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )
        lr.apply_hook(None, req)
        # System message extended, no new message inserted
        assert len(req.messages) == 2
        assert req.messages[0].role == "system"
        assert "You are helpful." in req.messages[0].content
        assert "[Genesis-PN16-V8]" in req.messages[0].content
        assert "150" in req.messages[0].content
        assert lr.get_stats()["tool_budget_prepended"] == 1

    def test_v8_idempotent_no_double_append(
        self, env_pn16_on, monkeypatch,
    ):
        """Calling apply_hook twice on the same request must not insert
        the budget hint twice."""
        monkeypatch.setenv("GENESIS_PN16_TOOL_THINK_BUDGET", "200")
        req = _make_request(
            messages=[_user_msg("hi")],
            tools=[{"type": "function", "function": {"name": "x"}}],
        )
        lr.apply_hook(None, req)
        lr.apply_hook(None, req)
        # Only ONE prepend in stats
        assert lr.get_stats()["tool_budget_prepended"] == 1
        # And the hint appears only once across all messages
        joined_content = " ".join(
            str(getattr(m, "content", "") if not isinstance(m, dict)
                else m.get("content", ""))
            for m in req.messages
        )
        assert joined_content.count("[Genesis-PN16-V8]") == 1

    def test_v8_works_with_dict_messages(
        self, env_pn16_on, monkeypatch,
    ):
        """Dict-shaped messages also accept V8 prepend."""
        monkeypatch.setenv("GENESIS_PN16_TOOL_THINK_BUDGET", "200")
        req = _make_request(
            messages=[{"role": "user", "content": "weather in Tokyo?"}],
            tools=[{"type": "function", "function": {"name": "x"}}],
        )
        lr.apply_hook(None, req)
        # New system inserted at index 0
        assert len(req.messages) == 2
        assert req.messages[0]["role"] == "system"
        assert "[Genesis-PN16-V8]" in req.messages[0]["content"]

    def test_v8_does_not_block_v3_explicit_choice(
        self, env_pn16_on, monkeypatch,
    ):
        """V8 should run BEFORE V3 — when client set explicit choice,
        V3 still wins for thinking decision but V8 already inserted
        the budget hint."""
        monkeypatch.setenv("GENESIS_PN16_TOOL_THINK_BUDGET", "200")
        req = _make_request(
            messages=[_user_msg("hi")],
            tools=[{"type": "function", "function": {"name": "x"}}],
            chat_template_kwargs={"enable_thinking": True},
        )
        lr.apply_hook(None, req)
        # V8 fired
        assert lr.get_stats()["tool_budget_prepended"] == 1
        # V3 also fired
        assert lr.get_stats()["respect_explicit_on"] == 1
        # chat_template_kwargs still respected
        assert req.chat_template_kwargs == {"enable_thinking": True}

    def test_v8_compatible_with_v7(self, env_pn16_on, monkeypatch):
        """V8 + V7 stack: tool request short prompt → V8 inserts system
        msg, V7 caps max_tokens (V7 only fires on no-tools paths per
        classifier — verify behavior)."""
        monkeypatch.setenv("GENESIS_PN16_TOOL_THINK_BUDGET", "200")
        monkeypatch.setenv("GENESIS_PN16_CLASSIFIER_MAX_TOKENS", "300")
        # short trivial prompt + tools → V8 fires, V7 doesn't (tools attached)
        req = _make_request(
            messages=[_user_msg("hi")],
            tools=[{"type": "function", "function": {"name": "x"}}],
            max_tokens=4096,
        )
        lr.apply_hook(None, req)
        # V8 fired
        assert lr.get_stats()["tool_budget_prepended"] == 1
        # V7 did NOT cap max_tokens (tools attached → classifier kept thinking on)
        assert req.max_tokens == 4096


# ─── Variant 1 LEGACY — opt-in only via GENESIS_PN16_V1_LEGACY=1 ────────


class TestVariant1Legacy:
    def test_v1_legacy_default_off_no_mutation(self, env_pn16_on, monkeypatch):
        """v2 default: V1 path is OFF without explicit opt-in."""
        monkeypatch.delenv("GENESIS_PN16_V1_LEGACY", raising=False)
        req = _make_request(messages=[_user_msg("hi")])
        lr.apply_hook(None, req)
        assert req.chat_template_kwargs is None

    def test_v1_legacy_opt_in_mutates_template_kwargs(
        self, env_pn16_on, monkeypatch,
    ):
        """With GENESIS_PN16_V1_LEGACY=1, V1 fires (legacy behavior)."""
        monkeypatch.setenv("GENESIS_PN16_V1_LEGACY", "1")
        req = _make_request(messages=[_user_msg("hi")])
        lr.apply_hook(None, req)
        assert req.chat_template_kwargs == {"enable_thinking": False}
        assert lr.get_stats()["disabled_by_heuristic"] == 1

    def test_v1_legacy_emits_one_shot_warning(
        self, env_pn16_on, monkeypatch, caplog,
    ):
        """V1-legacy must emit a regression warning so operators see it
        in their startup logs (one-shot, not per-request spam)."""
        import logging
        monkeypatch.setenv("GENESIS_PN16_V1_LEGACY", "1")
        with caplog.at_level(logging.WARNING,
                             logger="genesis.middleware.lazy_reasoner"):
            for _ in range(3):
                req = _make_request(messages=[_user_msg("hi")])
                lr.apply_hook(None, req)
        # Warning emitted exactly once across 3 requests
        warns = [r for r in caplog.records
                 if r.levelno == logging.WARNING and "V1-legacy" in r.message]
        assert len(warns) == 1
        assert "28%" in warns[0].message  # cites documented regression
        # Stats marker reflects warning state
        assert lr.get_stats()["v1_legacy_warned"] == 1

    def test_v7_takes_precedence_over_v1_legacy_when_both_set(
        self, env_pn16_on, monkeypatch,
    ):
        """When both V7 and V1-legacy are configured, V7 caps
        max_tokens AND V1-legacy still mutates kwargs (additive). The
        operator who explicitly enabled V1-legacy is acknowledging the
        documented regression — they get the V1 mutation too."""
        monkeypatch.setenv("GENESIS_PN16_CLASSIFIER_MAX_TOKENS", "200")
        monkeypatch.setenv("GENESIS_PN16_V1_LEGACY", "1")
        req = _make_request(messages=[_user_msg("hi")], max_tokens=4096)
        lr.apply_hook(None, req)
        # V7 fired
        assert req.max_tokens == 200
        # V1-legacy also fired
        assert req.chat_template_kwargs == {"enable_thinking": False}


# ─── Stats counters ─────────────────────────────────────────────────────


class TestStatsCounters:
    def test_get_stats_returns_copy(self, env_pn16_on):
        req = _make_request(messages=[_user_msg("hi")])
        lr.apply_hook(None, req)
        stats = lr.get_stats()
        stats["total_requests"] = 99999
        assert lr.get_stats()["total_requests"] == 1, (
            "get_stats() should return a copy, not the underlying dict"
        )

    def test_reset_stats_clears(self, env_pn16_on):
        req = _make_request(messages=[_user_msg("hi")])
        lr.apply_hook(None, req)
        lr.apply_hook(None, _make_request(messages=[_user_msg("hi")]))
        assert lr.get_stats()["total_requests"] == 2
        lr.reset_stats()
        assert lr.get_stats()["total_requests"] == 0


# ─── Dispatcher integration ─────────────────────────────────────────────


class TestDispatcherIntegration:
    def test_pn16_in_registry(self):
        from sndr.dispatcher import PATCH_REGISTRY
        assert "PN16" in PATCH_REGISTRY
        meta = PATCH_REGISTRY["PN16"]
        assert meta["env_flag"] == "GENESIS_ENABLE_PN16_LAZY_REASONER"
        assert meta["default_on"] is False

    def test_dispatcher_should_apply_default_off(self, monkeypatch):
        monkeypatch.delenv("GENESIS_ENABLE_PN16_LAZY_REASONER", raising=False)
        from sndr.dispatcher import should_apply
        decision, _ = should_apply("PN16")
        assert decision is False

    def test_dispatcher_should_apply_env_on(self, monkeypatch):
        monkeypatch.setenv("GENESIS_ENABLE_PN16_LAZY_REASONER", "1")
        from sndr.dispatcher import should_apply
        decision, _ = should_apply("PN16")
        assert decision is True
