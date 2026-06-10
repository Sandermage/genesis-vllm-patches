# SPDX-License-Identifier: Apache-2.0
"""Tests for `sndr.engines.vllm.middleware.long_ctx_tool_adherence` — P68 + P69.

Contract:

  1. _env_flag normalises 1/true/yes/on (any case) → True; everything
     else → False.
  2. _get_threshold_chars honors GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS
     with min floor 1000; falls back to 50000 on parse error.
  3. _estimate_prompt_chars counts string content + multimodal text parts.
  4. _extract_tool_names handles dict + pydantic-style tool definitions.
  5. _scan_schema_for_unsupported_key detects xgrammar-blocking keys
     recursively (depth-limited at 16 to avoid cyclic $ref).
  6. _find_xgrammar_incompat_tool returns (name, key) on first hit.
  7. _build_p69_reminder includes tool names and format requirements.
  8. apply_hook respects all 3 gates (tools present, tool_choice auto,
     prompt > threshold) before applying P68/P69.
  9. apply_hook with both env flags off short-circuits early.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from sndr.engines.vllm.middleware import long_ctx_tool_adherence as lc


# ─── _env_flag ────────────────────────────────────────────────────────


class TestEnvFlag:
    @pytest.mark.parametrize("val", ["1", "true", "True", "TRUE", "yes",
                                      "YES", "on", "ON"])
    def test_truthy_values(self, monkeypatch, val):
        monkeypatch.setenv("MY_FLAG", val)
        assert lc._env_flag("MY_FLAG")

    @pytest.mark.parametrize("val", ["", "0", "false", "no", "off",
                                      "random", "2"])
    def test_falsy_values(self, monkeypatch, val):
        monkeypatch.setenv("MY_FLAG", val)
        assert not lc._env_flag("MY_FLAG")

    def test_unset_is_false(self, monkeypatch):
        monkeypatch.delenv("MY_FLAG", raising=False)
        assert not lc._env_flag("MY_FLAG")

    def test_whitespace_stripped(self, monkeypatch):
        monkeypatch.setenv("MY_FLAG", "  true  ")
        assert lc._env_flag("MY_FLAG")


# ─── _get_threshold_chars ─────────────────────────────────────────────


class TestThresholdChars:
    def test_default_50000(self, monkeypatch):
        monkeypatch.delenv("GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS",
                            raising=False)
        assert lc._get_threshold_chars() == 50000

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS", "20000")
        assert lc._get_threshold_chars() == 20000

    def test_min_floor_clamps_low_values(self, monkeypatch):
        """Threshold is clamped to >= 1000 to avoid pathological firings."""
        monkeypatch.setenv("GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS", "500")
        assert lc._get_threshold_chars() == 1000

    def test_parse_error_falls_back(self, monkeypatch):
        monkeypatch.setenv("GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS",
                            "not-a-number")
        assert lc._get_threshold_chars() == 50000


# ─── _estimate_prompt_chars ───────────────────────────────────────────


class TestEstimatePromptChars:
    def test_empty_messages(self):
        assert lc._estimate_prompt_chars([]) == 0
        assert lc._estimate_prompt_chars(None) == 0

    def test_string_content(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        assert lc._estimate_prompt_chars(msgs) == 10

    def test_multimodal_text_parts(self):
        msgs = [
            {"role": "user", "content": [
                {"type": "text", "text": "hello"},
                {"type": "image", "url": "..."},  # non-text ignored
                {"type": "text", "text": "world"},
            ]},
        ]
        assert lc._estimate_prompt_chars(msgs) == 10

    def test_skips_non_dict_content_parts(self):
        msgs = [{"content": ["not a dict", {"type": "text", "text": "ok"}]}]
        assert lc._estimate_prompt_chars(msgs) == 2


# ─── _extract_tool_names ──────────────────────────────────────────────


class TestExtractToolNames:
    def test_dict_style_tools(self):
        tools = [
            {"function": {"name": "get_weather"}},
            {"function": {"name": "search_web"}},
        ]
        assert lc._extract_tool_names(tools) == ["get_weather", "search_web"]

    def test_pydantic_style_tools(self):
        tools = [
            SimpleNamespace(function=SimpleNamespace(name="get_time")),
        ]
        assert lc._extract_tool_names(tools) == ["get_time"]

    def test_empty(self):
        assert lc._extract_tool_names([]) == []
        assert lc._extract_tool_names(None) == []

    def test_skips_unnamed_tools(self):
        tools = [{"function": {}}]  # no name key
        assert lc._extract_tool_names(tools) == []


# ─── xgrammar schema scanning ─────────────────────────────────────────


class TestScanSchemaForUnsupportedKey:
    def test_returns_none_on_clean_schema(self):
        clean = {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }
        assert lc._scan_schema_for_unsupported_key(clean) is None

    def test_detects_pattern_properties(self):
        schema = {"type": "object",
                  "patternProperties": {"^x_": {"type": "string"}}}
        assert lc._scan_schema_for_unsupported_key(schema) == "patternProperties"

    def test_detects_property_names(self):
        schema = {"type": "object", "propertyNames": {"pattern": "^abc$"}}
        assert lc._scan_schema_for_unsupported_key(schema) == "propertyNames"

    def test_detects_ref_keyword(self):
        schema = {"type": "object", "properties": {
            "child": {"$ref": "#/definitions/foo"},
        }}
        assert lc._scan_schema_for_unsupported_key(schema) == "$ref"

    def test_detects_one_of(self):
        schema = {"oneOf": [{"type": "string"}, {"type": "number"}]}
        assert lc._scan_schema_for_unsupported_key(schema) == "oneOf"

    def test_recurses_into_nested_dicts(self):
        deep = {
            "type": "object",
            "properties": {
                "outer": {"type": "object", "properties": {
                    "inner": {"oneOf": [{}]},
                }},
            },
        }
        assert lc._scan_schema_for_unsupported_key(deep) == "oneOf"

    def test_recurses_into_lists(self):
        schema = {"type": "array", "items": [
            {"type": "string"},
            {"propertyNames": {}},
        ]}
        assert lc._scan_schema_for_unsupported_key(schema) == "propertyNames"

    def test_depth_limit_avoids_cyclic_recursion(self):
        """Cyclic schema must not StackOverflow — depth-limited at 16."""
        schema = {"type": "object"}
        # Build a deeply nested chain
        cur = schema
        for _ in range(50):
            cur["nested"] = {"type": "object"}
            cur = cur["nested"]
        # Should return None without raising (depth limit kicks in)
        assert lc._scan_schema_for_unsupported_key(schema) is None


class TestFindXgrammarIncompatTool:
    def test_clean_tools_returns_none(self):
        tools = [
            {"function": {"name": "t1", "parameters": {"type": "object"}}},
            {"function": {"name": "t2", "parameters": {"type": "object"}}},
        ]
        assert lc._find_xgrammar_incompat_tool(tools) is None

    def test_returns_first_offender(self):
        tools = [
            {"function": {"name": "clean", "parameters": {"type": "object"}}},
            {"function": {"name": "bad",
                          "parameters": {"propertyNames": {}}}},
        ]
        result = lc._find_xgrammar_incompat_tool(tools)
        assert result == ("bad", "propertyNames")

    def test_handles_pydantic_style(self):
        tool = SimpleNamespace(function=SimpleNamespace(
            name="x", parameters={"oneOf": [{}]}))
        result = lc._find_xgrammar_incompat_tool([tool])
        assert result == ("x", "oneOf")

    def test_no_parameters_skipped(self):
        """Tools without parameters → no offender."""
        tools = [{"function": {"name": "t1"}}]
        assert lc._find_xgrammar_incompat_tool(tools) is None


# ─── P69 reminder builder ─────────────────────────────────────────────


class TestBuildP69Reminder:
    def test_includes_tool_names(self):
        reminder = lc._build_p69_reminder(["get_weather", "search_web"])
        assert "get_weather" in reminder
        assert "search_web" in reminder

    def test_includes_format_marker(self):
        reminder = lc._build_p69_reminder(["x"])
        assert "<tool_call>" in reminder

    def test_fallback_when_no_names(self):
        reminder = lc._build_p69_reminder([])
        assert "the provided tools" in reminder

    def test_includes_forbidden_modes(self):
        reminder = lc._build_p69_reminder(["x"])
        # All 4 documented failure modes mentioned
        assert "DO NOT respond with plain text" in reminder
        assert "Python-style" in reminder


# ─── apply_hook integration ────────────────────────────────────────────


class TestApplyHookGates:
    def test_both_flags_off_short_circuits(self, monkeypatch):
        monkeypatch.setenv("GENESIS_ENABLE_P68_AUTO_FORCE_TOOL", "0")
        monkeypatch.setenv("GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER", "0")
        req = SimpleNamespace(tools=[{"function": {"name": "x"}}],
                              tool_choice="auto",
                              messages=[{"role": "user", "content": "x" * 100000}])
        result = lc.apply_hook(serving_chat=None, request=req)
        assert not result["applied_p68"]
        assert not result["applied_p69"]
        assert "both env flags off" in result["reason"]

    def test_no_tools_skips(self, monkeypatch):
        monkeypatch.setenv("GENESIS_ENABLE_P68_AUTO_FORCE_TOOL", "1")
        req = SimpleNamespace(tools=None, tool_choice="auto",
                              messages=[{"role": "user", "content": "x" * 100000}])
        result = lc.apply_hook(None, req)
        assert "no tools" in result["reason"]
        assert not result["applied_p68"]

    def test_explicit_tool_choice_respected(self, monkeypatch):
        """User-set tool_choice != 'auto' is respected."""
        monkeypatch.setenv("GENESIS_ENABLE_P68_AUTO_FORCE_TOOL", "1")
        req = SimpleNamespace(
            tools=[{"function": {"name": "x"}}],
            tool_choice="none",  # explicit
            messages=[{"role": "user", "content": "x" * 100000}],
        )
        result = lc.apply_hook(None, req)
        assert "explicit" in result["reason"]
        assert not result["applied_p68"]

    def test_short_prompt_skips(self, monkeypatch):
        monkeypatch.setenv("GENESIS_ENABLE_P68_AUTO_FORCE_TOOL", "1")
        req = SimpleNamespace(
            tools=[{"function": {"name": "x"}}],
            tool_choice="auto",
            messages=[{"role": "user", "content": "short"}],
        )
        result = lc.apply_hook(None, req)
        assert not result["applied_p68"]
        assert "< threshold" in result["reason"]


class TestApplyHookSuccess:
    def test_p68_upgrades_tool_choice(self, monkeypatch):
        monkeypatch.setenv("GENESIS_ENABLE_P68_AUTO_FORCE_TOOL", "1")
        monkeypatch.setenv("GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS", "1000")
        req = SimpleNamespace(
            tools=[{"function": {"name": "x", "parameters": {"type": "object"}}}],
            tool_choice="auto",
            messages=[{"role": "user", "content": "x" * 5000}],
        )
        result = lc.apply_hook(None, req)
        assert result["applied_p68"]
        assert req.tool_choice == "required"

    def test_p68_skipped_on_xgrammar_incompat(self, monkeypatch):
        monkeypatch.setenv("GENESIS_ENABLE_P68_AUTO_FORCE_TOOL", "1")
        monkeypatch.setenv("GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS", "1000")
        req = SimpleNamespace(
            tools=[{"function": {"name": "bad",
                                  "parameters": {"propertyNames": {}}}}],
            tool_choice="auto",
            messages=[{"role": "user", "content": "x" * 5000}],
        )
        result = lc.apply_hook(None, req)
        assert not result["applied_p68"]
        assert "xgrammar-unsupported" in result["reason"]
        # tool_choice unchanged
        assert req.tool_choice == "auto"

    def test_p68_force_overrides_xgrammar_check(self, monkeypatch):
        monkeypatch.setenv("GENESIS_ENABLE_P68_AUTO_FORCE_TOOL", "1")
        monkeypatch.setenv("GENESIS_P68_FORCE", "1")
        monkeypatch.setenv("GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS", "1000")
        req = SimpleNamespace(
            tools=[{"function": {"name": "bad",
                                  "parameters": {"propertyNames": {}}}}],
            tool_choice="auto",
            messages=[{"role": "user", "content": "x" * 5000}],
        )
        result = lc.apply_hook(None, req)
        assert result["applied_p68"]
        assert req.tool_choice == "required"

    def test_p69_appends_reminder_to_last_user(self, monkeypatch):
        monkeypatch.delenv("GENESIS_ENABLE_P68_AUTO_FORCE_TOOL", raising=False)
        monkeypatch.setenv("GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER", "1")
        monkeypatch.setenv("GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS", "1000")
        msgs = [{"role": "user", "content": "x" * 5000}]
        req = SimpleNamespace(
            tools=[{"function": {"name": "search"}}],
            tool_choice="auto",
            messages=msgs,
        )
        result = lc.apply_hook(None, req)
        assert result["applied_p69"]
        assert "[SYSTEM REMINDER" in msgs[-1]["content"]
        assert "search" in msgs[-1]["content"]
