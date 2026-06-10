# SPDX-License-Identifier: Apache-2.0
"""Phase B tests for PN288 — qwen3_coder tool_call finish_reason
override (§1.3 of the unified plan).

The PN288 decision logic lives in
``sndr.engines.vllm.middleware.pn288_finish_reason_override`` and is
unit-testable in isolation — the text-patch overlay
``sndr.engines.vllm.patches.serving.pn288_tool_finish_reason_override``
just wires the helper into ``OpenAIServingChat._create_chat_completion``.

These tests pin down:

  * Helper-level args-validity check.
  * Streaming + non-streaming dispatchers in 4 conditions each:
      - PN288 disabled → upstream verdict.
      - Enabled + args valid → upstream verdict (no intervention).
      - Enabled + args invalid + output.finish_reason != 'length' →
        upstream verdict.
      - Enabled + args invalid + output.finish_reason == 'length' +
        dry-run ON → log WOULD downgrade, return upstream verdict.
      - Enabled + args invalid + output.finish_reason == 'length' +
        dry-run OFF → return downgraded verdict.
  * Prometheus counters reflect each branch correctly under labeled
    (model, channel, action) tuples.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


def _import_mod():
    sys.path.insert(0, str(REPO_ROOT))
    try:
        mod = importlib.import_module(
            "sndr.engines.vllm.middleware.pn288_finish_reason_override"
        )
    finally:
        sys.path.pop(0)
    return mod


def _reset(mod) -> None:
    mod.counters.clear()


# ─── Stand-in objects ───────────────────────────────────────────────────


class _Request:
    def __init__(self, model: str = "qwen3.6-35b-a3b",
                 tool_choice: Any = "auto"):
        self.model = model
        self.tool_choice = tool_choice


class _Output:
    def __init__(self, finish_reason: str | None = "length"):
        self.finish_reason = finish_reason


class _Parser:
    """Minimal stand-in for Qwen3CoderToolParser exposing
    ``prev_tool_call_arr``."""
    def __init__(self, prev_tool_call_arr: list[dict] | None = None):
        self.prev_tool_call_arr = prev_tool_call_arr or []


_GOOD_ARGS = [{"name": "Read", "arguments": '{"file_path":"/a"}'}]
_BAD_ARGS = [{"name": "Read", "arguments": '{"file_path":"/some/lo'}]


# ─── Helper-level: _validate_tool_call_args ────────────────────────────


def test_validate_returns_true_on_no_parser():
    mod = _import_mod()
    assert mod._validate_tool_call_args(None) is True


def test_validate_returns_true_on_parser_without_prev_tool_call_arr():
    mod = _import_mod()

    class _Empty:
        pass

    assert mod._validate_tool_call_args(_Empty()) is True


def test_validate_returns_true_on_empty_or_placeholder_args():
    mod = _import_mod()
    parser = _Parser([
        {"name": "Read", "arguments": ""},
        {"name": "Edit", "arguments": "{}"},
        {"name": "Bash", "arguments": None},
    ])
    assert mod._validate_tool_call_args(parser) is True


def test_validate_returns_true_on_well_formed_json():
    mod = _import_mod()
    parser = _Parser(_GOOD_ARGS)
    assert mod._validate_tool_call_args(parser) is True


def test_validate_returns_false_on_truncated_json():
    mod = _import_mod()
    parser = _Parser(_BAD_ARGS)
    assert mod._validate_tool_call_args(parser) is False


def test_safe_model_name_handles_missing_request():
    mod = _import_mod()
    assert mod._safe_model_name(None) == "unknown"


def test_safe_model_name_handles_request_without_model_attr():
    mod = _import_mod()

    class _Naked:
        pass

    assert mod._safe_model_name(_Naked()) == "unknown"


# ─── Gate helpers ──────────────────────────────────────────────────────


def test_is_enabled_default_off(monkeypatch):
    mod = _import_mod()
    monkeypatch.delenv("GENESIS_ENABLE_PN288_TOOL_FINISH_REASON_OVERRIDE",
                       raising=False)
    assert mod.is_enabled() is False


def test_is_enabled_respects_env(monkeypatch):
    mod = _import_mod()
    monkeypatch.setenv(
        "GENESIS_ENABLE_PN288_TOOL_FINISH_REASON_OVERRIDE", "1",
    )
    assert mod.is_enabled() is True


def test_is_dry_run_default_true_when_enabled(monkeypatch):
    mod = _import_mod()
    monkeypatch.delenv("GENESIS_PN288_DRY_RUN", raising=False)
    assert mod.is_dry_run() is True


def test_is_dry_run_respects_explicit_off(monkeypatch):
    mod = _import_mod()
    monkeypatch.setenv("GENESIS_PN288_DRY_RUN", "0")
    assert mod.is_dry_run() is False


# ─── Streaming dispatcher ──────────────────────────────────────────────


def _streaming_kwargs(
    *, auto_tools_called: bool = True,
    output_finish_reason: str | None = "length",
    parser_prev_args: list[dict] | None = None,
) -> dict:
    return dict(
        auto_tools_called=auto_tools_called,
        tools_streamed_i=False,
        tool_choice_function_name=None,
        use_harmony=False,
        harmony_tools_streamed_i=False,
        output=_Output(output_finish_reason),
        request=_Request(),
        tool_parser=_Parser(parser_prev_args),
    )


def test_streaming_disabled_returns_upstream_verdict(monkeypatch):
    mod = _import_mod()
    _reset(mod)
    monkeypatch.delenv(
        "GENESIS_ENABLE_PN288_TOOL_FINISH_REASON_OVERRIDE",
        raising=False,
    )
    # Bad args + length truncation — but PN288 is OFF, so we keep upstream.
    out = mod.decide_streaming_finish_reason(
        **_streaming_kwargs(parser_prev_args=_BAD_ARGS),
    )
    assert out == "tool_calls"


def test_streaming_enabled_args_valid_returns_upstream_verdict(monkeypatch):
    mod = _import_mod()
    _reset(mod)
    monkeypatch.setenv(
        "GENESIS_ENABLE_PN288_TOOL_FINISH_REASON_OVERRIDE", "1",
    )
    out = mod.decide_streaming_finish_reason(
        **_streaming_kwargs(parser_prev_args=_GOOD_ARGS),
    )
    assert out == "tool_calls"
    # Counter records the kept-valid branch.
    assert mod.counters.get(("streaming", mod._ACTION_KEPT_VALID)) == 1


def test_streaming_enabled_args_invalid_but_not_length_keeps_upstream(
    monkeypatch,
):
    mod = _import_mod()
    _reset(mod)
    monkeypatch.setenv(
        "GENESIS_ENABLE_PN288_TOOL_FINISH_REASON_OVERRIDE", "1",
    )
    # output.finish_reason is "stop", not "length" — PN288 does NOT
    # intervene even though args are unparseable. The bug PN288 chases
    # is specifically max_tokens-truncated args, not arbitrary parse
    # failures.
    out = mod.decide_streaming_finish_reason(
        **_streaming_kwargs(
            output_finish_reason="stop",
            parser_prev_args=_BAD_ARGS,
        ),
    )
    assert out == "tool_calls"
    # No "would_downgrade" or "downgraded" recorded.
    assert ("streaming", mod._ACTION_WOULD_DOWNGRADE) not in mod.counters
    assert ("streaming", mod._ACTION_DOWNGRADED) not in mod.counters


def test_streaming_dry_run_records_would_downgrade_but_keeps_upstream(
    monkeypatch, caplog,
):
    mod = _import_mod()
    _reset(mod)
    monkeypatch.setenv(
        "GENESIS_ENABLE_PN288_TOOL_FINISH_REASON_OVERRIDE", "1",
    )
    monkeypatch.delenv("GENESIS_PN288_DRY_RUN", raising=False)  # default ON
    with caplog.at_level("WARNING",
                         logger="genesis.middleware."
                         "pn288_finish_reason_override"):
        out = mod.decide_streaming_finish_reason(
            **_streaming_kwargs(parser_prev_args=_BAD_ARGS),
        )
    # Dry-run: emit upstream verdict, just LOG the "would change".
    assert out == "tool_calls"
    assert mod.counters[("streaming", mod._ACTION_WOULD_DOWNGRADE)] == 1
    assert any("WOULD downgrade" in r.message for r in caplog.records)


def test_streaming_dry_run_off_actually_downgrades(monkeypatch):
    mod = _import_mod()
    _reset(mod)
    monkeypatch.setenv(
        "GENESIS_ENABLE_PN288_TOOL_FINISH_REASON_OVERRIDE", "1",
    )
    monkeypatch.setenv("GENESIS_PN288_DRY_RUN", "0")
    out = mod.decide_streaming_finish_reason(
        **_streaming_kwargs(parser_prev_args=_BAD_ARGS),
    )
    assert out == "length"
    assert mod.counters[("streaming", mod._ACTION_DOWNGRADED)] == 1


def test_streaming_does_not_intervene_when_upstream_says_stop(monkeypatch):
    """The PN288 trigger condition is auto_tools_called=True. When
    upstream wouldn't even emit tool_calls (auto_tools_called=False),
    we MUST pass through unchanged — there's no tool call to downgrade."""
    mod = _import_mod()
    _reset(mod)
    monkeypatch.setenv(
        "GENESIS_ENABLE_PN288_TOOL_FINISH_REASON_OVERRIDE", "1",
    )
    monkeypatch.setenv("GENESIS_PN288_DRY_RUN", "0")
    out = mod.decide_streaming_finish_reason(
        **_streaming_kwargs(
            auto_tools_called=False,
            output_finish_reason="stop",
            parser_prev_args=_BAD_ARGS,
        ),
    )
    assert out == "stop"
    assert mod.counters == {}


# ─── Non-streaming dispatcher ──────────────────────────────────────────


def _ns_kwargs(
    *, auto_tools_called: bool = True,
    output_finish_reason: str | None = "length",
    parser_prev_args: list[dict] | None = None,
) -> dict:
    return dict(
        auto_tools_called=auto_tools_called,
        request=_Request(),
        output=_Output(output_finish_reason),
        tool_parser=_Parser(parser_prev_args),
    )


def test_non_streaming_disabled_returns_upstream_bool(monkeypatch):
    mod = _import_mod()
    _reset(mod)
    monkeypatch.delenv(
        "GENESIS_ENABLE_PN288_TOOL_FINISH_REASON_OVERRIDE",
        raising=False,
    )
    out = mod.decide_non_streaming_is_tool_calls(
        **_ns_kwargs(parser_prev_args=_BAD_ARGS),
    )
    assert out is True  # auto_tools_called=True


def test_non_streaming_args_valid_keeps_upstream_true(monkeypatch):
    mod = _import_mod()
    _reset(mod)
    monkeypatch.setenv(
        "GENESIS_ENABLE_PN288_TOOL_FINISH_REASON_OVERRIDE", "1",
    )
    out = mod.decide_non_streaming_is_tool_calls(
        **_ns_kwargs(parser_prev_args=_GOOD_ARGS),
    )
    assert out is True


def test_non_streaming_dry_run_returns_upstream_true_and_records(
    monkeypatch, caplog,
):
    mod = _import_mod()
    _reset(mod)
    monkeypatch.setenv(
        "GENESIS_ENABLE_PN288_TOOL_FINISH_REASON_OVERRIDE", "1",
    )
    monkeypatch.delenv("GENESIS_PN288_DRY_RUN", raising=False)
    with caplog.at_level("WARNING",
                         logger="genesis.middleware."
                         "pn288_finish_reason_override"):
        out = mod.decide_non_streaming_is_tool_calls(
            **_ns_kwargs(parser_prev_args=_BAD_ARGS),
        )
    assert out is True
    assert mod.counters[("non_streaming",
                          mod._ACTION_WOULD_DOWNGRADE)] == 1
    assert any("WOULD downgrade" in r.message for r in caplog.records)


def test_non_streaming_dry_run_off_returns_false(monkeypatch):
    mod = _import_mod()
    _reset(mod)
    monkeypatch.setenv(
        "GENESIS_ENABLE_PN288_TOOL_FINISH_REASON_OVERRIDE", "1",
    )
    monkeypatch.setenv("GENESIS_PN288_DRY_RUN", "0")
    out = mod.decide_non_streaming_is_tool_calls(
        **_ns_kwargs(parser_prev_args=_BAD_ARGS),
    )
    assert out is False
    assert mod.counters[("non_streaming", mod._ACTION_DOWNGRADED)] == 1


# ─── Prometheus counter integration ────────────────────────────────────


def test_setup_prometheus_counter_is_idempotent(monkeypatch):
    pytest.importorskip("prometheus_client")
    mod = _import_mod()
    assert mod.setup_prometheus_counters() is True
    assert mod._prom_counter is not None
    assert mod.setup_prometheus_counters() is True  # no-op second call


def test_labeled_counter_increments_under_correct_tuple(monkeypatch):
    pytest.importorskip("prometheus_client")
    mod = _import_mod()
    _reset(mod)
    mod.setup_prometheus_counters()
    monkeypatch.setenv(
        "GENESIS_ENABLE_PN288_TOOL_FINISH_REASON_OVERRIDE", "1",
    )
    monkeypatch.delenv("GENESIS_PN288_DRY_RUN", raising=False)

    # Capture starting value for the (35B, streaming, would_downgrade)
    # series.
    starting = mod._prom_counter.labels(
        model="qwen3.6-35b-a3b",
        channel="streaming",
        action=mod._ACTION_WOULD_DOWNGRADE,
    )._value.get()

    mod.decide_streaming_finish_reason(
        **_streaming_kwargs(parser_prev_args=_BAD_ARGS),
    )

    after = mod._prom_counter.labels(
        model="qwen3.6-35b-a3b",
        channel="streaming",
        action=mod._ACTION_WOULD_DOWNGRADE,
    )._value.get()
    assert after == starting + 1


def test_labeled_counter_segregates_channels(monkeypatch):
    """Streaming and non_streaming counters track distinct time series.
    A would_downgrade on the streaming channel does NOT pollute the
    non_streaming bucket."""
    pytest.importorskip("prometheus_client")
    mod = _import_mod()
    _reset(mod)
    mod.setup_prometheus_counters()
    monkeypatch.setenv(
        "GENESIS_ENABLE_PN288_TOOL_FINISH_REASON_OVERRIDE", "1",
    )
    monkeypatch.delenv("GENESIS_PN288_DRY_RUN", raising=False)

    mod.decide_streaming_finish_reason(
        **_streaming_kwargs(parser_prev_args=_BAD_ARGS),
    )
    ns_value = mod._prom_counter.labels(
        model="qwen3.6-35b-a3b",
        channel="non_streaming",
        action=mod._ACTION_WOULD_DOWNGRADE,
    )._value.get()
    # No non_streaming event happened.
    assert ns_value == 0


# ─── Module exports surface ────────────────────────────────────────────


def test_public_exports_lock():
    mod = _import_mod()
    for name in [
        "is_enabled", "is_dry_run",
        "get_min_args_length", "get_max_args_length",
        "setup_prometheus_counters",
        "decide_streaming_finish_reason",
        "decide_non_streaming_is_tool_calls",
        "counters", "_validate_tool_call_args",
        "_args_length_in_band", "_safe_model_name",
        "_ACTION_WOULD_DOWNGRADE", "_ACTION_DOWNGRADED",
        "_ACTION_KEPT_VALID", "_ACTION_KEPT_NO_LENGTH",
        "_ACTION_KEPT_OUT_OF_RANGE",
    ]:
        assert name in mod.__all__, (
            f"{name!r} must be in __all__ for downstream stability"
        )


# ─── Phase C threshold guards ──────────────────────────────────────────


def test_get_min_max_args_length_defaults(monkeypatch):
    mod = _import_mod()
    monkeypatch.delenv("GENESIS_PN288_MIN_ARGS_LENGTH", raising=False)
    monkeypatch.delenv("GENESIS_PN288_MAX_ARGS_LENGTH", raising=False)
    assert mod.get_min_args_length() == 5
    assert mod.get_max_args_length() == 200


def test_get_args_length_respects_env(monkeypatch):
    mod = _import_mod()
    monkeypatch.setenv("GENESIS_PN288_MIN_ARGS_LENGTH", "10")
    monkeypatch.setenv("GENESIS_PN288_MAX_ARGS_LENGTH", "500")
    assert mod.get_min_args_length() == 10
    assert mod.get_max_args_length() == 500


def test_get_args_length_invalid_env_falls_back_to_default(monkeypatch):
    """Operator typos in env vars must not crash the observer."""
    mod = _import_mod()
    monkeypatch.setenv("GENESIS_PN288_MIN_ARGS_LENGTH", "not_a_number")
    monkeypatch.setenv("GENESIS_PN288_MAX_ARGS_LENGTH", "also_bad")
    assert mod.get_min_args_length() == 5
    assert mod.get_max_args_length() == 200


def test_args_length_in_band_true_for_canonical_truncated_args():
    """The PN287 evidence band (5-80 chars) is the typical truncated
    tool_call.arguments length when max_tokens hits mid-JSON-string."""
    mod = _import_mod()
    # 22 chars — exactly the PN287 evidence sample.
    parser = _Parser([{"name": "Read",
                       "arguments": '{"file_path":"/some/lo'}])
    assert mod._args_length_in_band(parser) is True


def test_args_length_in_band_false_below_min():
    mod = _import_mod()
    # 3 chars — too short, likely our probe parse-failure not a real
    # truncation.
    parser = _Parser([{"name": "Read", "arguments": '{"a'}])
    assert mod._args_length_in_band(parser) is False


def test_args_length_in_band_false_above_max():
    mod = _import_mod()
    # 250 chars — too long, likely a real structured tool call that
    # parse-failed for a different reason (we don't want to corrupt it).
    bad_long = '{"x":"' + ("y" * 240) + '"'
    parser = _Parser([{"name": "Read", "arguments": bad_long}])
    assert mod._args_length_in_band(parser) is False


def test_args_length_in_band_false_with_no_real_args():
    """Empty / placeholder args (`""` / `"{}"`) don't count as real
    truncation evidence — the band check returns False so PN288 takes
    no action."""
    mod = _import_mod()
    parser = _Parser([
        {"name": "Read", "arguments": ""},
        {"name": "Edit", "arguments": "{}"},
    ])
    assert mod._args_length_in_band(parser) is False


def test_args_length_in_band_all_must_pass(monkeypatch):
    """Mixed-length args: if ANY non-empty entry is out of band, the
    helper returns False — refuse to downgrade the whole request."""
    mod = _import_mod()
    parser = _Parser([
        {"name": "Read", "arguments": '{"x":"y'},        # 7 chars, in
        {"name": "Edit", "arguments": "x" * 500},        # 500, out
    ])
    assert mod._args_length_in_band(parser) is False


# ─── Streaming dispatcher Phase C path ────────────────────────────────


def test_streaming_phase_c_out_of_range_keeps_upstream(monkeypatch):
    """When PN288 is enabled, DRY_RUN=0 (Phase C), AND args length is
    OUT of band → upstream verdict preserved, counter records
    KEPT_OUT_OF_RANGE."""
    mod = _import_mod()
    _reset(mod)
    monkeypatch.setenv(
        "GENESIS_ENABLE_PN288_TOOL_FINISH_REASON_OVERRIDE", "1",
    )
    monkeypatch.setenv("GENESIS_PN288_DRY_RUN", "0")
    # Length 500 — outside default band.
    bad_long = [{"name": "Read", "arguments": "x" * 500}]
    out = mod.decide_streaming_finish_reason(
        **_streaming_kwargs(parser_prev_args=bad_long),
    )
    # Upstream stays — Phase C refuses to downgrade.
    assert out == "tool_calls"
    assert mod.counters[("streaming",
                          mod._ACTION_KEPT_OUT_OF_RANGE)] == 1
    assert ("streaming", mod._ACTION_DOWNGRADED) not in mod.counters


def test_streaming_phase_c_in_band_downgrades(monkeypatch):
    """Same setup as above but with in-band args — should downgrade."""
    mod = _import_mod()
    _reset(mod)
    monkeypatch.setenv(
        "GENESIS_ENABLE_PN288_TOOL_FINISH_REASON_OVERRIDE", "1",
    )
    monkeypatch.setenv("GENESIS_PN288_DRY_RUN", "0")
    out = mod.decide_streaming_finish_reason(
        **_streaming_kwargs(parser_prev_args=_BAD_ARGS),
    )
    assert out == "length"
    assert mod.counters[("streaming", mod._ACTION_DOWNGRADED)] == 1
    assert ("streaming", mod._ACTION_KEPT_OUT_OF_RANGE) not in mod.counters


def test_streaming_dry_run_out_of_range_records_correctly(monkeypatch):
    """In Phase B (dry-run): out-of-range args still surface the
    KEPT_OUT_OF_RANGE counter — operators learn the safety guard
    fired BEFORE flipping DRY_RUN=0."""
    mod = _import_mod()
    _reset(mod)
    monkeypatch.setenv(
        "GENESIS_ENABLE_PN288_TOOL_FINISH_REASON_OVERRIDE", "1",
    )
    monkeypatch.delenv("GENESIS_PN288_DRY_RUN", raising=False)
    bad_long = [{"name": "Read", "arguments": "x" * 500}]
    out = mod.decide_streaming_finish_reason(
        **_streaming_kwargs(parser_prev_args=bad_long),
    )
    assert out == "tool_calls"  # upstream stays in dry-run
    assert mod.counters[("streaming",
                          mod._ACTION_KEPT_OUT_OF_RANGE)] == 1
    # No would_downgrade event — the safety guard short-circuits BEFORE
    # the dry-run counter fires.
    assert ("streaming", mod._ACTION_WOULD_DOWNGRADE) not in mod.counters


def test_streaming_custom_band_via_env_alters_decision(monkeypatch):
    """When the operator tightens the band, Phase C declines requests
    that the default band would have downgraded. This is the key
    Phase C tuning lever — narrow the band first, observe via PN287,
    then widen."""
    mod = _import_mod()
    _reset(mod)
    monkeypatch.setenv(
        "GENESIS_ENABLE_PN288_TOOL_FINISH_REASON_OVERRIDE", "1",
    )
    monkeypatch.setenv("GENESIS_PN288_DRY_RUN", "0")
    # Tighten the band to [50, 200) — the 22-char canonical sample
    # falls below the new floor.
    monkeypatch.setenv("GENESIS_PN288_MIN_ARGS_LENGTH", "50")
    out = mod.decide_streaming_finish_reason(
        **_streaming_kwargs(parser_prev_args=_BAD_ARGS),
    )
    # 22 chars < new min (50) → kept_out_of_range, no downgrade.
    assert out == "tool_calls"
    assert mod.counters[("streaming",
                          mod._ACTION_KEPT_OUT_OF_RANGE)] == 1


# ─── Non-streaming dispatcher Phase C path ────────────────────────────


def test_non_streaming_phase_c_out_of_range_keeps_upstream(monkeypatch):
    mod = _import_mod()
    _reset(mod)
    monkeypatch.setenv(
        "GENESIS_ENABLE_PN288_TOOL_FINISH_REASON_OVERRIDE", "1",
    )
    monkeypatch.setenv("GENESIS_PN288_DRY_RUN", "0")
    bad_long = [{"name": "Read", "arguments": "x" * 500}]
    out = mod.decide_non_streaming_is_tool_calls(
        **_ns_kwargs(parser_prev_args=bad_long),
    )
    # Upstream True stays.
    assert out is True
    assert mod.counters[("non_streaming",
                          mod._ACTION_KEPT_OUT_OF_RANGE)] == 1


def test_non_streaming_phase_c_in_band_downgrades(monkeypatch):
    mod = _import_mod()
    _reset(mod)
    monkeypatch.setenv(
        "GENESIS_ENABLE_PN288_TOOL_FINISH_REASON_OVERRIDE", "1",
    )
    monkeypatch.setenv("GENESIS_PN288_DRY_RUN", "0")
    out = mod.decide_non_streaming_is_tool_calls(
        **_ns_kwargs(parser_prev_args=_BAD_ARGS),
    )
    assert out is False
    assert mod.counters[("non_streaming",
                          mod._ACTION_DOWNGRADED)] == 1
