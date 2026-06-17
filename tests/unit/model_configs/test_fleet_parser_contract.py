# SPDX-License-Identifier: Apache-2.0
"""Fleet validation contract locks (2026-06-11, pin 303916e93).

Source: docs/superpowers/journal/2026-06-11-fleet-validation-on-pin-303916e93.md

Finding 1 — tool/reasoning parsers are part of the validated fleet
contract. The 27B "breakthrough" container ran with NO tool parser at
all (Class-1 drift), and the 35B streamed raw tool-call XML because the
launcher had a tool parser but no reasoning parser (upstream
``parse_delta`` dead-zone: ``_in_tool_call_phase`` requires
``reasoning_ended=True`` which only the reasoning phase machine sets on
``</think>``). The ModelDef ``capabilities`` block is the source of
truth; every composed builtin preset and every emitted launch argv must
carry it.

Finding 2 — the hardware YAML prescribes ``shm_size: 8g``; the 27B
drift container ran with the 64 MB docker default and crashed NCCL the
moment ``NCCL_P2P_DISABLE=1`` was restored. No profile/preset layer may
shrink it (there is deliberately no override mechanism — this test is
the drift trap should one ever appear).
"""
from __future__ import annotations

import re
import warnings

import pytest

from sndr.model_configs.registry_v2 import (
    list_models,
    list_presets,
    load_alias,
    load_model,
    load_preset_def,
)
from sndr.model_configs.runtime_command import build_runtime_command


def _all_builtin_aliases() -> list[str]:
    return sorted(list_presets())


def _compose(alias: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return load_alias(alias)


# ─── Finding 1a: composed preset parsers == parent ModelDef parsers ────────


@pytest.mark.parametrize("alias", _all_builtin_aliases())
def test_preset_parsers_match_parent_modeldef(alias):
    """compose() must propagate capabilities.{tool_call,reasoning}_parser
    verbatim from the parent ModelDef — profiles/hardware own no parser
    fields and must never drop or shadow them."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        preset = load_preset_def(alias)
    model = load_model(preset.model)
    cfg = _compose(alias)
    assert cfg.tool_call_parser == model.capabilities.tool_call_parser, (
        f"preset {alias!r}: composed tool_call_parser="
        f"{cfg.tool_call_parser!r} != ModelDef capabilities "
        f"{model.capabilities.tool_call_parser!r}"
    )
    assert cfg.reasoning_parser == model.capabilities.reasoning_parser, (
        f"preset {alias!r}: composed reasoning_parser="
        f"{cfg.reasoning_parser!r} != ModelDef capabilities "
        f"{model.capabilities.reasoning_parser!r}"
    )
    assert cfg.enable_auto_tool_choice == (
        model.capabilities.enable_auto_tool_choice
    ), f"preset {alias!r}: enable_auto_tool_choice drifted from ModelDef"


# ─── Finding 1b: per-family parser values (fleet contract) ─────────────────


@pytest.mark.parametrize("model_id", sorted(list_models()))
def test_modeldef_parser_values_match_fleet_contract(model_id):
    """Validated fleet contract (2026-06-11; xml allowance 2026-06-14):
    - Qwen 3.6 family → tool_call_parser ∈ {qwen3_coder, qwen3_xml} +
      reasoning_parser=qwen3. Both are valid XML parsers for Qwen3.6's
      tool-call output; qwen3_xml is the engine-native streaming-robust
      variant (no Genesis-wrap dependency) adopted on the 35B 2026-06-14.
    - Gemma 4 family  → tool_call_parser=gemma4 (no reasoning parser)
    """
    model = load_model(model_id)
    caps = model.capabilities
    if model_id.startswith("qwen3.6"):
        assert caps.tool_call_parser in ("qwen3_coder", "qwen3_xml"), (
            f"{model_id}: Qwen 3.6 fleet contract requires "
            f"tool_call_parser ∈ {{qwen3_coder, qwen3_xml}}, "
            f"got {caps.tool_call_parser!r}"
        )
        assert caps.reasoning_parser == "qwen3", (
            f"{model_id}: Qwen 3.6 fleet contract requires "
            f"reasoning_parser=qwen3, got {caps.reasoning_parser!r}"
        )
    elif model_id.startswith("gemma-4"):
        assert caps.tool_call_parser == "gemma4", (
            f"{model_id}: Gemma 4 fleet contract requires "
            f"tool_call_parser=gemma4, got {caps.tool_call_parser!r}"
        )
    elif model_id.startswith("diffusiongemma"):
        # DiffusionGemma (block-diffusion Gemma 4 MoE) shares the Gemma 4
        # tool-call surface: tool_call_parser=gemma4, no reasoning parser
        # (block-diffusion denoising has no </think> reasoning split).
        assert caps.tool_call_parser == "gemma4", (
            f"{model_id}: DiffusionGemma fleet contract requires "
            f"tool_call_parser=gemma4, got {caps.tool_call_parser!r}"
        )
        assert caps.reasoning_parser is None, (
            f"{model_id}: DiffusionGemma fleet contract requires "
            f"no reasoning parser, got {caps.reasoning_parser!r}"
        )
    else:
        pytest.fail(f"unknown model family for fleet contract: {model_id!r}")


# ─── Finding 1c: qwen3_coder must never ship without a reasoning parser ────


@pytest.mark.parametrize("alias", _all_builtin_aliases())
def test_qwen3_coder_tool_parser_requires_reasoning_parser(alias):
    """Upstream parse_delta dead-zone guard: with --tool-call-parser
    qwen3_coder but NO --reasoning-parser, ``reasoning_ended`` is never
    set on streamed requests → both phases inactive → tool-call XML
    passes through as content (35B addendum, 2026-06-11). The fleet
    contract therefore forbids composing qwen3_coder without qwen3."""
    cfg = _compose(alias)
    if cfg.tool_call_parser == "qwen3_coder":
        assert cfg.reasoning_parser == "qwen3", (
            f"preset {alias!r}: tool_call_parser=qwen3_coder with "
            f"reasoning_parser={cfg.reasoning_parser!r} — streamed "
            "tool-calls would hit the parse_delta dead-zone "
            "(raw-XML passthrough)"
        )


# ─── Finding 1d: emitted launch argv carries the parser flags ──────────────


@pytest.mark.parametrize("alias", _all_builtin_aliases())
def test_launch_argv_carries_parser_flags(alias):
    """build_runtime_command() is the single argv source for compose/
    quadlet/k8s/bare-metal launchers — the parser flags must appear
    whenever the ModelDef declares them (the 2026-06-10 hand launcher
    that omitted them is exactly the drift class this traps)."""
    cfg = _compose(alias)
    argv = build_runtime_command(cfg).argv
    if cfg.tool_call_parser:
        assert "--tool-call-parser" in argv, (
            f"preset {alias!r}: --tool-call-parser missing from argv"
        )
        assert argv[argv.index("--tool-call-parser") + 1] == (
            cfg.tool_call_parser
        )
    if cfg.reasoning_parser:
        assert "--reasoning-parser" in argv, (
            f"preset {alias!r}: --reasoning-parser missing from argv"
        )
        assert argv[argv.index("--reasoning-parser") + 1] == (
            cfg.reasoning_parser
        )
    if cfg.enable_auto_tool_choice:
        assert "--enable-auto-tool-choice" in argv


# ─── Finding 2: shm_size must never shrink below the 8g hardware floor ─────


_SHM_RE = re.compile(r"^(\d+(?:\.\d+)?)([bkmg]?)$", re.IGNORECASE)
_SHM_MULT = {"": 1, "b": 1, "k": 1024, "m": 1024**2, "g": 1024**3}


def _shm_bytes(value: str) -> float:
    m = _SHM_RE.match(value.strip())
    assert m, f"unparseable shm_size value: {value!r}"
    return float(m.group(1)) * _SHM_MULT[m.group(2).lower()]


@pytest.mark.parametrize("alias", _all_builtin_aliases())
def test_composed_shm_size_at_least_8g(alias):
    """Hardware YAMLs prescribe shm_size 8g (NCCL TP shared segments).
    The 27B drift container ran at the docker 64 MB default and crashed
    NCCL on first all-reduce. ipc=host is the validated alternative
    (documented in the a5000-2x hardware YAML docker comment); the
    composed default must stay >= 8 GiB."""
    cfg = _compose(alias)
    if cfg.docker is None:
        pytest.skip(f"preset {alias!r}: no docker runtime block")
    assert cfg.docker.shm_size, (
        f"preset {alias!r}: docker.shm_size unset — container would fall "
        "back to the 64 MB docker default and crash NCCL under TP"
    )
    assert _shm_bytes(cfg.docker.shm_size) >= 8 * 1024**3, (
        f"preset {alias!r}: docker.shm_size={cfg.docker.shm_size!r} is "
        "below the validated 8g floor (hardware YAML owns this knob)"
    )
