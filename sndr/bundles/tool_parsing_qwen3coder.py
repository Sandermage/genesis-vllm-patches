# SPDX-License-Identifier: Apache-2.0
"""Bundle: qwen3coder tool-parser fixes (community tier).

Atomic apply of 4 patches that together harden the qwen3coder
tool-call streaming pipeline:

  P15  — Accept lowercase `none` alongside `null` in `_convert_param_value`
         (vllm#38996, Jinja `| string` quirk in qwen3.5+ chat template).
  P61c — Defer `is_tool_call_started=True` commit until `<function=`
         appears in 64-char slack (closes club-3090#72 — narrative
         `<tool_call>` mention causes 30-120s SSE silence).
  P64  — Remove early-return after parameter fragments + unify
         </function> emit (vllm#39598, MTP bundles last param +
         </function> in same delta). Touches 2 files: tool_parser.py
         AND entrypoints/openai/serving.py.
  PN56 — Restore prev_tool_call_arr arguments from streamed_args + "}"
         when XML parse fails (vllm#41466 — prevents leak of "{}"
         placeholder to strict OpenAI clients).

Why bundle:
  All 4 patches touch `tool_parsers/qwen3coder_tool_parser.py` (P64
  also touches serving.py). Today (pre-bundle) operator must enable
  4 separate flags AND test compatibility — the Stage 7 audit deep-
  dive showed this is error-prone (8 possible file states, only
  partially tested).

  This bundle commits all 5 patcher transactions (P15, P61c, P64-
  parser, P64-serving, PN56) atomically via MultiFilePatchTransaction.
  Either ALL apply or NONE — no partial state.

Tier:    community  (all sub-patches are upstream-PR backports)
Flag:    SNDR_ENABLE_BUNDLE_TOOL_PARSING_QWEN3CODER=1
Targets: tool_parsers/qwen3coder_tool_parser.py
         entrypoints/openai/chat_completion/serving.py

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

from vllm.sndr_core.env import Flags

from ._common import run_bundle

# Lazy imports of patch modules — done inside apply() to avoid top-level
# import chains pulling in _genesis at bundle module load time. Bundles
# may be imported by CLI tools that don't have vllm installed yet.


def apply() -> tuple[str, str]:
    """Apply qwen3coder tool-parser bundle atomically."""
    from sndr.engines.vllm.patches.tool_parsing import p15_qwen3_none_null as _p15
    from sndr.engines.vllm.patches.tool_parsing import p61c_qwen3coder_deferred_commit as _p61c

    from sndr.engines.vllm.patches.tool_parsing import p64_qwen3coder_mtp_streaming as _p64

    from sndr.engines.vllm.patches.tool_parsing import pn56_qwen3coder_xml_fallback as _pn56
    return run_bundle(
        name="tool_parsing_qwen3coder",
        umbrella_flag=Flags.BUNDLE_TOOL_PARSING_QWEN3CODER,
        tier="community",
        patcher_factories=[
            _p15._make_patcher,
            _p61c._make_patcher,
            _p64._make_qwen3cod_patcher,
            _p64._make_serving_patcher,
            _pn56._make_patcher,
        ],
    )


__all__ = ["apply"]
