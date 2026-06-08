# SPDX-License-Identifier: Apache-2.0
"""PN73 — safe `tool_calls.arguments` string→dict normalization.

Upstream vllm/entrypoints/chat_utils.py::_postprocess_messages already
attempts to convert tool_calls.arguments from JSON string to dict (the
form chat templates expect). However the conversion is unguarded:

    if content := item["function"].get("arguments"):
        if not isinstance(content, (dict, list)):
            item["function"]["arguments"] = json.loads(content)  # ← raises 500
    else:
        item["function"]["arguments"] = {}

Failure modes that bubble up as HTTP 500 instead of being handled:

1. Client replays a tool_call from a previous turn where the model
   produced an arguments string that wasn't strict JSON (a leading
   comment, trailing comma, single quotes — common Qwen3 glitches).
2. Client sends arguments as a non-string scalar (int / bool / float)
   — json.loads raises TypeError.
3. Client sends arguments as a pre-stringified-then-re-stringified
   nested JSON (double-encoded).

PN73 wraps the conversion in try/except: on JSON-decode failure the
arguments string is kept as-is so the chat template can render it
verbatim. On TypeError (non-string scalar) the value is coerced to
its string representation and stored. Log warning so the operator
sees the malformed payload, but never 500.

Env gate: `GENESIS_ENABLE_PN73_TOOL_ARGS_SAFE_NORMALIZE=1` (default OFF).

Complements PN72 (developer role) and the upstream tool_calls
normalizer. Strictly defensive — does not change behavior on
well-formed input.
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root
from sndr.kernel import TextPatch, TextPatcher

log = logging.getLogger("genesis.wiring.pn73_tool_args_safe_normalize")

GENESIS_MARKER = "Genesis PN73 tool_calls.arguments safe normalization v11.3.0_021x"


def _enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN73_TOOL_ARGS_SAFE_NORMALIZE", "0",
    ).strip().lower() in ("1", "true", "yes", "on")


# v11.3.0 P0.2 anchor rework (commit pending): on dev371+/0.21.x
# upstream `_postprocess_messages` was refactored to extract
# `function = item.get("function")` as a local variable + add type
# validation before touching arguments. The inner block now uses
# `function["arguments"]` instead of `item["function"]["arguments"]`.
# Original anchor on dev209+g5536fc0c0 inspected 2026-05-13.
# Functional semantics identical — only variable alias changed.
# v3 (2026-06-08, archive-drift forensics): upstream rewrote the inner
# block to ``parsed = json.loads(content); function["arguments"] = parsed
# if parsed is not None else {}`` — this only handles the ``null`` case
# and still raises ``JSONDecodeError`` on malformed JSON. PN73's
# defensive value (no 500 on glitchy ``arguments`` string) is unchanged;
# anchor updated to the new two-line shape.
PN73_OLD = (
    "                # if arguments is None or empty string, set to {}\n"
    "                if content := function.get(\"arguments\"):\n"
    "                    if not isinstance(content, (dict, list)):\n"
    "                        parsed = json.loads(content)\n"
    "                        function[\"arguments\"] = parsed if parsed is not None else {}\n"
    "                else:\n"
    "                    function[\"arguments\"] = {}\n"
)
PN73_NEW = (
    "                # if arguments is None or empty string, set to {}\n"
    "                if content := function.get(\"arguments\"):\n"
    "                    if not isinstance(content, (dict, list)):\n"
    "                        # [Genesis PN73] safe normalize: on JSON-decode\n"
    "                        # failure keep the string as-is rather than 500.\n"
    "                        # Non-string scalars are coerced via str().\n"
    "                        # Preserves upstream's null→{} handling.\n"
    "                        try:\n"
    "                            if isinstance(content, str):\n"
    "                                parsed = json.loads(content)\n"
    "                            else:\n"
    "                                parsed = json.loads(str(content))\n"
    "                            function[\"arguments\"] = parsed if parsed is not None else {}\n"
    "                        except (json.JSONDecodeError, TypeError, ValueError):\n"
    "                            import logging as _g_pn73_log\n"
    "                            _g_pn73_log.getLogger(\"genesis.pn73\").warning(\n"
    "                                \"[Genesis PN73] malformed tool_call.arguments \"\n"
    "                                \"kept as-is (no 500): type=%s len=%s\",\n"
    "                                type(content).__name__,\n"
    "                                len(content) if hasattr(content, '__len__') else 'n/a',\n"
    "                            )\n"
    "                            # keep original — template can render verbatim\n"
    "                else:\n"
    "                    function[\"arguments\"] = {}\n"
)


def _make_patcher() -> TextPatcher | None:
    if not _enabled():
        return None
    target = resolve_vllm_file("entrypoints/chat_utils.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN73 tool_calls.arguments safe normalization",
        target_file=str(target),
        marker=GENESIS_MARKER,
        sub_patches=[
            TextPatch(
                name="pn73_tool_args_safe_normalize",
                anchor=PN73_OLD,
                replacement=PN73_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "Genesis PN73",
            "_g_pn73_log",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN73 text-patch. Returns (wiring_status, message)."""
    if not _enabled():
        return "skipped", "PN73 disabled (set GENESIS_ENABLE_PN73_TOOL_ARGS_SAFE_NORMALIZE=1)"
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "target file chat_utils.py not resolvable"
    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as fh:
        content = fh.read()
    if patcher.marker in content:
        return "applied", "idempotent (marker present)"
    for m in patcher.upstream_drift_markers:
        if m in content:
            return "skipped", f"drift marker {m!r} already in file"
    result, failure = patcher.apply()
    from sndr.kernel import result_to_wiring_status
    return result_to_wiring_status(
        result, failure,
        applied_message="PN73 tool_calls.arguments now safe-normalized (no 500 on malformed JSON)",
        patch_name=patcher.patch_name,
    )
