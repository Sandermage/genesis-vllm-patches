# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Genesis G4_T1 v2 — vendored gemma4 tool-parser rewrite (PR #42237)
# ─────────────────────────────────────────────────────────────────────────
# This file is a verbatim copy of `vllm/tool_parsers/gemma4_tool_parser.py`
# from the head of `whytem/vllm` branch `codex/gemma4-hermes-style-parser`
# (vLLM upstream PR #42237, OPEN at vendor time 2026-05-31).
#
# Provenance + investigation discipline (Sander iron rule #10/#11):
#   1. STUDY:  read PR #42006 thread + PR #42237 thread end-to-end. Author
#              `whytem` filed #42006 first (segment-replay refactor of the
#              existing `_extract_streaming`, +203/-58 LOC). After reviewer
#              `bbrowning` raised concerns about subtle pre-processing bugs,
#              the author wrote a clean rewrite as #42237 (+203/-306 LOC,
#              net -103) using the Hermes/Kimi-style "scan accumulated text,
#              parse, then diff" pattern. Author quote: "I've been
#              torture-testing both approaches in Opencode and both appear
#              robust. I've observed zero tool-calling errors."
#   2. ANALYZE: the v1 vendor (G4_T1 v1 = PR #42006 = club-3090 stack) keeps
#              the original state-machine `_extract_streaming` and adds a
#              segment-replay layer on top. PR #42237's accumulated-text
#              rescan is *structurally* insensitive to MTP delta packing —
#              the same shape solved Hermes/Kimi for the same root cause
#              (multiple delimiter events in one streaming delta).
#   3. VERIFY: the v1 vendor file on our pin 626fa9bb is 967 LOC; this v2
#              vendor is 720 LOC. v2 drops a `regex` import (no third-party
#              `regex` package needed at runtime — the v1 file imports
#              `regex` which is why our G4_T1 v1 needed a marker-stub for
#              the apply contract).
#   4. SEARCH: noonghunna/club-3090 has only stacked the OLD PR #42006 +
#              PR #41991 in `models/gemma-4-31b/vllm/patches/
#              vllm-gemma4-tool-parser-fixes/`. NO public vendor of PR
#              #42237 exists at this date. We are the first.
#   5. COMPARE: PR #42237 author + reviewer have not closed PR #42006,
#              both remain OPEN at vendor time. The author's last comment
#              asks the reviewer which approach to merge. Our empirical
#              gating: the v1 vendor (segment-replay) on gemma4-31B AWQ-4bit
#              + TQ4bit_nc + MTP K=4 gave 5/7 deterministic tool-calls with
#              greedy decode (Class 13 residual at session 2026-05-30).
#              Hypothesis being tested with this v2 vendor: the residual
#              2/7 are MTP-K=4 multi-delimiter-per-delta cases that v1's
#              segment-replay still mis-handles but v2's accumulated-text
#              rescan does not.
#   6. THEN CHANGE: vendor PR #42237 head as v2 overlay. Keep v1 overlay
#              file on disk for git-blame + operator rollback path. Switch
#              the launcher bind-mount target from v1 to v2.
#
# Deployment (operator) — mount as overlay via the launcher's docker
# run `-v` flag:
#
#     -v $REPO/sndr/engines/vllm/patches/tool_parsing/g4_t1_v2_gemma4_tool_parser_pr42237_overlay.py:$TGT/tool_parsers/gemma4_tool_parser.py:ro
#
# (where $TGT = /usr/local/lib/python3.12/dist-packages/vllm). The
# launcher `start_gemma4-31b-tq-mtp-structured-k4.sh` mounts a copy of this
# file as `/tmp/gemma4_tool_parser_FIXED.py` per G4_T1 v2 session
# 2026-05-31.
#
# Genesis dispatcher does NOT call into this file directly — apply contract
# lives in `g4_t1_pr42006_marker.py` (preserved marker name; semantics now
# cover both v1 and v2 overlay variants). The dispatcher returns `skipped`
# with a message pointing at the launcher's bind-mount.
#
# Retire trigger (updated 2026-06-11): SIX upstream PRs now race on the
# same root issue #41967 — #42006 / #42237 / #42300 / #44741 / #45068 /
# #44844. Whichever merges first AND propagates to a pin we run decides
# retirement of the whole G4_T1 overlay stack (deep-diff first, iron
# rule #11). Track:
#
#     for pr in 42006 42237 42300 44741 45068 44844; do
#         gh pr view $pr --repo vllm-project/vllm --json state,mergedAt
#     done
#
# Canonical machine-readable rows: tools/upstream_watchlist.yaml (sweep:
# section, G4_T1 racing cluster). Once one merges + our pin contains the
# merge, delete this v2 file, the v1 rollback file
# `g4_t1_gemma4_tool_parser_pr42006_overlay.py`, AND the v3 prep file
# `g4_t1_v3_gemma4_tool_parser_pr44844_overlay.py`, then remove the
# corresponding `-v` mount in the gemma4 launcher.
#
# Empirical effect (gemma4-31B AWQ-4bit + TQ4bit_nc + MTP K=4 on
# pin 626fa9bb, 2026-05-31 session):
#
#   Bench harness: 7 edge cases × 5 runs = 35 total. Greedy decode
#   (T=0, top_k=1, top_p=1.0). Cases cover simple single-tool,
#   thinking-then-tool (CoT preamble), multi-tool sequential, string
#   args with special chars (apostrophe + comma), nested-object args,
#   mixed numeric+boolean, and two-tools-in-one-response.
#
#   Streaming SSE     + Connection:close header: 35/35 = 100.0%
#   Non-streaming     + Connection:close header: 35/35 = 100.0%
#   Streaming SSE     + HTTP keep-alive (default): 31/35 = 88.6%
#                       → failures are case 5 (nested object) runs
#                       2-5 of 5, missing the closing `"}` two chars
#                       of the args JSON. Diagnosed as PR #42237
#                       inter-request parser-instance state leak:
#                       vLLM appears to re-use the Gemma4ToolParser
#                       instance across requests on the same HTTP
#                       keep-alive socket, AND the `_reset_streaming_
#                       state` guard `if not previous_text:` does not
#                       fire when a follow-up request lands on the
#                       same parser instance with stale
#                       `streamed_args_for_tool` from the prior call.
#                       Workaround: clients SHOULD send the request
#                       header `Connection: close` for tool-call
#                       requests (forces vLLM to construct a fresh
#                       parser per request). Filed upstream feedback
#                       on PR #42237 thread (see commit message).
#
#   Pre-v2 baseline (G4_T1 v1 = PR #42006 segment-replay overlay) on
#   same hardware + same prompts: 5/7 deterministic single-run pass,
#   reproducible across many reruns of the previous session — this is
#   the 71.4% result documented as Class 13 (model-quality intrinsic
#   limit) in docs/TROUBLESHOOTING.md as of 2026-05-30. With this v2
#   vendor the diagnosis revises: the 5/7 ceiling was NOT a Class 13
#   intrinsic limit — it was the v1 parser's segment-replay refactor
#   missing exactly the multi-delimiter-per-delta cases that PR
#   #42237's accumulated-text rescan handles correctly. Class 13 has
#   been re-categorized to Class 7 (parser-bug + operator-side memory
#   tune: gpu-memory-utilization 0.92 → 0.80 to give Genesis P38 TQ
#   continuation-prefill workspace its 1 GiB alloc).
#
# Additional vendored hunk (2026-06-11): upstream PR #44717 (OPEN, issue
# #44715) — `_parse_gemma4_args()` strips `<|"|>` STRING_DELIM sentinels
# from value positions but not key positions, leaking sentinel bytes into
# dict keys for any dict-typed tool argument with string-quoted keys
# (e.g. `{'<|"|>3<|"|>': 'v'}` instead of `{'3': 'v'}`). The 8-line
# key-strip hunk was vendored verbatim into the key scanner below (and into
# the v1 legacy overlay, keeping the rollback path safe). Confirmed buggy
# in pin 0.22.1rc1.dev259+g303916e93 (`vllm/tool_parsers/
# gemma4_tool_parser.py:124`).
#
# SUPERSEDED same sweep (2026-06-11) by upstream PR #44877 (OPEN, same
# issue #44715) — full quoted-key branch vendored verbatim in place of the
# #44717 post-hoc strip. The branch parses a STRING_DELIM-wrapped key the
# same way string values are parsed (skip opening delimiter, read to
# closing delimiter, advance past whitespace to ':'), which additionally
# handles ':' inside a quoted key and withholds unterminated quoted keys
# during streaming. The #44717 strip became provably dead once the branch
# landed (the bare-key path can no longer see a key starting with
# STRING_DELIM) and was removed. Track BOTH racing PRs:
#
#     gh pr view 44717 --repo vllm-project/vllm --json state,mergedAt
#     gh pr view 44877 --repo vllm-project/vllm --json state,mergedAt
#
# When this overlay retires (PR #42237 merged + in pin), verify #44877 (or
# #44717) is ALSO merged in that pin; if not, re-vendor the quoted-key
# branch as a standalone text patch. Tests: tests/unit/integrations/
# tool_parsing/test_g4_t1_dict_key_sentinel_strip.py (AST-extracts the
# shipped source).
#
# Second vendored hunk (2026-06-11): upstream PR #45068 (OPEN, same issue
# #41967) — token-id start-gate in `extract_tool_calls_streaming`. The
# upstream PR rewrites the PRISTINE parser's gate from text-membership to
# `tool_call_start_token_id not in current_token_ids`; we transplant that
# gate (with Genesis adaptations documented inline) so lookalike marker
# TEXT produced by ordinary tokens stays content, and the O(len) rescan
# is skipped entirely until a real tool call starts. #45068's MTP-split
# regression corpus (a single delta carrying `}<tool_call|><|tool_call>
# call:next{`) is ported into tests/unit/integrations/tool_parsing/
# test_g4_t1_streaming_mtp_corpus.py — the FIRST streaming-parser tests
# this overlay family has. The same module carries the strict-xfail
# reproduction of the keep-alive 31/35 state leak documented above
# (test_keep_alive_instance_reuse_second_request_clean[v2_pr42237_
# current]): the leak is deliberately NOT fixed in v2 — the fix ships as
# the v3 prep overlay's reset-guard hardening and gets A/B'd at the
# server stage.
#
# ─────────────────────────────────────────────────────────────────────────
# BEGIN VERBATIM PR #42237 SOURCE (+ PR #44877 quoted-key branch)
# ─────────────────────────────────────────────────────────────────────────
"""
Tool call parser for Google Gemma4 models.

Gemma4 uses a custom serialization format (not JSON) for tool calls::

    <|tool_call>call:func_name{key:<|"|>value<|"|>,num:42}<tool_call|>

Strings are delimited by ``<|"|>`` (token 52), keys are unquoted, and
multiple tool calls are concatenated without separators.

Used when ``--enable-auto-tool-choice --tool-call-parser gemma4`` are set.

For offline inference tool call parsing (direct ``tokenizer.decode()`` output),
see ``vllm.tool_parsers.gemma4_utils.parse_tool_calls``.
"""

import json
from collections.abc import Sequence

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser
from vllm.tool_parsers.utils import find_common_prefix, partial_tag_overlap

logger = init_logger(__name__)

# Gemma4 special tokens for tool calls
TOOL_CALL_START = "<|tool_call>"
TOOL_CALL_END = "<tool_call|>"
STRING_DELIM = '<|"|>'


# ---------------------------------------------------------------------------
# Gemma4 argument parser (used by both streaming and non-streaming paths)
# ---------------------------------------------------------------------------


def _parse_gemma4_value(value_str: str) -> object:
    """Parse a single Gemma4 value (after key:) into a Python object."""
    value_str = value_str.strip()
    if not value_str:
        return value_str

    # Boolean
    if value_str == "true":
        return True
    if value_str == "false":
        return False

    # Null
    if value_str.lower() in ("null", "none", "nil"):
        return None

    # Number (int or float)
    try:
        if "." in value_str:
            return float(value_str)
        return int(value_str)
    except ValueError:
        pass

    # Bare string (no <|"|> delimiters — shouldn't happen but be safe)
    return value_str


def _parse_gemma4_args(args_str: str, *, partial: bool = False) -> dict:
    """Parse Gemma4's custom key:value format into a Python dict.

    Format examples::

        location:<|"|>Tokyo<|"|>
        location:<|"|>San Francisco<|"|>,unit:<|"|>celsius<|"|>
        count:42,flag:true
        nested:{inner_key:<|"|>val<|"|>}
        items:[<|"|>a<|"|>,<|"|>b<|"|>]

    Args:
        args_str: The raw Gemma4 argument string.
        partial: When True (streaming), bare values at end of string are
            omitted because they may be incomplete and type-unstable
            (e.g. partial boolean parsed as bare string).

    Returns a dict ready for ``json.dumps()``.
    """
    if not args_str or not args_str.strip():
        return {}

    result: dict = {}
    i = 0
    n = len(args_str)

    while i < n:
        # Skip whitespace and commas
        while i < n and args_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break

        # Parse key. Keys are usually bare identifiers terminated by ':', but
        # the model may wrap a string-typed key in STRING_DELIM (e.g.
        # <|"|>3<|"|>) to mark a numeric-looking key as a string. Handle that
        # case the same way as string values below so the delimiter is
        # stripped from the key rather than kept verbatim (vendored upstream
        # PR #44877, issue #44715; supersedes the PR #44717 post-hoc strip —
        # the inline branch additionally handles ':' inside a quoted key and
        # withholds unterminated quoted keys during streaming).
        if args_str[i:].startswith(STRING_DELIM):
            i += len(STRING_DELIM)
            key_start = i
            end_pos = args_str.find(STRING_DELIM, i)
            if end_pos == -1:
                # Unterminated key string — nothing parseable follows.
                break
            key = args_str[key_start:end_pos]
            i = end_pos + len(STRING_DELIM)
            # Skip to the ':' separator (whitespace may precede it).
            while i < n and args_str[i] != ":":
                i += 1
            if i >= n:
                break
            i += 1  # skip ':'
        else:
            key_start = i
            while i < n and args_str[i] != ":":
                i += 1
            if i >= n:
                break
            key = args_str[key_start:i].strip()
            i += 1  # skip ':'

        # Parse value
        if i >= n:
            if not partial:
                result[key] = ""
            break

        # Skip whitespace after ':'
        while i < n and args_str[i] in (" ", "\n", "\t"):
            i += 1
        if i >= n:
            if not partial:
                result[key] = ""
            break

        # String value: <|"|>...<|"|>
        if args_str[i:].startswith(STRING_DELIM):
            i += len(STRING_DELIM)
            val_start = i
            end_pos = args_str.find(STRING_DELIM, i)
            if end_pos == -1:
                # Unterminated string — take rest
                result[key] = args_str[val_start:]
                break
            result[key] = args_str[val_start:end_pos]
            i = end_pos + len(STRING_DELIM)

        # Nested object: {...}
        elif args_str[i] == "{":
            depth = 1
            obj_start = i + 1
            i += 1
            while i < n and depth > 0:
                if args_str[i:].startswith(STRING_DELIM):
                    # Skip over string contents to avoid counting { inside strings
                    i += len(STRING_DELIM)
                    next_delim = args_str.find(STRING_DELIM, i)
                    i = n if next_delim == -1 else next_delim + len(STRING_DELIM)
                    continue
                if args_str[i] == "{":
                    depth += 1
                elif args_str[i] == "}":
                    depth -= 1
                i += 1
            if depth > 0:
                # Incomplete nested object — use i (not i-1) to avoid
                # dropping the last char, and recurse as partial.
                result[key] = _parse_gemma4_args(args_str[obj_start:i], partial=True)
            else:
                result[key] = _parse_gemma4_args(args_str[obj_start : i - 1])

        # Array: [...]
        elif args_str[i] == "[":
            depth = 1
            arr_start = i + 1
            i += 1
            while i < n and depth > 0:
                if args_str[i:].startswith(STRING_DELIM):
                    i += len(STRING_DELIM)
                    next_delim = args_str.find(STRING_DELIM, i)
                    i = n if next_delim == -1 else next_delim + len(STRING_DELIM)
                    continue
                if args_str[i] == "[":
                    depth += 1
                elif args_str[i] == "]":
                    depth -= 1
                i += 1
            if depth > 0:
                result[key] = _parse_gemma4_array(args_str[arr_start:i], partial=True)
            else:
                result[key] = _parse_gemma4_array(args_str[arr_start : i - 1])

        # Bare value (number, boolean, etc.)
        else:
            val_start = i
            while i < n and args_str[i] not in (",", "}", "]"):
                i += 1
            if partial and i >= n:
                # Value may be incomplete (e.g. partial boolean) —
                # withhold to avoid type instability during streaming.
                break
            if i == val_start:
                logger.warning(
                    "Gemma4 args parser made no progress at position %d; "
                    "aborting on malformed input.",
                    i,
                )
                break
            result[key] = _parse_gemma4_value(args_str[val_start:i])

    return result


def _parse_gemma4_array(arr_str: str, *, partial: bool = False) -> list:
    """Parse a Gemma4 array content string into a Python list."""
    items: list = []
    i = 0
    n = len(arr_str)

    while i < n:
        while i < n and arr_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break

        # String element
        if arr_str[i:].startswith(STRING_DELIM):
            i += len(STRING_DELIM)
            end_pos = arr_str.find(STRING_DELIM, i)
            if end_pos == -1:
                items.append(arr_str[i:])
                break
            items.append(arr_str[i:end_pos])
            i = end_pos + len(STRING_DELIM)

        # Nested object
        elif arr_str[i] == "{":
            depth = 1
            obj_start = i + 1
            i += 1
            while i < n and depth > 0:
                if arr_str[i:].startswith(STRING_DELIM):
                    i += len(STRING_DELIM)
                    nd = arr_str.find(STRING_DELIM, i)
                    i = nd + len(STRING_DELIM) if nd != -1 else n
                    continue
                if arr_str[i] == "{":
                    depth += 1
                elif arr_str[i] == "}":
                    depth -= 1
                i += 1
            if depth > 0:
                items.append(_parse_gemma4_args(arr_str[obj_start:i], partial=True))
            else:
                items.append(_parse_gemma4_args(arr_str[obj_start : i - 1]))

        # Nested array
        elif arr_str[i] == "[":
            depth = 1
            sub_start = i + 1
            i += 1
            while i < n and depth > 0:
                if arr_str[i:].startswith(STRING_DELIM):
                    i += len(STRING_DELIM)
                    nd = arr_str.find(STRING_DELIM, i)
                    i = nd + len(STRING_DELIM) if nd != -1 else n
                    continue
                if arr_str[i] == "[":
                    depth += 1
                elif arr_str[i] == "]":
                    depth -= 1
                i += 1
            if depth > 0:
                items.append(_parse_gemma4_array(arr_str[sub_start:i], partial=True))
            else:
                items.append(_parse_gemma4_array(arr_str[sub_start : i - 1]))

        # Bare value
        else:
            val_start = i
            while i < n and arr_str[i] not in (",", "]"):
                i += 1
            if partial and i >= n:
                break
            if i == val_start:
                logger.warning(
                    "Gemma4 array parser made no progress at position %d; "
                    "aborting on malformed input.",
                    i,
                )
                break
            items.append(_parse_gemma4_value(arr_str[val_start:i]))

    return items


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class Gemma4ToolParser(ToolParser):
    """
    Tool call parser for Google Gemma4 models.

    Handles the Gemma4 function call format::

        <|tool_call>call:func_name{key:<|"|>value<|"|>}<tool_call|>

    Used when ``--enable-auto-tool-choice --tool-call-parser gemma4``
    are set.

    Streaming strategy: **scan accumulated text, parse, then diff**

    Instead of trying to convert Gemma4's custom format to JSON
    one token at a time (which fails because Gemma4 uses bare keys, custom
    delimiters, and structural braces that differ from JSON), this parser:

    1. Re-scans the accumulated model output to find tool-call regions
    2. Parses each visible region with ``_parse_gemma4_args()``
    3. Converts arguments to JSON with ``json.dumps()``
    4. Diffs against the JSON prefix already streamed for that tool index
    5. Emits only the new function names, argument fragments, and content

    This follows the same shape used by Hermes/Kimi-style parsers and makes
    the parser insensitive to whether speculative decoding delivers one token
    or several delimiter events in a single streaming delta.
    """

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

        # Token strings
        self.tool_call_start_token = TOOL_CALL_START
        self.tool_call_end_token = TOOL_CALL_END

        # Token IDs
        self.tool_call_start_token_id = self.vocab.get(TOOL_CALL_START)
        self.tool_call_end_token_id = self.vocab.get(TOOL_CALL_END)

        if self.tool_call_start_token_id is None:
            raise RuntimeError(
                "Gemma4 ToolParser could not locate the tool call start "
                f"token '{TOOL_CALL_START}' in the tokenizer!"
            )

        # Streaming state — reset per-request via _reset_streaming_state()
        self._reset_streaming_state()

    def _reset_streaming_state(self) -> None:
        """Reset all streaming state for a new request."""
        self.current_tool_id = -1
        self.prev_tool_call_arr: list[dict] = []
        self.streamed_args_for_tool: list[str] = []
        self._sent_content_idx = 0

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            # Don't skip special tokens — <|tool_call> etc. are needed for
            # the parser to detect tool calls. Apply to BOTH
            # ChatCompletionRequest and ResponsesRequest (the previous
            # isinstance(ChatCompletionRequest) guard caused tool-call
            # delimiters to be stripped on /v1/responses, leaking raw
            # `call:fn{...}` text via output_text.delta).
            request.skip_special_tokens = False
        return request

    # ------------------------------------------------------------------
    # Non-streaming extraction
    # ------------------------------------------------------------------

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            parsed_tool_calls = self._extract_tool_call_regions(
                model_output, include_partial=False
            )
            if not parsed_tool_calls:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

            tool_calls: list[ToolCall] = []
            for func_name, args_str, _ in parsed_tool_calls:
                arguments = _parse_gemma4_args(args_str)
                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=func_name,
                            arguments=json.dumps(arguments, ensure_ascii=False),
                        ),
                    )
                )

            # Content = text before first tool call (if any)
            content_end = model_output.find(self.tool_call_start_token)
            content = model_output[:content_end].strip() if content_end > 0 else None

            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if content else None,
            )

        except Exception:
            logger.exception("Error extracting tool calls from Gemma4 response")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    # ------------------------------------------------------------------
    # Streaming extraction — scan accumulated text, parse, then diff
    # ------------------------------------------------------------------

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        if not previous_text:
            self._reset_streaming_state()

        # Token-id start-gate (vendored from upstream PR #45068, OPEN at
        # vendor time 2026-06-11; same root issue #41967). Gate the
        # accumulated-text rescan on the START token ID instead of the
        # start token TEXT: token ids are the detokenizer's ground truth
        # — the special token always arrives atomically alongside its id,
        # while lookalike TEXT (a model spelling out "<|tool_call>" via
        # ordinary tokens) must stay on the content channel instead of
        # being swallowed as a phantom tool call. Genesis adaptations vs
        # the upstream hunk (iron rule #10):
        #   (1) advance _sent_content_idx so the rescan path does not
        #       re-emit gate-emitted content once a real tool call
        #       starts;
        #   (2) keep the membership check stateless per call (no latched
        #       boolean) — the documented keep-alive instance-reuse leak
        #       below makes latched per-request booleans hazardous on
        #       this class;
        #   (3) fall back to the upstream text check if the start token
        #       id is unresolvable (__init__ raises when it is missing,
        #       so this arm is defense in depth only).
        if self.tool_call_start_token_id is not None:
            no_tool_call_yet = (
                self.tool_call_start_token_id not in current_token_ids
            )
        else:
            no_tool_call_yet = self.tool_call_start_token not in current_text
        if no_tool_call_yet:
            self._sent_content_idx = len(current_text)
            if delta_text:
                return DeltaMessage(content=delta_text)
            return None

        try:
            return self._extract_streaming(current_text)
        except Exception:
            logger.exception("Error in Gemma4 streaming tool call extraction")
            return None

    def _extract_streaming(self, current_text: str) -> DeltaMessage | None:
        """Stream content and Gemma4 tool-call deltas from accumulated text."""
        content = self._extract_content(current_text)
        tool_calls = self._extract_tool_call_regions(current_text, include_partial=True)
        tool_call_deltas: list[DeltaToolCall] = []

        for index, (func_name, raw_args_str, is_complete) in enumerate(tool_calls):
            self._ensure_tool_state(index)

            if "name" not in self.prev_tool_call_arr[index]:
                self.prev_tool_call_arr[index] = {
                    "name": func_name,
                    "arguments": {},
                }
                tool_call_deltas.append(
                    DeltaToolCall(
                        index=index,
                        type="function",
                        id=make_tool_call_id(),
                        function=DeltaFunctionCall(
                            name=func_name,
                            arguments="",
                        ).model_dump(exclude_none=True),
                    )
                )

            args_delta = self._compute_args_diff(
                index=index,
                raw_args_str=raw_args_str,
                is_complete=is_complete,
            )
            if args_delta:
                tool_call_deltas.append(
                    DeltaToolCall(
                        index=index,
                        function=DeltaFunctionCall(arguments=args_delta).model_dump(
                            exclude_none=True
                        ),
                    )
                )

        if content or tool_call_deltas:
            return DeltaMessage(content=content, tool_calls=tool_call_deltas)
        return None

    def _extract_content(self, current_text: str) -> str | None:
        """Return unsent non-tool-call text, holding partial start tags."""
        content_parts: list[str] = []
        pos = self._sent_content_idx
        text_len = len(current_text)

        while pos < text_len:
            start = current_text.find(self.tool_call_start_token, pos)
            if start == -1:
                overlap = partial_tag_overlap(
                    current_text[pos:], self.tool_call_start_token
                )
                sendable_end = text_len - overlap
                if sendable_end > pos:
                    content_parts.append(current_text[pos:sendable_end])
                    pos = sendable_end
                break

            if start > pos:
                content_parts.append(current_text[pos:start])

            end = current_text.find(
                self.tool_call_end_token, start + len(self.tool_call_start_token)
            )
            if end == -1:
                pos = start
                break

            pos = end + len(self.tool_call_end_token)

        self._sent_content_idx = pos
        return "".join(content_parts) or None

    def _extract_tool_call_regions(
        self, current_text: str, *, include_partial: bool
    ) -> list[tuple[str, str, bool]]:
        """Extract visible Gemma4 tool-call regions from accumulated text."""
        tool_calls: list[tuple[str, str, bool]] = []
        pos = 0

        while True:
            start = current_text.find(self.tool_call_start_token, pos)
            if start == -1:
                break

            body_start = start + len(self.tool_call_start_token)
            end = current_text.find(self.tool_call_end_token, body_start)
            if end == -1:
                if not include_partial:
                    break
                body = current_text[body_start:]
                overlap = partial_tag_overlap(body, self.tool_call_end_token)
                if overlap:
                    body = body[:-overlap]
                is_complete = False
                pos = len(current_text)
            else:
                body = current_text[body_start:end]
                is_complete = True
                pos = end + len(self.tool_call_end_token)

            parsed = self._parse_tool_call_body(body)
            if parsed is not None:
                func_name, raw_args_str = parsed
                tool_calls.append((func_name, raw_args_str, is_complete))

            if not is_complete:
                break

        return tool_calls

    def _parse_tool_call_body(self, body: str) -> tuple[str, str] | None:
        """Parse ``call:name{args}`` from a raw Gemma4 tool-call body."""
        if not body.startswith("call:"):
            return None

        func_part = body[len("call:") :]
        open_brace = func_part.find("{")
        if open_brace == -1:
            return None

        func_name = func_part[:open_brace].strip()
        if not func_name:
            return None

        args_start = open_brace + 1
        args_end = self._find_outer_args_end(func_part, args_start)
        raw_args_str = (
            func_part[args_start:args_end]
            if args_end is not None
            else func_part[args_start:]
        )
        return func_name, raw_args_str

    def _find_outer_args_end(self, text: str, args_start: int) -> int | None:
        """Return the index of the outer Gemma4 argument brace, if present."""
        depth = 1
        i = args_start
        text_len = len(text)

        while i < text_len:
            if text[i:].startswith(STRING_DELIM):
                i += len(STRING_DELIM)
                next_delim = text.find(STRING_DELIM, i)
                if next_delim == -1:
                    return None
                i = next_delim + len(STRING_DELIM)
                continue

            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return i
            i += 1

        return None

    def _ensure_tool_state(self, index: int) -> None:
        """Ensure streaming state exists for tool-call index."""
        while len(self.prev_tool_call_arr) <= index:
            self.prev_tool_call_arr.append({})
        while len(self.streamed_args_for_tool) <= index:
            self.streamed_args_for_tool.append("")
        self.current_tool_id = max(self.current_tool_id, index)

    def _compute_args_diff(
        self, index: int, raw_args_str: str, is_complete: bool
    ) -> str | None:
        """Return new argument JSON not yet sent for tool-call index."""
        try:
            current_args = _parse_gemma4_args(raw_args_str, partial=not is_complete)
        except Exception:
            logger.debug(
                "Could not parse partial Gemma4 args yet: %s",
                raw_args_str[:100],
            )
            return None

        if not current_args and not is_complete:
            return None

        current_args_json = json.dumps(current_args, ensure_ascii=False)
        next_json = (
            current_args_json
            if is_complete
            else self._stable_partial_json_prefix(current_args_json)
        )

        prev_streamed = self.streamed_args_for_tool[index]
        if not next_json or next_json == prev_streamed:
            return None

        if not next_json.startswith(prev_streamed):
            prefix = find_common_prefix(prev_streamed, next_json)
            self.streamed_args_for_tool[index] = prefix
            logger.debug(
                "Skipping Gemma4 argument delta for tool %d because parsed "
                "JSON no longer extends the streamed prefix.",
                index,
            )
            return None

        diff = next_json[len(prev_streamed) :]
        if not diff:
            return None

        self.streamed_args_for_tool[index] = next_json
        self.prev_tool_call_arr[index]["arguments"] = current_args
        return diff

    @staticmethod
    def _stable_partial_json_prefix(json_text: str) -> str:
        """Trim closing chars that may move while a Gemma4 call is partial."""
        return json_text.rstrip('}"]<|\\>')
