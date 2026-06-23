# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Genesis Q3_T1 v1 — vendored qwen3_coder tool-parser streaming rewrite
# ─────────────────────────────────────────────────────────────────────────
# Replaces `extract_tool_calls_streaming` of vllm/tool_parsers/
# qwen3coder_tool_parser.py with a Hermes/Kimi-style accumulated-text
# rescan + diff pattern, structurally analogous to G4_T1 v2 (PR #42237
# vendored 2026-05-30 for gemma4_tool_parser).
#
# Why we need it (Bug Class 15, see docs/TROUBLESHOOTING.md):
#   The upstream qwen3_coder streaming implementation is a complex
#   state machine (is_tool_call_started + current_tool_index +
#   header_sent + json_started + json_closed + in_function + in_param
#   + accumulated_params + accumulated_text) that mis-handles the
#   multi-delimiter-per-delta packing MTP K=3 produces on the
#   Qwen3.6-A3B / Qwen3.6-27B PROD launchers. Empirical 2026-05-31
#   bench:
#       Non-streaming : 7/7 = 100%
#       Streaming     : 1/7 ~ 14%   (XML leaks into delta.content,
#                                    delta.tool_calls stays empty,
#                                    finish_reason="stop" instead of
#                                    "tool_calls")
#
# How this overlay fixes it:
#   The non-streaming extract_tool_calls() is the working ORACLE — it
#   uses well-tested tool_call_complete_regex + tool_call_function_regex
#   + tool_call_parameter_regex against the FULL accumulated text and
#   correctly populates ToolCall objects via _parse_xml_function_call.
#   The new streaming method calls the same oracle on each delta's
#   accumulated text, then diffs against per-tool-index streamed-args
#   state to emit only the new fragments. No state machine, no
#   position-tracked transitions, no per-token re-entry bugs.
#
# Algorithm:
#   1. Reset streamed state on `not previous_text` (new request boundary).
#   2. Rescan accumulated `current_text` for ALL tool-call regions
#      using existing regex pair (complete + partial).
#   3. For each region (index 0..N):
#        a. If this index's tool name not yet emitted, emit
#           DeltaToolCall(type="function", id=<new uuid>,
#                         function=DeltaFunctionCall(name=NAME, arguments=""))
#        b. Build current args dict via tool_call_parameter_regex
#           against the function-body region. Convert to JSON via
#           json.dumps with the existing schema-coercion helpers.
#        c. Compute prefix diff vs streamed_args_for_tool[index];
#           emit only the new tail as DeltaToolCall(arguments=diff).
#   4. Content-before-first-toolcall extraction stays the same shape
#      as the gateway expects (DeltaMessage.content for any leading
#      pre-tool text).
#
# Deployment (operator) — bind-mount the file via launcher `-v`:
#     -v $REPO/sndr/engines/vllm/patches/tool_parsing/q3_t1_qwen3coder_tool_parser_overlay.py:$TGT/tool_parsers/qwen3coder_tool_parser.py:ro
#   where $TGT = /usr/local/lib/python3.12/dist-packages/vllm
#   The rig launchers start_pn95_2xa5000_test.sh (27B) and
#   start_35b_prod_wave8.sh (35B) mount the file as
#   /tmp/qwen3coder_tool_parser_FIXED.py per Bug Class 15
#   resolution session 2026-05-31.
#
# Retire trigger:
#   When upstream lands a Hermes/Kimi-style rewrite of
#   qwen3coder_tool_parser:extract_tool_calls_streaming and our pin
#   bumps to include it — diff byte-by-byte against this overlay and
#   retire if equivalent.
#
# Empirical effect (qwen3.6-27B int4 + TQ + MTP K=3, 2026-05-31):
#   pre-Q3_T1   : 1/7 streaming   (14.3%)
#   post-Q3_T1  : <to be filled after bench>
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import uuid
from collections.abc import Sequence
from typing import Any

import regex as re

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
from vllm.envs import VLLM_ENFORCE_STRICT_TOOL_CALLING
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)
from vllm.tool_parsers.structural_tag_registry import (
    get_enable_structured_outputs_in_reasoning,
    get_model_structural_tag,
)
from vllm.tool_parsers.utils import (
    coerce_to_schema_type,
    extract_types_from_schema,
    find_tool_properties,
)

logger = init_logger(__name__)


class Qwen3CoderToolParser(ToolParser):
    supports_required_and_named: bool = not VLLM_ENFORCE_STRICT_TOOL_CALLING

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        # Override base class type - we use string IDs for tool calls
        self.current_tool_id: str | None = None  # type: ignore
        self.streamed_args_for_tool: list[str] = []

        # Sentinel tokens for streaming mode
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_prefix: str = "<function="
        self.function_end_token: str = "</function>"
        self.parameter_prefix: str = "<parameter="
        self.parameter_end_token: str = "</parameter>"
        self.is_tool_call_started: bool = False
        self.failed_count: int = 0

        # Enhanced streaming state - reset for each new message
        self._reset_streaming_state()

        # Regex patterns
        self.tool_call_complete_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>", re.DOTALL
        )
        self.tool_call_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL
        )
        self.tool_call_function_regex = re.compile(
            r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
        )
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
            re.DOTALL,
        )

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        if self.tool_call_start_token_id is None or self.tool_call_end_token_id is None:
            raise RuntimeError(
                "Qwen3 XML Tool parser could not locate tool call start/end "
                "tokens in the tokenizer!"
            )

        logger.debug(
            "vLLM Successfully import tool parser %s !", self.__class__.__name__
        )

        # Q3_T1 v1 (2026-05-31): strict parameter regex used by the
        # streaming-safe rebuild path. Requires </parameter> closing
        # (no lookahead alternatives) so partial trailing params are
        # excluded from streaming JSON emission.
        self._q3_complete_param_regex = re.compile(
            r"<parameter=(.*?)</parameter>",
            re.DOTALL,
        )

    def _generate_tool_call_id(self) -> str:
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:24]}"

    def _reset_streaming_state(self):
        """Reset all streaming state."""
        self.current_tool_index = 0
        self.is_tool_call_started = False
        self.header_sent = False
        self.current_tool_id = None
        self.current_function_name = None
        self.current_param_name = None
        self.current_param_value = ""
        self.param_count = 0
        self.in_param = False
        self.in_function = False
        self.accumulated_text = ""
        self.json_started = False
        self.json_closed = False
        # Store accumulated parameters for type conversion
        self.accumulated_params = {}
        self.streaming_request = None

    def _convert_param_value(
        self, param_value: str, param_name: str, param_config: dict, func_name: str
    ) -> Any:
        """Convert parameter value based on its type in the schema."""
        if not isinstance(param_value, str):
            return param_value
        param_schema = param_config.get(param_name, {})
        param_types = extract_types_from_schema(param_schema)
        return coerce_to_schema_type(param_value, param_types)

    def _parse_xml_function_call(self, function_call_str: str) -> ToolCall | None:
        # Extract function name
        end_index = function_call_str.find(">")
        # If there's no ">" character, this is not a valid xml function call
        if end_index == -1:
            return None
        function_name = function_call_str[:end_index]
        param_config = find_tool_properties(self.tools, function_name)
        parameters = function_call_str[end_index + 1 :]
        param_dict = {}
        for match_text in self.tool_call_parameter_regex.findall(parameters):
            idx = match_text.index(">")
            param_name = match_text[:idx]
            param_value = str(match_text[idx + 1 :])
            # Remove prefix and trailing \n
            if param_value.startswith("\n"):
                param_value = param_value[1:]
            if param_value.endswith("\n"):
                param_value = param_value[:-1]

            param_dict[param_name] = self._convert_param_value(
                param_value, param_name, param_config, function_name
            )
        return ToolCall(
            type="function",
            function=FunctionCall(
                name=function_name, arguments=json.dumps(param_dict, ensure_ascii=False)
            ),
        )

    def _get_function_calls(self, model_output: str) -> list[str]:
        # Find all tool calls
        matched_ranges = self.tool_call_regex.findall(model_output)
        raw_tool_calls = [
            match[0] if match[0] else match[1] for match in matched_ranges
        ]

        # Back-off strategy if no tool_call tags found
        if len(raw_tool_calls) == 0:
            raw_tool_calls = [model_output]

        raw_function_calls = []
        for tool_call in raw_tool_calls:
            raw_function_calls.extend(self.tool_call_function_regex.findall(tool_call))

        function_calls = [
            match[0] if match[0] else match[1] for match in raw_function_calls
        ]
        return function_calls

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        # Quick check to avoid unnecessary processing
        if self.tool_call_prefix not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            function_calls = self._get_function_calls(model_output)
            if len(function_calls) == 0:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

            tool_calls = [
                self._parse_xml_function_call(function_call_str)
                for function_call_str in function_calls
            ]
            # Populate prev_tool_call_arr for serving layer to set finish_reason
            self.prev_tool_call_arr.clear()  # Clear previous calls
            for tool_call in tool_calls:
                if tool_call:
                    self.prev_tool_call_arr.append(
                        {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        }
                    )

            # Extract content before tool calls
            content_index = model_output.find(self.tool_call_start_token)
            idx = model_output.find(self.tool_call_prefix)
            content_index = content_index if content_index >= 0 else idx
            content = model_output[:content_index]  # .rstrip()
            valid_tool_calls = [tc for tc in tool_calls if tc is not None]
            return ExtractedToolCallInformation(
                tools_called=(len(valid_tool_calls) > 0),
                tool_calls=valid_tool_calls,
                content=content if content else None,
            )

        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

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
        """Hermes/Kimi-style accumulated-text rescan + diff streaming.

        On each delta: re-extract ALL tool calls from accumulated
        current_text using the same regex-based oracle as the
        non-streaming path, then compute the JSON-args diff against
        per-tool-index streamed state and emit only the new fragment.
        """
        if not previous_text:
            self._reset_streaming_state()
            self.streaming_request = request
            # Per-tool-index emitted state (initialized lazily as new
            # tool indices appear during the stream).
            self._q3_tool_ids: list[str | None] = []
            self._q3_emitted_names: list[bool] = []
            self._q3_streamed_args: list[str] = []

        try:
            return self._q3_extract_streaming(current_text)
        except Exception:
            logger.exception(
                "Q3_T1: error in hermes-style qwen3_coder streaming "
                "extraction; falling back to None to avoid request crash"
            )
            return None

    def _q3_extract_streaming(self, current_text: str) -> DeltaMessage | None:
        """Re-scan accumulated text, build deltas vs streamed state."""
        # Reuse the existing non-streaming oracle for FULL extraction.
        # _get_function_calls handles partial (open-ended) functions
        # via the dual-form tool_call_function_regex.
        function_call_strs = self._get_function_calls(current_text)
        tool_call_deltas: list[DeltaToolCall] = []

        for index, fc_str in enumerate(function_call_strs):
            self._q3_ensure_index_state(index)

            # Extract function name from current accumulated fragment.
            end_marker = fc_str.find(">")
            if end_marker == -1:
                # Function name not yet fully streamed; defer this index.
                continue
            func_name = fc_str[:end_marker]
            if not func_name:
                continue

            # Emit tool_call name on first sight of this index.
            if not self._q3_emitted_names[index]:
                self._q3_emitted_names[index] = True
                if self._q3_tool_ids[index] is None:
                    self._q3_tool_ids[index] = self._generate_tool_call_id()
                tool_call_deltas.append(
                    DeltaToolCall(
                        index=index,
                        type="function",
                        id=self._q3_tool_ids[index],
                        function=DeltaFunctionCall(
                            name=func_name,
                            arguments="",
                        ).model_dump(exclude_none=True),
                    )
                )

            # Build current args dict from parameter regex on the
            # body region (after the function-name-closing ">").
            params_text = fc_str[end_marker + 1:]
            current_args = self._q3_build_args_dict(func_name, params_text)
            current_args_json = json.dumps(
                current_args, ensure_ascii=False
            )

            # Determine completeness: a tool-call is complete only when
            # its </function> closing tag is present in the underlying
            # function-regex match. We check by re-running the regex
            # tuple form against current_text and checking match[0].
            is_complete = self._q3_index_is_complete(current_text, index)

            # Per Hermes/Kimi stable-prefix pattern: when the tool
            # call is still in-flight, trim trailing structural chars
            # (`}`, `"`, `]`) from the JSON because those will move
            # as more parameters arrive. When the call is complete,
            # emit the full JSON including the closing chars.
            if is_complete:
                next_json = current_args_json
            else:
                next_json = self._q3_stable_partial_json_prefix(
                    current_args_json
                )

            # Diff vs already-streamed args for this index.
            prev_streamed = self._q3_streamed_args[index]
            if next_json == prev_streamed:
                continue
            if not next_json.startswith(prev_streamed):
                # Defensive: a re-parse changed the prefix (rare; can
                # happen with type-coercion flips on partial values).
                # Reset the prefix conservatively to the common head.
                common = 0
                for ca, pa in zip(next_json, prev_streamed):
                    if ca != pa:
                        break
                    common += 1
                self._q3_streamed_args[index] = next_json[:common]
                prev_streamed = self._q3_streamed_args[index]

            diff = next_json[len(prev_streamed):]
            if diff:
                tool_call_deltas.append(
                    DeltaToolCall(
                        index=index,
                        function=DeltaFunctionCall(
                            arguments=diff,
                        ).model_dump(exclude_none=True),
                    )
                )
                self._q3_streamed_args[index] = next_json

        # Also extract any leading pre-tool-call content from the
        # accumulated text on the first emission only. The streaming
        # API expects content separate from tool_calls.
        content = self._q3_extract_leading_content(current_text)

        if content or tool_call_deltas:
            # Q3_T1 v1.3 fix (2026-05-31): Pydantic v2 rejects None for
            # the DeltaMessage.tool_calls field — it must be either a
            # list or omitted entirely. Build kwargs dict so empty
            # lists/None get dropped via the `if x:` filter instead
            # of being passed explicitly. The same applies to content.
            kwargs: dict[str, Any] = {}
            if content:
                kwargs["content"] = content
            if tool_call_deltas:
                kwargs["tool_calls"] = tool_call_deltas
            return DeltaMessage(**kwargs)
        return None

    def _q3_ensure_index_state(self, index: int) -> None:
        """Initialize per-tool-index streamed state lazily."""
        while len(self._q3_tool_ids) <= index:
            self._q3_tool_ids.append(None)
        while len(self._q3_emitted_names) <= index:
            self._q3_emitted_names.append(False)
        while len(self._q3_streamed_args) <= index:
            self._q3_streamed_args.append("")

    def _q3_build_args_dict(self, func_name: str, params_text: str) -> dict:
        """Build args dict from a function-body params text fragment.

        STREAMING-SAFE: only includes parameters whose `</parameter>`
        closing tag has been observed. Excludes the in-flight last
        parameter — its value would be partial and could flip type
        (e.g. `""` placeholder becomes `4` once int value arrives),
        which would break the stable-prefix diff invariant required
        for incremental streaming JSON emission. The complete params
        from earlier in the same stream stay stable across deltas;
        only when the last param closes does its value flow through.
        """
        param_config = find_tool_properties(self.tools, func_name)
        param_dict: dict[str, Any] = {}
        # Strict regex: requires </parameter> closing (no lookahead
        # / no end-of-string alternative). Partial trailing parameter
        # is intentionally skipped.
        for m in self._q3_complete_param_regex.finditer(params_text):
            inner = m.group(1)
            try:
                idx = inner.index(">")
            except ValueError:
                continue
            param_name = inner[:idx]
            param_value = str(inner[idx + 1:])
            if param_value.startswith("\n"):
                param_value = param_value[1:]
            if param_value.endswith("\n"):
                param_value = param_value[:-1]
            param_dict[param_name] = self._convert_param_value(
                param_value, param_name, param_config, func_name
            )
        return param_dict

    @staticmethod
    def _q3_stable_partial_json_prefix(json_text: str) -> str:
        """Trim trailing structural chars that move as args grow.

        Strips trailing `}`, `\"`, `]`, and whitespace so the streamed
        prefix is a stable JSON prefix that the NEXT delta's
        re-parsed JSON can extend cleanly. Same pattern used by
        G4_T1 v2 (PR #42237) for gemma4.
        """
        return json_text.rstrip('}\"] ')

    def _q3_index_is_complete(
        self, current_text: str, index: int
    ) -> bool:
        """True if the index-th tool call has its </function> tag."""
        # Use the same complete-vs-partial regex tuple form. tuples
        # where [0] is non-empty mean the complete-form alternative
        # of the regex matched (i.e. has </function>).
        matches = self.tool_call_function_regex.findall(current_text)
        if index >= len(matches):
            return False
        return bool(matches[index][0])

    def _q3_extract_leading_content(
        self, current_text: str
    ) -> str | None:
        """Return any pre-tool-call leading content not yet streamed.

        First time through: returns everything before the first
        <tool_call> token; subsequent calls return None because
        leading content is emitted at most once per request.
        """
        if getattr(self, "_q3_leading_content_sent", False):
            return None
        first_marker = current_text.find(self.tool_call_start_token)
        if first_marker == -1:
            first_marker = current_text.find(self.tool_call_prefix)
        if first_marker <= 0:
            # No leading content OR no tool-call seen yet.
            return None
        content = current_text[:first_marker]
        if not content:
            return None
        self._q3_leading_content_sent = True
        return content

    def get_structural_tag(self, request: ChatCompletionRequest):
        return get_model_structural_tag(
            model="qwen_3_5",
            tools=request.tools,
            tool_choice=request.tool_choice,
            reasoning=get_enable_structured_outputs_in_reasoning(),
        )
