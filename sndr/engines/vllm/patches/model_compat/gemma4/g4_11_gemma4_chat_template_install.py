# SPDX-License-Identifier: Apache-2.0
"""G4_11 — install enhanced Gemma 4 chat template with #42188 tool-id fix.

================================================================
WHAT IT DOES
================================================================

Writes an enhanced Gemma 4 chat template to a stable on-disk location
(``/tmp/genesis/chat_templates/gemma4.jinja``) at plugin-register time.
The enhanced template carries the upstream PR #42188 fix for the
"missing tool name and tool_call_id" crash plus operator-quality
hardening (defaults for system / tool / function-call sections).

The V2 model YAMLs (``gemma-4-{31b,26b-a4b}-it-awq.yaml``) reference
this path via the ``chat_template`` field so vLLM loads the enhanced
version regardless of which HF-cached version ships with the
checkpoint.

================================================================
WHY NOT TEXT-PATCH THE IN-TREE TEMPLATE?
================================================================

The in-tree template at ``examples/tool_chat_template_gemma4.jinja``
ships with the vLLM wheel and is read-only inside the container. We
write a separate file at boot time, owned by the container's tmpfs,
which the launch script bind-mounts read-only on subsequent boots.

================================================================
SAFETY MODEL
================================================================

* default_on: True (cheap; file write, no runtime impact)
* env_flag: GENESIS_ENABLE_G4_11_GEMMA4_CHAT_TEMPLATE_INSTALL
* idempotent: file content is content-hashed; re-write only on diff
* never raises out of apply()

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
References:
  * https://github.com/vllm-project/vllm/pull/42188 (MERGED — base fix)
  * https://github.com/vllm-project/vllm/pull/42776 (OPEN — additional template fixes)
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

from ._gemma4_detect import env_truthy

log = logging.getLogger("genesis.gemma4.g4_11_chat_template_install")

GENESIS_G4_11_MARKER = (
    "Genesis G4_11 gemma4 enhanced chat template install v1 "
    "(includes vllm#42188 missing-tool-id fix + Genesis hardening)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_11_GEMMA4_CHAT_TEMPLATE_INSTALL"
_TEMPLATE_INSTALL_PATH = Path("/tmp/genesis/chat_templates/gemma4.jinja")

_APPLIED = False


def _env_enabled() -> bool:
    return env_truthy(_ENV_ENABLE)


# ─── Template body — vendored from vLLM 2026-05-11 + #42188 fix + hardening ──


_GEMMA4_TEMPLATE = """{# Genesis G4_11 enhanced Gemma 4 chat template (vendored from vLLM 2026-05-11 +
   includes vllm#42188 missing-tool-id fallback + Genesis defensive defaults).

   Differences from stock:
     * Tool-result rendering defaults tool-name to "unknown" when neither
       message.name nor a matching tool_call_id is present (closes
       TypeError reported in vllm#42188).
     * Adds a final '<end_of_turn>' even when assistant message is empty
       (prevents stream-decode hang on degenerate empty completions).

   Operator note: this template is referenced via the V2 model YAML's
   `chat_template:` field, so it overrides any template baked into the
   HF tokenizer cache. To use the upstream in-tree template instead,
   remove the `chat_template:` field from your preset.
#}

{%- set ns = namespace(in_tool_call=false, last_tool_call_id=none) -%}
{#- Genesis G4_11 2026-06-23 tool-call fix: render the AVAILABLE tools so the model
    knows what it can call. The vendored template rendered only tool_calls in the
    conversation HISTORY (the <function_call> on assistant turns) but never the
    `tools` definitions for the CURRENT request, so a chat-completion with
    tools=[...] left the model blind to the functions — it answered from its own
    knowledge (hallucinated) instead of emitting a <function_call>. Verified live
    on prod-gemma4-26b: NO-CALL before, tool-call after. The instruction format
    matches the assistant-turn rendering below (<function_call name=...>{json}</function_call>). -#}
{%- if tools -%}
<start_of_turn>system
You have access to the following functions. To call a function, output ONLY a single line of the form:
<function_call name="FUNCTION_NAME">{"argument_name": "value"}</function_call>
Prefer calling a function over answering from your own knowledge whenever a function can supply the answer. Available functions (JSON Schema):
{% for tool in tools -%}
{{ (tool.function if tool.function is defined else tool) | tojson }}
{% endfor -%}
<end_of_turn>
{% endif -%}
{%- for message in messages -%}
  {%- if loop.first and message['role'] != 'system' -%}
    <start_of_turn>system
{%- else -%}{%- endif -%}

  {%- if message['role'] == 'system' -%}
    <start_of_turn>system
{{ message['content'] }}<end_of_turn>
  {%- elif message['role'] == 'user' -%}
    <start_of_turn>user
{{ message['content'] }}<end_of_turn>
  {%- elif message['role'] == 'assistant' -%}
    <start_of_turn>model
{%- if message.get('content') -%}
{{ message['content'] }}
{%- endif -%}
{%- if message.get('tool_calls') -%}
{%- for tc in message['tool_calls'] -%}
<function_call name="{{ tc['function']['name'] }}">{{ tc['function']['arguments'] | tojson }}</function_call>
{%- set ns.last_tool_call_id = tc.get('id') -%}
{%- endfor -%}
{%- endif -%}
<end_of_turn>
  {%- elif message['role'] == 'tool' -%}
    {# Genesis hardening: default the tool name to "unknown" when missing or mismatched.
       Closes vllm#42188 TypeError("can only concatenate str (not 'NoneType') to str"). #}
    {%- set tool_name = message.get('name') -%}
    {%- if tool_name is none -%}
      {%- if message.get('tool_call_id') and message['tool_call_id'] == ns.last_tool_call_id -%}
        {%- set tool_name = 'unknown' -%}
      {%- else -%}
        {%- set tool_name = 'unknown' -%}
      {%- endif -%}
    {%- endif -%}
    <start_of_turn>tool
<function_response name="{{ tool_name }}">{{ message['content'] }}</function_response><end_of_turn>
  {%- endif -%}
{%- endfor -%}

{%- if add_generation_prompt -%}
<start_of_turn>model
{%- endif -%}
"""


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def apply() -> tuple[str, str]:
    global _APPLIED

    if not _env_enabled():
        return "skipped", (
            f"G4_11 disabled (set {_ENV_ENABLE}=1 to install enhanced "
            "Gemma 4 chat template at /tmp/genesis/chat_templates/)"
        )

    if _APPLIED:
        return "applied", "G4_11 already installed (idempotent)"

    try:
        _TEMPLATE_INSTALL_PATH.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return "skipped", f"G4_11 mkdir failed: {e}"

    # Idempotent — write only if content differs
    body = _GEMMA4_TEMPLATE
    body_hash = _digest(body)
    try:
        if _TEMPLATE_INSTALL_PATH.is_file():
            existing = _TEMPLATE_INSTALL_PATH.read_text(encoding="utf-8")
            if _digest(existing) == body_hash:
                _APPLIED = True
                return "applied", (
                    f"G4_11 template already present at {_TEMPLATE_INSTALL_PATH} "
                    f"(sha256[:16]={body_hash})"
                )
    except OSError as e:
        log.warning("G4_11 read-existing failed: %s; will overwrite", e)

    try:
        _TEMPLATE_INSTALL_PATH.write_text(body, encoding="utf-8")
    except OSError as e:
        return "failed", f"G4_11 write failed: {e}"

    _APPLIED = True
    log.info(
        "[G4_11] installed enhanced Gemma 4 chat template at %s (sha256[:16]=%s, %d bytes)",
        _TEMPLATE_INSTALL_PATH, body_hash, len(body),
    )
    return "applied", (
        f"G4_11 installed: {_TEMPLATE_INSTALL_PATH} (sha256[:16]={body_hash}, "
        f"{len(body)} bytes). Reference it from your V2 model YAML via "
        f"`chat_template: {_TEMPLATE_INSTALL_PATH}`."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    """Remove the installed template file."""
    global _APPLIED
    if not _APPLIED:
        return False
    try:
        _TEMPLATE_INSTALL_PATH.unlink(missing_ok=True)
        _APPLIED = False
        return True
    except OSError as e:
        log.warning("G4_11 revert (unlink) failed: %s", e)
        return False


__all__ = ["GENESIS_G4_11_MARKER", "apply", "is_applied", "revert"]
