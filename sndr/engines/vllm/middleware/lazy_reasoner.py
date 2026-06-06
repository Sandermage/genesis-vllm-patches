# SPDX-License-Identifier: Apache-2.0
"""Genesis PN16 — lazy reasoner middleware (v2 architecture, 2026-05-09).

================================================================
v2 RATIONALE — WHY V1 WAS RETIRED
================================================================

PN16 v1 default-on path mutated ``request.chat_template_kwargs[
"enable_thinking"] = False`` for short prompts. Live bench on Sander's
35B PROD (TQ k8v4 + MTP K=3 + 320K ctx) measured a 28% wall_TPS drop
with 6× CV amplification (236.24 → 166.25 TPS, 6.3% → 37.6% CV) when
V1 was the only PN16 variant active — see Wave 6 closure.

Root cause analysis (PROD has prefix-caching DISABLED due to
TQ+spec-decode crash, so cache miss is NOT the mechanism):

  1. **CUDA graph dispatch mismatch.** vllm captures cudagraphs for
     specific input-shape buckets at warmup. Forcing
     ``enable_thinking=False`` renders a different chat-template
     prefix (no ``<think>`` opener) → resulting input shape may not
     match a captured bucket → eager fallback for those requests
     (5-10× slower decode for the affected calls). The 37% CV is the
     mix of graph-hit (fast) and graph-miss (slow) requests.
  2. **MTP draft model bias.** Qwen3.6 MTP draft was trained on
     thinking-enabled traces. With ``enable_thinking=False`` the
     draft's predictions are systematically less aligned with target
     output → lower acceptance rate → more wasted spec-decode work.
  3. **Mixed-batch overhead.** When some scheduled requests are
     thinking-on and others thinking-off, the engine reshapes
     batches per pattern → pipeline bubbles.

V1's "save compute on trivial prompts" goal can be served WITHOUT
mutating the chat template via output-side mechanisms (V7 below),
which preserve graph dispatch and MTP draft compatibility.

================================================================
v2 PRODUCTION PATHS
================================================================

  • Variant 3 (client override) — DEFAULT ON: if the client
    explicitly set ``chat_template_kwargs.enable_thinking`` to True
    or False, respect it. Zero cost; no mutation.
  • Variant 5 (soft cap hint) — opt-in via
    GENESIS_PN16_MAX_THINKING_TOKENS > 0. Appends a "be concise in
    <think>" hint to the last user message. Soft cap (model decides
    to comply). Adds ~30 chars to last block — minor cache impact
    only at last block.
  • Variant 7 (max_tokens hard cap) — NEW; opt-in via
    GENESIS_PN16_CLASSIFIER_MAX_TOKENS > 0. When the classifier
    detects a short trivial prompt (same heuristic as legacy V1),
    instead of mutating the template flag, **lower the request's
    max_tokens** to the configured ceiling. The chat template is
    UNCHANGED → CUDA graphs hit, MTP draft compat preserved → no
    TPS regression. Hard cap on response length — the request
    stops generating cleanly when it hits the cap. Cache-safe.
  • Variant 8 (tool-presence think-budget system msg) — NEW; opt-in
    via GENESIS_PN16_TOOL_THINK_BUDGET > 0. When the request has
    `tools` attached, prepend a system-message hint capping
    reasoning at N tokens before the tool_call. Cache-stable (the
    hint is identical across every tool-attached request). Targets
    the london_think failure class — model graphomanizes inside
    `<think>` until max_tokens cap, never emits the tool_call.
    Soft guard via prompt engineering — works under spec-decode
    where V4 LogitsProcessor strict-cap is upstream-blocked.

================================================================
v2 DEFERRED / OPT-IN LEGACY
================================================================

  • Variant 1 (template-mutation) — RETIRED from default 2026-05-09.
    Still callable by setting GENESIS_PN16_V1_LEGACY=1 (acknowledges
    the documented regression on prefix-cache-reuse / cuda-graph
    workloads). For ad-hoc / single-shot RAG endpoints where neither
    cache nor cuda-graphs benefit applies, V1 is still useful.
  • Variant 4 (LogitsProcessor strict cap): vllm v1 rejects custom
    logits processors when ``speculative_config`` is set (Genesis
    PROD = MTP K=3). To revisit when upstream lifts the restriction.
  • Variant 6 (streaming-side <think> truncator): bound TTFT via
    inline ``</think>`` injection on the SSE stream. Cache-safe and
    deterministic but doesn't save compute. Future work; tracked in
    project memory (Wave 6 deferred items).

================================================================
COMPATIBILITY
================================================================

The hook signature ``apply_hook(serving, request)`` and import path
``vllm.sndr_core.middleware.lazy_reasoner.apply_hook`` are unchanged.
V2 is a behavioural refactor of the existing PN16 wiring text-patch.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
v1 → v2 refactor 2026-05-09 (Wave 6 closure: Sander's "PN16 can be
implemented differently so we don't lose speed").
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

log = logging.getLogger("genesis.middleware.lazy_reasoner")


# ─── Operator-tunable thresholds ─────────────────────────────────────────


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name, "")
    return val.strip().lower() in ("1", "true", "yes", "on") if val else default


def _is_enabled() -> bool:
    """Master gate — same env name as PATCH_REGISTRY['PN16'].env_flag."""
    return _env_bool("GENESIS_ENABLE_PN16_LAZY_REASONER")


def _threshold_chars() -> int:
    """Char count threshold below which thinking is candidate-for-disable."""
    return _env_int("GENESIS_PN16_THRESHOLD_CHARS", 300)


def _max_thinking_tokens() -> int:
    """Soft cap — max reasoning tokens when thinking IS allowed.

    0 disables the cap (default). Non-zero engages variant 5
    (prompt-engineering soft cap via hint injection).
    """
    return _env_int("GENESIS_PN16_MAX_THINKING_TOKENS", 0)


def _classifier_max_tokens() -> int:
    """V7 hard cap — when the short-prompt classifier hits, clamp the
    request's max_tokens to this value. Cache-safe (no template
    mutation). 0 disables V7 (default)."""
    return _env_int("GENESIS_PN16_CLASSIFIER_MAX_TOKENS", 0)


def _v1_legacy_enabled() -> bool:
    """Opt-in to legacy V1 (chat_template_kwargs mutation). Default OFF
    — V1 was retired from default behavior 2026-05-09 due to cuda-graph
    dispatch mismatch + MTP draft divergence (28% TPS drop on PROD)."""
    return _env_bool("GENESIS_PN16_V1_LEGACY", default=False)


def _tool_think_budget() -> int:
    """V8 — when a request carries `tools`, prepend a system-message
    hint capping reasoning at this many tokens before the tool_call.

    Cache-stable (system prompt constant per tool request) and engages
    via prompt engineering — works under spec-decode unlike V4. Targets
    the london_think failure class: model graphomanizes inside <think>
    until max_tokens cap, never emits the tool_call.

    0 disables V8 (default)."""
    return _env_int("GENESIS_PN16_TOOL_THINK_BUDGET", 0)


# ─── Reasoning-signal detector ───────────────────────────────────────────
# Patterns that suggest the request may need chain-of-thought even on a
# short prompt. Conservative — false POSITIVE here means "we left thinking
# ON when it might have been disposable", which is the safer error.

_REASONING_SIGNAL_PATTERNS = [
    # Math / problem-solving verbs
    r"\b(calculate|compute|solve|prove|derive|integrate|differentiate"
    r"|reason|estimate|optimi[sz]e|simplify|factor|expand)\b",
    # Math / CS nouns
    r"\b(prime|matrix|vector|tensor|equation|theorem|lemma|proof"
    r"|integral|derivative|algorithm|complexity)\b",
    # Code block fence
    r"```",
    # Inline LaTeX-ish math
    r"\$[^$]+\$",
    # Arithmetic operators next to digits (basic math)
    r"[+\-*/=<>%^]\s*\d",
    r"\d\s*[+\-*/=<>%^]",
    # Programming snippet smell — class/function/return on a single line
    r"\b(class|def|function|return|import|from|public|private)\b",
    # Step-by-step request markers
    r"\b(step[- ]by[- ]step|chain[- ]of[- ]thought|explain why|how does)\b",
]
_COMPILED_SIGNAL_PATTERNS: list[re.Pattern[str]] | None = None


def _signal_patterns() -> list[re.Pattern[str]]:
    global _COMPILED_SIGNAL_PATTERNS
    if _COMPILED_SIGNAL_PATTERNS is None:
        _COMPILED_SIGNAL_PATTERNS = [
            re.compile(p, re.IGNORECASE) for p in _REASONING_SIGNAL_PATTERNS
        ]
    return _COMPILED_SIGNAL_PATTERNS


def _has_reasoning_signal(text: str) -> bool:
    """True when text contains any pattern that hints reasoning is useful."""
    for pat in _signal_patterns():
        if pat.search(text):
            return True
    return False


# ─── Prompt-shape inspectors (defensive against schema variation) ────────


def _extract_text_from_message(msg: Any) -> str:
    """Pull plain-text content from a chat message (string or content-parts)."""
    content = getattr(msg, "content", None)
    if content is None and isinstance(msg, dict):
        content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") == "text":
                    parts.append(str(p.get("text", "")))
                elif "text" in p:
                    parts.append(str(p["text"]))
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts)
    return ""


def _total_chars(request: Any) -> int:
    messages = getattr(request, "messages", None) or []
    return sum(len(_extract_text_from_message(m)) for m in messages)


def _last_user_text(request: Any) -> str:
    messages = getattr(request, "messages", None) or []
    for m in reversed(messages):
        role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)
        if role == "user":
            return _extract_text_from_message(m)
    return ""


def _has_tools(request: Any) -> bool:
    tools = getattr(request, "tools", None)
    return bool(tools)


def _has_json_schema_format(request: Any) -> bool:
    """True when response_format constrains output to a JSON schema —
    schema-constrained generation typically benefits from reasoning."""
    rf = getattr(request, "response_format", None)
    if rf is None:
        return False
    type_field = getattr(rf, "type", None) or (rf.get("type") if isinstance(rf, dict) else None)
    return type_field in ("json_schema", "json_object")


def _client_explicit_thinking_choice(request: Any) -> bool | None:
    """Returns True/False if client set chat_template_kwargs.enable_thinking
    explicitly, None otherwise."""
    ctk = getattr(request, "chat_template_kwargs", None)
    if ctk is None:
        return None
    if isinstance(ctk, dict):
        if "enable_thinking" in ctk:
            return bool(ctk["enable_thinking"])
    return None


# ─── Core decision ───────────────────────────────────────────────────────


def _should_disable_thinking(request: Any) -> tuple[bool, str]:
    """Return (decision, reason). True = disable thinking for this request.

    Conservative — ALL of the following must hold to disable:
      1. Total prompt chars below threshold
      2. No tools attached
      3. No JSON-schema response_format
      4. Last user message has no reasoning-signal pattern hits

    Any single failure keeps thinking on. False positives are intentionally
    biased toward "leave thinking on".
    """
    threshold = _threshold_chars()
    char_count = _total_chars(request)
    if char_count >= threshold:
        return False, f"prompt {char_count} chars >= threshold {threshold}"
    if _has_tools(request):
        return False, "tools attached — keep thinking for tool-call planning"
    if _has_json_schema_format(request):
        return False, "json_schema response_format — keep thinking"
    last = _last_user_text(request)
    if last and _has_reasoning_signal(last):
        return False, "reasoning-signal pattern in last user message"
    return True, (
        f"short prompt ({char_count} chars), no tools, no schema, no signal "
        f"— thinking disabled"
    )


# ─── Stats counter ────────────────────────────────────────────────────────


_STATS: dict[str, int] = {
    "total_requests": 0,
    "respect_explicit_on": 0,
    "respect_explicit_off": 0,
    "disabled_by_heuristic": 0,        # V1 (legacy opt-in only)
    "left_on_by_heuristic": 0,
    "soft_cap_hint_injected": 0,       # V5 — prompt-engineering hint
    "max_tokens_capped": 0,            # V7 — classifier-driven max_tokens cap
    "tool_budget_prepended": 0,        # V8 — tool-presence think-budget system msg
    "v1_legacy_warned": 0,             # one-shot warning emitted
    "errors": 0,
}

def get_stats() -> dict[str, int]:
    """Return current counters (for diagnostic/`/v1/genesis/stats` etc.)."""
    return dict(_STATS)


def reset_stats() -> None:
    """Reset counters (for tests)."""
    for k in _STATS:
        _STATS[k] = 0


# ─── Variant 5 — prompt-engineering soft cap ──────────────────────────────
# Soft cap engaged when GENESIS_PN16_MAX_THINKING_TOKENS > 0. Instructs
# the model directly via a hint appended to the last user message.
# Model may ignore — but works with all engine configurations including
# spec-decode (where variant 4's strict LogitsProcessor cap is blocked
# upstream; see module docstring).

_SOFT_CAP_TEMPLATE = (
    "\n\n[Genesis hint] Keep your reasoning concise — under {tokens} "
    "tokens of `<think>` — and proceed to the final answer. Be brief in "
    "the thinking block."
)


def _inject_soft_cap_hint(request: Any, max_tokens: int) -> bool:
    """Append a soft-cap reasoning-budget hint to the last user message.

    Returns True if injection succeeded, False otherwise (e.g. no user
    message to append to, or message shape doesn't allow mutation).
    """
    messages = getattr(request, "messages", None) or []
    # Find the LAST user message (working backwards)
    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)
        if role == "user":
            last_user_idx = i
            break
    if last_user_idx is None:
        return False

    target = messages[last_user_idx]
    content = getattr(target, "content", None)
    if content is None and isinstance(target, dict):
        content = target.get("content")

    hint = _SOFT_CAP_TEMPLATE.format(tokens=max_tokens)

    if isinstance(content, str):
        new_content = content + hint
    elif isinstance(content, list):
        # Append a text part rather than mutating existing parts. Find the
        # last text part and extend it for cleanest tokenization.
        new_content = list(content)
        # Append a fresh text part at end (model concatenates content parts)
        new_content.append({"type": "text", "text": hint})
    else:
        # Unknown content shape — safer to skip than mutate blindly
        return False

    # Try setattr first (pydantic model), fall back to dict mutation.
    try:
        if isinstance(target, dict):
            target["content"] = new_content
        else:
            target.content = new_content
    except Exception:
        try:
            object.__setattr__(target, "content", new_content)
        except Exception:
            return False
    return True


# ─── Variant 7 — max_tokens hard cap (cache-safe replacement for V1) ────


def _apply_max_tokens_cap(request: Any, cap: int) -> bool:
    """Lower ``request.max_tokens`` to ``cap`` if currently higher (or
    unset). The chat template is NOT touched, so CUDA graph capture and
    MTP draft compatibility are preserved (cache-safe).

    Returns True iff a cap was actually applied.
    """
    current = getattr(request, "max_tokens", None)
    if current is None:
        new_value = cap
    elif int(current) <= cap:
        return False  # already tighter than our cap — leave alone
    else:
        new_value = cap

    try:
        request.max_tokens = new_value
    except Exception:
        try:
            object.__setattr__(request, "max_tokens", new_value)
        except Exception:
            return False
    return True


# ─── Variant 8 — tool-presence think-budget system message (cache-safe) ─


# Short, prefill-cheap. Goal: minimum tokens that reliably caps
# graphomania. Empirically validated 2026-05-09 (35B PROD): this short
# form fixes london_think while adding only ~25 tokens to prompt prefill.
# Unique tag "[Genesis-PN16-V8]" distinguishes V8 from V5's "[Genesis hint]"
# so the idempotent guard doesn't false-match.
_TOOL_BUDGET_TAG = "[Genesis-PN16-V8]"
_TOOL_BUDGET_TEMPLATE = (
    f"{_TOOL_BUDGET_TAG} Keep `<think>` under "
    "{tokens} tokens before emitting the tool call."
)


def _prepend_tool_budget_system_msg(request: Any, max_tokens: int) -> bool:
    """Prepend a fixed system-message budget hint when the request has
    tools attached. Cache-stable — the hint is identical across every
    tool-attached request, so prompt-prefix caching benefits.

    Strategy:
      • If first message is `system`: append the hint to its content.
      • Else: insert a new system message at index 0 with the hint.

    Returns True iff a hint was actually inserted/appended (False if
    the hint was already present — idempotent).
    """
    messages = getattr(request, "messages", None)
    if messages is None:
        return False

    hint = _TOOL_BUDGET_TEMPLATE.format(tokens=max_tokens)
    # Idempotent guard — don't append the same hint repeatedly on retry
    # paths or chained middleware passes.
    for m in messages:
        content = getattr(m, "content", None)
        if content is None and isinstance(m, dict):
            content = m.get("content")
        if isinstance(content, str) and _TOOL_BUDGET_TAG in content:
            return False
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text", "")
                    if _TOOL_BUDGET_TAG in str(text):
                        return False

    if not messages:
        # No messages — insert system message at index 0
        try:
            messages.append({"role": "system", "content": hint})
            return True
        except Exception:
            return False

    first = messages[0]
    first_role = (
        getattr(first, "role", None)
        or (first.get("role") if isinstance(first, dict) else None)
    )
    if first_role == "system":
        # Append to existing system message
        first_content = (
            getattr(first, "content", None)
            if not isinstance(first, dict)
            else first.get("content")
        )
        if isinstance(first_content, str):
            new_content = first_content + "\n\n" + hint
        elif isinstance(first_content, list):
            new_content = list(first_content)
            new_content.append({"type": "text", "text": "\n\n" + hint})
        else:
            new_content = hint
        try:
            if isinstance(first, dict):
                first["content"] = new_content
            else:
                first.content = new_content
        except Exception:
            try:
                object.__setattr__(first, "content", new_content)
            except Exception:
                return False
        return True

    # No leading system message — insert one. Try to mirror the message
    # shape (dict vs object) of the first user/assistant message.
    if isinstance(first, dict):
        try:
            messages.insert(0, {"role": "system", "content": hint})
            return True
        except Exception:
            return False
    # Object-shaped messages — use SimpleNamespace-like construction via
    # the same class as first if possible.
    try:
        cls = type(first)
        new_msg = cls.__new__(cls)
        try:
            new_msg.role = "system"
            new_msg.content = hint
        except Exception:
            object.__setattr__(new_msg, "role", "system")
            object.__setattr__(new_msg, "content", hint)
        messages.insert(0, new_msg)
        return True
    except Exception:
        # Fallback: dict insertion (vllm tolerates mixed shapes)
        try:
            messages.insert(0, {"role": "system", "content": hint})
            return True
        except Exception:
            return False


# ─── Public hook ──────────────────────────────────────────────────────────


def apply_hook(serving: Any, request: Any) -> None:
    """Mutate request in place per PN16 lazy-reasoner v2 policy.

    Called from ``OpenAIServingChat.create_chat_completion`` near the
    top via the text-patched hook injection in
    ``patches/middleware/pn16_lazy_reasoner.py``.

    v2 default behavior (cache-safe):
      1. V3 — respect explicit client ``enable_thinking`` choice
      2. V7 — when classifier flags a short trivial prompt and
         ``GENESIS_PN16_CLASSIFIER_MAX_TOKENS > 0``, clamp
         ``request.max_tokens`` to that value (NO template mutation)
      3. V5 — when ``GENESIS_PN16_MAX_THINKING_TOKENS > 0``, append
         a concision hint to the last user message
      4. V1 (legacy template-mutation) — only fires when
         ``GENESIS_PN16_V1_LEGACY=1`` (opt-in; documented regression)

    Failure mode: any exception is caught by the wiring's try/except and
    logged at debug; the request continues unchanged.
    """
    if not _is_enabled():
        return

    _STATS["total_requests"] += 1

    # V8 — tool-presence think-budget system message. Runs BEFORE V3
    # because it's a system-prompt prepend that's compatible with any
    # client thinking-choice. Cache-stable (constant hint per tool req).
    tool_budget = _tool_think_budget()
    if tool_budget > 0 and _has_tools(request):
        if _prepend_tool_budget_system_msg(request, tool_budget):
            _STATS["tool_budget_prepended"] += 1
            log.debug(
                "PN16 V8: tool think-budget system msg prepended (%d tokens)",
                tool_budget,
            )
        # No early return — V3/V5/V7/V1 may still apply on top of V8.

    # V3 — respect explicit client choice (zero cost; no mutation)
    explicit = _client_explicit_thinking_choice(request)
    if explicit is True:
        _STATS["respect_explicit_on"] += 1
        log.debug("PN16: client set enable_thinking=True explicitly — respect")
        return
    if explicit is False:
        _STATS["respect_explicit_off"] += 1
        log.debug("PN16: client set enable_thinking=False explicitly — respect")
        return

    # Performance optimization 2026-05-09: the classifier
    # (`_should_disable_thinking`) iterates messages, computes total
    # char count, and runs ~10 regex patterns. Skip it entirely when
    # neither V7 nor V1-legacy is configured to act on its decision —
    # its only remaining consumer would be the "left_on_by_heuristic"
    # stat counter, which we approximate as "called when V8 fired" so
    # operators still see PN16 doing work.
    v7_cap = _classifier_max_tokens()
    v1_legacy_on = _v1_legacy_enabled()
    if v7_cap <= 0 and not v1_legacy_on:
        # Nothing left to do — short-circuit. V5 hint (below) doesn't
        # need classifier output either; it just needs the cap > 0.
        cap = _max_thinking_tokens()
        if cap > 0 and _inject_soft_cap_hint(request, cap):
            _STATS["soft_cap_hint_injected"] += 1
        return

    # Run the classifier ONCE; its decision feeds both V7 and V1-legacy.
    disable, reason = _should_disable_thinking(request)

    # V7 — cache-safe hard cap on max_tokens for short trivial prompts.
    # Replaces V1's "save compute on trivial prompts" goal without
    # touching the chat template (no CUDA-graph dispatch miss; no MTP
    # draft incompatibility). v7_cap was already fetched above.
    if disable and v7_cap > 0:
        if _apply_max_tokens_cap(request, v7_cap):
            _STATS["max_tokens_capped"] += 1
            log.debug(
                "PN16 V7: max_tokens capped to %d — %s", v7_cap, reason,
            )

    # V1 LEGACY — only fires under explicit opt-in. Documented regression
    # on cuda-graph or prefix-cache workloads (28% TPS drop on PROD).
    # v1_legacy_on was already fetched above.
    if disable and v1_legacy_on:
        # One-shot warn so operator notices when V1-legacy is active.
        if _STATS["v1_legacy_warned"] == 0:
            _STATS["v1_legacy_warned"] = 1
            log.warning(
                "[PN16 V1-legacy] GENESIS_PN16_V1_LEGACY=1 active — "
                "chat_template_kwargs mutation enabled. Documented to "
                "cause 28%% wall_TPS drop on cuda-graph / prefix-cache "
                "workloads (Wave 6 closure 2026-05-09). Prefer V7 (set "
                "GENESIS_PN16_CLASSIFIER_MAX_TOKENS=N) for cache-safe "
                "compute capping."
            )
        ctk = dict(getattr(request, "chat_template_kwargs", None) or {})
        ctk["enable_thinking"] = False
        try:
            request.chat_template_kwargs = ctk
        except Exception:
            object.__setattr__(request, "chat_template_kwargs", ctk)
        _STATS["disabled_by_heuristic"] += 1
        log.debug("PN16 V1-legacy: thinking disabled — %s", reason)
        return

    if disable:
        # Classifier said "trivial" but neither V7 nor V1-legacy is
        # configured to act on it. Count for visibility.
        _STATS["left_on_by_heuristic"] += 1

    # V5 — soft cap via prompt-engineering hint when operator set
    # GENESIS_PN16_MAX_THINKING_TOKENS > 0. Affects last user message;
    # works under spec-decode (V4 LogitsProcessor cap is upstream-blocked).
    cap = _max_thinking_tokens()
    if cap > 0 and _inject_soft_cap_hint(request, cap):
        _STATS["soft_cap_hint_injected"] += 1
        log.debug(
            "PN16 V5: soft cap hint injected (max_thinking_tokens=%d)", cap,
        )

    if not disable:
        _STATS["left_on_by_heuristic"] += 1
        log.debug("PN16: thinking left on — %s", reason)
