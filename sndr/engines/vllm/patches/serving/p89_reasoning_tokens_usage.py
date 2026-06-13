# SPDX-License-Identifier: Apache-2.0
"""P89 — completion_tokens_details.reasoning_tokens in chat usage (vendor of vllm#45471).

Upstream gap (vllm#45471, DRAFT as of 2026-06-13, "[Frontend] Add
completion_token_details to usage object in Chat Completions response
body"): the ``/v1/chat/completions`` ``usage`` object exposes
``prompt_tokens_details`` but NOT ``completion_tokens_details``. OpenAI's
chat API surfaces ``completion_tokens_details.reasoning_tokens`` so a
caller can attribute decode cost between the chain-of-thought and the
answer. In our pin the ``/v1/responses`` path already surfaces it
(``responses/serving.py`` calls ``count_reasoning_tokens`` →
``OutputTokensDetails(reasoning_tokens=...)``) but the chat path — which
every Genesis client actually uses — does not.

LIVE RELEVANCE on our stack: all 4 PROD models run the qwen3 reasoning
parser (``<think>...</think>``). The chat usage object therefore drops
the single most useful per-request signal: how many of the completion
tokens were reasoning. Surfacing it is a TPOT-attribution lever
(reasoning vs answer) and a denominator for MTP/TurboQuant tuning, at
ZERO GPU cost — the count is one O(n) token-id walk over already-emitted
ids via the parser's existing ``count_reasoning_tokens`` (depth counter
over start/end ``<think>`` ids; ``basic_parsers.py`` in the pin).

Vendor of OPEN PR vllm#45471 (studied via ``gh pr view`` + ``gh pr
diff`` 2026-06-13), 3 source hunks, adapted to the pin's file layout:

  * ``entrypoints/openai/engine/protocol.py`` — add the
    ``CompletionTokenUsageInfo`` model and wire
    ``UsageInfo.completion_tokens_details``.
  * ``entrypoints/openai/chat_completion/serving.py`` — import the new
    model; in ``chat_completion_stream_generator`` declare a per-choice
    token-id accumulator, extend it each decode step, and attach the
    reasoning count to the final usage chunk; in
    ``chat_completion_full_generator`` attach the reasoning count to the
    response usage. Both attach sites are gated on
    ``self.reasoning_parser_cls`` so non-reasoning models are byte-
    inert.

GENESIS EXTENSION the PR explicitly defers (its model only carries
``reasoning_tokens``; the PR description: "can be extended in the future
to include stuff like specdec token data"): we add the OpenAI-spec-
aligned ``accepted_prediction_tokens`` / ``rejected_prediction_tokens``
fields to ``CompletionTokenUsageInfo`` (default ``None``), forward-
compatible for surfacing MTP K=3 spec-decode efficiency per request.

  HONEST PLUMBING NOTE (verified against pristine pin g303916e93, iron
  rule #11): per-request accepted/rejected spec-decode counts are NOT
  reachable at the frontend in this pin. ``SpecDecodingStats``
  (``v1/spec_decode/metrics.py``: ``num_draft_tokens`` /
  ``num_accepted_tokens``) lives on ``IterationStats`` — an engine-STEP
  aggregate consumed by Prometheus, NOT per request. The per-request
  ``RequestOutput.metrics`` is ``RequestStateStats``, which has NO
  accept/draft fields. Threading a per-request MTP counter through
  EngineCore → output processor → ``RequestOutput`` is a separate
  hot-path-touching change, well beyond this S-effort frontend vendor.
  So P89 ships the SCHEMA (so the cost-parsing in Genesis_proxy_ai /
  agregator and the bench/dashboard can rely on a stable field shape)
  with the two prediction-token fields ``None`` until that plumbing
  lands (tracked as a Wave-3 follow-up). The bench note keys on
  ``reasoning_tokens`` today: ``tools/genesis_chat_matrix_bench.py``
  reads ``usage.completion_tokens_details.reasoning_tokens`` from the
  final streaming usage chunk and reports a per-variant "reason tok"
  column (the reasoning-vs-answer TPOT-attribution denominator and the
  MTP/TurboQuant tuning lever). The prediction fields are reserved.

GENESIS DIVERGENCE — spelling only (documented per iron rule #10): the
PR attaches via a bare ``final_usage.completion_tokens_details =
CompletionTokenUsageInfo(...)``. Our final-usage attachment carries a
distinguishing inline comment so the PR's exact structural lines stay
usable as upstream drift markers without ever matching our own emitted
text (tools/lint_drift_markers.py self-collision contract). The
``CompletionTokenUsageInfo`` model body also diverges (it carries the
two extra prediction-token fields), so the PR's exact model block is a
clean drift marker too.

Rebind analysis (verified against pristine pin g303916e93): both targets
are pure-Python frontend modules imported at server start, before any
request; a source-level text patch applied before import needs NO
runtime rebind. ``CompletionTokenUsageInfo`` is referenced only from
``serving.py`` (which the same patch imports it into) and instantiated
at the two attach sites — no other consumer exists in the pin (zero hits
for ``completion_tokens_details`` outside the responses path's distinct
``OutputTokensDetails``).

Atomicity: the two edits are one logical change (the model must exist
before ``serving.py`` instantiates it), so they apply through a
``MultiFilePatchTransaction`` — both files take the patch or neither
does, with true rollback on a commit-phase race.

Activation: opt-in via ``GENESIS_ENABLE_P89_REASONING_TOKENS_USAGE=1``
(default OFF in the registry). Self-skips when #45471 lands upstream:
drift markers below are exact substrings of the PR's form.

Author backport: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Vendor target: vllm-project/vllm#45471 (DRAFT/OPEN as of 2026-06-13).
"""
from __future__ import annotations

import logging

from sndr.engines.vllm.detection.guards import resolve_vllm_file
from sndr.kernel import (
    MultiFilePatchTransaction,
    TextPatch,
    TextPatcher,
)

log = logging.getLogger("genesis.wiring.p89_reasoning_tokens_usage")

GENESIS_P89_PROTOCOL_MARKER = (
    "Genesis P89 completion_tokens_details model "
    "(vendor of vllm#45471 + spec-decode extension) v1"
)
GENESIS_P89_SERVING_MARKER = (
    "Genesis P89 chat-usage reasoning_tokens attach "
    "(vendor of vllm#45471) v1"
)

_PROTOCOL_REL = "entrypoints/openai/engine/protocol.py"
_SERVING_REL = "entrypoints/openai/chat_completion/serving.py"

# Drift markers — exact substrings of #45471's form, taken from
# `gh pr diff 45471` on 2026-06-13. Absent in the pristine pin tree and
# deliberately NOT substrings of our own replacement text: our model
# body carries the two extra prediction-token fields, and our attach
# lines carry a distinguishing comment (lint_drift_markers self-
# collision contract).
_PROTOCOL_DRIFT_MARKERS = (
    # The PR's exact model block FOLLOWED BY UsageInfo. In the PR
    # ``reasoning_tokens`` is the model's only field so the class is
    # immediately followed by the blank lines + ``class UsageInfo``;
    # OUR model inserts the two prediction-token fields between them, so
    # this exact sequence never appears in our emitted text (lint
    # self-collision contract) yet matches the merged form exactly.
    "    reasoning_tokens: int = 0\n\n\nclass UsageInfo(OpenAIBaseModel):\n",
)
_SERVING_DRIFT_MARKERS = (
    # The PR's exact reasoning-count line. The PR names its local
    # ``reasoning_parser``; we prefix ours ``p89_reasoning_parser`` so
    # this bare form never appears in our emitted text (lint self-
    # collision contract) yet matches the PR's merged form exactly.
    "                        reasoning_parser.count_reasoning_tokens(ids)\n",
)


# ── protocol.py sub-patches ──────────────────────────────────────────
# (1) Insert the CompletionTokenUsageInfo model immediately before
#     UsageInfo (so the forward reference on the field resolves), and
#     (2) wire the completion_tokens_details field onto UsageInfo. Both
#     anchors are unique in the pin file (count==1 byte-verified at
#     g303916e93).

# Anchor head shared by the two protocol edits: the pristine UsageInfo
# block with prompt_tokens_details as its last field.
_PROTOCOL_USAGE_OLD = (
    "class UsageInfo(OpenAIBaseModel):\n"
    "    prompt_tokens: int = 0\n"
    "    total_tokens: int = 0\n"
    "    completion_tokens: int | None = 0\n"
    "    prompt_tokens_details: PromptTokenUsageInfo | None = None\n"
)

# Replacement: prepend the new model (PR verbatim reasoning_tokens +
# Genesis-extended prediction-token fields) and append the new field.
_PROTOCOL_USAGE_NEW = (
    "# [Genesis P89 vendor of vllm#45471] completion_tokens_details:\n"
    "# the OpenAI chat API surfaces per-request reasoning-token count so a\n"
    "# caller can attribute decode cost between chain-of-thought and answer.\n"
    "# The two prediction-token fields are the Genesis-extended OpenAI-spec\n"
    "# fields the PR defers (spec-decode/MTP efficiency); they stay None\n"
    "# until a per-request accept counter is plumbed onto RequestOutput\n"
    "# (RequestStateStats has no accept/draft field in this pin — the\n"
    "# SpecDecodingStats accept counters are an engine-step aggregate, not\n"
    "# per request). Shipping the schema now keeps cost-parsing stable.\n"
    "class CompletionTokenUsageInfo(OpenAIBaseModel):\n"
    "    reasoning_tokens: int = 0\n"
    "    accepted_prediction_tokens: int | None = None\n"
    "    rejected_prediction_tokens: int | None = None\n"
    "\n"
    "\n"
    "class UsageInfo(OpenAIBaseModel):\n"
    "    prompt_tokens: int = 0\n"
    "    total_tokens: int = 0\n"
    "    completion_tokens: int | None = 0\n"
    "    prompt_tokens_details: PromptTokenUsageInfo | None = None\n"
    "    completion_tokens_details: CompletionTokenUsageInfo | None = None\n"
)


def _protocol_sub_patches() -> list[TextPatch]:
    return [
        TextPatch(
            name="p89_protocol_completion_token_usage_model",
            anchor=_PROTOCOL_USAGE_OLD,
            replacement=_PROTOCOL_USAGE_NEW,
            required=True,
            upstream_merged_markers=list(_PROTOCOL_DRIFT_MARKERS),
            on_upstream_merge="abort_bundle",
        ),
    ]


# ── serving.py sub-patches ───────────────────────────────────────────

# (1) Import CompletionTokenUsageInfo alongside the existing usage models.
_SERVING_IMPORT_OLD = (
    "from vllm.entrypoints.openai.engine.protocol import (\n"
    "    DeltaMessage,\n"
    "    ErrorResponse,\n"
    "    FunctionCall,\n"
    "    PromptTokenUsageInfo,\n"
    "    RequestResponseMetadata,\n"
    "    ToolCall,\n"
    "    UsageInfo,\n"
    ")\n"
)
_SERVING_IMPORT_NEW = (
    "from vllm.entrypoints.openai.engine.protocol import (\n"
    "    # [Genesis P89 vendor of vllm#45471] CompletionTokenUsageInfo\n"
    "    CompletionTokenUsageInfo,\n"
    "    DeltaMessage,\n"
    "    ErrorResponse,\n"
    "    FunctionCall,\n"
    "    PromptTokenUsageInfo,\n"
    "    RequestResponseMetadata,\n"
    "    ToolCall,\n"
    "    UsageInfo,\n"
    ")\n"
)

# (2) Declare the per-choice token-id accumulator next to previous_texts.
#     Faithful to the PR (its own unconditional accumulator), so the
#     attach works on every streaming path including harmony/edge cases.
_SERVING_ACCUM_DECL_OLD = (
    "        # Always track previous_texts for comprehensive output logging\n"
    "        previous_texts = [\"\"] * num_choices\n"
)
_SERVING_ACCUM_DECL_NEW = (
    "        # Always track previous_texts for comprehensive output logging\n"
    "        previous_texts = [\"\"] * num_choices\n"
    "        # [Genesis P89 vendor of vllm#45471] per-choice generated\n"
    "        # token-ids, accumulated so the final usage chunk can report\n"
    "        # completion_tokens_details.reasoning_tokens via the parser's\n"
    "        # existing count_reasoning_tokens (one O(n) walk, zero GPU).\n"
    "        per_choice_token_ids: list[list[int]] = [[] for _ in range(num_choices)]\n"
)

# (3) Extend the accumulator each decode step, right where the running
#     token count is already updated.
_SERVING_ACCUM_EXT_OLD = (
    "                    # set the previous values for the next iteration\n"
    "                    previous_num_tokens[i] += len(output.token_ids)\n"
)
_SERVING_ACCUM_EXT_NEW = (
    "                    # set the previous values for the next iteration\n"
    "                    previous_num_tokens[i] += len(output.token_ids)\n"
    "                    # [Genesis P89 vendor of vllm#45471] accumulate\n"
    "                    # this choice's generated token-ids for the final\n"
    "                    # reasoning-token count.\n"
    "                    per_choice_token_ids[i].extend(output.token_ids)\n"
)

# (4) Attach completion_tokens_details to the streaming final usage,
#     gated on the reasoning parser being configured. Inserted after the
#     prompt_tokens_details block and before the final usage chunk is
#     built (so it lands on the same UsageInfo object).
_SERVING_STREAM_ATTACH_OLD = (
    "                if self.enable_prompt_tokens_details and num_cached_tokens:\n"
    "                    final_usage.prompt_tokens_details = PromptTokenUsageInfo(\n"
    "                        cached_tokens=num_cached_tokens\n"
    "                    )\n"
    "\n"
    "                final_usage_chunk = ChatCompletionStreamResponse(\n"
)
_SERVING_STREAM_ATTACH_NEW = (
    "                if self.enable_prompt_tokens_details and num_cached_tokens:\n"
    "                    final_usage.prompt_tokens_details = PromptTokenUsageInfo(\n"
    "                        cached_tokens=num_cached_tokens\n"
    "                    )\n"
    "\n"
    "                # [Genesis P89 vendor of vllm#45471] surface\n"
    "                # completion_tokens_details.reasoning_tokens for\n"
    "                # reasoning models (qwen3 on all 4 PROD models). Sum\n"
    "                # the per-choice count via the parser's existing\n"
    "                # count_reasoning_tokens walk; non-reasoning models are\n"
    "                # byte-inert (reasoning_parser_cls is None).\n"
    "                if self.reasoning_parser_cls and any(per_choice_token_ids):\n"
    "                    p89_reasoning_parser = self.reasoning_parser_cls(\n"
    "                        tokenizer,\n"
    "                        chat_template_kwargs=(\n"
    "                            self._effective_chat_template_kwargs(request)\n"
    "                        ),\n"
    "                    )\n"
    "                    p89_reasoning_count = sum(\n"
    "                        p89_reasoning_parser.count_reasoning_tokens(ids)\n"
    "                        for ids in per_choice_token_ids\n"
    "                        if ids\n"
    "                    )\n"
    "                    final_usage.completion_tokens_details = (\n"
    "                        CompletionTokenUsageInfo(\n"
    "                            reasoning_tokens=p89_reasoning_count,\n"
    "                        )\n"
    "                    )\n"
    "\n"
    "                final_usage_chunk = ChatCompletionStreamResponse(\n"
)

# (5) Attach completion_tokens_details to the non-streaming response
#     usage, gated on the reasoning parser. Inserted after the
#     prompt_tokens_details block and before final_usage_info is set.
_SERVING_FULL_ATTACH_OLD = (
    "        if self.enable_prompt_tokens_details and final_res.num_cached_tokens:\n"
    "            usage.prompt_tokens_details = PromptTokenUsageInfo(\n"
    "                cached_tokens=final_res.num_cached_tokens\n"
    "            )\n"
    "\n"
    "        request_metadata.final_usage_info = usage\n"
)
_SERVING_FULL_ATTACH_NEW = (
    "        if self.enable_prompt_tokens_details and final_res.num_cached_tokens:\n"
    "            usage.prompt_tokens_details = PromptTokenUsageInfo(\n"
    "                cached_tokens=final_res.num_cached_tokens\n"
    "            )\n"
    "\n"
    "        # [Genesis P89 vendor of vllm#45471] non-streaming\n"
    "        # completion_tokens_details.reasoning_tokens (see streaming\n"
    "        # attach above). reasoning_parser_cls None => byte-inert.\n"
    "        if self.reasoning_parser_cls:\n"
    "            p89_reasoning_parser = self.reasoning_parser_cls(\n"
    "                tokenizer,\n"
    "                chat_template_kwargs=self._effective_chat_template_kwargs(request),\n"
    "            )\n"
    "            p89_reasoning_count = sum(\n"
    "                p89_reasoning_parser.count_reasoning_tokens(list(output.token_ids))\n"
    "                for output in final_res.outputs\n"
    "            )\n"
    "            usage.completion_tokens_details = CompletionTokenUsageInfo(\n"
    "                reasoning_tokens=p89_reasoning_count,\n"
    "            )\n"
    "\n"
    "        request_metadata.final_usage_info = usage\n"
)


def _serving_sub_patches() -> list[TextPatch]:
    return [
        TextPatch(
            name="p89_serving_import",
            anchor=_SERVING_IMPORT_OLD,
            replacement=_SERVING_IMPORT_NEW,
            required=True,
            upstream_merged_markers=list(_SERVING_DRIFT_MARKERS),
            on_upstream_merge="abort_bundle",
        ),
        TextPatch(
            name="p89_serving_accumulator_decl",
            anchor=_SERVING_ACCUM_DECL_OLD,
            replacement=_SERVING_ACCUM_DECL_NEW,
            required=True,
        ),
        TextPatch(
            name="p89_serving_accumulator_extend",
            anchor=_SERVING_ACCUM_EXT_OLD,
            replacement=_SERVING_ACCUM_EXT_NEW,
            required=True,
        ),
        TextPatch(
            name="p89_serving_stream_attach",
            anchor=_SERVING_STREAM_ATTACH_OLD,
            replacement=_SERVING_STREAM_ATTACH_NEW,
            required=True,
        ),
        TextPatch(
            name="p89_serving_full_attach",
            anchor=_SERVING_FULL_ATTACH_OLD,
            replacement=_SERVING_FULL_ATTACH_NEW,
            required=True,
        ),
    ]


def _make_protocol_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_PROTOCOL_REL)
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "P89 entrypoints/openai/engine/protocol.py — "
            "CompletionTokenUsageInfo model + UsageInfo field "
            "(vendor of vllm#45471 + spec-decode extension)"
        ),
        target_file=str(target),
        marker=GENESIS_P89_PROTOCOL_MARKER,
        sub_patches=_protocol_sub_patches(),
        upstream_drift_markers=list(_PROTOCOL_DRIFT_MARKERS),
    )


def _make_serving_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_SERVING_REL)
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "P89 entrypoints/openai/chat_completion/serving.py — "
            "chat-usage reasoning_tokens attach (vendor of vllm#45471)"
        ),
        target_file=str(target),
        marker=GENESIS_P89_SERVING_MARKER,
        sub_patches=_serving_sub_patches(),
        upstream_drift_markers=list(_SERVING_DRIFT_MARKERS),
    )


def _make_transaction() -> MultiFilePatchTransaction | None:
    """Build the atomic two-file transaction, or None if a target is
    unresolved (e.g. pin layout drift)."""
    protocol_patcher = _make_protocol_patcher()
    serving_patcher = _make_serving_patcher()
    if protocol_patcher is None or serving_patcher is None:
        return None
    # Order: protocol.py FIRST (define the model) then serving.py
    # (import + instantiate it) — mirrors the dependency direction.
    return MultiFilePatchTransaction(
        [protocol_patcher, serving_patcher],
        name="P89 reasoning-tokens chat usage (vllm#45471)",
    )


def _upstream_merged_reason(txn: MultiFilePatchTransaction) -> str | None:
    """If #45471 has landed upstream, return a clean skip reason.

    The MultiFilePatchTransaction's dry-run does not inspect patcher-level
    ``upstream_drift_markers`` (the merged form may leave our anchors
    intact while adding the field — so dry-run would pass and the commit
    phase would treat the per-file drift SKIP as a race-condition
    FAILURE). We therefore pre-flight the drift check here so a merged
    pin yields a clean ("skipped", "upstream_merged ...") instead of a
    misleading "failed".
    """
    for patcher in txn.patchers:
        # Already-applied markers mean we wrote it, not upstream — skip.
        try:
            with open(patcher.target_file, encoding="utf-8") as f:
                src = f.read()
        except (OSError, UnicodeDecodeError):
            continue
        if patcher.marker in src:
            continue
        for dm in patcher.upstream_drift_markers:
            if dm in src:
                return (
                    f"P89: upstream_merged — drift marker present in "
                    f"{patcher.target_file} (vllm#45471 appears merged); "
                    "skipping without touching the file"
                )
    return None


def apply() -> tuple[str, str]:
    """Apply P89 — chat-usage completion_tokens_details.reasoning_tokens.

    Never raises. Opt-in: gated through the dispatcher on
    ``GENESIS_ENABLE_P89_REASONING_TOKENS_USAGE`` (default_on=False in
    the registry).
    """
    from sndr.dispatcher import log_decision, should_apply

    decision, reason = should_apply("P89")
    log_decision("P89", decision, reason)
    if not decision:
        return "skipped", reason

    txn = _make_transaction()
    if txn is None:
        return (
            "skipped",
            "P89: target file(s) not found "
            f"({_PROTOCOL_REL} / {_SERVING_REL})",
        )

    # Pre-flight: if #45471 has merged upstream, return a clean skip
    # instead of letting the commit phase mistake the per-file drift
    # SKIP for a race-condition FAILURE.
    merged_reason = _upstream_merged_reason(txn)
    if merged_reason is not None:
        return "skipped", merged_reason

    # apply_or_skip returns the canonical (status, message) tuple:
    #   ("applied", ...) all sub-patches committed atomically
    #   ("skipped", ...) dry-run failed OR a drift marker fired
    #   ("failed",  ...) commit-phase race, rolled back
    return txn.apply_or_skip()


def is_applied() -> bool:
    """Return True iff BOTH file markers are present (the patch is a
    single logical change across two files)."""
    protocol_patcher = _make_protocol_patcher()
    serving_patcher = _make_serving_patcher()
    if protocol_patcher is None or serving_patcher is None:
        return False
    for patcher in (protocol_patcher, serving_patcher):
        try:
            with open(patcher.target_file, encoding="utf-8") as f:
                if patcher.marker not in f.read():
                    return False
        except (OSError, UnicodeDecodeError):
            return False
    return True
