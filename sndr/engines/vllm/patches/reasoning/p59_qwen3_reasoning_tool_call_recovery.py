# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 59 — Qwen3 reasoning embedded tool_call recovery.

Backport of vllm-project/vllm#39055 (ZenoAFfectionate, OPEN at time of writing).

================================================================
EMPIRICAL CANDIDATE for #40831 / our degenerate-output bug, after
P58 (#40768 backport) was empirically disproven 2026-04-25.
================================================================

What the bug looks like
-----------------------
On Qwen3.5-35B-A3B-FP8 / Qwen3.6-35B-A3B-FP8 with:

  - --reasoning-parser qwen3
  - --tool-call-parser qwen3_coder

tool-calling requests (`tools=[...]`) sometimes return:

  - empty `tool_calls` list with populated `reasoning`
  - OR garbage fragments like `parameter=city`, `<<argname>`, `</parameter`
    leaking into JSON arguments string
  - OR `<tool_call><<parameter name=...>` patterns (extra `<` + tag corruption)

Plain text requests (no `tools`) on the same model are clean.

Why it happens (per PR #39055 design doc)
-----------------------------------------
1. Model emits XML tool-call markup INSIDE the `<think>...</think>` block:

      <think>
      ... reasoning text ...
      <tool_call>
      <function=Finish>
      <parameter=answer>
      204
      </parameter>
      </function>
      </tool_call>
      </think>

2. `qwen3_reasoning_parser.extract_reasoning` partitions on `</think>` and
   puts everything before it (including the embedded `<tool_call>` block)
   into the `reasoning` field.

3. Downstream `qwen3_coder` tool parser only inspects `content`. The valid
   tool_call XML in `reasoning` never reaches it.

4. Result: empty `tool_calls`, OR fragments of incomplete XML elsewhere in
   the model output get mis-parsed as garbage tokens.

Note: our existing P12 already adds awareness of `<tool_call>` as an
implicit reasoning-end marker, but only triggers when `</think>` is
ABSENT from the output. If `</think>` is present and a `<tool_call>`
block is nested before it, P12's branch doesn't help. P59 is additive:
adds extraction of nested tool_call blocks regardless of where `</think>`
sits.

Community confirmations on PR #39055
------------------------------------
- @meitalbensinai: "Also happens for me with the new Qwen 3.6 30b" (our family)
- @epheien: "encountered with both 27b and 397b in streaming"
- @jogoossens: "very hard to get qwen stable on vllm"

Status: opt-in (`GENESIS_ENABLE_P59_QWEN3_TOOL_RECOVERY=1`).

Re-anchor history
-----------------
- 2026-06-11 (preflight residual triage plan §1b): IMPORT_OLD re-anchored
  on the pristine `Iterable, Sequence` import; wrap variants A/B removed
  (anchored on dead residue — see the commented history block below);
  variant C (chained on P27's post-apply output) and variant D (pristine,
  P27-absent deployments) added; apply() gained a require-at-least-one
  wrap gate; drift markers replaced with upstream-only typed-signature
  strings per the self-collision rule.

Compatibility
-------------
- Composes cleanly with P12 (Qwen3 tool_call reasoning fix v2): P12 handles
  the `</think>`-absent case via `<tool_call>` implicit-end; P59 handles
  the `</think>`-present case where `<tool_call>` is nested in reasoning.
  On dev259+ P12's sole surviving sub-patch only rewrites the
  extract_content_ids body — no anchor overlap with P59.
- CHAINS ON P27 (BEFORE-THINK fallback): P27 applies before P59 at boot
  (PROD boot log 2026-06-10: P27 applied at line 20, P59 evaluated at
  line 153) and rewrites the `</think>`-present return site that P59 must
  wrap. Variant C anchors on P27's post-apply output; variant D covers
  P27-absent deployments. See the chain declaration comment above
  RETURN_THINK_P27_CHAIN_OLD.
- Auto-no-op once #39055 lands upstream (drift markers: upstream's TYPED
  helper signatures — see UPSTREAM_DRIFT_MARKERS).
- Non-Qwen3 deployments: parser file simply isn't loaded → patcher skips.

Risks acknowledged
------------------
- The PR fix uses a regex to detect `<tool_call>` blocks. Edge cases:
  malformed/truncated XML inside reasoning may be partially extracted.
  PR #39055's tests cover this; we lean on those.
- Streaming path is NOT addressed by this PR (PR author's own caveat).
  Our streaming clients hit a separate bug class (#40816 family).
  CROSS-REF 2026-05-16: club-3090 issue #145 independently confirms
  the streaming gap on Qwen 3.6-27B + `--reasoning-parser qwen3` +
  `enable_thinking=true` + `--tool-call-parser qwen3_coder`. The
  qwen3_coder streaming state machine fails to engage at the
  `</think>` → `<tool_call>` boundary — entire XML arrives as raw
  `delta.content`, no `delta.tool_calls` chunks, finish_reason="stop"
  instead of "tool_calls". Universal workaround on every compose
  (Genesis or not): switch `--tool-call-parser qwen3_xml`. The xml
  parser engages correctly on the streaming transition and reasoning
  still flows on `delta.reasoning` under `--reasoning-parser qwen3`.
  club-3090 has a streaming-extractor fix in internal validation for
  their v0.8.0; we monitor upstream #39056 for the canonical merge.
- This patch is in the parser layer, NOT model generation. Model behavior
  unchanged. Worst case: extraction fails on edge cases → reasoning text
  preserved as-is, parse fallback to original behavior.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Investigation used automated source navigation.
"""
from __future__ import annotations

import logging

# Audit A-19 (2026-05-05): tightly coupled subpatches — both apply
# or both stay un-applied. Shared marker is acceptable here because the
# subpatches together form one logical fix; partial application is not
# desired anyway. _AUDIT_A19_EXEMPT documents this intentional design.
_AUDIT_A19_EXEMPT = True  # tightly coupled subpatches
import os  # noqa: E402  — import after A-19 exemption marker

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root  # noqa: E402
from sndr.kernel import (  # noqa: E402
    TextPatcher,
    TextPatchResult,
    TextPatch,
)
# Chain provider: variant C's anchor is imported from P27's module so it
# stays byte-identical to P27's post-apply output by construction (see the
# chain declaration comment above RETURN_THINK_P27_CHAIN_OLD).
from sndr.engines.vllm.patches.reasoning.p27_reasoning_before_think import (  # noqa: E402
    _NEW_NONSTREAM_RETURN_PR35687 as _P27_NONSTREAM_RETURN_POST_APPLY,
)

log = logging.getLogger("genesis.wiring.p59_qwen3_reasoning_tool_call_recovery")

# v7.14 (2026-06-11): content-version bump for the §1b re-anchor batch
# (variants C/D + require-at-least-one gate). v7.13-marked files never
# existed on dev259+ (old IMPORT_OLD predates the Iterable import).
GENESIS_P59_MARKER = "Genesis P59 Qwen3 reasoning embedded tool_call recovery v7.14"

UPSTREAM_DRIFT_MARKERS = [
    # Verified via `gh pr diff 39055` (2026-06-11, PR state OPEN): upstream's
    # helper carries TYPED signatures —
    #     def _split_embedded_tool_calls(
    #         reasoning: str | None,
    #         content: str | None,
    #     ) -> tuple[str | None, str | None]:
    # and
    #     def _collect_or_keep(match: re.Match[str]) -> str:
    # Our backport injects UNTYPED variants of both, so these strings can
    # only appear when #39055 itself lands in the pin. Self-collision rule
    # (triage plan §6 / PN353A bug class): a drift marker must never be a
    # substring of this patch's own replacement text or marker line; the
    # previous marker `_split_embedded_tool_calls` violated that.
    "def _split_embedded_tool_calls(\n        reasoning: str | None,",
    "def _collect_or_keep(match: re.Match[str]) -> str:",
]


def _is_enabled() -> bool:
    """Env-gate. Off by default — opt-in via:
    GENESIS_ENABLE_P59_QWEN3_TOOL_RECOVERY=1
    """
    return os.environ.get(
        "GENESIS_ENABLE_P59_QWEN3_TOOL_RECOVERY", ""
    ).strip().lower() in ("1", "true", "yes", "on")


# ─── Sub-patch 1: add `import re` before the existing collections import ────
# Re-anchored 2026-06-11: pristine dev259+ imports Iterable too. Pristine
# qwen3_reasoning_parser.py lines 4-5 (verified count==1):
#     from collections.abc import Iterable, Sequence
#     from typing import TYPE_CHECKING

IMPORT_OLD = (
    "from collections.abc import Iterable, Sequence\n"
    "from typing import TYPE_CHECKING"
)

IMPORT_NEW = (
    "import re  # [Genesis P59 vllm#39055]\n"
    "from collections.abc import Iterable, Sequence\n"
    "from typing import TYPE_CHECKING"
)


# ─── Sub-patch 2: insert _EMBEDDED_TOOL_CALL_RE module-level constant ─────
# Anchor on the blank line just before `class Qwen3ReasoningParser`.

REGEX_OLD = (
    "if TYPE_CHECKING:\n"
    "    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest\n"
    "    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest\n"
    "    from vllm.tokenizers import TokenizerLike\n"
    "\n"
    "\n"
    "class Qwen3ReasoningParser(BaseThinkingReasoningParser):"
)

REGEX_NEW = (
    "if TYPE_CHECKING:\n"
    "    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest\n"
    "    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest\n"
    "    from vllm.tokenizers import TokenizerLike\n"
    "\n"
    "\n"
    "# [Genesis P59 vllm#39055] regex for extracting nested tool_call blocks.\n"
    "_EMBEDDED_TOOL_CALL_RE = re.compile(\n"
    "    r\"<tool_call>(.*?)</tool_call>|<tool_call>.*$\",\n"
    "    re.DOTALL,\n"
    ")\n"
    "\n"
    "\n"
    "class Qwen3ReasoningParser(BaseThinkingReasoningParser):"
)


# ─── Sub-patch 3: insert _split_embedded_tool_calls staticmethod ───────────
# Anchor on the `end_token` property's body so we can insert just after it.

# Anchor only on the end_token property body — the line that follows differs
# between v5.12 monolith P12 (`is_reasoning_end(self, input_ids: Sequence[int])`)
# and modular P12 (`is_reasoning_end(self, input_ids):`). Insert helper right
# after end_token's return statement so it's the FIRST sibling member.
METHOD_OLD = (
    "    @property\n"
    "    def end_token(self) -> str:\n"
    "        \"\"\"The token that ends reasoning content.\"\"\"\n"
    "        return \"</think>\""
)

METHOD_NEW = (
    "    @property\n"
    "    def end_token(self) -> str:\n"
    "        \"\"\"The token that ends reasoning content.\"\"\"\n"
    "        return \"</think>\"\n"
    "\n"
    "    @staticmethod\n"
    "    def _split_embedded_tool_calls(\n"
    "        reasoning,\n"
    "        content,\n"
    "    ):\n"
    "        \"\"\"[Genesis P59 vllm#39055] Promote tool_call XML out of reasoning.\n"
    "\n"
    "        Qwen3.5/3.6 models can emit XML tool calls before </think>. The\n"
    "        downstream tool parser only inspects content, so embedded tool\n"
    "        calls would otherwise be lost. This helper extracts well-formed\n"
    "        <tool_call>...</tool_call> blocks from reasoning and prepends\n"
    "        them to content so qwen3_coder can parse them normally.\n"
    "        \"\"\"\n"
    "        if (\n"
    "            not reasoning\n"
    "            or \"<tool_call>\" not in reasoning\n"
    "            or \"<function=\" not in reasoning\n"
    "        ):\n"
    "            return reasoning, content\n"
    "\n"
    "        extracted_blocks = []\n"
    "\n"
    "        def _collect_or_keep(match):\n"
    "            block = match.group(0)\n"
    "            if \"<function=\" not in block:\n"
    "                return block\n"
    "            extracted_blocks.append(block.strip())\n"
    "            return \"\"\n"
    "\n"
    "        remaining_reasoning = _EMBEDDED_TOOL_CALL_RE.sub(\n"
    "            _collect_or_keep, reasoning\n"
    "        )\n"
    "        remaining_reasoning = remaining_reasoning.strip() or None\n"
    "\n"
    "        if not extracted_blocks:\n"
    "            return reasoning, content\n"
    "\n"
    "        content_parts = [\"\\n\\n\".join(extracted_blocks)]\n"
    "        if content:\n"
    "            content_parts.append(content)\n"
    "        merged_content = \"\\n\\n\".join(\n"
    "            part for part in content_parts if part\n"
    "        ) or None\n"
    "        return remaining_reasoning, merged_content"
)


# ─── Sub-patches 4C/4D: wrap the </think>-present return ─────────────────
# Two variants; we expect EXACTLY ONE to match per file state, and apply()
# enforces require-at-least-one (see _CORE_WRAP_SUB_NAMES).
#
# RETIRED variants A/B (2026-06-11, preflight residual triage plan §1b).
# Both anchored on DEAD RESIDUE and could never match dev259+:
#   - Variant A targeted the v5.12 MONOLITH P12 layout via its injected
#     comment line; P12's monolith sub-patches were retired 2026-06-08
#     (superseded by upstream #35687, MERGED 2026-04-24), so no live
#     sibling emits that text anymore.
#   - Variant B targeted the pre-#35687 `final_content` aliasing shape
#     that #35687 removed from upstream.
# With both soft-skipping (required=False) the patch false-reported
# "applied" while its core </think>-present wrap was missing — the helper
# was injected as dead code. Constants preserved as commented history:
#
# RETURN_THINK_MONOLITH_OLD = (
#     "        # [Genesis v5.12] PR #35687: 3-way branch with <tool_call>\n"
#     "        if self.end_token in model_output:\n"
#     "            reasoning, _, content = model_output.partition(self.end_token)\n"
#     "            return reasoning, content or None"
# )
#
# RETURN_THINK_MONOLITH_NEW = (
#     "        # [Genesis v5.12] PR #35687: 3-way branch with <tool_call>\n"
#     "        if self.end_token in model_output:\n"
#     "            reasoning, _, content = model_output.partition(self.end_token)\n"
#     "            # [Genesis P59 vllm#39055] extract nested tool_call from reasoning\n"
#     "            return self._split_embedded_tool_calls(reasoning, content or None)"
# )
#
# RETURN_THINK_MODULAR_OLD = (
#     "        final_content = content or None\n"
#     "        return reasoning, final_content"
# )
#
# RETURN_THINK_MODULAR_NEW = (
#     "        final_content = content or None\n"
#     "        # [Genesis P59 vllm#39055] extract nested tool_call from reasoning\n"
#     "        return self._split_embedded_tool_calls(reasoning, final_content)"
# )

# ── Chain declaration (preflight CHAINED_ANCHOR convention) ───────────────
# P59 CHAINS ON P27: apply-order dependency P27-before-P59. The boot
# orchestrator already runs them in that order (PROD boot log 2026-06-10:
# P27 applied at line 20, P59 evaluated at line 153). The registry
# `requires_patches` entry for P59 is intentionally NOT updated in this
# batch (registry.py is owned by the parallel hygiene batch); this comment
# plus the anchor-import below are the canonical chain declaration.
#
# Variant C's anchor is P27's post-apply output, imported from
# p27_reasoning_before_think._NEW_NONSTREAM_RETURN_PR35687 so it is
# byte-identical by construction. The preflight chain pass
# (tools/pin_preflight.py::reclassify_chained) recognizes the dependency
# because the anchor is a substring of P27's `_replacement_blob` on the
# same target file, and reclassifies an anchor miss on the pristine tree
# as CHAINED_ANCHOR instead of DRIFT_ANCHOR.

# Variant C: P27-applied layout (P27's BEFORE-THINK prepend before return).
RETURN_THINK_P27_CHAIN_OLD = _P27_NONSTREAM_RETURN_POST_APPLY

# Tail return line of P27's replacement that variant C wraps. Derivation
# is validated by _P27_CHAIN_DERIVATION_OK; if P27's injected text ever
# changes shape, variant C is dropped from the sub-patch list and the
# require-at-least-one gate in apply() fails loudly on post-P27 files
# instead of a no-op replacement masquerading as applied.
_P27_POST_APPLY_RETURN_LINE = "            return reasoning, content or None"

RETURN_THINK_P27_CHAIN_NEW = RETURN_THINK_P27_CHAIN_OLD.replace(
    _P27_POST_APPLY_RETURN_LINE,
    "            # [Genesis P59 vllm#39055] extract nested tool_call from reasoning\n"
    "            return self._split_embedded_tool_calls(reasoning, content or None)",
    1,
)

_P27_CHAIN_DERIVATION_OK = (
    RETURN_THINK_P27_CHAIN_OLD.count(_P27_POST_APPLY_RETURN_LINE) == 1
    and RETURN_THINK_P27_CHAIN_NEW != RETURN_THINK_P27_CHAIN_OLD
)

# Variant D: pristine layout for P27-absent deployments. Anchor quoted
# byte-exactly from pristine qwen3_reasoning_parser.py lines 142-144
# (verified count==1 on dev259+):
#         if self.end_token in model_output:
#             reasoning, _, content = model_output.partition(self.end_token)
#             return reasoning, content or None
RETURN_THINK_PRISTINE_OLD = (
    "        if self.end_token in model_output:\n"
    "            reasoning, _, content = model_output.partition(self.end_token)\n"
    "            return reasoning, content or None"
)

RETURN_THINK_PRISTINE_NEW = (
    "        if self.end_token in model_output:\n"
    "            reasoning, _, content = model_output.partition(self.end_token)\n"
    "            # [Genesis P59 vllm#39055] extract nested tool_call from reasoning\n"
    "            return self._split_embedded_tool_calls(reasoning, content or None)"
)

# Require-at-least-one set: the patch is only functional when ONE of the
# core </think>-present wrap variants landed; the helper alone is dead
# code. apply() refuses to report "applied" otherwise.
_CORE_WRAP_SUB_NAMES = (
    "p59_wrap_think_return_p27_chain",
    "p59_wrap_think_return_pristine",
)


# ─── Sub-patch 5: wrap the truncated-output return ───────────────────────
# Same shape in both layouts (P12/P27 don't touch this branch).

RETURN_TRUNC_OLD = (
    "        # Thinking enabled but no </think>: output was truncated.\n"
    "        # Everything generated so far is reasoning.\n"
    "        return model_output, None"
)

RETURN_TRUNC_NEW = (
    "        # Thinking enabled but no </think>: output was truncated.\n"
    "        # Everything generated so far is reasoning.\n"
    "        # [Genesis P59 vllm#39055] still try to extract embedded tool_call\n"
    "        return self._split_embedded_tool_calls(model_output, None)"
)


def _make_patcher_for_target(target_file: str) -> TextPatcher:
    """Build the P59 patcher for an explicit target path.

    Split out of _make_patcher so unit tests can exercise the REAL
    sub-patch layout against synthetic files without a vllm tree.
    """
    sub_patches = [
        TextPatch(
            name="p59_import_re",
            anchor=IMPORT_OLD,
            replacement=IMPORT_NEW,
            required=True,
        ),
        TextPatch(
            name="p59_module_regex",
            anchor=REGEX_OLD,
            replacement=REGEX_NEW,
            required=True,
        ),
        TextPatch(
            name="p59_helper_method",
            anchor=METHOD_OLD,
            replacement=METHOD_NEW,
            required=True,
        ),
    ]
    # Variants C/D for the </think>-present return — required=False so
    # whichever file state is present wins; apply() then enforces that at
    # least one of them actually landed (_CORE_WRAP_SUB_NAMES gate).
    if _P27_CHAIN_DERIVATION_OK:
        sub_patches.append(
            TextPatch(
                name="p59_wrap_think_return_p27_chain",
                anchor=RETURN_THINK_P27_CHAIN_OLD,
                replacement=RETURN_THINK_P27_CHAIN_NEW,
                required=False,
            )
        )
    else:
        # P27's injected text changed shape — fail loud via the apply()
        # gate on post-P27 files rather than ship a no-op replacement.
        log.warning(
            "[P59] P27 chain derivation failed — P27's injected return "
            "shape changed; variant C dropped. Re-derive "
            "RETURN_THINK_P27_CHAIN_* against the current P27 module."
        )
    sub_patches.extend(
        [
            TextPatch(
                name="p59_wrap_think_return_pristine",
                anchor=RETURN_THINK_PRISTINE_OLD,
                replacement=RETURN_THINK_PRISTINE_NEW,
                required=False,
            ),
            TextPatch(
                name="p59_wrap_trunc_return",
                anchor=RETURN_TRUNC_OLD,
                replacement=RETURN_TRUNC_NEW,
                required=False,
            ),
        ]
    )
    return TextPatcher(
        patch_name="P59 Qwen3 reasoning embedded tool_call recovery",
        target_file=target_file,
        marker=GENESIS_P59_MARKER,
        sub_patches=sub_patches,
        upstream_drift_markers=UPSTREAM_DRIFT_MARKERS,
    )


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("reasoning/qwen3_reasoning_parser.py")
    if target is None:
        return None
    return _make_patcher_for_target(str(target))


def apply() -> tuple[str, str]:
    """Apply P59 wiring (up to 6 sub-patches in one file). Never raises.

    All-or-nothing: if any required anchor drifts, abort the whole group.
    Require-at-least-one: APPLIED only counts when one of the core
    </think>-present wrap variants (C: P27 chain, D: pristine) actually
    landed — otherwise the injected helper is dead code and we report
    "failed" loudly instead of false-reporting success (2026-06-11 fix;
    the retired residue variants A/B used to mask exactly this).
    Idempotent + auto-no-op once #39055 lands upstream.
    """
    from sndr.dispatcher import should_apply, log_decision
    decision, reason = should_apply("P59")
    log_decision("P59", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "qwen3_reasoning_parser.py not found"

    result, failure = patcher.apply()

    if result == TextPatchResult.APPLIED:
        applied = list(patcher.applied_sub_patches)
        if not set(applied).intersection(_CORE_WRAP_SUB_NAMES):
            return (
                "failed",
                "P59 anchors partially applied but NO core </think>-present "
                "wrap variant matched (applied: "
                + (", ".join(applied) or "none")
                + ") — helper injected as dead code. Re-anchor variants C/D "
                "against the current pin (and check P27 apply order) before "
                "serving traffic.",
            )
        return (
            "applied",
            f"P59 backport applied ({len(applied)} sub-patches: "
            + ", ".join(applied)
            + ") in qwen3_reasoning_parser.py. Tool_call XML inside "
            "<think>...</think> reasoning now extracted and routed to "
            "content for qwen3_coder parser. Validate with blue/green "
            "reproducer suite before serving traffic.",
        )
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already applied this image layer (idempotent)"
    if result == TextPatchResult.SKIPPED:
        msg = failure.reason if failure else "anchor not found"
        detail = failure.detail if failure else ""
        return (
            "skipped",
            f"{msg} ({detail}) — likely #39055 already merged upstream OR "
            "anchor drifted (P27-modified file changed shape). "
            "Re-anchor needed.",
        )
    return "failed", failure.reason if failure else "unknown failure"
