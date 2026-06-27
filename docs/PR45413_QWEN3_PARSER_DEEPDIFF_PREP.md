# Deep-diff prep — vLLM #45413 (declarative state-machine Qwen3 tool-call parser) vs P64 / P61c / PN56

Status: **PREP / TRACKING ONLY** — no patch changed. Added 2026-06-28.
Trigger: `review-on-merge` (watchlist `sweep:` row `pr: 45413`).
Owner action: execute this checklist on the **next pin bump that carries #45413**.

## Why this prep exists (iron-rule #11)

vLLM PR **#45413** — *"declarative state-machine streaming tool-call parser for
Qwen3"* — **MERGED 2026-06-15** (Gemma4 follow-up **#45588**). It is a ground-up
rewrite of the Qwen3 streaming tool-call parser as an explicit declarative state
machine. It **overlaps our qwen3coder tool-parser defense stack**:

| Genesis patch | Origin | What it does | File / anchor |
|---|---|---|---|
| **P64** | vllm#39598 | MTP streaming early-return fix + unify `</function>` emit | `qwen3coder_tool_parser.py` (2 sub-patches) |
| **P61c** | club-3090#72 | defer commit until the `<function=` header is seen | `extract_tool_calls_streaming` commit-trigger block (1) |
| **PN56** | vllm#41466 | XML parse-failure fallback / restore | `_parse_xml_function_call` try/except (1) |

All three are consolidated in
`sndr/engines/vllm/patches/tool_parsing/p64_p61c_pn56_qwen3coder_consolidated.py`.
The live registry id is **P64** (P61c / PN56 ride as `env_flag_aliases` on the
merged entry). Each absorbed patch keeps its own `TextPatcher` + marker for
failure isolation (a P64 anchor miss must not skip P61c / PN56).

**Iron rule #11:** do NOT retire a Genesis patch on a title / abstract match.
Read both sides, diff line-by-line, classify, then act. This note records the
comparison inputs so the decision on the carrying pin is mechanical, not a
guess.

## The shared failure mode

Both the upstream state machine and our three patches exist to keep a single
boundary clean: **club-3090 #178 — MTP × `qwen3_coder` argument corruption.** A
streaming tool-call *argument* gets garbled when MTP speculative tokens cross a
tool-call boundary (the parser commits a partial / mis-split delta). The needle
ladder and a plain smoke test do not see this; it shows up as a corrupted
tool-call `arguments` payload under MTP + `--tool-call-parser qwen3_coder`.

## The classify-each-patch checklist (run on the carrying pin)

For **each** of P64 / P61c / PN56, read the merged #45413 state machine and the
Genesis sub-patch side by side and assign one of:

- **RETIRE** — the state machine subsumes the patch's behavior verbatim (the
  reproduction test below flips to PASS on a pristine #45413 tree).
- **PARTIAL** — borrow the upstream shape, keep the Genesis-unique guard.
- **KEEP** — the rewrite does not cover the case the patch covers.

Concrete questions to answer from the merged source:

1. **P64 / `</function>` + MTP early-return.** Does the new state machine emit
   `</function>` exactly **once** per call **without** P64's early-return guard on
   a multi-token MTP delta? If it can double-emit or drop the close tag on an MTP
   split, P64 is **KEEP / PARTIAL**.
2. **P61c / deferred commit.** Does the state machine **defer** the commit until
   the `<function=` header is fully seen, or can it still leak a partial header
   (emit a tool-call start before the name is known)? If it can leak, P61c is
   **KEEP**.
3. **PN56 / XML parse fallback.** Does the rewrite carry an XML parse-failure
   **fallback / restore** equivalent to PN56, or will a malformed Coder XML block
   now hard-fail (raise / drop the call) instead of degrading gracefully? If no
   fallback, PN56 is **KEEP**.
4. **Regression gate FIRST.** Before retiring **anything**, run the merged parser
   against our streaming-MTP corpus — the `#45068` / `#178` MTP-split tests
   (`tests/unit/integrations/tool_parsing/`) — at temperature 0 with MTP K≥3 and
   `--tool-call-parser qwen3_coder`. A retire is only valid if (a) the
   per-patch pristine-bug reproduction flips to PASS on the #45413 tree **and**
   (b) the full corpus stays green with the Genesis patch removed.

## Retire mechanics (if and only if the checklist says RETIRE)

- Retire via the registry lifecycle (do not delete the module blind); the
  consolidated module holds three logical patches — retiring P61c / PN56 means
  dropping their `TextPatcher` + `env_flag_alias`, not the whole file, unless all
  three retire together.
- Keep the pristine-bug reproduction test as a **strict-xfail** that flips to
  FAILED if a later pin reverts #45413 (the PN375 pattern) — that is the
  drift tripwire after retirement.
- Update this `sweep:` row's outcome and move it to a closed state.

## Do-NOT list

- Do **not** retire on this prep alone — it is the input, not the decision.
- Do **not** change P64 / P61c / PN56 now; #45413 is not in the current pin's
  validated baseline as a Genesis-relevant change until a bump carries it and the
  checklist runs.
- Do **not** assume #45588 (Gemma4 follow-up) changes the Qwen3 path; track it
  separately if a Gemma4 parser decision is needed.

## Sources

- vLLM PR #45413 (merged 2026-06-15), follow-up #45588.
- club-3090 #178 — MTP × `qwen3_coder` argument corruption (the shared failure).
- Genesis: P64 (vllm#39598), P61c (club-3090#72), PN56 (vllm#41466),
  consolidated module
  `sndr/engines/vllm/patches/tool_parsing/p64_p61c_pn56_qwen3coder_consolidated.py`.
- Watchlist row: `tools/upstream_watchlist.yaml` `sweep:` `pr: 45413`.
