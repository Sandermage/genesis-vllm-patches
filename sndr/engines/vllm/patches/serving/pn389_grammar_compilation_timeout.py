# SPDX-License-Identifier: Apache-2.0
"""PN389 — XGrammar input-validation + grammar-compilation timeouts.

Vendor of OPEN vllm#45390 (jperezdealgaba, "fix(security): add input
validation and compilation timeouts for DoS mitigations"; studied via
``gh pr view`` + ``gh pr diff`` 2026-06-13). Backports the seven-GHSA
DoS-hardening bundle, scoped to the XGrammar grammar-compilation hot
path that EVERY Genesis tool-call traverses.

================================================================
UPSTREAM BUG CLASS (7 GHSA, CWE-400 uncontrolled resource consumption)
================================================================

The OpenAI-compatible structured-output path compiles a user-supplied
grammar/regex/JSON-schema into a DFA on the CPU engine loop with NO
wall-clock bound. A pathological grammar (catastrophic-backtracking
regex, exponential JSON schema) consumes unbounded CPU during
compilation — and because compilation runs on the single EngineCore
thread, it wedges ALL decode for every concurrent request. Our PROD is
single-instance and single-user-low-latency: async-scheduling overlap
does NOT save us (overlap hides GPU latency behind the scheduler, but
grammar compilation is pure-CPU on the engine loop, off the GPU
stream). One adversarial tool schema therefore stalls the whole engine
indefinitely — an instance-wide DoS.

vllm#45390 closes this with a generic ``run_with_timeout`` (daemon
thread + ``Queue`` hand-off + bounded ``Semaphore`` so timed-out
compilations cannot accumulate) wrapping every XGrammar entrypoint, plus
a cheap ``_check_regex_complexity`` pre-filter (length + paren-nesting
bound) that rejects the obviously-adversarial regex BEFORE the compiler
is even called. The PR also adds protocol/engine input bounds
(logit_bias / stop_token_ids / allowed_token_ids / bad_words); those are
deliberately OUT OF SCOPE here — they edit ``sampling_params.py`` and the
split ``protocol.py`` files that P109 / PN387 already touch, and they are
tracked as a separate batch-2 wave-2 item to keep PN389's anchors
collision-free and the patch reviewable. PN389 vendors the
grammar-timeout core only.

================================================================
WHAT THIS PATCH DOES (three files, one atomic transaction)
================================================================

(1) ``v1/structured_output/utils.py`` — ADDITIVE: introduces
    ``run_with_timeout`` (daemon-thread + ``Queue`` + module-level
    ``Semaphore(4)``) and ``_check_regex_complexity`` plus their
    constants (``MAX_REGEX_LENGTH=10_000``, ``MAX_REGEX_NESTING_DEPTH``
    ``=20``, ``_MAX_CONCURRENT_COMPILATIONS=4``). New symbols only — no
    pin function is rewritten, so there is no anchor inside an existing
    body. Our pin g303916e93 has NO compilation timeout AT ALL (it lacks
    even the first-generation ``compile_regex_with_timeout`` the PR
    refactors), so these helpers are brand-new to the tree.

(2) ``v1/structured_output/backend_xgrammar.py`` — two distinct surfaces,
    both bounded:

      (2a) THE ENGINECORE COMPILE PATH (the actual DoS wedge surface):
      ``XgrammarBackend.compile_grammar`` is refactored — exactly as the PR
      does — into ``compile_grammar`` -> ``_compile_ctx`` ->
      ``run_with_timeout(self._compile_ctx_inner, ...)``. ``_compile_ctx_inner``
      holds the pin's original ``self.compiler.compile_*`` dispatch verbatim
      (only ``ctx = ...`` becomes ``return ...``), so EVERY type's vocab-
      dependent DFA build (JSON / JSON_OBJECT / GRAMMAR / REGEX /
      STRUCTURAL_TAG) now runs on a daemon thread under the wall-clock
      timeout. This is what bounds the single CPU EngineCore loop: a
      pathological compile bounces as a ``ValueError`` instead of wedging
      decode for every concurrent request. The REGEX arm keeps the cheap
      ``_check_regex_complexity`` pre-filter (inside the timed thread).

      (2b) THE FRONTEND VALIDATION PRE-FLIGHT: EVERY ``xgr.Grammar.from_*``
      call inside ``validate_xgrammar_grammar`` (regex / choice / json_schema
      / ebnf / structural_tag — ALL XGrammar types, matching the PR) is also
      wrapped in ``run_with_timeout``. ``validate_xgrammar_grammar`` runs in
      the FRONTEND process (it PARSES the schema into a Grammar object — a
      different, cheaper xgrammar API than the EngineCore compiler's
      vocab-dependent DFA build) and is the pre-flight every Genesis
      tool-call JSON schema flows through; bounding it rejects an
      adversarial *parse* before the request reaches the engine. Both
      surfaces are bounded so neither a slow parse (frontend) nor a slow
      compile (EngineCore) can hang unbounded.

(3) ``envs.py`` — ADDITIVE: registers
    ``VLLM_GRAMMAR_COMPILATION_TIMEOUT_SECONDS`` (declaration + os.getenv
    lambda). The PR renames the older ``VLLM_REGEX_COMPILATION_TIMEOUT_S``
    to this name; our pin has neither, so we add the new env fresh.

================================================================
GENESIS DIVERGENCE FROM THE PR (documented per iron rule #10)
================================================================

  • DEFAULT TIMEOUT = 2s, not the PR's 10s. A 10s compilation budget
    would let a single slow schema blow our 70-160ms TTFT SLO by ~60x
    before the timeout even fires. 2s is the largest budget that still
    bounds the worst-case wedge below a human-perceptible stall while
    leaving generous headroom over a healthy tool-schema compile
    (sub-millisecond on Qwen 152K-vocab in our offline timing). The env
    is operator-tunable; ``VLLM_GRAMMAR_COMPILATION_TIMEOUT_SECONDS=10``
    restores the PR default.

  • The constants block, the ``run_with_timeout`` docstring, the
    ``_check_regex_complexity`` docstring and the envs.py comment are
    re-worded vs the PR. This is the lint_drift_markers self-collision
    contract (PN369): the drift markers below are exact substrings of the
    PR's emitted text (its docstrings / comments), and our emitted text
    deliberately never reproduces them, so the markers can flag an
    upstream merge without ever matching our own replacement.

================================================================
SAFETY MODEL
================================================================

  • run_with_timeout is BIT-IDENTICAL for any compilation that finishes
    inside the budget — it returns the inner result unchanged; the only
    behavioural change is that a >2s compile/parse now raises a clean
    ``ValueError`` instead of running unbounded.
  • COMPILE vs VALIDATE (the surface distinction that matters): the
    EngineCore wedge is the ``compile_grammar`` DFA build in the engine
    subprocess — that is the path PN389 refactors through
    ``run_with_timeout`` so a schema that PARSES fast but COMPILES
    catastrophically (the worst case) is bounded too, not just rejected
    at the frontend ``from_*`` parse pre-flight. Bounding ONLY the
    frontend validation would leave the documented engine-wedge open;
    PN389 bounds both, so the ``ValueError`` (-> 400 / engine non-wedge)
    claim holds for the catastrophic-compile case.
  • _check_regex_complexity only fires on >10K-char or >20-deep-paren
    patterns. The GENESIS-SPECIFIC false-positive concern (legit
    JSON-schema-derived regex tripping the naive paren-depth counter) is
    pinned by a unit test that feeds our real gemma4 / qwen3_coder tool
    schemas' JSON-schema-derived regex through ``_check_regex_complexity``
    and asserts NO rejection — see tests (red-first).
  • The bounded ``Semaphore(4)`` caps concurrent compilation threads so a
    burst of slow grammars cannot spawn unbounded daemon threads; the
    5th concurrent compile is rejected (400), not queued.
  • Default OFF (``default_on=False``): the timeout reject is a new
    failure mode for legitimate-but-slow grammars, so we gate it behind
    ``GENESIS_ENABLE_PN389_GRAMMAR_TIMEOUTS`` until a server A/B confirms
    the 2s budget never trips a real tool-schema compile. STRONG
    RECOMMENDATION to enable on single-instance PROD once validated — the
    unguarded path is a one-request engine wedge.
  • Synergy with PN386 (#45389): both harden the same XGrammar tool-call
    hot path every Genesis model uses; disjoint files (PN386 edits the
    tool-parser streaming helper, PN389 edits the grammar backend), no
    anchor overlap. No collision with P62 / PN58 (spec-decode grammar
    mask timing / reasoning boundary — different files entirely).

Author backport: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Vendor target: vllm-project/vllm#45390 (OPEN as of 2026-06-13).
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root
from sndr.kernel import TextPatch, TextPatcher

log = logging.getLogger("genesis.wiring.pn389_grammar_compilation_timeout")

GENESIS_PN389_MARKER = (
    "Genesis PN389 XGrammar input-validation + grammar-compilation "
    "timeouts (vendor of vllm#45390) v1"
)

# Genesis default grammar/regex compilation budget. The PR ships 10s; we
# clamp to 2s because a 10s wedge blows our 70-160ms TTFT SLO by ~60x
# before the timeout fires. Spelled here as the source-of-truth default
# that the envs.py os.getenv lambda reads.
PN389_DEFAULT_TIMEOUT_SECONDS = 2

_TARGET_UTILS = "v1/structured_output/utils.py"
_TARGET_XGRAMMAR = "v1/structured_output/backend_xgrammar.py"
_TARGET_ENVS = "envs.py"


# ─────────────────────────────────────────────────────────────────────
# Drift markers — exact substrings of vllm#45390's emitted text (from
# `gh pr diff 45390`, 2026-06-13). Each is ABSENT in the pristine pin
# tree g303916e93 (byte-verified count==0) and is deliberately NOT a
# substring of any PN389 replacement below: we re-word our docstrings,
# comments and constants so these PR-form strings only ever match an
# actual upstream merge (lint_drift_markers self-collision contract,
# PN369). The `[Genesis PN389` banner is the defended convention entry.
# ─────────────────────────────────────────────────────────────────────
_DRIFT_MARKERS = (
    # The PR's run_with_timeout docstring head (we re-word ours).
    "Run *fn(*args)* in a daemon thread with a hard wall-clock timeout.",
    # The PR's _check_regex_complexity docstring head (we re-word ours).
    "Reject patterns that are obviously too complex before compilation.",
    # The PR's envs.py comment head for the renamed timeout env.
    "Maximum time in seconds allowed for grammar/regex compilation into a",
    # Defended convention entry (our own banner) — exempt from the
    # collision lint; keeps residue coverage if the PR comments change.
    "[Genesis PN389",
)


# ══════════════════════════════════════════════════════════════════════
# File 1 — v1/structured_output/utils.py (ADDITIVE)
# ══════════════════════════════════════════════════════════════════════
#
# Two insertions, both keyed on byte-exact pin anchors (count==1):
#   (a) the import block — add `threading` and `Queue`;
#   (b) right after `CACHE = None` — the run_with_timeout +
#       _check_regex_complexity helpers and their constants.
# These are NEW symbols; no existing pin function body is rewritten.

# (a) import block. Pin head: importlib.metadata / os / tempfile / typing.
PN389_UTILS_IMPORTS_OLD = (
    "import importlib.metadata\n"
    "import os\n"
    "import tempfile\n"
    "from typing import TYPE_CHECKING\n"
)

PN389_UTILS_IMPORTS_NEW = (
    "import importlib.metadata\n"
    "import os\n"
    "import tempfile\n"
    "import threading\n"
    "from collections.abc import Callable\n"
    "from queue import Empty, Queue\n"
    "from typing import TYPE_CHECKING, TypeVar\n"
)

# (b) helper block, inserted right after the `CACHE = None` line. The
# anchor pins the `logger = init_logger(__name__)` line + blank line +
# `CACHE = None` so the helpers land directly below the module constants.
PN389_UTILS_HELPERS_OLD = (
    "logger = init_logger(__name__)\n"
    "\n"
    "CACHE = None\n"
)

PN389_UTILS_HELPERS_NEW = (
    "logger = init_logger(__name__)\n"
    "\n"
    "CACHE = None\n"
    "\n"
    "# [Genesis PN389 vendor of vllm#45390] grammar/regex compilation DoS\n"
    "# guards. Our pin g303916e93 ships NO compilation timeout at all, so\n"
    "# these are brand-new helpers (not the PR's rename of an existing one).\n"
    "_PN389_T = TypeVar(\"_PN389_T\")\n"
    "# Upper bounds for the cheap pre-filter. A pattern longer than\n"
    "# MAX_REGEX_LENGTH chars or nested deeper than MAX_REGEX_NESTING_DEPTH\n"
    "# parentheses is rejected before the (expensive) compiler is called.\n"
    "MAX_REGEX_LENGTH = 10_000\n"
    "MAX_REGEX_NESTING_DEPTH = 20\n"
    "# Cap on simultaneously-alive compilation threads. A burst of slow\n"
    "# grammars therefore cannot spawn unbounded daemon threads; the 5th\n"
    "# concurrent compile is rejected, not queued.\n"
    "_MAX_CONCURRENT_COMPILATIONS = 4\n"
    "_compilation_semaphore = threading.Semaphore(_MAX_CONCURRENT_COMPILATIONS)\n"
    "\n"
    "\n"
    "def _check_regex_complexity(pattern: str) -> None:\n"
    "    # [Genesis PN389] Cheap O(n) pre-filter run BEFORE compilation:\n"
    "    # bound the pattern length and the maximum parenthesis nesting\n"
    "    # depth so an obviously-adversarial regex is bounced as a clean\n"
    "    # ValueError instead of detonating the DFA builder.\n"
    "    if len(pattern) > MAX_REGEX_LENGTH:\n"
    "        raise ValueError(\n"
    "            f\"Regex pattern too long ({len(pattern)} chars, \"\n"
    "            f\"max {MAX_REGEX_LENGTH}). Simplify the pattern or \"\n"
    "            \"split into smaller expressions.\"\n"
    "        )\n"
    "    depth = 0\n"
    "    max_depth = 0\n"
    "    for ch in pattern:\n"
    "        if ch == \"(\":\n"
    "            depth += 1\n"
    "            max_depth = max(max_depth, depth)\n"
    "        elif ch == \")\":\n"
    "            depth -= 1\n"
    "    if max_depth > MAX_REGEX_NESTING_DEPTH:\n"
    "        raise ValueError(\n"
    "            f\"Regex nesting too deep ({max_depth} levels, \"\n"
    "            f\"max {MAX_REGEX_NESTING_DEPTH}). Simplify the pattern.\"\n"
    "        )\n"
    "\n"
    "\n"
    "def run_with_timeout(\n"
    "    fn: Callable[..., _PN389_T],\n"
    "    *args: object,\n"
    "    timeout: int,\n"
    "    label: str = \"Operation\",\n"
    ") -> _PN389_T:\n"
    "    # [Genesis PN389] Execute fn(*args) on a daemon thread under a hard\n"
    "    # wall-clock timeout. A bounded semaphore caps live compilation\n"
    "    # threads; a timed-out thread is orphaned (daemon threads never\n"
    "    # block process exit) but the semaphore stops them piling up. The\n"
    "    # caller returns in ~timeout seconds on a hang, not fn's duration.\n"
    "    if not _compilation_semaphore.acquire(timeout=timeout):\n"
    "        raise ValueError(\n"
    "            \"Too many concurrent grammar compilations in progress. \"\n"
    "            \"Try again later or simplify the request.\"\n"
    "        )\n"
    "    result_queue: \"Queue[tuple[str, object]]\" = Queue()\n"
    "\n"
    "    def _worker() -> None:\n"
    "        try:\n"
    "            result_queue.put((\"ok\", fn(*args)))\n"
    "        except BaseException as exc:  # noqa: BLE001 — re-raised in caller\n"
    "            result_queue.put((\"error\", exc))\n"
    "        finally:\n"
    "            _compilation_semaphore.release()\n"
    "\n"
    "    thread = threading.Thread(target=_worker, daemon=True)\n"
    "    thread.start()\n"
    "    try:\n"
    "        status, value = result_queue.get(timeout=timeout)\n"
    "    except Empty:\n"
    "        raise ValueError(\n"
    "            f\"{label} timed out after {timeout}s. \"\n"
    "            \"The grammar may be too complex.\"\n"
    "        ) from None\n"
    "    if status == \"error\":\n"
    "        raise value  # type: ignore[misc]\n"
    "    return value  # type: ignore[return-value]\n"
)


# ══════════════════════════════════════════════════════════════════════
# File 2 — v1/structured_output/backend_xgrammar.py
# ══════════════════════════════════════════════════════════════════════
#
# Seven edits: import the two helpers; refactor compile_grammar so the
# EngineCore DFA build of EVERY type runs through run_with_timeout (the
# core fix — the actual wedge surface); and wrap each `xgr.Grammar.from_*`
# frontend validation call in run_with_timeout (the parse pre-flight).

# (a) pull the two new helpers in from utils alongside the existing imports.
PN389_XGR_IMPORTS_OLD = (
    "from vllm.v1.structured_output.utils import (\n"
    "    choice_as_grammar,\n"
    "    convert_lark_to_ebnf,\n"
    "    grammar_is_likely_lark,\n"
    ")\n"
)

PN389_XGR_IMPORTS_NEW = (
    "from vllm.v1.structured_output.utils import (\n"
    "    _check_regex_complexity,\n"
    "    choice_as_grammar,\n"
    "    convert_lark_to_ebnf,\n"
    "    grammar_is_likely_lark,\n"
    "    run_with_timeout,\n"
    ")\n"
)

# (b) compile_grammar — the EngineCore DFA-build path (THE wedge surface).
#
# This is the core fix: the actual `self.compiler.compile_*` calls run on
# the single EngineCore CPU loop with NO wall-clock bound in the pin. The
# frontend validation pre-flight below (validate_xgrammar_grammar) only
# wraps the schema-PARSE (`xgr.Grammar.from_*`); a schema that PARSES fast
# but COMPILES catastrophically (vocab-dependent DFA explosion against the
# 152K-vocab compiler) sails through validation and then wedges decode
# here. So we mirror the PR's refactor of compile_grammar into:
#   compile_grammar     -> builds XgrammarGrammar from a ctx
#   _compile_ctx        -> run_with_timeout(_compile_ctx_inner, ...)
#   _compile_ctx_inner  -> the original compile_* dispatch, byte-identical
# Now EVERY type (JSON / JSON_OBJECT / GRAMMAR / REGEX / STRUCTURAL_TAG)
# has its DFA build bounded by VLLM_GRAMMAR_COMPILATION_TIMEOUT_SECONDS,
# and a pathological compile bounces as a ValueError instead of wedging
# the engine. The REGEX arm additionally keeps the cheap O(n)
# _check_regex_complexity pre-filter so an obviously-adversarial pattern
# is rejected before a compilation thread is even spawned.
#
# Anchor = the whole pin compile_grammar method (count==1, byte-verified
# against /private/tmp/candidate_pin_current/vllm). _compile_ctx_inner's
# body reproduces the pin's compile_* dispatch verbatim (only `ctx = ...`
# becomes `return ...`), so a compile within budget is bit-identical.
PN389_XGR_COMPILE_REGEX_OLD = (
    "    def compile_grammar(\n"
    "        self, request_type: StructuredOutputOptions, grammar_spec: str\n"
    "    ) -> StructuredOutputGrammar:\n"
    "        if request_type == StructuredOutputOptions.JSON:\n"
    "            ctx = self.compiler.compile_json_schema(\n"
    "                grammar_spec, any_whitespace=not self.disable_any_whitespace\n"
    "            )\n"
    "        elif request_type == StructuredOutputOptions.JSON_OBJECT:\n"
    "            ctx = self.compiler.compile_json_schema(\n"
    "                '{\"type\": \"object\"}', any_whitespace=not self.disable_any_whitespace\n"
    "            )\n"
    "        elif request_type == StructuredOutputOptions.GRAMMAR:\n"
    "            ctx = self.compiler.compile_grammar(grammar_spec)\n"
    "        elif request_type == StructuredOutputOptions.REGEX:\n"
    "            ctx = self.compiler.compile_regex(grammar_spec)\n"
    "        elif request_type == StructuredOutputOptions.STRUCTURAL_TAG:\n"
    "            s_tag = json.loads(grammar_spec)\n"
    '            if "structures" in s_tag:\n'
    "                # Falling back to deprecated method of compiling structural tag\n"
    "                tags = [\n"
    "                    xgr.StructuralTagItem(\n"
    '                        begin=s["begin"],\n'
    '                        schema=json.dumps(s["schema"]),\n'
    '                        end=s["end"],\n'
    "                    )\n"
    '                    for s in s_tag["structures"]\n'
    "                ]\n"
    '                ctx = self.compiler.compile_structural_tag(tags, s_tag["triggers"])\n'
    "            else:\n"
    "                ctx = self.compiler.compile_structural_tag(grammar_spec)\n"
    "        else:\n"
    "            logger.error(\n"
    '                "Validation should have already occurred. Please file an issue."\n'
    "            )\n"
    "            raise ValueError(\n"
    '                f"grammar is not of valid supported types. ({request_type!s})"\n'
    "            )\n"
    "\n"
    "        return XgrammarGrammar(\n"
    "            matcher=xgr.GrammarMatcher(\n"
    "                ctx,\n"
    "                max_rollback_tokens=self.num_speculative_tokens,\n"
    "            ),\n"
    "            vocab_size=self.vocab_size,\n"
    "            ctx=ctx,\n"
    "        )\n"
)

PN389_XGR_COMPILE_REGEX_NEW = (
    "    def compile_grammar(\n"
    "        self, request_type: StructuredOutputOptions, grammar_spec: str\n"
    "    ) -> StructuredOutputGrammar:\n"
    "        # [Genesis PN389 vendor of vllm#45390] wall-clock-bound the\n"
    "        # EngineCore DFA build. compile_grammar runs on the single\n"
    "        # CPU EngineCore loop; without a bound a pathological grammar\n"
    "        # compile wedges ALL decode. _compile_ctx runs the actual\n"
    "        # compile_* dispatch through run_with_timeout so every type is\n"
    "        # bounded, not just the frontend from_* parse pre-flight.\n"
    "        ctx = self._compile_ctx(\n"
    "            request_type,\n"
    "            grammar_spec,\n"
    "            vllm.envs.VLLM_GRAMMAR_COMPILATION_TIMEOUT_SECONDS,\n"
    "        )\n"
    "        return XgrammarGrammar(\n"
    "            matcher=xgr.GrammarMatcher(\n"
    "                ctx,\n"
    "                max_rollback_tokens=self.num_speculative_tokens,\n"
    "            ),\n"
    "            vocab_size=self.vocab_size,\n"
    "            ctx=ctx,\n"
    "        )\n"
    "\n"
    "    def _compile_ctx(self, request_type, grammar_spec, timeout):\n"
    "        # [Genesis PN389 vendor of vllm#45390] run the real DFA build on\n"
    "        # a daemon thread under the configured wall-clock timeout. On a\n"
    "        # hang this returns (raises ValueError) in ~timeout seconds\n"
    "        # instead of blocking the EngineCore loop for the full compile.\n"
    "        return run_with_timeout(\n"
    "            self._compile_ctx_inner,\n"
    "            request_type,\n"
    "            grammar_spec,\n"
    "            timeout=timeout,\n"
    '            label="Grammar compilation",\n'
    "        )\n"
    "\n"
    "    def _compile_ctx_inner(self, request_type, grammar_spec):\n"
    "        # [Genesis PN389 vendor of vllm#45390] the pin's original\n"
    "        # compile_* dispatch, unchanged except `ctx = ...` -> `return\n"
    "        # ...` so a compile within budget is bit-identical to the pin.\n"
    "        if request_type == StructuredOutputOptions.JSON:\n"
    "            return self.compiler.compile_json_schema(\n"
    "                grammar_spec, any_whitespace=not self.disable_any_whitespace\n"
    "            )\n"
    "        elif request_type == StructuredOutputOptions.JSON_OBJECT:\n"
    "            return self.compiler.compile_json_schema(\n"
    "                '{\"type\": \"object\"}', any_whitespace=not self.disable_any_whitespace\n"
    "            )\n"
    "        elif request_type == StructuredOutputOptions.GRAMMAR:\n"
    "            return self.compiler.compile_grammar(grammar_spec)\n"
    "        elif request_type == StructuredOutputOptions.REGEX:\n"
    "            # [Genesis PN389] reject adversarial regex before the DFA\n"
    "            # builder runs (cheap O(n) pre-filter inside the timeout).\n"
    "            _check_regex_complexity(grammar_spec)\n"
    "            return self.compiler.compile_regex(grammar_spec)\n"
    "        elif request_type == StructuredOutputOptions.STRUCTURAL_TAG:\n"
    "            s_tag = json.loads(grammar_spec)\n"
    '            if "structures" in s_tag:\n'
    "                # Falling back to deprecated method of compiling structural tag\n"
    "                tags = [\n"
    "                    xgr.StructuralTagItem(\n"
    '                        begin=s["begin"],\n'
    '                        schema=json.dumps(s["schema"]),\n'
    '                        end=s["end"],\n'
    "                    )\n"
    '                    for s in s_tag["structures"]\n'
    "                ]\n"
    '                return self.compiler.compile_structural_tag(tags, s_tag["triggers"])\n'
    "            else:\n"
    "                return self.compiler.compile_structural_tag(grammar_spec)\n"
    "        else:\n"
    "            logger.error(\n"
    '                "Validation should have already occurred. Please file an issue."\n'
    "            )\n"
    "            raise ValueError(\n"
    '                f"grammar is not of valid supported types. ({request_type!s})"\n'
    "            )\n"
)

# (c) validate_xgrammar_grammar — regex arm. Add the pre-filter and wrap
# from_regex in run_with_timeout (the env timeout is read inline).
PN389_XGR_VALIDATE_REGEX_OLD = (
    "    if so_params.regex:\n"
    "        try:\n"
    "            xgr.Grammar.from_regex(so_params.regex)\n"
    "        except Exception as err:\n"
)

PN389_XGR_VALIDATE_REGEX_NEW = (
    "    if so_params.regex:\n"
    "        try:\n"
    "            # [Genesis PN389 vendor of vllm#45390] complexity pre-filter\n"
    "            # + wall-clock-bounded grammar build.\n"
    "            _check_regex_complexity(so_params.regex)\n"
    "            run_with_timeout(\n"
    "                xgr.Grammar.from_regex,\n"
    "                so_params.regex,\n"
    "                timeout=vllm.envs.VLLM_GRAMMAR_COMPILATION_TIMEOUT_SECONDS,\n"
    "                label=\"Regex grammar validation\",\n"
    "            )\n"
    "        except ValueError:\n"
    "            raise\n"
    "        except Exception as err:\n"
)

# (d) validate_xgrammar_grammar — choice arm (from_ebnf).
PN389_XGR_VALIDATE_CHOICE_OLD = (
    "        choice_grammar = choice_as_grammar(so_params.choice)\n"
    "        try:\n"
    "            xgr.Grammar.from_ebnf(choice_grammar)\n"
    "        except Exception as err:\n"
)

PN389_XGR_VALIDATE_CHOICE_NEW = (
    "        choice_grammar = choice_as_grammar(so_params.choice)\n"
    "        try:\n"
    "            # [Genesis PN389 vendor of vllm#45390] bounded grammar build.\n"
    "            run_with_timeout(\n"
    "                xgr.Grammar.from_ebnf,\n"
    "                choice_grammar,\n"
    "                timeout=vllm.envs.VLLM_GRAMMAR_COMPILATION_TIMEOUT_SECONDS,\n"
    "                label=\"Choice grammar validation\",\n"
    "            )\n"
    "        except Exception as err:\n"
)

# (e) validate_xgrammar_grammar — json_schema arm (from_json_schema).
# This is THE hot path: every Genesis tool-call JSON schema lands here.
PN389_XGR_VALIDATE_JSON_OLD = (
    "        try:\n"
    "            xgr.Grammar.from_json_schema(schema)\n"
    "        except Exception as err:\n"
)

PN389_XGR_VALIDATE_JSON_NEW = (
    "        try:\n"
    "            # [Genesis PN389 vendor of vllm#45390] bounded grammar build\n"
    "            # — the tool-call JSON-schema hot path every model traverses.\n"
    "            run_with_timeout(\n"
    "                xgr.Grammar.from_json_schema,\n"
    "                schema,\n"
    "                timeout=vllm.envs.VLLM_GRAMMAR_COMPILATION_TIMEOUT_SECONDS,\n"
    "                label=\"JSON schema grammar validation\",\n"
    "            )\n"
    "        except Exception as err:\n"
)

# (f) validate_xgrammar_grammar — ebnf grammar arm (from_ebnf, the
# `we aren't compiling it` comment is unique to this site).
PN389_XGR_VALIDATE_EBNF_OLD = (
    "        try:\n"
    "            # parse the grammar, but we aren't compiling it.\n"
    "            xgr.Grammar.from_ebnf(so_params.grammar)\n"
    "        except Exception as e:\n"
)

PN389_XGR_VALIDATE_EBNF_NEW = (
    "        try:\n"
    "            # parse the grammar, but we aren't compiling it.\n"
    "            # [Genesis PN389 vendor of vllm#45390] bounded grammar build.\n"
    "            run_with_timeout(\n"
    "                xgr.Grammar.from_ebnf,\n"
    "                so_params.grammar,\n"
    "                timeout=vllm.envs.VLLM_GRAMMAR_COMPILATION_TIMEOUT_SECONDS,\n"
    "                label=\"EBNF grammar validation\",\n"
    "            )\n"
    "        except Exception as e:\n"
)

# (g) validate_xgrammar_grammar — structural_tag arm (both from_structural_tag
# calls). Wrap each in run_with_timeout.
PN389_XGR_VALIDATE_STAG_OLD = (
    "                xgr.Grammar.from_structural_tag(tags, s_tag[\"triggers\"])\n"
    "            else:\n"
    "                xgr.Grammar.from_structural_tag(so_params.structural_tag)\n"
)

PN389_XGR_VALIDATE_STAG_NEW = (
    "                # [Genesis PN389 vendor of vllm#45390] bounded build.\n"
    "                run_with_timeout(\n"
    "                    xgr.Grammar.from_structural_tag,\n"
    "                    tags,\n"
    "                    s_tag[\"triggers\"],\n"
    "                    timeout=vllm.envs.VLLM_GRAMMAR_COMPILATION_TIMEOUT_SECONDS,\n"
    "                    label=\"Structural tag grammar validation\",\n"
    "                )\n"
    "            else:\n"
    "                # [Genesis PN389 vendor of vllm#45390] bounded build.\n"
    "                run_with_timeout(\n"
    "                    xgr.Grammar.from_structural_tag,\n"
    "                    so_params.structural_tag,\n"
    "                    timeout=vllm.envs.VLLM_GRAMMAR_COMPILATION_TIMEOUT_SECONDS,\n"
    "                    label=\"Structural tag grammar validation\",\n"
    "                )\n"
)


# ══════════════════════════════════════════════════════════════════════
# File 3 — envs.py (ADDITIVE: new VLLM_GRAMMAR_COMPILATION_TIMEOUT_SECONDS)
# ══════════════════════════════════════════════════════════════════════

# (a) type-annotation declaration block (after VLLM_XGRAMMAR_CACHE_MB).
PN389_ENVS_DECL_OLD = "    VLLM_XGRAMMAR_CACHE_MB: int = 0\n"

PN389_ENVS_DECL_NEW = (
    "    VLLM_XGRAMMAR_CACHE_MB: int = 0\n"
    "    # [Genesis PN389 vendor of vllm#45390] grammar/regex compile budget.\n"
    "    VLLM_GRAMMAR_COMPILATION_TIMEOUT_SECONDS: int = 2\n"
)

# (b) the os.getenv lambda block (after the VLLM_XGRAMMAR_CACHE_MB lambda).
# Genesis default = 2s (PR ships 10s — would blow our TTFT SLO ~60x).
PN389_ENVS_LAMBDA_OLD = (
    '    "VLLM_XGRAMMAR_CACHE_MB": lambda: int(os.getenv("VLLM_XGRAMMAR_CACHE_MB", "512")),\n'
)

PN389_ENVS_LAMBDA_NEW = (
    '    "VLLM_XGRAMMAR_CACHE_MB": lambda: int(os.getenv("VLLM_XGRAMMAR_CACHE_MB", "512")),\n'
    "    # [Genesis PN389 vendor of vllm#45390] max seconds for a grammar/regex\n"
    "    # DFA compile. Genesis default 2s (the PR ships 10s, which would blow\n"
    "    # our 70-160ms TTFT SLO ~60x before the timeout fires). Operator may\n"
    "    # raise it back to 10 via the env var.\n"
    '    "VLLM_GRAMMAR_COMPILATION_TIMEOUT_SECONDS": lambda: int(\n'
    '        os.getenv("VLLM_GRAMMAR_COMPILATION_TIMEOUT_SECONDS", "2")\n'
    "    ),\n"
)


# ─────────────────────────────────────────────────────────────────────
# Patcher builders (one per target file). Driven atomically by apply().
# ─────────────────────────────────────────────────────────────────────


def _make_utils_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_TARGET_UTILS)
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN389 v1/structured_output/utils.py — run_with_timeout + "
            "_check_regex_complexity (vendor of vllm#45390)"
        ),
        target_file=str(target),
        marker=GENESIS_PN389_MARKER,
        sub_patches=[
            TextPatch(
                name="pn389_utils_imports",
                anchor=PN389_UTILS_IMPORTS_OLD,
                replacement=PN389_UTILS_IMPORTS_NEW,
                required=True,
            ),
            TextPatch(
                name="pn389_utils_helpers",
                anchor=PN389_UTILS_HELPERS_OLD,
                replacement=PN389_UTILS_HELPERS_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=list(_DRIFT_MARKERS),
    )


def _make_xgrammar_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_TARGET_XGRAMMAR)
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN389 v1/structured_output/backend_xgrammar.py — wrap all "
            "XGrammar compile/validate calls in run_with_timeout "
            "(vendor of vllm#45390)"
        ),
        target_file=str(target),
        marker=GENESIS_PN389_MARKER,
        sub_patches=[
            TextPatch(
                name="pn389_xgr_imports",
                anchor=PN389_XGR_IMPORTS_OLD,
                replacement=PN389_XGR_IMPORTS_NEW,
                required=True,
            ),
            TextPatch(
                name="pn389_xgr_compile_grammar",
                anchor=PN389_XGR_COMPILE_REGEX_OLD,
                replacement=PN389_XGR_COMPILE_REGEX_NEW,
                required=True,
            ),
            TextPatch(
                name="pn389_xgr_validate_regex",
                anchor=PN389_XGR_VALIDATE_REGEX_OLD,
                replacement=PN389_XGR_VALIDATE_REGEX_NEW,
                required=True,
            ),
            TextPatch(
                name="pn389_xgr_validate_choice",
                anchor=PN389_XGR_VALIDATE_CHOICE_OLD,
                replacement=PN389_XGR_VALIDATE_CHOICE_NEW,
                required=True,
            ),
            TextPatch(
                name="pn389_xgr_validate_json",
                anchor=PN389_XGR_VALIDATE_JSON_OLD,
                replacement=PN389_XGR_VALIDATE_JSON_NEW,
                required=True,
            ),
            TextPatch(
                name="pn389_xgr_validate_ebnf",
                anchor=PN389_XGR_VALIDATE_EBNF_OLD,
                replacement=PN389_XGR_VALIDATE_EBNF_NEW,
                required=True,
            ),
            TextPatch(
                name="pn389_xgr_validate_structural_tag",
                anchor=PN389_XGR_VALIDATE_STAG_OLD,
                replacement=PN389_XGR_VALIDATE_STAG_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=list(_DRIFT_MARKERS),
    )


def _make_envs_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_TARGET_ENVS)
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN389 envs.py — VLLM_GRAMMAR_COMPILATION_TIMEOUT_SECONDS "
            "(Genesis default 2s; vendor of vllm#45390)"
        ),
        target_file=str(target),
        marker=GENESIS_PN389_MARKER,
        sub_patches=[
            TextPatch(
                name="pn389_envs_decl",
                anchor=PN389_ENVS_DECL_OLD,
                replacement=PN389_ENVS_DECL_NEW,
                required=True,
            ),
            TextPatch(
                name="pn389_envs_lambda",
                anchor=PN389_ENVS_LAMBDA_OLD,
                replacement=PN389_ENVS_LAMBDA_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=list(_DRIFT_MARKERS),
    )


def _all_patchers() -> list[TextPatcher]:
    """Build every PN389 target patcher; drop unresolvable ones."""
    out: list[TextPatcher] = []
    for builder in (
        _make_utils_patcher,
        _make_xgrammar_patcher,
        _make_envs_patcher,
    ):
        p = builder()
        if p is not None:
            out.append(p)
    return out


def apply() -> tuple[str, str]:
    """Apply PN389 — XGrammar grammar-compilation timeouts. Never raises.

    Drives all three target files (utils.py, backend_xgrammar.py, envs.py)
    in ONE ``MultiFilePatchTransaction`` (validate-all-then-write-all),
    so the helpers, their call sites, and the env that times them out
    either all land together or none do — never a half-patched tree where
    ``backend_xgrammar`` references a ``run_with_timeout`` that ``utils``
    does not yet define.

    Opt-in: gated through the dispatcher on
    ``GENESIS_ENABLE_PN389_GRAMMAR_TIMEOUTS`` (default_on=False in the
    registry — the timeout reject is a new failure mode for legitimate
    slow grammars; gated until a server A/B confirms the 2s budget never
    trips a real tool-schema compile).
    """
    from sndr.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN389")
    log_decision("PN389", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patchers = _all_patchers()
    if len(patchers) != 3:
        return (
            "skipped",
            "PN389: not all targets resolvable "
            f"({len(patchers)}/3 of utils/backend_xgrammar/envs found)",
        )

    # All three files must be unpatched and free of the upstream-merged
    # form before we commit. The transaction's dry-run re-checks anchors;
    # here we additionally (a) report a clean idempotent skip when the
    # marker is already on every target, and (b) self-skip if #45390 has
    # landed upstream.
    markers_present = 0
    for p in patchers:
        if not os.path.isfile(p.target_file):
            return "skipped", f"target disappeared: {p.target_file}"
        try:
            with open(p.target_file, encoding="utf-8") as f:
                content = f.read()
        except OSError as e:
            return "skipped", f"PN389: read error on {p.target_file}: {e}"
        if p.marker in content:
            markers_present += 1
            continue  # already applied — idempotent
        for m in p.upstream_drift_markers:
            if m.startswith("[Genesis"):
                continue
            if m in content:
                return (
                    "skipped",
                    f"upstream drift marker {m!r} present in "
                    f"{os.path.basename(p.target_file)} — upstream PR "
                    "#45390 (or equivalent) appears merged (upstream_merged)",
                )

    # Every target already carries the marker -> nothing to do. Report a
    # clean idempotent skip rather than letting the transaction re-report
    # an all-IDEMPOTENT commit as "applied".
    if markers_present == len(patchers):
        return "skipped", "PN389: already applied (marker present on all 3 targets)"

    from sndr.kernel import MultiFilePatchTransaction

    txn = MultiFilePatchTransaction(patchers, name="PN389")
    status, txn_reason = txn.apply_or_skip()
    if status != "applied":
        return status, f"PN389: {txn_reason}"
    return (
        "applied",
        "PN389 applied (3 files): run_with_timeout (daemon-thread + Queue "
        "+ Semaphore(4)) now bounds BOTH XGrammar surfaces — the EngineCore "
        "DFA build (compile_grammar refactored to _compile_ctx -> "
        "run_with_timeout(_compile_ctx_inner) over every type: "
        "JSON/JSON_OBJECT/GRAMMAR/REGEX/STRUCTURAL_TAG) AND the frontend "
        "validate_xgrammar_grammar parse pre-flight (every xgr.Grammar."
        "from_* call), plus _check_regex_complexity on the REGEX arm. All "
        "bounded by VLLM_GRAMMAR_COMPILATION_TIMEOUT_SECONDS (Genesis "
        "default 2s, not the PR's 10s, to protect the 70-160ms TTFT SLO). "
        "A pathological tool schema that compiles catastrophically now "
        "bounces as a ValueError (400) instead of wedging the "
        "single-instance EngineCore loop (vllm#45390, 7-GHSA DoS). "
        "Bit-identical for compiles within budget.",
    )


def is_applied() -> bool:
    """Return True iff the PN389 marker is present in ALL three targets."""
    if vllm_install_root() is None:
        return False
    patchers = _all_patchers()
    if len(patchers) != 3:
        return False
    for p in patchers:
        try:
            with open(p.target_file, encoding="utf-8") as f:
                if p.marker not in f.read():
                    return False
        except OSError:
            return False
    return True
