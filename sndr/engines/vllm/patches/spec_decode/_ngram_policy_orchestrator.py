# SPDX-License-Identifier: Apache-2.0
"""NGRAM speculative-decoding policy orchestrator.

Phase 6 P3.2 (master plan §3.2) — Genesis carries five patches in the
ngram speculative-decoding stack. Each is independent and composes via
distinct insertion points on the `NgramProposer` pipeline; this module
documents the unified composition contract + provides a runtime audit
helper without changing any patch's apply() semantics (byte-equivalent
to v11.1.0).

The five patches
----------------

  P70   — auto-strict-ngram (config-time K boundary):
          Hooks `SpeculativeConfig.__post_init__` to enforce
          `prompt_lookup_min >= 8` when the env flag is set.
          PURPOSE: eliminate spurious tool-call acceptance from short
          n-grams (closes vllm#40875).
          INSERTION POINT: SpeculativeConfig boot path (one-time).

  P77   — adaptive K controller (runtime K modulation):
          Monkey-patches `NgramProposer.propose()` with an
          AdaptiveNgramController state machine. EMA + hysteresis over
          recent acceptance, chooses K ∈ {0, 1, 3, 5} per batch.
          K=0 short-circuits the entire proposer call.
          PURPOSE: adapt K to acceptance feedback per workload window.
          INSERTION POINT: NgramProposer.propose() wrap.

  P86   — batch_propose O(N+K) linear (pure perf):
          Text-patches `v1/spec_decode/ngram_proposer.py:73-113` to
          replace the O(N*K) membership scan with direct-fill init +
          per-valid-index iteration. No policy logic — backport of
          vllm#40876.
          INSERTION POINT: batch_propose hot path.

  PN72  — frequency-based ngram post-filter (quality gate):
          Wraps `NgramProposer.propose()`; filters drafts by first-token
          frequency in a recent window (default 1024 tokens, MIN=4
          observations). Mirrors llama.cpp's `draft_min_sample_size`.
          INSERTION POINT: NgramProposer.propose() post-wrap.

  PN90  — probabilistic draft rejection (verifier enrichment):
          Backport of vllm#40269 (MERGED upstream 2026-05-14). Adds
          softmax over drafter logits → `_pn90_draft_probs` buffer →
          verifier reads instead of literal None. Switches verifier
          acceptance rule from argmax to `min(1, target/draft)`.
          INSERTION POINT: llm_base_proposer.py + gpu_model_runner.py
          rejection_sampler call site.

Composition matrix
------------------

| Patch | Default | Insertion point | Conflicts with | Notes |
|---|---|---|---|---|
| P70   | OFF | Config-time     | (none in family) | strictly idempotent boot hook |
| P77   | OFF | Runtime wrap    | (composes with P70 + PN72) | K=0 short-circuits PN72 |
| P86   | OFF | Hot-path patch  | (none) | pure perf, no policy |
| PN72  | OFF | Runtime wrap    | (composes with P77) | runs AFTER P77 if both ON |
| PN90  | OFF | Verifier wrap   | (independent) | upstream merged → self-skip |

All 5 are operational orthogonality (different insertion points). They
do NOT need a runtime orchestrator object — vLLM's existing dispatch
chain handles ordering via Python wrapping (later-registered wrap is
outermost; PN72 wrap fires AFTER P77 wrap inside the proposer).

Why this module exists
----------------------

1. **Composition documentation** — without this file the contract is
   spread across five docstrings. Centralizing here surfaces the
   ordering invariant (PN72 outermost, P77 inner, P86 hot-path,
   PN90 verifier-side, P70 boot-only) in ONE place.

2. **Drift surveillance** — provides `audit_ngram_stack_state()` to
   report which patches are reachable on the current pin. Useful when
   upstream refactors move the NgramProposer class or rename methods.

3. **Operator visibility** — `sndr patches show ngram_policy` calls
   into this module to print the unified composition + which patches
   are currently env-enabled.

4. **NOT a refactor of the patch LOGIC.** Each patch's apply() function
   is unchanged. This module is documentation + audit ONLY. A future
   v12.x or v13.x might consolidate P70 + P77 + PN72 into a single
   pluggable DecisionStrategy class; that's deferred until the
   ordering semantics are bench-verified across all workloads (multi-conc
   especially, where PN72 frequency window may interact with K=0
   short-circuit).

Future consolidation candidate (v12.x+)
---------------------------------------

A pluggable `NgramDecisionStrategy` base class could host:

    class NgramDecisionStrategy(Protocol):
        def adjust_k(self, ctx: ProposalContext) -> int: ...
        def should_short_circuit(self, ctx: ProposalContext) -> bool: ...
        def filter_drafts(self, drafts: list[int], ctx) -> list[int]: ...

With concrete implementations:
  - AutoStrictKStrategy (P70-derived)
  - AdaptiveKStrategy (P77-derived; encapsulates EMA + hysteresis)
  - FrequencyGateStrategy (PN72-derived)
  - DefaultStrategy (no-op, current vLLM behaviour)

Operators would pick via env: `GENESIS_NGRAM_STRATEGY=auto_strict+adaptive_k`
and the orchestrator would compose them. Estimated saving: ~600-700
LOC across the three patches, much of it duplicate "wrap propose +
check env + restore on error" boilerplate.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa
Status: v11.2.0+ P3.2 documentation + audit (no behavior change)
"""
from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger("genesis.spec_decode.ngram_policy")


__all__ = [
    "NGRAM_POLICY_PATCH_IDS",
    "audit_ngram_stack_state",
    "describe_policy",
    "verify_patch_composition",
]


NGRAM_POLICY_PATCH_IDS = ("P70", "P77", "P86", "PN72", "PN90")
"""All patches in the unified NGRAM speculative-decode policy stack."""


_POLICY_SUMMARY = """\
NGRAM speculative-decoding policy (Phase 6 P3.2)

Genesis carries 5 patches that compose the ngram speculative-decode
pipeline. They are operationally orthogonal — different insertion points
on the NgramProposer call graph. The "policy" each implements:

  P70   — config-time strict K bound (prompt_lookup_min >= 8)
  P77   — runtime adaptive K modulation (EMA + hysteresis state machine)
  P86   — pure perf backport (O(N*K) → O(N+K) batch fill)
  PN72  — post-filter by first-token frequency window
  PN90  — probabilistic verifier (target/draft prob ratio)

All five are default_on=False. Enable per workload. The composition is
already orthogonal — they don't need a runtime orchestrator object on
v11.2.0+ Genesis. A future v12.x+ may consolidate P70+P77+PN72 into a
pluggable DecisionStrategy class; until then this module provides the
unified documentation + audit surface only.

To diagnose the live stack on a running container:
  python -c 'from vllm.sndr_core.integrations.spec_decode._ngram_policy_orchestrator import audit_ngram_stack_state; import json; print(json.dumps(audit_ngram_stack_state(), indent=2, default=str))'
"""


def describe_policy() -> str:
    """Human-readable summary of the policy + composition contract."""
    return _POLICY_SUMMARY


def verify_patch_composition() -> dict:
    """Report composition state across the 5 ngram patches.

    Returns dict shape:
      {
        "composable": bool,
        "patches": {
          "P70": {"lifecycle": str, "default_on": bool,
                  "env_enabled": bool, "env_flag": str},
          ...
        },
        "conflicts": [str],
      }
    """
    try:
        from vllm.sndr_core.dispatcher.registry import PATCH_REGISTRY
    except ImportError:
        return {
            "composable": True,
            "patches": {
                pid: {"registry_present": False}
                for pid in NGRAM_POLICY_PATCH_IDS
            },
            "conflicts": [],
            "warning": "PATCH_REGISTRY not importable; audit limited",
        }

    patches: dict = {}
    conflicts: list[str] = []
    for pid in NGRAM_POLICY_PATCH_IDS:
        entry = PATCH_REGISTRY.get(pid)
        if entry is None:
            patches[pid] = {"registry_present": False}
            conflicts.append(f"{pid} not in PATCH_REGISTRY")
            continue
        env_flag = entry.get("env_flag")
        env_enabled = (
            env_flag is not None
            and os.environ.get(env_flag, "").strip() in ("1", "true", "True")
        )
        patches[pid] = {
            "registry_present": True,
            "lifecycle": entry.get("lifecycle"),
            "default_on": entry.get("default_on", False),
            "env_flag": env_flag,
            "env_enabled": env_enabled,
            "family": entry.get("family"),
            "upstream_pr": entry.get("upstream_pr"),
        }
        cw = entry.get("conflicts_with") or []
        for other in NGRAM_POLICY_PATCH_IDS:
            if other != pid and other in cw:
                conflicts.append(f"{pid}.conflicts_with contains {other}")

    return {
        "composable": not conflicts,
        "patches": patches,
        "conflicts": conflicts,
    }


def audit_ngram_stack_state(verbose: bool = False) -> dict:
    """Full runtime audit — composition + reachability + active K hint.

    `reachability` checks whether the upstream NgramProposer class is
    importable at the current pin, which is the prerequisite for ALL
    five patches.

    Returns dict shape:
      {
        "policy_summary": str (when verbose),
        "composition": {...verify_patch_composition output...},
        "reachability": {
          "ngram_proposer_class": bool,
          "speculative_config_class": bool,
          "error": str,
        },
        "summary": {
          "composable": bool,
          "any_patch_env_enabled": bool,
          "all_patches_in_registry": bool,
        },
      }
    """
    composition = verify_patch_composition()

    reachability: dict = {
        "ngram_proposer_class": False,
        "speculative_config_class": False,
    }
    try:
        # The two classes touched by the stack:
        from vllm.v1.spec_decode.ngram_proposer import (  # noqa: F401
            NgramProposer,
        )
        reachability["ngram_proposer_class"] = True
    except Exception as e:
        reachability["ngram_proposer_error"] = str(e)
    try:
        from vllm.config import SpeculativeConfig  # noqa: F401
        reachability["speculative_config_class"] = True
    except Exception as e:
        reachability["speculative_config_error"] = str(e)

    any_enabled = any(
        p.get("env_enabled", False)
        for p in composition["patches"].values()
    )

    summary = {
        "composable": composition.get("composable", False),
        "any_patch_env_enabled": any_enabled,
        "all_patches_in_registry": all(
            p.get("registry_present", False)
            for p in composition["patches"].values()
        ),
    }

    if verbose:
        log.info("NGRAM policy audit summary: %s", summary)

    return {
        "policy_summary": _POLICY_SUMMARY if verbose else None,
        "composition": composition,
        "reachability": reachability,
        "summary": summary,
    }


def main_cli() -> int:
    """Human-readable audit — `sndr explain ngram_policy` etc."""
    print(_POLICY_SUMMARY)
    print()
    result = audit_ngram_stack_state(verbose=False)
    print("Composition:")
    for pid, p in result["composition"]["patches"].items():
        print(
            f"  {pid}: lifecycle={p.get('lifecycle')} "
            f"default_on={p.get('default_on')} "
            f"env_enabled={p.get('env_enabled')} "
            f"env_flag={p.get('env_flag')}"
        )
    if result["composition"]["conflicts"]:
        print(f"Conflicts: {result['composition']['conflicts']}")
    print()
    print("Reachability:")
    for k, v in result["reachability"].items():
        print(f"  {k}: {v}")
    print()
    print(f"Summary: {result['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
