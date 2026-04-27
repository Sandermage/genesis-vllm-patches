# SPDX-License-Identifier: Apache-2.0
"""Genesis Dispatcher v2 — unified patch decision matrix + diagnostics.

Builds on top of `model_detect.py` and `config_detect.py` to provide:

  1. **Per-patch should_apply()** — single-line gate decision for each patch.
     Wraps `config_detect.recommend()` + `model_detect` checks + env-flag
     overrides into one consistent API.

  2. **Apply matrix dump** — diagnostic command-line entry-point that prints
     the full per-patch decision table for the current vllm config. Useful
     for operators to see WHY a patch was applied or skipped without grep-ing
     boot logs.

  3. **Startup logging** — single condensed line at boot summarizing which
     patches got applied, skipped (with reason), or failed. Replaces the
     scattered per-patch INFO lines that flood the boot log.

Usage at runtime
----------------
From a patch wiring (`patch_NN_*.py::apply()`):

    from vllm._genesis.dispatcher import should_apply, log_decision

    decision, reason = should_apply("P60")
    if not decision:
        log_decision("P60", decision, reason)
        return "skipped", reason
    # ... do the patching ...

From CLI / diagnostic:

    python3 -m vllm._genesis.dispatcher

Output: full apply matrix as ASCII table.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("genesis.dispatcher")


# ─── Patch metadata registry ───────────────────────────────────────────────
# Each patch declares what it touches + which env flag enables/disables it.
# This is the SINGLE source of truth for patch-to-feature mapping.

PATCH_REGISTRY: dict[str, dict[str, Any]] = {
    "P56": {
        "title": "TQ spec-decode safe-path guard (deprecated — superseded by P65)",
        "env_flag": "GENESIS_ENABLE_P56_SPEC_DECODE_GUARD",
        "default_on": False,
        "deprecated": True,
        "category": "spec_decode",
        "credit": "noonghunna (#40807, #40831)",
        "upstream_pr": None,
        "deprecation_note": (
            "P56 was a routing-layer workaround forcing spec-decode through a "
            "'safe' path when CG-aware buffers misaligned. Real fix is P65 "
            "(TurboQuant CG downgrade) which addresses the root cause in the "
            "full-attention path under FULL cudagraph capture. Kept opt-in for "
            "configurations where P65 is intentionally disabled and a routing "
            "guard is still desired (no such config verified in production)."
        ),
    },
    "P57": {
        "title": "TQ spec-decode capture-safe buffers (deprecated — research artifact)",
        "env_flag": "GENESIS_ENABLE_P57_SPEC_DECODE_CAPTURE_SAFE",
        "default_on": False,
        "deprecated": True,
        "category": "spec_decode",
        "credit": "noonghunna (#40831), gdn_attn.py reference",
        "upstream_pr": None,
        "deprecation_note": (
            "P57 v2 enlarges per-layer capture buffers from ~530 KiB to ~2.1 MiB "
            "(see wiring/patch_57 docstring for derivation), which is the "
            "MINIMAL sufficient fix for the original symptom but pushes total "
            "spec-decode buffer memory from ~270 MiB to ~1080 MiB across 32 "
            "layers — unacceptable on consumer Ampere with 24 GB VRAM. P65 "
            "(CG downgrade) achieves the same correctness without the memory "
            "blow-up. Kept opt-in as a research artifact / reference for "
            "future hardware with larger VRAM budgets."
        ),
    },
    "P58": {
        "title": "Async-scheduler -1 placeholder fix",
        "env_flag": "GENESIS_ENABLE_P58_ASYNC_PLACEHOLDER_FIX",
        "default_on": False,
        "category": "spec_decode",
        "credit": "z1ying (vllm#40768)",
        "upstream_pr": 40768,
    },
    "P59": {
        "title": "Qwen3 reasoning embedded tool_call recovery",
        "env_flag": "GENESIS_ENABLE_P59_QWEN3_TOOL_RECOVERY",
        "default_on": False,
        "category": "structured_output",
        "credit": "ZenoAFfectionate (vllm#39055)",
        "upstream_pr": 39055,
    },
    "P60": {
        "title": "GDN+ngram state recovery (Phase 1: SSM pre-copy)",
        "env_flag": "GENESIS_ENABLE_P60_GDN_NGRAM_FIX",
        "default_on": False,
        "category": "spec_decode",
        "credit": "tdoublep (vllm#40738), bhaktatejas922 (#39273)",
        "upstream_pr": 40738,
    },
    "P60b": {
        "title": "GDN+ngram Triton kernel offset (Phase 2)",
        "env_flag": "GENESIS_ENABLE_P60B_TRITON_KERNEL",
        "default_on": False,
        "category": "spec_decode",
        "credit": "tdoublep (vllm#40738)",
        "upstream_pr": 40738,
    },
    "P61": {
        "title": "Qwen3 multi-tool first-occurrence",
        "env_flag": "GENESIS_ENABLE_P61_QWEN3_MULTI_TOOL",
        "default_on": False,
        "category": "structured_output",
        "credit": "ExtReMLapin (vllm#40783)",
        "upstream_pr": 40783,
    },
    "P62": {
        "title": "Structured-output spec-decode reasoning-end timing fix",
        "env_flag": "GENESIS_ENABLE_P62_STRUCT_OUT_SPEC_TIMING",
        "default_on": False,
        "category": "structured_output",
        "credit": "sfbemerk (vllm#36138), cicirori (vllm#34650)",
        "upstream_pr": 36138,
    },
    "P61b": {
        "title": "Qwen3 streaming partial-tag overlap guard",
        "env_flag": "GENESIS_ENABLE_P61B_STREAMING_OVERLAP",
        "default_on": False,
        "category": "structured_output",
        "credit": "ExtReMLapin (vllm#40783)",
        "upstream_pr": 40783,
    },
    "P63": {
        "title": "MTP/Eagle drafter GDN state recovery (deprecated — wrong layer)",
        "env_flag": "GENESIS_ENABLE_P63_MTP_GDN_STATE_RECOVERY",
        "default_on": False,
        "deprecated": True,
        "category": "spec_decode",
        "credit": "Genesis-original (hypothesis disproven 2026-04-25)",
        "upstream_pr": None,
        "deprecation_note": (
            "P63 hypothesis was wrong: MTP module uses layer_type='full_attention' "
            "(Qwen3NextAttention), NOT GDN. GDNAttentionMetadataBuilder.build_for_drafting "
            "is never called for MTP drafter. Real fix is P65 (TurboQuant CG downgrade) — "
            "the bug is in the full_attention path under FULL cudagraph capture, not GDN. "
            "P63 may still be relevant for eagle/draft_model methods that use a separate "
            "drafter model with hybrid layers, but no such configuration is verified yet."
        ),
    },
    "P64": {
        "title": "qwen3coder MTP streaming early-return fix",
        "env_flag": "GENESIS_ENABLE_P64_QWEN3CODER_MTP_STREAMING",
        "default_on": False,
        "category": "structured_output",
        "credit": "kotori-yan (vllm#39598)",
        "upstream_pr": 39598,
    },
    "P65": {
        "title": "TurboQuant spec-decode cudagraph downgrade",
        "env_flag": "GENESIS_ENABLE_P65_TURBOQUANT_SPEC_CG_DOWNGRADE",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Genesis-original (root cause for noonghunna #40880)",
        "upstream_pr": None,
    },
    "P66": {
        "title": "cudagraph_capture_sizes spec-decode divisibility filter",
        "env_flag": "GENESIS_ENABLE_P66_CUDAGRAPH_SIZE_FILTER",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Genesis-original (mirrors fhl2000 vllm#23679 closed)",
        "upstream_pr": 23679,
    },
    "P68": {
        "title": "Auto force tool_choice=required for long-context tool calls",
        "env_flag": "GENESIS_ENABLE_P68_AUTO_FORCE_TOOL",
        "default_on": False,
        "category": "structured_output",
        "credit": "Genesis-original (long-ctx tool adherence mitigation)",
        "upstream_pr": None,
    },
    "P69": {
        "title": "Long-context tool-format reminder injection",
        "env_flag": "GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER",
        "default_on": False,
        "category": "structured_output",
        "credit": "Genesis-original (long-ctx tool adherence mitigation)",
        "upstream_pr": None,
    },
    "P70": {
        "title": "Auto-strict-ngram (force prompt_lookup_min>=8)",
        "env_flag": "GENESIS_ENABLE_P70_AUTO_STRICT_NGRAM",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Genesis-original (vllm#40875 enforcement)",
        "upstream_pr": None,
    },
    "P67": {
        "title": "TurboQuant multi-query kernel for spec-decode K+1",
        "env_flag": "GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Genesis-original (proper fix for noonghunna #40880; replaces P65 workaround)",
        "upstream_pr": None,
    },
    "P72": {
        "title": "profile_run M cap (unblocks --max-num-batched-tokens>4096 on MoE)",
        "env_flag": "GENESIS_ENABLE_P72_PROFILE_RUN_CAP",
        "default_on": False,
        "category": "compile_safety",
        "credit": "Genesis-original (Dynamo fake-tensor mismatch workaround for moe_align_block_size symbolic shape)",
        "upstream_pr": None,
    },
    "P71": {
        "title": "Block-verify rejection sampler (Sun 2024 ICLR)",
        "env_flag": "GENESIS_ENABLE_P71_BLOCK_VERIFY",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Backport of vllm#40819 (Z. Golpayegani draft) + Sun et al. arXiv 2403.10444 + 2 critical fixes from gemini-code-assist review (shared u per request, denom==0 → 1.0)",
        "upstream_pr": 40819,
    },
    "P74": {
        "title": "Auto chunk-clamp via long_prefill_token_threshold (P72 companion)",
        "env_flag": "GENESIS_ENABLE_P74_CHUNK_CLAMP",
        "default_on": False,
        "category": "compile_safety",
        "credit": "Genesis-original (zero-VRAM-cost prealloc-overflow safety net for P72-unblocked batched_tokens>4096)",
        "upstream_pr": None,
    },
    "P75": {
        "title": "Auto-enable Suffix Decoding (Arctic Inference, vllm#25784)",
        "env_flag": "GENESIS_ENABLE_P75_SUFFIX_DECODING",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Backport-enabler of vllm#25784 (Arctic Inference Suffix Decoding) — operator convenience: auto-swap method=ngram→suffix when env enabled. Algorithm: arxiv 2411.04975.",
        "upstream_pr": 25784,
    },
    "P77": {
        "title": "Adaptive ngram K controller (EMA + hysteresis + auto-disable)",
        "env_flag": "GENESIS_ENABLE_P77_ADAPTIVE_NGRAM_K",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Genesis-original (port of SGLang adaptive_spec_params.py EMA+hysteresis Apache-2.0 + Nightjar arXiv 2512.22420 auto-disable extension). Targets free-form ngram pathology (46 tok/s).",
        "upstream_pr": None,
    },
    "P78": {
        "title": "TurboQuant .tolist() capture-guard (adapted from noonghunna)",
        "env_flag": "GENESIS_ENABLE_P78_TOLIST_CAPTURE_GUARD",
        "default_on": False,
        "category": "compile_safety",
        "credit": "Adapted from noonghunna's patch_tolist_cudagraph.py (Apache-2.0, github.com/noonghunna/qwen36-27b-single-3090). Surgical safety-net for cudagraph capture; complements our P22/P26/P44 prealloc.",
        "upstream_pr": None,
    },
    "P79b": {
        "title": "Async × spec-decode proposer-sync backport (vllm#40610)",
        "env_flag": "GENESIS_ENABLE_P79B_ASYNC_PROPOSER_SYNC",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Backport of vllm#40610 (OPEN draft, tracked from #40608). Re-records prepare_inputs_event AFTER spec-decode proposer GPU work in sample_tokens(). Fixes async × spec-decode race where next batch _update_states could mutate block_table while previous batch's proposer was still reading on GPU. Genesis prod uses sync ngram so direct value is minimal; protects users on async+EAGLE/MTP/ngram_gpu.",
        "upstream_pr": 40610,
    },
    "P79c": {
        "title": "Stale spec_token_ids cleanup for unscheduled requests (vllm#37629)",
        "env_flag": "GENESIS_ENABLE_P79C_STALE_SPEC_TOKEN_CLEANUP",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Backport of vllm#37629 (OPEN, fixes #36906). Cleanup pass after main scheduling loop clears spec_token_ids for unscheduled running requests. Prevents -1 placeholder leak into F.embedding() under budget-exhausted high-concurrency on async + EAGLE/MTP. Genesis prod (max_num_seqs=2, sync ngram) gains nothing direct; protects high-concurrency multimodal users.",
        "upstream_pr": 37629,
    },
    "P81": {
        "title": "fp8 block-scaled MM low-M decode tuning (vllm#40925)",
        "env_flag": "GENESIS_ENABLE_P81_FP8_BLOCK_SCALED_M_LE_8",
        "default_on": False,
        "category": "kernel_perf",
        "credit": "Backport of vllm#40925 (tonyliu312, OPEN). Specializes w8a8_triton_block_scaled_mm default config for M<=8 (single-request decode + MTP K=3 verify): BLOCK_SIZE_M 64->16, num_stages 2->3 (non-ROCm). Empirical +23% median decode on GB10. Direct hit for Genesis prod (Qwen3.6-A3B FP8 + max_num_seqs=2 + no pre-tuned JSON for A5000).",
        "upstream_pr": 40925,
    },
    "P82": {
        "title": "SGLang threshold_single OR-clause acceptance (BIASED — opt-in research)",
        "env_flag": "GENESIS_ENABLE_P82",
        "default_on": False,
        "category": "spec_decode",
        "credit": "SGLang team (sgl-project/sglang) speculative_sampling.cuh — port of the threshold_single OR-clause that breaks the structural ceiling clean_rate ≈ accept_rate^num_spec. Targets v7.13 strict-ngram acceptance gap. BIASED rule (loses unbiased-sampling guarantee); requires empirical quality validation before prod. Threshold baked from env GENESIS_P82_THRESHOLD_SINGLE (default 0.3) at server start.",
        "upstream_pr": None,
    },
    "P83": {
        "title": "MTP keep-last-cached-block (vllm#38182 downstream symptom — P84 is real fix)",
        "env_flag": "GENESIS_ENABLE_P83",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Root-cause analysis: vllm#38182 by uOnePiece + @Angazenn comment identifying single_type_kv_cache_manager.py:457 force-pop last cached block when use_eagle=True. MTP gets caught up via config/speculative.py:890-891 (use_eagle returns True for 'mtp'). EMPIRICALLY DISPROVEN as the actual cause: Genesis debug instrumentation showed find_longest_cache_hit was NEVER called for our workload because num_hashes=0 (block_size > prompt_len after P5 LCM-pad). The L457 pop is a downstream symptom, not the upstream cause. P84 (hash_block_size override) is the real fix. P83 kept as opt-in research artifact for future workloads where the pop site IS reached.",
        "upstream_pr": None,
    },
    "P84": {
        "title": "hash_block_size override (vllm#38182 actual root cause)",
        "env_flag": "GENESIS_ENABLE_P84",
        "default_on": False,
        "category": "kv_cache",
        "credit": "Genesis-original discovery 2026-04-27 via P83 DEBUG instrumentation. scheduler.py:234 hard-codes hash_block_size=self.block_size; on hybrid Qwen3.6-MoE with P5 LCM-pad this becomes 2048+, so request_block_hasher computes 0 hashes for prompts < 2048 tokens. Cache machinery runs with overhead but never produces hits. P84 text-patches scheduler.py to read hash_block_size from env GENESIS_P84_HASH_BLOCK_SIZE (recommended value: 16 = full-attention default). Engage via GENESIS_ENABLE_P84=1 + GENESIS_P84_HASH_BLOCK_SIZE=16. Constraint: must divide every group's block_size, else vLLM's own assertion fires at startup. Related: vllm#38182 identified WRONG root cause (the L457 pop); P84 attacks the upstream cause.",
        "upstream_pr": None,
    },
    "P85": {
        "title": "Hybrid fine-shadow prefix cache (vllm#38182 followup, MambaManager fix)",
        "env_flag": "GENESIS_ENABLE_P85",
        "default_on": False,
        "category": "kv_cache",
        "credit": "Genesis-original 2026-04-27 — synthesis of 6-round empirical investigation + deep code analysis. Identified TWO mismatches in hybrid prefix cache: (A) MambaManager.cache_blocks early-returns for prompts < self.block_size (e.g., 1424 < 2048); (B) Mamba align-mode pads with null_blocks so num_full_blocks > 0 still inserts 0 entries. P85 patches MambaManager to: (1) register shadow fine-grained hash entries (scale_factor=block_size/hash_block_size duplicates) when caching, (2) walk fine hashes on lookup with eviction-safety re-derive verify. Memory layout / ref-count untouched. Requires P84 (fine hashes computed). Architectural limit: cannot help prompts < block_size (Mamba state genuinely uncached at sub-block boundaries).",
        "upstream_pr": None,
    },
    "P86": {
        "title": "ngram batch_propose O(N*K) → O(N+K) direct-fill (vllm#40876)",
        "env_flag": "GENESIS_ENABLE_P86",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Backport of vllm#40876 (aaronagent, OPEN). Replaces O(N*K) `i in valid_ngram_requests` membership scan in NgramProposer.batch_propose with O(N+K) direct-fill loop iterating only the valid ngram requests. Algorithmic improvement, no behavioral change. Negligible at Genesis prod max_num_seqs=2 (~ns); meaningful at high-concurrency multi-user serving (e.g. N=64, K=32 saves ~1952 list-membership ops per batch step).",
        "upstream_pr": 40876,
    },
}


# ─── Single-call gate ─────────────────────────────────────────────────────

def should_apply(patch_id: str) -> tuple[bool, str]:
    """Unified gate: returns (apply_decision, reason).

    Combines:
      - env-flag check (`GENESIS_ENABLE_P<patch>=1` opt-in)
      - `config_detect.recommend(patch_id)` (model+config-aware decision)
      - `model_detect` arch checks if applicable

    The decision rule:

      1. If env flag is set to truthy → always apply (operator override)
      2. If env flag is unset/falsy AND patch is `default_on=False` → skip (opt-in)
      3. Otherwise consult `config_detect.recommend()`:
         - "skip:..." → don't apply
         - "redundant:..." → don't apply
         - "deprecated:..." → don't apply
         - "neutral" / "apply" → apply

    Returns:
        (True, reason) — patch should apply
        (False, reason) — patch should skip, with human-readable reason
    """
    meta = PATCH_REGISTRY.get(patch_id)
    if meta is None:
        return False, f"unknown patch_id {patch_id!r}"

    env_flag = meta.get("env_flag")
    env_value = os.environ.get(env_flag, "") if env_flag else ""
    env_truthy = env_value.strip().lower() in ("1", "true", "yes", "on")

    # Operator override: env truthy = always apply (subject to anchor presence)
    if env_truthy:
        # But still consult config_detect to PRINT the recommendation as info
        try:
            from vllm._genesis.config_detect import recommend
            verdict, reason = recommend(patch_id)
            if verdict == "apply":
                return True, f"opt-in env + config recommends apply: {reason}"
            elif verdict == "neutral":
                return True, f"opt-in env (config: neutral)"
            else:
                return True, (
                    f"opt-in env OVERRIDE (config recommends {verdict}: "
                    f"{reason}) — proceeding because operator forced it"
                )
        except Exception as e:
            return True, f"opt-in env (config_detect probe failed: {e})"

    # Env flag unset/falsy
    if not meta.get("default_on", False):
        if meta.get("deprecated", False):
            return False, (
                f"opt-in only AND empirically deprecated — "
                f"keeping skip; set {env_flag}=1 only for diagnostics"
            )
        return False, f"opt-in only — set {env_flag}=1 to engage"

    # default_on=True patches still consult config_detect
    try:
        from vllm._genesis.config_detect import recommend
        verdict, reason = recommend(patch_id)
        return (verdict in ("apply", "neutral")), f"config_detect: {verdict}:{reason}"
    except Exception as e:
        return False, f"config_detect failed: {e}"


# ─── Decision logging ─────────────────────────────────────────────────────

# Module-level cache of decisions made this boot, for matrix dump.
_DECISIONS: list[dict[str, Any]] = []


def log_decision(patch_id: str, applied: bool, reason: str) -> None:
    """Log + record a patch decision for the boot-time matrix dump.

    Single condensed line per patch. Operator can see all decisions at boot
    via `Genesis Dispatcher v2 decisions:` log block (called from apply_all).
    """
    meta = PATCH_REGISTRY.get(patch_id, {})
    title = meta.get("title", patch_id)
    status = "APPLY" if applied else "SKIP "
    log.info(
        "[Genesis Dispatcher] %s %s — %s | %s",
        status, patch_id, title, reason[:120],
    )
    _DECISIONS.append({
        "patch_id": patch_id,
        "title": title,
        "applied": applied,
        "reason": reason,
        "env_flag": meta.get("env_flag", ""),
        "credit": meta.get("credit", ""),
        "upstream_pr": meta.get("upstream_pr"),
    })


def get_apply_matrix() -> list[dict[str, Any]]:
    """Return the recorded apply matrix for this boot.

    Useful for tests + diagnostic dump.
    """
    return list(_DECISIONS)


def dump_apply_matrix() -> str:
    """Format the apply matrix as ASCII table (string for printing).

    Columns: patch_id, status, title, reason (truncated), credit.
    """
    if not _DECISIONS:
        return "(no decisions recorded — Genesis Dispatcher hasn't been used yet)"

    # Compute column widths
    rows = [
        (
            d["patch_id"],
            "APPLY" if d["applied"] else "SKIP",
            d["title"][:45],
            d["reason"][:60],
            d.get("credit", "")[:30],
        )
        for d in _DECISIONS
    ]
    widths = [max(len(r[i]) for r in rows) for i in range(5)]
    widths = [max(w, len(h)) for w, h in zip(widths,
              ["Patch", "Status", "Title", "Reason", "Credit"])]

    def _fmt_row(r):
        return " | ".join(c.ljust(widths[i]) for i, c in enumerate(r))

    lines = []
    lines.append(_fmt_row(["Patch", "Status", "Title", "Reason", "Credit"]))
    lines.append("-+-".join("-" * w for w in widths))
    for r in rows:
        lines.append(_fmt_row(r))
    return "\n".join(lines)


def log_apply_matrix() -> None:
    """Emit the apply matrix as a multi-line INFO block.

    Called by apply_all at end of boot to give operator a single readable
    summary instead of grep-ing through scattered INFO lines.
    """
    matrix = dump_apply_matrix()
    log.info(
        "[Genesis Dispatcher v2] apply matrix:\n%s",
        matrix,
    )


# ─── CLI entry-point ──────────────────────────────────────────────────────

def main() -> int:
    """`python3 -m vllm._genesis.dispatcher` — print apply matrix as table.

    Useful for diagnostics OUTSIDE a vllm boot (e.g. dry-run profiling).
    Walks the patch registry and dry-evaluates should_apply() for each.
    """
    print("Genesis Dispatcher v2 — patch decision matrix")
    print("=" * 80)
    print()

    # Run should_apply against every registered patch
    for patch_id in PATCH_REGISTRY:
        decision, reason = should_apply(patch_id)
        log_decision(patch_id, decision, reason)

    print(dump_apply_matrix())
    print()
    print("Note: this is a STATIC dispatch view. Some recommendations are")
    print("'skip' because vllm config isn't set in this dry-run context.")
    print("In real boot, get_runtime_profile() returns the actual config.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
