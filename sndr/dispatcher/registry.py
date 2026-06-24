# SPDX-License-Identifier: Apache-2.0
"""SNDR Core dispatcher — PATCH_REGISTRY data (single source of truth).

This file is DATA-ONLY. It exceeds the 300-LOC cap on logic files
because it's a registry of all known patches with their metadata
(2000+ lines of dict entries). The cap exemption was approved by
Sander on 2026-05-07 (Idea 2 of Stage 2 acceptance) — data files
that grow with the patch count are not covered by the structural
LOC limit. Logic files MUST stay under 300 LOC.

Each entry declares:
  - title: short human-readable description
  - env_flag: env var that toggles this patch (without prefix; the
    SNDR_ENABLE_/GENESIS_ENABLE_ prefix is added at lookup time)
  - default_on: bool — whether the patch applies without explicit env
  - category: family bucket (deprecated by Stage 6 family taxonomy)
  - tier: "community" | "engine"  (added at Stage 5)
  - lifecycle: "active" | "deprecated" | "retired" | "legacy"
  - credit: provenance + design notes
  - upstream_pr: int | None (vllm-project/vllm PR number, if applicable)
  - applies_to: dict of profile-key → allowed values (model_class etc.)
  - requires_patches / conflicts_with: dependency declarations
  - composes_with: informational soft-link to related patches
  - apply_module: dotted path to the module providing apply()
    (added at Stage 6 reorg)

Migration history:
  - Original location: vllm/_genesis/dispatcher.py (2828-LOC monolith).
  - Stage 3 (CURRENT): split out as data-only module here.
"""
from __future__ import annotations

from typing import Any


# ─── Patch metadata registry ───────────────────────────────────────────────
# Each patch declares what it touches + which env flag enables/disables it.
# This is the SINGLE source of truth for patch-to-feature mapping.

PATCH_REGISTRY: dict[str, dict[str, Any]] = {
    # P56 + P57: archived 2026-05-05 to
    # ../Genesis_internal_docs/_archive/dead_patches/p56_p57_tq_specdec_deadends/
    # P56 = TQ spec-decode safe-path guard (superseded by P65)
    # P57 = TQ spec-decode capture-safe buffers (~1080 MiB regression, dead-end)
    # See archive README for the full investigation thread.
    "P58": {
        "title": "Async-scheduler -1 placeholder fix",
        "tier": "community",
        "family": "scheduler",
        "env_flag": "GENESIS_ENABLE_P58_ASYNC_PLACEHOLDER_FIX",
        "default_on": False,
        "category": "spec_decode",
        "credit": "z1ying (vllm#40768)",
        "upstream_pr": 40768,
        "upstream_pr_relationship": "backport",
        "apply_module": "sndr.engines.vllm.patches.scheduler.p58_async_scheduler_placeholder_fix",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    # P59 (Qwen3 reasoning embedded tool_call recovery, vllm#39055) was
    # consolidated into the P61b entry on 2026-06-20 — all three reasoning
    # parser patches (P61b + P59 + PN51) share one apply_module
    # (p61b_p59_pn51_qwen3_reasoning_consolidated). P59's enable flag
    # GENESIS_ENABLE_P59_QWEN3_TOOL_RECOVERY is retained as an env_flag_alias
    # on P61b so existing YAML opt-ins keep engaging the merged module.
    "P60": {
        "title": "GDN+ngram state recovery (Phase 1: SSM pre-copy)",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_P60_GDN_NGRAM_FIX",
        "default_on": False,
        "category": "spec_decode",
        "credit": "tdoublep (vllm#40738), bhaktatejas922 (#39273)",
        "upstream_pr": 40738,
        "upstream_pr_relationship": "backport",
        "applies_to": {"is_hybrid": [True]},
        "apply_module": "sndr.engines.vllm.patches.attention.gdn.p60_gdn_ngram_state_recovery",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P60b": {
        "title": "GDN+ngram Triton kernel offset (Phase 2)",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_P60B_TRITON_KERNEL",
        "default_on": False,
        "category": "spec_decode",
        "credit": "tdoublep (vllm#40738)",
        "upstream_pr": 40738,
        "upstream_pr_relationship": "backport",
        "applies_to": {"is_hybrid": [True]},
        "requires_patches": ["P60"],
        "apply_module": "sndr.engines.vllm.patches.attention.gdn.p60b_gdn_ngram_triton_kernel",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P61": {
        "title": "Qwen3 multi-tool first-occurrence (RETIRED — fully superseded by P12 v2)",
        "tier": "community",
        "family": "reasoning",
        "env_flag": "GENESIS_ENABLE_P61_QWEN3_MULTI_TOOL",
        "default_on": False,
        "lifecycle": "retired",  # migrated from `deprecated: True` to explicit retired lifecycle. P12 v2 directly emits FIRST-occurrence; P61 anchor never matched in active form. Safety-gated.
        "category": "structured_output",
        "credit": "ExtReMLapin (vllm#40783) — P61 was supposed to flip P12's LAST-occurrence to FIRST via post-anchor replacement, but its anchor 'tool_call_index = ...' never matched P12-emitted 'idx = ...' form, so it silent-skipped when P12 was active. v7.62.5 (2026-04-28): P12 emit updated to FIRST directly; P61 retired. Setting GENESIS_ENABLE_P61=1 is now a harmless no-op (anchor not found vs already-FIRST P12 output).",
        "upstream_pr": 40783,
        "upstream_pr_relationship": "related_not_superseding",
        "applies_to": {"model_class": ["qwen3", "qwen3_5", "qwen3_6", "qwen3_moe", "qwen3_next"]},
        "apply_module": "sndr.engines.vllm._archive.p61_qwen3_multi_tool_first_occurrence",
        "retired_waiver": True,  # P12 v2 directly emits FIRST-occurrence; P61 anchor never matched in active form. Harmless no-op.
        "implementation_status": "full",
    },
    "P62": {
        "title": "Structured-output spec-decode reasoning-end timing fix",
        "tier": "community",
        "family": "serving",
        "env_flag": "GENESIS_ENABLE_P62_STRUCT_OUT_SPEC_TIMING",
        "default_on": False,
        "category": "structured_output",
        "credit": "sfbemerk (vllm#36138), cicirori (vllm#34650)",
        "upstream_pr": 36138,
        "upstream_pr_relationship": "backport",
        "applies_to": {"model_class": ["qwen3", "qwen3_5", "qwen3_6", "qwen3_moe", "qwen3_next"]},
        "conflicts_with": ["PN58"],
        "apply_module": "sndr.engines.vllm.patches.serving.p62_structured_output_spec_decode_timing",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    # P61c (Qwen3Coder deferred-commit, club-3090#72) was consolidated into
    # the P64 entry on 2026-06-20 — all three qwen3coder parser patches
    # (P64 + P61c + PN56) share one apply_module
    # (p64_p61c_pn56_qwen3coder_consolidated). P61c's enable flag
    # GENESIS_ENABLE_P61C_QWEN3CODER_DEFERRED_COMMIT is retained as an
    # env_flag_alias on P64 so existing YAML opt-ins keep engaging the merged
    # module.
    "P61b": {
        "title": (
            "qwen3_reasoning_parser consolidated: streaming partial-tag "
            "overlap guard (vllm#40783) + embedded tool_call recovery "
            "(vllm#39055) + thinking-disabled content routing (vllm#40816)"
        ),
        "tier": "community",
        "family": "reasoning",
        "env_flag": "GENESIS_ENABLE_P61B_STREAMING_OVERLAP",
        # P59 + PN51 were consolidated into this entry on 2026-06-20 (all three
        # patch reasoning/qwen3_reasoning_parser.py at disjoint regions). Their
        # enable flags are recognized aliases so existing builtin YAMLs keep
        # working — should_apply (decision.py::_resolve_env_state) honors
        # env_flag_aliases at the ENTRY level, then the merged module's apply()
        # gates each feature by its own flag + replicated version gate:
        #   GENESIS_ENABLE_P61B_STREAMING_OVERLAP            -> p61b group
        #   GENESIS_ENABLE_P59_QWEN3_TOOL_RECOVERY           -> p59 group
        #   GENESIS_ENABLE_PN51_QWEN3_STREAMING_THINKING_DISABLED -> pn51 group
        "env_flag_aliases": [
            "GENESIS_ENABLE_P59_QWEN3_TOOL_RECOVERY",
            "GENESIS_ENABLE_PN51_QWEN3_STREAMING_THINKING_DISABLED",
        ],
        "default_on": False,
        "category": "structured_output",
        "credit": (
            "Consolidated 2026-06-20 (maintainability refactor, runtime-"
            "neutral): P61b + P59 + PN51 all patch reasoning/"
            "qwen3_reasoning_parser.py at disjoint regions, now share one "
            "apply_module (p61b_p59_pn51_qwen3_reasoning_consolidated) with "
            "three independently env-gated feature groups. Each group is its "
            "OWN TextPatcher carrying its ORIGINAL marker verbatim (failure "
            "isolation: a P61b anchor drift must not skip P59's subs; and no "
            "Layer-2 marker cross-shadowing) — so the applied bytes are "
            "byte-identical to P61b+P59+PN51 applied separately, INCLUDING "
            "the per-feature marker lines (verified: 8/8 flag-combo md5 "
            "match on a simulated <0.23.0 pristine tree). "
            "(1) P61b — backport slice of vllm#40783 (ExtReMLapin): streaming "
            "partial-tag overlap guard (holds back half-formed <tool_call> "
            "fragments assembled across deltas). Primary flag "
            "GENESIS_ENABLE_P61B_STREAMING_OVERLAP. P61b is the merged "
            "primary because it is co-enabled in every builtin YAML; P59 is "
            "enabled in ZERO YAMLs. "
            "(2) P59 — backport of vllm#39055 (ZenoAFfectionate): promotes "
            "tool_call XML emitted inside <think>...</think> out of reasoning "
            "into content so qwen3_coder can parse it. Flag (alias) "
            "GENESIS_ENABLE_P59_QWEN3_TOOL_RECOVERY. "
            "(3) PN51 — backport of vllm#40816 (fixed upstream by #40820): "
            "defensive streaming short-circuit routing deltas to "
            "delta.content when thinking is disabled. Flag (alias) "
            "GENESIS_ENABLE_PN51_QWEN3_STREAMING_THINKING_DISABLED. "
            "VERSION GATE: all three carry vllm_version_range "
            "(>=0.20.0,<0.23.0); the #45413/#45588 parser reorg (MERGED in "
            "dev148) restructured the qwen3 reasoning+tool target and the "
            "native engine parser owns embedded-tool-call recovery / "
            "tag-overlap on 0.23.x. The merged module replicates this version "
            "gate inside each per-group helper (check_version_constraints "
            "under live GENESIS_ENFORCE_VERSION_RANGE=1) so a >=0.23.0 pin "
            "where the file is still present version-SKIPs every group "
            "instead of corrupting the native parser — matching what the "
            "standalone originals (routing through should_apply) did."
        ),
        "upstream_pr": 40783,
        "upstream_pr_relationship": "backport",
        # All three absorbed patches share this identical range, so the
        # entry-level range is correct and does NOT over-gate. should_apply's
        # version-only gate (decision.py::_check_version_gate) fires BEFORE
        # the env branch and is LIVE on the rig (GENESIS_ENFORCE_VERSION_RANGE
        # composed from a5000-2x hardware yaml), so the whole module
        # version-gate-SKIPs on dev148.
        "applies_to": {"model_class": ["qwen3", "qwen3_5", "qwen3_6", "qwen3_moe", "qwen3_next"], "vllm_version_range": (">=0.20.0", "<0.23.0")},
        "apply_module": "sndr.engines.vllm.patches.reasoning.p61b_p59_pn51_qwen3_reasoning_consolidated",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P63": {
        "title": "MTP/Eagle drafter GDN state recovery (RETIRED — hypothesis disproven 2026-04-25)",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_P63_MTP_GDN_STATE_RECOVERY",
        "default_on": False,
        "lifecycle": "retired",  # migrated from `deprecated: True`. Hypothesis disproven empirically; safety-gated.
        "retired_waiver": True,  # Genesis-original hypothesis disproven 2026-04-25; no upstream supersession (not a backport)
        "apply_module": "sndr.engines.vllm._archive.p63_mtp_gdn_state_recovery",
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
        "implementation_status": "full",
    },
    "P64": {
        "title": (
            "qwen3coder_tool_parser consolidated: MTP streaming early-return "
            "fix (vllm#39598) + deferred-commit until <function= "
            "(club-3090#72) + XML parse fallback (vllm#41466)"
        ),
        "tier": "community",
        "family": "tool_parsing",
        "env_flag": "GENESIS_ENABLE_P64_QWEN3CODER_MTP_STREAMING",
        # P61c + PN56 were consolidated into this entry on 2026-06-20 (all
        # three patch tool_parsers/qwen3coder_tool_parser.py at disjoint
        # regions; P64 also has a second serving.py target). Their enable
        # flags are recognized aliases so existing builtin YAMLs keep working
        # — should_apply honors env_flag_aliases at the ENTRY level, then the
        # merged module's apply() gates each feature by its own flag +
        # replicated version gate:
        #   GENESIS_ENABLE_P64_QWEN3CODER_MTP_STREAMING       -> p64 group
        #   GENESIS_ENABLE_P61C_QWEN3CODER_DEFERRED_COMMIT    -> p61c group
        #   GENESIS_ENABLE_PN56_QWEN3CODER_XML_FALLBACK       -> pn56 group
        "env_flag_aliases": [
            "GENESIS_ENABLE_P61C_QWEN3CODER_DEFERRED_COMMIT",
            "GENESIS_ENABLE_PN56_QWEN3CODER_XML_FALLBACK",
        ],
        "default_on": False,
        "category": "structured_output",
        "credit": (
            "Consolidated 2026-06-20 (maintainability refactor, runtime-"
            "neutral): P64 + P61c + PN56 all patch tool_parsers/"
            "qwen3coder_tool_parser.py at disjoint regions, now share one "
            "apply_module (p64_p61c_pn56_qwen3coder_consolidated) with three "
            "independently env-gated feature groups. Each group is its OWN "
            "TextPatcher carrying its ORIGINAL marker verbatim (failure "
            "isolation + no Layer-2 marker cross-shadowing) — so the applied "
            "bytes are byte-identical to P64+P61c+PN56 applied separately, "
            "INCLUDING the per-feature marker lines (verified: 8/8 flag-combo "
            "md5 match on a simulated <0.23.0 pristine tree). "
            "(1) P64 — backport of vllm#39598 (kotori-yan): removes the "
            "early-return after parameter fragments so MTP-bundled "
            "last-param+</function> deltas don't drop the closing brace. P64 "
            "ALSO carries TWO serving.py sub-patches "
            "(p64_safety_net_widen + p64_callsite_guard) that are RETIRED-by-"
            "design on dev259+ (required=False, 0-match a pristine tree — the "
            "helper they anchored on was refactored out; P107 carries the "
            "serving-side role now). They are kept as a separate serving "
            "patcher: serving.py stays BYTE-UNTOUCHED on pristine (verified). "
            "Primary flag GENESIS_ENABLE_P64_QWEN3CODER_MTP_STREAMING. "
            "(2) P61c — local mitigation for club-3090#72 (troymroberts): "
            "defers is_tool_call_started=True until <function= appears in a "
            "64-char slack window, so a narrative <tool_call> mention no "
            "longer causes 30-120s SSE silence. Flag (alias) "
            "GENESIS_ENABLE_P61C_QWEN3CODER_DEFERRED_COMMIT. "
            "(3) PN56 — backport of vllm#41466 (ToastyTheBot): on XML parse "
            "failure restores prev_tool_call_arr arguments from "
            "streamed_args + closing brace so the serving layer does not "
            "double-emit the '{}' placeholder. Flag (alias) "
            "GENESIS_ENABLE_PN56_QWEN3CODER_XML_FALLBACK. "
            "VERSION GATE: all three carry vllm_version_range "
            "(>=0.20.0,<0.23.0); #45171/#45588 deleted/remapped the "
            "qwen3coder/qwen3xml tool parsers in the dev148-era engine, which "
            "owns streaming tool-call extraction natively. The merged module "
            "replicates this version gate inside each per-group helper "
            "(check_version_constraints under live GENESIS_ENFORCE_VERSION_"
            "RANGE=1) so a >=0.23.0 pin where the files are still present "
            "version-SKIPs every group instead of leaking tool-call XML to "
            "content on the native parser — matching what the standalone "
            "originals (routing through should_apply) did."
        ),
        "upstream_pr": 39598,
        "upstream_pr_relationship": "backport",
        # All three absorbed patches share this identical range, so the
        # entry-level range is correct and does NOT over-gate. should_apply's
        # version-only gate fires BEFORE the env branch and is LIVE on the rig
        # (GENESIS_ENFORCE_VERSION_RANGE), so the whole module version-gate-
        # SKIPs on dev148.
        "applies_to": {"model_class": ["qwen3", "qwen3_5", "qwen3_6", "qwen3_moe", "qwen3_next"], "vllm_version_range": (">=0.20.0", "<0.23.0")},
        "apply_module": "sndr.engines.vllm.patches.tool_parsing.p64_p61c_pn56_qwen3coder_consolidated",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P65": {
        "title": "TurboQuant spec-decode cudagraph downgrade (FALLBACK — see P67 root-cause fix)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_P65_TURBOQUANT_SPEC_CG_DOWNGRADE",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Genesis-original (root cause for noonghunna #40880). Fallback safety-net workaround that downgrades cudagraph PIECEWISE → ~30% TPS hit. SUPERSEDED by P67/P67b root-cause fix (TurboQuant multi-query kernel for spec-decode K+1 verify). Default OFF; opt-in only when P67 unavailable or unstable.",
        "upstream_pr": None,
        "applies_to": {"is_turboquant": [True]},
        # PN353B added 2026-06-10: it downgrades the same cudagraph
        # ClassVar (see PN353B credit), so the conflict is mutual.
        "conflicts_with": ["P67", "P67b", "PN353B"],
        # NOTE: P67/P67b is the root-cause fix (multi-query kernel) and
        # P65 is the safety-net fallback. Relationship explained in
        # `credit`. Not using `superseded_by` because P65 has no pin-gate
        # boundary (it's a runtime fallback choice, not version retire).
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.p65_turboquant_spec_cg_downgrade",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P66": {
        "title": "cudagraph_capture_sizes spec-decode divisibility filter",
        "tier": "community",
        "family": "compile_safety",
        "env_flag": "GENESIS_ENABLE_P66_CUDAGRAPH_SIZE_FILTER",
        "default_on": False,
        "category": "spec_decode",
        "credit": (
            "Genesis-original runtime filter (mirrors fhl2000 vllm#23679, "
            "closed-without-merge 2026-03-13). Audit 2026-05-24 "
            "(PIN.R-P66-METADATA.1): relationship reclassified from "
            "`backport` to `related_not_superseding` — Genesis P66 is a "
            "narrow runtime filter on cudagraph_capture_sizes at the "
            "consumer call site, while the cited upstream PR was a "
            "broader config-time refactor of the size-derivation logic. "
            "Same bug class (#28015 family), different layer, coverage "
            "does not overlap. Status-only retire FORBIDDEN — "
            "audit_upstream_status.py routes this entry to "
            "RELATED-NOT-SUPERSEDING bucket; iron-rule-#11 deep-parity "
            "remains the only retire path."
        ),
        "upstream_pr": 23679,
        "upstream_pr_relationship": "related_not_superseding",
        "apply_module": "sndr.engines.vllm.patches.compile_safety.p66_cudagraph_size_divisibility_filter",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P68": {
        "title": "Auto force tool_choice=required for long-context tool calls",
        "tier": "community",
        "family": "serving",
        "env_flag": "GENESIS_ENABLE_P68_AUTO_FORCE_TOOL",
        "default_on": False,
        "category": "structured_output",
        "credit": "Genesis-original (long-ctx tool adherence mitigation)",
        "upstream_pr": None,
        "applies_to": {"model_class": ["qwen3", "qwen3_5", "qwen3_6", "qwen3_moe", "qwen3_next"]},
        "apply_module": "sndr.engines.vllm.patches.serving.p68_69_long_ctx_tool_adherence",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P69": {
        "title": "Long-context tool-format reminder injection",
        "tier": "community",
        "family": "serving",
        "env_flag": "GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER",
        "default_on": False,
        "category": "structured_output",
        "credit": "Genesis-original (long-ctx tool adherence mitigation)",
        "upstream_pr": None,
        "applies_to": {"model_class": ["qwen3", "qwen3_5", "qwen3_6", "qwen3_moe", "qwen3_next"]},
        "apply_module": "sndr.engines.vllm.patches.serving.p68_69_long_ctx_tool_adherence",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P70": {
        "title": "Auto-strict-ngram (force prompt_lookup_min>=8)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_P70_AUTO_STRICT_NGRAM",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Genesis-original (vllm#40875 enforcement)",
        "upstream_pr": None,
        "applies_to": {
            # [Genesis pin-gate 2026-05-11] PROD-active (GroupAB component
            # +9.2% on 27B). Validated dev9 → dev93.
            "vllm_version_range": (">=0.20.0", "<0.24.0"),
        },
        "apply_module": "sndr.engines.vllm.patches.spec_decode.p70_auto_strict_ngram",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN72": {
        "title": "Frequency-based ngram draft post-filter (llama.cpp-style)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_PN72_FREQUENCY_NGRAM_DRAFTER",
        "default_on": False,
        "category": "spec_decode",
        "credit": (
            "Genesis-original 2026-05-06. Mirrors llama.cpp's "
            "draft_min_sample_size + draft_min_percent heuristic from "
            "common/ngram-cache.cpp — but as POST-filter on vllm's "
            "longest-match ngram drafter (because vllm uses numba JIT, "
            "text-patching it is risky). Wraps NgramProposer.propose, "
            "rejects drafts whose first token has < MIN_OBS occurrences "
            "in the recent window. Tunables: "
            "GENESIS_PN72_MIN_OBSERVATIONS (default 4), "
            "GENESIS_PN72_FREQUENCY_WINDOW (default 1024). Composable with "
            "P70 (orthogonal — both can be on for stricter drafting). "
            "Goal: reject spurious draft-acceptances from chat-template "
            "ambiguity (Qwen3-coder `<<` 2-token-suffix bug class — see "
            "Genesis v7.13 BREAKTHROUGH). Safety: graceful fallback to "
            "unfiltered drafts on any internal error."
        ),
        "upstream_pr": None,
        "apply_module": "sndr.engines.vllm.patches.spec_decode.pn72_frequency_ngram_drafter",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN77": {
        "title": "FP8 lm_head compression (BF16→FP8 e4m3 + per-channel scale)",
        "tier": "community",
        "family": "quantization",
        "env_flag": "GENESIS_ENABLE_PN77_FP8_LM_HEAD",
        "default_on": False,
        "category": "memory_savings",
        "credit": (
            "Genesis-original 2026-05-07 (Phase E MVP). Backport pattern of "
            "vllm PR #35696 (lucaspirola, OPEN) + PR #35694 (FP8 weight "
            "storage in unquantized linear/embedding apply). Compresses "
            "lm_head BF16 weight to float8_e4m3fn with per-channel scale. "
            "Saves ~606 MiB/rank on 27B Qwen3.6 (vocab=248320, hidden=5120), "
            "~243 MiB/rank on 35B (hidden=2048). MVP forward path: cast-back "
            "to BF16 on each call (~3 ms/token, absorbed by spec-decode "
            "latency). Phase E.3+ will replace with apply_fp8_marlin_linear "
            "for weight-only FP8 GEMM (no per-call cast). Quality gate: "
            "cosine_sim ≥ 0.999 verified in unit tests; integration A/B "
            "(tool-call 10/10 + reasoning probe) required before promoting. "
            "Tied-embedding detection prevents poisoning embed_tokens."
        ),
        "upstream_pr": None,  # PR #35696 OPEN; Genesis is preemptive backport
        "apply_module": "sndr.engines.vllm.patches.quantization.pn77_fp8_lm_head",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN80": {
        "title": "LoRA tensorizer device kwarg (vllm#41845 backport)",
        "tier": "community",
        "family": "lora",
        "env_flag": "GENESIS_ENABLE_PN80_LORA_TENSORIZER_DEVICE",
        "default_on": False,
        "lifecycle": "retired",  # 2026-05-11 v2 audit: byte-equivalent in dev209
        "vllm_version_range": "<0.20.2rc1.dev93",  # mirrors applies_to (gated SoT); reconciled D17 2026-06-17 (was >=...dev16+g7a1eb8ac2,<...dev209+g5536fc0c0)
        "apply_module": "sndr.engines.vllm._archive.pn80_lora_tensorizer_device",
        "category": "memory_savings",
        "credit": (
            "Backport of vllm#41845 (Or Ozeri @ IBM, MERGED 2026-05-07 main "
            "HEAD — not in nightly image as of dev93+g51f22dcfd). Single-line "
            "fix: pass `device=device` to TensorDeserializer in "
            "lora/lora_model.py LoRA loading path. Without device kwarg, "
            "tensorizer first deserializes to host RAM (full tensor size, "
            "potentially 2-50 GB depending on LoRA rank), then transfers to "
            "GPU — peak host RAM blows up causing OOM on memory-constrained "
            "rigs. With kwarg, tensorizer streams directly to GPU. Genesis "
            "35B/27B PROD does not currently use LoRA — patch ready for "
            "community deployments or Sander LoRA workloads."
        ),
        "upstream_pr": 41845,
        "upstream_pr_relationship": "backport",
        "superseded_by": "vllm#41845 (merged 2026-05-07, byte-identical in dev209 lora_model.py:206 has `device=device` arg in TensorDeserializer call)",
        "applies_to": {
            # [Iron rule #11 retire 2026-05-11 v2 audit] PN80 was
            # never deployed on Genesis PROD (no LoRA workload). On
            # dev209 deep-diff verified: lora_model.py:203-206 has
            # the exact `device=device` arg PN80 was backporting.
            "vllm_version_range": "<0.20.2rc1.dev93",
        },
        "requires_patches": [],
        "conflicts_with": [],
        "implementation_status": "full",
    },
    "PN79": {
        "title": "In-place SSM state for GDN chunk prefill (vllm#41824 backport)",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_PN79_INPLACE_SSM_STATE",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.attention.gdn.pn79_inplace_ssm_state",
        "lifecycle": "experimental",
        "category": "memory_savings",
        "credit": (
            "Backport of vllm#41824 (Kermit-C, OPEN as of 2026-05-06). "
            "Eliminates per-decode-step gather/scatter copies of "
            "initial_state and final_state in chunk_gated_delta_rule_fwd_h "
            "by passing ssm_state_indices directly to the Triton kernel. "
            "Author claims 4.5-36 GiB cumulative fp32 traffic eliminated "
            "per multi-turn session (Qwen3.5-0.8B → Qwen3.6-27B scale). "
            "ORTHOGONAL TO PN59: PN59 fixes prefill peak (h-allocation), "
            "PN79 fixes decode steady-state allocator pressure. Empirical "
            "Genesis 2026-05-06: PN59 _streaming_path almost never fires "
            "in real workloads (chunked-prefill T=64 chunks always "
            "bypass T<1024 gate; multi-seq batches always bypass single-seq "
            "guard) — therefore PN79 has higher actual ROI on our stack. "
            "Status 2026-05-07: FULL IMPLEMENTATION LANDED + LIVE A/B "
            "VALIDATED — 17 anchors across 3 files (Sub-1 chunk.py: 7 "
            "[1A/1B/1C/1D + 1E_SIG/1E_VAL/1E_APPLY_CALL high-level API], "
            "Sub-2 chunk_delta_h.py: 7 [2A heuristics, 2B kernel sig, 2C "
            "kernel main, 2D kernel epilogue, 2E wrapper sig, 2F wrapper "
            "body strides, 2G wrapper kernel call], Sub-3 gdn_linear_attn.py: "
            "3 [3A forward_cuda fallback, 3B forward_native passthrough, "
            "3C _forward_core gather/scatter elim]). All 17 OLD anchors "
            "match server live source (vllm pin dev60+ge47c98ef7) uniquely; "
            "pristine dev9 chunk_delta_h.py is bit-identical to live dev60 "
            "(kernel unchanged across 51 dev versions). Live boot 27B "
            "Lorbus INT4+TQ k8v4 successful (135-155s); GDN warmup all 32 "
            "layers clean. A/B bench 27B 2026-05-07: TPS +1.1% within "
            "noise (105.3 vs 104.2), VRAM identical (45485 MiB), tool 10/10 "
            "match — single-shot win unproven, multi-turn evidence pending. "
            "58 PN79 tests + 2260 full Genesis suite pass on Mac. Atomic "
            "3-file MultiFilePatchTransaction + conflicts_with [PN59, "
            "PN54] guard. Default OFF, lifecycle experimental. PN59/PN54 "
            "lifecycle migration pending Stage 4 multi-turn evidence "
            "(currently both stable in registry; PN79 docstring describes "
            "intended final state, not current registry state). "
            "K.2 RE-ANCHOR 2026-06-10 on pin 0.22.1rc1.dev259+g303916e93 "
            "(post-#44700 GDN mixed-batch split + core_attn_out): all "
            "anchors re-derived from upstream #41824 rebased diff "
            "(2026-06-09) and verified byte-exact against LIVE PROD "
            "container files. Sub-1 now 8 anchors (1D split into "
            "decorator/sig+contiguity/inner-call; input_guard IMPORT kept, "
            "torch.accelerator.device_index wrapper skipped — single-device "
            "TP workers). Sub-3 re-targeted to mamba/gdn/qwen_gdn_linear_"
            "attn.py with backend gate (in-place kwargs only when "
            "self.gdn_prefill_backend == 'triton'; flashinfer/cutedsl keep "
            "upstream gather/scatter — old 3A forward_cuda anchor retired). "
            "Sub-4 re-targeted to mamba/gdn/olmo_gdn_linear_attn.py "
            "(models/olmo_hybrid.py still exists on the pin but lost the "
            "GDN code — stale target would have aborted the transaction). "
            "P103 chunked_fwd bail-out extended with ssm_state_indices "
            "guard (Cliff-2 protection preserved). Still opt-in; "
            "enable+bench on PROD pending."
        ),
        "upstream_pr": 41824,
        "upstream_pr_relationship": "backport",
        "applies_to": {"is_hybrid": [True]},
        "requires_patches": [],
        "conflicts_with": ["PN59", "PN54"],
        "implementation_status": "full",
    },
    "PN79_V2_MD5_CHUNK": {
        "title": "PN79 v2 — md5+full-file PoC (PN119 reference pattern, chunk.py scope)",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_PN79_V2_MD5_CHUNK",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Genesis PoC of the PN119 md5+full-file pattern applied to "
            "pn79's chunk.py target — sibling 1 of pn79's multi-file "
            "conversion. pn79 originally targets 4 files (chunk.py, "
            "chunk_delta_h.py, gdn_linear_attn.py, olmo_hybrid.py); the "
            "last 2 have drifted out of upstream entirely (gdn split into "
            "model-specific files under gdn/{kimi,olmo,qwen}_gdn_linear_"
            "attn.py; olmo_hybrid.py removed). Drift finding during scout "
            "(2026-06-03): pn79 silently applies only 3 of its 7 chunk.py "
            "anchors on current pin (4 ANCHOR_1B/1D/1E_SIG/1E_APPLY_CALL "
            "do not match upstream). md5+full-file pattern documents "
            "this drift transparently and prevents the silent partial "
            "apply. Composes with PN79 (Genesis marker prevents "
            "re-anchoring on chunk.py post-v2). Default OFF — operator "
            "opt-in for PoC validation."
        ),
        "upstream_pr": 41824,
        "upstream_pr_relationship": "backport",
        "applies_to": {"model_class": ["qwen3_5", "qwen3_6", "qwen3_next"]},
        "implementation_status": "full",
        "apply_module": "sndr.engines.vllm.patches.attention.gdn.pn79_v2_md5_chunk",
        "source": "vllm_pr_backport",
        "lifecycle": "experimental",
        "conflicts_with": [],
        "requires_patches": [],
    },
    "PN79_V2_MD5_CHUNK_DELTA_H": {
        "title": "PN79 v2 — md5+full-file PoC (PN119 reference pattern, chunk_delta_h.py scope)",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_PN79_V2_MD5_CHUNK_DELTA_H",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Genesis PoC sibling 2 — chunk_delta_h.py scope. 3/4 pn79 "
            "anchors apply cleanly on current pin (ANCHOR_2B_KERNEL_SIG "
            "drifted). md5+full-file pattern catches this. Composes with "
            "PN79 + PN79_V2_MD5_CHUNK. Default OFF."
        ),
        "upstream_pr": 41824,
        "upstream_pr_relationship": "backport",
        "applies_to": {"model_class": ["qwen3_5", "qwen3_6", "qwen3_next"]},
        "implementation_status": "full",
        "apply_module": "sndr.engines.vllm.patches.attention.gdn.pn79_v2_md5_chunk_delta_h",
        "source": "vllm_pr_backport",
        "lifecycle": "experimental",
        "conflicts_with": [],
        "requires_patches": [],
    },
    "PN78": {
        "title": "[RETIRED] One-shot empty_cache() after CG warmup",
        "tier": "community",
        "family": "memory",
        "env_flag": "GENESIS_ENABLE_PN78_POST_WARMUP_CACHE_RELEASE",
        "default_on": False,
        "lifecycle": "retired",  # migrated from "deprecated" — upstream pin handles cache release internally; this wrap is permanent no-op.
        "vllm_version_range": "<0.20.2rc1.dev9",  # mirrors applies_to (gated SoT); reconciled D17 2026-06-17 (was >=...dev16+g7a1eb8ac2,<...dev9+g01d4d1ad3)
        "apply_module": "sndr.engines.vllm._archive.pn78_post_warmup_cache_release",
        "category": "memory_savings",
        "credit": (
            "DEPRECATED 2026-05-07: see deprecation_note. Patch retained "
            "for documentation; env flag honored but executes no-op."
        ),
        "deprecation_note": (
            "Investigation 2026-05-07 (MEMORY_DEEP_PLAN Phase 2.1): vllm "
            "pin already calls torch.accelerator.empty_cache() inside "
            "GPUModelRunner.capture_model (gpu_model_runner.py:6213 "
            "BEFORE capture, :6244 AFTER capture, before lock_workspace). "
            "PN78 wrap would be redundant 3rd call. Additionally, "
            "apply_all()-time monkey patches do NOT reach worker "
            "processes (VLLM_WORKER_MULTIPROC_METHOD=spawn → fresh "
            "interp). Genesis pattern for worker-visible patches is "
            "source-level edits to vllm core (e.g. PN59 in "
            "fla/ops/chunk.py). No upstream PR needed — upstream is "
            "already correct."
        ),
        "upstream_pr": None,
        "implementation_status": "retired",
        "superseded_by": "upstream torch.accelerator.empty_cache() calls in GPUModelRunner.capture_model (gpu_model_runner.py:6213/6244 in dev9+) — PN78 would be redundant 3rd call",
        "applies_to": {
            # [Iron rule #11 formal retire 2026-05-11] Promoted from
            # waiver — deprecation_note identified specific supersession
            # points in upstream code. PN78 wrap permanent no-op since
            # dev9+; pin-gate documents the boundary.
            "vllm_version_range": "<0.20.2rc1.dev9",
        },
    },
    "P67": {
        "title": "TurboQuant multi-query kernel for spec-decode K+1",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Genesis-original (proper fix for noonghunna #40880; replaces P65 workaround). 2026-05-05 NOTE: alternative upstream fix is OPEN as vllm#40914 (Sandermage) — uses synth_seq_lens routing through existing decode kernel instead of new Genesis-original kernel. If #40914 merges, P67 becomes one of two equivalent paths; defer retirement decision until empirical TPS A/B (P67 currently delivers +32% on 35B-A3B-FP8 PROD vs upstream baseline). Watch_for_drift_via vllm#40914.",
        "upstream_pr": None,
        "applies_to": {
            "is_turboquant": [True],
            # [Genesis pin-gate 2026-05-11] PROD-active patch (35B +32% TPS).
            # Validated dev16 → dev93. Broad range; drift detector handles
            # anchor-line breakage on bumps.
            # 2026-06-17 pin-bump 0.23.1: cap <0.23.0 -> <0.24.0. ROOT CAUSE
            # of the 0.23.1 MTP degenerate-loop: P67 (the TurboQuant
            # multi-query kernel for the spec-decode K+1 verify batch) was
            # version-gated OFF on 0.23.1 despite GENESIS_ENABLE_P67=1 in the
            # 35B config -> the native multi-query TQ spec-decode path ran ->
            # garbage drafts rubber-stamped (93% accept) -> constant-token
            # loop. MTP-off was unaffected (P67 only fires for spec-decode).
            # P67b (spec-verify forward routing) requires P67, so this unblocks
            # both. Drift detector confirmed P67 anchors still resolve on 0.23.1.
            "vllm_version_range": (">=0.20.0", "<0.24.0"),
        },
        "conflicts_with": ["P65", "G4_67"],
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.p67_tq_multi_query_kernel",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P67b": {
        "title": "TurboQuant spec-verify forward() routing (FULL CG enable)",
        "tier": "community",
        "family": "attention.turboquant",
        # P67b reuses P67's env flag intentionally — they're a coupled pair,
        # P67b is the forward() routing companion that bypasses
        # _prefill_attention for K+1 verify batches (cudagraph-safe).
        "env_flag": "GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Genesis-original (FULL CG enable for P67 multi-query kernel)",
        "upstream_pr": None,
        "applies_to": {
            "is_turboquant": [True],
        },
        "requires_patches": ["P67"],
        "conflicts_with": ["P65", "G4_67"],
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.p67b_spec_verify_routing",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN398": {
        "title": "Async spec-decode accepted-counts race fix (vllm#45100 backport)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_PN398_ASYNC_ACCEPTED_RACE",
        "default_on": False,
        "category": "spec_decode",
        "credit": (
            "Backport of vllm#45100 (OPEN, verified+approved tdoublep/ZJY0516) "
            "— fixes the 0.23.x async spec-decode num_accepted_tokens race for "
            "hybrid GDN/Mamba + MTP (Qwen3.5/3.6). Async scheduling became "
            "default-on for MTP via #27614/#31998 (2025-12/2026-01), exercising "
            "a racy stale-CPU-copy path -> GDN recurrence restored from the "
            "wrong slot -> prompt-memory-loss constant-token loop (93% accept, "
            "K=1 too). MTP-off unaffected. Diagnosed 2026-06-17 via structural "
            "research + confirmed live (--no-async-scheduling A/B). Keeps async "
            "scheduling ON. Auto-no-ops on upstream merge (drift marker "
            "'needs_cpu_accepted_counts')."
        ),
        "upstream_pr": 45100,
        "upstream_pr_relationship": "backport",
        "applies_to": {
            "is_hybrid": [True],
            # 0.23.x regression only (async became MTP-default on 0.23.0).
            # dev491/0.22.1 never hit it -> gate >=0.23.0.
            "vllm_version_range": (">=0.23.0", "<0.24.0"),
        },
        "apply_module": "sndr.engines.vllm.patches.spec_decode.pn398_async_accepted_counts_race",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P72": {
        "title": "profile_run M cap (unblocks --max-num-batched-tokens>4096 on MoE)",
        "tier": "community",
        "family": "worker",
        "env_flag": "GENESIS_ENABLE_P72_PROFILE_RUN_CAP",
        "default_on": False,
        "category": "compile_safety",
        "credit": "Genesis-original (Dynamo fake-tensor mismatch workaround for moe_align_block_size symbolic shape)",
        "upstream_pr": None,
        "applies_to": {
            # [Genesis pin-gate 2026-05-11] PROD-active (caps profile_run
            # M to GENESIS_PROFILE_RUN_CAP_M, unblocks batched_tokens>4096
            # on MoE). Validated dev9 → dev93.
            "vllm_version_range": (">=0.20.0", "<0.24.0"),
        },
        "apply_module": "sndr.engines.vllm.patches.worker.p72_profile_run_cap",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P71": {
        "title": "Block-verify rejection sampler (Sun 2024 ICLR) + PN369 relaxed acceptance",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_P71_BLOCK_VERIFY",
        # 2026-06-19: PN369 (relaxed acceptance) was consolidated INTO this
        # entry — both text-patch v1/sample/rejection_sampler.py at disjoint
        # regions (P71 ~:471 block-verify branch; PN369 ~:489-506 kernel
        # signature + OR-compose body + launch-site mask). One apply_module
        # (p71_pn369_rejection_sampler_consolidated) carries all four sub-
        # patches; each feature is independently gated by its own env flag
        # inside apply() so the applied kernel-code bytes are byte-identical
        # to P71+PN369 applied separately. PN369's enable flag is retained as
        # an env_flag_alias below so its existing YAML opt-in still engages
        # the merged module. NOTE: PN369 carried vllm_version_range
        # (>=0.22.0,<0.24.0); this entry has none, so the consolidated
        # apply() replicates PN369's version gate internally (the dispatcher
        # version-only gate fires before env-override and is LIVE on the rig
        # via GENESIS_ENFORCE_VERSION_RANGE=1).
        "env_flag_aliases": ["GENESIS_ENABLE_PN369_RELAXED_ACCEPTANCE"],
        "default_on": False,
        "category": "spec_decode",
        "credit": "Backport of vllm#40819 (Z. Golpayegani draft) + Sun et al. arXiv 2403.10444 + 2 critical fixes from gemini-code-assist review (shared u per request, denom==0 → 1.0). Consolidated 2026-06-19 with PN369 relaxed acceptance (Genesis-original, TRT-LLM-style top-K + delta window; opt-in, BIASED rule; the P71 block-verify path tail-extends the Sun-2024 accepted length while the relaxed window holds).",
        "upstream_pr": 40819,
        "upstream_pr_relationship": "backport",
        "apply_module": "sndr.engines.vllm.patches.spec_decode.p71_pn369_rejection_sampler_consolidated",
        "lifecycle": "experimental",
        "implementation_status": "full",
        # drift D1 (2026-06-15): P71's block-verify branch references the dense
        # target_probs buffer that PN390 removes (rejection_sampler.py:518/525)
        # -> latent NameError if both fire under probabilistic draft. Symmetric
        # to PN390.conflicts_with. Dormant on PROD (greedy draft gates it off).
        # PN369 shared the SAME PN390 conflict (it reads the same full-vocab
        # target_probs local), so the merge is symmetric — no new conflict.
        "conflicts_with": ["PN390"],
        # Merged from PN369.composes_with (deduped; P71-self dropped). P82
        # (per-token threshold), PN90 (probabilistic draft), PN361 (fail-
        # closed missing probs).
        "composes_with": ["P82", "PN90", "PN361"],
    },
    "P74": {
        "title": "Auto chunk-clamp via long_prefill_token_threshold (P72 companion)",
        "tier": "community",
        "family": "scheduler",
        "env_flag": "GENESIS_ENABLE_P74_CHUNK_CLAMP",
        "default_on": False,
        "category": "compile_safety",
        "credit": "Genesis-original (zero-VRAM-cost prealloc-overflow safety net for P72-unblocked batched_tokens>4096)",
        "upstream_pr": None,
        "applies_to": {"model_class": ["qwen3", "qwen3_5", "qwen3_6", "qwen3_moe", "qwen3_next"]},
        "requires_patches": ["P72"],
        "apply_module": "sndr.engines.vllm.patches.scheduler.p74_chunk_clamp",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P75": {
        "title": "Auto-enable Suffix Decoding (Arctic Inference, vllm#25784)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_P75_SUFFIX_DECODING",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Backport-enabler of vllm#25784 (Arctic Inference Suffix Decoding) — operator convenience: auto-swap method=ngram→suffix when env enabled. Algorithm: arxiv 2411.04975.",
        "upstream_pr": 25784,
        "upstream_pr_relationship": "enables_upstream",
        # [Iron rule #11 audit 2026-05-11 v2] P75 is NOT a backport —
        # it's a convenience activator ON TOP of merged upstream feature
        # (#25784 in pin since 2025-11). Audit script routes via the
        # explicit `upstream_pr_relationship: "enables_upstream"` field
        # to exclude from NEWLY-MERGED categorization. KEEP active —
        # convenience value preserved.
        "apply_module": "sndr.engines.vllm.patches.spec_decode.p75_suffix_decoding_enable",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P77": {
        "title": "Adaptive ngram K controller (EMA + hysteresis + auto-disable)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_P77_ADAPTIVE_NGRAM_K",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Genesis-original (port of SGLang adaptive_spec_params.py EMA+hysteresis Apache-2.0 + Nightjar arXiv 2512.22420 auto-disable extension). Targets free-form ngram pathology (46 tok/s).",
        "upstream_pr": None,
        "applies_to": {
            # 2026-06-19 drift audit: the engine CHANGED the NgramProposer.propose()
            # API on 0.22.x — propose() now takes an explicit first positional
            # `num_speculative_tokens: int` (+ `assert num_speculative_tokens <=
            # self.k` + `batch_propose(..., k)`), so K flows per-call instead of via
            # `self.k`. P77's "override self.k then restore" mechanism is anchor-dead
            # AND partly defeated by the new fixed-width valid_ngram_draft buffers.
            # Re-engaging would be a REWRITE, not a re-anchor. Capped <0.22.0 (the
            # signature change predates the dev148 pin); the adaptive-K idea is still
            # valid but must be re-authored against the per-call-K API before lifting.
            # default_off + in zero builtin YAMLs, so no runtime change.
            "vllm_version_range": (">=0.20.0", "<0.22.0"),
        },
        "apply_module": "sndr.engines.vllm.patches.spec_decode.p77_adaptive_ngram_k",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P78": {
        "title": "TurboQuant .tolist() capture-guard (adapted from noonghunna) — RETIRED 2026-06-11",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_P78_TOLIST_CAPTURE_GUARD",
        "default_on": False,
        "category": "compile_safety",
        "credit": "Adapted from noonghunna's patch_tolist_cudagraph.py (Apache-2.0, github.com/noonghunna/qwen36-27b-single-3090). Was the surgical safety-net for cudagraph capture; complemented our P22/P26/P44 prealloc. RETIRED — upstream absorbed all guarded sites.",
        # [Retire 2026-06-11, preflight residual triage §3, byte-verified]
        # Upstream absorbed Sites B/C/D/E on pin 0.22.1rc1.dev259: CPU-
        # mirror metadata fields query_start_loc_cpu / seq_lens_cpu
        # (pristine turboquant_attn.py:190-193), build() wiring populating
        # them (237-238), prefill_max_seq taken from seq_lens_cpu with
        # max_seq_len fallback (486-489), and the continuation path doing
        # CPU-first .tolist() with an explicit "otherwise .tolist() on GPU
        # tensors forces a synchronizing copy" comment (601-610). The
        # buggy GPU-tensor .tolist() pattern P78 guarded is gone from the
        # file. Module archived.
        "superseded_by": "upstream-native CPU-mirror metadata path — Sites B/C/D/E absorbed (pristine turboquant_attn.py:190-193, 237-238, 486-489, 601-610); GPU .tolist() pattern gone (byte-verified 2026-06-11)",
        "vllm_version_range": "<0.22.1rc1.dev259",  # supersession byte-verified on this pin's pristine tree
        "upstream_pr": None,
        "applies_to": {
            "is_turboquant": [True],
            "quant_format": ["fp8", "compressed_tensors"],
        },
        "apply_module": "sndr.engines.vllm._archive.p78_tolist_capture_guard",
        "lifecycle": "retired",
        "implementation_status": "full",
    },
    "P79b": {
        "title": "Async × spec-decode proposer-sync backport (vllm#40610)",
        "tier": "community",
        "family": "worker",
        "env_flag": "GENESIS_ENABLE_P79B_ASYNC_PROPOSER_SYNC",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Backport of vllm#40610 (OPEN draft, tracked from #40608). Re-records prepare_inputs_event AFTER spec-decode proposer GPU work in sample_tokens(). Fixes async × spec-decode race where next batch _update_states could mutate block_table while previous batch's proposer was still reading on GPU. Genesis prod uses sync ngram so direct value is minimal; protects users on async+EAGLE/MTP/ngram_gpu.",
        "upstream_pr": 40610,
        "upstream_pr_relationship": "backport",
        "apply_module": "sndr.engines.vllm.patches.worker.p79b_async_proposer_sync",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P79c": {
        "title": "Stale spec_token_ids cleanup for unscheduled requests (vllm#37629)",
        "tier": "community",
        "family": "scheduler",
        "env_flag": "GENESIS_ENABLE_P79C_STALE_SPEC_TOKEN_CLEANUP",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Backport of vllm#37629 (OPEN, fixes #36906). Cleanup pass after main scheduling loop clears spec_token_ids for unscheduled running requests. Prevents -1 placeholder leak into F.embedding() under budget-exhausted high-concurrency on async + EAGLE/MTP. Genesis prod (max_num_seqs=2, sync ngram) gains nothing direct; protects high-concurrency multimodal users.",
        "upstream_pr": 37629,
        "upstream_pr_relationship": "backport",
        "apply_module": "sndr.engines.vllm.patches.scheduler.p79c_stale_spec_token_cleanup",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P79d": {
        "title": "Preempt async-discard credit grant (vllm#38624 v2 rewrite)",
        "tier": "community",
        "family": "scheduler",
        "env_flag": "GENESIS_ENABLE_P79D_PREEMPT_ASYNC_DISCARD",
        "default_on": False,
        "category": "spec_decode",
        "credit": "v2 rewrite (2026-06-11) of the STALE v1 backport of vllm#38624 (CodersAcademy006, OPEN) — staleness surfaced by the #45146 study (pr-sweep-50 roadmap chunk 2). v1 wrote the dead boolean discard_latest_async_tokens (0 hits on pin 0.22.1rc1.dev259; upstream migrated to integer async_tokens_to_discard) and would have tripped 'assert request.num_output_placeholders >= 0' (async_scheduler.py:60) on the first preempt-resume. v2 grants TOKEN-denominated discard credit BEFORE zeroing placeholders on every preemption path ('+=' so undrained debt survives), neutralizes the reset_prefix_cache '=' credit wipe, drains a stale frame's rejected drafts from credit instead of live counters, and makes the async drain consume len(new_token_ids) per stale frame (upstream's 1-per-frame under-drains under MTP K=3, silently swallowing legitimate post-resume frames). Atomic 2-file transaction (scheduler.py + async_scheduler.py). Coexists with P58 in either apply order (narrow num_preemptions anchor). Genesis 35B PROD (async + MTP K=3, 280K agent ctx) is the exact profile where KV-pressure preemptions hit this path.",
        "upstream_pr": 38624,
        "upstream_pr_relationship": "backport",
        "applies_to": {
            # Anchor-tight lower bound = verification point (integer
            # async_tokens_to_discard credit pattern byte-verified on the
            # 0.22.1rc1.dev259 pristine tree; first-appearance dev not
            # bisected — widen only after study). Older pins carry the
            # boolean-era code this rewrite replaced.
            "vllm_version_range": (">=0.22.1rc1.dev259", "<0.24.0"),
        },
        "apply_module": "sndr.engines.vllm.patches.scheduler.p79d_preempt_async_discard",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P81": {
        "title": "fp8 block-scaled MM low-M decode tuning (vllm#40925)",
        "tier": "community",
        "family": "quantization",
        "env_flag": "GENESIS_ENABLE_P81_FP8_BLOCK_SCALED_M_LE_8",
        "default_on": False,
        "category": "kernel_perf",
        "credit": "Backport of vllm#40925 (tonyliu312, OPEN). Specializes w8a8_triton_block_scaled_mm default config for M<=8 (single-request decode + MTP K=3 verify): BLOCK_SIZE_M 64->16, num_stages 2->3 (non-ROCm). Empirical +23% median decode on GB10. Direct hit for Genesis prod (Qwen3.6-A3B FP8 + max_num_seqs=2 + no pre-tuned JSON for A5000).",
        "upstream_pr": 40925,
        "upstream_pr_relationship": "backport",
        "applies_to": {
            "quant_format": ["fp8"],
        },
        "apply_module": "sndr.engines.vllm.patches.quantization.p81_fp8_block_scaled_m_le_8",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P82": {
        "title": "SGLang threshold_single OR-clause acceptance (BIASED — opt-in research)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_P82",
        "default_on": False,
        "category": "spec_decode",
        "credit": "SGLang team (sgl-project/sglang) speculative_sampling.cuh — port of the threshold_single OR-clause that breaks the structural ceiling clean_rate ≈ accept_rate^num_spec. Targets v7.13 strict-ngram acceptance gap. BIASED rule (loses unbiased-sampling guarantee); requires empirical quality validation before prod. Threshold baked from env GENESIS_P82_THRESHOLD_SINGLE (default 0.3) at server start.",
        "upstream_pr": None,
        "applies_to": {
            # [Genesis pin-gate 2026-05-11] PROD-active (Sprint 1 winner
            # +5.23% TPS at thr=0.1). Validated dev9 → dev93.
            "vllm_version_range": (">=0.20.0", "<0.24.0"),
        },
        "apply_module": "sndr.engines.vllm.patches.spec_decode.p82_sglang_acceptance_threshold",
        "lifecycle": "research",
        "research_note": (
            "BIASED rule — gives up the unbiased-sampling guarantee in "
            "exchange for breaking the clean_rate ≈ accept_rate^num_spec "
            "structural ceiling. Sprint 1 +5.23% TPS at threshold=0.1, "
            "but mathematical bias means output distribution drifts vs "
            "vanilla speculative decoding. Kept as opt-in research; "
            "promotion requires per-workload quality validation (tool-call "
            "score, JSON correctness, reasoning quality) — not just TPS."
        ),
        "implementation_status": "full",
    },
    "P83": {
        "title": "MTP keep-last-cached-block (vllm#38182 downstream symptom — P84 was real fix) — RETIRED 2026-06-11",
        "tier": "community",
        "family": "kv_cache",
        "env_flag": "GENESIS_ENABLE_P83",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Root-cause analysis: vllm#38182 by uOnePiece + @Angazenn comment identifying single_type_kv_cache_manager.py:457 force-pop last cached block when use_eagle=True. MTP gets caught up via config/speculative.py:890-891 (use_eagle returns True for 'mtp'). EMPIRICALLY DISPROVEN as the actual cause: Genesis debug instrumentation showed find_longest_cache_hit was NEVER called for our workload because num_hashes=0 (block_size > prompt_len after P5 LCM-pad). The L457 pop is a downstream symptom, not the upstream cause. P84 (hash_block_size override) was the real fix (itself retired 2026-06-11 — upstream-native). RETIRED.",
        # [Retire 2026-06-11, preflight residual triage §3, byte-verified]
        # Upstream renamed use_eagle → drop_eagle_block in single_type_
        # kv_cache_manager.py (parameter at pristine :405/:530/:608) and
        # added coordinator-level lookahead with eagle_verified
        # bookkeeping (kv_cache_coordinator.py:643-650; the lookahead
        # window logic at 565-571) — this supersedes the convergence-
        # interaction cost P83 tried to dodge; the residual tail-block
        # drop is tracked via open #44986 (OPEN, gh-verified 2026-06-11).
        # Re-anchoring P83 onto the new code is UNSAFE: a skip-pop breaks
        # the monotonic-decrease convergence invariant (coordinator
        # L588-593) and corrupts eagle_verified bookkeeping. Already
        # "empirically disproven" as the root cause per the research_note
        # below. Module archived.
        "superseded_by": "upstream use_eagle→drop_eagle_block rename + coordinator lookahead with eagle_verified bookkeeping (pristine kv_cache_coordinator.py:643-650, 565-571) — supersedes the convergence-interaction cost; residual tail-block drop tracked via open #44986",
        "vllm_version_range": "<0.22.1",  # plan-mandated cap (§3); rename verified on 0.22.1rc1.dev259 pristine
        "upstream_pr": None,
        "applies_to": {
            "is_hybrid": [True],
            "vllm_version_range": "<0.22.1",  # pin-gate mirror of the retire cap
        },
        "apply_module": "sndr.engines.vllm._archive.p83_mtp_keep_last_cached_block",
        "lifecycle": "retired",
        "research_note": (
            "Empirically disproven as the root cause of vllm#38182: "
            "Genesis instrumentation found `find_longest_cache_hit` was "
            "never called for our workload (num_hashes=0 after P5 "
            "LCM-pad makes block_size > prompt_len). The L457 force-pop "
            "is a downstream symptom; P84 (hash_block_size override) is "
            "the actual upstream fix. P83 kept as opt-in research for "
            "future workloads where the pop site IS reachable (e.g. "
            "longer prompts that produce num_hashes>0 with smaller blocks)."
        ),
        "implementation_status": "full",
    },
    "P84": {
        "title": "hash_block_size override (vllm#38182 actual root cause) — RETIRED 2026-06-11",
        "tier": "community",
        "family": "scheduler",
        "env_flag": "GENESIS_ENABLE_P84",
        "default_on": False,
        "category": "kv_cache",
        "credit": "Genesis-original discovery 2026-04-27 via P83 DEBUG instrumentation. scheduler.py hard-coded hash_block_size=self.block_size; on hybrid Qwen3.6-MoE with P5 LCM-pad this became 2048+, so request_block_hasher computed 0 hashes for prompts < 2048 tokens. P84 text-patched scheduler.py to read hash_block_size from env GENESIS_P84_HASH_BLOCK_SIZE. Related: vllm#38182 identified WRONG root cause (the L457 pop); P84 attacked the upstream cause. RETIRED — both sites upstream-native.",
        # [Retire 2026-06-11, preflight residual triage §3, byte-verified]
        # Both P84 sites are upstream-native on pin 0.22.1rc1.dev259:
        # (1) Scheduler.__init__ accepts an explicit `hash_block_size:
        # int | None = None` parameter (pristine v1/core/sched/
        # scheduler.py:72, default resolution at 229-230, passed to the
        # hasher at 242); (2) `resolve_kv_cache_block_sizes` (pristine
        # v1/core/kv_cache_utils.py:593) provides the GCD default, the
        # explicit `cache_config.hash_block_size` override, and the
        # divisibility ValueError — a superset of P84's env override.
        # CAVEAT (verifier, plan §3): the Mamba back-off
        # (kv_cache_utils.py:639-644) runs BEFORE both the GCD default
        # AND the explicit override — before declaring full behavioral
        # equivalence on a prod prefix-caching config, verify
        # mamba_cache_mode='align' resolution actually yields
        # num_hashes>0 (server-side check, queued with the §5 P85 work).
        # Cascade: P85's requires_patches=["P84"] re-triage note added to
        # the P85 entry; the P85 fix itself is the §5 follow-up, NOT part
        # of this batch. Module archived.
        "superseded_by": "upstream-native hash_block_size: scheduler param (pristine v1/core/sched/scheduler.py:72,229-230,242) + resolve_kv_cache_block_sizes with GCD default / cache_config.hash_block_size override / divisibility ValueError (kv_cache_utils.py:593) — byte-verified 2026-06-11; Mamba back-off (639-644) equivalence check queued with §5 P85 work",
        "vllm_version_range": "<0.22.1rc1.dev259",  # supersession byte-verified on this pin's pristine tree
        "upstream_pr": None,
        "applies_to": {"is_hybrid": [True]},
        "apply_module": "sndr.engines.vllm._archive.p84_hash_block_size_override",
        "lifecycle": "retired",
        "implementation_status": "full",
    },
    "P85": {
        "title": "Hybrid fine-shadow prefix cache (vllm#38182 followup, MambaManager fix)",
        "tier": "community",
        "family": "kv_cache",
        "env_flag": "GENESIS_ENABLE_P85",
        "default_on": False,
        "category": "kv_cache",
        "credit": "Genesis-original 2026-04-27 — synthesis of 6-round empirical investigation + deep code analysis. Identified TWO mismatches in hybrid prefix cache: (A) MambaManager.cache_blocks early-returns for prompts < self.block_size (e.g., 1424 < 2048); (B) Mamba align-mode pads with null_blocks so num_full_blocks > 0 still inserts 0 entries. P85 patches MambaManager to: (1) register shadow fine-grained hash entries (scale_factor=block_size/hash_block_size duplicates) when caching, (2) walk fine hashes on lookup with eviction-safety re-derive verify. Memory layout / ref-count untouched. Fine hashes come from upstream-native --hash-block-size (cache_config.hash_block_size, replaces retired P84). Architectural limit: cannot help prompts < block_size (Mamba state genuinely uncached at sub-block boundaries). v2 (2026-06-11, plan §5 both-sites fix): Site 1 re-anchored to the upstream retention_interval cache_blocks signature; Site 2 carries dual anchor variants (pristine-shaped + post-PN346-shaped assembled from PN346's own constants, required-at-least-one) because PN346 — effectively default-ON, boot-dispatched before P85 — rewrites a byte-identical 4-line subsequence mid-anchor; the post-PN346 replacement carries PN346's drop_eagle_block guard.",
        "upstream_pr": None,
        "applies_to": {"is_hybrid": [True]},
        # [Preflight triage §3 cascade — re-triage RESOLVED 2026-06-11 §5]
        # P84 retired (hash_block_size is upstream-native via the
        # scheduler param + resolve_kv_cache_block_sizes, byte-verified
        # in the P84 entry). requires_patches=["P84"] dropped: fine
        # hashes now come from upstream cache_config.hash_block_size
        # (engine arg --hash-block-size) instead of P84's env override
        # — operator-config prerequisite, not a Genesis patch
        # dependency. CAVEAT carried from the verifier: the Mamba
        # back-off (kv_cache_utils.py:639-644) precedes both the GCD
        # default AND the explicit override, so fine hashing only
        # engages with mamba_cache_mode="align"; num_hashes>0 must
        # still be verified server-side on the prod prefix-caching
        # config before any P85 enablement bench.
        "requires_patches": [],
        "composes_with": ["PN346"],  # Site 2 dual anchor variants; PN346 boot-dispatches first
        "apply_module": "sndr.engines.vllm.patches.kv_cache.p85_hybrid_fine_shadow_prefix_cache",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P86": {
        "title": "ngram batch_propose O(N*K) → O(N+K) direct-fill (vllm#40876)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_P86",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Backport of vllm#40876 (aaronagent, OPEN). Replaces O(N*K) `i in valid_ngram_requests` membership scan in NgramProposer.batch_propose with O(N+K) direct-fill loop iterating only the valid ngram requests. Algorithmic improvement, no behavioral change. Negligible at Genesis prod max_num_seqs=2 (~ns); meaningful at high-concurrency multi-user serving (e.g. N=64, K=32 saves ~1952 list-membership ops per batch step).",
        "upstream_pr": 40876,
        "upstream_pr_relationship": "backport",
        "apply_module": "sndr.engines.vllm.patches.spec_decode.p86_ngram_batch_propose_linear",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P87": {
        "title": "Marlin W4A16/W8A16 sub-tile output dim pad-on-load (vllm#40361)",
        "tier": "community",
        "family": "kernels",
        "env_flag": "GENESIS_ENABLE_P87",
        "default_on": False,
        "category": "kernel",
        "credit": "Backport of vllm#40361 (OPEN). MarlinLinearKernel requires per-rank out_features divisible by GPTQ_MARLIN_MIN_THREAD_N=64. Sub-tile shards (e.g. Qwen3.5 GatedDeltaNet.in_proj_ba at TP>=2 with num_v_heads=64, or Intel/Qwen3.6-35B-A3B-int4-AutoRound n=32 shard at TP=2) fail can_implement and force a slow non-Marlin fallback (or refuse to load entirely on Ampere where Machete/CutlassW4A8/AllSpark are unavailable or restricted). P87 wraps three MarlinLinearKernel methods to zero-pad qweight/scales/qzeros/bias along the output dim at load, swap config.partition_weight_shape to padded value so downstream transforms see consistent layout, and slice the extra columns off the output in apply_weights. Runtime cost is zero — padding is one-time at load. PR bench: +24% on 2x RTX 3090 SM 8.6 with Intel/Qwen3.6-35B-A3B-int4-AutoRound TP=2 (137 -> 170 t/s). Closes vllm#35924 generically.",
        "upstream_pr": 40361,
        "upstream_pr_relationship": "backport",
        "applies_to": {
            "quant_format": [
                "int8_w8a16", "int4_w4a16",
                "autoround_int8", "autoround_int4",
                "gptq_int4", "awq_int4", "compressed_tensors",
            ],
        },
        "apply_module": "sndr.engines.vllm.patches.kernels.p87_marlin_pad_sub_tile",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN8": {
        "title": "MTP/draft online-quant propagation (vllm#40849)",
        "tier": "community",
        "family": "loader",
        "env_flag": "GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT",
        "default_on": False,
        "category": "spec_decode",
        "credit": (
            "Backport of vllm#40849 (bhoomit, OPEN). Modifies "
            "`get_draft_quant_config()` so that, when the spec-decode draft "
            "model has no explicit quantization config, it inherits the "
            "target's `OnlineQuantizationConfig` (e.g. fp8_per_tensor). "
            "Frees ~600 MiB on FP8-target + Eagle3 / DFlash / MTP-as-external-"
            "draft worker (1.45 GiB BF16 → 0.88 GiB FP8 on Qwen3-32B + Eagle3 "
            "per PR author bench). Also catches ValueError/FileNotFoundError "
            "in the existing draft lookup path (online-quant methods crash "
            "through checkpoint-config because hf_overrides is callable). "
            "NO-OP for current Genesis prod (Lorbus/Minachist 27B do not run "
            "online-quant + external draft). Becomes valuable when DFlash / "
            "Eagle3 / FP8 stacks roll out."
        ),
        "upstream_pr": 40849,
        "upstream_pr_relationship": "backport",
        "applies_to": {
            # Predicate enforced naturally by the patched function — when
            # spec-decode is off OR target is not online-quantized, the new
            # branch falls through identical to vanilla. No model gating.
        },
        "apply_module": "sndr.engines.vllm.patches.loader.pn8_mtp_draft_online_quant_propagation",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN9": {
        "title": "Independent drafter attention backend (vllm#39930)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_PN9_INDEPENDENT_DRAFTER_ATTN",
        "default_on": False,
        "category": "spec_decode",
        "credit": (
            "Backport of vllm#39930 (MatthewBonanni, MERGED). Allows the "
            "spec-decode drafter to use a different attention backend than "
            "the target model. Unblocks drafters with incompatible "
            "requirements (e.g. DFlash needs non-causal attention support, "
            "which TRITON_ATTN does not provide → ValueError on boot). "
            "Modifies `LLMBaseProposer._create_draft_vllm_config()` to "
            "always reset the drafter's attention backend (None = "
            "auto-select). Genesis minimal port: env "
            "GENESIS_PN9_DRAFTER_BACKEND chooses backend (e.g. FLASH_ATTN); "
            "unset/auto → drafter auto-selects. Does NOT add the new "
            "SpeculativeConfig.attention_backend pydantic field (too "
            "invasive at runtime for a frozen dataclass + field_validator). "
            "Unblocks DFlash spike sprint task without full pin bump risk "
            "from #40860 mega-merge. NO-OP for current Genesis prod (PROD "
            "uses ngram drafter, no attention backend conflict)."
        ),
        "upstream_pr": 39930,
        "upstream_pr_relationship": "backport",
        "superseded_by": "vllm#39930 (merged 2026-04-28, in dev9+) — upstream provides full feature including SpeculativeConfig.attention_backend pydantic field; our PN9 backported only the env-driven subset (less invasive at runtime). Upstream is strictly more capable on dev9+.",
        "vllm_version_range": "<0.20.2rc1.dev9",  # mirrors applies_to (gated SoT); reconciled D17 2026-06-17 (dropped cosmetic +g01d4d1ad3 suffix)
        "apply_module": "sndr.engines.vllm._archive.pn9_independent_drafter_attn_backend",
        "applies_to": {
            # Patch only takes effect inside _create_draft_vllm_config which
            # is only called when spec-decode is active. No additional gate.

            # [Genesis iron-rule-#11 retire 2026-05-11 v2 audit] #39930
            # MERGED 2026-04-28 → in dev9+. Upstream provides FULL feature
            # (including new SpeculativeConfig.attention_backend pydantic
            # field) which our PN9 deliberately did NOT backport (too
            # invasive at runtime for a frozen dataclass + field_validator).
            # Upstream is strictly more capable on dev9+; PN9 was a stop-gap.
            # Pin-gate upper bound documents the boundary. Wiring auto-skips
            # on dev9+. Cleanup: delete patch file in next refactor pass.
            "vllm_version_range": "<0.20.2rc1.dev9",
        },
        "lifecycle": "retired",  # 2026-05-11 v2 audit: upstream more capable, soft-retire formalized
        "implementation_status": "full",
    },
    "PN38": {
        "title": "DFlash drafter quantization support (PR #40425 backport)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_PN38_DFLASH_QUANT_DRAFTER",
        "default_on": False,
        "category": "spec_decode",
        "credit": (
            "Backport of vllm-project/vllm#40425 (infatoshi, OPEN). "
            "Enables quantized DFlash drafter checkpoints (FP8 W8A8, "
            "NVFP4, AWQ, etc.) — correctness/compat fix per PR title, "
            "NOT throughput improvement claim. Today NO-OP for configured BF16 "
            "Qwen3.6-{27B,35B-A3B}-DFlash drafters "
            "(quant_config is None → original dense fast-path runs). "
            "Tomorrow: drop-in support for FP8/NVFP4 drafter checkpoints "
            "(e.g. AEON-7/Qwen3.6-NVFP4-DFlash, llm-compressor self-quant). "
            "Memory savings on adoption: BF16 drafter ~2.4 GB → FP8 ~1.2 GB "
            "per worker, ~2.4 GB total at TP=2 — frees KV-cache headroom. "
            "3 sub-patches into qwen3_dflash.py (Site A: F.linear→qkv_proj; "
            "C: conditional fused-KV; D: quantized fallback in precompute). "
            "Site B (pass quant_config to layer) retired 2026-06-11 — "
            "upstream-native since 0.22.1rc1.dev259 (get_draft_quant_config "
            "init + decoder-layer kwarg); apply() presence-guards both "
            "native lines, loud skip when absent. Composable with PN40-A "
            "(different anchor surfaces in same file)."
        ),
        "upstream_pr": 40425,
        "upstream_pr_relationship": "backport",
        "applies_to": {
            "spec_method": ["dflash"],
        },
        "apply_module": "sndr.engines.vllm.patches.spec_decode.pn38_dflash_quant_drafter",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN40-classifier": {
        "title": "PN40 sub-D workload classifier (chat_completion middleware)",
        "tier": "community",
        # Phase 3B.1 (2026-05-22): family changed from 'middleware' to
        # 'spec_decode' to align with on-disk location
        # (integrations/spec_decode/pn40_workload_classifier_hook.py) and
        # runtime ownership group (PN40 omnibus). The chat-completion
        # middleware mechanism is documented in the title + credit
        # blurb; `family` describes ownership / consumer area, not the
        # text-patch mechanism.
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_PN40_DFLASH_OMNIBUS",
        "default_on": False,
        "category": "spec_decode",
        "credit": (
            "Genesis-original 2026-05-04 — companion to PN40 sub-D. "
            "Text-patches vllm/entrypoints/openai/chat_completion/serving.py "
            "(audit A-13 fix 2026-05-05: was incorrectly listed as "
            "serving_chat.py — actual target is chat_completion/serving.py) "
            "to classify each request as code/short_ctx/long_ctx/free_form "
            "and stash on `request._genesis_pn40_workload_class`. "
            "Consumer is the runtime K-trim hook in PN40 sub-C "
            "(scheduler.update_draft_token_ids). Toggled jointly with "
            "PN40 master via GENESIS_ENABLE_PN40_DFLASH_OMNIBUS — no "
            "separate enable flag (sub-D is universal companion to sub-C). "
            "Tier bias: code +1, long_ctx -1, others 0. Defensive on "
            "unknown class names (falls through to neutral bias)."
        ),
        "upstream_pr": None,
        "applies_to": {
            "spec_method": ["dflash", "mtp"],
        },
        "apply_module": "sndr.engines.vllm.patches.spec_decode.pn40_workload_classifier_hook",  # explicit ref — file exists but auto-derivation can't infer from PN40-classifier ID
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN40": {
        "title": "Spec-decode omnibus (A DFlash K-norm + B pool + C adaptive K + D sentinel)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_PN40_DFLASH_OMNIBUS",
        "default_on": False,
        "category": "spec_decode",
        "credit": (
            "Genesis-original 2026-05-04 — 4-component omnibus spec-decode "
            "optimization with strict no-regression contract. "
            "Sub-A (DFlash-only): fused per-layer K-norm Triton kernel "
            "replaces L-iteration loop in qwen3_dflash.py:397-404. "
            "Numerical TDD 12/12 PASS rel_avg=0.0000. Microbench vs "
            "_custom_ops.rms_norm: 3.22x (27B L=5) / 5.32x (35B L=8). "
            "Per-draft-step saving +37us (27B) / +70us (35B). "
            "Sub-B (DFlash-only MVP): persistent K/V buffer pool, "
            "LRU-bounded, hit-rate tracked. Saves cudaMalloc churn. "
            "Sub-C (UNIVERSAL): adaptive K/N controller, mirrors SGLang "
            "tier policy + EMA hysteresis. Default tiers MTP K=3 [0,1,3], "
            "DFlash N=5 [0,1,3,5], DFlash N=3 [0,1,3]. NaN-trip safety. "
            "Applies to ALL 4 configs (27B/35B x MTP/DFlash). "
            "Sub-D (UNIVERSAL): workload classifier (code/short/long/"
            "free-form) + stability sentinel (sliding-window AL drop "
            "detector + NaN trip). Applies to ALL 4 configs. "
            "TDD: 12/12 (sub-A numerical) + 35/35 (sub-B/C/D logic). "
            "Per-sub env toggles GENESIS_PN40_ENABLE_SUB_{A,B,C,D}=0 "
            "to disable individually. Strict-superset throughout: "
            "any eligibility failure falls through to baseline. Default "
            "OFF master-gated until A/B prod-validates. Composes "
            "additively with PN21/PN23/PN24/P77."
        ),
        "upstream_pr": None,
        "applies_to": {
            "spec_method": ["dflash", "mtp"],  # C+D universal across both
        },
        # v11.3.0 BUG #8 fix: PN40 spec apply_module is now the canonical
        # omnibus orchestrator (`pn40_dflash_omnibus`). The omnibus wires
        # sub-A K-norm Triton kernel + sub-B persistent K/V pool + sub-C
        # adaptive K/N controller + sub-D classifier_hook (called
        # internally at line 381-382). Previously incorrectly pointed at
        # sub-D classifier_hook only, which on v12.0.0 spec-mode flip
        # would drop sub-A/B/C — silent regression of K-norm fusion and
        # adaptive K/N control. The standalone `PN40-classifier` spec
        # entry stays for callers wanting classifier-only activation
        # (sub-D's apply is idempotent via marker — safe if omnibus
        # already ran).
        "apply_module": "sndr.engines.vllm.patches.spec_decode.pn40_dflash_omnibus",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    # PN37 archived 2026-05-04 to vllm/_genesis/_not_used_artifact/.
    # Premise (FA2 dead-zone for tiny-Q non-causal) was disproved by
    # microbench: torch SDPA already routes to FA2 packed-GQA path well.
    # Kernel + TDD (rel_avg < 0.01) preserved as research artifact;
    # entry intentionally NOT in PATCH_REGISTRY (no dispatcher matrix
    # row, no apply_all skip-noise on every boot).
    # PN36 was removed 2026-05-04 — was a misdiagnosis. The 5 `self.reasoner`
    # call-sites I found were inserted by OUR P62 backport (vllm#36138, still
    # OPEN), NOT by upstream. Upstream PR #41199 (MERGED 2026-05-01, included
    # in our pin) intentionally moved reasoner to per-request lazy build via
    # `self._get_reasoner(request)`. Pristine upstream code does NOT reference
    # `self.reasoner`. Fix path: disable P62 on this pin (it collides with
    # the rename refactor) and backport PR #40962 separately for the
    # post-reasoning-boundary spec-decode case.
    "PN50": {
        "title": "GDN proj fusion (SGLang#21019 backport — Qwen3.5/3.6 contiguous-projection Triton kernel)",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_PN50_GDN_FUSED_PROJ",
        "default_on": False,
        "category": "perf_kernel",
        "credit": (
            "Backport of SGLang PR #21019 (MERGED 2026-03-23, commit "
            "5bdc07d). Original Triton kernel by Yuan Luo (@yuan-luo), "
            "Apache-2.0. Replaces the unfused split/reshape/cat/.contiguous() "
            "chain (5-6 launches + 2 explicit copies) in `gdn_linear_attn.py:562-566` "
            "Qwen3.5/3.6 contiguous projection branch with single fused Triton "
            "kernel `fused_qkvzba_split_reshape_cat_contiguous` (310 LOC). "
            "Pure data-copy kernel — no math, no reductions, no numerical drift. "
            "Output layout bit-identical to unfused PyTorch. Wrapper falls "
            "through to PyTorch reference on: non-contiguous input, non-pow2 "
            "head_dim, V_PER_GROUP non-integer, kernel launch failure. "
            "CORRECTED 2026-06-14 (drift D6): the 35B is hybrid_gdn_moe — 30 "
            "GDN/Mamba2 + 10 full-attn layers (NOT 'Qwen3MoE, no GDN' as "
            "previously claimed). head_dim=256 is pow2, so PN50's fused proj "
            "kernel DOES engage on those 30 GDN layers and has NOT been "
            "deliberately A/B'd on the 35B — pin PN50 and bench code+tool_call "
            "(the GDN-heaviest variants) before trusting it on 35B. Claimed "
            "gain on H200/Qwen3.5-35B-A3B (SGLang "
            "naming): +7.4% TPS, -10.8% TTFT, -31.2% ITL P95. On A5000 + "
            "27B Lorbus expect modest gain (memory-bound layer, A5000 PCIe "
            "slower than H200). Composable with PN11/PN29/PN32/P103 — verified "
            "no overlap (PN11 acts in interleaved branch, others in different "
            "files). Default OFF until live A/B prod-validates. "
            "RE-ANCHOR 2026-06-09 (PROD pin 0.22.1rc1.dev259+g303916e93): "
            "the `b, a = ba.chunk(2, dim=-1)` line in the Qwen3.5 contiguous "
            "branch became `b, a = self.split_ba(ba)` via vllm#41126's "
            "`mamba/` → `mamba/gdn/qwen_gdn_linear_attn.py` rename. Re-anchored "
            "to the new shape; forward_cpu() still uses the old chunk(2,...) "
            "form but lacks the .contiguous() pair, so the 9-line block "
            "remains unique to forward()."
        ),
        "upstream_pr": None,
        "applies_to": {
            "model_class": ["qwen3_5", "qwen3_6"],
            # [2026-06-20] Explicit upper cap. PN50's anchor STILL matches the
            # live >=0.23 tree, but native fla/ops fused GDN kernels
            # (fused_gdn_prefill_post_conv / fused_sigmoid_gating) now supersede
            # it. Only the default-OFF env flag held it back — without a cap a
            # stray flag-flip on dev148+ would stack the SGLang Triton kernel on
            # native-superseded code. PN50 routes through should_apply, so this
            # gate fires under GENESIS_ENFORCE_VERSION_RANGE=1. Matches the
            # sibling Qwen patch bound; excludes 0.23.1rc1.dev148.
            "vllm_version_range": (">=0.20.0", "<0.23.0"),
        },
        "apply_module": "sndr.engines.vllm.patches.attention.gdn.pn50_gdn_fused_proj",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN59": {
        "title": "Streaming-GDN orchestrator (Variant D Phase 2) — true Cliff 2b OOM fix",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_PN59_STREAMING_GDN",
        "default_on": False,
        "category": "hybrid",
        "credit": (
            "Genesis-original 2026-05-05, Variant D Phase 2. Replaces the "
            "(B, NT, H, V, K) full materialization in `chunk_gated_delta_rule_"
            "fwd_h → chunk_fwd_o` consumer pair with window-iterative driver. "
            "Eliminates Cliff 2b multi-turn OOM (Issue #19) — root cause: 805 "
            "MiB single allocation per layer per forward at T=64K Genesis 27B "
            "Lorbus shapes. Cross-engine validation: llama.cpp + MLX-LM use "
            "pure-streaming register-resident state, survive multi-turn; "
            "vLLM/SGLang/FLA materialize-full, hit Cliff 2b. **Independent "
            "confirmation** by noonghunna (issue #20, 2026-05-05): 'the "
            "limitation is the triton kernel for cliff 2; doesn't appear with "
            "llama.cpp'. Phase 1 numerical TDD proves bit-equivalence "
            "(rtol<1e-5) on 10 Genesis 27B shape cases. Composes with "
            "PN50/PN54/PN26b/P67 (orthogonal). Supersedes P103 outer chunked "
            "wrapper when both ON. Default OFF until live A/B prod-validates."
        ),
        "upstream_pr": None,  # FLA RFC #485, #190 pending — first-mover position
        "applies_to": {
            "model_class": ["qwen3_5", "qwen3_6"],  # 27B Lorbus hybrid only
        },
        # Symmetric with PN79's declaration (audit 2026-05-12): PN79 wraps
        # the GDN chunk-prefill path with in-place SSM state; this
        # streaming-GDN orchestrator targets the same call site differently.
        "conflicts_with": ["PN79"],
        "apply_module": "sndr.engines.vllm.patches.attention.gdn.pn59_streaming_gdn",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN58": {
        "title": "Spec-decode reasoning boundary validation — narrower alt to P62 (vllm#40962)",
        "tier": "community",
        "family": "reasoning",
        "env_flag": "GENESIS_ENABLE_PN58_SPEC_REASONING_BOUNDARY",
        "default_on": False,
        "category": "structured_output",
        "credit": (
            "Backport of vllm#40962 (OPEN). NARROWER "
            "alternative to our existing P62 (vllm#36138 broader pipeline-"
            "level fix). MUTUALLY EXCLUSIVE with P62 — both patch the same "
            "`should_advance` block in scheduler.update_from_output(). "
            "Apply check enforces P62 OFF requirement; SKIPS otherwise. "
            "PN58 modifies ONLY commit-time validation, doesn't touch "
            "bitmask/draft validation. Author warns: significant perf drop "
            "with multi-token reasoning markers (per-token boundary scan "
            "expensive). Engineering tradeoff: P62 = more correct (per-pos "
            "grammar masks), more invasive; PN58 = less correct in edge "
            "cases (commit-time only), cheaper hot-path. Multi-file "
            "(envs.py + abs_reasoning_parsers.py + basic_parsers.py + "
            "v1/structured_output/__init__.py + v1/core/sched/scheduler.py "
            "= 5 files, 6 sub-patches). Default OFF; current Genesis PROD "
            "uses P62 (broader). Enable PN58 only after measuring P62 perf "
            "hit on YOUR specific reasoning parser."
        ),
        "upstream_pr": 40962,
        "upstream_pr_relationship": "backport",
        "applies_to": {},
        "conflicts_with": ["P62"],
        "apply_module": "sndr.engines.vllm.patches.reasoning.pn58_spec_reasoning_boundary",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P107": {
        "title": "MTP truncation detector at reasoning→tool_call boundary (vllm#41467)",
        "tier": "community",
        "family": "serving",
        "env_flag": "GENESIS_ENABLE_P107_MTP_TRUNCATION_DETECTOR",
        "default_on": False,
        "category": "structured_output",
        "credit": (
            "Backport of vllm#41467 (ToastyTheBot, OPEN). With MTP K>=1 + "
            "tools + reasoning_parser a rare (~0.25% per author measurement "
            "on Qwen3.6 27B-FP8) condition arises: the model emits EOS at "
            "the reasoning->tool_call boundary. finish_reason=stop, neither "
            "tool_calls nor content. A defensive guard in "
            "chat_completion_stream_generator detects the combo and raises "
            "GenerationError (retryable) — the client retries instead of "
            "silent stop. Author explicitly references our P58/P59/P60/P61/"
            "P64 path. EXACT match for our PROD config (27B Lorbus + MTP K=3 "
            "+ tools). Defensive safety-net, not a root-cause fix. Default "
            "OFF until live verify tool-call sweep on 27B PROD."
        ),
        "upstream_pr": 41467,
        "upstream_pr_relationship": "backport",
        "applies_to": {
            # [Genesis pin-gate 2026-05-11] Defensive safety-net (default
            # OFF). Validated dev9 → dev93. Self-retires when #41467 merges.
            "vllm_version_range": (">=0.20.0", "<0.24.0"),
        },
        "apply_module": "sndr.engines.vllm.patches.serving.p107_mtp_truncation_detector",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    # PN56 (Qwen3Coder XML parse fallback, vllm#41466) was consolidated into
    # the P64 entry on 2026-06-20 — all three qwen3coder parser patches
    # (P64 + P61c + PN56) share one apply_module
    # (p64_p61c_pn56_qwen3coder_consolidated). PN56's enable flag
    # GENESIS_ENABLE_PN56_QWEN3CODER_XML_FALLBACK is retained as an
    # env_flag_alias on P64 so existing YAML opt-ins keep engaging the merged
    # module.
    "PN57": {
        "title": "TurboQuant centroids disk-persistent cache (vllm#41418-inspired)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_PN57_TQ_CENTROIDS_DISK_CACHE",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Inspired by vllm#41418 (TheTom, OPEN). Upstream PR pre-bakes 9 "
            "(d,bits) centroid tables inline (~1500 LOC of constants). "
            "Genesis approach: disk-persistent cache `~/.cache/genesis/"
            "turboquant_centroids.pkl` instead of inline constants. "
            "Lloyd-Max solver fully deterministic given (d,bits) → bit-"
            "identical to upstream pre-baked tables. Cold start: 200ms × N "
            "first-time shapes per fresh container. Subsequent boots / "
            "worker restarts: instant lookup. Saves ~205ms per worker on "
            "k8v4 path. Atomic write (tempfile+rename), defensive fall-"
            "through to solver on any cache failure. Default OFF until "
            "live-verified cold-start savings."
        ),
        "upstream_pr": 41418,
        "upstream_pr_relationship": "backport",
        "applies_to": {},
        # Mutually exclusive with PN26 — both implement vllm#41418 by
        # rewriting the same get_centroids() in centroids.py (PN57 = disk
        # cache, PN26 = pre-baked tables). See PN26.conflicts_with
        # (cross-patch lint, deep-audit 2026-06-14 #2).
        "conflicts_with": ["PN26"],
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.pn57_tq_centroids_disk_cache",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN282": {
        # dev491 drift note (2026-06-16): dev491's rejection_sample() grew
        # use_fp64_gumbel (vllm#43150), passed unconditionally by RejectionSampler
        # .forward (rejection_sampler.py:182-184), after synthetic_mode/synthetic_
        # conditional_rates. The old explicit-signature wrapper TypeError'd EVERY
        # spec-decode step once enabled. FIXED 2026-06-16: PN282 (and the PN248
        # sibling) now forward (*args, **kwargs) transparently and read the
        # side-channel via inspect.signature(original).bind() — signature-agnostic
        # and forward-proof, so NO version-cap is needed. Regression test:
        # tests/unit/integrations/observability/test_pn282_pn248_forward_proof.py.
        "title": "Spec-decode acceptance proxy metric (Prometheus, non-dispatcher boot patch)",
        "tier": "community",
        "family": "observability",
        "env_flag": "SNDR_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC",
        "default_on": False,
        "category": "observability",
        "implementation_status": "marker_only",
        "source": "genesis_original",
        "apply_module": None,
        "lifecycle": "coordinator",
        "credit": (
            "STAGE_6_HARDENING.2C registration (2026-05-28). PN282 is a "
            "production sibling of PN248's debug-log trace: wraps "
            "rejection_sample and emits sndr_spec_decode_* Prometheus "
            "series on the worker's existing /metrics endpoint. "
            "Boot-applied directly from sndr_core/__init__.py (matches "
            "PN248 sibling pattern), not via the dispatcher pipeline — "
            "hence apply_module=None and lifecycle=coordinator. Canonical "
            "env name is SNDR_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC; the "
            "legacy alias GENESIS_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC is "
            "accepted with a one-shot deprecation warning. Registering "
            "this coordinator entry closes the Stage-6 orphan-flag gap "
            "and makes the metric discoverable via standard tooling "
            "(`sndr explain PN282`, generated patch docs)."
        ),
        "upstream_pr": None,
        "requires_patches": [],
        "conflicts_with": [],
        # PN248 was a debug-log trace predecessor referenced in the credit
        # text. It was never promoted to a registry entry — drop the
        # composes_with reference (K.1.R.R.1 cleanup 2026-05-28). The
        # text reference in the credit stays as historical context.
        "composes_with": [],
        "applies_to": {"model_arch": ["*"]},
    },
    "PN283": {
        "title": "vLLM v1 multiprocess Prometheus directory bootstrap (non-dispatcher boot patch)",
        "tier": "community",
        "family": "observability",
        "env_flag": "SNDR_ENABLE_PN283_PROC_BRIDGE",
        "default_on": False,
        "category": "observability",
        "implementation_status": "marker_only",
        "source": "genesis_original",
        "apply_module": None,
        "lifecycle": "coordinator",
        "credit": (
            "Genesis-original — Sandermage; PN283 / 2026-05-20. Sibling "
            "of PN282 (same coordinator pattern): boots the multiprocess "
            "Prometheus directory referenced by PROMETHEUS_MULTIPROC_DIR "
            "before any patch hook runs, so PN282's worker-process "
            "Counters have a writable value-file dir by the time "
            "prometheus_client opens its first Counter/Gauge. Boot-applied "
            "directly from sndr_core/__init__.py (matches PN248/PN282 "
            "coordinator pattern), not via the dispatcher pipeline — "
            "hence apply_module=None and lifecycle=coordinator. Canonical "
            "env name SNDR_ENABLE_PN283_PROC_BRIDGE (mirrors PN282 "
            "SNDR_ENABLE_* naming for non-dispatcher coordinator boot "
            "patches). Registering this coordinator entry closes the "
            "orphan-flag gap surfaced by audit_config_keys / "
            "audit_v2_env_keys after the chat-K3 profile promotion "
            "declared the env in gemma4-31b-tq-mtp-{chat-k3,structured-k4} "
            "profiles."
        ),
        "upstream_pr": None,
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["PN282"],
        "applies_to": {"model_arch": ["*"]},
    },
    "PN122": {  # renamed 2026-05-14 from SPRINT26_CG_DISPATCH_TRACE — long ID violated P[N]?\d+ convention + auto-derivation
        "title": "Sprint 2.6 v2 — CUDA graph dispatch trace wire-in (formerly SPRINT26_CG_DISPATCH_TRACE)",
        "tier": "community",
        "family": "observability",  # 2026-05-11 audit fix: was "worker" but file lives under integrations/observability/ + category is observability
        "env_flag": "GENESIS_ENABLE_PN122_CG_DISPATCH_TRACE",  # legacy GENESIS_ENABLE_SPRINT26_CG_DISPATCH_TRACE accepted for 1 release
        "default_on": False,
        "category": "observability",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": (
            "sndr.engines.vllm.patches.observability."
            "pn122_sprint26_cudagraph_dispatch_trace"
        ),
        "lifecycle": "experimental",
        "experimental_note": (
            "Text-patch into gpu_model_runner.py decoder dispatch site "
            "to call record_dispatch(matched). Default OFF — opt in via "
            "this flag PLUS the runtime GENESIS_CUDAGRAPH_DISPATCH_TRACE=1 "
            "to actually emit per-N-requests summary lines. Wave 6 PN16 "
            "V1 regression root-cause: dispatch mismatch invisible to "
            "wall_TPS averaging."
        ),
        "credit": (
            "Genesis-original — Sandermage. Sprint 2.6 v2 closes the "
            "Sprint 2.6 v1 'instrumentation only, no wire-in' gap."
        ),
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {},
    },
    "PN132": {
        "title": "Triton top-k/top-p contiguous logits fix (backport vllm#42739)",
        "tier": "community",
        "family": "compile_safety",
        "env_flag": "GENESIS_ENABLE_PN132_TOPK_TOPP_CONTIGUOUS",
        "default_on": False,
        "category": "stability",
        "implementation_status": "full",
        "source": "vllm_pr_backport",
        "apply_module": (
            "sndr.engines.vllm.patches.compile_safety."
            "pn132_triton_topk_topp_contiguous"
        ),
        "lifecycle": "retired",
        "retired_waiver": True,
        "experimental_note": (
            "Backport vllm#42739 (merged 2026-05-23, commit "
            "d19db10974587335ec3a37e0a424abb57430574e). RETIRED 2026-05-30 "
            "after iron-rule-#11 deep-diff against the upstream merge "
            "(verified live on pin 626fa9bb 2026-05-28 via the running "
            "gemma4 container, function source inspection). VERDICT: "
            "upstream solves the SAME bug at the ROOT — the Triton "
            "kernel itself now takes `LOGITS_STRIDE_0` and computes row "
            "pointers via `LOGITS + row_id * LOGITS_STRIDE_0` (stride-aware "
            "addressing), with a contiguous-temporary fallback for "
            "stride(1) != 1 layouts. PN132's wrapper added a "
            "`.contiguous()` guarantee at the Python boundary — a "
            "WORKAROUND, not the root fix; the kernel-level fix in #42739 "
            "is strictly superior. PN132 also has signature drift: our "
            "wrapper exposes `(logits, k, p)` but upstream's post-merge "
            "signature is `(logits, k, p, mask_value=-inf)` — enabling "
            "PN132 on 626fa9bb+ would drop the `mask_value` kwarg if a "
            "caller passes it positionally or as a kwarg. Defense-in-depth "
            "rationale also moot: VLLM_USE_FLASHINFER_SAMPLER=1 keeps "
            "Triton on the fallback path, AND the upstream fix removes "
            "the underlying bug we were defending against. PN132 was "
            "default_off so no PROD impact from the retire."
        ),
        "credit": "Backport vllm#42739 by Sandermage 2026-05-15. Retired 2026-05-30 (upstream merged + root-cause fix is strictly better than our wrapper workaround; signature drift detected on post-merge pin).",
        "upstream_pr": 42739,
        "upstream_pr_relationship": "backport",
        "superseded_by": "vllm#42739 (merged 2026-05-23, commit d19db10974587335ec3a37e0a424abb57430574e)",
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {
            "vllm_version_range": (">=0.20.0", "<0.21.1rc0+g626fa9bba5"),
        },
    },
    "PN133": {
        "title": "MTP scheduler empty-output accounting fix (backport vllm#42722)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_PN133_MTP_EMPTY_OUTPUT_FIX",
        "default_on": False,
        "category": "stability",
        "implementation_status": "full",
        "source": "vllm_pr_backport",
        "apply_module": (
            "sndr.engines.vllm.patches.spec_decode."
            "pn133_mtp_scheduler_empty_output"
        ),
        "lifecycle": "experimental",
        "experimental_note": (
            "Backport vllm#42722 (OPEN). Fixes permanently-stuck request "
            "in MTP/spec-decode when model_runner returns empty "
            "generated_token_ids (request abortion, async race, OOM "
            "partial output). Pre-fix: scheduler doesn't account "
            "scheduled draft tokens as rejected → num_computed_tokens "
            "stays caught up → scheduler stops issuing work for the "
            "unfinished request. Also fixes the pre-existing crash via "
            "len([])-1 = -1 → Prometheus counter ValueError."
        ),
        "credit": "Backport vllm#42722 by Sandermage 2026-05-15.",
        "upstream_pr": 42722,
        "upstream_pr_relationship": "backport",
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {
            "vllm_version_range": (">=0.20.0", "<0.23.0"),
        },
    },
    "PN134": {
        "title": "torch.compile fullgraph patch for PyTorch 2.11 (backport vllm#42686) — BENCH-VALIDATED REGRESSOR, DO NOT ENABLE",
        "tier": "community",
        "family": "compile_safety",
        "env_flag": "GENESIS_ENABLE_PN134_TORCH_COMPILE_FULLGRAPH_211",
        "default_on": False,
        "category": "perf_hotfix",
        "implementation_status": "full",
        "source": "vllm_pr_backport",
        "apply_module": (
            "sndr.engines.vllm.patches.compile_safety."
            "pn134_torch_compile_fullgraph_211"
        ),
        "lifecycle": "retired",
        "retired_waiver": True,
        "retired_reason": (
            "Bench 2026-05-15 on Qwen3.6-35B-A3B-FP8 (dev371 nightly-bf610c2f, "
            "2× A5000 TP=2): enabling PN134 caused -25% TPS regression "
            "(211.38 → 158.0), TPOT +37%, TTFT 85→196 ms. Monkey-patching "
            "torch._inductor.ir.StorageBox.should_realize_on_reuse affects "
            "the ENTIRE model compilation graph, not just attention path — "
            "the size-aware cost model materializes too many intermediates "
            "for hybrid_gdn_moe layout, blowing the Inductor cache and "
            "forcing recompilation on every batch shape variant. "
            "DO NOT ENABLE on this model class. Module kept on disk for "
            "future investigation on dense-attention models where the "
            "cost model may behave correctly. See plan section 12.x for "
            "regression report."
        ),
        "experimental_note": (
            "RETIRED 2026-05-15 — bench-validated regressor. Original "
            "design backport vllm#42686 (OPEN), closes vLLM issue #27828. "
            "Patches torch._inductor.ir.StorageBox.should_realize_on_reuse "
            "with a size-aware cost model for PyTorch 2.11 (fix landed in "
            "torch 2.12). Theory: without the fix Inductor inlines residual "
            "into fused_add_rms_norm every time → cascade re-computation. "
            "Reality on hybrid_gdn_moe: -25% TPS regression."
        ),
        "credit": "Backport vllm#42686 (pytorch#176994 simplified) by Sandermage 2026-05-15. Retired same day after bench regression.",
        "upstream_pr": 42686,
        "upstream_pr_relationship": "backport",
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {
            "vllm_version_range": (">=0.20.0", "<0.22.0"),
        },
    },
    "PN128": {
        "title": "Spec-decode helper kernel warmup (backport vllm#41481, 4 eagle kernels)",
        "tier": "community",
        "family": "compile_safety",
        "env_flag": "GENESIS_ENABLE_PN128_SPEC_DECODE_WARMUP",
        "default_on": False,
        "category": "perf_hotfix",
        "implementation_status": "full",
        "source": "vllm_pr_backport",
        "apply_module": (
            "sndr.engines.vllm.patches.compile_safety."
            "pn128_spec_decode_helper_warmup"
        ),
        "lifecycle": "experimental",
        "experimental_note": (
            "Genesis backport of vllm-project/vllm#41481 (OPEN). Closes "
            "4 of 8 JIT spikes on the first user request: "
            "eagle_prepare_next_token_padded_kernel, "
            "eagle_prepare_inputs_padded_kernel, "
            "copy_and_expand_eagle_inputs_kernel, "
            "eagle_step_slot_mapping_metadata_kernel. Wraps "
            "Worker.compile_or_warm_up_model and, after the original "
            "warmup, invokes 4 dummy Triton kernels with synthetic shapes "
            "(next_power_of_2(num_spec_tokens + 1)). Auto-skip on "
            "V2_MODEL_RUNNER=1 and enforce_eager=True. Issue #39790 H100 "
            "repro showed a 25x first-request regression pre-fix."
        ),
        "credit": "Backport of vllm-project/vllm#41481 by Sandermage 2026-05-15.",
        "upstream_pr": 41481,
        "upstream_pr_relationship": "backport",
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {
            "model_arch": [
                "Qwen3_5ForConditionalGeneration",
                "Qwen3_5MoeForConditionalGeneration",
                "Qwen3NextForCausalLM",
                "Qwen3MoeForCausalLM",
            ],
            "vllm_version_range": (">=0.20.0", "<0.24.0"),
        },
    },
    "PN129": {
        "title": "V1 slot mapping kernel warmup (backport vllm#42165, 1 kernel + do_not_specialize)",
        "tier": "community",
        "family": "compile_safety",
        "env_flag": "GENESIS_ENABLE_PN129_SLOT_MAPPING_WARMUP",
        "default_on": False,
        "category": "perf_hotfix",
        "implementation_status": "full",
        "source": "vllm_pr_backport",
        "apply_module": (
            "sndr.engines.vllm.patches.compile_safety."
            "pn129_slot_mapping_warmup"
        ),
        "lifecycle": "experimental",
        "experimental_note": (
            "Genesis backport of vllm-project/vllm#42165 (OPEN). Closes "
            "the _compute_slot_mapping_kernel JIT spike + (attempts) "
            "do_not_specialize='num_tokens' via the private Triton API. "
            "If do_not_specialize injection does not work on our Triton "
            "version, only the warmup hook remains — the kernel JITs at "
            "boot instead of on the first user request. Best-effort fix."
        ),
        "credit": "Backport of vllm-project/vllm#42165 by Sandermage 2026-05-15.",
        "upstream_pr": 42165,
        "upstream_pr_relationship": "backport",
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {
            "vllm_version_range": (">=0.20.0", "<0.24.0"),
        },
    },
    "PN130": {
        "title": "TurboQuant decode kernel warmup (backport vllm#42215, 1 kernel + workspace prealloc)",
        "tier": "community",
        "family": "compile_safety",
        "env_flag": "GENESIS_ENABLE_PN130_TQ_DECODE_WARMUP",
        "default_on": False,
        "category": "perf_hotfix",
        "implementation_status": "full",
        "source": "vllm_pr_backport",
        "apply_module": (
            "sndr.engines.vllm.patches.compile_safety."
            "pn130_turboquant_decode_warmup"
        ),
        "lifecycle": "experimental",
        "experimental_note": (
            "Genesis backport of vllm-project/vllm#42215 (OPEN). Closes "
            "the _tq_grouped_decode_stage1 JIT spike and prevents workspace "
            "re-allocation after lock_workspace(). Iterates the model's "
            "Attention layers, dedupes by config-tuple, calls "
            "impl._decode_attention() with synthetic tensors. Auto-skip "
            "when kv_cache_dtype != turboquant_*."
        ),
        "credit": "Backport of vllm-project/vllm#42215 by Sandermage 2026-05-15.",
        "upstream_pr": 42215,
        "upstream_pr_relationship": "backport",
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {
            "vllm_version_range": (">=0.20.0", "<0.24.0"),
        },
    },
    "PN127": {
        "title": "Qwen 3.5/3.6 enhanced chat-template auto-install (closes club-3090#53/#72)",
        "tier": "community",
        "family": "serving",
        "env_flag": "GENESIS_ENABLE_PN127_AUTO_CHAT_TEMPLATE",
        "default_on": False,
        "category": "stability",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": (
            "sndr.engines.vllm.patches.serving."
            "pn127_chat_template_qwen36"
        ),
        "lifecycle": "experimental",
        "experimental_note": (
            "Genesis-original 2026-05-15. Closes operator pain: the "
            "enhanced chat-template for Qwen 3.5/3.6 hybrid_gdn_moe "
            "(interleaved-thinking + XML tool_call) previously lived in "
            "HF repos (froggeric / Sandermage / club-3090) and the "
            "operator had to know where to look and copy the .jinja by "
            "hand. PN127 bakes the enhanced template as a Genesis asset "
            "(sndr/assets/chat_templates/qwen3.6_enhanced.jinja) "
            "and at apply() copies it into a writable location "
            "(/tmp/genesis/chat_templates/ or GENESIS_CHAT_TEMPLATE_DIR). "
            "The operator receives the canonical path through a log line "
            "and launches vllm with --chat-template <path>. Closes 7 bugs "
            "in the default template: empty <think></think>, </thinking> "
            "hallucination, unclosed think before tool_call, no-user-query "
            "crash, developer role, multi-turn tool-call SSE deadlock "
            "(club-3090#72), think->tool_call boundary truncation."
        ),
        "credit": (
            "Genesis-original — Sandermage. Combines Sandermage v7.62 "
            "interleaved-thinking template + 7 froggeric fixes + "
            "club-3090 live-verify (30/30 tool regression PASS)."
        ),
        "upstream_pr": None,
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {
            "model_arch": [
                "Qwen3_5ForConditionalGeneration",
                "Qwen3_5MoeForConditionalGeneration",
                "Qwen3NextForCausalLM",
                "Qwen3MoeForCausalLM",
            ],
            "vllm_version_range": (">=0.20.0", "<0.24.0"),
        },
    },
    "PN126": {
        "title": "V1 decode + spec-decode kernel warmup orchestrator (fixes JIT spikes on first request)",
        "tier": "community",
        "family": "compile_safety",
        "env_flag": "GENESIS_ENABLE_PN126_V1_DECODE_WARMUP",
        "default_on": False,  # bench-gate required
        "category": "perf_hotfix",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": (
            "sndr.engines.vllm.patches.compile_safety."
            "pn126_v1_decode_kernel_warmup"
        ),
        "lifecycle": "experimental",
        "experimental_note": (
            "Genesis-original 2026-05-15. Closes V1 vs V2 model runner gap: "
            "V2 calls warmup_kernels() at end of compile_or_warm_up_model "
            "which exercises decode + spec-decode + TQ kernels at boot; "
            "V1 only runs a sampler-only dummy_run with cudagraph_mode=NONE. "
            "Result on V1: vLLM jit_monitor warns 8-10 kernel JIT events "
            "on FIRST user request, causing TTFT spike of 5-25 s. "
            "PN126 wraps Worker.compile_or_warm_up_model to add 2 extra "
            "_dummy_run() passes after the original sampler warmup: "
            "(Pass 1) prefill at max_num_batched_tokens with PIECEWISE "
            "cudagraph dispatch — covers Mamba causal_conv1d at large T + "
            "prefill attention; (Pass 2) uniform decode at "
            "max_num_seqs × (1 + num_speculative_tokens) with FULL "
            "cudagraph dispatch — covers decode attention + TQ kernels + "
            "spec-decode draft prep kernels. Expected TTFT CV drop "
            "30%→~15% post-warmup. Auto-skips when VLLM_USE_V2_MODEL_RUNNER=1 "
            "(V2 native) or enforce_eager=True (no cudagraphs). "
            "Default OFF until bench: 35B 8K TPS unchanged, TTFT CV < 20%, "
            "boot time +3-10s acceptable."
        ),
        "credit": (
            "Genesis-original — Sandermage 2026-05-15. Pattern from V2 "
            "warmup_kernels (vllm/v1/worker/gpu/warmup.py in dev338+) "
            "adapted to V1 model runner via wrapped "
            "compile_or_warm_up_model. Related: PR #39822 (SSD kernel "
            "warmup at profile_run — landed upstream but only covers "
            "MambaMixer2 path, not GDN+spec_decode+TQ kernels that fire "
            "on first user request)."
        ),
        "upstream_pr": None,
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {
            # Hybrid models with spec_decode benefit most. Dense models
            # (Llama, Mistral) still benefit from prefill+decode warmup
            # but the gap there is smaller (no Mamba JIT, no TQ).
            "model_arch": [
                "Qwen3_5ForConditionalGeneration",
                "Qwen3_5MoeForConditionalGeneration",
                "Qwen3NextForCausalLM",
                "Qwen3MoeForCausalLM",
            ],
            "vllm_version_range": (">=0.20.0", "<0.24.0"),
        },
    },
    "PN125": {
        "title": "Hybrid Qwen3.5/3.6 FULL_AND_PIECEWISE cudagraph_mode (redundant on dev338+)",
        "tier": "community",
        "family": "compile_safety",
        "env_flag": "GENESIS_ENABLE_PN125_HYBRID_FULL_AND_PIECEWISE",
        "default_on": False,  # measured no-op on dev338
        "category": "perf_hotfix",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": (
            "sndr.engines.vllm.patches.compile_safety."
            "pn125_hybrid_full_and_piecewise"
        ),
        "lifecycle": "experimental",
        "experimental_note": (
            "POST-BENCH NOTE (2026-05-15): on dev338 PN125 is empirically "
            "REDUNDANT. The vllm v1 default cudagraph resolver in "
            "config/compilation.py already sets FULL_AND_PIECEWISE when "
            "splitting_ops_contain_attention(); for hybrid_gdn_moe this "
            "branch fires unconditionally. Bench on 2× A5000 + 35B-A3B-FP8 "
            "+ TQ k8v4 + MTP K=3 (n=25, 5 runs × 5 prompts × 384 tok):\n"
            "  - PN125 OFF: 206.26 TPS / CV 5.5%\n"
            "  - PN125 ON:  206.23 TPS / CV 5.5%\n"
            "Delta within noise band (CV exceeds the 0.01% TPS gap by 500×).\n"
            "ORIGINAL motivation (likely incorrect for dev338): "
            "Qwen3_5ForConditionalGenerationConfig only updates "
            "mamba_ssm_cache_dtype and never calls "
            "MambaModelConfig.verify_and_update_config — assumed this "
            "leaked the FULL_AND_PIECEWISE setup. In practice the v1 "
            "resolver runs LATER in init and re-applies FULL_AND_PIECEWISE "
            "via the splitting-ops path regardless. PN125 monkey-patches "
            "verify_and_update_config to also invoke MambaModelConfig — "
            "essentially a NO-OP given the v1 resolver. Retained as "
            "audit-trail registry entry + safety net in case future vllm "
            "pins change the resolver default (auto-retire when "
            "splitting_ops_contain_attention drops the FULL branch). "
            "Default OFF until a bench shows measurable benefit on some "
            "pin/workload combination."
        ),
        "credit": (
            "Genesis-original 2026-05-15 — Sandermage. Source: "
            "https://pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm/ "
            "(PyTorch blog claim of up to 91% throughput gain on hybrid "
            "Mamba models — measured 0% on our 35B / dev338 stack since "
            "vllm v1 default already engages FULL_AND_PIECEWISE)."
        ),
        "upstream_pr": None,
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {
            "model_arch": [
                "Qwen3_5ForConditionalGeneration",
                "Qwen3_5MoeForConditionalGeneration",
                "Qwen3NextForCausalLM",
                "Qwen3MoeForCausalLM",
            ],
            # Valid for pins where MambaModelConfig.verify_and_update_config exists.
            "vllm_version_range": (">=0.20.0", "<0.22.0"),
        },
    },
    "PN96b": {
        "title": "Persistent Marlin MoE workspace (Wave 9 dev209 perf-restore)",
        "tier": "community",
        "family": "moe",
        "env_flag": "GENESIS_ENABLE_PN96B",  # renamed from PN96 2026-05-14 to avoid collision with kv_cache/PN96 emergency demote
        "default_on": True,
        "category": "perf_hotfix",
        "credit": (
            "Genesis-original 2026-05-12 (Wave 9 dev209 35B regression "
            "RCA). Upstream `experts/marlin_moe.py::MarlinExperts.apply` "
            "calls `fused_marlin_moe(...)` without passing the `workspace` "
            "kwarg, so `_fused_marlin_moe` allocates fresh via "
            "`marlin_make_workspace_new(device, 4)` on every MoE call. "
            "PN96 wraps MarlinExperts.apply to cache a persistent "
            "workspace per-instance + monkey-patches fused_marlin_moe to "
            "honor a thread-local default when workspace=None. Target: "
            "recover part of the -2.82% TPS / +2.86% TPOT 35B A3B-FP8 "
            "regression seen between dev93 and dev209. NO-OP on "
            "non-Marlin paths (27B hybrid GDN+Mamba INT4 unaffected). "
            "Auto-skips on dev93-era layout where experts/marlin_moe.py "
            "doesn't exist. Self-retires when upstream adds workspace= "
            "to the apply→fused_marlin_moe call site."
        ),
        "upstream_pr": None,  # not yet proposed upstream
        "applies_to": {},
        "apply_module": "sndr.engines.vllm.patches.moe.pn96b_marlin_persistent_workspace",
        "lifecycle": "experimental",
        "experimental_note": (
            "Runtime hook (no _make_patcher). default_on=True for 35B "
            "PROD. Recovers a portion of the dev209 MoE regression "
            "(exact gain TBD by post-deploy A/B bench). Composes with "
            "P17/P18 (per-SM tuning) + P22/P38 (TQ workspace) — "
            "different optimization vectors, no aliasing."
        ),
        "conflicts_with": [],
        # P18 is not a separate registry entry — it's the companion "B"
        # variant of P17 (Marlin MoE per-SM tuning) handled inline.
        "composes_with": ["P17", "P22", "P38"],
        "implementation_status": "full",
    },
    "PN95": {
        "title": "PN95 — Tier-aware KV cache + CPU offload + boot-time expansion (Path C, club-3090 #58)",
        "tier": "community",
        "family": "kv_cache",
        "env_flag": "GENESIS_ENABLE_PN95_TIER_AWARE_CACHE",
        "default_on": False,
        "category": "kv_cache",
        "implementation_status": "partial",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.kv_cache.pn95_tier_aware_cache",
        "lifecycle": "experimental",
        "experimental_note": (
            "Eleven text-patch anchors wired into vLLM v1 KV cache hot path. "
            "Phase 1 (anchors 1-3): cache_blocks admit, block_pool touch, "
            "KVCacheManager mamba classifier + lazy TierManager init. "
            "Phase 2 (anchors 4-5): worker register_kv_caches + Scheduler tick. "
            "Phase 4 (anchors 6-8): prefix cache extension — block_pool register, "
            "demote-on-evict, promote-on-miss. Phase 5 partial (anchors 9, 11, 12): "
            "boot-time available_memory inflation, BlockPool metadata side-table, "
            "get_new_blocks virtual materialization. "
            "Phase 5 anchor #10 (KVCacheTensor physical-allocation cap via "
            "pn95_physical_num_blocks_cap()) helper exists in _pn95_runtime.py "
            "but text-patch wire-in is pending — boot-time virtualization needs "
            "live GPU validation before flipping that switch. CPU slab is "
            "torch.empty(pin_memory=True) when torch+CUDA available; "
            "MambaSpec groups are filtered from demote candidates."
        ),
        "credit": (
            "Genesis-original implementation. Addresses club-3090 #58 — Mamba "
            "SSM state lives outside the KV pool, so all upstream CPU-offload "
            "paths (vLLM cpu-offload-gb, SimpleCPUOffloadConnector, LMCache, "
            "SGLang HiCache) crash on hybrid-GDN models. PN95 filters MambaSpec "
            "groups out of demote candidates and drains MM/vision pages first "
            "(image tokens carry lower attention re-use than text-prefix pages). "
            "Reuses PN91 eviction policy ABC (LRU/2Q/ARC) per tier. Coordinates "
            "with P83 and P85 in the same vLLM source files."
        ),
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {},
        "related_upstream_prs": [],
    },
    "PN90": {
        # Drift cap: PN90 self-skips on dev338+ via its own
        # _PROPOSER_DRIFT_MARKERS, so the upper bound silences the
        # spurious version-mismatch WARN that would stack on top of the
        # drift-marker skip. NOT a supersession boundary — vllm#40269 is
        # a DIFFERENT-APPROACH (related_not_superseding) landing, not a
        # backport of PN90 (see credit + upstream_pr_relationship below).
        "vllm_version_range": (">=0.20.0", "<0.22.0"),
        "title": "Probabilistic draft rejection (vllm#40269 backport) — propagate draft_probs to verifier",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_PN90_PROBABILISTIC_DRAFT",
        "default_on": False,
        # Reconciled 2026-06-19: lifecycle restored to "experimental"
        # (was erroneously flipped to "retired" + superseded_by="vllm#40269"
        # by the 4c8d992b P3-reverify sweep, whose own commit message
        # lists PN396 as its ONLY retire and does NOT mention PN90). The
        # flip contradicted iron rule #11 verdict (c) documented at length
        # in this entry's `credit` ("KEEP PN90 ... lifecycle stays
        # experimental ... Do NOT retire") AND the
        # `upstream_pr_relationship: related_not_superseding` field below.
        # vllm#40269 implements the same goal via a DIFFERENT approach
        # (config-knob `draft_sample_method=probabilistic`), empirically
        # rejected on our shape — so this is not a supersession. The
        # false-positive lock test (test_audit_upstream_status
        # TestLiveRegistryFalsePositiveLock) correctly demanded
        # NEEDS-DEEP-PARITY, which the retired flip was masking as
        # ALREADY-RETIRED.
        "lifecycle": "experimental",
        "implementation_status": "full",
        "category": "spec_decode",
        "apply_module": "sndr.engines.vllm.patches.spec_decode.pn90_probabilistic_draft_rejection",
        "credit": (
            "Backport of vllm-project/vllm#40269 (MERGED upstream 2026-05-14). "
            "Stock vLLM dev93 passed literal `None` for draft_probs in "
            "gpu_model_runner.py:3416 → rejection_sampler fell back to "
            "argmax-or-bonus rule. PN90 captures softmax probs in "
            "_greedy_sample (proposer side), accumulates per K-step, "
            "stacks into [total_drafts, vocab] 2D layout, and feeds to "
            "rejection_sampler. Probabilistic acceptance rule "
            "min(1, target_prob/draft_prob) is mathematically tighter "
            "and accepts more borderline tokens. Expected: +0.5-2% "
            "acceptance rate on MTP K=3 workloads. "
            "Genesis contribution: full text-patch implementation across "
            "llm_base_proposer.py (3 anchors) + gpu_model_runner.py "
            "(1 anchor), MultiFilePatchTransaction atomicity, drift "
            "markers, env-gated for safe back-compat. "
            "POST-MERGE NOTE (2026-05-15): upstream landed the same "
            "feature via `speculative_config.draft_sample_method = "
            "\"probabilistic\"` + `take_last_draft_probs()`. PN90 drift "
            "markers detect upstream-native symbols and self-skip on "
            "pins ≥dev338. Prefer the native path: set draft_sample_method "
            "(see commit 0e877eaf exposing the knob on SpecDecodeConfig) "
            "instead of GENESIS_ENABLE_PN90_*. "
            "[Phase 3D 2026-05-22] Live-network verification on dev371 "
            "(canonical pin bf610c2f56764e1b30bc6065f4ceace3d6e59036): "
            "(1) PN90 merge commit f51f6844f99aa38547f1fcae6516da31997bda50 "
            "is IN dev371 baseline — `gh api .../compare/f51f6844f...bf610c2f5` "
            "shows dev371 is 38 commits ahead of the PN90 merge, behind_by=0. "
            "(2) All four upstream-native symbols are present in dev371 "
            "source at vllm/v1/spec_decode/llm_base_proposer.py: "
            "`take_last_draft_probs` (×1), `_enable_probabilistic_draft_probs` "
            "(×3), `_last_draft_probs` (×6), `draft_sample_method` (×1) — "
            "verified by direct `gh api .../contents` fetch. "
            "(3) PN90's drift markers (`_PROPOSER_DRIFT_MARKERS`, "
            "`_RUNNER_DRIFT_MARKERS`) include these upstream symbols, so "
            "apply() returns `\"skipped\"` at boot on dev371 — PN90 is "
            "functionally inert on the current pin even when "
            "GENESIS_ENABLE_PN90_PROBABILISTIC_DRAFT=1 is set. "
            "(4) The upstream-native knobs `draft_sample_method: "
            "probabilistic` and `rejection_sample_method: standard` are "
            "INTENTIONALLY commented out in the qwen3.6-35b-a3b-fp8.yaml "
            "ModelDef (and siblings) with note `→ regression on our shape` "
            "— empirical bench rejected upstream's path on the current "
            "pin/hardware combo. "
            "(5) Net effect on dev371 PROD: neither PN90's text-patch nor "
            "upstream's native probabilistic path is active. Spec-decode "
            "rejection falls back to the original argmax-or-bonus rule. "
            "(6) The +1.4-2.8% accept_rate measurement quoted above is "
            "HISTORICAL evidence from dev93 / dev209 (pre-upstream-merge) "
            "and is not reproducible on dev371. "
            "Decision (verdict c per iron rule #11 — different approach, "
            "same goal): KEEP PN90 in registry. Do NOT retire because: "
            "(i) upstream implementation is structurally different (config-knob "
            "vs Genesis text-patch), so a clean byte-identical retire claim "
            "would be inaccurate; (ii) upstream variant has been empirically "
            "rejected for our shape, so retiring implies a fallback path that "
            "operator has chosen not to use; (iii) older pins (pre-dev338) "
            "where the upstream symbols are absent still benefit from PN90's "
            "text-patch path — retiring would surprise any operator running "
            "an older pin. Module stays at "
            "integrations/spec_decode/pn90_probabilistic_draft_rejection.py; "
            "lifecycle stays experimental; ModelDef YAMLs untouched. The "
            "patch self-skips correctly on dev371+ via its own drift-marker "
            "mechanism — registry state matches runtime reality."
        ),
        "upstream_pr": 40269,
        "upstream_pr_relationship": "related_not_superseding",
        "applies_to": {
            # Only fires for MTP/Eagle/DFlash drafters that go through
            # llm_base_proposer._greedy_sample. ngram drafter has its
            # own propose path that doesn't touch this code.

            # [Genesis pin-gate] PN90 anchors live in llm_base_proposer.py
            # (3 sites) + gpu_model_runner.py (1 site). Validated against:
            #   - dev93+g51f22dcfd (2026-05-07, PROD baseline, +7.4% TPS bench)
            # On <0.20.2 the anchor target lines do not yet exist (different
            # rejection_sampler path). On future pins, drift detector
            # auto-skips if anchors moved — pin-gate prevents the attempt
            # when version is known-incompatible.
            # v11.3.0 BUG #14 fix: upper bound bumped to <0.22.0 — PN90
            # self-skips via _PROPOSER_DRIFT_MARKERS on dev371+ regardless
            # of range; bumping eliminates the spurious version-mismatch
            # WARN log that stacks on top of the drift-marker skip.
            "vllm_version_range": (">=0.20.2rc1.dev9", "<0.22.0"),
        },
        "requires_patches": [],
        "conflicts_with": [],
    },
    "SNDR_WORKSPACE_001": {
        "title": "SNDR-WORKSPACE-001 — workspace grow-after-lock graceful fix",
        "tier": "community",
        "family": "worker",
        "env_flag": "GENESIS_ENABLE_SNDR_WORKSPACE_001",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "perf_hotfix",
        "apply_module": "sndr.engines.vllm.patches.worker.sndr_workspace_001_grow_after_lock",
        "source": "genesis_original",
        "credit": (
            "Genesis-original fix for the upstream vllm v1 workspace.py guard "
            "that raises AssertionError when any post-warmup path "
            "(decode_attention, continuation_prefill, Marlin GEMM scratch) "
            "needs to grow the GPU workspace. Without this patch the engine "
            "crashes on every request that touches a code path warmup did not "
            "pre-size. The patch replaces the raise with a warn + grow: the "
            "torch CUDA allocator handles the resize, the first call takes a "
            "non-graph slow path, subsequent calls hit the graph normally. "
            "Net effect: engine keeps serving instead of crashing. "
            "Update 2026-05-14 PR sweep: upstream vllm#42551 (jasonboukheir, "
            "DRAFT) proposes a more invasive fix to the same fault class — "
            "pre-reserve decode workspace in TurboQuantAttentionImpl.__init__ "
            "plus non-raising try_get_simultaneous() + torch.empty fallback "
            "in _decode_attention. When #42551 merges and a pin bump absorbs "
            "it, this patch self-retires via drift-marker. Until then we "
            "stay on the lighter warn+grow shape which is already running "
            "in PROD without issue (27B INT4 + TQ k8v4, 256K hardware-"
            "verified Wave 9). PN34 is being retired in this same wave "
            "(duplicate of this patch with the same fault site)."
        ),
        "applies_to": {
            "vllm_version_range": (">=0.20.2rc1.dev9", "<0.24.0"),
        },
        "requires_patches": [],
        "conflicts_with": [],
        "upstream_pr": 42551,  # 2026-05-14 PR sweep — pin-bump retire trigger
        "upstream_pr_relationship": "backport",
        "implementation_status": "full",
    },
    "SNDR_MTP_DYNAMIC_K_001": {
        "vllm_version_range": (">=0.20.0", "<0.23.0"),  # retired-provenance drift cap (native vllm#32374 in dev148)
        "title": "SNDR-MTP-DYNAMIC-K-001 — adaptive K MTP proposer (vllm#26504 port to DraftModelProposer base)",
        # Phase 10.5 edition-boundary fix (2026-06-01): tier corrected
        # from 'engine' to 'community'. P0-3/P0-4 audit (2026-05-08)
        # policy: public Genesis repo carries no engine-tier patches —
        # PN72 was the last one and moved community as Genesis-original
        # community code. SNDR_MTP_DYNAMIC_K_001 title + credit both
        # reference vllm#26504 (the upstream PR it ports from), which
        # violates the strict-AND engine-tier rule (no upstream_pr / no
        # PR ref / no author ref simultaneously). community tier is the
        # correct classification for backported / PR-derived work, even
        # when Sandermage authored the port itself.
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_SNDR_MTP_DYNAMIC_K_001",
        "default_on": False,
        # RETIRED 2026-06-19 (dev148 TIER-1 audit): superseded by native
        # vllm#32374 "[V1][Spec Decode] Add Dynamic SD" (MERGED 2026-06-14,
        # in-pin on dev148). The engine now provides dynamic speculative
        # decoding natively, so the #26504 DraftModelProposer monkey-patch
        # port is redundant. Three independent benches (single-stream, dual
        # 35B/27B, multi-turn) measured the K_001 effect as indistinguishable
        # from noise on the qwen3.6 stack, so retiring loses nothing. default_on
        # stays False; the monkey-patch code remains in the registry for the
        # audit trail.
        "lifecycle": "retired",
        "superseded_by": ["vllm#32374"],
        "retired_reason": (
            "native vllm#32374 (Dynamic SD, MERGED 2026-06-14, in-pin on "
            "dev148) provides dynamic speculative decoding in-engine; the "
            "#26504 DraftModelProposer port is redundant. K_001 effect was "
            "bench-measured NOT_SIGNIFICANT across all qwen3.6 workloads."
        ),
        "category": "perf_hotfix",
        "apply_module": "sndr.engines.vllm.patches.spec_decode.g_dynamic_k_mtp_proposer",
        "source": "genesis_original",
        "credit": (
            "Genesis-original port of vllm#26504 (whytem's DynamicProposer "
            "extending EagleProposer) to the DraftModelProposer base used "
            "by qwen3.6-27B/35B assistant-model MTP. Algorithm verbatim "
            "from PR #26504: per-seq SequenceState with rolling "
            "acceptance-rate window (len=10), K-adjustment with "
            "hysteresis (avg_acc >= threshold+0.05 -> K++ up to launcher "
            "cap; avg_acc <= threshold-0.05 -> K-- down to MIN=1), called "
            "via monkey-patch on DraftModelProposer.__init__ and "
            ".propose. SCOPE CORRECTION (2026-05-31): gemma4-31B/26B "
            "MTP is NO-OP for this patch — `Gemma4Proposer` MRO is "
            "`[Gemma4Proposer, SpecDecodeBaseProposer, object]` and "
            "does NOT inherit from `DraftModelProposer`, so the "
            "monkey-patch never reaches the gemma4 propose hot-path. "
            "A separate Gemma4Proposer-targeted patch would be needed "
            "for gemma4 adaptive K (future work). Empirical claim from "
            "PR #26504 author: +5-12% TPS on mixed workload vs static K. "
            "Operator value: for qwen3.6 models, removes the need for "
            "the chat-K=3 + structured-K=4 launcher split — single "
            "launcher converges to the right K per-sequence at runtime "
            "instead of requiring the gateway to route per request "
            "signal. The workload-class semantic split (different "
            "compression_plan / drafter behavior) stays valid; only "
            "the K choice becomes self-adapting. Default-off — operator "
            "must explicitly set the env flag after A/B benching against "
            "the static-K baselines on qwen3.6 specifically. "
            "FIRST EMPIRICAL BENCH (2026-06-02, qwen3.6-35b-multiconc "
            "preset, 2x A5000 TP=2, vllm 0.21.1rc1.dev354+g626fa9bba, "
            "n=25 per arm, genesis_bench_suite.py --quick): "
            "wall_TPS control 211.84 +- 11.98% CV vs treatment 208.62 "
            "+- 11.08% CV (delta -1.52%, Welch t=-0.118 p=0.9063 "
            "NOT_SIGNIFICANT). Boot log confirms `status=applied "
            "elapsed_ms=0.08 ordinal=48` — patch wiring works on "
            "qwen3.6 (DraftModelProposer MRO matches, unlike gemma4 "
            "no-op). Plan section 15.1 forecasted +5-12% on mixed "
            "workload; this single-stream-batched preset shows no "
            "measurable benefit. Default OFF empirically validated "
            "for prod-qwen3.6-35b-multiconc. Operator follow-up "
            "candidates: threshold sweep (default 0.7; try 0.5/0.8), "
            "multi-turn agentic workload (bench_agentic.py), wider "
            "K-range workloads (k1+k4 mixed) where adaptive K saves "
            "cycles vs fixed K=3, 27B Lorbus INT4 preset for "
            "different drafter-path comparison. "
            "SECOND EMPIRICAL BENCH (2026-06-03, BOTH "
            "qwen3.6-35b-multiconc + qwen3.6-27b-multiconc presets, "
            "same pin + same n=25 protocol): 35B Δwall_TPS=-1.66% "
            "(t=-0.570 p=0.5688 NOT_SIGNIFICANT, OFF=214.04 ON=210.48); "
            "27B Δwall_TPS=+0.21% (t=+0.088 p=0.9295 NOT_SIGNIFICANT, "
            "OFF=118.54 ON=118.78). Both qwen3.6-applicable production "
            "presets confirm K_001 effect indistinguishable from noise "
            "under short-prompt batched workload. Default OFF empirically "
            "ratified across full qwen3.6 production scope. "
            "THIRD EMPIRICAL BENCH (2026-06-03, multi-turn workload via "
            "tools/bench_multiturn_tps.py — explicitly designed to "
            "exercise per-seq SequenceState window-maturation, the only "
            "remaining viable hypothesis for the +5-12% forecast): "
            "qwen3.6-35b-multiconc, 12 turns × 2 sessions = n=24 per arm, "
            "same pin. OVERALL: Δwall_TPS=+1.40% (t=+0.169 p=0.8656 "
            "NOT_SIGNIFICANT, OFF=47.34 ON=48.01). LATE WINDOW (turns "
            "10-12, SequenceState matured): Δ=+1.20% (t=+0.906 p=0.3651 "
            "NOT_SIGNIFICANT, OFF=43.19 ON=43.71). The multi-turn "
            "hypothesis is now ALSO FALSIFIED — K_001 produces no "
            "measurable improvement even with mature SequenceState "
            "across 24 conversation turns. PR #26504 author's +5-12% "
            "forecast does not materialize on Genesis's qwen3.6 stack "
            "on any tested workload. Default OFF is the empirically "
            "correct setting. Evidence: "
            "evidence/bench/v11.2.0_k001_validation/{35b,27b}_k001_"
            "{off,on}.json + 35b_multiturn_k001_{off,on}.json + "
            "SUMMARY.md + tools/bench_multiturn_tps.py for the "
            "reusable multi-turn measurement harness."
        ),
        "applies_to": {
            "spec_decode_method": ["mtp"],
            "vllm_version_range": (">=0.21.0", "<0.22.0"),
            # Self-gating: monkey-patch installs on DraftModelProposer
            # only. Models whose Proposer class doesn't inherit from
            # DraftModelProposer (e.g. Gemma4Proposer extends
            # SpecDecodeBaseProposer directly) are unaffected by MRO.
            "proposer_mro_must_include": "DraftModelProposer",
        },
        "requires_patches": [],
        "conflicts_with": [],
        "upstream_pr": 26504,
        "upstream_pr_relationship": "backport",
        "implementation_status": "retired",
    },
    "SNDR_EAGLE3_AUX_HIDDEN_001": {
        "title": "SNDR-EAGLE3-AUX-HIDDEN-001 — model-side prep for EAGLE-3 (arXiv 2503.01840)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_SNDR_EAGLE3_AUX_HIDDEN_001",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "perf_hotfix",
        "apply_module": (
            "sndr.engines.vllm.patches.spec_decode."
            "sndr_eagle3_aux_hidden_001"
        ),
        "source": "genesis_original",
        "credit": (
            "Genesis-original Phase 7 EAGLE-3 model-side preparation. "
            "EAGLE-3 (arXiv 2503.01840) drafters fuse {input_embeds, "
            "last_hidden, aux_hidden_states} from intermediate target "
            "layers. vLLM landed EAGLE-3 in V2 ModelRunner via PR #35029 "
            "(2026-02-21) + #35040 (CUDA graph). PR #43132 (Qwen3) is "
            "still open as of 2026-06. A trained Qwen3.6 EAGLE-3 drafter "
            "checkpoint does NOT exist publicly yet. This patch ships "
            "the safe API surface — register_aux_hidden_state_hooks() + "
            "pop_aux_hidden_states() — so when a checkpoint lands the "
            "drafter wire-up is <1 day. Default OFF; with no caller "
            "invoking the helpers, zero runtime cost on the target "
            "model. Layer-id selection via "
            "GENESIS_SNDR_EAGLE3_AUX_LAYER_IDS env (comma-separated). "
            "Forward hook lifecycle is idempotent + thread-safe. "
            "References: vllm#35029, vllm#35040, vllm#43132 (Qwen3 "
            "EAGLE-3 open), G4_71-G4_76 (Gemma4 drafter routing "
            "template Genesis can reuse for the future drafter wire-up)."
        ),
        "applies_to": {
            # Self-gating — patch is a NO-OP on the target-model forward
            # until a future drafter explicitly calls
            # register_aux_hidden_state_hooks(). No model_class predicate
            # because the helper auto-detects layer attribute path.
            "spec_decode_method": ["eagle3"],
        },
        "requires_patches": [],
        "conflicts_with": [],
        "upstream_pr": 35029,
        "upstream_pr_relationship": "enables_upstream",
        "implementation_status": "experimental",
    },
    "PN202": {
        "title": "PN202 — per-layer KV tensor split (Tier 2.A enabler)",
        "tier": "community",
        "family": "streaming",  # was "kv_cache"; integration lives at integrations/streaming/
        "env_flag": "GENESIS_ENABLE_PN202_PER_LAYER_KV_SPLIT",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "memory",
        "apply_module": "sndr.engines.vllm.patches.streaming.pn202_per_layer_kv_split",
        "source": "genesis_original",
        "credit": (
            "Genesis-original Tier 2.A enabler. Replaces Branch-C of "
            "kv_cache_utils.py::get_kv_cache_config_from_groups (which "
            "emits group_size slabs each shared by representative layers) "
            "with one KVCacheTensor per layer (Branch-A semantics). Net "
            "bytes identical; enables per-layer memory policies "
            "(offload/evict/quantize) required by PN203. Zero speed "
            "and quality impact."
        ),
        "applies_to": {"vllm_version_range": (">=0.20.2rc1.dev9", "<0.24.0")},
        "requires_patches": [],
        "conflicts_with": [],
        "implementation_status": "full",
    },
    "PN203": {
        "title": "PN203 — cold-prefix CPU offload manager (Tier 3.A)",
        "tier": "community",
        "family": "streaming",  # was "kv_cache"; integration lives at integrations/streaming/
        "env_flag": "GENESIS_ENABLE_PN203_COLD_PREFIX_OFFLOAD",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "memory",
        "apply_module": "sndr.engines.vllm.patches.streaming.pn203_cold_prefix_offload",
        "source": "genesis_original",
        "credit": (
            "Genesis-original Tier 3.A — the architectural breakthrough "
            "piece. Demotes full-attention layer blocks older than "
            "active_window_tokens (default 32768) to PN95 pinned host RAM "
            "L2. Reuses existing PN95 infrastructure: pinned pool, stream "
            "pool, prefetch API, demote_on_evict. Mamba/GDN state kept "
            "GPU-resident (small fixed cost). Requires PN202 per-layer "
            "split. Math: 256K context fp8 KV = 8 GiB attention KV; "
            "active window 32K = 1 GiB → 7 GiB offloaded. Quality "
            "preserved (math identical). Decode adds 5-15ms per token "
            "when paging hits."
        ),
        "applies_to": {"vllm_version_range": (">=0.20.2rc1.dev9", "<0.24.0")},
        "requires_patches": ["PN202"],
        "conflicts_with": [],
        "implementation_status": "full",
    },
    "PN200": {
        "title": "PN200 — GDN outer-forward scratch pool (Tier 1.B) — RETIRED 2026-06-11",
        "tier": "community",
        "family": "streaming",  # was "kv_cache"; integration lives at integrations/streaming/
        "env_flag": "GENESIS_ENABLE_PN200_GDN_SCRATCH_REUSE",
        "default_on": False,
        # [Retire 2026-06-11, preflight residual triage §3 decision
        # EXECUTED — was PENDING DECISION earlier tonight; byte-verified]
        # Option (a) retire, NOT option (b) P28-chain re-anchor.
        # Evidence on pristine pin 0.22.1rc1.dev259
        # (mamba/gdn/qwen_gdn_linear_attn.py):
        # (1) P28 (PROD-applied, default_on=True legacy auto-apply) owns
        #     the unique forward_cuda site — its anchor is the
        #     #28182-comment-disambiguated SUPERSET that textually
        #     CONTAINS PN200's entire anchor, so the byte range PN200
        #     targets is consumed by P28's rewrite (1 match pristine).
        # (2) P28's replacement delivers the same buffer-reuse+zero
        #     contract PN200 promised: reused buffer
        #     (self._genesis_gdn_core_attn_buf[:num_tokens], attached at
        #     __init__ per CRIT-HW-1) + explicit .zero_() honoring the
        #     #28182 zero contract + torch.zeros fallback. PN200's
        #     pn106_get_pooled_buf(..., zero=True) route promised
        #     exactly reuse + .zero_() + torch.zeros fallback.
        # (3) PN200's bare anchor is ambiguous either way: 3 matches
        #     pristine (forward_cuda:950 / forward_xpu:991 /
        #     forward_cpu:1046), 2 post-P28 — both non-CUDA paths our
        #     2x A5000 never executes. TextPatcher demands exactly 1;
        #     PN200 can never apply on this pin family again.
        # (4) A chain variant would only pool-route P28's eager fallback
        #     branch (taken only when the prealloc buffer is absent or
        #     over capacity) and would reintroduce the in-forward env
        #     read + import that CRIT-HW-1 forbids (PN200's replacement
        #     did `import os` + env.get per forward_cuda call).
        # Flag was never enabled anywhere — every launcher exported '0';
        # exports removed with this retire per the journal corollary
        # (range-capping is NOT retirement while launchers still export
        # the flag). Module archived.
        "superseded_by": (
            "INTERNAL: P28 GDN core_attn_out prealloc (PROD-applied, "
            "default_on=True) — anchor superset of PN200's on the unique "
            "forward_cuda site; same buffer-reuse + explicit .zero_() "
            "(#28182 contract) + torch.zeros fallback, plus the capacity "
            "guard and CRIT-HW-1 init-time attach PN200 lacked — "
            "byte-verified on pristine 0.22.1rc1.dev259, 2026-06-11"
        ),
        "lifecycle": "retired",
        # Top-level mirror of the applies_to cap below (retired_provenance
        # contract: superseded_by + vllm_version_range, P78 precedent).
        # Anchor last byte-verified on nightly dcacdf9a (2026-05-14);
        # supersession by P28 byte-verified on 0.22.1rc1.dev259.
        "vllm_version_range": (">=0.20.2rc1.dev9", "<0.21.0"),  # mirrors applies_to (gated SoT); reconciled D17 2026-06-17 (added >=...dev9 lower bound)
        "category": "memory",
        "apply_module": "sndr.engines.vllm._archive.pn200_gdn_scratch_reuse",
        "source": "genesis_original",
        "credit": (
            "Genesis-original Tier 1.B of three-tier memory plan. Routed "
            "gdn_linear_attn.py:765 core_attn_out (32 MiB × 48 layers per "
            "step) through the PN106 named-pool API with zero=True. "
            "Honored the vllm PR #28182 'must be zeroed' contract via "
            "explicit .zero_() on pool slice. Composed with PN106 (inner "
            "FLA chunk scratch) and PN201 (scheduler empty_cache) for "
            "Tier 1 coverage. RETIRED — P28 owns the CUDA site and "
            "delivers the same buffer-reuse+zero; see retire note."
        ),
        # Pin-gate mirror of the retire: the pre-existing upper cap
        # already excludes every 0.22.x pin; anchor last byte-verified
        # on nightly dcacdf9a (2026-05-14, single-file gdn_linear_attn).
        "applies_to": {"vllm_version_range": (">=0.20.2rc1.dev9", "<0.21.0")},
        "requires_patches": [],
        "conflicts_with": [],
        "implementation_status": "full",
    },
    "PN201": {
        "title": "PN201 — scheduler empty_cache hook (Tier 1.C)",
        "tier": "community",
        "family": "streaming",  # was "kv_cache"; integration lives at integrations/streaming/
        "env_flag": "GENESIS_ENABLE_PN201_SCHEDULER_EMPTY_CACHE",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "memory",
        "apply_module": "sndr.engines.vllm.patches.streaming.pn201_scheduler_empty_cache",
        "source": "genesis_original",
        "credit": (
            "Genesis-original Tier 1.C. Threshold-gated torch.cuda."
            "empty_cache() called from PN95 scheduler_tick when free_mib "
            "< 256 or free_blocks < threshold AND cooldown elapsed. "
            "Reclaims the 'reserved but unallocated' fragmentation that "
            "OOM logs show as 319 MiB stuck after long prefill runs. "
            "Cooldown (default 50 ticks ~= 5s) prevents hot-path stall. "
            "No text-patch — runtime hook in existing scheduler_tick path."
        ),
        "applies_to": {"vllm_version_range": (">=0.20.2rc1.dev9", "<0.24.0")},
        "requires_patches": [],
        "conflicts_with": [],
        "implementation_status": "full",
    },
    "PN106": {
        "title": "PN106 — GDN scratch tensor pool (architectural memory mgr)",
        "tier": "community",
        "family": "kv_cache",
        "env_flag": "GENESIS_ENABLE_PN106_GDN_H_POOL",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "memory",
        "apply_module": "sndr.engines.vllm.patches.kv_cache.pn106_gdn_h_pool",
        "source": "genesis_original",
        "credit": (
            "Genesis-original architectural memory manager. Replaces the "
            "per-call torch.empty / torch.empty_like patterns inside the "
            "48-GDN-layer hot path (chunk_delta_h.py, chunk_o.py) with "
            "slice views into named persistent pools. Eliminates 2.4-5.7 GiB "
            "of alloc/free traffic per chunked-prefill step and 200-400 MiB "
            "of steady-state fragmentation. Includes generic "
            "pn106_get_pooled_buf(name, shape, dtype, device) API for "
            "extending to other hot-path allocations (Marlin scratch, "
            "FlashAttention k_full/v_full, etc). Targets the exact crash "
            "site observed at chunk_o.py:168 on Qwen3.6-27B + 156K context."
        ),
        "applies_to": {
            "vllm_version_range": (">=0.20.2rc1.dev9", "<0.24.0"),
        },
        "requires_patches": [],
        "conflicts_with": [],
        "implementation_status": "full",
    },
    "PN105": {
        "title": "PN105 — PrefetchOffloader AutoRound INT4 compat (pin-assert relax)",
        "tier": "community",
        "family": "offload",
        "env_flag": "GENESIS_ENABLE_PN105_AUTOROUND_OFFLOAD_COMPAT",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "stability",  # was "compat"; normalize to existing VALID_CATEGORIES (AutoRound INT4 compat / pin-assert relax)
        "apply_module": "sndr.engines.vllm.patches.offload.pn105_prefetch_autoround_compat",
        "source": "genesis_original",
        "credit": (
            "Genesis-original fix unblocking vllm's PrefetchOffloader on "
            "AutoRound INT4 models. AutoRound's process_weights_after_"
            "loading replaces param.data with non-pinned INT tensors "
            "(g_idx, qzeros, scales). PrefetchOffloader asserts ALL "
            "cpu_storage pinned and crashes at engine startup. PN105 "
            "replaces the assertion with conditional blocking copy: "
            "non-blocking when pinned (fast path), blocking when not "
            "(slow fallback). Blocking copy from pageable memory IS "
            "correct, just slower. Combined with PN104 (UVA→Prefetch "
            "redirect) and Tier 1 GDN scratch pool, this unblocks "
            "cpu_offload_gb=8 on AutoRound → KV pool grows 4 GB → "
            "9-10 GB → 156K-176K context on single A5000."
        ),
        "applies_to": {"vllm_version_range": (">=0.20.2rc1.dev9", "<0.24.0")},
        "requires_patches": ["PN104"],
        "conflicts_with": [],
        "implementation_status": "full",
    },
    "PN104": {
        "title": "PN104 — redirect --cpu-offload-gb from UVA to Prefetch backend",
        "tier": "community",
        "family": "offload",
        "env_flag": "GENESIS_ENABLE_PN104_OFFLOAD_PREFETCH_REDIRECT",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "kernel",  # was "perf_critical"; normalize (kernel-level redirect from UVA to Prefetch backend)
        "apply_module": "sndr.engines.vllm.patches.offload.pn104_offload_backend_redirect",
        "source": "genesis_original",
        "credit": (
            "Genesis-original critical perf patch. vllm's --cpu-offload-gb "
            "defaults to UVAOffloader which uses cudaHostGetDevicePointer to "
            "map pinned host RAM as GPU-visible pointer — every GEMM kernel "
            "issues PCIe reads on every load instruction, with zero GPU "
            "caching. Empirically observed 24x slowdown (50K prefill: 30s "
            "→ 720s) on Genesis 27B INT4 single-A5000 with cpu_offload_gb=8. "
            "PN104 monkey-patches create_offloader so cpu_offload_gb auto-"
            "translates to PrefetchOffloader params (offload_group_size + "
            "num_in_group + prefetch_step=2). PrefetchOffloader does explicit "
            "cudaMemcpyAsync into static GPU buffers on a side stream, "
            "compute reads from GPU buffer, copy hidden behind previous "
            "layer's compute. Expected +30-50% TPS recovery — the single "
            "biggest win for any cpu_offload deployment."
        ),
        "applies_to": {
            "vllm_version_range": (">=0.20.2rc1.dev9", "<0.24.0"),
        },
        "requires_patches": [],
        "conflicts_with": [],
        "implementation_status": "full",
    },
    "PN383": {
        "title": (
            "PN383 — KV-offload + MTP cuMemcpyBatchAsync segfault gate "
            "(vendor of vllm#44784) + Qwen3.6 narrowing + pre-DMA bounds check"
        ),
        "tier": "community",
        "family": "offload",
        "env_flag": "GENESIS_ENABLE_PN383_OFFLOAD_MTP_EAGLE_GATE",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "stability",
        "apply_module": "sndr.engines.vllm.patches.offload.pn383_offload_mtp_eagle_gate",
        "source": "vllm_pr_backport",
        "upstream_pr": 44784,
        "upstream_pr_relationship": "backport",
        "credit": (
            "Vendor of OPEN PR vllm#44784 (issue #44780): "
            "OffloadingConnectorScheduler schedules EAGLE/MTP draft-attention "
            "groups into KV-offload store/load; the drafter's volatile "
            "trailing block (no stable hash, tiny gpu_block_size) yields an "
            "out-of-bounds GPU block index that segfaults silently in "
            "cuMemcpyBatchAsync, blocking native CPU KV offload on every MTP "
            "config (Qwen3.6 MTP K=3 included). Four scheduler hunks gate the "
            "eagle groups; one worker hunk re-adds the pre-DMA bounds check "
            "the PR dropped (raises RuntimeError instead of segfaulting). "
            "Genesis extension: Qwen3.6-specific is_eagle_group flagging "
            "narrows the all-groups fallback to the real 'mtp'-prefix drafter "
            "group so the offload lookup keeps prefix-cache hit-rate. Dormant "
            "until a KV-offload backend is configured."
        ),
        "applies_to": {"vllm_version_range": (">=0.22.0", "<0.23.0")},
        "composes_with": ["PN104", "PN105", "PN102"],
        "requires_patches": [],
        "conflicts_with": [],
        "implementation_status": "full",
    },
    # ── 2026-06-13 50-PR sweep WAVE 1 (batch 2) — five LIVE-bug vendors
    # of OPEN upstream PRs, all opt-in (default_on=False) pending the
    # server A/B. Spec-driven from inception (apply_module + own apply(),
    # no legacy @register_patch hook) — same class as PN371/PN373/PN383.
    # See journal docs/superpowers/journal/2026-06-11-pr-sweep-62-batch2-
    # roadmap.md. PN384 + PN388 both touch KV/scheduler split paths and
    # are designed to coexist (see their credit notes).
    "PN384": {
        "title": (
            "PN384 — Eagle/MTP prefix-cache prefill fix "
            "(vendor of vllm#44986); thread skip_eagle_pop=is_prefill_phase"
        ),
        "tier": "community",
        "family": "kv_cache",
        "env_flag": "GENESIS_ENABLE_PN384_EAGLE_PREFIX_CACHE_PREFILL",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "kv_cache",
        "implementation_status": "full",
        "source": "vllm_pr_backport",
        "apply_module": "sndr.engines.vllm.patches.kv_cache.pn384_eagle_prefix_cache_prefill",
        # category=kv_cache (family default; VALID_CATEGORIES has no
        # 'performance' — this is a prefix-cache TTFT correctness win).
        "credit": (
            "PR-sweep batch-2 wave-1 (2026-06-13). Vendor of OPEN "
            "vllm#44986 (closes #44858): when EAGLE/MTP + "
            "--enable-prefix-caching, get_computed_blocks drops the last "
            "matched prefix-cache block so the drafter can recompute "
            "hidden states. Correct in decode, WRONG in prefill — no draft "
            "tokens exist yet (num_output_tokens == 0) so the dropped block "
            "is pure loss. Filed on Qwen3.6-27B block_size=1536 — our exact "
            "27B int4 PROD shape. Fix: thread skip_eagle_pop=is_prefill_phase "
            "through find_longest_cache_hit (Unitary + Hybrid coordinators + "
            "the manager caller) so the drop is suppressed in prefill only. "
            "Recovers 1 prefix-cache block per prefill on every MTP request; "
            "on short prompts (block_size > prompt) recovers the ENTIRE hit "
            "(0% -> full). Direct TTFT win, ZERO decode cost (decode path "
            "byte-unchanged). SUPERSEDES retired P83/P84: it skips only in "
            "prefill, preserving the convergence invariant P83 flagged. "
            "COORDINATION: PN346 (#43650) touches the sibling file "
            "single_type_kv_cache_manager.py (MambaManager) — PN384 patches "
            "the coordinator/manager one level up; zero anchor overlap, the "
            "two compose (in prefill skip_eagle_pop=True makes PN346's "
            "drop_eagle_block guard a clean no-op). Genesis spelling "
            "divergence keeps the PR's exact lines as merged-form drift "
            "markers. STRONG RECOMMENDATION: enable on 27B int4 "
            "(block_size=1536) + 35B FP8 MTP-K=3 after the server A/B "
            "confirms the TTFT recovery. COEXISTS with PN388 (scheduler "
            "split): disjoint files/functions (find_longest_cache_hit vs "
            "_mamba_block_aligned_split), no anchor overlap."
        ),
        "upstream_pr": 44986,
        "upstream_pr_relationship": "backport",
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["PN346"],
        "applies_to": {"vllm_version_range": (">=0.22.0", "<0.24.0")},
        "vllm_version_range": (">=0.22.0", "<0.24.0"),
    },
    "PN385": {
        "title": "Forced-named empty-params tool schema -> JSON object (vendor of vllm#45290)",
        "tier": "community",
        "family": "tool_parsing",
        "env_flag": "GENESIS_ENABLE_PN385_FORCED_NAMED_EMPTY_PARAMS",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "structured_output",
        "implementation_status": "full",
        "source": "vllm_pr_backport",
        "apply_module": "sndr.engines.vllm.patches.tool_parsing.pn385_forced_named_empty_params",
        "credit": (
            "PR-sweep batch-2 wave-1 (2026-06-13). Vendor of OPEN "
            "vllm#45290: the public get_json_schema_from_tools returns a "
            "forced-named tool's parameters verbatim on both the Responses "
            "and ChatCompletion branches. For a no-arg tool (end_turn / "
            "noop / handoff) that is None (free-form text) or {} "
            "(unconstrained), so the model can emit a bare string/number "
            "as arguments instead of {} -> agent-loop parse-500 on "
            "parameterless tools for qwen3_xml (35B/27B) and gemma4 "
            "(26B/31B); qwen3coder shielded. Fix: normalize both "
            "forced-named branches to {\"type\": \"object\", "
            "\"properties\": {}} exactly like the in-pin `required` path "
            "(_get_tool_schema_from_tool). Genesis divergence: inline the "
            "in-pin idiom instead of the PR's _params_or_empty_object "
            "helper (no 3rd anchor; the helper name stays a clean upstream "
            "drift marker). Disjoint from PN70 (internal "
            "_get_json_schema_from_tools required path) and P68 "
            "(auto->required gate) — verified no anchor collision. "
            "Anchors byte-verified count==1 vs pristine pin g303916e93. "
            "Candidate default-ON after fleet test on the three exposed "
            "families."
        ),
        "upstream_pr": 45290,
        "upstream_pr_relationship": "backport",
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["PN70", "P68"],
        "applies_to": {"vllm_version_range": (">=0.22.0", "<0.24.0")},
        "vllm_version_range": (">=0.22.0", "<0.24.0"),
    },
    "PN386": {
        "title": "Required-tool streaming brace JSON-string-awareness (vendor of vllm#45389)",
        "tier": "community",
        "family": "tool_parsing",
        "env_flag": "GENESIS_ENABLE_PN386_REQUIRED_STREAMING_STRING_AWARE",
        "default_on": False,
        "lifecycle": "experimental",
        # category=structured_output (the tool_parsing-sibling convention;
        # VALID_CATEGORIES has no 'tool_calling'). PN385 uses the same.
        "category": "structured_output",
        "implementation_status": "full",
        "source": "vllm_pr_backport",
        "apply_module": "sndr.engines.vllm.patches.tool_parsing.pn386_required_streaming_brace_string_aware",
        "credit": (
            "PR-sweep batch-2 wave-1 (2026-06-13). Vendor of OPEN "
            "vllm#45389 (Sunt-ing; related #41111). In tool_choice="
            "'required' STREAMING, vLLM trims the streamed tool-call JSON "
            "by counting the wrapper's bracket level in "
            "tool_parsers/streaming.py. _bracket_level and "
            "filter_delta_text treat { } , as structural even inside a "
            "JSON string VALUE, so a valid argument like {\"city\": "
            "\"a } b\"} is mis-trimmed into malformed function.arguments "
            "(client JSONDecodeError) for any string carrying { } \\\" \\\\ "
            "(file paths, regex, shell, nested JSON). Fix: "
            "_bracket_level_state tracks in_string/escaped and skips "
            "brace counting inside strings; filter_delta_text carries "
            "that state across deltas and only breaks on a top-level "
            "comma when not in_string; the parameter body is trimmed "
            "against its own prefix (current_text[:param_match.start(1)]) "
            "instead of previous_text (greedy .*\"parameters\" match kept "
            "for multi-tool streaming). PREREQUISITE for safely enabling "
            "Genesis P68 (long-ctx auto-force-required funnels agent "
            "traffic into this exact helper). Sibling #45310 (Hermes "
            "</tool_call> boundary, same string-awareness bug class one "
            "layer up) is the wave-2 pairing. Genesis divergence: spells "
            "the prefix slice without the space after [: and the thin "
            "wrapper unpacks named throwaways, so the PR's exact lines "
            "stay usable as merge drift markers without self-collision. "
            "Anchors byte-verified count==1 on pristine pin g303916e93."
        ),
        "upstream_pr": 45389,
        "upstream_pr_relationship": "backport",
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["P68"],
        "applies_to": {"vllm_version_range": (">=0.22.0", "<0.24.0")},
        "vllm_version_range": (">=0.22.0", "<0.24.0"),
    },
    "PN387": {
        "title": "Reject degenerate structured_outputs (DoS guard, vendor of vllm#45346)",
        "tier": "community",
        "family": "serving",
        "env_flag": "GENESIS_ENABLE_PN387_REJECT_DEGENERATE_STRUCTURED_OUTPUTS",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "stability",
        "implementation_status": "full",
        "source": "vllm_pr_backport",
        "apply_module": "sndr.engines.vllm.patches.serving.pn387_reject_degenerate_structured_outputs",
        "credit": (
            "PR-sweep batch-2 wave-1 (2026-06-13). Vendor of OPEN vllm#45346 "
            "(Sunt-ing) — CONFIRMED instance-wide DoS on single-instance PROD. "
            "A single request with structured_outputs={'json_object': false} "
            "or {'json': ''} crashes EngineCore (EngineDeadError) and bricks "
            "the instance for everyone: both inputs pass "
            "StructuredOutputsParams.__post_init__'s `is not None` exclusivity "
            "check but have no key in get_structured_output_key, so they raise "
            "inside the per-request-isolation-free EngineCore step loop. "
            "Two layers, one flag: (1) SOURCE OVERLAY (verbatim PR backport) "
            "adds two guards in SamplingParams._validate_structured_outputs "
            "after the pin's empty-grammar guard (line 888) — json='' and "
            "json_object=False -> ValueError (frontend 400). (2) GENESIS EDGE "
            "GUARD injects a request-validation hook at the top of "
            "_create_chat_completion that returns a clean 400 BadRequestError "
            "BEFORE the request reaches the engine loop (defence-in-depth; "
            "composes with P68/P69 + PN16 on the same anchor pair). The single "
            "PN387 apply() drives both files atomically via "
            "MultiFilePatchTransaction. default_on=False — pure safety reject, "
            "gated so the rejection criteria can be A/B'd first; STRONG "
            "RECOMMENDATION to enable on every single-instance PROD. "
            "Self-skips when #45346 merges (drift markers = the PR's exact "
            "guard comment lines). Bit-identical for valid inputs."
        ),
        "upstream_pr": 45346,
        "upstream_pr_relationship": "backport",
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["P68", "P69", "PN16", "P109"],
        "applies_to": {"vllm_version_range": (">=0.22.0", "<0.24.0")},
        "vllm_version_range": (">=0.22.0", "<0.24.0"),
    },
    "PN388": {
        "title": "Mamba-block-aligned intermediate prefill split (vendor of vllm#45477)",
        "tier": "community",
        "family": "scheduler",
        "env_flag": "GENESIS_ENABLE_PN388_MAMBA_BLOCK_ALIGNED_SPLIT",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "correctness",
        "implementation_status": "full",
        "source": "vllm_pr_backport",
        "apply_module": "sndr.engines.vllm.patches.scheduler.pn388_mamba_block_aligned_prefill_split",
        "credit": (
            "PR-sweep batch-2 wave-1 (2026-06-13), HIGHEST value LIVE bug. "
            "Vendor of OPEN vllm#45477 (FIX #43559, root cause of the "
            "Qwen3.5/3.6 + MTP + --enable-prefix-caching accuracy collapse). "
            "Scheduler._mamba_block_aligned_split keeps prefill chunk ends on "
            "mamba block boundaries (align cache mode). With spec decode "
            "(use_eagle) the eagle prune zeroes last_cache_position for prompts "
            "< 2*block_size (mamba block 1600; a 2002-tok prompt qualifies), so "
            "the old else:pass fall-through accepts an unaligned chunk end the "
            "moment the per-step budget fragments concurrent prefills. The GDN "
            "kernel writes that mid-block state into the position-0 mamba slot "
            "and cache_blocks hashes it as the boundary snapshot — poisoning "
            "the prefix cache (garbled output / stray </think> / malformed tool "
            "calls / runaway gens) for every request that resumes from it; "
            "persists until restart. Single requests are accidentally safe so "
            "it only shows under concurrent unequal prefixes — our exact PROD "
            "multiconc shape (Qwen3.6 35B FP8 + 27B int4, GDN+Mamba, MTP K=3, "
            "APC). Fix: round the chunk END position (not the LENGTH) so every "
            "non-final chunk ends on a block boundary; a budget-collapsed first "
            "chunk defers via num_new_tokens==0 (scheduler-handled). Verified: "
            "unpatched pin+P34 yields unaligned non-final ends [364,1064]; "
            "patched yields [1600,2002] (poison-free). Genesis divergences "
            "(iron rule #10): the PR's Marconi common-prefix admission tail is "
            "OMITTED (its param is absent from our pin's signature — would "
            "NameError); round_down is inlined as (x//block_size)*block_size "
            "(single anchor site; gives drift-marker spelling divergence). P34 "
            "COEXISTENCE: P34 (effectively always-on) rewrites the exact "
            "first-branch lines PN388 deletes, so PN388 carries a DUAL ANCHOR "
            "(pristine-shaped + post-P34-shaped, required-at-least-one, "
            "apply()-pre-gated; P85-on-PN346 convention) and requires_patches "
            "P34 so P34 dispatches first; PN388's deferral subsumes P34's "
            "zero-collapse guard. Composes with PN346 (#43650) + P85 "
            "(hit-side, complementary per the PR author — PN346 alone does not "
            "cover non-final-block poisoning under unequal concurrent "
            "prefixes; we were CURRENTLY EXPOSED). Disjoint from PN384/#44986 "
            "(find_longest_cache_hit, different file+function) — PN384 + PN388 "
            "COEXIST cleanly. ASYNC CAVEAT: the PR validated "
            "--no-async-scheduling; our PROD runs async overlap ON — enable "
            "only after a server A/B confirms boundary-timing parity vs GDN "
            "state-write (10-way 2002-tok tool-call fanout replay + bench). "
            "default_on=False until that A/B."
        ),
        "upstream_pr": 45477,
        "upstream_pr_relationship": "backport",
        "upstream_issue": 43559,
        "requires_patches": ["P34"],
        "conflicts_with": [],
        "composes_with": ["PN346", "P85"],
        # 2026-06-17 (0.23.1 reverify): kept capped <0.23.0. Bug still live
        # (#45477 OPEN). REDESIGNED 2026-06-17: anchor narrowed to the live
        # 0.23.1 POST-P34 inner branch (verified byte-exact, count==1);
        # cap bumped <0.24.0.
        "applies_to": {"vllm_version_range": (">=0.22.0", "<0.24.0")},
        "vllm_version_range": (">=0.22.0", "<0.24.0"),
    },
    # ── 2026-06-13 50-PR sweep BATCH-3 — four more vendors of OPEN
    # upstream PRs (PN389/PN390/PN391 + P89), spec-driven from inception
    # (apply_module + own apply(), no legacy @register_patch hook; same
    # class as PN383-PN388). All opt-in (default_on=False), server A/B
    # pending. See journal docs/superpowers/journal/2026-06-11-pr-sweep-
    # 62-batch2-roadmap.md.
    "PN389": {
        "title": "XGrammar input-validation + grammar-compilation timeouts (vendor of vllm#45390)",
        "tier": "community",
        "family": "serving",
        "env_flag": "GENESIS_ENABLE_PN389_GRAMMAR_TIMEOUTS",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "stability",
        "implementation_status": "full",
        "source": "vllm_pr_backport",
        "apply_module": "sndr.engines.vllm.patches.serving.pn389_grammar_compilation_timeout",
        "credit": (
            "PR-sweep batch-3 (2026-06-13). Vendor of OPEN vllm#45390 "
            "(jperezdealgaba), grammar-timeout core of the 7-GHSA DoS bundle "
            "(CWE-400 uncontrolled resource consumption). XGrammar grammar/"
            "regex/JSON-schema DFA compilation runs on the CPU EngineCore loop "
            "with NO wall-clock bound; a pathological tool schema wedges ALL "
            "decode indefinitely (instance-wide DoS on single-instance PROD — "
            "async-scheduling overlap does NOT save us, compilation is pure-CPU "
            "off the GPU stream). CRITICAL pin finding: our pin g303916e93 has "
            "NO compilation timeout AT ALL (lacks even the first-gen "
            "compile_regex_with_timeout the PR refactors), so PN389 vendors the "
            "FULL run_with_timeout machinery, not the PR's rename. THREE files, "
            "one atomic MultiFilePatchTransaction: (1) utils.py ADDITIVE — "
            "run_with_timeout (daemon-thread + Queue + Semaphore(4)) + "
            "_check_regex_complexity (length 10K + paren-nesting 20 pre-filter) "
            "+ constants; (2) backend_xgrammar.py — TWO surfaces bounded: (2a) "
            "the EngineCore DFA build (compile_grammar refactored exactly as the "
            "PR into compile_grammar -> _compile_ctx -> "
            "run_with_timeout(self._compile_ctx_inner) so EVERY type's "
            "vocab-dependent compile — JSON/JSON_OBJECT/GRAMMAR/REGEX/"
            "STRUCTURAL_TAG — is wall-clock-bounded; _compile_ctx_inner holds "
            "the pin's compile_* dispatch verbatim so a compile within budget is "
            "bit-identical), and (2b) the frontend validate_xgrammar_grammar "
            "parse pre-flight (every xgr.Grammar.from_* call wrapped); (3) "
            "envs.py ADDITIVE — VLLM_GRAMMAR_COMPILATION_TIMEOUT_SECONDS. "
            "GENESIS DIVERGENCE (iron rule #10): default 2s, NOT the PR's 10s "
            "(10s blows our 70-160ms TTFT SLO ~60x before the timeout fires; "
            "operator can raise it back to 10 via the env). Bit-identical for "
            "compiles within budget; a >2s compile (frontend parse OR EngineCore "
            "DFA build) now bounces as a clean ValueError (400) instead of "
            "wedging the engine — including the catastrophic-compile case "
            "(schema parses fast, compiles unbounded) that frontend-only "
            "bounding would miss. The PR's sampling_params/protocol input-bounds "
            "half is OUT OF SCOPE (overlaps P109/PN387 surface; separate item). "
            "SYNERGY with PN386 (#45389): both harden the same XGrammar "
            "tool-call hot path every model uses (disjoint files, no anchor "
            "overlap). No collision with P62/PN58 (spec-decode mask timing / "
            "reasoning boundary). default_on=False — the timeout reject is a new "
            "failure mode for legit-but-slow grammars; gated until a server A/B "
            "confirms the 2s budget never trips a real gemma4/qwen3_coder "
            "tool-schema compile. Self-skips when #45390 merges (drift markers = "
            "the PR's exact docstring/comment lines)."
        ),
        "upstream_pr": 45390,
        "upstream_pr_relationship": "backport",
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["PN386", "P62", "P109", "PN387"],
        # 2026-06-17 (0.23.1 reverify): kept capped <0.23.0. Bug live
        # (#45390 OPEN). REDESIGNED 2026-06-17: collapsed to the live
        # backend_xgrammar.compile_grammar arm (composes with upstream
        # compile_regex_with_timeout); cap bumped <0.24.0. Tests: follow-up.
        "applies_to": {"vllm_version_range": (">=0.22.0", "<0.24.0")},
        "vllm_version_range": (">=0.22.0", "<0.24.0"),
    },
    "PN390": {
        "title": "Streaming-LSE rejection sampler — no full-vocab target_probs materialize (vendor of vllm#45369)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_PN390_STREAMING_LSE_SAMPLER",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "spec_decode",
        "implementation_status": "full",
        "source": "vllm_pr_backport",
        "apply_module": "sndr.engines.vllm.patches.spec_decode.pn390_streaming_lse_rejection_sampler",
        "credit": (
            "PR-sweep batch-3 (2026-06-13). Vendor of OPEN/DRAFT "
            "vllm#45369: the non-greedy speculative rejection sampler no "
            "longer materializes the full-vocab target_probs = "
            "target_logits.softmax(...) buffer. compute_target_lse + "
            "target_lse_kernel produce one logsumexp per row and the "
            "rejection / recovery Triton kernels reconstruct each probability "
            "as exp(logit - lse) (exact identity softmax(x)[i] == exp(x[i] - "
            "logsumexp(x)), ULP-stable). LIVE on both PROD MTP-K=3 models, "
            "which run GREEDY draft (draft_probs=None) so the NO_DRAFT_PROBS "
            "kernel arm fires every decode step — NOT the heavy with-draft-"
            "probs 'else' arm (corrected 2026-06-16: PN90 probabilistic is "
            "version-gated OFF on dev491 and P71 is inert; the earlier "
            "'with-draft-probs arm fires' claim was wrong). PN390 accelerates "
            "that arm's prob LOADS. Reclaims the transient buffer (vocab 151936: "
            "3.6 MB single row, 14.6 MB at K=3 batch-8 burst) and its HBM "
            "traffic — PR's A100 sweep shows -8..-11% mechanism latency on "
            "with-draft-probs rows, expected larger on byte-bound A5000. "
            "Line-orthogonal to PN378 (which masks the score product in the "
            "SAME kernel; PN390 rewrites the prob LOADS) — they compose. "
            "PN369's torch-side prob read is NOT in this kernel rewrite "
            "(documented out-of-scope). Genesis divergence: body constant "
            "named GENESIS_PN390_LSE_BLOCK_SIZE and the LSE store factored "
            "through a named intermediate, so upstream's 'BLOCK_SIZE: "
            "tl.constexpr = 8192' and 'tl.store(target_lse_ptr + row, m + "
            "tl.log(s))' lines stay usable as drift markers disjoint from our "
            "emitted text. default_on=False pending the A/B BLOCK_SIZE "
            "{8192/16384/...} + num_warps sweep on the server."
        ),
        "upstream_pr": 45369,
        "upstream_pr_relationship": "backport",
        "requires_patches": [],
        # PN390 rewrites the rejection sampler to drop the full-vocab
        # target_probs local that the relaxed-acceptance mask reads
        # (compute_relaxed_ok_mask). With both enabled the relaxed path
        # silently degrades (its read is try/except-guarded so it does not
        # crash, but goes dark and floods per-decode-step warnings). They do
        # NOT compose. Caught by deep-audit 2026-06-14.
        # drift D1 (deep-audit 2026-06-14): P71 moved composes->conflicts. PN390
        # rewrites rejection_sample to drop the dense [num_tokens,vocab]
        # target_probs buffer (computes only target_lse), but the P71
        # block-verify branch (rejection_sampler.py:518/525) still references
        # target_probs — a latent NameError, dormant ONLY because PROD runs
        # greedy draft (draft_probs is None gates the branch off).
        # 2026-06-19: PN369 was consolidated into the P71 entry (the merged
        # apply_module carries the PN369 relaxed-mask read). The old
        # ["PN369", "P71"] pair collapses to ["P71"] — PN369 is no longer a
        # registry id, and the single P71 conflict now covers both reasons.
        "conflicts_with": ["P71"],
        "composes_with": ["PN378", "PN90", "P82"],
        "applies_to": {"vllm_version_range": (">=0.22.0", "<0.24.0")},
        "vllm_version_range": (">=0.22.0", "<0.24.0"),
    },
    "PN391": {
        "title": "/health/decode forward-progress watchdog (vendor of vllm#45453)",
        "tier": "community",
        "family": "observability",
        "env_flag": "GENESIS_ENABLE_PN391_HEALTH_DECODE_WATCHDOG",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "stability",
        "implementation_status": "full",
        "source": "vllm_pr_backport",
        "apply_module": "sndr.engines.vllm.patches.observability.pn391_health_decode_watchdog",
        "credit": (
            "PR-sweep batch-3 (2026-06-13). Vendor of OPEN vllm#45453 "
            "(terafin, re-file of #45097) — additive GET /health/decode "
            "forward-progress watchdog. The stock /health only checks 'is the "
            "engine task alive?'; on a TP>1 NCCL P2P deadlock that survives a "
            "container restart (vllm#45094) the FastAPI task stays alive, "
            "/health keeps returning 200, and every request hangs at 0 tok/s. "
            "This is OUR EXACT topology: 2x A5000 PCIe, NCCL_P2P_DISABLE=1, "
            "--disable-custom-all-reduce (NCCL is the collective path), TP=2, "
            "restart:unless-stopped. The new route reports ok/prefilling/idle/"
            "stalled from StatLoggerManager per-step bookkeeping (2 attr writes "
            "+ 1 time.monotonic() read per step; zero GPU cost, no new locks). "
            "503 'stalled' only when running>0 AND decode-stall window exceeded "
            "AND prefill not recent. 6-file ATOMIC additive overlay "
            "(MultiFilePatchTransaction): envs.py (2 vars), health.py (route), "
            "metrics.py (Prometheus exclude), protocol.py (EngineClient default "
            "(0,None,None)), async_llm.py (AsyncLLM accessor + scale_elastic_ep "
            "carry-forward), loggers.py (per-engine bookkeeping). Byte-identical "
            "for anything that does not probe the new route. GENESIS value-add: "
            "tune VLLM_DECODE_LIVENESS_STALL_SECONDS ~20-30s and keep "
            "VLLM_PREFILL_LIVENESS_STALL_SECONDS >=30s so the 'prefilling' arm "
            "shields our 4.4s@32K GDN prefill from a false 503; follow-up wires "
            "the endpoint into tools/safe_container_recreate.py (currently polls "
            "the weak /health) as the readiness gate. DP partial-stall path "
            "inert-but-harmless on single-engine TP=2. default_on=False — ships "
            "dormant until the safe_container_recreate.py gate swap lands. "
            "Self-skips when #45453 merges (drift markers = the PR's exact "
            "docstring/comment heads, which this overlay re-words per iron rule "
            "#10). Genesis divergence: prose reworded, behavioral code "
            "byte-faithful."
        ),
        "upstream_pr": 45453,
        "upstream_pr_relationship": "backport",
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": [],
        "applies_to": {"vllm_version_range": (">=0.22.0", "<0.24.0")},
        "vllm_version_range": (">=0.22.0", "<0.24.0"),
    },
    "PN392": {
        "title": "qwen3_coder streaming tool-call within-call coalescing (dev491 regression fix)",
        "tier": "community",
        "family": "tool_parsing",
        "env_flag": "GENESIS_ENABLE_PN392_QWEN3CODER_STREAMING_COALESCE",
        "default_on": False,
        # RETIRED 2026-06-14: misdiagnosis. PN392 was built to "fix" the
        # dev491 streaming tool-call regression by coalescing the coder
        # parser's single-emission. But live raw smoke proved the dev491
        # native Qwen3CoderToolParser is self-sufficient — the regression was
        # caused by Genesis's OWN dev259-era qwen3coder wraps (P64/P61c/PN56),
        # not a parser bug. PN392 is unnecessary and was itself part of the
        # active wrap stack. Retired; version-capped so it never engages.
        "lifecycle": "retired",
        "retired_reason": "misdiagnosis — dev491 native parser self-sufficient; regression was Genesis dev259-era wraps (P64/P61c/PN56), not the parser",
        "superseded_by": "dev491 upstream native Qwen3CoderToolParser (self-sufficient for streaming; #45171). PN392 was a misdiagnosis — the regression was Genesis's own dev259-era qwen3coder wraps, not the parser. Proven by live raw smoke 2026-06-14.",
        # widened to <0.23.0 2026-06-19 (dev148 TIER-1 audit): tool_parsers/
        # qwen3coder_tool_parser.py + gemma4 parser DELETED by #45588; engine
        # state machine supersedes. The prior <0.22.1rc1.dev491 bound did NOT
        # exclude the 0.23.1 dev148 pin (version-semantics gap).
        "vllm_version_range": (">=0.20.0", "<0.23.0"),
        "category": "tool_parsing",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.tool_parsing.pn392_qwen3coder_streaming_coalesce",
        "credit": (
            "Genesis-original 2026-06-13 (dev491 pin-bump promotion blocker). "
            "Fixes the streaming tool-call regression introduced by the "
            "dev259->dev491 (0.22.1rc1.dev259+g303916e93 -> "
            "0.22.1rc1.dev491+g1033ffac2) upstream refactor: vllm#45171-era "
            "deleted tool_parsers/qwen3xml_tool_parser.py and remapped the "
            "qwen3_xml registry key from the dedicated Qwen3XMLToolParser to "
            "Qwen3CoderToolParser (tool_parsers/__init__.py). The coder "
            "parser's extract_tool_calls_streaming is single-emission "
            "(one structural delta per call, returns None to advance) and "
            "assumes token-by-token feeding. But parser/abstract_parser.py "
            "parse_delta feeds the WHOLE accumulated tool XML as ONE "
            "delta_text at the reasoning->tool boundary (sets "
            "tool_call_text_started, delta_text=current_text). On that single "
            "call the coder parser flips is_tool_call_started and returns "
            "None, emitting ZERO delta.tool_calls — the tool call is silently "
            "dropped (finish_reason=stop, no tool_calls). NON-streaming is "
            "unaffected. The dev259 Qwen3XMLToolParser coalesced multiple "
            "deltas per call via _merge_new_deltas_to_single_response and did "
            "NOT have this defect. PN392 restores that semantics: a runtime "
            "monkey-patch wraps extract_tool_calls_streaming on BOTH "
            "Qwen3CoderToolParser AND Qwen3XMLToolParser (dual-target so it "
            "works on dev259 PROD and dev491 candidate regardless of which "
            "class qwen3_xml/qwen3_coder resolves to), calling the original "
            "core once then draining it while pending tool structure remains "
            "(in_function open / unprocessed <tool_call> starts / "
            "closed-but-not-advanced tool), merging every emitted "
            "DeltaToolCall into ONE DeltaMessage. Token-by-token happy path, "
            "pure-content passthrough, and multi-tool-in-one-delta all "
            "preserved (TDD: 7 streaming-coalescing scenarios + lifecycle). "
            "Pure control-flow wrap, no text anchor -> no drift markers; "
            "opaque to dynamo (API-server process, not compiled forward "
            "path); idempotent class-marker; self-retires per class on "
            "upstream _within_call_coalescing drift. Composes with PN287 "
            "(same method, read-only observer; order-robust) and P107 (which "
            "becomes the inert safety net once PN392 makes tools_streamed[i] "
            "True). default_on=False (runtime monkey-patch convention); "
            "STRONGLY recommended ON for streaming tool-call workloads on "
            "dev491. No upstream PR coalesces the coder parser's streaming "
            "within a single call as of 2026-06-13."
        ),
        "upstream_pr": None,
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["PN287", "P107", "PN288"],
        "applies_to": {
            "tool_call_parser": ["qwen3_coder", "qwen3_xml"],
        },
    },
    "PN394": {
        "title": "qwen3 partial-param value `<` truncation fix (vllm#46047)",
        "tier": "community",
        "family": "tool_parsing",
        "env_flag": "GENESIS_ENABLE_PN394_QWEN3_PARTIAL_PARAM_LT_FIX",
        "default_on": True,
        "category": "stability",
        "credit": (
            "Backport of vllm#46047 (MERGED 2026-06-18, AFTER our "
            "0.23.1rc1.dev148+gb4c80ec0f pin — so NOT in the deployed "
            "engine). On dev148 the engine-native qwen3 parser "
            "vllm/parser/qwen3.py builds the streaming partial-parameter "
            "regex with the value group `([^<]*)$`, which stops at the "
            "first `<`. A still-open (partial=True) tool-call argument "
            "value containing a literal `<` (code `if a < b`, HTML "
            "`<div>`, generics `List<T>`) is SILENTLY TRUNCATED at that "
            "`<` — the model emits `{\"expr\": \"a < b\"}` but the client "
            "receives `{\"expr\": \"a \"}`. Real tool-call correctness "
            "bug on the hot streaming path: our 27B/35B presets run "
            "--tool-call-parser qwen3_xml which resolves to this engine "
            "parser on 0.23.x. PN394 is a byte-exact text-patch of the "
            "single line, widening the value group to `(.*)$` (re.DOTALL "
            "already spans newlines) — verbatim the #46047 fix. "
            "required=True (anchor present on dev148, byte-verified "
            "count==1 against vllm/parser/qwen3.py@b4c80ec0f); a missing "
            "anchor means parser drift and the patch SKIPs loudly. "
            "Self-skips once a future pin carries #46047: the post-fix "
            "spelling `>(.*)$` is the upstream_drift_marker (checked AFTER "
            "the idempotency marker, so it never trips on PN394's own "
            "output). Default ON — a pure correctness widening with no "
            "behavior change on values that contain no `<`."
        ),
        "upstream_pr": 46047,
        "upstream_pr_relationship": "backport",
        # RETIRED 2026-06-24 (pin bump dev148 -> dev301): #46047 is IN
        # dev301 (deep-diff iron-rule-#11 + dev301 anchor-SOT regen
        # confirm the anchor drifted on dev301). The runtime drift-marker
        # already self-skips on dev301; this is the FORMAL retire marking.
        # Cap the applies_to range upper bound to <dev301 so the version
        # gate also self-skips on dev301+, AND mirror it at top level for
        # iron-rule-#11 provenance. Still applies through dev148 (the
        # previous/rollback pin), where the bug is live.
        "applies_to": {"vllm_version_range": (">=0.23.0", "<0.23.1rc1.dev301")},
        "vllm_version_range": "<0.23.1rc1.dev301",
        "superseded_by": "vllm#46047 (MERGED 2026-06-18, IN dev301 — qwen3 partial-param value group widened to (.*)$; anchor drifted on dev301 per the anchor-SOT regen). Applies through dev148; retired on dev301+.",
        "apply_module": "sndr.engines.vllm.patches.tool_parsing.pn394_qwen3_partial_param_lt_fix",
        "lifecycle": "retired",
        "implementation_status": "full",
    },
    "P89": {
        "title": "completion_tokens_details.reasoning_tokens in chat usage (vendor of vllm#45471)",
        "tier": "community",
        "family": "serving",
        "env_flag": "GENESIS_ENABLE_P89_REASONING_TOKENS_USAGE",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "observability",
        "implementation_status": "full",
        "source": "vllm_pr_backport",
        "apply_module": "sndr.engines.vllm.patches.serving.p89_reasoning_tokens_usage",
        "credit": (
            "PR-sweep batch-3 (2026-06-13). Vendor of OPEN/DRAFT vllm#45471 "
            "([Frontend] Add completion_token_details to usage object in Chat "
            "Completions response body, nv-nedelman-1). The /v1/chat/completions "
            "usage object exposes prompt_tokens_details but NOT "
            "completion_tokens_details; OpenAI's chat API surfaces "
            "completion_tokens_details.reasoning_tokens so a caller can attribute "
            "decode cost between chain-of-thought and answer. Our /v1/responses "
            "path already surfaces it; the chat path every Genesis client uses "
            "does not. All 4 PROD models run the qwen3 reasoning parser, so this "
            "is a TPOT-attribution lever (reasoning vs answer) and an MTP/"
            "TurboQuant tuning denominator at ZERO GPU cost (one O(n) token-id "
            "walk via the parser's existing count_reasoning_tokens). Genesis "
            "extension over the PR: adds the OpenAI-spec accepted/"
            "rejected_prediction_tokens schema fields (default None) for future "
            "per-request MTP K=3 spec-decode efficiency — shipped as schema only "
            "(per-request accept counts are an engine-step SpecDecodingStats "
            "aggregate, not on RequestOutput in this pin). Bench plumbing lands "
            "in tools/genesis_chat_matrix_bench.py (reason-tok column). Atomic "
            "two-file MultiFilePatchTransaction (protocol.py model + serving.py "
            "import/attach). Self-skips on #45471 merge (drift markers = the "
            "PR's exact model/count lines)."
        ),
        "upstream_pr": 45471,
        "upstream_pr_relationship": "backport",
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": [],
        # 2026-06-17 (0.23.1 reverify): kept capped <0.23.0. Bug live
        # (#45471 OPEN). REDESIGNED 2026-06-17: new dev101 stream+full attach
        # variants on serving.py (_make_prompt_tokens_details), verified
        # byte-exact count==1 on live; cap bumped <0.24.0.
        "applies_to": {"vllm_version_range": (">=0.22.0", "<0.24.0")},
        "vllm_version_range": (">=0.22.0", "<0.24.0"),
    },
    "PN97": {
        "title": "PN97 — physical-cap on KV tensor allocation (Phase 7 PoC)",
        "tier": "community",
        "family": "kv_cache",
        "env_flag": "GENESIS_ENABLE_PN97_TENSOR_PHYSICAL_CAP",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "memory",
        "apply_module": "sndr.engines.vllm.patches.kv_cache.pn97_tensor_physical_cap",
        "source": "genesis_original",
        "credit": (
            "Genesis-original Phase 7 PoC — missing Anchor #10. Caps "
            "each KVCacheTensor.size to physical GPU budget so VIRT=1 "
            "inflation does NOT cause CUDA OOM. Per-tensor cap derived "
            "from torch.cuda.mem_get_info(0) × 0.80 / n_tensors, or "
            "operator override via GENESIS_PN97_PHYSICAL_CAP_GIB. "
            "Prerequisite for 156K+ single-card via virtual block "
            "addressing — full support also requires PN98 (attention "
            "block_id translation) which is future work."
        ),
        "applies_to": {
            "vllm_version_range": (">=0.20.2rc1.dev9", "<0.24.0"),
        },
        "requires_patches": [],
        "conflicts_with": [],
        "implementation_status": "full",
    },
    "PN96": {
        "title": "PN96 — emergency-demote hook (Phase 6 PoC; allocator intercept)",
        "tier": "community",
        "family": "kv_cache",
        "env_flag": "GENESIS_ENABLE_PN96_EMERGENCY_DEMOTE",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "memory",
        "apply_module": "sndr.engines.vllm.patches.kv_cache.pn96_emergency_demote",
        "source": "genesis_original",
        "credit": (
            "Genesis-original Phase 6 PoC. Inserts an emergency-rescue "
            "branch at the entry of BlockPool.get_new_blocks: when vllm "
            "would raise 'Cannot get N free blocks', PN96 walks the "
            "free queue for already-cached (ref_cnt=0, block_hash!=None) "
            "blocks, captures their bytes to PN95 L2 via demote_on_evict, "
            "and clears the hash so vllm sees clean free slots. Recovers "
            "the engine from the allocation cliff without preempting "
            "active sequences. Helps multi-prefix workloads; does NOT "
            "extend single-user max_model_len above the physical pool — "
            "that requires virtual block_table addressing inside attention "
            "(Phase 7 / scheduler refactor)."
        ),
        "applies_to": {
            "vllm_version_range": (">=0.20.2rc1.dev9", "<0.24.0"),
        },
        "requires_patches": [],
        "conflicts_with": [],
        "implementation_status": "full",
    },
    "PN92": {
        "title": "PN92 — nixl_ep/deep_ep/mori trial-import guard (vllm PR #40154)",
        "tier": "community",
        "family": "worker",
        "env_flag": "GENESIS_ENABLE_PN92_NIXL_EP_TRIAL_IMPORT",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "loader",  # was "compat"; normalize (trial-import guard for nixl_ep/deep_ep/mori loader-time)
        "apply_module": "sndr.engines.vllm.patches.worker.pn92_nixl_ep_trial_import",
        "source": "genesis_original",
        "credit": (
            "Genesis-original backport of upstream vllm PR #40154 / "
            "issue #42525. New nightlies (>= dcacdf9a 2026-05-13) ship "
            "nixl_ep C++ extension compiled against CUDA 12 inside a "
            "CUDA-13 image. find_spec-only check in has_nixl_ep() lets "
            "the broken import cascade through fused_moe → all2all_utils "
            "and break ALL hybrid-MoE models (Qwen3.5/3.6, DeepSeek, "
            "Mixtral) on inspect. PN92 replaces with try/except trial "
            "import. Until upstream PR lands."
        ),
        "applies_to": {
            "vllm_version_range": (">=0.20.2rc1.dev209", "<0.24.0"),
        },
        "requires_patches": [],
        "conflicts_with": [],
        "implementation_status": "full",
    },
    "PN71": {
        "title": "PN71 — `</thinking>` hallucination runtime normalizer",
        "tier": "community",
        "family": "reasoning",
        "env_flag": "GENESIS_ENABLE_PN71_THINKING_TAG_NORMALIZE",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "reasoning",  # was "compat"; normalize (reasoning-subsystem thinking-tag hallucination normalizer)
        "apply_module": "sndr.engines.vllm.patches.reasoning.pn71_thinking_token_hallucination",
        "source": "genesis_original",
        "credit": (
            "Genesis-original parser-side normalizer for Qwen 3.6 "
            "`</thinking>` hallucination. The model occasionally emits "
            "the full word instead of the canonical `</think>` token. "
            "Without this patch, Qwen3ReasoningParser.extract_reasoning "
            "routes ALL output to reasoning channel with content=None. "
            "PN71 normalizes the tag at parser entry so the partition "
            "logic stays correct regardless of which chat template is "
            "active. Complements froggeric template fix (which handles "
            "the prompt side); PN71 handles the live-generation side."
        ),
        "applies_to": {
        # 2026-06-17 (0.23.1 reverify): kept capped <0.23.0. Target file
        # reasoning/qwen3_reasoning_parser.py was DELETED by the #45588
        # parser reorg (now parser/qwen3.py token state-machine). Bug is
        # REDESIGNED 2026-06-17 for 0.23.1 (verified byte-exact on live): the
        # fix now targets parser/qwen3.py via a Qwen3Parser._preprocess_feed
        # override (</thinking>-></think> normalize). Cap bumped <0.24.0.
            "vllm_version_range": (">=0.20.2rc1.dev9", "<0.24.0"),
        },
        # [2026-06-17 redesign] On >=0.23.0 PN71 targets parser/qwen3.py
        # (the new token state-machine parser), NOT P27's engine-deleted
        # reasoning/qwen3_reasoning_parser.py. PN71's anchor no longer
        # contains P27's injected text (live-verified on dev148) and
        # applies standalone. requires_patches=["P27"] is retained only as
        # a legacy ordering hint for older pins where both co-existed; on
        # >=0.23.0 P27 is version-capped <0.23.0 (out of range), so the
        # dep-graph emits a HARMLESS advisory dep_missing WARNING that
        # never blocks (failed=0) — the dependency is structurally
        # unsatisfiable on this pin, not a wiring error.
        "requires_patches": ["P27"],
        "conflicts_with": [],
        "implementation_status": "full",
    },
    "PN73": {
        "title": "PN73 — safe `tool_calls.arguments` string→dict normalization",
        "tier": "community",
        "family": "serving",
        "env_flag": "GENESIS_ENABLE_PN73_TOOL_ARGS_SAFE_NORMALIZE",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "tool_parsing",  # was "compat"; normalize (tool_calls.arguments string→dict normalization)
        "apply_module": "sndr.engines.vllm.patches.serving.pn73_tool_args_safe_normalize",
        "source": "genesis_original",
        "credit": (
            "Genesis-original defensive normalizer for malformed "
            "tool_calls.arguments. Upstream chat_utils.py runs unguarded "
            "json.loads which raises HTTP 500 on (a) non-strict JSON, "
            "(b) non-string scalars, (c) double-encoded payloads. PN73 "
            "wraps in try/except and keeps the original string on failure "
            "instead of 500'ing. Strictly defensive."
        ),
        "applies_to": {
            "vllm_version_range": (">=0.20.2rc1.dev9", "<0.24.0"),
        },
        "requires_patches": [],
        "conflicts_with": [],
        "implementation_status": "full",
    },
    "PN91": {
        "title": "PN91 — `developer` role pre-render normalizer (OpenAI Responses API compat)",
        "tier": "community",
        "family": "serving",
        "env_flag": "GENESIS_ENABLE_PN91_DEVELOPER_ROLE",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "request_middleware",  # was "compat"; normalize (OpenAI `developer` role pre-render normalizer in request path)
        "apply_module": "sndr.engines.vllm.patches.serving.pn91_developer_role_normalizer",
        "source": "genesis_original",
        "credit": (
            "Genesis-original fix for OpenAI Responses API `role=developer` "
            "support. Maps developer→system at parser layer "
            "(_parse_chat_message_content) BEFORE chat template renders, so "
            "the fix holds regardless of which chat template is active — "
            "complements froggeric/enhanced jinja but does not require them."
        ),
        "applies_to": {
            "vllm_version_range": (">=0.20.2rc1.dev9", "<0.24.0"),
        },
        "requires_patches": [],
        "conflicts_with": [],
        "implementation_status": "full",
    },
    "PN82": {
        "title": "Mamba CUDA-graph stale `is_prefilling` padded rows — vllm#41873 backport (RETIRED — merged in dev371→626fa9bb window)",
        "tier": "community",
        "family": "worker",
        "env_flag": "GENESIS_ENABLE_PN82_MAMBA_CUDAGRAPH_PREFILL_ZERO",
        "default_on": False,
        "lifecycle": "retired",
        "category": "perf_hotfix",
        "apply_module": "sndr.engines.vllm._archive.pn82_mamba_cudagraph_prefill_zero",
        "superseded_by": (
            "vllm#41873 (merged 2026-05-21T22:42:43Z at commit "
            "39d5fa96a7c687f9ed7e14a5a52064965356cede — in window dev371 → "
            "626fa9bba566; K.1.R deep-diff 2026-05-28 confirmed byte-equivalent "
            "`is_prefilling[num_reqs:] = False` insertion at the same "
            "post-assignment location in vllm/v1/worker/gpu_model_runner.py:2270 "
            "— Genesis 3-line in-place comment vs upstream 2-line comment is "
            "the only cosmetic delta, functional change identical)"
        ),
        "vllm_version_range": "<0.21.1rc0+g626fa9bba5",
        "credit": (
            "Backport of vllm-project/vllm#41873 (OPEN as of 2026-05-07). "
            "After CUDA-graph batch padding via condense(), the boolean "
            "`is_prefilling` slice keeps stale True values for padded "
            "rows beyond num_reqs. Mamba/hybrid attention backends read "
            "this slice to decide between prefill vs decode kernel paths "
            "and would treat padding rows as prefill, occasionally "
            "corrupting Mamba state on hybrid models. Fix is a single "
            "extra line right after the existing assignment: "
            "`is_prefilling[num_reqs:] = False`. Tiny, safe, default OFF "
            "until live smoke on hybrid CUDA-graph configs (27B INT4 + "
            "DFlash; 35B-A3B-FP8 not affected — no Mamba). Genesis "
            "contribution: integration, gating, idempotent TextPatch, "
            "drift markers, tests. "
            "[Phase 3D 2026-05-22] Upstream vllm#41873 merged on "
            "2026-05-21 at commit "
            "39d5fa96a7c687f9ed7e14a5a52064965356cede. The merge is "
            "199 commits AHEAD of our current dev371 baseline "
            "(bf610c2f56764e1b30bc6065f4ceace3d6e59036), confirmed "
            "by `gh api .../compare/bf610c2f5...39d5fa96a` and by "
            "direct inspection of gpu_model_runner.py at dev371 "
            "(the `is_prefilling[num_reqs:] = False` line is NOT "
            "present at line 2233+ in the dev371 source). PN82 is "
            "therefore still load-bearing on the current pin — do "
            "NOT retire yet. The patch's existing upstream_drift_markers "
            "(`is_prefilling[num_reqs:] = False`) will auto-trigger "
            "apply()'s `skipped` return path on the next pin bump "
            "that pulls in commit 39d5fa96a or a successor containing "
            "the upstream fix. Functional change is byte-identical "
            "vs upstream (Genesis adds a 3-line in-place comment; "
            "upstream uses 2 lines; both inject the same single "
            "`is_prefilling[num_reqs:] = False` line at the same "
            "post-assignment location). Upstream additionally adds "
            "a regression test in tests/v1/attention/test_attention_splitting.py "
            "(+65 LOC); Genesis does not mirror that test."
        ),
        "upstream_pr": 41873,
        "upstream_pr_relationship": "backport",
        "applies_to": {"is_hybrid": [True]},
        "implementation_status": "full",
    },
    "PN55": {
        "title": "wake_up nested KV cache crash fix — vllm#41602+#41896 unified backport",
        "tier": "community",
        "family": "worker",
        "env_flag": "GENESIS_ENABLE_PN55_WAKE_UP_HYBRID_KV",
        "default_on": False,
        "category": "perf_hotfix",
        "apply_module": "sndr.engines.vllm.patches.worker.pn55_wake_up_hybrid_kv",
        "credit": (
            "PR38 Day 2 (2026-05-08) upgrade from PN55v1 → PN55v2. "
            "Unified backport of vllm-project/vllm#41602 (kevglynn / "
            "Mistral, OPEN as of 2026-05-04) AND vllm-project/vllm#41896 "
            "(nested KV cache class). v1 only handled `list[Tensor]` "
            "for Mamba/DeltaNet hybrid; #41896 surfaced the same bug "
            "class with `tuple` and `Mapping` (block-scaled FP8 KV). "
            "Both PRs target the SAME wake_up zeroing site, so a "
            "separate PN83 would conflict on the anchor. PN55v2 collapses "
            "them into one recursive iterator that zero-s only real "
            "tensors and silently skips None / non-tensor sentinels. "
            "Affects 27B Lorbus Qwen3.6 hybrid (GDN = MambaSpec layers) "
            "and any future FP8 nested-KV deployment. Crash trigger: "
            "/sleep + /wake_up via management API. Genesis active "
            "scripts don't use sleep, but defensive backport recommended "
            "for any external mgmt-API trigger. Default OFF; enable "
            "when sleep/wake actively used."
        ),
        "upstream_pr": 41602,
        "upstream_pr_relationship": "backport",
        # 2026-06-13 wave-2: #44778 (exec-patched-text regression-test
        # technique + companion #44779 review) added per the upstream
        # review of the same wake_up zeroing site.
        "related_upstream_prs": [41896, 44778],
        "applies_to": {},
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN54": {
        "title": "GDN contiguous-call deduplication (P0.7 Cliff 2b OOM mitigation) — RETIRED 2026-06-11",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_PN54_GDN_CONTIGUOUS_DEDUP",
        "default_on": False,
        "category": "perf_hotfix",
        # [Retire 2026-06-11, preflight residual triage §3, byte-verified]
        # Both sub-anchors are upstream-native / gone on pin
        # 0.22.1rc1.dev259+g303916e93: sub-A — pristine
        # gdn/qwen_gdn_linear_attn.py:1504-1514 gathers WITHOUT
        # .contiguous() ("initial_state = ssm_state[prefill_state_indices]"
        # at :1513); sub-B — the LoRA chunk-halves branch was removed
        # upstream (only the CPU assert "lora isn't supported on CPU."
        # remains at :1023). SSM_STATE_OLD / LORA_BA_OLD both count=0
        # (byte-verified). Latent self-substring marker bug (marker text
        # appears inside the pristine anchor → false-skip PRE-apply) fed
        # to the §6 self-collision lint corpus. Do NOT chase the
        # b/a.contiguous() pair at pristine 942-943 — different site,
        # real copy (PN350/PN365 territory). conflicts_with PN79 kept
        # (documents the historical double-pad interaction).
        "superseded_by": "upstream-native dedup — sub-A gather without .contiguous() (pristine gdn/qwen_gdn_linear_attn.py:1513); sub-B LoRA branch removed (CPU assert only at :1023); anchors count=0 (byte-verified 2026-06-11)",
        "vllm_version_range": "<0.22.1rc1.dev259",  # anchors dead on this pin's pristine tree
        "credit": (
            "Genesis-original 2026-05-04, inspired by MLX-LM PR #1077 "
            "(adurham, MIT) root-cause analysis: shared-buffer/slice-keeps-"
            "parent-alive class of bug. Removes 2 redundant `.contiguous()` "
            "calls in `gdn_linear_attn.py` already guaranteed contiguous by "
            "operator semantics OR re-enforced by FLA `@input_guard`. "
            "Sub-A: `ssm_state[non_spec_state_indices_tensor].contiguous()` "
            "(line ~985) — advanced index already produces fresh allocation; "
            "saves one full ssm_state-shape copy per prefill batch. Sub-B: "
            "LoRA branch `b/a.contiguous()` after `chunk(2, -1)` (lines 551-"
            "552) — chunk on last dim returns contiguous halves; LoRA-only, "
            "no-op on Genesis non-LoRA PROD. Target: Cliff 2b multi-turn "
            "OOM (Genesis Issue #19) — observed +1400 MiB/turn allocator "
            "delta; estimated saving 300-600 MiB/turn. Models: 27B Lorbus "
            "INT4 (all configs with GDN) — sub-A fires; 35B Qwen3MoE no GDN "
            "— patch never fires. Default OFF until live A/B Cliff 2b "
            "reproducer shows per-turn delta drops below ~900 MiB."
        ),
        "upstream_pr": None,
        "applies_to": {
            "model_class": ["qwen3_5", "qwen3_6"],
        },
        # Symmetric with PN79's declaration (audit 2026-05-12): PN79 wraps
        # `chunk_gated_delta_rule` while PN54 dedups `.contiguous()` calls
        # in the same gdn_linear_attn.py codepath. Together they double-pad
        # the same allocation in some prefill regimes.
        "conflicts_with": ["PN79"],
        "apply_module": "sndr.engines.vllm._archive.pn54_gdn_contiguous_dedup",
        "lifecycle": "retired",
        "implementation_status": "full",
    },
    "PN52": {
        "title": "prompt_logprobs eviction fix during chunked prefill (vllm#41411 backport)",
        "tier": "community",
        "family": "worker",
        "env_flag": "GENESIS_ENABLE_PN52_PROMPT_LOGPROBS_EVICTION",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Backport of vllm-project/vllm#41411 (MERGED 2026-05-04 18:46 UTC "
            "by Joachim Studnia, Mistral). Fixes TWO bugs in v1 gpu_worker "
            "prompt_logprobs path: (1) overly aggressive `-1` in "
            "`includes_prompt = computed_prefill < prompt_lens - 1` skipped "
            "the last prompt token's logprob when chunked-prefill boundary "
            "fell on `prompt_lens - 1`; (2) `in_progress_prompt_logprobs_cpu` "
            "stored on `input_batch` per-batch dict was lost on request "
            "eviction → silent corruption / IndexError on re-schedule. "
            "Multi-file text-patch: prompt_logprob.py + gpu_input_batch.py "
            "(field move) + gpu_model_runner.py (read/write per-request). "
            "Affects Genesis configs that combine `--enable-chunked-prefill` "
            "(all of ours) + spec-decode (MTP K=3 on PROD) + clients passing "
            "`prompt_logprobs=N`. Default OFF until live verify with "
            "an OpenAI-compatible streaming-client workload that "
            "exercises prompt_logprobs."
        ),
        "upstream_pr": 41411,
        "upstream_pr_relationship": "backport",
        "superseded_by": "vllm#41411 (merged 2026-05-04, byte-equivalent on dev209+g5536fc0c0)",
        "vllm_version_range": "<0.20.2rc1.dev209",  # mirrors applies_to (gated SoT); reconciled D17 2026-06-17 (dropped cosmetic +g5536fc0c0 suffix)
        "apply_module": "sndr.engines.vllm._archive.pn52_prompt_logprobs_eviction",
        "applies_to": {
            # [Genesis pin-gate 2026-05-11 iron rule #11] Deep-diff'd
            # upstream code in dev209: both fixes from PN52 now in vllm
            # natively — `< prompt_lens` (no -1) in prompt_logprob.py,
            # `in_progress_prompt_logprobs_cpu` moved to CachedRequestState
            # (gpu_input_batch.py:54 + gpu_model_runner.py:5200/5207/5264).
            # Auto-skip working via wiring's drift detector. Pin-gate adds
            # explicit boundary for `genesis explain` / audit reports.
            # Cleanup: delete patch file in next refactor pass.
            "vllm_version_range": "<0.20.2rc1.dev209",
        },
        "lifecycle": "retired",  # 2026-05-11: byte-equivalent with #41411 in dev209
        "implementation_status": "full",
    },
    # PN51 (Qwen3 streaming enable_thinking=false content routing,
    # vllm#40816/#40820) was consolidated into the P61b entry on
    # 2026-06-20 — all three reasoning parser patches (P61b + P59 + PN51)
    # share one apply_module (p61b_p59_pn51_qwen3_reasoning_consolidated).
    # PN51's enable flag GENESIS_ENABLE_PN51_QWEN3_STREAMING_THINKING_DISABLED
    # is retained as an env_flag_alias on P61b so existing YAML opt-ins keep
    # engaging the merged module.
    "PN252": {
        "title": "M-RoPE prompt_embeds-only DoS fix (vllm#45252 / GHSA-33cg-gxv8-3p8g)",
        "tier": "community",
        "family": "worker",
        "env_flag": "GENESIS_ENABLE_PN252_MROPE_PROMPT_EMBEDS_DOS",
        "default_on": True,
        "category": "stability",
        "credit": (
            "Backport of vllm#45252 (security advisory GHSA-33cg-gxv8-3p8g). "
            "GpuModelRunner._init_mrope_positions asserted "
            "`req_state.prompt_token_ids is not None`; on any M-RoPE model a "
            "single /v1/completions request with prompt_embeds set and "
            "prompt=None tripped the assert and crashed EngineCore — an "
            "unprivileged remote DoS. The fix drops the fatal assert and "
            "derives a non-None token sequence at the call site (real "
            "prompt_token_ids when present, else dummy positional IDs sized "
            "from prompt_embeds — M-RoPE only needs the LENGTH for a "
            "passthrough modality without grid_thw; clean ValueError when "
            "BOTH are absent). Directly relevant on the PROD fleet: Gemma-4 "
            "26B-A4B / 31B-AWQ are M-RoPE and accept prompt_embeds. Host-side "
            "Python assertion — identical on Ampere A5000 / Ada / Hopper / "
            "Blackwell 5090, so NOT arch-gated (recipe E, universal "
            "correctness/security guard). Verified byte-identical anchor on "
            "dev259 (PROD) + dev491 (candidate); self-skips when upstream "
            "merges the fix (anchor stops matching). default_on=True "
            "(informational under strict-opt-in) — engage via "
            "GENESIS_ENABLE_PN252_MROPE_PROMPT_EMBEDS_DOS=1."
        ),
        "upstream_pr": 45252,
        "upstream_pr_relationship": "backport",
        "applies_to": {
            # vllm#45252 / GHSA-33cg-gxv8-3p8g MERGED upstream at/before
            # b4c80ec0f (dev148): _init_mrope_positions now derives a non-None
            # input_tokens (range(prompt_embeds.shape[0]) for embeds-only) and
            # raises a per-request ValueError instead of the fatal assert — the
            # DoS is closed BY THE ENGINE. Cap <0.23.0 so PN252 version-gate-
            # skips cleanly on dev148+ (was an incidental anchor-miss skip
            # because it had no cap) while still auto-applying (default_on) on
            # rollback pins <0.23.0 where the buggy assert remains. Verified
            # 2026-06-19: engine fix present at gpu_model_runner.py:1648/1652;
            # both PN252 required anchors count=0 on dev148 (self-skip, no
            # interim risk). Lower bound >=0.20.0 matches the sibling host-side
            # parser/worker patches (P61b/PN56/PN287).
            "vllm_version_range": (">=0.20.0", "<0.23.0"),
        },
        "apply_module": "sndr.engines.vllm.patches.worker.pn252_mrope_prompt_embeds_dos",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN517": {
        "title": "Take init MemorySnapshot before NCCL — asymmetric TP+PP OOM guard + startup_free observability (vllm#45517)",
        "tier": "community",
        "family": "worker",
        "env_flag": "GENESIS_ENABLE_PN517_INIT_SNAPSHOT_BEFORE_NCCL",
        "default_on": False,
        "category": "memory",
        "credit": (
            "Backport of vllm#45517 (RFC #34303). Worker.init_device took "
            "its baseline MemorySnapshot AFTER init_worker_distributed_"
            "environment (NCCL), so on an asymmetric TP+PP topology — where "
            "a PP-terminal rank carries far more NCCL workspace than rank 0 "
            "— gpu_memory_utilization was budgeted against post-NCCL free "
            "memory and OOMed the heaviest rank on init. The opt-in env "
            "VLLM_INIT_SNAPSHOT_BEFORE_NCCL snapshots before NCCL and reuses "
            "it; pre-NCCL free bytes are stashed in self._startup_free_bytes "
            "for observability. Genesis-installed code reads the env via "
            "os.environ (no dependency on a vllm.envs entry the pin may "
            "lack). On our TP=2 PP=1 PROD the VRAM guard is dormant — the "
            "live value is startup observability; the guard future-proofs "
            "any PP>1 config. Host-side accounting — NOT arch-gated (recipe "
            "E/G). default_on=False; engage via "
            "GENESIS_ENABLE_PN517_INIT_SNAPSHOT_BEFORE_NCCL=1 AND set "
            "VLLM_INIT_SNAPSHOT_BEFORE_NCCL=1 to fire the pre-NCCL branch. "
            "Byte-verified identical on dev259 (PROD) + dev491 (candidate); "
            "self-skips when upstream merges."
        ),
        "upstream_pr": 45517,
        "upstream_pr_relationship": "backport",
        "applies_to": {},
        "apply_module": "sndr.engines.vllm.patches.worker.pn517_init_snapshot_before_nccl",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN35": {
        "title": "Skip inputs_embeds buffer for text-only models (vllm#35975 backport)",
        "tier": "community",
        "family": "worker",
        "env_flag": "GENESIS_ENABLE_PN35_INPUTS_EMBEDS_OPTIONAL",
        "default_on": True,
        "category": "perf_hotfix",
        "credit": (
            "Backport of vllm-project/vllm#35975 by AjAnubolu (OPEN since "
            "2026-03-04). Skips the (max_num_tokens, hidden_size) GPU "
            "buffer allocation for text-only models in BOTH "
            "gpu_model_runner (~64 MiB GPU + 64 MiB pinned CPU) AND "
            "llm_base_proposer spec-decode proposer (~64 MiB GPU). For "
            "Qwen3.6-27B at max_num_tokens=4096 and hidden_size=8192: "
            "freed ~128 MiB GPU + 64 MiB pinned CPU per worker. "
            "Particularly relevant on borderline-OOM single-24GB-GPU "
            "configs (Cliff 2 fires at 50 MiB-free thresholds) and "
            "WSL2 setups with extra display overhead. Pattern credit: "
            "noonghunna club-3090 setup-time sidecar "
            "patch_inputs_embeds_optional.py 2026-05-02. Originally "
            "raised by club-3090#32 (RossNE99, GuiPerPT WSL2 OOM "
            "reports). Default ON — strict memory savings, no "
            "regression possible (the `if` guard preserves original "
            "allocation for multimodal models). Auto-retires when "
            "vllm#35975 merges upstream."
        ),
        "upstream_pr": 35975,
        "upstream_pr_relationship": "backport",
        # Promoted 2026-05-12 (Wave 9 dev209 + STABLE-prep): full ratchet
        # satisfied — `register_for_manifest()` added in wiring;
        # anchor_manifest.json covers PN35.Sub-1 (gpu_model_runner.py) +
        # PN35.Sub-2 (llm_base_proposer.py) with pristine fixtures from
        # the dev209 image. Production-validated default_on across Wave
        # 6→9 + dev93/dev209 with zero regressions. Strict-superset
        # (text-only guard preserves multimodal path verbatim).
        # Upstream vllm#35975 still OPEN — Genesis durably ahead;
        # auto-retires when upstream merges (registry-driven gate).
        "apply_module": "sndr.engines.vllm.patches.worker.pn35_inputs_embeds_optional",
        "lifecycle": "stable",
        "stable_kind": "text-patch",
        "stable_since": "v11.0.0+wave9_dev209",
        "implementation_status": "full",
    },
    "PN34": {
        "title": "WorkspaceManager runtime lock relaxation (PN33 companion for runtime decode)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_PN34_WORKSPACE_LOCK_RELAX",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Companion to PN33 — same root cause class but on the runtime "
            "decode path. PN33 fixes BOOT-time _dummy_sampler_run "
            "under-counting; PN34 relaxes the strict "
            "WorkspaceManager._ensure_workspace_size AssertionError that "
            "still fires at runtime decode "
            "(turboquant_attn.py:1350:_decode_attention) on rare paths. "
            "Direct port of noonghunna's club-3090 setup-time sidecar "
            "patch_workspace_lock_disable.py. "
            "Retired 2026-05-14 PR sweep: this entry is a near-duplicate "
            "of SNDR_WORKSPACE_001 (same fault site, same warn+grow "
            "shape, same vllm.v1.worker.workspace anchor). Both were "
            "carried side-by-side during the v7 → v11 migration; the "
            "audit identified PN34 as the older variant. Consolidated "
            "into SNDR_WORKSPACE_001 going forward. Upstream replacement "
            "is vllm#42551 (jasonboukheir, DRAFT — pre-reserve + non-"
            "raising try API in _decode_attention) — see "
            "SNDR_WORKSPACE_001 credit for the merge-and-retire plan."
        ),
        "upstream_pr": 42551,  # 2026-05-14 PR sweep — same retire trigger as SNDR_WORKSPACE_001
        # relationship corrected 2026-05-30: PN34 was retired as an
        # INTERNAL duplicate of SNDR_WORKSPACE_001, NOT because
        # vllm#42551 merged (it's still DRAFT/OPEN). `backport` was
        # incorrect — it implied PR-merge would supersede us. The
        # correct framing is `related_not_superseding`: same fault
        # site as #42551, but the retire driver was de-duplication
        # with SNDR_WORKSPACE_001. audit_upstream_status now classifies
        # PN34 as RELATED-NOT-SUPERSEDING (informational) instead of
        # STALE-RETIRED (weird-state alarm).
        "upstream_pr_relationship": "related_not_superseding",
        "requires_patches": ["PN33"],
        "superseded_by": "SNDR_WORKSPACE_001",
        "lifecycle": "retired",  # 2026-05-14 PR sweep audit — duplicate of SNDR_WORKSPACE_001
        "vllm_version_range": ">=0.20.1rc1.dev16+g7a1eb8ac2,<0.20.2rc1.dev338+gbf0d2dc6d",  # was active on dev16-dev209; retired on dev338 pin in favor of SNDR_WORKSPACE_001
        "implementation_status": "retired",
    },
    "PN33": {
        "title": "Spec-decode warmup K-aware sizing (vllm#37521 extended to MTP/ngram)",
        "tier": "community",
        "family": "worker",
        "env_flag": "GENESIS_ENABLE_PN33_SPEC_DECODE_WARMUP_K",
        "default_on": True,
        "category": "spec_decode",
        "credit": (
            "Backport of vllm-project/vllm#37521 (itailang, OPEN at "
            "backport time 2026-05-02) EXTENDED beyond use_eagle() "
            "gate to cover all spec-decode methods (EAGLE + MTP + "
            "ngram + draft-model). Root-cause fix: gpu_model_runner."
            "_dummy_sampler_run() warmed up with dummy K=1 instead of "
            "real num_speculative_tokens, causing (a) KV-cache profile "
            "to over-estimate available headroom → mid-stream OOM via "
            "propose_draft_token_ids (ampersandru, club-3090#16 "
            "2026-05-01) AND (b) TurboQuant WorkspaceManager lock fails "
            "when real spec-decode tries to grow workspace beyond "
            "warmup-reserved size (noonghunna, club-3090 disc #19 "
            "2026-05-01). Same root cause for both bugs; one fix "
            "closes both. Default ON when spec-decode active — real "
            "correctness fix, not experimental. Disable via "
            "GENESIS_DISABLE_PN33_SPEC_DECODE_WARMUP_K=1 if K-sized "
            "warmup itself OOMs (better-than-runtime-OOM diagnosis)."
        ),
        "upstream_pr": 37521,
        "upstream_pr_relationship": "backport",
        "applies_to": {
            # Only fires when speculative_config is present at runtime.
            # The text-patch site itself is gated `if self.speculative_config:`
            # so non-spec-decode boots are NULL on this path.
        },
        # Promoted 2026-05-12 (Wave 9 dev209 + STABLE-prep): full ratchet
        # satisfied — `register_for_manifest()` added in wiring;
        # anchor_manifest.json covers PN33.Sub-1 (gpu_model_runner.py).
        # Correctness fix (not perf-experimental). Production-validated
        # default_on across Wave 6→9 + dev93/dev209 + both 27B+35B PROD
        # with zero regressions. EXTENDED upstream vllm#37521 beyond
        # use_eagle() to cover all spec-decode methods (EAGLE + MTP +
        # ngram + draft-model) so Genesis stays ahead until upstream
        # broadens its merge — disabling carries real correctness risk
        # on MTP/ngram/draft paths.
        "apply_module": "sndr.engines.vllm.patches.worker.pn33_spec_decode_warmup_k",
        "lifecycle": "stable",
        "stable_kind": "text-patch",
        "stable_since": "v11.0.0+wave9_dev209",
        "implementation_status": "full",
    },
    "PN32": {
        "title": "GDN _forward_core chunked-prefill v2 (Cliff 2 fix for single-24GB-GPU OOM)",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_PN32_GDN_CHUNKED_PREFILL",
        "default_on": False,
        "category": "hybrid",
        # NOTE: v2 (v7.69) supersedes v1 (v7.65). v1 chunked at the WRONG
        # level (forward_cuda outer, didn't propagate cu_seqlens to inner
        # FLA call → empirically OOM'd EARLIER on club-3090 cross-rig).
        # v2 chunks _forward_core directly with chunk-local cu_seqlens
        # and threaded initial_state. See club-3090#19 finding 3.
        #
        # COMPOSITION: v2 chunks the OUTER FLA call (chunk_gated_delta_rule).
        # P103 chunks INSIDE chunk_gated_delta_rule_fwd's h tensor. Both
        # default OFF, COMPLEMENTARY. Recommended together for single-24GB-
        # GPU users hitting Cliff 2 (>50K single-prompt prefill on 1×3090
        # /4090/5090).
        #
        # DEPENDENCY: P28 (legacy persistent buffer pool) conflicts with
        # PN32 v2 — both modify gdn_linear_attn.py overlapping paths.
        # Operator MUST disable P28 before enabling PN32. P28 IS in this
        # dispatcher (legacy lifecycle since v7.65) — symmetric back-link
        # declared via conflicts_with below.
        "credit": (
            "Genesis-original v7.69 v2 (2026-05-02) — Cliff 2 fix per "
            "noonghunna's CLIFF2_INVESTIGATION_20260430.md + cross-rig "
            "club-3090#19 finding 3. v2 supersedes v1 (v7.65) which "
            "chunked at wrong level. v2: when single-seq prefill T > "
            "16384 (env-tunable), splits chunk_gated_delta_rule call "
            "into chunks of 8192. Each chunk: slice query/key/value/g/"
            "beta along T, build chunk-local cu_seqlens=[0, chunk_len], "
            "thread initial_state via prior chunk's last_recurrent_state, "
            "concat outputs. Multi-seq prefill bypasses to original. "
            "Default OFF. Composes with P103 (P103 chunks INSIDE FLA "
            "kernel; PN32 chunks the FLA CALL). Recommended pairing for "
            "single-24GB-GPU Cliff 2: GENESIS_ENABLE_P103=1 + GENESIS_"
            "ENABLE_PN32_GDN_CHUNKED_PREFILL=1. Cross-rig validation "
            "required (our 2×A5000 PROD with TP=2 doesn't hit Cliff 2)."
        ),
        "upstream_pr": None,
        "applies_to": {
            # Triggers in any hybrid GDN model with long single-prompts.
            # NULL on non-GDN paths (no GDN layers in 35B Qwen3MoE).
        },
        "requires_patches": [],
        "conflicts_with": ["P28", "PN108"],  # PN32+PN108 both touch GDN chunked prefill orchestrator — incompatible
        "apply_module": "sndr.engines.vllm.patches.attention.gdn.pn32_gdn_chunked_prefill",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN102": {
        "title": "PrefetchOffloader pinned-allocator prewarm pool",
        "tier": "community",
        "family": "offload",
        "env_flag": "GENESIS_ENABLE_PN102_PARAM_POOL",
        "default_on": False,
        "category": "memory_pool",  # was "performance"; normalize (PrefetchOffloader pinned-allocator prewarm pool)
        # Relevant only when --cpu-offload-gb > 0. Without prewarm,
        # vllm's PrefetchOffloader calls torch.empty_strided(pin_memory=True)
        # once per offloaded param: 27B INT4 Qwen3.6 with cpu_offload_gb=8
        # has ~768 cudaHostAlloc calls × ~50 ms each ≈ 38 s of pure
        # pinning overhead before the model is ready. PN102 prewarms a
        # single contiguous slab (configurable via
        # GENESIS_PN102_PREWARM_MB, default 1024) so the subsequent
        # per-param pinnings are served from PyTorch's cached pinned
        # allocator — drops startup overhead to single-digit seconds.
        # Idempotent; harmless if cpu_offload is off (then the
        # PrefetchOffloader is never touched).
        "credit": (
            "Genesis-original 2026-05-14. Closes the 38 s startup-wall "
            "introduced by per-param `cudaHostAlloc` when "
            "`--cpu-offload-gb > 0` on AutoRound INT4 models. Companion "
            "to PN104 (cpu_offload → Prefetch redirect) and PN105 "
            "(AutoRound metadata exclusion)."
        ),
        "upstream_pr": None,
        "applies_to": {},
        "requires_patches": [],
        "conflicts_with": [],
        "apply_module": "sndr.engines.vllm.patches.offload.pn102_pinned_alloc_pool",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN204": {
        "title": "GDN dual-stream input projection (port of vllm#42301)",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_PN204_DUAL_STREAM_INPROJ",
        "default_on": False,
        "category": "kernel_perf",  # was "performance"; normalize (GDN dual-stream input projection kernel-level perf win)
        # PN204 replaces legacy P7 (lifecycle='legacy', boot status =
        # skipped/deferred — raw torch.cuda.Stream not SymPy-graphable
        # inside torch.compile fullgraph; wording aligned 2026-06-11
        # during the P7b retire: P7 was never formally retired).
        # PN204 uses upstream vllm.utils.multi_stream_utils
        # .maybe_execute_in_parallel which is torch.compile-safe and
        # available in the pinned nightly dcacdf9a.
        #
        # Direct port of vllm PR #42301. Single text-patch anchor on
        # gdn_linear_attn.py::forward_cuda Part 1: replaces the serial
        # in_proj_qkvz/in_proj_ba pair with the parallel helper. Stream
        # and events are created lazily on the first forward call (naive
        # __init__ allocation crashed worker with a torch.Event type
        # error in our pinned vLLM). Auto-SKIPs when upstream lands
        # #42301 (drift marker `_in_proj_aux_stream`).
        "credit": (
            "Port of vllm-project/vllm#42301 (open as of 2026-05-14). "
            "Upstream measures -2.9% TPOT at qps=0.5 on Qwen3.5-35B-A3B "
            "via overlapping in_proj_qkvz/in_proj_ba GEMMs on an aux "
            "CUDA stream. PN204 ports the same change as a Genesis "
            "text-patch to unblock the win without waiting for upstream "
            "merge."
        ),
        "upstream_pr": 42301,
        "upstream_pr_relationship": "backport",
        "applies_to": {
            # Hybrid GDN models (Qwen3.5/3.6) on CUDA-alike platforms.
        },
        "requires_patches": [],
        # Mutually exclusive with legacy P7 (same forward_cuda Part 1
        # target). Operator must keep P7 disabled when enabling PN204.
        # Also mutually exclusive with PN365 (port of vllm#42746 fuses
        # the two in_proj GEMMs into one — nothing to overlap, AND PN204
        # anchor no longer matches the patched file).
        "conflicts_with": ["P7", "PN365"],
        "apply_module": "sndr.engines.vllm.patches.attention.gdn.pn204_dual_stream_inproj",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN108": {
        "title": "GDN fused_recurrent prefill dispatch (Cliff 2 memory-bound fix)",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_PN108_FUSED_RECURRENT_PREFILL",
        "default_on": False,
        "category": "hybrid",
        # PN108 takes a different approach from PN32 / PN59. Instead of
        # chunking the chunk_gated_delta_rule call (PN32) or the inner
        # kernel orchestrator (PN59), PN108 switches BACKENDS for long
        # single-sequence prefill: it dispatches to fla's
        # fused_recurrent_gated_delta_rule, which has NO chunk-state h
        # buffer at all (the Cliff 2 OOM trigger).
        #
        # Trade-off: fused_recurrent is purely token-recurrent compute,
        # so prefill is ~3-8× slower above the threshold. In exchange:
        # zero h-tensor allocation, predictable memory across any T,
        # no fragmentation from a chunked-dispatch loop. For long-context
        # workflows on memory-constrained single-GPU rigs (1×A5000 24GB,
        # 1×3090, 1×4090), this is the only path that achieves stable
        # 150-200K prefill without TP=2.
        #
        # Mutually exclusive with PN32 (both patch the same _forward_core
        # prefill branch in gdn_linear_attn.py). PN108 also conflicts
        # with P28 for the same reason as PN32 — operator must disable
        # P28 before enabling PN108.
        "credit": (
            "Genesis-original 2026-05-14 — Cliff 2 memory-bound fix on "
            "single 24GB GPU. PN32/PN59 attempted to keep the chunkwise-"
            "parallel kernel and chunk dispatch around it; both either "
            "hit anchor drift (PN59, upstream added `core_attn_out` "
            "param) or added a torch.cat memory peak that didn't fit on "
            "saturated single-card budget (PN32). PN108 sidesteps by "
            "switching to fla.fused_recurrent_gated_delta_rule for long "
            "single-seq prefill — same output contract (B,T,HV,V), no "
            "chunk-state h buffer. Threshold env-tunable via GENESIS_"
            "PN108_FUSED_RECURRENT_THRESHOLD (default 32768)."
        ),
        "upstream_pr": None,
        "applies_to": {
            # Hybrid GDN/DeltaNet models with single-sequence long prefill.
            # NULL on non-GDN paths or short prompts.
        },
        "requires_patches": [],
        "conflicts_with": ["P28", "PN32"],
        "lifecycle": "retired",  # docstring says TOMBSTONED — fla recurrent kernel design conflict; synced to registry 2026-05-14
        "apply_module": "sndr.engines.vllm._archive.pn108_fused_recurrent_prefill",
        "retired_waiver": True,  # design conflict — see docstring tombstone
        "implementation_status": "full",
    },
    "PN31": {
        "title": "FA varlen persistent out buffer (issue #15, sister to P38)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_PN31_FA_VARLEN_PERSISTENT_OUT",
        "default_on": False,
        "category": "memory_pool",
        "credit": (
            "Genesis-original sister patch to P38 (K_full/V_full persistent "
            "buffers). Closes issue #15 — OOM at flash_attn_varlen_func on "
            "budget-constrained single-GPU configs. Per-shape persistent "
            "out buffer eliminates per-call malloc pressure inside FA C "
            "extension. Memory cost: ~16-64 MiB per shape × layer. NULL "
            "impact on 2×A5000 PROD (we have headroom); designed for "
            "1×3090 / 1×4090 single-GPU community users."
        ),
        "upstream_pr": None,
        "applies_to": {
            # Triggers when TurboQuant attention is active. NULL on
            # non-TQ paths (FP8 KV, BF16 KV).
        },
        "conflicts_with": [],
        "requires_patches": [],
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.pn31_fa_varlen_persistent_out",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN30": {
        "title": "DS conv state layout + spec-decode AL>1 fix (issue #17)",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_PN30_DS_LAYOUT_SPEC_DECODE",
        "default_on": False,
        "category": "model_correctness",
        "credit": (
            "Genesis-original fix for issue #17 (noonghunna, 2026-05-01). "
            "Replaces upstream NotImplementedError raise in "
            "`get_conv_copy_spec` for DS conv state layout + "
            "num_accepted_tokens > 1 (every spec-decode AL>1 prefill on "
            "DS-enabled hybrid GDN configs). Two-file text-patch: "
            "(1) mamba_utils.py uses .contiguous() copy + module-level "
            "temp tensor list; (2) v1/worker/mamba_utils.py wraps "
            "do_mamba_copy_block with stream sync + list clear after "
            "batch_memcpy. Cost: ~10-50us per batch when path active. "
            "Closes 50/50 LCB v6 failure on structured-CoT workloads."
        ),
        "upstream_pr": None,
        "applies_to": {
            # Triggers in any hybrid GDN model with DS layout + spec-decode.
            # Genesis A5000 PROD doesn't have --structured-outputs-config so
            # may not exercise this path; community single-3090 + structured
            # CoT does.
            #
            # 2026-06-17 (0.23.1 pin-bump, iron-rule #11 outcome-a — verified
            # by reading BOTH our part1 anchor AND live 0.23.1 source):
            # upstream REWROTE get_conv_copy_spec. The NotImplementedError
            # this patch worked around ("DS conv state layout does not yet
            # support speculative decoding ... num_accepted_tokens > 1") is
            # GONE — replaced by `assert offset == 0, "...must be handled by
            # the fused postprocess kernel, not get_conv_copy_spec"`. Upstream
            # now handles DS conv state + num_accepted_tokens > 1 natively via
            # a fused postprocess kernel (a superset of our memcpy workaround).
            # Confirmed live in 0.23.1rc1.dev101+g4c6266331
            # (mamba_utils.get_conv_copy_spec). The cap retires PN30's whole
            # 3-part apply() on >=0.23.0 via should_apply (all-or-nothing — a
            # part1-only drift-skip would leave the half-patched state the
            # apply() docstring forbids). PN30 stays ACTIVE on <0.23.0 (dev491
            # still raises NotImplementedError on the DS+align path).
            "vllm_version_range": (">=0.20.0", "<0.23.0"),
        },
        "conflicts_with": [],
        "requires_patches": [],
        "apply_module": "sndr.engines.vllm.patches.attention.gdn.pn30_ds_layout_spec_decode_align",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P67c": {
        "title": "Per-row vote sparse-V integration into P67 split-M kernel",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_P67_SPARSE_V",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Genesis-original 2026-05-01 — synthesizes PN26b proven uniform-"
            "scalar `if` pattern (Triton 3.6 scf.if), TRT-LLM #9821 sink "
            "protection design, TheTom #41422 threshold=0 bit-exact contract. "
            "Per-q_t skip via `if SPARSE_V: ...` constexpr-DCE'd to nothing "
            "at SPARSE_V=0 → byte-equivalent to pre-sparse-V P67. "
            "When SPARSE_V=1 + threshold=0: bit-exact (P_t = exp2(...) >= 0, "
            "so `p_t_max < 0` always False). When threshold > 0: per-q_t "
            "max-prob check skips V@P tl.dot for cold tiles past sink window. "
            "Greenfield in spec-decode K+1 verify (no upstream impl exists). "
            "Expected gain: +5-22% on long-context (16K+); NULL on short ctx."
        ),
        "upstream_pr": None,
        "applies_to": {"is_turboquant": [True]},
        "requires_patches": ["P67"],
        "conflicts_with": [],
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.p67c_sparse_v",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    # PN29 (GDN chunk_o scale-fold, vllm#41446 pattern (c)) was CONSOLIDATED
    # into the PN298 entry on 2026-06-19 (maintainability refactor — both
    # patches target the SAME engine file model_executor/layers/fla/ops/
    # chunk_o.py at disjoint regions). The PN29 scale-fold and PN298 arch-
    # aware NUM_WARPS prune now share one apply_module
    # (pn29_pn298_chunk_o_consolidated) with two independently-gated sub-
    # patches. The PN29 env flag (GENESIS_ENABLE_PN29_GDN_SCALE_FOLD) is
    # retained as a recognized alias on the PN298 entry's `env_flag_aliases`
    # so existing builtin YAMLs keep working unchanged. Runtime-neutral:
    # the applied kernel-code bytes are byte-identical to PN29+PN298 applied
    # separately (only the single shared wiring-marker comment differs). See
    # the PN298 entry below.
    "PN11": {
        "title": "GDN a/b contiguity in fix_query_key_value_ordering (vllm#41142)",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_PN11_GDN_AB_CONTIGUOUS",
        "default_on": False,
        "category": "model_correctness",
        "credit": (
            "Backport of vllm#41142 (Yeuvoir, OPEN). Fixes upstream issue "
            "#41112: in `GatedDeltaNetAttention.fix_query_key_value_ordering` "
            "the reshape of `b` and `a` returns a non-contiguous view when "
            "num_v_heads == num_k_heads (np/ng == 1), breaking "
            "`fused_post_conv_prep` Triton kernel which assumes head-dim "
            "stride 1. Adds `.contiguous()` to both lines (zero cost when "
            "already contiguous; copy only on the buggy path). Symptom on "
            "affected configs: silent quality drift, no crash. For Genesis "
            "prod (Qwen3.6 27B has np/ng=8, 35B has no GDN) this is "
            "DEFENSIVE — installs guard against future model swaps."
        ),
        "upstream_pr": 41142,
        "upstream_pr_relationship": "backport",
        "applies_to": {
            # Patch only matters when GDN layer's fix_query_key_value_ordering
            # runs with np/ng==1. Genesis prod doesn't trigger it but the
            # patch is harmless (no-op .contiguous() call).
        },
        "apply_module": "sndr.engines.vllm.patches.attention.gdn.pn11_gdn_a_b_contiguous",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN12": {
        "title": "FFN intermediate scratch pool — Cliff 1 fix on TQ3 path",
        "tier": "community",
        "family": "kernels",
        "env_flag": "GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL",
        "default_on": False,
        "category": "memory_savings",
        "credit": (
            "Genesis-original 2026-04-29 — Cliff 1 fix on TQ3 path. "
            "Closes 138 MiB OOM at 192K + tool-call on RTX 3090 (noonghunna "
            "report). PN8 closes Cliff 1 on FP8 by freeing ~600 MiB persistent "
            "draft VRAM, but on TQ3 frees only ~230 MiB — not enough slack "
            "for the 138 MiB transient. Different memory class. PN12 pools "
            "the SiluAndMul output across layers (single buffer per "
            "(intermediate_size, dtype, device)) — reduces per-step allocator "
            "churn from ~4.7-18 GiB to ~73-285 MiB on Lorbus 27B-int4. "
            "Pointer-stable (cudagraph-safe). Cross-engine reference: "
            "TensorRT-LLM live-range activation reuse (gold standard); "
            "alternative paths: vLLM PR #34207 (silu_and_mul.out variant), "
            "SGLang PR #15927 (piecewise CUDA graph private pool). Tested "
            "via 17 unit tests in tests/test_ffn_intermediate_cache.py."
        ),
        "upstream_pr": 34207,  # would obsolete this patch if merged
        "upstream_pr_relationship": "backport",
        "applies_to": {
            # Patch matters when SiluAndMul / MulAndSilu is on the hot path
            # (any model with FFN gate-up + silu activation — qwen3, llama,
            # mistral, deepseek, etc.). For MoE models impact is per-expert.

            # [Genesis pin-gate 2026-05-11] PROD-active (GroupAB component).
            # Validated dev9 → dev93. Self-retires when #34207 merges.
            "vllm_version_range": (">=0.20.0", "<0.24.0"),
        },
        "apply_module": "sndr.engines.vllm.patches.kernels.pn12_ffn_intermediate_pool",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN19": {
        "title": "Scoped max_split_size_mb during model load (vllm#41268)",
        "tier": "community",
        "family": "memory",
        "env_flag": "GENESIS_ENABLE_PN19_SCOPED_MAX_SPLIT",
        "default_on": False,
        "category": "memory_savings",
        "credit": (
            "Backport of vllm#41268 (MatthewBonanni, OPEN 2026-04-30). "
            "PyTorch 2.10+ introduced load-time allocator fragmentation: "
            "weight segments split inside other segments, leaving "
            "200-500 MiB unusable. Mitigation: temporarily set "
            "max_split_size_mb=20 (PyTorch minimum) for the duration of "
            "model load, restore prior on exit. Cudagraph-safe (load-"
            "time only; capture phase uses restored allocator). "
            "Default OFF — operator should measure fragmentation gap "
            "via nvidia-smi peak during load before vs after to confirm "
            "win on Ampere SM 8.6 (PR #41268 measured on H100; A5000 "
            "behavior unverified). Cross-reference Genesis memory "
            "feedback_p104_l2_persistence_thrashing — hardware-mismatch "
            "patches are anti-pattern; measure first."
        ),
        "upstream_pr": 41268,
        "upstream_pr_relationship": "backport",
        "superseded_by": "vllm#41268 (merged 2026-04-30, byte-equivalent on dev209+g5536fc0c0)",
        "vllm_version_range": "<0.20.2rc1.dev93",  # mirrors applies_to (gated SoT); reconciled D17 2026-06-17 (was <0.20.2rc1.dev209+g5536fc0c0)
        "apply_module": "sndr.engines.vllm._archive.pn19_scoped_max_split",
        "applies_to": {
            # Always applicable on CUDA. Self-detects torch < 2.11 lack
            # of _accelerator_setAllocatorSettings and falls through
            # unchanged.

            # [Iron rule #11 retire 2026-05-11] Deep-diff confirmed
            # #41268 byte-equivalent in dev209: `_scoped_allocator_max_split`
            # symbol present in vllm/v1/worker/gpu_worker.py (upstream's
            # context-manager-based scoped allocator with same 20 MiB
            # minimum + prior-value restoration semantics that PN19
            # implemented). Wiring auto-skips correctly; pin-gate
            # formalizes the retire boundary. Was actually mergeable into
            # dev93+ already (merge date 2026-04-30 < dev93 SHA 2026-05-07)
            # but lifecycle wasn't updated until this audit.
            # Cleanup: delete patch file in next refactor pass.
            "vllm_version_range": "<0.20.2rc1.dev93",
        },
        "lifecycle": "retired",  # 2026-05-11: byte-equivalent with #41268 in dev209
        "implementation_status": "full",
    },
    "PN23": {
        "title": "DFlash combine_hidden_states dtype cast (vllm#40334)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_PN23_DFLASH_DTYPE_FIX",
        "default_on": False,
        "category": "spec_decode",
        "credit": (
            "Backport of vllm#40334 (ciphernaut, OPEN 2026-05-01). Six-line "
            "defensive cast in Qwen3DFlashModel.combine_hidden_states to handle "
            "mixed-precision targets (AWQ + non-quantized layers, FP8 + BF16 mix). "
            "Casts hidden_states to fc.params_dtype before FC layer call. Fixes "
            "RuntimeError on mixed-precision DFlash configs."
        ),
        "upstream_pr": 40334,
        "upstream_pr_relationship": "backport",
        "applies_to": {
            # DFlash-specific; auto-no-op when qwen3_dflash.py absent or anchor
            # already has params_dtype cast (upstream merge).
        },
        "conflicts_with": [],
        "requires_patches": [],
        "apply_module": "sndr.engines.vllm.patches.spec_decode.pn23_dflash_combine_hidden_dtype",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN21": {
        "title": "DFlash SWA support partial backport (vllm#40898)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_PN21_DFLASH_SWA",
        "default_on": False,
        "category": "spec_decode",
        "credit": (
            "Partial backport of vllm#40898 (jianc99, OPEN 2026-05-01). "
            "Adds SWA config preservation in speculators/algos.py and forces "
            "causal=True on sliding-window layer attention metadata in "
            "v1/spec_decode/dflash.py. The qwen3_dflash.py model class "
            "changes (7+ sub-patches) are NOT backported. EMPIRICAL on 35B-A3B "
            "DFlash 160K: tool-call regresses 5-6/7 vs 7/7 baseline (without PN21) — "
            "metadata/compute mismatch (config says SWA, model computes full attn). "
            "DEFAULT OFF, NOT enabled in any launch script. Wait for upstream merge "
            "or full manual model class backport before enabling. Composes (no conflict) "
            "with PN24 if/when full enabler lands."
        ),
        "upstream_pr": 40898,
        "upstream_pr_relationship": "backport",
        "applies_to": {},
        "conflicts_with": [],
        "requires_patches": [],  # Pairs with PN24 but does not strictly require it
        "apply_module": "sndr.engines.vllm.patches.spec_decode.pn21_dflash_swa_support",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN22": {
        "vllm_version_range": (">=0.20.0", "<0.23.0"),  # retired-provenance drift cap (superseded; obsolete on dev148)
        "title": "Local argmax for TP draft (vllm#39419 backport)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_PN22_LOCAL_ARGMAX_TP",
        "default_on": False,
        "category": "spec_decode",
        "credit": (
            "Backport of vllm#39419 (EanWang; MERGED upstream "
            "2026-06-10T07:59Z as LocalArgmaxMixin — AFTER our pin "
            "0.22.1rc1.dev259+g303916e93). Adds get_top_tokens() plumbing "
            "to Qwen3, Qwen3-DFlash and Qwen3_5MTP model classes, enabling "
            "vocab-parallel argmax on each TP rank instead of all-gathering "
            "full logits. Wins +9.4-30.6% TPS on TP>=2 + draft model per "
            "PR author. LogitsProcessor.get_top_tokens() callsite is "
            "already in our pin (PR #34049 merged). "
            "[v2 2026-06-10 dead-binding audit fix] Original backport "
            "covered qwen3.py + qwen3_dflash.py only, but the live 35B MTP "
            "drafter is Qwen3_5MTP in qwen3_5_mtp.py (imports from "
            "qwen3_5.py, NOT qwen3.py) — the local-argmax path never "
            "engaged on 35B PROD. v2 adds the qwen3_5_mtp.py subpatch per "
            "the merged implementation (D2T remap parity included); the "
            "merged PR's proposer delta is logging-only vs our pin. Drift "
            "marker LocalArgmaxMixin self-skips all subpatches once a pin "
            "bump includes the merge. Llama, Eagle3 and DeepSeek parts of "
            "the upstream PR are not backported — Genesis does not run "
            "those models in production."
        ),
        "upstream_pr": 39419,
        "upstream_pr_relationship": "backport",
        "applies_to": {
            "vllm_version_range": "<0.23.1rc1.dev101",  # pin-gate cap: vllm#39419 (LocalArgmaxMixin) merged at bd2d83ff, ancestor of 0.23.1rc1.dev101+g4c6266331 — mixin native in interfaces.py:1285 on the deployed pin (live-verified 2026-06-17)
        },
        "conflicts_with": [],
        "requires_patches": [],
        "apply_module": "sndr.engines.vllm._archive.pn22_local_argmax_tp",  # moved to _archive/ 2026-06-21 (retired: superseded by vllm#39419 LocalArgmaxMixin, native on dev148)
        "superseded_by": "vllm#39419 (LocalArgmaxMixin)",
        "lifecycle": "retired",
        "implementation_status": "full",
    },
    "PN24": {
        "title": "DFlash aux layer +1 indexing fix (vllm#40727)",
        "tier": "community",
        "family": "worker",
        "env_flag": "GENESIS_ENABLE_PN24_DFLASH_AUX_LAYER_FIX",
        "default_on": False,
        "category": "spec_decode",
        "credit": (
            "Backport of vllm#40727 (benchislett, OPEN 2026-05-01). One-line "
            "semantic fix in _get_eagle3_aux_layers_from_config. DFlash stores "
            "target_layer_ids as 0-indexed; downstream Eagle3 aux machinery "
            "expects 1-indexed (layer 0 = embedding). +1 shift converts. "
            "Empirical: AL gsm8k 6.18→6.42 per PR author. "
            "[Phase 3D 2026-05-22] Live-network verification on dev371 "
            "(canonical pin bf610c2f56764e1b30bc6065f4ceace3d6e59036): "
            "(1) PN24 merge commit c628a93a64fb4929c3c11d8e2c7244c4826b4f76 "
            "(merged 2026-05-20) is NOT in our dev371 baseline — "
            "`gh api .../compare/c628a93a6...bf610c2f5` shows "
            "{ahead_by: 0, behind_by: 136} → PN24 merge is 136 commits "
            "AHEAD of dev371. Upstream fix not yet in our runtime. "
            "(2) dev371 source at vllm/v1/worker/gpu_model_runner.py:5171 "
            "shows the pre-fix shape `layer_ids = dflash_config.get(\"target_layer_ids\")` "
            "(no +1 shift) — Genesis PN24 anchor matches verbatim, so the "
            "TextPatch will find its anchor and apply normally on every "
            "DFlash boot. "
            "(3) Mechanism asymmetry vs upstream: upstream PR #40727 is a "
            "2-SITE fix (subtract 1 at storage in algos.py + add 1 at read "
            "in gpu_model_runner.py — coherent round-trip), while Genesis "
            "PN24 is a 1-SITE fix (only the +1 at the consumer). The two "
            "are NOT byte-identical even after restoring upstream — they "
            "are STRUCTURALLY different approaches to the same off-by-one. "
            "(4) DFlash validation evidence: both DFlash ModelDefs "
            "(qwen3.6-27b-dflash.yaml and qwen3.6-35b-a3b-fp8-dflash.yaml) "
            "explicitly enable PN24 with `GENESIS_ENABLE_PN24_DFLASH_AUX_LAYER_FIX: '1'` "
            "and declare `patches_attribution.PN24.role: load_bearing`. "
            "Phase 2.4 sprint Q27-DFlash M8 PASS (commit 7e310b25) and "
            "Q35-DFlash M8 PASS + quick bench (commit 8b635f90) both ran "
            "on dev371 with PN24 active — coherent output produced (Paris, "
            "4, finish_reason=stop, n_completion=2). The +1 shift is "
            "empirically correct on our DFlash checkpoint format on dev371. "
            "(5) Self-skip mechanism: wiring module declares "
            "upstream_drift_markers `[Genesis PN24]` + `i + 1 for i in "
            "dflash_config.get` — the second marker watches for the upstream-"
            "merged form. When a future pin bump pulls in commit c628a93a6 "
            "or successor that lands the consumer-side +1, PN24's apply() "
            "will auto-return `\"skipped\"` (upstream-merged) before any "
            "text-patching is attempted. No operator action needed at the "
            "future pin bump. "
            "Decision (verdict A annotate per iron rule #11; PN82-style "
            "with extra DFlash-evidence emphasis): KEEP PN24 active. Do "
            "NOT retire because (i) pin state — upstream merge is 136 "
            "commits ahead of dev371; (ii) load-bearing — DFlash ModelDefs "
            "depend on it and validation evidence from Phase 2.4 PASS "
            "smokes confirms active state was correct; (iii) mechanism "
            "asymmetry — Genesis 1-site vs upstream 2-site is not a clean "
            "byte-identical retire claim even when upstream lands later. "
            "Phase 5 follow-up queued: before the next pin bump that "
            "pulls in c628a93a6+, do an empirical determination of the "
            "DFlash checkpoint's `aux_hidden_state_layer_ids` indexing "
            "convention (0-indexed per Genesis assumption vs 1-indexed "
            "per upstream PR assumption) so the eventual retirement and "
            "any potential indexing reconciliation is fully understood. "
            "Both interpretations produce correct output for OUR "
            "checkpoint format on dev371 (validated by Phase 2.4 PASS), "
            "but the conceptual model matters when upstream's algos.py "
            "subtract-1 change lands alongside the consumer add-1 fix."
        ),
        "upstream_pr": 40727,
        "upstream_pr_relationship": "related_not_superseding",
        "applies_to": {},
        "conflicts_with": [],
        "requires_patches": [],
        "apply_module": "sndr.engines.vllm.patches.worker.pn24_dflash_aux_layer_indexing",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN28": {
        "title": "merge_attn_states NaN guard (vllm#39148 backport)",
        "tier": "community",
        "family": "kernels",
        "env_flag": "GENESIS_ENABLE_PN28_MERGE_ATTN_NAN_GUARD",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Backport of vllm#39148 (jasonkim8652, OPEN 2026-05-01). "
            "Branchless NaN guard in Triton merge_attn_states kernel for "
            "both-LSE-(-inf) edge case (zero-context-length chunked prefill). "
            "Without guard: NaN propagates through exp()/division and silently "
            "corrupts output — one bad token can break tool-call JSON parsing. "
            "Fix: clamp max_lse to -1e30 finite floor + add 1e-10 epsilon to "
            "denominator. Quality-only — no perf impact. CUDA merge_attn_states "
            "kernel already had this guard; PN28 brings Triton to parity."
        ),
        "upstream_pr": 39148,
        "upstream_pr_relationship": "backport",
        "applies_to": {},
        "conflicts_with": [],
        "requires_patches": [],
        "apply_module": "sndr.engines.vllm.patches.kernels.pn28_merge_attn_states_nan_guard",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P15B": {
        "title": "FA varlen max_seqlen_k clamp on TQ path (Issue #15 fix)",
        "tier": "community",
        "family": "memory",
        "env_flag": "GENESIS_ENABLE_P15B_FA_VARLEN_CLAMP",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Genesis-original 2026-05-01 fix for noonghunna's Issue #15. "
            "PN17 clamps max_seqlen_k on the FA2 backend path, but TurboQuant "
            "code path bypasses PN17's coverage by calling vllm_flash_attn's "
            "vendored wrapper via turboquant_attn.py:_flash_attn_varlen. P15B "
            "extends the same clamp logic to that callsite via text-patch — "
            "computes actual span from cu_seqlens_k and clamps max_seqlen_k "
            "before invocation. Prevents 50 MiB workspace OOM on long-context "
            "continuation-prefill on tight VRAM (24 GB consumer cards). "
            "Trade-off: adds one GPU->CPU sync per call on the infrequent "
            "continuation-prefill path."
        ),
        "upstream_pr": None,
        "applies_to": {},
        "conflicts_with": [],
        "requires_patches": [],
        "apply_module": "sndr.engines.vllm.patches.memory.p15b_fa_varlen_clamp",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P38B": {
        "title": "P38 compile-safe in-source hook (Issue #14 fix)",
        "tier": "community",
        "family": "memory",
        "env_flag": "GENESIS_ENABLE_P38B_COMPILE_SAFE",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Genesis-original 2026-05-01 fix for noonghunna's Issue #14. "
            "Root cause: aot_compile_fullgraph captures _continuation_prefill "
            "original body at engine init; Python class-attribute rebind "
            "(P38's mechanism) doesn't propagate to compiled artifact. "
            "P38B injects an in-source delegate hook at the start of "
            "_continuation_prefill body via text-patch. Hook calls a "
            "dispatcher that returns Genesis result OR None (fall-through). "
            "Source-level edit means aot_compile captures the hook itself. "
            "Affects ALL TQ KV users with V0/V1 compile pipeline; fp8 KV "
            "configs unaffected (different code path). Composes with P38 "
            "(both share _genesis_continuation_prefill impl)."
        ),
        "upstream_pr": None,
        "applies_to": {},
        "conflicts_with": [],
        "requires_patches": [],  # P38 install order: P38 first (provides impl), P38B second (installs hook)
        "apply_module": "sndr.engines.vllm.patches.memory.p38b_compile_safe_hook",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN26b": {
        "title": "Sparse-V tile-skip Genesis kernel (BLASST λ=a/L for SM86)",
        "tier": "community",
        # Family corrected 2026-05-12: kernel operates on attention V tensors
        # (sparse-V tile-skip in the TQ decode path), not on memory pools.
        # Wiring lives in integrations/attention/turboquant/pn26_sparse_v_kernel.py.
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_PN26_SPARSE_V",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Genesis-original Triton kernel fork — first sparse-V tile-skip "
            "deployed for SM86 (Ampere consumer). Synthesized from 4-agent "
            "research 2026-05-01: vllm#41422 (TheTom, AMD-only validated) "
            "design template + BLASST arXiv 2512.12087 (Yuan et al. Dec 2025) "
            "λ=a/L threshold formula + tq-kv reference (CUDA, SM86-compatible) "
            "acc*re_scale skip semantics + StreamingLLM (arXiv 2309.17453) "
            "sink token protection (first 4 KV positions never skipped). "
            "Mechanism: when tl.max(p) < threshold for a KV tile, skip V load + "
            "dequant + weighted sum, just decay accumulator. Online softmax "
            "denominator/max still update so totals stay numerically exact "
            "for non-skipped tiles. Composes with PN26 main (centroids "
            "prebake) + P98 (workspace revert) + P67 (multi-query — separate "
            "code path, not affected). Default OFF; opt-in via "
            "GENESIS_ENABLE_PN26_SPARSE_V=1 + GENESIS_PN26_SPARSE_V_THRESHOLD "
            "(fixed) OR GENESIS_PN26_SPARSE_V_SCALE_FACTOR (BLASST adaptive)."
        ),
        "upstream_pr": 41422,
        "upstream_pr_relationship": "backport",
        "applies_to": {},
        "conflicts_with": [],
        "requires_patches": [],
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.pn26_sparse_v_kernel",
        "lifecycle": "research",
        "research_note": (
            "First sparse-V tile-skip kernel deployed on SM86 (Ampere). "
            "Upstream vllm#41422 (TheTom) is AMD-only validated; Genesis "
            "port adds SM86 Triton path + BLASST adaptive λ=a/L threshold "
            "+ StreamingLLM sink-token protection. Default OFF until "
            "long-context (32k+) bench data lands — at typical 4k-8k "
            "decode the per-tile skip-probability check overhead exceeds "
            "the savings. Quality risk: sparse-V drops V loads for tiles "
            "below threshold; for biased workloads (concentrated attention) "
            "this is exact, but for diffuse-attention workloads the "
            "BLASST adaptive threshold can drop signal. Promotion requires "
            "(a) 32k+ ctx bench data, (b) tool-call score ≥ 8/10 across "
            "3+ models, (c) decode_tpot regression < 2% on short-ctx."
        ),
        "implementation_status": "full",
    },
    "PN27": {
        "title": "Revert MoERunnerInterface PluggableLayer (vllm#41440)",
        "tier": "community",
        "family": "moe",
        "env_flag": "GENESIS_ENABLE_PN27_REVERT_PLUGGABLE_MOE",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Backport of vllm#41440 (auto-generated CI failure analyzer, OPEN "
            "2026-05-01). Reverts vllm#35178 (b55b2652, merged 2026-04-30) "
            "which made MoERunnerInterface inherit from PluggableLayer + "
            "introduced DefaultMoERunner split/recombine. Issue #41306 "
            "reports +21% TPOT / +59% TTFT / -19% throughput on Mixtral-8x7B "
            "(8× H200), with bnellnm confirming `--moe-backend=triton` "
            "restores v0.19 perf. Our pin (0.20.1rc1.dev16+g7a1eb8ac2) "
            "predates the merge by 2 days — PN27 is a PROACTIVE SCAFFOLD "
            "for the case when we eventually pin-bump past b55b2652 BEFORE "
            "#41440 (or equivalent fix-forward) merges. On our current pin, "
            "all 3 sub-patches SKIP as intended (anchors are pre-#35178)."
        ),
        "upstream_pr": 41440,
        "upstream_pr_relationship": "backport",
        "applies_to": {},
        "conflicts_with": [],
        "requires_patches": [],
        "apply_module": "sndr.engines.vllm.patches.moe.pn27_revert_pluggable_moe",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN26": {
        "title": "TQ unified perf pack (centroids prebake + sparse V scaffold)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_PN26_TQ_UNIFIED",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Genesis-original 2026-05-01 unification of three OPEN upstream "
            "PRs (jasonkim8652): #41418 pre-baked Lloyd-Max centroids (drop-in "
            "safe, eliminates 50ms-2.5s JIT solver per shape on cold boot); "
            "#41422 sparse V tile-skip in decode kernel (scaffolded, OFF by "
            "default until NVIDIA Ampere correctness validation — author "
            "validated AMD MI300X only); #41414 head_dim pow-2 padding "
            "DROPPED — Qwen3.6 head_dim=128 already pow-2, would add dead "
            "code overhead. Genesis defensive addition: self-check at "
            "module-init asserts prebaked centroids equal solver output; on "
            "drift (e.g. upstream changes Lloyd-Max algo) auto-disables "
            "prebake and falls through to runtime solver with WARNING. No "
            "silent staleness. Composes with P67/P98/PN8 — orthogonal code "
            "paths."
        ),
        "upstream_pr": 41418,
        "upstream_pr_relationship": "backport",
        "applies_to": {},
        # PN26 (pre-baked Lloyd-Max tables) and PN57 (disk-persistent cache)
        # are two DIFFERENT implementations of the SAME upstream vllm#41418,
        # both rewriting the identical get_centroids() in centroids.py. Their
        # required anchors destroy each other (BOTH directions) — co-enabling
        # boots the second one FAILED with the file half-patched. Mutually
        # exclusive alternatives; pick one. Surfaced by the cross-patch
        # anchor-overlap lint (deep-audit 2026-06-14 #2).
        "conflicts_with": ["PN57"],
        "requires_patches": [],
        # v11.3.0 BUG #10 fix: PN26 spec apply_module is the canonical
        # unified-perf orchestrator (`pn26_tq_unified_perf`) which wires
        # the centroids prebake — the safe/drop-in component. The
        # sparse-V kernel sub-component has its own spec entry (`PN26b`,
        # env GENESIS_ENABLE_PN26_SPARSE_V) for opt-in activation after
        # NVIDIA Ampere validation. Previously incorrectly pointed at
        # `pn26_sparse_v_kernel` — on v12.0.0 spec-flip an operator with
        # GENESIS_ENABLE_PN26_TQ_UNIFIED=1 would activate the risky
        # sparse-V path instead of the safe centroids prebake.
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.pn26_tq_unified_perf",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN25": {
        "title": "SiluAndMul.forward_native opaque-op pool (Cliff 1 mech B compile path)",
        "tier": "community",
        "family": "kernels",
        "env_flag": "GENESIS_ENABLE_PN25_SILU_INDUCTOR_SAFE",
        "default_on": False,
        "category": "memory_savings",
        "credit": (
            "Genesis-original 2026-05-01 in response to noonghunna's "
            "club-3090#16 (VolandBerlioz/ampersandru cross-rig OOM trace, "
            "RTX 3090 24 GB + Lorbus 27B + long-prefill IDE workload, 29K). PN12 "
            "patches eager `forward_cuda` but `custom_ops=['none']` (default "
            "under V1 aot_compile_fullgraph) routes dispatch through "
            "`forward_native` which Inductor inlines and lowers to "
            "`empty_strided_cuda(...)`, bypassing PN12's pool. "
            "Sister-patch PN25 patches `forward_native` to dispatch through "
            "an opaque `genesis::silu_and_mul_pooled` torch.library.custom_op "
            "(Inductor cannot inline opaque ops). Both patches share the "
            "same FFNIntermediateCache pool. Recommended pairing for any "
            "inductor-heavy config."
        ),
        "upstream_pr": None,
        "applies_to": {},
        "conflicts_with": [],
        "requires_patches": [],  # complements PN12 but does not require it
        "apply_module": "sndr.engines.vllm.patches.kernels.pn25_silu_inductor_safe_pool",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN286": {
        "title": "FA KV cache layout revert for Ampere SM 8.6 (closes #42095 MTP K=3 regression)",
        "tier": "community",
        "family": "attention.flash",
        "env_flag": "GENESIS_ENABLE_PN286_FA_LAYOUT_REVERT_SM86",
        # K.1.R.R.7 (2026-05-29): flip default_on=True after empirical
        # validation. Patch self-skips on non-SM-8.6 hardware via
        # current_platform.is_device_capability(86) check. On SM 8.6:
        # - 35B FP8 dense MoE: +6.6% TPS (validated, 2 runs)
        # - 27B Lorbus + TQ k8v4: neutral (MTP layer in TQ overlay group)
        # No negative cases observed. Safe to enable by default.
        "default_on": True,
        "category": "perf_hotfix",
        "credit": (
            "Genesis-original 2026-05-29 (K.1.R.R.5). Upstream vllm#42095 "
            "(merged 2026-05-27, commit 7e33081ce) flipped FlashAttention "
            "KV cache layout from (2, num_blocks, ...) to (num_blocks, 2, "
            "...) — physically interleaving K and V per block. On Hopper "
            "SM 9.0+ with TMA this is performance-neutral or slightly "
            "faster. On Ampere SM 8.6 (A5000/A6000) with 6 MB L2 and no "
            "TMA, the new unbind(1) view has 2x outer stride producing "
            "L2 prefetch miss amplification during paged decode. "
            "Under MTP K=3 each accepted token costs 4 KV cache reads, "
            "amplifying the regression to 9% wall TPS empirically. "
            "Math derivation (K.1.R.R.5 diagnostic): base TPS 57.75, "
            "K=1 mult 1.45x (T_K=0.24), K=3 mult 2.07x (T_K=0.17) vs "
            "dev371 T_K 0.155 -> 31% slower draft step -> 9% wall TPS. "
            "35B FP8 dense MoE on SAME pin is +3.67% TPS (no MTP, FP8 "
            "cache fits L2) -- confirms hybrid/SM8.6-specific issue. "
            "PN286 monkey-patches FlashAttentionBackend.get_kv_cache_shape "
            "to (2, N, B, H, D), get_kv_cache_stride_order to pre-#42095 "
            "tuples, GPUModelRunner._update_hybrid_attention_mamba_layout "
            "to skip FA backends, and TextPatches FA forward/do_kv_cache_"
            "update to use unbind(0). Strictly SM 8.6 gated; SM 8.0 (A100), "
            "SM 9.0+ (Hopper, Blackwell) skip self-detection. "
            "Genesis contribution: mechanism identification via 5-test "
            "diagnostic, sm-strict gating, monkey-patch + TextPatch hybrid "
            "design, idempotent apply, drift markers. "
            "[Empirical validation 2026-05-29 K.1.R.R.5]: "
            "27B Lorbus INT4 + TQ k8v4 + MTP K=3 — NEUTRAL (119.93 TPS, "
            "same as baseline); reason: 27B MTP draft layer "
            "(mtp.layers.0.self_attn.attn) is in TQFullAttentionSpec group "
            "and uses TQ overlay kernels, not native FA. PN286 patches FA "
            "backend which is bypassed for TQ-quantized layers. "
            "35B FP8 dense MoE + MTP K=3 — POSITIVE: 254.97 TPS with PN286 "
            "vs 239.17 TPS without (+6.6% TPS, 50% variance reduction). "
            "Reason: 35B uses native FA for attention layers (not TQ), so "
            "FA backend monkey-patches restore pre-#42095 cache locality "
            "on A5000 SM 8.6. Combined with prior +3.67% pin improvement: "
            "net ~+10% TPS for 35B vs dev371 baseline. "
            "RECOMMENDED for any model that uses native FlashAttention on "
            "SM 8.6 hardware. Self-skips silently on TQ-only paths. "
            "Tool-call quality preserved (7/7 on 27B verified)."
        ),
        "upstream_pr": None,
        "applies_to": {
            "platform": "cuda",
            "device_capability": "8.6",
        },
        "apply_module": "sndr.engines.vllm.patches.attention.flash.pn286_fa_layout_revert_sm86",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN287": {
        "title": "qwen3_coder × MTP arg-corruption frequency observer (club-3090 #178)",
        "tier": "community",
        "family": "tool_parsing",
        "env_flag": "GENESIS_ENABLE_PN287_QWEN3CODER_ARGS_OBSERVER",
        "default_on": False,
        "category": "observability",
        "credit": (
            "Genesis-original 2026-05-29 (club-3090 cross-reference wave). "
            "Surfaces frequency of the qwen3_coder × MTP arg-corruption bug "
            "club-3090 maintainer flagged on noonghunna/club-3090#178 as "
            "\"distinct from streaming bug #145\". Server-validated bench on "
            "35B-A3B FP8 PROD (pin 626fa9bb, MTP K=3, max_tokens=150, agentic "
            "12-turn × 2-session) hit the symptom 4/24 times — HTTP 400 "
            "\"Unterminated string starting at: line 1 column 13\" cascading "
            "from turn 9 onwards after a truncated tool_call.arguments at "
            "turn 8 poisoned the chat history. Observation patch wraps "
            "Qwen3CoderToolParser.extract_tool_calls_streaming via runtime "
            "monkey-patch (idempotent, opaque to dynamo since not in "
            "compile path), inspects prev_tool_call_arr post-invocation, "
            "emits structured WARN log + counter increment when arguments "
            "field is non-empty and json.loads() raises. Read-only — does "
            "NOT mutate output. Companion to tools/bench_agentic.py "
            "JSON-validate defense (client-side cascade prevention, ships "
            "in same wave). Default OFF — opt-in to surface prod frequency "
            "before deciding the proper behavior-changing fix (override "
            "finish_reason → length, auto-close JSON, etc.). After ~weeks "
            "of prod observation: if frequency negligible, close as "
            "observed; if meaningful, escalate to PN288 behavior fix or "
            "file vllm upstream PR. Auto-skips if upstream adds its own "
            "`_args_validation_installed` marker. Existing P64+PN56+P61C "
            "3-layer defense does NOT cover this surface — they fix "
            "streaming-extractor early-return / XML-parse fallback / SSE "
            "deferred-commit respectively, none validate final args JSON. "
            "§2.4 Phase A (2026-05-30): Prometheus counters relabeled "
            "with (model, ctx_bucket) — buckets 0-5K / 5-15K / 15-30K / "
            "30K+ on len(current_token_ids); model from request.model. "
            "Cardinality = ~3 models × 4 buckets = 12 series per counter, "
            "well within Prometheus best-practice ceiling. The labeled "
            "data unlocks evidence-based PN288 trigger criteria (e.g. "
            "'fire only on 35B-A3B + ctx≥15K') instead of a global flag. "
            "v2 2026-06-10 (call-site drift audit finding): PROD container "
            "vllm-qwen3.6-35b-balanced-k3 (pin 0.22.1rc1.dev259+g303916e93) "
            "runs --tool-call-parser qwen3_xml → Qwen3XMLToolParser with "
            "its OWN extract_tool_calls_streaming; the v1 coder-only wrap "
            "never fired there (Prometheus counters permanently zero, "
            "~236ms+6.5MB APIServer boot cost for nothing). v2 wraps BOTH "
            "Qwen3CoderToolParser AND Qwen3XMLToolParser — whichever "
            "import; the inactive parser's wrap is inert since the serving "
            "layer instantiates only the configured parser. XML parser "
            "ACCUMULATES arguments incrementally per delta (+= fragment), "
            "unlike the coder parser which writes complete JSON only at "
            "function close — so the XML wrap gates json.loads validation "
            "on tool-call completeness (count of </tool_call> end tokens "
            "in current_text vs entry index) with a per-instance "
            "validated-index set reset on new stream (empty previous_text, "
            "same signal upstream uses). Avoids mid-stream partial-JSON "
            "false positives. Scope: completed-call corruption (framing "
            "intact, args corrupt — the #178 MTP mode); final-call "
            "max_tokens truncation stays PN288's serving-layer surface. "
            "Prometheus counters gain third label `parser` "
            "(qwen3_coder|qwen3_xml) — cardinality 3 models × 4 buckets "
            "× 2 parsers = 24 series per counter. Same env flag (live "
            "launcher unchanged); v2 also fixes is_applied()/revert() to "
            "resolve the new vllm.tool_parsers.* module layout (v1 only "
            "tried the legacy entrypoints path there)."
        ),
        "upstream_pr": None,
        # version-capped <0.23.0 2026-06-19 (dev148 TIER-1 audit): tool_parsers/
        # qwen3coder_tool_parser.py + gemma4 parser DELETED by #45588; engine
        # state machine supersedes. The observer wraps the deleted/restructured
        # Qwen3CoderToolParser/Qwen3XMLToolParser classes, so it correctly skips
        # on 0.23.x rather than file-missing-skip.
        "applies_to": {
            "tool_call_parser": ["qwen3_coder", "qwen3_xml"],
            "vllm_version_range": (">=0.20.0", "<0.23.0"),
        },
        "apply_module": "sndr.engines.vllm.patches.tool_parsing.pn287_qwen3coder_args_validity_observer",
        "lifecycle": "experimental",
        "implementation_status": "full",
        # 2026-06-20: PN56 + P61c were consolidated into the P64 entry, so the
        # former ["P64", "PN56", "P61c"] composes_with is deduped to the
        # surviving P64 (otherwise dangling -> test_composes_with_targets_exist
        # would fail). Same parser-layer defense cohort, now one registry id.
        "composes_with": ["P64"],
    },
    "PN288": {
        "title": "qwen3_coder tool_call finish_reason override — Phase B+C with length-band safety guard",
        "tier": "community",
        "family": "serving",
        "env_flag": "GENESIS_ENABLE_PN288_TOOL_FINISH_REASON_OVERRIDE",
        "default_on": False,
        "category": "stability",
        "credit": (
            "Genesis-original 2026-05-30 (§1.3 of the unified plan, "
            "Phase B). Mutating companion to PN287's observation patch: "
            "when upstream serving.py would emit finish_reason='tool_calls' "
            "BUT the accumulated tool_call.arguments doesn't parse as JSON "
            "AND the underlying output.finish_reason is 'length' (max_tokens "
            "cut mid-JSON-string), PN288 downgrades the response to "
            "finish_reason='length'. OpenAI-format clients (Cline, Claude "
            "Code, openai-python, openai-node) treat 'length' as the "
            "canonical retry-with-higher-max_tokens signal, so the downgrade "
            "lets them auto-recover instead of cascading a poisoned "
            "tool_call into chat history. Text-patches "
            "OpenAIServingChat._create_chat_completion at two anchors "
            "(serving.py:884-893 streaming if-block + serving.py:1306-1310 "
            "non-streaming bool, both verified live on pin 626fa9bb 2026-05-30). "
            "Decision logic lives in the companion middleware module "
            "sndr.engines.vllm.middleware.pn288_finish_reason_override so it "
            "is unit-testable without applying the overlay; both call sites "
            "are wrapped in try/except so any failure falls back to upstream. "
            "Phase B (this commit): full text-patch + helper + Prometheus "
            "labels (model, channel, action), but GENESIS_PN288_DRY_RUN=1 "
            "is the default when PN288 is enabled — the patch LOGS 'WOULD "
            "downgrade' + counter-increments, then emits upstream's verdict "
            "unchanged. Phase C (future, operator decision): flip "
            "GENESIS_PN288_DRY_RUN=0 after 2-4 weeks of Phase A (PN287 "
            "labeled counters) + Phase B (would_downgrade counter) evidence "
            "justifies the behavior change. Default OFF; only the PN287 "
            "observation surface is on by operator action. Risk: behavior "
            "change is irreversible per request; 'length' for tool_calls "
            "is unusual-but-valid in the OpenAI spec — strict clients may "
            "behave unexpectedly. Composes with P64+PN56+P61c (parser-layer "
            "defenses) — PN288 sits at serving layer, no anchor overlap. "
            "Prometheus counter: vllm:pn288_finish_reason_override_total "
            "with labels (model, channel ∈ {streaming, non_streaming}, "
            "action ∈ {would_downgrade, downgraded, kept_tool_calls_args_valid, "
            "kept_tool_calls_no_length_trunc, "
            "kept_tool_calls_args_length_out_of_range}). Cardinality ~3 × 2 × 5 = 30 "
            "series. References §1.3 of UNIFIED_DEVELOPMENT_PLAN. "
            "PHASE C SAFETY GUARDS (2026-05-30): added length-band check "
            "before downgrade — args length must fall in "
            "[GENESIS_PN288_MIN_ARGS_LENGTH (default 5), "
            "GENESIS_PN288_MAX_ARGS_LENGTH (default 200)). Real "
            "max_tokens-truncated tool_call args are typically 5-80 chars "
            "(PN287 evidence band); args outside the window are more likely "
            "a different parse-failure mode where downgrade would corrupt "
            "a real long tool call. The kept_tool_calls_args_length_out_of_range "
            "action surfaces the guard's firings so operators can tune the "
            "band BEFORE flipping DRY_RUN=0. ACTIVATION RUNBOOK (B→C): "
            "(1) enable PN287+PN288 in PROD launcher (DRY_RUN default=1); "
            "(2) observe vllm:pn288_finish_reason_override_total{action="
            "'would_downgrade'} + PN287 malformed_total{model,ctx_bucket} "
            "for 2-4 weeks; (3) verify *_out_of_range stays low; (4) if "
            "frequency concentrates on (35B-A3B + ctx≥15K) per PN287 "
            "labels, tighten MIN/MAX_ARGS_LENGTH to empirical p10/p90 of "
            "malformed args lengths; (5) flip GENESIS_PN288_DRY_RUN=0 — "
            "downgraded action label starts firing in PROD."
        ),
        "upstream_pr": None,
        "applies_to": {
            "tool_call_parser": "qwen3_coder",
            # 2026-06-19 drift audit: 0.23.x (#45413/#45588 parser reorg era)
            # SIMPLIFIED the streaming finish_reason block — the harmony
            # OR-clause + both use_harmony/harmony_tools_streamed vars were
            # removed, so PN288's dev259-era harmony anchor is count=0 on dev148
            # (genuine drift). Capped <0.23.0: PN288 stays valid on pre-0.23
            # rollback pins (where its anchor matches) and version-gate-skips
            # cleanly on dev148. The load-bearing sibling P107 is dual-anchor and
            # already handles dev148; PN288 is the UNUSED dry-run companion
            # (default_off, in zero builtin YAMLs), so no runtime change. Bring
            # to dev148 as a net-new dual-anchor patch only if the malformed-args
            # downgrade is ever wanted on 0.23.x (re-author, not re-anchor).
            "vllm_version_range": (">=0.20.0", "<0.23.0"),
        },
        # Apply-order chain (2026-06-11, pin 0.22.1rc1.dev259 re-anchor):
        # P107 v3's streaming anchor spans the same pristine finish_reason
        # block PLUS the following choice_data line; P107's replacement
        # keeps the block verbatim, so the pair composes ONLY in
        # P107-then-PN288 order (PN288 re-indents the block inside its
        # except-fallback, destroying P107's anchor). Ordering-only
        # dependency: PN288 still applies standalone on pristine when
        # P107 is disabled — the dep-graph dep_missing WARNING is
        # advisory. Proven in
        # tests/unit/integrations/serving/test_pn288_p107_anchor_coordination.py.
        "requires_patches": ["P107"],
        "apply_module": "sndr.engines.vllm.patches.serving.pn288_tool_finish_reason_override",
        "lifecycle": "experimental",
        "implementation_status": "full",
        # 2026-06-20: PN56 + P61c were consolidated into the P64 entry, so the
        # former ["P64", "PN56", "P61c", "PN287"] composes_with is deduped to
        # ["P64", "PN287"] (otherwise dangling -> test_composes_with_targets_
        # exist would fail).
        "composes_with": ["P64", "PN287"],
    },
    "PN289": {
        "title": "Genesis process-info Prometheus gauge (§6.H10 enterprise observability)",
        # v11.3.0 bug fix: was tier="engine" which silently disabled the
        # patch on every boot via `_check_tier_gate` requiring the
        # commercial sndr_engine package. But the implementation lives
        # at sndr/observability/genesis_process_info.py (public
        # sndr_core, Apache 2.0, Genesis-original 2026-05-30 per credit
        # field below). It is community-tier code; tier="engine" was a
        # registry-side mistake that broke §6.H10 enterprise
        # observability for every operator who enabled
        # GENESIS_ENABLE_PN289_PROCESS_INFO=1.
        "tier": "community",
        "family": "observability",
        "env_flag": "GENESIS_ENABLE_PN289_PROCESS_INFO",
        "default_on": False,
        "category": "observability",
        "credit": (
            "Genesis-original 2026-05-30 (§6.H10 closure). Canonical "
            "Prometheus *_info* pattern: emits genesis_process_info "
            "gauge with value 1 labeled by (preset, profile, "
            "workload_class, K, backend, patch_hash, model, pin). "
            "Downstream PromQL queries JOIN against this via "
            "`* on(instance) group_left(preset, ...) "
            "genesis_process_info` to pivot vllm-builtin counters "
            "(num_requests_running, e2e_request_latency_seconds, "
            "etc.) by Genesis operator metadata. Modifying vLLM's "
            "core metrics code to add labels directly was out of "
            "scope and pin-bump fragile — the *_info* pattern is the "
            "Grafana/Prometheus best-practice for late-binding label "
            "augmentation. Label values resolved from env "
            "(GENESIS_PRESET / GENESIS_PROFILE / GENESIS_WORKLOAD_CLASS) "
            "+ argv (--speculative-config / --attention-backend / "
            "--served-model-name) + git rev-parse in GENESIS_REPO + "
            "vllm.__version__. Operator queries: "
            "(a) `count by (preset, profile, K, backend) "
            "(genesis_process_info)` for active fleet configs; "
            "(b) histogram_quantile pivot by preset for per-preset "
            "p99 latency; (c) `count by (patch_hash) "
            "(genesis_process_info)` for fleet patch-hash drift "
            "detection. Cardinality: 1 row per container instance. "
            "Composes with PN287 (qwen3_coder args observer Counter), "
            "PN288 (finish_reason override Counter), PN95 Prometheus "
            "metrics — all use the same labeled-Counter idiom for "
            "per-request signals; PN289 adds the launch-time process "
            "metadata that completes the operator's mental model."
        ),
        "upstream_pr": None,
        "applies_to": {},
        "apply_module": "sndr.observability.genesis_process_info",
        "lifecycle": "experimental",
        "implementation_status": "full",
        "composes_with": ["PN287", "PN288"],
    },
    "PN302": {
        "title": "Genesis Model Profile boot-time initializer (model-aware decision API)",
        "tier": "community",
        "family": "detection",
        "env_flag": "GENESIS_ENABLE_PN302_MODEL_PROFILE_INIT",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.detection.pn302_model_profile_init",
        "lifecycle": "experimental",
        "category": "config_auto_tune",
        "credit": (
            "Genesis-original 2026-06-05 — companion to PN296 (arch "
            "profile). Detects MODEL architecture (Qwen3.6 / Gemma / "
            "Llama / Mamba), quantization (FP8 / INT4 / GPTQ / AWQ / "
            "AutoRound), topology (Dense / MoE / Hybrid GDN+Attn), "
            "spec-decode method (MTP K / EAGLE / none), and parallelism "
            "(TP size). Emits GENESIS_MODEL_* env stamps for downstream "
            "patches: uses_gdn, uses_marlin, uses_tq, has_mtp, is_fp8, "
            "is_moe, hot_kernels. Together with PN296 forms a 2D "
            "decision matrix: f(arch_profile, model_profile). Enables "
            "model-aware patches (e.g. PN303 Marlin FP8 fires ONLY when "
            "model is FP8 AND arch has no native FP8 TCs)."
        ),
        "upstream_pr": None,
        "applies_to": {},
        "implementation_status": "full",
        "composes_with": ["PN296"],
    },
    "PN300": {
        "title": "Universal Triton Autotune Arch-Aware Wrapper (replaces per-file PN298/299)",
        "tier": "community",
        "family": "detection",
        "env_flag": "GENESIS_ENABLE_PN300_UNIVERSAL_TRITON_AUTOTUNE_WRAPPER",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.detection.pn300_universal_triton_autotune_wrapper",
        "lifecycle": "experimental",
        "category": "kernel_perf",
        "credit": (
            "Genesis-original 2026-06-05 — enterprise-grade architectural "
            "solution. Instead of patching each FLA/Mamba/attention ops "
            "file individually (PN298=1 file, PN299=3 files), monkey-"
            "patches triton.runtime.autotuner.Autotuner.__init__ to "
            "filter `configs` list using get_gpu_arch_profile()'s "
            "max_safe_num_warps + max_safe_num_stages. Coverage: ALL "
            "@triton.autotune decorators across vllm package — past, "
            "present, future. Idempotent (skip if already wrapped). "
            "Safety: empty filter result keeps original (never breaks "
            "autotune). Escape: GENESIS_PN300_DISABLE=1. No-op on "
            "Hopper+ (max_warps=8 doesn't prune anything). Critical for "
            "SM 8.x consumer (Ampere RTX 30xx/A5000/A6000) where "
            "num_warps=8 spills 100KB shared mem budget. Composes with "
            "PN296. Supersedes PN298, PN299 (those can be disabled once "
            "PN300 verified, but composability is benign — both filters "
            "produce same result)."
        ),
        "upstream_pr": None,
        "applies_to": {},
        "implementation_status": "full",
        "composes_with": ["PN296"],
    },
    "PN362": {
        "title": "Triton autotune determinism — VLLM_TRITON_FORCE_FIRST_CONFIG (vendor of vllm#42425)",
        "tier": "community",
        "family": "kernels",
        "env_flag": "GENESIS_ENABLE_PN362",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.kernels.pn362_triton_force_first_config",
        "lifecycle": "experimental",
        "category": "bench_methodology",
        "credit": (
            "Genesis vendoring of OPEN upstream PR vllm#42425 "
            "(Francesco Fusco / IBM, OPEN as of 2026-06-09). Adds the "
            "VLLM_TRITON_FORCE_FIRST_CONFIG env knob: monkey-patches "
            "triton.runtime.autotuner.Autotuner.run to pick the first "
            "VALID config (walking past OutOfResources / "
            "CompileTimeAssertionFailure / PTXASError) instead of "
            "benchmarking all candidates. Kills run-to-run autotune "
            "variance — the SAME jitter that produced the false "
            "'199 vs 228 wall_TPS regression' alarm on 2026-06-09. "
            "Author cites GDN prefill + MTP non-determinism (PR "
            "#40172 debugging) as motivation — exactly our hybrid + "
            "MTP K=3 hot path. Default off (no PROD behaviour "
            "change); opt-in for bench A/B and determinism debugging. "
            "Composes cleanly with PN345 (shmem-aware pre-autotune "
            "pruner): PN345 drops OOR configs at decorator time; "
            "PN362 picks first surviving at runtime. No anchor "
            "overlap — PN345 patches FLA kernel source files, PN362 "
            "patches env_override.py only. Implementation: single "
            "text-patch sub-patch inlining the upstream 107-LOC "
            "force_first_config.install() helper into env_override.py "
            "after _patch_inductor_fallback_allow_list() (the file's "
            "canonical last call). Detects post-merge state via "
            "vllm/triton_utils/force_first_config.py existence "
            "and skips. Risk: very low — opt-in, default off, "
            "additive only."
        ),
        "upstream_pr": 42425,
        "upstream_pr_relationship": "backport",
        "applies_to": {"vllm_version_range": (">=0.21.0", "<0.23.0")},
        "implementation_status": "full",
        "composes_with": ["PN345", "PN340", "PN341"],
    },
    "PN350": {
        "title": "Fused GDN Q/K/V split Triton kernel (SGLang#26206 + TRT-LLM#12966 convergent)",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_PN350",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.attention.gdn.pn350_gdn_qkv_fused_split",
        "lifecycle": "experimental",
        "category": "kernel_perf",
        "credit": (
            "Genesis convergent port of SGLang PR #26206 (Qwen3.6-35B-A3B "
            "+2.66% output TPS on B200) and TensorRT-LLM PR #12966 — both "
            "engines independently introduced the same fused GDN post-conv "
            "Q/K/V split Triton kernel. Replaces upstream "
            "Qwen3_GatedDeltaNet.rearrange_mixed_qkv torch.cat-based path "
            "(1 full-buffer memcpy + 4-5 ATen kernel launches per call) "
            "with a single Triton launch — 1 program per token row, 1 read "
            "+ 3 writes. Per-layer split time 18.97ms → 3.33ms (5.7× "
            "kernel speedup, SGLang bench). End-to-end Qwen3.6-35B-A3B "
            "+2.66% output TPS. On Ampere SM 8.6 the kernel speedup carries "
            "(memory-bandwidth-bound, no SM-specific intrinsics). Absolute "
            "μs savings per layer scale with bandwidth ratio (768 GB/s "
            "A5000 vs 8 TB/s B200 ≈ 10×). As fraction of slower A5000 "
            "forward → +1-1.5% single-stream wall_TPS estimated. Strict "
            "no-regression fallback: kernel exception or env "
            "GENESIS_DISABLE_PN350=1 routes back to upstream cat-based "
            "split. Kernel module: sndr/engines/vllm/kernels/pn350_gdn_qkv_"
            "fused_split.py (~80 LOC pure Triton). Integration: 1 text-"
            "patch on qwen_gdn_linear_attn.py replacing rearrange_mixed_qkv "
            "body. Shmem budget: 16 KiB per program at qkv_dim=8192 BF16 "
            "≪ 99 KiB A5000 opt-in. No autotune (single-config kernel). "
            "Composes with PN340+PN341+PN345 (different files), PN204 "
            "(in_proj upstream of conv), PN54 (.contiguous() dedup — "
            "PN350 outputs already contiguous), PN29 (chunk_o downstream), "
            "P28 (gdn_core pool downstream). No anchor overlap."
        ),
        "upstream_pr": 26206,  # SGLang #26206; TRT-LLM #12966 convergent algorithm
        "upstream_pr_relationship": "backport",
        "applies_to": {"vllm_version_range": (">=0.21.0", "<0.24.0")},
        "implementation_status": "full",
        "composes_with": ["PN340", "PN341", "PN345", "PN204", "PN54", "PN298", "P28"],  # PN29 consolidated into PN298 (2026-06-19)
    },
    "PN354": {
        "title": "GDN chunked-prefill exp2 gate decay (extends vllm#43195 KDA pattern to GDN)",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_PN354_GDN_USE_EXP2",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.attention.gdn.pn354_gdn_use_exp2",
        "lifecycle": "experimental",
        "category": "kernel_perf",
        "credit": (
            "Genesis extension of MERGED vllm#43195 (KDA-only in our "
            "pin) to the GDN chunked-prefill consumers. #43195 scales "
            "the chunk-local cumulative gate ONCE after cumsum "
            "(g = g * RCP_LN2, RCP_LN2=1.4426950216 from "
            "vllm.utils.math_utils) and uses exp2 in every downstream "
            "consumer — one fewer fp32 fmul per element per exp site "
            "(exp(x) lowers to exp2(x*log2e) on NVIDIA; the pre-scale "
            "hoists the multiply out of the kernels). chunk_delta_h.py "
            "ALREADY carries the USE_EXP2 dual branches upstream (KDA "
            "uses them); PN354 adds the same constexpr+wrapper plumbing "
            "to the remaining GDN-path exp sites: chunk_o.py (2 sites), "
            "chunk_scaled_dot_kkt.py (1 site), wy_fast.py (1 raw "
            "tl.exp site), pre-scales g once in chunk.py after "
            "chunk_local_cumsum (template: kda.py g = g * RCP_LN2 "
            "after cumsum), and threads use_exp2 through the PN59 "
            "streaming driver (our file, direct edit) — its vanilla "
            "AND windowed paths. Decode paths (fused_recurrent / "
            "fused_sigmoid_gating) stay natural-base — zero-win there, "
            "upstream keeps them on exp too. State values are domain-"
            "unchanged (exp2(g*RCP_LN2) == exp(g) numerically) so "
            "prefill-ON/decode-OFF mixing is safe — KDA ships exactly "
            "this split. Runtime-conditional text: env read ONCE at "
            "module import in the patched files; flag off -> no "
            "pre-scale, NO use_exp2 kwarg passed (empty splat) -> "
            "bit-identical to upstream in every partial-apply state. "
            "Consumer kernels patched FIRST, chunk.py dispatcher only "
            "when all consumers live. P103's own chunked path stays "
            "natural-base end-to-end (self-consistent, correct, "
            "un-optimized when it engages). Anchors verified unique "
            "against live pin 0.22.1rc1.dev259+g303916e93."
        ),
        "upstream_pr": 43195,
        "upstream_pr_relationship": "backport",
        "applies_to": {"vllm_version_range": (">=0.22.0", "<0.24.0")},
        "implementation_status": "full",
        "composes_with": ["PN59", "P103", "PN106", "PN298", "PN299", "PN345", "PN350"],  # PN29 consolidated into PN298 (2026-06-19)
    },
    "PN396": {
        "vllm_version_range": (">=0.20.0", "<0.23.0"),  # retired-provenance drift cap (GDN spec-decode num_warps dead-end; n/a on dev148)
        "title": "GDN spec-decode recurrent kernel num_warps 4->1 (SM 8.6 row-per-thread reduction)",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_PN396_GDN_SPEC_DECODE_WARPS",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.attention.gdn.pn396_gdn_spec_decode_num_warps",
        # 2026-06-17 (0.23.1 reverify): RETIRED as a tested-negative dead-end.
        # The anchor still resolves byte-exact on 0.23.1 (upstream still hardcodes
        # num_warps=4), so the mechanical rule would bump_cap — but that is
        # OVERRIDDEN: the A/B on 2026-06-14 (PROD 35B chat-matrix n=5) proved
        # num_warps=1 REGRESSES everywhere vs upstream 4 (code -4.2%, tool_call
        # -10.8%, short_chat -5.5%, thinking_off -2.6%, thinking_on -3.8%). 4 warps
        # win because extra threads hide per-token q/k/v global-load latency on the
        # gating+varlen path. Restoring it on 0.23.1 would ARM a known regression
        # whenever an operator sets the flag — strictly worse than the silent
        # no-op. Kept capped <0.23.0 AND retired so the refuted 4-vs-1 result is
        # preserved (dead-ends ledger) and the question is not re-opened.
        "lifecycle": "retired",
        "superseded_by": "dead-end (tested-negative A/B 2026-06-14; num_warps=4 wins)",
        "category": "kernel_perf",
        "credit": (
            "TESTED-NEGATIVE — DO NOT ENABLE. A/B 2026-06-14 on dev491 (PROD "
            "35B, chat-matrix n=5): num_warps=1 REGRESSED vs upstream 4 — "
            "thinking_off 247.6 vs 254.2 (-2.6%), thinking_on 244.6 vs 254.3 "
            "(-3.8%), code 209.0 vs 218.1 (-4.2%), short_chat -5.5%, tool_call "
            "-10.8%; multi_turn flat, long_gen -1.1%. The static hypothesis "
            "(1 warp => intra-thread BK reduction, no shuffle, matches the "
            "fused_recurrent siblings) was empirically wrong: 4 warps win here "
            "because the extra threads hide the per-token q/k/v global-load "
            "latency and the gating+varlen path differs from the pure-recurrent "
            "siblings. Kept default_on=False + the anchor/dispatch as a "
            "documented dead-end so the '4 vs 1' question is not re-opened. "
            "Original (refuted) rationale follows.\n"
            "Genesis-original SM 8.6 tune for the dominant GDN spec-decode "
            "kernel fused_sigmoid_gating_delta_rule_update_kernel (30 of 41 "
            "layers route through it under MTP K=3, IS_VARLEN=True). The "
            "launcher (fla/ops/fused_sigmoid_gating.py:212) hardcodes "
            "num_warps=4 while every sibling recurrent kernel in the same "
            "family (fla/ops/fused_recurrent.py:199,439) uses num_warps=1 for "
            "the identical [BV=32,BK=128] fp32 state tile. At 1 warp (32 "
            "threads) Triton maps the 32 state rows one-per-thread, so the two "
            "per-token reductions over BK=128 (b_v=-sum(b_h*b_k), b_o=sum("
            "b_h*b_q)) are intra-thread (sequential, no cross-warp shuffle) "
            "and b_h stays register-resident across the T loop; at 4 warps "
            "each row's reduction is split across 4 threads, adding a 4-way "
            "shuffle/shared-mem reduction every token. 1 warp also packs more "
            "(sequence,head) blocks per SM on the 28-SM A5000. The gating "
            "(softplus/exp/sigmoid) is scalar-per-head so warp count is "
            "irrelevant to it. LAUNCH-PARAM change only -> bit-identical "
            "output. Opt-in (default OFF) pending the decode-TPOT A/B; "
            "GENESIS_DISABLE_PN396=1 force-reverts. Anchor 'num_stages=3 / "
            "num_warps=4' is unique to fused_sigmoid_gating.py in fla/ops."
        ),
        "upstream_pr": None,
        "applies_to": {"vllm_version_range": (">=0.22.0", "<0.23.0")},
        "implementation_status": "full",
        "composes_with": ["PN354", "PN59", "PN345", "PN299"],
    },
    "PN367": {
        "title": "CUDA graph memory estimate clamp (vendor of OPEN vllm#44745, ex-vllm#45076)",
        "tier": "community",
        "family": "compile_safety",
        "env_flag": "GENESIS_ENABLE_PN367",
        "default_on": True,
        "apply_module": "sndr.engines.vllm.patches.compile_safety.pn367_cudagraph_mem_estimate_clamp",
        "lifecycle": "experimental",
        "category": "stability",
        "credit": (
            "Genesis vendor of vllm PR #44745 (fixes #44740) — negative "
            "CUDA graph memory estimation under MTP spec-decode. "
            "History: v1 vendored OPEN PR #45076 (Oxygen56); on "
            "2026-06-10 the author CLOSED #45076 and consolidated into "
            "#44745 (same clamp + 1 MiB first-capture floor + unit "
            "tests, OPEN) — PN367 v2 tracks #44745. Verified at pin "
            "g303916e93: decoder profiling path appends raw mem_before "
            "- free_after (can go negative via CachingAllocator "
            "freelist consolidation or MTP lazy buffer GC between "
            "measurements) while the encoder path in the same function "
            "already clamps to >= 0. A negative first_capture "
            "understates graph memory -> KV cache sized larger than the "
            "card affords -> silent headroom loss / OOM risk on 24 GB "
            "A5000 at gpu_memory_utilization=0.9 + MTP K=3. Vendored: "
            "per-sample clamp + WARNING on negative delta (visible at "
            "PROD's VLLM_LOGGING_LEVEL=WARNING) + 1 MiB first-capture "
            "floor in gpu_model_runner.py + final non-negative guard in "
            "gpu_worker.py. NOT vendored (documented divergence): the "
            "PR's per-measurement empty_cache() — measurement-stability "
            "improvement with boot-time cost; clamp alone removes the "
            "correctness hazard. Defensive: zero behavior change when "
            "estimates are positive. Self-skips when #44745 lands — v2 "
            "drift markers are exact substrings of its merged form "
            "(v1's markers could never fire; fixed 2026-06-11)."
        ),
        "upstream_pr": 44745,
        "upstream_pr_relationship": "backport",
        "related_upstream_prs": [45076],
        # 2026-06-17 (0.23.1 pin-bump): cap bumped <0.23.0 -> <0.24.0. default_on
        # negative-cudagraph-memory-estimate clamp silently no-op'd on 0.23.1 with
        # the stale cap. PR #44745 OPEN (not merged); both required anchors
        # (gpu_model_runner.py + gpu_worker.py) verified byte-present in upstream
        # source at the 0.23.1 build commit 4c626633; drift markers ABSENT.
        "applies_to": {"vllm_version_range": (">=0.21.0", "<0.24.0")},
        "implementation_status": "full",
        "composes_with": ["P66", "PN125", "PN126", "PN364"],
    },
    "PN352": {
        "title": "Triton moe_sum for unsupported topk (counterpart of OPEN vllm#44557)",
        "tier": "community",
        "family": "moe",
        "env_flag": "GENESIS_ENABLE_PN352",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.moe.pn352_moe_sum_topk8",
        "lifecycle": "experimental",
        "category": "kernel_perf",
        "credit": (
            "Genesis counterpart of OPEN vllm PR #44557 (xyang16, 'Support "
            "more topk values in moe_sum kernel'). The PR extends the "
            "compiled CUDA switch in csrc/moe/moe_align_sum_kernels.cu — "
            "unvendorable onto a prebuilt nightly wheel — so Genesis ships "
            "the equivalent as a Triton kernel routed from the single hot "
            "Python call site (fused_experts -> ops.moe_sum). Verified at "
            "pin g303916e93: the compiled switch covers topk 2/3/4 only; "
            "topk=8 (Qwen3.6-35B-A3B num_experts_per_tok=8, 40 layers) "
            "falls back to at::sum_out generic TensorIterator reduce per "
            "layer per step. PR author measures ~-700 us per decode step "
            "on a 40-layer topk=8 MoE; est -1-3 % decode TPOT on our "
            "shape. fp32 accumulation (ATen acc_type parity for "
            "half/bf16). Text install is always-on (runtime branch is "
            "env-gated and bit-equivalent when GENESIS_ENABLE_PN352 is "
            "unset); GENESIS_DISABLE_PN352_INSTALL=1 skips even the text "
            "for hygiene. Single-strike disable on any Triton failure -> "
            "upstream fallback. Iron-rule-#12 log on first hot-path call. "
            "Self-skips when upstream lands moe_sum_kernel<scalar_t, 8> "
            "(drift marker). Kernel: sndr/engines/vllm/kernels_legacy/"
            "pn352_moe_sum_topk.py. Composes with P24 (fused_moe config "
            "overlay — different site), PN96b (Marlin workspace), P31 "
            "(router softmax)."
        ),
        "upstream_pr": 44557,
        "upstream_pr_relationship": "backport",
        "applies_to": {"vllm_version_range": (">=0.21.0", "<0.24.0")},
        "implementation_status": "full",
        "composes_with": ["P24", "PN96b", "P31"],
    },
    "PN352B": {
        "title": "Marlin MoE topk=8 reduce via Genesis Triton kernel (right call-site for the FP8 Marlin path)",
        "tier": "community",
        "family": "moe",
        "env_flag": "GENESIS_ENABLE_PN352B_MARLIN_MOE_SUM",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.moe.pn352b_marlin_moe_sum",
        "lifecycle": "experimental",
        "category": "kernel_perf",
        "credit": (
            "A/B 2026-06-15 on dev491 (PROD 35B): BUILT + VALIDATED but NOT a "
            "significant single-stream win — keep default OFF. Numeric gate: "
            "BIT-IDENTICAL to ops.moe_sum (max_abs_diff=0.0 at M=4/8/16). "
            "Stability: 3-gen crash-test PASSED + kernel fires (moe_sum_topk "
            "tokens=4 topk=8 hidden=2048) — DEFEATS the parked-PN352 stream race "
            "(the Marlin-site override runs inside the FULL-capture apply). "
            "Perf: clean decode-TPOT n=125 baseline 4.776ms vs on 4.725ms = "
            "-1.07%, Welch p=0.38 NOT SIGNIFICANT (the at::sum_out is fast for "
            "the tiny M=8 decode; the reduce is not the latency-bound bottleneck "
            "— consistent with the regression-rootcause model). Candidate for a "
            "MULTI-CONC A/B (M=64 at conc=8 makes the reduce bigger and the "
            "generic at::sum_out relatively slower). Supersedes the broken "
            "parked PN352 (wrong call-site + stream race) regardless of the "
            "single-stream null result. Original design follows.\n"
            "Genesis-original (sibling of PN352 / counterpart of OPEN "
            "vllm#44557). The parked PN352 text-patched "
            "fused_moe.py::fused_experts_impl, but the live FP8 Marlin MoE "
            "decode NEVER executes that site — MarlinExpertsBase returns "
            "TopKWeightAndReduceNoOP for the modular finalize and reduces via "
            "self.moe_sum (marlin_moe.py:487/959 -> :996 ops.moe_sum). "
            "_moe_C.moe_sum has fast paths only for topk 2/3/4; Qwen3.6-A3B "
            "routes 8 experts/token so it falls through to the generic "
            "at::sum_out reduce fired 40x/forward on the decode critical path. "
            "PN352B monkey-patches MarlinExpertsBase.moe_sum (PN96b style, the "
            "RIGHT site) to route topk not in (2,3,4) through the verified "
            "moe_sum_topk Triton kernel, falling back to ops.moe_sum on any "
            "failure. Avoids the parked-PN352 stream race because the override "
            "runs inside apply() on the FULL_AND_PIECEWISE capture/replay "
            "stream (the Triton launch is captured on the correct stream) + "
            "pre-warms the kernel at install for the decode shapes so it never "
            "JITs during capture. Non-regressing by construction (removes a "
            "serial fixed-latency reduction; does NOT touch parallelism/"
            "occupancy/batch). Est -1..3% decode TPOT, helps ALL variants incl. "
            "temp=0. fp32 accumulate (same tolerance class as topk 2/3/4 CUDA "
            "kernels, not bit-identical). A/B + numeric-gate + crash-watch "
            "pending. Supersedes the parked PN352 on this model."
        ),
        "upstream_pr": 44557,
        "upstream_pr_relationship": "backport",
        "applies_to": {"vllm_version_range": (">=0.21.0", "<0.24.0")},
        "implementation_status": "full",
        "composes_with": ["P24", "PN96b", "P31", "PN368"],
    },
    "PN368": {
        "title": "Marlin MoE w13 reduce-mode wire (env-gated atomic-add, dense-path heuristic parity)",
        "tier": "community",
        "family": "moe",
        "env_flag": "GENESIS_ENABLE_PN368_MARLIN_MOE_ATOMIC_ADD",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.moe.pn368_marlin_moe_atomic_add_wire",
        "lifecycle": "experimental",
        "category": "kernel_perf",
        "credit": (
            "Genesis-original 2026-06-10 — wires upstream's OWN dense-"
            "path reduce-mode heuristic (should_use_atomic_add_reduce, "
            "marlin_utils.py) into the MoE Marlin w13 GEMM, where "
            "upstream hardcodes use_atomic_add=False, "
            "use_fp32_reduce=True at both moe_wna16_marlin_gemm call "
            "sites (experts/marlin_moe.py). Verified at pin g303916e93 "
            "(0.22.1rc1.dev259, live container 2026-06-10): Qwen3.6-35B-"
            "A3B-FP8 runs MarlinExperts on SM 8.6 (TritonExperts "
            "excluded — supports_fp8() False on Ampere); for the w13 "
            "GEMM the heuristic APPROVES atomic-add (n=w13_num_shards*N"
            "=512<2048, k=K=2048>=2048, --dtype float16, PROD launcher "
            "sets VLLM_MARLIN_USE_ATOMIC_ADD=1; the sm8x refusal applies "
            "only to bfloat16). The w2 GEMM (n=2048, k=256) fails the "
            "heuristic — v1 deliberately leaves it untouched. Mutual-"
            "exclusion VERIFIED in csrc at the same commit: the fp32 "
            "global-reduce c_tmp buffer is allocated only under "
            "use_fp32_reduce && !use_atomic_add (moe ops.cu L692) and "
            "the kernel consults use_fp32_reduce only inside the "
            "!use_atomic_add global-reduce branch (marlin_template.h "
            "L2162-2165) — so use_fp32_reduce stays True, exactly like "
            "the dense path (which passes the heuristic result alongside "
            "USE_FP32_REDUCE_DEFAULT=True). Heuristic replicated inline "
            "in the patched text (it ignores m at this pin) so the "
            "upstream function name stays a clean drift marker. Text "
            "install always-on (runtime branch env-gated, bit-identical "
            "when unset); GENESIS_DISABLE_PN368_INSTALL=1 skips even the "
            "text. First enabled call logs the resolved reduce mode "
            "(iron rule #3 ON-vs-OFF observability). Composes with P37 "
            "(same file, disjoint anchors), PN96b (Marlin workspace "
            "runtime hook), PN352 (fused_moe.py — different file), P24."
        ),
        "upstream_pr": None,
        "applies_to": {"vllm_version_range": (">=0.22.0", "<0.24.0")},
        "implementation_status": "full",
        # Anchor collision with P23_WIRE on the marlin_moe.py w13
        # use_fp32_reduce block — see P23_WIRE.conflicts_with (deep-audit #2).
        "conflicts_with": ["P23_WIRE"],
        "composes_with": ["P37", "PN96b", "PN352", "P24"],
    },
    "PN365": {
        "title": "Fused GDN qkv|z|b|a single-GEMM input projection (port of OPEN vllm#42746)",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_PN365_GDN_GEMM_FUSE",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.attention.gdn.pn365_gdn_qkvz_ba_fuse_gemm",
        "lifecycle": "experimental",
        "category": "kernel_perf",
        "credit": (
            "Port of OPEN vllm#42746 (author forrestl111, 2026-05-15). "
            "Collapses the 2 GDN linear_attn input GEMMs (in_proj_qkvz + "
            "in_proj_ba) into a single MergedColumnParallelLinear named "
            "in_proj_qkvzba with output_sizes [key, key, value, value, "
            "n_v, n_v]. Bit-equivalent at the matmul level — same weight "
            "values, just concatenated along output dim. Win comes from "
            "(a) one kernel launch instead of two (saves ~5-10us/layer on "
            "A5000 from cudaLaunchKernel overhead); (b) larger N "
            "(12352 vs 12288 + 64 separately) gives cuBLASLt a better tile "
            "selection on Ampere SM 8.6 — the in_proj_ba GEMM at N=64 "
            "wastes most of the SM throughput because tiles pad to 64x64 "
            "minimum. Author bench (RTX PRO 6000 sm_120, Qwen3.5-35B-A3B "
            "NVFP4, TP=1): +3.7% TPOT @ C=3, +3.3% @ C=5, +2.4% @ C=8; "
            "at SLO TPOT<=10ms max concurrency rises 5 -> 6 (+20%), "
            "request throughput 2.95 -> 3.50 req/s (+19%). On Ampere SM "
            "8.6 the launch-overhead win carries fully (~+1.5-2%); the "
            "cuBLASLt tile win partially carries (~+0.5-1%) -> combined "
            "+1-3% wall_TPS single-stream estimate on Qwen3.6-35B-A3B "
            "FP8 / 2x A5000. Strict no-regression: when env flag is unset, "
            "the patched conditional branches are bit-equivalent to "
            "upstream. Default OFF. Three text-patch anchors (GDN ctor, "
            "forward_cuda Part 1, forward_cuda gqa-split short-circuit) "
            "on qwen_gdn_linear_attn.py + one anchor (load_weights "
            "stacked_params_mapping) on qwen3_5.py. Runtime detection of "
            "the fused Linear (no env-flag read inside load_weights). "
            "LoRA-incompatible (auto-disabled when vllm_config.lora_"
            "config is set). HARD CONFLICT with PN204: PN204 wraps the "
            "two in_proj GEMMs in dual streams; PN365 fuses them into "
            "one GEMM with nothing to overlap, AND PN204's anchor no "
            "longer matches. Operator must set GENESIS_ENABLE_PN204_DUAL_"
            "STREAM_INPROJ=0 when enabling PN365 (apply() refuses with "
            "'failed' status if both env flags are on). Composes cleanly "
            "with PN350 (downstream of conv1d, different site), PN54 "
            "(.contiguous() dedup — PN365 fused path already emits "
            "contiguous), PN11 (a/b contiguous — PN365 path already "
            "calls .contiguous()), P28 (gdn_core_attn pool, downstream), "
            "PN50 (GDN fused proj, post-conv). Drift markers auto-SKIP "
            "if upstream lands #42746 (in_proj_qkvzba / "
            "VLLM_GDN_FUSE_QKVZBA / create_in_proj_qkvzba)."
        ),
        "upstream_pr": 42746,
        "upstream_pr_relationship": "backport",
        "applies_to": {"vllm_version_range": (">=0.21.0", "<0.24.0")},
        "requires_patches": [],
        "conflicts_with": ["PN204"],  # same forward_cuda Part 1 site + semantic conflict
        "implementation_status": "full",
        "composes_with": ["PN350", "PN54", "PN11", "P28", "PN50", "PN340", "PN341", "PN345"],
    },
    "PN349": {
        "title": "Gemma 4 KV-shared k_norm/v_norm skip (vendor of OPEN vllm#44797)",
        "tier": "community",
        "family": "model_compat.gemma4",
        "env_flag": "GENESIS_ENABLE_PN349",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.model_compat.gemma4.pn349_gemma4_kv_shared_norm_skip",
        "lifecycle": "experimental",
        "category": "correctness",
        "credit": (
            "Genesis vendoring of OPEN upstream PR vllm#44797 (Anai-Guo, "
            "2026-06-08). Gemma4Attention.__init__ unconditionally registers "
            "self.k_norm and self.v_norm RMSNorm Modules for every layer, "
            "but KV-shared layers (last N layers per num_kv_shared_layers) "
            "have checkpoints that OMIT k_norm/v_norm weights. Result: "
            "Module params allocated at default-init (ones for k_norm, "
            "no-weight for v_norm), never receive checkpoint values, "
            "silent ~1% logit drift class if a future refactor accidentally "
            "removes the `if not self.is_kv_shared_layer:` guard around "
            "norm application (line 522 in our pin). Fix: 2 sub-patches: "
            "(1) drop unconditional K/V norm allocation (keep q_norm), "
            "(2) gated K/V norm allocation AFTER is_kv_shared_layer is "
            "determined — None for shared, RMSNorm for owners. Direct "
            "hit for Gemma 4 26B-A4B + 31B PROD; no-op on Qwen3.6. "
            "Composes with all G4_* patches (different concerns: AWQ, "
            "FP8 block, Marlin K-pad, etc.). Risk: low — behaviour-"
            "preserved on both branches."
        ),
        "upstream_pr": 44797,
        "upstream_pr_relationship": "backport",
        "applies_to": {"vllm_version_range": (">=0.21.0", "<0.24.0")},
        "implementation_status": "full",
        "composes_with": ["G4_01", "G4_02", "G4_06", "G4_19"],
    },
    "PN351": {
        "title": "Triton unified_attention head_dim>=512 tune (vendor of OPEN vllm#43257)",
        "tier": "community",
        "family": "attention",
        "env_flag": "GENESIS_ENABLE_PN351",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.attention.pn351_triton_unified_attention_large_head",
        "lifecycle": "experimental",
        "category": "kernel_perf",
        "credit": (
            "Genesis vendoring of OPEN upstream PR vllm#43257 (ShuaiShao93, "
            "2026-05-20). triton_unified_attention.py hardcodes num_warps=4 "
            "+ num_stages=3 + 32-tile for all head sizes. For head_dim >= "
            "512 (Gemma 4 31B + 26B-A4B global-attention heads) this hits "
            "a register cliff that caps occupancy at 6-13% (Hopper) "
            "instead of 25-40% achievable with num_warps=8 + num_stages=2 "
            "+ 64-tile. Same architectural class on Ampere SM 8.6 (A5000). "
            "Two sub-patches: (1) _get_tile_size head_dim>=512 + FP8 + "
            "prefill branch returns 64 (vs default 32); (2) kernel launch "
            "adds num_warps=8/num_stages=2 conditional on head_dim>=512. "
            "Sub-2 (kernel launch) is MULTI-ANCHOR (batch-3 2026-06-13, "
            "corrected after review): three mutually-exclusive launch-site "
            "variants under required-at-least-one semantics — (A) current "
            "pin g303916e93 (no **launch_kwargs), (B) upstream main "
            "pre-vllm#45151 (main refactored the launch to a **launch_kwargs "
            "splat), (C) upstream main post-vllm#45151 (7 fused-quant kwargs "
            "spliced before **launch_kwargs). In B/C our literals are "
            "inserted BEFORE the splat; they cannot collide with main's "
            "native launch_kwargs['num_warps'] because that fires only on "
            "the head_size==256 B200 tuned_large_head path, disjoint from "
            "PN351's head_size>=512 FP8 gate. So PN351 keeps applying across "
            "both the launch_kwargs refactor and the #45151 insertion. "
            "FP8 gate keeps shmem within ~99 KiB opt-in budget. Expected "
            "-3-7% decode_TPOT on Gemma 4 31B FP8 prefill. No-op on "
            "Qwen3.6 head_dim=128. Composes with PN29x + PN345 (different "
            "files); composes with G4_* family. Risk: LOW (gated path)."
        ),
        "anchor_breaker_watch": {
            "pr": 45151,
            "also": "upstream-main launch_kwargs refactor",
            "mitigation": (
                "multi-anchor (A current-pin / B main-pre-45151 / "
                "C main-post-45151)"
            ),
        },
        "upstream_pr": 43257,
        "upstream_pr_relationship": "backport",
        "applies_to": {"vllm_version_range": (">=0.21.0", "<0.24.0")},
        "implementation_status": "full",
        "composes_with": ["PN298", "PN299", "PN299B", "PN299C", "PN299D", "PN299E", "PN345"],  # PN29 consolidated into PN298 (2026-06-19)
    },
    "PN353A": {
        "title": "TurboQuant MetadataBuilder workspace reserve (backport OPEN vllm#44053)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_PN353A",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.pn353a_tq_builder_workspace_reserve",
        "lifecycle": "experimental",
        "category": "stability",
        "credit": (
            "Genesis backport of OPEN vllm#44053 (Bot1822, 2026-06-04, "
            "supersedes #40798). Closes long-context "
            "AssertionError: workspace locked, cannot grow on TQ + chunked "
            "prefill. Reserves max decode + continuation-prefill scratch "
            "from TurboQuantMetadataBuilder.__init__ BEFORE CUDA-graph "
            "capture lock. Composes additively with PN118 (PN118 covers "
            "per-ubatch decode slots via its custom reserve(); PN353A "
            "covers continuation-prefill K/V dequant buffers PN118 misses, "
            "plus reserves via stock get_simultaneous so it does not "
            "require PN118's WorkspaceManager method additions). Direct "
            "hit for our PROD 35B-A3B FP8 + MTP K=3 + TQ k8v4 + 320K ctx."
        ),
        "upstream_pr": 44053,
        "upstream_pr_relationship": "backport",
        # KEPT ACTIVE 2026-06-24 (pin bump dev148 -> dev301): #44053 is IN
        # dev301 (the TQ MetadataBuilder reserve is fixed native on dev301;
        # anchor drifted per the anchor-SOT regen) so #44053 supersedes this
        # patch on its own. BUT NOT version-gated out: PN399.requires_patches
        # includes PN353A (verified) — PN399 anchors PN353A's LIVE-applied
        # reserve block. Coupled retirement waits for PN399's upstream
        # (#46067). Range deliberately left open through <0.24.0 so PN353A
        # stays applicable on dev301 and the PN399 chain does not break.
        "superseded_by": (
            "vllm#44053 (MERGED, IN dev301 — TQ MetadataBuilder workspace "
            "reserve is fixed native on dev301+; anchor drifted per the "
            "anchor-SOT regen). RETAINED active on dev301: PN399."
            "requires_patches anchors PN353A's live output, so coupled "
            "retirement waits for PN399's upstream (#46067). NOT "
            "version-gated out of dev301 — see applies_to range."
        ),
        "applies_to": {
            "is_turboquant": True,
            "vllm_version_range": (">=0.21.0", "<0.24.0"),
        },
        "implementation_status": "full",
        "composes_with": ["PN118", "PN353B", "PN399"],  # PN399 (when ON)
                                        # anchors PN353A's live output and
                                        # removes ONLY the now-dead decode-
                                        # scratch get_simultaneous reservation,
                                        # keeping the continuation-prefill K/V
                                        # reservation byte-intact. PN353A source
                                        # is NOT edited.
        "conflicts_with": [],
    },
    "PN353B": {
        "title": "TurboQuant prefill CUDA-graph capture safety (backport OPEN vllm#43747, closes vllm#40807)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_PN353B",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.pn353b_tq_prefill_cg_capture_safety",
        "lifecycle": "experimental",
        "category": "spec_decode",
        "credit": (
            "Genesis backport of OPEN vllm#43747 (oneraghavan, 2026-05-15) "
            "closing vllm#40807 (noonghunna). Three coupled fixes for the "
            "engine-init crash on TQ + MTP + chunked-prefill (our exact "
            "PROD config): (1) downgrade _cudagraph_support from "
            "UNIFORM_BATCH to UNIFORM_SINGLE_TOKEN_DECODE so spec-decode "
            "K+1 verify batches fall to PIECEWISE instead of hitting the "
            "continuation .tolist() crash; (2) build_for_cudagraph_capture "
            "always populates seq_lens_cpu / query_start_loc_cpu so "
            "_prefill_attention reads from CPU tensors even on the "
            "fallback branch; (3) defense-in-depth zero-return in "
            "_prefill_attention continuation when "
            "torch.cuda.is_current_stream_capturing() is True. Conflicts "
            "with P65 (P65 also downgrades the same ClassVar via a "
            "classmethod approach; P65 is DEFAULT OFF so no live conflict "
            "today). Supersedes-in-effect: with PN353B applied, P65's "
            "downgrade is redundant. P78 reference removed 2026-06-11: "
            "P78 retired (upstream absorbed its Sites B/C/D/E — CPU-"
            "mirror metadata is native in turboquant_attn.py), so the "
            "former 'composes with P78 Sites C/D/E' relationship is now "
            "PN353B-on-upstream-native, no companion patch involved. "
            "Composes with PN116 / P101 (different code paths). Risk "
            "LOW-MEDIUM: ~5-8 %% TPS hit "
            "on K+1 batches but crash without it. Direct hit for our "
            "PROD 35B-A3B FP8 + MTP K=3 + TQ k8v4."
        ),
        "upstream_pr": 43747,
        "upstream_pr_relationship": "backport",
        "applies_to": {
            "is_turboquant": True,
            "vllm_version_range": (">=0.21.0", "<0.24.0"),
        },
        "implementation_status": "full",
        # P78 dropped 2026-06-11 (retired — its Sites B/C/D/E are
        # upstream-native on this pin; see P78 retire note).
        "composes_with": ["P101", "PN116", "PN118", "PN353A"],
        "conflicts_with": ["P65"],
    },
    "PN364": {
        "title": "Hybrid GDN/Mamba/MRoPE startup warmup (vendor of OPEN vllm#43642)",
        "tier": "community",
        "family": "compile_safety",
        "env_flag": "GENESIS_ENABLE_PN364_HYBRID_GDN_WARMUP",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.compile_safety.pn364_hybrid_gdn_mamba_warmup",
        "lifecycle": "experimental",
        "category": "ttft_warmup",
        "credit": (
            "Genesis backport of OPEN PR vllm#43642 (hybrid GDN/Mamba/MRoPE "
            "kernel warmup). Closes the LAST 4-5 first-request JIT-spike "
            "kernels that PN126/PN128/PN129/PN130 do NOT cover: "
            "_causal_conv1d_update_kernel (DECODE shape, different from "
            "PN126 Pass 1 prefill), "
            "fused_recurrent_gated_delta_rule_packed_decode_kernel, "
            "MRotaryEmbedding.forward_cuda first-shape, _kv_block_zeroer "
            "warmup, extra capture-size single-token-decode shapes. "
            "Wraps Worker.compile_or_warm_up_model AFTER PN126's chain "
            "to issue extra warmup passes with single-token decode shape "
            "(num_tokens = max_num_seqs × 1, distinct from PN126's "
            "spec-decode-uniform max_num_seqs × (1+num_spec) shape). "
            "Per previous-session journal section 'Next actionable steps "
            "#1', 10 JIT-spike kernels were listed as still firing on "
            "first user request; PN126/128/129/130 covered 6, PN364 "
            "closes the remaining 4-5. Expected: TTFT -200-1500 ms on "
            "first user request after restart; CV tightening on bench "
            "mean (less variance from mid-bench JIT events). No effect "
            "on steady-state wall_TPS in mean. Auto-skip on V2 model "
            "runner (V2 has equivalent built-in), enforce_eager=True "
            "(no cudagraphs), or non-hybrid models. Strict no-regression "
            "fallback: try/except wraps every pass, partial completion "
            "acceptable, engine continues. Composes with PN126/PN128/"
            "PN129/PN130 (same wrapper-chaining pattern, distinct "
            "kernel-target sets, zero overlap)."
        ),
        "upstream_pr": 43642,
        "upstream_pr_relationship": "backport",
        "applies_to": {"vllm_version_range": (">=0.21.0", "<0.24.0")},
        "implementation_status": "full",
        "composes_with": ["PN126", "PN128", "PN129", "PN130"],
    },
    "PN363": {
        "title": "force_max_spec_tokens for suffix decoding — FULL CG dispatch (vendor of OPEN vllm#43114)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_PN363",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.spec_decode.pn363_force_max_spec_tokens",
        "lifecycle": "experimental",
        "category": "spec_decode",
        "credit": (
            "Genesis vendor of OPEN PR vllm#43114 (Csrayz, 2026-05-19, "
            "last touched 2026-06-01). Adds force_max_spec_tokens=True "
            "path to SuffixDecodingProposer so short draft lists get "
            "padded to num_speculative_tokens with eos_token_id. "
            "Padding gives the target side uniform num_scheduled_tokens "
            "per request → get_uniform_token_count() returns a non-None "
            "value → dispatcher selects CUDAGraphMode.FULL instead of "
            "PIECEWISE. Author measured ~88% of decode steps were "
            "PIECEWISE without padding and ~99.7% become FULL after, "
            "with avg ITL -15% at 8-concurrency on MiniMaxM2 (TP8+EP). "
            "Bit-equivalent output on the GREEDY rejection sampler "
            "(pad token is rejected deterministically). UNSAFE for "
            "PROBABILISTIC rejection (draft_p == 0 for eos pad token "
            "breaks the min(1, target_p/draft_p) ratio). Genesis PROD "
            "uses MTP K=3 with draft_sample_method=GREEDY (probabilistic "
            "is commented out in qwen3.6-35b-a3b-fp8.yaml:60-73 — a proven "
            "-5.9% TPS / -10% accept regression on our shape, 2026-05-15 "
            "rollback) so (a) the GREEDY rejection sampler is active and "
            "this patch is BIT-EQUIVALENT on it (pad rejected "
            "deterministically) and (b) SuffixDecodingProposer is DEFAULT "
            "OFF so not instantiated anyway. Ships as DEFAULT OFF for "
            "audit clarity + A/B reuse (suffix is a candidate bench "
            "lever for chat workloads via P75). MTP-side adaptation "
            "(scheduler num_scheduled_tokens padding + draft_probs "
            "one-hot injection) deferred to PN364 — see PN363 module "
            "docstring 'PN364 design note' for the correct "
            "probabilistic-safe approach. Composes cleanly with "
            "PN340 / PN341 / PN348 / PN357 / PN361 / PN133 / "
            "G_DYNAMIC_K_MTP (all different files or different "
            "anchors). Composes with P75 (extends the suffix proposer "
            "P75 enables). Disable via GENESIS_DISABLE_PN363=1. "
            "Auto-no-op when vllm#43114 merges (drift marker: "
            "'[Genesis PN363' on every injected block; TextPatcher "
            "idempotency guards re-apply)."
        ),
        "upstream_pr": 43114,
        "upstream_pr_relationship": "backport",
        "applies_to": {"vllm_version_range": (">=0.21.0", "<0.24.0")},
        "implementation_status": "full",
        "composes_with": ["P75", "PN340", "PN341", "PN348", "PN357", "PN361"],
    },
    "PN361": {
        "title": "Spec-decode fail-closed on missing draft probs (vendor of OPEN vllm#44869)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_PN361",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.spec_decode.pn361_spec_decode_fail_closed_missing_probs",
        "lifecycle": "experimental",
        "category": "observability",
        "credit": (
            "Genesis vendoring of OPEN upstream PR vllm#44869 (masterFoad, "
            "2026-06-08). GPUModelRunner._get_spec_decode_draft_probs "
            "today silently returns None + logs warning when a request "
            "with drafted tokens has no cached draft-probability row → "
            "caller silently downgrades from probabilistic to greedy "
            "rejection sampler. Our PROD spec_decode_config uses "
            "draft_sample_method=GREEDY (probabilistic is commented out in "
            "qwen3.6-35b-a3b-fp8.yaml:60-73 — proven -5.9% TPS / -10% "
            "accept regression on our shape), so this guard is DEAD "
            "INSURANCE on PROD: greedy draft emits one-hot draft_probs and "
            "never produces a missing-probs row, so the fail-closed raise "
            "never fires here — it protects only a future config that "
            "flips draft_sample_method to probabilistic. 20-LOC "
            "fix replaces logger.warning + return None with raise "
            "RuntimeError carrying a precise message. Fail-closed pattern: "
            "converts silent quality regression to visible exception. "
            "Disable via GENESIS_DISABLE_PN361=1 if PROD needs the "
            "silent-fallback behaviour. Composes with PN340 + PN341 "
            "(different methods in same file)."
        ),
        "upstream_pr": 44869,
        "upstream_pr_relationship": "backport",
        "applies_to": {"vllm_version_range": (">=0.21.0", "<0.24.0")},
        "implementation_status": "full",
        "composes_with": ["PN340", "PN341"],
    },
    "PN357": {
        "title": "Optimize remapped greedy draft token selection (vendor of OPEN vllm#43349)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_PN357",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.spec_decode.pn357_draft_greedy_speedup",
        "lifecycle": "experimental",
        "category": "spec_decode",
        "credit": (
            "Genesis vendor of OPEN PR vllm#43349 (yewentao256, 2026-06-XX). "
            "Bypasses dense [N_tokens, target_vocab] scatter in greedy "
            "spec-decode draft path for Eagle3 / DFlash / Eagle3-DeepSeek "
            "models that use draft_id_to_target_id remap. Replaces "
            "compute_logits().argmax() with argmax-in-draft-vocab + "
            "remap-add. Bit-identical to prior path (PR author "
            "mismatch_count tests). Author measures 37-81% kernel speedup "
            "across batch 16-1024 on Llama_eagle3. Adds class attribute "
            "supports_remapped_top_tokens=True + new get_top_tokens method "
            "+ proposer-side auto-resolver for use_local_argmax_reduction. "
            "MTP IMPACT: zero — MTP draft model has no draft_id_to_target_id "
            "(shares target lm_head), so auto-resolver returns False and "
            "the existing MTP path is unchanged. Patch is INSURANCE for "
            "when we A/B 27B + DFlash drafting. Supersedes PN22's "
            "if-fallback branch when both are on (PN357 detects PN22 "
            "marker and swaps the dense-scatter fallback). Composes with "
            "PN22, PN340, PN341, PN348, PN361. Disable via "
            "GENESIS_DISABLE_PN357=1. Auto-no-op when vllm#43349 merges "
            "(drift markers on supports_remapped_top_tokens class attr "
            "and _resolve_local_argmax_reduction method name)."
        ),
        "upstream_pr": 43349,
        "upstream_pr_relationship": "backport",
        "applies_to": {"vllm_version_range": (">=0.21.0", "<0.24.0")},
        "implementation_status": "full",
        "composes_with": ["PN22", "PN340", "PN341", "PN348", "PN361"],
    },
    "PN346": {
        "title": "Mamba/GDN cache hit boundary fix for MTP + prefix caching (vendor of OPEN vllm#43650)",
        "tier": "community",
        "family": "kv_cache",
        "env_flag": "GENESIS_ENABLE_PN346",
        # [Preflight triage 2026-06-11 §2] Honest alignment with module
        # reality: pn346_mamba_mtp_apc_boundary.apply() ignores
        # GENESIS_ENABLE_PN346 entirely and honors ONLY the opt-out
        # GENESIS_DISABLE_PN346 → the patch is effectively default-ON
        # (PROD boot line 87 shows it applied with no enable flag set).
        # Registry default_on=True records that truth; module behavior
        # deliberately unchanged (correctness vendor, opt-out-only by
        # design). Operators disable via GENESIS_DISABLE_PN346=1.
        "default_on": True,
        "apply_module": "sndr.engines.vllm.patches.kv_cache.pn346_mamba_mtp_apc_boundary",
        "lifecycle": "experimental",
        "category": "correctness",
        "credit": (
            "Genesis vendoring of OPEN upstream PR vllm#43650 (6-LOC). "
            "MambaManager.find_longest_cache_hit (line 986 in our pin) "
            "matches the full prefix-cache hit window when EAGLE/MTP is "
            "active, but its state-cache layout is [null, ..., null, "
            "state_block] — the LAST block IS the SSM state itself. "
            "FullAttentionManager handles this by popping the last matched "
            "block AFTER the loop (safe for token KVs). Mamba can't pop — "
            "that destroys the state — so it leaves the partially-accepted "
            "final state block in the result, which the MTP verify step "
            "reads stale → quality drift. PR author measured silent "
            "-1.6 pp GSM8K accuracy on Qwen3.5/3.6-35B-A3B FP8 + MTP K=3 + "
            "--enable-prefix-caching (our exact PROD shape). The 6-LOC "
            "fix walks max_num_blocks back by 1 BEFORE the search loop "
            "when drop_eagle_block=True, so the final state block is "
            "never considered. Trade-off: small QPS regression on the "
            "prefix-cache overlap path (author: 18.6 QPS → 15.5; output "
            "TPS 2494 → 1983) in exchange for accuracy parity with the "
            "no-MTP-no-APC baseline (0.916 → 0.914). Trans-anchor note: "
            "the upstream PR uses local var name 'use_eagle' (author's "
            "fork); our pin (= upstream main) uses 'drop_eagle_block' as "
            "the function parameter — Genesis patch uses the parameter "
            "name that actually exists in our file. Composes with PN340 "
            "+ PN341 + PN345 (different files / layers); independent of "
            "P83 (FullAttentionManager path, retired 2026-06-11). "
            "Composes with P85 via P85's Site 2 dual anchor variants "
            "(2026-06-11 §5): PN346's 4-line anchor is a byte-identical "
            "subsequence inside P85's Site 2 anchor; PN346 "
            "boot-dispatches BEFORE P85, so P85 carries a "
            "post-PN346-shaped variant assembled from PN346's own "
            "constants."
        ),
        "upstream_pr": 43650,
        "upstream_pr_relationship": "backport",
        "upstream_issue": 43559,
        # 2026-06-17 (0.23.1 pin-bump): cap bumped <0.23.0 -> <0.24.0. PN346 is
        # default_on but the stale <0.23.0 cap silently no-op'd this default-on
        # MTP+APC+mamba accuracy fix on the deployed 0.23.1 pin. PR #43650 is
        # OPEN (NOT merged/superseded); anchor (MambaManager.find_longest_cache_hit
        # 4-line block) verified byte-present in pristine upstream source at the
        # 0.23.1 build commit 4c626633. Restored for the 0.23.x line; next minor
        # forces a re-verify. Lockstep with sibling PN346B (coordinator half).
        "applies_to": {"vllm_version_range": (">=0.21.0", "<0.24.0")},
        "implementation_status": "full",
        # PN346B: coordinator half of the SAME fix (#45614) — manager
        # half (PN346) and coordinator half MUST ship together.
        "composes_with": ["PN340", "PN341", "PN345", "P85", "PN346B"],  # P85: Site 2 dual variants, PN346 first
    },
    "PN346B": {
        "title": "Mamba/GDN + EAGLE/MTP + APC coordinator curr_hit_length clamp (coordinator half of OPEN vllm#45614; sibling of PN346)",
        "tier": "community",
        "family": "kv_cache",
        "env_flag": "GENESIS_ENABLE_PN346B",
        # Mirrors PN346: the module is opt-out-only —
        # pn346b_mamba_mtp_apc_coordinator_clamp.apply() ignores
        # GENESIS_ENABLE_PN346B and honors ONLY the opt-out
        # GENESIS_DISABLE_PN346B → effectively default-ON. This is the
        # MISSING half of a correctness fix Genesis already ships
        # default-ON (PN346); a half-fix is worse than none, so both
        # halves enable together. Operators disable via
        # GENESIS_DISABLE_PN346B=1.
        "default_on": True,
        "apply_module": "sndr.engines.vllm.patches.kv_cache.pn346b_mamba_mtp_apc_coordinator_clamp",
        "lifecycle": "experimental",
        "category": "correctness",
        "credit": (
            "Genesis vendoring of OPEN upstream PR vllm#45614 "
            "(\"[Bugfix][Core] Fix Mamba prefix cache EAGLE hit\", closes "
            "vllm#43559) — the COORDINATOR half of the same fix whose "
            "MANAGER half Genesis already vendors as PN346. "
            "HybridKVCacheCoordinator.find_longest_cache_hit runs a "
            "while/for fixed-point loop that unconditionally overwrites "
            "curr_hit_length = _new_hit_length. On the eagle-drop branch "
            "_new_hit_length can be LONGER than the current candidate, so "
            "the naked assignment GROWS curr_hit_length on a verify pass "
            "and re-admits the partially-accepted final SSM state block — "
            "the exact poison PN346 walks back in the manager. The 1-LOC "
            "fix clamps curr_hit_length = min(curr_hit_length, "
            "_new_hit_length) so the hit length is monotonically "
            "non-increasing across the iteration, matching the "
            "manager-half guard. Pin-agnostic anchor: the 4-line region "
            "(elif _new_hit_length < curr_hit_length: → curr_hit_length = "
            "_new_hit_length) is byte-identical and grep-unique on BOTH "
            "PROD (0.21.1rc0 g626fa9bba, `if use_eagle:` above) and live "
            "dev491 (0.22.1rc1.dev491 g1033ffac2, `if drop_eagle_block:` "
            "above) — the divergent if-line is deliberately excluded from "
            "the anchor. `min(curr_hit_length` verified ABSENT on both "
            "pins → the clamp is genuinely missing. MUST compose with "
            "PN346 (manager half) — the two halves of #45614 are a unit. "
            "Self-skips once #45614 merges via the exact merged-shape "
            "drift marker. Upstream regression: "
            "test_hybrid_mamba_eagle_does_not_reuse_lookahead_state in "
            "tests/v1/core/test_prefix_caching.py."
        ),
        "upstream_pr": 45614,
        "upstream_pr_relationship": "backport",
        "upstream_issue": 43559,
        # 2026-06-17 (0.23.1 pin-bump): cap bumped <0.23.0 -> <0.24.0, lockstep
        # with sibling PN346. default_on coordinator-half clamp (#45614 OPEN, not
        # superseded); anchor (kv_cache_coordinator curr_hit_length 4-line block)
        # verified byte-present in upstream v0.23.1rc0 source. A half-fix (only
        # one of PN346/PN346B uncapped) is worse than none, so both move together.
        "applies_to": {"vllm_version_range": (">=0.21.0", "<0.24.0")},
        "implementation_status": "full",
        # PN346: the sibling MANAGER half — the two MUST ship together.
        "composes_with": ["PN346", "PN340", "PN341", "PN345"],
    },
    "PN347": {
        "title": "MarlinFP8 N==K silent corruption correctness fix (vendor of CLOSED vllm#44113; superseded by MERGED vllm#44735 on dev491+ — active only on <dev491 rollback pins, version-gated)",
        "tier": "community",
        "family": "quantization.marlin",
        "env_flag": "GENESIS_ENABLE_PN347",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.quantization.marlin.pn347_marlin_fp8_nk_correctness",
        "lifecycle": "experimental",
        "category": "correctness",
        "credit": (
            "Genesis vendoring of OPEN upstream PR vllm#44113 (shernshiou, "
            "Closes vllm#44110). MarlinFP8ScaledMMLinearKernel."
            "process_weights_after_loading uses a shape-tuple guard "
            "`if w_q.shape != (in, out)` to decide whether to transpose "
            "the weight to (K, N). For SQUARE weights (N == K) the "
            "tuples (N, K) and (K, N) are identical → the transpose is "
            "silently skipped, regardless of actual memory layout → "
            "Marlin multiplies a wrongly-laid-out weight → silent data "
            "corruption. Bug fires on sm_75-88 (no native FP8 compute, "
            "Marlin emulates) — our 2× A5000 (sm_86) is in scope. Direct "
            "hit for Qwen3.6 27B INT4 (square 4096² q/k/v/o_proj) and "
            "35B FP8 (square 5120² q/o_proj). Upstream A40 test result: "
            "BF16 output coherent vs FP8 output total token-stream "
            "collapse (',,,,,,,,...') on square layers. Fix: switch to "
            "`w_q.is_contiguous()` (a `.t()` view is non-contiguous "
            "regardless of square shape). One method, ~6 net lines. "
            "Risk: LOW (behaviour-preserved on non-square, on modelopt "
            "pre-transposed input, on sm_89+, on block-quant branch). "
            "Composes with PN77 (FP8 lm_head — different layer), PN81 "
            "(FP8 block-scaled — different branch), PN91/PN91B (INT4 "
            "AutoRound — different scheme), P87 (Marlin INT4 sub-tile "
            "pad — different kernel). CORRECTNESS category — not perf — "
            "quality gain is restoration, not improvement."
        ),
        "upstream_pr": 44113,
        "upstream_pr_relationship": "backport",
        "upstream_issue": 44110,
        # Upper bound capped <dev491 2026-06-14 (do NOT retire — load-bearing on
        # the dev259 rollback pin). VERIFIED against both live images: dev259
        # (kernels/linear/scaled_mm/marlin.py:87) STILL has the buggy
        # `if w_q.shape != (...)` transpose guard → bug present → PN347 applies
        # and protects the 35B square 5120² FP8 q/o_proj. dev491 REFACTORED the
        # method (guard removed; transpose responsibility moved to the caller via
        # the explicit `size_k_first` contract + `prepare_fp8_layer_for_marlin`),
        # so the square-matrix corruption cannot occur and PN347's anchor is
        # correctly absent. vllm#44113 was CLOSED-unmerged because upstream
        # solved it structurally, not via the PR. Capping the range makes PN347
        # version-SKIP (benign) on dev491+ instead of emitting a per-boot DRIFT
        # WARNING (required_anchor_missing), while staying ACTIVE on dev259.
        # Re-widen the upper bound only if a future pin reintroduces the guard.
        "applies_to": {"vllm_version_range": (">=0.21.0", "<0.22.1rc1.dev491")},
        "implementation_status": "full",
        # Composes with: PN77 (FP8 lm_head — different layer), P81 (FP8 block-
        # scaled M≤8 — different branch), P91/P91B (AutoRound INT4 — different
        # scheme), P87 (Marlin INT4 sub-tile pad — different kernel). NO overlap.
        # Fixed 2026-06-09: previous list referenced PN81/PN91/PN91B which do
        # not exist (correct prefixes are P, not PN).
        "composes_with": ["PN77", "P81", "P91", "P91B", "P87"],
    },
    "PN-FP8MOE-KPAD": {
        "title": "FP8 MoE intermediate thread-tile pad (FP8-core backport of OPEN vllm#45703)",
        "tier": "community",
        "family": "quantization.marlin",
        "env_flag": "GENESIS_ENABLE_PN_FP8MOE_KPAD",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.quantization.marlin.pn_fp8moe_kpad_marlin_moe",
        "lifecycle": "experimental",
        "category": "kernel",
        "credit": (
            "FP8-core backport of OPEN upstream PR vllm#45703 ('[Kernel] "
            "Extend Marlin thread-tile padding to MoE (WNA16 + FP8/MXFP8)'). "
            "The fused Marlin MoE kernel needs gate-up 2*intermediate % 128 "
            "== 0 and down intermediate % 64 == 0 (i.e. intermediate % 64 == "
            "0). A tile-misaligned MoE *intermediate* dim crashes with "
            "'Invalid thread config ... MKN=[16384,352,2816] num_bits=8 "
            "group_size=-1' — the DiffusionGemma N=352 case (352 % 64 == 32) "
            "on the dev491 pin (vllm/vllm-openai:nightly-1033ffac2). PN-FP8MOE-"
            "KPAD adds marlin_moe_padded_intermediate(intermediate, group) = "
            "round_up(intermediate, lcm(64, max(group,1))) and pads the "
            "intermediate at WEIGHT PREP: w13 gate/up shard-rows + w2 last-dim "
            "+ the CONVERTED FP8 scales (incl. the block-FP8 scale-row padding "
            "club-3090 punts on), then widens check_moe_marlin_supports_layer "
            "with allow_tile_padding so the misaligned FP8 layer USES fast "
            "Marlin instead of the slow WNA16 fallback. The pad is "
            "intrinsically SHAPE-gated (padded_n == n -> no-op): an "
            "already-tile-aligned intermediate (e.g. the PROD 35B's, dense "
            "FP8 routed through Marlin on Ampere) passes through UNCHANGED at "
            "ZERO cost -> NO 35B regression even when enabled. Padded region "
            "self-cancels (FP8 zero decodes to 0.0; scales zero-padded). Pad "
            "multiple is %64 (lcm(64,group)), NOT %128 — does NOT double-pad "
            "on top of the PR45295 round_up dense base dev491 already carries "
            "(P87/#40361 dense sibling). FP8-CORE SCOPE: exactly 3 vLLM files "
            "(marlin_utils.py, marlin_utils_fp8.py, compressed_tensors_moe.py); "
            "the mxfp8 hunk + INT-WNA16 oracle module of the current #45703 "
            "HEAD are out of scope. DiffusionGemma boot rig-validated 2026-06-17 "
            "(coherent TP=2 serve on dev101; drift-audited clean on dev148 "
            "2026-06-19); the 35B-regression bench + block-diffusion speed bench "
            "remain operator-gated -> default OFF, opt-in via "
            "GENESIS_ENABLE_PN_FP8MOE_KPAD=1, committing is zero-risk."
        ),
        "upstream_pr": 45703,
        "upstream_pr_relationship": "backport",
        # Lower bound: the PR45295 dense round_up base (round_up, math,
        # marlin_padded_nk) that this patch builds on appeared by the dev491
        # pin. VERIFIED present on 0.22.1rc1.dev491+g1033ffac2; the
        # marlin_moe_padded_intermediate func is ABSENT there (the delta we
        # add). No upper bound yet — the marlin_utils patcher's
        # `def marlin_moe_padded_intermediate` upstream-merge drift marker
        # self-skips the patch once #45703 lands (iron-rule-#11 outcome (a)).
        "applies_to": {
            "vllm_version_range": (">=0.22.1rc1.dev491", "<1.0.0"),
            "quant_format": [
                "fp8", "fp8_block", "compressed_tensors",
            ],
        },
        "implementation_status": "full",
        # Composes with the dense Marlin pad siblings (different code paths):
        # P87 (Marlin INT4 sub-tile output-dim pad — dense #40361), PN347
        # (MarlinFP8 N==K correctness — weight transpose, different hook).
        # On #45703 merge: retire PN-FP8MOE-KPAD + re-decide G4_08 / G4_02
        # per the retirement note.
        "composes_with": ["P87", "PN347"],
    },
    "PN348": {
        "title": "Qwen3.5/3.6 MTP backbone dedup (vendor of OPEN vllm#44644) — ~1 GiB/rank peak load VRAM + 1-3s boot",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_PN348",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.spec_decode.pn348_qwen3_mtp_backbone_dedup",
        "lifecycle": "experimental",
        "category": "memory_savings",
        "credit": (
            "Genesis vendoring of OPEN upstream PR vllm#44644 (Ntropic / "
            "Michael Schilling, 2026-06-05). Qwen3.5 MTP backbone in "
            "models/qwen3_5_mtp.py unconditionally allocates a fresh "
            "VocabParallelEmbedding for embed_tokens AND a fresh "
            "ParallelLMHead, even when the model config opts into sharing "
            "with the target via text_config.mtp_use_dedicated_embeddings"
            "=False (which Qwen3.5/3.6 checkpoints do — VERIFIED on our "
            "PROD Qwen3.6-35B-A3B-FP8 config.json). At "
            "vocab=248320 × hidden=2048 × 2B BF16 = 1.0 GiB per tensor, "
            "duplicate per worker = ~1.0 GiB embed_tokens + lm_head "
            "(TP-sharded). GAIN CLAIM CORRECTED 2026-06-11 (50-PR sweep "
            "re-study of #44644): the pin's proposer already reclaims the "
            "duplicate at steady-state, so the benefit is ~1 GiB/rank "
            "lower PEAK load-time VRAM + 1-3s faster boot — steady-state "
            "VRAM/TPS unchanged. ENABLED on the qwen3.6-35b-a3b-fp8 "
            "profile 2026-06-11 (enable+measure; A/B peak VRAM + boot "
            "time at next 35B restart). Three sub-patches text-patch "
            "qwen3_5_mtp.py: (1) embed_tokens predicate, "
            "(2) lm_head PPMissingLayer fallthrough, (3) weight-loader "
            "skip for embed_tokens/lm_head names when sharing. Required "
            "gate: PP=1 (matches our TP=2 PP=1). Per harsha20032020 "
            "PR #44720 already in pin, Qwen3.6 reuses the "
            "Qwen3_5MoeForConditionalGeneration class — single-file "
            "patch covers both 27B and 35B SKUs. Composes with PN108, "
            "PN133, PN290, PN340, PN341 (MTP runtime patches in "
            "different files; no anchor overlap). Composes with PN77 "
            "FP8 lm_head (targets the TARGET model's lm_head dtype; "
            "this gates lm_head EXISTENCE on MTP backbone). Risk: low "
            "(getattr default preserves legacy path on models that "
            "don't opt in; world_size==1 gate preserves legacy on PP>1). "
            "Watchlist: retire-on-merge row in tools/upstream_watchlist."
            "yaml sweep section; vllm#44943 (pre-fused expert loader) "
            "touches the same file — coordinate anchors if vendored."
        ),
        "upstream_pr": 44644,
        "upstream_pr_relationship": "backport",
        "applies_to": {"vllm_version_range": (">=0.21.0", "<0.24.0")},
        "implementation_status": "full",
        "composes_with": ["PN108", "PN133", "PN290", "PN340", "PN341", "PN77"],
    },
    "PN345": {
        "title": "Shmem-aware Triton autotune pruner (vendor of vllm#43047) for FLA chunk kernels",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_PN345",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.attention.gdn.pn345_shmem_aware_autotune_pruner",
        "lifecycle": "experimental",
        "category": "kernel_perf",
        "credit": (
            "Genesis vendoring of OPEN upstream PR vllm#43047 (shmem-aware "
            "autotune pruner). Triton autotune configs in chunk_delta_h.py "
            "and chunk_o.py were tuned for the largest opt-in shared-memory "
            "budget H100/H200 support (228 KiB). Smaller-budget GPUs hit "
            "triton.runtime.errors.OutOfResources at JIT time on configs "
            "that don't fit: Turing T4 ~64 KiB, Ampere A100 ~163 KiB, "
            "consumer Ampere A5000/3090 ~99 KiB (VERIFIED via "
            "torch.cuda.get_device_properties.shared_memory_per_block_optin "
            "= 101376 bytes on our PROD), Blackwell SM_120 ~99 KiB. "
            "Concrete math: chunk_gated_delta_rule_fwd_kernel_h_blockdim64 "
            "at BV=64 BT=64 num_stages=4 needs 4*64*64*4 (persistent fp32 "
            "b_h) + 4*(2*64*64*2 + 64*64*2) (per-stage bf16 b_w/b_k/b_v) "
            "+ 4096 = 160 KiB — exceeds A5000 budget by 64 KiB. The pruner "
            "drops it; smaller BV=32 num_stages=2 config (76 KiB) survives. "
            "Same shape for chunk_fwd_kernel_o. Author's SM_120 bench "
            "claims +3-7% GDN prefill TPS. We inline a minimal helper "
            "(~30 LOC) into each file as text-patch (no new file). Four "
            "sub-patches total across 2 files. STRICTLY DIFFERENT FROM "
            "Genesis PN298+PN299+PN299B+PN299C+PN299D+PN299E coarse "
            "env-based warps cap on 6 OTHER files — no anchor overlap, "
            "approaches compose (PN29x coarse pre-filter + PN345 precise "
            "shmem-budget filter). Closes upstream #36598; partially "
            "addresses #38918 + #36802 + #41063 + #32826. Risk: low "
            "(no-op when configs fit; on estimator failure keeps config). "
            "Composes with PN125 + PN204 + PN286 + PN340 + PN341 + PN29x."
        ),
        "upstream_pr": 43047,
        "upstream_pr_relationship": "backport",
        "applies_to": {"vllm_version_range": (">=0.21.0", "<0.24.0")},
        "implementation_status": "full",
        "composes_with": ["PN125", "PN204", "PN286", "PN340", "PN341"],
    },
    "PN341": {
        "title": "MTP decode bubbles reduction in gpu_model_runner (vendor of vllm#43955, sister to PN340)",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_PN341",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.attention.gdn.pn341_mtp_decode_bubbles_gpu_runner",
        "lifecycle": "experimental",
        "category": "perf_hotfix",
        "credit": (
            "Genesis vendoring of OPEN upstream PR vllm#43955 — second "
            "half (gpu_model_runner.py portion). Sister to PN340 which "
            "vendored the gdn_attn.py portion of the same PR. "
            "Closes the per-step ``num_accepted_tokens_event."
            "synchronize()`` CPU bubble on hybrid + MTP K=3 decode steps. "
            "Four sub-patches: (a) __init__ flag gating the GPU-only "
            "path on hybrid + spec_tokens > 0 + mamba_cache_mode != "
            "'align'; (b) _update_states_after_model_execute early "
            "return that captures the req_ids dict snapshot; "
            "(c) _compute_prev_positions optional prev_req_id_to_index "
            "param so the GPU-only path can pass its captured dict; "
            "(d) _prepare_inputs new branch that does GPU gather + "
            "masked_fill_ instead of waiting on the event. Net effect: "
            "no CPU-GPU round-trip, no NumPy intermediate, no event "
            "synchronize on every decode step. Our 35B PROD config "
            "(hybrid + MTP K=3 + mamba_cache_mode='none') is the target. "
            "Risk: medium-high (touches the sampling-adjacent code "
            "path). Each sub-patch required=False to soft-skip on "
            "anchor drift. Composes with PN125 + PN204 + PN286 + PN340."
        ),
        "upstream_pr": 43955,
        "upstream_pr_relationship": "backport",
        "applies_to": {"vllm_version_range": (">=0.21.0", "<0.24.0")},
        "implementation_status": "full",
        "composes_with": ["PN125", "PN204", "PN286", "PN340"],
        "requires_patches": [],  # PN340 + PN341 are independent halves of #43955
    },
    "PN340": {
        "title": "MTP decode bubbles reduction in GDN backend (vendor of vllm#43955)",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_PN340",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.attention.gdn.pn340_mtp_decode_bubbles_gdn_attn",
        "lifecycle": "experimental",
        "category": "perf_hotfix",
        "credit": (
            "Genesis vendoring of OPEN upstream PR vllm#43955 (Nekofish-L, "
            "REVIEW_REQUIRED as of 2026-06-09). The PR identifies that "
            "the GDN backend's metadata-build hot path launches a tiny "
            "CUDA kernel (torch.arange) on every call, plus does CPU-mask "
            "indexing (block_table_tensor[spec_sequence_masks_cpu, ...]) "
            "when a simple forward slice would suffice (spec rows are "
            "already compacted to the front of the batch; padded rows "
            "live at the back). Three sub-patches: (a) preallocate a "
            "spec_token_arange buffer at __init__, (b) build() slices "
            "into it instead of running torch.arange, (c) conditional "
            "copy_ that skips the no-op device-to-device copy when "
            "spec_token_indx already points at the preallocated arange. "
            "Author's profile screenshots show measurably smaller inter-"
            "step gaps. Direct hit for our Qwen3.6-A3B FP8 + TQ k8v4 + "
            "MTP K=3 stack — fires every decode step. Risk: medium-low "
            "(open PR, anchors may rebase before upstream merges; three "
            "sub-patches each required=False so partial drift is safe)."
        ),
        "upstream_pr": 43955,
        "upstream_pr_relationship": "backport",
        "applies_to": {"vllm_version_range": (">=0.21.0", "<0.24.0")},
        "implementation_status": "full",
        "composes_with": ["PN125", "PN204", "PN286"],
    },
    # ─── 2026-06-11 50-PR sweep wave 1 (PN370-PN375) ──────────────────
    # NOTE: PN370 must stay AFTER PN341 in this dict — the spec-driven
    # loop iterates insertion order and PN370's post-PN341 anchor
    # variant expects PN341's _prepare_inputs rewrite to land first
    # (legacy parking-lot order already PN341 → PN370; parity here).
    "PN370": {
        "title": "Async spec-decode accepted-counts race fix (vendor of OPEN vllm#45100)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_PN370_ASYNC_ACCEPT_RACE",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.spec_decode.pn370_async_accepted_counts_race",
        "lifecycle": "experimental",
        "category": "spec_decode",
        "credit": (
            "Genesis vendor of OPEN PR vllm#45100 (2026-06-11). Fixes an "
            "async speculative-decoding race for hybrid non-align "
            "Mamba/GDN models — the EXACT 35B PROD config (Qwen3.6-35B-"
            "A3B FP8 hybrid GDN+MoE, MTP K=3, async-scheduling ON). Two "
            "sub-fixes: (1) gpu_model_runner._prepare_inputs skips the "
            "racy CPU accepted-counts read under async + mamba_cache_mode "
            "!= 'align' — the CPU mirror races with the in-flight "
            "non-blocking D2H copy and with input-batch row moves "
            "(swap_states/condense); at a prefill-to-first-spec-decode "
            "transition GDN consumes another row's count and restores the "
            "wrong recurrent-state slot (prompt-memory loss, garbled "
            "early-EOS — upstream A/B: 16/20480 corrupted unpatched vs "
            "0/20480 patched). Stays device-authoritative: counts default "
            "to 1, the GPU correction kernel overwrites draft rows from "
            "valid_sampled_token_count; align mode keeps the synchronized "
            "CPU path. BONUS: deletes the per-step num_accepted_tokens_"
            "event.synchronize() + NumPy gather + copy_to_gpu on that "
            "path (~2-5% TPOT est. on 35B). (2) gdn_attn.py build() "
            "sizes FULL-cudagraph per-request metadata (spec_state_"
            "indices_tensor / spec_sequence_masks / spec_query_start_loc "
            "/ num_accepted_tokens + non-spec decode views) by m.num_reqs "
            "instead of token-padded m.num_actual_tokens. Anchors "
            "byte-verified count=1 on pin g303916e93 (0.22.1rc1.dev259). "
            "COMPOSITION: PN341 sub-patch 4 anchors the IDENTICAL "
            "_prepare_inputs block — PN370 carries dual anchor variants "
            "(pristine-shaped + post-PN341-shaped; the post anchor is "
            "imported from PN341_PREPARE_NEW, required-at-least-one, "
            "PN32/PN79 chain convention). ORDER: this entry must stay "
            "AFTER PN341's in PATCH_REGISTRY (SNDR_APPLY_VIA_SPECS "
            "parity) and the dispatch block already sits after PN341's "
            "in the parking lot; the reverse order is ast-valid but "
            "soft-skips PN341 sub-patch 4 (roadmap-sanctioned). PN290 "
            "composes (producer vs consumer side); PN340 composes "
            "(disjoint gdn_attn anchors). Drift markers self-skip when "
            "vllm#45100 merges: needs_cpu_accepted_counts (runner) + "
            "'token-padded for FULL graph replay' (gdn_attn). Opt-in "
            "DEFAULT OFF pending per-model A/B on 35B PROD async profile "
            "(adopt the PR's min-len/short-output distribution scoring "
            "as the corruption detector per roadmap synergy note)."
        ),
        "upstream_pr": 45100,
        "upstream_pr_relationship": "backport",
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {"vllm_version_range": (">=0.22.0", "<0.23.0")},
        "implementation_status": "full",
        "composes_with": ["PN341", "PN340", "PN290"],
    },
    "PN371": {
        "title": "Deferred ref-pinned encoder-cache eviction (vendor of CLOSED vllm#45199)",
        "tier": "community",
        "family": "multimodal",
        "env_flag": "GENESIS_ENABLE_PN371_ENCODER_CACHE_EVICTION",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.multimodal.pn371_encoder_cache_deferred_eviction",
        "lifecycle": "experimental",
        "category": "stability",
        "credit": (
            "Genesis vendor of vllm PR #45199 (fixes #38551) — whole-engine-"
            "fatal 'AssertionError: Encoder cache miss' when the scheduler "
            "frees an encoder-cache entry an in-flight request can still "
            "read: under async scheduling num_computed_tokens is advanced "
            "speculatively and rolled back on draft-token rejection, and "
            "entries are shared across requests with the same mm_hash — the "
            "exact Gemma-4 vision + MTP K=3 + async-scheduling triple of "
            "our gemma4 composes. Vendored at pin g303916e93 (11 anchors "
            "byte-verified, count==1, 2026-06-11): ref-counted EncoderCache "
            "(eager_eviction + caller-owned encoder_outputs dict + "
            "update_request + deferred free) in gpu/mm/encoder_cache.py; "
            "modular-runner eager_eviction=is_encoder_decoder; 5 legacy-"
            "runner tracker points (tracker attr _g_pn371_ec_tracker — "
            "renamed from upstream's encoder_cache_tracker for drift-marker "
            "hygiene). GENESIS EXTEND: the fatal assert in "
            "_gather_mm_embeddings is demoted to logger.warning_once + "
            "feature skip in the DRAFTER path only (shift_computed_tokens "
            "!= 0, sole non-zero caller is the MTP draft proposal); the "
            "verifier path keeps the hard assert — the target model "
            "verifies every draft token, so a skipped feature only degrades "
            "draft quality. Zero impact on text-only Qwen PROD (no mm "
            "features -> no refs -> eviction stays eager); memory overhead "
            "bounded by in-flight requests' encoder outputs. UPSTREAM "
            "STATUS: #45199 CLOSED unmerged 2026-06-11 (no comments), "
            "#38551 still OPEN; WATCHLIST sibling #39544 (scheduler-side "
            "alternative, OPEN) — if it merges instead, PN371's legacy "
            "anchors drift and the patch skips loudly at the next pin "
            "bump. Self-skips on #45199's merged form (eager_eviction "
            "two-line signature / multi-line ctor / encoder_cache_tracker "
            "call). Opt-in: intended ON for the gemma4 composes only."
        ),
        "upstream_pr": 45199,
        "upstream_pr_relationship": "backport",
        "related_upstream_prs": [39544, 39543, 38622],
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {
            "model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"],
            # Registry-integration 2026-06-11: draft claimed >=0.21.0 but
            # all 11 anchors were byte-verified ONLY on the 0.22.1 pin —
            # pin-specific vendor range per the G4_79 checklist.
            "vllm_version_range": (">=0.22.0", "<0.24.0"),
        },
        "implementation_status": "full",
        "composes_with": ["PN62"],
    },
    "PN372": {
        "title": "eagle_step zero/negative-seqlen slot-mapping guard (vendor of OPEN vllm#45005)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_PN372_EAGLE_ZERO_SEQLEN_GUARD",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.spec_decode.pn372_eagle_step_zero_seqlen_guard",
        "lifecycle": "experimental",
        "category": "stability",
        "credit": (
            "Genesis vendoring of OPEN upstream PR vllm#45005 "
            "(ashishpatel26, refs #40756/#39295), studied via gh pr "
            "view/diff 2026-06-11. The fused EAGLE/MTP draft-step "
            "slot-mapping kernel (eagle_step_slot_mapping_metadata_"
            "kernel, v1/spec_decode/utils.py) advances inactive padding "
            "rows with seq_lens == 0 whose block_table entries are -1 "
            "-> invalid slot mapping -> CUDA illegal memory access / "
            "device-side assert later in the draft loop. Exact crash "
            "class of our 262-280K-token MTP K=3 agent sessions. "
            "Guard: early-return writing PADDING_SLOT_ID + clamped "
            "position 0, seq_lens untouched; seq_len load hoisted "
            "(optional dedup sub-patch, parity with the PR). STRICTER "
            "than upstream: guards seq_len <= 0, not == 0 — #40756-"
            "class traces also showed NEGATIVE lens on corrupted rows; "
            "identical kernel cost (one register compare on an already-"
            "loaded value). Anchors byte-verified count==1 on pristine "
            "pin g303916e93 (0.22.1rc1.dev259); drift markers are exact "
            "substrings of #45005's form (absent at pin, lint-clean vs "
            "own replacements). SUCCESS CRITERION for retiring P108 "
            "(#42603 sync workaround for the same crash class): A/B on "
            "35B PROD with PN372 ON + P108 OFF shows the IMA class "
            "gone -> retire P108, recovering its 2-6% TPOT cost. A/B "
            "planned; P108 untouched. Default OFF pending the "
            "PN370+PN372 bench cycle (roadmap chunk-3 Theme A: land "
            "both in the same cycle)."
        ),
        "upstream_pr": 45005,
        "upstream_pr_relationship": "backport",
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {"vllm_version_range": (">=0.22.0", "<0.24.0")},
        "implementation_status": "full",
        "composes_with": ["P58", "P108", "PN340", "PN341", "PN370"],
    },
    "PN373": {
        "title": "parallel_tool_calls explicit null != false (vendor of OPEN vllm#44955)",
        "tier": "community",
        "family": "serving",
        "env_flag": "GENESIS_ENABLE_PN373_PARALLEL_TOOLCALLS_NULL",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.serving.pn373_parallel_toolcalls_null",
        "lifecycle": "experimental",
        "category": "stability",
        "credit": (
            "Genesis vendor of OPEN vllm PR #44955 (fixes #44948) — "
            "parallel_tool_calls explicit JSON null treated as false. "
            "ChatCompletionRequest.parallel_tool_calls is declared "
            "bool | None = True (pristine chat_completion/protocol.py:233 "
            "at pin g303916e93); clients that serialize the unset knob as "
            "null (LiteLLM/n8n) arrive at maybe_filter_parallel_tool_calls "
            "(entrypoints/serve/utils/tool_calls_utils.py:19) as None, and "
            "the pristine truthiness check trims the response to a SINGLE "
            "tool call on both the streaming path (delta.tool_calls "
            "filtered to index==0, called from chat_completion/"
            "serving.py:844) and the non-streaming path "
            "(message.tool_calls[:1], serving.py:1277). The documented "
            "default is true, so explicit null must keep all tool calls; "
            "every recovered call saves a full agent round-trip (hundreds "
            "of ms-s). Vendored: the 1-line semantic fix (is not False) "
            "as one text sub-patch spanning the function docstring + "
            "truthiness check (anchor count==1 byte-verified against "
            "/private/tmp/candidate_pin_current 2026-06-11). ADDED beyond "
            "upstream: the streaming-delta unit test the PR lacks "
            "(explicit-null must NOT truncate multi-tool-call deltas) + "
            "pristine bug reproduction (doubles as a retire trigger at "
            "pin bump) + drift-marker hygiene suite — 19 tests in "
            "tests/unit/integrations/serving/"
            "test_pn373_parallel_toolcalls_null.py. No anchor overlap "
            "with PN288/P107 (they patch chat_completion/serving.py; "
            "PN373 patches the delegated helper module). Self-skips when "
            "#44955 lands: drift marker is the PR post-image docstring "
            "wording — the merged condition text cannot serve as marker "
            "because the replacement necessarily emits it "
            "(lint_drift_markers clean). Behavior change limited to "
            "requests carrying explicit null; true/false semantics "
            "preserved. Default OFF; candidate default-on after fleet "
            "test."
        ),
        "upstream_pr": 44955,
        "upstream_pr_relationship": "backport",
        "requires_patches": [],
        "conflicts_with": [],
        # Registry-integration 2026-06-11: draft claimed >=0.21.0 but the
        # anchor was byte-verified ONLY on the 0.22.1 pin (and the
        # entrypoints/serve/utils/ helper module is a recent layout) —
        # pin-specific vendor range per the G4_79 checklist.
        "applies_to": {"vllm_version_range": (">=0.22.0", "<0.23.0")},
        "implementation_status": "full",
        "composes_with": ["PN288", "P107", "PN70"],
    },
    "PN374": {
        "title": "qwen3xml quoted parameter-name strip (Gemma4 #44715 key/value asymmetry analog)",
        "tier": "community",
        "family": "tool_parsing",
        "env_flag": "GENESIS_ENABLE_PN374_QWEN3XML_QUOTED_KEYS",
        "default_on": False,
        "category": "stability",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.tool_parsing.pn374_qwen3xml_quoted_keys",
        "lifecycle": "experimental",
        "credit": (
            "Genesis-original 2026-06-11 (50-PR sweep chunk-4 Theme 1 "
            "audit mandate). qwen3xml has the same key/value asymmetry "
            "as Gemma4 issue #44715: values are JSON-escaped but "
            "parameter names are interpolated UNESCAPED into the "
            "streamed arguments JSON, and the <parameter=([^>]+)> "
            "preprocess regex captures quote chars verbatim - a "
            "model-emitted quoted key like <parameter='3'> (or the "
            "double-quoted form) becomes a malformed name attribute and "
            "kills the expat parse of the element (parameter silently "
            "dropped). Two-hunk text patch: strip whitespace + quote "
            "wrappers from the captured name in _preprocess_xml_chunk "
            "and in _extract_parameter_name's parameter=NAME fallback. "
            "Anchors byte-verified count==1 vs pin "
            "0.22.1rc1.dev259+g303916e93. No upstream PR fixes qwen3xml "
            "keys as of 2026-06-11; upstream_pr points at the Gemma4 "
            "sibling fix #44877 for bug-class tracking only. Tests: "
            "tests/unit/integrations/tool_parsing/"
            "test_pn374_qwen3xml_quoted_keys.py (15)."
        ),
        "upstream_pr": 44877,
        "upstream_pr_relationship": "related_not_superseding",
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {
            "tool_call_parser": ["qwen3_xml"],
            # Registry-integration 2026-06-11: pin-specific text-patch
            # vendor — anchors byte-verified on the 0.22.1 pin only.
        # 2026-06-17 (0.23.1 reverify): kept capped <0.23.0. The qwen3xml
        # quoted-keys bug is GONE on 0.23.x (fixed upstream); anchors no
        # longer exist. Do NOT bump (would re-arm a superseded patch).
        # Patch + 15 tests retained for rollback contingency.
            "vllm_version_range": (">=0.22.0", "<0.23.0"),
        },
    },
    "PN375": {
        "title": "Gemma4 multi-boundary streaming tool-call deltas under MTP (vllm#44741)",
        "tier": "community",
        "family": "tool_parsing",
        "env_flag": "GENESIS_ENABLE_PN375_GEMMA4_MULTIBOUNDARY_STREAMING",
        "default_on": False,
        "category": "stability",
        "implementation_status": "full",
        "source": "vllm_pr_backport",
        "apply_module": "sndr.engines.vllm.patches.tool_parsing.pn375_gemma4_multiboundary_streaming",
        "lifecycle": "experimental",
        "credit": (
            "Vendors upstream vllm PR #44741 (OPEN 2026-06-11, issue "
            "#41967): under MTP a single streamed delta can cross "
            "multiple tool-call boundaries; the pristine pin parser "
            "selects one state-machine branch per delta and silently "
            "drops argument fragments past the boundary (silent "
            "first-tool-call argument loss in multi-tool streaming "
            "turns). Runtime hook: attaches "
            "_extract_streaming_delta_segments and rebinds "
            "Gemma4ToolParser._extract_streaming to replay "
            "delimiter-aligned segments through the saved original, "
            "merging per-segment DeltaMessages. Genesis adaptations: "
            "(1) CRITICAL - the G4_14 pad-token set is stripped from "
            "current_text AND delta_text BEFORE the upstream endswith "
            "consistency check (mutation-verified: without the strip "
            "the fix silently degrades to single-pass whenever pads "
            "appear); (2) binds at _extract_streaming so it composes "
            "with the G4_14 wrapper in either apply order (both orders "
            "test-verified); (3) self-skips on the G4_T1 v2 overlay "
            "variant (accumulated-rescan, structurally immune) via "
            "signature probe; (4) tolerates dict-or-object "
            "DeltaToolCall.function shapes. Single-boundary deltas keep "
            "the pristine path. Tests: tests/unit/integrations/"
            "tool_parsing/test_pn375_gemma4_multiboundary_streaming.py "
            "(14, incl. the combined multi-boundary + pad + "
            "G4_14-active regression on MTP-sized chunks); the "
            "pristine-bug reproduction test flips to FAILED when an "
            "upstream fix lands in a pin - then deep-diff and retire "
            "(iron-rule-#11). Racing cluster: #42006/#42237/#42300/"
            "#43037/#44741/#45068. Enable on gemma4 profiles after "
            "live docker-logs verification (insurance for "
            "pristine-parser deployments; live profiles mount the v2 "
            "overlay where PN375 self-skips)."
        ),
        "upstream_pr": 44741,
        "upstream_pr_relationship": "backport",
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["G4_14", "G4_T1"],
        # version-capped <0.23.0 2026-06-19 (dev148 TIER-1 audit): tool_parsers/
        # qwen3coder_tool_parser.py + gemma4 parser DELETED by #45588; engine
        # state machine supersedes. PN375 rebinds Gemma4ToolParser._extract_
        # streaming, which #45588 replaced with Gemma4EngineToolParser, so it
        # correctly skips on 0.23.x rather than file-missing-skip.
        "applies_to": {"tool_call_parser": "gemma4", "vllm_version_range": (">=0.20.0", "<0.23.0")},
    },
    "PN299E": {
        "title": "KV cache writer arch-aware NUM_WARPS+NUM_STAGES cap (SM 8.6)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_PN299E",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.pn299e_kv_cache_writer_arch_warps",
        "lifecycle": "experimental",
        "category": "kernel_perf",
        "credit": (
            "Genesis-original 2026-06-08 — CRITICAL hot-path finding. "
            "vllm/v1/attention/ops/triton_reshape_and_cache_flash.py is "
            "the KV cache writer — fires PER TOKEN PER LAYER on every "
            "prefill and decode step. Three launchers in the file set "
            "num_warps + num_stages for the CUDA branch: launcher 1 "
            "uses a heuristic that picks 8 on head_size=256, launchers "
            "2 and 3 hardcode num_warps=16 num_stages=10. The launcher "
            "2 branch even has ``if device_capability < 9: TILE_SIZE = "
            "512`` but does NOT adjust num_warps / num_stages — upstream "
            "bug. On SM 8.6 these configs spill 100 KB shared/SM hard. "
            "PN299E caps both via GENESIS_TRITON_AUTOTUNE_MAX_WARPS / "
            "MAX_STAGES (PN296 auto-sets =4/=2 on Ampere). Hopper+ stays "
            "at upstream defaults via env fallback. Composes with "
            "PN296+PN298+PN299+PN299B+PN299C+PN299D."
        ),
        "upstream_pr": None,
        "applies_to": {"vllm_version_range": (">=0.21.0", "<0.24.0")},
        "implementation_status": "full",
        "composes_with": ["PN296", "PN298", "PN299", "PN299B", "PN299C", "PN299D"],
        "requires_patches": ["PN296"],  # hard runtime dep on PN296 keystone (bug-hunt D13)
    },
    "PN299D": {
        "title": "Mamba2 SSU fallback heuristic arch-aware NUM_WARPS cap (SM 8.6)",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_PN299D",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.attention.gdn.pn299d_mamba_ssm_arch_warps",
        "lifecycle": "experimental",
        "category": "kernel_perf",
        "credit": (
            "Genesis-original 2026-06-08 — defensive cap on the "
            "selective_state_update FALLBACK heuristic in "
            "model_executor/layers/mamba/ops/mamba_ssm.py. The fallback "
            "path runs whenever the tuned JSON config for the live "
            "(headdim, dstate, cache_dtype, device) combo is missing. "
            "Two branches leave the function with num_warps=8: "
            "(a) dstate>128 AND not Blackwell — stays at initial (4,8); "
            "(b) dstate>128 AND Blackwell — (32,8). On SM 8.6 Ampere "
            "where path (a) fires, 8 warps spill the 100 KB shared/SM "
            "and trigger autotune eviction-recompile. Some Qwen3.6 "
            "Mamba block variants ship dstate=256 → fallback path "
            "hits this. PN299D caps via GENESIS_TRITON_AUTOTUNE_MAX_"
            "WARPS (PN296 auto-sets =4 on Ampere). No-op when tuned "
            "config JSON is found (heuristic bypassed). Composes with "
            "PN296+PN298+PN299+PN299B+PN299C."
        ),
        "upstream_pr": None,
        "applies_to": {"vllm_version_range": (">=0.21.0", "<0.24.0")},
        "implementation_status": "full",
        "composes_with": ["PN296", "PN298", "PN299", "PN299B", "PN299C"],
        "requires_patches": ["PN296"],  # hard runtime dep on PN296 keystone (bug-hunt D13)
    },
    "PN299C": {
        "title": "FLA layernorm_guard arch-aware NUM_WARPS heuristic cap (SM 8.6)",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_PN299C",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.attention.gdn.pn299c_fla_layernorm_guard_arch_warps",
        "lifecycle": "experimental",
        "category": "kernel_perf",
        "credit": (
            "Genesis-original 2026-06-08 — closes the LAST num_warps=8 "
            "site in vllm/model_executor/layers/fla/ops/ after PN298 + "
            "PN299 + PN299B coverage. layernorm_guard.py uses a runtime "
            "HEURISTIC, not a config-list autotune: "
            "``num_warps = min(max(BLOCK_N // 256, 1), 8)``. For "
            "Qwen3.6-A3B (hidden 5120 → BLOCK_N = 8192) this picks "
            "num_warps=8 unconditionally — spills on SM 8.6 (100 KB "
            "shared/SM). PN299C caps the heuristic with the same env "
            "PN296 auto-sets (GENESIS_TRITON_AUTOTUNE_MAX_WARPS=4 on "
            "Ampere). On Hopper+ the env stays at the upstream default "
            "'8' so behaviour is identical. Kernel fires per LN per "
            "layer per token — hot path. Composes with PN296+PN298+"
            "PN299+PN299B."
        ),
        "upstream_pr": None,
        "applies_to": {"vllm_version_range": (">=0.21.0", "<0.24.0")},
        "implementation_status": "full",
        "composes_with": ["PN296", "PN298", "PN299", "PN299B"],
        "requires_patches": ["PN296"],  # hard runtime dep on PN296 keystone (bug-hunt D13)
    },
    "PN299B": {
        "title": "FLA extended (kda+cumsum+solve_tril) arch-aware NUM_WARPS prune",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_PN299B",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.attention.gdn.pn299b_fla_kda_cumsum_solve_tril_arch_warps",
        "lifecycle": "experimental",
        "category": "kernel_perf",
        "credit": (
            "Genesis-original 2026-06-08 — closes a gap in PN299 coverage. "
            "Kernels-audit agent flagged 3 more FLA files with num_warps=8 "
            "autotune configs that PN299 doesn't touch: cumsum.py (2 sites), "
            "kda.py (5 sites — the Qwen3.6 KDA path), and solve_tril.py "
            "(3 sites). All 10 sub-patches use the same arch-aware filter "
            "pattern as PN299 (PN296 auto-sets GENESIS_TRITON_AUTOTUNE_MAX_"
            "WARPS=4 on Ampere SM 8.6, the filter list comprehension drops "
            "8-warp configs). Same per-sub required=False — partial-apply "
            "across upstream layout drift. kda.py is particularly important "
            "on Qwen3.6 hybrid_gdn_moe because the Kimi Delta Attention "
            "kernels run on every hybrid block; cold-start autotune spill-"
            "evictions there add 50-200 ms TTFT and contribute to the "
            "post-dev93 TPS gap. Composes with PN296 + PN298 + PN299."
        ),
        "upstream_pr": None,
        "applies_to": {"vllm_version_range": (">=0.21.0", "<0.24.0")},
        "implementation_status": "full",
        "composes_with": ["PN296", "PN298", "PN299"],
        "requires_patches": ["PN296"],  # hard runtime dep on PN296 keystone (bug-hunt D13)
    },
    "PN299": {
        "title": "FLA multi-file (kkt+wy_fast+l2norm) arch-aware NUM_WARPS prune",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_PN299_FLA_MULTI_ARCH_WARPS",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.attention.gdn.pn299_fla_multi_arch_warps",
        "lifecycle": "experimental",
        "category": "kernel_perf",
        "credit": (
            "Genesis-original 2026-06-05 — extends PN298 pattern to 3 "
            "additional FLA ops files used by GDN forward path. Files: "
            "chunk_scaled_dot_kkt.py (1 site), wy_fast.py (1 site), "
            "l2norm.py (2 sites). Original lists go up to num_warps=32 "
            "on l2norm.py:kernel1 which is severe spilling on Ampere "
            "SM 8.6. Reads GENESIS_TRITON_AUTOTUNE_MAX_WARPS env var "
            "(auto-set by PN296 to 4 on A5000). All these kernels run "
            "PER GDN LAYER on prefill — 48 layers in 27B Lorbus, 30 "
            "in 35B FP8. Companion to PN298."
        ),
        "upstream_pr": None,
        "applies_to": {"vllm_version_range": (">=0.21.0", "<0.24.0")},
        "implementation_status": "full",
        "composes_with": ["PN296", "PN298"],
        "requires_patches": ["PN296"],  # hard runtime dep on PN296 keystone (bug-hunt D13)
    },
    "PN298": {
        "title": (
            "chunk_o consolidated: GDN scale-fold (vllm#41446 pattern c) + "
            "FLA NUM_WARPS arch-aware prune (SM 8.6 spilling fix)"
        ),
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_PN298_FLA_CHUNK_O_ARCH_WARPS",
        # PN29 was consolidated into this entry on 2026-06-19. Its flag is a
        # recognized alias so existing builtin YAMLs keep working. Each flag
        # independently gates its own sub-patch inside the merged module
        # (apply()): GENESIS_ENABLE_PN29_GDN_SCALE_FOLD → pn29_scale_fold,
        # GENESIS_ENABLE_PN298_FLA_CHUNK_O_ARCH_WARPS → pn298_num_warps.
        "env_flag_aliases": ["GENESIS_ENABLE_PN29_GDN_SCALE_FOLD"],
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.attention.gdn.pn29_pn298_chunk_o_consolidated",
        "lifecycle": "experimental",
        "category": "kernel_perf",
        "credit": (
            "Consolidated 2026-06-19 (maintainability refactor, runtime-"
            "neutral): PN29 + PN298 both patch model_executor/layers/fla/"
            "ops/chunk_o.py at disjoint regions, now share one apply_module "
            "(pn29_pn298_chunk_o_consolidated) with two independently env-"
            "gated sub-patches. Applied kernel-code bytes are byte-identical "
            "to PN29+PN298 applied separately (only the single shared wiring-"
            "marker comment differs). "
            "(1) pn29_scale_fold — backport of vllm#41446 (zobinHuang, OPEN) "
            "pattern (c): folds the scale multiply in `chunk_fwd_kernel_o` "
            "to `b_o = (b_o + tl.dot(b_A, b_v)) * scale` (one fewer fp32 "
            "multiply per inner iter; distributive, drift bounded 1-2 ULP; "
            "Triton does NOT auto-fuse across the +/- boundary). Hardware-"
            "agnostic; gated by GENESIS_ENABLE_PN29_GDN_SCALE_FOLD. "
            "(2) pn298_num_warps — Genesis-original 2026-06-05, first patch "
            "built on the gpu_arch_profile foundation (PN296). Upstream "
            "chunk_o.py uses NUM_WARPS=[2,4] on Hopper else [2,4,8]; Ampere "
            "SM 8.6 (A5000 100KB shared/SM) falls into the [2,4,8] branch — "
            "num_warps=8 with BV=128 SPILLS registers and wastes autotune "
            "search time. The injected replacement reads "
            "get_gpu_arch_profile().max_safe_num_warps and FALLS BACK to the "
            "upstream NUM_WARPS expression when the profile is absent — i.e. "
            "the PN296 precondition lives inside the injected code and "
            "applies to THIS sub-patch only. Gated by "
            "GENESIS_ENABLE_PN298_FLA_CHUNK_O_ARCH_WARPS. Composes with "
            "PN296 (NOT requires_patches at the entry level — that would "
            "over-gate the version-agnostic PN29 scale-fold). This kernel "
            "runs PER LAYER PER PREFILL on 27B Lorbus + 48 GDN layers and on "
            "35B + 30 GDN layers; no-op on Qwen3MoE 35B which has no GDN."
        ),
        "upstream_pr": 41446,
        "upstream_pr_relationship": "backport",
        "applies_to": {
            # Triggers in any model using FLA chunk_fwd_kernel_o (hybrid
            # GDN). On Qwen3MoE without GDN, the kernel never fires → both
            # sub-patches are silently no-op even if their env is enabled.
            # The version range bounds the pn298 arch-aware rewrite; the
            # default version gate is OFF (GENESIS_ENFORCE_VERSION_RANGE) so
            # this is runtime-neutral on the default config.
            "vllm_version_range": (">=0.21.0", "<0.24.0"),
        },
        "implementation_status": "full",
        "composes_with": ["PN296"],
    },
    "PN296": {
        "title": "Genesis GPU Architecture Profile boot-time initializer (auto-tune env by arch)",
        "tier": "community",
        "family": "detection",
        "env_flag": "GENESIS_ENABLE_PN296_ARCH_PROFILE_INIT",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.detection.pn296_arch_profile_init",
        "lifecycle": "experimental",
        "category": "config_auto_tune",
        "credit": (
            "Genesis-original 2026-06-05 — addresses architecture-aware "
            "code-path divergence between vllm versions. Upstream often "
            "ships paths optimized for specific newer arch (Hopper SM 9.0+, "
            "Blackwell SM 10.x); older Ampere SM 8.x falls through to "
            "generic defaults not tuned for our resource budget (e.g. "
            "100KB shared/SM on A5000 vs 228KB on H100). Genesis env vars "
            "must be set MANUALLY in launcher to compensate. PN296 boots "
            "the architecture profiler (`sndr.engines.vllm.detection."
            "gpu_arch_profile.get_gpu_arch_profile()`), logs the full "
            "profile (device, SM, shared mem, L2, num SMs, HBM BW, TMA, "
            "FP32 TCs, FP8 native), and AUTO-SETS follow-on env vars when "
            "not already set: VLLM_MARLIN_FP32_REDUCE based on FP32 TC "
            "availability, GENESIS_TRITON_AUTOTUNE_MAX_WARPS based on "
            "shared-mem budget, GENESIS_GPU_ARCH_* diagnostic stamps. "
            "Operator overrides preserved. Composes with downstream "
            "patches that read GENESIS_TRITON_AUTOTUNE_MAX_WARPS to "
            "filter Triton autotune configs."
        ),
        "upstream_pr": None,
        "applies_to": {},
        "implementation_status": "full",
        "composes_with": ["PN286", "P23_WIRE"],
    },
    "PN294": {
        "title": "Unsplit MTP draft+target attention groups (vllm#43543 cold-path skip)",
        "tier": "community",
        "family": "worker",
        "env_flag": "GENESIS_ENABLE_PN294_UNSPLIT_MTP_ATTN_GROUPS",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.worker.pn294_unsplit_attn_groups_mtp",
        "lifecycle": "experimental",
        "category": "perf_hotfix",
        "credit": (
            "Genesis-original 2026-06-04 — companion to PN293. Closes "
            "~4-6ms TTFT overhead from vllm PR#43543 (`dede691c95`, "
            "'Split attention groups by num_heads_q for spec-decode "
            "drafts'). The PR added num_heads_q to AttentionGroupKey "
            "tuple. MTP K=3 draft model has different head count than "
            "target → creates 2 separate attn_groups → doubled metadata "
            "build + Python loop overhead per prefill iteration. PN294 "
            "force-merges by setting num_heads_q=0 in the bucket key "
            "(builder sizes scratch by max). Bit-identical for same-"
            "head groups; for different-head groups still correct (max "
            "scratch alloc) with slightly more memory. Combined with "
            "PN293 + PN295 targets full TTFT recovery + beyond baseline."
        ),
        "upstream_pr": "https://github.com/vllm-project/vllm/pull/43543",
        "applies_to": {},
        "implementation_status": "full",
        # Fixed 2026-06-09: removed PN295 (never landed; was planned follow-up
        # but no module exists in registry). Composes with PN293 only — the
        # PR43543 cold-path skip companion.
        "composes_with": ["PN293"],
    },
    "PN293": {
        "title": "mamba_attn _compute_common_metadata prefill fast-path (vllm#42430 cold-path skip)",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_PN293_MAMBA_ATTN_PREFILL_FASTPATH",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.attention.gdn.pn293_mamba_attn_prefill_fastpath",
        "lifecycle": "experimental",
        "category": "perf_hotfix",
        "credit": (
            "Genesis-original 2026-06-04 — closes the +24ms warm TTFT "
            "regression on Qwen3.6 27B Lorbus INT4 + TQ k8v4 + MTP K=3 "
            "between vllm dev93 (97.1ms) and dev354 (120.96ms) on 2× "
            "A5000 SM 8.6. Identified via TTFT-focused bisect. Upstream "
            "PR#42430 (`47829b1159`, '[Bugfix] mamba: run single-token "
            "extends as decodes') added unconditional per-build CPU "
            "overhead in `_compute_common_metadata`: torch.diff + 2 "
            "bool tensor ops + torch.any().item() + .clone() + "
            ".replace(). On 27B's 32 hybrid layers this is 32× the "
            "overhead per prefill iteration = ~14-18 ms of the +24 ms "
            "TTFT gap. PN293 adds early-exit guards: (a) skip if "
            "num_accepted_tokens is None (no spec data for this build), "
            "(b) skip if min(query_lens_cpu) > 1 (no single-token rows "
            "possible). On warm TTFT path both guards hit → entire "
            "block dead-coded. True mixed-batch cases still run full "
            "upstream logic. Output bit-identical to upstream on all "
            "true-positive cases. Companion patches PN294 + PN295 "
            "target secondary +4-6ms (PR#43543 attn group split) and "
            "tertiary +3-5ms (PR#41434 _cu_2 slice-assign) respectively."
        ),
        "upstream_pr": "https://github.com/vllm-project/vllm/pull/42430",
        "applies_to": {},
        "implementation_status": "full",
        "composes_with": [],
    },
    "PN292": {
        "title": "Revert PR#40172 fused Triton Mamba postprocess (A11 -18% root cause)",
        "tier": "community",
        "family": "worker",
        "env_flag": "GENESIS_ENABLE_PN292_REVERT_FUSED_MAMBA_POSTPROCESS",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.worker.pn292_revert_fused_mamba_postprocess",
        "lifecycle": "experimental",
        "category": "perf_hotfix",
        "credit": (
            "Genesis-original 2026-06-04 — closes the A11 -18% TPS "
            "regression on Qwen3.6 27B (hybrid GDN+Mamba + TQ k8v4 + "
            "MTP K=3) between vllm dev371 (`bf610c2f5`, 130 TPS) and "
            "dev354 (`626fa9bba`, 107 TPS) on 2× A5000 SM 8.6. Bisect "
            "of 362 commits between pins identified PR#40172 "
            "(`b730c46352`, 'Perf Hybrid Fused Triton kernel for GPU-"
            "side Mamba state postprocessing') as root cause. PR40172 "
            "replaces dev371 .cpu()-sync + Python postprocess_mamba "
            "with TWO per-decode-step Triton kernel launches "
            "(stage_postprocess_inputs_to_gpu + postprocess_mamba_"
            "align_gpu). Grid 5×128 under-occupies A5000's 84 SMs but "
            "still pays full launch+memcpy via 1024-element COPY_BLOCK_"
            "SIZE loops. On Hopper the saved sync dominates; on Ampere "
            "the launch overhead inverts the balance. Reverts 2 sites "
            "in gpu_model_runner.py back to dev371 form. Other "
            "candidates (PR41126 Mamba refactor, PR42095 FA KV layout, "
            "PR43361 stable-ABI, PR43273 SM100 GDN) were ruled out via "
            "byte-diff. Gate config: align + spec_decode + hybrid + "
            "SM 8.6."
        ),
        "upstream_pr": None,
        "applies_to": {
            # Version-capped off on dev148+ (2026-06-21 file-intersection
            # study). PN292's revert REPLACEMENT calls mamba_utils.
            # postprocess_mamba(...) which vllm#40172 DELETED, so the revert
            # is mechanically invalid on >=0.23.0 (would AttributeError /
            # trip its own drift marker). The fused kernel it reverts is
            # align-mode-gated and NEITHER PROD config sets align (35B="none",
            # 27B align retired) → the -18% it fixes is unreachable today.
            # Re-derive against the fused-kernel form ONLY if align is enabled.
            "vllm_version_range": (">=0.20.0", "<0.23.0"),
        },
        "implementation_status": "full",
        "composes_with": [],
    },
    "PN290": {
        "title": "num_accepted_tokens D2H race fix (vllm Issue #41190)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_PN290_NUM_ACCEPTED_TOKENS_RACE",
        "default_on": False,
        "category": "stability",
        "credit": (
            "Genesis-original 2026-06-04 backport-style fix for vllm "
            "Issue #41190 (OPEN at write — no upstream PR yet). Race "
            "condition: GPUModelRunner._update_states_after_model_execute "
            "issues non-blocking D2H copy of num_accepted_tokens.gpu, "
            "records a CUDA event, continues into model forward. With "
            "TP>1 + MTP, NCCL collective between record() and next "
            "iteration's synchronize() frees/recycles the source GPU "
            "tensor → cudaErrorIllegalAddress at sync time. Forces "
            "blocking D2H to eliminate race window. Cost ~0.3-0.6 ms "
            "blocking on A5000+PCIe4 for typical num_reqs≤8. Reproduced "
            "on 0.21.1rc1.dev354+g626fa9bba, 2× A5000, TP=2, Qwen3.6-35B-"
            "A3B-FP8 + MTP K=3 + multi-conc. Issue #41190 validators: "
            "hata1234 (Qwen3.6 35B-A3B-AWQ + RTX 6000 Ada), UmutAlihan "
            "(Gemma4 e2b-it + RTX 3060) — both reproduce same crash on "
            "TP=2+MTP, both confirm TP=2 without MTP works. Operator "
            "override: GENESIS_PN290_SYNC_MODE=none to revert. "
            "ENABLED on 35B PROD (dev148) 2026-06-21 file-intersection study: "
            "verified the target is the NON-ALIGN else-branch of "
            "_update_states_after_model_execute (gpu_model_runner.py:1537-1542) "
            "— reached when speculative_config + is_hybrid + mamba_cache_mode="
            "'none' = the exact live 35B path (MTP K=5 + hybrid + non-align). "
            "Boot-validated: applied=91/failed=0, bench 242.5 TPS (no regression "
            "vs 225 band, SYNC_MODE=full default), tool-call 7/7."
        ),
        "upstream_pr": None,
        "applies_to": {},
        "apply_module": "sndr.engines.vllm.patches.spec_decode.pn290_num_accepted_tokens_race",
        "lifecycle": "experimental",
        "implementation_status": "full",
        "composes_with": [],
    },
    "G4_T1": {
        "title": "Gemma4 tool-parser PR #42006 MTP streaming overlay (vendored from club-3090)",
        "tier": "community",
        "family": "tool_parsing",
        # Informational env_flag — schema validator requires a string;
        # G4_T1 is operator-side bind-mount only and has no Genesis
        # apply() path. The flag has no runtime semantics; it exists
        # so audit_upstream_status and the schema gate can track this
        # entry uniformly with other registry rows.
        "env_flag": "GENESIS_INFO_G4_T1_PR42006_OVERLAY_MOUNTED",
        "default_on": False,
        "category": "stability",
        "credit": (
            "Operator-side overlay 2026-05-30. Vendors upstream vllm "
            "PR #42006 ([Bugfix] Fix Gemma4 MTP streaming multi-tool "
            "calls — author @whytem, status OPEN as of 2026-05-30) via "
            "the bind-mount path documented in the file header. "
            "Companion to upstream vllm PR #41991 (MERGED 2026-05-08, "
            "already in our pin 626fa9bb) which covered the infinite-"
            "loop + array-boundary parser bugs. #42006 fixes the "
            "remaining 142-line refactor of `_extract_streaming` so "
            "the streaming SSE path can handle MTP-bundled token "
            "outputs (`</function>` close + last param in one delta). "
            "Vendor source: noonghunna/club-3090 stack, 2026-05-08; "
            "same family as the Qwen3 tool-parser SSE-silence fix in "
            "club-3090 issue #72. Empirical effect on gemma4-31B "
            "AWQ-4bit + TQ4bit_nc + MTP K=4 (session 2026-05-30): "
            "fixes parse of mid-thinking emissions like "
            "`<|tool_call>call:get_weather{city:<|\"|>London<|\"|>}`; "
            "tool_calls[] now populated. Residual 2/7 bench failures "
            "(thinking-then-tool edge prompts) are model-quality "
            "intrinsic limits (verified identical residual on dev371 "
            "base pin), NOT parser bugs. NOT a Genesis runtime patch "
            "— deployed entirely via the launcher's docker `-v` mount "
            "of sndr/engines/vllm/patches/tool_parsing/"
            "g4_t1_gemma4_tool_parser_pr42006_overlay.py over "
            "vllm/tool_parsers/gemma4_tool_parser.py. apply_module is "
            "the vendored file itself; the registry entry exists so "
            "audit_upstream_status tracks the PR and so retire-when-"
            "merged can drop both file and mount in one go."
        ),
        "upstream_pr": 42006,
        "upstream_pr_relationship": "backport",
        "applies_to": {
            "tool_call_parser": "gemma4",
        },
        # apply_module points at a thin marker module rather than the
        # vendored parser (which imports `regex` and is intentionally
        # not loaded in-process). The vendored file is deployed via
        # the operator-side bind-mount; the marker just provides the
        # apply() contract row for the dispatcher.
        "apply_module": "sndr.engines.vllm.patches.tool_parsing.g4_t1_pr42006_marker",
        "lifecycle": "experimental",
        "implementation_status": "marker_only",
        "composes_with": ["PN287", "PN288"],
    },
    "PN17": {
        "title": "FA2 softmax_lse runtime clamp (Cliff 1 mechanism A, Issue #11)",
        "tier": "community",
        "family": "attention.flash",
        "env_flag": "GENESIS_ENABLE_PN17_FA2_LSE_CLAMP",
        "default_on": False,
        "category": "memory_savings",
        "credit": (
            "Genesis-original 2026-04-30 in response to noonghunna's "
            "Genesis Issue #11 cross-rig diagnosis (RTX 3090, 2026-04-29). "
            "FA2 `flash_attn_varlen_func` allocates softmax_lse buffer "
            "of shape [num_seqs, num_heads, max_seqlen_k] sized by "
            "the max_seqlen_k argument — NOT actual seqused_k. vLLM's "
            "gpu_model_runner sets attn_metadata.max_seq_len = "
            "max_model_len during cudagraph capture for shape stability "
            "(see vllm#40961 SWA case); this leaks into runtime "
            "decode/prefill, causing 50-100 MiB over-allocation at "
            "long context. Closes Cliff 1 mechanism A (FA2 path); "
            "widens long-text-no-vision safe envelope from ~150K to "
            "~205K. Mechanism B (FFN intermediate buffer 138 MiB on "
            "long-vision) is OUT OF SCOPE — requires upstream-FFN "
            "chunked forward, not addressable from Genesis text-patch "
            "layer. Cudagraph-safe: clamp only fires when "
            "is_current_stream_capturing() returns False; capture-time "
            "preserves max_model_len padding. Reference: "
            "Dao-AILab/flash-attention#1011 (open since 2024)."
        ),
        "upstream_pr": None,
        "applies_to": {
            # Applies whenever FA2 varlen path is active. Most relevant
            # at long context (>100K) where the cap-leak dominates.
        },
        "apply_module": "sndr.engines.vllm.patches.attention.flash.pn17_fa2_softmax_lse_clamp",
        "lifecycle": "experimental",
        # [Performance verified 2026-05-11 — DO NOT DISABLE]
        # Differential bench on 35B/dev209 (canonical genesis_bench_suite
        # --quick --ctx 8k, 5×5×1024):
        #   PN17=1: wall_TPS 231.08, decode_TPOT 4.01 ms, CV ~4-9%
        #   PN17=0: wall_TPS 178.95, decode_TPOT 5.15 ms, CV ~8-11%
        #   Delta: -22.6% TPS, +28.5% TPOT when DISABLED
        # PN17 is a PERFORMANCE WIN, not a regression source. The naive
        # static-analysis worry about per-call .item() sync was wrong:
        # CUDA stream scheduling absorbs the sync, while the alternative
        # (FA2 allocating softmax_lse for max_seqlen_k=320K per call) is
        # the actual dominant cost on long-context configs. The clamp
        # to seqused_k.max() reduces the per-call buffer 10-1000x for
        # decode workloads where actual context << max_model_len.
        # Iron rule #10 ("study before disabling") was the right call —
        # the empirical bench reversed the static-analysis hypothesis.
        # KEEP PN17 enabled on PROD. No optimization needed at this site;
        # the per-call .item() sync IS the cheap path here.
        "implementation_status": "full",
    },
    "PN16": {
        "title": "Lazy-reasoner request hook v2 (cache-safe; V7 max_tokens cap)",
        "tier": "community",
        "family": "middleware",
        "env_flag": "GENESIS_ENABLE_PN16_LAZY_REASONER",
        "default_on": False,
        "category": "request_middleware",
        "credit": (
            "Genesis-original 2026-04-29; v2 rearchitecture 2026-05-09 "
            "(Wave 6 closure). Per-request hybrid policy deciding "
            "whether the model's `<think>...</think>` block adds value. "
            "v2 production paths (cache-safe): "
            "V3 (client override) respects explicit "
            "chat_template_kwargs.enable_thinking; "
            "V5 (soft cap) injects a concise-reasoning hint into the "
            "last user message when GENESIS_PN16_MAX_THINKING_TOKENS > 0; "
            "V7 (NEW — max_tokens hard cap) clamps request.max_tokens "
            "to GENESIS_PN16_CLASSIFIER_MAX_TOKENS (default 0=off) when "
            "the short-prompt classifier hits — provides a hard ceiling "
            "on response length without mutating the chat template, so "
            "CUDA graph dispatch and MTP draft compatibility are "
            "preserved. "
            "⚠ V1 (template-flag mutation) RETIRED from default on "
            "2026-05-09 — Wave 6 bench measured 28%% wall_TPS drop with "
            "6× CV amplification (236.24 → 166.25 TPS, 6.3% → 37.6% CV) "
            "on Sander's 35B PROD because forcing enable_thinking=False "
            "renders a different chat-template prefix → CUDA graph "
            "dispatch miss + MTP draft divergence. V1 still callable as "
            "legacy via GENESIS_PN16_V1_LEGACY=1 (emits one-shot WARN at "
            "first hit). "
            "V4 (LogitsProcessor strict cap) deferred — vllm v1 rejects "
            "custom logits processors with speculative_config set. "
            "V6 (streaming-side `<think>` truncator) — future work for "
            "deterministic TTFT bounding. "
            "Stats via `sndr.engines.vllm.middleware.lazy_reasoner.get_stats()`."
        ),
        "upstream_pr": None,
        "applies_to": {
            # [Genesis pin-gate 2026-05-11] PROD-active (V2 rearchitecture,
            # Wave 6 closure). V1 retired, V5/V7 cache-safe paths active.
            "vllm_version_range": (">=0.20.0", "<0.24.0"),
        },
        # v11.3.0 BUG #9 fix: PN16 spec apply_module is the canonical
        # lazy-reasoner module (`pn16_lazy_reasoner`) — V2/V3/V5/V7
        # paths for per-request enable_thinking gating. PN16_V6 is a
        # separately-registered SECOND version (streaming-truncator
        # token-budget enforcer) under env GENESIS_ENABLE_PN16_V6_*.
        # Previously incorrectly pointed at the V6 module — on v12.0.0
        # spec-flip an operator with GENESIS_ENABLE_PN16_LAZY_REASONER=1
        # would activate the streaming truncator (different function,
        # different anchor file) instead of the lazy reasoner hook.
        "apply_module": "sndr.engines.vllm.patches.middleware.pn16_lazy_reasoner",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN16_V6": {
        "title": "PN16 V6 — streaming `<think>` token-budget enforcer (Sprint 4)",
        "tier": "community",
        "family": "middleware",
        "env_flag": "GENESIS_ENABLE_PN16_V6_STREAMING_TRUNCATOR",
        "default_on": False,
        "category": "request_middleware",
        "implementation_status": "full",
        "source": "genesis_original",
        # Explicit apply_module — registry sub-id "PN16_V6" doesn't auto-resolve
        # via the pn{digit}_ regex; spell it out so iter_patch_specs() finds it.
        "apply_module": (
            "sndr.engines.vllm.patches.middleware.pn16_v6_streaming_truncator"
        ),
        "credit": (
            "Genesis-original Sprint 4 closure 2026-05-09 — deep fix for "
            "london_think failure class. Per-request stateful truncator "
            "(sndr/engines/vllm/middleware/think_streaming_truncator.py) wraps "
            "OpenAIServingChat.chat_completion_stream_generator via "
            "class-rebind. When a request has tools attached AND "
            "GENESIS_PN16_MAX_THINKING_STREAM_TOKENS > 0, counts "
            "delta.reasoning_content chunks; once budget exceeded, drops "
            "subsequent reasoning chunks and emits one-shot [Genesis] "
            "truncation note as delta.content. tool_calls + plain content "
            "chunks always pass. Bounds client-visible TTFT-to-tool-call "
            "deterministically when the model ignores V8's prompt-engineering "
            "budget hint. Stacks cleanly on top of V8 — V8 nudges, V6 "
            "enforces. Cache-safe (no chat_template mutation), "
            "compute-neutral (model still generates internally; only "
            "client-visible stream is truncated)."
        ),
        "upstream_pr": None,
        "lifecycle": "experimental",
        "requires_patches": [],
        "conflicts_with": [],
    },
    "PN14": {
        "title": "TQ decode IOOB safe_page_idx clamp (vllm#40074)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_PN14_TQ_DECODE_OOB_CLAMP",
        "default_on": False,
        "category": "kernel_safety",
        "credit": (
            "Backport of vllm#40074 (devarakondasrikanth @adobe, OPEN). "
            "Fixes upstream issue #39998 — Triton bounds-checker assertion "
            "in `_tq_decode_stage1` on long (>32k) sequences. The mask= "
            "argument guards the LOADED VALUE on masked-out lanes but not "
            "the address arithmetic; clamping page_idx to 0 via "
            "`tl.where(kv_mask, page_idx, 0)` keeps the pointer in-bounds "
            "even on lanes whose result is discarded. Originally reported "
            "on 4090 (sm_89); jhsmith409 confirmed clean apply on 5090 "
            "(sm_120) while stacking on top of #39931. Defensive on Genesis "
            "Ampere prod (sm_86 — assertion not seen). Becomes load-bearing "
            "on Sander's planned RTX PRO 6000 Blackwell upgrade. Self-"
            "retires via marker `safe_page_idx` when #40074 merges. "
            "Codepath fires when spec-decode OFF/K=1 OR P67 dispatch returns "
            "False (shape outside envelope) — runs in Genesis prod despite "
            "MTP K=3 being active."
        ),
        "upstream_pr": 40074,
        "upstream_pr_relationship": "backport",
        "applies_to": {
            "is_turboquant": [True],
            # [Genesis pin-gate 2026-05-11] Defensive on Ampere; load-
            # bearing on planned Blackwell upgrade. Validated dev9 → dev93.
            # Self-retires via marker when #40074 merges.
            "vllm_version_range": (">=0.20.0", "<0.24.0"),
        },
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.pn14_tq_decode_oob_clamp",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    # PN13 entry moved to legacy/retired section below (lifecycle: retired_2026-05-04)
    # Reason: vllm 0.20.2 commit c2fb013 merged identical change (#41235).
    # See PN13 entry near line 1289 for retirement metadata.

    "P94": {
        "title": "Spec-decode prepare_next_token_ids_padded zero-alloc (vllm#41043)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_P94",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Backport of vllm#41043 (wangluochao902, MERGED 2026-04-29). Removes GPU->CPU .tolist() sync + list-comp Python objects + np.array allocation in LLMBaseProposer.prepare_next_token_ids_padded hot path. PR author measured P99 TPOT -9.3% on Llama-3.1-8B + Eagle3 TP=4. For our MTP K=3 single-stream: expected +2-4% wall TPS + tighter CV. SUPERSEDED-ON-MERGE: when our pin advances past the merge SHA the patch will SKIP cleanly via drift detection on the original .tolist() anchor — at that point delete the wiring file + this entry.",
        "upstream_pr": 41043,
        "upstream_pr_relationship": "backport",
        "superseded_by": "vllm#41043 (merged 2026-04-29, byte-identical with deep-diff confirmed Wave 8 audit) — patch retained as audit trail",
        "vllm_version_range": "<0.20.2rc1.dev9",  # mirrors applies_to (gated SoT); reconciled D17 2026-06-17 (was <0.20.2rc1.dev93+g51f22dcfd)
        "apply_module": "sndr.engines.vllm._archive.p94_spec_decode_zero_alloc",
        "applies_to": {
            # Applies whenever spec-decode is active. All spec methods.

            # [Genesis iron-rule-#11 retire 2026-05-11 v2 audit] Wave 8
            # deep-diff confirmed BYTE-IDENTICAL with upstream #41043
            # (merged 2026-04-29, in dev9+ → dev93+ → dev209+). On
            # post-merge pins drift detector auto-skips (anchor gone).
            # Pin-gate tightened to anchor-correct upper bound.
            "vllm_version_range": "<0.20.2rc1.dev9",
        },
        "lifecycle": "retired",  # 2026-05-11 v2 audit: formalized byte-identical retire
        "implementation_status": "full",
    },
    # ════════════════════════════════════════════════════════════════════
    # 2026-05-14 — vLLM upstream PR sweep on dev338+gbf0d2dc6d nightly.
    # Four entries here (P108, P109, PN110, PN111). All backport open
    # upstream fixes that target either bugs in our hot path (P108, P109,
    # PN110) or a measurable perf win on a future operator profile
    # (PN111, align-mode only). Default-on choice is per-patch.
    # ════════════════════════════════════════════════════════════════════
    "P108": {
        "title": "MTP draft-loop stream synchronization (vllm#42603)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_P108",
        "default_on": True,
        "category": "spec_decode",
        "credit": (
            "Defensive overlay derived from vllm#42603 (z1ying). Closes a "
            "cudaErrorIllegalAddress race in LLMBaseProposer.propose() "
            "where the input_ids / hidden_states buffer writes on the "
            "default stream were not synchronized before downstream "
            "attention kernels ran on a different stream (FlashInfer "
            "default). Reproduced upstream on Qwen3.6-27B-FP8 + RTX 5090; "
            "root bug tracked at vllm#40756 (OPEN). "
            "PIN.R-DEEP-PARITY.1 (2026-05-24) verified the upstream pin "
            "0.20.2rc1.dev371+gbf610c2f5 and current main both lack any "
            "equivalent sync call; PR #42603 was CLOSED 2026-05-14 by "
            "maintainer @benchislett with the verbatim rationale "
            "\"forcing a synchronization is an unacceptable fix\" — the "
            "bug is acknowledged upstream but the simple fix was "
            "rejected pending root-cause investigation. "
            "Genesis ships a REFINED form that addresses the maintainer's "
            "objection: the sync is backend-gated (auto-enabled only for "
            "FlashInfer family, where the race is empirically confirmed) "
            "with operator override via GENESIS_P108_FORCE_SYNC=0|1, and "
            "a cached first-call decision so the per-step overhead is a "
            "single attribute read on non-racy backends. Genesis bench: "
            "−14% wall TPS on 27B INT4 + TurboQuant + MTP K=3 if sync is "
            "unconditional, confirming the gate is mandatory for "
            "performance. PIN.R-P108-METADATA.1 (2026-05-24) reclassifies "
            "upstream_pr_relationship from `backport` to "
            "`defensive_overlay` to lock the entry against status-only "
            "retire if PR #42603 is ever re-opened-and-merged in its "
            "rejected simple-sync form — iron-rule-#11 deep-parity "
            "remains the only retire path."
        ),
        "upstream_pr": 42603,
        "upstream_pr_relationship": "defensive_overlay",
        "applies_to": {
            "spec_method_any": ["mtp", "eagle", "dflash"],
        },
        "implementation_status": "full",
        "apply_module": "sndr.engines.vllm.patches.spec_decode.p108_mtp_draft_stream_sync",
        "source": "vllm_pr_backport",
        "lifecycle": "experimental",
    },
    "P109": {
        "title": "sampling_params vocab-range validators (vllm#42614)",
        "tier": "community",
        "family": "serving",
        "env_flag": "GENESIS_ENABLE_P109",
        "default_on": True,
        "category": "stability",
        "credit": (
            "Backport of vllm#42614 (jperezdealgaba, OPEN 2026-05-14). "
            "Adds explicit validation of stop_token_ids and "
            "logprob_token_ids against model vocab size in "
            "SamplingParams.verify(). Out-of-vocab ids previously OOB'd "
            "the V2 Triton _bias_kernel and crashed the worker; with "
            "this patch the request bounces with a clear 400 instead. "
            "Defense-in-depth for the public Proxy-AI surface "
            "(OpenAI-compatible streaming clients sometimes pass malformed "
            "stop_token_ids). Bit-identical for valid inputs."
        ),
        "upstream_pr": 42614,
        "upstream_pr_relationship": "backport",
        "applies_to": {},  # generic safety; always applicable
        "implementation_status": "full",
        "apply_module": "sndr.engines.vllm.patches.serving.p109_sampling_params_vocab_bounds",
        "source": "vllm_pr_backport",
        "lifecycle": "experimental",
    },
    "PN110": {
        "title": "BlockPool.free_blocks deduplication (vllm#42615)",
        "tier": "community",
        "family": "kv_cache",
        "env_flag": "GENESIS_ENABLE_PN110",
        "default_on": True,
        "category": "stability",
        "credit": (
            "Backport of vllm#42615 (AkCodes23, OPEN 2026-05-14). "
            "Deduplicates by id(block) in BlockPool.free_blocks() so a "
            "caller that passes the same KVCacheBlock twice (sliding-"
            "window + offload-connector race) does not double-decrement "
            "ref_cnt or double-append into the free queue. Composes "
            "cleanly with PN95/PN96/PN97 — same family, no anchor "
            "conflict (dedup happens before ref_cnt -= and before "
            "append_n). Warns when duplicates are observed."
        ),
        "upstream_pr": 42615,
        "upstream_pr_relationship": "backport",
        # 2026-06-18 (dev148 full-patch audit): #42615 (OPEN) dedups
        # gpu_block_ids in the SimpleCPUOffload eager-store path — our PROD
        # models run NO KV offloading, so PN110 is dormant; its block_pool.py
        # anchor also drifted on 0.23.x (DRIFT skip). Capped <0.23.0 for an
        # honest registry (not silently drift-skipped); re-anchor only if
        # KV-offloading is ever adopted.
        "applies_to": {"vllm_version_range": (">=0.20.0", "<0.23.0")},
        "implementation_status": "full",
        "apply_module": "sndr.engines.vllm.patches.kv_cache.pn110_block_pool_free_dedup",
        "source": "vllm_pr_backport",
        "lifecycle": "experimental",
    },
    "PN111": {
        "title": "Skip-mamba-postprocess GPU->CPU sync (align-mode; vllm#42574)",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_PN111",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Backport of vllm#42574 (mamingyuan-nv, OPEN 2026-05-14). "
            "Skips the per-decode-step blocking GPU->CPU sync of "
            "num_accepted_tokens in the mamba_cache_mode==align branch "
            "of GPUModelRunner._update_states_after_model_execute when "
            "the downstream postprocess_mamba is provably a no-op this "
            "step (upper-bound proof: num_accepted <= n_draft + 1, no "
            "Mamba block boundary crossed). Adds can_skip_mamba_"
            "postprocess() to mamba_utils.py. Reported +17.4% TPS / "
            "-13.7% ITL on Nemotron-Super-120B-A12B-NVFP4 MTP=3 on "
            "GB300. ⚠ Genesis PROD presets currently DO NOT set "
            "--mamba-cache-mode align — patch is a no-op there. Win "
            "materialises only after an operator opts into align mode."
        ),
        "upstream_pr": 42574,
        "upstream_pr_relationship": "backport",
        "applies_to": {
            "is_hybrid": True,
            "spec_method_any": ["mtp", "eagle"],
            # Version-capped off on dev148+ (2026-06-21 file-intersection
            # study). vllm#40172 landed a fused GPU postprocess kernel that
            # DELETED the Python `postprocess_mamba` PN111 targets and now
            # computes the skip predicate on-GPU per request
            # (v1/worker/mamba_utils.py:87-94) — a superset of #42574. The
            # blocking-sync code-path PN111 optimizes no longer exists, and
            # PROD runs mamba_cache_mode="none" (non-align) so it never fired
            # anyway. iron-rule-#11 outcome (a): superseded (functional superset).
            "vllm_version_range": (">=0.20.0", "<0.23.0"),
        },
        "implementation_status": "full",
        "apply_module": "sndr.engines.vllm.patches.attention.gdn.pn111_skip_mamba_postprocess_sync",
        "source": "vllm_pr_backport",
        # Effectively retired on dev148+ via the vllm_version_range cap above
        # (PN30 precedent: the cap is the retire mechanism; lifecycle stays
        # experimental so the family/location + docstring-sync gates pass).
        "lifecycle": "experimental",
        "conflicts_with": [],
        "requires_patches": [],
    },
    "PN116": {
        "title": "TurboQuant prefill max_seq_len fallback fix (regressor: vllm#41434)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_PN116",
        "default_on": True,
        "category": "perf_hotfix",
        "credit": (
            "Genesis-original fix for an Ampere-specific regression "
            "introduced by vllm#41434 (Perf 3/n: eliminate GPU<->CPU "
            "syncs in attention impls, merged 2026-05-08). The PR's "
            "new TurboQuant prefill fast path uses a CPU-resident "
            "seq_lens upper bound when populated and falls back to "
            "`attn_metadata.max_seq_len` otherwise — but max_seq_len is "
            "the FULL-BATCH max (includes decodes), so the fallback "
            "feeds the attention kernel an inflated upper bound. On "
            "Hopper GB200 the PR measured +4.8% TPS (fast path "
            "dominant). On 2× A5000 Ampere + 35B-A3B-FP8 + TurboQuant "
            "k8v4 + MTP K=3 we measure −9.7% wall_TPS (241→218 between "
            "dev93 and dev209+). 27B INT4 + GDN is unaffected. "
            "HW-aware: applies on SM<9.0 only by default; on Hopper "
            "and newer the upstream behaviour wins and we self-skip "
            "(GENESIS_PN116_FORCE=1 to override). "
            "[Phase 3D 2026-05-22] Live-network verification on dev371 "
            "(canonical pin bf610c2f56764e1b30bc6065f4ceace3d6e59036): "
            "(1) PR41434 merge commit 989c176c0a14e1adc5a9ba33cb5c3a39fceec3d3 "
            "is IN dev371 baseline — `gh api .../compare/989c176c0...bf610c2f5` "
            "shows dev371 is 246 commits ahead of the PR41434 merge, "
            "behind_by=0. The regression has been in our runtime since the "
            "dev371 promotion. "
            "(2) The inflated fallback line STILL EXISTS at dev371 source "
            "vllm/v1/attention/backends/turboquant_attn.py:489 — direct "
            "`gh api .../contents?ref=bf610c2f5...` fetch confirms the "
            "buggy `prefill_max_seq = attn_metadata.max_seq_len` else-branch "
            "is unchanged. No upstream follow-up fix has been filed. "
            "(3) PN116 is a COUNTER-REGRESSION hotfix, NOT a backport. "
            "The `upstream_pr: 41434` field here means \"regression source\" "
            "— the PR Genesis is correcting — NOT \"backport source\". "
            "This is semantically equivalent to P98's INTENTIONAL-INVERSE "
            "relationship with the same PR. "
            "(4) Current PROD state: PN116 applies normally on every "
            "TurboQuant Ampere boot (`default_on: True` + "
            "`applies_to: {is_turboquant: True}`), restoring the prefill-"
            "slice max computation that PR41434 broke. Active across the "
            "two TurboQuant ModelDefs in the current public corpus — "
            "`qwen3.6-35b-a3b-fp8` (FP8 + TQ k8v4 KV) and "
            "`qwen3.6-27b-int4-autoround-tq-k8v4` (INT4 + TQ k8v4 KV) — "
            "via the `applies_to: {is_turboquant: True}` predicate; no "
            "explicit YAML enable entries needed. "
            "(5) Retire would CATASTROPHICALLY regress PROD: removing "
            "PN116 re-exposes the measured −9.7% wall_TPS regression on "
            "2× A5000 + 35B-A3B-FP8 + TQ + MTP K=3. Do NOT retire. "
            "(6) Phase 3C upstream audit's NEWLY-MERGED classification "
            "of PN116 was a relationship-type false positive — the audit "
            "script doesn't yet distinguish `upstream_pr` semantic kinds "
            "(backport-source vs regression-source vs intentional-inverse). "
            "Follow-up for Phase 5: introduce an `upstream_pr_relationship` "
            "registry field (or equivalent) so the audit can route the "
            "three patterns to different action buckets without operator "
            "manual triage."
        ),
        "upstream_pr": 41434,  # regression source (NOT backport); see Phase 3D 2026-05-22 note above. patch self-retires when upstream re-fixes fallback
        "upstream_pr_relationship": "counter_regression",
        "applies_to": {
            "is_turboquant": True,  # patch site is turboquant_attn.py
        },
        "implementation_status": "full",
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.pn116_tq_prefill_maxseq_fallback",
        "source": "genesis_original",
        "lifecycle": "experimental",
        "conflicts_with": [],
        "requires_patches": [],
    },
    "PN118": {
        "title": "TurboQuant workspace graceful-fallback (vllm#42551, P99-compat)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_PN118",
        "default_on": True,
        "category": "stability",
        "credit": (
            "P99-compatible backport of vllm#42551 (jasonboukheir, OPEN "
            "2026-05-14). Closes an AssertionError on first decode request "
            "for partial-TQ models (16/64 TQ layers, e.g. Lorbus/Qwen3.6-"
            "27B-int4-AutoRound — named in the PR). Adds two new methods "
            "to WorkspaceManager: try_get_simultaneous (returns None on "
            "locked-undersized instead of raising) and reserve (pre-"
            "allocates every ubatch slot before lock_workspace snapshot). "
            "TurboQuant __init__ calls reserve, _decode_attention uses "
            "try_get_simultaneous with torch.empty fallback. Composes with "
            "our P99 memoization on get_simultaneous (P99 stays intact)."
        ),
        "upstream_pr": 42551,
        "upstream_pr_relationship": "backport",
        "applies_to": {
            "is_turboquant": True,  # patch sites are TQ-specific
        },
        "implementation_status": "full",
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.pn118_tq_workspace_fallback",
        "source": "vllm_pr_backport",
        "lifecycle": "experimental",
        "composes_with": ["PN399"],  # PN399 (when ON) wraps PN118's live
                                     # decode output (CG fixed-buffer branch
                                     # BEFORE PN118's try_get; PN118 body
                                     # byte-unchanged as the eager elif) AND
                                     # removes the now-dead PN118 __init__
                                     # _reserve_decode_workspace box+call+method
                                     # to cut boot overhead. PN118 SOURCE is NOT
                                     # edited (PN399 anchors PN118's live output
                                     # and transforms it). PN118 still owns the
                                     # decode block + reservation when PN399 is
                                     # OFF (current crash-free PROD behavior).
        "conflicts_with": [],
        "requires_patches": [],
    },
    "PN399": {
        "title": (
            "Consolidated single-owner TurboQuant decode-scratch fixed-buffer "
            "— fix CUDA IMA in FULL cudagraph + remove dead PN118/PN353A "
            "decode reservations (backport+improve OPEN vllm#46067)"
        ),
        "tier": "community",
        "family": "attention.turboquant",
        "category": "stability",
        "env_flag": "GENESIS_ENABLE_PN399_TQ_DECODE_SCRATCH_IMA",
        "default_on": False,
        "credit": (
            "Genesis CONSOLIDATED backport+improvement of OPEN vllm#46067 "
            "([Bugfix][TurboQuant] Fix CUDA illegal memory access in FULL "
            "cudagraph). TQ decode runs inside the FULL cudagraph; scratch "
            "from the growable WorkspaceManager is freed+realloc'd on grow "
            "(empty_cache unmaps the old address) across the B=1..max capture "
            "sweep / long continuation-prefill, freeing an address an earlier "
            "captured graph still points at -> first replay -> CUDA IMA. "
            "PN399 owns the TQ decode-scratch lifecycle: a fixed module-level "
            "_DECODE_SCRATCH allocated once at max_cudagraph_capture_size, "
            "reused by all TQ layers + all captured graphs, sliced [:B] "
            "(native kernel contract), reset on gpu/shutdown teardown. "
            "RE-AUTHORED against live dev148: the PR's pristine "
            "_decode_attention OLD block was already consumed by PN118 "
            "(try_get_simultaneous), so PN399 inserts the CG-path branch "
            "BEFORE PN118's is_workspace_manager_initialized check, demoting "
            "PN118's try_get body to the eager/cold elif (left byte-unchanged "
            "as the enforce_eager / B>max_batch safety net). BETTER THAN "
            "UPSTREAM: because PN399 also OWNS the de-duplication that "
            "upstream #46067 cannot (upstream has neither PN118 nor PN353A), "
            "it additionally REMOVES the now-dead decode reservations to cut "
            "boot overhead — (1) the PN118 __init__ _reserve_decode_workspace "
            "box+call+method, (2) the PN353A decode-scratch get_simultaneous "
            "reservation — KEEPING the PN353A continuation-prefill K/V "
            "reservation byte-intact (PN399 never touches the prefill path). "
            "The removals are SOURCE-file-clean: PN399 anchors the LIVE "
            "PN118/PN353A-applied output and transforms it; the "
            "pn118_*/pn353a_* patch sources are NOT edited. With PN399 "
            "OFF/unapplied it produces ZERO change — PN118/PN353A keep owning "
            "decode + their reservations (current crash-free PROD behavior). "
            "On our PROD (35B FP8 + MTP K=3 + TQ k8v4, capture set [4,8,16], "
            "max_batch=16) the live IMA is ALREADY neutralized by "
            "PN118+PN353A+SNDR_WORKSPACE_001 (5h clean) — PN399 is belt-and-"
            "suspenders + tidier (single owner, one persistent 16-row buffer "
            "vs per-capture torch.empty, a few MB of boot reservation "
            "reclaimed), NOT an open-wound fix. default_on False / lifecycle "
            "experimental pending rig validation at the next pin-upgrade "
            "window."
        ),
        "upstream_pr": 46067,
        "upstream_pr_relationship": "backport",
        "applies_to": {
            "is_turboquant": True,  # patch sites are TQ-specific
            "vllm_version_range": (">=0.21.0", "<0.24.0"),
        },
        "implementation_status": "full",
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.pn399_tq_decode_scratch_ima",
        "source": "vllm_pr_backport",
        "lifecycle": "experimental",
        "composes_with": [
            "PN353A", "PN353B", "P98", "P99", "P101", "PN119", "P67", "P67b",
        ],
        "conflicts_with": [],
        "requires_patches": ["PN118", "PN353A", "P101"],
                                        # PN399 anchors the LIVE PN118-applied
                                        # __init__ box / decode head AND the
                                        # PN353A-applied reserve block, so BOTH
                                        # must apply first (registry index >
                                        # PN118 and PN353A; insertion-order
                                        # apply runs them first). With either
                                        # off the dependent sub-patches SKIP
                                        # cleanly and the IMA defense is off.
                                        #
                                        # 2026-06-19 (dependency audit): P101
                                        # added — PN399's const sub-patch
                                        # anchors the LIVE P101-APPLIED output
                                        # `_CONTINUATION_DECODE_THRESHOLD = 64`
                                        # +`_CONTINUATION_DECODE_MAX_CACHED_LEN
                                        # = 32768` (pristine has `= 128` and no
                                        # MAX_CACHED_LEN — confirmed against
                                        # the dev148 pristine tree). With P101
                                        # OFF that anchor is absent and the
                                        # const sub-patch skips. P101 is
                                        # default_on=False but co-enabled on
                                        # 35B PROD (GENESIS_ENABLE_P101=1 in the
                                        # live YAML), same situation as PN353A
                                        # — so requires (not just composes) is
                                        # correct: it mirrors the existing
                                        # anchors-the-LIVE-applied-output edges.
    },
    "PN118_V2_MD5_WORKSPACE": {
        "title": "PN118 v2 — md5+full-file PoC (PN119 reference pattern, workspace.py scope only)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_PN118_V2_MD5_WORKSPACE",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Genesis PoC of the PN119 single-file md5 + full-file "
            "replacement pattern, applied to pn118's TurboQuant "
            "workspace.py target. Master plan Phase 6 P3.1 closeout. "
            "SCOPE CORRECTION from v11.1.0 spec: spec assumed pn118 "
            "patches a single file at v1/attention/ops/workspace.py — "
            "reality is pn118 patches TWO files (v1/worker/workspace.py "
            "AND v1/attention/backends/turboquant_attn.py, 4 anchors "
            "total). This v2 PoC is scoped to workspace.py only; the "
            "original PN118 retains full coverage of turboquant_attn.py "
            "via its anchors. PN118 self-detects v2's Genesis marker on "
            "workspace.py and skips re-anchoring there, so the two "
            "compose (not conflict) — both can be enabled simultaneously "
            "without overlap. Default OFF so operators opt-in to A/B "
            "test the md5 pattern against the anchor-based original. If "
            "the PoC ships clean, future work converts pn79 (35 anchors "
            "across 3 files) to a multi-file md5 pattern — that's "
            "separate v11.2.0+ refactor because multi-file md5 pattern "
            "is not yet validated."
        ),
        "upstream_pr": 42551,
        "upstream_pr_relationship": "backport",
        "applies_to": {
            "is_turboquant": True,  # same target file as PN118 (workspace.py)
        },
        "implementation_status": "full",
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.pn118_v2_md5_workspace",
        "source": "vllm_pr_backport",
        "lifecycle": "experimental",
        "conflicts_with": [],
        "requires_patches": [],
    },
    "PN118_V2_MD5_TURBOQUANT_ATTN": {
        "title": "PN118 v2 — md5+full-file PoC (PN119 reference pattern, turboquant_attn.py scope)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_PN118_V2_MD5_TURBOQUANT_ATTN",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Genesis PoC sibling to PN118_V2_MD5_WORKSPACE — applies the "
            "PN119 md5+full-file pattern to pn118's second target file "
            "(v1/attention/backends/turboquant_attn.py). Together the two "
            "v2 patches replace pn118's anchor-based coverage of its full "
            "2-file scope with md5+full-file replacements (one v2 patch "
            "per target). Master plan Phase 6 P3.1 continuation in "
            "v11.2.0 — closes the deferred 'pn118 multi-file md5' work "
            "from v11.1.0. Drift finding during scout (2026-06-02): "
            "pn118's TQ_ANCHOR_INIT_OLD does not match current upstream — "
            "pn118 silently no-ops on that anchor at the current pin. "
            "This silent partial-apply is exactly the failure mode "
            "md5+full-file pattern prevents. Composes with PN118 + "
            "PN118_V2_MD5_WORKSPACE — all three can be enabled "
            "simultaneously via Genesis marker detection; pn118 detects "
            "both v2 markers and skips re-anchoring on both files. "
            "Default OFF — operators opt-in for A/B validation."
        ),
        "upstream_pr": 42551,
        "upstream_pr_relationship": "backport",
        "applies_to": {
            "is_turboquant": True,  # second target file of PN118 (turboquant_attn.py)
        },
        "implementation_status": "full",
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.pn118_v2_md5_turboquant_attn",
        "source": "vllm_pr_backport",
        "lifecycle": "experimental",
        "conflicts_with": [],
        "requires_patches": [],
    },
    "PN119": {
        "title": "TurboQuant k8v4 GQA head grouping kernel (vllm#40792)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_PN119",
        "default_on": True,
        "category": "kernel_perf",
        "credit": (
            "Backport of vllm#40792 (hoseung2, OPEN 2026-05-11). Adds "
            "_tq_grouped_decode_stage1 Triton kernel that loads K once "
            "per BLOCK_H q-head tile (sharing across the GQA group) and "
            "uses tl.dot (tensor cores) instead of element-wise scoring. "
            "Updates dispatch in triton_turboquant_decode_attention to "
            "route GQA-ratio>1 traffic to the grouped kernel. Upstream "
            "measured +16.5–27.2 % TPS on A100/H100 with GQA-ratio "
            "∈ {4, 8, 24}. Our 27B + 35B both run GQA-ratio 8 so the "
            "expected win is at the high end on Ampere SM 8.6. Applied "
            "via bundled diff + `patch` subprocess with md5 pre-patch "
            "guard; self-retires on drift."
        ),
        "upstream_pr": 40792,
        "upstream_pr_relationship": "backport",
        "applies_to": {
            "is_turboquant": True,  # k8v4 decode path
        },
        "implementation_status": "full",
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.pn119_tq_gqa_grouping",
        "source": "vllm_pr_backport",
        "lifecycle": "experimental",
        "conflicts_with": [],
        "requires_patches": [],
    },
    "P100": {
        "title": "FlashInfer FULL CUDA graph for spec-decode (vllm#41127)",
        "tier": "community",
        "family": "attention.flash",
        "env_flag": "GENESIS_ENABLE_P100",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": "Backport of vllm#41127 (open 2026-04-28). Per Sander: 'don't wait — study, import'. Native FlashInfer can route uniform query_len>1 (1+num_spec_tokens) batches through prefill wrapper in cudagraph mode (zero_rows padding bit-identical). Adds FISpecDecode dataclass + _get_spec_decode_prefill_wrapper method + per-row qo_indptr delta scan in build() + FISpecDecode case in forward(). 11 sub-patches on flashinfer.py. NO-OP for PROD (turboquant_attn). Active for 27B variants (FlashInfer + spec-decode + non-DCP). Expected: +5-10% TPS on Ampere SM 8.6. RECOMMENDED on Blackwell consumer (sm_120) where FlashInfer is the default backend and PIECEWISE downgrade was observed (apnar club-3090#51). Recommendation surfaced via gpu_profile.PATCH_RECOMMENDATIONS rule.",
        "upstream_pr": 41127,
        "upstream_pr_relationship": "backport",
        "applies_to": {},  # FlashInfer auto-selected; gating via env_flag only
        "apply_module": "sndr.engines.vllm.patches.attention.flash.p100_flashinfer_full_cg_specdec",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P103": {
        "title": "FLA Cliff 2 chunked fwd_h+fwd_o orchestrator (qwen36-27b-single-3090#1)",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_P103",
        "default_on": False,
        "category": "memory_hotfix",
        "credit": "Genesis-original 2026-04-28 in response to noonghunna Cliff 2 OOM report (qwen36-27b-single-3090#1). Wraps chunk.py::chunk_gated_delta_rule_fwd to split T-dim into MAX_T sub-prompts; runs fwd_h + fwd_o per sub-call, chains final_state, never materializes full (B, NT, H, V, K) h tensor. For Qwen3.6-27B at T=64K: peak h drops 4x (805 → 200 MiB per rank). Saves ~600 MiB headroom for long-context single-GPU users. Falls back to original for cu_seqlens != None or T <= MAX_T. Default OFF; opt-in via GENESIS_ENABLE_P103=1. Threshold: GENESIS_FLA_FWD_H_MAX_T (default 16384, rounded down to FLA_CHUNK_SIZE multiple). KDA path uncovered (separate model class).",
        "upstream_pr": None,
        "applies_to": {
            "model_arch": [
                "Qwen3MoeForCausalLM",
                "Qwen3_5ForConditionalGeneration",
                "Qwen3_5MoeForConditionalGeneration",
                "Qwen3_5MoeForCausalLM",
                "Qwen3_6MoeForConditionalGeneration",
                "Qwen3_6MoeForCausalLM",
                "Qwen3NextForCausalLM",
                "*",  # defer to runtime self-guard in apply()
            ],
            # [Genesis pin-gate 2026-05-11] PROD-active (GroupAB component
            # + long-context single-GPU users). Validated dev9 → dev93.
            "vllm_version_range": (">=0.20.0", "<0.24.0"),
        },
        "apply_module": "sndr.engines.vllm.patches.attention.gdn.p103_fla_cliff2_chunked",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P101": {
        "title": "TQ continuation 64-token slicing (vllm#41123 SELECTIVE)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_P101",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": "Selective backport of vllm#41123 TQ on hybrid models. TAKE: _CONTINUATION_DECODE_THRESHOLD 128→64 + _CONTINUATION_DECODE_MAX_CACHED_LEN=32K + 64-token slicing loop in _prefill_attention. SKIP: cudagraph_support downgrade (would hurt PROD), hybrid boundary-skip (would break our explicit skip-layers). Expected: +3-12% TPS on PROD long-context. Composes with P98/P99.",
        "upstream_pr": 41123,
        "upstream_pr_relationship": "backport",
        "applies_to": {
            "kv_cache_dtype": [
                "turboquant_k8v4", "turboquant_4bit_nc",
                "turboquant_k3v4_nc", "turboquant_3bit_nc",
            ],
        },
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.p101_tq_continuation_slicing",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P99": {
        "title": "WorkspaceManager.get_simultaneous memoization (perf hotfix)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_P99",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": "Per Sander 2026-04-28: 'if revert gives speedup, look at kernel — maybe rewrite'. P99 keeps upstream WorkspaceManager design (shared memory, 60x savings) but adds memoization to bypass per-call list-comp + accumulate + _ensure_workspace_size. Cache hit ~5x faster than full computation. Composes with P98 (P98 reverts turboquant_attn to per-layer; P99 helps any other backend using WorkspaceManager).",
        "upstream_pr": 40941,
        "upstream_pr_relationship": "enables_upstream",
        # [Iron rule #11 audit 2026-05-11 v2] P99 AUGMENTS the merged
        # upstream feature (#40941 WorkspaceManager) by wrapping
        # `get_simultaneous()` with memoization — it does NOT backport
        # the PR (the PR is already in our pin since dev9+). Case (b)
        # of iron rule #11: we do MORE on top of upstream. Audit script
        # routes via the explicit
        # `upstream_pr_relationship: "enables_upstream"` field to
        # exclude from NEWLY-MERGED categorization. KEEP active.
        # Cleanup queue: if upstream upstreams the memoization,
        # retire then.
        "applies_to": {},  # applies whenever WorkspaceManager is used
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.p99_workspace_manager_memoize",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P98": {
        "title": "TQ WorkspaceManager revert (vllm#40941 perf hotfix)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_P98",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": "Reverts upstream PR #40941 (MERGED 2026-04-27). PR introduced WorkspaceManager indirection in turboquant_attn._decode_attention hot path. Diagnosis 2026-04-28: caused 17% TPS regression on PROD (200 → 167 TPS) due to current_workspace_manager().get_simultaneous() Python lookup × N layers × per-step. Restores OLD per-layer cached buffer pattern. Memory cost: O(num_layers) extra dequant buffers (~1GB for 64-layer model). DO NOT enable on H100/H200 high-concurrency where WorkspaceManager amortizes better. NOTE: this patch is a DELIBERATE INVERSE of merged upstream behavior (NOT a backport) — it remains a perf hotfix specifically for Ampere small-batch single-stream workloads even though the upstream PR is merged. Author: Sandermage.",
        "upstream_pr": 40941,
        "upstream_pr_relationship": "intentional_inverse",
        "applies_to": {
            "kv_cache_dtype": [
                "turboquant_k8v4", "turboquant_4bit_nc",
                "turboquant_k3v4_nc", "turboquant_3bit_nc",
            ],
        },
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.p98_tq_workspace_revert",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P95": {
        "title": "Marlin TP cudagraph cap on Ampere (vllm#40385)",
        "tier": "community",
        "family": "compile_safety",
        "env_flag": "GENESIS_ENABLE_P95",
        "default_on": False,
        "category": "stability",
        "credit": "Backport of vllm#40385 (OPEN as of 2026-04-28). Defensive cap of max_cudagraph_capture_size to 8 when ALL of: TP>1, Ampere SM 8.0 family (covers SM 8.6 A5000), quantization endswith '_marlin', AND user did NOT set explicit cudagraph sizing. Mitigates vllm#40121 (illegal memory access during CG replay on TP>1 + Marlin + Ampere). NO-OP for our PROD (FP8, not Marlin); ACTIVE for Lorbus INT4 + Minachist gs128 (Marlin path). Operator override via --compilation-config bypasses entirely.",
        "upstream_pr": 40385,
        "upstream_pr_relationship": "backport",
        "applies_to": {
            "quant_format": [
                "gptq_int4", "gptq_int8", "awq_int4", "awq_int8",
                "compressed_tensors", "int4_w4a16", "int8_w8a16",
                "autoround_int4", "autoround_int8",
            ],
        },
        "apply_module": "sndr.engines.vllm.patches.compile_safety.p95_marlin_tp_cudagraph_cap",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P91": {
        "title": "AutoRound row-parallel group cdiv + start-idx fix (vllm#39460)",
        "tier": "community",
        "family": "quantization",
        "env_flag": "GENESIS_ENABLE_P91",
        "default_on": False,
        "category": "quantization",
        "credit": "Backport of non-MoE-specific portion of vllm#39460 (CLOSED, supersession chain #40281/#41588 also closed without merge, fix abandoned upstream). {auto_gptq,gptq_marlin}.py computes scales_and_zp_size = input_size_per_partition // group_size — when input_size_per_partition % group_size != 0 (AutoRound INT4/INT8 checkpoints with awkward shard sizes), this floor-div drops the trailing partial group of scales. Combined with parameter.py:222-225 load_row_parallel_weight using `tp_rank * shard_size` as start_idx (in scale-rows units, but the source tensor is indexed in scales-rows that map to input-element groups), rank-1 scales load from the wrong offset for partial-group shards → silent dequant corruption or fallback to slow non-Marlin path. P91 (a) replaces both floor-divs with cdiv(), (b) tags scales/qzeros with row_group_size + row_input_size_per_partition, (c) makes load_row_parallel_weight compute start_idx as (tp_rank * input_partition_size) // group_size when those tags present. Hypothesized as dominant cause of Lorbus INT4 < INT8 perf gap on our 2x A5000 (87/61/67 vs 93/77/86 t/s) — sister bug #38064 had 2.72x latency improvement when fixed. We do NOT port the MoE/gate_linear/gemma4 changes (those are Gemma4-specific). v7.62.2 refresh (2026-05-25): upstream renamed gptq_marlin.py → auto_gptq.py between dev338 and dev371; P91 now resolves the target file with a fallback (auto_gptq first, gptq_marlin fallback) so a single patch source covers both allowlisted pins. Idempotency uses the version-agnostic marker base so an upgrade from v7.62.1 is recognized as already-applied. Companion patch P91B covers the same bug class in inc.py + compressed_tensors_wNa16.py + compressed_tensors_w4a8_fp8.py for checkpoints that go through those code paths.",
        "upstream_pr": 39460,
        "upstream_pr_relationship": "backport",
        "applies_to": {
            "quant_format": [
                "autoround_int8", "autoround_int4",
                "gptq_int4", "int8_w8a16", "int4_w4a16",
                "compressed_tensors",
            ],
        },
        "apply_module": "sndr.engines.vllm.patches.quantization.p91_autoround_row_group_cdiv",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P91B": {
        "title": "AutoRound row-group cdiv defensive coverage for INC + compressed-tensors schemes (P91 sibling, vllm#39460-derived)",
        "tier": "community",
        "family": "quantization",
        "env_flag": "GENESIS_ENABLE_P91B",
        "default_on": False,
        "category": "quantization",
        "credit": "Defensive sibling of P91. Same bug class (silent dequant corruption when input_size % group_size != 0 or input_size_per_partition % group_size != 0 for row-group quantized layers) in files that vllm#39460 did NOT touch: inc.py (Intel Neural Compressor linear method) and compressed_tensors/schemes/{compressed_tensors_wNa16,compressed_tensors_w4a8_fp8}.py. cdiv-only fix (3 sub-patches across 3 files). inc.py carries a cross-pin anchor drift between dev338 (`self.group_size`) and dev371 (bare `group_size`); P91B uses two independent factories per Option A from the Step 0 anchor manifest, whichever matches the live pin applies. compressed_tensors_w4a8_int.py is NOT covered — its function-head assert proactively rejects partial-group shards, so the floor-div there is always exact and there is no silent-corruption surface to fix. The inc.py setattr companion for P91's parameter.py loader gate is also NOT covered: it is infrastructure for an existing fix rather than a new bug fix, deferred to a future refresh if INC enters Genesis prod use. Relationship to vllm#39460 is `related_not_superseding` (not `backport`) because the PR did not touch these files; status-based retire on upstream merge does not apply to P91B.",
        "upstream_pr": 39460,
        "upstream_pr_relationship": "related_not_superseding",
        "applies_to": {
            "quant_format": [
                "compressed_tensors",
                "int4_w4a16", "int8_w8a16",
                "gptq_int4", "gptq_int8",
            ],
        },
        "apply_module": "sndr.engines.vllm.patches.quantization.p91b_autoround_row_group_cdiv_multi_scheme",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN400": {
        "title": "Restore is_sym qzeros guard for symmetric AutoRound/GPTQ Marlin MoE (vllm#45656 backport; fixes vllm#43409 regression)",
        "tier": "community",
        "family": "quantization",
        "env_flag": "GENESIS_ENABLE_PN400",
        "default_on": False,
        "category": "quantization",
        "credit": "P0 CORRECTNESS regression found by the /loop upstream sweep 2026-06-20. vllm#43409 (merged 06-12, IN dev148) removed the `if not self.quant_config.is_sym else None` guard in AutoGPTQMoEMethod.get_fused_moe_quant_config so the CPU fused_experts_cpu path could receive synthesized zero points for symmetric models. On NVIDIA GPUs this regressed SYMMETRIC (is_sym=True) AutoRound/GPTQ Marlin MoE: the method then passes the meaningless w13_qzeros/w2_qzeros to the Marlin kernel -> INCORRECT expert outputs. Our 27B = Qwen3.6-27B-int4-AutoRound is CONFIRMED on this path (live checkpoint config.json: quant_method=auto-round, sym=True, bits=4, group_size=128, packing auto_round:auto_gptq -> AutoGPTQMoEMethod) on 2x A5000. Fix vllm#45656 ('Restore is_sym guard for zp in GPTQ/CT MoE', merged 06-18 16:20Z) landed ~12h AFTER the dev148 base commit -> NOT in pin. PN400 backports the auto_gptq.py half: gate w1_zp/w2_zp on `not is_sym` before the gptq_marlin_moe_quant_config return. The upstream `or backend==CPU` clause is intentionally dropped (NVIDIA-only rig; avoids the WNA16MoEBackend import + a NameError-if-half-applied surface). The compressed-tensors twin (file 2 of #45656) is a different checkpoint format we do not run; add PN400B if one enters rotation. Anchor SELF-GATES: the unconditional `w1_zp=getattr(layer,'w13_qzeros',None),` text exists ONLY on pins with #43409 and without #45656 (pre-#43409 / post-#45656 it is `... if <cond> else None` -> self-skip). lifecycle=experimental: the text transform is unit-tested (dev148-broken -> fixed) but the semantic 27B greedy A/B (dev148 vs +PN400) is operator-gated (displaces the 35B PROD).",
        "upstream_pr": 45656,
        "upstream_pr_relationship": "backport",
        # RETIRED 2026-06-24 (pin bump dev148 -> dev301): the NVIDIA-scoped
        # subset of #45656 (the auto_gptq.py is_sym qzeros guard) is IN
        # dev301 — deep-diff iron-rule-#11 + the dev301 anchor-SOT regen
        # confirm the anchor drifted on dev301. The upstream `or
        # backend==CPU` clause was never used on our NVIDIA-only rig. The
        # anchor self-gates (the unconditional `w1_zp=...` text is gone
        # post-#45656), so this is the FORMAL retire marking. Cap the
        # applies_to range upper bound to <dev301 so the version gate also
        # self-skips on dev301+, AND mirror it at top level for
        # iron-rule-#11 provenance. Still applies through dev148 (the
        # previous/rollback pin), where the #43409 regression is live.
        "applies_to": {
            "quant_format": [
                "autoround_int4", "autoround_int8",
                "gptq_int4", "gptq_int8",
                "int4_w4a16", "int8_w8a16",
            ],
            "vllm_version_range": (">=0.23.0", "<0.23.1rc1.dev301"),
        },
        "vllm_version_range": "<0.23.1rc1.dev301",
        "superseded_by": "vllm#45656 (MERGED 2026-06-18, IN dev301 — restores the is_sym qzeros guard in AutoGPTQMoEMethod; the NVIDIA auto_gptq.py half we backported is native on dev301, anchor drifted per the anchor-SOT regen). CPU clause never used on our rig. Applies through dev148; retired on dev301+.",
        "apply_module": "sndr.engines.vllm.patches.quantization.pn400_marlin_moe_sym_zp_guard",
        "lifecycle": "retired",
        "implementation_status": "full",
    },

    # ─── Legacy patches (P1–P46 series, pre-dispatcher era) ─────────────
    # These patches predate the PATCH_REGISTRY metadata system. They have
    # been live in PROD since pre-v7.0 and don't currently read an env
    # flag — they apply unconditionally as part of `apply_all`. The
    # synthetic `GENESIS_LEGACY_P*` env_flags below exist purely so the
    # dispatcher / validator / `genesis explain` see a coherent registry
    # entry; setting them has no runtime effect (yet). Future work: wire
    # actual opt-out gating where it makes sense.
    #
    # Why register them: lets `apply_all_dispatcher_sync` test pass,
    # surfaces these patches in `genesis list-patches` / `genesis explain`,
    # and provides a stable shape for documentation tooling.

    "P1": {
        "title": "FP8 kernel dispatcher (P1/P2 — Ampere FP8 viability)",
        "tier": "community",
        "family": "quantization",
        "env_flag": "GENESIS_LEGACY_P1",
        "default_on": True,
        "implementation_status": "marker_only",
        "lifecycle": "legacy",
        "category": "quantization",
        "credit": "Pre-dispatcher legacy patch. Wires Ampere SM86 to FP8 kernel paths so consumer 3090/A5000 can serve FP8-quantized models.",
    },
    "P3": {
        "title": "TurboQuant BF16→FP8 cast (Ampere fix)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_LEGACY_P3",
        "default_on": True,
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.p3_tq_bf16_cast",
        "lifecycle": "legacy",
        "category": "quantization",
        "credit": "Pre-dispatcher legacy patch. Inserts BF16→FP8 cast on TQ ingress for SM86 where FP8 is software-emulated.",
        "implementation_status": "full",
    },
    "P4": {
        "title": "TurboQuant hybrid model support — RETIRED 2026-06-11",
        "tier": "community",
        "family": "scheduler",
        "env_flag": "GENESIS_LEGACY_P4",
        "default_on": False,
        "apply_module": "sndr.engines.vllm._archive.p4_tq_hybrid",
        "lifecycle": "retired",
        "category": "kv_cache",
        "credit": "Pre-dispatcher legacy patch. Removed hybrid (GDN + full attention) model rejection in TQ path, enabling Qwen3.5/3.6 hybrid serving with TQ k8v4. RETIRED — upstream supports hybrid TQ natively via vllm#39931.",
        # [Retire 2026-06-11, preflight residual triage §3 — executes the
        # registry's own overdue 2026-05-05 plan] vllm#39931 MERGED
        # 2026-05-05 00:14 UTC (gh-verified 2026-06-11; JartX + jhsmith409
        # + Sandermage co-authors). Upstream detects hybrid via
        # layer_types/layers_block_type/attn_type_list — near-verbatim our
        # `_genesis_p4_full_attention_indices` logic — and computes TQ
        # page-size via lcm in `_align_hybrid_block_size`. PROD boot on
        # pin 0.22.1rc1.dev259+g303916e93 already self-skips P4 via the
        # upstream marker ("skipped: P4 ... upstream_merged", deduped boot
        # line 152) — P4-OFF is the live steady state, and hybrid-TQ
        # boots clean on this pin (fleet validation journal 2026-06-11).
        # Module archived; delete file in next cleanup batch. Regenerate
        # docs/PATCHES_AUTO.md + run pin-gate / iron-rule-11 gates.
        "superseded_by": "vllm#39931 (MERGED 2026-05-05 00:14 UTC, gh-verified 2026-06-11) — upstream hybrid detection + lcm page-size in _align_hybrid_block_size",
        "vllm_version_range": "<0.22.1rc1.dev259",  # supersession byte-verified on this pin's pristine tree
        "implementation_status": "full",
    },
    "P5": {
        "title": "KV cache page size unification",
        "tier": "community",
        "family": "kv_cache",
        "env_flag": "GENESIS_LEGACY_P5",
        "default_on": True,
        "apply_module": "sndr.engines.vllm.patches.kv_cache.p5_page_size",
        "lifecycle": "legacy",
        "category": "kv_cache",
        "credit": "Pre-dispatcher legacy patch. Unifies per-layer page size across hybrid attention layers so block manager doesn't fragment.",
        "implementation_status": "full",
    },
    "P5b": {
        "title": "KV page-size pad-smaller-to-max (env-flag coordinator for P5)",
        "tier": "community",
        "family": "memory",
        # Audit P2 fix 2026-05-05: registry was `GENESIS_ENABLE_P5B_PAGE_PAD`
        # but wiring code + docstrings use `GENESIS_ENABLE_P5B`. Aligned.
        "env_flag": "GENESIS_ENABLE_P5B",
        "default_on": False,
        # `coordinator` lifecycle (introduced 2026-05-06): the wiring module
        # has no real binding — it only reads the env gate and reports
        # `applied`/`skipped`. The actual text-patch (selecting v2 pad-smaller-
        # to-max body in `_align_hybrid_block_size` + stamping
        # `real_page_size_bytes` on TQFullAttentionSpec) is performed by P5
        # when GENESIS_ENABLE_P5B=1 is set. Kept as a separate registry entry
        # so operators can grep for the feature flag and so audit/preflight
        # can flag P5+P5b composability concerns explicitly.
        "apply_module": "sndr.engines.vllm.patches.memory.p5b_page_size_pad_smaller",
        "lifecycle": "coordinator",
        "category": "kv_cache",
        "credit": "Pre-dispatcher legacy patch. Opt-in companion to P5 — pads smaller pages up to max so all layers share one block-pool stride. Guarded by env (was always opt-in). Coordinator pattern: real binding in P5; this entry is a documented feature-flag handle.",
        "implementation_status": "full",
    },
    "P6": {
        "title": "TurboQuant-aware attention page size — RETIRED 2026-06-11",
        "tier": "community",
        "family": "compile_safety",
        "env_flag": "GENESIS_LEGACY_P6",
        "default_on": False,
        "apply_module": "sndr.engines.vllm._archive.p6_tq_block_size_align",
        "lifecycle": "retired",
        "category": "kv_cache",
        "credit": "Pre-dispatcher legacy patch. Selected TQ-aware page size (matches TQ packed slot stride) when TQ KV is active. RETIRED — vllm#39931 merged a corrected superset.",
        # [Retire 2026-06-11, preflight residual triage §3, formalizes the
        # §1 neutralization (commit 23033ddb)] vllm#39931 (MERGED
        # 2026-05-05) ships a corrected superset: upstream
        # `_align_hybrid_block_size` uses `lcm(tq_page, skip_page)` — not
        # `max` — at pristine platforms/interface.py:573-609 (TQ branch
        # imports TQFullAttentionSpec lazily at :582), fixing the max-vs-
        # lcm bug our P6 carried. Mis-apply hazard that forced the §1
        # hard-skip: P6's drift marker `'TQFullAttentionSpec,'` (trailing
        # comma) never matches the pristine lazy import, while BOTH P6
        # anchors still match pristine — an enabled P6 would "apply" and
        # inject a dead duplicate elif + redundant import on top of the
        # merged superset. Module archived.
        "superseded_by": "vllm#39931 (MERGED 2026-05-05) — corrected superset: lcm-based _align_hybrid_block_size at pristine platforms/interface.py:573-609",
        "vllm_version_range": "<0.22.1rc1.dev259",  # supersession byte-verified on this pin's pristine tree
        "implementation_status": "full",
    },
    "P7": {
        "title": "GDN dual-stream in_proj parallelism",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_LEGACY_P7",
        "default_on": True,
        "apply_module": "sndr.engines.vllm.patches.attention.gdn.p7_gdn_dual_stream",
        "lifecycle": "legacy",
        "category": "kernel_perf",
        "credit": "Pre-dispatcher legacy patch. Splits GDN in_proj across two CUDA streams so q/k/v projections overlap. Validated +8% decode on 35B.",
        "conflicts_with": ["P7b", "PN204"],  # PN204 = port of vllm#42301, same site as P7
        "applies_to": {
            # 2026-06-19 drift audit: superseded by PN204 (port of vllm#42301 —
            # same forward_cuda Part-1 in_proj site, but compile-safe via the native
            # maybe_execute_in_parallel at utils/multi_stream_utils.py). P7 can never
            # apply on dev148: its raw torch.cuda.Stream apply() is deferred under
            # torch.compile, the #41126 gdn module split moved the anchored region
            # (8-space, no hasattr branch), and all 5 builtin YAMLs set
            # GENESIS_LEGACY_P7:'0'. Sibling P7b is already retired -> PN204 for the
            # identical reason. Cap off 0.22.1rc1.dev259+ (where PN204 became the
            # chosen path). legacy lifecycle is exempt from the stale-range gate.
            "vllm_version_range": (">=0.20.0", "<0.22.1rc1.dev259"),
        },
        "implementation_status": "full",
    },
    "P7b": {
        "title": "GDN dual-stream via torch.library.custom_op — RETIRED 2026-06-11",
        "tier": "community",
        "family": "attention.gdn",
        # Audit P2 fix 2026-05-05: registry was `GENESIS_ENABLE_P7B_DUAL_STREAM_CUSTOM_OP`
        # but wiring code + docstrings use `GENESIS_ENABLE_P7B`. Aligned.
        "env_flag": "GENESIS_ENABLE_P7B",
        "default_on": False,
        "apply_module": "sndr.engines.vllm._archive.p7b_gdn_dual_stream_customop",
        "lifecycle": "retired",
        "category": "kernel_perf",
        "credit": "Pre-dispatcher legacy patch. Custom-op variant of P7 dual-stream — was the opt-in alternative for cudagraph capture compatibility experiments. RETIRED — superseded by PN204 + PN365; both anchors dead on current pin.",
        # [Retire 2026-06-11, preflight residual triage §3, byte-verified]
        # Both P7b anchors are dead post-refactor (#41126 split moved the
        # target to gdn/qwen_gdn_linear_attn.py): `get_tensor_model_
        # parallel` has 0 hits in the target file, and the 12-space serial
        # in_proj pair ("else:\n mixed_qkvz, _ = self.in_proj_qkvz(...)\n
        # ba, _ = self.in_proj_ba(...)") counts 0 in pristine. Superseded
        # by PN204 (port of vllm#42301 — upstream helper
        # `maybe_execute_in_parallel` present in pin at
        # utils/multi_stream_utils.py:20) and PN365 (port of vllm#42746
        # single-GEMM in_proj fuse — applied in one PROD container, boot
        # line 100; skipped-disabled in the other). Extra retire reason:
        # PN204_FWD_NEW / PN365_FWD_NEW emit P7b's anchor-2 text in their
        # fallback branches → a resurrected P7b risks sibling-matching
        # another patch's post-apply output. conflicts_with kept on
        # purpose: the mutual-exclusion contract still documents the
        # shared forward_cuda Part 1 site.
        "superseded_by": "PN204 (vllm#42301 port; maybe_execute_in_parallel native in pin at utils/multi_stream_utils.py:20) + PN365 (vllm#42746 port, applied in one PROD container) — P7b anchors count=0 on pristine gdn/qwen_gdn_linear_attn.py (byte-verified 2026-06-11)",
        "vllm_version_range": "<0.22.1rc1.dev259",  # anchors dead from this pin's gdn split onward
        "conflicts_with": ["P7"],
        "implementation_status": "full",
    },
    "PN13": {
        "title": "CUDAGraphWrapper lambda arity (vllm#41235 backport) — RETIRED 2026-05-04",
        "tier": "community",
        "family": "compile_safety",
        "env_flag": "GENESIS_ENABLE_PN13_CUDA_GRAPH_LAMBDA_ARITY",
        "default_on": False,
        "lifecycle": "retired",
        "notes": (
            "upstream_native_via_pr41235 — vllm 0.20.2 (commit "
            "c2fb013) merged identical change in cuda_graph.py: "
            "patch(\"gc.collect\", lambda *args, **kwargs: None) + "
            "patch(\"torch.accelerator.empty_cache\", lambda *args, "
            "**kwargs: None). Upstream code now matches our PN13 "
            "replacement byte-for-byte. PN13 anchor (pre-fix lambda: "
            "None pattern) no longer matches → silent skip. Per Sander "
            "rule (2026-05-04): 'when upstream code matches a patch, "
            "retire the patch'. Retired."
        ),
        "category": "compile_safety",
        "credit": "Backport of vllm#41235 by Roi Koren (NVIDIA). RETIRED — upstream natively fixes after vllm v0.20.2.",
        "upstream_pr": 41235,
        "upstream_pr_relationship": "backport",
        "superseded_by": "vllm#41235 (merged 2026-04-29, in commit c2fb013 / v0.20.2 — byte-equivalent on dev93+dev209)",
        "vllm_version_range": "<0.20.2",  # active before upstream merge in v0.20.2
        "apply_module": "sndr.engines.vllm._archive.pn13_cuda_graph_lambda_arity",
        "applies_to": {
            # [Genesis pin-gate 2026-05-11] #41235 MERGED 2026-04-29 (in
            # commit c2fb013, vllm 0.20.2). Upper bound formalizes retire
            # — pin-gate adds belt-to-the-suspenders (lifecycle="retired"
            # already de-facto skips, this makes the version boundary
            # explicit for `genesis explain` and audit reports).
            "vllm_version_range": "<0.20.2",
        },
        "implementation_status": "full",
    },
    "P8": {
        "title": "KV hybrid reporting (per-token capacity) — RETIRED 2026-05-04",
        "tier": "community",
        "family": "scheduler",
        "env_flag": "GENESIS_LEGACY_P8",
        "default_on": False,
        "lifecycle": "retired",
        "notes": (
            "upstream_native_via_get_max_concurrency_refactor — vllm "
            "v0.20.2rc1.dev9 (commit 01d4d1ad3) refactored "
            "_report_kv_cache_config to use "
            "get_max_concurrency_for_kv_cache_config(vllm_config, "
            "kv_cache_config) which natively handles hybrid layouts "
            "(SWA / chunked-local groups with per-request block count "
            "capped by window). The new formula `max_concurrency * "
            "max_model_len` supersedes our P8 approach (excluding O(1) "
            "Mamba groups from per-token divisor). Engine now reports "
            "correct capacity natively without our patch. P8 anchors "
            "no longer match (kv_cache_utils.py refactored)."
        ),
        "category": "kv_cache",
        "credit": "Pre-dispatcher legacy patch. Reports KV capacity per-token (not per-block) for hybrid models so scheduler doesn't over-admit. RETIRED upstream natively fixes after vllm v0.20.2.",
        "superseded_by": "vllm dev9 commit 01d4d1ad3 — get_max_concurrency_for_kv_cache_config refactor handles hybrid layouts natively (no specific PR captured; verified via anchor mismatch on dev9+)",
        "vllm_version_range": "<0.20.2rc1.dev9",  # mirrors applies_to (gated SoT); reconciled D17 2026-06-17 (dropped cosmetic +g01d4d1ad3 suffix)
        "apply_module": "sndr.engines.vllm._archive.p8_kv_hybrid_reporting",
        "applies_to": {
            # [Iron rule #11 formal retire 2026-05-11] Promoted from
            # waiver — notes already had identifiable supersession.
            # Anchors don't match dev9+, so pin-gate upper bound is
            # cosmetic on the legacy auto-apply path (synthetic env_flag),
            # but documents the supersession boundary for `genesis explain`.
            "vllm_version_range": "<0.20.2rc1.dev9",
        },
        "implementation_status": "full",
    },
    "P12": {
        "title": "Qwen3 <tool_call> implicit reasoning end",
        "tier": "community",
        "family": "reasoning",
        "env_flag": "GENESIS_LEGACY_P12",
        "default_on": True,
        "apply_module": "sndr.engines.vllm.patches.reasoning.p12_tool_call_reasoning",
        "lifecycle": "legacy",
        "category": "structured_output",
        "credit": "Pre-dispatcher legacy patch. Treats <tool_call> emission as implicit </think>, fixing Qwen3 reasoning models that omit explicit </think> before tool calls. Updated v7.62.5 to FIRST-occurrence (was LAST), retiring P61.",
        "superseded_by": "upstream qwen3_parser refactor (marker '_tool_call_token_id' present on dev93+; auto-skip via wiring drift detector)",
        # [Iron rule #11 audit 2026-05-11] Wire detector auto-skips on
        # dev93+dev209 — upstream qwen3 reasoning_parser refactor added
        # `_tool_call_token_id` attribute which handles the implicit
        # </think> case natively (different impl but same outcome).
        # Lifecycle stays "legacy" (architectural — pre-dispatcher
        # auto-apply pattern); skip is correct + safe.
        # 2026-06-19 (dev148 TIER-1 audit): capped <0.23.0 — the parser
        # reorg #45413/#45588 (MERGED in dev148) deleted/restructured P12's
        # qwen3 reasoning target; the native engine parser handles the
        # implicit </think>-before-tool_call case. Honest cap.
        "applies_to": {"vllm_version_range": (">=0.20.0", "<0.23.0")},
        "implementation_status": "full",
    },
    "P14": {
        "title": "block_table tail zero-fill",
        "tier": "community",
        "family": "kv_cache",
        "env_flag": "GENESIS_LEGACY_P14",
        "default_on": True,
        "apply_module": "sndr.engines.vllm.patches.kv_cache.p14_block_table",
        "lifecycle": "legacy",
        "category": "kernel_safety",
        "credit": "Pre-dispatcher legacy patch. Zero-fills block_table tail past valid sequences so out-of-bounds prefetch doesn't read stale page indices.",
        "implementation_status": "full",
    },
    "P15": {
        "title": "Qwen3 None/null tool arg parser",
        "tier": "community",
        "family": "tool_parsing",
        "env_flag": "GENESIS_LEGACY_P15",
        "default_on": True,
        "apply_module": "sndr.engines.vllm.patches.tool_parsing.p15_qwen3_none_null",
        "lifecycle": "legacy",
        "category": "structured_output",
        "credit": "Pre-dispatcher legacy patch. Tolerates None / null tool arguments in Qwen3 parser instead of raising.",
        "implementation_status": "full",
    },
    "P17": {
        "title": "Marlin MoE per-SM tuning (P17/P18)",
        "tier": "community",
        "family": "moe",
        "env_flag": "GENESIS_LEGACY_P17",
        "default_on": True,
        "implementation_status": "marker_only",
        "lifecycle": "legacy",
        "category": "kernel_perf",
        "credit": "Pre-dispatcher legacy patch. Per-SM (SM86) tuned configs for Marlin MoE kernel — bsm=8 selected on Ampere consumer cards.",
    },
    "P18b": {
        "title": "TurboQuant decode stage1 tune",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_LEGACY_P18B",
        "default_on": True,
        "implementation_status": "marker_only",
        "lifecycle": "legacy",
        "category": "kernel_perf",
        "credit": "Pre-dispatcher legacy patch. Tuned launch config for TQ decode stage1 kernel on SM86.",
    },
    "P18B_TEXT": {
        "title": "TurboQuant decode stage1 kernel-literal tune (TEXT-PATCH)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_P18B_TEXT",
        "default_on": True,
        "category": "kernel_perf",
        "credit": (
            "Genesis-original 2026-06-08. Kernels-audit agent flagged "
            "the original P18b (kernels_legacy/tq_decode_tune.py + "
            "_per_patch_dispatch.py:6275) as DEAD CODE: it only logs "
            "the resolved VLLM_TQ_DECODE_{BLOCK_KV,NUM_WARPS,NUM_STAGES} "
            "env vars and never patches the actual Triton launcher. "
            "Result: 35B + 27B production has been running the upstream "
            "H100 defaults (num_warps=4/1, num_stages=2/1) regardless "
            "of env overrides — under-utilising Ampere SM 8.6 "
            "(RTX A5000 / 3090) shared-memory budgets. "
            "This patch is the missing text-patch half: rewrites the "
            "two launch-parameter blocks of "
            "vllm/v1/attention/ops/triton_turboquant_decode.py in place "
            "at boot, using the values from resolve_decode_tune(). "
            "SM-8.6-validated default is num_warps=8 num_stages=3 "
            "(per master plan section 15.1 empirical note: num_stages=2 "
            "measured -2% to -9% on A5000). "
            "Two sub-patches: GQA branch (line ~790) and MHA branch "
            "(line ~830). Both required=False — partial-apply is allowed."
        ),
        "applies_to": {
            "is_turboquant": True,
            "sm_min": (8, 0),
            "sm_max": (9, 0),  # H100+ keep upstream defaults (already H100-tuned)
        },
        "implementation_status": "full",
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.p18b_kernel_literals_textpatch",
        "source": "genesis_original",
        "lifecycle": "experimental",
        "conflicts_with": [],
        # PN119's replacement CREATES the GQA/MHA launcher blocks this
        # patch anchors on (verified vs pristine 0.22.1 tree 2026-06-10:
        # pristine has a single 8-space launcher; the 12-space if/else
        # is PN119 output). Preflight reports this as CHAINED_ANCHOR.
        "requires_patches": ["PN119"],
    },
    "P20": {
        "title": "TurboQuant continuation-prefill FP16 rotate — RETIRED 2026-06-11",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_LEGACY_P20",
        "default_on": False,
        "implementation_status": "marker_only",
        "lifecycle": "retired",
        "category": "kernel_perf",
        "credit": "Pre-dispatcher legacy patch. FP16 rotation for TQ continuation-prefill path (JartX/vllm#11 prerequisite for v7.0+). RETIRED — upstream superset native; P20 was marker_only and never bound.",
        # [Retire 2026-06-11, preflight residual triage §3, byte-verified
        # — zero-risk: implementation_status='marker_only', the helper
        # never bound to the live file] Upstream ships a SUPERSET of
        # JartX/vllm#11: per-layer `_tq_Pi_half` fp16 rotation cache
        # (pristine turboquant_attn.py:352 "layer._tq_Pi_half =
        # H.to(torch.float16)", read at :791 in the 789-797 block) PLUS
        # preallocated k_full/v_full that eliminate the torch.cat
        # transient — better than our approach. No wiring module to
        # archive (registry-only entry, like PN34). upstream_compat.py
        # PR_JARTX_11 merged_date updated in the same batch.
        "superseded_by": "upstream superset of JartX/vllm#11 — per-layer _tq_Pi_half fp16 cache (pristine turboquant_attn.py:352, 789-797) + prealloc k_full/v_full removing the torch.cat transient (byte-verified 2026-06-11)",
        "vllm_version_range": "<0.22.1rc1.dev259",  # supersession byte-verified on this pin's pristine tree
    },
    "P22": {
        "title": "TurboQuant shared dequant prealloc",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_LEGACY_P22",
        "default_on": True,
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.p22_tq_prealloc",
        "lifecycle": "legacy",
        "category": "memory_pool",
        "credit": "Pre-dispatcher legacy patch. Preallocates shared dequant scratch buffer so TQ doesn't allocate-per-step (Genesis-original).",
        # [Iron rule #11 investigation queued 2026-05-11]
        # On dev209+g5536fc0c0 the drift detector auto-skips P22 because
        # upstream removed `_init_turboquant_buffers` (PR #40655-style
        # restructure landed without a formal merge). However, per our
        # patch's own docstring (lines 43-55), the alternative upstream
        # approaches LACK profiler visibility — which is the specific
        # value-add P22 provides (visible to memory-profiler → correct
        # KV cache sizing → no #40420-class OOM at long context).
        # Per iron rule #11 case (b) "our patch does MORE": should be
        # UPDATED to re-hook profiler visibility on top of the new
        # upstream restructure, NOT silently retired. Currently auto-skip
        # is safe (no crash, just loses our improvement); investigation
        # queued — read dev209's gpu_model_runner.capture_model + the
        # current TurboQuantAttentionImpl class to design new hook site.
        "implementation_status": "full",
    },
    "P23": {
        "title": "Marlin FP32_REDUCE env override",
        "tier": "community",
        "family": "kernels",
        "env_flag": "GENESIS_LEGACY_P23",
        "default_on": True,
        "implementation_status": "marker_only",
        "lifecycle": "legacy",
        "category": "kernel_perf",
        "credit": "Pre-dispatcher legacy patch. Honors VLLM_MARLIN_FP32_REDUCE env to force FP32 reduction in Marlin matmul (numerical-stability hedge).",
    },
    "P23_WIRE": {
        "title": "Marlin FP32_REDUCE env wire (P23 companion, fix-wire 2026-06-04)",
        "tier": "community",
        "family": "kernels",
        "env_flag": "GENESIS_ENABLE_P23_MARLIN_FP32_REDUCE_WIRE",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.kernels.p23_marlin_fp32_reduce_wire",
        "lifecycle": "experimental",
        "category": "kernel_perf",
        "credit": (
            "Genesis-original 2026-06-04 — fixes the long-standing P23 "
            "wire gap. Previously P23 shipped a working env reader + "
            "platform auto-detect but the dispatch handler only LOGGED "
            "the decision, never propagated it. P23_WIRE text-patches "
            "two upstream call sites: marlin_utils.py:36 (module-level "
            "USE_FP32_REDUCE_DEFAULT constant) + marlin_moe.py:158,217 "
            "(W13 + W2 hardcoded use_fp32_reduce=True). Both now read "
            "VLLM_MARLIN_FP32_REDUCE env. Effect on PROD 27B Qwen3.6 "
            "MoE Marlin (GPTQ, 2× A5000 SM 8.6) with "
            "VLLM_MARLIN_FP32_REDUCE=0: +1.5-3% TGS per Genesis empirical "
            "data, no quality drop on GSM8K/MMLU. Contra-productive on "
            "SM 9.0+ (Hopper has native FP32 tensor cores)."
        ),
        "upstream_pr": None,
        "applies_to": {},
        "implementation_status": "full",
        # Anchor collision: P23_WIRE and PN368 both rewrite the SAME w13
        # use_fp32_reduce block in marlin_moe.py. P23_WIRE registers first,
        # rewrites the anchor, then PN368's required=True sub-patch finds its
        # anchor gone and boots FAILED with marlin_moe.py half-patched.
        # Declared mutually exclusive until PN368 grows a dual-anchor. Caught
        # by deep-audit 2026-06-14 (#2).
        "conflicts_with": ["PN368"],
        "composes_with": ["P23"],
    },
    "P24": {
        "title": "fused_moe num_warps/num_stages overlay",
        "tier": "community",
        "family": "moe",
        "env_flag": "GENESIS_LEGACY_P24",
        "default_on": True,
        "apply_module": "sndr.engines.vllm.patches.moe.p24_moe_tune",
        "lifecycle": "legacy",
        "category": "kernel_perf",
        "credit": "Pre-dispatcher legacy patch. Overlays SM86-tuned num_warps/num_stages on fused_moe kernel selection.",
        "implementation_status": "full",
    },
    "P26": {
        "title": "TurboQuant prefill output prealloc",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_LEGACY_P26",
        "default_on": True,
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.p26_prefill_output",
        "lifecycle": "legacy",
        "category": "memory_pool",
        "credit": "Pre-dispatcher legacy patch. Preallocates TQ prefill output buffer to avoid per-step allocation churn.",
        "superseded_by": "PARTIAL — cu_2 half only (upstream `_cu_2` lazy-init guard, #40420-class OOM concern, marker 'if not hasattr(self, \"_cu_2\")' present on dev93+); the output_alloc prealloc (~32 MiB/call) is NOT upstream and still applies. Do NOT retire P26 on a title match.",
        # [Preflight triage 2026-06-11 §2, supersedes the 2026-05-11
        # iron-rule-#11 audit note] The earlier wording implied FULL
        # supersession; byte-level re-check shows upstream only covers
        # the cu_2 half (lazy `_cu_2` hasattr guard). P26's prefill
        # output_alloc prealloc (~32 MiB per call) has no upstream
        # equivalent and remains a live perf win. The PARTIAL prefix
        # guards against a future title-matching retire (iron rule #11
        # anti-pattern). Lifecycle stays "legacy" (architectural).
        "implementation_status": "full",
    },
    "P27": {
        "title": "Qwen3 BEFORE-THINK fallback",
        "tier": "community",
        "family": "reasoning",
        "env_flag": "GENESIS_LEGACY_P27",
        "default_on": True,
        "apply_module": "sndr.engines.vllm.patches.reasoning.p27_reasoning_before_think",
        "lifecycle": "legacy",
        "category": "structured_output",
        "credit": "Pre-dispatcher legacy patch. Falls back to BEFORE-THINK parsing path when Qwen3 model emits tool_call before <think>.",
        # 2026-06-19 (dev148 TIER-1 audit): capped <0.23.0 — the parser reorg
        # #45413/#45588 (MERGED in dev148) deleted/restructured P27's qwen3
        # BEFORE-THINK fallback target; the native engine parser handles the
        # tool_call-before-<think> ordering. Honest cap.
        "applies_to": {"vllm_version_range": (">=0.20.0", "<0.23.0")},
        "implementation_status": "full",
    },
    "P28": {
        "title": "GDN core_attn_out prealloc",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_LEGACY_P28",
        "default_on": True,
        "apply_module": "sndr.engines.vllm.patches.attention.gdn.p28_gdn_core_attn",
        "lifecycle": "legacy",
        "category": "memory_pool",
        "credit": "Pre-dispatcher legacy patch. Preallocates GDN core_attn_out as a layer-persistent buffer + zero()-on-reuse instead of torch.zeros() per-step. Reduces allocator pressure on GDN forward.",
        "conflicts_with": ["PN32", "PN108"],  # PN108 = GDN fused_recurrent prefill switches backend; incompatible with P28 buffer reuse
        "implementation_status": "full",
    },
    "P29": {
        "title": "tool parser IndexError guard",
        "tier": "community",
        "family": "tool_parsing",
        "env_flag": "GENESIS_LEGACY_P29",
        "default_on": True,
        "implementation_status": "marker_only",
        "lifecycle": "legacy",
        "category": "structured_output",
        "credit": "Pre-dispatcher legacy patch. Wraps tool-arg index access so malformed parser state returns empty instead of raising IndexError.",
    },
    "P29_HEAL": {
        "title": "qwen3coder tool parser index heal (P29 companion, fix-wire 2026-06-04)",
        "tier": "community",
        "family": "tool_parsing",
        "env_flag": "GENESIS_ENABLE_P29_QWEN3CODER_INDEX_HEAL",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.tool_parsing.p29_qwen3coder_index_heal",
        "lifecycle": "experimental",
        "category": "structured_output",
        "credit": (
            "Genesis-original 2026-06-04 — extends P29's coverage. "
            "Upstream qwen3coder_tool_parser.py added bounded-index "
            "guards at lines 372 (tool_start_positions lookup) + 619 "
            "(combined emit). But TWO hot sites remain raw: line 287 "
            "(self.current_tool_index += 1 — bare advance with no "
            "symmetric streamed_args_for_tool grow) + line 442 "
            "(self.streamed_args_for_tool[self.current_tool_index] += "
            "'{' — symptom site that raises IndexError when 287 has "
            "advanced past the list). P29_HEAL adds heal-on-advance "
            "(grow list to match new index) + heal-on-write (grow on "
            "demand before indexed write). Closes a class of 500 "
            "errors on streaming tool-call deltas. Defensive — no "
            "happy-path semantics change."
        ),
        "upstream_pr": None,
        # 2026-06-17 (0.23.1 pin-bump): P29_HEAL heals raw index sites in the
        # OLD qwen3coder_tool_parser.py (lines 287/442). That file was DELETED
        # by the #45588 parser reorg — the new vllm/tool_parsers/
        # qwen3_engine_tool_parser.py is a different module with its own bounds
        # guards. The heal premise is gone on >=0.23.0, and streaming multi-
        # tool deltas were verified clean (no IndexError/500) on 0.23.1 live.
        # Cap with the same window as its sibling qwen3coder parser patches
        # (PN56/P64/P61c, all "<0.22.1rc1.dev491") so it stays active only on
        # the older parser layout.
        "applies_to": {
            "vllm_version_range": (">=0.20.0", "<0.22.1rc1.dev491"),
        },
        "implementation_status": "full",
        "composes_with": ["P29"],
    },
    "P31": {
        "title": "MoE router fp32 softmax",
        "tier": "community",
        "family": "moe",
        "env_flag": "GENESIS_LEGACY_P31",
        "default_on": True,
        "apply_module": "sndr.engines.vllm.patches.moe.p31_router_softmax",
        "lifecycle": "legacy",
        "category": "model_correctness",
        "credit": "Pre-dispatcher legacy patch. Upcasts MoE router softmax to fp32 (DeepSeek-V3 pattern, deepseek_v2.py:345 reference). Improves expert routing stability on consumer Ampere.",
        "implementation_status": "full",
    },
    "P32": {
        "title": "TurboQuant cu_2 + synth_seq_lens preallocs (P32/P33)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_LEGACY_P32",
        "default_on": True,
        "implementation_status": "marker_only",
        "lifecycle": "legacy",
        "category": "memory_pool",
        "credit": "Pre-dispatcher legacy patch. Preallocates cu_2 and synth_seq_lens TQ scratch tensors as persistent buffers.",
    },
    "P34": {
        "title": "Mamba zero-collapse deadlock guard",
        "tier": "community",
        "family": "scheduler",
        "env_flag": "GENESIS_LEGACY_P34",
        "default_on": True,
        "apply_module": "sndr.engines.vllm.patches.scheduler.p34_mamba_deadlock_guard",
        "lifecycle": "legacy",
        "category": "stability",
        "credit": "Pre-dispatcher legacy patch. Guards against Mamba state collapse-to-zero deadlock when delta is exactly zero on hybrid models.",
        "implementation_status": "full",
    },
    "P36": {
        "title": "TurboQuant shared decode buffers — RETIRED 2026-06-11",
        "tier": "community",
        "family": "kernels",
        "env_flag": "GENESIS_LEGACY_P36",
        "default_on": False,
        "apply_module": "sndr.engines.vllm._archive.p36_tq_shared_decode_buffers",
        "lifecycle": "retired",
        "category": "memory_pool",
        "credit": "Pre-dispatcher legacy patch. Shared decode-stage scratch buffers across TQ layers to amortize allocation. RETIRED — upstream WorkspaceManager provides the same sharing natively.",
        # [Retire 2026-06-11, preflight residual triage §3, byte-verified]
        # Upstream WorkspaceManager.get_simultaneous is native at pristine
        # v1/worker/workspace.py:92 with the identical shared-scratch
        # contract (multiple shapes/dtypes carved from one allocation,
        # 256-byte aligned, sequential-layer reuse invariant), consumed by
        # turboquant_attn.py:747 (k_buf/v_buf) and :879. The patched
        # 3x register_buffer site (_tq_mid_o_buf/_tq_output_buf/_tq_lse_buf
        # in attention layer init) no longer exists — those names now
        # live as lazy holder attributes in triton_turboquant_decode.py
        # (548-608), a different mechanism. Boot already skips P36 via the
        # pr40798 probe (deduped boot line 149). gh check 2026-06-11:
        # vllm#40798 itself is CLOSED-unmerged — the equivalent landed via
        # the WorkspaceManager refactor, NOT that PR; note kept audit-
        # accurate. POOL_TQ_DECODE_SHARED consumer check 2026-06-11: the
        # constant is defined in sndr/runtime/persistent_buffer_registry.py
        # (stays) and consumed only by this module + its unit test —
        # module ARCHIVED, not deleted.
        "superseded_by": "upstream-native WorkspaceManager.get_simultaneous (pristine v1/worker/workspace.py:92; consumed at turboquant_attn.py:747/879) — identical shared-scratch invariant; patched register_buffer site gone. vllm#40798 CLOSED-unmerged (gh-verified 2026-06-11); equivalence landed via the WorkspaceManager refactor",
        "vllm_version_range": "<0.22.1rc1.dev259",  # supersession byte-verified on this pin's pristine tree
        "implementation_status": "full",
    },
    "P37": {
        "title": "MoE intermediate cache pool (opt-in)",
        "tier": "community",
        "family": "moe",
        # Audit P1 fix 2026-05-05 (genesis_local_consistency_audit + runtime audit):
        # registry was `GENESIS_ENABLE_P37_MOE_INTER_CACHE` but wiring code,
        # apply_all docstring, AND launch scripts all use `GENESIS_ENABLE_P37`.
        # env_flag_guard was reporting it as suspicious typo. Aligned to short form.
        "env_flag": "GENESIS_ENABLE_P37",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.moe.p37_moe_intermediate_cache",
        "lifecycle": "legacy",
        "category": "memory_pool",
        "credit": "Pre-dispatcher legacy patch. Opt-in pool for MoE intermediate activations. noonghunna's club-3090 long-text recipe ships with this enabled.",
        "implementation_status": "full",
    },
    "P38": {
        "title": "TQ _continuation_prefill persistent workspace",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_LEGACY_P38",
        "default_on": True,
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.p38_tq_continuation_memory",
        "lifecycle": "legacy",
        "category": "memory_pool",
        "credit": "Pre-dispatcher legacy patch. Persistent workspace tensor for TQ continuation-prefill, addresses VolandBerlioz's OOM site at turboquant_attn.py. Companion: P38B (compile-safe in-source hook, see PATCH_REGISTRY).",
        "implementation_status": "full",
    },
    "P39a": {
        "title": "FLA chunk_scaled_dot_kkt persistent A pool",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_LEGACY_P39A",
        "default_on": True,
        "apply_module": "sndr.engines.vllm.patches.attention.gdn.p39a_fla_kkt_buffer",
        "lifecycle": "legacy",
        "category": "memory_pool",
        "credit": "Pre-dispatcher legacy patch. Persistent pool for FLA chunk_scaled_dot_kkt's A matrix to avoid per-step allocation in GDN backward.",
        "implementation_status": "full",
    },
    "P40": {
        "title": "TurboQuant GQA-grouped decode stage1 (opt-in)",
        "tier": "community",
        "family": "attention.turboquant",
        # Audit P1 fix 2026-05-05: same class as P37 — registry was
        # `GENESIS_ENABLE_P40_GQA_GROUPED_DECODE` but wiring/kernel/scripts
        # use `GENESIS_ENABLE_P40`. compat/presets.py also used yet another
        # variant (`GENESIS_ENABLE_P40_TQ_GROUPED_DECODE`). Aligned to short form.
        "env_flag": "GENESIS_ENABLE_P40",
        "default_on": False,
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.p40_tq_grouped_decode",
        "lifecycle": "legacy",
        "category": "kernel_perf",
        "credit": "Pre-dispatcher legacy patch (vllm#40792 backport candidate). Opt-in GQA-grouped TQ decode stage1 kernel. Welch t-test on 2x A5000 single-stream: not significant (p=0.284 vs baseline 183 TPS) — kept opt-in pending Blackwell retest.",
        "implementation_status": "full",
    },
    "P44": {
        "title": "TQ mixed-batch attn_out pool",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_LEGACY_P44",
        "default_on": True,
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.p44_tq_mixed_attn_out",
        "lifecycle": "legacy",
        "category": "memory_pool",
        "credit": "Pre-dispatcher legacy patch. Pool for TQ attn_out tensor under mixed prefill+decode batches.",
        "implementation_status": "full",
    },
    "P46": {
        "title": "GDN gating buffer pool",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_LEGACY_P46",
        "default_on": True,
        "apply_module": "sndr.engines.vllm.patches.attention.gdn.p46_gdn_gating_buffers",
        "lifecycle": "legacy",
        "category": "memory_pool",
        "credit": "Pre-dispatcher legacy patch. Pool for GDN gating tensor to avoid per-layer allocation.",
        "implementation_status": "full",
    },
    "P51": {
        "title": "TQ-active runtime layer-level guard",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_LEGACY_P51",
        "default_on": True,
        "implementation_status": "marker_only",
        "lifecycle": "legacy",
        "category": "quantization",
        "credit": "Pre-dispatcher library patch. Runtime layer-level TQ-active detection in kernels/dequant_buffer.py — skips TQ preallocs on layers where TQ is not active. No env toggle (defensive runtime check). Companion to model_detect's config-level TQ check.",
    },
    "P102": {
        "title": "Unified spec-decode metadata + disagreement tracker (TRT-LLM style)",
        "tier": "community",
        "family": "kv_cache",
        "env_flag": "GENESIS_ENABLE_P102",
        "default_on": False,
        # Phase 3B.2 (2026-05-22): impl_status changed from
        # 'marker_only' to 'placeholder'. 'marker_only' implies the
        # registry row documents an already-active behavior (env
        # consumed elsewhere). P102 is the opposite — it's a future
        # planned feature (TRT-LLM-style spec_meta + disagreement
        # tracker) that has no on-disk wiring yet. 'placeholder' is
        # the canonical semantic for "entry exists but apply path TBD".
        "implementation_status": "placeholder",
        "category": "spec_decode",
        "credit": "Genesis-original (Sander 2026-04-29). First-class spec_meta module that wraps spec-decode metadata into a unified object + tracks predicate disagreement (e.g. should_dispatch_p67 disagreements between proposer and verify paths). Diagnostic-only opt-in observability layer; emits log lines when divergence detected. Future hook for unified spec-decode dispatcher refactor.",
        "upstream_pr": None,
        "lifecycle": "experimental",
    },
    "PN60": {
        "title": "Quant arg vs config.json validator (preflight DX)",
        "tier": "community",
        "family": "compile_safety",
        "env_flag": "GENESIS_ENABLE_PN60",
        "default_on": True,
        "implementation_status": "marker_only",  # advisory validation, registered as recommendation
        "category": "stability",
        "credit": "Genesis-original 2026-05-05 (apnar club-3090#51 finding). Cross-checks operator's --quantization CLI arg against the model's config.json:quantization_config.quant_method BEFORE vLLM loads. Emits one-line remediation hint instead of a 30-line pydantic ValidationError. Doctor extension; runs at preflight, no monkey-patch.",
        "upstream_pr": None,
        "applies_to": {},
        "lifecycle": "legacy",
    },
    "PN61": {
        "title": "qwen3_vl loader KeyError → text-only auto-fallback (vllm-loader guard)",
        "tier": "community",
        "family": "loader",
        "env_flag": "GENESIS_ENABLE_PN61",
        "default_on": False,
        "category": "stability",
        "credit": "Genesis-original 2026-05-05 (apnar club-3090#51 NVFP4 finding). Catches `KeyError: 'blocks.0.attn.proj.weight'` in qwen3_vl.load_weights when an NVFP4 quant strips the ViT tower; emits WARN + auto-sets language_model_only=True instead of crashing. Same defensive pattern as P29 IndexError guard.",
        "upstream_pr": None,
        "applies_to": {"model_class": ["qwen3_vl"]},
        "apply_module": "sndr.engines.vllm.patches.loader.pn61_qwen3_vl_keyerror_guard",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN62": {
        "title": "Text-only ViT scratch skip via skip_mm_profiling flip (3-5 GiB save)",
        "tier": "community",
        "family": "multimodal",
        "env_flag": "GENESIS_ENABLE_PN62",
        "default_on": False,
        "category": "memory_savings",
        "credit": "Genesis-original 2026-05-05 (apnar club-3090#51); Wave 6 real hook 2026-05-09. When mm_limits_all_zero AND --language-model-only, PN62 wraps GPUModelRunner.profile_run and flips MultiModalConfig.skip_mm_profiling=True before profile_run executes, so vllm dev93's native short-circuit at the encoder profiling branch fires. Saves ~3-5 GiB ViT scratch on qwen3_vl + NVFP4 single-card boot. Real hook landed (Wave 6) — replaces the prior marker-only scaffold. Sister to PN35 (text-only inputs_embeds skip, vllm#35975 merged).",
        "upstream_pr": None,
        "applies_to": {"model_class": ["qwen3_vl"]},
        "implementation_status": "full",
        "apply_module": "sndr.engines.vllm.patches.multimodal.pn62_text_only_vit_skip",
        "lifecycle": "experimental",  # awaiting cross-rig live validation
    },
    "PN63": {
        "title": "fp8_e5m2 advisory for consumer Blackwell (gpu_profile recommendation)",
        "tier": "community",
        "family": "compile_safety",
        "env_flag": "GENESIS_ENABLE_PN63",
        "default_on": True,
        "implementation_status": "marker_only",  # advisory only, no patch site
        "category": "stability",
        "credit": "Genesis-original 2026-05-05 (apnar club-3090#51 empirical). Adds a per-GPU advisory entry to gpu_profile.PATCH_RECOMMENDATIONS that recommends --kv-cache-dtype fp8_e5m2 over fp8_e4m3 on consumer Blackwell (sm 12.0) until vLLM e4m3 codepath matures. Suggest-only; operator passes via CLI.",
        "upstream_pr": None,
        "lifecycle": "legacy",
    },
    "PN64": {
        "title": "Marlin MoE per-SM tuning placeholder for SM 12.0 (consumer Blackwell)",
        "tier": "community",
        "family": "kernels",
        "env_flag": "GENESIS_ENABLE_PN64",
        "default_on": False,
        "implementation_status": "placeholder",  # SM 12.0 stub, awaiting empirical Blackwell measurement
        "category": "kernel_perf",
        "credit": "Genesis-original 2026-05-05 (apnar club-3090#51 — boot log shows `[Genesis] skipped: P17/P18 Marlin MoE per-SM tuning — no tuning entry for SM (12, 0)`). PN64 adds a placeholder entry copying SM (9, 0) Hopper config until empirical sweep data lands from sm_120. Author-blocked: needs real 5090 sweep — solicit from apnar/jhsmith409.",
        "upstream_pr": None,
        "applies_to": {},
        # Phase 3B.3 (2026-05-22): lifecycle changed from 'experimental'
        # to 'research'. 'experimental' implies the patch can plausibly
        # be tried on currently-supported hardware; PN64 is a
        # hardware-forward placeholder for SM 12.0 (Blackwell) which
        # no Genesis-validated rig has access to. 'research' is the
        # honest tag — opt-in, not for PROD, future hardware.
        "lifecycle": "research",
    },
    "PN65": {
        "title": "Genesis structured API access log middleware (operator UX)",
        "tier": "community",
        "family": "middleware",
        "env_flag": "GENESIS_ENABLE_PN65",
        "default_on": False,
        "category": "request_middleware",
        "credit": "Genesis-original 2026-05-05 (Sander request: 'the API log is bare — needs polish too'). Replaces uvicorn's bare `INFO: 127.0.0.1:45116 - GET /v1/models 401 Unauthorized` with `[Genesis-API] 200  POST /v1/chat/completions  34ms  prompt=46t  completion=400t  tools=1  client=127.0.0.1`. Suppresses /health polling by default (GENESIS_PN65_LOG_HEALTH=1 to include). Status-aware level (2xx INFO / 4xx WARN / 5xx ERROR + exception type).",
        "upstream_pr": None,
        "applies_to": {},
        "apply_module": "sndr.engines.vllm.patches.middleware.pn65_access_log",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN66": {
        "title": "Multiturn </think> leak fix in DelegatingParser (vllm#41696 backport)",
        "tier": "community",
        "family": "reasoning",
        "env_flag": "GENESIS_ENABLE_PN66",
        "default_on": False,
        "category": "structured_output",
        "credit": "Backport of vllm#41696 (panpan0000, OPEN as of 2026-05-05). Removes the buggy `prompt_reasoning_checked` short-circuit in `vllm.parser.abstract_parser.DelegatingParser.parse_delta` that walked the FULL prompt looking for `</think>` and prematurely set `reasoning_ended=True` from a previous turn's `</think>`. Defensive backport for multi-turn DSML/Hermes/Qwen3 chat clients sending full history. Original report: DeepSeek V3.2 reasoning users.",
        "upstream_pr": 41696,
        "upstream_pr_relationship": "backport",
        # 2026-06-18 (dev148 full-patch audit): #45588 reorganized the parser
        # into vllm/parser/abstract_parser.py — PN66's DelegatingParser anchor
        # is gone (DRIFT skip on 0.23.x). A LIVE multiturn-reasoning probe on
        # dev148 showed NO </think> leak (the new engine parser handles the
        # reasoning+tool compose), and #41696 is CLOSED-unmerged. Capped
        # <0.23.0 so the registry is honest (correctly inert, not silently
        # drift-skipped). Re-anchor to the new parser only if a </think> leak
        # is ever observed live on 0.23.x+.
        "applies_to": {"vllm_version_range": (">=0.20.0", "<0.23.0")},
        "apply_module": "sndr.engines.vllm.patches.reasoning.pn66_multiturn_think_leak",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN67": {
        "title": "thinking_token_budget inverted-bool fix (vllm#41674 backport, 1-line)",
        "tier": "community",
        "family": "worker",
        "env_flag": "GENESIS_ENABLE_PN67",
        "default_on": False,
        "category": "stability",
        "credit": "Backport of vllm#41674 (JasonKeyiL). Single-token fix in `vllm/v1/worker/gpu_input_batch.py:879` — removes `not` from `or not thinking_budget_tracks_reqs`. Bug: thinking_token_budget was silently ignored for any request without penalty parameters. NULL on Genesis PROD (we don't enable thinking_token_budget); defensive for users who experiment with it. Trivial backport, zero risk. Retired 2026-05-22 (Phase 3D) — upstream PR merged 2026-05-15 at commit bf610c2f56764e1b30bc6065f4ceace3d6e59036, which IS our dev371 canonical pin baseline. Genesis PN67 has an apply-time pre-flight check that auto-skips when the anchor `or not thinking_budget_tracks_reqs` is gone, so the patch is functionally inert on dev371 and later; this retire makes the registry state match runtime reality.",
        "upstream_pr": 41674,
        "upstream_pr_relationship": "backport",
        "vllm_version_range": "<0.20.2rc1.dev371",
        "superseded_by": "vllm#41674 (merged 2026-05-15 at commit bf610c2f56764e1b30bc6065f4ceace3d6e59036 — the dev371 canonical pin baseline; functionally identical 1-line removal of `not` from gpu_input_batch.py thinking_budget_tracks_reqs condition; Genesis 3-line in-place comment is the only delta, no behavioral difference)",
        "applies_to": {},
        "apply_module": "sndr.engines.vllm._archive.pn67_thinking_budget_inverted_bool",
        "lifecycle": "retired",
        "implementation_status": "retired",
    },
    "PN70": {
        "title": "Tool schema subset filter (combined `anyOf` xgrammar-clean) — companion to P68 v7.72.1",
        "tier": "community",
        "family": "serving",
        "env_flag": "GENESIS_ENABLE_PN70_TOOL_SCHEMA_FILTER",
        "default_on": False,
        "category": "structured_output",
        "credit": "Genesis-original — implements lexhoefsloot's option-3 fix for noonghunna/club-3090#57. Wraps `vllm.tool_parsers.utils._get_json_schema_from_tools` and filters tools containing xgrammar-unsupported JSON Schema keys (patternProperties / propertyNames / $ref / oneOf / etc.) BEFORE the combined `anyOf` is built and handed to xgrammar. Companion to P68's option-1 skip: where P68 refuses to upgrade tool_choice on dirty catalogs, PN70 keeps the upgrade and filters dirty tools out of grammar enforcement (model can still SEE all tools in context but grammar restricts callable subset). Reuses P68's `_scan_schema_for_unsupported_key` so the unsupported-key set is single-sourced. Off by default; enable per workload.",
        "applies_to": {},
        "composes_with": ["P68"],
        "apply_module": "sndr.engines.vllm.patches.serving.pn70_tool_schema_subset_filter",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },

    # ── Gemma 4 family (2026-05-17) ────────────────────────────────────
    # 21 patches addressing every known Gemma 4 issue on Ampere SM 8.6:
    # FP8 double-scale (#39407), Marlin K-divisibility (#40354),
    # non-causal attention wall (#40382), AWQ MoE keys (#40886),
    # DFlash backend autoselect (#42069), KV-projection optimization
    # (#41944), SWA/global prefill chunker (#39914), FP8 e4nv guard
    # (#41014), per-token-head KV asymmetry (#40388 + WIP #40391),
    # tool-call-parser pad-token (#39392), vision-tower text-only,
    # FP16 overflow (#40124), perf kernels (fused RMSNorm + softcap),
    # FULL_AND_PIECEWISE cudagraph mode parallel to PN125.
    # Family: gemma4, location: sndr/engines/vllm/patches/model_compat/gemma4/
    "G4_01": {
        "title": "Refuse FP8_BLOCK Gemma 4 checkpoint on Ampere SM 8.6 (closes vllm#39407 user pain)",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_01_GEMMA4_FP8_BLOCK_GUARD",
        "default_on": True,
        "category": "stability",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.model_compat.gemma4.g4_01_gemma4_ampere_fp8_block_guard",
        "lifecycle": "stable",
        "stable_kind": "runtime-hook",
        "production_validated_pins": [
            ("v12.0.0", "0.20.2rc1.dev338+gbf0d2dc6d"),
            ("v12.0.0", "0.20.2rc1.dev371+gbf610c2f5"),
        ],
        "credit": "Refuses the known-broken FP8_BLOCK + Ampere combo at process_weights_after_loading. Saves operators a 30-min cold-boot-to-garbage debug cycle.",
        "upstream_pr": "https://github.com/vllm-project/vllm/issues/39407",
        "requires_patches": [],
        # Phase 5.3.C (2026-05-22): removed overloaded `superseded_by:
        # ["G4_07"]`. G4_07 is an experimental alternative — not a
        # supersessor in the canonical retire-on-merged-upstream sense.
        # Mutual exclusion is fully expressed by `conflicts_with`.
        "conflicts_with": ["G4_07"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_02": {
        "title": "Refuse Marlin MoE with K%64≠0 on Gemma 4 26B-A4B (closes vllm#40354 user pain)",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_02_GEMMA4_MARLIN_KDIM_GUARD",
        "default_on": True,
        "category": "stability",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.model_compat.gemma4.g4_02_gemma4_ampere_marlin_kdim_guard",
        "lifecycle": "stable",
        "stable_kind": "runtime-hook",
        "production_validated_pins": [
            ("v12.0.0", "0.20.2rc1.dev338+gbf0d2dc6d"),
            ("v12.0.0", "0.20.2rc1.dev371+gbf610c2f5"),
        ],
        "credit": "Refuses 26B-A4B + Ampere Marlin combo at apply_weights time. K=352 (704/2 at TP=2) fails Marlin's min_thread_k=64 divisibility.",
        "upstream_pr": "https://github.com/vllm-project/vllm/issues/40354",
        "requires_patches": [],
        # Phase 5.3.C (2026-05-22): see G4_01 for rationale. G4_08 is an
        # experimental alternative; `conflicts_with` carries the
        # mutual-exclusion contract.
        "conflicts_with": ["G4_08"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration"]},
    },
    "G4_03": {
        "title": "Refuse non-causal drafter (Eagle3/DFlash) on Ampere SM 8.6 (closes vllm#40382 user pain)",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_03_GEMMA4_NON_CAUSAL_DRAFTER_GUARD",
        "default_on": True,
        "category": "stability",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.model_compat.gemma4.g4_03_gemma4_ampere_non_causal_drafter_guard",
        "lifecycle": "stable",
        "stable_kind": "runtime-hook",
        "production_validated_pins": [
            ("v12.0.0", "0.20.2rc1.dev338+gbf0d2dc6d"),
            ("v12.0.0", "0.20.2rc1.dev371+gbf610c2f5"),
        ],
        "credit": "Refuses Eagle3/DFlash drafter on Gemma 4 + Ampere — no Ampere backend supports head_dim=256 + non-causal.",
        "upstream_pr": "https://github.com/vllm-project/vllm/issues/40382",
        "requires_patches": [],
        # Phase 5.3.C (2026-05-22): see G4_01 for rationale. G4_10 is an
        # experimental alternative; `conflicts_with` carries the
        # mutual-exclusion contract.
        "conflicts_with": ["G4_10"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_04": {
        "title": "Gemma 4 AWQ MoE keys remap (vendors vllm#40886)",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_04_GEMMA4_AWQ_MOE_KEYS_REMAP",
        "default_on": True,
        "category": "loader",
        "implementation_status": "full",
        "source": "vendor_backport",
        "apply_module": "sndr.engines.vllm.patches.model_compat.gemma4.g4_04_gemma4_awq_moe_keys_remap",
        "lifecycle": "stable",
        "stable_kind": "text-patch",
        "credit": (
            "AUDIT NOTE 2026-06-10 (call-site drift sweep): the wrap "
            "lands on grouped_topk_router.py correctly, but Qwen3.6-A3B "
            "uses the NON-grouped FusedTopKRouter (qwen3_next.py FusedMoE "
            "passes no use_grouped_topk) — the wrapped fn never runs for "
            "this model. Harmless but the 'applied' status is misleading "
            "on 35B; effective only on grouped-topk MoE models. "

            "Vendors vllm#40886 — AWQ MoE keys remap for Gemma 4 26B-A4B "
            "checkpoint compatibility. "
            "Cross-file supersession watchlist (PIN.R-DRIFT-MARKER-AUDIT "
            "2026-05-24): patch's `upstream_merged_markers` watch only "
            "the target `vllm/model_executor/models/gemma4.py`, but the "
            "upstream fix could plausibly land in a shared MoE / AWQ "
            "key-mapping abstraction. At next pin bump, BEFORE concluding "
            "no-supersession, grep for `moe.gate_up_proj_packed` and "
            "`moe.down_proj_packed` across "
            "`vllm/model_executor/layers/quantization/compressed_tensors/`, "
            "`vllm/model_executor/layers/quantization/awq*.py`, and "
            "`vllm/model_executor/layers/quantization/utils/marlin_utils*.py`. "
            "Same blind-spot pattern that PIN.R-G4_05-RETIRE.1 (2026-05-24) "
            "uncovered for G4_05 — recorded preventively here to make the "
            "next parity pass actionable."
        ),
        "upstream_pr": "https://github.com/vllm-project/vllm/pull/40886",
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration"]},
    },
    "G4_05": {
        "title": "DFlash drafter backend autoselect (retired — superseded by vllm#39930)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_G4_05_GEMMA4_DFLASH_BACKEND_AUTOSELECT",
        "default_on": False,
        "category": "loader",
        "implementation_status": "full",
        "source": "vendor_backport",
        "apply_module": "sndr.engines.vllm._archive.g4_05_dflash_backend_autoselect",
        "lifecycle": "retired",
        "credit": (
            "Originally vendored from vllm#42069 (mikeumus, OPEN at backport "
            "time; later CLOSED 2026-05-19 by maintainer @MatthewBonanni "
            "with comment \"This is already fixed by "
            "https://github.com/vllm-project/vllm/pull/39930\"). "
            "PIN.R-DEEP-PARITY.2 (2026-05-24) verdict: "
            "RETIRE_CANDIDATE_AFTER_PARITY. Deep-parity established that "
            "PR #39930 (merged 2026-04-28 at commit "
            "fd74c90d9c3b5c35308f1f0ab308469235fa5277) adds "
            "SpecDecodeConfig.attention_backend (default None=autoselect) "
            "plus base-class autoselect path in "
            "LLMBaseProposer._create_draft_vllm_config (verbatim: \"we "
            "always independently autoselect unless explicitly specified "
            "in the speculative config\"). Genesis G4_05's backend=None "
            "insertion in DFlash._create_draft_vllm_config is functionally "
            "REDUNDANT on all Genesis-supported pins (the first allowlist "
            "pin 0.20.1rc1.dev16+g7a1eb8ac2 was committed 2h after #39930 "
            "merged); in the explicit "
            "--speculative-config.attention_backend=TRITON_ATTN edge case "
            "G4_05 INVERTS operator intent against upstream's documented "
            "fail-loudly design. No production Genesis preset uses that "
            "override (grep clean). Sister patch PN9 retired for the same "
            "vllm#39930 supersession at the matching boundary. "
            "Cross-file drift-marker blind spot (G4_05 watched dflash.py "
            "for `backend=None,` while the supersession landed in "
            "llm_base_proposer.py) flagged for a separate "
            "PIN.R-DRIFT-MARKER-AUDIT tooling phase."
        ),
        "upstream_pr": 39930,
        "upstream_pr_relationship": "backport",
        "superseded_by": (
            "vllm#39930 (merged 2026-04-28 at commit "
            "fd74c90d9c3b5c35308f1f0ab308469235fa5277, in dev16+ — "
            "first known-good Genesis allowlist pin 0.20.1rc1.dev16+g7a1eb8ac2 "
            "was committed 2026-04-28T04:52:54Z, ~2h after the merge). "
            "Upstream adds full SpecDecodeConfig.attention_backend field + "
            "base-class autoselect path that obsoletes Genesis G4_05's "
            "DFlash-override-layer insertion. Matches PN9's earlier retire "
            "for the same upstream PR; G4_05 was a parallel "
            "vendor-backport at the same root cause."
        ),
        "vllm_version_range": "<0.20.2rc1.dev9+g01d4d1ad3",
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_06": {
        "title": "v_head_size=0 for k_eq_v attention layers (vendors vllm#41944)",
        "tier": "community",
        "family": "kv_cache",
        "env_flag": "GENESIS_ENABLE_G4_06_GEMMA4_KV_PROJ_V0",
        "default_on": False,
        "category": "perf_hotfix",
        "implementation_status": "partial",
        "source": "vendor_backport",
        "apply_module": "sndr.engines.vllm.patches.kv_cache.g4_06_kv_proj_v_head_size_zero",
        "lifecycle": "experimental",
        "credit": (
            "Vendors __init__ portion of vllm#41944. ~3% memory savings on "
            "V-slot weights for global attention layers. "
            "Cross-file supersession watchlist (PIN.R-DRIFT-MARKER-AUDIT "
            "2026-05-24): patch's `upstream_merged_markers` watch only "
            "the target `vllm/model_executor/models/gemma4.py`, but the "
            "k_eq_v / v_head_size optimization could plausibly be "
            "promoted to a generic attention-layer abstraction during "
            "upstream review. At next pin bump, BEFORE concluding "
            "no-supersession, grep for `v_head_size=0` and "
            "`use_k_eq_v` across `vllm/attention/layer.py`, "
            "`vllm/model_executor/layers/attention.py`, and any new "
            "AttentionConfig surface in `vllm/config/`. Same blind-spot "
            "pattern that PIN.R-G4_05-RETIRE.1 (2026-05-24) uncovered "
            "for G4_05 — recorded preventively here to make the next "
            "parity pass actionable."
        ),
        "upstream_pr": "https://github.com/vllm-project/vllm/pull/41944",
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_07": {
        "title": "FP8_BLOCK double-scale fix — custom quant config (closes vllm#39407)",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_07_GEMMA4_FP8_BLOCK_FIX",
        "default_on": False,
        "category": "kernel",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.model_compat.gemma4.g4_07_gemma4_fp8_block_double_scale_fix",
        "lifecycle": "experimental",
        "credit": "Registers gemma4_fp8_block_fix quantization config — bypasses double-scale bug by skipping the second activation quantization.",
        "upstream_pr": "https://github.com/vllm-project/vllm/issues/39407",
        "requires_patches": [],
        "conflicts_with": ["G4_01"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_08": {
        "title": "Marlin K-pad Triton MoE fallback (closes vllm#40354)",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_08_MARLIN_KDIM_PAD",
        "default_on": False,
        "category": "kernel",
        "implementation_status": "partial",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.model_compat.gemma4.g4_08_gemma4_marlin_kdim_pad_fallback",
        "lifecycle": "experimental",
        "credit": "Genesis Triton zero-pad MoE GEMM kernel routes K%64≠0 cases. Unblocks Gemma 4 26B-A4B at TP=2 on Ampere SM 8.6.",
        "upstream_pr": "https://github.com/vllm-project/vllm/issues/40354",
        "requires_patches": [],
        "conflicts_with": ["G4_02"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration"]},
    },
    "G4_09": {
        "title": "SWA→global prefill chunker (workaround vllm#39914 engine hang)",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_09_GEMMA4_SWA_PREFILL_CHUNKER",
        "default_on": True,
        "category": "stability",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.model_compat.gemma4.g4_09_gemma4_swa_global_prefill_chunker",
        "lifecycle": "stable",
        "stable_kind": "runtime-hook",
        "production_validated_pins": [
            ("v12.0.0", "0.20.2rc1.dev338+gbf0d2dc6d"),
            ("v12.0.0", "0.20.2rc1.dev371+gbf610c2f5"),
        ],
        "credit": "Clamps scheduler.max_num_batched_tokens to 2048 + forces enable_chunked_prefill=True on Gemma 4 — bypasses #39914 engine hang at prefill>4K.",
        "upstream_pr": "https://github.com/vllm-project/vllm/issues/39914",
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_10": {
        "title": "Gemma-4 non-causal drafter enablement on Ampere (stock TRITON_ATTN; vllm#40382 closed on dev491)",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_10_GEMMA4_AMPERE_NON_CAUSAL_BACKEND",
        "default_on": False,
        "category": "loader",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.model_compat.gemma4.g4_10_gemma4_ampere_non_causal_attn_backend",
        "lifecycle": "experimental",
        "credit": "Enablement guard: lifts the G4_03 refusal so an EAGLE-3/DFlash (non-causal) drafter can run on Ampere SM 8.6 + Gemma 4 via STOCK TRITON_ATTN (verified dev491: supports_non_causal=True, supports_head_size>=32 covers 256+512). Per-layer drafter routing owned by g4_71b (256) + g4_75 (512). The old bespoke head_dim=256 Triton kernel + custom backend were RETIRED 2026-06-16 (no-op registration on dev491 + redundant with stock TRITON_ATTN + untested/256-only). vllm#39930 (independent drafter backend selection) makes the per-drafter route legitimate.",
        "upstream_pr": "https://github.com/vllm-project/vllm/issues/40382",
        "requires_patches": [],
        "conflicts_with": ["G4_03"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_11": {
        "title": "Gemma 4 enhanced chat template install",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_11_GEMMA4_CHAT_TEMPLATE_INSTALL",
        "default_on": True,
        "category": "loader",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.model_compat.gemma4.g4_11_gemma4_chat_template_install",
        "lifecycle": "stable",
        "stable_kind": "runtime-hook",
        "production_validated_pins": [
            ("v12.0.0", "0.20.2rc1.dev338+gbf0d2dc6d"),
            ("v12.0.0", "0.20.2rc1.dev371+gbf610c2f5"),
        ],
        "credit": "Writes enhanced gemma4.jinja chat template to /tmp/genesis/chat_templates/ — supports tool calls + system role + thinking blocks.",
        "upstream_pr": None,
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_12": {
        "title": "Refuse FP8 e4nv on Ampere SM 8.6 (closes vllm#41014)",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_12_GEMMA4_FP8_E4NV_GUARD",
        "default_on": True,
        "category": "stability",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.model_compat.gemma4.g4_12_gemma4_fp8_e4nv_ampere_guard",
        "lifecycle": "stable",
        "stable_kind": "runtime-hook",
        "production_validated_pins": [
            ("v12.0.0", "0.20.2rc1.dev338+gbf0d2dc6d"),
            ("v12.0.0", "0.20.2rc1.dev371+gbf610c2f5"),
        ],
        "credit": (
            "Refuses FP8 e4nv Gemma 4 checkpoint on Ampere SM 8.6 at "
            "config-verify time. Ampere tensor cores don't support e4nv "
            "natively. Audit 2026-05-24 (PIN.R-REFS-CLOSED-PR.R): "
            "upstream issue #41014 verified HTTP 200 + state=open (title "
            "matches patch purpose); KEEP — G4_12 is the defensive guard "
            "that closes the bug locally while upstream issue remains "
            "open. No deep-parity required (genesis_original guard, not "
            "a backport)."
        ),
        "upstream_pr": "https://github.com/vllm-project/vllm/issues/41014",
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_13": {
        "title": "Refuse asymmetric per-layer KV-head config (closes vllm#40388 silent corruption)",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_13_GEMMA4_PER_TOKEN_HEAD_KV_GUARD",
        "default_on": True,
        "category": "stability",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.model_compat.gemma4.g4_13_gemma4_per_token_head_kv_guard",
        "lifecycle": "stable",
        "stable_kind": "runtime-hook",
        "production_validated_pins": [
            ("v12.0.0", "0.20.2rc1.dev338+gbf0d2dc6d"),
            ("v12.0.0", "0.20.2rc1.dev371+gbf610c2f5"),
        ],
        "credit": "Refuses 26B-A4B (sliding=8 KV-heads, full=2 KV-heads) at config-verify. Prevents silent quality regression from KV page-size mismatch.",
        "upstream_pr": "https://github.com/vllm-project/vllm/issues/40388",
        "requires_patches": [],
        # Phase 5.3.C (2026-05-22): see G4_01 for rationale. G4_18 is an
        # experimental alternative; `conflicts_with` carries the
        # mutual-exclusion contract.
        "conflicts_with": ["G4_18"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration"]},
    },
    "G4_14": {
        "title": "Gemma 4 tool-call parser pad-token strip (closes vllm#39392)",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_14_GEMMA4_TOOL_CALL_PARSER_PAD",
        "default_on": True,
        "category": "stability",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.model_compat.gemma4.g4_14_gemma4_tool_call_parser_pad_token",
        "lifecycle": "stable",
        "stable_kind": "runtime-hook",
        "production_validated_pins": [
            ("v12.0.0", "0.20.2rc1.dev338+gbf0d2dc6d"),
            ("v12.0.0", "0.20.2rc1.dev371+gbf610c2f5"),
        ],
        "credit": "Strips <pad>/<eos>/turn-boundary control tokens from streaming tool-call JSON deltas. Fixes malformed function.arguments JSON.",
        "upstream_pr": "https://github.com/vllm-project/vllm/issues/39392",
        "requires_patches": [],
        "conflicts_with": [],
        # 2026-06-18 (0.23.1 migration, iron-rule-#11 deep-diff): upstream
        # tool-parser reorg #45588 DELETED Gemma4ToolParser; the new
        # Gemma4EngineToolParser (vllm/tool_parsers/gemma4_engine_tool_parser.py)
        # + gemma4_utils.parse_tool_calls is a full rewrite — it decodes with
        # skip_special_tokens=False and extracts args via a structured
        # vllm.parser.gemma4._parse_gemma4_args pass, NOT the old raw-token
        # streaming JSON path. The #39392 pad-leak mode ("holds raw token IDs
        # for partial-JSON parsing and the pad sneaks through") no longer exists
        # in the architecture G4_14 wraps; G4_14._find_gemma_tool_parser()
        # targets only the deleted class -> graceful skip on 0.23.1. Capped
        # <0.23.0 so the registry is honest (correctly inert, not silently
        # broken), consistent with PN30/PN56/P64. #39392 is still OPEN upstream:
        # if a live gemma4 tool-call repro shows the pad leak on
        # Gemma4EngineToolParser, redesign against the new class with a failing
        # test FIRST, then lift the cap.
        "applies_to": {
            "model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"],
            "vllm_version_range": (">=0.20.0", "<0.23.0"),
        },
    },
    "G4_15": {
        "title": "Fused RMSNorm Triton kernels for Gemma 4 (ported from SGLang)",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_15_GEMMA4_FUSED_RMSNORM",
        "default_on": False,
        "category": "kernel",
        "implementation_status": "partial",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.model_compat.gemma4.g4_15_gemma4_fused_rmsnorm_route",
        "lifecycle": "experimental",
        "credit": "Triton kernels port + integration hooks for Gemma 4 RMSNorm fusion (Q/K/V per-head + residual+scalar + dual-norm MoE reduction). Expected +5-10% TPS on decode at low concurrency. SM 8.6 budget-tuned.",
        "upstream_pr": None,
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_84": {
        "title": "MoE-geometry advisor + wna16 tuned-config provider (generic)",
        "tier": "community",
        "family": "moe",
        "env_flag": "GENESIS_ENABLE_G4_84_MOE_GEOMETRY_ADVISOR",
        "default_on": True,
        "category": "perf_hotfix",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.moe.g4_84_moe_geometry_advisor",
        "lifecycle": "experimental",
        "credit": (
            "Generic fix for the structural int4-MoE-Marlin-ineligible trap "
            "(intermediate_per_partition %% max(64,group_size) != 0 -> silent "
            "fallback to the slow CUDA-core moe_wna16 kernel). Verified on "
            "Gemma-4-26B-A4B (moe_intermediate=704, TP=2 -> N=352, 352%%64=32). "
            "ADVISOR: loud warning naming the shape + FP8/int8 remedy (turns a "
            "silent ~1.5-1.85x slowdown into a visible signal). PROVIDER: inject "
            "Genesis-tuned moe_wna16 block configs for table-listed shapes "
            "(generic, fail-open, extend per rig sweep). Found by the /loop "
            "deep MoE-kernel audit 2026-06-21 (4 research agents + code)."
        ),
        "applies_to": {},
        "composes_with": [],
    },
    "G4_85": {
        "title": "TurboMind int4 grouped-MoE kernel (tensor-core moe_wna16 replacement)",
        "tier": "community",
        "family": "moe",
        "env_flag": "GENESIS_ENABLE_G4_85",
        "default_on": False,
        "category": "kernel",
        "implementation_status": "partial",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.moe.g4_85_tm_int4_moe_kernel",
        "lifecycle": "experimental",
        "credit": (
            "Routes Marlin-ineligible int4 MoE layers through TurboMind's "
            "tensor-core sm80_16816 int4 grouped-MoE GEMM (third_party/tm_int4_moe) "
            "instead of vLLM's slow CUDA-core moe_wna16_gemm. 3-6x faster on "
            "SM80/86, 0.036%% rel-err vs FP16 (rig-proven). Targets the LIVE "
            "CompressedTensorsWNA16MoEMethod.apply (the slow moe_wna16 path). "
            "Gated on GENESIS_ENABLE_G4_85 (default OFF, experimental) AND "
            "is_moe_model() AND G4_84's marlin_moe_marginal() detector, so it "
            "only fires on Marlin-ineligible int4 MoE. JIT-builds the vendored "
            "torch extension on first apply. Fail-open: any error degrades to "
            "the original moe_wna16 path. LIVE INVESTIGATION (rig, pin dev148, "
            "2026-06-23): the patch APPLIES but NEVER fires on the prod preset — "
            "prod-gemma4-26b-default runs --enable-expert-parallel, EP keeps "
            "intermediate_per_partition=704 (not 352) so Marlin stays eligible "
            "and vLLM picks the SIBLING CompressedTensorsWNA16MarlinMoEMethod "
            "(not hooked here). No perf loss: the 26B is already FAST via Marlin "
            "(~136-169 tok/s), not the slow moe_wna16 path. Dead-on-prod but "
            "harmless. Second blocker on pure-TP no-EP (where moe_wna16 IS "
            "selected): the vendored build is incomplete — build_kernels.sh "
            "missed -fPIC (fixed 2026-06-23) and compiles only kernels/gemm/* "
            "(13 TUs), so genesis_tm.so fails to dlopen (undefined symbol: "
            "vtable for turbomind::LinearWeight; the linear_weight/LlamaLinear/"
            "core TUs are never built). Full enablement deferred pending a "
            "design decision: also hook the Marlin sibling, OR bench/claim "
            "G4_85 only on a pure-TP no-EP config. implementation_status=partial "
            "reflects this. Built by the deep TurboMind-int4-MoE port 2026-06-22."
        ),
        "upstream_pr": None,
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["G4_84"],
        "applies_to": {},
    },
    "G4_83": {
        "title": "Gemma 4 per-layer attention backend on Ampere (#38891 backport)",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_83_GEMMA4_PER_LAYER_BACKEND",
        "default_on": True,
        "category": "perf_hotfix",
        "implementation_status": "full",
        "source": "vllm_pr_backport",
        "upstream_pr": 38891,
        "upstream_pr_relationship": "backport",
        "apply_module": "sndr.engines.vllm.patches.model_compat.gemma4.g4_83_per_layer_flashattn",
        "lifecycle": "experimental",
        "credit": (
            "Backport of vllm PR #38891 (OPEN, fixes #38887). On Ampere (no "
            "FA4) Gemma4Config force-sets TRITON_ATTN for ALL layers — a "
            "~5-11x attention tax on the ~80%% of layers (sliding-window, "
            "head_dim=256) that can run FlashAttention. G4_83 undoes the "
            "global force (only when the engine itself set it, never an "
            "explicit operator backend) so each layer picks its own backend: "
            "sliding-256 -> FlashAttention, global-512 -> Triton. The "
            "kv_sharing contract (G4_69 skip-list [58,59] + G4_71b/G4_75 "
            "drafter override) is preserved independently. Rig-validated "
            "2026-06-21 (Gemma-4-31B-AWQ, 2x A5000): correctness intact "
            "(7x6->42, no mixed-backend corruption), decode TPOT 11.9->~10.9ms "
            "median/3 runs (-8.5%%), TPS 65->~70 (+8%%)."
        ),
        "applies_to": {},
        "composes_with": ["G4_69", "G4_71B", "G4_75", "G4_16"],
    },
    "G4_16": {
        "title": "Gemma 4 FULL_AND_PIECEWISE cudagraph mode (parallel to PN125 for gemma4)",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_16_GEMMA4_FULL_AND_PIECEWISE",
        "default_on": True,
        "category": "perf_hotfix",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.model_compat.gemma4.g4_16_gemma4_full_piecewise_cudagraph",
        "lifecycle": "stable",
        "stable_kind": "runtime-hook",
        "production_validated_pins": [
            ("v12.0.0", "0.20.2rc1.dev338+gbf0d2dc6d"),
            ("v12.0.0", "0.20.2rc1.dev371+gbf610c2f5"),
        ],
        "credit": "Forces FULL_AND_PIECEWISE cudagraph mode on Gemma 4 dense path. Upstream's splitting_ops heuristic doesn't catch gemma4 model_type. Expected +10-30% TPS on decode at low batch.",
        "upstream_pr": None,
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_17": {
        "title": "Gemma 4 vision-tower text-only skip",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_17_GEMMA4_VISION_SKIP",
        "default_on": False,
        "category": "memory_savings",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.model_compat.gemma4.g4_17_gemma4_vision_tower_text_only_skip",
        "lifecycle": "experimental",
        "credit": (
            "Stubs vision tower + multi_modal_projector when "
            "GENESIS_GEMMA4_TEXT_ONLY=1. Saves ~2.3 GB VRAM + ~30 sec "
            "cold boot. Audit 2026-05-24 (PIN.R-REFS-CLOSED-PR.R): the "
            "previously recorded upstream reference vllm#41565 was found "
            "to be a wrong-number — issue #41565 is the TurboQuant "
            "`_continuation_prefill` workspace bug (the correct upstream "
            "tracker for G4_61 / G4_62), not a multimodal-text-only "
            "report. No matching upstream issue for the vision-tower "
            "text-only skip was found via `gh search issues` "
            "(2026-05-24). `upstream_pr=None` (genesis_original, no "
            "upstream tracker) until a correct upstream issue is located."
        ),
        "upstream_pr": None,
        "requires_patches": [],
        "conflicts_with": ["G4_23"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration"]},
    },
    "G4_18": {
        "title": "Per-layer KV cache page-size for 26B-A4B (vendors WIP vllm#40391)",
        "tier": "community",
        "family": "kv_cache",
        "env_flag": "GENESIS_ENABLE_G4_18_GEMMA4_PER_LAYER_KV_PAGE_SIZE",
        "default_on": False,
        "category": "kernel",
        "implementation_status": "partial",
        "source": "vendor_backport",
        "apply_module": "sndr.engines.vllm.patches.kv_cache.g4_18_per_layer_kv_page_size",
        "lifecycle": "experimental",
        "credit": "Hooks ModelConfig.get_num_kv_heads to return per-layer-type KV-head counts for asymmetric 26B-A4B. Closes vllm#40388 root cause (not just guard). CAVEAT (2026-06-16 dev491 audit): on vLLM >=0.22 (dev491) get_num_kv_heads(self, parallel_config) has NO layer_idx kwarg, so the per-layer branch cannot fire — G4_18 falls through to the original and does NOT supersede G4_13 there. Keep default_on=False and rely on G4_13 until #40391 lands; the 26B YAML disabling G4_13 'because G4_18 supersedes it' does not hold on dev491.",
        "upstream_pr": "https://github.com/vllm-project/vllm/pull/40391",
        "requires_patches": [],
        "conflicts_with": ["G4_13"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration"]},
    },
    "G4_23": {
        "title": "Gemma 4 vision-tower FP16 overflow fix (closes vllm#40124)",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_23_GEMMA4_VISION_FP16_OVERFLOW",
        "default_on": True,
        "category": "stability",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.model_compat.gemma4.g4_23_gemma4_vision_fp16_overflow_fix",
        "lifecycle": "stable",
        "stable_kind": "runtime-hook",
        "production_validated_pins": [
            ("v12.0.0", "0.20.2rc1.dev338+gbf0d2dc6d"),
            ("v12.0.0", "0.20.2rc1.dev371+gbf610c2f5"),
        ],
        "credit": "Forces vision tower to BF16 (or soft-clip fallback) when operator chose FP16. Prevents NaN propagation from patch-embed overflow.",
        "upstream_pr": "https://github.com/vllm-project/vllm/issues/40124",
        "requires_patches": [],
        "conflicts_with": ["G4_17"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration"]},
    },
    "G4_24": {
        "vllm_version_range": (">=0.20.0", "<0.23.0"),  # retired-provenance drift cap (native softcap LogitsProcessor in-pin)
        "title": "Fused softcap Triton kernel route for Gemma 4 (FINAL logits only; G4_24b will cover attention)",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_24_GEMMA4_FUSED_SOFTCAP",
        "default_on": False,
        "category": "kernel",
        # RETIRED 2026-06-19 (dev148 TIER-1 audit): vLLM's native softcap
        # LogitsProcessor supersedes the fused-softcap route. The Genesis
        # fused-softcap kernel introduced a per-token GPU->CPU sync stall on
        # the final-logit path (the wrapper read a scalar back to host each
        # decode step) that negated the kernel-fusion win on the A5000 decode
        # hot path. The native LogitsProcessor does the softcap on-device with
        # no host round-trip. default_on stays False. The Triton kernel file
        # (kernels/g4_softcap_triton.py) is KEPT as a library — it is not
        # deleted, only de-wired from the active route.
        "implementation_status": "retired",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.model_compat.gemma4.g4_24_gemma4_fused_softcap_route",
        "lifecycle": "retired",
        "superseded_by": "vLLM native softcap LogitsProcessor (on-device softcap, no per-token GPU->CPU sync; the Genesis fused-softcap route's host scalar read-back stalled the A5000 decode hot path)",
        "retired_reason": (
            "native softcap LogitsProcessor supersedes; the fused-softcap "
            "kernel route had a per-token GPU->CPU sync stall that negated "
            "the fusion win. Kernel file kept as a library, de-wired only."
        ),
        "credit": "Triton kernel fuses div+tanh+mul for softcap calls. Routes final-logit softcap via wrapper. Expected +3-5% TPS on decode at low batch.",
        "upstream_pr": None,
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["G4_15"],
        # iron-rule-#11 pin-gate for the retire: the native softcap
        # LogitsProcessor supersedes on the current support window; cap at
        # <0.23.0 so the fused-softcap route only ever decides on pre-0.23.0
        # pins (where it was active). De-wired, kernel file kept as a library.
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"], "vllm_version_range": (">=0.20.0", "<0.23.0")},
    },
    "G4_19": {
        "title": "Genesis G4-TurboQuant KV cache for Gemma 4 (3/4-bit VQ, unlocks 256K context on 2× A5000)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_G4_19_GEMMA4_TURBOQUANT_KV",
        "default_on": False,
        "category": "memory_savings",
        "implementation_status": "experimental",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.g4_19_turboquant_kv_cache",
        "lifecycle": "experimental",
        "credit": "Genesis-original — TurboQuant-style vector-quantized KV cache adapted for Gemma 4 (head_dim=256, interleaved sliding/global attention, Lloyd-Max 3/4-bit codebooks). Parallel pattern to our Qwen 3.5/3.6 P67/PN116/PN118/PN119 production stack. Triton kernels at integrations/gemma4/kernels/turboquant/. Unlocks 256K context on 2× A5000 (48GB total) which is infeasible on fp16 KV (~22GB just for KV at 256K + 20GB weights = 42GB no margin). Expected quality: -0.5 to -1.5 pp MMLU vs fp16 KV, top-1 retrieval 81-95%.",
        "upstream_pr": "https://github.com/vllm-project/vllm/issues/38171",
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["G4_09"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_19B": {
        "title": "G4-TurboQuant KV spec integration with vLLM v1 _check_enough_kv_cache_memory",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_G4_19B_GEMMA4_TQ_KV_SPEC",
        "default_on": False,
        "category": "kernel",
        "implementation_status": "experimental",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.g4_19b_tq_kv_spec_integration",
        "lifecycle": "experimental",
        "credit": "Genesis-original — companion to G4_19. Hooks vllm.v1.core.kv_cache_utils._check_enough_kv_cache_memory to multiply available KV cache memory by G4-TurboQuant compression factor. Without G4_19b, vLLM's preflight memory check rejects 256K context boot because it doesn't know about our compressed cache. With G4_19b, vLLM accepts compressed cache as logically smaller. Temporary monkey-patch until upstream vllm#38171 ships compression-aware KV spec.",
        "upstream_pr": "https://github.com/vllm-project/vllm/issues/38171",
        "requires_patches": ["G4_19"],
        "conflicts_with": [],
        "composes_with": ["G4_19"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_60A": {
        "title": "Inject TQSlidingWindowSpec into vllm.v1.kv_cache_interface (PR #42637 cherry-pick)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_G4_60A_TQ_SLIDING_SPEC",
        "default_on": False,
        "category": "memory_savings",
        "implementation_status": "experimental",
        "source": "vllm_pr_backport",
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.g4_60a_tq_sliding_window_spec",
        "lifecycle": "experimental",
        "credit": "Project-owned cherry-pick from vllm PR #42637 (lesj0610). Adds TQSlidingWindowSpec frozen dataclass with tq_slot_size field + tightens TQFullAttentionSpec.merge isinstance assertion. Prerequisite for G4_60g per-layer TQ dispatch and G4_60e mixed-route detection. Source: vllm/v1/kv_cache_interface.py lines 501-522 in PR HEAD fdeb14981. Retirement is an operator decision (switch overlay strategy or supersede), NOT a wait on upstream merge — see sndr_private/planning/audits/LOCAL_PR42637_CLOSURE_R_2026-05-28_RU.md.",
        "upstream_pr": "https://github.com/vllm-project/vllm/pull/42637",
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["G4_60E", "G4_60G"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_60E": {
        "title": "kv_cache_utils.py TQ/native mixed-layout dispatch (PR #42637 cherry-pick)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_G4_60E_KV_CACHE_UTILS",
        "default_on": False,
        "category": "memory_savings",
        "implementation_status": "experimental",
        "source": "vllm_pr_backport",
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.g4_60e_kv_cache_utils",
        "lifecycle": "experimental",
        "credit": "Upstream cherry-pick from vllm PR #42637 (lesj0610). Patches 4 symbols on vllm.v1.core.kv_cache_utils: is_kv_cache_spec_uniform (detect TQ+native mix), unify_kv_cache_spec_page_size (TQ-aware padded path), inject _is_tq_native_mixed_kv_cache_spec predicate, wrap get_kv_cache_groups dispatch. Source: PR HEAD fdeb14981 lines 854-881, 1019-1063, 1484-1512, 1696-1706. Requires G4_60A.",
        "upstream_pr": "https://github.com/vllm-project/vllm/pull/42637",
        # 2026-06-13 wave-2 reconciliation: the unify_kv_cache_spec_page_size
        # ladder + _reshape_attention_kv_cache hardening fold in OPEN
        # vllm#45207 (MambaSpec page padding) and vllm#45181 (generic
        # attention reshape) — tracked here per the related_upstream_prs
        # convention (the PR fixes referenced by the patch text).
        "related_upstream_prs": [45207, 45181],
        "requires_patches": ["G4_60A"],
        "conflicts_with": [],
        "composes_with": ["G4_60A", "G4_60G"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_60G": {
        "title": "Attention.get_kv_cache_spec per-layer TQ-first dispatch (PR #42637 cherry-pick)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_G4_60G_TQ_DISPATCH",
        "default_on": False,
        "category": "memory_savings",
        "implementation_status": "experimental",
        "source": "vllm_pr_backport",
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.g4_60g_attention_dispatch",
        "lifecycle": "experimental",
        "credit": "Upstream cherry-pick from vllm PR #42637 (lesj0610). Replaces Attention.get_kv_cache_spec to dispatch turboquant_* layers FIRST (TQSlidingWindowSpec for sliding, TQFullAttentionSpec for full) before the plain SlidingWindowSpec/FullAttentionSpec branches. Fixes dev371 behaviour where sliding layers got plain SlidingWindowSpec and TQ compression was silently disabled on the sliding tier. Source: PR HEAD fdeb14981 vllm/model_executor/layers/attention/attention.py lines 575-633. Requires G4_60A.",
        "upstream_pr": "https://github.com/vllm-project/vllm/pull/42637",
        "requires_patches": ["G4_60A"],
        "conflicts_with": [],
        "composes_with": ["G4_60A", "G4_60E", "G4_60H"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_60L": {
        "title": "TurboQuantBackend supports_mm_prefix=True monkey-patch (Gemma 4 MM-prefix LM unblock)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_G4_60L_TQ_BACKEND_MM_PREFIX",
        "default_on": False,
        "category": "stability",
        "implementation_status": "experimental",
        "source": "vllm_pr_backport",
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.g4_60l_tq_backend_mm_prefix",
        "lifecycle": "experimental",
        "credit": "Phase 7.G4.31B.K4-TURBOQUANT-BACKEND-MM-PREFIX (2026-05-23). Stock vllm pin 0.20.2rc1.dev371+gbf610c2f5 ships TurboQuantBackend without the supports_mm_prefix=True classmethod override that PR #42637 overlays/pr42637/turboquant_attn.py lines 221-223 adds. Without it, Gemma 4 31B AWQ + TURBOQUANT backend hard-fails engine init with 'partial multimodal token full attention not supported' (model_config.is_mm_prefix_lm=True propagates through --language-model-only because the flag only skips vision/audio towers, not the Gemma4ForMultimodalLM wrapper). This patch monkey-patches the missing override into the stock class at apply() time. Idempotent: if the overlay file is bind-mounted (β'-A hand-launcher path) the override is already present and apply() no-ops. Companion to G4_60B (overlay verifier, requires bind-mount) — G4_60L is the Python-side equivalent for the V2 compose path that does not bind-mount.",
        "upstream_pr": "https://github.com/vllm-project/vllm/pull/42637",
        "vllm_version_range": "<0.21",
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["G4_60A", "G4_60B", "G4_60K", "G4_19", "G4_19B"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_60H": {
        "title": "TurboQuantConfig KV-sharing skip-layer helpers (PR #42637 cherry-pick)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_G4_60H_TQ_CONFIG_AUGMENT",
        "default_on": False,
        "category": "memory_savings",
        "implementation_status": "experimental",
        "source": "vllm_pr_backport",
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.g4_60h_turboquant_config_augment",
        "lifecycle": "experimental",
        "credit": "Upstream cherry-pick from vllm PR #42637 (lesj0610). Injects 4 missing symbols into vllm.model_executor.layers.quantization.turboquant.config: TurboQuantConfig.align_kv_sharing_skip_layers + get_kv_sharing_target_skip_layers (static methods); module-level _sort_skip_layers + _get_kv_sharing_target_fanout helpers. Required by G4_60K for skip-layer union with high-fanout KV-sharing target protection. Source: PR HEAD fdeb14981 lines 220-396.",
        "upstream_pr": "https://github.com/vllm-project/vllm/pull/42637",
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["G4_60K"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_60B": {
        "title": "Verify turboquant_attn.py PR #42637 overlay is bind-mounted",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_G4_60B_TQ_ATTN_OVERLAY",
        "default_on": False,
        "category": "memory_savings",
        "implementation_status": "experimental",
        "source": "vllm_pr_backport",
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.g4_60b_turboquant_attn_overlay_loader",
        "lifecycle": "experimental",
        "credit": "Verifies turboquant_attn.py bind-mount overlay from vllm PR #42637 (lesj0610). Inspects TurboQuantAttentionImpl for _decode_prefill_from_cache, _continuation_prefill, _cache_prefill_attention methods (PR #42637 signatures). Returns error if missing (overlay not bind-mounted at boot). Companion to G4_60c/d (Triton kernel overlays). Source file in overlays/pr42637/turboquant_attn.py.",
        "upstream_pr": "https://github.com/vllm-project/vllm/pull/42637",
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["G4_60C", "G4_60D"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_60C": {
        "title": "Verify triton_turboquant_decode.py PR #42637 overlay is bind-mounted",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_G4_60C_TQ_DECODE_OVERLAY",
        "default_on": False,
        "category": "memory_savings",
        "implementation_status": "experimental",
        "source": "vllm_pr_backport",
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.g4_60c_triton_decode_overlay_loader",
        "lifecycle": "experimental",
        "credit": "Verifies triton_turboquant_decode.py bind-mount overlay from vllm PR #42637. Inspects triton_turboquant_decode_attention launcher signature for sliding_window + mm_prefix_range kwargs. Source: overlays/pr42637/triton_turboquant_decode.py (756 LOC).",
        "upstream_pr": "https://github.com/vllm-project/vllm/pull/42637",
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["G4_60B", "G4_60D"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_60D": {
        "title": "Verify triton_turboquant_store.py PR #42637 overlay is bind-mounted",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_G4_60D_TQ_STORE_OVERLAY",
        "default_on": False,
        "category": "memory_savings",
        "implementation_status": "experimental",
        "source": "vllm_pr_backport",
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.g4_60d_triton_store_overlay_loader",
        "lifecycle": "experimental",
        "credit": "Verifies triton_turboquant_store.py bind-mount overlay from vllm PR #42637. Store kernel changes minimal (+4/-4) — module-import verification only. Source: overlays/pr42637/triton_turboquant_store.py (447 LOC).",
        "upstream_pr": "https://github.com/vllm-project/vllm/pull/42637",
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["G4_60B", "G4_60C"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_61": {
        "title": "Share TQ decode workspace across layers (PR #40798 cherry-pick)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_G4_61_TQ_SHARED_WORKSPACE",
        "default_on": False,
        "category": "memory_savings",
        "implementation_status": "experimental",
        "source": "vllm_pr_backport",
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.g4_61_tq_shared_workspace",
        "lifecycle": "experimental",
        "credit": "Upstream cherry-pick from vllm PR #40798 (Bot1822, Guipeng Zhang, OPEN). Per-layer _tq_mid_o_buf / _tq_output_buf / _tq_lse_buf allocations replaced with shared WorkspaceManager acquisition. capture_model pre-reserves max-shape workspace before lock_workspace fires. PR validated 105->66 GiB model loading drop, 3.7x KV pool boost on Llama-3.1-70B. Closes issue #41565 (continuation_prefill workspace fails long-ctx, MidasMining's bisect: regression source = #40941 WorkspaceManager merge 2026-04-27).",
        "upstream_pr": "https://github.com/vllm-project/vllm/pull/40798",
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["G4_62", "PN118"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_71": {
        "title": "Force FlashAttn backend for Gemma 4 MTP drafter Attention layers",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_G4_71_DRAFTER_NATIVE_BACKEND",
        "default_on": False,
        "category": "kernel",
        "implementation_status": "experimental",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.spec_decode.g4_71_drafter_native_attn_backend",
        "lifecycle": "experimental",
        "credit": "PN261-C production fix per user 2026-05-19. PN260 trace proved drafter's Attention objects (prefix 'draft_model.*') were instantiating TurboQuantAttentionImpl despite their physical KV cache being native FlashAttn-shaped (5-dim bf16). G4_69 did not reroute because drafter's self.kv_cache_dtype stays 'turboquant_4bit_nc' (drafter is not in cache_config.kv_cache_dtype_skip_layers). Result: TurboQuant Triton kernel _tq_decode_stage1 launched on a native cache, walked out-of-bounds, asynchronous cudaErrorIllegalAddress reported by NCCL watchdog. G4_71 wraps Attention.__init__: when prefix starts with 'draft_model.' (configurable via GENESIS_G4_71_DRAFTER_PREFIX), substitute attn_backend=FlashAttention before original init. Drafter then never has TurboQuant impl. Companion to G4_69 (target skip layers), G4_72 (drafter native spec), and PN261-A pre-launch assert.",
        "upstream_pr": None,
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["G4_69", "G4_60K", "G4_60G", "G4_60H", "G4_72"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_72": {
        "title": "Force native FullAttentionSpec/SlidingWindowSpec for Gemma 4 MTP drafter layers",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_G4_72_DRAFTER_NATIVE_SPEC",
        "default_on": False,
        "category": "kernel",
        "implementation_status": "experimental",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.spec_decode.g4_72_drafter_native_kv_cache_spec",
        "lifecycle": "experimental",
        "credit": "PN261-D production fix per user 2026-05-19. After G4_71 forced drafter Attention impl to FlashAttn, K=2 produced a clean ValueError at flash_attn.py:744 'key_cache, value_cache = kv_cache.unbind(0)' because drafter's allocated KV cache had axis-order (num_blocks, 2, block_size, num_kv_heads, head_dim) — the TQFullAttentionSpec layout — instead of FlashAttn's expected (2, num_blocks, ...). Root cause: G4_60g's get_kv_cache_spec still routed drafter into TQFullAttentionSpec because drafter's self.kv_cache_dtype stays 'turboquant_4bit_nc'. G4_72 wraps Attention.get_kv_cache_spec AFTER G4_60g: when self._genesis_g4_71_is_drafter is True (marker set by G4_71), returns native FullAttentionSpec or SlidingWindowSpec with dtype = vllm_config.model_config.dtype (bf16) and kv_quant_mode = no-quant. Non-drafter layers fall through to G4_60g unchanged. Companion to G4_71 (impl-level fix) and PN261-A assert.",
        "upstream_pr": None,
        "requires_patches": ["G4_71"],
        "conflicts_with": [],
        "composes_with": ["G4_71", "G4_60G", "G4_60A", "G4_69"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_76": {
        "title": "Disable Gemma4Proposer._setup_gemma4_kv_sharing (PN265 — make drafter fully independent)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_G4_76_DISABLE_DRAFTER_KV_SHARING",
        "default_on": False,
        "category": "kernel",
        "implementation_status": "experimental",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.spec_decode.g4_76_disable_drafter_kv_sharing",
        "lifecycle": "experimental",
        "credit": "PN265 architectural fix per user 2026-05-19. After G4_71/G4_72/G4_73/G4_74-cap/G4_75 unblocked K=2 first prompt, multi-prompt H8-0 probe found CUDA illegal memory access on 14-token prompt. Root cause: contradictory state — G4_74 broke physical kv_cache alias to give drafter independent HND tensor capped at 256 blocks, but Gemma4Proposer._setup_gemma4_kv_sharing still set attn.kv_sharing_target_layer_name=target_layer on drafter Attention. vllm then uses target's slot_mapping (block ids up to 24987) for drafter writes — drafter has only 256 blocks → OOB → CUDA illegal access. G4_76 wraps Gemma4Proposer._setup_gemma4_kv_sharing to be a no-op. Drafter then has kv_sharing_target_layer_name=None and is treated as fully independent: own kv_cache_groups entry (via G4_72 native spec), own block_table from kv_cache_manager, own slot_mapping referencing drafter's own block range. Writes stay in bounds. Trade-off: drafter has cold kv_cache at request start (no inherited target context); acceptance will be 0% until G4_77 warm-up is added. Companion to G4_71/G4_72/G4_73/G4_74/G4_75; precedes G4_77 (warm-up restoration of drafter context).",
        "upstream_pr": None,
        # NOTE 2026-06-16 (dev491): tried relaxing requires to [] to run G4_76
        # STANDALONE on a pure-TQ (memory-safe) drafter, since the no-op of
        # _setup_gemma4_kv_sharing is backend-independent. It boot-FAILED in
        # _reshape_kv_cache_tensors: "shape '[78104,32,8,262]' invalid for input
        # of size 10237247488" (= 512 bytes/slot bf16 buffer vs 262 TQ slot) —
        # disabling kv_sharing without G4_72's native spec leaves a TQ sliding
        # layer with a bf16-sized buffer. So the dependency is REAL: G4_76 needs
        # the native-drafter spec companions. On dev491 those (G4_71/G4_72) make
        # the drafter bf16 -> +9.27GiB OOM at 64K ctx. MTP-on-31B-tq is thus a
        # 3-way bind (kv_sharing OOB / native OOM / pure-TQ reshape-mismatch);
        # see journal 2026-06-16-g4_82-native-tq-headdim512-fix.md. Requires kept.
        "requires_patches": ["G4_71", "G4_72"],
        "conflicts_with": [],
        "composes_with": ["G4_71", "G4_72", "G4_73", "G4_74", "G4_75", "PN262"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_75": {
        "title": "Per-layer drafter backend split: route head_size=512 to TRITON_ATTN (PN264 fix)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_G4_75_DRAFTER_HEAD512_TRITON",
        "default_on": False,
        "category": "kernel",
        "implementation_status": "experimental",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.spec_decode.g4_75_drafter_head512_triton",
        "lifecycle": "experimental",
        "credit": "PN264 fix per user 2026-05-19. After G4_71/G4_72/G4_73/G4_74-cap unblocked drafter sliding layers (head=256), first prompt failed with 'FlashAttention forward only supports head dimension at most 256' on drafter layer 3 (head_size=512). Backend capability probe in this pin: FLASH_ATTN caps 256, FLASHINFER supports [64,128,256], TRITON_ATTN supports head_size>=32 (covers 512). G4_75 wraps Attention.__init__ AFTER G4_71: when drafter prefix + head_size==512, kwargs['attn_backend'] is overridden to AttentionBackendEnum.TRITON_ATTN.get_class(). Also stamps self._genesis_g4_75_drafter_triton=True so G4_74 skips HND transpose for the Triton-routed layer (Triton uses NHD natively). Sliding drafter layers 0..2 stay on FlashAttn + HND; layer 3 uses Triton + NHD. Companion to G4_71 (impl), G4_72 (spec), G4_73 (profile skip), G4_74 (HND conv + cap), PN262 (forward trace).",
        "upstream_pr": None,
        # 2026-06-16: requires_patches CORRECTED [G4_71,G4_74]->[]. G4_71/G4_74 were the
        # RETIRED independent-drafter stack; the breakthrough kv_sharing MTP config
        # (G4_75=1 + G4_71B=1 + G4_76=0 + G4_67->G4_81) runs G4_75 STANDALONE — validated
        # coherent 40.7 t/s on 31B-tq (run bcfojku36). G4_75 only needs the drafter to exist
        # (head_size=512 layer present), not the G4_71 backend. Stale dep caused a false
        # dispatcher validator ERROR + risked auto-disabling G4_75 on a strict validator.
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["G4_71B", "G4_76", "G4_67", "G4_81", "G4_82", "PN262"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_74": {
        "title": "Drafter HND layout enforcement post-reshape (PN263 fix for FlashAttn unbind(0))",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_G4_74_DRAFTER_HND_LAYOUT",
        "default_on": False,
        "category": "kernel",
        "implementation_status": "experimental",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.spec_decode.g4_74_drafter_hnd_layout",
        "lifecycle": "experimental",
        "credit": "PN263 fix per user 2026-05-19. PN262 with fixed args[4] index revealed actual drafter kv_cache shape at FlashAttn forward is NHD: (num_blocks=4, 2, block_size, num_kv_heads, head_dim) — contiguous, bf16, 5-D. FlashAttn at flash_attn.py:744 expects HND: (2, num_blocks, ...) — its 'key_cache, value_cache = kv_cache.unbind(0)' splits k/v on the leading axis. NHD has num_blocks as leading axis, so unbind(0) returns num_blocks tensors → 'too many values to unpack (expected 2)'. Path A (upstream SpeculativeConfig.attention_backend=FLASH_ATTN) produced bit-identical NHD shape, confirming the field doesn't propagate to physical kv_cache layout. G4_74 wraps GPUModelRunner._reshape_kv_cache_tensors. After the original returns kv_caches dict, for each layer whose name starts with 'draft_model.' and is 5-D: if shape[0]==2 (already HND) no-op; if shape[1]==2 (NHD) replace with kv_cache.transpose(0, 1).contiguous(); else fail-fast. Mutation occurs before bind_kv_cache stores the tensor in static_forward_context, so attention context delivers HND tensor to FlashAttn forward. Drafter-only — target TQ layers are not touched. Companion to G4_71 (impl), G4_72 (spec), G4_73 (profile skip), PN262 (forward-side fail-fast trace).",
        "upstream_pr": None,
        "requires_patches": ["G4_71"],
        "conflicts_with": [],
        "composes_with": ["G4_71", "G4_72", "G4_73", "G4_60G", "PN262"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_73": {
        "title": "Skip drafter.dummy_run during profile_run (PN262-D pragmatic boot unblock)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_G4_73_DRAFTER_PROFILE_SKIP",
        "default_on": False,
        "category": "kernel",
        "implementation_status": "experimental",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.spec_decode.g4_73_drafter_profile_skip",
        "lifecycle": "experimental",
        "credit": "PN262-D pragmatic boot unblock per user 2026-05-19. After G4_71 (impl reroute) and G4_72 (spec reroute), K=2 still crashed inside determine_available_memory -> profile_run because drafter's profile-time KV cache placeholder was sized by the GROUP's backend (TQ) instead of the per-layer impl (FlashAttn). PN262-A fail-fast captured the shape (8192, 8, 256) = (num_blocks*block_size, num_kv_heads, head_dim) — exactly what TurboQuantAttentionBackend.get_kv_cache_shape() returns. A/B with PN259c=0 was bit-identical, ruling out the split allocator. The wrong-shape tensor reaches FlashAttn via profile_run -> _dummy_run -> self.drafter.dummy_run -> drafter.model.forward -> unified_attention_with_output. G4_73 wraps GPUModelRunner._dummy_run to set a thread-local in_profile flag, and wraps SpecDecodeBaseProposer.dummy_run to return early when the flag is set. Profile completes (memory estimate omits drafter — small relative to 31B target). Engine then proceeds to initialize_kv_cache(real_config), which sub-groups layers by per-layer attn_backend (G4_71 honored), so runtime KV cache allocation respects the FlashAttn shape contract for drafter. Companion to G4_71 (impl), G4_72 (spec), PN262 (fail-fast trace).",
        "upstream_pr": None,
        "requires_patches": ["G4_71"],
        "conflicts_with": [],
        "composes_with": ["G4_71", "G4_72", "G4_60G", "PN262"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_71B": {
        "title": "Per-layer drafter backend force: route head_size=256 sliding to TRITON_ATTN (β'-A K=4 enabler)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_G4_71B_DRAFTER_SLIDING_TRITON",
        "default_on": False,
        "category": "kernel",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.spec_decode.g4_71b_drafter_sliding_triton",
        "lifecycle": "experimental",  # operationally validated by the gemma4-31b-tq-mtp-structured-k4 profile (status=validated), but registry-lifecycle stays experimental until production-default cutover
        "credit": "Phase 3 bucket 3 registration (2026-05-21): G4_71B is load-bearing for the validated β'-A K=4 structured path (gemma4-31b-tq-mtp-structured-k4 profile). The structured profile declares it via backend_plan.drafter_sliding=TRITON_ATTN. Companion to G4_75 (head_size=512 → TRITON_ATTN) — each owns a disjoint drafter head_size class. β control + PN271b proved the canonical TQ+MTP launcher has a kernel-vs-storage contract mismatch on drafter[0..2] (TurboQuantAttentionImpl reading native bf16 bytes as TQ-packed → acceptance=0). G4_71B forces drafter sliding layers to Triton NHD native bf16 so the safety guard accepts the configuration as EXACT_COPY. Required for production opt-in of the structured profile.",
        "upstream_pr": None,
        # 2026-06-16: requires_patches CORRECTED [G4_71]->[]. G4_71 was the retired
        # independent-drafter backend; G4_71B forces drafter sliding(head256) layers to
        # TRITON_ATTN native-bf16 for the kv_sharing path and runs STANDALONE (validated
        # in the working 31B-tq MTP config alongside G4_75, NOT G4_71). Stale dep produced
        # a false validator ERROR.
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["G4_75", "G4_76", "G4_67", "G4_81", "G4_82", "PN262", "PN271"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_78": {
        "title": "Drafter K/V bridge from target[58]/[59] (RETIRED — superseded by P1.8 A2 declarative physical kv_sharing)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_G4_78_DRAFTER_TARGET_KV_BRIDGE",
        "default_on": False,
        "category": "kernel",
        "implementation_status": "retired",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm._archive.g4_78_drafter_target_kv_bridge",
        "lifecycle": "retired",
        "retired_waiver": True,
        "credit": "RETIRED 2026-05-21 (Phase 3 bucket 3). G4_78 was an investigative drafter K/V bridge fallback developed during the H8/PN267-PN269 cycle when the kv_sharing path was unknown. The validated β'-A K=4 path (P1.8 A2 declarative backend_plan.drafter_kv_sharing=physical) proved physical kv_sharing is the correct production-supported drafter contract. The bridge approach is not needed for the validated structured profile and was never the production path. File moved to integrations/_retired/ to preserve git history while removing it from the active spec_decode/ namespace. Not enabled by any V2 profile.",
        "upstream_pr": None,
        "superseded_by": ["drafter_kv_sharing=physical (backend_plan, P1.8 A2)"],
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "PN262B": {
        "title": "KV cache allocator/reshape/proposer-init diagnostic trace (PN262-A D-3 deep-dive)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_PN262B_KV_ALLOC_TRACE",
        "default_on": False,
        "category": "observability",  # Phase 3A.4 2026-05-22: was 'diagnostic' (not in VALID_CATEGORIES enum)
        "implementation_status": "experimental",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.spec_decode.probes.pn262b_kv_alloc_trace",
        "lifecycle": "experimental",
        "credit": "PN262-A K=2 (2026-05-19) proved the wrong-axis tensor reaches FlashAttn.forward as a contiguous 3-dim (8192, 8, 256) tensor — not a view, not aliasing, not VLLM_KV_CACHE_LAYOUT, not PN259c (A/B identical). The wrong physical shape comes from GpuModelRunner._reshape_kv_cache_tensors line ~6846 where shape = attn_backend.get_kv_cache_shape(...). attn_backend is the *group*'s backend, and the group's kv_cache_spec drives the contract. PN262-B wraps _reshape_kv_cache_tensors AND LLMBaseProposer.initialize_attn_backend with pre+post diagnostic logs that dump kv_cache_groups (id, spec class, backend class, layer_names), raw_tensor info per drafter layer, final reshape tensor info, proposer-side gid + per-AttentionGroup backend/spec/layers. Identifies which of (e1) wrong group backend, (e2) wrong group spec, (e3) UniformTypeKVCacheSpecs missing per-layer entries for drafter is the actual root cause. Companion to PN262 (FlashAttn forward trace + fail-fast), G4_71 (impl reroute), G4_72 (spec reroute).",
        "upstream_pr": None,
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["PN262", "G4_71", "G4_72", "G4_60G"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "PN262": {
        "title": "FlashAttn drafter KV cache shape/stride trace + fail-fast (PN261-D D-3 localization)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_PN262_FLASH_ATTN_DRAFTER_TRACE",
        "default_on": False,
        "category": "observability",  # Phase 3A.4 2026-05-22: was 'diagnostic' (not in VALID_CATEGORIES enum)
        "implementation_status": "experimental",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.spec_decode.probes.pn262_flash_attn_drafter_trace",
        "lifecycle": "experimental",
        "credit": "PN261-D follow-up. After G4_71 (impl reroute) + G4_72 (spec reroute) were verified by 8/8 marker logs in the 2026-05-19 K=2 gate, the first forward STILL crashed at flash_attn.py:744 'key_cache, value_cache = kv_cache.unbind(0)' — drafter received a tensor with leading dim != 2. Spec and impl were native; the wrong layout must come from one of: (a) allocator built wrong physical shape, (b) bind/view applied a transpose between allocator and forward, (c) drafter aliases another layer's TQ cache via kv_sharing_target_layer_name, (d) global VLLM_KV_CACHE_LAYOUT=NHD forces NHD on all layers. PN262 wraps FlashAttentionImpl.forward and, for drafter layers only, logs shape+stride+dtype+is_contiguous+data_ptr+kv_sharing_target+layout-env BEFORE the unbind call, then raises a clean RuntimeError with full context. Diagnostic only — does not change behaviour for non-drafter or for correct drafter shapes. Companion to G4_71/G4_72/PN261-A.",
        "upstream_pr": None,
        "requires_patches": ["G4_71"],
        "conflicts_with": [],
        "composes_with": ["G4_71", "G4_72", "G4_60G"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_19C": {
        "title": "Round-trip K/V through G4-TurboQuant inside Gemma4Attention.forward",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_G4_19C_ATTN_WRAP",
        "default_on": False,
        "category": "kernel",
        "implementation_status": "partial",
        "source": "genesis_original",
        "apply_module": None,
        "lifecycle": "retired",
        "retired_waiver": True,
        "retired_reason": (
            "2026-05-29 boot failure on rig (dev371+, gemma4-31b-tq-mtp-structured-k4 "
            "container): `_g4_19c_roundtrip_tensor` custom kernel is invoked from "
            "Gemma4Attention.forward path that Dynamo traces under torch.compile. "
            "The kernel was not wrapped as an opaque custom op (no torch.library."
            "custom_op or allow_in_graph at the entry), so Dynamo attempts to "
            "fake-tensor-trace through it and raises: 'Cannot access data pointer "
            "of Tensor (e.g. FakeTensor, FunctionalTensor). If you're using torch."
            "compile/export/fx, it is likely that we are erroneously tracing into "
            "a custom kernel.' This crashes the engine core with Worker died. "
            "Workaround in production: `GENESIS_ENABLE_G4_19C_ATTN_WRAP=0` set on "
            "the launcher (gemma4 prod uses this). Proper fix requires wrapping "
            "`_g4_19c_roundtrip_tensor` via torch.library.custom_op with a fake-"
            "tensor meta — see https://pytorch.org/tutorials/advanced/custom_ops"
            "_landing_page.html. P7b's import-time-cached opaque-op pattern is "
            "the reference (sndr/engines/vllm/patches/attention/gdn/p7b_*.py). "
            "Until then, the TQ KV cache contract for gemma4 is INCOMPLETE on "
            "the hot path — TQ allocation happens (G4_19) and memory accounting "
            "is correct (G4_19B) but actual K/V round-tripping does not engage. "
            "Operationally OK because gemma4-31b-tq-mtp-structured-k4 runs at very "
            "small max_model_len=4096 / max_num_seqs=1, so TQ compression is not "
            "the bottleneck. Retirement keeps the wiring intact (no file move) "
            "for diff against a future fix candidate."
        ),
        "credit": "Phase 3 bucket 4 registration (2026-05-21). G4_19C wraps Gemma4Attention.forward to round-trip K and V through the G4-TurboQuant write+read kernels — was intended to complete the TQ KV cache contract started by G4_19 (KV cache registration) and G4_19B (memory accounting). Skip optimization: sliding layers (window=1024) bypass TQ since their cache is already small. Companion to G4_19 (KV cache), G4_19B (spec integration), G4_31 (dtype preservation). RETIRED 2026-05-29 — torch.compile FakeTensor incompatibility.",
        "upstream_pr": None,
        "requires_patches": ["G4_19"],
        "conflicts_with": [],
        "composes_with": ["G4_19", "G4_19B", "G4_31", "G4_60B", "G4_60C", "G4_60D"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_31": {
        "title": "Preserve turboquant_* kv_cache_dtype against AWQ quant-config override",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_G4_31_TQ_DTYPE_PRESERVE",
        "default_on": False,
        "category": "kernel",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.g4_31_preserve_tq_dtype",
        "lifecycle": "experimental",
        "credit": "Phase 3 bucket 4 registration (2026-05-21). When the AWQ-compressed checkpoint declares kv_cache_scheme, vLLM's quant-config layer overrides the operator-supplied turboquant_* kv_cache_dtype back to whatever AWQ recommended (typically auto or fp16). G4_31 wraps the post-load dtype reconciliation to preserve turboquant_* if the operator explicitly requested it. Required for TQ k8v4 / 4bit_nc to actually be applied to AWQ-compressed Gemma checkpoints. Companion to G4_19 (KV cache install) and G4_32 (validation bypass).",
        "upstream_pr": None,
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["G4_19", "G4_19B", "G4_19C", "G4_32"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_32": {
        "title": "Bypass TurboQuantAttentionBackend.validate_configuration for Gemma 4",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_G4_32_TQ_VALIDATION_BYPASS",
        "default_on": False,
        "category": "kernel",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.g4_32_tq_validation_bypass",
        "lifecycle": "experimental",
        "credit": "Phase 3 bucket 4 registration (2026-05-21). TurboQuantAttentionBackend.validate_configuration refuses Gemma 4's interleaved sliding+global attention combo because the upstream validator was tuned for uniform attention layouts. G4_32 wraps the validator to skip the refusal when Gemma 4 arch is detected. Required to boot Gemma 4 with TURBOQUANT attention backend. Companion to G4_69 (skip-layer routing), G4_60K (skip-list plumbing).",
        "upstream_pr": None,
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["G4_19", "G4_60K", "G4_69"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_79": {
        "title": "TQ backend supports_mm_prefix for Gemma 4 MM (0.22.1 validity-gate unblock, mm half)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_G4_79_TQ_MM_PREFIX",
        "default_on": False,
        "category": "kernel",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.g4_79_tq_mm_prefix_support",
        "lifecycle": "experimental",
        "credit": (
            "Fleet validation 2026-06-11: pin 0.22.1rc1.dev259 added a NEW "
            "validity gate (v1/attention/backend.py:301-304) requiring "
            "supports_mm_prefix() from every backend serving an "
            "is_mm_prefix_lm model; Gemma 4 MM + TURBOQUANT boot rejects. "
            "Read-only investigation verified Gemma 4 vision/audio towers "
            "run EAGER (gemma4_mm.py:1037/1053) and never reach vLLM "
            "attention — TQ only quantizes the text-decoder KV, so "
            "declaring support is semantically safe (same basis as "
            "Triton/FlexAttention which declare it without mm-specific "
            "decode logic). Fixes ONLY the mm_prefix refusal; the sibling "
            "kv_cache_dtype refusal is the G4_31 class one stage earlier "
            "— first instrumented 31B boot discriminates (module "
            "docstring recipe; G4_32 blanket bypass is the fallback). "
            "Surgical successor to the G4_32-era approach."
        ),
        "upstream_pr": None,
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["G4_19", "G4_31", "G4_32"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
        "vllm_version_range": (">=0.22.0", "<0.23.0"),
    },
    "G4_80": {
        "title": "Allow fp8_e5m2 KV cache for weight-only quantized checkpoints (vllm#45040)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_G4_80_FP8E5M2_KV",
        "default_on": False,
        "category": "kernel",
        "implementation_status": "full",
        "source": "vllm_pr_backport",
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.g4_80_fp8e5m2_kv_weight_only",
        "lifecycle": "experimental",
        "credit": (
            "PR-sweep wave 1 vendoring (2026-06-11, roadmap chunk-3 Theme B; "
            "review fix same day). Upstream vllm#45040 (OPEN; closes #39137): "
            "the pristine _init_kv_cache_quant gate (attention.py:167-168 on "
            "pin 0.22.1rc1.dev259+g303916e93, byte-verified) rejects "
            "--kv-cache-dtype fp8_e5m2 for EVERY compressed-tensors checkpoint "
            "because CompressedTensorsConfig.get_quant_method returns "
            "CompressedTensorsKVCacheMethod for all Attention layers "
            "(compressed_tensors.py:205-206) regardless of kv_cache_scheme. "
            "Weight-only AWQ/GPTQ carry no fp8 KV scales, so fp8 KV was "
            "unreachable for Gemma-4-31B AWQ (CT checkpoint). TWO ARMS: arm 1 "
            "= module-symbol rebind (NOT body copy) masking layer.kv_cache_dtype "
            "around the original call when the upstream "
            "_checkpoint_has_fp8_kv_scales predicate allows (gate is the sole "
            "kv_cache_dtype read in the pristine body); arm 2 (Genesis extra, "
            "no upstream fix exists — gh-searched 2026-06-11) = "
            "Attention.__init__ wrap nulling query_quant for fp8_e5m2 layers: "
            "TritonAttentionImpl sets supports_quant_query_input "
            "unconditionally on CUDA (triton_attn.py:502), so the e4m3-only "
            "QuantFP8 query quantizer is created and the FIRST forward (boot "
            "memory profiling) dies on assert kv_cache_dtype in {fp8, "
            "fp8_e4m3, nvfp4} (attention.py:467). Impl forwards handle "
            "unquantized queries natively (q_descale dtype-gated, "
            "triton_attn.py:607-614). Genesis extras: BOTH import sites "
            "rebound (mla_attention.py:219 imports by value), install-time "
            "drift guard refusing on gate-signature loss (retire trigger: "
            "#45040 merge), GENESIS_G4_80_FORCE_ALLOW_WITH_KV_SCHEME escape "
            "hatch (accuracy-unvalidated). Pairs with G4_31 arm 2 (vllm#45038 "
            "sub-SM90 fp8-auto guard): guard protects the kv-auto interim "
            "state, G4_80 is the escape. Consumed by profile "
            "gemma4-31b-fp8e5m2-fallback on TRITON_ATTN — the ONLY viable "
            "backend on this pin: FA2/FLEX have no fp8 KV (flash_attn.py:70-74, "
            "flex_attention.py:86-90); FLASHINFER has true e5m2 + no query "
            "quant sub-SM90 but fails the Gemma-4 mm-prefix validity gate "
            "(supports_mm_prefix False, backend.py:301-303 — the G4_60L/G4_79 "
            "gate class). KNOWN MASQUERADE (audited): TRITON_ATTN stores+loads "
            "quantized KV as platform e4m3fn regardless of the e5m2 string "
            "(triton_reshape_and_cache_flash.py:364-376, triton_attn.py:597-602) "
            "— 1-byte KV with e4m3 numerics + unit scales; e5m2 selector still "
            "correct vs plain fp8 because it keeps the query unquantized "
            "(#44879 IMA surface). 31B KV halves (~9.4 -> ~4.7 GiB @200K) "
            "toward full 256K ctx. Boot+bench validation pending (incl. Triton "
            "emulated e4m3 cast compile on SM 8.6); 30 torch-less unit tests."
        ),
        "upstream_pr": 45040,
        "upstream_pr_relationship": "backport",
        # Pin bump dev148 -> dev301 (2026-06-24): NOT retired this bump.
        # #45040 merged into vllm main AFTER dev301 (it is NOT in dev301 —
        # the dev301 deep-diff/anchor-SOT regen does not show G4_80's anchor
        # drifting), so arm 1 (the kv_cache_dtype mask rebind) still applies
        # on dev301. arm 2 (Attention.__init__ query_quant nulling) is
        # Genesis-original (no upstream fix exists). CREDIT: arm 1 will
        # retire when #45040 lands in a future pin; arm 2 stays. Left active.
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["G4_31", "G4_79"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
        # Registry-integration 2026-06-11: G4_79-template parity — the
        # rebind anchors + drift guard target the 0.22.1 validity-gate
        # generation (pin-specific vendor range per the G4_79 checklist).
        "vllm_version_range": (">=0.22.0", "<0.23.0"),
    },
    # ─── 50-PR sweep WAVE 2 (2026-06-13) ────────────────────────────────
    # Ten new vendors/blueprints from the roadmap. All pin-specific
    # (g303916e93 / 0.22.1rc1.dev259), all opt-in (default OFF) except
    # PN377 (clamp-only, provably inert on every current PROD model).
    # Registered spec-only (apply_module with own apply(), no
    # apply_patch_* legacy hook) EXCEPT PN377 which has a legacy parking-
    # lot hook — same convention as the wave-1 G4_79/G4_80 + PN371/PN373
    # block above. vllm_version_range top-level per the G4_79/G4_80
    # template (pin-specific vendor range).
    "P88": {
        "title": "Prefix-cache stats retry de-duplication (rewrite of vllm#45202)",
        "tier": "community",
        "family": "observability",
        "env_flag": "GENESIS_ENABLE_P88_PREFIX_CACHE_STATS_DEDUP",
        "default_on": False,
        "category": "observability",
        "implementation_status": "full",
        "source": "vllm_pr_backport",
        "apply_module": "sndr.engines.vllm.patches.observability.p88_prefix_cache_stats_dedup",
        "lifecycle": "experimental",
        "credit": (
            "PR-sweep wave 2 (2026-06-13). Genesis REWRITE of OPEN "
            "vllm#45202 (fixes #43736), NOT the upstream diff: "
            "KVCacheManager.get_computed_blocks records the prefix-cache "
            "query/hit stats at LOOKUP time, so a waiting request whose "
            "allocate_slots then fails (no free blocks) re-counts the "
            "stats on every later scheduler step. Under KV-pressure "
            "burst retries (long-context agent profile) prefix_hit_rate "
            "inflates by tens of percent, poisoning the P85 / TQ-KV A/B "
            "conclusions read off /metrics. Upstream moves the record "
            "into the ~2000-line Scheduler.schedule() waiting loop; P88 "
            "instead keeps BOTH sites inside kv_cache_manager.py "
            "(P79d-style minimal-anchor): the LOOKUP site stashes a "
            "single pending record on _genesis_p88_pending_stats and "
            "allocate_slots COMMITS it exactly once after its last "
            "failure return (request-id matched, slot cleared so a "
            "running-loop second allocate cannot double-record). More "
            "faithful than upstream for our configs (records iff a real "
            "lookup happened — enable_caching=False records nothing). "
            "Metrics-only; fallback-disables when a KV connector is "
            "configured. Self-skips on #45202 merge (the LOOKUP record() "
            "anchor disappears)."
        ),
        "upstream_pr": 45202,
        # Genesis rewrites #45202's fix at a different layer
        # (kv_cache_manager.py lookup-stash + allocate-commit, NOT the
        # scheduler-side record the PR moves it to) — same bug class,
        # non-overlapping site.
        "upstream_pr_relationship": "related_not_superseding",
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["P85"],
        "applies_to": {"vllm_version_range": (">=0.22.0", "<0.24.0")},
        "vllm_version_range": (">=0.22.0", "<0.24.0"),
    },
    "PN358": {
        "title": "FULL cudagraph forward-context refresh, data_ptr-pruned (vendor of vllm#44868)",
        "tier": "community",
        "family": "compile_safety",
        "env_flag": "GENESIS_ENABLE_PN358_FULL_CG_CONTEXT_REFRESH",
        "default_on": False,
        "category": "stability",
        "implementation_status": "full",
        "source": "vllm_pr_backport",
        "apply_module": "sndr.engines.vllm.patches.compile_safety.pn358_full_cg_context_refresh",
        "lifecycle": "experimental",
        "credit": (
            "PR-sweep wave 2 (2026-06-13). Vendor of OPEN vllm#44868 "
            "(weicj): during FULL CUDA-graph capture the graph entry "
            "bakes in references to the forward-context tensors that "
            "existed at capture time (attn metadata, slot mappings, "
            "ubatch slices, DP metadata, additional kwargs); on replay "
            "fresh tensors leave the captured graph reading stale "
            "metadata — silent wrong-continuation / degenerate-output "
            "class under spec-decode, not a crash. Verified at pin "
            "g303916e93: compilation/cuda_graph.py has NO refresh on the "
            "replay path. FULL_AND_PIECEWISE via PN125 + MTP K=3 + the "
            "287-patch overlay is exactly the exposure surface. Genesis "
            "extras over the PR: (1) data_ptr-pruned copy — only leaves "
            "whose live tensor moved storage are copied (kills the PR's "
            "1-3% unconditional per-replay copy cost); (2) "
            "GENESIS_PN358_MODE=detect log-only audit of stale-metadata "
            "hazards (zero hazard lines == the overlay is clean); plus "
            "shape-mismatch skip-not-crash, cycle guard, self-disable on "
            "internal error. Engages via GENESIS_ENABLE_PN358_FULL_CG_"
            "CONTEXT_REFRESH (install gate); MODE selects refresh|detect. "
            "Composes with PN353B / PN118 (turboquant_attn.py / "
            "workspace.py — no file overlap on cuda_graph.py). Self-skips "
            "on #44868 merge (drift markers = the PR's helper names)."
        ),
        "upstream_pr": 44868,
        "upstream_pr_relationship": "backport",
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["PN353B", "PN118"],
        "applies_to": {"vllm_version_range": (">=0.22.0", "<0.24.0")},
        "vllm_version_range": (">=0.22.0", "<0.24.0"),
    },
    "PN376": {
        "title": "FP8 modules_to_not_convert substring match (vendor of vllm#44628)",
        "tier": "community",
        "family": "quantization",
        "env_flag": "GENESIS_ENABLE_PN376_FP8_IGNORE_SUBSTRING",
        "default_on": False,
        "category": "quantization",
        "implementation_status": "full",
        "source": "vllm_pr_backport",
        "apply_module": "sndr.engines.vllm.patches.quantization.pn376_fp8_ignore_substring",
        "lifecycle": "experimental",
        "credit": (
            "PR-sweep wave 2 (2026-06-13). Vendor of OPEN vllm#44628 "
            "(fixes #21669): Fp8Config.get_quant_method calls "
            "is_layer_skipped with exact prefix matching, so HF-style "
            "short modules_to_not_convert patterns "
            "(e.g. 'linear_attn.in_proj_qkv') never match the "
            "fully-qualified runtime prefix and the checkpoint-excluded "
            "layer silently loads as FP8 without its weight_scale — "
            "gibberish output, no exception, no log. The AWQ family "
            "fixed this class with an opt-in substring match "
            "(#26909/#27416/#29774); #44628 opts FP8 in. Adapted per "
            "iron rule #10: the pin has ONE more is_layer_skipped call "
            "site than the PR base (LinearBase + RoutedExperts in "
            "fp8.py), and the quant_utils experts branch keeps "
            "parent-in-child MoE containment in substring mode. CORE "
            "pair (fp8.py + quant_utils.py) lands atomically via "
            "MultiFilePatchTransaction; parity one-liners "
            "(fbgemm/mxfp4/modelopt) best-effort. Genesis impact: "
            "Qwen3.6-VL FP8 is broken TODAY on the pin by this class. "
            "VALIDATION GATE before any default_on: per-layer "
            "quant-scheme log diff on 35B PROD + the two Gemma-4 AWQ "
            "models (shared quant_utils experts branch)."
        ),
        "upstream_pr": 44628,
        "upstream_pr_relationship": "backport",
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["P81", "P91", "P91B"],
        "applies_to": {"vllm_version_range": (">=0.22.0", "<0.24.0")},
        "vllm_version_range": (">=0.22.0", "<0.24.0"),
    },
    "PN377": {
        "title": "moe_wna16 BLOCK_SIZE_K legality clamp (vendor of vllm#44563)",
        "tier": "community",
        "family": "moe",
        "env_flag": "GENESIS_ENABLE_PN377_MOE_WNA16_BSK_CLAMP",
        "default_on": True,
        "category": "kernel",
        "implementation_status": "full",
        "source": "vllm_pr_backport",
        "apply_module": "sndr.engines.vllm.patches.moe.pn377_moe_wna16_bsk_clamp",
        "lifecycle": "experimental",
        "credit": (
            "PR-sweep wave 2 (2026-06-13). Vendor of OPEN vllm#44563 "
            "(fixes #36008): GPTQ/AWQ int4 MoE models with group_size=32 "
            "abort moe_wna16_gemm at warmup with 'BLOCK_SIZE_K // "
            "group_size must be one of [1, 2, 4, 8]' — "
            "get_moe_wna16_block_config grows BLOCK_SIZE_K to 512 "
            "(ratio 16) on the first small decode batch. The wna16 path "
            "is the LIVE Marlin fallback (awq_marlin.py / auto_gptq.py) "
            "for unsupported RoutedExperts layers. Fix: cap block_size_k "
            "at group_size*8 before the divisibility step; gs 64/128 can "
            "mathematically never overshoot so legal configs are "
            "untouched. DEFAULT ON (the clamp only rewrites "
            "kernel-illegal configs — provably inert for every current "
            "PROD model: 35B FP8 never takes the wna16 path, gs>=64 AWQ "
            "MoE can never overshoot); GENESIS_ENABLE_PN377_MOE_WNA16_"
            "BSK_CLAMP=0 skips. Install additionally gated on "
            "is_moe_model() (P52 dispatch, P24 pattern). Genesis extra: "
            "boot-time legality assert sweeps the on-disk heuristic over "
            "the actual model MoE grid and fires a loud actionable ERROR "
            "instead of the cryptic warmup abort. Composes with P24 "
            "(same file, get_default_config — disjoint anchors, "
            "byte-verified) and PN352/PN368 (different files). Unblocks "
            "gs=32 int4 MoE benchmarking (roadmap chunk-5 Theme D)."
        ),
        "upstream_pr": 44563,
        "upstream_pr_relationship": "backport",
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["P24", "PN352", "PN368"],
        # 2026-06-17 (0.23.1 pin-bump): cap bumped <0.23.0 -> <0.24.0 (both the
        # applies_to gate AND its SoT mirror, kept in sync). default_on moe_wna16
        # BLOCK_SIZE_K legality clamp silently no-op'd on 0.23.1. PR #44563 OPEN
        # (merged-form marker "max_block_size_k = group_size * 8" ABSENT in
        # v0.23.1rc0); anchor (fused_moe.py _ensure_block_size_k_divisible call)
        # byte-present. Inert on kernel-legal configs (gs>=64 can't overshoot).
        "applies_to": {"vllm_version_range": (">=0.22.0", "<0.24.0")},
        "vllm_version_range": (">=0.22.0", "<0.24.0"),
    },
    "PN378": {
        "title": "Recovered-token vocab-pad -inf mask (vendor of vllm#45060, kernel half)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_PN378_VOCAB_PAD_MASK",
        "default_on": False,
        "category": "spec_decode",
        "implementation_status": "full",
        "source": "vllm_pr_backport",
        "apply_module": "sndr.engines.vllm.patches.spec_decode.pn378_recovered_token_vocab_pad_mask",
        "lifecycle": "experimental",
        "credit": (
            "PR-sweep wave 2 (2026-06-13). Vendor of OPEN vllm#45060 "
            "(KERNEL HALF only; root cause of #26372/#33729/#42722): "
            "sample_recovered_tokens_kernel tiles the vocab in "
            "BLOCK_SIZE chunks and the final tile's padding lanes load "
            "other=0.0. On all-NaN target_probs the NaN-propagating "
            "tl.max lets that zero-score padding run win, returning "
            "recovered_id == vocab_size — an out-of-vocab id that "
            "parse_output drops, collapsing the row to [] and "
            "livelocking the request. LIVE on our stack: wrapper "
            "hardcodes BLOCK_SIZE=8192 and Qwen vocab 151936 % 8192 != "
            "0, so every recovered-token sample of both PROD MTP models "
            "carries padding lanes. Fix: mask padding lanes to -inf "
            "before the tile reduction (recovered_id keeps in-vocab "
            "init 0 on NaN rows; healthy rows byte-identical). Genesis "
            "divergence: spells float('-inf') so the PR's -float('inf') "
            "line stays usable as a drift marker. The SCHEDULER half is "
            "NOT vendored — PN133 v2 is the safer half (keeps the "
            "request schedulable + log.error on the invariant). "
            "Composes with PN133 (removes the out-of-vocab source; PN133 "
            "keeps accounting correct — different files, zero overlap). "
            "Roadmap: land with PN372 (#45005) in one MTP-hardening "
            "bench cycle."
        ),
        "upstream_pr": 45060,
        "upstream_pr_relationship": "backport",
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["PN133", "PN372"],
        "applies_to": {"vllm_version_range": (">=0.22.0", "<0.23.0")},
        "vllm_version_range": (">=0.22.0", "<0.23.0"),
    },
    "PN379": {
        "title": "LoadConfig / DefaultModelLoader fail-fast validation (vendor of vllm#45196)",
        "tier": "community",
        "family": "loader",
        "env_flag": "GENESIS_ENABLE_PN379_LOAD_CONFIG_FAIL_FAST",
        "default_on": False,
        "category": "stability",
        "implementation_status": "full",
        "source": "vllm_pr_backport",
        "apply_module": "sndr.engines.vllm.patches.loader.pn379_load_config_fail_fast",
        "lifecycle": "experimental",
        "credit": (
            "PR-sweep wave 2 (2026-06-13). Vendor of OPEN vllm#45196 "
            "(Sunt-ing): three silent-misconfig classes -> loud "
            "ValueErrors. (1) LoadConfig typing — load_format: str | "
            "LoadFormats is str | Any at runtime so pydantic accepted "
            "any type, and a typo'd safetensors_load_strategy silently "
            "fell back to lazy; fixed with load_format: str + a Literal "
            "re-derived from THIS pin's weight_utils dispatch sites. "
            "(2) DefaultModelLoader extra-config validation — non-dict "
            "extra config, non-bool enable_multithread_load, "
            "non-positive num_threads, and multithread+non-lazy-strategy "
            "(byte-verified: the multi-thread iterator drops the "
            "strategy on this pin). (3) explicit-safetensors .pt "
            "fallback guard. Six anchored edits across config/load.py + "
            "default_loader.py, atomic 2-file transaction. Zero hot-path "
            "cost (constructor-only); safety prerequisite for the "
            "multithread-load restart-time experiment (server-stage). "
            "Static pre-deploy mirror in scripts/audit_config_keys.py "
            "loader-key pass."
        ),
        "upstream_pr": 45196,
        "upstream_pr_relationship": "backport",
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": [],
        "applies_to": {"vllm_version_range": (">=0.22.0", "<0.24.0")},
        "vllm_version_range": (">=0.22.0", "<0.24.0"),
    },
    "PN380": {
        "title": "Qwen3.5/3.6 MTP pre-fused expert loader + load-coverage guard (vendor of vllm#44943)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_PN380_MTP_PREFUSED_LOADER",
        "default_on": False,
        "category": "spec_decode",
        "implementation_status": "full",
        "source": "vllm_pr_backport",
        "apply_module": "sndr.engines.vllm.patches.spec_decode.pn380_qwen3_mtp_prefused_expert_loader",
        "lifecycle": "experimental",
        "credit": (
            "PR-sweep wave 2 (2026-06-13). Vendor of OPEN vllm#44943 "
            "(Qwen3.5/3.6 MTP pre-fused expert loader) + Genesis "
            "draft-weight load-coverage guard. Qwen3_5MultiTokenPredictor"
            ".load_weights only recognizes experts.gate_up_proj / "
            "down_proj as checkpoint SOURCE names; a checkpoint storing "
            "expert tensors under the fused names directly (community "
            "AutoRound/GPTQ INT4 quants) falls through — quantized MTP "
            "boots with random expert weights and accept rate silently "
            "collapses (PR A/B: 65.0% -> 41.9%), unquantized MTP crashes "
            "with a weight_loader TypeError. Adapted per iron rule #10 "
            "(pin carries the older STATIC two-entry mapping; we append "
            "two static pre-fused entries instead of the PR's loop-built "
            "alt_ckpt_name). Both PROD SKUs use split-form names "
            "(unaffected today) — this is INSURANCE for the planned INT4 "
            "35B-A3B trial. Genesis extra (sub-4..6): coverage guard "
            "converts the engine's quantization-disabled silent partial "
            "load into ONE log.error on any checkpoint/param gap. SAME "
            "FILE as PN348 (qwen3_5_mtp.py) — disjoint anchors (PN348 "
            "outside load_weights, PN380 inside), both co-apply orders + "
            "cross-module drift-marker hygiene asserted. Composes with "
            "PN348 + PN108/PN133/PN290/PN340/PN341/PN370 (different "
            "files)."
        ),
        "upstream_pr": 44943,
        "upstream_pr_relationship": "backport",
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["PN348", "PN133", "PN340", "PN341", "PN370"],
        "applies_to": {
            "model_class": ["qwen3_5", "qwen3_6"],
            "vllm_version_range": (">=0.22.0", "<0.24.0"),
        },
        "vllm_version_range": (">=0.22.0", "<0.24.0"),
    },
    "PN381": {
        "title": "allowed_token_ids spec-decode metadata hardening (vendor of vllm#44742)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_PN381_ALLOWED_TOKEN_IDS_METADATA",
        "default_on": False,
        "category": "spec_decode",
        "implementation_status": "full",
        "source": "vllm_pr_backport",
        "apply_module": "sndr.engines.vllm.patches.spec_decode.pn381_allowed_token_ids_spec_metadata",
        "lifecycle": "experimental",
        "credit": (
            "PR-sweep wave 2 (2026-06-13). Vendor of OPEN vllm#44742 "
            "(fixes GHSA-8c65-hq7q-r7jm): when a request sets ONLY "
            "allowed_token_ids, InputBatch._make_sampling_metadata ships "
            "output_token_ids == [] while allowed_token_ids_mask and the "
            "draft-token counts are non-empty; any consumer that derives "
            "the request count from len(output_token_ids) mis-expands "
            "the mask rows during draft verification. Single anchored "
            "sub-patch on gpu_input_batch.py adding the "
            "allowed_token_ids clause to needs_output_token_ids. "
            "DEFENSE-IN-DEPTH: the pin's consumer fix #35654 already "
            "sizes the draft expansion by len(num_draft_tokens) so no "
            "in-tree consumer trips today; PN381 populates once at the "
            "PRODUCER so every present/future consumer (the P71+PN369 "
            "rewritten rejection-sampler paths) inherits row-"
            "consistency. Zero perf either way (NULL on the entire "
            "Genesis PROD workload, which never sets allowed_token_ids). "
            "Genesis emits the parenthesized clause so the PR's "
            "unparenthesized form is the drift marker. Same playbook as "
            "retired PN67 (#41674), one clause further. Composes with "
            "P71 (rejection_sampler.py — different file; PN369 was "
            "consolidated into the P71 entry 2026-06-19)."
        ),
        "upstream_pr": 44742,
        "upstream_pr_relationship": "backport",
        "requires_patches": [],
        "conflicts_with": [],
        # 2026-06-19: PN369 consolidated into the P71 entry; the pair
        # ["PN369", "P71"] collapses to ["P71"] (same rejection_sampler.py
        # consumer paths, now one registry id).
        "composes_with": ["P71"],
        "applies_to": {"vllm_version_range": (">=0.22.0", "<0.24.0")},
        "vllm_version_range": (">=0.22.0", "<0.24.0"),
    },
    "PN382": {
        "title": "DecodeBenchConnector hybrid per-block KV fill (vendor of vllm#45080)",
        "tier": "community",
        "family": "kv_cache",
        "env_flag": "GENESIS_ENABLE_PN382_DECODE_BENCH_HYBRID_FILL",
        "default_on": False,
        "category": "kv_cache",
        "implementation_status": "full",
        "source": "vllm_pr_backport",
        "apply_module": "sndr.engines.vllm.patches.kv_cache.pn382_decode_bench_hybrid_fill",
        "lifecycle": "experimental",
        "credit": (
            "PR-sweep wave 2 (2026-06-13). Vendor of OPEN vllm#45080 + "
            "two Genesis extensions. DecodeBenchConnectorWorker._fill_"
            "blocks assumes every layer's KV cache is a single "
            "block-indexed tensor and dies with AttributeError: 'list' "
            "object has no attribute 'device' on the first decode batch "
            "for hybrid / linear-attention models (Mamba/GDN register a "
            "LIST of state tensors) — making decode-TPOT-vs-depth "
            "benching impossible on our GDN hybrids. Upstream splits the "
            "fill (tensors keep block-row fill; list/tuple caches get "
            "each state tensor filled in its entirety). Genesis "
            "extensions (iron rule #10): (1) PER-BLOCK fill for the "
            "list/tuple path — on this pin every MambaSpec state tensor "
            "is block-indexed (num_blocks, *shape), so the whole-pool "
            "fill would clobber the recurrent state of every concurrent "
            "request; (2) REAL group_idx -> layer_names map from "
            "kv_cache_config.kv_cache_groups (upstream maps all layers "
            "to group 0, which on hybrids fills Mamba pools with the "
            "attention group's block ids). Bench-infrastructure only "
            "(never in a PROD --kv-transfer-config); MTP must be OFF for "
            "sweeps. Unlocks the 8K/32K/128K/280K sweep on Qwen3.6 "
            "hybrids (roadmap chunk-3 Theme D)."
        ),
        "upstream_pr": 45080,
        # RECLASSIFIED 2026-06-24 (pin bump dev148 -> dev301): #45080 is IN
        # dev301 (its anchor drifted per the dev301 anchor-SOT regen), but
        # the merge is a DIVERGENT FORK, not a supersession — KEEP ACTIVE.
        # `related_not_superseding` is the valid-enum equivalent of a
        # "divergent_fork" (status-based retire is NOT eligible; see
        # divergence_note). Was "backport".
        "upstream_pr_relationship": "related_not_superseding",
        "notes": (
            "vllm#45080 merged a WEAKER whole-pool/group-0 variant of the "
            "DecodeBenchConnector hybrid fill; our per-block fill + the real "
            "kv_cache_groups group_idx->layer_names map are correctness-"
            "critical for GDN hybrids (Mamba/GDN block-indexed state) and are "
            "ABSENT from the merge — upstream maps all layers to group 0 and "
            "fills whole pools, clobbering concurrent recurrent state. Keep "
            "active; do NOT retire on #45080's merge into dev301."
        ),
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": [],
        "applies_to": {"vllm_version_range": (">=0.22.0", "<0.24.0")},
        "vllm_version_range": (">=0.22.0", "<0.24.0"),
    },
    "G4_81": {
        "title": "TQ multi-query DIRECT decode routing for Gemma-4-31B MTP (vllm#45144 blueprint)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_G4_81_TQ_MQ_DIRECT_ROUTE",
        "default_on": False,
        "category": "kernel",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.g4_81_tq_multi_query_direct_route",
        "lifecycle": "experimental",
        "credit": (
            "PR-sweep wave 2 (2026-06-13). Variant B multi-query DIRECT "
            "decode routing, vllm#45144 BLUEPRINT (ROCm MTP + fp8 KV + "
            "AITER Shuffle-KV — studied, NOT vendored: no ROCm code). "
            "MTP K=3 x TurboQuant on Gemma-4-31B dense is blocked — "
            "spec-verify batches (uniform max_query_len=K+1 with prior "
            "cached KV) fall into the per-request _prefill_attention "
            "continuation path that does GPU->CPU syncs (cudagraph-"
            "hostile #40880 class) and routes through the "
            "PN255/PN256-broken cache-read path (the 4.9x-slowdown "
            "workaround). #45144 is the second independent upstream "
            "validation of the Genesis P67/P67b technique (PROD-active "
            "on 35B, +32% TPS): route uniform multi-query verify batches "
            "through the single-token decode kernel. ADAPTATION (iron "
            "rule #10): synthetic per-token expansion (P67b / G4_67 / our "
            "#40914) — each of B*(K+1) rows becomes a virtual "
            "single-token decode. Runtime monkey-patch of "
            "TurboQuantAttentionImpl.forward; batch predicate is "
            "arithmetic (no GPU sync); ANY non-routable shape or routing "
            "failure falls through to the original forward. Genesis "
            "extras over G4_67: sliding-window + mm-prefix forwarding "
            "under a launcher capability gate, engine output-buffer "
            "contract respected, per-K1 buffer holders on the impl. "
            "Expected +20-40% decode TPS on the 31B TQ profile. Composes "
            "with G4_79/G4_31/G4_80 (31B boot-gate companions — needs "
            "G4_79's mm-prefix unblock first); supersedes the G4_67 "
            "verify-path predecessor (enable ONE)."
        ),
        "upstream_pr": 45144,
        # #45144 is a ROCm/AITER blueprint, not a port target — Genesis
        # reimplements the technique on the TQ CUDA decode kernel via
        # synthetic per-token expansion; no shared code, different layer.
        "upstream_pr_relationship": "related_not_superseding",
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["G4_79", "G4_80", "G4_31", "G4_67", "P67", "P67b"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
        "vllm_version_range": (">=0.22.0", "<0.23.0"),
    },
    "G4_82": {
        "title": "TQ prefill SDPA fallback for head_dim>256 (Ampere FA2 256-cap, vllm#38887)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_G4_82_TQ_PREFILL_SDPA_HEADDIM",
        "default_on": False,
        "category": "kernel",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.g4_82_tq_prefill_sdpa_headdim",
        "lifecycle": "experimental",
        "credit": (
            "dev491 31B-tq bring-up (2026-06-16). Native vllm "
            "TurboQuantAttentionImpl routes ALL non-decode-kernel attention "
            "compute through _flash_attn_varlen, which calls FA2's "
            "flash_attn_varlen_func unconditionally (turboquant_attn.py:311/"
            "322). FA2 caps head_dim at 256 on SM 8.x (no Ampere/Ada 512 "
            "kernel, vllm#38887) -> the Gemma-4-31B GLOBAL layers "
            "(head_dim=512) crash the worker on first-request prefill "
            "('FlashAttention forward only supports head dimension at most "
            "256'); async-scheduling masks it as a scheduler KeyError "
            "(core.py:578 batch-queue desync), --no-async-scheduling "
            "unmasks it. The pr42637 overlay had this fallback "
            "(_can_use_flash_prefill + _sdpa_causal_prefill) but is now "
            "stale on dev491 (sibling overlays lack get_kv_cache_capacity / "
            "register_all_kvcache_specs). This patch ports ONLY the "
            "head_dim>256 fallback onto the native backend: runtime "
            "monkey-patch of _flash_attn_varlen dispatching on "
            "self.head_size -> per-sequence torch SDPA for >256 (math/"
            "efficient backend, any head_dim), original FA2 fast path "
            "byte-unchanged for <=256. Provably FA2-equivalent (iron rule "
            "#11): same q/k/v inputs, FA2's exact causal mask reproduced "
            "(is_causal for q_len==k_len; bottom-right offset mask for "
            "q_len<k_len continuation); GQA via enable_gqa. Cost: one "
            "cu_seqlens.tolist() sync per call, only on the head_dim>256 "
            "prefill path (eager, off the decode hot path); zero sync for "
            "<=256. Complementary to G4_81 (which wraps .forward for the "
            "decode verify path and explicitly leaves first-chunk prefill "
            "to the flash path G4_82 repairs). No-op for every head_dim<=256 "
            "model. Required by the Gemma-4-31B TQ profile."
        ),
        "upstream_pr": 38887,
        # #38887 documents the Ampere FA2 256 head_dim cap (the constraint),
        # not a fix to port; G4_82 is the Genesis runtime workaround for it.
        "upstream_pr_relationship": "related_not_superseding",
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["G4_81", "G4_79", "G4_80", "G4_31", "G4_69", "G4_60A"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
        "vllm_version_range": (">=0.22.0", "<0.23.0"),
    },
    "G4_70": {
        "title": "PN259-B mixed-allocator path for TQ skip-list layers (PR42637 overlay control)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_G4_70_PN259B_MIXED_ALLOC",
        "default_on": False,
        "category": "kernel",
        "implementation_status": "marker_only",
        "source": "genesis_original",
        "apply_module": None,
        "lifecycle": "experimental",
        "credit": "R3 registration (2026-05-21). Companion env to G4_69 / G4_60K. Emitted by spec_decode/backend_router.py compose path when a per-layer compression plan exists with native_source_layers. Consumed by the PR42637 bind-mount overlay at attention/turboquant/overlays/pr42637/kv_cache_utils.py to route mixed-allocator KV layout for the skip-list layers (58, 59 in the validated β'-A K=4 profile). Marker-only registry entry: the env is read inside the bind-mount overlay, not by an apply_module patch — registering it here closes the R3 audit gap and makes operator-facing config-keys catalog complete. Companions: G4_70B (FAIL_FAST), G4_69 (native-backend reroute), G4_60K (skip-list plumbing). The structured profile gemma4-31b-tq-mtp-structured-k4 declares both G4_70 envs via patches_delta.enable.",
        "upstream_pr": None,
        "requires_patches": ["G4_69"],
        "conflicts_with": [],
        "composes_with": ["G4_69", "G4_60E", "G4_60G", "G4_60K"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_70B": {
        "title": "PN259-B fail-fast guard on mixed-allocator KV layout mismatch",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_G4_70_PN259B_FAIL_FAST",
        "default_on": False,
        "category": "kernel",
        "implementation_status": "marker_only",
        "source": "genesis_original",
        "apply_module": None,
        "lifecycle": "experimental",
        "credit": "R3 registration (2026-05-21). Fail-fast variant of G4_70. When mixed-allocator routing (G4_70) is active, FAIL_FAST=1 promotes a soft-warning on KV layout mismatch into a hard RuntimeError at allocator-time so the boot stops cleanly instead of producing degenerate kernel runs. Read by attention/turboquant/overlays/pr42637/kv_cache_utils.py alongside the primary G4_70 env. The validated β'-A K=4 structured profile enables this guard to make boot regressions noisy. Marker-only registry entry (env consumed by bind-mount overlay).",
        "upstream_pr": None,
        "requires_patches": ["G4_70"],
        "conflicts_with": [],
        "composes_with": ["G4_70", "G4_69", "G4_60K"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_70C": {
        "title": "PN259-C Route B split allocator for mixed TQ/native KV layout",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_G4_70_PN259C_ROUTE_B",
        "default_on": False,
        "category": "kernel",
        "implementation_status": "marker_only",
        "source": "genesis_original",
        "apply_module": None,
        "lifecycle": "experimental",
        "credit": "R3 follow-up registration (2026-05-21). Control-A output-smoke showed PN259-C Route B is load-bearing for the validated Gemma 4 β'-A K=4 TQ+MTP path. The PR42637 overlay reads GENESIS_ENABLE_G4_70_PN259C_ROUTE_B in kv_cache_utils.py to choose the split allocator: one shared TurboQuant KV tensor for compressed target layers plus native per-layer tensors for skip-listed source layers 58/59. Without this env, the V2-rendered structured launcher can boot and pass the guard but produce corrupt unicode output because spec-verify reuses the wrong TQ/native allocation shape. Marker-only registry entry; runtime code lives in the bind-mounted PR42637 overlay.",
        "upstream_pr": None,
        "requires_patches": ["G4_70", "G4_70B"],
        "conflicts_with": [],
        "composes_with": ["G4_69", "G4_70", "G4_70B", "G4_60K", "PN256", "PN261"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "PN256": {
        "title": "K+1 spec-verify routing through raw-K/V continuation prefill (PR42637 overlay control)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_PN256_KPLUS1_RAW_KV",
        "default_on": False,
        "category": "kernel",
        "implementation_status": "marker_only",
        "source": "genesis_original",
        "apply_module": None,
        "lifecycle": "experimental",
        "credit": "R3 registration (2026-05-21). Runtime control env consumed by the PR42637 bind-mount overlay at attention/turboquant/overlays/pr42637/turboquant_attn.py inside the _prefill_attention() path. When PN256=1, the K+1 spec-verify step routes through raw-K/V _continuation_prefill() instead of the TQ-quantized fast path — required for the validated β'-A K=4 acceptance contract because the spec-verify step reads native bf16 K/V from the kv-shared target slots (layers 58, 59), not TQ-packed bytes. Marker-only registry entry: env is read inside the overlay, not by an apply_module patch. Companion to G4_67 (TQ spec-verify routing), G4_68 (CG downgrade), P65 (TQ spec-decode CG downgrade base).",
        "upstream_pr": None,
        "requires_patches": ["G4_67"],
        "conflicts_with": [],
        "composes_with": ["G4_67", "G4_68", "P65", "PN261"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "PN261": {
        "title": "TQ native cache layout assert (opt-in guard; PR42637 overlay)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_PN261_TQ_NATIVE_CACHE_ASSERT",
        "default_on": False,
        "category": "kernel_safety",
        "implementation_status": "marker_only",
        "source": "genesis_original",
        "apply_module": None,
        "lifecycle": "experimental",
        "credit": "R3 registration (2026-05-21). Opt-in cache-layout assert inside the PR42637 bind-mount overlay at attention/turboquant/overlays/pr42637/triton_turboquant_decode.py. When PN261=1, asserts kv_cache.ndim==4 and kv_cache.dtype==torch.uint8 before the TQ decode kernel runs — catches KERNEL_STORAGE_DTYPE_MISMATCH conditions at the earliest possible point with a clean RuntimeError instead of producing garbage output. Default OFF until the assert is proven stable across all production paths. The validated β'-A K=4 structured profile enables this guard. Marker-only registry entry (env is read inside the overlay).",
        "upstream_pr": None,
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["G4_19", "G4_60B", "G4_60C", "PN256", "PN262"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "PN271": {
        "title": "SpecDecode KV-contract audit (model-agnostic, read-only)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_PN271_KV_CONTRACT_AUDIT",
        "default_on": False,
        "category": "observability",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.spec_decode.pn271_kv_contract_audit",
        "lifecycle": "experimental",
        "credit": "R3 registration (2026-05-21). Phase 3 bucket 1 relocated this audit from gemma4/ to spec_decode/ (operator-decision: runtime guard, not pure probe). PN271 is a model-agnostic, read-only audit run at boot: per (drafter_attn, target_attn) layer pair it inspects shape contract, KV layout (HND vs NHD), dtype, sliding window, kv_cache_dtype string, attention numerics (RoPE, softcap, scale), and quantization-mode mismatch. Output is a per-pair verdict (EXACT_COPY / GQA_REPEAT / LAYOUT_ADAPTER / DEQUANT / UNSUPPORTED). The structured profile's safety guard reads PN271's verdict + the functional artifact's config_hash to decide whether MTP is allowed to run with that configuration. Mapping providers (e.g. gemma4) are pluggable; the audit itself does not know about Gemma. Companion to G4_71 (impl reroute), G4_72 (spec reroute), G4_76 (drafter kv_sharing control), safety_guard.py (verdict consumer), functional_artifact.py (artifact-locked path).",
        "upstream_pr": None,
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["G4_71", "G4_71B", "G4_72", "G4_75", "G4_76", "PN262", "PN262B"],
        "applies_to": {"model_arch": ["*"]},
    },
    "PN274": {
        "title": "Spec-decode KV-adapter safety opt-in (operator-facing control)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "SNDR_ALLOW_SPEC_DECODE_KV_ADAPTER",
        "default_on": False,
        "category": "stability",
        "implementation_status": "marker_only",
        "source": "genesis_original",
        "apply_module": None,
        "lifecycle": "coordinator",
        "credit": "R3 registration (2026-05-21). Operator-facing safety opt-in env consumed by spec_decode/functional_artifact.py and spec_decode/safety_guard.py. When the PN271 KV-contract audit returns a non-EXACT verdict (LAYOUT_ADAPTER / GQA_REPEAT / DEQUANT), the runtime requires explicit operator consent to proceed with MTP — this env is the consent signal for the structural-adapter dimension. With a matching functional artifact (config_hash gate), only SNDR_ALLOW_SPEC_DECODE_KV_ADAPTER is required; without one, both this and SNDR_ALLOW_SPEC_DECODE_FUNCTIONAL_UNKNOWN are required. GENESIS_* aliases still resolve via get_sndr_env() with deprecation warning. Coordinator lifecycle: this is a control/policy env, not a runtime patch — registering it here closes the R3 audit gap and makes operator introspection complete. Companion to functional_artifact.py (artifact gate), safety_guard.py (verdict consumer), PN271 (verdict producer).",
        "upstream_pr": None,
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["PN271"],
        "applies_to": {"model_arch": ["*"]},
    },
    "PN275": {
        "title": "DFlash drafter VllmConfig max_cgs alignment (dev371 compat)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_PN275_DFLASH_MAX_CGS_ALIGN",
        "default_on": False,
        "category": "stability",
        "implementation_status": "experimental",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.spec_decode.pn275_dflash_max_cgs_align",
        "lifecycle": "experimental",
        "credit": "Genesis-original 2026-05-21 — dev371 compat overlay for the DFlash drafter VllmConfig re-validation defect. Q27-DFlash dev371 boot fails at vllm/v1/spec_decode/dflash.py:74 with `pydantic ValidationError: customized max_cudagraph_capture_size(=8) should be consistent with the max value of cudagraph_capture_sizes(=6)`. 3-layer root cause: parent VllmConfig init aligns the two fields (vllm/config/vllm.py:1722) — Genesis P95 then resets max=8 post-init for TP>1+Marlin+Ampere without touching sizes (desync state) — DFlash's `_create_draft_vllm_config` calls `replace(base, attention_config=...)` which rebuilds VllmConfig via `cls(**dict)` and re-runs the dev371-only cross-validator at vllm/config/vllm.py:1703-1715. dev338 had no such validator. Fix: wrap `vllm.config.utils.replace` to detect VllmConfig sources whose compilation_config has `max != max(sizes)` and inject an aligned compilation_config before delegating to the original replace. Narrow scope: only fires when sizes/max are present, inconsistent, and caller doesn't supply compilation_config kwarg explicitly. Does NOT change P95's contract; does NOT touch any ModelDef pin. Opt-in via env (default_on=False) until smoke validates Q27-DFlash + Q35-DFlash on dev371. See sndr_private/planning/audits/P2_DFLASH_CANDIDATE_A_DESIGN_REFINED_2026-05-21_RU.md for the upstream-source-cited root cause and B1 design rationale, plus sndr_private/planning/audits/P2_DFLASH_DEV371_INCOMPATIBILITY_DESIGN_2026-05-21_RU.md for the original hold posture (commit 6d40c768).",
        "upstream_pr": None,
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": [],
        "vllm_version_range": ">=0.20.2rc1.dev371",
        "applies_to": {"model_arch": ["*"]},
    },
    "G4_69": {
        "title": "Per-layer native attention backend dispatch for skip-listed TurboQuant layers",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_G4_69_SKIP_LAYERS_NATIVE_BACKEND",
        "default_on": False,
        "category": "kernel",
        "implementation_status": "experimental",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.g4_69_skip_layers_native_backend",
        "lifecycle": "experimental",
        "credit": "Genesis-original route fix that unblocks GENESIS_G4_TQ_FORCE_SKIP_LAYERS under explicit --attention-backend TURBOQUANT. The H8 cycle (Genesis investigation 2026-05-18) proved that drafter acceptance recovers (~62%) when KV-sharing target layers 58,59 are not TQ-compressed. G4_60K plumbs the skip list into cache_config.kv_cache_dtype_skip_layers, G4_60G returns native FullAttentionSpec, but Attention.__init__ still instantiates TurboQuantAttentionImpl because --attention-backend TURBOQUANT forces selected_backend at the v1 attention selector level. G4_69 wraps CudaPlatformBase.get_attn_backend_cls to clear selected_backend (=> auto-priority fall-through) only when both selected_backend==TURBOQUANT and attn_selector_config.kv_cache_dtype=='auto'. Non-skipped layers continue to dispatch TURBOQUANT verbatim; only the skip-listed 'auto'-dtype layers reroute to FLASH_ATTN. Companion to G4_32 (TQ validation bypass) and G4_60K (skip-layer plumbing).",
        "upstream_pr": None,
        "requires_patches": ["G4_60K"],
        "conflicts_with": [],
        "composes_with": ["G4_32", "G4_60E", "G4_60G", "G4_60H", "G4_60K", "G4_60B", "G4_67", "G4_68"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_68": {
        "title": "TQ spec-decode cudagraph downgrade for PR #42637 overlay (P65 v2 inline verifier)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_G4_68_TQ_SPEC_CG_DOWNGRADE_OVERLAY",
        "default_on": False,
        "category": "stability",
        "implementation_status": "marker_only",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.g4_68_tq_spec_cg_downgrade_overlay",
        "lifecycle": "experimental",
        "credit": "Genesis-original verifier for the P65 v2 cudagraph downgrade inlined directly into the PR #42637 overlay file (sndr/engines/vllm/patches/attention/turboquant/overlays/pr42637/turboquant_attn.py). Stock P65 cannot apply because the overlay is bind-mounted read-only, so P65 v2 logic was inlined as `TurboQuantMetadataBuilder.get_cudagraph_support` classmethod returning AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE when speculative_config is active. G4_68 verifies the inline marker is present at boot and reports applied/error/skipped to the dispatcher. Companion to PN256 raw-K/V continuation route (also inlined in overlay). Restores target correctness for Gemma 4 + TurboQuant + MTP under default CUDA graph mode, but does NOT recover MTP speedup (acceptance remains 0% — see H8 follow-up). Diagnostic chain PN253-PN257a, 2026-05-18.",
        "upstream_pr": None,
        "requires_patches": ["G4_60B"],
        "conflicts_with": [],
        "composes_with": ["G4_60B", "G4_60C", "G4_60D", "G4_60E", "G4_60G", "G4_60H", "G4_60K", "G4_61", "G4_62"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_67": {
        "title": "TQ K+1 spec-verify routing through decode kernel (PR #40914 backport)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_G4_67_TQ_SPEC_VERIFY_ROUTE",
        "default_on": False,
        "category": "spec_decode",
        "implementation_status": "experimental",
        "source": "vllm_pr_backport",
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.g4_67_tq_spec_verify_routing",
        "lifecycle": "experimental",
        "credit": "Backport of my upstream PR #40914 (Sandermage, OPEN) adapted for Gemma 4. Monkey-patches TurboQuantAttentionImpl.forward to detect MTP K+1 spec-verify batches (uniform max_query_len > 1 with prior cached KV) and route them through triton_turboquant_decode_attention via synth_seq_lens trick instead of default _prefill_attention. Default path has query_start_loc.tolist() GPU-CPU sync incompatible with CUDA graph capture — root cause of issue #40880 degenerate output. Empirical 4.9x slowdown observed when using cudagraph=NONE workaround (Genesis bench 2026-05-17 A5000); G4_67 removes need for workaround. Alternative to P67 (Genesis-original kernel, pin-gated dev16-dev93). G4_67 path uses existing decode kernel, no new Triton variant.",
        "upstream_pr": "https://github.com/vllm-project/vllm/pull/40914",
        "requires_patches": [],
        "conflicts_with": ["P67", "P67b"],
        "composes_with": ["G4_60B", "G4_61", "G4_62"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_62": {
        "title": "Warm up TQ decode kernels before lock_workspace (PR #42215 cherry-pick)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_G4_62_TQ_KERNEL_WARMUP",
        "default_on": False,
        "category": "stability",
        "implementation_status": "experimental",
        "source": "vllm_pr_backport",
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.g4_62_tq_kernel_warmup",
        "lifecycle": "experimental",
        "credit": "Upstream cherry-pick from vllm PR #42215 (lesj0610, OPEN). Adds turboquant_decode_warmup function that walks TQ attention layers, deduplicates by 13-field _TurboQuantDecodeWarmupKey, and calls impl._decode_attention with synthetic inputs to JIT-compile _tq_decode_stage1 + _tq_decode_stage2 BEFORE lock_workspace. Companion to G4_61: G4_62 compiles + allocates, G4_61 reserves max-shape. Either resolves issue #41565 family; together = belt-and-suspenders.",
        "upstream_pr": "https://github.com/vllm-project/vllm/pull/42215",
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["G4_61"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_60K": {
        "title": "EngineArgs.create_engine_config TQ skip-layer union + FA2 force (PR #42637 cherry-pick)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_G4_60K_TQ_ENGINE_CONFIG",
        "default_on": False,
        "category": "stability",
        "implementation_status": "experimental",
        "source": "vllm_pr_backport",
        "apply_module": "sndr.engines.vllm.patches.attention.turboquant.g4_60k_arg_utils",
        "lifecycle": "experimental",
        "credit": "Upstream cherry-pick from vllm PR #42637 (lesj0610). Wraps EngineArgs.create_engine_config to (1) union TurboQuantConfig.get_boundary_skip_layers + get_kv_sharing_target_skip_layers into cache_config.kv_cache_dtype_skip_layers and align via align_kv_sharing_skip_layers; (2) force attention_config.flash_attn_version=2 for turboquant_* dtypes (FA3 conflicts with TurboQuantAttentionImpl). G4_60H provides the required static methods. Source: PR HEAD fdeb14981 vllm/engine/arg_utils.py lines 1717-1732 and 2050-2061.",
        "upstream_pr": "https://github.com/vllm-project/vllm/pull/42637",
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["G4_60H"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_25": {
        "title": "Gemma 4 dual-RoPE base-freq divergence guard (long-context quality)",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_25_GEMMA4_RoPE_DUAL_BASE_GUARD",
        "default_on": True,
        "category": "stability",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "sndr.engines.vllm.patches.model_compat.gemma4.g4_25_gemma4_rope_dual_base_freq_guard",
        "lifecycle": "stable",
        "stable_kind": "runtime-hook",
        "production_validated_pins": [
            ("v12.0.0", "0.20.2rc1.dev338+gbf0d2dc6d"),
            ("v12.0.0", "0.20.2rc1.dev371+gbf610c2f5"),
        ],
        "credit": "Diagnoses single-table-collapse when rope_theta == global_rope_theta. Warns operator to fix config.json to distinct values.",
        "upstream_pr": None,
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_26": {
        "title": "Fix DiffusionGemma self-conditioning soft-embed for TP>1 (vocab-sharded embed_weight all-gather) — backport open vLLM PR #45774",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_26_DIFFUSIONGEMMA_TP_VOCAB",
        "default_on": False,
        "category": "correctness",
        "implementation_status": "full",
        "source": "vllm_pr_backport",
        "apply_module": "sndr.engines.vllm.patches.model_compat.gemma4.g4_26_diffusiongemma_tp_vocab_soft_embed",
        "lifecycle": "experimental",
        "credit": "Backports the TP-correctness half of open PR #45774: DiffusionGemmaForBlockDiffusion self-conditioning does probs@embed_weight over FULL vocab (262144); at TP=2 embed_tokens.weight is vocab-sharded to [131072,2816] -> RuntimeError reduction-dim mismatch. Adds get_tensor_model_parallel_world_size/tensor_model_parallel_all_gather import + _get_full_embed_weight helper + line-853 swap. SKIPs XPU/UVA hunks. Intrinsically TP-gated (helper returns .weight unchanged at TP=1). Self-skips once #45774 merges (upstream_drift_marker 'def _get_full_embed_weight').",
        "upstream_pr": "https://github.com/vllm-project/vllm/pull/45774",
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {"model_arch": ["DiffusionGemmaForBlockDiffusion"], "vllm_version_range": (">=0.22.1rc1.dev491", "<1.0.0")},
    },
}


# ─── Legacy-register patch index (Entry 17 §6.8 P-2 documentation) ────
# `_apply_module_overlay.APPLY_MODULE_OVERLAY` is the index of 127
# patches that today still live in the monolithic
# `sndr/apply/_per_patch_dispatch.py`. The patches-prove gate
# (§6.8 rule P-2) consults `apply._state.PATCH_REGISTRY` for legacy
# register membership — Phase 10 migration moves each patch into
# `integrations/<family>/<patch>.py`, and the integration-tree walk in
# `dispatcher.spec._build_apply_module_map` picks up the new home
# automatically.
#
# IMPORTANT: this index is NOT applied to PATCH_REGISTRY entries. The
# spec-loop dry-run requires a callable `apply()` in the resolved
# module, which `_per_patch_dispatch.py` doesn't expose (it has 95
# `apply_patch_X` functions, not a generic apply). Writing the overlay
# into PATCH_REGISTRY would mislead the spec-loop into trying to import
# a non-existent `apply()`. Keeping the index file as documentation
# avoids that pitfall while still letting `discover_apply_modules.py`
# track migration progress.
