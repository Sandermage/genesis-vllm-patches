# SPDX-License-Identifier: Apache-2.0
"""Audit rules database — exhaustive catalog of "X requires Y" relationships
across patches, model classes, KV dtypes, and spec_decode setups.

Each rule describes a CONDITION (triggered by config attributes) and a
REQUIREMENT (env vars / patches / settings that must be present). When
the condition matches but the requirement is missing, audit() emits a
warning naming the rule.

Rules sourced from:
  - CHANGELOG.md (P98 requires for TQ k8v4 hybrid, etc.)
  - PATCH_REGISTRY conflicts_with / requires_patches fields
  - docs/_internal/* operator runbooks (when available)
  - Empirical bench history

This file is the single source of truth for "what should be on for
this combination" — operators no longer rely on tribal knowledge.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class AuditRule:
    """One audit rule.

    `applies_to(cfg)` returns True if this rule's condition triggers.
    `check(cfg)` returns None if compliant or a string explaining the
    violation. `severity` is 'error' / 'warning' / 'info'.
    """
    rule_id: str
    title: str
    description: str
    severity: str  # 'error' / 'warning' / 'info'
    applies_to: Callable[..., bool]  # cfg → bool
    check: Callable[..., Optional[str]]  # cfg → None | str (violation message)


# ─── Helper predicates ─────────────────────────────────────────────────


def _is_hybrid_gdn(cfg) -> bool:
    """Heuristic: hybrid GDN models (Qwen3.5/3.6 with mamba layers)."""
    p = cfg.model_path.lower()
    # 27B Lorbus INT4 = hybrid; 35B-A3B-FP8 = dense MoE (NOT hybrid)
    if "int4-autoround" in p or "lorbus" in p:
        return True
    if cfg.genesis_env.get("GENESIS_ENABLE_PN59_STREAMING_GDN") == "1":
        return True
    return False


def _is_moe_fp8(cfg) -> bool:
    """MoE + FP8 stack (e.g. 35B-A3B-FP8)."""
    p = cfg.model_path.lower()
    return "a3b" in p and "fp8" in p


def _is_marlin_path(cfg) -> bool:
    """Quantization that routes through Marlin kernel."""
    if cfg.quantization in ("auto_round", "gptq", "awq"):
        return True
    if "marlin" in (cfg.kv_cache_dtype or "").lower():
        return True
    return False


def _has_spec_decode(cfg) -> bool:
    return cfg.spec_decode is not None


def _is_tq_k8v4(cfg) -> bool:
    return cfg.kv_cache_dtype == "turboquant_k8v4"


def _has_long_ctx(cfg, threshold: int = 65536) -> bool:
    return cfg.max_model_len >= threshold


def _has_chunked_prefill(cfg) -> bool:
    return cfg.enable_chunked_prefill


def _env_on(cfg, key: str) -> bool:
    return cfg.genesis_env.get(key) == "1"


def _env_off_or_missing(cfg, key: str) -> bool:
    return cfg.genesis_env.get(key) != "1"


# Empirical per-request mamba state size for hybrid GDN models.
# Sources: boot-log peak deltas observed on Qwen3.6-27B Lorbus + 35B-A3B.
# Conservative — actual is ~200-260 MiB; 250 leaves margin against drift.
_R018_MAMBA_STATE_PER_REQUEST_MIB = 250

# Threshold = fraction of per-GPU budget reserved purely for mamba state.
# 30% leaves 70% for weights (largest), KV pool, CG capture, activations.
_R018_BUDGET_FRACTION = 0.30


def _check_hybrid_capacity_R018(cfg) -> Optional[str]:
    """R-018 helper: warn on hybrid mamba REQUEST_CONSTANT capacity overflow.

    Mamba state is replicated across all TP ranks (NOT sharded), so the
    per-GPU footprint is `max_num_seqs × per-request state` regardless of
    n_gpus. If this exceeds `_R018_BUDGET_FRACTION` of per-GPU budget,
    the rest of the stack (weights + KV pool + CG capture + activations)
    has no headroom and OOM is likely at boot or first request.

    Phase D (2026-05-07): if `cfg.reference_metrics.mamba_state_mib_per_request`
    is set (empirical-bake), use that exact value. Otherwise fall back to the
    250 MiB conservative heuristic.
    """
    # Phase D empirical-bake — prefer model-specific measured value if available
    per_request_mib = _R018_MAMBA_STATE_PER_REQUEST_MIB
    source = "heuristic"
    ref = getattr(cfg, "reference_metrics", None)
    if ref is not None:
        baked = getattr(ref, "mamba_state_mib_per_request", None)
        if baked is not None and baked > 0:
            per_request_mib = baked
            source = "empirical"

    estimated_mamba_per_gpu = cfg.max_num_seqs * per_request_mib
    budget_per_gpu = cfg.hardware.min_vram_per_gpu_mib * cfg.gpu_memory_utilization
    safe_threshold = budget_per_gpu * _R018_BUDGET_FRACTION

    if estimated_mamba_per_gpu > safe_threshold:
        return (
            f"hybrid mamba state budget tight: "
            f"max_num_seqs={cfg.max_num_seqs} × "
            f"~{per_request_mib:.0f} MiB ({source}) ≈ "
            f"{estimated_mamba_per_gpu:.0f} MiB per-GPU mamba state "
            f"(>{int(_R018_BUDGET_FRACTION * 100)}% of budget "
            f"{int(budget_per_gpu)} MiB). Consider: lower max_num_seqs, "
            f"raise gpu_memory_utilization, or use larger-VRAM hardware."
        )
    return None


# ─── R-019 helpers (W-runtime 2026-05-06) ─────────────────────────────


def _has_symbolic_mounts(cfg) -> bool:
    """Config has any ${var} reference in docker.mounts."""
    if cfg.docker is None or not cfg.docker.mounts:
        return False
    return any("${" in m for m in cfg.docker.mounts)


def _check_symbolic_mounts_resolvable_R019(cfg) -> Optional[str]:
    """All ${var} in docker.mounts must be resolvable via host.yaml or auto-detect.

    Returns None if all vars resolve, else error string listing missing vars.

    F-013/F-016 fix (audit 2026-05-07): builtin configs are now symbolic
    too, so `host.yaml` may be missing or partial in fresh checkouts and
    test environments. The check now merges `detect_paths()` (auto-probe)
    with `load_host_config()` so the audit doesn't false-positive on
    a newly-cloned repo where install.sh hasn't run yet — auto-probe
    finds anything sitting at canonical default locations, and only
    truly-missing vars surface as errors.
    """
    import re
    var_pat = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
    referenced: set[str] = set()
    for m in cfg.docker.mounts:
        referenced.update(var_pat.findall(m))
    if not referenced:
        return None

    # Merge: auto-detect first, host.yaml overrides on collision.
    available: dict[str, str] = {}
    try:
        from vllm.sndr_core.model_configs.host import detect_paths
        available.update(detect_paths())
    except Exception:
        pass
    try:
        from vllm.sndr_core.model_configs.host import load_host_config
        available.update(load_host_config().paths)
    except Exception:
        pass

    missing = referenced - set(available.keys())
    if not missing:
        return None
    return (
        f"symbolic mounts reference {sorted(missing)} but neither "
        f"host.yaml nor auto-detect supplied them on this host. "
        f"Run `genesis install` OR add them to ~/.genesis/host.yaml "
        f"`paths:` section. Available now: {sorted(available.keys())}"
    )


# ─── Rules database ────────────────────────────────────────────────────


RULES: list[AuditRule] = [
    # ─── Hybrid GDN + TQ ──────────────────────────────────────────
    AuditRule(
        rule_id="R-001",
        title="P98 required for TQ k8v4 + hybrid GDN",
        description=(
            "vllm#40941 introduced WorkspaceManager._locked flag that "
            "fires AssertionError on first decode for TQ + hybrid + MTP. "
            "P98 reverts the lock semantics. Without P98, container "
            "boots but crashes on first inference call."
        ),
        severity="error",
        applies_to=lambda cfg: _is_tq_k8v4(cfg) and _is_hybrid_gdn(cfg),
        check=lambda cfg: (
            None if _env_on(cfg, "GENESIS_ENABLE_P98")
            else "Set GENESIS_ENABLE_P98=1 (WorkspaceManager fix)"
        ),
    ),

    # ─── Spec decode + TQ k8v4 ────────────────────────────────────
    AuditRule(
        rule_id="R-002",
        title="P67 (multi-query kernel) required for spec_decode + TQ",
        description=(
            "TQ k8v4 spec-decode K+1 verify requires P67 multi-query "
            "kernel routing for correctness + perf. Without it, spec "
            "decode falls through broken `_prefill_attention` .tolist() "
            "path on non-pow-2 GQA."
        ),
        severity="warning",
        applies_to=lambda cfg: (
            _has_spec_decode(cfg) and _is_tq_k8v4(cfg)
        ),
        check=lambda cfg: (
            None if _env_on(cfg, "GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL")
            else "Recommend GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL=1"
        ),
    ),

    # ─── Spec decode + cudagraph ──────────────────────────────────
    AuditRule(
        rule_id="R-003",
        title="P58 required for spec_decode + cudagraph (async-scheduler -1)",
        description=(
            "vllm#40768 backport. Without P58, async scheduler hands the "
            "draft model a -1 placeholder which crashes downstream."
        ),
        severity="warning",
        applies_to=lambda cfg: (
            _has_spec_decode(cfg) and not cfg.enforce_eager
        ),
        check=lambda cfg: (
            None if _env_on(cfg, "GENESIS_ENABLE_P58_ASYNC_PLACEHOLDER_FIX")
            else "Set GENESIS_ENABLE_P58_ASYNC_PLACEHOLDER_FIX=1"
        ),
    ),

    AuditRule(
        rule_id="R-004",
        title="P60+P60B required for GDN+ngram spec decode",
        description=(
            "vllm#40738 (tdoublep). Without these, GDN+ngram spec produces "
            "stale state. P60 = SSM pre-copy; P60B = Triton kernel offset fix."
        ),
        severity="warning",
        applies_to=lambda cfg: (
            _has_spec_decode(cfg)
            and cfg.spec_decode.method == "ngram"
            and _is_hybrid_gdn(cfg)
        ),
        check=lambda cfg: (
            None if (_env_on(cfg, "GENESIS_ENABLE_P60_GDN_NGRAM_FIX")
                     and _env_on(cfg, "GENESIS_ENABLE_P60B_TRITON_KERNEL"))
            else "Set GENESIS_ENABLE_P60_GDN_NGRAM_FIX=1 + "
                 "GENESIS_ENABLE_P60B_TRITON_KERNEL=1"
        ),
    ),

    # ─── Long-ctx GDN ─────────────────────────────────────────────
    AuditRule(
        rule_id="R-005",
        title="PN59 streaming-GDN recommended for long-ctx hybrid",
        description=(
            "Cliff 2b OOM risk: at 60K+ context the per-layer h-tensor "
            "allocation reaches 805 MiB. PN59 streams in window "
            "iterations, reducing peak alloc by ~95%."
        ),
        severity="warning",
        applies_to=lambda cfg: _is_hybrid_gdn(cfg) and _has_long_ctx(cfg),
        check=lambda cfg: (
            None if _env_on(cfg, "GENESIS_ENABLE_PN59_STREAMING_GDN")
            else "Recommend GENESIS_ENABLE_PN59_STREAMING_GDN=1 "
                 "for context >= 64K"
        ),
    ),

    # ─── Marlin path (quantized weights, NOT native FP8) ──────────
    AuditRule(
        rule_id="R-006",
        title="P87 + P91 required for Marlin-path quantized weights",
        description=(
            "P87 sub-tile pad-on-load + P91 row-parallel scales. "
            "Required for AutoRound/GPTQ/AWQ on Ampere Marlin kernel. "
            "Native FP8 (W8A8) does NOT use Marlin — these patches are "
            "no-ops there."
        ),
        severity="info",
        applies_to=lambda cfg: _is_marlin_path(cfg),
        check=lambda cfg: (
            None if (_env_on(cfg, "GENESIS_ENABLE_P87")
                     and _env_on(cfg, "GENESIS_ENABLE_P91"))
            else "Recommend GENESIS_ENABLE_P87=1 + GENESIS_ENABLE_P91=1 "
                 "for Marlin path"
        ),
    ),

    # ─── Chunked prefill + MoE ────────────────────────────────────
    AuditRule(
        rule_id="R-007",
        title="P72 required for chunked-prefill on MoE with batched > 4096",
        description=(
            "P72 caps profile_run M to 4096. Without it, MoE profile_run "
            "OOMs trying to estimate full --max-num-batched-tokens. "
            "Required when --max-num-batched-tokens > 4096 + MoE."
        ),
        severity="warning",
        applies_to=lambda cfg: (
            _is_moe_fp8(cfg) and cfg.max_num_batched_tokens > 4096
            and _has_chunked_prefill(cfg)
        ),
        check=lambda cfg: (
            None if _env_on(cfg, "GENESIS_ENABLE_P72_PROFILE_RUN_CAP")
            else "Set GENESIS_ENABLE_P72_PROFILE_RUN_CAP=1 + "
                 "GENESIS_PROFILE_RUN_CAP_M=4096"
        ),
    ),

    # ─── Qwen3 tool-calling ────────────────────────────────────────
    AuditRule(
        rule_id="R-008",
        title="P61/P64/P68/P69 recommended for Qwen3 tool-call quality",
        description=(
            "Empirical: tool_call clean rate stock 2-6/10 → Genesis 10/10 "
            "comes from this stack. Each patch addresses a specific "
            "Qwen3 parser bug."
        ),
        severity="info",
        applies_to=lambda cfg: (
            cfg.tool_call_parser == "qwen3_coder"
            and cfg.enable_auto_tool_choice
        ),
        check=lambda cfg: _check_qwen3_tool_stack(cfg),
    ),

    # ─── Hybrid GDN + prefix caching (DANGER) ─────────────────────
    AuditRule(
        rule_id="R-009",
        title="--enable-prefix-caching crashes hybrid GDN with TQ + spec",
        description=(
            "DS conv state layout error on hybrid GDN when "
            "--enable-prefix-caching + MTP + accept>1. Empirically "
            "verified 2026-04-28 (memory record)."
        ),
        severity="error",
        applies_to=lambda cfg: _is_hybrid_gdn(cfg) and _is_tq_k8v4(cfg),
        check=lambda cfg: (
            "DO NOT set --enable-prefix-caching for hybrid GDN + TQ k8v4 "
            "+ spec_decode (DS conv state crash)"
            if "--enable-prefix-caching" in cfg.vllm_extra_args
            else None
        ),
    ),

    # ─── 27B GQA=6 needs P67 v7.63.x non-pow-2 split-M fix ────────
    AuditRule(
        rule_id="R-010",
        title="27B Lorbus INT4 + TQ k8v4 requires P67 v7.63.x non-pow-2 GQA fix",
        description=(
            "27B has GQA=24/4=6 (non-pow-2). Pre-v7.63.x P67 split-M "
            "kernel rejected non-pow-2 head_dim and fell through to "
            "broken upstream _prefill_attention .tolist() during "
            "cudagraph capture. Fix landed in v7.63.x_nopow2_gqa "
            "(BLOCK_QH=next_power_of_2(hpk) + lane_valid mask). "
            "vllm default cudagraph_mode=FULL_AND_PIECEWISE is fine "
            "with that fix in place — Genesis stack does NOT inject "
            "any --cudagraph-mode flag (the flag doesn't exist in "
            "this vllm pin anyway)."
        ),
        severity="warning",
        applies_to=lambda cfg: (
            _is_tq_k8v4(cfg) and "int4-autoround" in cfg.model_path.lower()
        ),
        check=lambda cfg: _check_known_good_genesis_pin(cfg),
    ),

    # ─── Schema cross-check: typo in env name ─────────────────────
    AuditRule(
        rule_id="R-011",
        title="genesis_env keys must exist in PATCH_REGISTRY",
        description=(
            "Catches typos like GENESIS_ENABLE_PXX9 that would silently "
            "be ignored at runtime."
        ),
        severity="error",
        applies_to=lambda cfg: bool(cfg.genesis_env),
        check=lambda cfg: _check_env_keys_exist(cfg),
    ),

    # ─── VRAM math sanity ─────────────────────────────────────────
    AuditRule(
        rule_id="R-012",
        title="VRAM math sanity (rough estimate)",
        description=(
            "Quick check: model_size + KV at max context fits in "
            "n_gpus × min_vram × gpu_memory_utilization. Rough heuristic "
            "to catch obviously-too-big configs early."
        ),
        severity="warning",
        applies_to=lambda cfg: cfg.hardware.min_vram_per_gpu_mib > 0,
        check=lambda cfg: _check_vram_math(cfg),
    ),

    # ─── Pin gate (config required pin must be in allowlist) ──────
    AuditRule(
        rule_id="R-013",
        title="vllm_pin_required must be in KNOWN_GOOD_VLLM_PINS",
        description=(
            "If config pins a vllm version not on Genesis's allowlist, "
            "operators will hit the pin-gate at boot. Either bump "
            "allowlist (PR review) or change config."
        ),
        severity="error",
        applies_to=lambda cfg: cfg.vllm_pin_required is not None,
        check=lambda cfg: _check_pin_in_allowlist(cfg),
    ),

    # ─── Reference metrics provenance ─────────────────────────────
    AuditRule(
        rule_id="R-014",
        title="reference_metrics.vllm_pin should match vllm_pin_required",
        description=(
            "If config requires pin X but reference was bench'd on pin Y, "
            "verify will likely fail or compare apples to oranges."
        ),
        severity="warning",
        applies_to=lambda cfg: (
            cfg.reference_metrics is not None
            and cfg.vllm_pin_required is not None
        ),
        check=lambda cfg: (
            None if cfg.reference_metrics.vllm_pin == cfg.vllm_pin_required
            else f"reference_metrics.vllm_pin "
                 f"({cfg.reference_metrics.vllm_pin}) != "
                 f"vllm_pin_required ({cfg.vllm_pin_required})"
        ),
    ),

    # ─── Stable lifecycle requires reference_metrics ──────────────
    AuditRule(
        rule_id="R-015",
        title="stable lifecycle requires reference_metrics",
        description=(
            "Operators can't run `verify` against a stable config without "
            "baseline numbers. Either bench it (`bench-and-update`) or "
            "demote to lifecycle: experimental."
        ),
        severity="warning",
        applies_to=lambda cfg: cfg.lifecycle == "stable",
        check=lambda cfg: (
            None if cfg.reference_metrics is not None
            else "stable lifecycle without reference_metrics — operators "
                 "can't run verify. Run bench-and-update or demote to "
                 "lifecycle: experimental."
        ),
    ),

    # ─── Genesis pin pinning ──────────────────────────────────────
    AuditRule(
        rule_id="R-016",
        title="genesis_pin should be set for stable configs",
        description=(
            "Without genesis_pin, operators can't reproduce the exact "
            "patcher state that was bench'd. Required for community "
            "PRs."
        ),
        severity="info",
        applies_to=lambda cfg: cfg.lifecycle == "stable",
        check=lambda cfg: (
            None if cfg.genesis_pin
            else "Set genesis_pin to git short SHA at time of bench"
        ),
    ),

    # ─── cudagraph_mode drift detector ────────────────────────────
    AuditRule(
        rule_id="R-017",
        title="cudagraph_mode = FULL_AND_PIECEWISE on Genesis stack",
        description=(
            "Genesis stack standardizes on vllm default "
            "FULL_AND_PIECEWISE — covers both decode-only FULL graph "
            "and mixed prefill/decode PIECEWISE. Other modes work but "
            "are not bench-validated, so flag the divergence so the "
            "operator can confirm they really meant it."
        ),
        severity="warning",
        applies_to=lambda cfg: not cfg.enforce_eager,
        check=lambda cfg: (
            None if cfg.cudagraph_mode == "FULL_AND_PIECEWISE"
            else (
                f"cudagraph_mode='{cfg.cudagraph_mode}' is non-default. "
                f"Genesis configs ship with FULL_AND_PIECEWISE — "
                f"set enforce_eager: true if you really want eager "
                f"fallback, else revert to FULL_AND_PIECEWISE."
            )
        ),
    ),

    # ─── R-018: hybrid mamba REQUEST_CONSTANT capacity ────────────
    # Genesis-side equivalent of upstream #41495 phase 2 cudagraph
    # capacity validator (Sander 2026-05-07: chose Genesis-side R-018
    # over full PN75/PN76 backport because R-018 = zero core touch).
    #
    # Why: hybrid mamba models (Qwen3.5/3.6) allocate REQUEST_CONSTANT
    # state per active request — independent of sequence length, but
    # replicated across ALL TP ranks (not sharded). High `max_num_seqs`
    # × per-request mamba state can OOM at boot or at first cudagraph
    # capture. Empirical mamba state per request on Qwen3.6-27B Lorbus
    # hybrid INT4 ≈ 250 MiB (41 mamba layers × ssm_state + conv_state).
    #
    # Heuristic: warn if `max_num_seqs × 250 MiB` exceeds 30% of
    # per-GPU budget (`min_vram_per_gpu_mib × gpu_memory_utilization`).
    # 30% threshold is conservative — leaves headroom for weights, KV
    # pool, CG capture, activations.
    #
    # Auto-retire path: when vllm#41495 lands and we bump pin, this
    # rule's classifier `_is_hybrid_gdn` still fires but the runtime
    # validator from upstream catches the same class earlier with
    # better precision. Genesis rule remains as an authoring-time gate.
    AuditRule(
        rule_id="R-018",
        title="hybrid mamba REQUEST_CONSTANT state capacity",
        description=(
            "Hybrid mamba models (Qwen3.5/3.6 with Mamba layers) "
            "allocate ~200-300 MiB of REQUEST_CONSTANT mamba state "
            "per active request, replicated across all TP ranks. "
            "High max_num_seqs × per-request state can OOM at boot. "
            "Catches misconfigurations (e.g. max_num_seqs=32 on a "
            "24 GB single-GPU hybrid config) before deploy. "
            "Genesis-side equivalent of upstream #41495 phase 2 "
            "validator (chosen 2026-05-07 over full backport because "
            "zero vllm core touch + zero pin-bump fragility)."
        ),
        severity="warning",
        applies_to=lambda cfg: _is_hybrid_gdn(cfg),
        check=_check_hybrid_capacity_R018,
    ),
    AuditRule(
        rule_id="R-019",
        title="unresolved ${var} in mounts requires host.yaml",
        description=(
            "W-runtime 2026-05-06: configs may reference paths via "
            "symbolic ${var} (e.g. ${models_dir}, ${hf_cache}) for "
            "portability across community rigs. Render layer resolves "
            "via ~/.genesis/host.yaml. This rule fires if mounts contain "
            "${var} but neither host.yaml nor auto-detect can supply it. "
            "F-013 (audit 2026-05-07): downgraded to WARNING — builtin "
            "configs are now also symbolic, so the rule fires on fresh "
            "checkouts before `genesis install` has a chance to write "
            "host.yaml. The actual deploy-time resolution check lives "
            "in preflight (`genesis model-config preflight`), which is "
            "the right gate to block a real launch."
        ),
        severity="warning",
        applies_to=lambda cfg: _has_symbolic_mounts(cfg),
        check=_check_symbolic_mounts_resolvable_R019,
    ),
]


# ─── Helpers ───────────────────────────────────────────────────────────


def _check_qwen3_tool_stack(cfg) -> Optional[str]:
    needed = {
        "GENESIS_ENABLE_P61_QWEN3_MULTI_TOOL": "P61 multi-tool first-occurrence",
        "GENESIS_ENABLE_P64_QWEN3CODER_MTP_STREAMING":
            "P64 qwen3coder MTP streaming",
        "GENESIS_ENABLE_P68_AUTO_FORCE_TOOL": "P68 auto force tool_choice",
        "GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER":
            "P69 long-ctx tool reminder",
    }
    missing = [
        f"{k} ({desc})" for k, desc in needed.items()
        if not _env_on(cfg, k)
    ]
    if not missing:
        return None
    return (
        "Recommend enabling Qwen3 tool-call stack:\n    "
        + "\n    ".join(missing)
    )


def _check_env_keys_exist(cfg) -> Optional[str]:
    """Validate every key in genesis_env against PATCH_REGISTRY env_flags."""
    try:
        from vllm.sndr_core.dispatcher import PATCH_REGISTRY
    except Exception:
        return None  # registry not available; skip

    # Build set of all known env flag names
    known_flags = set()
    for pid, meta in PATCH_REGISTRY.items():
        flag = meta.get("env_flag")
        if flag:
            known_flags.add(flag)

    # Tunable knobs (vs patch-enable flags) live in their own registry —
    # see `vllm/sndr_core/runtime_tunables.py::TUNABLE_KNOBS` for the
    # canonical list with type / default / doc per knob.
    from vllm.sndr_core.runtime_tunables import is_known_tunable

    # Stub keys used as marker placeholders (e.g. _DUP suffix).
    stub_suffixes = ("_DUP",)

    unknown = []
    for k in cfg.genesis_env:
        if k in known_flags:
            continue
        if is_known_tunable(k):
            continue
        if any(k.endswith(s) for s in stub_suffixes):
            continue
        unknown.append(k)

    if not unknown:
        return None
    return (
        "genesis_env contains unknown keys (typo or removed patch?):\n"
        "    " + "\n    ".join(unknown)
    )


def _check_vram_math(cfg) -> Optional[str]:
    """Rough memory math check.

    Heuristic only — actual usage depends on activations, KV layout,
    workspace etc. Catches obviously-too-big configs (e.g. 35B FP8 on
    1× 24GB).
    """
    # Rough model size estimate in bytes
    est = _estimate_model_size_bytes(cfg)
    if est == 0:
        return None  # unknown model — skip

    total_mib_available = (
        cfg.hardware.n_gpus * cfg.hardware.min_vram_per_gpu_mib
        * cfg.gpu_memory_utilization
    )
    est_mib = est / (1024 * 1024)
    # KV memory per 1K tokens depends on dtype:
    #   fp16 → ~50 MiB/1K (baseline)
    #   fp8_e5m2 → ~25 MiB/1K (half)
    #   turboquant_k8v4 → ~10 MiB/1K (packed slot layout)
    kv_dtype = cfg.kv_cache_dtype or "fp16"
    kv_mib_per_1k = {
        "fp16": 50.0, "bf16": 50.0,
        "fp8_e5m2": 25.0, "fp8_e4m3": 25.0, "fp8": 25.0,
        "turboquant_k8v4": 10.0, "turboquant_3bit_nc": 8.0,
        "turboquant_4bit_nc": 9.0,
    }.get(kv_dtype, 50.0)
    # Realistic working ctx: vllm rarely fills max_model_len in one slot
    # (multi-seq sharing); discount to ~50% for the heuristic
    effective_ctx_k = (cfg.max_model_len / 1024) * 0.5
    kv_est_mib = effective_ctx_k * kv_mib_per_1k * cfg.max_num_seqs
    total_needed = est_mib + kv_est_mib

    if total_needed > total_mib_available * 1.15:  # 15% headroom
        return (
            f"VRAM math fails rough check: model ~{est_mib:.0f} MiB + "
            f"KV ~{kv_est_mib:.0f} MiB = {total_needed:.0f} MiB needed, "
            f"but {total_mib_available:.0f} MiB available "
            f"({cfg.hardware.n_gpus}× {cfg.hardware.min_vram_per_gpu_mib} "
            f"MiB × {cfg.gpu_memory_utilization})"
        )
    return None


def _estimate_model_size_bytes(cfg) -> int:
    """Rough param count × bytes/param estimate. Returns 0 if unknown."""
    p = cfg.model_path.lower()
    # Param count guesses
    n_params_b = 0
    if "35b" in p:
        n_params_b = 35
    elif "27b" in p:
        n_params_b = 27
    elif "14b" in p:
        n_params_b = 14
    elif "7b" in p:
        n_params_b = 7
    if n_params_b == 0:
        return 0

    # Bytes per param by quant
    bpp = 2.0  # default fp16
    if "int4" in p or cfg.quantization == "auto_round":
        bpp = 0.5
    elif "fp8" in p or cfg.quantization == "fp8":
        bpp = 1.0
    elif cfg.quantization in ("gptq", "awq"):
        bpp = 0.5

    return int(n_params_b * 1_000_000_000 * bpp)


def _check_pin_in_allowlist(cfg) -> Optional[str]:
    """vllm_pin_required must appear in KNOWN_GOOD_VLLM_PINS."""
    try:
        from vllm.sndr_core.detection.guards import KNOWN_GOOD_VLLM_PINS
    except Exception:
        return None
    if cfg.vllm_pin_required in KNOWN_GOOD_VLLM_PINS:
        return None
    return (
        f"vllm_pin_required '{cfg.vllm_pin_required}' is NOT in "
        f"KNOWN_GOOD_VLLM_PINS. Either bump the allowlist (PR to "
        f"guards.py) or change config.vllm_pin_required to a known pin: "
        f"{list(KNOWN_GOOD_VLLM_PINS)}"
    )


def _check_known_good_genesis_pin(cfg) -> Optional[str]:
    """R-010 helper — genesis_pin must be in KNOWN_GOOD_GENESIS_PINS,
    OR enforce_eager fallback engaged. Used to gate 27B INT4 + TQ k8v4
    configs against the P67 non-pow-2 GQA split-M fix presence.
    """
    if cfg.enforce_eager:
        return None
    try:
        from vllm.sndr_core.detection.guards import is_genesis_pin_validated
    except Exception:
        return None
    if is_genesis_pin_validated(cfg.genesis_pin):
        return None
    return (
        f"27B INT4 + TQ k8v4 requires the P67 non-pow-2 GQA split-M fix "
        f"(v7.63.x_nopow2_gqa, present in commits ≥ 991dc1a). Current "
        f"genesis_pin='{cfg.genesis_pin}' is NOT in KNOWN_GOOD_GENESIS_PINS. "
        f"Either bump the allowlist in guards.py (PR review) or set "
        f"enforce_eager: true as fallback."
    )


# ─── Public entry point ────────────────────────────────────────────────


def audit(cfg) -> list[tuple[str, str, str, str]]:
    """Run all rules against cfg.

    Returns list of (rule_id, severity, title, message) for each
    triggered rule whose check fails. Empty list = clean.
    """
    issues: list[tuple[str, str, str, str]] = []
    for rule in RULES:
        try:
            if not rule.applies_to(cfg):
                continue
            msg = rule.check(cfg)
            if msg:
                issues.append((rule.rule_id, rule.severity, rule.title, msg))
        except Exception as e:
            # Defensive — a buggy rule shouldn't break the whole audit
            issues.append((
                rule.rule_id, "error",
                f"rule {rule.rule_id} raised", str(e),
            ))
    return issues
