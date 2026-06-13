# SPDX-License-Identifier: Apache-2.0
"""PN377 — moe_wna16 BLOCK_SIZE_K legality clamp (vendor of OPEN vllm#44563).

Problem (upstream #36008, fix proposed in OPEN PR vllm#44563)
------------------------------------------------------------------
GPTQ/AWQ int4 MoE models with ``group_size=32`` crash on the CUDA
``moe_wna16_gemm`` path with ``RuntimeError: BLOCK_SIZE_K // group_size
must be one of [1, 2, 4, 8]``. Root cause:
``get_moe_wna16_block_config`` (model_executor/layers/fused_moe/
fused_moe.py) can grow ``BLOCK_SIZE_K`` to 512 for small decode batches
(the ``num_m_blocks <= 16`` doubling step fires whenever
``num_valid_tokens <= 16``); with ``group_size=32`` the ratio becomes
16 on the FIRST small forward — i.e. a deterministic warmup abort, not
a tail-risk. ``group_size`` 64/128 can never overshoot (thresholds
512/1024 are never exceeded by the 512 cap), so already-legal configs
are untouched by construction.

Why we care on this rig (verified at pin g303916e93 / 0.22.1rc1.dev259)
------------------------------------------------------------------
The moe_wna16 path is the LIVE Marlin fallback for our model family:
``awq_marlin.py`` (check_moe_marlin_supports_layer fails -> "Falling
back to Moe WNA16 kernels", line ~307) and ``auto_gptq.py`` (same
fallback, line ~242) both route RoutedExperts layers to
``MoeWNA16Config`` when Marlin cannot take them. A gs=32 int4 MoE quant
of the Qwen3.6 family (e.g. the #36008 reporter's Qwen3.5-35B-A3B-GPTQ
gate_up gemm: size_k=2048, group_size=32) aborts at warmup on the pin.
PN377 unblocks gs=32 int4 MoE quant benchmarking (roadmap 2026-06-11,
chunk-5 Theme D).

Fix (the PR's 4-line clamp, adapted)
------------------------------------------------------------------
Cap ``block_size_k`` at ``group_size * 8`` immediately BEFORE the
existing ``_ensure_block_size_k_divisible`` step, so the divisor search
starts from a kernel-legal value and its result keeps the ratio <= 8.
The clamp can only ever rewrite an otherwise-illegal config. Upstream's
``num_experts=B.size(1)`` call-site oddity (#36026) is deliberately NOT
vendored — per the PR's measured analysis it does not fix the crash and
is a small perf regression where it changes anything; it belongs to the
#40547 heuristic cleanup (P24 MoE-tiling research track).

Genesis extra: boot-time legality assert for the actual model grid
------------------------------------------------------------------
``run_boot_legality_check`` extracts the two heuristic functions from
the target file AS PATCHED ON DISK (post-PN377, idempotent re-boot, or
a future upstream-merged form alike — both are pure Python at this
pin), sweeps them over the ACTUAL model's MoE GEMM grid (gate_up + w2
shapes derived from hf_config, both TP=1 and the configured TP, real
expert count plus the buggy ``B.size(1)`` surrogate upstream currently
passes as ``num_experts``), and asserts the three moe_wna16_gemm
legality invariants. A violation produces a loud, actionable ERROR log
at boot (what would crash, why, and the remedies) instead of the
cryptic warmup abort. Grid discovery is best-effort: dense models,
non-4-bit quants, group_size not in {32, 64, 128}, or an unavailable
vllm config skip the check quietly.

Env gating
----------
Default ON (the clamp only rewrites kernel-illegal configs — provably
inert for every current PROD model: 35B FP8 never takes the wna16
path; gs>=64 AWQ MoE can never overshoot). Set
``GENESIS_ENABLE_PN377_MOE_WNA16_BSK_CLAMP=0`` to skip the install.
Install is additionally gated on ``is_moe_model()`` (P52 dispatch, P24
pattern) — dense models never dispatch fused_moe.

Drift markers: #44563's exact clamp line and comment. Our emitted text
uses the ``_g_pn377_`` prefix and different comment wording so the
markers can never fire on our own text (lint_drift_markers contract).

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import ast
import logging
import os
from pathlib import Path
from typing import Any, Callable, Sequence

from sndr.engines.vllm.detection.guards import resolve_vllm_file
from sndr.kernel import TextPatch, TextPatcher, result_to_wiring_status

log = logging.getLogger("genesis.wiring.pn377_moe_wna16_bsk_clamp")

GENESIS_PN377_MARKER = "Genesis PN377 moe_wna16 BLOCK_SIZE_K legality clamp v1"

_TARGET_REL = "model_executor/layers/fused_moe/fused_moe.py"
_ENV_FLAG = "GENESIS_ENABLE_PN377_MOE_WNA16_BSK_CLAMP"

# #44563's exact added clamp line and comment. PN377's own replacement
# deliberately uses the _g_pn377_ variable prefix and different comment
# wording, so neither marker can ever match our own emitted text.
_UPSTREAM_DRIFT_MARKERS = [
    "max_block_size_k = group_size * 8",
    "Clamp to at most 8 groups per block row",
]

# Anchor: the existing divisibility step inside get_moe_wna16_block_config
# (single occurrence at the pin; the def site of the helper does not
# contain this two-line call form). P24 anchors live in
# get_default_config — disjoint text, byte-verified non-colliding.
PN377_CLAMP_OLD = (
    "        # Ensure BLOCK_SIZE_K is a divisor of size_k"
    " for CUDA kernel compatibility\n"
    "        block_size_k = _ensure_block_size_k_divisible"
    "(size_k, block_size_k, group_size)"
)

PN377_CLAMP_NEW = (
    "        # [Genesis PN377] moe_wna16_gemm legality clamp (vendor of OPEN\n"
    "        # vllm#44563, fixes #36008): the CUDA kernel aborts unless\n"
    "        # BLOCK_SIZE_K // group_size is one of (1, 2, 4, 8), and the\n"
    "        # heuristic above can overshoot for group_size=32 models\n"
    "        # (block_size_k reaches 512 -> ratio 16 on the first small\n"
    "        # decode batch, killing serving at warmup). Cap the ratio at 8\n"
    "        # BEFORE the divisibility step so the divisor search starts\n"
    "        # from a kernel-legal value. group_size 64/128 thresholds\n"
    "        # (512/1024) are never exceeded -> legal configs unchanged.\n"
    "        _g_pn377_bsk_cap = group_size * 8\n"
    "        if block_size_k > _g_pn377_bsk_cap:\n"
    "            block_size_k = _g_pn377_bsk_cap\n"
    "\n"
    "        # Ensure BLOCK_SIZE_K is a divisor of size_k"
    " for CUDA kernel compatibility\n"
    "        block_size_k = _ensure_block_size_k_divisible"
    "(size_k, block_size_k, group_size)"
)


def _enabled() -> bool:
    """Default ON; explicit 0/false/no/off skips the install."""
    return os.environ.get(_ENV_FLAG, "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_TARGET_REL)
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN377 moe_wna16 BLOCK_SIZE_K legality clamp",
        target_file=str(target),
        marker=GENESIS_PN377_MARKER,
        sub_patches=[
            TextPatch(
                name="pn377_bsk_clamp",
                anchor=PN377_CLAMP_OLD,
                replacement=PN377_CLAMP_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=list(_UPSTREAM_DRIFT_MARKERS),
    )


# ── Genesis extra: boot-time legality assert ─────────────────────────

_HEURISTIC_FUNCS = ("_ensure_block_size_k_divisible", "get_moe_wna16_block_config")

_LEGAL_RATIOS = (1, 2, 4, 8)

# Token counts seen between single-token decode and a full prefill
# batch; the small ones (<= 16) are where the doubling step overshoots.
_NUM_VALID_TOKENS_SWEEP = (1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096)

# BLOCK_SIZE_M values upstream's default configs feed into the call
# sites (config["BLOCK_SIZE_M"]) plus the PR test's small-batch floor.
_BLOCK_SIZE_M_SWEEP = (1, 16, 32, 64)


def load_block_config_heuristic(target_file: str) -> Callable[..., Any] | None:
    """Extract ``get_moe_wna16_block_config`` (+ its divisor helper)
    from ``target_file`` via ast and exec them in an isolated namespace.

    Works on the pristine pin form, the PN377-patched form, and
    #44563's merged form alike — both functions are pure Python at this
    pin (no torch / no triton / no platform probe). Returns None when
    either function is missing or the file does not parse (heuristic
    refactored upstream -> the boot check abstains rather than lies).
    """
    try:
        src = Path(target_file).read_text(encoding="utf-8")
        tree = ast.parse(src)
        wanted = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in _HEURISTIC_FUNCS
        ]
        if {node.name for node in wanted} != set(_HEURISTIC_FUNCS):
            return None
        namespace: dict[str, Any] = {}
        module = ast.Module(body=wanted, type_ignores=[])
        exec(compile(module, target_file, "exec"), namespace)  # noqa: S102
        return namespace.get("get_moe_wna16_block_config")
    except Exception as e:
        log.debug("[Genesis PN377] heuristic extraction failed: %s", e)
        return None


def check_block_config_legality(
    get_block_config: Callable[..., Any],
    *,
    group_size: int,
    gemm_shapes: Sequence[tuple[int, int]],
    num_experts_candidates: Sequence[int],
    top_k: int,
    num_valid_tokens_sweep: Sequence[int] = _NUM_VALID_TOKENS_SWEEP,
    block_size_m_sweep: Sequence[int] = _BLOCK_SIZE_M_SWEEP,
) -> list[str]:
    """Sweep the heuristic over a model grid; return violation strings.

    Asserts the three moe_wna16_gemm legality invariants on every
    returned config: ``BLOCK_SIZE_K % group_size == 0``,
    ``size_k % BLOCK_SIZE_K == 0`` and ``BLOCK_SIZE_K // group_size``
    in {1, 2, 4, 8}. Pure CPU, a few hundred dict computations.
    """
    violations: list[str] = []
    for size_k, size_n in gemm_shapes:
        for num_experts in num_experts_candidates:
            for nvt in num_valid_tokens_sweep:
                for bsm in block_size_m_sweep:
                    cfg = get_block_config(
                        config={},
                        use_moe_wna16_cuda=True,
                        num_valid_tokens=nvt,
                        size_k=size_k,
                        size_n=size_n,
                        num_experts=num_experts,
                        group_size=group_size,
                        real_top_k=top_k,
                        block_size_m=bsm,
                    )
                    bsk = cfg.get("BLOCK_SIZE_K")
                    if bsk is None:
                        continue
                    legal = (
                        bsk % group_size == 0
                        and size_k % bsk == 0
                        and bsk // group_size in _LEGAL_RATIOS
                    )
                    if not legal:
                        violations.append(
                            f"size_k={size_k} size_n={size_n}"
                            f" num_experts={num_experts}"
                            f" num_valid_tokens={nvt} block_size_m={bsm}"
                            f" -> BLOCK_SIZE_K={bsk}"
                            f" (group_size={group_size},"
                            f" ratio={bsk // group_size};"
                            f" legal ratios {_LEGAL_RATIOS})"
                        )
    return violations


def _int_attr(obj: Any, *names: str) -> int | None:
    """First positive-int attribute (or dict key) among ``names``."""
    for name in names:
        value = getattr(obj, name, None)
        if value is None and isinstance(obj, dict):
            value = obj.get(name)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _model_wna16_grid() -> dict[str, Any] | None:
    """Best-effort: the actual model's moe_wna16 CUDA GEMM grid.

    Returns None whenever the model cannot hit ``moe_wna16_gemm``
    (config unavailable, no quantization_config, not 4-bit, group_size
    not in {32, 64, 128}, geometry attributes missing) — the boot check
    then abstains. Mirrors ``should_moe_wna16_use_cuda``'s eligibility
    (bit == 4, group_size in [32, 64, 128]) without importing vllm
    model code.
    """
    try:
        from vllm.config import get_current_vllm_config

        cfg = get_current_vllm_config()
        hf = cfg.model_config.hf_config
    except Exception as e:
        log.debug("[Genesis PN377] vllm config unavailable: %s", e)
        return None
    if hf is None:
        return None

    qcfg = getattr(hf, "quantization_config", None)
    if qcfg is None and isinstance(hf, dict):
        qcfg = hf.get("quantization_config")
    if not isinstance(qcfg, dict):
        return None
    bits = qcfg.get("bits") or qcfg.get("weight_bits")
    group_size = qcfg.get("group_size")
    if bits != 4 or group_size not in (32, 64, 128):
        return None

    # Multimodal configs nest the language model under text_config.
    text = getattr(hf, "text_config", None) or hf
    hidden = _int_attr(text, "hidden_size")
    intermediate = _int_attr(text, "moe_intermediate_size", "intermediate_size")
    num_experts = _int_attr(
        text,
        "num_experts", "n_routed_experts", "num_local_experts",
        "moe_num_experts",
    )
    top_k = _int_attr(text, "num_experts_per_tok", "moe_top_k", "top_k") or 1
    if not hidden or not intermediate or not num_experts:
        return None

    try:
        tp = int(cfg.parallel_config.tensor_parallel_size)
    except Exception:
        tp = 1

    # gate_up (w13) and down (w2) GEMM shapes, at TP=1 and the
    # configured TP (the intermediate dim is the sharded one).
    shapes: set[tuple[int, int]] = set()
    for ways in {1, max(tp, 1)}:
        inter_part = max(intermediate // ways, 1)
        shapes.add((hidden, 2 * inter_part))
        shapes.add((inter_part, hidden))

    # Upstream currently passes num_experts=B.size(1) (the N dim — the
    # #36026 oddity, not vendored); sweep BOTH the real expert count and
    # that surrogate so the assert covers the call as it actually fires
    # today AND after a future upstream num_experts correction.
    num_experts_candidates = sorted({num_experts} | {n for _, n in shapes})

    return {
        "group_size": group_size,
        "gemm_shapes": sorted(shapes),
        "num_experts_candidates": num_experts_candidates,
        "top_k": top_k,
    }


def run_boot_legality_check(target_file: str) -> tuple[bool | None, str]:
    """Boot-time legality assert for the actual model grid.

    Returns (verdict, detail): verdict True = grid legal, False =
    violations found (a loud ERROR log has fired), None = check
    abstained (reason in detail). Never raises.
    """
    try:
        grid = _model_wna16_grid()
        if grid is None:
            return None, (
                "model grid unavailable or model cannot hit the"
                " moe_wna16 CUDA path"
            )
        heuristic = load_block_config_heuristic(target_file)
        if heuristic is None:
            return None, (
                "heuristic functions not extractable from"
                f" {target_file} (upstream refactor?)"
            )
        violations = check_block_config_legality(heuristic, **grid)
        if violations:
            shown = "; ".join(violations[:5])
            detail = (
                f"{len(violations)} kernel-illegal BLOCK_SIZE_K configs"
                f" for the model grid {grid} — first 5: {shown}"
            )
            log.error(
                "[Genesis PN377] BOOT LEGALITY CHECK FAILED:"
                " get_moe_wna16_block_config in %s returns kernel-illegal"
                " BLOCK_SIZE_K for this model's MoE grid. Serving WOULD"
                " abort at warmup with RuntimeError: 'BLOCK_SIZE_K //"
                " group_size must be one of [1, 2, 4, 8]' on the"
                " moe_wna16 CUDA path (the Marlin fallback both"
                " awq_marlin.py and auto_gptq.py take for unsupported"
                " RoutedExperts layers). %s. Remedies: keep PN377"
                " applied; re-quantize with group_size 64/128 or a"
                " size_k divisible by the group size; report the grid"
                " on vllm#44563.",
                target_file,
                detail,
            )
            return False, detail
        return True, (
            f"model grid legal: group_size={grid['group_size']},"
            f" {len(grid['gemm_shapes'])} GEMM shapes x"
            f" {len(grid['num_experts_candidates'])} expert counts x"
            f" {len(_NUM_VALID_TOKENS_SWEEP)} token counts x"
            f" {len(_BLOCK_SIZE_M_SWEEP)} block_size_m values"
        )
    except Exception as e:  # pragma: no cover - belt and braces
        log.debug("[Genesis PN377] boot legality check errored: %s", e)
        return None, f"boot legality check errored: {e}"


# ── apply ────────────────────────────────────────────────────────────


def apply() -> tuple[str, str]:
    """Install the clamp + run the boot legality assert. Never raises."""
    if not _enabled():
        return "skipped", (
            f"PN377 install disabled via {_ENV_FLAG}=0 (default ON —"
            " the clamp only rewrites kernel-illegal configs)"
        )

    # P52 MoE dispatch gate (P24 pattern): dense models never dispatch
    # fused_moe, the text would be dead weight.
    try:
        from sndr.engines.vllm.detection.model_detect import (
            is_moe_model,
            log_skip,
        )

        if not is_moe_model():
            log_skip(
                "PN377 moe_wna16 BLOCK_SIZE_K legality clamp",
                "dense model (no fused_moe dispatch)",
            )
            return "skipped", "P52 dispatch: model has no MoE layers"
    except Exception as e:
        log.debug("[Genesis PN377] model_detect probe failed (proceeding): %s", e)

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", f"PN377: target {_TARGET_REL} not resolvable"

    result, failure = patcher.apply()
    status, message = result_to_wiring_status(
        result,
        failure,
        applied_message=(
            "PN377 applied: BLOCK_SIZE_K capped at group_size * 8 before"
            " the divisibility step in get_moe_wna16_block_config —"
            " group_size=32 int4 MoE no longer aborts moe_wna16_gemm at"
            " warmup (vendor of OPEN vllm#44563, fixes #36008; gs 64/128"
            " mathematically unaffected)"
        ),
        patch_name="PN377 moe_wna16 BLOCK_SIZE_K legality clamp",
    )

    # Genesis extra: legality assert against the file AS ON DISK —
    # runs after applied, idempotent re-boot, and upstream-merged
    # drift-skip alike; only a hard text-patch failure bypasses it.
    if status != "failed":
        verdict, detail = run_boot_legality_check(patcher.target_file)
        if verdict is False:
            message += f" | BOOT LEGALITY CHECK FAILED — {detail}"
        elif verdict is True:
            message += f" | boot legality check passed ({detail})"
        else:
            log.debug("[Genesis PN377] boot legality check abstained: %s", detail)

    return status, message
