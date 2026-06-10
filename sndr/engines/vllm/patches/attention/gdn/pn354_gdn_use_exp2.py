# SPDX-License-Identifier: Apache-2.0
"""PN354 — exp2 extension for the GDN chunked-prefill kernel chain.

Mechanism (vllm#43195, merged, KDA-only in our pin)
===================================================

vllm#43195 scales the chunk-local cumulative gate ONCE after cumsum
(``g = g * RCP_LN2``, RCP_LN2 = 1.4426950216 from
``vllm.utils.math_utils``) and uses ``exp2`` in every downstream
consumer — eliminating one fp32 fmul per element per exp site
(``exp(x)`` lowers to ``exp2(x * log2e)`` on NVIDIA; pre-scaling hoists
the multiply out of the kernels). Upstream applied this to KDA only:
``kda.py`` pre-scales after its cumsum and calls
``chunk_gated_delta_rule_fwd_h(..., use_exp2=True)`` —
``chunk_delta_h.py`` already carries the dual ``USE_EXP2`` branches.

PN354 extends the same pattern to the remaining GDN-path consumers of
the cumulative gate:

  1. ``fla/ops/chunk_o.py`` — 2 exp sites in ``chunk_fwd_kernel_o``
     (``b_o * exp(b_g)`` and ``b_A * exp(b_g_diff)``)
  2. ``fla/ops/chunk_scaled_dot_kkt.py`` — 1 exp site
     (``b_A * exp(b_g_diff)``)
  3. ``fla/ops/wy_fast.py`` — 1 exp site (raw ``tl.exp``; the file does
     not import from ``.op`` so the new branch uses ``tl.exp2``)
  4. ``fla/ops/chunk.py`` — the dispatcher: pre-scale ``g`` once after
     ``chunk_local_cumsum`` + thread ``use_exp2`` into all 4 consumers
     (``chunk_delta_h.py`` needs NO kernel edit — upstream already has
     the branches; we only pass ``use_exp2``).

Decode paths stay natural-base (``fused_recurrent`` /
``fused_sigmoid_gating``) — confirmed zero-win there, and upstream
keeps them on ``exp`` too. State values are domain-unchanged
(``exp2(g * RCP_LN2) == exp(g)`` numerically), so prefill-exp2 /
decode-exp mixing is safe — KDA ships exactly this split.

Runtime-conditional design (env read ONCE at import)
====================================================

Env flag: ``GENESIS_ENABLE_PN354_GDN_USE_EXP2`` (default OFF). The
text patches are themselves runtime-conditional: the patched
``chunk.py`` reads the env ONCE at module import into
``_GENESIS_PN354_USE_EXP2`` / ``_GENESIS_PN354_KW``. With the flag off:

  * no pre-scale multiply runs,
  * NO ``use_exp2`` kwarg is passed at all (``**{}`` splat), so the
    consumer wrappers run their default ``use_exp2=False`` → the
    kernels compile the ``exp`` branch → bit-identical to upstream,
    even if some consumer file was left unpatched.

The kernel-side edits are purely additive: a ``USE_EXP2: tl.constexpr``
parameter mirroring ``chunk_delta_h.py``'s existing dual-branch
pattern, plumbed from a ``use_exp2: bool = False`` wrapper kwarg.

Apply ordering & partial-failure safety
=======================================

Consumer files (chunk_o, chunk_scaled_dot_kkt, wy_fast) are patched
FIRST — they are inert until called with ``use_exp2=True``. The
``chunk.py`` dispatcher patch is applied ONLY when all three consumers
are live (applied or idempotent), so the flag can never route
``use_exp2=True`` into an unpatched wrapper.

Composition notes
=================

  * **PN59 streaming driver** (``sndr/engines/vllm/kernels_legacy/
    streaming_gdn_driver.py``): the live ``chunk.py`` dispatches
    chunked-prefill calls into the driver, whose ``_vanilla_path`` /
    ``_streaming_path`` is what actually runs. The driver is OUR file —
    it carries the same env-gated pre-scale + conditional ``use_exp2``
    threading directly (edited in-repo, not text-patched), reading the
    same env flag once at module import.
  * **P103** (Cliff-2 chunked wrapper): its own chunked path computes
    cumsum + consumers natural-base end-to-end (self-consistent, no
    mixing within a call) — correct but un-optimized when it engages
    (T > P103 MAX_T direct path). Its hot fallthrough delegates to the
    original fwd → PN59 driver → PN354-optimized. No conflict.
  * **PN345 / PN298 / PN299 / PN29 / PN106**: same files, different
    anchors (autotune config lists, scale-fold, pool sites) — verified
    zero anchor overlap against the live pin.
  * **PN350 / PN340 / PN341**: different files entirely.

Verified against live pin 0.22.1rc1.dev259+g303916e93 (container
vllm-qwen3.6-35b-balanced-k3, 2026-06-10): all OLD anchors unique
(count=1), no ``use_exp2`` / ``RCP_LN2`` text in the 4 target files.

Author: Sander (Sandermage / Aleksandr Barzov), Odessa, Ukraine.
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root
from sndr.kernel import TextPatch, TextPatcher, TextPatchResult

log = logging.getLogger("genesis.wiring.pn354_gdn_use_exp2")

ENV_FLAG = "GENESIS_ENABLE_PN354_GDN_USE_EXP2"

GENESIS_PN354_MARKER_CHUNK_O = (
    "Genesis PN354 GDN chunked-prefill exp2 (vllm#43195 KDA pattern -> GDN) v1 :: chunk_o.py"
)
GENESIS_PN354_MARKER_KKT = (
    "Genesis PN354 GDN chunked-prefill exp2 (vllm#43195 KDA pattern -> GDN) v1 :: chunk_scaled_dot_kkt.py"
)
GENESIS_PN354_MARKER_WY = (
    "Genesis PN354 GDN chunked-prefill exp2 (vllm#43195 KDA pattern -> GDN) v1 :: wy_fast.py"
)
GENESIS_PN354_MARKER_CHUNK = (
    "Genesis PN354 GDN chunked-prefill exp2 (vllm#43195 KDA pattern -> GDN) v1 :: chunk.py"
)

_REL_CHUNK_O = "model_executor/layers/fla/ops/chunk_o.py"
_REL_KKT = "model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py"
_REL_WY = "model_executor/layers/fla/ops/wy_fast.py"
_REL_CHUNK = "model_executor/layers/fla/ops/chunk.py"


# ════════════════════════════════════════════════════════════════════
# chunk_o.py — 4 sub-patches
# ════════════════════════════════════════════════════════════════════

# (1) import exp2 alongside exp (full-line anchor, unique at line 63)
CHUNK_O_IMPORT_OLD = "from .op import exp\n"
CHUNK_O_IMPORT_NEW = "from .op import exp, exp2\n"

# (2) kernel signature — add USE_EXP2 constexpr (mirrors chunk_delta_h.py)
CHUNK_O_SIG_OLD = (
    "    USE_G: tl.constexpr,\n"
    "    IS_VARLEN: tl.constexpr,\n"
    "):\n"
    "    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)\n"
)
CHUNK_O_SIG_NEW = (
    "    USE_G: tl.constexpr,\n"
    "    IS_VARLEN: tl.constexpr,\n"
    "    USE_EXP2: tl.constexpr,\n"
    "):\n"
    "    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)\n"
)

# (3) the two exp sites — dual branch exactly mirroring
#     chunk_delta_h.py:271-277's `if USE_EXP2: ... exp2 ... else: ... exp`
CHUNK_O_EXP_OLD = (
    "        b_o = b_o * exp(b_g)[:, None]\n"
    "        b_A = b_A * exp(b_g[:, None] - b_g[None, :])\n"
)
CHUNK_O_EXP_NEW = (
    "        # [Genesis PN354] g pre-scaled by RCP_LN2 -> exp2(g') == exp(g)\n"
    "        if USE_EXP2:\n"
    "            b_o = b_o * exp2(b_g)[:, None]\n"
    "            b_A = b_A * exp2(b_g[:, None] - b_g[None, :])\n"
    "        else:\n"
    "            b_o = b_o * exp(b_g)[:, None]\n"
    "            b_A = b_A * exp(b_g[:, None] - b_g[None, :])\n"
)

# (4a) wrapper signature — plumb use_exp2 kwarg (default False = upstream)
CHUNK_O_WRAP_SIG_OLD = (
    "    chunk_size: int = FLA_CHUNK_SIZE,\n"
    "    core_attn_out: torch.Tensor | None = None,\n"
    ") -> torch.Tensor:\n"
)
CHUNK_O_WRAP_SIG_NEW = (
    "    chunk_size: int = FLA_CHUNK_SIZE,\n"
    "    core_attn_out: torch.Tensor | None = None,\n"
    "    use_exp2: bool = False,\n"
    ") -> torch.Tensor:\n"
)

# (4b) kernel launch — pass USE_EXP2
CHUNK_O_LAUNCH_OLD = (
    "        K=K,\n"
    "        V=V,\n"
    "        BT=BT,\n"
    "    )\n"
    "    return o\n"
)
CHUNK_O_LAUNCH_NEW = (
    "        K=K,\n"
    "        V=V,\n"
    "        BT=BT,\n"
    "        USE_EXP2=use_exp2,\n"
    "    )\n"
    "    return o\n"
)


# ════════════════════════════════════════════════════════════════════
# chunk_scaled_dot_kkt.py — 4 sub-patches
# ════════════════════════════════════════════════════════════════════

KKT_IMPORT_OLD = "from .op import exp\n"
KKT_IMPORT_NEW = "from .op import exp, exp2\n"

KKT_SIG_OLD = (
    "    IS_VARLEN: tl.constexpr,\n"
    "    USE_G: tl.constexpr,\n"
    "):\n"
)
KKT_SIG_NEW = (
    "    IS_VARLEN: tl.constexpr,\n"
    "    USE_G: tl.constexpr,\n"
    "    USE_EXP2: tl.constexpr,\n"
    "):\n"
)

KKT_EXP_OLD = (
    "        b_g_diff = b_g[:, None] - b_g[None, :]\n"
    "        b_A = b_A * exp(b_g_diff)\n"
)
KKT_EXP_NEW = (
    "        b_g_diff = b_g[:, None] - b_g[None, :]\n"
    "        # [Genesis PN354] g pre-scaled by RCP_LN2 -> exp2(g') == exp(g)\n"
    "        if USE_EXP2:\n"
    "            b_A = b_A * exp2(b_g_diff)\n"
    "        else:\n"
    "            b_A = b_A * exp(b_g_diff)\n"
)

KKT_WRAP_SIG_OLD = (
    "    chunk_size: int = FLA_CHUNK_SIZE,\n"
    "    output_dtype: torch.dtype = torch.float32,\n"
    ") -> torch.Tensor:\n"
)
KKT_WRAP_SIG_NEW = (
    "    chunk_size: int = FLA_CHUNK_SIZE,\n"
    "    output_dtype: torch.dtype = torch.float32,\n"
    "    use_exp2: bool = False,\n"
    ") -> torch.Tensor:\n"
)

KKT_LAUNCH_OLD = (
    "        Hg=Hg,\n"
    "        K=K,\n"
    "        BT=BT,\n"
    "    )\n"
    "    return A\n"
)
KKT_LAUNCH_NEW = (
    "        Hg=Hg,\n"
    "        K=K,\n"
    "        BT=BT,\n"
    "        USE_EXP2=use_exp2,\n"
    "    )\n"
    "    return A\n"
)


# ════════════════════════════════════════════════════════════════════
# wy_fast.py — 3 sub-patches (file does NOT import from .op — the new
# branch uses raw tl.exp2 next to the existing raw tl.exp)
# ════════════════════════════════════════════════════════════════════

WY_SIG_OLD = (
    "    BV: tl.constexpr,\n"
    "    IS_VARLEN: tl.constexpr,\n"
    "):\n"
)
WY_SIG_NEW = (
    "    BV: tl.constexpr,\n"
    "    IS_VARLEN: tl.constexpr,\n"
    "    USE_EXP2: tl.constexpr,\n"
    "):\n"
)

WY_EXP_OLD = (
    "    b_g = tl.exp(tl.load(p_g, boundary_check=(0,)))\n"
)
WY_EXP_NEW = (
    "    # [Genesis PN354] g pre-scaled by RCP_LN2 -> tl.exp2(g') == exp(g)\n"
    "    if USE_EXP2:\n"
    "        b_g = tl.exp2(tl.load(p_g, boundary_check=(0,)))\n"
    "    else:\n"
    "        b_g = tl.exp(tl.load(p_g, boundary_check=(0,)))\n"
)

WY_WRAP_SIG_OLD = (
    "    cu_seqlens: torch.Tensor | None,\n"
    "    chunk_indices: torch.Tensor | None = None,\n"
    ") -> tuple[torch.Tensor, torch.Tensor]:\n"
)
WY_WRAP_SIG_NEW = (
    "    cu_seqlens: torch.Tensor | None,\n"
    "    chunk_indices: torch.Tensor | None = None,\n"
    "    use_exp2: bool = False,\n"
    ") -> tuple[torch.Tensor, torch.Tensor]:\n"
)

WY_LAUNCH_OLD = (
    "        BT=BT,\n"
    "        BK=BK,\n"
    "        BV=BV,\n"
    "    )\n"
    "    return w, u\n"
)
WY_LAUNCH_NEW = (
    "        BT=BT,\n"
    "        BK=BK,\n"
    "        BV=BV,\n"
    "        USE_EXP2=use_exp2,\n"
    "    )\n"
    "    return w, u\n"
)


# ════════════════════════════════════════════════════════════════════
# chunk.py — 6 sub-patches (dispatcher side). Template: kda.py:1421-1423
# (`g = g * RCP_LN2` after cumsum, then use_exp2=True downstream).
#
# WARNING (anchor discipline): lines immediately above the cumsum block
# are Genesis PN59 dispatch text — anchors below deliberately do NOT
# overlap any PN59 / P103 text.
# ════════════════════════════════════════════════════════════════════

# (1) module-scope flag — env read ONCE at import, near the imports
CHUNK_IMPORTS_OLD = (
    "from .utils import FLA_CHUNK_SIZE, SUPPRESS_LEVEL, input_guard\n"
    "from .wy_fast import recompute_w_u_fwd\n"
)
CHUNK_IMPORTS_NEW = (
    "from .utils import FLA_CHUNK_SIZE, SUPPRESS_LEVEL, input_guard\n"
    "from .wy_fast import recompute_w_u_fwd\n"
    "\n"
    "# [Genesis PN354] GDN chunked-prefill exp2 (vllm#43195 KDA pattern\n"
    "# extended to GDN consumers). Env read ONCE at import. With the flag\n"
    "# off, NO use_exp2 kwarg is passed at all and no pre-scale runs —\n"
    "# every code path is bit-identical to upstream.\n"
    "try:\n"
    "    import os as _genesis_pn354_os\n"
    "    from vllm.utils.math_utils import RCP_LN2 as _GENESIS_PN354_RCP_LN2\n"
    "    _GENESIS_PN354_USE_EXP2 = _genesis_pn354_os.environ.get(\n"
    "        \"GENESIS_ENABLE_PN354_GDN_USE_EXP2\", \"0\",\n"
    "    ).strip().lower() in (\"1\", \"true\", \"yes\", \"on\")\n"
    "except Exception:  # noqa: BLE001\n"
    "    _GENESIS_PN354_RCP_LN2 = 1.4426950216  # fp32 1/ln(2) (vllm constant)\n"
    "    _GENESIS_PN354_USE_EXP2 = False\n"
    "_GENESIS_PN354_KW = {\"use_exp2\": True} if _GENESIS_PN354_USE_EXP2 else {}\n"
)

# (2) pre-scale g once after the chunk-local cumsum
CHUNK_CUMSUM_OLD = (
    "    g = chunk_local_cumsum(\n"
    "        g, chunk_size=FLA_CHUNK_SIZE, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices\n"
    "    )\n"
)
CHUNK_CUMSUM_NEW = (
    "    g = chunk_local_cumsum(\n"
    "        g, chunk_size=FLA_CHUNK_SIZE, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices\n"
    "    )\n"
    "    # [Genesis PN354] scale the cumulative gate ONCE so every consumer\n"
    "    # can use exp2 (one fewer fp32 fmul per element per exp site).\n"
    "    # exp2(g * RCP_LN2) == exp(g) — state domain unchanged, safe to mix\n"
    "    # with natural-base decode kernels (KDA ships exactly this split).\n"
    "    if _GENESIS_PN354_USE_EXP2:\n"
    "        g = g * _GENESIS_PN354_RCP_LN2\n"
)

# (3) thread use_exp2 into chunk_scaled_dot_kkt_fwd
CHUNK_KKT_CALL_OLD = (
    "    A = chunk_scaled_dot_kkt_fwd(\n"
    "        k=k,\n"
    "        beta=beta,\n"
    "        g=g,\n"
    "        cu_seqlens=cu_seqlens,\n"
    "        chunk_indices=chunk_indices,\n"
    "        output_dtype=torch.float32,\n"
    "    )\n"
)
CHUNK_KKT_CALL_NEW = (
    "    A = chunk_scaled_dot_kkt_fwd(\n"
    "        k=k,\n"
    "        beta=beta,\n"
    "        g=g,\n"
    "        cu_seqlens=cu_seqlens,\n"
    "        chunk_indices=chunk_indices,\n"
    "        output_dtype=torch.float32,\n"
    "        **_GENESIS_PN354_KW,\n"
    "    )\n"
)

# (4) thread use_exp2 into recompute_w_u_fwd
CHUNK_WY_CALL_OLD = (
    "    w, u = recompute_w_u_fwd(\n"
    "        k=k,\n"
    "        v=v,\n"
    "        beta=beta,\n"
    "        A=A,\n"
    "        g_cumsum=g,\n"
    "        cu_seqlens=cu_seqlens,\n"
    "        chunk_indices=chunk_indices,\n"
    "    )\n"
)
CHUNK_WY_CALL_NEW = (
    "    w, u = recompute_w_u_fwd(\n"
    "        k=k,\n"
    "        v=v,\n"
    "        beta=beta,\n"
    "        A=A,\n"
    "        g_cumsum=g,\n"
    "        cu_seqlens=cu_seqlens,\n"
    "        chunk_indices=chunk_indices,\n"
    "        **_GENESIS_PN354_KW,\n"
    "    )\n"
)

# (5) thread use_exp2 into chunk_gated_delta_rule_fwd_h (upstream kernel
#     already has the USE_EXP2 branches — kwarg exists in our pin)
CHUNK_FWD_H_CALL_OLD = (
    "    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(\n"
    "        k=k,\n"
    "        w=w,\n"
    "        u=u,\n"
    "        g=g,\n"
    "        initial_state=initial_state,\n"
    "        output_final_state=output_final_state,\n"
    "        cu_seqlens=cu_seqlens,\n"
    "        chunk_indices=chunk_indices,\n"
    "        chunk_offsets=chunk_offsets,\n"
    "    )\n"
)
CHUNK_FWD_H_CALL_NEW = (
    "    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(\n"
    "        k=k,\n"
    "        w=w,\n"
    "        u=u,\n"
    "        g=g,\n"
    "        initial_state=initial_state,\n"
    "        output_final_state=output_final_state,\n"
    "        cu_seqlens=cu_seqlens,\n"
    "        chunk_indices=chunk_indices,\n"
    "        chunk_offsets=chunk_offsets,\n"
    "        **_GENESIS_PN354_KW,\n"
    "    )\n"
)

# (6) thread use_exp2 into chunk_fwd_o
CHUNK_FWD_O_CALL_OLD = (
    "    o = chunk_fwd_o(\n"
    "        q=q,\n"
    "        k=k,\n"
    "        v=v_new,\n"
    "        h=h,\n"
    "        g=g,\n"
    "        scale=scale,\n"
    "        cu_seqlens=cu_seqlens,\n"
    "        chunk_indices=chunk_indices,\n"
    "        core_attn_out=core_attn_out,\n"
    "    )\n"
)
CHUNK_FWD_O_CALL_NEW = (
    "    o = chunk_fwd_o(\n"
    "        q=q,\n"
    "        k=k,\n"
    "        v=v_new,\n"
    "        h=h,\n"
    "        g=g,\n"
    "        scale=scale,\n"
    "        cu_seqlens=cu_seqlens,\n"
    "        chunk_indices=chunk_indices,\n"
    "        core_attn_out=core_attn_out,\n"
    "        **_GENESIS_PN354_KW,\n"
    "    )\n"
)


# Drift markers: 'use_exp2' / 'USE_EXP2' appearing in the pristine file
# means upstream extended #43195 to the GDN path itself — our patch is
# obsolete there. '[Genesis PN354' is the self-marker belt (replacement
# text present without the top-of-file wiring marker => never re-splice).
_CONSUMER_DRIFT_MARKERS = ["use_exp2", "USE_EXP2", "[Genesis PN354"]
_CHUNK_DRIFT_MARKERS = ["use_exp2", "RCP_LN2", "[Genesis PN354"]


def _env_enabled() -> bool:
    return os.environ.get(ENV_FLAG, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _apply_one(
    rel_path: str,
    marker: str,
    sub_patches: list[TextPatch],
    drift_markers: list[str],
) -> tuple[str, bool]:
    """Apply one TextPatcher. Returns (status_msg, is_live).

    is_live=True for both APPLIED (fresh splice) and IDEMPOTENT (marker
    already present from a prior boot — the patched vllm tree persists
    across container restarts).
    """
    target = resolve_vllm_file(rel_path)
    if target is None:
        return f"{rel_path}: file not found", False
    patcher = TextPatcher(
        patch_name=f"PN354 {rel_path} — GDN exp2 gate decay",
        target_file=str(target),
        marker=marker,
        sub_patches=sub_patches,
        upstream_drift_markers=drift_markers,
    )
    result, failure = patcher.apply()
    if result == TextPatchResult.FAILED:
        return f"{rel_path}: FAILED — {failure.reason if failure else 'unknown'}", False
    if result == TextPatchResult.SKIPPED:
        return f"{rel_path}: skipped — {failure.reason if failure else 'unknown'}", False
    if result == TextPatchResult.IDEMPOTENT:
        return f"{rel_path}: idempotent", True
    return (
        f"{rel_path}: applied {len(patcher.applied_sub_patches)}"
        f"/{len(sub_patches)} sub-patches",
        True,
    )


def apply() -> tuple[str, str]:
    """Apply PN354 — exp2 extension across the GDN chunked-prefill chain.

    Order: consumer kernels first (inert until called with
    use_exp2=True), chunk.py dispatcher LAST and only if all consumers
    are live. Any other outcome reports failed with per-file detail —
    runtime stays upstream-correct in every partial state by design.
    """
    if not _env_enabled():
        return "skipped", (
            f"PN354 default OFF — set {ENV_FLAG}=1. Extends vllm#43195 "
            "(KDA exp2 gate decay) to GDN chunked-prefill consumers."
        )

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    results: list[str] = []

    msg, ok_chunk_o = _apply_one(
        _REL_CHUNK_O,
        GENESIS_PN354_MARKER_CHUNK_O,
        [
            TextPatch(name="pn354_chunk_o_import", anchor=CHUNK_O_IMPORT_OLD,
                      replacement=CHUNK_O_IMPORT_NEW, required=True),
            TextPatch(name="pn354_chunk_o_sig", anchor=CHUNK_O_SIG_OLD,
                      replacement=CHUNK_O_SIG_NEW, required=True),
            TextPatch(name="pn354_chunk_o_exp_sites", anchor=CHUNK_O_EXP_OLD,
                      replacement=CHUNK_O_EXP_NEW, required=True),
            TextPatch(name="pn354_chunk_o_wrap_sig", anchor=CHUNK_O_WRAP_SIG_OLD,
                      replacement=CHUNK_O_WRAP_SIG_NEW, required=True),
            TextPatch(name="pn354_chunk_o_launch", anchor=CHUNK_O_LAUNCH_OLD,
                      replacement=CHUNK_O_LAUNCH_NEW, required=True),
        ],
        _CONSUMER_DRIFT_MARKERS,
    )
    results.append(msg)

    msg, ok_kkt = _apply_one(
        _REL_KKT,
        GENESIS_PN354_MARKER_KKT,
        [
            TextPatch(name="pn354_kkt_import", anchor=KKT_IMPORT_OLD,
                      replacement=KKT_IMPORT_NEW, required=True),
            TextPatch(name="pn354_kkt_sig", anchor=KKT_SIG_OLD,
                      replacement=KKT_SIG_NEW, required=True),
            TextPatch(name="pn354_kkt_exp_site", anchor=KKT_EXP_OLD,
                      replacement=KKT_EXP_NEW, required=True),
            TextPatch(name="pn354_kkt_wrap_sig", anchor=KKT_WRAP_SIG_OLD,
                      replacement=KKT_WRAP_SIG_NEW, required=True),
            TextPatch(name="pn354_kkt_launch", anchor=KKT_LAUNCH_OLD,
                      replacement=KKT_LAUNCH_NEW, required=True),
        ],
        _CONSUMER_DRIFT_MARKERS,
    )
    results.append(msg)

    msg, ok_wy = _apply_one(
        _REL_WY,
        GENESIS_PN354_MARKER_WY,
        [
            TextPatch(name="pn354_wy_sig", anchor=WY_SIG_OLD,
                      replacement=WY_SIG_NEW, required=True),
            TextPatch(name="pn354_wy_exp_site", anchor=WY_EXP_OLD,
                      replacement=WY_EXP_NEW, required=True),
            TextPatch(name="pn354_wy_wrap_sig", anchor=WY_WRAP_SIG_OLD,
                      replacement=WY_WRAP_SIG_NEW, required=True),
            TextPatch(name="pn354_wy_launch", anchor=WY_LAUNCH_OLD,
                      replacement=WY_LAUNCH_NEW, required=True),
        ],
        _CONSUMER_DRIFT_MARKERS,
    )
    results.append(msg)

    if not (ok_chunk_o and ok_kkt and ok_wy):
        # Do NOT touch chunk.py — never route use_exp2=True into an
        # unpatched consumer. Consumer-only edits already applied are
        # inert (default use_exp2=False compiles the exp branch).
        return "failed", (
            "PN354 consumer kernels incomplete — chunk.py dispatcher "
            "NOT patched (runtime stays upstream-correct). Details: "
            + " | ".join(results)
        )

    msg, ok_chunk = _apply_one(
        _REL_CHUNK,
        GENESIS_PN354_MARKER_CHUNK,
        [
            TextPatch(name="pn354_chunk_module_flag", anchor=CHUNK_IMPORTS_OLD,
                      replacement=CHUNK_IMPORTS_NEW, required=True),
            TextPatch(name="pn354_chunk_g_prescale", anchor=CHUNK_CUMSUM_OLD,
                      replacement=CHUNK_CUMSUM_NEW, required=True),
            TextPatch(name="pn354_chunk_kkt_call", anchor=CHUNK_KKT_CALL_OLD,
                      replacement=CHUNK_KKT_CALL_NEW, required=True),
            TextPatch(name="pn354_chunk_wy_call", anchor=CHUNK_WY_CALL_OLD,
                      replacement=CHUNK_WY_CALL_NEW, required=True),
            TextPatch(name="pn354_chunk_fwd_h_call", anchor=CHUNK_FWD_H_CALL_OLD,
                      replacement=CHUNK_FWD_H_CALL_NEW, required=True),
            TextPatch(name="pn354_chunk_fwd_o_call", anchor=CHUNK_FWD_O_CALL_OLD,
                      replacement=CHUNK_FWD_O_CALL_NEW, required=True),
        ],
        _CHUNK_DRIFT_MARKERS,
    )
    results.append(msg)

    if not ok_chunk:
        return "failed", (
            "PN354 consumer kernels live but chunk.py dispatcher did not "
            "apply — exp2 path will not engage via chunk.py (the PN59 "
            "driver edit still covers dispatched calls; runtime stays "
            "correct either way). Details: " + " | ".join(results)
        )

    return "applied", (
        "PN354 applied: GDN chunked-prefill gate decay now exp2-capable "
        "across chunk_o + chunk_scaled_dot_kkt + wy_fast (+ existing "
        "chunk_delta_h USE_EXP2 branches), with the cumulative gate "
        "pre-scaled once by RCP_LN2 in chunk.py. Engages at runtime "
        f"only while {ENV_FLAG}=1 (read once at import). Decode paths "
        "stay natural-base (upstream-equivalent split, as KDA ships). "
        "Details: " + " | ".join(results)
    )


def is_applied() -> bool:
    target = resolve_vllm_file(_REL_CHUNK)
    if target is None:
        return False
    try:
        with open(str(target), encoding="utf-8") as f:
            return GENESIS_PN354_MARKER_CHUNK in f.read()
    except (OSError, UnicodeDecodeError):
        return False
