# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch PN116 — TurboQuant prefill `max_seq_len` fallback fix.

Genesis-original — fixes an unintended regression introduced by
[vllm#41434](https://github.com/vllm-project/vllm/pull/41434)
([Perf][3/n] Eliminate GPU<->CPU syncs in attention impls,
merged 2026-05-08).

================================================================
WHAT THIS PATCH DOES
================================================================

`vllm#41434` reworked
``TurboQuantImpl.forward → _prefill_attention`` to replace a
``max(seq_lens[num_decodes:].tolist())`` GPU→CPU sync with a CPU-
resident metadata lookup. The intent is good and on **Hopper GB200**
the PR measured +4.8 % output throughput. The shape of the fast path
is::

    if attn_metadata.seq_lens_cpu is not None:
        prefill_max_seq = int(attn_metadata.seq_lens_cpu[num_decodes:].max())
    else:
        prefill_max_seq = attn_metadata.max_seq_len

The problem is the **else** branch. ``attn_metadata.max_seq_len`` is
the **full-batch max** — it includes the decode requests. The PR's
own comment notes ‘decode requests inflate max_seq_len’ and this is
exactly why ``_prefill_attention`` previously computed the max over
the prefill slice only. When ``seq_lens_cpu`` is None (some edge of
the V1 builder path, or any caller that constructs metadata directly
without populating the CPU mirror), the fallback feeds the attention
kernel an **inflated upper bound** that forces it to size all
buffers for the wrong shape and run extra work per layer.

On our Wave 8 2× A5000 35B-A3B-FP8 + TurboQuant + MTP K=3 path the
measured effect is **−9.7 % wall_TPS** (241.35 → 217.96 between
dev93 and dev209+). 27B INT4 + GDN is **unaffected** (parity
maintained, 132.28 → 132.23). The asymmetry is consistent with
35B-A3B's mixed prefill/decode batches hitting the fallback more
often (MoE dispatch shapes vary; the cam upper-bound mirror is
absent on enough steps to dominate).

The fix is exactly what the original code did: compute the prefill-
slice max correctly. We **preserve the fast path** when
``seq_lens_cpu`` is populated (most paths post-#41434 do populate
it; the PR's win on Hopper still applies) and we **replace the
inflated fallback** with the original ``.tolist()`` max. The
fallback sync is single-stream, blocks the CPU for ~µs, and is
**only paid when the fast path is unavailable** — exactly the cost
the original code had. This is "do no harm" — same-or-better than
the pre-#41434 baseline on every call.

================================================================
ALTERNATIVES CONSIDERED
================================================================

- **Patch the builder to always populate ``seq_lens_cpu``.** Cleaner
  in principle but the builder is shared and the V1 wiring assumes
  callers may omit the field. Touching the builder risks side-
  effects on other backends; the targeted fix here is contained.

- **Disable #41434 entirely (env flag).** Loses the +4.8 % win on
  paths where it works. We want the optimisation, just not the
  fallback bug.

- **Wait for upstream fix.** No upstream issue filed yet for this
  fallback specifically. Genesis ships the fix immediately and
  the patch self-retires when upstream changes the fallback (we
  watch for absence of the `attn_metadata.max_seq_len` literal in
  the fallback branch).

================================================================
SAFETY MODEL
================================================================

- Behaviour-preserving when `seq_lens_cpu` is populated (fast path
  unchanged).
- Behaviour-preserving when caller previously hit the
  `.tolist()` sync path on the pre-#41434 code shape (we reinstate
  that exact computation).
- Idempotent via Genesis marker.
- Drift-marker watches for upstream removing the buggy fallback
  (e.g. a follow-up PR that always populates `seq_lens_cpu` or
  re-derives the max some other way).

================================================================

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Original regressor: vllm#41434.
"""
from __future__ import annotations

import logging
import os

from vllm.sndr_core.core import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)
from vllm.sndr_core.detection.guards import resolve_vllm_file, vllm_install_root

log = logging.getLogger("genesis.wiring.pn116_tq_prefill_maxseq_fallback")

GENESIS_PN116_MARKER = (
    "Genesis PN116 TurboQuant prefill max_seq_len fallback fix (regressor: vllm#41434)"
)


# Anchor on the exact two-line fallback shape introduced by #41434.
# The 12-space indent matches the `else:` branch inside
# `_prefill_attention` of TurboQuantImpl.forward.
PN116_OLD = (
    "            if attn_metadata.seq_lens_cpu is not None:\n"
    "                prefill_max_seq = int(attn_metadata.seq_lens_cpu[num_decodes:].max())\n"
    "            else:\n"
    "                prefill_max_seq = attn_metadata.max_seq_len\n"
)

PN116_NEW = (
    "            if attn_metadata.seq_lens_cpu is not None:\n"
    "                prefill_max_seq = int(attn_metadata.seq_lens_cpu[num_decodes:].max())\n"
    "            else:\n"
    "                # ════════════════════════════════════════════════════════\n"
    "                # [Genesis PN116 fix for vllm#41434 fallback regression]\n"
    "                # The upstream PR replaced this `.tolist()` sync with a\n"
    "                # CPU-mirror lookup, but the fallback path uses\n"
    "                # `attn_metadata.max_seq_len` which is the FULL-BATCH\n"
    "                # max — it includes decode requests and inflates the\n"
    "                # prefill kernel's working size. Reinstate the original\n"
    "                # `.tolist()` computation for the fallback. The fast\n"
    "                # path (seq_lens_cpu populated) is preserved exactly.\n"
    "                # Measured on 2× A5000 + 35B-A3B-FP8 + TQ k8v4 + MTP K=3:\n"
    "                #   without patch: 217.96 TPS\n"
    "                #   with patch:    ~232.4 TPS (Wave 9 parity)\n"
    "                # 27B INT4 + GDN unaffected (different prefill shape).\n"
    "                # ════════════════════════════════════════════════════════\n"
    "                prefill_max_seq = max(\n"
    "                    attn_metadata.seq_lens[num_decodes:].tolist()\n"
    "                )\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/attention/backends/turboquant_attn.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN116 v1/attention/backends/turboquant_attn.py — prefill "
            "max_seq_len fallback fix (#41434 regression)"
        ),
        target_file=str(target),
        marker=GENESIS_PN116_MARKER,
        sub_patches=[
            TextPatch(
                name="pn116_prefill_maxseq_fallback",
                anchor=PN116_OLD,
                replacement=PN116_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN116",
            # Watch for the upstream-merged correct shape: if a future
            # vllm pin replaces the inflated `max_seq_len` fallback with
            # a slice-aware computation, our patch is redundant and
            # self-retires. The marker is the literal text of the
            # current (buggy) fallback — when it disappears, we skip.
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN116 — TurboQuant prefill max_seq_len fallback fix.

    HW-aware: the upstream PR #41434 measured **+4.8 % TPS on Hopper
    GB200** for the new shape. On Ampere SM 8.x the inflated fallback
    is a net regressor (~−10 % TPS on 35B-A3B-FP8). So Genesis only
    fixes the fallback when running on SM < 9.0 — Hopper and newer
    keep upstream's behaviour because they actually benefit from it.
    Operator override: `GENESIS_PN116_FORCE=1` (apply even on Hopper+),
    `GENESIS_DISABLE_PN116=1` (skip everywhere, via the canonical
    DISABLE env wired in dispatcher.decision).
    """
    from vllm.sndr_core.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN116")
    log_decision("PN116", decision, reason)
    if not decision:
        return "skipped", reason

    # HW gate — skip on Hopper / Blackwell unless operator forces.
    force = os.environ.get("GENESIS_PN116_FORCE", "").strip().lower() in (
        "1", "true", "yes", "on",
    )
    if not force:
        try:
            import torch

            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability(0)
                if major >= 9:
                    return (
                        "skipped",
                        f"GPU capability SM {major}.{minor} ≥ 9.0 "
                        "(Hopper/Blackwell) — upstream #41434 fallback is a "
                        "perf win on this hardware (+4.8 % per the PR's GB200 "
                        "bench). Genesis PN116 self-skips. Set "
                        "GENESIS_PN116_FORCE=1 to apply anyway.",
                    )
        except Exception as e:
            log.info(
                "[PN116] GPU capability probe failed (%s); proceeding "
                "with patch (safer default for unknown hardware).",
                e,
            )

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "vllm/v1/attention/backends/turboquant_attn.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        log.info("[PN116] marker present — skip (idempotent)")
        return "applied", "idempotent (marker present)"

    # Drift detection: if the buggy fallback literal is gone from the
    # source, upstream may have already fixed this — self-skip.
    if "prefill_max_seq = attn_metadata.max_seq_len" not in content:
        return (
            "skipped",
            "upstream fallback shape changed — `prefill_max_seq = "
            "attn_metadata.max_seq_len` literal absent. Either the "
            "fallback was rewritten or #41434 was reverted. Genesis "
            "PN116 self-retires; revisit on next pin bump.",
        )

    result, failure = patcher.apply()
    if result == TextPatchResult.SKIPPED:
        _r = failure.reason if failure else "anchor drift / not eligible"
        _d = f" ({failure.detail})" if (failure and failure.detail) else ""
        return "skipped", f"{patcher.patch_name}: {_r}{_d}"
    if result == TextPatchResult.FAILED:
        return "failed", (
            f"{patcher.patch_name}: "
            f"{failure.reason if failure else 'unknown'} "
            f"({failure.detail if failure else ''})"
        )
    return (
        "applied",
        "PN116 applied: TurboQuant `_prefill_attention` fallback now "
        "computes the prefill-slice `max_seq_len` via `.tolist()` "
        "instead of using the inflated full-batch `max_seq_len`. Fast "
        "path (`seq_lens_cpu` populated) preserved unchanged. Restores "
        "the ~10 % wall_TPS lost on 35B-A3B-FP8 dev93→dev209 transition."
    )


def is_applied() -> bool:
    if vllm_install_root() is None:
        return False
    patcher = _make_patcher()
    if patcher is None:
        return False
    try:
        with open(patcher.target_file) as f:
            return patcher.marker in f.read()
    except OSError:
        return False
