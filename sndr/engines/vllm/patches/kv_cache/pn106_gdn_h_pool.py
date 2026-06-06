# SPDX-License-Identifier: Apache-2.0
"""PN106 — pool the GDN `h` chunk-state tensor (kill per-layer allocs).

Root cause of allocator fragmentation under long context: each of the
48 GDN/Mamba layers in Qwen3.6-27B-Next calls
`chunk_gated_delta_rule_fwd_h()`, which on every invocation does

    h = k.new_empty(B, NT, H, V, K)   # chunk_delta_h.py:336

Size for B=1, H=48, V=K=128, fp16:
   - T=2048  chunk:  NT=32   → 50 MiB
   - T=5000  chunk:  NT=79   → 124 MiB
   - T=156K  full:   NT=2438 → 3.8 GiB (caps via chunked prefill)

Per chunked-prefill step this is allocated + freed **48 times**. The
PyTorch caching allocator does coalesce same-size slabs, but each new
chunk size (T-not-aligned-to-block-size) yields a fresh slab class,
and 48 mamba layers × N chunk variants generates persistent
fragmentation — the 319 MiB "reserved but unallocated" in the OOM
crash log is largely this.

PN106 replaces `k.new_empty(B, NT, H, V, K)` with a slice from a single
PN95-managed pool sized to the max observed (NT, H, V, K). The pool
grows on demand (max NT seen so far × H × V × K) and is reused across
all 48 layers + all steps. Net effect:

- ~200-400 MiB fragmentation reclaimed (steady-state)
- 0 alloc/free traffic in the GDN forward hot-path
- No speed impact (slice view is zero-cost; downstream Triton kernels
  read the underlying storage identically to fresh `new_empty`)

This is a TEXT-PATCH (modifies `chunk_delta_h.py` source) so it applies
to every worker process. Env gate:
`GENESIS_ENABLE_PN106_GDN_H_POOL=1` (default OFF).
"""
from __future__ import annotations

import logging
import os

from vllm.sndr_core.detection.guards import resolve_vllm_file, vllm_install_root
from vllm.sndr_core.core import TextPatch, TextPatcher

log = logging.getLogger("genesis.wiring.pn106_gdn_h_pool")

GENESIS_MARKER = "Genesis PN106 GDN scratch pool (multi-anchor architectural memory mgr) v11.3.0_hotpath"


def _enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN106_GDN_H_POOL", "0",
    ).strip().lower() in ("1", "true", "yes", "on")


# Anchor verified on vllm nightly dcacdf9a (2026-05-13) at
# vllm/model_executor/layers/fla/ops/chunk_delta_h.py:~332.
# ── Anchor 1: chunk_delta_h.py — h-state tensor pool ─────────────────────
PN106_H_OLD = (
    "    h = k.new_empty(B, NT, H, V, K)\n"
)
PN106_H_NEW = (
    "    # [Genesis PN106 v2 — hot-path optimized] pool h-state.\n"
    "    # 48-layer × N-chunk × per-step. v2 caches the get-fn in this\n"
    "    # module's globals so per-call env-var lookup + .strip().lower()\n"
    "    # + import is eliminated after the first call.\n"
    "    h = None\n"
    "    _g_pn106_get = globals().get('_GENESIS_PN106_get_fn')\n"
    "    if _g_pn106_get is None:\n"
    "        try:\n"
    "            import os as _g_pn106_os\n"
    "            if _g_pn106_os.environ.get(\n"
    "                \"GENESIS_ENABLE_PN106_GDN_H_POOL\", \"0\",\n"
    "            ).strip().lower() in (\"1\", \"true\", \"yes\", \"on\"):\n"
    "                from vllm.sndr_core.cache._pn95_runtime import pn106_get_pooled_buf as _g_pn106_get\n"
    "            else:\n"
    "                _g_pn106_get = False\n"
    "        except Exception:\n"
    "            _g_pn106_get = False\n"
    "        globals()['_GENESIS_PN106_get_fn'] = _g_pn106_get\n"
    "    if _g_pn106_get:\n"
    "        try:\n"
    "            h = _g_pn106_get(\"gdn_h\", (B, NT, H, V, K), k.dtype, k.device)\n"
    "        except Exception:\n"
    "            h = None\n"
    "    if h is None:\n"
    "        h = k.new_empty(B, NT, H, V, K)\n"
)

# ── Anchor 2: chunk_delta_h.py — v_new tensor pool ────────────────────────
# Anchor exists ONLY when save_new_value=True; mark required=False so the
# patcher does not fail if the line text drifts (rare path).
PN106_VNEW_OLD = (
    "    v_new = torch.empty_like(u) if save_new_value else None\n"
)
PN106_VNEW_NEW = (
    "    # [Genesis PN106 v2 — hot-path optimized] pool v_new.\n"
    "    v_new = None\n"
    "    if save_new_value:\n"
    "        _g_pn106_get = globals().get('_GENESIS_PN106_get_fn')\n"
    "        if _g_pn106_get is None:\n"
    "            try:\n"
    "                import os as _g_pn106_os\n"
    "                if _g_pn106_os.environ.get(\n"
    "                    \"GENESIS_ENABLE_PN106_GDN_H_POOL\", \"0\",\n"
    "                ).strip().lower() in (\"1\", \"true\", \"yes\", \"on\"):\n"
    "                    from vllm.sndr_core.cache._pn95_runtime import pn106_get_pooled_buf as _g_pn106_get\n"
    "                else:\n"
    "                    _g_pn106_get = False\n"
    "            except Exception:\n"
    "                _g_pn106_get = False\n"
    "            globals()['_GENESIS_PN106_get_fn'] = _g_pn106_get\n"
    "        if _g_pn106_get:\n"
    "            try:\n"
    "                v_new = _g_pn106_get(\"gdn_v_new\", tuple(u.shape), u.dtype, u.device)\n"
    "            except Exception:\n"
    "                v_new = None\n"
    "        if v_new is None:\n"
    "            v_new = torch.empty_like(u)\n"
)

# ── Anchor 3: chunk_o.py — `o = torch.empty_like(v)` (the CRASH SITE) ─────
PN106_O_OLD = (
    "    else:\n"
    "        o = torch.empty_like(v)\n"
)
PN106_O_NEW = (
    "    else:\n"
    "        # [Genesis PN106 v2 — hot-path optimized] pool `o` — crash site.\n"
    "        o = None\n"
    "        _g_pn106_get = globals().get('_GENESIS_PN106_get_fn')\n"
    "        if _g_pn106_get is None:\n"
    "            try:\n"
    "                import os as _g_pn106_os\n"
    "                if _g_pn106_os.environ.get(\n"
    "                    \"GENESIS_ENABLE_PN106_GDN_H_POOL\", \"0\",\n"
    "                ).strip().lower() in (\"1\", \"true\", \"yes\", \"on\"):\n"
    "                    from vllm.sndr_core.cache._pn95_runtime import pn106_get_pooled_buf as _g_pn106_get\n"
    "                else:\n"
    "                    _g_pn106_get = False\n"
    "            except Exception:\n"
    "                _g_pn106_get = False\n"
    "            globals()['_GENESIS_PN106_get_fn'] = _g_pn106_get\n"
    "        if _g_pn106_get:\n"
    "            try:\n"
    "                o = _g_pn106_get(\"gdn_o\", tuple(v.shape), v.dtype, v.device)\n"
    "            except Exception:\n"
    "                o = None\n"
    "        if o is None:\n"
    "            o = torch.empty_like(v)\n"
)


def _make_chunk_delta_h_patcher() -> TextPatcher | None:
    """Patcher for h + v_new pools in chunk_delta_h.py."""
    target = resolve_vllm_file(
        "model_executor/layers/fla/ops/chunk_delta_h.py"
    )
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN106-A chunk_delta_h h + v_new pools",
        target_file=str(target),
        marker=GENESIS_MARKER,
        sub_patches=[
            TextPatch(
                name="pn106_chunk_delta_h_h_pool",
                anchor=PN106_H_OLD,
                replacement=PN106_H_NEW,
                required=True,
            ),
            TextPatch(
                name="pn106_chunk_delta_h_vnew_pool",
                anchor=PN106_VNEW_OLD,
                replacement=PN106_VNEW_NEW,
                required=False,  # exists only on save_new_value paths
            ),
        ],
        upstream_drift_markers=[
            "Genesis PN106",
            "pn106_get_pooled_buf",
        ],
    )


def _make_chunk_o_patcher() -> TextPatcher | None:
    """Patcher for `o = torch.empty_like(v)` — the OOM crash site."""
    target = resolve_vllm_file(
        "model_executor/layers/fla/ops/chunk_o.py"
    )
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN106-B chunk_o output pool (crash-site fix)",
        target_file=str(target),
        marker=GENESIS_MARKER,
        sub_patches=[
            TextPatch(
                name="pn106_chunk_o_pool",
                anchor=PN106_O_OLD,
                replacement=PN106_O_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "Genesis PN106",
            "pn106_get_pooled_buf",
        ],
    )


def _make_patcher():  # legacy single-patcher (kept for back-compat)
    return _make_chunk_delta_h_patcher()


def _apply_one(patcher) -> tuple[str, str]:
    if patcher is None:
        return "skipped", "patcher None"
    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as fh:
        content = fh.read()
    if patcher.marker in content:
        return "applied", "idempotent"
    for m in patcher.upstream_drift_markers:
        if m in content:
            return "skipped", f"drift marker {m!r} already in file"
    result, failure = patcher.apply()
    from vllm.sndr_core.core import result_to_wiring_status
    return result_to_wiring_status(
        result, failure,
        applied_message=f"{patcher.patch_name}: applied",
        patch_name=patcher.patch_name,
    )


def apply() -> tuple[str, str]:
    if not _enabled():
        return "skipped", "PN106 disabled (set GENESIS_ENABLE_PN106_GDN_H_POOL=1)"
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    results = []
    for patcher_factory in (_make_chunk_delta_h_patcher,
                             _make_chunk_o_patcher):
        try:
            patcher = patcher_factory()
        except Exception as e:
            results.append(("skipped", f"factory error: {e}"))
            continue
        status, reason = _apply_one(patcher)
        results.append((status, reason))

    applied_count = sum(1 for s, _ in results if s == "applied")
    if applied_count == 0:
        return "skipped", "; ".join(f"{s}:{r[:80]}" for s, r in results)
    return "applied", f"{applied_count}/{len(results)} sub-patches applied"
