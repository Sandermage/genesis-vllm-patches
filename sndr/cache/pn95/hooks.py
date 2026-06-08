# SPDX-License-Identifier: Apache-2.0
"""PN95 hook entry points — the text-patch anchor surface.

The five public hooks here are the surface that vllm's source itself
calls into after the PN95 text-patches have been applied. Every name
in this file is referenced by an embedded
``from sndr.cache._pn95_runtime import <name>`` string
inside ``integrations/kv_cache/pn95_tier_aware_cache.py``:

  pn95_tier_aware_cache.py:61   notify_admit                 (Anchor 1)
  pn95_tier_aware_cache.py:86   notify_touch                 (Anchor 2)
  pn95_tier_aware_cache.py:115  init_mamba_exclusions_from_kv_groups (Anchor 3)
  pn95_tier_aware_cache.py:144  register_kv_caches           (Anchor 4)
  pn95_tier_aware_cache.py:160  scheduler_tick               (Anchor 5)

The legacy module MUST keep working re-export shims byte-identical
for all five names — text-anchor strings hard-code the
``_pn95_runtime`` import path, and any drift would regenerate the
anchor hash + break ``apply.shadow --strict``.

Two supporting helpers travel with the hooks:

  _refresh_env_cache  — one-shot env read for `_TICK_EVERY_CACHED` /
                        `_THRESHOLD_CACHED` / `_DEMOTE_BATCH_CACHED`;
                        called once on first scheduler_tick
  _gpu_free_mib       — best-effort torch.cuda.mem_get_info wrapper;
                        called by scheduler_tick under the
                        `_FREE_MIB_CACHE_TTL` cache TTL

M.4.2.I scope: function extraction only. The seven mutable state
singletons stay in ``_pn95_runtime``:

  _TICK_COUNTER             — REBOUND on every tick; 7 test sites
                               rebind it via ``monkeypatch.setattr``
                               in test_pn95_obs1_observability.py
  _TICK_LAST_FREE_MIB       — REBOUND inside scheduler_tick
  _TICK_EVERY_CACHED        — REBOUND inside _refresh_env_cache
  _THRESHOLD_CACHED         — REBOUND inside _refresh_env_cache
  _DEMOTE_BATCH_CACHED      — REBOUND inside _refresh_env_cache
  _FREE_MIB_CACHE_TTL       — read-only constant (5)
  _FREE_MIB_CACHE_VALID     — REBOUND inside scheduler_tick

Same alias-fragility class as previous slices — moving the names
would break the `_TICK_COUNTER` monkeypatch sites. The moved
functions read/rebind via lazy ``_rt._TICK_X = …`` attribute
mutation at the same module-attribute slot the original ``global``
declaration mutated.

The two ``global _TM`` declarations inside the original
``register_kv_caches`` / ``init_mamba_exclusions_from_kv_groups``
are NO-OPS (no rebind happens inside — only ``init_from_config``
rebinds, and that lives in `pn95.runtime_state` which already writes
via ``_rt._TM = …``). We drop those harmless declarations in the
moved versions to avoid the cross-module ``global`` no-op pattern.

Log/warning strings are preserved byte-identical — operators grep
``[PN95]`` / ``[PN95 v1.0]`` markers in production logs.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from .gates import (
    _enabled,
    _pn95_prefetch_neighbors_enabled,
    _pn95_prefetch_window,
    _read_env_int,
)

log = logging.getLogger("genesis.pn95")


# ─── Anchor 1: notify_admit ─────────────────────────────────────────────


def notify_admit(request: Any, prev_n_cached: int, new_n_cached: int,
                 group_id: int, block_size: int = 0) -> None:
    """Hook called from the cache_blocks() text-patch.

    `request` is a vllm Request; `prev_n_cached`/`new_n_cached` are the
    block index range that just got cached (newly_cached =
    range(prev_n_cached, new_n_cached)). `group_id` is the KV cache
    group id for the manager that produced these blocks. `block_size`
    is the manager's per-block token count — required for Day 5
    per-block MM tagging.

    Day 5: per-block mm_origin computed from `request.mm_features` (the
    list of `MultiModalFeatureSpec` objects, each carrying
    `mm_position: PlaceholderRange(offset, length)`). Falls back to
    coarse `has_mm_input` boolean when block_size is 0 or mm_features
    is missing (callers from older patch versions get a clean degrade).
    """
    from sndr.cache import _pn95_runtime as _rt
    if _rt._TM is None:
        return
    try:
        gid_str = f"g{group_id}"
        rid = getattr(request, "request_id", None) or getattr(
            request, "id", None) or "unknown"
        blk_range = range(prev_n_cached, new_n_cached)

        # Day 5 fast-path: real per-block MM tagging if data available
        from .transfer import _mm_block_overlap_set
        mm_block_set: set[int] = set()
        mm_features = getattr(request, "mm_features", None)
        if mm_features and block_size > 0:
            mm_block_set = _mm_block_overlap_set(
                mm_features, blk_range, block_size)
        else:
            # Coarse fallback (skeleton behavior — whole request marked
            # mm_origin if any MM input present)
            coarse_mm = bool(getattr(request, "has_mm_input", False)
                              or getattr(request, "mm_inputs", None)
                              or getattr(request, "multi_modal_inputs", None))
            if coarse_mm:
                mm_block_set = set(blk_range)

        for blk_idx in blk_range:
            key = (rid, gid_str, blk_idx)
            _rt._TM.admit(key, mm_origin=(blk_idx in mm_block_set),
                       group_id=gid_str)

        # Auto-warm L1 from L2/disk for predicted-near neighbors. The admit
        # call just observed a real prefix-cache event, which is the cheapest
        # signal we have that this request stream will keep traversing the
        # adjacent block_hashes. We pull the trailing N entries from
        # _admit_order — those are the freshest hits, most likely co-locality
        # candidates — and ask pn95_prefetch_blocks to move them L2->L1.
        # Pure host-side memcpy; no GPU touch. Skipped when env-gated off.
        if _pn95_prefetch_neighbors_enabled():
            window = _pn95_prefetch_window()
            if window > 0:
                try:
                    from .prefetch import pn95_prefetch_blocks
                    tail = _rt._TM._admit_order[-window:]
                    if tail:
                        pn95_prefetch_blocks(list(tail))
                except Exception:
                    pass
    except Exception as e:  # pragma: no cover — defensive
        log.warning("[PN95] notify_admit failed silently: %s", e)


# ─── Anchor 2: notify_touch ─────────────────────────────────────────────


def notify_touch(block_hash: Any, group_ids: list,
                 cached_blocks: Optional[list]) -> None:
    """Hook called from the get_cached_block() text-patch.

    Records that `block_hash` was hit. The skeleton just records via
    the TierManager.touch(); promote-on-hit logic stays inside the
    manager (returns demoted bytes on tier-1 hit; caller promotes).

    For the skeleton we don't actually do GPU promotion since that
    requires a real cuda buffer reference — Day 7 (live integration)
    swaps in the real promote path.
    """
    from sndr.cache import _pn95_runtime as _rt
    if _rt._TM is None:
        return
    try:
        # We don't have the (request, group_idx, block_idx) triple at
        # this site; instead use the block_hash as the key. The Day 5
        # plumbing canonicalizes (admit uses one key shape, touch
        # uses another) — for skeleton we record the touch by hash.
        # When a tier-aware system is fully wired, admit + touch
        # share the same key namespace via canonical_block_key().
        key = ("h", block_hash) if not isinstance(block_hash, tuple) \
            else block_hash
        # Best-effort: TierManager.touch returns bytes if demoted.
        # In the skeleton the caller can't do anything with bytes;
        # just record and discard.
        _rt._TM.touch(key)
    except Exception as e:  # pragma: no cover — defensive
        log.warning("[PN95] notify_touch failed silently: %s", e)


# ─── Anchor 4: register_kv_caches ───────────────────────────────────────


def register_kv_caches(kv_caches: Any, kv_cache_groups: Any) -> int:
    """Path C v1.0 Phase 1 (UNIFIED_CONFIG plan 2026-05-09): bridge from
    vLLM worker-level GPU tensor refs to the TierManager.

    Called from the 4th PN95 text-patch in `gpu_model_runner.py`
    immediately after `kv_caches = self.initialize_kv_cache_tensors(...)`.

    `kv_caches` is the vLLM worker's per-layer KV tensor list (or dict)
    — typically `dict[layer_name, Tensor]` or `list[Tensor]`. Each
    tensor has shape `(2, num_blocks, block_size, num_kv_heads, head_dim)`
    for attention layers, or `(num_blocks, conv_state_dim, ...)` for
    Mamba SSM layers (which we already exclude via Day 6).

    Phase 1 records the shape + tensor refs into TierManager metadata
    so Phase 2 can later cudaMemcpyAsync slices to/from CPU pinned slots.
    Phase 1 is observability-only — no actual copies happen yet.

    Returns the count of layer tensors successfully registered.
    Fail-silent: never raises.
    """
    from sndr.cache import _pn95_runtime as _rt
    # DEBUG sentinel for live verification
    try:
        with open("/tmp/pn95_init_called.log", "a") as fh:
            import os as _os
            shape_repr = (
                f"dict[{len(kv_caches)}]" if isinstance(kv_caches, dict)
                else f"list[{len(kv_caches)}]" if isinstance(kv_caches, (list, tuple))
                else type(kv_caches).__name__
            )
            fh.write(
                f"[{_os.getpid()}] register_kv_caches called: kv_caches={shape_repr} "
                f"enabled={_enabled()} tm={'set' if _rt._TM else 'None'}\n"
            )
    except Exception:
        pass
    # 2026-06-04 observability fix: banner moved AFTER lazy-init so the
    # reported TierManager state reflects post-install reality. Operators
    # used to see "TierManager=None" on every call (since banner ran
    # before lazy-init triggered) even when install succeeded on the
    # same call. Now: explicit warning for each silent-failure path,
    # banner reports final state.
    if not _enabled():
        log.warning(
            "[PN95 v1.0] register_kv_caches: PN95 disabled — "
            "GENESIS_ENABLE_PN95_TIER_AWARE_CACHE not set in worker env",
        )
        return 0
    # Lazy install of singleton if missing — workers spawn fresh Python
    # so the EngineCore-side init from init_mamba_exclusions doesn't
    # propagate. Re-do it here from the same env var.
    if _rt._TM is None:
        cfg_key = os.environ.get("GENESIS_PN95_CONFIG_KEY", "").strip()
        if not cfg_key:
            log.warning(
                "[PN95 v1.0] register_kv_caches: GENESIS_PN95_CONFIG_KEY "
                "absent from worker env — TierManager will stay None. "
                "Check YAML genesis_env + VLLM_WORKER_MULTIPROC_METHOD=spawn.",
            )
        else:
            try:
                from .runtime_state import init_from_config
                from .tier_config_loader import load_by_key
                # Resolution order (V1→V2 architectural unblock 2026-06-01):
                #   1. PN95-internal tier_configs/<key>.yaml (preferred)
                #   2. V1 ModelConfig.get(<key>) fallback (backward compat)
                cfg = load_by_key(cfg_key)
                cfg_source = f"tier_configs/{cfg_key}.yaml"
                if cfg is None:
                    from sndr.model_configs.registry import get
                    cfg = get(cfg_key)
                    cfg_source = f"V1 builtin/{cfg_key}.yaml"
                if cfg is None:
                    log.warning(
                        "[PN95 v1.0] register_kv_caches: cfg_key=%r "
                        "resolved to NO cfg (neither tier_configs/ nor "
                        "V1 builtin/). TierManager will stay None.",
                        cfg_key,
                    )
                else:
                    installed = init_from_config(cfg)
                    log.warning(
                        "[PN95 v1.0] register_kv_caches lazy-init from "
                        "%s: installed=%s, TierManager=%s",
                        cfg_source, installed,
                        "set" if _rt._TM else "None",
                    )
            except Exception as e:
                log.warning(
                    "[PN95 v1.0] register_kv_caches lazy-init failed: %s", e,
                )
    log.warning(
        "[PN95 v1.0] register_kv_caches called: PN95 enabled=%s, "
        "TierManager=%s",
        _enabled(), "installed" if _rt._TM else "None",
    )
    if _rt._TM is None:
        return 0
    try:
        # vLLM stores kv_caches in different shapes depending on version.
        # Common shapes:
        #   - list[torch.Tensor]: indexed by layer index
        #   - dict[str, torch.Tensor]: keyed by layer name
        # We handle both.
        # Phase 2 (UNIFIED_CONFIG plan 2026-05-09): vllm dev93 stores
        # per-layer KV caches in two distinct shapes:
        #   - Attention layers (`*self_attn.attn`): bare torch.Tensor
        #     of shape (num_blocks, block_size, K_or_V, packed_features)
        #     dtype=uint8 (TQ packed) — ELIGIBLE for demote.
        #   - Mamba/linear_attn layers: list[2 torch.Tensor]
        #     of shape (num_blocks, hidden_dim, conv_state_dim) fp16 —
        #     EXCLUDE from demote (SSM state stays GPU-resident).
        #
        # We register both shapes for observability but only attention
        # layers get the per-layer view registry that demote_block()
        # uses. Mamba layers are tracked by group_id only.
        n_registered = 0
        n_attention_eligible = 0
        per_layer_meta: dict = {}
        # Per-attention-layer view registry: {layer_name: {tensor, num_blocks, bytes_per_block}}
        attention_views: dict = {}

        if isinstance(kv_caches, dict):
            iterable = kv_caches.items()
        elif isinstance(kv_caches, (list, tuple)):
            iterable = enumerate(kv_caches)
        else:
            log.warning(
                "[PN95 v1.0] register_kv_caches: unrecognized kv_caches "
                "shape %s — skipping", type(kv_caches).__name__,
            )
            return 0

        for layer_id, val in iterable:
            try:
                layer_key = str(layer_id)
                # Mamba/linear_attn = list[Tensor]
                if isinstance(val, (list, tuple)):
                    inner_shapes = []
                    for t in val:
                        if hasattr(t, "shape"):
                            inner_shapes.append(tuple(t.shape))
                    per_layer_meta[layer_key] = {
                        "kind": "mamba_list",
                        "n_inner": len(val),
                        "inner_shapes": inner_shapes,
                        "demote_eligible": False,
                    }
                    n_registered += 1
                    continue
                # Attention bare Tensor — Phase 2 demote target
                shape = tuple(getattr(val, "shape", ()))
                dtype = str(getattr(val, "dtype", "?"))
                device = str(getattr(val, "device", "?"))
                if not shape or len(shape) < 2:
                    per_layer_meta[layer_key] = {
                        "kind": "unknown",
                        "shape": shape, "demote_eligible": False,
                    }
                    n_registered += 1
                    continue
                # Convention from dev93: shape[0] = num_blocks (TQ k8v4)
                num_blocks = int(shape[0])
                # Per-block byte size = product of remaining dims × elem_size
                elem_size = getattr(val, "element_size", lambda: 1)()
                tail_elems = 1
                for d in shape[1:]:
                    tail_elems *= int(d)
                bytes_per_block = tail_elems * elem_size
                per_layer_meta[layer_key] = {
                    "kind": "attention_tensor",
                    "shape": shape, "dtype": dtype, "device": device,
                    "num_blocks": num_blocks,
                    "bytes_per_block": bytes_per_block,
                    "demote_eligible": True,
                }
                # Stash the live tensor ref for demote_block / promote_block
                attention_views[layer_key] = {
                    "tensor": val,
                    "num_blocks": num_blocks,
                    "bytes_per_block": bytes_per_block,
                    "device": str(device),
                }
                n_registered += 1
                n_attention_eligible += 1
            except Exception as e:
                log.warning(
                    "[PN95 v1.0] register_kv_caches: layer %s failed: %s",
                    layer_id, e,
                )

        # Stash on the TierManager for Phase 2 demote/promote bridge
        _rt._TM._kv_caches_ref = kv_caches  # type: ignore[attr-defined]
        _rt._TM._kv_caches_meta = per_layer_meta  # type: ignore[attr-defined]
        _rt._TM._attention_views = attention_views  # type: ignore[attr-defined]
        log.warning(
            "[PN95 v1.0] register_kv_caches: %d layers (mamba+attn), "
            "%d attention layers eligible for demote",
            n_registered, n_attention_eligible,
        )
        # Sentinel for live integration verification — RICH dump of
        # actual structure (Phase 2 inspection): we need to know what
        # vllm dev93 puts in kv_caches[layer_name] since shape came
        # back () in v1.0 Phase 1.
        try:
            with open("/tmp/pn95_init_called.log", "a") as fh:
                fh.write(f"  → registered {n_registered} layers\n")
                # Dump first 2 entries with FULL introspection
                # Pick samples: 2 mamba layers + 2 attention layers
                items_iter = []
                if isinstance(kv_caches, dict):
                    all_items = list(kv_caches.items())
                    mamba_items = [(k, v) for k, v in all_items
                                    if "linear_attn" in k][:2]
                    attn_items = [(k, v) for k, v in all_items
                                   if "self_attn" in k or "attn.attn" in k][:2]
                    items_iter = mamba_items + attn_items
                    if not items_iter:
                        items_iter = all_items[:2]
                else:
                    items_iter = list(enumerate(kv_caches))[:2]
                for key, val in items_iter:
                    fh.write(f"    [{key}] type={type(val).__name__}\n")
                    if hasattr(val, "shape"):
                        fh.write(f"      shape={tuple(val.shape)}\n")
                    if hasattr(val, "dtype"):
                        fh.write(f"      dtype={val.dtype}\n")
                    if hasattr(val, "device"):
                        fh.write(f"      device={val.device}\n")
                    # Show available attrs (filter to non-dunder)
                    attrs = [a for a in dir(val) if not a.startswith("_")][:25]
                    fh.write(f"      attrs(first 25): {attrs}\n")
                    # If it's a list/tuple/dict-like, dig 1 level deeper
                    if isinstance(val, (list, tuple)) and len(val) > 0:
                        fh.write(f"      (list[{len(val)}] of {type(val[0]).__name__})\n")
                        if hasattr(val[0], "shape"):
                            fh.write(f"      [0].shape={tuple(val[0].shape)}\n")
                            fh.write(f"      [0].dtype={val[0].dtype}\n")
                    elif isinstance(val, dict) and val:
                        first_k = next(iter(val))
                        fh.write(f"      (dict[{len(val)}], first key={first_k!r}, val type={type(val[first_k]).__name__})\n")
        except Exception as e:
            try:
                with open("/tmp/pn95_init_called.log", "a") as fh:
                    fh.write(f"    SENTINEL DUMP FAILED: {e}\n")
            except Exception:
                pass
        return n_registered
    except Exception as e:  # pragma: no cover — defensive
        log.warning("[PN95 v1.0] register_kv_caches failed silently: %s", e)
        return 0


# ─── Anchor 3: init_mamba_exclusions_from_kv_groups ─────────────────────


def init_mamba_exclusions_from_kv_groups(kv_cache_groups: Any) -> int:
    """Day 6 (UNIFIED_CONFIG plan 2026-05-09): walk KVCacheGroupSpec list,
    register every MambaSpec group as excluded from demotion.

    Returns the count of groups marked excluded. Idempotent (safe to
    re-call). Fail-silent: never raises — all errors logged + swallowed.

    Called from the PN95 text-patch in `KVCacheManager.__init__`. ALSO
    triggers lazy TierManager init from env (`GENESIS_PN95_CONFIG_KEY`)
    if no manager has been installed yet — so workers spawned with
    `VLLM_WORKER_MULTIPROC_METHOD=spawn` get the singleton on first use.
    """
    from sndr.cache import _pn95_runtime as _rt
    n_groups = len(list(kv_cache_groups or []))
    # DEBUG sentinel — writes to /tmp to prove the hook fired
    try:
        with open("/tmp/pn95_init_called.log", "a") as fh:
            import os as _os
            fh.write(
                f"[{_os.getpid()}] init_mamba called n_groups={n_groups} "
                f"enabled={_enabled()}\n"
            )
    except Exception:
        pass
    log.warning(
        "[PN95] init_mamba_exclusions_from_kv_groups called: %d groups, "
        "PN95 enabled=%s",
        n_groups, _enabled(),
    )
    if not _enabled():
        return 0
    try:
        # Lazy install of singleton if missing — read config from env.
        if _rt._TM is None:
            cfg_key = os.environ.get("GENESIS_PN95_CONFIG_KEY", "").strip()
            log.info("[PN95] lazy init: cfg_key=%r", cfg_key)
            if cfg_key:
                try:
                    from .runtime_state import init_from_config
                    from .tier_config_loader import load_by_key
                    # Resolution order (V1→V2 architectural unblock 2026-06-01):
                    #   1. PN95-internal tier_configs/<key>.yaml (preferred)
                    #   2. V1 ModelConfig.get(<key>) fallback (backward
                    #      compat for operators still pointing at V1 keys
                    #      like `a5000-2x-tier-aware-example`)
                    cfg = load_by_key(cfg_key)
                    cfg_source = "tier_configs/" + cfg_key + ".yaml"
                    if cfg is None:
                        from sndr.model_configs.registry import get
                        cfg = get(cfg_key)
                        cfg_source = "V1 builtin/" + cfg_key + ".yaml"
                    if cfg is not None:
                        log.info("[PN95] resolved cfg from %s", cfg_source)
                        init_from_config(cfg)
                        log.info("[PN95] singleton installed: %s",
                                 _rt._TM.stats() if _rt._TM else "FAILED")
                except Exception as e:
                    log.warning(
                        "[PN95] lazy init from GENESIS_PN95_CONFIG_KEY=%s "
                        "failed: %s", cfg_key, e,
                    )

        if _rt._TM is None:
            return 0

        n_excluded = 0
        for idx, group in enumerate(kv_cache_groups or []):
            spec = getattr(group, "kv_cache_spec", None)
            cls_name = type(spec).__name__ if spec is not None else "<None>"
            log.warning(
                "[PN95] group %d: spec_class=%s layers=%s",
                idx, cls_name, getattr(group, "layer_names", "?"),
            )
            if spec is None:
                continue
            # Detect MambaSpec by name + check known mamba-spec classes
            # in case vllm renamed (Mamba2Spec, ShortConvSpec, etc.)
            mamba_class_names = (
                "MambaSpec", "Mamba2Spec", "ShortConvSpec",
                "GdnAttentionSpec", "MambaAttentionSpec",
            )
            if cls_name in mamba_class_names:
                gid = f"g{idx}"
                _rt._TM.register_mamba_excluded(gid)
                n_excluded += 1
                log.warning(
                    "[PN95] excluding %s group %s (layers=%s) from demotion",
                    cls_name, gid, getattr(group, "layer_names", "?"),
                )

        if n_excluded > 0:
            log.info(
                "[PN95] Mamba exclusion init complete — %d groups excluded "
                "out of %d total. TierManager stats: %s",
                n_excluded, len(list(kv_cache_groups or [])), _rt._TM.stats(),
            )
        return n_excluded
    except Exception as e:  # pragma: no cover — defensive
        log.warning("[PN95] init_mamba_exclusions failed silently: %s", e)
        return 0


# ─── Supporting helpers (scheduler-tick infrastructure) ─────────────────


def _refresh_env_cache() -> None:
    """Re-read env vars into module-local cache. Called once on first tick."""
    from sndr.cache import _pn95_runtime as _rt
    # Path C Phase 3 default: TICK_EVERY=10 (was 100 — too slow for single-stream
    # workloads where Scheduler.schedule() fires only ~30 times per long request).
    _rt._TICK_EVERY_CACHED = max(1, _read_env_int("GENESIS_PN95_TICK_EVERY", 10))
    _rt._THRESHOLD_CACHED = _read_env_int("GENESIS_PN95_DEMOTE_FREE_MIB_THRESHOLD", 2048)
    _rt._DEMOTE_BATCH_CACHED = max(1, _read_env_int("GENESIS_PN95_DEMOTE_BATCH", 8))


def _gpu_free_mib() -> int:
    """Best-effort: returns GPU 0 free VRAM in MiB. 0 if torch/cuda missing.

    Note: torch.cuda.mem_get_info costs ~800-1200 μs per call (cudaMemGetInfo
    syscall round-trip). Caller responsible for caching across multiple ticks
    via _FREE_MIB_CACHE_VALID counter.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return 0
        free, _total = torch.cuda.mem_get_info(0)
        return free // (1 << 20)
    except Exception:
        return 0


# ─── Anchor 5: scheduler_tick ───────────────────────────────────────────


def scheduler_tick() -> None:
    """Path C v1.0 Phase 4.1 — smart proactive scheduler-tick hook.

    Strategy:
      1. Fast-path early return (~50 ns) when disabled
      2. Throttled by GENESIS_PN95_TICK_EVERY (default 10)
      3. Cached _gpu_free_mib (TTL=5 ticks → amortizes cudaMemGetInfo)
      4. When pressure detected (free < threshold), select COLD cached
         blocks via BlockPool's own LRU queue (head of free_block_queue
         = next-to-evict). These blocks are ref_cnt=0 = no readers =
         safe to copy. Skip hot-ring members (last N spec-decode targets).
      5. demote_on_evict captures bytes BEFORE vllm's own eviction —
         turns vllm's reset_hash into a no-op (bytes already preserved).

    Result: real LRU-based demote instead of dummy block_idx=0. Released
    GPU memory comes from vllm's normal eviction path (no race).

    Fail-silent — never raises into scheduler hot path.
    """
    from sndr.cache import _pn95_runtime as _rt
    if not _enabled() or _rt._TM is None:
        return
    _rt._TICK_COUNTER += 1
    _rt._PN95_STATS["ticks_total"] += 1

    # OBS1 — periodic stats dump to JSON file for operator visibility
    # Throttled by GENESIS_PN95_STATS_INTERVAL (default 100 ticks),
    # disabled via GENESIS_PN95_STATS_FILE="" env. Fail-silent.
    from .metrics import _pn95_dump_stats_if_due
    _pn95_dump_stats_if_due()

    if _rt._TICK_EVERY_CACHED is None:
        _refresh_env_cache()

    if _rt._TICK_COUNTER % _rt._TICK_EVERY_CACHED != 0:
        return

    _rt._PN95_STATS["ticks_pressure_check"] += 1
    try:
        if _rt._FREE_MIB_CACHE_VALID <= 0:
            free_mib = _gpu_free_mib()
            _rt._TICK_LAST_FREE_MIB = free_mib
            _rt._PN95_STATS["last_free_mib"] = free_mib
            _rt._FREE_MIB_CACHE_VALID = _rt._FREE_MIB_CACHE_TTL
        else:
            free_mib = _rt._TICK_LAST_FREE_MIB
            _rt._FREE_MIB_CACHE_VALID -= 1

        if free_mib <= 0 or free_mib >= _rt._THRESHOLD_CACHED:
            return

        _rt._FREE_MIB_CACHE_VALID = 0

        # [Genesis PN203] cold-prefix offload sweep — Tier 3.A core.
        # Runs BEFORE empty_cache so the demote path can populate L2 (PN95
        # pinned pool) with bytes that would otherwise be discarded.
        # Requires per-layer KV split (PN202) for correctness on hybrid models.
        from .shared_buffers import pn203_cold_prefix_sweep, pn201_maybe_empty_cache
        try:
            pn203_cold_prefix_sweep()
        except Exception:
            pass

        # [Genesis PN201] threshold-gated empty_cache for fragmentation
        # reclaim. Fires after PN203 has captured what's worth saving.
        try:
            pn201_maybe_empty_cache(free_mib)
        except Exception:
            pass

        # smart proactive demote via vllm LRU. Falls back to
        # legacy block_idx=0 path if no BlockPools registered (dispatcher
        # not wired) or no cached candidates found.
        from .prefix_store import _proactive_demote_cold
        target = _rt._DEMOTE_BATCH_CACHED
        n_demoted = _proactive_demote_cold(target)

        if n_demoted == 0:
            # Legacy fallback — only fires if BlockPool refs not registered
            # or no cached blocks exist yet (cold start)
            views = getattr(_rt._TM, "_attention_views", {}) or {}
            for layer_name, info in list(views.items())[:target]:
                num_blocks = int(info.get("num_blocks", 0))
                if num_blocks <= 0:
                    continue
                if _rt._TM.demote_block(layer_name, 0):
                    n_demoted += 1
                if n_demoted >= target:
                    break

        if n_demoted > 0:
            _rt._PN95_STATS["ticks_demote_triggered"] += 1
            _rt._PN95_STATS["blocks_demoted_total"] += n_demoted
            _rt._PN95_STATS["last_demote_count"] = n_demoted
            log.warning(
                "[PN95 v1.0 Phase 4.1] scheduler_tick: pressure (free=%d MiB "
                "< %d MiB) — demoted %d cold blocks via LRU "
                "(total demoted=%d, prefix_store_entries=%d)",
                free_mib, _rt._THRESHOLD_CACHED, n_demoted,
                _rt._PN95_STATS["blocks_demoted_total"],
                len(_rt._PN95_PREFIX_STORE),
            )
    except Exception as e:  # pragma: no cover — defensive
        log.warning("[PN95 v1.0 Phase 4.1] scheduler_tick failed: %s", e)
