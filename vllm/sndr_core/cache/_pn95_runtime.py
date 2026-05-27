# SPDX-License-Identifier: Apache-2.0
"""PN95 v7.73.x runtime hooks — notify_admit / notify_touch.

Module-level singleton TierManager. Both hooks are designed to be
**fail-silent**: if GENESIS_ENABLE_PN95_TIER_AWARE_CACHE is unset OR
the singleton hasn't been initialized OR any error occurs inside the
notification, the call must return cleanly so the surrounding vLLM
code path is never destabilized.

Public entry points:
  - `init_from_config(cfg)` — install the singleton from a ModelConfig.
    Idempotent. Called once at engine startup by the dispatcher hook.
  - `notify_admit(request, prev_n_cached, new_n_cached, group_id)` —
    called from the cache_blocks() text-patch site after vLLM's
    cache_full_blocks() returns.
  - `notify_touch(block_hash, group_ids, cached_blocks)` — called
    from the get_cached_block() text-patch site before return.
  - `tier_manager()` — accessor for live observability / tests.
  - `reset_for_tests()` — drop the singleton.

Vision-token tagging + Mamba exclusion plumbing wires through this
module so the text-patches stay tiny.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

log = logging.getLogger("genesis.pn95")

_LOCK = threading.Lock()
_TM: Optional[Any] = None  # vllm.sndr_core.cache.tier_manager.TierManager
_LAST_GROUP_IDS_BY_HASH: dict = {}  # cleared on reset_for_tests


# M.4.1 — env-gate predicates extracted to `.pn95.gates`. The
# re-exports below keep ``_pn95_runtime._enabled`` / ``_phase5_virt_enabled``
# importable at the original dotted path for tests and text-patch anchors.
from .pn95.gates import _enabled, _phase5_virt_enabled  # noqa: E402


# M.4.2.F — `pn95_extra_logical_memory_bytes` extracted to
# `.pn95.virtual_blocks`. Sibling-patch text-anchor in
# pn95_tier_aware_cache.py:393 imports this name through _pn95_runtime
# so the shim below MUST keep working byte-identically.
from .pn95.virtual_blocks import pn95_extra_logical_memory_bytes  # noqa: E402


_PN95_CUDA_STREAM: Optional[Any] = None
# Phase 5 Session 2 — side-table for block metadata.
# KVCacheBlock is @dataclass(slots=True) → cannot add fields directly.
# Side-table keyed by (id(pool), block_id) → {"physical_resident": bool,
# "physical_block_id": Optional[int], "last_access_tick": int}.
_PN95_BLOCK_METADATA: dict = {}
_PN95_POOL_LOGICAL_NUM_BLOCKS: dict = {}  # id(pool) → logical num_blocks


# M.4.1 — `_pn95_async_enabled` + `_pn95_use_stream_pool` extracted to
# `.pn95.gates` (re-exports preserved here so test sites that do
# `rt._pn95_async_enabled` keep resolving — e.g.
# tests/unit/cache/test_pn95_b1_async_stream.py).
from .pn95.gates import _pn95_async_enabled, _pn95_use_stream_pool  # noqa: E402


# M.4.2.G — seven transfer helpers (_pn95_stream + 6 byte-copy primitives)
# extracted to `.pn95.transfer`. State `_PN95_CUDA_STREAM` stays here (5
# test sites rebind it via `monkeypatch.setattr(rt, ...)`); the moved
# `_pn95_stream` lazy-imports `_rt` and writes `_rt._PN95_CUDA_STREAM = …`
# via attribute mutation at the same module-attribute slot the original
# `global` declaration mutated.
from .pn95.transfer import (  # noqa: E402
    _pn95_stream,
    _pn95_gpu_to_cpu_bytes,
    _pn95_gpu_to_cpu_bytes_batch_v2,
    _pn95_cpu_to_gpu_copy_batch_v2,
    _pn95_gpu_to_cpu_bytes_batch,
    _pn95_cpu_to_gpu_copy_batch,
    _pn95_cpu_to_gpu_copy,
)


# M.4.2.F — `pn95_phase5_init_block_pool` extracted to
# `.pn95.virtual_blocks`. Sibling-patch text-anchor in
# pn95_tier_aware_cache.py:348 imports this name through _pn95_runtime
# so the shim below MUST keep working byte-identically.
from .pn95.virtual_blocks import pn95_phase5_init_block_pool  # noqa: E402


# M.4.2.H — `pn95_materialize_virtual_block` extracted to
# `.pn95.prefix_store`. Was parked here in M.4.2.F because it calls
# `demote_on_evict` (now also in prefix_store.py). Reaches state via
# lazy `_rt.X`; the virtual_blocks helpers that call this function
# (pn95_guard_get_new_blocks, pn95_anchor12_post_popleft) keep using
# `_rt.pn95_materialize_virtual_block` → resolves through the shim.
from .pn95.prefix_store import pn95_materialize_virtual_block  # noqa: E402


# M.4.2.E — `pn106_get_gdn_h_buf`, `_pn106_legacy_h_impl`,
# `pn106_get_pooled_buf` extracted to `.pn95.shared_buffers`. State
# (`_PN106_POOLS`, `_PN106_NAMED_POOLS`) stays defined in this module
# (see block below); the moved functions reach it via lazy `_rt.X`.
# The legacy import path stays byte-identical for sibling-patch
# text-anchor strings (PN106 / PN200 anchor strings hard-code
# ``from vllm.sndr_core.cache._pn95_runtime import pn106_get_pooled_buf``).
from .pn95.shared_buffers import (  # noqa: E402
    pn106_get_gdn_h_buf,
    _pn106_legacy_h_impl,
    pn106_get_pooled_buf,
)


_PN106_POOLS: dict = {}
_PN106_NAMED_POOLS: dict = {}  # name -> torch.Tensor (flat backing buffer)


_PN201_LAST_EMPTY_CACHE_TICK: int = 0

# PN203 cold-prefix offload settings (set by PN203 apply hook at boot).
# Read by scheduler_tick to decide whether to do window-aware demote
# of full-attention layer blocks older than _PN203_ACTIVE_WINDOW_TOKENS.
_PN203_ENABLED: bool = False
_PN203_ACTIVE_WINDOW_TOKENS: int = 32768
_PN203_ATTENTION_ONLY: bool = True


# M.4.2.E — `pn203_cold_prefix_sweep`, `pn201_maybe_empty_cache`,
# `pn106_periodic_empty_cache` extracted to `.pn95.shared_buffers`.
# State (`_PN201_LAST_EMPTY_CACHE_TICK`, `_PN203_*`) stays defined in
# this module; the moved functions read/rebind via lazy `_rt.X` (the
# `global _PN201_LAST_EMPTY_CACHE_TICK` rebind is replicated via
# `_rt._PN201_LAST_EMPTY_CACHE_TICK = tick` attribute write).
from .pn95.shared_buffers import (  # noqa: E402
    pn203_cold_prefix_sweep,
    pn201_maybe_empty_cache,
    pn106_periodic_empty_cache,
)


# M.4.2.E — `pn97_physical_cap_bytes` + `pn96_emergency_rescue`
# extracted to `.pn95.shared_buffers`. The legacy import path stays
# byte-identical for sibling-patch text-anchor strings (PN96 / PN97
# hard-code the ``from vllm.sndr_core.cache._pn95_runtime import …``
# string).
from .pn95.shared_buffers import (  # noqa: E402
    pn97_physical_cap_bytes,
    pn96_emergency_rescue,
)


# M.4.2.F — six virtual-block helpers extracted to
# `.pn95.virtual_blocks`. Sibling-patch text-anchor in
# pn95_tier_aware_cache.py:250 imports `pn95_anchor12_post_popleft`
# through _pn95_runtime; the shim below MUST keep working byte-identically.
# `pn95_guard_get_new_blocks` and `pn95_anchor12_post_popleft` call
# `pn95_materialize_virtual_block` (stays here, calls demote_on_evict /
# promote_on_miss) via lazy `_rt.pn95_materialize_virtual_block`.
from .pn95.virtual_blocks import (  # noqa: E402
    pn95_block_is_physical_resident,
    pn95_guard_get_new_blocks,
    pn95_anchor12_post_popleft,
    pn95_block_metadata,
    pn95_pool_logical_num_blocks,
    pn95_physical_num_blocks_cap,
)


# M.4.2.A — `_detect_upstream_offload_connector` + `init_from_config`
# extracted to `.pn95.runtime_state`. State ownership (`_TM`, `_LOCK`)
# stays in this module; the moved functions write through `_rt._TM = ...`
# via lazy late-import so the 36 reader sites here see the canonical
# binding.
from .pn95.runtime_state import (  # noqa: E402
    _detect_upstream_offload_connector,
    init_from_config,
)


# M.4.2.G — `_mm_block_overlap_set` extracted to `.pn95.transfer`. Pure
# helper (no torch / no CUDA) but it's used by notify_admit alongside the
# other transfer primitives, so it lives in transfer.py for cohesion.
# Tests in test_pn95_day5_mm_overlap.py import it directly from
# `_pn95_runtime`; the shim keeps that path working.
from .pn95.transfer import _mm_block_overlap_set  # noqa: E402


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
    if _TM is None:
        return
    try:
        gid_str = f"g{group_id}"
        rid = getattr(request, "request_id", None) or getattr(
            request, "id", None) or "unknown"
        blk_range = range(prev_n_cached, new_n_cached)

        # Day 5 fast-path: real per-block MM tagging if data available
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
            _TM.admit(key, mm_origin=(blk_idx in mm_block_set),
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
                    tail = _TM._admit_order[-window:]
                    if tail:
                        pn95_prefetch_blocks(list(tail))
                except Exception:
                    pass
    except Exception as e:  # pragma: no cover — defensive
        log.warning("[PN95] notify_admit failed silently: %s", e)


# M.4.1 — prefetch env gates extracted to `.pn95.gates`.
from .pn95.gates import (  # noqa: E402
    _pn95_prefetch_neighbors_enabled,
    _pn95_prefetch_window,
)


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
    if _TM is None:
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
        _TM.touch(key)
    except Exception as e:  # pragma: no cover — defensive
        log.warning("[PN95] notify_touch failed silently: %s", e)


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
    global _TM
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
                f"enabled={_enabled()} tm={'set' if _TM else 'None'}\n"
            )
    except Exception:
        pass
    log.warning(
        "[PN95 v1.0] register_kv_caches called: PN95 enabled=%s, "
        "TierManager=%s",
        _enabled(), "installed" if _TM else "None",
    )
    if not _enabled():
        return 0
    # Lazy install of singleton if missing — workers spawn fresh Python
    # so the EngineCore-side init from init_mamba_exclusions doesn't
    # propagate. Re-do it here from the same env var.
    if _TM is None:
        cfg_key = os.environ.get("GENESIS_PN95_CONFIG_KEY", "").strip()
        if cfg_key:
            try:
                from vllm.sndr_core.model_configs.registry import get
                cfg = get(cfg_key)
                if cfg is not None:
                    init_from_config(cfg)
            except Exception as e:
                log.warning(
                    "[PN95 v1.0] register_kv_caches lazy-init failed: %s", e,
                )
    if _TM is None:
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
        _TM._kv_caches_ref = kv_caches  # type: ignore[attr-defined]
        _TM._kv_caches_meta = per_layer_meta  # type: ignore[attr-defined]
        _TM._attention_views = attention_views  # type: ignore[attr-defined]
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
        global _TM
        if _TM is None:
            cfg_key = os.environ.get("GENESIS_PN95_CONFIG_KEY", "").strip()
            log.info("[PN95] lazy init: cfg_key=%r", cfg_key)
            if cfg_key:
                try:
                    from vllm.sndr_core.model_configs.registry import get
                    cfg = get(cfg_key)
                    if cfg is not None:
                        init_from_config(cfg)
                        log.info("[PN95] singleton installed: %s",
                                 _TM.stats() if _TM else "FAILED")
                except Exception as e:
                    log.warning(
                        "[PN95] lazy init from GENESIS_PN95_CONFIG_KEY=%s "
                        "failed: %s", cfg_key, e,
                    )

        if _TM is None:
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
                _TM.register_mamba_excluded(gid)
                n_excluded += 1
                log.warning(
                    "[PN95] excluding %s group %s (layers=%s) from demotion",
                    cls_name, gid, getattr(group, "layer_names", "?"),
                )

        if n_excluded > 0:
            log.info(
                "[PN95] Mamba exclusion init complete — %d groups excluded "
                "out of %d total. TierManager stats: %s",
                n_excluded, len(list(kv_cache_groups or [])), _TM.stats(),
            )
        return n_excluded
    except Exception as e:  # pragma: no cover — defensive
        log.warning("[PN95] init_mamba_exclusions failed silently: %s", e)
        return 0


_TICK_COUNTER = 0
_TICK_LAST_FREE_MIB = 0
# Path C v1.0 Phase 3 — observability counters.
#
# M.4.1 note: ownership stays in this module (not `.pn95.metrics`) because
# ~10 test sites rebind `rt._PN95_STATS` via ``monkeypatch.setattr``,
# which would break a cross-module name alias. The functions that READ
# this dict (``get_pn95_stats`` / ``_pn95_dump_stats_if_due``) live in
# `.pn95.metrics` and late-import this name so the monkeypatch path
# continues to work. State ownership reorganization is deferred to M.4.2.
_PN95_STATS = {
    "ticks_total": 0,
    "ticks_pressure_check": 0,
    "ticks_demote_triggered": 0,
    "blocks_demoted_total": 0,
    "blocks_promoted_total": 0,
    "last_free_mib": 0,
    "last_demote_count": 0,
}
# Cache config envs — read once at module init, not on every tick
# (was causing measurable overhead per call). Override via reset_env_cache().
_TICK_EVERY_CACHED: Optional[int] = None
_THRESHOLD_CACHED: Optional[int] = None
_DEMOTE_BATCH_CACHED: Optional[int] = None
_FREE_MIB_CACHE_TTL: int = 5  # cache mem_get_info for N consecutive ticks
_FREE_MIB_CACHE_VALID: int = 0


# M.4.1 — `_read_env_int` extracted to `.pn95.gates`.
from .pn95.gates import _read_env_int  # noqa: E402


def _refresh_env_cache() -> None:
    """Re-read env vars into module-local cache. Called once on first tick."""
    global _TICK_EVERY_CACHED, _THRESHOLD_CACHED, _DEMOTE_BATCH_CACHED
    # Path C Phase 3 default: TICK_EVERY=10 (was 100 — too slow for single-stream
    # workloads where Scheduler.schedule() fires only ~30 times per long request).
    _TICK_EVERY_CACHED = max(1, _read_env_int("GENESIS_PN95_TICK_EVERY", 10))
    _THRESHOLD_CACHED = _read_env_int("GENESIS_PN95_DEMOTE_FREE_MIB_THRESHOLD", 2048)
    _DEMOTE_BATCH_CACHED = max(1, _read_env_int("GENESIS_PN95_DEMOTE_BATCH", 8))


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


# M.4.1 — `get_pn95_stats` and `_pn95_dump_stats_if_due` extracted to
# `.pn95.metrics`. They late-import the foreign state (`_PN95_PREFIX_STORE`,
# `_PN95_PREFETCH_STATS`, `_PN95_LAYER_ACCESS_COUNTS`, `_PN95_COMPRESS_LIB`,
# `_pn95_l1_pool`, `_TICK_COUNTER`) from this module to avoid a circular
# import; M.4.2 will move those singletons into focused modules too.
from .pn95.metrics import get_pn95_stats, _pn95_dump_stats_if_due  # noqa: E402


# ─── Path C v1.0 Phase 4 — prefix-cache extension to CPU pinned RAM ──────
#
# Strategy: instead of demoting ARBITRARY GPU blocks (race-prone), we
# intercept exactly two BlockPool events that are already safe:
#
#   1. demote_on_evict — called from `_maybe_evict_cached_block` AFTER
#      the block has been removed from `cached_block_hash_to_block` and
#      BEFORE `block.reset_hash()`. At this moment the block has ref_cnt=0
#      (no readers), is not in any active `block_table`, and vllm is
#      about to recycle the GPU slot. We safely copy the bytes to CPU.
#
#   2. promote_on_miss — called from `get_cached_block` when vllm's own
#      lookup returned None (cache miss). We check our CPU store; if the
#      hash is there, we allocate a fresh GPU block via `get_new_blocks(1)`,
#      copy CPU→GPU, re-insert into vllm's prefix cache, and return it.
#      vllm sees a normal cache hit — no further changes needed.
#
# Effect: prefix cache effective capacity = N_gpu_blocks + N_cpu_entries
# Multi-turn / long-history workloads see dramatically higher hit rate
# without any CUDA OOM risk and without any hot-path overhead (no polling,
# no per-tick mem_get_info — the path only fires on actual eviction events).
#
# Compatible with hybrid-GDN models (Mamba SSM groups never enter the
# prefix cache to begin with — only attention groups have block hashes).
# Compatible with TP=2+ — each worker has its own _PN95_PREFIX_STORE
# scoped to that worker's GPU.

from collections import OrderedDict as _OrderedDict  # noqa: E402  — block-local import after PN95 section header
_PN95_PREFIX_STORE: "_OrderedDict[Any, list]" = _OrderedDict()
_PN95_PREFIX_STORE_BYTES_USED: int = 0
_PN95_PREFIX_STORE_MAX_BYTES_CACHED: Optional[int] = None
_PN95_BLOCK_POOL_REFS: list = []
# Lock protecting concurrent writers to _PN95_PREFIX_STORE +
# _PN95_PREFIX_STORE_BYTES_USED. Multiple paths can mutate the store:
# demote_on_evict (scheduler thread), prefetch_blocks (prefetch worker
# thread), _prefix_store_evict_until_fit (called recursively from demote).
# Pre-PN95 the dict was single-threaded so a lock would have been overhead;
# with the new prefetch API + worker_side_proactive_demote we explicitly
# advertise thread-safety, so the lock is required (review finding #12).
_PN95_PREFIX_STORE_LOCK: threading.Lock = threading.Lock()


# ── L1 pinned host cache (optional, gated by GENESIS_ENABLE_PN95_PINNED_POOL).
# Held in a separate module (_pn95_pinned_pool) so the heavy import (torch
# pin_memory) doesn't run at sndr_core boot when the feature is OFF.
#
# Layer payload (list of (layer_name, bytes)) is serialized to a single bytes
# blob via pickle.HIGHEST_PROTOCOL before being placed in the pool — the pool
# itself works on byte slabs of equal slot size. Unpack reverses pickle.
# Pickle overhead is ~5-10 μs per blob, dwarfed by the PCIe transfer savings
# from non-pageable memory (3-5 GB/s pinned vs ~600 MB/s pageable bounce).
# M.4.2.C — `_pn95_pack_layer_data` + `_pn95_unpack_layer_data` extracted
# to `.pn95.compression`. Pure pickle helpers with no state touch.
from .pn95.compression import (  # noqa: E402
    _pn95_pack_layer_data,
    _pn95_unpack_layer_data,
)


# M.4.2.H — `_pn95_l1_pool` extracted to `.pn95.prefix_store`. Pure
# accessor for the pinned-pool singleton.
from .pn95.prefix_store import _pn95_l1_pool  # noqa: E402


# ── Prefetch / warmup API ───────────────────────────────────────────────
# Inspired by SGLang HiCache layer-by-layer prefetch overlap: the engine
# tells PN95 which block_hashes are about to be needed; PN95 warms up the
# fast L1 pinned pool from the slow L2 OrderedDict (or, if not in L2, the
# disk tier) so the actual `promote_on_miss` call lands in L1.
#
# Without prefetch the path on a cold block is:
#   promote_on_miss → L2 OrderedDict.get → numpy.frombuffer → torch.from_numpy
#   → .to(cuda, non_blocking=True from pageable mem) → bounce-buffer copy
#   ~400 μs for a 32 KB block (single attention layer's K+V for one block).
#
# With prefetch the L1 slot is already pinned by the time vllm calls
# promote_on_miss; the GPU read is a single pinned-host DMA at PCIe Gen4
# line rate, ~80 μs.
#
# Stats track hits/misses so operators can see whether prefetch is paying
# off (vs raw L1 demote-side fills).
_PN95_PREFETCH_STATS = {
    "prefetch_calls": 0,
    "prefetch_block_hashes": 0,
    "prefetch_l2_hits_promoted": 0,  # L2 entry copied into L1 pinned
    "prefetch_l2_already_in_l1": 0,  # L1 already warm — no-op
    "prefetch_missing": 0,           # not in L2 or disk — nothing to do
    "prefetch_disk_hits_promoted": 0,
    "prefetch_pool_full_skips": 0,
}


# M.4.2.B — `pn95_prefetch_blocks` + `pn95_get_prefetch_stats` extracted
# to `.pn95.prefetch`. The `_PN95_PREFETCH_STATS` dict + every other state
# singleton this code reads (`_PN95_PREFIX_STORE`, the L1 pool, prefix
# store helpers, the packer) stay defined in this module; the moved
# functions mutate them through `_rt.X` via lazy late-import — including
# the `_rt._PN95_PREFIX_STORE_BYTES_USED += …` attribute rebind that
# replaces the original `global` declaration.
from .pn95.prefetch import (  # noqa: E402
    pn95_prefetch_blocks,
    pn95_get_prefetch_stats,
)


# ── Layer-aware demote priority ──────────────────────────────────────────
# Tracks per-layer access frequency from the promote path so demote can
# prioritize cold layers when capacity is constrained. Implementation is a
# small dict keyed by layer_name; on a 17-attention-layer Qwen3.6 27B model
# the structure stays trivial (<200 bytes). Single-process, single-rank —
# no cross-worker sync needed (each rank decides its own demote order).
#
# Update on every promote restoration; read on every demote sort. The
# heuristic is intentionally simple: layers with the highest cumulative
# promote-read count are deemed "hot" and pushed to the end of the demote
# queue. Cold layers (low counts) are demoted first, freeing GPU memory
# along the path the GPU's attention forward least frequently touches.
#
# Bounded growth: counts are reset on overflow (>10M) to prevent integer
# bloat. The relative ordering is what matters, not absolute values.
_PN95_LAYER_ACCESS_COUNTS: dict = {}
_PN95_LAYER_ACCESS_RESET_THRESHOLD = 10_000_000


# ── store_threshold reuse-frequency gate (upstream PR #40020 pattern) ─────
#
# Tracks how many times each block_hash has been *looked up* during
# promote_on_miss. Blocks with hits below GENESIS_PN95_STORE_THRESHOLD
# are NOT demoted on evict — the engine pays no compression/copy cost
# on a block that's about to disappear from the request stream forever.
#
# Inspired by upstream `FilterReusedOffloadingManager` in cpu/manager.py
# (only stores keys observed `store_threshold` times via lookup).
#
# Default off (threshold=0). Operators set >=2 when serving chat workloads
# where most prefill blocks are one-shot.
# Lookup hit tracker — ownership stays here for M.4.1 (same reason
# as ``_PN95_STATS``: test sites may rebind via monkeypatch). The
# ``_pn95_record_lookup`` function lives in `.pn95.metrics` and
# late-imports this state.
_PN95_HIT_COUNTS: dict = {}
_PN95_HIT_TRACKER_MAX = 64_000

from .pn95.metrics import _pn95_record_lookup  # noqa: E402


# M.4.1 — `_pn95_store_threshold` extracted to `.pn95.gates`.
from .pn95.gates import _pn95_store_threshold  # noqa: E402


# M.4.2.D — `_pn95_should_demote` extracted to `.pn95.demote_policy`.
# Reads `_PN95_HIT_COUNTS` (stays here) via lazy `_rt.X`.
from .pn95.demote_policy import _pn95_should_demote  # noqa: E402


# ── block_size_factor — PCIe transaction amortization ────────────────────
#
# Upstream PR #40020 lets the offload layer operate on `block_size_factor`
# adjacent KV blocks as a single super-block. This amortizes the PCIe
# transaction setup cost (~10-20us per DMA submit) over a larger payload,
# critical when each KV block is small (Qwen3.6 fp8 32KB/block).
#
# At factor=4 we batch four ~32KB blocks into one ~128KB transfer:
#   - submit/sync overhead drops 4×
#   - PCIe is more BW-efficient on larger packets (closer to line rate)
#   - tradeoff: the L1 pinned pool slot_size auto-derives from first
#     super-block payload, so 4× larger slots → fewer slots within
#     GENESIS_PN95_PINNED_POOL_MB budget
#
# Default 1 (no grouping). 2-4 typical sweet spots for production.
# M.4.1 — `_pn95_block_size_factor` extracted to `.pn95.gates`.
from .pn95.gates import _pn95_block_size_factor  # noqa: E402


# M.4.2.H — `pn95_demote_batch` extracted to `.pn95.prefix_store`.
# Was parked here in M.4.2.D because it calls demote_on_evict.
from .pn95.prefix_store import pn95_demote_batch  # noqa: E402


# M.4.1 — `_pn95_layer_aware_enabled` extracted to `.pn95.gates`.
from .pn95.gates import _pn95_layer_aware_enabled  # noqa: E402


# M.4.2.D — `_pn95_record_layer_promote` + `_pn95_sort_layers_cold_first`
# extracted to `.pn95.demote_policy`. `_PN95_LAYER_ACCESS_COUNTS` +
# `_PN95_LAYER_ACCESS_RESET_THRESHOLD` stay defined in this module; the
# moved functions read/rebind them via lazy `_rt.X` (the «halve all
# counters» overflow path uses explicit attribute mutation that hits
# the same module-attribute slot the original `global` declaration did).
from .pn95.demote_policy import (  # noqa: E402
    _pn95_record_layer_promote,
    _pn95_sort_layers_cold_first,
)

# Path C v1.0 Quality-First Sprint Q1 A1 — lossless CPU prefix compression.
# Reduces effective CPU tier capacity 2-3× via zstd (or 1.5-2× via lz4).
# Detection at decompress is via magic bytes — no per-entry header overhead.
# Quality: 100% (lossless by construction).
_PN95_COMPRESS_LIB: Optional[str] = None  # 'zstd'|'lz4'|'zlib'|'none'|None
_PN95_COMPRESS_LEVEL: Optional[int] = None
_PN95_COMPRESS_MIN_BYTES = 256  # entries smaller skip compression (overhead)
# Sprint Q1 B6 — per-thread cached compressor/decompressor instances.
# threading.local ensures each ThreadPool worker has own cached instance
# (avoids race in singleton init AND any potential thread-safety nuance
# of underlying C library context state).
_PN95_ZSTD_TL = threading.local()


# M.4.2.C — six compression helpers extracted to `.pn95.compression`.
# State singletons (`_PN95_COMPRESS_LIB`, `_PN95_COMPRESS_LEVEL`,
# `_PN95_COMPRESS_MIN_BYTES`, `_PN95_ZSTD_TL`, `_PN95_COMPRESS_POOL`)
# stay defined in this module — four test files actively rebind them via
# ``monkeypatch.setattr(rt, ...)`` and direct ``rt._PN95_COMPRESS_POOL = None``
# writes, so moving the names would break the test contract. The moved
# functions reach the state through lazy ``_rt.X`` and replicate the
# original ``global ... = …`` rebinds via explicit attribute mutation.
# ``_PN95_COMPRESS_POOL`` keeps its definition in this module — see the
# state-singleton block below.
from .pn95.compression import (  # noqa: E402
    _pn95_init_compression,
    _pn95_compress_bytes,
    _pn95_compress_pool,
    _pn95_compress_bytes_batch,
    _pn95_decompress_bytes_batch,
    _pn95_decompress_bytes,
)

_PN95_COMPRESS_POOL: Optional[Any] = None  # ThreadPoolExecutor для parallel compress


# M.4.2.H — prefix-store accounting + register_block_pool extracted to
# `.pn95.prefix_store`. State (`_PN95_PREFIX_STORE`,
# `_PN95_PREFIX_STORE_BYTES_USED`, `_PN95_PREFIX_STORE_MAX_BYTES_CACHED`,
# `_PN95_BLOCK_POOL_REFS`, `_PN95_PREFIX_STORE_LOCK`) stays in this
# module. The original `global` rebinds become `_rt.X` attribute writes.
# Sibling-patch anchors at pn95_tier_aware_cache.py:181/334/342 import
# `register_block_pool` through _pn95_runtime — shim preserves the path.
from .pn95.prefix_store import (  # noqa: E402
    _prefix_store_max_bytes,
    _prefix_store_evict_until_fit,
    register_block_pool,
)


# M.4.2.H — `demote_on_evict` extracted to `.pn95.prefix_store`.
# Sibling-patch text-anchor in pn95_tier_aware_cache.py:213 imports
# this name through _pn95_runtime — shim preserves the path.
from .pn95.prefix_store import demote_on_evict  # noqa: E402


# M.4.2.H — `promote_on_miss` extracted to `.pn95.prefix_store`.
# Sibling-patch text-anchor in pn95_tier_aware_cache.py:437 imports
# this name through _pn95_runtime — shim preserves the path.
from .pn95.prefix_store import promote_on_miss  # noqa: E402


# M.4.2.D — `_select_cold_blocks_via_bpool_lru` extracted to
# `.pn95.demote_policy`. Reads `_PN95_BLOCK_POOL_REFS`, `_TM`, and
# `_PN95_PREFIX_STORE` (all stay here) via lazy `_rt.X`.
from .pn95.demote_policy import _select_cold_blocks_via_bpool_lru  # noqa: E402


# M.4.2.H — `worker_side_proactive_demote` + `_proactive_demote_cold`
# extracted to `.pn95.prefix_store`. Both were parked here in M.4.2.D
# because they call demote_on_evict.
from .pn95.prefix_store import (  # noqa: E402
    worker_side_proactive_demote,
    _proactive_demote_cold,
)


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
    if not _enabled() or _TM is None:
        return
    global _TICK_COUNTER, _TICK_LAST_FREE_MIB, _FREE_MIB_CACHE_VALID
    _TICK_COUNTER += 1
    _PN95_STATS["ticks_total"] += 1

    # OBS1 — periodic stats dump к JSON file для operator visibility
    # Throttled by GENESIS_PN95_STATS_INTERVAL (default 100 ticks),
    # disabled via GENESIS_PN95_STATS_FILE="" env. Fail-silent.
    _pn95_dump_stats_if_due()

    if _TICK_EVERY_CACHED is None:
        _refresh_env_cache()

    if _TICK_COUNTER % _TICK_EVERY_CACHED != 0:
        return

    _PN95_STATS["ticks_pressure_check"] += 1
    try:
        if _FREE_MIB_CACHE_VALID <= 0:
            free_mib = _gpu_free_mib()
            _TICK_LAST_FREE_MIB = free_mib
            _PN95_STATS["last_free_mib"] = free_mib
            _FREE_MIB_CACHE_VALID = _FREE_MIB_CACHE_TTL
        else:
            free_mib = _TICK_LAST_FREE_MIB
            _FREE_MIB_CACHE_VALID -= 1

        if free_mib <= 0 or free_mib >= _THRESHOLD_CACHED:
            return

        _FREE_MIB_CACHE_VALID = 0

        # [Genesis PN203] cold-prefix offload sweep — Tier 3.A core.
        # Runs BEFORE empty_cache so the demote path can populate L2 (PN95
        # pinned pool) with bytes that would otherwise be discarded.
        # Requires per-layer KV split (PN202) for correctness on hybrid models.
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
        target = _DEMOTE_BATCH_CACHED
        n_demoted = _proactive_demote_cold(target)

        if n_demoted == 0:
            # Legacy fallback — only fires if BlockPool refs not registered
            # or no cached blocks exist yet (cold start)
            views = getattr(_TM, "_attention_views", {}) or {}
            for layer_name, info in list(views.items())[:target]:
                num_blocks = int(info.get("num_blocks", 0))
                if num_blocks <= 0:
                    continue
                if _TM.demote_block(layer_name, 0):
                    n_demoted += 1
                if n_demoted >= target:
                    break

        if n_demoted > 0:
            _PN95_STATS["ticks_demote_triggered"] += 1
            _PN95_STATS["blocks_demoted_total"] += n_demoted
            _PN95_STATS["last_demote_count"] = n_demoted
            log.warning(
                "[PN95 v1.0 Phase 4.1] scheduler_tick: pressure (free=%d MiB "
                "< %d MiB) — demoted %d cold blocks via LRU "
                "(total demoted=%d, prefix_store_entries=%d)",
                free_mib, _THRESHOLD_CACHED, n_demoted,
                _PN95_STATS["blocks_demoted_total"],
                len(_PN95_PREFIX_STORE),
            )
    except Exception as e:  # pragma: no cover — defensive
        log.warning("[PN95 v1.0 Phase 4.1] scheduler_tick failed: %s", e)


# M.4.2.A — `tier_manager` + `reset_for_tests` extracted to
# `.pn95.runtime_state`. State ownership (`_TM`, `_LOCK`,
# `_LAST_GROUP_IDS_BY_HASH`) stays in this module; the moved functions
# read/rebind via lazy late-import (`_rt._TM = None` / `return _rt._TM`)
# so the local module attribute remains the canonical name.
from .pn95.runtime_state import tier_manager, reset_for_tests  # noqa: E402
