# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 85 — hybrid fine-shadow prefix cache (vllm#38182 followup).

================================================================
ROOT CAUSE (proven empirically + via deep code analysis)
================================================================

vLLM v1's prefix-cache for hybrid models (Qwen3.6-MoE GDN+Mamba+attention)
has TWO distinct mismatches that combine to make caching non-functional
for short-prompt single-user workloads:

**Mismatch A (short prompts < largest spec.block_size, e.g., < 2048):**

`single_type_kv_cache_manager.py:251` `SingleTypeKVCacheManager.cache_blocks`:

    num_full_blocks = num_tokens // self.block_size

For 1424-token requests with `MambaManager.self.block_size = 2048`:
- `num_full_blocks = 1424 // 2048 = 0` → early return → nothing stored.

The HybridKVCacheCoordinator then gates the final hit_length on the MIN
across all groups (kv_cache_coordinator.py:497-540 iterative loop).
Even though FullAttentionManager correctly stores 89 fine-grained hashes
for the 1424-token request, the Mamba group returns 0 hits → final
hit_tokens = 0.

**Mismatch B (long prompts ≥ 2048 tokens, e.g., 5018):**

`MambaManager.allocate_new_blocks` in align mode pads the prefix with
`null_block`s — only the LAST `1 + num_speculative_blocks` real blocks
are populated. So for `num_full_blocks = 5018 // 2048 = 2`:
- `new_full_blocks = blocks[0:2]` = `[null_block, null_block]`
- `block_pool.cache_full_blocks` skip-loops at `if blk.is_null: continue`
- **Zero entries actually inserted in `cached_block_hash_to_block`.**

Both mismatches manifest empirically as `hit_tokens = 0` even on three
identical requests in a row.

================================================================
PRIOR PATCHES (insufficient alone)
================================================================

- **P83** (skip Eagle pop): correct fix for one downstream symptom but
  pop site is never reached because hashes / coordinator paths cut earlier.
- **P84** (dual-site `hash_block_size` override): enables fine-grained
  hash COMPUTATION (89 hashes for 1424-token request) but doesn't fix
  the STORE/LOOKUP mismatch on Mamba group.

P83 + P84 together produce: `num_hashes=89` ✓ but still `hit_tokens=0`.

================================================================
P85 FIX — fine-shadow entries on Mamba store + lookup
================================================================

Approach (a) from architectural analysis: when MambaManager stores a
coarse Mamba block, ALSO register `scale_factor = mamba_block_size /
hash_block_size` shadow fine-hash entries in `cached_block_hash_to_block`,
all pointing to the SAME `KVCacheBlock`.

On Mamba lookup (find_longest_cache_hit), when env P85 is set, walk
fine hashes (request.block_hashes directly) instead of coarse-adapted
hashes. This finds matches that the coarse scan would miss.

**Memory model invariants preserved:**
- Same `KVCacheBlock` objects, no new allocations.
- No ref-count changes (shadow entries are pure lookup keys).
- Eviction safety: lookup branch verifies the cached block's
  `block_hash` field still matches our expected coarse hash before
  returning. On mismatch (block was recycled), treat as miss.

**For Mismatch A (short prompts):** Mamba never stores → no shadows
created → architectural limit honored (Mamba state genuinely cannot be
recovered from cache for prompts < block_size). P85 doesn't claim to
fix this — it's a fundamental limitation of incremental Mamba state.

**For Mismatch B (long prompts):** Mamba stores 1+ blocks → shadows
register → Mamba lookup finds matches → coordinator gate passes →
real cache hits → multi-turn TTFT improves.

================================================================
SAFETY MODEL
================================================================

- Default OFF. Opt-in via `GENESIS_ENABLE_P85=1`.
- Both store and lookup hunks gated on the same env. Only ONE side
  enabled would be a no-op (lookup finds nothing, or store creates
  unused shadows — both safe).
- Stale-shadow eviction safety: lookup branch verifies
  `cached_block.block_hash == expected_coarse_hash` before returning.
- Drift detection on both anchor sites.

Status: opt-in via `GENESIS_ENABLE_P85=1`. Default OFF.

================================================================
v2 (2026-06-11) — both-sites re-anchor + PN346 composition
================================================================

Preflight residual triage action plan section 5 (pin
0.22.1rc1.dev259+g303916e93):

**Site 1 re-anchor:** upstream widened `MambaManager.cache_blocks` to
the new `retention_interval` keyword signature (pristine
`v1/core/single_type_kv_cache_manager.py` lines 1211-1229). The v1
single-line-signature anchor matched zero times on this pin, so the
required Site 1 sub-patch failed every boot. v2 re-derives
P85_SITE1_OLD/NEW byte-exact from pristine and forwards
`retention_interval` through `super().cache_blocks` unchanged.

**Site 2 dual anchor variants (verifier-mandated):** sibling PN346
(vendor of vllm#43650; effectively default-ON — only
`GENESIS_DISABLE_PN346` is honored; boot-dispatched BEFORE P85)
rewrites a byte-identical 4-line subsequence inside P85_SITE2_OLD,
inserting 12 lines mid-anchor. A pristine-only Site 2 anchor passes
pristine preflight but fails on every real (post-PN346) boot — the
P18B-on-PN119 mirror class. v2 therefore carries TWO Site 2 variants
with required-at-least-one semantics (both `required=False`; kernel
soft-skips the absent variant), per the P18B / PN32-on-PN79 chain
convention:

  - pristine-shaped (`P85_SITE2_OLD`) — PN346 disabled;
  - post-PN346-shaped (`P85_SITE2_OLD_POST_PN346`) — assembled
    textually from PN346's own PN346_ANCHOR_OLD/NEW constants so the
    two modules cannot silently diverge. Its replacement carries
    PN346's `drop_eagle_block` boundary guard in the coarse fallback
    (P85 must not undo PN346's GSM8K accuracy fix).

Because Site 1 stays `required=True`, the kernel's all-miss SKIP can
never fire for a Site-2-only drift; `apply()` adds an explicit
`site2_anchor_present` pre-gate returning a structured skip BEFORE any
write when neither Site 2 variant matches (a Site-1-only half-apply
would be a store-side no-op hiding the drift behind the marker).

Apply order: PN346 boot-dispatches before P85 (post-PN346 variant
fires). The reverse order also composes — P85's pristine-variant
replacement re-emits the 4-line coarse fallback, so PN346's anchor
still matches exactly once inside it.

**P84 dependency retired:** P84 (env-based hash_block_size override)
was retired 2026-06-11 — both its sites are upstream-native on this
pin: the Scheduler accepts an explicit `hash_block_size` parameter
(pristine v1/core/sched/scheduler.py:72,229-230,242) and
`resolve_kv_cache_block_sizes` (kv_cache_utils.py:593) provides the
GCD default + the `cache_config.hash_block_size` override + the
divisibility ValueError. P85's fine hashes now come from the
upstream-native `--hash-block-size <N>` engine arg
(cache_config.hash_block_size) instead of GENESIS_P84_HASH_BLOCK_SIZE.
CAVEAT (verifier, plan section 3): the Mamba back-off
(kv_cache_utils.py:639-644) returns the coarse scheduler block size
BEFORE consulting the override whenever a Mamba group's block_size
diverges from cache_config.block_size (mamba_cache_mode != "align") —
fine hashing only engages with mamba_cache_mode="align"; verify
num_hashes>0 server-side on the prod prefix-caching config.

NOTE (plan section 6, pending batch): the `_g_p85_` upstream drift
marker is a substring of this patch's own replacement text
(self-collision lint class). Remediation belongs to the section 6
batch, sequenced after this fix.

Tunable knobs
-------------
- `GENESIS_ENABLE_P85` (default unset/0): master switch
- Requires upstream-native `--hash-block-size <N>`
  (cache_config.hash_block_size) with N dividing every group
  block_size (typically 16), and mamba_cache_mode="align" so the
  Mamba back-off in resolve_kv_cache_block_sizes does not disable
  fine hashing. (Replaces the retired P84 env override.)

Compatibility
-------------
- Mamba / GDN / hybrid models (where this patch is needed).
- Pure attention models: P85 hunks would activate but shadow registration
  is a no-op (block_size == hash_block_size → scale_factor = 1, shadows
  are duplicates of coarse).
- MTP / Eagle / ngram: cache layer agnostic to spec method.

Author: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa.
Discovery: 6-round empirical investigation 2026-04-27 + deep code analysis
synthesis. See sprint report SPRINT_REPORT_20260427_phase4_*.md.
Related: P83 (Eagle pop skip), P84 (dual-site hash_block_size override)
— both retired 2026-06-11 (upstream-native on pin 0.22.1rc1.dev259);
PN346 (Mamba/GDN MTP+APC boundary fix, composes via Site 2 dual
variants — see v2 section above).
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root
from sndr.engines.vllm.patches.kv_cache.pn346_mamba_mtp_apc_boundary import (
    PN346_ANCHOR_NEW as _PN346_NEW,
    PN346_ANCHOR_OLD as _PN346_OLD,
)
from sndr.kernel import (
    TextPatcher,
    TextPatch,
)

log = logging.getLogger("genesis.wiring.p85_hybrid_fine_shadow_prefix_cache")


GENESIS_P85_MARKER = "Genesis P85 hybrid fine-shadow prefix cache (vllm#38182 followup) v7.53.8_debug"


# ─── Site 1: MambaManager.cache_blocks adds shadow entries ────────────────
#
# v3 re-anchor (2026-06-24, pin 0.23.1rc1.dev301+g04c2a8dea): upstream
# rewrote the cache_blocks loop body — the old
# ``if block.is_null: continue / assert block.block_hash is not None``
# pair was folded into a single ``if block.is_null or block.block_hash
# is None: continue`` plus a 4-line sparse-retention comment (pristine
# v1/core/single_type_kv_cache_manager.py lines 1268-1287, byte-exact,
# count==1 verified against the live dev301 pristine tree). The
# ``retention_interval`` signature (v2 re-anchor, dev259) is unchanged.
# P85 only APPENDS the shadow registration block — the upstream body
# (including the new combined null/no-hash guard) is preserved verbatim,
# em-dash included.

_P85_SITE1_TAIL = "\n    def new_step_starts(self) -> None:\n"

_P85_SITE1_BODY = (
    "    def cache_blocks(\n"
    "        self,\n"
    "        request: Request,\n"
    "        num_tokens: int,\n"
    "        retention_interval: int | None = None,\n"
    "    ) -> None:\n"
    "        num_cached_blocks_before = self.num_cached_block.get(request.request_id, 0)\n"
    "        super().cache_blocks(request, num_tokens, retention_interval=retention_interval)\n"
    "        num_cached_blocks_after = self.num_cached_block.get(request.request_id, 0)\n"
    "        if num_cached_blocks_after > num_cached_blocks_before:\n"
    "            for block in self.req_to_blocks[request.request_id][\n"
    "                num_cached_blocks_before:num_cached_blocks_after\n"
    "            ]:\n"
    "                # Skip null blocks (align-mode skipped states) and blocks that\n"
    "                # were not cached this step — with sparse retention\n"
    "                # (reachable_block_mask) the intermediate state snapshots carry\n"
    "                # no hash and must not be recorded as cached-this-step.\n"
    "                if block.is_null or block.block_hash is None:\n"
    "                    continue\n"
    "                self.cached_blocks_this_step.add(block.block_hash)\n"
)

P85_SITE1_OLD = _P85_SITE1_BODY + _P85_SITE1_TAIL

_P85_SITE1_SHADOW_BLOCK = (
    "        # ════════════════════════════════════════════════════════════════\n"
    "        # [Genesis P85] Shadow fine-grained hash entries for hybrid lookup.\n"
    "        # When GENESIS_ENABLE_P85=1 and self.block_size != hash_block_size,\n"
    "        # also register fine sub-block hash entries (one per hash_block_size\n"
    "        # tokens) pointing at the coarse Mamba block. This lets lookup find\n"
    "        # multi-turn cache hits that the coarse-only store would miss.\n"
    "        # ════════════════════════════════════════════════════════════════\n"
    "        import os as _g_p85_os\n"
    "        if _g_p85_os.environ.get('GENESIS_ENABLE_P85', '').strip().lower() in (\n"
    "                '1', 'true', 'yes', 'on'):\n"
    "            from vllm.v1.core.kv_cache_utils import (\n"
    "                make_block_hash_with_group_id as _g_p85_mk)\n"
    "            _g_p85_hbs = self.block_pool.hash_block_size\n"
    "            if self.block_size != _g_p85_hbs and self.block_size % _g_p85_hbs == 0:\n"
    "                _g_p85_scale = self.block_size // _g_p85_hbs\n"
    "                _g_p85_blocks = self.req_to_blocks[request.request_id]\n"
    "                _g_p85_fine = request.block_hashes\n"
    "                _g_p85_committed = self.num_cached_block.get(request.request_id, 0)\n"
    "                for _g_p85_i in range(_g_p85_committed):\n"
    "                    _g_p85_blk = _g_p85_blocks[_g_p85_i]\n"
    "                    if _g_p85_blk.is_null:\n"
    "                        continue\n"
    "                    _g_p85_base = _g_p85_i * _g_p85_scale\n"
    "                    _g_p85_end = _g_p85_base + _g_p85_scale\n"
    "                    if _g_p85_end > len(_g_p85_fine):\n"
    "                        break\n"
    "                _g_p85_inserted = 0\n"
    "                _g_p85_skipped_null = 0\n"
    "                for _g_p85_i in range(_g_p85_committed):\n"
    "                    _g_p85_blk = _g_p85_blocks[_g_p85_i]\n"
    "                    if _g_p85_blk.is_null:\n"
    "                        _g_p85_skipped_null += 1\n"
    "                        continue\n"
    "                    _g_p85_base = _g_p85_i * _g_p85_scale\n"
    "                    _g_p85_end = _g_p85_base + _g_p85_scale\n"
    "                    if _g_p85_end > len(_g_p85_fine):\n"
    "                        break\n"
    "                    for _g_p85_j in range(_g_p85_base, _g_p85_end):\n"
    "                        _g_p85_key = _g_p85_mk(\n"
    "                            _g_p85_fine[_g_p85_j], self.kv_cache_group_id)\n"
    "                        self.block_pool.cached_block_hash_to_block.insert(\n"
    "                            _g_p85_key, _g_p85_blk)\n"
    "                        _g_p85_inserted += 1\n"
    "                if _g_p85_os.environ.get('GENESIS_P85_DEBUG', '') == '1':\n"
    "                    import sys as _g_p85_sys\n"
    "                    _g_p85_sys.stderr.write(\n"
    "                        '[GENESIS_P85_STORE] req=' + request.request_id[:8]\n"
    "                        + ' bs=' + str(self.block_size)\n"
    "                        + ' hbs=' + str(_g_p85_hbs)\n"
    "                        + ' scale=' + str(_g_p85_scale)\n"
    "                        + ' committed=' + str(_g_p85_committed)\n"
    "                        + ' skipped_null=' + str(_g_p85_skipped_null)\n"
    "                        + ' shadows_inserted=' + str(_g_p85_inserted)\n"
    "                        + ' fine_hashes=' + str(len(_g_p85_fine))\n"
    "                        + '\\n')\n"
    "                    _g_p85_sys.stderr.flush()\n"
)

P85_SITE1_NEW = _P85_SITE1_BODY + _P85_SITE1_SHADOW_BLOCK + _P85_SITE1_TAIL


# ─── Site 2: MambaManager.find_longest_cache_hit walks fine hashes ────────
#
# The pristine-shaped variant below byte-matches pristine lines 988-1016
# (count==1 verified 2026-06-11 against the live pristine tree and the
# committed fixture). The post-PN346 variant is assembled from PN346's
# own constants further down.

P85_SITE2_OLD = (
    "        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(\n"
    "            [] for _ in range(len(kv_cache_group_ids))\n"
    "        )\n"
    "\n"
    "        block_size = kv_cache_spec.block_size\n"
    "        max_num_blocks = max_length // block_size\n"
    "        # Search from right to left and early stop when a match is found.\n"
    "        for i in range(max_num_blocks - 1, -1, -1):\n"
    "            if cached_block := block_pool.get_cached_block(\n"
    "                block_hashes[i], kv_cache_group_ids\n"
    "            ):\n"
    "                # When enable Mamba prefix caching, `block_size` will be aligned\n"
    "                # across full attention layers and Mamba layers to ensure the\n"
    "                # prefix hit length aligned at block\n"
    "                if (\n"
    "                    block_size != alignment_tokens  # Faster for common case.\n"
    "                    and (i + 1) * block_size % alignment_tokens != 0\n"
    "                ):\n"
    "                    continue\n"
    "                for computed, cached in zip(computed_blocks, cached_block):\n"
    "                    # the hit length logic later assumes:\n"
    "                    #  hit_length = len(hit_blocks_other_attn[0])\n"
    "                    #               * self.other_block_size\n"
    "                    # so we insert dummy blocks at the beginning:\n"
    "                    computed.extend([block_pool.null_block] * i)\n"
    "                    computed.append(cached)\n"
    "                break  # we just need the last match - early stopping\n"
    "\n"
    "        return computed_blocks\n"
)

P85_SITE2_NEW = (
    "        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(\n"
    "            [] for _ in range(len(kv_cache_group_ids))\n"
    "        )\n"
    "\n"
    "        # ════════════════════════════════════════════════════════════════\n"
    "        # [Genesis P85] Fine-shadow lookup branch.\n"
    "        # When env is set AND fine hashes are available (BlockHashListWithBlockSize\n"
    "        # adapter NOT applied — i.e. coordinator passed raw fine hashes),\n"
    "        # scan at fine granularity to find shadows registered by\n"
    "        # MambaManager.cache_blocks. Eviction-safe: re-derive the coarse\n"
    "        # hash and verify cached_block.block_hash matches before returning.\n"
    "        # ════════════════════════════════════════════════════════════════\n"
    "        import os as _g_p85_os2\n"
    "        if _g_p85_os2.environ.get('GENESIS_ENABLE_P85', '').strip().lower() in (\n"
    "                '1', 'true', 'yes', 'on'):\n"
    "            from vllm.v1.core.kv_cache_utils import (\n"
    "                BlockHashListWithBlockSize as _g_p85_BHLBS,\n"
    "                make_block_hash_with_group_id as _g_p85_mk2,\n"
    "            )\n"
    "            _g_p85_hbs2 = block_pool.hash_block_size\n"
    "            if (kv_cache_spec.block_size != _g_p85_hbs2\n"
    "                    and kv_cache_spec.block_size % _g_p85_hbs2 == 0\n"
    "                    and not isinstance(block_hashes, _g_p85_BHLBS)):\n"
    "                _g_p85_scale2 = kv_cache_spec.block_size // _g_p85_hbs2\n"
    "                _g_p85_max_fine = max_length // _g_p85_hbs2\n"
    "                _g_p85_max_fine = min(_g_p85_max_fine, len(block_hashes))\n"
    "                _g_p85_align_fine = alignment_tokens // _g_p85_hbs2\n"
    "                # Search from right to left at fine granularity, but only\n"
    "                # consider indices that align with both scale and alignment.\n"
    "                for _g_p85_i in range(_g_p85_max_fine - 1, -1, -1):\n"
    "                    if (_g_p85_i + 1) % _g_p85_scale2 != 0:\n"
    "                        continue\n"
    "                    if _g_p85_align_fine > 0 and (_g_p85_i + 1) % _g_p85_align_fine != 0:\n"
    "                        continue\n"
    "                    _g_p85_fine_key = _g_p85_mk2(\n"
    "                        block_hashes[_g_p85_i], kv_cache_group_ids[0])\n"
    "                    _g_p85_cached = block_pool.get_cached_block(\n"
    "                        block_hashes[_g_p85_i], kv_cache_group_ids)\n"
    "                    if _g_p85_cached is None:\n"
    "                        continue\n"
    "                    # Eviction safety: the cached block's block_hash field\n"
    "                    # is the COARSE hash (set by block_pool.cache_full_blocks).\n"
    "                    # Re-derive it and verify match. If block was evicted +\n"
    "                    # recycled, the field will mismatch → treat as miss.\n"
    "                    _g_p85_coarse_base = (_g_p85_i + 1 - _g_p85_scale2)\n"
    "                    _g_p85_coarse_end = _g_p85_i + 1\n"
    "                    if _g_p85_coarse_end > len(block_hashes):\n"
    "                        continue\n"
    "                    _g_p85_merged = bytes(block_hashes[_g_p85_coarse_base])\n"
    "                    for _g_p85_j2 in range(_g_p85_coarse_base + 1, _g_p85_coarse_end):\n"
    "                        _g_p85_merged += bytes(block_hashes[_g_p85_j2])\n"
    "                    _g_p85_expected_coarse = _g_p85_mk2(\n"
    "                        _g_p85_merged, kv_cache_group_ids[0])\n"
    "                    _g_p85_first = _g_p85_cached[0]\n"
    "                    if _g_p85_first.block_hash != _g_p85_expected_coarse:\n"
    "                        # Stale shadow — block was evicted/recycled. Skip.\n"
    "                        continue\n"
    "                    # Coarse-level index from fine: (i+1)/scale - 1\n"
    "                    _g_p85_coarse_idx = (_g_p85_i + 1) // _g_p85_scale2 - 1\n"
    "                    for computed, cached in zip(computed_blocks, _g_p85_cached):\n"
    "                        computed.extend(\n"
    "                            [block_pool.null_block] * _g_p85_coarse_idx)\n"
    "                        computed.append(cached)\n"
    "                    return computed_blocks\n"
    "                # No fine match — fall through to coarse logic below\n"
    "                # (which will likely return empty for short prompts).\n"
    "        block_size = kv_cache_spec.block_size\n"
    "        max_num_blocks = max_length // block_size\n"
    "        # Search from right to left and early stop when a match is found.\n"
    "        for i in range(max_num_blocks - 1, -1, -1):\n"
    "            if cached_block := block_pool.get_cached_block(\n"
    "                block_hashes[i], kv_cache_group_ids\n"
    "            ):\n"
    "                # When enable Mamba prefix caching, `block_size` will be aligned\n"
    "                # across full attention layers and Mamba layers to ensure the\n"
    "                # prefix hit length aligned at block\n"
    "                if (\n"
    "                    block_size != alignment_tokens  # Faster for common case.\n"
    "                    and (i + 1) * block_size % alignment_tokens != 0\n"
    "                ):\n"
    "                    continue\n"
    "                for computed, cached in zip(computed_blocks, cached_block):\n"
    "                    # the hit length logic later assumes:\n"
    "                    #  hit_length = len(hit_blocks_other_attn[0])\n"
    "                    #               * self.other_block_size\n"
    "                    # so we insert dummy blocks at the beginning:\n"
    "                    computed.extend([block_pool.null_block] * i)\n"
    "                    computed.append(cached)\n"
    "                break  # we just need the last match - early stopping\n"
    "\n"
    "        return computed_blocks\n"
)


# ─── Site 2, post-PN346 variant (chain convention) ────────────────────────
#
# PN346 (vendor of vllm#43650, boot-dispatched BEFORE P85, effectively
# default-ON) rewrites the 4-line coarse-search subsequence inside
# P85_SITE2_OLD, inserting 12 lines mid-anchor. The post-PN346 variant
# is assembled textually from PN346's own PN346_ANCHOR_OLD/NEW constants
# (PN32-imports-PN79 precedent) so the two modules cannot silently
# diverge. The replacement's coarse fallback thereby carries PN346's
# drop_eagle_block boundary guard — P85 must not undo PN346's GSM8K
# accuracy fix.

if (
    P85_SITE2_OLD.count(_PN346_OLD) == 1
    and P85_SITE2_NEW.count(_PN346_OLD) == 1
):
    P85_SITE2_OLD_POST_PN346 = P85_SITE2_OLD.replace(_PN346_OLD, _PN346_NEW, 1)
    P85_SITE2_NEW_POST_PN346 = P85_SITE2_NEW.replace(_PN346_OLD, _PN346_NEW, 1)
else:
    # PN346's anchor constants changed shape — the post-PN346 variant
    # can no longer be assembled safely. Disable it with a
    # never-matching sentinel (the pristine variant still works for
    # PN346-disabled deployments) and fail loudly in the unit test
    # (test_post_pn346_old_built_from_pn346_constants).
    log.warning(
        "[P85 v2] PN346_ANCHOR_OLD no longer appears exactly once in "
        "P85_SITE2_OLD/NEW — disabling the post-PN346 anchor variant. "
        "Re-verify P85/PN346 composition."
    )
    P85_SITE2_OLD_POST_PN346 = (
        "# [Genesis P85 v2 sentinel — post-PN346 variant disabled, "
        "PN346 anchors drifted]\n"
    )
    P85_SITE2_NEW_POST_PN346 = P85_SITE2_OLD_POST_PN346


def build_sub_patches() -> list[TextPatch]:
    """Site 1 (required) + the two Site 2 anchor variants.

    Site 2 variants are both `required=False` (required-at-least-one
    semantics — the kernel soft-skips the variant whose anchor is
    absent). They are mutually exclusive by construction: the
    post-PN346 anchor contains PN346's `[Genesis PN346` comment lines,
    absent from pristine, and PN346's apply breaks the contiguous
    4-line run the pristine anchor needs. Verified in
    tests/unit/integrations/kv_cache/test_p85_dual_anchor_2026_06_11.py.

    Because Site 1 is required=True the kernel's all-miss SKIP cannot
    fire for a Site-2-only drift — `apply()` pre-gates on
    `site2_anchor_present` before any write.
    """
    return [
        TextPatch(
            name="p85_mamba_cache_blocks_shadow",
            anchor=P85_SITE1_OLD,
            replacement=P85_SITE1_NEW,
            required=True,
        ),
        TextPatch(
            name="p85_mamba_find_longest_cache_hit_fine_pristine",
            anchor=P85_SITE2_OLD,
            replacement=P85_SITE2_NEW,
            required=False,
        ),
        TextPatch(
            name="p85_mamba_find_longest_cache_hit_fine_post_pn346",
            anchor=P85_SITE2_OLD_POST_PN346,
            replacement=P85_SITE2_NEW_POST_PN346,
            required=False,
        ),
    ]


def site2_anchor_present(content: str) -> bool:
    """True iff at least one Site 2 anchor variant matches `content`.

    Required-at-least-one belt for `apply()`: a Site-1-only half-apply
    would be a store-side no-op (shadows registered, never looked up)
    that hides Site 2 anchor drift behind the idempotency marker.
    """
    return (
        P85_SITE2_OLD in content
        or P85_SITE2_OLD_POST_PN346 in content
    )


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/core/single_type_kv_cache_manager.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "P85 v1/core/single_type_kv_cache_manager.py — hybrid fine-shadow "
            "prefix cache (vllm#38182 followup)"
        ),
        target_file=str(target),
        marker=GENESIS_P85_MARKER,
        sub_patches=build_sub_patches(),
        # Self-collision lint (triage plan §6 2026-06-11, remediated in the
        # section 6 batch): former entry "_g_p85_" was a substring of this
        # patch's own replacement text — false "upstream_merged" skip on
        # partial file_cache divergence. Residue coverage stays with the
        # "[Genesis P85" banner.
        upstream_drift_markers=[
            "[Genesis P85",
            # Upstream-side markers if vLLM ships its own hybrid fine cache:
            "MambaManager.find_longest_cache_hit_fine",
            "fine_shadow_prefix_cache",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P85 — hybrid fine-shadow prefix cache."""
    from sndr.dispatcher import should_apply, log_decision
    decision, reason = should_apply("P85")
    log_decision("P85", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "vllm/v1/core/single_type_kv_cache_manager.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        log.info("[P85] marker present — skip (idempotent)")
        return "applied", "idempotent (marker present)"
    for m in patcher.upstream_drift_markers:
        if m == "[Genesis P85" and m in content:
            continue
        if m in content:
            return (
                "skipped",
                f"upstream drift marker {m!r} in {patcher.target_file} — "
                "upstream may have absorbed hybrid fine-cache fix",
            )

    # Required-at-least-one pre-gate (v2, plan section 5): Site 2's two
    # variants are required=False so the kernel cannot abort on a
    # Site-2-only drift while Site 1 still matches. Skip BEFORE any
    # write — a Site-1-only half-apply would be a store-side no-op
    # hiding the drift behind the idempotency marker.
    if not site2_anchor_present(content):
        return (
            "skipped",
            "P85: Site 2 anchor (MambaManager.find_longest_cache_hit) "
            "absent in both pristine-shaped and post-PN346-shaped "
            "variants — anchor drift; file left untouched",
        )

    result, failure = patcher.apply()
    # Audit P1 fix 2026-05-05: route SKIPPED/IDEMPOTENT honestly via shared helper
    from sndr.kernel import result_to_wiring_status
    return result_to_wiring_status(
        result, failure,
        applied_message=(
            "P85 applied: hybrid fine-shadow prefix cache installed at MambaManager. "
            "cache_blocks now registers fine-grained shadow entries; "
            "find_longest_cache_hit prefers fine-scan with eviction-safety verify "
            "(variant: " + ", ".join(patcher.applied_sub_patches or ["?"]) + "). "
            "Requires upstream-native --hash-block-size <N> "
            "(cache_config.hash_block_size) + mamba_cache_mode=align for fine "
            "hashes to be computed in the first place (P84 retired 2026-06-11)."
        ),
        patch_name=patcher.patch_name,
    )
