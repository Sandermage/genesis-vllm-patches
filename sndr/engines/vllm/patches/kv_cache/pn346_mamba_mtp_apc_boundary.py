# SPDX-License-Identifier: Apache-2.0
"""PN346 — vendor of OPEN PR vllm#43650 (Mamba/GDN prefix-cache + MTP boundary fix).

Background — silent accuracy regression on our exact PROD shape
==============================================================

Issue vllm#43559 reports that **Qwen3.5/3.6-35B-A3B FP8 hybrid GDN +
MTP K=3 + ``--enable-prefix-caching``** silently loses 1.6 pp accuracy
on GSM8K compared to the no-MTP-no-APC baseline. The bug is a
mismatch between how ``FullAttentionManager`` and ``MambaManager``
treat the *final* matched block during prefix-cache lookup when
EAGLE-style speculation drops the last block.

Concrete behaviour per upstream PR #43650 author bench (3-run):

  * **no MTP + APC**:           GSM8K accuracy 0.916 (baseline)
  * **old MTP + APC** (bug):    0.900 (-1.6 pp, silent)
  * **new MTP + APC** (fix):    0.914 (recovered, within noise of baseline)

Root cause — different state structure
======================================

``FullAttentionManager.find_longest_cache_hit`` (line 523 in our pin's
``vllm/v1/core/single_type_kv_cache_manager.py``) handles the EAGLE
case by **popping the last matched block AFTER the search loop**::

  for i in range(max_num_blocks):
      if cached := block_pool.get_cached_block(block_hashes[i], ...):
          for computed, c in zip(computed_blocks, cached):
              computed.append(c)
      else:
          break
  if drop_eagle_block and computed_blocks[0]:
      # Need to drop the last matched block if eagle is enabled.
      for computed in computed_blocks:
          computed.pop()

The pop is safe because full-attention cache blocks hold token KVs;
removing the last one just walks the hit length back by one block.

``MambaManager.find_longest_cache_hit`` (line 986 in our pin) cannot
do this — Mamba/GDN state cache layout is::

  [null_block, null_block, ..., null_block, state_block]

i.e. the LAST block in the returned list IS the SSM state itself, and
every block before it is a placeholder ``null_block``. Popping the
last block destroys the only state we have — and the downstream code
then keeps the placeholders, which encode "we have all of these
tokens covered" but with no actual state to start from. The MTP
verify step reads stale/null state, produces a slightly off
distribution, and the K-step accept rate drops just enough to shift
accuracy by ~1-1.6 pp on knowledge-recall benchmarks like GSM8K.

The right fix: instead of matching one extra block then popping it,
**search up to but not including the final block** when EAGLE is
active. This is what PR #43650 does in six lines.

Translation note — anchor variable name
=======================================

The upstream PR's author worked off a private fork where the local
variable was renamed ``use_eagle``. Our pin (and upstream main as of
2026-06-09) still uses the function parameter ``drop_eagle_block``.
The Genesis patch uses the parameter name that exists in our actual
file. Verified live in the container at
``/usr/local/lib/python3.12/dist-packages/vllm/v1/core/single_type_kv_cache_manager.py``
line 986 (MambaManager.find_longest_cache_hit signature has
``drop_eagle_block: bool`` — no ``use_eagle`` exists).

Why we vendor this OPEN PR (not just wait for upstream merge)
=============================================================

  * Bug actively hits us — Qwen3.5/3.6 + MTP + APC is our exact PROD
    config; the model class is the same one the author benched.
  * The fix is **6 lines**, surgical, no perf cost when EAGLE-off.
  * Latency cost when EAGLE-on is real (author measured: QPS dropped
    18.6 → 15.5, output tps 2494 → 1983) but **accuracy is restored**
    to baseline. This is correctness, not perf.
  * The risk of an undetected ~1.6 pp accuracy hole in our prod stack
    is higher than the cost of a 15-20 % QPS hit on the prefix-cache
    overlap path. (Most of our requests don't hit overlapping prefix
    cache, so the QPS impact is much smaller in practice than the
    author's worst-case bench.)
  * Author claim: "The increase in latency is expected as the fix makes
    MTP prefix-cache lookup recompute one block at the speculative
    boundary instead of reusing it, this aligns with the full attention
    behavior."

What this patch changes
=======================

Adds a 6-line guard inside ``MambaManager.find_longest_cache_hit``
just AFTER the ``max_num_blocks = max_length // block_size`` line and
BEFORE the ``for i in range(max_num_blocks - 1, -1, -1):`` search
loop. The guard decrements ``max_num_blocks`` by 1 when
``drop_eagle_block`` is True (i.e. EAGLE / MTP is active), so the
search never touches the final state block.

The function signature already has ``drop_eagle_block: bool`` — no
new parameter is introduced, no caller change needed.

Composition + safety
====================

  * ANCHOR OVERLAP with P85 (correction 2026-06-11, plan section 5 —
    the original "no anchor overlap" claim was wrong): PN346's 4-line
    anchor is a byte-identical subsequence inside P85's Site 2 anchor
    (``MambaManager.find_longest_cache_hit``). PN346 boot-dispatches
    BEFORE P85; P85 v2 composes via dual Site 2 anchor variants
    (pristine-shaped + post-PN346-shaped assembled from THIS module's
    PN346_ANCHOR_OLD/NEW constants — do not rename them without
    updating P85 and its dual-anchor test). P85 remains strict opt-in
    via GENESIS_ENABLE_P85, default OFF.
  * No-op when ``drop_eagle_block=False`` (EAGLE / MTP disabled).
  * Composes with PN340 + PN341 + PN345 + PN29 + PN204 + PN286 + P85.

Risk: LOW — six-line surgical guard, only fires on EAGLE / MTP path.
Effort: XS.

Source rationale: Sander 2026-06-09 — comprehensive 3-week
multi-agent research synthesis (see journal
2026-06-09-comprehensive-research-roadmap.md).

Author: Sander Barzov Aleksandr (Sandermage, Ukraine, Odessa).
Vendor target: vllm-project/vllm#43650 (open as of 2026-06-09).
Closes vllm#43559.
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file
from sndr.kernel import TextPatch, TextPatcher, TextPatchResult

log = logging.getLogger("genesis.wiring.pn346_mamba_mtp_apc_boundary")

GENESIS_PN346_MARKER = (
    "Genesis PN346 vendor of vllm#43650 (Mamba/GDN + MTP + APC accuracy fix) v1"
)


# Anchor: the three-line sequence inside MambaManager.find_longest_cache_hit
# right after ``computed_blocks`` is initialized but before the search loop.
# We match on 4 lines of context (block_size + max_num_blocks + the comment
# + the for-loop opening) so the anchor is uniquely owned by MambaManager —
# the ChunkedLocalAttentionManager has a similar-shaped sequence but uses
# ``max_length // kv_cache_spec.block_size`` (with the dot access in place
# of the locally-bound ``block_size``) so the anchor below is unique to
# MambaManager.
PN346_ANCHOR_OLD = (
    "        block_size = kv_cache_spec.block_size\n"
    "        max_num_blocks = max_length // block_size\n"
    "        # Search from right to left and early stop when a match is found.\n"
    "        for i in range(max_num_blocks - 1, -1, -1):\n"
)

PN346_ANCHOR_NEW = (
    "        block_size = kv_cache_spec.block_size\n"
    "        max_num_blocks = max_length // block_size\n"
    "        # [Genesis PN346 vendor of vllm#43650] Mamba/GDN state cache layout is\n"
    "        # [null, null, ..., null, state_block] where the LAST block holds the\n"
    "        # only SSM state. FullAttentionManager's drop_eagle_block path matches\n"
    "        # one extra block then pops it AFTER the loop, which is safe for token\n"
    "        # KVs but on Mamba would destroy the state. Instead, when EAGLE / MTP\n"
    "        # is active, walk the search boundary back by one block BEFORE the loop\n"
    "        # so we never include the partially-accepted final state block.\n"
    "        # This recovers the 1.6 pp GSM8K accuracy lost on MTP + prefix-caching\n"
    "        # overlap (vllm#43559). Cost: small QPS regression on overlapping prefix\n"
    "        # path; benefit: parity with full-attention drop_eagle_block semantics.\n"
    "        if drop_eagle_block and max_num_blocks > 0:\n"
    "            max_num_blocks -= 1\n"
    "        # Search from right to left and early stop when a match is found.\n"
    "        for i in range(max_num_blocks - 1, -1, -1):\n"
)


def _env_disabled() -> bool:
    return os.environ.get("GENESIS_DISABLE_PN346", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def apply() -> tuple[str, str]:
    """Apply PN346 — Mamba/GDN cache hit boundary fix on EAGLE / MTP path."""
    if _env_disabled():
        return "skipped", "PN346 disabled via GENESIS_DISABLE_PN346=1"

    target = resolve_vllm_file("v1/core/single_type_kv_cache_manager.py")
    if target is None:
        return "skipped", "v1/core/single_type_kv_cache_manager.py not found"

    patcher = TextPatcher(
        patch_name=(
            "PN346 vllm/v1/core/single_type_kv_cache_manager.py — "
            "Mamba/GDN cache hit boundary fix on EAGLE / MTP path"
        ),
        target_file=str(target),
        marker=GENESIS_PN346_MARKER,
        sub_patches=[
            TextPatch(
                name="pn346_mamba_drop_eagle_boundary",
                anchor=PN346_ANCHOR_OLD,
                replacement=PN346_ANCHOR_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN346",
            # upstream sentinel if PR #43650 merges as-is with use_eagle
            "if use_eagle and max_num_blocks > 0:",
            # NOTE: the renamed form `if drop_eagle_block and max_num_blocks > 0:`
            # is intentionally NOT a drift marker — it self-collides with
            # PN346_ANCHOR_NEW (this patch's own replacement), which
            # tools/lint_drift_markers.py flags. The renamed-fork merge case is
            # still covered by anchor-absence at apply time.
        ],
    )

    try:
        result, failure = patcher.apply()
    except Exception as e:  # noqa: BLE001
        return "failed", f"PN346 apply raised {e!r}"

    if result == TextPatchResult.FAILED:
        reason = failure.reason if failure else "unknown"
        return "failed", f"PN346: {reason}"
    if result == TextPatchResult.SKIPPED:
        reason = failure.reason if failure else "unknown"
        return "skipped", f"PN346: {reason}"
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "PN346 already applied (idempotent)"

    return "applied", (
        "PN346 applied: MambaManager.find_longest_cache_hit now walks the "
        "search boundary back by one block when drop_eagle_block=True. "
        "Recovers ~1.6 pp GSM8K accuracy on the Qwen3.5/3.6-A3B + MTP K=3 + "
        "--enable-prefix-caching overlap path. Vendor of OPEN PR "
        "vllm#43650 (closes #43559). 6-LOC surgical patch. No-op when EAGLE "
        "/ MTP is disabled. Composes with PN340 + PN341 + PN345."
    )


def is_applied() -> bool:
    from pathlib import Path
    target = resolve_vllm_file("v1/core/single_type_kv_cache_manager.py")
    if target is None:
        return False
    try:
        return GENESIS_PN346_MARKER in Path(str(target)).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
