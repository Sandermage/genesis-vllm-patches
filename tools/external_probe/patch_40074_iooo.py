"""
DEPRECATED 2026-04-29 — superseded by Genesis PN14 (vllm/_genesis/wiring/
patch_N14_tq_decode_oob_clamp.py, commit 0d92e5b). Kept for compatibility
with the 35B FP8 PROD launch path (`scripts/launch/start_35b_fp8_PROD*.sh`)
which has not yet been migrated to rely solely on Genesis apply_all.

When migrating a launch script: drop the `python3 /external_probe/
patch_40074_iooo.py` line and set `GENESIS_ENABLE_PN14_TQ_DECODE_OOB_CLAMP=1`
in the env. PN14's `safe_page_idx` drift marker makes it self-skip if
this probe was already applied — running both is harmless, just redundant.

See `external_probe/README.md` for full migration status.

────────────────────────────────────────────────────────────────────────

Apply PR #40074 IOOB fix to triton_turboquant_decode.py.

5-line fix: clamp masked-out SIMD lanes to page_idx=0 before block_table
pointer arithmetic. Triton's bounds checker fires on the address even
when the output is masked.

Source: https://github.com/vllm-project/vllm/pull/40074
Fixes: https://github.com/vllm-project/vllm/issues/39998
        possibly also https://github.com/vllm-project/vllm/issues/40831

This is an upstream-author-credited backport applied for probe purposes
on Genesis pin fe9c3d6c5. Will be retired when #40074 merges upstream.
"""
import logging, sys

log = logging.getLogger("pr40074_backport")
log.setLevel(logging.INFO)
if not log.handlers:
    log.addHandler(logging.StreamHandler())

TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_turboquant_decode.py"

OLD = """        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        block_nums = tl.load(
            Block_table_ptr + bt_base + page_idx,
            mask=kv_mask,
            other=0,
        ).to(tl.int64)"""

NEW = """        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        # [PR #40074 backport] Clamp OOB lanes to index 0 before pointer
        # arithmetic so Triton's bounds checker does not fire on masked-out
        # lanes (mask only guards the output value, not the address computation).
        safe_page_idx = tl.where(kv_mask, page_idx, 0)
        block_nums = tl.load(
            Block_table_ptr + bt_base + safe_page_idx,
            mask=kv_mask,
            other=0,
        ).to(tl.int64)"""

try:
    with open(TARGET) as f:
        content = f.read()
    if "safe_page_idx" in content:
        log.info("[PR40074] already applied — skipping")
        sys.exit(0)
    if OLD not in content:
        log.error("[PR40074] anchor not found — vLLM SHA may differ")
        sys.exit(1)
    with open(TARGET, "w") as f:
        f.write(content.replace(OLD, NEW))
    log.info("[PR40074] applied: clamp masked SIMD lanes to safe_page_idx=0")
except Exception as e:
    log.error(f"[PR40074] failed: {e}")
    sys.exit(2)
