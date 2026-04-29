"""
Disk-edit patch for turboquant_attn.py: fix CUDA graph capture crashes
at `.tolist()` calls inside the backend.

There are two sites in turboquant_attn.py that force GPU->CPU syncs via
`.tolist()` inside code paths that can execute under active CUDA stream
capture:

  A) forward() mixed-batch branch:
       prefill_max_seq = max(attn_metadata.seq_lens[num_decodes:].tolist())
     Hit when the engine builds a mixed batch (decodes + prefills) during
     warmup/capture.

  B) _prefill_attention() continuation branch:
       qsl = query_start_loc.tolist()
       seq_lens_list = attn_metadata.seq_lens.tolist()
     Hit when max_query_len != max_seq_len during warmup/capture (which is
     what happens with spec-decode + chunked-prefill: warmup's dummy batch
     simulates continuation chunks).

Fix:
  A) During capture, use `attn_metadata.max_seq_len` (Python int, batch-level
     upper bound — safe overestimate for the prefill portion; flash_attn
     just uses it as a grid sizing bound).
  B) During capture, early-return the graph-safe fast path
     (flash_attn_varlen_func with cu_seqlens == query_start_loc). Attention
     is a splitting_op in V1 PIECEWISE mode, so capture-time values
     only drive memory profiling, not graph content. At inference (non-
     capture), the original correct continuation path runs.

Runs AFTER Genesis patches (text anchors verified to survive Genesis's
disk edits — Genesis does not touch either tolist site).
"""

import logging
import os

log = logging.getLogger("tolist_cudagraph_fix")
log.setLevel(logging.INFO)
if not log.handlers:
    log.addHandler(logging.StreamHandler())

TARGET = (
    "/usr/local/lib/python3.12/dist-packages/"
    "vllm/v1/attention/backends/turboquant_attn.py"
)

# Site B: _prefill_attention continuation branch (crash we actually hit).
# Anchor spans from the comment block above Hk through both tolist calls.
SITE_B_OLD = """        # Continuation or no flash_attn: per-request attention.
        # For continuation chunks (seq_len > q_len), we must attend to
        # previously cached K/V from the TQ cache, not just the current
        # chunk's raw K/V.
        Hk = key.shape[1]
        use_gqa = Hk < Hq
        query_start_loc = attn_metadata.query_start_loc
        num_reqs = query_start_loc.shape[0] - 1

        output = torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype)

        # Convert to Python lists once (single CPU-GPU sync) instead of
        # per-request .item() calls that each force a sync.
        qsl = query_start_loc.tolist()
        seq_lens_list = attn_metadata.seq_lens.tolist()"""

SITE_B_NEW = """        # [tolist_cudagraph_fix] During CUDA graph capture, the continuation
        # branch below calls .tolist() which forces a GPU->CPU sync -- illegal
        # under torch.cuda.graph(). vLLM V1 PIECEWISE mode lists
        # unified_attention_with_output as a splitting_op, so the captured
        # piece does not include attention outputs; capture-time values only
        # need to drive memory profiling. Fall back to the graph-safe fast
        # path (same shape output (N,Hq,D), similar workspace). At inference
        # (non-capture), is_current_stream_capturing() returns False and the
        # original per-request continuation path runs unchanged.
        if torch.cuda.is_current_stream_capturing():
            if _HAS_FLASH_ATTN:
                return flash_attn_varlen_func(
                    q=query,
                    k=key,
                    v=value,
                    cu_seqlens_q=attn_metadata.query_start_loc,
                    cu_seqlens_k=attn_metadata.query_start_loc,
                    max_seqlen_q=attn_metadata.max_query_len,
                    max_seqlen_k=attn_metadata.max_query_len,
                    softmax_scale=self.scale,
                    causal=True,
                )
            return torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype)

        # Continuation or no flash_attn: per-request attention.
        # For continuation chunks (seq_len > q_len), we must attend to
        # previously cached K/V from the TQ cache, not just the current
        # chunk's raw K/V.
        Hk = key.shape[1]
        use_gqa = Hk < Hq
        query_start_loc = attn_metadata.query_start_loc
        num_reqs = query_start_loc.shape[0] - 1

        output = torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype)

        # Convert to Python lists once (single CPU-GPU sync) instead of
        # per-request .item() calls that each force a sync.
        qsl = query_start_loc.tolist()
        seq_lens_list = attn_metadata.seq_lens.tolist()"""

# Site A: forward() mixed-batch prefill_max_seq tolist.
SITE_A_OLD = """            # Use CPU-side max to avoid GPU→CPU sync from .item()
            prefill_max_seq = max(attn_metadata.seq_lens[num_decodes:].tolist())"""

SITE_A_NEW = """            # Use CPU-side max to avoid GPU→CPU sync from .item()
            # [tolist_cudagraph_fix] During CUDA graph capture, substitute a
            # safe upper bound (batch-level max_seq_len, a Python int) to
            # avoid the tolist() sync. Overestimates prefill_max_seq but
            # flash_attn uses it only as a grid upper bound, and real
            # inference (non-capture) takes the else branch unchanged.
            if torch.cuda.is_current_stream_capturing():
                prefill_max_seq = attn_metadata.max_seq_len
            else:
                prefill_max_seq = max(attn_metadata.seq_lens[num_decodes:].tolist())"""


def _apply(label, src, old, new):
    if old not in src:
        log.error(
            "[tolist_cudagraph_fix] %s anchor NOT FOUND -- patch NOT applied. "
            "Upstream may have changed or Genesis rewrote this area.",
            label,
        )
        return src, False
    return src.replace(old, new), True


def main():
    if not os.path.exists(TARGET):
        log.error(f"[tolist_cudagraph_fix] target missing: {TARGET}")
        return

    with open(TARGET, "r") as f:
        src = f.read()

    if "[tolist_cudagraph_fix]" in src:
        log.info("[tolist_cudagraph_fix] already applied (idempotent)")
        return

    applied_any = False
    src, ok_b = _apply("Site B (_prefill_attention)", src, SITE_B_OLD, SITE_B_NEW)
    applied_any = applied_any or ok_b
    src, ok_a = _apply("Site A (forward mixed-batch)", src, SITE_A_OLD, SITE_A_NEW)
    applied_any = applied_any or ok_a

    if applied_any:
        with open(TARGET, "w") as f:
            f.write(src)
        log.info(
            "[tolist_cudagraph_fix] Patched %s. Site A: %s, Site B: %s",
            TARGET,
            "ok" if ok_a else "skip",
            "ok" if ok_b else "skip",
        )
    else:
        log.error("[tolist_cudagraph_fix] NO sites patched -- file not modified")


if __name__ == "__main__":
    main()
