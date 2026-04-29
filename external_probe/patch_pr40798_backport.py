"""
Backport of vllm-project/vllm PR #40798 onto Genesis pin fe9c3d6c5.

PR #40798 — "[TurboQuant] Share decode scratch workspace across layers"
https://github.com/vllm-project/vllm/pull/40798

Hypothesis (H14, this session):
  PR #40798 may already fix vllm-project/vllm#40831 (TurboQuant ×
  spec-decode degenerate token loops) as a side-effect of stabilizing
  the workspace pointer across spec-decode shape transitions.
  WorkspaceManager.get_simultaneous() returns persistent base buffer
  views — captured cudagraph references stable data_ptr; runtime
  spec-decode call returns view into the SAME base → no pointer drift
  between capture and replay.

This script applies all 4 production-code file changes (skip the
test file). Returns non-zero on any anchor miss so we can detect
incompatibility with our pin / Genesis patches early.

Apply order: BEFORE Genesis apply_all (anchors are upstream-pristine).

Author: Sandermage (Sander) Barzov Aleksandr.
Original PR author: vLLM core team.
"""
import logging
import sys

log = logging.getLogger("pr40798_backport")
log.setLevel(logging.INFO)
if not log.handlers:
    log.addHandler(logging.StreamHandler())

VLLM_ROOT = "/usr/local/lib/python3.12/dist-packages/vllm"

# ─── Patch 1/4: attention.py — remove per-layer register_buffer ────────
ATTENTION_OLD = '''        self._tq_config = tq_config

        # Pre-allocate decode intermediate buffers so model.to(device) moves
        # them to GPU *before* the memory profiler runs.  Without this the
        # profiler gives all free memory to KV cache blocks and the first
        # decode OOMs when these buffers are lazily allocated.
        _vllm_cfg = get_current_vllm_config()
        B = _vllm_cfg.scheduler_config.max_num_seqs
        Hq = self.num_heads
        S = _vllm_cfg.attention_config.tq_max_kv_splits_for_cuda_graph
        D = head_size
        self.register_buffer(
            "_tq_mid_o_buf",
            torch.empty(B, Hq, S, D + 1, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_tq_output_buf",
            torch.empty(B, Hq, D, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_tq_lse_buf",
            torch.empty(B, Hq, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,'''

ATTENTION_NEW = '''        self._tq_config = tq_config

        # [PR #40798 backport] TQ decode scratch space is allocated through
        # the v1 workspace manager at runtime. Shared across layers, no
        # per-layer multiplication of scratch memory.

    def forward(
        self,'''

# ─── Patch 2/4: turboquant_attn.py — remove buf args from _decode_attention ─
TQ_ATTN_OLD_1 = '''        PiT: torch.Tensor | None = None,
        layer: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        # Grab cached decode buffers from the layer (lazily allocated).
        mid_o_buf = output_buf = lse_buf = None
        if layer is not None:
            mid_o_buf = getattr(layer, "_tq_mid_o_buf", None)
            output_buf = getattr(layer, "_tq_output_buf", None)
            lse_buf = getattr(layer, "_tq_lse_buf", None)

        result = triton_turboquant_decode_attention('''

TQ_ATTN_NEW_1 = '''        PiT: torch.Tensor | None = None,
        layer: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        # [PR #40798 backport] buffers fetched via WorkspaceManager inside
        # triton_turboquant_decode_attention; no per-layer attr lookup needed.
        result = triton_turboquant_decode_attention('''

TQ_ATTN_OLD_2 = '''            norm_correction=self.tq_config.norm_correction,
            PiT=PiT,
            mid_o_buf=mid_o_buf,
            output_buf=output_buf,
            lse_buf=lse_buf,
            buf_holder=layer,
            max_num_kv_splits=self.max_num_kv_splits,
        )'''

TQ_ATTN_NEW_2 = '''            norm_correction=self.tq_config.norm_correction,
            PiT=PiT,
            max_num_kv_splits=self.max_num_kv_splits,
        )'''

# ─── Patch 3/4: triton_turboquant_decode.py — WorkspaceManager fallback ─
TRITON_OLD_1 = '''import math
from typing import Any

import torch'''

TRITON_NEW_1 = '''import math

import torch'''

TRITON_OLD_2 = '''    mid_o_buf: torch.Tensor | None = None,
    output_buf: torch.Tensor | None = None,
    lse_buf: torch.Tensor | None = None,
    buf_holder: Any = None,
    max_num_kv_splits: int = 32,  # fixed split count (must be constant for cudagraph)
) -> torch.Tensor:'''

TRITON_NEW_2 = '''    mid_o_buf: torch.Tensor | None = None,
    output_buf: torch.Tensor | None = None,
    lse_buf: torch.Tensor | None = None,
    max_num_kv_splits: int = 32,  # fixed split count (must be constant for cudagraph)
) -> torch.Tensor:'''

TRITON_OLD_3 = '''    NUM_KV_SPLITS = max_num_kv_splits

    if (
        mid_o_buf is not None
        and mid_o_buf.shape[0] >= B'''

TRITON_NEW_3 = '''    NUM_KV_SPLITS = max_num_kv_splits

    if mid_o_buf is None or output_buf is None or lse_buf is None:
        from vllm.v1.worker.workspace import (
            current_workspace_manager,
            is_workspace_manager_initialized,
        )

        if is_workspace_manager_initialized():
            mid_o_buf, output_buf, lse_buf = (
                current_workspace_manager().get_simultaneous(
                    ((B, Hq, NUM_KV_SPLITS, D + 1), torch.float32),
                    ((B, Hq, D), torch.float32),
                    ((B, Hq), torch.float32),
                )
            )

    if (
        mid_o_buf is not None
        and mid_o_buf.shape[0] >= B'''

TRITON_OLD_4 = '''            dtype=torch.float32,
            device=device,
        )
        if buf_holder is not None:
            buf_holder._tq_mid_o_buf = mid_o

    # Stage 1:'''

TRITON_NEW_4 = '''            dtype=torch.float32,
            device=device,
        )

    # Stage 1:'''

TRITON_OLD_5 = '''        output = output_buf[:B, :Hq, :D]
    else:
        output = torch.empty(B, Hq, D, dtype=torch.float32, device=device)
        if buf_holder is not None:
            buf_holder._tq_output_buf = output
    if lse_buf is not None and lse_buf.shape[0] >= B:
        lse = lse_buf[:B, :Hq]
    else:
        lse = torch.empty(B, Hq, dtype=torch.float32, device=device)
        if buf_holder is not None:
            buf_holder._tq_lse_buf = lse'''

TRITON_NEW_5 = '''        output = output_buf[:B, :Hq, :D]
    else:
        output = torch.empty(B, Hq, D, dtype=torch.float32, device=device)
    if lse_buf is not None and lse_buf.shape[0] >= B:
        lse = lse_buf[:B, :Hq]
    else:
        lse = torch.empty(B, Hq, dtype=torch.float32, device=device)'''

# ─── Patch 4/4: gpu_model_runner.py — _reserve_turboquant_decode_workspace ─
RUNNER_OLD_IMPORT = '''from vllm.v1.worker.workspace import lock_workspace'''
RUNNER_NEW_IMPORT = '''from vllm.v1.worker.workspace import current_workspace_manager, lock_workspace'''

RUNNER_OLD_CAPTURE = '''    @instrument(span_name="Capture model")
    def capture_model(self) -> int:
        if self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:'''

RUNNER_NEW_CAPTURE = '''    @instrument(span_name="Capture model")
    def capture_model(self) -> int:
        # [PR #40798 backport] reserve TQ decode workspace BEFORE capture
        self._reserve_turboquant_decode_workspace()

        if self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:'''

# Inject _reserve_turboquant_decode_workspace method after capture_model.
# Anchor: the closing return of capture_model. Insert new method right after.
RUNNER_OLD_INJECT = '''        )
        return cuda_graph_size

    def _warmup_and_capture('''

RUNNER_NEW_INJECT = '''        )
        return cuda_graph_size

    def _reserve_turboquant_decode_workspace(self) -> None:
        # [PR #40798 backport] pre-reserve TQ decode workspace at the
        # max_num_seqs shape so the captured cudagraph sees stable
        # data_ptr that runtime spec-decode views also alias into.
        if not self.cache_config.cache_dtype.startswith("turboquant_"):
            return
        if not self.attn_groups:
            return

        max_num_reqs = self.scheduler_config.max_num_seqs
        num_heads = self.model_config.get_num_attention_heads(self.parallel_config)
        head_size = self.model_config.get_head_size()
        max_num_splits = (
            self.vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph
        )

        for groups in self.attn_groups:
            for group in groups:
                if group.backend.get_name() != "TURBOQUANT":
                    continue

                current_workspace_manager().get_simultaneous(
                    (
                        (max_num_reqs, num_heads, max_num_splits, head_size + 1),
                        torch.float32,
                    ),
                    ((max_num_reqs, num_heads, head_size), torch.float32),
                    ((max_num_reqs, num_heads), torch.float32),
                )
                return

    def _warmup_and_capture('''


# ──────────────────────────────────────────────────────────────────────
# Apply
# ──────────────────────────────────────────────────────────────────────

def apply_patch(target: str, edits: list[tuple[str, str, str]]) -> int:
    """Apply (name, old, new) edit list to file. Returns 0 on success."""
    try:
        with open(target) as f:
            content = f.read()
    except FileNotFoundError:
        log.error(f"target not found: {target}")
        return 1

    if "[PR #40798 backport]" in content:
        log.info(f"  already applied — skipping {target}")
        return 0

    for name, old, new in edits:
        if old not in content:
            log.error(f"  ANCHOR MISS in {target} — {name}")
            log.error(f"    Looking for: {repr(old[:120])}")
            return 2
        content = content.replace(old, new, 1)
        log.info(f"  applied: {name}")

    with open(target, "w") as f:
        f.write(content)
    log.info(f"  wrote {target}")
    return 0


def main() -> int:
    rc = 0
    log.info("=== PR #40798 backport ===")

    log.info("[1/4] attention.py — remove _tq_*_buf register_buffer calls")
    rc |= apply_patch(
        f"{VLLM_ROOT}/model_executor/layers/attention/attention.py",
        [("attention_remove_register_buffer", ATTENTION_OLD, ATTENTION_NEW)],
    )

    log.info("[2/4] turboquant_attn.py — remove buf args from _decode_attention")
    rc |= apply_patch(
        f"{VLLM_ROOT}/v1/attention/backends/turboquant_attn.py",
        [
            ("tq_attn_remove_buf_lookup", TQ_ATTN_OLD_1, TQ_ATTN_NEW_1),
            ("tq_attn_remove_buf_args", TQ_ATTN_OLD_2, TQ_ATTN_NEW_2),
        ],
    )

    log.info("[3/4] triton_turboquant_decode.py — WorkspaceManager fallback")
    rc |= apply_patch(
        f"{VLLM_ROOT}/v1/attention/ops/triton_turboquant_decode.py",
        [
            ("triton_remove_any_import", TRITON_OLD_1, TRITON_NEW_1),
            ("triton_remove_buf_holder_arg", TRITON_OLD_2, TRITON_NEW_2),
            ("triton_workspace_fallback", TRITON_OLD_3, TRITON_NEW_3),
            ("triton_remove_buf_holder_set_mid", TRITON_OLD_4, TRITON_NEW_4),
            ("triton_remove_buf_holder_set_out_lse", TRITON_OLD_5, TRITON_NEW_5),
        ],
    )

    log.info("[4/4] gpu_model_runner.py — _reserve_turboquant_decode_workspace")
    rc |= apply_patch(
        f"{VLLM_ROOT}/v1/worker/gpu_model_runner.py",
        [
            ("runner_import_workspace_manager", RUNNER_OLD_IMPORT, RUNNER_NEW_IMPORT),
            ("runner_capture_call_reserve", RUNNER_OLD_CAPTURE, RUNNER_NEW_CAPTURE),
            ("runner_inject_reserve_method", RUNNER_OLD_INJECT, RUNNER_NEW_INJECT),
        ],
    )

    if rc == 0:
        log.info("✅ PR #40798 backport applied cleanly across 4 files")
    else:
        log.error(f"❌ PR #40798 backport failed (rc={rc})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
