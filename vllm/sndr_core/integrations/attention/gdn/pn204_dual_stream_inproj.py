# SPDX-License-Identifier: Apache-2.0
"""PN204 v2 — GDN dual-stream input projection overlap (port of vllm PR #42301).

================================================================
v2 redesign 2026-05-15: PN204 v1 crashed during `profile_run` on
dev371 with `CUDA driver error: invalid argument` raised inside
`torch/_inductor/runtime/static_triton_launcher.py:291`. Root
cause: when Inductor's torch.compile path executes a Triton kernel,
it expects the kernel to launch on the CURRENT default stream that
the compile context registered. Launching the same kernel on a
freshly-created auxiliary stream invalidates Inductor's stream
binding for that kernel cache entry → CUDA driver rejects the
launch with "invalid argument".

v2 guards added (in priority order):
  1. `torch.compiler.is_compiling()` check — when Inductor is
     actively tracing/compiling the model, take the SERIAL path
     unconditionally. Avoids registering a kernel on aux stream
     during compile.
  2. `_g_pn204_armed` thread-local flag — only enable dual-stream
     AFTER profile_run completes. Set by a worker post-warmup hook;
     stays False during model load, warmup, cudagraph capture, and
     profile_run.
  3. Explicit `device=` on `torch.cuda.Stream(device=...)` — use the
     device of `hidden_states` so the stream lands on the same GPU
     the layer is on (was implicit current-device before, can mismatch
     under TP).
  4. Single-strike disable: any runtime exception in the parallel
     path permanently disables PN204 for THAT layer instance (sets
     `_g_pn204_aux_stream = None`) — siblings continue on serial.

Composes with PN50, PN54, PN59. Replaces retired P7. Auto-SKIPs when
upstream #42301 lands (drift marker `_in_proj_aux_stream`).

Env gate: `GENESIS_ENABLE_PN204_DUAL_STREAM_INPROJ=1` (default OFF).
Arming hook env: `GENESIS_PN204_ARM_AFTER_WARMUP=1` (default ON when
PN204 is enabled — set =0 to keep dual-stream OFF even after warmup
for A/B testing).

Background
----------
`GatedDeltaNetAttention.forward_cuda` issues two independent GEMMs on the
same `hidden_states` tensor back-to-back:

    mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
    ba,         _ = self.in_proj_ba  (hidden_states)

They share an input but produce disjoint outputs; their kernels can run
concurrently on separate CUDA streams. vllm PR #42301 measures -2.9%
TPOT at qps=0.5 on Qwen3.5-35B-A3B with this single change.

Credit
------
- Upstream change being ported: vllm-project/vllm#42301.
- Genesis v1 backport (lazy stream init): Sander 2026-05-04.
- Genesis v2 redesign (is_compiling guard + arming flag + explicit
  device): Sander 2026-05-15 after CUDA driver crash on dev371.
"""
from __future__ import annotations

import logging
import os

from vllm.sndr_core.detection.guards import resolve_vllm_file, vllm_install_root
from vllm.sndr_core.core import TextPatch, TextPatcher, result_to_wiring_status

log = logging.getLogger("genesis.wiring.pn204_dual_stream_inproj")

GENESIS_PN204_MARKER = "Genesis PN204 dual-stream input projection (port of vllm#42301)"


def _enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN204_DUAL_STREAM_INPROJ", "0",
    ).strip().lower() in ("1", "true", "yes", "on")


# Single text-patch on the `forward_cuda` Part 1. We do not modify
# `__init__` — instead the stream/events are created lazily inside the
# forward path on the first call (and cached on `self`). This is more
# robust against early-init CUDA context issues we observed when
# creating stream/events directly in `__init__`.
PN204_FWD_OLD = (
    "        # ============================================================\n"
    "        # Part 1: Input Projection\n"
    "        # ============================================================\n"
    "        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)\n"
    "        ba, _ = self.in_proj_ba(hidden_states)\n"
)
PN204_FWD_NEW = (
    "        # ============================================================\n"
    "        # Part 1: Input Projection\n"
    "        # ============================================================\n"
    "        # [Genesis PN204 v2 port of vllm#42301] Overlap in_proj_qkvz\n"
    "        # and in_proj_ba on independent CUDA streams. v2 guards:\n"
    "        # (a) skip while torch.compiler.is_compiling() — avoids the\n"
    "        #     CUDA driver invalid-argument crash during Inductor\n"
    "        #     compile we saw on dev371;\n"
    "        # (b) skip until armed by post-warmup hook (env\n"
    "        #     GENESIS_PN204_ARM_AFTER_WARMUP defaults to 1);\n"
    "        # (c) explicit device=hidden_states.device on Stream;\n"
    "        # (d) single-strike disable on layer instance after any\n"
    "        #     runtime fault.\n"
    "        import os as _g_pn204_os\n"
    "        if _g_pn204_os.environ.get(\n"
    "            'GENESIS_ENABLE_PN204_DUAL_STREAM_INPROJ', '0'\n"
    "        ).strip().lower() in ('1', 'true', 'yes', 'on'):\n"
    "            try:\n"
    "                _g_pn204_compiling = torch.compiler.is_compiling()\n"
    "            except Exception:\n"
    "                _g_pn204_compiling = False\n"
    "            from vllm.sndr_core.integrations.attention.gdn import (\n"
    "                pn204_dual_stream_inproj as _g_pn204_mod,\n"
    "            )\n"
    "            _g_pn204_armed = getattr(_g_pn204_mod, '_PN204_ARMED', False)\n"
    "            if (not _g_pn204_compiling) and _g_pn204_armed:\n"
    "                if not hasattr(self, '_g_pn204_aux_stream'):\n"
    "                    try:\n"
    "                        _g_pn204_dev = hidden_states.device\n"
    "                        self._g_pn204_aux_stream = torch.cuda.Stream(\n"
    "                            device=_g_pn204_dev,\n"
    "                        )\n"
    "                        self._g_pn204_events = (\n"
    "                            torch.cuda.Event(), torch.cuda.Event(),\n"
    "                        )\n"
    "                    except Exception:\n"
    "                        self._g_pn204_aux_stream = None\n"
    "                        self._g_pn204_events = (None, None)\n"
    "                if self._g_pn204_aux_stream is not None:\n"
    "                    from vllm.utils.multi_stream_utils import (\n"
    "                        maybe_execute_in_parallel as _g_pn204_parallel,\n"
    "                    )\n"
    "                    try:\n"
    "                        (mixed_qkvz, _), (ba, _) = _g_pn204_parallel(\n"
    "                            lambda: self.in_proj_qkvz(hidden_states),\n"
    "                            lambda: self.in_proj_ba(hidden_states),\n"
    "                            self._g_pn204_events[0],\n"
    "                            self._g_pn204_events[1],\n"
    "                            self._g_pn204_aux_stream,\n"
    "                        )\n"
    "                    except Exception:\n"
    "                        # single-strike disable for THIS layer\n"
    "                        self._g_pn204_aux_stream = None\n"
    "                        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)\n"
    "                        ba, _ = self.in_proj_ba(hidden_states)\n"
    "                else:\n"
    "                    mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)\n"
    "                    ba, _ = self.in_proj_ba(hidden_states)\n"
    "            else:\n"
    "                # compile in progress, OR not yet armed: serial\n"
    "                mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)\n"
    "                ba, _ = self.in_proj_ba(hidden_states)\n"
    "        else:\n"
    "            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)\n"
    "            ba, _ = self.in_proj_ba(hidden_states)\n"
)


# ─── Module-level arming flag ──────────────────────────────────────────
# Toggled True by a post-warmup hook after profile_run + cudagraph capture
# completes. While False, all layers take the serial path even if env
# flag is set. This is what guards dev371 from the CUDA invalid-argument
# crash during Inductor compile.
_PN204_ARMED = False


def arm_after_warmup() -> None:
    """Called by a worker post-warmup hook (or manually).

    Once armed, the patched forward path uses dual-stream — but only
    when also outside `torch.compiler.is_compiling()` (belt-and-suspenders).
    Operators wanting to keep dual-stream OFF after warmup for A/B testing
    can set `GENESIS_PN204_ARM_AFTER_WARMUP=0`.
    """
    import os as _os
    if _os.environ.get(
        "GENESIS_PN204_ARM_AFTER_WARMUP", "1",
    ).strip().lower() not in ("1", "true", "yes", "on"):
        return
    global _PN204_ARMED
    _PN204_ARMED = True
    log.info(
        "[PN204] v2 armed — dual-stream activates on next forward "
        "after profile_run + cudagraph capture completed"
    )


def _make_patcher() -> TextPatcher | None:
    if not _enabled():
        return None
    target = resolve_vllm_file(
        "model_executor/layers/mamba/gdn_linear_attn.py"
    )
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN204 GDN dual-stream input projection (port of vllm#42301)"
        ),
        target_file=str(target),
        marker=GENESIS_PN204_MARKER,
        sub_patches=[
            TextPatch(
                name="pn204_forward_parallel_proj",
                anchor=PN204_FWD_OLD,
                replacement=PN204_FWD_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "_in_proj_aux_stream",
            "_in_proj_events",
            "maybe_execute_in_parallel(",
            "_g_pn204_aux_stream",
        ],
    )


def _install_arm_hook() -> bool:
    """Wrap gpu_worker.determine_available_memory so PN204 arms after
    the worker finishes profile_run + cudagraph capture. Idempotent.
    """
    try:
        from vllm.v1.worker import gpu_worker as _gw
    except Exception as e:
        log.warning("[PN204] cannot import gpu_worker for arm-hook: %s", e)
        return False
    if getattr(_gw.Worker, "_genesis_pn204_arm_wrapped", False):
        return True
    _orig = _gw.Worker.determine_available_memory

    def _wrapped(self, *args, **kwargs):
        result = _orig(self, *args, **kwargs)
        try:
            arm_after_warmup()
        except Exception as e:
            log.warning("[PN204] arm-after-warmup failed: %s", e)
        return result

    _gw.Worker.determine_available_memory = _wrapped
    _gw.Worker._genesis_pn204_arm_wrapped = True
    return True


def apply() -> tuple[str, str]:
    if not _enabled():
        return (
            "skipped",
            "PN204 disabled (set GENESIS_ENABLE_PN204_DUAL_STREAM_INPROJ=1)",
        )
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "gdn_linear_attn.py not resolvable"
    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    result, failure = patcher.apply()
    hook_ok = _install_arm_hook()
    return result_to_wiring_status(
        result, failure,
        applied_message=(
            "PN204 v2 applied: GDN in_proj_qkvz/in_proj_ba dual-stream "
            "via vllm.utils.multi_stream_utils.maybe_execute_in_parallel. "
            f"Arm-hook on Worker.determine_available_memory={'installed' if hook_ok else 'NOT-installed (manual arm needed)'}. "
            "Dual-stream activates only after profile_run completes AND "
            "outside torch.compiler.is_compiling() — addresses CUDA "
            "driver crash seen in v1. Expected: -2.9% TPOT per vllm "
            "PR #42301."
        ),
        patch_name="PN204 v2 GDN dual-stream input projection",
    )
