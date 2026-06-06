# SPDX-License-Identifier: Apache-2.0
"""``build_vllm_cmd(cfg)`` — emit vllm serve command parts.

Pure function; takes a ``ModelConfig`` and returns the ``list[str]``
of CLI parts (no exec/docker prefix). Previously
``ModelConfig._build_vllm_cmd`` in ``model_configs/schema.py``. Body
unchanged — only the call site changed from method to free function.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .shell import shell_quote

if TYPE_CHECKING:
    from ..schema import ModelConfig


def build_vllm_cmd(cfg: "ModelConfig") -> list[str]:
    """vllm serve command parts (without exec/docker prefix)."""
    parts = [
        "vllm serve",
        f"--model {shell_quote(cfg.model_path)}",
        f"--tensor-parallel-size {cfg.hardware.n_gpus}",
        f"--gpu-memory-utilization {cfg.gpu_memory_utilization}",
        f"--max-model-len {cfg.max_model_len}",
        f"--max-num-seqs {cfg.max_num_seqs}",
        f"--max-num-batched-tokens {cfg.max_num_batched_tokens}",
        f"--dtype {shell_quote(cfg.dtype)}",
    ]
    if cfg.kv_cache_dtype:
        parts.append(f"--kv-cache-dtype {shell_quote(cfg.kv_cache_dtype)}")
    if cfg.quantization:
        parts.append(f"--quantization {shell_quote(cfg.quantization)}")
    if cfg.served_model_name:
        parts.append(f"--served-model-name {shell_quote(cfg.served_model_name)}")
    if cfg.tool_call_parser:
        parts.append(f"--tool-call-parser {shell_quote(cfg.tool_call_parser)}")
    if cfg.reasoning_parser:
        parts.append(f"--reasoning-parser {shell_quote(cfg.reasoning_parser)}")
    if cfg.enable_chunked_prefill:
        parts.append("--enable-chunked-prefill")
    if cfg.enforce_eager:
        parts.append("--enforce-eager")
    if cfg.disable_custom_all_reduce:
        parts.append("--disable-custom-all-reduce")
    if cfg.language_model_only:
        parts.append("--language-model-only")
    if cfg.trust_remote_code:
        parts.append("--trust-remote-code")
    if cfg.enable_auto_tool_choice:
        parts.append("--enable-auto-tool-choice")
    parts.append(f"--api-key {shell_quote(cfg.api_key)}")
    parts.append(f"--host {shell_quote(cfg.host)}")
    if cfg.docker:
        # Y4: pass container-side port to vllm serve (the port it
        # listens on inside the container). Falls back to legacy
        # `port` field when host_port/container_port are not split.
        parts.append(f"--port {cfg.docker.effective_container_port()}")
    if cfg.spec_decode:
        parts.append(
            f"--speculative-config '{cfg.spec_decode.to_vllm_arg()}'"
        )
    for extra in cfg.vllm_extra_args:
        parts.append(extra)
    # club-3090 #58 Path A: cpu offload knobs become engine flags.
    # OffloadConfig.validate() already blocked hybrid-GDN combos.
    if cfg.offload is not None:
        parts.extend(cfg.offload.to_vllm_args())
    return parts
