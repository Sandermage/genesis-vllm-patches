# SPDX-License-Identifier: Apache-2.0
"""Canonical vllm serve command builder for all deployment adapters.

Etap 2.1 (audit 2026-05-12): compose / quadlet / k8s previously had
independent `_container_command` functions that diverged from the canonical
`ModelConfig._build_vllm_cmd` (used by bare-metal `to_launch_script`).
Example of divergence: compose used `vllm serve <path>` (positional)
instead of `vllm serve --model <path>` and did not add `--language-model-only`.

This module is the single source of truth for argv form. Compose/Quadlet/K8s
emitters call `build_runtime_command(cfg).argv`. The bash launch script
receives the same argv via `argv_to_shell(argv)` → string list for joining.

Security note (Etap 0.4 reference): `--api-key` is NOT added to argv.
vLLM picks up `VLLM_API_KEY` from an env var, which is rendered through
compose interpolation / quadlet Environment= / k8s Secret refs.
This prevents leaking the key into process listings / docker inspect.
"""
from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import ModelConfig


__all__ = [
    "RuntimeCommandSpec",
    "build_runtime_command",
    "argv_to_shell",
]


@dataclass(frozen=True)
class RuntimeCommandSpec:
    """Canonical `vllm serve …` invocation for a preset.

    `argv` — list of arguments starting with `["vllm", "serve", ...]`.
    Used uniformly by all deployment adapters. The
    `test_runtime_command_parity.py` tests guarantee that compose,
    quadlet, k8s and bare-metal emit identical argv.
    """
    argv: list[str] = field(default_factory=list)


def build_runtime_command(cfg: "ModelConfig") -> RuntimeCommandSpec:
    """Build canonical argv from `ModelConfig`.

    Contract:
      - `argv[0:2] == ["vllm", "serve"]`.
      - `--model <path>` is passed as a named flag (NOT positional).
      - `--api-key` is NOT included (env-based — Etap 0.4).
      - All non-None / truthy ModelConfig flags are converted into
        CLI args in a deterministic order.
      - `vllm_extra_args` are appended at the end (operator override).
    """
    argv: list[str] = ["vllm", "serve", "--model", cfg.model_path]

    # Identity / served name
    if cfg.served_model_name:
        argv += ["--served-model-name", cfg.served_model_name]
    if cfg.quantization:
        argv += ["--quantization", cfg.quantization]
    if cfg.kv_cache_dtype:
        argv += ["--kv-cache-dtype", cfg.kv_cache_dtype]

    # Sizing
    argv += ["--max-model-len", str(cfg.max_model_len)]
    argv += ["--gpu-memory-utilization", f"{cfg.gpu_memory_utilization}"]
    argv += ["--max-num-seqs", str(cfg.max_num_seqs)]
    argv += ["--max-num-batched-tokens", str(cfg.max_num_batched_tokens)]
    argv += ["--tensor-parallel-size", str(cfg.hardware.n_gpus)]
    argv += ["--dtype", cfg.dtype]

    # Behavior flags (alphabetic by flag name for deterministic order)
    if cfg.disable_custom_all_reduce:
        argv.append("--disable-custom-all-reduce")
    if cfg.enable_auto_tool_choice:
        argv.append("--enable-auto-tool-choice")
    if cfg.enable_chunked_prefill:
        argv.append("--enable-chunked-prefill")
    if cfg.enforce_eager:
        argv.append("--enforce-eager")
    if cfg.language_model_only:
        argv.append("--language-model-only")
    if cfg.trust_remote_code:
        argv.append("--trust-remote-code")

    # Parsers
    if cfg.tool_call_parser:
        argv += ["--tool-call-parser", cfg.tool_call_parser]
    if cfg.reasoning_parser:
        argv += ["--reasoning-parser", cfg.reasoning_parser]

    # Spec decode
    if cfg.spec_decode is not None:
        argv += ["--speculative-config", cfg.spec_decode.to_vllm_arg()]

    # Endpoint
    argv += ["--host", cfg.host]
    container_port = (
        cfg.docker.effective_container_port() if cfg.docker else 8000
    )
    argv += ["--port", str(container_port)]
    # `--api-key` intentionally omitted — see Etap 0.4 (compose secret leak).

    # Offload (club-3090 #58 Path A) — validate() has already rejected hybrid-GDN
    # combinations, so it is safe here to add the flags if they are set.
    if cfg.offload is not None:
        argv.extend(cfg.offload.to_vllm_args())

    # Operator override hatch — last
    argv.extend(cfg.vllm_extra_args)
    return RuntimeCommandSpec(argv=argv)


def argv_to_shell(argv: list[str]) -> list[str]:
    """Shell-quote each argv entry — for bash join.

    `shlex.quote` correctly escapes spaces, quotes, $, etc. Used by
    `ModelConfig._build_vllm_cmd` for backward-compat (legacy bash launch
    scripts expect the shell-quoted form).
    """
    return [shlex.quote(a) for a in argv]
