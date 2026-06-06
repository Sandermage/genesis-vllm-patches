#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""kv_calc — standalone KV cache + VRAM capacity calculator.

Port of the `kv-calc.py` utility frequently asked for by club-3090
community operators. Wraps Genesis's already-built `memory_estimator`
(vllm/sndr_core/runtime/memory_estimator.py) with a friendly CLI that
answers the canonical question: **"will this model fit on my GPU?"**.

## Usage

  # By registered preset key (fastest):
  python3 tools/kv_calc.py --preset a5000-2x-35b-prod
  python3 tools/kv_calc.py --preset a5000-2x-35b-prod --gpu-vram 24
  # NOTE: V1 alias `a5000-2x-35b-prod` is Phase-9-frozen; V2 successor
  # is `prod-qwen3.6-35b-balanced` (passable to --preset transparently
  # via load_alias resolution).

  # By local model directory (reads config.json + sums safetensors):
  python3 tools/kv_calc.py --model /models/Qwen3.6-27B-int4-AutoRound \\
      --ctx 131072 --kv-dtype fp8_e5m2 --tp 1 --gpu-vram 24

  # JSON for scripting:
  python3 tools/kv_calc.py --preset a5000-2x-35b-prod --json

## Verdict semantics

  GREEN  — utilization < 80% of declared GPU
  YELLOW — 80% ≤ util < 95% (tight but bootable)
  RED    — util ≥ 95% (likely OOM, suggests CPU offload or smaller ctx)

Verdict is computed against `--gpu-vram` (defaults to 24 GiB — the
community 3090 baseline). Configs without a `--gpu-vram` flag still
print the breakdown but skip the verdict.

The estimator deliberately does NOT include any safety belt vLLM
adds at boot (cudagraph reservation grows with batch, Marlin scratch
varies with intermediate size). It uses Genesis's bench-validated
heuristic floors. Treat the verdict as "first sanity check"; if it
says GREEN at 75%, you'll boot. If it says RED, look at --ctx and
--kv-dtype before launch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Make `vllm.sndr_core.*` imports work when run from a repo checkout.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from sndr.runtime.memory_estimator import (  # noqa: E402
    MemoryEstimate, MemoryComponent, read_model_shape,
    estimate_kv_cache, estimate_weights, estimate_activations,
    estimate_cuda_graph_reserve, estimate_marlin_scratch,
    estimate_for_config, _humanize,
)


# ─── Verdict ───────────────────────────────────────────────────────────


def _verdict(used_bytes: int, gpu_vram_gib: float) -> tuple[str, str]:
    """Return (color_code, label) for the GPU utilization."""
    gpu_bytes = int(gpu_vram_gib * (1 << 30))
    if gpu_bytes <= 0:
        return ("", "n/a")
    util = used_bytes / gpu_bytes
    if util < 0.80:
        return ("\033[32m", "GREEN")
    if util < 0.95:
        return ("\033[33m", "YELLOW")
    return ("\033[31m", "RED")


def _kv_bytes_of(estimate: MemoryEstimate) -> int:
    for c in estimate.components:
        if c.name == "KV cache":
            return c.bytes_
    return 0


def _suggestions(estimate: MemoryEstimate, kv_dtype: str) -> list[str]:
    """Return human-readable suggestions when the verdict is YELLOW or RED."""
    out: list[str] = []
    kv = _kv_bytes_of(estimate)
    total = estimate.total_bytes
    kv_share = kv / max(1, total)
    if kv_share > 0.25:
        out.append(
            f"KV cache is {kv_share:.0%} of the total — halve --max-model-len "
            f"or switch to fp8 KV (--kv-cache-dtype fp8_e5m2) to free "
            f"~{_humanize(kv // 2)}."
        )
    if kv_dtype.lower() in ("float16", "fp16", "bf16", "bfloat16"):
        out.append(
            "KV dtype is full-precision; consider TurboQuant k8v4 (saves "
            "~75% KV memory) or fp8_e5m2 (saves 50%). Beware: TQ requires "
            "Genesis P98 if hybrid GDN model."
        )
    out.append(
        "Or use CPU offload: add `offload.cpu_offload_gib: 8` to the "
        "config (Path A — DENSE models only, NOT hybrid GDN). See "
        "V2 preset `example-3090-dense-cpu-offload` (in "
        "builtin/presets/) for a template."
    )
    return out


# ─── CLI ───────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kv_calc",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--preset", default=None,
        help="Registered model_config key (e.g. 'a5000-2x-35b-prod').",
    )
    src.add_argument(
        "--model", default=None,
        help="Path to local model directory (must contain config.json).",
    )

    # Overrides for the --model path (ignored when --preset wins)
    p.add_argument("--ctx", default=None, type=str,
                   help="Context length override (e.g. '131072', '128k').")
    p.add_argument("--seqs", type=int, default=2,
                   help="max_num_seqs for cudagraph reserve (default 2).")
    p.add_argument("--kv-dtype", default="fp16",
                   help="KV element dtype: fp16/bf16/fp8_e5m2/turboquant_k8v4. "
                        "Default fp16.")
    p.add_argument("--tp", type=int, default=1,
                   help="Tensor parallel size (default 1).")
    p.add_argument("--quant-method", default=None,
                   help="Quant method override for Marlin scratch estimate.")

    # GPU verdict
    p.add_argument("--gpu-vram", type=float, default=24.0,
                   help="GPU VRAM in GiB for verdict (default 24 — 3090 baseline).")

    # Output
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON; disable colors.")
    return p


def _parse_ctx(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    s = s.strip().lower()
    if s.endswith("k"):
        return int(float(s[:-1]) * 1024)
    if s.endswith("m"):
        return int(float(s[:-1]) * 1024 * 1024)
    return int(s)


def _resolve_preset(key: str):
    """Returns (cfg, ctx, kv_dtype, tp_size, max_num_seqs) from registry."""
    try:
        from sndr.model_configs.registry import get
    except ImportError as e:
        print(f"ERROR: registry not importable: {e}", file=sys.stderr)
        return None
    cfg = get(key)
    if cfg is None:
        from sndr.model_configs.registry import list_keys
        print(f"ERROR: unknown preset key {key!r}", file=sys.stderr)
        print(f"Available: {', '.join(sorted(list_keys()))}", file=sys.stderr)
        return None
    return cfg


def _estimate_from_model_path(args: argparse.Namespace,
                               gpu_vram_gib: float) -> MemoryEstimate:
    """Build a MemoryEstimate by reading config.json from --model path."""
    shape = read_model_shape(args.model)
    if not shape.n_layers:
        print(f"ERROR: couldn't read config.json from {args.model}",
              file=sys.stderr)
        sys.exit(2)

    ctx = _parse_ctx(args.ctx) or 4096
    seqs = args.seqs
    tp = max(1, args.tp)
    kv_dtype = args.kv_dtype

    weights = estimate_weights(shape, tp_size=tp)
    kv = estimate_kv_cache(shape,
                            max_model_len=ctx,
                            kv_dtype=kv_dtype,
                            tp_size=tp)
    activations = estimate_activations(shape, max_num_batched_tokens=4096)
    cuda_graph = estimate_cuda_graph_reserve(max_num_seqs=seqs)
    marlin = estimate_marlin_scratch(shape)

    components = (
        MemoryComponent(name="Model weights (post-TP)", bytes_=weights,
                        confidence="high" if shape.weights_size_bytes else "medium"),
        MemoryComponent(name="KV cache", bytes_=kv,
                        notes=f"ctx={ctx}, dtype={kv_dtype}, tp={tp}"),
        MemoryComponent(name="Activations / scratch", bytes_=activations,
                        confidence="low"),
        MemoryComponent(name="CUDA graph reserve", bytes_=cuda_graph,
                        confidence="low"),
        MemoryComponent(name="Marlin repack scratch (peak)", bytes_=marlin,
                        notes="transient — only during weight load",
                        confidence="medium"),
    )
    return MemoryEstimate(
        preset_key=f"<model:{Path(args.model).name}>",
        model_path=args.model,
        gpu_count=tp,
        gpu_vram_bytes=int(gpu_vram_gib * (1 << 30)),
        components=components,
        warnings=(),
        recommendations=(),
        shape=shape,
    )


# ─── Render ────────────────────────────────────────────────────────────


def _render_human(estimate: MemoryEstimate, gpu_vram_gib: float,
                  use_color: bool, ctx: int, kv_dtype: str) -> int:
    color, label = _verdict(estimate.total_bytes, gpu_vram_gib)
    reset = "\033[0m" if use_color else ""
    color = color if use_color else ""

    print("kv_calc — VRAM breakdown")
    print("─" * 60)
    print(f"  Preset/model:  {estimate.preset_key}")
    print(f"  Model path:    {estimate.model_path}")
    print(f"  ctx:           {ctx} tokens")
    print(f"  kv_dtype:      {kv_dtype}")
    print(f"  tp_size:       {estimate.gpu_count}")
    print()
    for c in estimate.components:
        marker = " (transient)" if "transient" in c.notes else ""
        print(f"  {c.name:<32s} {c.human:>12s}  [{c.confidence}]{marker}")
    print("  ─────────────")
    print(f"  TOTAL (committed):              {_humanize(estimate.total_bytes):>12s}")
    print()

    util = (estimate.total_bytes / max(1, gpu_vram_gib * (1 << 30))) * 100
    print(f"  GPU declared:   {gpu_vram_gib:.1f} GiB")
    print(f"  Utilization:    {color}{util:.1f}%  →  {label}{reset}")

    if estimate.warnings:
        print()
        print("  Warnings:")
        for w in estimate.warnings:
            print(f"    - {w}")

    if label in ("YELLOW", "RED"):
        print()
        print("  Suggestions:")
        for s in _suggestions(estimate, kv_dtype):
            print(f"    · {s}")

    return 0 if label in ("GREEN", "n/a") else (1 if label == "YELLOW" else 2)


def _render_json(estimate: MemoryEstimate, gpu_vram_gib: float,
                 ctx: int, kv_dtype: str) -> int:
    color, label = _verdict(estimate.total_bytes, gpu_vram_gib)
    util = estimate.total_bytes / max(1, gpu_vram_gib * (1 << 30))
    out = {
        "preset_key": estimate.preset_key,
        "model_path": estimate.model_path,
        "ctx": ctx,
        "kv_dtype": kv_dtype,
        "tp_size": estimate.gpu_count,
        "components": [
            {"name": c.name, "bytes": c.bytes_, "human": c.human,
             "confidence": c.confidence, "notes": c.notes}
            for c in estimate.components
        ],
        "total_bytes": estimate.total_bytes,
        "total_human": _humanize(estimate.total_bytes),
        "gpu_vram_gib": gpu_vram_gib,
        "utilization": util,
        "verdict": label,
        "warnings": list(estimate.warnings),
        "recommendations": list(estimate.recommendations),
    }
    if label in ("YELLOW", "RED"):
        out["suggestions"] = _suggestions(estimate, kv_dtype)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if label in ("GREEN", "n/a") else (1 if label == "YELLOW" else 2)


# ─── main ──────────────────────────────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    use_color = sys.stdout.isatty() and not args.json

    if args.preset:
        cfg = _resolve_preset(args.preset)
        if cfg is None:
            return 2
        ctx = _parse_ctx(args.ctx) or cfg.max_model_len
        kv_dtype = args.kv_dtype if args.kv_dtype != "fp16" else (
            cfg.kv_cache_dtype or "fp16"
        )
        estimate = estimate_for_config(cfg)
    else:
        ctx = _parse_ctx(args.ctx) or 4096
        kv_dtype = args.kv_dtype
        estimate = _estimate_from_model_path(args, args.gpu_vram)

    if args.json:
        return _render_json(estimate, args.gpu_vram, ctx, kv_dtype)
    return _render_human(estimate, args.gpu_vram, use_color, ctx, kv_dtype)


if __name__ == "__main__":
    sys.exit(main())
