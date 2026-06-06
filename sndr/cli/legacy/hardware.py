# SPDX-License-Identifier: Apache-2.0
"""V2 layered config — `sndr hardware` subcommand (Phase 4, P1).

Two subcommands surface the V2 HardwareDef layer:

  sndr hardware list
      List every HardwareDef under `builtin/hardware/*.yaml`.

  sndr hardware show <id>
      Print the resolved HardwareDef: identity, sizing knobs, runtime block,
      system env. Read-only.

This is the V2 counterpart to existing `sndr model-config list` / `sndr config
list` (V1 monolithic preset surface). The V2 surface answers operator
questions like "what rigs does this repo know about?" without coupling
to a particular ModelDef.
"""
from __future__ import annotations

import argparse
import json
from typing import Any

from . import _io


__all__ = ["add_argparser", "run_list", "run_show"]


def add_argparser(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "hardware",
        help="V2 hardware layer — list/show HardwareDef definitions.",
        description=(
            "Inspect V2 HardwareDef layer (model_configs/builtin/hardware/*.yaml). "
            "Sister command of `sndr model` (V2 ModelDef) and `sndr profile` (V2 ProfileDef)."
        ),
    )
    sub = p.add_subparsers(dest="hardware_cmd", required=True)

    p_list = sub.add_parser("list", help="List all HardwareDef ids + titles.")
    p_list.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON.")
    p_list.set_defaults(func=run_list)

    p_show = sub.add_parser("show",
                            help="Print resolved HardwareDef (sizing, runtime, system env).")
    p_show.add_argument("hw_id", help="hardware id (e.g. 'a5000-2x-24gbvram-16cpu-128gbram')")
    p_show.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON.")
    p_show.set_defaults(func=run_show)


def _hw_summary(hw_id: str) -> dict:
    """Lightweight summary derived from a HardwareDef without rendering
    the full nested structure. Used by `list` to avoid loading every file
    fully when only the index is needed."""
    from vllm.sndr_core.model_configs.registry_v2 import load_hardware
    hw = load_hardware(hw_id)
    return {
        "id": hw.id,
        "title": hw.title,
        "n_gpus": hw.hardware.n_gpus,
        "min_vram_per_gpu_mib": hw.hardware.min_vram_per_gpu_mib,
        "cuda_capability_min": list(hw.hardware.cuda_capability_min)
            if hw.hardware.cuda_capability_min else None,
        "runtime_default": hw.runtime.default,
        "runtime_supported": list(hw.runtime.supported),
    }


# ─── list

def run_list(args: argparse.Namespace) -> int:
    from vllm.sndr_core.model_configs.registry_v2 import list_hardware
    from vllm.sndr_core.model_configs.schema import SchemaError

    ids = list_hardware()
    summaries: list[dict] = []
    errors: list[tuple[str, str]] = []
    for hw_id in ids:
        try:
            summaries.append(_hw_summary(hw_id))
        except (SchemaError, Exception) as e:
            errors.append((hw_id, f"{type(e).__name__}: {e}"))

    if args.json:
        out = {"hardware": summaries, "errors": errors}
        print(json.dumps(out, indent=2, sort_keys=True))
        return 1 if errors else 0

    print("sndr hardware list — V2 HardwareDef registry")
    print("─" * 60)
    if not summaries and not errors:
        print("  (no V2 hardware files found under model_configs/builtin/hardware/)")
        return 0
    for s in summaries:
        n_gpu = s["n_gpus"]
        vram = s["min_vram_per_gpu_mib"]
        cc = s["cuda_capability_min"] or [0, 0]
        print(f"  {s['id']}")
        print(f"      {s['title']}")
        print(f"      n_gpus={n_gpu}  min_vram_per_gpu={vram} MiB  "
              f"cuda_cap≥{cc[0]}.{cc[1]}  runtime={s['runtime_default']}")
    if errors:
        print()
        print("  Errors loading these IDs:")
        for hw_id, msg in errors:
            print(f"    {hw_id}: {msg}")
    print()
    print(f"  Total: {len(summaries)} hardware definitions"
          + (f" ({len(errors)} errors)" if errors else ""))
    return 1 if errors else 0


# ─── show

def run_show(args: argparse.Namespace) -> int:
    from vllm.sndr_core.model_configs.registry_v2 import load_hardware
    from vllm.sndr_core.model_configs.schema import SchemaError

    try:
        hw = load_hardware(args.hw_id)
    except SchemaError as e:
        _io.warn(f"hardware id {args.hw_id!r}: {e}")
        return 2

    if args.json:
        # Convert to a JSON-safe dict via dataclass walk.
        from dataclasses import asdict
        print(json.dumps(asdict(hw), indent=2, sort_keys=True, default=str))
        return 0

    print(f"sndr hardware show '{hw.id}'")
    print("─" * 60)
    print(f"  title:                 {hw.title}")
    print(f"  maintainer:            {hw.maintainer}")
    print()
    print("  Hardware:")
    print(f"    gpu_match_keys:      {list(hw.hardware.gpu_match_keys)}")
    print(f"    n_gpus:              {hw.hardware.n_gpus}")
    print(f"    min_vram_per_gpu:    {hw.hardware.min_vram_per_gpu_mib} MiB")
    cc = hw.hardware.cuda_capability_min
    if cc:
        print(f"    cuda_capability_min: {cc[0]}.{cc[1]}")
    print()
    s = hw.sizing
    print("  Sizing (defaults; profile may override):")
    print(f"    max_model_len:           {s.max_model_len}")
    print(f"    gpu_memory_utilization:  {s.gpu_memory_utilization}")
    print(f"    max_num_seqs:            {s.max_num_seqs}")
    print(f"    max_num_batched_tokens:  {s.max_num_batched_tokens}")
    print(f"    enable_chunked_prefill:  {s.enable_chunked_prefill}")
    print(f"    enforce_eager:           {s.enforce_eager}")
    print(f"    disable_custom_all_reduce: {s.disable_custom_all_reduce}")
    print()
    rt = hw.runtime
    print("  Runtime:")
    print(f"    default:    {rt.default}")
    print(f"    supported:  {list(rt.supported)}")
    if rt.docker is not None:
        print(f"    docker.image:           {rt.docker.image}")
        if rt.docker.image_digest:
            print(f"    docker.image_digest:    {rt.docker.image_digest}")
        print(f"    docker.host_port:       {rt.docker.host_port}")
        print(f"    docker.container_port:  {rt.docker.container_port}")
        print(f"    docker.shm_size:        {rt.docker.shm_size}")
        print(f"    docker.network:         {rt.docker.network}")
        print(f"    docker.mounts:          {len(list(rt.docker.mounts))} entries")
    print()
    print(f"  System env entries:    {len(hw.system_env)}")
    if hw.system_env:
        for k, v in sorted(hw.system_env.items()):
            print(f"    {k}={v}")
    if hw.notes:
        print()
        print("  Notes:")
        for n in hw.notes:
            print(f"    • {n}")
    return 0
