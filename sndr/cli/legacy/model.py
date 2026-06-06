# SPDX-License-Identifier: Apache-2.0
"""C4 (UNIFIED_CONFIG plan 2026-05-09) — `sndr model` subcommand tree.

Promotes `python3 -m vllm.sndr_core.compat.models.pull` to a first-class
CLI surface. Today this is a thin delegation; B4 (rewrite of pull using
ModelConfig + artifacts.models block) is a separate larger refactor.

Subcommands:

  sndr model pull <key> [...flags...]
      Download a registered model from HuggingFace + generate a launch
      script. Delegates to `compat.models.pull.main()`. All flags from
      that module are accepted verbatim.

  sndr model list
      List registered models (delegates to `compat.models.list_cli`).
"""
from __future__ import annotations

import argparse
import sys
from typing import Any


__all__ = ["add_argparser", "run_pull", "run_list"]


def add_argparser(subparsers: Any) -> None:
    """Register the `sndr model` subcommand tree."""
    p = subparsers.add_parser(
        "model",
        help="Model registry / pull / verify (UNIFIED_CONFIG C4).",
        description=(
            "Manage registered models: list available keys, pull "
            "weights from HuggingFace, generate launch scripts. "
            "Replaces the legacy `scripts/fetch_models.sh` shell "
            "wrapper."
        ),
    )
    sub = p.add_subparsers(dest="model_cmd", required=True)

    # ── pull
    p_pull = sub.add_parser(
        "pull",
        help="Download a registered model from HuggingFace.",
        # Delegate flag handling to the underlying compat module so we
        # don't fork the schema. `--help` works because we forward `--help`.
        add_help=False,
    )
    p_pull.add_argument(
        "args", nargs=argparse.REMAINDER,
        help="Pull arguments — see `sndr model pull --help` for the full list.",
    )
    p_pull.set_defaults(func=run_pull)

    # ── list (V1 registry)
    p_list = sub.add_parser(
        "list",
        help="List registered models (V1 compat registry).",
        add_help=False,
    )
    p_list.add_argument(
        "args", nargs=argparse.REMAINDER,
        help="Forwarded verbatim to the underlying compat list module.",
    )
    p_list.set_defaults(func=run_list)

    # ── list-v2 (V2 layered registry — Wave 10 contract drift fix)
    # The V2 layered model registry lives in
    # `vllm/sndr_core/model_configs/builtin/model/*.yaml`. Each file is a
    # ModelDef (schema_v2.py:ModelDef). Use this command to list them
    # separately from the V1 monolithic registry.
    p_list_v2 = sub.add_parser(
        "list-v2",
        help="List V2 layered models (builtin/model/*.yaml).",
    )
    p_list_v2.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON.",
    )
    p_list_v2.set_defaults(func=run_list_v2)


def run_pull(args: argparse.Namespace) -> int:
    """Forward to `compat.models.pull.main(argv)`."""
    from vllm.sndr_core.compat.models import pull as _pull
    return _pull.main(args.args or [])


def run_list(args: argparse.Namespace) -> int:
    """Forward to `compat.models.list_cli` if it has a main entry point."""
    try:
        from vllm.sndr_core.compat.models import list_cli as _list
    except ImportError:
        from vllm.sndr_core.cli import _io
        _io.error("compat.models.list_cli not importable")
        return 2

    if hasattr(_list, "main"):
        return _list.main(args.args or [])
    # Fallback: some compat modules expose `cli` instead of `main`
    if hasattr(_list, "cli"):
        # cli() typically reads sys.argv directly; reconstruct it.
        old_argv = sys.argv
        try:
            sys.argv = ["sndr-model-list"] + (args.args or [])
            rc = _list.cli()
            return rc if isinstance(rc, int) else 0
        finally:
            sys.argv = old_argv

    from vllm.sndr_core.cli import _io
    _io.error("compat.models.list_cli has no callable entry point")
    return 2


def run_list_v2(args: argparse.Namespace) -> int:
    """List V2 layered model files (builtin/model/*.yaml)."""
    import json
    from pathlib import Path
    from vllm.sndr_core.model_configs.registry_v2 import load_model

    model_dir = (
        Path(__file__).resolve().parents[1]
        / "model_configs" / "builtin" / "model"
    )
    if not model_dir.is_dir():
        from vllm.sndr_core.cli import _io
        _io.error(f"V2 model registry not found: {model_dir}")
        return 2

    models = []
    for f in sorted(model_dir.glob("*.yaml")):
        try:
            m = load_model(f.stem)
            models.append({
                "id": m.id,
                "title": m.title,
                "maintainer": m.maintainer,
                "last_validated": m.last_validated or "n/a",
                "model_path": m.model_path,
                "kv_cache_dtype": m.capabilities.kv_cache_dtype,
                "spec_method": m.capabilities.spec_decode.method if m.capabilities.spec_decode else "none",
            })
        except Exception as e:
            models.append({"id": f.stem, "error": str(e)})

    if args.json:
        print(json.dumps(models, indent=2, sort_keys=True))
    else:
        print(f"V2 layered models ({len(models)} entries in {model_dir.relative_to(model_dir.parents[3])}):")
        for m in models:
            if "error" in m:
                print(f"  {m['id']:42s} ERROR: {m['error']}")
                continue
            spec = m.get("spec_method", "—")
            kv = m.get("kv_cache_dtype") or "default"
            print(f"  {m['id']:42s}  KV={kv:20s}  spec={spec}")
    return 0
