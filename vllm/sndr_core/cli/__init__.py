# SPDX-License-Identifier: Apache-2.0
"""SNDR Core CLI — `sndr` command-line entry points.

Subcommands surface (DA-006 audit closure 2026-05-08):

  Native (in this package):
    sndr install [--dry-run] [-y] [--channel ...]   — setup wizard
    sndr launch [config_key] [--dry-run] [--port]   — render + apply + exec

  Bridged from `vllm.sndr_core.compat.cli` (lazy-imported on first use):
    sndr doctor               — diagnostic / health-check
    sndr verify               — post-apply rebind verification
    sndr self-test            — structural sanity (matches CI gate)
    sndr model-config <args>  — model_config registry browser
    sndr lifecycle-audit      — registry lifecycle drift detection
    sndr validate-schema      — PATCH_REGISTRY schema validator
    sndr explain <patch_id>   — full patch metadata + rationale
    sndr list-models          — Genesis-recognized models
    sndr categories           — patch category browser
    sndr plugins              — plugin discovery + status
    sndr telemetry            — opt-in usage signals
    sndr update-channel       — release channel selector
    sndr preflight            — pre-launch sanity checks
    sndr bench                — run benchmark suite
    sndr migrate              — migration helpers
    sndr recipe               — recipe browser
    sndr preset               — preset matcher + launch script writer
    sndr init                 — interactive init wizard

The bridge keeps a single canonical CLI (`sndr`) while the legacy
`genesis` console script (mapping to `compat.cli`) continues to work
verbatim for back-compat with v7.x operators.

Lazy import contract: the bridge populates a stub argparser; only when
the user actually picks a bridged subcommand does the corresponding
compat module get imported. This keeps `sndr --help` cold-import-fast
even though `compat.cli` pulls in heavier modules.

Entry points:
  python -m vllm.sndr_core.cli ...   # always works
  sndr ...                            # when installed via pip with console_scripts
"""
from __future__ import annotations

import argparse
import sys
from typing import Any

from .community import add_argparser as _community_argparser  # Phase 5 community SDK
from .install import add_argparser as _install_argparser
from .launch import add_argparser as _launch_argparser
from .memory import add_argparser as _memory_argparser
from .patches import add_argparser as _patches_argparser
from .report import add_argparser as _report_argparser
from .deps import add_argparser as _deps_argparser  # C2 (UNIFIED_CONFIG plan)
from .model import add_argparser as _model_argparser  # C4 (UNIFIED_CONFIG plan)
from .upstream import add_argparser as _upstream_argparser  # C17 (UNIFIED_CONFIG plan)
from .caveats import add_argparser as _caveats_argparser  # C22 (UNIFIED_CONFIG plan)
from .doctor_system import add_argparser as _doctor_system_argparser  # C1 (UNIFIED_CONFIG plan)
from .config import add_argparser as _config_argparser  # C8 (UNIFIED_CONFIG plan)
from .service import add_argparser as _service_argparser  # C13 (UNIFIED_CONFIG plan)
from .tune import add_argparser as _tune_argparser  # C14 (UNIFIED_CONFIG plan)
from .migrate import add_argparser as _migrate_argparser  # C9 (UNIFIED_CONFIG plan)
from .image import add_argparser as _image_argparser  # C3 (UNIFIED_CONFIG plan)
from .k8s import add_argparser as _k8s_argparser  # C10 (UNIFIED_CONFIG plan)
from .proxmox import add_argparser as _proxmox_argparser  # C11 (UNIFIED_CONFIG plan)
from .bootstrap import add_argparser as _bootstrap_argparser  # C12 (UNIFIED_CONFIG plan)
from .host import add_argparser as _host_argparser  # P3-B audit 2026-05-12
from .compose import add_argparser as _compose_argparser  # S3.1 audit P3-1 2026-05-12
from .quadlet import add_argparser as _quadlet_argparser  # S3.2 audit P3-2 2026-05-12
from . import bench_compare as _bench_compare  # S2.5

__all__ = ["cli_main"]


# Map of bridged subcommand → compat.cli subcommand name.
# Entries here register a stub argparser that, on dispatch, delegates
# to `vllm.sndr_core.compat.cli` with the original argv.
_BRIDGED: dict[str, str] = {
    "doctor": "doctor",
    "verify": "verify",
    "self-test": "self-test",
    "model-config": "model-config",
    "lifecycle-audit": "lifecycle-audit",
    "validate-schema": "validate-schema",
    "explain": "explain",
    "list-models": "list-models",
    "categories": "categories",
    "plugins": "plugins",
    "telemetry": "telemetry",
    "update-channel": "update-channel",
    "preflight": "preflight",
    "bench": "bench",
    # "migrate": "migrate",  # C9 (UNIFIED_CONFIG plan 2026-05-09): native sndr/cli/migrate.py wins
    "recipe": "recipe",
    "preset": "preset",
    "init": "init",
    "pull": "pull",
}


def _make_bridge_handler(compat_cmd: str):
    """Build a `func(args)` that re-dispatches to compat.cli with the
    original argv tail. Lazy-imports `compat.cli` only on first call."""
    def _bridge(args: argparse.Namespace) -> int:
        from vllm.sndr_core.compat import cli as _compat_cli
        # Reconstruct argv for the compat parser:
        #   sndr <compat_cmd> [extra...]
        # The args namespace carries `_extra_argv` populated by the
        # `argparse.REMAINDER` trick in the stub argparser.
        extra = getattr(args, "_extra_argv", []) or []
        return _compat_cli.main([compat_cmd, *extra])
    return _bridge


def _add_bridged_argparser(subparsers, name: str, compat_cmd: str) -> None:
    """Register a stub argparser for a bridged subcommand."""
    p = subparsers.add_parser(
        name,
        help=f"(bridged) → vllm.sndr_core.compat.cli {compat_cmd}",
        description=(
            f"Bridged subcommand. Delegates to "
            f"`vllm.sndr_core.compat.cli {compat_cmd}` — pass any "
            f"flags/args after `sndr {name}` and they are forwarded "
            "verbatim. For full help, run "
            f"`python -m vllm.sndr_core.compat.cli {compat_cmd} --help`."
        ),
        # Disable add_help so `--help` falls through to compat.cli.
        add_help=False,
    )
    p.add_argument(
        "_extra_argv", nargs=argparse.REMAINDER,
        help=argparse.SUPPRESS,
    )
    p.set_defaults(func=_make_bridge_handler(compat_cmd))


def cli_main(argv: list[str] | None = None) -> int:
    """Top-level dispatch. Returns process exit code."""
    if argv is None:
        argv = sys.argv[1:]

    # DA-006 fast-path: if the FIRST positional arg is a bridged
    # subcommand, delegate the entire argv tail to compat.cli without
    # involving argparse here. This guarantees `--help` and any
    # subcommand-specific flags pass through verbatim.
    if argv and argv[0] in _BRIDGED:
        from vllm.sndr_core.compat import cli as _compat_cli
        return _compat_cli.main([_BRIDGED[argv[0]], *argv[1:]])

    # S2.5 (audit closure 2026-05-08): `sndr bench-compare A.json B.json`
    # delegates to bench_compare.main without involving argparse here.
    if argv and argv[0] == "bench-compare":
        return _bench_compare.main(argv[1:])

    # C4 (UNIFIED_CONFIG plan 2026-05-09): `sndr model pull <args>` and
    # `sndr model list <args>` bypass argparse so the inner module's
    # `--help` and flags pass through verbatim.
    if (len(argv) >= 2 and argv[0] == "model"
            and argv[1] in ("pull", "list")):
        from vllm.sndr_core.cli import model as _model_mod
        if argv[1] == "pull":
            from vllm.sndr_core.compat.models import pull as _pull
            return _pull.main(argv[2:])
        else:  # list
            from vllm.sndr_core.compat.models import list_cli as _list
            if hasattr(_list, "main"):
                return _list.main(argv[2:])
            old_argv = sys.argv
            try:
                sys.argv = ["sndr-model-list"] + list(argv[2:])
                rc = _list.cli() if hasattr(_list, "cli") else 0
                return rc if isinstance(rc, int) else 0
            finally:
                sys.argv = old_argv

    parser = argparse.ArgumentParser(
        prog="sndr",
        description="SNDR Core / Genesis CLI — vllm patcher + preset launcher.",
    )
    parser.add_argument(
        "--version", action="store_true",
        help="Print SNDR Core version and exit.",
    )
    subparsers = parser.add_subparsers(
        dest="cmd", title="Subcommands",
        metavar="{install,launch,doctor,verify,model-config,patches,...}",
    )
    # Native subcommands (live in this package).
    _community_argparser(subparsers)  # Phase 5 community SDK validator + scaffold
    _install_argparser(subparsers)
    _launch_argparser(subparsers)
    _memory_argparser(subparsers)  # T1.3 (audit closure 2026-05-09)
    _patches_argparser(subparsers)  # T1.2 (audit closure 2026-05-09)
    _report_argparser(subparsers)  # T1.1 (audit closure 2026-05-09)
    _deps_argparser(subparsers)  # C2 (UNIFIED_CONFIG plan 2026-05-09)
    _model_argparser(subparsers)  # C4 (UNIFIED_CONFIG plan 2026-05-09)
    _upstream_argparser(subparsers)  # C17 (UNIFIED_CONFIG plan 2026-05-09)
    _caveats_argparser(subparsers)  # C22 (UNIFIED_CONFIG plan 2026-05-09)
    _doctor_system_argparser(subparsers)  # C1 (UNIFIED_CONFIG plan 2026-05-09)
    _config_argparser(subparsers)  # C8 (UNIFIED_CONFIG plan 2026-05-09)
    _service_argparser(subparsers)  # C13 (UNIFIED_CONFIG plan 2026-05-09)
    _tune_argparser(subparsers)  # C14 (UNIFIED_CONFIG plan 2026-05-09)
    _migrate_argparser(subparsers)  # C9 (UNIFIED_CONFIG plan 2026-05-09)
    _image_argparser(subparsers)  # C3 (UNIFIED_CONFIG plan 2026-05-09)
    _k8s_argparser(subparsers)  # C10 (UNIFIED_CONFIG plan 2026-05-09)
    _proxmox_argparser(subparsers)  # C11 (UNIFIED_CONFIG plan 2026-05-09)
    _bootstrap_argparser(subparsers)  # C12 (UNIFIED_CONFIG plan 2026-05-09)
    _host_argparser(subparsers)  # P3-B audit 2026-05-12 — host profile manager
    _compose_argparser(subparsers)  # S3.1 audit P3-1 2026-05-12 — docker-compose renderer
    _quadlet_argparser(subparsers)  # S3.2 audit P3-2 2026-05-12 — podman quadlet renderer

    # S2.5 (audit closure 2026-05-08): bench-compare A.json B.json
    p_bcmp = subparsers.add_parser(
        "bench-compare",
        help="A/B compare two genesis_bench_suite JSON outputs",
        add_help=False,  # forwarded to bench_compare.main(argv)
    )
    p_bcmp.add_argument(
        "args", nargs=argparse.REMAINDER,
        help="bench-compare arguments — see `sndr bench-compare --help`",
    )
    p_bcmp.set_defaults(func=lambda a: _bench_compare.main(a.args))

    # DA-006 (audit 2026-05-08): register bridged subcommands so they
    # appear in `sndr --help`. Actual dispatch went through the
    # fast-path above before reaching this argparse call.
    for name, compat_cmd in _BRIDGED.items():
        _add_bridged_argparser(subparsers, name, compat_cmd)

    args = parser.parse_args(argv)

    if args.version:
        from vllm.sndr_core import SNDR_CORE_VERSION
        print(f"SNDR Core {SNDR_CORE_VERSION}")
        return 0

    if not getattr(args, "cmd", None):
        parser.print_help()
        return 0

    func = getattr(args, "func", None)
    if func is None:
        parser.error(f"command {args.cmd!r} has no handler")
        return 2
    return func(args)
