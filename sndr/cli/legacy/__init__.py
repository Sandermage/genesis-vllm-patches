# SPDX-License-Identifier: Apache-2.0
"""SNDR Core CLI — `sndr` command-line entry points.

Subcommands surface (DA-006 audit closure 2026-05-08):

  Native (in this package):
    sndr install [--dry-run] [-y] [--channel ...]   — setup wizard
    sndr launch [config_key] [--dry-run] [--port]   — render + apply + exec

  Bridged from `sndr.compat.cli` (lazy-imported on first use):
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
  python -m sndr.cli.legacy ...      # always works
  sndr ...                            # when installed via pip with console_scripts
"""
from __future__ import annotations

import argparse
import sys

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

# Wave 10 (2026-05-16) — close P1-1 CLI contract drift from production-
# readiness audit. These modules had `add_argparser()` factories but
# were not registered in this `__init__.py`, so the subcommands returned
# argparse `invalid choice` errors despite live code, tests, and docs.
from .bench import add_argparser as _bench_argparser  # Phase 6 bench-validate / bench-methodology
from .config_keys import add_argparser as _config_keys_argparser  # config-keys-list (env var registry)
from .findings import add_argparser as _findings_argparser  # findings (bench evidence audit)
from .hardware import add_argparser as _hardware_argparser  # hardware (V2 hardware registry)
from .license import add_argparser as _license_argparser  # license (key generation + check)
from .profile import add_argparser as _profile_argparser  # profile (V2 profile registry)
from .gateway import add_argparser as _gateway_argparser  # P1.6 thin gateway wrapper
from .routing import add_argparser as _routing_argparser  # Phase 7.G4.WORKLOAD-GATE-POLICY.IMPLEMENT
from .preset import add_argparser as _preset_argparser  # CONFIG-UX.3 native preset surface
from .config_catalog import add_argparser as _config_catalog_argparser  # CONFIG-UX.5.2 derived catalog CLI
from .gui_api import add_argparser as _gui_api_argparser  # GUI Product API daemon
from .trace import add_argparser as _trace_argparser  # §6.H6 trace catalog CLI
from .trace import (
    add_support_bundle_argparser as _support_bundle_argparser,
)  # §6.H9 support-bundle verb (top-level, sibling of `trace`)

__all__ = ["cli_main"]


# Map of bridged subcommand → compat.cli subcommand name.
# Entries here register a stub argparser that, on dispatch, delegates
# to `sndr.compat.cli` with the original argv.
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
    # CONFIG-UX.3 (2026-05-24): `preset` graduated from bridged stub to
    # native CLI module (`cli/preset.py`). The compat.cli `preset`
    # subcommand remains reachable via direct invocation
    # (`python -m sndr.compat.cli preset ...`) for back-compat,
    # but `sndr preset` now resolves to the native CLI.
    "init": "init",
    "pull": "pull",
}


def _make_bridge_handler(compat_cmd: str):
    """Build a `func(args)` that re-dispatches to compat.cli with the
    original argv tail. Lazy-imports `compat.cli` only on first call."""
    def _bridge(args: argparse.Namespace) -> int:
        from sndr.compat import cli as _compat_cli
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
        help=f"(bridged) → sndr.compat.cli {compat_cmd}",
        description=(
            f"Bridged subcommand. Delegates to "
            f"`sndr.compat.cli {compat_cmd}` — pass any "
            f"flags/args after `sndr {name}` and they are forwarded "
            "verbatim. For full help, run "
            f"`python -m sndr.compat.cli {compat_cmd} --help`."
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
        from sndr.compat import cli as _compat_cli
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
        if argv[1] == "pull":
            from sndr.compat.models import pull as _pull
            return _pull.main(argv[2:])
        else:  # list
            from sndr.compat.models import list_cli as _list
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

    # Wave 10 (2026-05-16) — P1-1 CLI contract drift fix. These six
    # native subparsers had factories under cli/* but were never
    # registered here, so `sndr profile list`, `sndr config-keys-list`,
    # etc. failed with `invalid choice` despite live code + tests + docs
    # expecting them.
    _bench_argparser(subparsers)         # bench-validate, bench-methodology (Phase 6)
    _config_keys_argparser(subparsers)   # config-keys-list (env var registry dump)
    _findings_argparser(subparsers)      # findings (bench evidence audit)
    _hardware_argparser(subparsers)      # hardware (V2 hardware list/show)
    _license_argparser(subparsers)       # license (key generation + check)
    _profile_argparser(subparsers)       # profile (V2 profile list/show)
    _gateway_argparser(subparsers)       # gateway (P1.6 thin CLI wrapper)
    _routing_argparser(subparsers)       # routing-table (Phase 7.G4.WORKLOAD-GATE-POLICY.IMPLEMENT)
    _preset_argparser(subparsers)        # CONFIG-UX.3 preset list/show/explain/recommend
    _config_catalog_argparser(subparsers) # CONFIG-UX.5.2 derived catalog build/verify/show/query
    _gui_api_argparser(subparsers)       # read-only Product API daemon for GUI/web clients
    _trace_argparser(subparsers)         # §6.H6 trace catalog (sndr trace list)
    _support_bundle_argparser(subparsers)  # §6.H9 sndr support-bundle

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
        from sndr import SNDR_CORE_VERSION
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
