# SPDX-License-Identifier: Apache-2.0
"""Canonical promotions of high-value legacy commands.

The v12 split-brain seam: the canonical ``sndr`` CLI (``sndr.cli.main``)
shipped ~7 commands, while the ~40 commands with the real operator
functionality lived only on the legacy ``genesis`` entry point
(``sndr.cli.legacy``) and the ``sndr.compat.cli`` bridge.

This module promotes the high-value subset to the canonical surface
**by wiring, not rewriting**: each promoted command is a thin
:class:`~sndr.cli.commands.Command` that re-dispatches to the existing
legacy implementation with the original argv tail. The legacy code path
stays the single source of truth, so behavior cannot drift between the
two entry points.

Promoted (priority order): report, doctor, preset, bench, tune, config.

Delegation targets:
  - report / preset / bench / tune / config → ``sndr.cli.legacy:cli_main``
    (these are *native* legacy subcommands with ``add_argparser`` factories).
  - doctor → ``sndr.compat.cli:main`` (doctor is a *bridged* subcommand on
    the legacy tree, so we delegate to the same bridge target directly).

Each promoted command:
  - declares ``add_help = False`` so ``sndr <cmd> --help`` falls through to
    the delegate's own ``--help`` (verbatim, no stub duplication);
  - captures the argv tail via ``argparse.REMAINDER``;
  - is import-light — the delegate modules are imported lazily inside
    ``execute()`` so registering these commands never pulls heavy deps.
"""
from __future__ import annotations

import argparse


class _PassthroughCommand:
    """Base for a thin canonical command that delegates to legacy.

    Subclasses set ``name``, ``help``, and ``_target`` (one of
    ``"legacy"`` or ``"compat"``). ``--help`` and every flag/arg after
    ``sndr <name>`` are forwarded verbatim to the legacy implementation.
    """

    name: str = ""
    help: str = ""
    # Tell the registrar to build the subparser with ``add_help=False`` so
    # ``-h``/``--help`` is forwarded to the delegate instead of being
    # intercepted by the stub subparser.
    add_help: bool = False
    # Which legacy entry point to dispatch through.
    _target: str = "legacy"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "_extra_argv",
            nargs=argparse.REMAINDER,
            help=argparse.SUPPRESS,
        )

    def _argv_tail(self, args: argparse.Namespace) -> list[str]:
        return list(getattr(args, "_extra_argv", []) or [])

    def execute(self, args: argparse.Namespace) -> int:
        tail = self._argv_tail(args)
        if self._target == "compat":
            from sndr.compat import cli as _compat_cli
            return _compat_cli.main([self.name, *tail])
        # default: native legacy subcommand tree
        from sndr.cli import legacy as _legacy
        return _legacy.cli_main([self.name, *tail])


class ReportCommand(_PassthroughCommand):
    name = "report"
    help = "Generate diagnostic report bundles for support / issues."


class DoctorCommand(_PassthroughCommand):
    name = "doctor"
    help = "System diagnostic — hardware + software + model + patches."
    _target = "compat"


class PresetCommand(_PassthroughCommand):
    name = "preset"
    help = "V2 preset surface — list / show / explain / recommend."


class BenchCommand(_PassthroughCommand):
    name = "bench"
    help = "Benchmark suite — bench-validate / bench-methodology helpers."


class TuneCommand(_PassthroughCommand):
    name = "tune"
    help = "GPU power/clock tuning from a preset's Y8 gpu_tuning block."


class ConfigCommand(_PassthroughCommand):
    name = "config"
    help = "Preset config browser — diff / explain / new (scaffold)."


# ── UX R2 (v12) — beginner-verb cohesion ────────────────────────────────────
#
# These five verbs were the canonical-vs-legacy split-brain: a novice reading
# the docs typed ``sndr verify`` / ``sndr pull`` / ``sndr list-models`` /
# ``sndr model-config`` and hit ``invalid choice`` — the verbs lived ONLY on the
# legacy ``genesis`` tree and the ``sndr.compat.cli`` bridge. We promote them to
# the canonical surface as thin pass-throughs through ``_target = "compat"``,
# i.e. the SAME bridge target the legacy tree uses, so the two entry points
# cannot drift. (``doctor`` was already promoted in the first v12 batch above.)


class VerifyCommand(_PassthroughCommand):
    name = "verify"
    help = "Post-install smoke test (--quick / --boot / --full)."
    _target = "compat"


class PullCommand(_PassthroughCommand):
    name = "pull"
    help = "HF download + tailored launch script for a curated model."
    _target = "compat"


class ListModelsCommand(_PassthroughCommand):
    name = "list-models"
    help = "Browse the curated model registry."
    _target = "compat"


class ModelConfigCommand(_PassthroughCommand):
    name = "model-config"
    help = "Vetted model launch configs — list / show / render / launch / verify."
    _target = "compat"


PROMOTED_COMMANDS = (
    ReportCommand(),
    DoctorCommand(),
    PresetCommand(),
    BenchCommand(),
    TuneCommand(),
    ConfigCommand(),
    # UX R2 beginner verbs:
    VerifyCommand(),
    PullCommand(),
    ListModelsCommand(),
    ModelConfigCommand(),
)


__all__ = [
    "ReportCommand",
    "DoctorCommand",
    "PresetCommand",
    "BenchCommand",
    "TuneCommand",
    "ConfigCommand",
    "VerifyCommand",
    "PullCommand",
    "ListModelsCommand",
    "ModelConfigCommand",
    "PROMOTED_COMMANDS",
]
