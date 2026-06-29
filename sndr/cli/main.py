# SPDX-License-Identifier: Apache-2.0
"""sndr CLI dispatcher.

Examples. Resource commands have a dotted canonical name (``engines.list``)
and a spaced ergonomic alias (``engines list``) — BOTH resolve to the same
command::

    sndr --version
    sndr engines.list            # or: sndr engines list
    sndr engines.info vllm        # or: sndr engines info vllm
    sndr pins.list --engine vllm  # or: sndr pins list --engine vllm
    sndr health
    sndr preflight prod-qwen3.6-35b-balanced
    sndr preflight prod-gemma4-26b-default --rig single-3090-24gbvram

Promoted operator + beginner commands (v12 split-brain closure / UX R2) —
thin pass-throughs to the legacy/compat implementation, so the canonical and
``genesis`` entry points cannot drift::

    sndr report bundle --preset a5000-2x-35b-prod
    sndr doctor --full
    sndr verify --quick
    sndr pull Qwen/Qwen3-32B      # or: sndr model pull Qwen/Qwen3-32B
    sndr list-models
    sndr model-config list
    sndr preset list
    sndr preset recommend --workload agentic-coding
    sndr bench --help
    sndr tune plan a5000-2x-35b-prod
    sndr config explain a5000-2x-35b-prod

The CLI exists for headless automation (CI scripts, cron jobs, scripts);
operators primarily use the GUI.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import sys

from sndr.cli.commands import COMMAND_REGISTRY, build_subparsers
from sndr.cli.commands.promoted import PROMOTED_COMMANDS
from sndr.version import __version__

# Env markers that signal a non-interactive / CI context. When any is set we
# never auto-launch the wizard on a bare ``sndr`` invocation — scripted callers
# (and dashboards that scrape the help text) must keep the old help behaviour
# even if they happen to run under a pseudo-TTY.
_CI_ENV_MARKERS = ("CI", "CONTINUOUS_INTEGRATION", "GITHUB_ACTIONS", "SNDR_NO_WIZARD")


def _interactive_no_args() -> bool:
    """True when a bare ``sndr`` should drop into the interactive wizard.

    Gated on BOTH stdin and stdout being a real TTY (so piped / redirected
    callers keep the help output) AND the absence of any CI marker. This is the
    only thing that turns the Ollama-style "type one thing → chatting" first
    experience on; everything else keeps the legacy help wall.
    """
    if any(os.environ.get(name) for name in _CI_ENV_MARKERS):
        return False
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except Exception:
        return False


def _run_wizard_no_args(argv: list[str]) -> int:
    """Dispatch a bare ``sndr`` to the interactive launch wizard.

    A bare invocation maps to ``sndr launch`` with no preset — the existing
    rig→preset→fit wizard. Kept as a separate seam (overridable in tests) so
    the no-args dispatch can be exercised without spinning up the whole wizard.
    """
    from sndr.cli.commands.launch import LaunchCommand

    parser = build_parser()
    args = parser.parse_args(["launch", *argv])
    return LaunchCommand().execute(args)


# Promoted pass-through commands delegate their entire argv tail to the
# legacy implementation. They must bypass the top-level argparse so that a
# leading ``--help`` (and any delegate-specific flags) forward verbatim —
# ``argparse.REMAINDER`` does not capture a leading optional like ``--help``.
_PASSTHROUGH_COMMANDS: dict[str, object] = {
    cmd.name: cmd for cmd in PROMOTED_COMMANDS
}


# UX R2 (v12) — spaced-verb cohesion. The canonical resource commands use
# dotted names (``engines.list``, ``engines.info``, ``pins.list``) but a
# beginner naturally types the spaced Docker/git-style form (``engines list``).
# We alias the two-token spaced prefix to its canonical single token so BOTH
# resolve to the SAME command — the dotted form stays primary in ``--help``,
# the spaced form is a silent ergonomic alias. ``model pull`` is the spaced
# alias of the promoted ``pull`` verb (kept for parity with the legacy
# ``sndr model pull`` special-case). Nothing is removed; this only adds
# resolutions that previously raised ``invalid choice``.
_SPACED_ALIASES: dict[tuple[str, str], str] = {
    ("engines", "list"): "engines.list",
    ("engines", "info"): "engines.info",
    ("pins", "list"): "pins.list",
    ("model", "pull"): "pull",
}

# Bare resource-group verbs default to their ``list`` ("show me") — a beginner
# who types just ``sndr engines`` / ``sndr pins`` (no subcommand) should get the
# obvious listing, not a raw argparse ``invalid choice`` wall. Mirrors R1's
# bare-verb philosophy (bare ``sndr`` → wizard). Only groups that HAVE a natural
# list default are listed here (``model`` has no list, so it is left alone).
_BARE_GROUP_DEFAULTS: dict[str, str] = {
    "engines": "engines.list",
    "pins": "pins.list",
}


def _leading_verb_index(argv: list[str]) -> int:
    """Index of the leading subcommand token, skipping any global options that
    precede it (``--output VALUE`` / ``--output=VALUE``). Returns ``-1`` when
    there is no verb to interpret — an introspection flag (``--version`` /
    ``-h`` / ``--help``) or only globals — so those fall through to argparse
    untouched. Without this, a global flag before the verb (``sndr --output json
    engines``) defeats both the spaced-verb normalization and the friendly
    unknown-command handler, dropping the user back onto the raw argparse wall.
    """
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--output":          # global opt that consumes the next token
            i += 2
            continue
        if tok.startswith("--output="):  # --output=json form (self-contained)
            i += 1
            continue
        if tok.startswith("-"):          # --version / -h / --help / other flag
            return -1
        return i                          # first positional → the verb
    return -1


def _normalize_spaced_verbs(argv: list[str]) -> list[str]:
    """Rewrite a leading spaced compound verb to its canonical token.

    ``["engines", "list", ...]`` -> ``["engines.list", ...]`` (dotted alias).
    ``["model", "pull", ...]``   -> ``["pull", ...]``          (promoted verb).
    ``["pins"]`` / ``["engines", "--flag"]`` -> the group's ``list`` default
    (bare group verb, or group + only flags, → "show me").
    Anything else is returned unchanged. Only a leading known spaced alias OR a
    bare group verb is ever rewritten — a real positional like
    ``engines.info engines`` is never mangled.
    """
    if len(argv) >= 2:
        canonical = _SPACED_ALIASES.get((argv[0], argv[1]))
        if canonical is not None:
            return [canonical, *argv[2:]]
    # Bare group verb (or group followed only by flags) → its list default.
    if argv:
        default = _BARE_GROUP_DEFAULTS.get(argv[0])
        if default is not None and (len(argv) == 1 or argv[1].startswith("-")):
            return [default, *argv[1:]]
    return argv


# UX R5 (v12) — friendly unknown-command handling. Before R5 a typo'd verb
# (``sndr lauch``) fell through to argparse's raw ``invalid choice`` wall: a
# dump of every command with no pointer to the next action. We intercept the
# unknown command BEFORE argparse and print a short, rustup/uv-style message
# (matching install.sh's tone) that names the nearest valid verb AND the next
# step. A bare ``sndr``, an introspection flag (``-h``/``--version``), and
# every valid/aliased/promoted verb are left for the normal dispatch — only a
# genuinely-unknown leading subcommand token reaches this path.


def _known_verbs() -> list[str]:
    """Every leading token a user could legitimately type as a subcommand.

    The canonical registry verbs (``launch``, ``run``, ``doctor``, the dotted
    resource names …) PLUS the first token of each spaced alias (``engines``,
    ``pins``, ``model``) so ``engines list`` is never mis-flagged as unknown.
    Builds the parser once to guarantee the registry is populated (the same
    registration path ``main`` uses).
    """
    build_parser()  # ensure COMMAND_REGISTRY is populated
    verbs = set(COMMAND_REGISTRY.keys())
    verbs.update(prefix for (prefix, _verb) in _SPACED_ALIASES)
    return sorted(verbs)


# Muscle-memory synonyms (Docker / Ollama / systemd) → the canonical sndr verb.
# A beginner's first reflex (``serve`` / ``ps`` / ``stop`` / ``models`` …) should
# be pointed at the REAL verb — a curated, correct hint that beats difflib's
# fuzzy guess (which mis-matched ``ps`` → ``pins`` and ``models`` → ``model``).
_SYNONYM_HINTS: dict[str, str] = {
    "serve": "run",          # serve/host a model → sndr run
    "start": "run",          # start a model → sndr run  (or `sndr up` for the stack)
    "up-model": "run",
    "stop": "down",          # stop the engine/stack → sndr down
    "kill": "down",
    "status": "engines",     # what's running → sndr engines
    "ps": "engines",         # docker ps → sndr engines
    "ls": "list-models",     # list available models → sndr list-models
    "models": "list-models",
    "model-list": "list-models",
}


def _friendly_unknown_command(token: str) -> str:
    """Build the rustup-style message for an unknown leading subcommand.

    A curated muscle-memory synonym (``serve`` → ``run``, ``ps`` → ``engines`` …)
    wins first with a precise pointer. Otherwise names the nearest valid verb via
    ``difflib`` (only when genuinely close — no fabricated suggestion for a
    far-off token). Always points at the next action: the bare ``sndr`` guided
    menu and ``sndr --help``.
    """
    lines = []
    synonym = _SYNONYM_HINTS.get(token)
    if synonym is not None:
        lines.append(f"sndr: no {token!r} command — use 'sndr {synonym}' instead.")
    else:
        verbs = _known_verbs()
        # Suggest the dotted/spaced canonical for resource verbs, but match
        # against the bare prefix too so ``enginez`` → ``engines`` reads naturally.
        suggest_pool = sorted(set(verbs) | {v.split(".", 1)[0] for v in verbs})
        near = difflib.get_close_matches(token, suggest_pool, n=1, cutoff=0.6)
        if near:
            lines.append(f"sndr: unknown command {token!r} — did you mean {near[0]!r}?")
        else:
            lines.append(f"sndr: unknown command {token!r}.")
    lines.append(
        "      Run 'sndr' for the guided menu or 'sndr --help' for all commands."
    )
    return "\n".join(lines)


def _interrupt() -> int:
    """Uniform Ctrl-C handling: one clean line on stderr, the conventional
    130 exit code (128 + SIGINT), no traceback. Shared by every dispatch path
    so a Ctrl-C anywhere in the CLI exits the same friendly way."""
    sys.stderr.write("\nInterrupted.\n")
    return 130


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="sndr",
        description="sndr-platform — multi-engine inference patch orchestrator.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--output",
        choices=("json", "yaml", "text"),
        default="text",
        help="Output format (default: text)",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        metavar="COMMAND",
    )
    build_subparsers(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns exit code."""
    if argv is None:
        argv = sys.argv[1:]

    # UX R2: rewrite a leading spaced compound verb to its canonical token
    # (``engines list`` -> ``engines.list``, ``model pull`` -> ``pull``) before
    # any dispatch. A bare ``sndr`` (empty argv) is left untouched so the R1
    # no-args wizard gate still fires.
    # Locate the verb (skipping any leading global options) and normalize the
    # spaced/bare-group form there — so a global flag before the verb doesn't
    # defeat the alias/bare-group handling or the friendly unknown-command path.
    _vi = _leading_verb_index(argv)
    if _vi >= 0:
        argv = argv[:_vi] + _normalize_spaced_verbs(argv[_vi:])

    # Fast-path for promoted pass-through commands (report / doctor / preset /
    # bench / tune / config + the R2 beginner verbs verify / pull / list-models
    # / model-config). Delegate the whole tail to the legacy/compat impl before
    # argparse runs, so ``sndr <cmd> --help`` and every flag forward verbatim.
    # Mirrors the legacy ``cli_main`` bridge fast-path.
    if argv and argv[0] in _PASSTHROUGH_COMMANDS:
        cmd = _PASSTHROUGH_COMMANDS[argv[0]]
        ns = argparse.Namespace(_extra_argv=list(argv[1:]))
        try:
            return cmd.execute(ns)  # type: ignore[attr-defined]
        except KeyboardInterrupt:
            return _interrupt()

    # UX R5: a leading positional token that is neither a flag (``-h`` etc.)
    # nor a known verb is a typo / unknown command. Intercept it BEFORE
    # argparse so the user gets the friendly "did you mean …" + next-step hint
    # instead of the raw ``invalid choice`` wall. Exit code stays non-zero (2,
    # matching argparse's usage-error convention). Flags and valid/aliased
    # verbs fall through untouched.
    if _vi >= 0 and not argv[_vi].startswith("-") and argv[_vi] not in _known_verbs():
        sys.stderr.write(_friendly_unknown_command(argv[_vi]) + "\n")
        return 2

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        # Ollama-style first experience: a bare ``sndr`` on an interactive TTY
        # drops straight into the launch wizard. ``-h``/``--help`` never reaches
        # here (argparse exits first), so this only fires on a truly bare call.
        # Non-TTY / piped / CI keeps the legacy help output unchanged.
        if not argv and _interactive_no_args():
            sys.stderr.write(
                "Welcome to sndr — let's get a model running. "
                "(Ctrl-C to exit, `sndr --help` for all commands)\n"
            )
            try:
                return _run_wizard_no_args([])
            except KeyboardInterrupt:
                return _interrupt()
        parser.print_help()
        return 0

    command = COMMAND_REGISTRY.get(args.command)
    if command is None:
        parser.error(f"Unknown command: {args.command}")

    try:
        return command.execute(args)
    except KeyboardInterrupt:
        return _interrupt()


if __name__ == "__main__":
    sys.exit(main())
