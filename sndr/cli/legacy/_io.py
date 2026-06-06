# SPDX-License-Identifier: Apache-2.0
"""SNDR Core CLI — terminal I/O helpers (colored output, prompts).

Rustup/uv-style output — colored when stdout is a TTY, plain otherwise
(safe for CI logs / piped output). Stage 11 (2026-05-07).

Usage:

    from vllm.sndr_core.cli import _io
    _io.step(1, 8, "Detecting hardware")
    _io.success("Found 2× RTX A5000 (Ampere SM 8.6)")
    _io.info("Free disk: 287 GiB")
    _io.error("vllm not installable on Python <3.10")
    answer = _io.prompt("Install vllm nightly?", default="Y", choices="Yn")
"""
from __future__ import annotations

import os
import sys
from typing import NoReturn, Optional


# ─── Color codes (rustup-style) ───────────────────────────────────────
_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR", "") == ""

_RESET = "\033[0m"   if _USE_COLOR else ""
_BOLD  = "\033[1m"   if _USE_COLOR else ""
_RED   = "\033[31m"  if _USE_COLOR else ""
_GREEN = "\033[32m"  if _USE_COLOR else ""
_YELLOW = "\033[33m" if _USE_COLOR else ""
_BLUE  = "\033[34m"  if _USE_COLOR else ""
_DIM   = "\033[2m"   if _USE_COLOR else ""


def step(n: int, total: int, label: str) -> None:
    """Numbered step header: `[3/8] Detecting hardware...`"""
    print(f"\n{_BOLD}[{n}/{total}]{_RESET} {label}...")


def success(msg: str) -> None:
    print(f"  {_GREEN}✓{_RESET} {msg}")


def info(msg: str) -> None:
    print(f"  {_DIM}{msg}{_RESET}")


def warn(msg: str) -> None:
    print(f"  {_YELLOW}⚠{_RESET}  {msg}")


def error(msg: str) -> None:
    print(f"  {_RED}✗{_RESET} {msg}", file=sys.stderr)


def fatal(msg: str, exit_code: int = 1) -> "NoReturn":
    """Print error and exit. Returns never (typed as NoReturn for mypy)."""
    error(msg)
    sys.exit(exit_code)


def question(msg: str) -> None:
    print(f"  {_BLUE}?{_RESET} {_BOLD}{msg}{_RESET}")


def prompt(
    msg: str,
    *,
    default: Optional[str] = None,
    choices: Optional[str] = None,
    non_interactive: bool = False,
) -> str:
    """Interactive Y/N or free-text prompt. Returns user's answer (stripped).

    Args:
      msg: prompt text shown to user.
      default: default value if user just presses ENTER (or non_interactive=True).
      choices: e.g. "Yn" — case sensitivity preserved; first letter is default.
      non_interactive: if True, return `default` immediately without reading
        stdin. Used by `-y`/`--non-interactive` CLI flag and by tests.

    Aborts with SystemExit(1) if user provides invalid input 3× or stdin EOF.
    """
    if non_interactive:
        if default is None:
            fatal(f"non-interactive mode but no default for: {msg}", 1)
        return default

    suffix = ""
    if choices:
        suffix = f" [{choices}]"
    elif default is not None:
        suffix = f" [{default}]"

    for _attempt in range(3):
        try:
            line = input(f"  {_BLUE}?{_RESET} {_BOLD}{msg}{_RESET}{suffix}: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            fatal("aborted by user", 130)

        if not line and default is not None:
            return default
        if not line:
            warn("input required (no default set)")
            continue
        if choices:
            # Validate against choice letters case-insensitively
            if line.lower() in [c.lower() for c in choices]:
                return line
            warn(f"please answer one of: {', '.join(choices)}")
            continue
        return line

    fatal("too many invalid attempts", 1)


def banner(title: str, subtitle: str = "") -> None:
    """Box-drawn header at start of CLI command."""
    width = max(len(title), len(subtitle)) + 4
    print()
    print(f"  {_BOLD}┌{'─' * width}┐{_RESET}")
    print(f"  {_BOLD}│{_RESET}  {_BOLD}{title.ljust(width - 2)}{_RESET}{_BOLD}│{_RESET}")
    if subtitle:
        print(f"  {_BOLD}│{_RESET}  {_DIM}{subtitle.ljust(width - 2)}{_RESET}{_BOLD}│{_RESET}")
    print(f"  {_BOLD}└{'─' * width}┘{_RESET}")
    print()


__all__ = [
    "banner", "error", "fatal", "info", "prompt", "question",
    "step", "success", "warn",
]
