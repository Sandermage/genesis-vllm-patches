# SPDX-License-Identifier: Apache-2.0
"""P1.6 — `sndr gateway` thin CLI wrapper around the FastAPI dispatcher.

Equivalent to:

    python -m sndr.engines.vllm.patches.spec_decode.gateway

with operator-friendly argparse flags that map 1:1 onto the SNDR_GATEWAY_*
env surface the gateway already reads.

What this wrapper does (intentionally narrow):

  * Parse CLI args → set SNDR_GATEWAY_* env vars
  * Lazy-import the gateway module (no FastAPI / uvicorn / httpx import
    at CLI registration time, so `sndr --help` and `sndr gateway --help`
    stay torch/vllm/uvicorn-free)
  * Call ``sndr.engines.vllm.patches.spec_decode.gateway.app.main()``

What it does NOT do:

  * No new routing logic — uses the existing FastAPI router as-is
  * No prompt-text inspection
  * No safety guard / planner / contract changes
  * No new metric emission
  * No production launcher edits
  * No Qwen touch

If an operator explicitly sets the corresponding env var BEFORE running
``sndr gateway``, the CLI flag overrides it (CLI is authoritative for
this invocation). Unset CLI flags leave the existing env untouched —
the gateway falls back to its built-in defaults (see app.py docstring).
"""
from __future__ import annotations

import argparse
import os
from typing import Any


__all__ = ["add_argparser", "run_gateway"]


# CLI flag → SNDR_GATEWAY_* env var name. Maps directly onto the env
# surface documented in sndr/engines/vllm/patches/spec_decode/gateway/app.py
# (and the gateway README). Both the SNDR_* canonical and the legacy
# GENESIS_GATEWAY_* aliases are read by the gateway via `get_sndr_env`;
# this wrapper sets the SNDR_* canonical form.
_FLAG_TO_ENV: dict[str, str] = {
    "default_url":       "SNDR_GATEWAY_DEFAULT_URL",
    "structured_url":    "SNDR_GATEWAY_STRUCTURED_URL",
    "host":              "SNDR_GATEWAY_BIND_HOST",
    "port":              "SNDR_GATEWAY_BIND_PORT",
    "profile":           "SNDR_GATEWAY_PROFILE",
    "health_interval":   "SNDR_GATEWAY_HEALTH_INTERVAL",
    "timeout":           "SNDR_GATEWAY_TIMEOUT",
    "log_level":         "SNDR_GATEWAY_LOG_LEVEL",
}


def add_argparser(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "gateway",
        help="Run the SNDR spec-decode gateway (FastAPI reverse proxy).",
        description=(
            "Thin wrapper around "
            "`python -m sndr.engines.vllm.patches.spec_decode.gateway`. "
            "CLI flags map onto SNDR_GATEWAY_* env vars 1:1. Flags "
            "override pre-existing env for the lifetime of this "
            "invocation."
        ),
    )
    p.add_argument(
        "--host", default=None,
        help="Bind host (default: 0.0.0.0 unless SNDR_GATEWAY_BIND_HOST set).",
    )
    p.add_argument(
        "--port", default=None,
        help="Bind port (default: 8100 unless SNDR_GATEWAY_BIND_PORT set).",
    )
    p.add_argument(
        "--default-url", default=None,
        help="Default upstream URL "
             "(default: http://localhost:8101 unless env set).",
    )
    p.add_argument(
        "--structured-url", default=None,
        help="Structured upstream URL "
             "(default: http://localhost:8102 unless env set).",
    )
    p.add_argument(
        "--profile", default=None,
        help="Profile/artifact id to load "
             "(default: gemma4-31b-tq-mtp-structured-k4 unless env set).",
    )
    p.add_argument(
        "--health-interval", default=None,
        help="Health-probe interval in seconds (default: 5).",
    )
    p.add_argument(
        "--timeout", default=None,
        help="Upstream request timeout in seconds (default: 120).",
    )
    p.add_argument(
        "--admin-allow-remote", action="store_true",
        help="Permit non-localhost admin endpoints "
             "(off by default; sets SNDR_GATEWAY_ADMIN_ALLOW_REMOTE=1).",
    )
    p.add_argument(
        "--log-level", default=None,
        help="Gateway log level (default: INFO).",
    )
    p.set_defaults(func=run_gateway)


def _apply_env(args: argparse.Namespace) -> None:
    """Map CLI args to SNDR_GATEWAY_* env vars.

    Unset args leave the env untouched (gateway falls back to its own
    defaults). Empty string args also pass through unchanged — operators
    who want to "clear" an env var explicitly should unset it via shell
    before invoking the CLI.
    """
    for attr, env_name in _FLAG_TO_ENV.items():
        val = getattr(args, attr, None)
        if val is None:
            continue
        os.environ[env_name] = str(val)
    # Boolean flag: only set when True (operator opted in)
    if getattr(args, "admin_allow_remote", False):
        os.environ["SNDR_GATEWAY_ADMIN_ALLOW_REMOTE"] = "1"


def run_gateway(args: argparse.Namespace) -> int:
    """Handler: map args to env, then exec the gateway main loop.

    Returns the int exit code the underlying ``app.main()`` produces;
    on KeyboardInterrupt (operator ^C), returns 130.
    """
    _apply_env(args)
    # Lazy import — keeps the rest of the `sndr` CLI free of FastAPI
    # / uvicorn / httpx imports until the operator actually runs the
    # gateway subcommand.
    try:
        from sndr.engines.vllm.patches.spec_decode.gateway.app import main
    except ImportError as e:
        # Heavy deps missing (fastapi / uvicorn / httpx). Distinct from
        # exit code 2 (which is for tooling failure) to make the error
        # cause obvious to operators.
        print(
            "sndr gateway: missing runtime dependency for the gateway "
            f"FastAPI app: {e}\n"
            "Install with: pip install fastapi uvicorn httpx",
        )
        return 3
    try:
        main()
    except KeyboardInterrupt:
        return 130
    return 0
