# SPDX-License-Identifier: Apache-2.0
"""Phase 4.6 — `sndr license` CLI surface.

Subcommands:

  sndr license status [--json]
      Print public-core license boundary status. Always reports
      `core: public (unlicensed)`. Detects optional private engine.

  sndr license verify --file <path> [--offline]
      Verify a license file's signature. Public-core deferred — needs
      vllm.sndr_engine for real verification.

INVARIANTS (tested in test_license_boundary.py):

- This module imports cleanly without network access.
- `status` never returns a non-zero exit code on a clean public install.
- `verify` does NOT invent verification success — it reports deferred
  when the engine is missing.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Any


__all__ = ["add_argparser", "run_status", "run_verify"]


def add_argparser(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "license",
        help="License + public/private boundary status (Phase 4.6).",
        description=(
            "Inspect the public/private license boundary. Public core "
            "NEVER requires a license — these commands surface whether "
            "the optional `vllm.sndr_engine` module is installed and "
            "whether a license file is loaded."
        ),
    )
    sub = p.add_subparsers(dest="license_cmd", required=True)

    p_status = sub.add_parser(
        "status",
        help="Print public-core license boundary status.",
    )
    p_status.add_argument("--json", action="store_true",
                          help="Emit machine-readable JSON.")
    p_status.set_defaults(func=run_status)

    p_verify = sub.add_parser(
        "verify",
        help="Verify a license file's signature (offline).",
    )
    p_verify.add_argument("--file", required=True,
                          help="Path to license file to verify.")
    p_verify.add_argument("--offline", action="store_true",
                          help="Enforce offline-only verification (default).")
    p_verify.add_argument("--json", action="store_true",
                          help="Emit machine-readable JSON.")
    p_verify.set_defaults(func=run_verify)


def run_status(args: argparse.Namespace) -> int:
    from vllm.sndr_core.license import core_license_status
    status = core_license_status()

    if args.json:
        # asdict handles nested EngineDetection cleanly.
        payload = asdict(status)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print("sndr license status")
    print("─" * 60)
    print(f"  Core:                  {status.core}")
    if status.engine is None:
        print("  Engine (private):      not detected")
        print("  Premium patches:       none")
        print()
        print("  Capabilities available: all public-core features.")
    else:
        eng = status.engine
        print(f"  Engine (private):      {eng.module_name}"
              + (f" v{eng.version}" if eng.version else ""))
        if status.license_path:
            print(f"  License:               {status.license_path}")
            if status.license_subject:
                print(f"    Subject:             {status.license_subject}")
            if status.license_tier:
                print(f"    Tier:                {status.license_tier}")
            if status.license_expires:
                print(f"    Expires:             {status.license_expires}")
            if status.license_signature_valid is not None:
                ok = "valid" if status.license_signature_valid else "INVALID"
                print(f"    Signature:           {ok}")
        else:
            print("  License:               not loaded")
        print(f"  Premium patches:       {status.premium_patches_enabled} enabled")
    return 0


def run_verify(args: argparse.Namespace) -> int:
    from vllm.sndr_core.license import verify_license_file
    result = verify_license_file(args.file)

    if args.json:
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
        # Deferred verification is exit 0 (not a failure — public-core
        # correctly declines to verify). Invalid signature is exit 1.
        return 0 if (result.valid or "deferred" in result.reason) else 1

    print(f"sndr license verify '{args.file}'")
    print("─" * 60)
    if result.valid:
        print("  ✓ Signature: valid")
        if result.subject:
            print(f"  Subject:     {result.subject}")
        if result.expires:
            print(f"  Expires:     {result.expires}")
        return 0
    print(f"  ⊘ Result: {result.reason}")
    return 0 if "deferred" in result.reason else 1
