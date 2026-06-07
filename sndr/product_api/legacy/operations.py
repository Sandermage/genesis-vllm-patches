# SPDX-License-Identifier: Apache-2.0
"""Project Operations console — run sndr_core's canonical maintenance and
diagnostic workflows from the GUI as live-monitored background jobs.

sndr_core exposes a large CLI surface (doctor, self-test, registry audits,
proof/release checks, …) that operators run regularly from a terminal. This
module surfaces a *curated allowlist* of those read-only operations to the
Product API so the GUI becomes the operator's hub for running them — without
ever accepting an arbitrary command from the client.

Safety model (matches the rest of the Product API):
* The client only sends an ``operation`` id. The shell command is looked up
  from the server-side ``_OPERATIONS`` table — never user input → no injection.
* Every operation here is read-only (no host/registry mutation). Execution
  still respects the apply gate: with ``--enable-apply`` it runs for real and
  streams output; otherwise it returns a dry-run job mirroring the command.
* Commands invoke the installed package (``python -m sndr.cli``), so
  they work wherever the daemon runs — no dependence on the dev repo / Makefile.
"""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass

_CLI = f"{sys.executable} -m sndr.cli"


@dataclass(frozen=True)
class Operation:
    id: str
    label: str
    group: str
    description: str
    command: str
    mutating: bool
    estimate: str


_OPERATIONS: tuple[Operation, ...] = (
    # ── Diagnostics ────────────────────────────────────────────────────────
    Operation("doctor", "System doctor", "Diagnostics",
              "Full environment, dependency and readiness health report.",
              f"{_CLI} doctor", False, "~10s"),
    Operation("self-test", "Self-test", "Diagnostics",
              "Structural sanity of the patch apply layer and registry bindings.",
              f"{_CLI} self-test --json", False, "~10s"),
    Operation("preflight", "Pre-launch preflight", "Diagnostics",
              "Pre-launch sanity checks before serving a preset.",
              f"{_CLI} preflight", False, "~5s"),
    # ── Registry audits ────────────────────────────────────────────────────
    Operation("validate-schema", "Validate patch schema", "Registry audits",
              "Validate every PATCH_REGISTRY entry against the schema contract.",
              f"{_CLI} validate-schema", False, "~5s"),
    Operation("patches-doctor", "Patch registry doctor", "Registry audits",
              "Cross-check patch lifecycle, env flags and apply modules.",
              f"{_CLI} patches doctor", False, "~5s"),
    Operation("lifecycle-audit", "Lifecycle drift audit", "Registry audits",
              "Detect registry lifecycle vs docstring/implementation drift.",
              f"{_CLI} lifecycle-audit", False, "~10s"),
    Operation("dead-patch", "Dead-patch detector", "Registry audits",
              "Static proof scan for patches that no longer fire (dead code).",
              f"{_CLI} patches prove --dead-detect", False, "~15s"),
    # ── Config & catalog ───────────────────────────────────────────────────
    Operation("config-catalog", "Verify config catalog", "Config & catalog",
              "Verify the derived preset-card catalog is fresh and redacted.",
              f"{_CLI} config-catalog verify", False, "~10s"),
    Operation("list-models", "List recognized models", "Config & catalog",
              "List every model Genesis recognizes with its config bindings.",
              f"{_CLI} list-models", False, "~5s"),
    # ── Proof & release ────────────────────────────────────────────────────
    Operation("proof-status", "Proof artefact status", "Proof & release",
              "State of proof artefacts attached across the patch registry.",
              f"{_CLI} patches proof-status", False, "~5s"),
    Operation("release-check", "Release readiness check", "Proof & release",
              "Static proof requirement gate for a release cut.",
              f"{_CLI} patches release-check", False, "~5s"),
)

_BY_ID = {op.id: op for op in _OPERATIONS}


def list_operations() -> list[dict]:
    """The curated operation catalogue (display metadata + the exact command)."""
    return [asdict(op) for op in _OPERATIONS]


def get_operation(op_id: str) -> Operation | None:
    return _BY_ID.get(op_id)


def _run_background(**kwargs):
    """Indirection so tests can substitute the executor without spawning."""
    from .background_exec import run_background_command

    return run_background_command(**kwargs)


def run_operation(op_id: str, *, apply_on: bool = False):
    """Run a curated operation. Raises ``KeyError`` for an unknown id.

    With ``apply_on`` the command executes as a live background job; otherwise
    a dry-run job is returned (command mirrored, nothing executed) — identical
    to how the bench/evidence/model-download endpoints behave.
    """
    op = get_operation(op_id)
    if op is None:
        raise KeyError(op_id)

    summary = {"operation": op.id, "group": op.group, "command": op.command}
    if apply_on:
        return _run_background(
            kind=f"op.{op.id}",
            title=op.label,
            summary=summary,
            command=op.command,
        )

    from .jobs import create_dry_run_job

    return create_dry_run_job(
        kind=f"op.{op.id}",
        title=op.label,
        summary=summary,
        steps=[(op.label, op.command)],
        cli_mirror=[op.command],
        note="Read-only operation — start the daemon with --enable-apply to run it here, or copy the command.",
    )
