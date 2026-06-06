# SPDX-License-Identifier: Apache-2.0
"""Gated, opt-in executor for service lifecycle actions.

This is the controlled-side-effect layer (GUI.P5). It is OFF by default and
only runs when the operator explicitly enables apply (``SNDR_ENABLE_APPLY=1``
or ``create_app(enable_apply=True)``). Even then:

  * read-only actions (``status`` / ``logs``) run freely;
  * mutating actions (``start`` / ``stop`` / ``restart``) require an explicit
    ``confirm=True`` from the caller;
  * follow flags (``-f``) are stripped so execution cannot block forever;
  * remote hosts are reached over SSH (``ssh <ssh_target> <command>``), so the
    daemon never needs elevated network exposure.

Commands come verbatim from :func:`build_service_plan` — the same plan the
read-only GUI shows. This executor does not invent commands.
"""
from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass

from .service_plan import MUTATING, build_service_plan


class ApplyDisabledError(RuntimeError):
    """Raised when execution is attempted while apply is disabled."""


class ConfirmationRequiredError(RuntimeError):
    """Raised when a mutating action is attempted without ``confirm=True``."""


@dataclass(frozen=True)
class StepResult:
    order: int
    title: str
    command: str
    exit_code: int
    stdout: str
    stderr: str
    status: str  # "ok" | "failed"


def apply_enabled() -> bool:
    """True only when apply is explicitly enabled via env (default OFF)."""
    return os.environ.get("SNDR_ENABLE_APPLY", "").strip().lower() in ("1", "true", "yes", "on")


def exec_safe_command(command: str) -> str:
    """Drop follow flags so a command terminates (logs/journalctl ``-f``)."""
    tokens = [tok for tok in command.split() if tok != "-f"]
    return " ".join(tokens)


def wrap_command(command: str, *, transport: str, ssh_target: str) -> str:
    """Wrap a command for the target transport. Remote → single SSH invocation."""
    if transport == "ssh" and ssh_target:
        return f"ssh {ssh_target} {shlex.quote(command)}"
    return command


def run_steps(
    steps: list[tuple[str, str]],
    *,
    transport: str = "local",
    ssh_target: str = "",
    timeout: int = 120,
) -> list[StepResult]:
    """Run ``(title, command)`` steps, capturing real exit/stdout/stderr."""
    results: list[StepResult] = []
    for index, (title, command) in enumerate(steps):
        safe = exec_safe_command(command)
        wrapped = wrap_command(safe, transport=transport, ssh_target=ssh_target)
        try:
            proc = subprocess.run(
                wrapped,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            code, out, err = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            code, out, err = 124, "", f"timeout after {timeout}s"
        except Exception as exc:  # pragma: no cover - defensive
            code, out, err = 1, "", str(exc)
        results.append(
            StepResult(
                order=index + 1,
                title=title,
                command=wrapped,
                exit_code=code,
                stdout=(out or "").strip()[:8000],
                stderr=(err or "").strip()[:4000],
                status="ok" if code == 0 else "failed",
            )
        )
    return results


def execute_service_action(
    *,
    preset_id: str,
    action: str = "status",
    runtime_target: str = "docker",
    host: str = "127.0.0.1",
    transport: str = "local",
    ssh_target: str = "",
    confirm: bool = False,
    timeout: int = 120,
    enabled: bool | None = None,
):
    """Execute a service action for real, returning a non-dry-run Job.

    ``enabled`` overrides the env gate (the daemon resolves apply state once and
    passes it in). When None, falls back to :func:`apply_enabled`.

    Raises :class:`ApplyDisabledError` unless apply is enabled, and
    :class:`ConfirmationRequiredError` for mutating actions without confirm.
    """
    from .jobs import create_executed_job

    is_enabled = apply_enabled() if enabled is None else bool(enabled)
    if not is_enabled:
        raise ApplyDisabledError(
            "Apply is disabled. Start the daemon with --enable-apply "
            "(or SNDR_ENABLE_APPLY=1) to execute service actions."
        )
    plan = build_service_plan(
        preset_id=preset_id, action=action, runtime_target=runtime_target, host=host
    )
    if plan.action in MUTATING and not confirm:
        raise ConfirmationRequiredError(
            f"Action {plan.action!r} mutates {plan.container_name}; "
            "resend with confirm=true to execute."
        )

    step_results = run_steps(
        [(step.title, step.command) for step in plan.steps],
        transport=transport,
        ssh_target=ssh_target,
        timeout=timeout,
    )
    ok = all(result.status == "ok" for result in step_results)

    return create_executed_job(
        kind=f"service.{plan.action}",
        title=f"{plan.action} {preset_id}",
        summary={
            "preset_id": preset_id,
            "action": plan.action,
            "runtime_target": runtime_target,
            "host": host,
            "transport": transport,
            "container": plan.container_name,
            "mutating": plan.mutating,
            "confirmed": confirm,
            "plan_id": plan.plan_id,
        },
        step_results=step_results,
        cli_mirror=list(plan.cli_mirror),
        ok=ok,
        note=(
            f"Executed on {ssh_target or host} via {transport}. "
            + ("All steps ok." if ok else "One or more steps failed.")
        ),
    )


__all__ = [
    "ApplyDisabledError",
    "ConfirmationRequiredError",
    "StepResult",
    "apply_enabled",
    "exec_safe_command",
    "execute_service_action",
    "run_steps",
    "wrap_command",
]
