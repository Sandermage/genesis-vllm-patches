# SPDX-License-Identifier: Apache-2.0
"""Background command runner with live job progress.

Runs a (gated) command as a detached subprocess, streaming its output into a
persisted job's log and updating its progress/status as it goes — the engine
behind real model downloads and other long-running apply actions. Output is
flushed to the job store on a throttle so a chatty download doesn't write the
JSON state on every line.
"""
from __future__ import annotations

import re
import subprocess
import threading
import time
from typing import Callable, Optional

from .jobs import Job, create_running_job, update_job
from .runtime_exec import exec_safe_command, wrap_command

_PERCENT = re.compile(r"(\d{1,3})\s*%")


def _default_progress(line: str) -> Optional[float]:
    match = _PERCENT.search(line)
    if not match:
        return None
    value = float(match.group(1))
    return value if 0 <= value <= 100 else None


def run_background_command(
    *,
    kind: str,
    title: str,
    summary: dict,
    command: str,
    transport: str = "local",
    ssh_target: str = "",
    progress_fn: Optional[Callable[[str], Optional[float]]] = None,
    flush_interval: float = 0.5,
    _spawn: bool = True,
) -> Job:
    """Start ``command`` in the background and return the ``running`` job at once."""
    wrapped = wrap_command(exec_safe_command(command), transport=transport, ssh_target=ssh_target)
    job = create_running_job(kind=kind, title=title, summary=summary, command=wrapped)
    progress = progress_fn or _default_progress

    def worker() -> None:
        try:
            proc = subprocess.Popen(  # noqa: S602 - operator-controlled command, gated by apply
                wrapped,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:  # pragma: no cover - spawn failure
            update_job(job.job_id, status="failed", append_log=f"[error] failed to start: {exc}")
            return
        buffer: list[str] = []
        last_flush = time.monotonic()
        last_progress: Optional[float] = None
        latest_progress: Optional[float] = None
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            if not line:
                continue
            buffer.append(line)
            pct = progress(line)
            if pct is not None:
                latest_progress = pct
            now = time.monotonic()
            if (latest_progress is not None and latest_progress != last_progress) or (now - last_flush >= flush_interval):
                update_job(job.job_id, append_log="\n".join(buffer), progress=latest_progress)
                buffer = []
                last_flush = now
                last_progress = latest_progress
        if buffer:
            update_job(job.job_id, append_log="\n".join(buffer))
        code = proc.wait()
        update_job(
            job.job_id,
            status="succeeded" if code == 0 else "failed",
            append_log=f"[exit {code}]",
            progress=100.0 if code == 0 else None,
            note="Completed." if code == 0 else f"Failed (exit {code}).",
        )

    if _spawn:
        threading.Thread(target=worker, daemon=True, name=f"sndr-job-{job.job_id}").start()
    else:  # synchronous path for tests
        worker()
        refreshed = update_job(job.job_id)  # no-op refresh returns current state
        return refreshed or job
    return job


__all__ = ["run_background_command"]
