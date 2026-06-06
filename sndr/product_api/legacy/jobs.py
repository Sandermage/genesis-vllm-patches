# SPDX-License-Identifier: Apache-2.0
"""Persistent job + event store for GUI actions.

Jobs and the event feed are persisted under ``$SNDR_HOME/state`` (atomic JSON)
so history survives a daemon/container restart. Jobs can be **dry-run** (records
the exact commands, no execution) or **executed** — including long-running
background jobs (e.g. a model download) whose status and log are updated live as
the work progresses.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

_LOCK = threading.RLock()
_EVENT_CAP = 500
_JOB_CAP = 200


def _state_dir() -> Path:
    home = os.environ.get("SNDR_HOME") or os.environ.get("GENESIS_HOME")
    base = Path(home).expanduser() if home else (Path.home() / ".sndr")
    return base / "state"


def _atomic_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=0), "utf-8")
    os.replace(tmp, path)


class _Store:
    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}
        self.events: list[dict] = []
        self.job_seq = 0
        self.event_seq = 0
        self._loaded = False

    def ensure(self) -> None:
        if self._loaded:
            return
        directory = _state_dir()
        try:
            self.jobs = json.loads((directory / "jobs.json").read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            self.jobs = {}
        try:
            feed = json.loads((directory / "events.json").read_text("utf-8"))
            self.events = feed.get("events", [])
            self.event_seq = int(feed.get("seq", len(self.events)))
        except (OSError, json.JSONDecodeError):
            self.events, self.event_seq = [], 0
        self.job_seq = max([int(j.get("_seq", 0)) for j in self.jobs.values()] + [0])
        self._loaded = True

    def save_jobs(self) -> None:
        if len(self.jobs) > _JOB_CAP:
            keep = sorted(self.jobs.values(), key=lambda j: j.get("created_at", 0), reverse=True)[:_JOB_CAP]
            self.jobs = {j["job_id"]: j for j in keep}
        _atomic_write(_state_dir() / "jobs.json", self.jobs)

    def save_events(self) -> None:
        _atomic_write(_state_dir() / "events.json", {"seq": self.event_seq, "events": self.events})


_STORE = _Store()


def _reset_state() -> None:
    """Drop the in-memory cache so the next access reloads from disk (tests /
    simulating a daemon restart)."""
    with _LOCK:
        _STORE.jobs = {}
        _STORE.events = []
        _STORE.job_seq = 0
        _STORE.event_seq = 0
        _STORE._loaded = False


# --------------------------------------------------------------------------- events

def record_event(kind: str, message: str, detail: dict | None = None) -> dict:
    """Append a structured event to the persisted, bounded feed."""
    with _LOCK:
        _STORE.ensure()
        _STORE.event_seq += 1
        event = {"seq": _STORE.event_seq, "ts": time.time(), "kind": kind, "message": message, "detail": detail or {}}
        _STORE.events.append(event)
        if len(_STORE.events) > _EVENT_CAP:
            del _STORE.events[: len(_STORE.events) - _EVENT_CAP]
        _STORE.save_events()
        return event


def list_events(since_seq: int = 0, limit: int = 100) -> list[dict]:
    with _LOCK:
        _STORE.ensure()
        fresh = [event for event in _STORE.events if event["seq"] > since_seq]
        return fresh[-limit:]


# --------------------------------------------------------------------------- jobs

@dataclass(frozen=True)
class JobStep:
    order: int
    title: str
    command: str
    status: str
    log: str


@dataclass(frozen=True)
class Job:
    job_id: str
    kind: str
    title: str
    status: str  # "dry_run" | "queued" | "running" | "succeeded" | "failed"
    dry_run: bool
    created_at: float
    summary: dict
    steps: tuple[JobStep, ...]
    log: tuple[str, ...]
    cli_mirror: tuple[str, ...]
    note: str = ""
    progress: float | None = None  # 0..100 for long-running jobs


def _job_from_dict(data: dict) -> Job:
    steps = tuple(
        JobStep(order=s["order"], title=s["title"], command=s["command"], status=s["status"], log=s["log"])
        for s in data.get("steps", [])
    )
    return Job(
        job_id=data["job_id"],
        kind=data["kind"],
        title=data["title"],
        status=data["status"],
        dry_run=data["dry_run"],
        created_at=data["created_at"],
        summary=data.get("summary", {}),
        steps=steps,
        log=tuple(data.get("log", [])),
        cli_mirror=tuple(data.get("cli_mirror", [])),
        note=data.get("note", ""),
        progress=data.get("progress"),
    )


def _next_id(kind: str) -> str:
    _STORE.job_seq += 1
    seed = f"{kind}:{_STORE.job_seq}:{time.time()}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8]
    return f"job_{_STORE.job_seq:04d}_{digest}"


def _put(data: dict) -> Job:
    _STORE.jobs[data["job_id"]] = data
    _STORE.save_jobs()
    return _job_from_dict(data)


def update_job(
    job_id: str,
    *,
    status: str | None = None,
    append_log: str | None = None,
    progress: float | None = None,
    note: str | None = None,
) -> Job | None:
    """Mutate a stored job (status / log / progress) and persist + emit an event."""
    with _LOCK:
        _STORE.ensure()
        data = _STORE.jobs.get(job_id)
        if data is None:
            return None
        if status is not None:
            data["status"] = status
        if append_log is not None:
            data.setdefault("log", []).append(append_log)
            if len(data["log"]) > 2000:
                data["log"] = data["log"][-2000:]
        if progress is not None:
            data["progress"] = max(0.0, min(100.0, progress))
        if note is not None:
            data["note"] = note
        _STORE.save_jobs()
        if status is not None:
            record_event("job", f"job {status}: {data['title']}", {"job_id": job_id, "kind": data["kind"], "status": status})
        return _job_from_dict(data)


def create_dry_run_job(*, kind: str, title: str, summary: dict, steps: list, cli_mirror: list, note: str = "") -> Job:
    """Record a dry-run job: ``steps`` are ``(title, command)`` pairs."""
    with _LOCK:
        _STORE.ensure()
        job_steps = [
            {"order": i + 1, "title": t, "command": c, "status": "previewed", "log": f"DRY RUN — would execute: {c}"}
            for i, (t, c) in enumerate(steps)
        ]
        log = [f"[plan] dry-run job created for {title}"] + [s["log"] for s in job_steps] + ["[plan] no host mutation performed (dry-run)"]
        data = {
            "job_id": _next_id(kind),
            "kind": kind,
            "title": title,
            "status": "dry_run",
            "dry_run": True,
            "created_at": time.time(),
            "summary": summary,
            "steps": job_steps,
            "log": log,
            "cli_mirror": list(cli_mirror),
            "note": note or "Dry-run only — real execution requires an explicit apply API.",
            "_seq": _STORE.job_seq,
        }
        job = _put(data)
    record_event("job", f"dry-run job created: {title}", {"job_id": job.job_id, "kind": kind, "status": job.status})
    return job


def create_executed_job(*, kind: str, title: str, summary: dict, step_results: list, cli_mirror: list, ok: bool, note: str = "") -> Job:
    """Record a real (executed) job from a list of ``StepResult`` objects."""
    job_steps = [
        {
            "order": r.order,
            "title": r.title,
            "command": r.command,
            "status": r.status,
            "log": "\n".join(p for p in (f"$ {r.command}", r.stdout, (f"[stderr] {r.stderr}" if r.stderr else ""), f"[exit {r.exit_code}]") if p),
        }
        for r in step_results
    ]
    log = [f"[apply] executed job for {title}"] + [s["log"] for s in job_steps] + [f"[apply] result: {'succeeded' if ok else 'failed'}"]
    with _LOCK:
        _STORE.ensure()
        data = {
            "job_id": _next_id(kind),
            "kind": kind,
            "title": title,
            "status": "succeeded" if ok else "failed",
            "dry_run": False,
            "created_at": time.time(),
            "summary": summary,
            "steps": job_steps,
            "log": log,
            "cli_mirror": list(cli_mirror),
            "note": note or "Executed action (apply enabled).",
            "_seq": _STORE.job_seq,
        }
        job = _put(data)
    record_event("job", f"executed job {job.status}: {title}", {"job_id": job.job_id, "kind": kind, "status": job.status})
    return job


def create_running_job(*, kind: str, title: str, summary: dict, command: str, note: str = "") -> Job:
    """Create a job in the ``running`` state for live background execution."""
    with _LOCK:
        _STORE.ensure()
        data = {
            "job_id": _next_id(kind),
            "kind": kind,
            "title": title,
            "status": "running",
            "dry_run": False,
            "created_at": time.time(),
            "summary": summary,
            "steps": [{"order": 1, "title": title, "command": command, "status": "running", "log": f"$ {command}"}],
            "log": [f"[apply] started: {command}"],
            "cli_mirror": [command],
            "note": note or "Running in the background.",
            "progress": 0.0,
            "_seq": _STORE.job_seq,
        }
        job = _put(data)
    record_event("job", f"job started: {title}", {"job_id": job.job_id, "kind": kind, "status": "running"})
    return job


def list_jobs(limit: int = 25) -> list[Job]:
    with _LOCK:
        _STORE.ensure()
        ordered = sorted(_STORE.jobs.values(), key=lambda j: j.get("created_at", 0), reverse=True)[:limit]
        return [_job_from_dict(j) for j in ordered]


def get_job(job_id: str) -> "Job | None":
    with _LOCK:
        _STORE.ensure()
        data = _STORE.jobs.get(job_id)
        return _job_from_dict(data) if data else None


def apply_service_action(*, preset_id: str, action: str = "status", runtime_target: str = "docker_compose", host: str = "127.0.0.1") -> Job:
    """Create a dry-run job from a service lifecycle plan."""
    from .service_plan import build_service_plan

    plan = build_service_plan(preset_id=preset_id, action=action, runtime_target=runtime_target, host=host)
    return create_dry_run_job(
        kind=f"service.{plan.action}",
        title=f"{plan.action} {preset_id}",
        summary={
            "preset_id": preset_id,
            "action": plan.action,
            "runtime_target": runtime_target,
            "host": host,
            "container": plan.container_name,
            "mutating": plan.mutating,
            "plan_id": plan.plan_id,
        },
        steps=[(step.title, step.command) for step in plan.steps],
        cli_mirror=list(plan.cli_mirror),
        note=plan.action_reason,
    )


__all__ = [
    "Job",
    "JobStep",
    "apply_service_action",
    "create_dry_run_job",
    "create_executed_job",
    "create_running_job",
    "get_job",
    "list_events",
    "list_jobs",
    "record_event",
    "update_job",
]
