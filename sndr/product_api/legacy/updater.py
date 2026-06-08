# SPDX-License-Identifier: Apache-2.0
"""Pin-gated self-updater for the GUI + ``sndr_core`` patcher.

Design constraints (mirror the project's vLLM pin policy):

* **Read-only by default.** ``collect_status`` / ``check_remote`` / ``build_plan``
  run no mutating command — they inspect the local checkout and (for the remote
  check) do a read-only ``git ls-remote``.
* **Pin-gated.** An update may only move the vLLM pin to a value the patcher
  *declares it supports* (``vllm_pin_required`` in the builtin model configs).
  A requested pin outside that set blocks the plan. Newer pins are never
  pulled implicitly — the operator picks a supported target.
* **The server docker pin step stays manual.** Pulling/re-tagging the vLLM image
  on the GPU host is the highest-risk, policy-governed action, so the plan emits
  it as a copyable command rather than auto-running it.
* **Apply is gated + confirmed + clean-tree-guarded.** Local steps only run when
  ``SNDR_ENABLE_APPLY`` is on, ``confirm`` is true, the pin gate passes and the
  working tree is clean.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Optional

from sndr.version import SNDR_CORE_VERSION

_PIN_RE = re.compile(r"^\s*vllm_pin_required:\s*([^\s#]+)")


def _repo_root() -> Optional[Path]:
    """The git checkout root, or None when running from an installed package."""
    here = Path(__file__).resolve()
    if len(here.parents) >= 4:
        root = here.parents[3]  # product_api -> sndr_core -> vllm -> <repo>
        if (root / ".git").exists():
            return root
    return None


def _builtin_dir() -> Path:
    # parents[0]=legacy, [1]=product_api, [2]=sndr after the relocation.
    return Path(__file__).resolve().parents[2] / "model_configs" / "builtin"


def _git(root: Path, *args: str, timeout: float = 10.0) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True, text=True, timeout=timeout,
        )
        return proc.returncode, (proc.stdout or "").strip()
    except Exception as exc:  # git missing / timeout
        return 1, f"{type(exc).__name__}: {exc}"


def supported_pins(configs_dir: Optional[Path] = None) -> list[str]:
    """Distinct ``vllm_pin_required`` values the patcher declares, most-common first."""
    base = configs_dir or _builtin_dir()
    counts: dict[str, int] = {}
    try:
        for path in base.rglob("*.yaml"):
            try:
                for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                    m = _PIN_RE.match(line)
                    if m:
                        counts[m.group(1)] = counts.get(m.group(1), 0) + 1
            except OSError:
                continue
    except Exception:
        return []
    return sorted(counts, key=lambda k: (-counts[k], k))


def _gui_build_info() -> dict[str, Any]:
    """Best-effort info about the published web build under ``web_static``."""
    static = Path(__file__).resolve().parent / "web_static"
    index = static / "index.html"
    info: dict[str, Any] = {"published": index.is_file()}
    if index.is_file():
        try:
            info["built_at"] = int(index.stat().st_mtime)
        except OSError:
            pass
        try:
            assets = sorted((static / "assets").glob("index-*.js"))
            info["bundle"] = assets[-1].name if assets else None
        except OSError:
            pass
    return info


def collect_status() -> dict[str, Any]:
    """Read-only snapshot: patcher version, supported pins, git + GUI build state."""
    root = _repo_root()
    git: dict[str, Any] = {"is_repo": root is not None}
    if root is not None:
        _, branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
        _, commit = _git(root, "rev-parse", "--short", "HEAD")
        _, porcelain = _git(root, "status", "--porcelain")
        _, remote = _git(root, "remote", "get-url", "origin")
        git.update(branch=branch, commit=commit, dirty=bool(porcelain.strip()), remote=remote)
    pins = supported_pins()
    return {
        "sndr_core_version": SNDR_CORE_VERSION,
        "supported_pins": pins,
        "canonical_pin": pins[0] if pins else None,
        "git": git,
        "gui_build": _gui_build_info(),
        "apply_enabled": _apply_enabled(),
    }


def check_remote(timeout: float = 20.0) -> dict[str, Any]:
    """Read-only remote check (``git ls-remote``) — is a newer commit available?"""
    root = _repo_root()
    if root is None:
        return {"is_repo": False, "update_available": False, "error": "not a git checkout"}
    _, branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    _, local = _git(root, "rev-parse", "HEAD")
    rc, out = _git(root, "ls-remote", "origin", branch, timeout=timeout)
    if rc != 0 or not out:
        return {"is_repo": True, "branch": branch, "local_commit": local[:12], "remote_commit": None,
                "update_available": False, "error": "could not reach remote"}
    remote_commit = out.split()[0]
    update_available = bool(remote_commit and local and remote_commit != local)
    return {
        "is_repo": True, "branch": branch, "local_commit": local[:12],
        "remote_commit": remote_commit[:12], "update_available": update_available, "error": None,
    }


def pin_gate(target_pin: Optional[str], supported: list[str]) -> dict[str, Any]:
    """Enforce: the target pin must be one the patcher declares it supports."""
    if not target_pin:
        return {"ok": True, "target_pin": supported[0] if supported else None, "reason": None}
    if target_pin in supported:
        return {"ok": True, "target_pin": target_pin, "reason": None}
    return {
        "ok": False, "target_pin": target_pin,
        "reason": f"pin {target_pin} is not declared supported by this patcher "
                  f"(supported: {', '.join(supported) or 'none'})",
    }


def build_plan(target_pin: Optional[str] = None) -> dict[str, Any]:
    """Read-only, ordered, pin-gated update plan. Runs nothing."""
    status = collect_status()
    root = _repo_root()
    repo = str(root) if root else "<repo>"
    static = str(Path(__file__).resolve().parent / "web_static")
    gate = pin_gate(target_pin, status["supported_pins"])
    tgt = gate["target_pin"]

    steps = [
        {"order": 1, "title": "Pull latest patcher + GUI source", "kind": "local",
         "cmd": f"git -C {repo} pull --ff-only"},
        {"order": 2, "title": "Reinstall the sndr_core patcher", "kind": "local",
         "cmd": f"python3 -m pip install -e '{repo}'"},
        {"order": 3, "title": "Rebuild the web GUI", "kind": "local",
         "cmd": f"npm --prefix '{repo}/gui/web' ci && npm --prefix '{repo}/gui/web' run build"},
        {"order": 4, "title": "Publish the GUI build to the daemon", "kind": "local",
         "cmd": f"rm -rf '{static}' && cp -R '{repo}/gui/web/dist' '{static}'"},
        {"order": 5, "title": f"Set the server vLLM pin to {tgt or '<supported pin>'} (MANUAL — pin policy)",
         "kind": "server-manual", "pin": tgt,
         "cmd": f"# on the GPU host: docker pull vllm/vllm-openai:{tgt} "
                f"&& docker tag vllm/vllm-openai:{tgt} vllm/vllm-openai:nightly  "
                f"# keep the prior pin as :nightly-previous for rollback"},
        {"order": 6, "title": "Restart the daemon", "kind": "local",
         "cmd": "# restart: re-run `python3 -m sndr.cli gui-api ...`"},
    ]

    blocked: list[str] = []
    if not gate["ok"]:
        blocked.append(gate["reason"])
    if status["git"].get("dirty"):
        blocked.append("working tree has uncommitted changes — commit or stash before applying")
    if not status["git"].get("is_repo"):
        blocked.append("not a git checkout — update via your package manager instead")

    return {
        "valid": not blocked,
        "blocked_reasons": blocked,
        "pin_gate": gate,
        "target_pin": tgt,
        "current_version": status["sndr_core_version"],
        "steps": steps,
    }


def _apply_enabled() -> bool:
    return os.environ.get("SNDR_ENABLE_APPLY", "").strip().lower() in ("1", "true", "yes", "on")


def apply_plan(*, confirm: bool, apply_enabled: Optional[bool] = None, target_pin: Optional[str] = None) -> dict[str, Any]:
    """Apply the LOCAL update steps. Gated + confirmed + pin-gated + clean-tree.

    The server docker-pin step is never auto-run — it is returned as a manual
    command for the operator to execute deliberately (pin policy)."""
    enabled = _apply_enabled() if apply_enabled is None else bool(apply_enabled)
    if not enabled:
        return {"applied": False, "status": "disabled",
                "message": "Updater disabled — start the daemon with SNDR_ENABLE_APPLY=1."}
    if not confirm:
        return {"applied": False, "status": "needs_confirm", "message": "confirm=true is required to apply."}
    plan = build_plan(target_pin)
    if not plan["valid"]:
        return {"applied": False, "status": "blocked", "message": "; ".join(plan["blocked_reasons"]), "plan": plan}

    from .runtime_exec import run_steps

    local = [(s["title"], s["cmd"]) for s in plan["steps"] if s["kind"] == "local" and not s["cmd"].lstrip().startswith("#")]
    results = run_steps(local, timeout=900)
    ok = all(r.status == "ok" for r in results)
    return {
        "applied": True,
        "status": "done" if ok else "partial",
        "results": [
            {"order": r.order, "title": r.title, "command": r.command,
             "exit_code": r.exit_code, "status": r.status, "stdout": r.stdout[-4000:], "stderr": r.stderr[-2000:]}
            for r in results
        ],
        "manual_steps": [s for s in plan["steps"] if s["kind"] != "local"],
        "plan": plan,
    }


__all__ = [
    "apply_plan",
    "build_plan",
    "check_remote",
    "collect_status",
    "pin_gate",
    "supported_pins",
]
