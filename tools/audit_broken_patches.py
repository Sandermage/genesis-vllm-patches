#!/usr/bin/env python3
"""tools/audit_broken_patches.py — find Genesis patches whose apply_module fails to import.

Distinguishes between:

  * **REAL broken** — module file is missing, or import raises
    ImportError for a sibling-Genesis-module reason. Critical: these
    fail at boot on the live container too.
  * **Local-only false positive** — module imports torch / triton /
    vllm at top level. Those are present on the PROD container but
    absent on the developer machine. Not a bug.

Usage::

    # Local audit (Mac dev box)
    python3 tools/audit_broken_patches.py

    # Remote audit (live container — bypasses local torch absence)
    python3 tools/audit_broken_patches.py --live

The --live flag runs the same audit *inside* the running container via
``ssh + docker exec``, so torch/triton/vllm are all present and only
*real* genesis-side broken imports surface.

Exit codes
==========

  * 0 — no real broken patches
  * 1 — at least one real broken patch found
  * 2 — invocation error
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from typing import Any

# Heuristic: these top-level imports inside a patch module mean local
# audit can't load them, but the live container can. NOT a broken
# patch — just an environment-mismatch with the dev machine.
_LIVE_ONLY_DEPS = frozenset({
    "torch", "torch.nn", "torch.cuda", "torch.distributed",
    "triton", "triton.language", "triton.runtime",
    "vllm", "vllm.model_executor", "vllm.v1", "vllm.config",
})


def _classify_import_error(exc: BaseException) -> str:
    """Return 'live_only' | 'real_broken' | 'unknown'."""
    msg = str(exc)
    if isinstance(exc, ModuleNotFoundError):
        # ModuleNotFoundError: No module named 'XYZ'
        name = getattr(exc, "name", None) or ""
        if name in _LIVE_ONLY_DEPS or any(
            name.startswith(prefix) for prefix in ("torch.", "triton.", "vllm.")
            if name not in {"vllm.sndr_core"}  # legacy alias, real if missing
        ):
            return "live_only"
        if name in _LIVE_ONLY_DEPS:
            return "live_only"
        # missing genesis-side sibling module → real broken
        if "sndr." in name or "sndr/" in msg:
            return "real_broken"
        # Anything else (e.g. missing _retired/ stub) is real broken
        return "real_broken"
    return "unknown"


def audit_local() -> dict[str, Any]:
    """Audit all patches via local import. Classify broken vs live-only."""
    from sndr.dispatcher.registry import PATCH_REGISTRY  # type: ignore

    real_broken: list[dict[str, str]] = []
    live_only: list[dict[str, str]] = []
    unconstrained: list[str] = []
    ok_count = 0

    for patch_id, entry in PATCH_REGISTRY.items():
        if not isinstance(entry, dict):
            continue
        mod_path = entry.get("apply_module")
        if mod_path is None:
            unconstrained.append(patch_id)
            continue
        try:
            importlib.import_module(mod_path)
            ok_count += 1
        except Exception as exc:  # noqa: BLE001
            klass = _classify_import_error(exc)
            info = {
                "patch_id": patch_id,
                "apply_module": mod_path,
                "error_type": type(exc).__name__,
                "error_msg": str(exc)[:200],
                "default_on": str(entry.get("default_on", False)),
                "lifecycle": str(entry.get("lifecycle", "")),
            }
            if klass == "live_only":
                live_only.append(info)
            else:
                real_broken.append(info)

    return {
        "ok_count": ok_count,
        "real_broken": real_broken,
        "live_only_count": len(live_only),
        "live_only": live_only,
        "unconstrained_count": len(unconstrained),
        "unconstrained": unconstrained,
    }


def audit_live(ssh_target: str, container: str) -> dict[str, Any]:
    """Run the same audit inside the live container — torch is present."""
    # Copy this script into the container and run it locally.
    script = """
import importlib
import json
try:
    from sndr.dispatcher.registry import PATCH_REGISTRY
except Exception as e:
    print(json.dumps({"error": f"cannot import registry: {e!r}"}))
    raise SystemExit(2)

real_broken = []
ok_count = 0
unconstrained = []
for pid, entry in PATCH_REGISTRY.items():
    if not isinstance(entry, dict):
        continue
    mod = entry.get("apply_module")
    if mod is None:
        unconstrained.append(pid)
        continue
    try:
        importlib.import_module(mod)
        ok_count += 1
    except Exception as exc:
        real_broken.append({
            "patch_id": pid,
            "apply_module": mod,
            "error_type": type(exc).__name__,
            "error_msg": str(exc)[:200],
            "default_on": str(entry.get("default_on", False)),
            "lifecycle": str(entry.get("lifecycle", "")),
        })

print(json.dumps({
    "ok_count": ok_count,
    "real_broken": real_broken,
    "unconstrained_count": len(unconstrained),
    "unconstrained": unconstrained,
}, indent=2))
"""
    # Write script to a tmp file and copy over — avoids shell escape hell.
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tf:
        tf.write(script)
        tmp_path = tf.name
    # Copy to host then exec inside container via bind-mounted home dir
    proc1 = subprocess.run(
        ["scp", tmp_path, f"{ssh_target}:/tmp/_audit_broken.py"],
        capture_output=True, text=True, timeout=30,
    )
    if proc1.returncode != 0:
        return {"error": f"scp failed: {proc1.stderr[:300]}"}
    cmd = [
        "ssh", ssh_target,
        f"docker cp /tmp/_audit_broken.py {container}:/tmp/_audit_broken.py && "
        f"docker exec {container} python3 /tmp/_audit_broken.py",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        return {"error": f"ssh+docker exec failed: {proc.stderr[:300]}"}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        return {"error": f"could not parse live output: {e!s}",
                "stdout_head": proc.stdout[:400]}


def render(audit: dict[str, Any], live: bool) -> str:
    """Render audit result as human-readable markdown."""
    lines: list[str] = []
    head = "LIVE-CONTAINER" if live else "LOCAL"
    lines.append(f"# Patch import audit — {head}")
    lines.append("")

    if "error" in audit:
        lines.append(f"**ERROR**: {audit['error']}")
        return "\n".join(lines)

    ok = audit.get("ok_count", 0)
    real = audit.get("real_broken", [])
    live_only = audit.get("live_only", [])
    unconstr = audit.get("unconstrained_count", 0)

    lines.append(f"* ✓ OK imports: **{ok}**")
    lines.append(f"* ⚠️ no apply_module (unconstrained): **{unconstr}**")
    if not live:
        lines.append(
            f"* ℹ️ live-only deps (torch/triton/vllm at module top): "
            f"**{audit.get('live_only_count', 0)}** "
            f"(present on PROD, absent on dev — not a bug)"
        )
    lines.append(f"* 🚨 **REAL broken**: **{len(real)}**")
    lines.append("")

    if real:
        lines.append("## Real broken patches (require fix)")
        lines.append("")
        lines.append("| Patch ID | default_on | lifecycle | error | message |")
        lines.append("|---|---|---|---|---|")
        for r in real:
            msg = r['error_msg'].replace('|', '\\|')[:120]
            lines.append(
                f"| {r['patch_id']} | {r['default_on']} | {r['lifecycle']} | "
                f"{r['error_type']} | `{msg}` |"
            )
        lines.append("")
    else:
        lines.append("✅ **No real broken patches.**")

    if not live and live_only:
        lines.append("")
        lines.append(
            f"## Live-only false positives ({len(live_only)} — not a bug)"
        )
        lines.append("")
        for lo in live_only[:5]:
            lines.append(f"* {lo['patch_id']} — depends on `{lo['error_msg'][:60]}…`")
        if len(live_only) > 5:
            lines.append(f"* ... and {len(live_only) - 5} more")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--live", action="store_true",
                    help="Run audit inside the live container via ssh+docker exec")
    ap.add_argument("--ssh-target", default=os.environ.get("SSH_HOST", ""),
                    help="ssh target <user>@<host> (default: $SSH_HOST)")
    ap.add_argument("--container", default="vllm-qwen3.6-35b-balanced-k3")
    ap.add_argument("--ci-strict", action="store_true",
                    help="Exit 1 if any real broken patch found")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.live:
        if not args.ssh_target:
            print("error: --live needs an ssh target; pass --ssh-target "
                  "or set SSH_HOST", file=sys.stderr)
            return 2
        audit = audit_live(args.ssh_target, args.container)
    else:
        audit = audit_local()

    if args.json:
        print(json.dumps(audit, indent=2, ensure_ascii=False))
    else:
        print(render(audit, live=args.live))

    if args.ci_strict and audit.get("real_broken"):
        print(f"\nCI-strict: {len(audit['real_broken'])} real broken — exit 1",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
