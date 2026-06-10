# SPDX-License-Identifier: Apache-2.0
"""SNDR Core CLI — `sndr report` diagnostic bundle exporter.

DA-018 / production roadmap §18.5: collects a structured tarball of
operator-shareable diagnostic artifacts so support / GitHub issues
have full context out-of-the-box.

Subcommands:

  sndr report bundle [--output PATH] [--no-redact]
                     [--preset KEY] [--container NAME] [--remote HOST]

Bundle contents (9 artifacts):

  1. doctor.json          — `sndr doctor --full --json` output
  2. patches.json         — `sndr patches list --json` output
  3. launch_dryrun.sh     — `sndr launch --dry-run <preset>` rendered script
  4. vllm_boot.log        — last 200 lines of vllm boot log (if container)
  5. host_yaml.txt        — `~/.sndr/host.yaml` content (redacted)
  6. nvidia_smi.txt       — `nvidia-smi -q` (if available)
  7. pip_freeze.txt       — operator's env pip freeze
  8. git_log.txt          — last 10 commits of Genesis checkout
  9. image_inspect.json   — vLLM image digest + docker inspect summary

Output: `~/.sndr/reports/<timestamp>.tar.gz` by default. Override via
`--output`.

Redaction: ON by default. `--no-redact` for internal-only bundles
(rare; default-on protects operators against accidental leaks).

Author: Sandermage(Sander)-Barzov Aleksandr.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

from . import _io


# ─── Per-artifact collectors ─────────────────────────────────────────────


def _collect_doctor() -> dict[str, Any]:
    """Run the doctor module and return its JSON-serializable report."""
    try:
        from sndr.compat.doctor import collect_report
        report = collect_report()
        # Ensure JSON-serializable (collect_report may include Path objects).
        return json.loads(json.dumps(report, default=str))
    except Exception as e:
        return {
            "error": f"doctor.collect_report() raised: {type(e).__name__}: {e}"
        }


def _collect_patches() -> dict[str, Any]:
    """Snapshot of PATCH_REGISTRY + apply_module coverage."""
    try:
        from sndr.dispatcher import PATCH_REGISTRY
        from sndr.dispatcher.spec import iter_patch_specs
        specs = {s.patch_id: s.apply_module for s in iter_patch_specs()}
        out: dict[str, Any] = {
            "total": len(PATCH_REGISTRY),
            "by_tier": {},
            "by_lifecycle": {},
            "by_implementation_status": {},
            "specs_with_apply_module": sum(1 for v in specs.values() if v),
            "entries": [],
        }
        for pid, meta in PATCH_REGISTRY.items():
            if not isinstance(meta, dict):
                continue
            tier = meta.get("tier", "unknown")
            lc = meta.get("lifecycle", "unset")
            impl = meta.get("implementation_status", "unset")
            out["by_tier"][tier] = out["by_tier"].get(tier, 0) + 1
            out["by_lifecycle"][lc] = out["by_lifecycle"].get(lc, 0) + 1
            out["by_implementation_status"][impl] = (
                out["by_implementation_status"].get(impl, 0) + 1
            )
            out["entries"].append({
                "patch_id": pid,
                "title": meta.get("title", "")[:120],
                "tier": tier,
                "lifecycle": lc,
                "implementation_status": impl,
                "default_on": meta.get("default_on"),
                "env_flag": meta.get("env_flag"),
                "upstream_pr": meta.get("upstream_pr"),
                "apply_module": specs.get(pid),
            })
        return out
    except Exception as e:
        return {"error": f"registry snapshot failed: {type(e).__name__}: {e}"}


def _collect_launch_dryrun(preset_key: str | None) -> str:
    """Render the launch dry-run script for the given preset (if any)."""
    if not preset_key:
        return "(no --preset specified; rendered script omitted)\n"
    try:
        from sndr.model_configs.registry import get as _get_cfg
        cfg = _get_cfg(preset_key)
    except Exception as e:
        return f"(failed to load preset {preset_key!r}: {e})\n"
    if cfg is None:
        return f"(preset {preset_key!r} not found)\n"
    try:
        from sndr.model_configs.host import load_host_config
        host_paths = dict(load_host_config().paths) or {}
    except Exception:
        host_paths = {}
    try:
        return cfg.to_launch_script(host_paths=host_paths, strict_mounts=False)
    except Exception as e:
        return f"(render failed: {type(e).__name__}: {e})\n"


def _collect_host_yaml() -> str:
    """Read host.yaml (canonical or legacy)."""
    candidates = [
        Path.home() / ".sndr" / "host.yaml",
        Path.home() / ".genesis" / "host.yaml",
    ]
    for p in candidates:
        if p.is_file():
            try:
                return f"# source: {p}\n{p.read_text(encoding='utf-8')}"
            except OSError:
                continue
    return "(no host.yaml found at ~/.sndr/host.yaml or ~/.genesis/host.yaml)\n"


def _run_cmd(argv: list[str], *, timeout: int = 10) -> str:
    """Execute a subprocess; return combined stdout/stderr or error string."""
    try:
        r = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, check=False,
        )
        return r.stdout + ("\n--- stderr ---\n" + r.stderr if r.stderr else "")
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return f"(failed to run {' '.join(argv)}: {e})\n"


def _collect_nvidia_smi() -> str:
    if not shutil.which("nvidia-smi"):
        return "(nvidia-smi not available on this host)\n"
    return _run_cmd(["nvidia-smi", "-q"])


def _collect_pip_freeze() -> str:
    return _run_cmd([sys.executable, "-m", "pip", "freeze", "--all"], timeout=20)


def _collect_git_log() -> str:
    """Last 10 commits of the Genesis checkout (if running from source)."""
    repo_root = Path(__file__).resolve().parents[3]
    if not (repo_root / ".git").exists():
        return "(running outside a git checkout)\n"
    return _run_cmd(
        ["git", "-C", str(repo_root), "log", "-10",
         "--pretty=format:%h %ad %s", "--date=short"],
        timeout=10,
    )


def _collect_vllm_boot_log(container: str | None) -> str:
    """Last 200 lines of `docker logs <container>`. Empty if no container."""
    if not container:
        return "(no --container specified; vllm boot log omitted)\n"
    if not shutil.which("docker"):
        return "(docker CLI not available)\n"
    return _run_cmd(
        ["docker", "logs", "--tail", "200", container], timeout=15,
    )


def _collect_image_inspect(container: str | None) -> dict[str, Any]:
    """Container image digest + minimal inspect."""
    out: dict[str, Any] = {"container": container}
    if not container:
        out["status"] = "skipped (no --container)"
        return out
    if not shutil.which("docker"):
        out["status"] = "skipped (docker CLI absent)"
        return out
    try:
        r = subprocess.run(
            ["docker", "inspect", container],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        out["error"] = str(e)
        return out
    if r.returncode != 0:
        out["error"] = r.stderr.strip()
        return out
    try:
        data = json.loads(r.stdout)
        if data:
            entry = data[0]
            out["image"] = entry.get("Config", {}).get("Image")
            out["image_id"] = entry.get("Image")
            out["created"] = entry.get("Created")
            out["state"] = entry.get("State", {}).get("Status")
            out["mounts"] = [
                {"src": m.get("Source"), "dst": m.get("Destination"),
                 "mode": m.get("Mode")}
                for m in entry.get("Mounts", [])
            ]
    except (json.JSONDecodeError, IndexError, KeyError) as e:
        out["error"] = f"parse: {e}"
    return out


# ─── Bundle assembly ─────────────────────────────────────────────────────


_SCOPE_ARTIFACTS: dict[str, set[str]] = {
    # C19 (UNIFIED_CONFIG plan 2026-05-09): scope filter for bundle.
    # Operators sharing a bundle for ONE topic (e.g. dependency
    # planning, launch debugging) don't need ALL 9 artifacts.
    #
    # patch_plan.json — patch_plan
    # resolver snapshot for compat / safe / minimal under the given
    # --preset. Lets the reviewer see exactly which toggles each
    # policy would have produced without re-running the resolver.
    "all": {
        "doctor.json", "patches.json", "patch_plan.json",
        "launch_dryrun.sh", "vllm_boot.log",
        "host_yaml.txt", "nvidia_smi.txt", "pip_freeze.txt",
        "git_log.txt", "image_inspect.json",
    },
    "deps": {  # for `sndr deps plan` / install issues
        "doctor.json", "host_yaml.txt", "nvidia_smi.txt",
        "pip_freeze.txt", "git_log.txt",
    },
    "launch": {  # for boot/launch issues
        "doctor.json", "launch_dryrun.sh", "vllm_boot.log",
        "host_yaml.txt", "image_inspect.json", "patch_plan.json",
    },
    "quality": {  # for tool-call / model quality regressions
        "doctor.json", "patches.json", "patch_plan.json",
        "launch_dryrun.sh", "vllm_boot.log", "git_log.txt",
    },
    "patches": {  # for patch apply / drift questions
        "doctor.json", "patches.json", "patch_plan.json", "git_log.txt",
    },
}


def _collect_all(
    preset: str | None,
    container: str | None,
    scope: str = "all",
) -> dict[str, Any]:
    """Run scope-filtered collectors. Returns a dict keyed by artifact filename.

    `scope` is one of: 'all' (default), 'deps', 'launch', 'quality',
    'patches'. Unknown scopes degrade to 'all'.
    """
    selected = _SCOPE_ARTIFACTS.get(scope, _SCOPE_ARTIFACTS["all"])
    full: dict[str, Any] = {}
    if "doctor.json" in selected:
        full["doctor.json"] = _collect_doctor()
    if "patches.json" in selected:
        full["patches.json"] = _collect_patches()
    if "launch_dryrun.sh" in selected:
        full["launch_dryrun.sh"] = _collect_launch_dryrun(preset)
    if "vllm_boot.log" in selected:
        full["vllm_boot.log"] = _collect_vllm_boot_log(container)
    if "host_yaml.txt" in selected:
        full["host_yaml.txt"] = _collect_host_yaml()
    if "nvidia_smi.txt" in selected:
        full["nvidia_smi.txt"] = _collect_nvidia_smi()
    if "pip_freeze.txt" in selected:
        full["pip_freeze.txt"] = _collect_pip_freeze()
    if "git_log.txt" in selected:
        full["git_log.txt"] = _collect_git_log()
    if "image_inspect.json" in selected:
        full["image_inspect.json"] = _collect_image_inspect(container)
    if "patch_plan.json" in selected:
        pp = _collect_patch_plan(preset)
        if pp is not None:
            full["patch_plan.json"] = pp
    return full


def _collect_patch_plan(preset: str | None) -> dict[str, Any] | None:
    """capture patch_plan resolver output for the preset.

    Runs the resolver for compat / safe / minimal in parallel and
    summarises included / excluded / passthrough counts plus the
    full env map per policy. Returns None when no --preset was
    supplied (no anchor for the resolver). Returns an error marker
    dict when the preset can't be resolved — failure-to-collect is
    not a bundle blocker (the rest of the artifacts still ship).
    """
    if not preset:
        return None
    try:
        from sndr.cli.legacy.memory import _resolve_preset_v1_or_v2
        from sndr.model_configs.patch_plan import resolve_patch_plan
    except Exception as e:
        return {"preset": preset, "error": f"import failed: {e}"}

    try:
        cfg = _resolve_preset_v1_or_v2(preset)
    except Exception as e:
        return {
            "preset": preset,
            "error": f"preset resolution failed: {type(e).__name__}: {e}",
        }

    out: dict[str, Any] = {"preset": preset, "plans": {}}
    for policy in ("compat", "safe", "minimal"):
        try:
            plan = resolve_patch_plan(cfg, policy=policy)
        except Exception as e:
            out["plans"][policy] = {"error": f"{type(e).__name__}: {e}"}
            continue
        out["plans"][policy] = {
            "included_count": len(plan.included),
            "excluded_count": len(plan.excluded),
            "passthrough_count": len(plan.passthrough),
            "warnings": list(plan.warnings),
            "included_env_flags": [d.env_flag for d in plan.included],
            "excluded_env_flags": [d.env_flag for d in plan.excluded],
            "passthrough_keys": sorted(plan.passthrough),
        }
    return out


def _maybe_redact(artifacts: dict[str, Any], do_redact: bool
                  ) -> tuple[dict[str, Any], dict[str, int]]:
    """Apply redaction in-place. Returns (redacted_artifacts, hit_counts)."""
    if not do_redact:
        return artifacts, {}
    from sndr.runtime.redact import Redactor, load_user_rules, DEFAULT_RULES
    user_rules = load_user_rules()
    rules = DEFAULT_RULES + user_rules
    r = Redactor(rules=rules)
    out: dict[str, Any] = {}
    for name, artifact in artifacts.items():
        if isinstance(artifact, str):
            out[name] = r.redact(artifact)
        elif isinstance(artifact, (dict, list)):
            from sndr.runtime.redact import _walk
            out[name] = _walk(artifact, r)
        else:
            out[name] = artifact
    return out, dict(r.counts)


def _serialize(artifact: Any) -> bytes:
    """JSON for dicts/lists, raw text otherwise."""
    if isinstance(artifact, (dict, list)):
        return json.dumps(artifact, indent=2, default=str).encode("utf-8")
    if isinstance(artifact, str):
        return artifact.encode("utf-8")
    return str(artifact).encode("utf-8")


def _write_bundle(
    artifacts: dict[str, Any],
    redaction_counts: dict[str, int],
    output_path: Path,
    metadata: dict[str, Any],
) -> Path:
    """Write all artifacts + a top-level manifest into a tar.gz."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, "w:gz") as tar:
        # Top-level manifest
        manifest = {
            "schema_version": 1,
            "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "metadata": metadata,
            "redaction": {
                "enabled": metadata.get("redaction_enabled", True),
                "hit_counts": redaction_counts,
            },
            "artifact_index": list(artifacts.keys()),
        }
        manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
        info = tarfile.TarInfo(name="manifest.json")
        info.size = len(manifest_bytes)
        info.mtime = int(time.time())
        tar.addfile(info, io.BytesIO(manifest_bytes))

        for name, artifact in artifacts.items():
            payload = _serialize(artifact)
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            info.mtime = int(time.time())
            tar.addfile(info, io.BytesIO(payload))
    return output_path


# ─── CLI argparse + dispatch ─────────────────────────────────────────────


def add_argparser(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "report",
        help="Generate diagnostic report bundles for support / issues.",
        description="`sndr report bundle` — collect 9 artifacts into a redacted tar.gz.",
    )
    rsub = p.add_subparsers(dest="report_cmd", title="Subcommands")

    bundle = rsub.add_parser(
        "bundle",
        help="Create a tar.gz bundle of operator-shareable diagnostics.",
    )
    bundle.add_argument(
        "--output", type=Path, default=None,
        help="Output tar.gz path. Default: ~/.sndr/reports/<timestamp>.tar.gz",
    )
    bundle.add_argument(
        "--preset", type=str, default=None,
        help="Optional preset key to render via launch --dry-run (e.g. a5000-2x-35b-prod).",
    )
    bundle.add_argument(
        "--container", type=str, default=None,
        help="Optional Docker container name for `docker logs` + `docker inspect`.",
    )
    bundle.add_argument(
        "--no-redact", action="store_true",
        help="Skip default redaction (use ONLY for internal bundles).",
    )
    bundle.add_argument(
        "--print-summary", action="store_true",
        help="Print a one-line summary of the bundle to stdout after writing.",
    )
    bundle.add_argument(
        "--scope", choices=list(_SCOPE_ARTIFACTS.keys()),
        default="all",
        help="C19 (UNIFIED_CONFIG plan): filter which artifacts to "
             "include. 'deps' = inventory only; 'launch' = launch + "
             "boot logs; 'quality' = patches + boot; 'patches' = "
             "patches + git only; 'all' = everything (default).",
    )
    bundle.set_defaults(func=run_bundle)

    # Sprint 2.6: cudagraph dispatch hit-rate snapshot — surfaces
    # CUDA graph dispatch coverage per current process so operators
    # can detect V1-style regressions where prompt mutations push
    # requests into eager fallback (low hit-rate).
    cgrep = rsub.add_parser(
        "cudagraph-coverage",
        help=(
            "Print current CUDA graph dispatch hit-rate (Sprint 2.6). "
            "Requires GENESIS_CUDAGRAPH_DISPATCH_TRACE=1 + the dispatch "
            "wire-in to have recorded events."
        ),
    )
    cgrep.add_argument(
        "--json", action="store_true",
        help="Emit JSON (machine-readable) instead of human summary.",
    )
    cgrep.set_defaults(func=run_cudagraph_coverage)

    p.set_defaults(func=lambda args: (p.print_help(), 0)[1])


def run_cudagraph_coverage(opts: argparse.Namespace) -> int:
    """`sndr report cudagraph-coverage` — print current hit-rate."""
    import json as _json
    from sndr.observability import (
        get_cudagraph_summary,
        emit_cudagraph_summary,
    )

    snap = get_cudagraph_summary()
    payload = {
        "hits": snap.hits,
        "misses": snap.misses,
        "total": snap.total,
        "hit_rate_pct": snap.hit_rate_pct,
        "miss_rate_pct": snap.miss_rate_pct,
    }

    if getattr(opts, "json", False):
        print(_json.dumps(payload, indent=2))
        return 0

    if snap.total == 0:
        print(
            "[Genesis cudagraph] no dispatch events recorded yet. Set "
            "GENESIS_CUDAGRAPH_DISPATCH_TRACE=1 + run live workload, then "
            "re-invoke. The trace wire-in must be in place at the dispatch "
            "site (Sprint 2.6 follow-up patch)."
        )
        return 0

    print(
        f"[Genesis cudagraph] dispatch hit-rate {snap.hit_rate_pct}% "
        f"({snap.hits} captured-graph hits / {snap.total} total)\n"
        f"  eager fallback: {snap.misses} ({snap.miss_rate_pct}%)"
    )
    if snap.miss_rate_pct is not None and snap.miss_rate_pct > 10.0:
        print(
            "  ⚠ High eager-fallback rate — investigate prompt-shape or "
            "config changes that may have pushed requests off the captured "
            "graphs (Wave 6 PN16 V1 regression pattern)."
        )
    # Also emit through the structured logger for log aggregation
    emit_cudagraph_summary()
    return 0


def _default_output_path() -> Path:
    """`~/.sndr/reports/sndr-report-<ISO>.tar.gz`."""
    home = Path(
        os.environ.get("SNDR_HOME") or os.environ.get("GENESIS_HOME")
        or str(Path.home() / ".sndr")
    )
    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    return home / "reports" / f"sndr-report-{stamp}.tar.gz"


def run_bundle(opts: argparse.Namespace) -> int:
    """Orchestrate collection → redaction → tar.gz write."""
    output_path = opts.output or _default_output_path()
    do_redact = not opts.no_redact

    _io.banner(
        "sndr report bundle",
        f"redact={do_redact}  preset={opts.preset!r}  container={opts.container!r}",
    )
    _io.step(1, 4, "Collecting artifacts")
    scope = getattr(opts, "scope", "all")
    artifacts = _collect_all(preset=opts.preset, container=opts.container,
                              scope=scope)
    _io.success(f"collected {len(artifacts)} artifacts")

    _io.step(2, 4, "Applying redaction" if do_redact else "Skipping redaction")
    redacted, counts = _maybe_redact(artifacts, do_redact)
    if do_redact:
        n_hits = sum(counts.values())
        _io.success(f"redacted {n_hits} sensitive references "
                    f"({len(counts)} rule classes)")

    _io.step(3, 4, f"Writing tar.gz → {output_path}")
    metadata = {
        "preset": opts.preset,
        "container": opts.container,
        "redaction_enabled": do_redact,
    }
    try:
        _write_bundle(redacted, counts, output_path, metadata)
    except OSError as e:
        _io.error(f"write failed: {e}")
        return 1

    size_mib = output_path.stat().st_size / (1024 * 1024)
    _io.step(4, 4, f"Done — {size_mib:.2f} MiB at {output_path}")
    if opts.print_summary:
        print(f"sndr-report: {output_path} ({size_mib:.2f} MiB, "
              f"{len(redacted)} artifacts, "
              f"{sum(counts.values())} redactions)")
    return 0


__all__ = ["add_argparser", "run_bundle"]
