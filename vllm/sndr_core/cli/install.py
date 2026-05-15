# SPDX-License-Identifier: Apache-2.0
"""SNDR Core CLI — `sndr install` canonical bootstrap + wizard.

S-05 fix (audit 2026-05-08): this module is the SINGLE source of truth
for Genesis install logic. The root `install.sh` is now a thin
bootstrap that ensures Python + pip + this package, then exec's
`sndr install` with the operator's flags.

Everything that used to live in 783 lines of bash now lives here:

  Pre-flight  : OS, Python, git/curl, disk space
  Hardware    : GPU class hint (32 product strings), driver version,
                CUDA runtime, RAM, free disk
  Runtime     : Proxmox VE caveat (auto-flips to bare-metal mode)
  Workload    : balanced / long_context / high_throughput / tool_agent
  Pin resolve : stable=latest tag (GitHub API), dev=branch tip,
                or explicit ref
  Clone       : git clone or fetch+checkout into ~/.sndr (or $SNDR_HOME)
  Install     : pip install + plugin install (vllm.general_plugins)
  Host paths  : auto-detect models_dir/hf_cache/etc → ~/.sndr/host.yaml
  Launch gen  : pick preset by (gpu, n_gpus, workload) → bash script
  Smoke       : `apply.run(apply=False)` dispatcher dry-run
  Next steps  : rustup-style summary

Sub-commands:

  sndr install                     # interactive bootstrap + wizard
  sndr install --dry-run           # report-only, no clone / no pip
  sndr install -y --workload tool_agent
  sndr install --uninstall         # remove plugin + host.yaml hint

Author: Sandermage (Sander) Barzov Aleksandr.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from . import _io


# ─── Constants ───────────────────────────────────────────────────────────


_DEFAULT_REPO = "https://github.com/Sandermage/genesis-vllm-patches.git"
_DEFAULT_HOME = Path.home() / ".sndr"
_LEGACY_HOME = Path.home() / ".genesis"  # back-compat with v7.x operators
_MIN_PYTHON = (3, 10)
_MIN_DISK_MB = 200

WORKLOADS = {
    "balanced":        "Default-safe — chat + occasional long ctx + tools",
    "long_context":    "Single long prompt (>50K), low concurrency",
    "high_throughput": "Many short prompts in parallel, max TPS",
    "tool_agent":      "IDE coding agents (Cline / Claude Code / OpenCode)",
}


# ─── Step result types ───────────────────────────────────────────────────


@dataclass
class StepResult:
    """Generic step outcome — keys are step-specific, used to thread
    state between steps and into the final summary."""
    data: dict[str, Any]


# ─── Pre-flight ──────────────────────────────────────────────────────────


def step_preflight(opts: argparse.Namespace) -> StepResult:
    """OS check, Python ≥3.10, git/curl present, disk space."""
    _io.step(1, 11, "Pre-flight checks")

    out: dict[str, Any] = {}

    osname = platform.system()
    out["os"] = osname
    if osname == "Linux":
        _io.success("OS: Linux")
    elif osname == "Darwin":
        _io.warn("OS: macOS — install will set up the package, but vllm "
                 "serve won't run here (Genesis targets Linux/CUDA).")
    else:
        _io.fatal(f"unsupported OS: {osname!r}", 2)

    pyver = sys.version_info
    out["python_version"] = f"{pyver.major}.{pyver.minor}.{pyver.micro}"
    if (pyver.major, pyver.minor) < _MIN_PYTHON:
        _io.fatal(f"Python {out['python_version']} too old "
                  f"(≥{_MIN_PYTHON[0]}.{_MIN_PYTHON[1]} required)", 2)
    _io.success(f"Python {out['python_version']} (≥3.10)")

    for tool in ("git", "curl"):
        if shutil.which(tool):
            _io.success(f"{tool} available")
        else:
            _io.fatal(f"{tool} not found — required for install", 2)

    home = Path(opts.home or os.environ.get("SNDR_HOME") or
                os.environ.get("GENESIS_HOME") or _DEFAULT_HOME)
    out["sndr_home"] = str(home)
    parent = home.parent
    parent.mkdir(parents=True, exist_ok=True)
    try:
        free_b = shutil.disk_usage(parent).free
        free_mb = free_b // (1024 * 1024)
        out["free_mb"] = free_mb
        if free_mb < _MIN_DISK_MB:
            _io.warn(f"only {free_mb} MiB free at {parent} — clone may fail")
        else:
            _io.success(f"{free_mb} MiB free at {parent}")
    except OSError as e:
        _io.warn(f"disk_usage({parent}) failed: {e}")
        out["free_mb"] = None

    return StepResult(out)


# ─── Hardware detection ──────────────────────────────────────────────────


def step_detect_hardware(opts: argparse.Namespace) -> StepResult:
    """GPU(s), class hint, driver version, CUDA runtime, RAM, free disk."""
    _io.step(2, 11, "Detecting hardware")

    from vllm.sndr_core.detection.gpu_class_map import classify_gpu
    from vllm.sndr_core.detection.driver_check import probe_driver

    out: dict[str, Any] = {}

    # GPU enumeration via nvidia-smi (the Python torch path is unreliable
    # on bare-machine bootstrap before vllm is installed).
    n_gpus, gpu_name = _query_gpu_via_nvidia_smi()
    out["n_gpus"] = n_gpus
    out["gpu_name"] = gpu_name

    if n_gpus == 0:
        _io.warn("no CUDA GPU detected via nvidia-smi (Mac dev / CPU-only?)")
        out["gpu_class_hint"] = ""
    else:
        hint = classify_gpu(gpu_name)
        out["gpu_class_hint"] = hint
        if hint:
            _io.success(f"Found {n_gpus}× {gpu_name} → preset class {hint!r}")
        else:
            _io.warn(f"GPU {gpu_name!r} not in Genesis preset matrix — "
                     "install will continue without preset hint")

    # Driver
    drv = probe_driver()
    out["driver_version"] = drv.raw_driver_version
    out["driver_major"] = drv.driver_major
    out["cuda_runtime"] = drv.cuda_runtime_reported
    out["driver_below_recommended"] = drv.below_recommended

    if drv.below_recommended:
        _io.warn(drv.recommendation)
        if not _confirm_continue(opts):
            _io.fatal("aborted — upgrade driver to ≥580.x and re-run", 1)
    elif drv.nvidia_smi_present:
        _io.success(drv.recommendation)

    # RAM (best-effort, not blocking)
    try:
        if hasattr(os, "sysconf"):
            mem_b = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            out["ram_gib"] = round(mem_b / (1024 ** 3), 1)
            _io.success(f"RAM: {out['ram_gib']} GiB")
    except (OSError, ValueError):
        out["ram_gib"] = None

    return StepResult(out)


def _query_gpu_via_nvidia_smi() -> tuple[int, str]:
    """Return (n_gpus, first_gpu_name). Either zero/empty if unavailable."""
    if not shutil.which("nvidia-smi"):
        return (0, "")
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return (0, "")
    if r.returncode != 0:
        return (0, "")
    names = [line.strip() for line in r.stdout.splitlines() if line.strip()]
    if not names:
        return (0, "")
    return (len(names), names[0])


def _confirm_continue(opts: argparse.Namespace) -> bool:
    """Return True if operator is OK to continue (or non-interactive)."""
    if getattr(opts, "non_interactive", False) or not sys.stdin.isatty():
        _io.warn("non-interactive — proceeding regardless")
        return True
    ans = _io.prompt("Continue anyway?", default="N", non_interactive=False)
    return ans.strip().lower() in ("y", "yes", "1")


# ─── vLLM detection ──────────────────────────────────────────────────────


def step_detect_vllm(opts: argparse.Namespace) -> StepResult:
    """Probe for an existing vllm install."""
    _io.step(3, 11, "Detecting existing vllm install")
    out: dict[str, Any] = {}
    try:
        import vllm
        # `vllm.__file__` may be None if the operator's environment has a
        # namespace shim or a sys.modules redirect rather than a real
        # vllm package — treat that as "not really installed".
        f = getattr(vllm, "__file__", None)
        if f:
            out["installed"] = True
            out["version"] = getattr(vllm, "__version__", "unknown")
            out["root"] = os.path.dirname(f)
            _io.success(f"vllm {out['version']} at {out['root']}")
            if "0.20" not in out["version"] and out["version"] != "unknown":
                _io.warn(f"Genesis is pinned to vllm 0.20.x — your "
                         f"{out['version']} may have anchor drift")
        else:
            out["installed"] = False
            _io.warn("vllm importable but has no __file__ (namespace "
                     "shim?) — treating as not installed")
    except ImportError:
        out["installed"] = False
        _io.warn("vllm not importable — Genesis installs anyway, but "
                 "patches need vllm before they can apply")
    return StepResult(out)


# ─── Runtime caveat probe ────────────────────────────────────────────────


def step_runtime_caveat(opts: argparse.Namespace) -> StepResult:
    """Detect Proxmox VE host → auto-flip to bare-metal mode."""
    _io.step(4, 11, "Container-runtime caveat probe")
    from vllm.sndr_core.detection.runtime_caveat import probe_caveats

    cav = probe_caveats()
    out: dict[str, Any] = {
        "proxmox_detected": cav.proxmox_detected,
        "kernel": cav.kernel_release,
    }
    if cav.proxmox_detected:
        _io.warn(cav.reason)
        if not opts.bare_metal:
            opts.bare_metal = True
            _io.info("auto-enabled --bare-metal")
        out["bare_metal_auto"] = True
    else:
        _io.success(cav.reason)
        out["bare_metal_auto"] = False
    return StepResult(out)


# ─── Workload picker ─────────────────────────────────────────────────────


def step_pick_workload(opts: argparse.Namespace) -> StepResult:
    """Interactive workload selection (or env-driven)."""
    _io.step(5, 11, "Pick workload")

    # 1. Explicit flag wins
    wl = opts.workload
    if wl:
        if wl not in WORKLOADS:
            _io.fatal(f"invalid --workload {wl!r}. One of: "
                      f"{', '.join(WORKLOADS)}", 2)
        _io.success(f"workload: {wl} (from --workload)")
        return StepResult({"workload": wl})

    # 2. Non-interactive default
    if opts.non_interactive or not sys.stdin.isatty():
        wl = "balanced"
        _io.success(f"workload: {wl} (non-interactive default)")
        return StepResult({"workload": wl})

    # 3. Interactive picker
    print("\nPick the workload Genesis should optimize for:\n")
    keys = list(WORKLOADS)
    for i, k in enumerate(keys, 1):
        print(f"  {i}) {k:<18s} — {WORKLOADS[k]}")
    print()
    while True:
        ans = _io.prompt("Choice", default="1", non_interactive=False).strip()
        try:
            idx = int(ans) - 1
            if 0 <= idx < len(keys):
                wl = keys[idx]
                break
        except ValueError:
            if ans in WORKLOADS:
                wl = ans
                break
        print(f"  invalid — pick 1-{len(keys)} or a workload name")
    _io.success(f"workload: {wl}")
    return StepResult({"workload": wl})


# ─── Pin resolution ──────────────────────────────────────────────────────


def step_resolve_pin(opts: argparse.Namespace) -> StepResult:
    """Resolve `--pin <ref>` to a concrete git ref.

    `stable` = latest GitHub tag (via API; falls back to `main` on
    network failure). `dev` = `dev` branch tip. Anything else: passed
    through verbatim.
    """
    _io.step(6, 11, "Resolve Genesis pin")
    pin = opts.pin

    if pin == "stable":
        tag = _resolve_latest_tag()
        if tag:
            _io.success(f"pin: {tag} (latest stable tag)")
            return StepResult({"pin": tag, "kind": "stable"})
        _io.warn("could not query GitHub tags API — falling back to 'main'")
        return StepResult({"pin": "main", "kind": "stable_fallback"})

    if pin == "dev":
        _io.success("pin: dev (branch tip — mutable)")
        _io.info("dev is mutable; use --pin <commit> for reproducible builds")
        return StepResult({"pin": "dev", "kind": "dev"})

    _io.success(f"pin: {pin} (explicit ref)")
    return StepResult({"pin": pin, "kind": "explicit"})


def _resolve_latest_tag(timeout: float = 10.0) -> Optional[str]:
    """Query GitHub tags API for the most recent tag. None on failure."""
    url = ("https://api.github.com/repos/Sandermage/genesis-vllm-patches/"
           "tags?per_page=10")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # nosec B310
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, list) or not data:
        return None
    first = data[0]
    if isinstance(first, dict) and isinstance(first.get("name"), str):
        return first["name"]
    return None


# ─── Clone or update ─────────────────────────────────────────────────────


def step_clone_or_update(
    opts: argparse.Namespace, preflight: dict[str, Any], pin: dict[str, Any],
) -> StepResult:
    """`git clone` or `git fetch+checkout` Genesis into SNDR_HOME."""
    home = Path(preflight["sndr_home"])
    _io.step(7, 11, f"Genesis source ({home})")

    if opts.dry_run:
        _io.info(f"(dry-run: would clone {opts.repo} → {home} @ {pin['pin']})")
        return StepResult({"home": str(home), "head": "<dry-run>"})

    if (home / ".git").is_dir():
        _io.info("found existing clone — updating")
        _run_git(home, ["fetch", "--tags", "origin"])
        _run_git(home, ["checkout", "--quiet", pin["pin"]])
        # If on a branch, fast-forward
        try:
            _run_git(home, ["symbolic-ref", "-q", "HEAD"], check=False)
            _run_git(home, ["pull", "--ff-only", "--quiet", "origin",
                            pin["pin"]], check=False)
        except subprocess.CalledProcessError:
            pass  # detached HEAD — fast-forward not applicable
    else:
        _io.info(f"cloning from {opts.repo}")
        _run([
            "git", "clone", "--quiet", opts.repo, str(home),
        ])
        _run_git(home, ["checkout", "--quiet", pin["pin"]])

    head = _run_git(home, ["rev-parse", "--short", "HEAD"]).stdout.strip()
    _io.success(f"Genesis at {head} (ref: {pin['pin']})")

    # Sanity — required canonical files
    required = (
        "vllm/sndr_core/__init__.py",
        "vllm/sndr_core/apply/orchestrator.py",
        "vllm/sndr_core/compat/cli.py",
    )
    for rel in required:
        if not (home / rel).is_file():
            _io.fatal(f"Genesis tree at {pin['pin']!r} missing {rel} "
                      "— wrong pin?", 2)

    return StepResult({"home": str(home), "head": head})


def _run(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
    """Subprocess wrapper that captures + returns CompletedProcess."""
    kw.setdefault("capture_output", True)
    kw.setdefault("text", True)
    kw.setdefault("check", True)
    return subprocess.run(cmd, **kw)


def _run_git(cwd: Path, args: list[str], *, check: bool = True
             ) -> subprocess.CompletedProcess[str]:
    """Run `git` in `cwd`."""
    return _run(["git", "-C", str(cwd), *args], check=check)


# ─── Plugin install ──────────────────────────────────────────────────────


def _resolve_plugin_src(home: Path) -> Path | None:
    """Locate the Genesis vllm-plugin source directory.

    Search order (first directory containing pyproject.toml wins):
      1. ${GENESIS_PLUGIN_SRC} env var (operator override)
      2. <Genesis checkout home>/tools/genesis_vllm_plugin  (default
         operator-side repo layout)
      3. <home>/genesis_vllm_plugin                          (alt layout)

    Returns None if no candidate is a directory with a pyproject.toml.
    """
    import os
    candidates: list[Path] = []
    env_override = os.environ.get("GENESIS_PLUGIN_SRC", "").strip()
    if env_override:
        candidates.append(Path(env_override).expanduser())
    candidates.append(home / "tools" / "genesis_vllm_plugin")
    candidates.append(home / "genesis_vllm_plugin")
    for c in candidates:
        if c.is_dir() and (c / "pyproject.toml").is_file():
            return c
    return None


def step_install_plugin(
    opts: argparse.Namespace, clone_info: dict[str, Any],
) -> StepResult:
    """`pip install -e <plugin-src>` so vLLM auto-loads Genesis in spawn
    workers. Plugin source location resolved via _resolve_plugin_src()
    — operators can override with GENESIS_PLUGIN_SRC."""
    if opts.no_plugin:
        _io.warn("skipping plugin install (--no-plugin)")
        return StepResult({"installed": False, "reason": "--no-plugin"})

    _io.step(8, 11, "Install genesis-vllm-plugin (vllm.general_plugins)")
    if opts.dry_run:
        _io.info("(dry-run: would pip install -e <Genesis plugin source>)")
        return StepResult({"installed": False, "reason": "dry-run"})

    home = Path(clone_info["home"])
    plugin = _resolve_plugin_src(home)
    if plugin is None:
        _io.warn(
            "Genesis plugin source not found in this checkout — skipping. "
            "Set GENESIS_PLUGIN_SRC to override."
        )
        return StepResult({"installed": False, "reason": "missing"})

    pip_args = []
    if opts.system:
        _io.info("using system pip (--system)")
    else:
        pip_args.append("--user")

    cmd = [sys.executable, "-m", "pip", "install", "-q", *pip_args, "-e",
           str(plugin)]
    _io.info(f"$ {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        _io.warn("plugin pip install failed — Genesis still works via "
                 "PYTHONPATH but won't auto-load in spawn workers")
        if r.stderr:
            _io.info(r.stderr.strip()[:400])
        return StepResult({"installed": False, "reason": "pip_failed"})
    _io.success("genesis-vllm-plugin installed")

    # Verify entry point registered
    try:
        from importlib.metadata import entry_points
        eps = entry_points(group="vllm.general_plugins")
        names = [ep.name for ep in eps]
        if "genesis_v7" in names:
            _io.success("vllm.general_plugins → genesis_v7 registered")
        else:
            _io.warn(f"entry point 'genesis_v7' not found post-install "
                     f"(got {names!r})")
    except Exception as e:
        _io.warn(f"entry-point verification raised {type(e).__name__}: {e}")

    return StepResult({"installed": True})


# ─── Host paths detection ────────────────────────────────────────────────


def step_detect_host_paths(
    opts: argparse.Namespace, clone_info: dict[str, Any],
) -> StepResult:
    """Auto-detect `models_dir`/`hf_cache`/etc → write `~/.sndr/host.yaml`."""
    _io.step(9, 11, "Auto-detect host paths (host.yaml)")
    if opts.dry_run:
        _io.info("(dry-run: would write host.yaml)")
        return StepResult({"written": False, "reason": "dry-run"})

    try:
        from vllm.sndr_core.model_configs.host import detect_and_save
        hc, path = detect_and_save(create_missing_caches=True)
    except Exception as e:
        _io.warn(f"host path auto-detect failed ({type(e).__name__}: {e})")
        return StepResult({"written": False, "reason": str(e)})

    _io.success(f"host paths → {path}")
    for k in sorted(hc.paths):
        _io.info(f"  {k}: {hc.paths[k]}")
    return StepResult({"written": True, "path": str(path),
                       "n_paths": len(hc.paths)})


# ─── Generate launch script ──────────────────────────────────────────────


def step_generate_launch(
    opts: argparse.Namespace,
    hw: dict[str, Any],
    workload: dict[str, Any],
    clone_info: dict[str, Any],
) -> StepResult:
    """Pick the best matching preset for (gpu_class, n_gpus, workload)
    and render its launch script via `cfg.to_launch_script(host_paths)`.
    """
    _io.step(10, 11, "Generate launch script")

    if not hw["gpu_class_hint"] or not hw["n_gpus"]:
        _io.warn("no GPU detected — skipping launch script generation")
        _io.info("Pick a preset manually: sndr launch --dry-run <key>")
        return StepResult({"path": None, "reason": "no_gpu"})

    cfg, key = _match_preset(hw["gpu_class_hint"], hw["n_gpus"],
                             workload["workload"])
    if cfg is None:
        _io.warn(f"no preset matches ({hw['gpu_class_hint']} × "
                 f"{hw['n_gpus']} × {workload['workload']})")
        _io.info("List configs: sndr launch  (interactive picker)")
        return StepResult({"path": None, "reason": "no_match"})

    home = Path(clone_info["home"])
    out_dir = home / "launch"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_gpu = hw["gpu_class_hint"].replace(" ", "_")
    out = (out_dir / f"start_{safe_gpu}_{hw['n_gpus']}x_"
                     f"{workload['workload']}.sh")

    try:
        from vllm.sndr_core.model_configs.host import load_host_config
        host_paths = dict(load_host_config().paths) if home else None
    except Exception:
        host_paths = None

    script = cfg.to_launch_script(host_paths=host_paths)
    out.write_text(script, encoding="utf-8")
    out.chmod(0o755)
    _io.success(f"wrote launch script: {out} (preset: {key})")
    return StepResult({"path": str(out), "preset": key})


def _match_preset(
    gpu_class: str, n_gpus: int, workload: str,
) -> tuple[Optional[Any], Optional[str]]:
    """Heuristic preset match. Returns (cfg, key) or (None, None).

    Strategy: walk all configs, filter by `hardware.n_gpus == n_gpus`
    and `hardware.gpu_match_keys` membership, prefer ones that mention
    `workload` in title/notes; else pick the first matching.

    B3 fix (UNIFIED_CONFIG plan 2026-05-09): the schema field is
    `gpu_match_keys: list[str]` (e.g. `['rtx a5000']`); the previous
    code looked for non-existent `gpu_class` / `gpu_name` attributes
    and silently matched nothing on every detected GPU.
    """
    try:
        from vllm.sndr_core.model_configs.registry import get, list_keys
    except ImportError:
        return (None, None)

    needle = (gpu_class or "").lower().strip()
    candidates: list[tuple[str, Any, int]] = []
    for k in list_keys():
        cfg = get(k)
        if cfg is None:
            continue
        hw = getattr(cfg, "hardware", None)
        if hw is None:
            continue
        if getattr(hw, "n_gpus", None) != n_gpus:
            continue
        keys = getattr(hw, "gpu_match_keys", None) or []
        if not _gpu_keys_match(needle, keys):
            continue
        # Workload preference score
        text = f"{getattr(cfg, 'title', '')} {' '.join(getattr(cfg, 'notes', []) or [])}".lower()
        score = 1 if workload.replace("_", " ") in text else 0
        candidates.append((k, cfg, score))

    if not candidates:
        return (None, None)
    candidates.sort(key=lambda t: -t[2])
    return (candidates[0][1], candidates[0][0])


def _gpu_keys_match(needle: str, keys: list[str]) -> bool:
    """True if any of `keys` matches `needle` either way (substring).

    Both directions: a config key 'rtx a5000' matches a detected
    'NVIDIA RTX A5000', and a config key 'rtx pro 6000 blackwell'
    matches a detected 'rtx pro 6000'. Empty needle never matches.
    """
    if not needle:
        return False
    for key in keys:
        kl = (key or "").lower().strip()
        if not kl:
            continue
        if kl in needle or needle in kl:
            return True
    return False


# ─── Smoke test ──────────────────────────────────────────────────────────


# P1-8 fail-class taxonomy (audit 2026-05-08).
# The previous smoke conflated:
#   (a) wiring regressions ("name X is not defined")
#   (b) anchor mismatches (text-patch failed to find target)
#   (c) missing-runtime imports (no torch / no vllm / no triton)
# all under "failed". Operators on Mac dev rigs got false positives
# because every torch-touching patch errored on import. Production
# servers got muddled diagnostics because real wiring bugs hid behind
# environment-related noise.
#
# The new taxonomy classifies each failure into one of these buckets
# so the installer can fail-closed on real bugs while skipping clean
# on environment gaps.
_RUNTIME_GAP_TOKENS = ("torch", "triton", "flashinfer", "vllm install root")
_ANCHOR_TOKENS = ("anchor", "marker not found", "drift")
_WIRING_TOKENS = ("name", "is not defined", "attributeerror", "cannot import name")


def _classify_failure(reason: str) -> str:
    """Return one of: 'runtime_gap', 'anchor_drift', 'wiring_bug', 'unknown'.

    Mutually exclusive — first match wins. The order matters: runtime
    gap is checked first because a missing torch can SURFACE as
    `cannot import name X from torch.nn` which would otherwise look
    like a wiring bug.
    """
    r = reason.lower()
    for tok in _RUNTIME_GAP_TOKENS:
        if tok in r:
            return "runtime_gap"
    for tok in _ANCHOR_TOKENS:
        if tok in r:
            return "anchor_drift"
    for tok in _WIRING_TOKENS:
        if tok in r:
            return "wiring_bug"
    return "unknown"


def step_smoke_test(opts: argparse.Namespace) -> StepResult:
    """Run apply_all in dry-run mode + classify failures.

    P1-8 fix (audit 2026-05-08): split failures into buckets so
    runtime gaps (Mac/CI without torch) don't trip the same wire as
    real wiring regressions. Only `wiring_bug` and `unknown` count as
    install-blocking; `runtime_gap` is downgraded to a structured
    skip with a clear message.
    """
    if opts.no_verify:
        _io.warn("skipping smoke test (--no-verify)")
        return StepResult({"ran": False})

    _io.step(11, 11, "Smoke test (dispatcher dry-run)")
    try:
        from vllm.sndr_core.apply import run
        stats = run(verbose=False, apply=False)
    except Exception as e:
        _io.warn(f"smoke test crashed: {type(e).__name__}: {e}")
        return StepResult({"ran": True, "ok": False,
                           "fail_class": "smoke_crash",
                           "exception": repr(e)})

    _io.success(
        f"applied={stats.applied_count}, skipped={stats.skipped_count},"
        f" failed={stats.failed_count}"
    )

    # Classify each failure into the right bucket.
    buckets: dict[str, list[Any]] = {
        "runtime_gap": [],
        "anchor_drift": [],
        "wiring_bug": [],
        "unknown": [],
    }
    for r in stats.failed:
        buckets[_classify_failure(r.reason)].append(r)

    blocking = buckets["wiring_bug"] + buckets["unknown"]
    runtime_gap_n = len(buckets["runtime_gap"])
    anchor_drift_n = len(buckets["anchor_drift"])

    # Surface buckets to the operator so they can act.
    if runtime_gap_n:
        _io.info(
            f"  ℹ {runtime_gap_n} patches reported missing runtime "
            "(torch / triton / vllm) — expected on hosts without GPU "
            "stack. Counted as SKIP, not failure."
        )
    if anchor_drift_n:
        _io.warn(
            f"  ⚠ {anchor_drift_n} patches reported anchor drift — "
            "vllm refactored the target region. Run the operator-facing "
            "upstream-drift detector script for detail (see docs/INSTALL.md)."
        )
        for r in buckets["anchor_drift"][:3]:
            _io.info(f"    - {r.name}: {r.reason[:120]}")
    if blocking:
        _io.error(
            f"  ✗ {len(blocking)} patches failed with WIRING / UNKNOWN "
            "errors — these are real bugs, not environment gaps:"
        )
        for r in blocking[:5]:
            _io.info(f"    - {r.name}: {r.reason[:120]}")

    ok = len(blocking) == 0  # runtime_gap + anchor_drift do NOT block install
    return StepResult({
        "ran": True,
        "ok": ok,
        "fail_classes": {
            "runtime_gap": runtime_gap_n,
            "anchor_drift": anchor_drift_n,
            "wiring_bug": len(buckets["wiring_bug"]),
            "unknown": len(buckets["unknown"]),
        },
        "stats": stats.summary(),
    })


# ─── Next steps banner ───────────────────────────────────────────────────


def print_next_steps(summary: dict[str, dict[str, Any]],
                     opts: argparse.Namespace) -> None:
    """rustup-style summary at the end of a successful install."""
    print()
    _io.banner("Genesis installed",
               f"home={summary['preflight']['sndr_home']}")
    print()
    if "clone" in summary:
        print(f"  Pin:       {summary['clone'].get('head', '?')}")
    print(f"  Workload:  {summary.get('workload', {}).get('workload', '?')}")
    if summary.get("plugin", {}).get("installed"):
        print("  Plugin:    installed (auto-loads in vllm serve)")
    else:
        print("  Plugin:    skipped")
    launch = summary.get("launch", {}).get("path")
    if launch:
        print(f"  Launch:    {launch}")

    print("\nNext:")
    if opts.bare_metal:
        print("  Bare-metal mode (Proxmox VE caveat or --bare-metal):")
        print(f"      {sys.executable} -m pip install --user vllm==0.20.1")
        print("      sndr verify")
        print("      vllm serve <model> --tensor-parallel-size <N> ...")
    elif launch:
        print(f"      bash {launch}        # docker-based launch")
    else:
        print("      sndr launch            # interactive picker")

    print("\nUseful commands:")
    print("  sndr doctor             # full system diagnostic")
    print("  sndr verify             # re-run smoke test")
    print("  sndr launch <preset>    # bring up vllm")
    print()


# ─── Uninstall ───────────────────────────────────────────────────────────


def run_uninstall(opts: argparse.Namespace) -> int:
    """Remove plugin install + delete legacy `vllm._genesis` symlink
    if present. Source tree at SNDR_HOME left in place — operator can
    `rm -rf` manually."""
    _io.banner("Uninstalling Genesis")

    # Plugin via pip
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "show", "genesis-vllm-plugin"],
            capture_output=True, text=True, check=False,
        )
        if r.returncode == 0:
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", "-q",
                 "genesis-vllm-plugin"],
                check=False,
            )
            _io.success("uninstalled genesis-vllm-plugin")
        else:
            _io.info("genesis-vllm-plugin was not installed via pip")
    except Exception as e:
        _io.warn(f"pip uninstall raised {type(e).__name__}: {e}")

    # Legacy `vllm/_genesis` symlink (v7.x; harmless on v11+ where it's gone)
    try:
        import vllm
        vllm_file = getattr(vllm, "__file__", None)
        if vllm_file:
            vllm_dir = Path(vllm_file).parent
            legacy = vllm_dir / "_genesis"
            if legacy.is_symlink():
                legacy.unlink()
                _io.success(f"removed legacy symlink {legacy}")
    except ImportError:
        pass

    home = Path(opts.home or os.environ.get("SNDR_HOME") or _DEFAULT_HOME)
    _io.warn(f"source tree at {home} left in place")
    _io.info(f"to fully remove:  rm -rf {home}")
    _io.warn("text-patches in vllm/ install were NOT reverted — "
             "run `pip install --force-reinstall vllm` to get clean upstream")
    return 0


# ─── Argparse + top-level dispatch ───────────────────────────────────────


def add_argparser(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "install",
        help="Bootstrap Genesis from a fresh machine (S-05 canonical wizard).",
        description="`sndr install` — bootstrap + setup wizard. "
                    "Replaces the legacy install.sh (now a thin shim).",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Report-only mode: no clone, no pip, no host.yaml.")
    p.add_argument("-y", "--non-interactive", action="store_true",
                   help="Accept all defaults (CI-friendly).")
    p.add_argument("--workload", default=None,
                   choices=list(WORKLOADS),
                   help="Genesis workload optimization target.")
    p.add_argument("--pin", default="stable",
                   help="`stable` (latest tag), `dev` (branch tip), or any "
                        "git ref. Default: stable.")
    p.add_argument("--repo", default=_DEFAULT_REPO,
                   help="Genesis git remote (default: official upstream).")
    p.add_argument("--home", default=None,
                   help="Where to clone Genesis. Default: $SNDR_HOME or "
                        "~/.sndr.")
    p.add_argument("--bare-metal", action="store_true",
                   help="Skip docker hints; emit native venv next-steps.")
    p.add_argument("--no-plugin", action="store_true",
                   help="Skip pip install of genesis-vllm-plugin.")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip the post-install smoke test.")
    p.add_argument("--allow-smoke-fail", action="store_true",
                   help="Print success banner + exit 0 even when the smoke "
                        "test reports failed patches. P0-7 escape hatch — "
                        "use only when you understand which patches "
                        "intentionally fail in the current env.")
    p.add_argument("--system", action="store_true",
                   help="Use system pip (default: --user).")
    p.add_argument("--uninstall", action="store_true",
                   help="Remove plugin + legacy symlink. Source left in place.")
    p.add_argument("--pro", action="store_true",
                   help="Also wire vllm.sndr_engine commercial package.")
    p.add_argument("--license-key", default="",
                   help="Commercial license key for --pro tier.")
    # C5 (UNIFIED_CONFIG plan 2026-05-09): direct-config preparation mode.
    # Bypasses workload/hardware heuristics — pulls hardware/workload/pin
    # from the named preset and runs only the prepare-side steps
    # (clone, plugin, host_paths, generate_launch). Operator already
    # knows which preset they want; --prepare hands them a ready-to-go
    # launcher script without the wizard prompts.
    p.add_argument("--config", default=None,
                   help="C5: model_config preset key (e.g. a5000-2x-35b-prod). "
                        "When set, --prepare bypasses workload heuristics.")
    p.add_argument("--prepare", action="store_true",
                   help="C5: prepare-only mode. Requires --config <key>. "
                        "Skips smoke test + interactive prompts; emits a "
                        "launcher script for the named preset.")
    p.set_defaults(func=run_install)


def run_install_prepare(opts: argparse.Namespace, cfg_key: str) -> int:
    """C5 (UNIFIED_CONFIG plan 2026-05-09): direct-config preparation flow.

    Bypasses the wizard's hardware/workload heuristics. Pulls
    declared metadata from the named preset and runs only the
    prepare-side steps:
      1. preflight (host info)
      2. resolve clone target (--home / SNDR_HOME)
      3. clone or update repo at the preset's pinned vllm/genesis revs
      4. install plugin (unless --no-plugin)
      5. detect host paths
      6. render the preset's launch script via to_launch_script

    Skips: detect_hardware (use cfg.hardware), pick_workload (use
    cfg.workload_tag), resolve_pin (use cfg.vllm_pin_required),
    smoke_test (operator runs `sndr launch <key>` separately).

    Exit codes: 0 on success, 2 on bad inputs, 1 on subprocess fail.
    """
    from vllm.sndr_core.model_configs.registry import get
    cfg = get(cfg_key)
    if cfg is None:
        _io.error(f"unknown preset key {cfg_key!r}")
        try:
            from vllm.sndr_core.model_configs.registry import list_keys
            _io.info(f"available: {', '.join(sorted(list_keys()))}")
        except Exception:
            pass
        return 2

    _io.banner(
        "SNDR Core Installer — prepare mode (C5)",
        f"preset={cfg_key} pin={cfg.vllm_pin_required or '_unset_'} "
        f"workload={cfg.workload_tag or '_unset_'}",
    )
    if opts.dry_run:
        _io.warn("DRY-RUN — no clone / no pip / no host.yaml writes")

    # Pull defaults from the preset onto opts (downstream steps read these)
    if cfg.workload_tag:
        opts.workload = cfg.workload_tag
    if cfg.vllm_pin_required:
        opts.pin = cfg.vllm_pin_required

    summary: dict[str, dict[str, Any]] = {}
    try:
        summary["preflight"] = step_preflight(opts).data
        # Skip detect_hardware/pick_workload/resolve_pin in prepare mode
        summary["pin"] = {
            "pin": cfg.vllm_pin_required or "stable",
            "kind": "from_config",
        }
        summary["clone"] = step_clone_or_update(
            opts, summary["preflight"], summary["pin"]).data
        summary["plugin"] = step_install_plugin(opts, summary["clone"]).data
        summary["host_paths"] = step_detect_host_paths(
            opts, summary["clone"]).data
        # Generate launch directly from the preset
        summary["launch"] = _emit_preset_launch_script(
            opts, cfg, summary["clone"], summary["host_paths"],
        )
    except SystemExit:
        raise
    except subprocess.CalledProcessError as e:
        _io.error(f"subprocess {e.cmd!r} failed (exit {e.returncode})")
        return 1
    except Exception as e:
        _io.error(f"prepare crashed: {type(e).__name__}: {e}")
        return 1

    _io.success(
        f"Preset {cfg_key!r} prepared. Launch with:\n"
        f"  sndr launch {cfg_key}"
    )
    return 0


def _emit_preset_launch_script(
    opts: argparse.Namespace,
    cfg: Any,
    clone: dict,
    host_paths: dict,
) -> dict:
    """Render `cfg.to_launch_script()` into scripts/launch/start_<key>.sh.

    Honors --dry-run (skips writing) but always returns the path it
    WOULD have written.
    """
    from pathlib import Path
    home = clone.get("home")
    out_dir = Path(home) / "scripts" / "launch" if home else Path.cwd() / "scripts" / "launch"
    target = out_dir / f"start_{cfg.key}.sh"
    if opts.dry_run:
        _io.info(f"[dry-run] would write launch script: {target}")
        return {"path": str(target), "wrote": False}
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        body = cfg.to_launch_script(
            host_paths=host_paths.get("paths") if isinstance(host_paths, dict)
            else None,
            strict_mounts=False,  # operators may render before host.yaml is set
        )
    except Exception as e:
        _io.error(f"to_launch_script failed: {e}")
        return {"path": str(target), "wrote": False, "error": str(e)}
    target.write_text(body)
    target.chmod(0o755)
    _io.success(f"wrote launch script: {target}")
    return {"path": str(target), "wrote": True}


def run_install(opts: argparse.Namespace) -> int:
    """Top-level orchestrator."""
    if opts.uninstall:
        return run_uninstall(opts)

    # C5 (UNIFIED_CONFIG plan 2026-05-09): --prepare requires --config
    if getattr(opts, "prepare", False):
        cfg_key = getattr(opts, "config", None)
        if not cfg_key:
            _io.error("--prepare requires --config <preset-key>")
            return 2
        return run_install_prepare(opts, cfg_key)

    _io.banner("SNDR Core Installer",
               f"channel={opts.pin} workload={opts.workload or 'pick'}")
    if opts.dry_run:
        _io.warn("DRY-RUN — no clone / no pip / no host.yaml writes")

    summary: dict[str, dict[str, Any]] = {}
    try:
        summary["preflight"] = step_preflight(opts).data
        summary["hardware"] = step_detect_hardware(opts).data
        summary["vllm"] = step_detect_vllm(opts).data
        summary["caveat"] = step_runtime_caveat(opts).data
        summary["workload"] = step_pick_workload(opts).data
        summary["pin"] = step_resolve_pin(opts).data
        summary["clone"] = step_clone_or_update(
            opts, summary["preflight"], summary["pin"]).data
        summary["plugin"] = step_install_plugin(opts, summary["clone"]).data
        summary["host_paths"] = step_detect_host_paths(
            opts, summary["clone"]).data
        summary["launch"] = step_generate_launch(
            opts, summary["hardware"], summary["workload"],
            summary["clone"]).data
        summary["smoke"] = step_smoke_test(opts).data
    except SystemExit:
        raise
    except subprocess.CalledProcessError as e:
        _io.error(f"subprocess {e.cmd!r} failed (exit {e.returncode})")
        if e.stderr:
            _io.info(e.stderr.strip()[:600])
        return 1
    except Exception as e:
        _io.error(f"installer crashed: {type(e).__name__}: {e}")
        return 1

    # P0-7 fix (audit 2026-05-08): exit non-zero when smoke detected
    # failed patches. Previously the banner said "Genesis installed"
    # regardless. Operators reading exit codes (CI, install.sh) need
    # the failure surfaced. `--no-verify` and `--allow-smoke-fail`
    # are the explicit opt-outs.
    smoke = summary.get("smoke", {})
    smoke_ok = smoke.get("ok", True) if smoke.get("ran") else True
    if not smoke_ok and not getattr(opts, "allow_smoke_fail", False):
        _io.error(
            "smoke test reported failed patches — install incomplete. "
            "Inspect with `sndr doctor`. Pass --allow-smoke-fail to "
            "ship anyway, or --no-verify to skip the check entirely."
        )
        return 2

    print_next_steps(summary, opts)
    return 0


__all__ = ["add_argparser", "run_install", "run_uninstall"]
