# SPDX-License-Identifier: Apache-2.0
"""Genesis doctor — single-command unified diagnostic.

Usage:
  python3 -m vllm.sndr_core.compat.doctor
  python3 -m vllm.sndr_core.compat.doctor --json
  python3 -m vllm.sndr_core.compat.doctor --explain PN14

Sections:
  1. Hardware            — GPUs, compute capabilities
  2. Software            — vllm / torch / triton / cuda / driver / python
  3. Model               — currently-loaded model + detected profile
  4. Patches that APPLY  — full registry walk, why each is on / off
  5. Lifecycle audit     — experimental / deprecated / research breakdown
  6. Recommendations     — actionable suggestions

Output is human-readable by default (color-free, copy-paste friendly)
or JSON via `--json` for machine consumers (CI, dashboards).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any

log = logging.getLogger("genesis.compat.doctor")


# ─── Sections ─────────────────────────────────────────────────────────────


def _section_hardware() -> dict[str, Any]:
    """Detect GPUs + compute capabilities. Wraps gpu_profile if available."""
    out: dict[str, Any] = {"gpus": [], "errors": []}
    try:
        import torch
        if not torch.cuda.is_available():
            out["errors"].append("torch.cuda.is_available() == False")
            return out
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            cc = torch.cuda.get_device_capability(i)
            props = torch.cuda.get_device_properties(i)
            out["gpus"].append({
                "index": i, "name": name,
                "compute_capability": f"{cc[0]}.{cc[1]}",
                "compute_capability_tuple": list(cc),
                "vram_total_gb": round(props.total_memory / 1e9, 2),
                "multi_processor_count": props.multi_processor_count,
            })
    except Exception as e:
        out["errors"].append(f"torch GPU probe: {e}")

    # Fold in our gpu_profile classification if available (datasheet bw/L2/etc)
    try:
        from vllm.sndr_core.compat.gpu_profile import (
            detect_gpu_class as _classify,
        )
        if out["gpus"]:
            try:
                out["gpu_class"] = _classify()
            except Exception as e:
                log.debug("gpu_profile.detect_gpu_class failed: %s", e,
                          exc_info=True)
    except Exception as e:
        log.debug("torch CUDA section probe failed: %s", e, exc_info=True)
    return out


def _section_environment() -> dict[str, Any]:
    """Detect host environment quirks that affect Genesis behavior.

    Currently surfaces:
      * WSL2 (Windows Subsystem for Linux 2) host — display overhead +
        DirectX shim eats VRAM, narrows borderline-OOM headroom; some
        kernels (notably P104 L2 persistence) misbehave under WSL paging.
      * Blackwell-class GPU on WSL2 — R6000 Pro 96GB on WSL2 is an
        atypical combo; Sander's planned upgrade target. Warn that NVFP4
        + PN38 FP8 paths assume bare-metal Linux/Windows, not WSL.
      * PCIe lane width (per-GPU) via nvidia-smi when available; warns
        when any GPU is wired below x16 (cuts P2P/host bandwidth and
        affects TQ continuation-prefill perf in TP=2 setups).

    All probes are best-effort and silently no-op when their data source
    is missing (Mac, container without nvidia-smi, etc.).
    """
    import os
    import shutil
    import subprocess

    out: dict[str, Any] = {
        "is_wsl": False,
        "wsl_version": None,
        "pcie_lanes": [],
        "errors": [],
    }

    # WSL2 detection — /proc/version contains "microsoft" or "WSL"
    proc_version_path = "/proc/version"
    if os.path.exists(proc_version_path):
        try:
            with open(proc_version_path, encoding="utf-8") as f:
                content = f.read().lower()
            if "microsoft" in content or "wsl" in content:
                out["is_wsl"] = True
                if "wsl2" in content:
                    out["wsl_version"] = "WSL2"
                elif "wsl" in content:
                    out["wsl_version"] = "WSL1"
        except Exception as e:
            out["errors"].append(f"/proc/version probe: {e}")

    # PCIe lane width per GPU — nvidia-smi --query-gpu=pcie.link.width.current
    if shutil.which("nvidia-smi"):
        try:
            res = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,name,pcie.link.gen.current,pcie.link.width.current,pcie.link.gen.max,pcie.link.width.max",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if res.returncode == 0:
                for line in res.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 6:
                        out["pcie_lanes"].append({
                            "index": int(parts[0]) if parts[0].isdigit() else parts[0],
                            "name": parts[1],
                            "gen_current": parts[2],
                            "width_current": parts[3],
                            "gen_max": parts[4],
                            "width_max": parts[5],
                        })
        except Exception as e:
            out["errors"].append(f"nvidia-smi pcie probe: {e}")

    return out


def _section_software() -> dict[str, Any]:
    """Versions of vllm / torch / triton / cuda / driver / python."""
    from vllm.sndr_core.compat.version_check import detect_versions
    p = detect_versions()
    return {
        "vllm": p.vllm, "vllm_commit": p.vllm_commit,
        "torch": p.torch, "triton": p.triton,
        "cuda_runtime": p.cuda_runtime, "nvidia_driver": p.nvidia_driver,
        "python": p.python,
        "compute_capabilities": [list(c) for c in p.compute_capabilities],
        "errors": list(p.errors),
    }


def _section_model_profile() -> dict[str, Any]:
    """Resolve the model profile via model_detect.get_model_profile."""
    out: dict[str, Any] = {"resolved": False, "errors": []}
    try:
        from vllm.sndr_core.compat.model_detect import get_model_profile
        profile = get_model_profile()
        out.update(profile)
    except Exception as e:
        out["errors"].append(f"model_detect: {e}")
    return out


def _section_patches() -> dict[str, Any]:
    """Walk every patch in PATCH_REGISTRY and decide apply/skip with reason."""
    from vllm.sndr_core.dispatcher import PATCH_REGISTRY, should_apply

    decisions = []
    apply_count = 0
    skip_count = 0
    for pid in PATCH_REGISTRY:
        try:
            decision, reason = should_apply(pid)
        except Exception as e:
            decision, reason = False, f"should_apply raised: {e}"
        meta = PATCH_REGISTRY.get(pid, {})
        decisions.append({
            "patch_id": pid,
            "title": meta.get("title", pid),
            "category": meta.get("category", "uncategorized"),
            "decision": "APPLY" if decision else "SKIP",
            "reason": reason,
            "env_flag": meta.get("env_flag", ""),
            "default_on": meta.get("default_on", False),
        })
        if decision:
            apply_count += 1
        else:
            skip_count += 1
    return {
        "total": len(decisions),
        "apply": apply_count,
        "skip": skip_count,
        "decisions": decisions,
    }


def _section_lifecycle() -> dict[str, Any]:
    """Run the lifecycle audit on the registry."""
    from vllm.sndr_core.dispatcher import PATCH_REGISTRY
    from vllm.sndr_core.compat.lifecycle import audit_registry

    entries = audit_registry(PATCH_REGISTRY)
    by_state: dict[str, list[dict]] = {}
    for e in entries:
        by_state.setdefault(e.state, []).append({
            "patch_id": e.patch_id, "note": e.note, "severity": e.severity,
        })
    return {
        "by_state": by_state,
        "total": len(entries),
    }


def _section_validator() -> dict[str, Any]:
    """Run the A3/D2 validator on the live registry + apply set."""
    try:
        from vllm.sndr_core.dispatcher import (
            validate_registry, validate_apply_plan, get_apply_matrix,
        )
        static = validate_registry()
        applied = {d["patch_id"] for d in get_apply_matrix() if d["applied"]}
        plan = validate_apply_plan(applied) if applied else []
        return {
            "static_issues": [
                {"severity": i.severity, "patch_id": i.patch_id, "message": i.message}
                for i in static
            ],
            "plan_issues": [
                {"severity": i.severity, "patch_id": i.patch_id, "message": i.message}
                for i in plan
            ],
        }
    except Exception as e:
        return {"error": str(e)}


def _section_recommendations(report: dict[str, Any]) -> list[str]:
    """Heuristic operator-actionable suggestions."""
    rec: list[str] = []

    # Validator errors → actionable
    val = report.get("validator", {})
    for issue in val.get("static_issues", []):
        rec.append(
            f"[{issue['severity']}] PATCH_REGISTRY: {issue['patch_id']} — "
            f"{issue['message']}"
        )
    for issue in val.get("plan_issues", []):
        rec.append(
            f"[{issue['severity']}] APPLY plan: {issue['patch_id']} — "
            f"{issue['message']}"
        )

    # Software errors → block deployment
    sw_errs = report.get("software", {}).get("errors", [])
    for e in sw_errs:
        rec.append(f"[ERROR] software detection: {e}")

    # Hardware errors → at least warn
    hw_errs = report.get("hardware", {}).get("errors", [])
    for e in hw_errs:
        rec.append(f"[WARN] hardware detection: {e}")

    # If model not resolved
    model = report.get("model_profile", {})
    if not model.get("resolved"):
        rec.append(
            "[INFO] no model loaded yet (run this command on a live vllm "
            "container for model-aware analysis)"
        )

    # Environment quirks (P1.11 / P2.11)
    env = report.get("environment", {})
    hw = report.get("hardware", {})
    if env.get("is_wsl"):
        rec.append(
            f"[WARN] {env.get('wsl_version','WSL')} detected — extra display "
            "overhead eats VRAM (~200-400 MiB on 24GB cards). On borderline "
            "configs (Cliff 2 single-card 24GB, club-3090 setups) expect "
            "tighter mem-utilization headroom. Verified noonghunna fix: "
            "set --gpu-memory-utilization to 0.85 (default 0.90 may trip)."
        )
        # Blackwell + WSL2 = atypical combo
        for g in hw.get("gpus", []):
            cc = g.get("compute_capability_tuple", [])
            if cc and len(cc) >= 1 and cc[0] >= 12:
                rec.append(
                    f"[WARN] Blackwell-class GPU '{g.get('name','?')}' "
                    f"detected on {env.get('wsl_version','WSL')}. PN38 NVFP4 "
                    "drafter path + Genesis sm_120 kernel autotune "
                    "assume bare-metal Linux/Windows. WSL paging may "
                    "reduce TPS by 5-10%; report results to "
                    "Genesis_internal_docs/wsl_blackwell_observations.md "
                    "if you have ground-truth bare-metal numbers."
                )
                break

    # PCIe lane warnings — flag any GPU wired below max width
    for lane in env.get("pcie_lanes", []):
        try:
            current = int(lane.get("width_current", "0").lstrip("x"))
            maximum = int(lane.get("width_max", "0").lstrip("x"))
        except (ValueError, AttributeError):
            continue
        if current > 0 and maximum > 0 and current < maximum:
            rec.append(
                f"[WARN] GPU {lane.get('index','?')} ({lane.get('name','?')}) "
                f"is wired x{current} but supports x{maximum} (gen {lane.get('gen_current','?')}/"
                f"{lane.get('gen_max','?')}). On TP=2 with TQ continuation-prefill "
                "this caps host↔device bandwidth and can cost 3-8% TPS. "
                "Check motherboard slot allocation (often x8/x8 vs x16/x16 BIOS "
                "setting) or PCIe riser cable integrity."
            )

    # Show at least one recommendation if everything is clean
    if not rec:
        rec.append("[OK] no issues detected. System is healthy.")

    return rec


# ─── Output formatters ────────────────────────────────────────────────────


def _format_text(report: dict[str, Any]) -> list[str]:
    L: list[str] = []
    L.append("=" * 72)
    L.append("Genesis doctor — system diagnostic")
    L.append("=" * 72)

    # Hardware
    L.append("")
    L.append("[1/6] Hardware")
    hw = report.get("hardware", {})
    if hw.get("gpus"):
        for g in hw["gpus"]:
            L.append(
                f"  GPU {g['index']}: {g['name']:<30} sm_{g['compute_capability'].replace('.', '')}  "
                f"VRAM {g['vram_total_gb']:.1f} GB"
            )
    else:
        L.append("  (no GPUs detected)")
    for e in hw.get("errors", []):
        L.append(f"  ⚠ {e}")

    # Environment quirks (WSL2, PCIe lanes) — only show when interesting
    env = report.get("environment", {})
    if env.get("is_wsl") or env.get("pcie_lanes"):
        L.append("")
        L.append("[1b] Host environment")
        if env.get("is_wsl"):
            L.append(f"  WSL:           {env.get('wsl_version','WSL')} (display "
                     "overhead +200-400 MiB; tighten gpu-mem-util)")
        for lane in env.get("pcie_lanes", []):
            L.append(
                f"  PCIe GPU {lane.get('index','?')}:    "
                f"gen {lane.get('gen_current','?')} x{lane.get('width_current','?')} "
                f"(max gen {lane.get('gen_max','?')} x{lane.get('width_max','?')})"
            )
        for e in env.get("errors", []):
            L.append(f"  ⚠ {e}")

    # Software
    L.append("")
    L.append("[2/6] Software")
    sw = report.get("software", {})
    L.append(f"  vllm:          {sw.get('vllm') or '(not installed)'}")
    if sw.get("vllm_commit"):
        L.append(f"    commit:      {sw['vllm_commit']}")
    L.append(f"  torch:         {sw.get('torch') or '(not installed)'}")
    L.append(f"  triton:        {sw.get('triton') or '(not installed)'}")
    L.append(f"  cuda runtime:  {sw.get('cuda_runtime') or '(none)'}")
    L.append(f"  nvidia driver: {sw.get('nvidia_driver') or '(unavailable)'}")
    L.append(f"  python:        {sw.get('python')}")

    # Model
    L.append("")
    L.append("[3/6] Model profile")
    mp = report.get("model_profile", {})
    if mp.get("resolved"):
        L.append(f"  model_class:   {mp.get('model_class', '?')}")
        L.append(f"  is_hybrid:     {mp.get('hybrid', mp.get('is_hybrid', '?'))}")
        L.append(f"  is_moe:        {mp.get('moe', mp.get('is_moe', '?'))}")
        L.append(f"  is_turboquant: {mp.get('turboquant', mp.get('is_turboquant', '?'))}")
        L.append(f"  quant_format:  {mp.get('quant_format', '?')}")
    else:
        L.append("  (model not loaded — run this on a live vllm container)")
        for e in mp.get("errors", []):
            L.append(f"  ⚠ {e}")

    # Patches
    L.append("")
    L.append("[4/6] Patch registry decisions")
    p = report.get("patches", {})
    L.append(f"  total: {p.get('total', 0)}, "
             f"APPLY: {p.get('apply', 0)}, SKIP: {p.get('skip', 0)}")
    apply_decisions = [d for d in p.get("decisions", []) if d["decision"] == "APPLY"]
    skip_decisions = [d for d in p.get("decisions", []) if d["decision"] == "SKIP"]
    if apply_decisions:
        L.append(f"  Applied ({len(apply_decisions)}):")
        for d in apply_decisions:
            L.append(f"    ✓ {d['patch_id']:<8} {d['title'][:55]}")
    if skip_decisions:
        # Hide the long list of opt-in skips by default; show count
        opt_in = [d for d in skip_decisions if "opt-in" in d.get("reason", "")]
        other_skips = [d for d in skip_decisions if "opt-in" not in d.get("reason", "")]
        if other_skips:
            L.append(f"  Skipped (non-opt-in, {len(other_skips)}):")
            for d in other_skips[:10]:  # cap at 10 to keep output readable
                L.append(f"    • {d['patch_id']:<8} {d['title'][:50]} — {d['reason'][:60]}")
            if len(other_skips) > 10:
                L.append(f"    ... and {len(other_skips) - 10} more")
        if opt_in:
            L.append(f"  Skipped (opt-in only, not engaged): {len(opt_in)}")

    # Lifecycle
    L.append("")
    L.append("[5/6] Lifecycle audit")
    lc = report.get("lifecycle", {})
    for state, ents in lc.get("by_state", {}).items():
        L.append(f"  {state}: {len(ents)}")
    if lc.get("total", 0) == 0:
        L.append("  (registry empty)")

    # Validator
    L.append("")
    L.append("[6/6] Validator")
    val = report.get("validator", {})
    if "error" in val:
        L.append(f"  ⚠ validator error: {val['error']}")
    else:
        si = val.get("static_issues", [])
        pi = val.get("plan_issues", [])
        if not si and not pi:
            L.append("  ✓ clean — no validator issues")
        else:
            for i in si:
                L.append(f"  [{i['severity']}] STATIC {i['patch_id']}: {i['message']}")
            for i in pi:
                L.append(f"  [{i['severity']}] PLAN   {i['patch_id']}: {i['message']}")

    # Recommendations
    L.append("")
    L.append("=" * 72)
    L.append("Recommendations")
    L.append("=" * 72)
    for r in report.get("recommendations", []):
        L.append(f"  {r}")
    L.append("=" * 72)
    return L


# ─── Driver ──────────────────────────────────────────────────────────────


def _section_preflight() -> dict[str, Any]:
    """PN60 + club#34 + club#43 doctor rules.

    Audit P2 fix 2026-05-05 (genesis_deep_cross_audit): PN60 was
    `default_on=True` in registry with credit "Doctor extension; runs at
    preflight" but `collect_report()` never called `run_all_preflight_checks()`.
    Operator running `genesis doctor` got no preflight signal.

    Now: doctor invokes preflight checks against the live container's logs
    (best-effort) and reports any WARN/ERROR findings under a dedicated
    `preflight` section. Operator-supplied `--quantization` and `--model`
    args are not available here (doctor takes no model context), so PN60
    quant validator only fires when the model_profile is resolved.
    """
    findings: list[dict[str, Any]] = []
    try:
        from vllm.sndr_core.compat.preflight_checks import (
            check_grammar_rejection_pattern,
            check_quant_arg,
            check_spec_decode_token_loop,
            fetch_container_logs,
        )
    except Exception as e:
        return {"status": "preflight module unavailable", "error": str(e),
                "findings": findings}

    # PN60 quant validator — only when we can locate config.json on disk.
    try:
        from vllm.sndr_core.detection.model_detect import get_model_profile
        profile = get_model_profile()
        model_dir = profile.get("model_dir") or profile.get("model_path")
        cli_quant = os.environ.get("GENESIS_DOCTOR_CLI_QUANT", None)
        if model_dir and cli_quant:
            r = check_quant_arg(cli_quant, model_dir)
            findings.append({"name": r.name, "severity": r.severity,
                             "message": r.message,
                             "remediation": r.remediation})
    except Exception:
        pass

    # club#34 + club#43 — log-driven, fire only if container logs available.
    log_text = fetch_container_logs(container_name="vllm-server-mtp-test")
    if log_text:
        for r in (check_grammar_rejection_pattern(log_text),
                  check_spec_decode_token_loop(log_text)):
            findings.append({"name": r.name, "severity": r.severity,
                             "message": r.message,
                             "remediation": r.remediation})
    return {"status": "ok", "findings": findings}


# ─── --full extended sections (T1.4 / audit §18.1) ───────────────────────


def _section_wsl() -> dict[str, Any]:
    """WSL2 detection + pin-memory + Docker GPU runtime probe.

    Mirrors the existing `_section_environment` WSL detection but
    explicitly extracts pin-memory and Docker runtime fields, plus
    formats recommendations specific to WSL2 quirks (PCIe pass-through,
    pin-memory paging). Useful for the audit closure §17.3 deliverable
    where doctor needs to surface WSL caveats up-front rather than as
    follow-on warnings.
    """
    import shutil
    import subprocess

    out: dict[str, Any] = {
        "is_wsl": False,
        "kernel": "",
        "distro": "",
        "pin_memory_ok": True,
        "docker_gpu_runtime": False,
        "recommendations": [],
        "errors": [],
    }
    # WSL detection via /proc/version
    try:
        with open("/proc/version", encoding="utf-8") as f:
            version = f.read()
        out["kernel"] = version.strip()
        if "microsoft" in version.lower() or "wsl" in version.lower():
            out["is_wsl"] = True
    except OSError:
        pass

    # Distro
    try:
        with open("/etc/os-release", encoding="utf-8") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    out["distro"] = line.split("=", 1)[1].strip().strip('"')
                    break
    except OSError:
        pass

    # Docker GPU runtime — best-effort, skips if docker missing
    if shutil.which("docker"):
        try:
            res = subprocess.run(
                ["docker", "info"],
                capture_output=True, text=True, timeout=5,
            )
            if res.returncode == 0 and "nvidia" in res.stdout.lower():
                out["docker_gpu_runtime"] = True
        except (subprocess.TimeoutExpired, OSError) as e:
            out["errors"].append(f"docker info probe failed: {e}")

    # Pin-memory probe — only matters under WSL2 + heavy concurrency.
    # Read /proc/meminfo for MemAvailable; if it's tiny relative to
    # vllm working set, pin-memory will fight with WSL's host shim.
    if out["is_wsl"]:
        try:
            with open("/proc/meminfo", encoding="utf-8") as f:
                meminfo = f.read()
            # crude heuristic: MemAvailable < 4 GiB on WSL is risky for vllm
            for line in meminfo.splitlines():
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    if kb < 4 * (1 << 20):  # 4 GiB in KB
                        out["pin_memory_ok"] = False
                        out["recommendations"].append(
                            "WSL2 + low MemAvailable: increase WSL "
                            "memory limit in .wslconfig "
                            "([wsl2] memory=24GB)"
                        )
        except (OSError, ValueError) as e:
            out["errors"].append(f"meminfo probe failed: {e}")

        # Standard WSL2 caveats applicable across the board
        out["recommendations"].append(
            "WSL2 detected — verify GPU pass-through with "
            "`nvidia-smi` and ensure CUDA toolkit ≥12.x for stable ops."
        )
        if not out["docker_gpu_runtime"]:
            out["recommendations"].append(
                "WSL2 + no Docker NVIDIA runtime: containers won't see "
                "GPUs. Install nvidia-container-toolkit on the WSL distro."
            )

    return out


def _section_image() -> dict[str, Any]:
    """Container image digest verification (T1.4 / audit §7.4 link).

    Reads `SNDR_DOCTOR_CONTAINER` env var (set by `--container` CLI
    flag forwarded into `os.environ`), inspects the image, and reports
    its digest + drift status. The "expected_digest" comes from
    `~/.sndr/host.yaml` if present (operators pin a known-good).

    Returns "skipped" when no container is in scope.
    """
    import shutil
    import subprocess

    out: dict[str, Any] = {
        "container": os.environ.get("SNDR_DOCTOR_CONTAINER"),
        "expected_digest": None,
        "actual_digest": None,
        "drift": False,
        "allowlist_status": "unknown",
        "status": "skipped",
        "errors": [],
    }
    if not out["container"]:
        out["status"] = "skipped (no --container)"
        return out
    if not shutil.which("docker"):
        out["status"] = "skipped (docker not on PATH)"
        return out

    # Resolve image from container
    try:
        image = subprocess.run(
            ["docker", "inspect", "-f", "{{.Image}}", out["container"]],
            capture_output=True, text=True, timeout=5,
        )
        if image.returncode != 0:
            out["status"] = f"docker inspect failed: {image.stderr.strip()}"
            return out
        actual = image.stdout.strip()
        out["actual_digest"] = actual
    except (subprocess.TimeoutExpired, OSError) as e:
        out["errors"].append(f"docker inspect failed: {e}")
        return out

    # Try to load expected digest from host.yaml
    try:
        from vllm.sndr_core.model_configs.host import load_host_config
        hc = load_host_config()
        if hc and hasattr(hc, "expected_image_digest"):
            out["expected_digest"] = getattr(hc, "expected_image_digest", None)
    except Exception:
        pass

    if out["expected_digest"] and out["actual_digest"]:
        out["drift"] = (out["expected_digest"] != out["actual_digest"])
        out["allowlist_status"] = "known_good" if not out["drift"] else "unknown"

    # Wave 4.1 (audit closure 2026-05-09): cross-check actual_digest
    # against KNOWN_GOOD_IMAGES (club-3090 #60). Even when the host.yaml
    # has no `expected_digest` pin, we can still classify the running
    # image as known-good / pin-match / unknown / historical.
    try:
        from vllm.sndr_core.compat.image_allowlist import (
            status_for as _allowlist_status,
            lookup_by_digest as _allowlist_lookup,
        )
        if out["actual_digest"]:
            # Try to resolve vllm pin from the running container
            actual_vllm_pin = ""
            try:
                vllm_probe = subprocess.run(
                    ["docker", "exec", out["container"], "python3", "-c",
                     "import vllm;print(vllm.__version__)"],
                    capture_output=True, text=True, timeout=5,
                )
                if vllm_probe.returncode == 0:
                    actual_vllm_pin = vllm_probe.stdout.strip()
            except (subprocess.TimeoutExpired, OSError):
                pass

            classification = _allowlist_status(
                digest=out["actual_digest"],
                vllm_pin=actual_vllm_pin,
            )
            out["allowlist_classification"] = classification
            entry = _allowlist_lookup(out["actual_digest"])
            if entry is not None:
                out["allowlist_match"] = {
                    "vllm_pin": entry.vllm_pin,
                    "validated_at": entry.validated_at,
                    "validated_on": entry.validated_on,
                    "bench_url": entry.bench_url,
                    "notes": entry.notes,
                }
            # If host.yaml had no pin but allowlist says known_good,
            # promote allowlist_status accordingly.
            if (out["allowlist_status"] == "unknown"
                    and classification == "known_good"):
                out["allowlist_status"] = "known_good"
    except Exception as e:
        out["errors"].append(f"image_allowlist check failed: {e}")

    out["status"] = "ok"
    return out


def _section_mounts() -> dict[str, Any]:
    """Mount writability scan — flags read-only mounts blocking patches.

    Implements club-3090 #47 (audit §17.4) detection: when the operator
    bind-mounts the SNDR Core tree read-only into a container, the
    text-patcher silently no-ops. This section scans `/proc/mounts` (or
    `docker inspect` if a container is in scope) for ro flags on paths
    inside vllm site-packages.
    """
    import os.path

    out: dict[str, Any] = {
        "mounts": [],
        "writability_violations": [],
        "errors": [],
    }
    # Best-effort: read /proc/mounts for the host. Container scan
    # would require docker inspect which we keep light here.
    try:
        with open("/proc/mounts", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 4:
                    continue
                src, dst, fstype, opts = parts[0], parts[1], parts[2], parts[3]
                # Only surface mounts that touch typical vllm paths.
                if any(needle in dst for needle in
                       ("vllm", "sndr", "genesis", "site-packages")):
                    is_ro = "ro" in opts.split(",")
                    writable = os.access(dst, os.W_OK) if not is_ro else False
                    out["mounts"].append({
                        "src": src, "dst": dst, "fstype": fstype,
                        "ro": is_ro, "writable": writable,
                    })
                    if is_ro:
                        out["writability_violations"].append(
                            f"{dst} mounted read-only — text-patcher "
                            "cannot apply edits. Use overlay mount or "
                            "rebind read-write."
                        )
    except OSError as e:
        out["errors"].append(f"/proc/mounts read failed: {e}")
    return out


def _section_license() -> dict[str, Any]:
    """License gate status — engine tier eligibility check.

    Surfaces the structured outcome of `check_engine_tier_eligible()`
    (used by dispatcher.should_apply for tier=engine patches), the
    trust-anchor source (zero-trust default vs operator-signed root),
    and whether legacy mode is active.
    """
    out: dict[str, Any] = {
        "trust_anchor": "zero",
        "license_present": False,
        "license_status": None,
        "legacy_mode_active": False,
        "engine_tier_eligible": False,
        "reason": "",
        "errors": [],
    }
    try:
        from vllm.sndr_core.license import (
            check_engine_tier_eligible,
            _engine_overlay_available,
        )
        result = check_engine_tier_eligible()
        out["engine_tier_eligible"] = bool(result.eligible)
        out["reason"] = str(getattr(result, "reason", ""))
        out["license_status"] = str(getattr(result, "status", "unknown"))
        out["license_present"] = bool(_engine_overlay_available())
    except Exception as e:
        out["errors"].append(f"license probe failed: {type(e).__name__}: {e}")

    # Legacy mode = user explicitly opted into community-only via
    # SNDR_ENABLE_TIER_OVERRIDE=1 (skips engine patches even if licensed).
    out["legacy_mode_active"] = bool(
        os.environ.get("SNDR_ENABLE_TIER_OVERRIDE")
        or os.environ.get("GENESIS_ENABLE_TIER_OVERRIDE")
    )

    # Trust anchor — currently always "zero" (no signed root configured).
    # Future: read from ~/.sndr/trust_anchor.json or a vendor cert.
    if os.path.isfile(os.path.expanduser("~/.sndr/trust_anchor.json")):
        out["trust_anchor"] = "real"

    return out


def _section_engine() -> dict[str, Any]:
    """Optional engine overlay status — what's installed beyond the core."""
    out: dict[str, Any] = {
        "engine_available": False,
        "overlay_packages": [],
        "version": None,
        "errors": [],
    }
    try:
        from vllm.sndr_core.license import _engine_overlay_available
        out["engine_available"] = bool(_engine_overlay_available())
    except Exception as e:
        out["errors"].append(f"overlay probe failed: {e}")

    if out["engine_available"]:
        try:
            from vllm.sndr_core.license import _engine_package_version
            out["version"] = _engine_package_version()
        except Exception as e:
            out["errors"].append(f"engine version probe failed: {e}")

        # Best-effort: walk vllm.sndr_engine.patches/* if present
        try:
            import importlib
            try:
                eng = importlib.import_module("vllm.sndr_engine")
                if hasattr(eng, "__path__"):
                    import pkgutil
                    out["overlay_packages"] = sorted(
                        m.name for m in pkgutil.iter_modules(eng.__path__)
                    )
            except ImportError:
                pass
        except Exception as e:
            out["errors"].append(f"overlay enumerate failed: {e}")
    return out


def _section_remote_capability() -> dict[str, Any]:
    """Remote/SSH support sanity probe.

    Used by `sndr doctor --remote <user>@host` planning. Surfaces
    whether the operator has SSH keys + can resolve a target host.
    Doesn't actually SSH anywhere — that's the launch path's job.
    """
    out: dict[str, Any] = {
        "ssh_keys_present": False,
        "ssh_agent_running": False,
        "can_resolve_remote_targets": False,
        "key_files": [],
        "errors": [],
    }
    home = os.path.expanduser("~")
    ssh_dir = os.path.join(home, ".ssh")
    if os.path.isdir(ssh_dir):
        for cand in ("id_rsa", "id_ed25519", "id_ecdsa"):
            p = os.path.join(ssh_dir, cand)
            if os.path.isfile(p):
                out["ssh_keys_present"] = True
                out["key_files"].append(p)

    # Agent socket
    if os.environ.get("SSH_AUTH_SOCK"):
        out["ssh_agent_running"] = True

    # Target resolution — only fires if --remote was passed via env
    target = os.environ.get("SNDR_DOCTOR_REMOTE")
    if target:
        # Don't actually probe DNS — keep this synchronous + safe
        out["target"] = target
        out["can_resolve_remote_targets"] = bool(out["ssh_keys_present"])
    return out


def collect_report(*, full: bool = False) -> dict[str, Any]:
    """Run all sections and return the unified report.

    `full=True` (T1.4) enables the 6 extended sections from audit §18.1:
    wsl, image, mounts, license, engine, remote. Default `False`
    preserves the back-compat 12-section shape for old tooling.
    """
    report: dict[str, Any] = {}
    report["hardware"] = _section_hardware()
    report["environment"] = _section_environment()
    report["software"] = _section_software()
    report["model_profile"] = _section_model_profile()
    report["patches"] = _section_patches()
    report["lifecycle"] = _section_lifecycle()
    report["validator"] = _section_validator()
    report["preflight"] = _section_preflight()

    if full:
        report["wsl"] = _section_wsl()
        report["image"] = _section_image()
        report["mounts"] = _section_mounts()
        report["license"] = _section_license()
        report["engine"] = _section_engine()
        report["remote"] = _section_remote_capability()

    report["recommendations"] = _section_recommendations(report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m vllm.sndr_core.compat.doctor",
        description="Genesis unified diagnostic — hardware + software + model "
                    "+ patches + validator + lifecycle.",
    )
    parser.add_argument("--json", action="store_true",
                        help="Output the full report as JSON (for CI / dashboards)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress section headers; print only critical issues")
    parser.add_argument("--full", action="store_true",
                        help="Enable 6 extended sections: wsl, image, mounts, "
                             "license, engine, remote (T1.4 / audit §18.1)")
    parser.add_argument("--container", default=None,
                        help="Container name to inspect for image digest "
                             "(passes through SNDR_DOCTOR_CONTAINER env)")
    parser.add_argument("--remote", default=None,
                        help="Remote target (user@host) for capability probe "
                             "(passes through SNDR_DOCTOR_REMOTE env)")
    parser.add_argument("--redact", action="store_true",
                        help="Mask IPs / hostnames / tokens before output")
    args = parser.parse_args(argv)

    # Forward CLI flags into env so the section probes (which already
    # read SNDR_DOCTOR_* env vars) pick them up. Avoids passing kwargs
    # through every layer.
    if args.container:
        os.environ["SNDR_DOCTOR_CONTAINER"] = args.container
    if args.remote:
        os.environ["SNDR_DOCTOR_REMOTE"] = args.remote

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    report = collect_report(full=args.full)

    if args.redact:
        # Apply the same redactor used by `sndr report bundle`.
        try:
            from vllm.sndr_core.runtime.redact import redact_dict
            report = redact_dict(report)
        except Exception as e:
            log.warning("redaction failed (%s) — emitting unredacted output", e)

    if args.json:
        # Convert any non-JSON-serializable types
        print(json.dumps(report, indent=2, default=str))
        return 0

    if args.quiet:
        # Only print recommendations
        for r in report.get("recommendations", []):
            print(r)
        # Exit non-zero if there are ERROR-level recommendations
        for r in report.get("recommendations", []):
            if r.startswith("[ERROR]"):
                return 1
        return 0

    for line in _format_text(report):
        print(line)
    # Exit non-zero if validator found errors
    val = report.get("validator", {})
    has_errors = any(
        i["severity"] == "ERROR"
        for i in (val.get("static_issues", []) + val.get("plan_issues", []))
    )
    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
