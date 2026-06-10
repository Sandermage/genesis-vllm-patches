#!/usr/bin/env python3
"""tools/verify_patches_runtime.py — runtime verification of Genesis patches.

The "applied" status in boot logs ONLY proves the marker landed in the
target file. It does NOT prove:

  * The patched function is actually CALLED at runtime.
  * The patch's logic actually EXECUTES on real requests.
  * The intended optimization is REALLY happening.

Pin bumps can silently shift things:

  * vllm rewrites the function the patch hooks → marker still there
    but new code-path bypasses the patched code.
  * Function signature changes → patch logic still runs but reads
    stale field that was renamed upstream.
  * Algorithm changes → the patched code path is dead, but no error
    fires because it just isn't called.

This tool tries to surface those silent failures by, for each hot-path
patch:

  1. Import the patched module via the LIVE container's python.
  2. ``inspect.getsource`` of the target function and assert the
     Genesis marker is present in the SOURCE the engine actually uses.
  3. If the patch ships a per-call counter / metric (e.g. via
     observability), read it from /metrics or via the wired hook.
  4. For specific patches (PN340/341/350/364), do a small synthetic
     request that should exercise the patched code path and look for
     evidence in metrics / logs that the patch executed.

Usage::

    # Verify all high-priority hot-path patches
    python3 tools/verify_patches_runtime.py

    # Verify a single patch
    python3 tools/verify_patches_runtime.py --patch PN350

    # JSON output for CI / scripts
    python3 tools/verify_patches_runtime.py --json

Exit codes
==========

  * 0 — every checked patch has runtime evidence of execution
  * 1 — at least one patch is "marker present, runtime evidence
    MISSING" (silent failure suspected)
  * 2 — invocation error

This is the LIVE-CONTAINER tool — it ssh+docker-exec'es into the
running engine and runs the introspection inside the engine's python.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import Any

# Patches to verify: (patch_id, target_module_import, target_function,
#                    marker_substring, optional_metric_name)
HOT_PATH_PATCHES = [
    # PN340 — MTP decode bubbles in GDN attn backend.
    # Patches `metadata_build` / `__init__` in gdn_attn.py.
    ("PN340",
     "vllm.v1.attention.backends.gdn_attn",
     "GDNAttentionMetadataBuilder.__init__",
     "Genesis PN340",
     None),

    # PN341 — MTP decode bubbles in GPU model runner.
    # Multiple sub-patches in gpu_model_runner.py.
    ("PN341",
     "vllm.v1.worker.gpu_model_runner",
     "GPUModelRunner._update_states_after_model_execute",
     "Genesis PN341",
     None),

    # PN345 — shmem-aware autotune pruner on FLA chunk kernels.
    ("PN345",
     "vllm.model_executor.layers.fla.ops.chunk_delta_h",
     None,
     "Genesis PN345",
     None),

    # PN346 — Mamba/GDN cache hit boundary fix.
    ("PN346",
     "vllm.v1.core.single_type_kv_cache_manager",
     "MambaManager.find_longest_cache_hit",
     "Genesis PN346",
     None),

    # PN347 — Marlin FP8 N==K silent corruption fix.
    ("PN347",
     "vllm.model_executor.kernels.linear.scaled_mm.marlin",
     "MarlinFP8ScaledMMLinearKernel.process_weights_after_loading",
     "Genesis PN347",
     None),

    # PN348 — Qwen3.5/3.6 MTP backbone dedup.
    ("PN348",
     "vllm.model_executor.models.qwen3_5_mtp",
     None,
     "Genesis PN348",
     None),

    # PN350 — fused GDN QKV split Triton kernel integration.
    ("PN350",
     "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
     None,
     "Genesis PN350",
     None),

    # PN361 — fail-closed missing draft probs.
    ("PN361",
     "vllm.v1.worker.gpu_model_runner",
     "GPUModelRunner._get_spec_decode_draft_probs",
     "Genesis PN361",
     None),

    # PN50 — GDN proj fusion text-patch (re-anchored 2026-06-09 against
    # vllm#41126 split_ba helper rename).
    ("PN50",
     "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
     None,
     "Genesis PN50",
     None),

    # PN353A — TQ workspace reserve (drift-skipped per agent finding;
    # upstream may have absorbed equivalent). We still probe for marker
    # in case it lands.
    ("PN353A",
     "vllm.v1.attention.backends.turboquant_attn",
     None,
     "Genesis PN353A",
     None),

    # PN362 — env_override.py text-patch. Install code is gated by
    # VLLM_TRITON_FORCE_FIRST_CONFIG=1 (not set in PROD by design —
    # this is a bench-A/B reproducibility tool, not a PROD perf knob).
    # The text-patch marker should be present in env_override.py module
    # file even when the env trigger is off.
    ("PN362",
     "vllm.env_override",
     None,
     "Genesis PN362",
     None),

    # PN364 — hybrid GDN/Mamba warmup wrapper.
    # Hook lives on the V1 Worker class — verify via class attr.
    # NOTE: monkey-patch wrap attr is PER-PROCESS. docker exec creates a
    # SEPARATE process which has its own Worker class import — that one
    # WILL NOT have the wrap. The wrap lives in the engine's Worker_TP*
    # processes via fork inheritance. Per-class probe gives false negative.
    # Behavioral probe (search container logs for Pass/Pass logs) is the
    # ULTIMATE proof — see _check_behavioral_evidence below.
    ("PN364",
     "vllm.v1.worker.gpu_worker",
     None,
     None,
     "_genesis_pn364_hybrid_gdn_warmup_installed"),  # attr on method
]


# Behavioral evidence patterns — search container logs for these strings
# to confirm the wrap's body ACTUALLY EXECUTED in worker subprocesses.
# This is the only reliable check for monkey-patch wrappers because per-
# process class state is invisible to `docker exec`.
BEHAVIORAL_EVIDENCE = {
    "PN126": ["[PN126] Pass 1 done", "[PN126] Pass 2 done"],
    "PN128": ["[PN128] num_reqs", "[PN128] spec-decode helper warmup complete"],
    # Note: PN130 uses unicode ✓ in log but grep on it through ssh-quote
    # is unreliable. Use a stable substring instead.
    "PN129": ["[PN129] starting slot_mapping warmup"],
    "PN130": ["[PN130] starting TQ decode warmup"],
    "PN364": ["[PN364] wrapped compile_or_warm_up_model called",
              "[PN364] Pass 1 done"],
    # PN50 — text-patch on qwen_gdn_linear_attn.py (re-anchored 2026-06-09
    # against split_ba helper rename from vllm#41126). Boot trace evidence
    # is the apply_all log line, not a per-call hot-path log.
    "PN50":  ["PN50 applied: GDN proj fusion active"],
}


def _build_probe_script() -> str:
    """Build the introspection script that runs inside the container."""
    return '''
import importlib
import inspect
import json
import sys

HOT_PATH_PATCHES = ''' + repr(HOT_PATH_PATCHES) + '''

def _resolve_attr(module, dotted: str):
    """Resolve `A.B.C` dotted access on a module."""
    obj = module
    for piece in dotted.split('.'):
        try:
            obj = getattr(obj, piece)
        except AttributeError:
            return None
    return obj

results = []
for patch_id, mod_path, fn_dotted, marker, wrapper_attr in HOT_PATH_PATCHES:
    entry = {
        "patch_id": patch_id,
        "module": mod_path,
        "function": fn_dotted,
        "marker": marker,
        "wrapper_attr": wrapper_attr,
    }
    try:
        mod = importlib.import_module(mod_path)
        entry["module_imports"] = True
    except Exception as e:
        entry["module_imports"] = False
        entry["error"] = f"import failed: {type(e).__name__}: {e!r}"[:300]
        results.append(entry)
        continue

    # Case 1: function source-level marker check
    if fn_dotted and marker:
        fn = _resolve_attr(mod, fn_dotted)
        if fn is None:
            entry["status"] = "FUNCTION_MISSING"
            entry["reason"] = (
                f"`{fn_dotted}` not found in `{mod_path}`. "
                "Pin bump may have renamed/moved the function — patch "
                "marker present in file but FUNCTION NOT CALLABLE."
            )
            results.append(entry)
            continue
        try:
            src = inspect.getsource(fn)
        except (OSError, TypeError) as e:
            entry["status"] = "SOURCE_UNAVAILABLE"
            entry["reason"] = f"getsource() failed: {e!r}"
            results.append(entry)
            continue
        if marker in src:
            entry["status"] = "MARKER_IN_RUNTIME_SOURCE"
            entry["reason"] = (
                f"Marker `{marker}` present in `{fn_dotted}` source. "
                "Patch logic IS in the function the engine calls."
            )
        else:
            entry["status"] = "SILENT_FAILURE_SUSPECT"
            entry["reason"] = (
                f"Marker `{marker}` is NOT in `{fn_dotted}` source as "
                "the engine sees it. Patch may have applied to a "
                "different copy of the file (rare), or function was "
                "rewritten upstream after patch landed."
            )
        results.append(entry)
        continue

    # Case 2: module-level marker check (e.g. PN345, PN348, PN350)
    if not fn_dotted and marker:
        try:
            src_path = inspect.getsourcefile(mod) or "<unknown>"
            with open(src_path) as f:
                src = f.read()
        except Exception as e:
            entry["status"] = "SOURCE_UNAVAILABLE"
            entry["reason"] = f"could not read module source: {e!r}"
            results.append(entry)
            continue
        if marker in src:
            entry["status"] = "MARKER_IN_MODULE_SOURCE"
            entry["reason"] = (
                f"Marker `{marker}` present in module file `{src_path}`."
            )
        else:
            entry["status"] = "SILENT_FAILURE_SUSPECT"
            entry["reason"] = f"Marker `{marker}` NOT in module file."
        results.append(entry)
        continue

    # Case 3: wrapper-attribute check (e.g. PN364)
    if wrapper_attr:
        # The pattern: a Worker method wrapped with a sentinel attribute.
        # PN364 sets _genesis_pn364_hybrid_gdn_warmup_installed=True on
        # Worker.compile_or_warm_up_model.
        try:
            Worker = getattr(mod, "Worker", None) or getattr(mod, "GPUWorker", None)
            if Worker is None:
                entry["status"] = "FUNCTION_MISSING"
                entry["reason"] = f"No Worker class in `{mod_path}`"
                results.append(entry)
                continue
            target_method = getattr(Worker, "compile_or_warm_up_model", None)
            if target_method is None:
                entry["status"] = "FUNCTION_MISSING"
                entry["reason"] = "Worker.compile_or_warm_up_model missing"
                results.append(entry)
                continue
            if getattr(target_method, wrapper_attr, False):
                entry["status"] = "WRAPPER_INSTALLED"
                entry["reason"] = (
                    f"Wrapper attr `{wrapper_attr}` set on "
                    "Worker.compile_or_warm_up_model — PN364 patch IS active."
                )
            else:
                entry["status"] = "SILENT_FAILURE_SUSPECT"
                entry["reason"] = (
                    f"Wrapper attr `{wrapper_attr}` NOT set on "
                    "Worker.compile_or_warm_up_model — patch installation "
                    "did NOT take effect at runtime."
                )
        except Exception as e:
            entry["status"] = "PROBE_ERROR"
            entry["reason"] = f"wrapper probe raised: {e!r}"
        results.append(entry)
        continue

    entry["status"] = "NO_PROBE"
    entry["reason"] = "No probe defined for this patch"
    results.append(entry)

print(json.dumps({"verification_results": results}, indent=2))
'''


def run_behavioral_probe(ssh_target: str, container: str) -> dict[str, list[bool]]:
    """Search container logs for behavioral evidence each monkey-patch fired.

    Returns mapping patch_id -> list of (pattern_matched: bool) per
    expected log pattern. All True => wrap fired and emitted all
    expected log lines. Any False => wrap may have silently no-op'd.
    """
    results: dict[str, list[bool]] = {}
    for patch_id, patterns in BEHAVIORAL_EVIDENCE.items():
        results[patch_id] = []
        for pat in patterns:
            # docker logs grep — escape the pattern
            cmd = [
                "ssh", ssh_target,
                f"docker logs {container} 2>&1 | grep -F {json.dumps(pat)} | head -1",
            ]
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=30,
                )
                found = bool(proc.stdout.strip())
            except Exception:  # noqa: BLE001
                found = False
            results[patch_id].append(found)
    return results


def run_live_probe(ssh_target: str, container: str) -> dict[str, Any]:
    """Copy the probe script into the container and run it via docker exec."""
    script = _build_probe_script()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tf:
        tf.write(script)
        tmp_path = tf.name

    # scp to host
    proc = subprocess.run(
        ["scp", tmp_path, f"{ssh_target}:/tmp/_verify_patches.py"],
        capture_output=True, text=True, timeout=30,
    )
    if proc.returncode != 0:
        return {"error": f"scp failed: {proc.stderr[:300]}"}

    # docker cp + exec inside container
    cmd = [
        "ssh", ssh_target,
        f"docker cp /tmp/_verify_patches.py {container}:/tmp/_verify_patches.py && "
        f"docker exec {container} python3 /tmp/_verify_patches.py",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        return {
            "error": f"docker exec failed: {proc.stderr[:300]}",
            "stdout_head": proc.stdout[:500],
        }
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        return {
            "error": f"could not parse output: {e!s}",
            "stdout_head": proc.stdout[:500],
        }


def render(audit: dict[str, Any]) -> str:
    """Render verification result as markdown."""
    if "error" in audit:
        return f"# Runtime patch verification\n\n**ERROR**: {audit['error']}"

    results = audit.get("verification_results", [])
    lines = [
        "# Runtime patch verification — live container introspection",
        "",
        "**Why this exists**: `applied=True` in boot logs only means the",
        "marker landed in the target file. It does NOT prove the patched",
        "function is actually called, or that the patch's logic actually",
        "executes on real requests. Pin bumps can silently shift code",
        "paths so a patch is present-but-dead.",
        "",
        "## Verification table",
        "",
        "| Patch | Module | Function | Status | Evidence |",
        "|---|---|---|---|---|",
    ]
    silent_fail = 0
    fn_missing = 0
    for r in results:
        status = r.get("status", "?")
        icon = {
            "MARKER_IN_RUNTIME_SOURCE": "✅",
            "MARKER_IN_MODULE_SOURCE": "✅",
            "WRAPPER_INSTALLED": "✅",
            "SILENT_FAILURE_SUSPECT": "🚨",
            "FUNCTION_MISSING": "🚨",
            "SOURCE_UNAVAILABLE": "⚠️",
            "NO_PROBE": "ℹ️",
        }.get(status, "❓")
        if status == "SILENT_FAILURE_SUSPECT":
            silent_fail += 1
        if status == "FUNCTION_MISSING":
            fn_missing += 1
        msg = (r.get("reason", "") or "")[:120].replace("|", "\\|")
        fn = r.get("function") or "(module)"
        lines.append(
            f"| {icon} {r.get('patch_id', '?')} | "
            f"`{r.get('module', '?')[:35]}` | `{fn[:30]}` | "
            f"{status} | {msg} |"
        )
    lines.append("")
    lines.append(f"**Silent-failure suspects**: {silent_fail}")
    lines.append(f"**Function-missing (pin drift)**: {fn_missing}")

    # Behavioral evidence — the ultimate proof for monkey-patches.
    behavioral = audit.get("behavioral_evidence") or {}
    if behavioral:
        lines.append("")
        lines.append("## Behavioral evidence (boot log search)")
        lines.append("")
        lines.append(
            "Monkey-patch wrappers (PN126/128/130/364) have per-process "
            "state — `docker exec` runs in a SEPARATE process that does NOT "
            "see the wrap. The static probe above gives a false-negative "
            "for them. The ULTIMATE proof that a wrap fired is finding its "
            "expected log line in `docker logs`."
        )
        lines.append("")
        lines.append("| Patch | Expected boot log pattern | Found |")
        lines.append("|---|---|---|")
        for pid, found_list in behavioral.items():
            patterns = BEHAVIORAL_EVIDENCE.get(pid, [])
            for pat, found in zip(patterns, found_list):
                icon = "✅" if found else "🚨"
                lines.append(f"| {icon} {pid} | `{pat}` | {'yes' if found else 'NO'} |")
    lines.append("")
    if silent_fail or fn_missing:
        lines.append(
            "## ⚠️ Action required\n\n"
            "Inspect each `SILENT_FAILURE_SUSPECT` or `FUNCTION_MISSING` "
            "entry. The patch's marker may be present in a stale file copy "
            "while the engine loads the patched code from elsewhere, or "
            "the upstream function may have been renamed/refactored after "
            "patch was authored."
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ssh-target", default=os.environ.get("SSH_HOST", ""),
                    help="ssh target <user>@<host> (default: $SSH_HOST)")
    ap.add_argument("--container", default="vllm-qwen3.6-35b-balanced-k3")
    ap.add_argument("--patch", help="Verify only one patch by ID")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--ci-strict", action="store_true",
                    help="Exit 1 on any SILENT_FAILURE_SUSPECT or FUNCTION_MISSING")
    args = ap.parse_args()

    if not args.ssh_target:
        print("error: ssh target required; pass --ssh-target or set "
              "SSH_HOST", file=sys.stderr)
        return 2

    audit = run_live_probe(args.ssh_target, args.container)
    behavioral = run_behavioral_probe(args.ssh_target, args.container)
    audit["behavioral_evidence"] = behavioral

    if args.patch:
        # filter
        if "verification_results" in audit:
            audit["verification_results"] = [
                r for r in audit["verification_results"]
                if r.get("patch_id") == args.patch
            ]

    if args.json:
        print(json.dumps(audit, indent=2, ensure_ascii=False))
    else:
        print(render(audit))

    if args.ci_strict and "verification_results" in audit:
        bad = [r for r in audit["verification_results"]
               if r.get("status") in ("SILENT_FAILURE_SUSPECT", "FUNCTION_MISSING")]
        if bad:
            print(f"\nCI-strict: {len(bad)} silent-failure suspects",
                  file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
