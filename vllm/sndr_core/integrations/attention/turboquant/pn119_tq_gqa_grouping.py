# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch PN119 — TurboQuant k8v4 GQA head grouping kernel.

Backport of [vllm#40792](https://github.com/vllm-project/vllm/pull/40792)
by `hoseung2` (OPEN at the time of backport).

================================================================
WHAT THIS PATCH DOES
================================================================

Adds the GQA-grouped variant of TurboQuant decode stage-1 kernel
``_tq_grouped_decode_stage1`` (~195 lines of new Triton code) and
updates the dispatch in ``triton_turboquant_decode_attention`` to
select the grouped kernel when GQA is active (k8v4 FP8 keys path).

The upstream PR measured **+16.5% – 27.2% TPS** on A100 / H100 with
GQA-ratio ∈ {4, 8, 24}. Our 27B and 35B both run **GQA-ratio 8**
(num_q_heads=32, num_kv_heads=4) so the win should be near the high
end on Ampere SM 8.6 hardware.

The grouped kernel:
  - Loads K once per ``BLOCK_H`` query-head tile and shares it across
    that whole tile of q-heads (the q-heads of one GQA group all read
    the same K vectors).
  - Uses ``tl.dot`` instead of element-wise products → routes through
    tensor cores instead of CUDA cores → 4-8× FLOPS density.
  - Falls back to the legacy ``_tq_decode_stage1`` kernel for MSE-tier
    presets that store quantized K but not FP8 K (k8v4 path only).

================================================================
IMPLEMENTATION APPROACH
================================================================

The kernel diff is +201 / -8 lines in
``vllm/v1/attention/ops/triton_turboquant_decode.py``, split across
two hunks: insertion of the new kernel after ``_tq_decode_stage1``,
and dispatch refactor in ``triton_turboquant_decode_attention``.

Both hunks are too large for inline anchor strings to be stable. We
bundle the upstream diff (``pn119_kernel.diff`` sibling file) and
apply it via ``subprocess`` (``patch -p4`` with filename rewriting).
A pre-patch md5 guard prevents application against drifted code; if
the file has changed in upstream relative to our cached diff, we
self-skip (safe — no partial application).

================================================================
SAFETY MODEL
================================================================

- **md5 pre-patch guard**: if current file md5 != expected pre-patch
  md5, the diff was authored against different code → skip.
- **Idempotency**: a marker line is injected at file head right after
  apply. On subsequent boots we check the marker before attempting
  re-patch.
- **Drift retreat**: when upstream merges (or rewrites) the kernel,
  our md5 guard catches it and PN119 self-retires.
- **No fallback path needed**: PN119 is additive on top of an
  unchanged dispatch entry point. If the patch fails to apply,
  vLLM continues with the original scalar kernel.

================================================================
HW GATE
================================================================

This patch is *active* on Ampere (SM 8.x) and Hopper (SM 9.x). It is
*not* expected to crash on newer HW, but the win was only measured
on A100/H100. Operators may disable with ``GENESIS_DISABLE_PN119=1``.

================================================================

Author: Genesis backport, original by hoseung2.
"""
from __future__ import annotations

import hashlib
import logging
import subprocess
from pathlib import Path

from vllm.sndr_core.detection.guards import resolve_vllm_file, vllm_install_root

log = logging.getLogger("genesis.wiring.pn119_tq_gqa_grouping")

GENESIS_PN119_MARKER = (
    "Genesis PN119 TurboQuant k8v4 GQA head grouping (backport: vllm#40792)"
)
GENESIS_PN119_MARKER_LINE = f"# {GENESIS_PN119_MARKER}\n"

# Expected md5 of the pre-patch ``triton_turboquant_decode.py`` against
# which the bundled ``pn119_kernel.diff`` was authored. Computed against
# vllm 0.20.2rc1.dev338+gbf0d2dc6d (our current PROD pin). If a future
# pin bump changes the file we'll see a mismatch and self-skip cleanly.
PN119_PRE_PATCH_MD5 = "e93d6f9eb591e0b68a50b0fc2eb689c3"

# Path to the bundled diff (sibling of this module).
PN119_DIFF_PATH = Path(__file__).parent / "pn119_kernel.diff"


def _target_path() -> Path | None:
    p = resolve_vllm_file("v1/attention/ops/triton_turboquant_decode.py")
    if p is None:
        return None
    return Path(p)


def _file_md5(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def apply() -> tuple[str, str]:
    """Apply PN119 — TurboQuant k8v4 GQA head grouping kernel."""
    from vllm.sndr_core.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN119")
    log_decision("PN119", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    target = _target_path()
    if target is None or not target.is_file():
        return "skipped", "triton_turboquant_decode.py not found"

    if not PN119_DIFF_PATH.is_file():
        return "skipped", f"bundled diff missing: {PN119_DIFF_PATH}"

    with open(target) as f:
        content = f.read()
    if GENESIS_PN119_MARKER in content:
        log.info("[PN119] marker present — skip (idempotent)")
        return "applied", "idempotent (marker present)"

    current_md5 = _file_md5(target)
    if current_md5 != PN119_PRE_PATCH_MD5:
        # File has drifted from the version the diff was authored
        # against. Either upstream merged the PR (or a competing PR),
        # or a pin bump changed unrelated code in the same file. Safe
        # default: skip and let the operator revisit on next pin bump.
        return (
            "skipped",
            f"drift: file md5 {current_md5} != expected "
            f"{PN119_PRE_PATCH_MD5}. Diff was authored against the dev338+"
            "gbf0d2dc6d pin; upstream may have merged #40792 or another "
            "PR has touched the same file. Genesis PN119 self-retires; "
            "regenerate diff + md5 on next pin bump.",
        )

    # Apply the diff via `patch`. Rewrite the unified-diff filename
    # headers so the tool finds the target (`patch` does not need a
    # subdirectory; we feed it the absolute path).
    try:
        with open(PN119_DIFF_PATH) as f:
            diff_text = f.read()
        # Strip git's a/ b/ prefixes — patch -p4 would expect a/vllm/v1/...
        # but we want it to operate directly on our absolute target.
        diff_text = diff_text.replace(
            "a/vllm/v1/attention/ops/triton_turboquant_decode.py",
            str(target),
        ).replace(
            "b/vllm/v1/attention/ops/triton_turboquant_decode.py",
            str(target),
        )

        # Dry-run first to validate.
        dry = subprocess.run(
            ["patch", "--dry-run", str(target)],
            input=diff_text,
            text=True,
            capture_output=True,
        )
        if dry.returncode != 0:
            return (
                "failed",
                f"patch dry-run failed: rc={dry.returncode} "
                f"stderr={dry.stderr[:200]} stdout={dry.stdout[:200]}",
            )

        # Apply for real.
        result = subprocess.run(
            ["patch", str(target)],
            input=diff_text,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            return (
                "failed",
                f"patch apply failed: rc={result.returncode} "
                f"stderr={result.stderr[:200]}",
            )

        # Inject marker at file head (line 2, after SPDX header line).
        with open(target) as f:
            patched = f.read()
        if GENESIS_PN119_MARKER in patched:
            return "applied", "PN119 applied (marker re-emitted by diff)"

        # Insert marker after the SPDX line or as a new line 1.
        lines = patched.split("\n", 1)
        if lines and lines[0].startswith("#"):
            new_content = lines[0] + "\n" + GENESIS_PN119_MARKER_LINE + (
                lines[1] if len(lines) > 1 else ""
            )
        else:
            new_content = GENESIS_PN119_MARKER_LINE + patched
        with open(target, "w") as f:
            f.write(new_content)

        return (
            "applied",
            "PN119 applied: TurboQuant k8v4 GQA-grouped decode stage1 "
            "kernel inserted; dispatch in triton_turboquant_decode_"
            "attention routed to grouped variant for GQA-ratio > 1. "
            "Upstream measured +16-27% TPS on A100/H100 GQA-{4,8,24}."
        )
    except Exception as e:
        return "failed", f"PN119 apply exception: {type(e).__name__}: {e}"


def is_applied() -> bool:
    if vllm_install_root() is None:
        return False
    target = _target_path()
    if target is None:
        return False
    try:
        with open(target) as f:
            return GENESIS_PN119_MARKER in f.read()
    except OSError:
        return False
