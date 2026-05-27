# SPDX-License-Identifier: Apache-2.0
"""N104 — dump worker thread stacks on EngineCore RPC timeout.

Diagnostic wiring for multiproc_executor.py.

When the EngineCore's `get_response()` hits a `TimeoutError` waiting for a
worker RPC to complete, we send SIGUSR2 to the worker processes before
re-raising. The workers have a `faulthandler` signal handler installed at
startup that dumps all thread stacks to stderr (captured by Docker logging).

The goal is to determine WHAT the workers are doing during the 20+ minute
hang: are they stuck in a CUDA kernel (synchronize), compiling Triton kernels,
or deadlocked at the Python level?

Two sub-patches:
  1. Add `import faulthandler` at top + register SIGUSR2 handler in worker_main()
  2. Add SIGUSR2 send in get_response() TimeoutError handler

Auto-applies (no opt-in env). Safe: only affects the error path + adds one
signal handler. No behavior change during normal operation.
"""
from __future__ import annotations

import logging

from vllm._genesis.guards import resolve_vllm_file
from vllm._genesis.wiring.text_patch import TextPatcher, TextPatch, TextPatchResult

log = logging.getLogger("genesis.wiring.dump_stacks_on_timeout")

GENESIS_MARKER = "Genesis N104 dump stacks on timeout v1"

# ─── Sub-patch 1: faulthandler import + SIGUSR2 handler in worker_main ────

IMPORT_OLD = "import traceback\n"
IMPORT_NEW = "import traceback\nimport faulthandler\n"

SIGUSR2_HANDLER_OLD = (
    "        # Either SIGTERM or SIGINT will terminate the worker\n"
    "        signal.signal(signal.SIGTERM, signal_handler)\n"
    "        signal.signal(signal.SIGINT, signal_handler)\n"
)

SIGUSR2_HANDLER_NEW = (
    "        # Either SIGTERM or SIGINT will terminate the worker\n"
    "        signal.signal(signal.SIGTERM, signal_handler)\n"
    "        signal.signal(signal.SIGINT, signal_handler)\n"
    "\n"
    "        # SIGUSR2 dumps all thread stacks to stderr (for hang diagnosis)\n"
    "        signal.signal(signal.SIGUSR2,\n"
    "                      lambda s, f: faulthandler.dump_traceback(all_threads=True))\n"
)

# ─── Sub-patch 2: SIGUSR2 workers before propagating timeout ──────────────

TIMEOUT_OLD = (
    "                except TimeoutError as e:\n"
    "                    raise TimeoutError(f\"RPC call to {method} timed out.\") from e\n"
)

TIMEOUT_NEW = (
    "                except TimeoutError as e:\n"
    "                    import os\n"
    "                    for _w in self.workers:\n"
    "                        try:\n"
    "                            os.kill(_w.proc.pid, signal.SIGUSR2)\n"
    "                        except (ProcessLookupError, OSError):\n"
    "                            pass\n"
    "                    raise TimeoutError(f\"RPC call to {method} timed out.\") from e\n"
)


def _make_patcher():
    target = resolve_vllm_file("v1/executor/multiproc_executor.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="N104 dump stacks on timeout",
        target_file=str(target),
        marker=GENESIS_MARKER,
        sub_patches=[
            TextPatch(
                name="faulthandler_import",
                anchor=IMPORT_OLD,
                replacement=IMPORT_NEW,
                required=True,
            ),
            TextPatch(
                name="sigusr2_worker_handler",
                anchor=SIGUSR2_HANDLER_OLD,
                replacement=SIGUSR2_HANDLER_NEW,
                required=True,
            ),
            TextPatch(
                name="sigusr2_before_timeout",
                anchor=TIMEOUT_OLD,
                replacement=TIMEOUT_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "Genesis N104 dump stacks on timeout",
            "faulthandler.dump_traceback",
        ],
    )


def apply():
    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "multiproc_executor.py not found"

    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        return "applied", (
            "N104: faulthandler + SIGUSR2 wired into worker_main() and "
            "get_response() timeout path"
        )
    if result == TextPatchResult.IDEMPOTENT:
        return "skipped", "N104: already applied (marker present)"
    if result == TextPatchResult.SKIPPED:
        r = failure.reason if failure else "unknown_skip"
        d = f" ({failure.detail})" if (failure and failure.detail) else ""
        return "skipped", f"N104: {r}{d}"
    r = failure.reason if failure else "unknown"
    d = f" ({failure.detail})" if (failure and failure.detail) else ""
    return "failed", f"N104: {r}{d}"
