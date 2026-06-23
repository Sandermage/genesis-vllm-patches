# SPDX-License-Identifier: Apache-2.0
"""SNDR Core — resolve_vllm_file() forwarding.

v10 (2026-05-07): canonical impl now lives at
`sndr.engines.vllm.detection.guards.resolve_vllm_file` — `vllm._genesis.guards`
is a sys.modules redirect to that module. The wiring tests monkey-patch
`guards.vllm_install_root` and expect `resolve_vllm_file()` to honor
the patch — which still works because both module names resolve to the
same module object.

Recommended usage pattern (new code):

    from sndr.engines.vllm.locations import engine_targets, resolve_vllm_file
    target = resolve_vllm_file(engine_targets.QWEN3CODER_TOOL_PARSER)
"""
from __future__ import annotations

from sndr.engines.vllm.detection.guards import resolve_vllm_file  # noqa: F401

__all__ = ["resolve_vllm_file"]
