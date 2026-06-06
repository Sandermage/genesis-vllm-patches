# SPDX-License-Identifier: Apache-2.0
"""SNDR Core — vllm install root discovery.

Stage 3 (revised, 2026-05-07): canonical impl lives in
`vllm._genesis.guards.vllm_install_root` for back-compat with test
mocks (tests monkey-patch `guards.vllm_install_root = lambda: td` and
expect propagation via `resolve_vllm_file()`). This module forwards
to the canonical home so imports via `from vllm.sndr_core.locations import
vllm_install_root` continue to work.

Public-facing usage (recommended for new code):

    from vllm.sndr_core.locations import vllm_install_root
    root = vllm_install_root()  # str or None

Test-mock pattern (for back-compat):

    import vllm.sndr_core.detection.guards as guards
    guards.vllm_install_root = lambda: "/tmp/test-vllm"
    # resolve_vllm_file() honors the monkey-patch
"""
from __future__ import annotations

# Forward to canonical location. Stage 6+ may move impl here once
# all monkey-patch test contracts have migrated.
from vllm.sndr_core.detection.guards import vllm_install_root  # noqa: F401

__all__ = ["vllm_install_root"]
