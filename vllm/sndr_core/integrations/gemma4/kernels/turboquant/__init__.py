# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim — Genesis G4-TurboQuant kernels relocated.

Real package: vllm.sndr_core.integrations.attention.turboquant.kernels

The TurboQuant kernel implementations were moved 2026-05-21 (Phase 3
bucket 4) from ``gemma4/kernels/turboquant/`` to
``attention/turboquant/kernels/`` so their ownership matches their
technical area of influence rather than the first model that used
them. The kernels remain applicable to any TurboQuant consumer.

Shim window: one release. Remove this file (and the empty parent
``gemma4/kernels/turboquant/`` directory) after external imports
migrate.
"""
from vllm.sndr_core.integrations.attention.turboquant.kernels import *  # noqa: F401,F403
from vllm.sndr_core.integrations.attention.turboquant.kernels import (  # noqa: F401
    GENESIS_G4_TQ_VERSION,
    g4_tq_cache,
    g4_tq_codebook,
    g4_tq_packed_triton,
    g4_tq_packed_wht_triton,
    g4_tq_packing,
    g4_tq_read_triton,
    g4_tq_reference,
    g4_tq_rotor,
    g4_tq_tight_triton,
    g4_tq_write_triton,
)
