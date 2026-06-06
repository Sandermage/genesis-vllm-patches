# SPDX-License-Identifier: Apache-2.0
"""Genesis G4-TurboQuant — KV cache compression kernels.

Vector-quantization KV cache adapted from TurboQuant (arXiv:2504.19874)
+ RotorQuant variant (Mimi8298 / scrya-com proposal in vllm#38291),
tuned for the Gemma 4 architecture (head_dim=256 decomposed into
2× 128-blocks for rotor application; mixed sliding+global attention;
GQA with uniform num_kv_heads per layer in current cyankiwi
checkpoints; native vLLM v1 KV-cache integration).

Relocated 2026-05-21 (Phase 3 bucket 4) from
``integrations/gemma4/kernels/turboquant/`` to
``integrations/attention/turboquant/kernels/`` so the TurboQuant
kernels are owned by their technical area (attention) rather than
by the first model that used them (Gemma 4). The kernels remain
applicable to any TurboQuant consumer.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""

from .g4_tq_codebook import (
    BITS_3_LLOYD_MAX_CENTROIDS,
    BITS_4_LLOYD_MAX_CENTROIDS,
    BITS_5_LLOYD_MAX_CENTROIDS,
    get_centroids,
    lloyd_max_codebook,
)
from .g4_tq_rotor import (
    build_randomized_hadamard_seed,
    clifford_rotate_full,
    clifford_rotor_layer,
    randomized_hadamard_apply,
    randomized_hadamard_apply_blocked,
)

__all__ = [
    "BITS_3_LLOYD_MAX_CENTROIDS",
    "BITS_4_LLOYD_MAX_CENTROIDS",
    "BITS_5_LLOYD_MAX_CENTROIDS",
    "get_centroids",
    "lloyd_max_codebook",
    "build_randomized_hadamard_seed",
    "clifford_rotate_full",
    "clifford_rotor_layer",
    "randomized_hadamard_apply",
    "randomized_hadamard_apply_blocked",
]

GENESIS_G4_TQ_VERSION = "g4_tq_v1.0_genesis"
