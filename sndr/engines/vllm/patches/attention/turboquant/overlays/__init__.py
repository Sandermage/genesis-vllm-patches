# SPDX-License-Identifier: Apache-2.0
"""TurboQuant attention-backend overlay packages.

This namespace holds the read-only bind-mount overlay subtrees that the
rendered launcher mounts into the vLLM container at boot. Each
sub-package represents a single upstream PR cherry-pick (or an
equivalent in-tree overlay) whose files replace stock vLLM modules via
``-v <host>:<container>:ro`` flags emitted by ``cli/profile.py``.

Current members:
  * ``pr42637/`` — vllm PR #42637 (TurboQuant attention backend +
    Triton decode/store kernels + KV cache interface/utils/block-pool
    + Gemma 4 reasoning/tool parsers + turboquant config). Loaded at
    boot by G4_60a..k loader patches (in this same family) and
    verified by G4_60b/c/d + G4_68. Relocated 2026-05-22 from
    ``integrations/gemma4/upstream_overlay_pr42637/`` to its
    technical-area canonical home; container paths unchanged.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
