# SPDX-License-Identifier: Apache-2.0
"""Verbatim copies of vllm PR #42637 (Mixed-attention TurboQuant for Gemma 4).

Used by Genesis G4_60b/c/d loader patches as bind-mount overlay sources.
NOT importable as Python — Triton kernels and frozen dataclasses inside
require live vllm module context.

See README.md for full inventory + bind-mount mapping.

Source: https://github.com/vllm-project/vllm/pull/42637 HEAD fdeb14981.
License of copied files: Apache-2.0 (vllm contributors).

Author of this loader package: Sandermage (Sander) Barzov Aleksandr,
Ukraine, Odessa.
"""
__all__: list[str] = []
