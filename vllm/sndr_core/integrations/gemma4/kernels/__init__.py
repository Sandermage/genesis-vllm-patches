# SPDX-License-Identifier: Apache-2.0
"""Custom Triton kernels for Gemma 4 family patches.

Kernels here are wired in by the G4_NN patches that need them. Each
kernel keeps its own module so the corresponding patch can import it
lazily — pure-CPU test collection (e.g. on the maintainer's laptop)
won't fail just because triton isn't installed.
"""
