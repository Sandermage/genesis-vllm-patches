# SPDX-License-Identifier: Apache-2.0
"""G4_19c — Round-trip K,V through G4-TurboQuant inside Gemma4Attention.

================================================================
WHAT THIS DOES (and what it doesn't)
================================================================

For every ``Gemma4Attention.forward`` call we **compress and immediately
decompress** the post-norm post-RoPE K and V tensors via the G4-TurboQuant
kernels selected by the active pack/wht mode. The round-tripped tensors
are then passed to the underlying ``self.attn(q, k_rt, v_rt)`` call.

Effect:

  * The quantization error from TurboQuant **is applied** to every K, V
    that enters the attention math.
  * The vLLM-managed KV cache itself remains **standard fp16/bf16** —
    this hook does NOT replace the cache buffer.
  * Reduction in MEMORY (the headline TurboQuant promise) is NOT
    achieved by this hook — it requires a separate KV-cache-substitution
    patch (PN-future) that overrides vllm's KVCacheSpec.

So G4_19c is a **quality + speed A/B harness**:
  * Quality: measure perplexity / NIAH retrieval under quantization noise
  * Speed: measure TPOT cost of the extra compress+decompress per layer
  * Memory: NO change (memory savings come from a separate patch)

================================================================
WHY THIS IS THE RIGHT FIRST STEP
================================================================

Real KV-cache substitution requires touching vllm v1's
``KVCacheSpec`` + ``KVCacheManager`` + attention-backend block-table
plumbing. Each of those has dev371-specific signatures that change
upstream. A reliable A/B harness that injects the **same numerical
error** into attention without touching the cache buffer gives us the
EMPIRICAL data needed before committing engineering effort to the
buffer rewrite:

  * If the round-trip TPS hit is e.g. -5% per layer, full substitution
    saves memory **and** improves perf via smaller buffers — go for it.
  * If TPS hit is -25%, full substitution can't save enough to pay
    for compression overhead — explore tensor-core batching first.
  * If quality (NIAH @ 128K+) collapses under round-trip noise, abandon
    aggressive bit-widths and go to 4/5-bit before cache substitution.

================================================================
SERVER A/B RESULT 2026-05-17 — quality regression observed
================================================================

First end-to-end bench against live Gemma 4 31B AWQ + 256K context,
pack=uint32 wht=signs_only (3-bit Lloyd-Max, no real Hadamard):

  Baseline (G4_19c OFF)            G4_19c ON (uint32+signs)
  - "2+2?"        → "4" ✓         → "4! (Wait, it's 4!) ..." LOOPING
  - "primary cols" → "Red, blue,  → "//" BROKEN
                     yellow"
  - "WWII ended?"  → "1945" ✓     → "Historically, the** (Wait wait..." LOOPING
  - "Tokyo?"       → "Tokyo" ✓    → "Tokyo" ✓
  - "3 planets"    → "Mercury,    → "Mercury, Venus, **Venus**" HALLUCIN.
                     Venus, Earth"
  - long-ctx 8K needle: FOUND in both
  - TPS: 55-66 (CV ~40%, warmup variability)

**Root cause hypothesis**: Lloyd-Max codebooks are calibrated for unit-
variance Gaussian marginals, but Gemma 4 K/V tensors AFTER q_norm +
k_norm + v_norm + RoPE have a different empirical distribution. The
sign-only "rotation" (the placeholder path) doesn't Gaussianize them
either. Quantization error is asymmetric enough to skew attention
scores → divergent token sampling → looping / wrong answers.

**Next steps (separate session)**:
  1. Probe real K/V distribution stats (mean, std, kurtosis, %outliers)
     from a warmup pass; calibrate per-layer scale + codebook from
     observed data instead of assuming unit-variance Gaussian.
  2. Try full_wht mode (butterfly Hadamard actually applied) — paper
     claims Beta-concentration restores Gaussian marginals.
  3. Try 4-bit and 5-bit codebooks (more aggressive than 3-bit means
     less quantization-error margin; raising bits should ease quality).
  4. Per-layer adaptive bit-width: profile per-layer attention
     sensitivity and use 5-bit on attention-dominant layers, 3-bit on
     redundant ones.

Default status of G4_19c: **OFF in production launcher** until quality
regression is resolved.

================================================================
ENV FLAG
================================================================

``GENESIS_ENABLE_G4_19C_ATTN_WRAP=1`` enables this hook. Default OFF
because it adds compute on every attention layer; operators must
explicitly opt in for A/B benches.

Requires G4_19 to have already populated the config registry — if
``g4_19_config_registry.is_active()`` is False at apply time, G4_19c
skips with a clear message.

================================================================
PER-LAYER ROTATION SEEDS
================================================================

Each layer gets a distinct random-sign vector via
``build_randomized_hadamard_seed(head_dim, layer_idx)``. We extract
``layer_idx`` from ``self.prefix`` (e.g. "model.layers.5.self_attn" → 5)
at first call and cache the device-resident sign tensor on ``self``
to avoid the numpy→torch transfer cost on hot path.

================================================================
PER-LAYER BIT-WIDTH
================================================================

If the active config has ``per_layer_types`` set (populated by G4_19's
``verify_and_update_config`` wrap), we use ``bits_sliding`` on
sliding-attention layers and ``bits_global`` on full-attention ones.
Otherwise (eager-registered config from worker subprocess), we use
``bits_global`` for all layers — conservative (more compression =
more error). Operator can override via env.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from typing import Optional

log = logging.getLogger("genesis.gemma4.g4_19c")

GENESIS_G4_19C_MARKER = (
    "Genesis G4_19c attention K/V round-trip wrapper v1 "
    "(injects TurboQuant quantization noise into Gemma4Attention math; "
    "memory layout unchanged — companion patch needed for buffer savings)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_19C_ATTN_WRAP"
_ENV_DEBUG = "GENESIS_G4_19C_DEBUG"

_APPLIED = False
_ORIGINAL_FORWARD = None
_ORIGINAL_INIT = None

_PREFIX_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _env_debug() -> bool:
    return os.environ.get(_ENV_DEBUG, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _env_truthy_local(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _extract_layer_idx(prefix: str) -> int:
    """Parse layer_idx from a ``self.prefix`` string like
    ``"model.layers.5.self_attn"`` → 5. Falls back to 0 if unparseable."""
    m = _PREFIX_LAYER_RE.search(prefix or "")
    return int(m.group(1)) if m else 0


def _select_bits(config, layer_idx: int) -> int:
    """Pick bits-per-coord for this layer.

    With per_layer_types: sliding → bits_sliding, full → bits_global.
    Without per_layer_types: bits_global (conservative).
    """
    lt = getattr(config, "per_layer_types", None)
    if lt is not None and 0 <= layer_idx < len(lt):
        return config.bits_sliding if lt[layer_idx] == "sliding_attention" else config.bits_global
    return config.bits_global


_KERNEL_LOCK = threading.Lock()


def _resolve_kernels(config) -> tuple:
    """Return (write_fn, read_fn, kernel_name) for the active (pack, wht) mode.

    Cached at first call to avoid repeated import overhead on hot path.
    """
    pack_mode = config.pack_mode
    wht_mode = config.wht_mode

    if pack_mode == "tight":
        from .kernels.g4_tq_tight_triton import (
            g4_tq_write_tight_3bit, g4_tq_read_tight_3bit,
        )
        apply_wht = (wht_mode == "full_wht")
        name = f"tight+{wht_mode}"

        def _write(x, signs, head_dim, block_size):
            return g4_tq_write_tight_3bit(
                x, signs, head_dim=head_dim, block_size=block_size,
                apply_wht=apply_wht,
            )

        def _read(packed, scale, signs, head_dim, block_size, dtype):
            return g4_tq_read_tight_3bit(
                packed, scale, signs,
                head_dim=head_dim, block_size=block_size,
                apply_wht=apply_wht, dtype=dtype,
            )
        return _write, _read, name

    if wht_mode == "full_wht":
        from .kernels.g4_tq_packed_wht_triton import (
            g4_tq_write_packed_wht_3bit, g4_tq_read_packed_wht_3bit,
        )

        def _write(x, signs, head_dim, block_size):
            return g4_tq_write_packed_wht_3bit(
                x, signs, head_dim=head_dim, block_size=block_size,
            )

        def _read(packed, scale, signs, head_dim, block_size, dtype):
            return g4_tq_read_packed_wht_3bit(
                packed, scale, signs,
                head_dim=head_dim, block_size=block_size, dtype=dtype,
            )
        return _write, _read, "uint32+full_wht"

    from .kernels.g4_tq_packed_triton import (
        g4_tq_write_packed_3bit, g4_tq_read_packed_3bit,
    )

    def _write(x, signs, head_dim, block_size):
        return g4_tq_write_packed_3bit(
            x, signs, head_dim=head_dim, block_size=block_size,
        )

    def _read(packed, scale, signs, head_dim, block_size, dtype):
        return g4_tq_read_packed_3bit(
            packed, scale, signs,
            head_dim=head_dim, block_size=block_size, dtype=dtype,
        )
    return _write, _read, "uint32+signs_only"


_SIGNS_LOCK = threading.Lock()
# Process-global pre-built sign tensors, keyed by (layer_idx, head_dim,
# seed_base, device-string). Built once at apply()-time or on first use
# OUTSIDE any traced region; the forward path only reads from this dict
# so CUDA-graph capture never sees a numpy/CPU op.
_SIGNS_CACHE: dict[tuple, "object"] = {}


def _build_signs_torch(head_dim: int, layer_idx: int, seed_base: int):
    """CPU fp32 sign tensor matching numpy reference
    ``build_randomized_hadamard_seed`` (same seed → same signs).

    Uses numpy (not torch.Generator) because vllm sets
    ``torch.set_default_device("cuda")`` during model construction;
    a torch CPU-generator inside that scope fails with
    "Expected a 'cuda' device type for generator but found 'cpu'".

    Called ONLY from non-traced regions (Gemma4Attention.__init__ or
    a cold-cache lookup). Forward path always hits the pre-built device
    tensor and never re-enters this function.
    """
    import numpy as np
    import torch
    seed_raw = (seed_base ^ (0x9E3779B97F4A7C15 + layer_idx))
    rng = np.random.default_rng(seed_raw)
    bits = rng.choice([-1.0, 1.0], size=head_dim).astype(np.float32)
    return torch.from_numpy(bits)


def _get_or_build_signs(
    layer_idx: int, head_dim: int, seed_base: int, device,
    attn_layer=None,  # kept for API back-compat; ignored after CUDA-graph bug
):
    """Lookup the device-resident signs tensor from the process-global cache.

    We intentionally do NOT attach signs to ``nn.Module`` instances —
    torch.compile / inductor captures module attributes into the
    compiled graph, and a CPU tensor attribute fails CUDA-graph capture
    with::

        RuntimeError: Cannot copy between CPU and CUDA tensors during
        CUDA graph capture unless the CPU tensor is pinned.

    The global cache is keyed by ``(layer_idx, head_dim, seed_base,
    device_str)`` and holds device-resident tensors only. Cold-path
    builds may still allocate inside a traced region — to avoid that,
    the wrap on ``Gemma4Attention.__init__`` pre-populates the cache
    BEFORE any forward runs.
    """
    del attn_layer  # signs are NOT attached to layers — see docstring
    key = (layer_idx, head_dim, seed_base, str(device))
    cached = _SIGNS_CACHE.get(key)
    if cached is not None:
        return cached
    with _SIGNS_LOCK:
        cached = _SIGNS_CACHE.get(key)
        if cached is not None:
            return cached
        signs_cpu = _build_signs_torch(head_dim, layer_idx, seed_base)
        signs = signs_cpu.to(device)
        _SIGNS_CACHE[key] = signs
        return signs


def prewarm_signs(num_layers: int, head_dim: int, seed_base: int, device) -> int:
    """Pre-populate the sign cache for ``num_layers`` × head_dim before any
    CUDA-graph capture runs. Returns the number of new entries added.

    Called from apply() once we know the model dimensions. Safe to call
    repeatedly — cache hits short-circuit.
    """
    added = 0
    for layer_idx in range(num_layers):
        key = (layer_idx, head_dim, seed_base, str(device))
        if key in _SIGNS_CACHE:
            continue
        signs_cpu = _build_signs_torch(head_dim, layer_idx, seed_base)
        with _SIGNS_LOCK:
            _SIGNS_CACHE[key] = signs_cpu.to(device)
        added += 1
    return added


def _roundtrip(x, signs, head_dim, block_size, write_fn, read_fn):
    """Compress + decompress x once. Returns a tensor of the same
    shape/dtype as x with the quantization noise applied.
    """
    # x is (..., head_dim_total) where head_dim_total = num_kv_heads * head_dim
    # The kernel expects shape (M, num_kv_heads, head_dim). Reshape:
    orig_shape = x.shape
    # The K/V tensor coming out of qkv_proj.split has shape
    #   (num_tokens, num_kv_heads * head_dim)
    num_kv_heads = orig_shape[-1] // head_dim
    M = x.numel() // (num_kv_heads * head_dim)
    x_3d = x.contiguous().view(M, num_kv_heads, head_dim)

    packed, scale = write_fn(x_3d, signs, head_dim, block_size)
    x_rt = read_fn(packed, scale, signs, head_dim, block_size, x.dtype)
    return x_rt.view(orig_shape)


def _make_wrapped_forward(original_forward):
    """Wrap ``Gemma4Attention.forward`` to round-trip K, V through TurboQuant.

    Only fires when:
      * the registry has an active config (G4_19 ran)
      * the layer has ``self.attn`` (vllm Attention instance — not the
        embedding-only layers)
      * the layer is NOT KV-shared (shared layers re-use the previous
        layer's K, so we skip to avoid corrupting it)
    """

    def _wrapped_g4_attn_forward(self, positions, hidden_states, **kwargs):
        from .g4_19_config_registry import get_active_config
        config = get_active_config()
        if config is None:
            return original_forward(self, positions, hidden_states, **kwargs)

        # Replicate the original forward up to attention call:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1,
        )
        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = self.q_norm(q)
        q = q.flatten(-2, -1)

        if not self.is_kv_shared_layer:
            k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
            k = self.k_norm(k)
            k = k.flatten(-2, -1)
            q, k = self.rotary_emb(positions, q, k)
            v = v.unflatten(-1, (self.num_kv_heads, self.head_dim))
            v = self.v_norm(v)
            v = v.flatten(-2, -1)

            # ── G4_19c hook: round-trip K and V through TurboQuant ──
            #
            # ARCHITECTURAL PERF FIX (2026-05-17):
            # Gemma 4 has 50 sliding (1024 ctx) + 10 full (256K ctx)
            # attention layers. 99.7% of KV cache memory lives in the
            # 10 full-attention layers. Sliding layers are tiny and
            # compression-overhead-bound: round-trip costs more wall
            # time than the storage saving justifies.
            #
            # Skip sliding-attention layers entirely:
            #   * Sliding layers attribute ``is_sliding=True`` (per
            #     gemma4.py Gemma4Attention.__init__ — line 436).
            #   * Skipping cuts the round-trip count from 60 → 10 per
            #     decode token (~6× less work).
            # Operator can disable the skip via
            # ``GENESIS_G4_19C_FORCE_ALL_LAYERS=1``.
            if (
                getattr(self, "is_sliding", False)
                and not _env_truthy_local("GENESIS_G4_19C_FORCE_ALL_LAYERS")
            ):
                # No round-trip for sliding layers — fall through with
                # un-modified K, V. KV cache contribution is negligible
                # at sliding_window=1024.
                pass
            else:
                try:
                    layer_idx = _extract_layer_idx(getattr(self, "prefix", ""))
                    write_fn, read_fn, _kernel_name = _resolve_kernels(config)
                    head_dim = self.head_dim
                    block_size = getattr(config, "block_size", 128)
                    seed_base = getattr(config, "seed_base", 0xC0FFEE)
                    signs = _get_or_build_signs(
                        layer_idx, head_dim, seed_base, k.device,
                        attn_layer=self,
                    )
                    k = _roundtrip(k, signs, head_dim, block_size, write_fn, read_fn)
                    v = _roundtrip(v, signs, head_dim, block_size, write_fn, read_fn)
                    if _env_debug():
                        log.debug(
                            "[G4_19c] layer=%d %s round-tripped K/V "
                            "(shape=%s dtype=%s)",
                            layer_idx, _kernel_name, tuple(k.shape), k.dtype,
                        )
                except Exception as e:  # noqa: BLE001
                    # Fail-open: log once, fall through with un-modified K, V.
                    if not getattr(self, "_genesis_g4_19c_warned", False):
                        log.warning(
                            "[G4_19c] round-trip failed at layer=%s (%r); "
                            "falling through with untouched K,V",
                            getattr(self, "prefix", "?"), e,
                        )
                        self._genesis_g4_19c_warned = True
        else:
            # Shared-K layer: only apply RoPE to Q
            q = self.rotary_emb(positions, q, k)[0]

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    _wrapped_g4_attn_forward._genesis_g4_19c_wrapped = True
    _wrapped_g4_attn_forward.__wrapped__ = original_forward
    return _wrapped_g4_attn_forward


def apply() -> tuple[str, str]:
    """Install K/V round-trip wrapper on Gemma4Attention.forward."""
    global _APPLIED, _ORIGINAL_FORWARD

    if not _env_enabled():
        return "skipped", (
            f"G4_19c disabled (set {_ENV_ENABLE}=1 to enable the K/V "
            "round-trip wrapper for A/B quality+perf benchmarking)"
        )

    # Need an active config from G4_19 — otherwise no kernels to select
    try:
        from .g4_19_config_registry import is_active
        if not is_active():
            return "skipped", (
                "G4_19c needs G4_19 to have populated the config registry first "
                "(set GENESIS_ENABLE_G4_19_GEMMA4_TURBOQUANT_KV=1 + bring G4_19 "
                "into the apply chain before G4_19c)"
            )
    except Exception as e:  # noqa: BLE001
        return "skipped", f"registry lookup failed: {e!r}"

    if _APPLIED:
        return "applied", "G4_19c already installed (idempotent)"

    try:
        from vllm.model_executor.models import gemma4 as _g4
    except ImportError as e:
        return "skipped", f"vllm.model_executor.models.gemma4 not importable: {e}"

    if not hasattr(_g4, "Gemma4Attention"):
        return "skipped", "Gemma4Attention class not found on this pin"

    original = _g4.Gemma4Attention.forward
    if getattr(original, "_genesis_g4_19c_wrapped", False):
        _APPLIED = True
        return "applied", "Gemma4Attention.forward already wrapped (idempotent)"

    _ORIGINAL_FORWARD = original
    _g4.Gemma4Attention.forward = _make_wrapped_forward(original)

    # Also wrap __init__ so signs are built BEFORE first forward (which
    # may run inside a CUDA-graph capture region — at-runtime allocation
    # inside a captured graph is unsupported and triggers a Triton trace
    # error 'attempted to trace numpy function unsupported by PyTorch').
    global _ORIGINAL_INIT
    original_init = _g4.Gemma4Attention.__init__
    if not getattr(original_init, "_genesis_g4_19c_init_wrapped", False):
        _ORIGINAL_INIT = original_init

        def _wrapped_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            try:
                from .g4_19_config_registry import get_active_config
                config = get_active_config()
                if config is None:
                    return
                layer_idx = _extract_layer_idx(getattr(self, "prefix", ""))
                head_dim = getattr(self, "head_dim", 256)
                seed_base = getattr(config, "seed_base", 0xC0FFEE)

                # Build signs and put DIRECTLY on the current CUDA device.
                # We MUST NOT attach to ``self`` as an nn.Module attribute
                # — torch.compile would then capture it into the compiled
                # graph, and a CPU tensor on a CUDA-target graph triggers
                # "Cannot copy between CPU and CUDA tensors during CUDA
                # graph capture" at runtime.
                #
                # Instead we publish to the process-global cache; the
                # forward path looks it up by key.
                import torch
                if torch.cuda.is_available():
                    device_str = f"cuda:{torch.cuda.current_device()}"
                else:
                    device_str = "cpu"
                key = (layer_idx, head_dim, seed_base, device_str)
                if key not in _SIGNS_CACHE:
                    signs_cpu = _build_signs_torch(
                        head_dim, layer_idx, seed_base,
                    )
                    with _SIGNS_LOCK:
                        if key not in _SIGNS_CACHE:
                            _SIGNS_CACHE[key] = signs_cpu.to(device_str)
            except Exception as e:  # noqa: BLE001
                log.warning(
                    "[G4_19c] sign pre-build at __init__ failed (%r); "
                    "falling back to lazy build (may fail in CUDA-graph)",
                    e,
                )

        _wrapped_init._genesis_g4_19c_init_wrapped = True
        _wrapped_init.__wrapped__ = original_init
        _g4.Gemma4Attention.__init__ = _wrapped_init

    _APPLIED = True

    log.info(
        "[G4_19c] installed: Gemma4Attention.forward now round-trips K,V "
        "through G4-TurboQuant on every call. Note: cache BUFFER unchanged."
    )
    return "applied", (
        "G4_19c installed: Gemma4Attention.forward now round-trips K,V "
        "through G4-TurboQuant kernels. Active kernel selected from registry "
        "(env-driven pack_mode / wht_mode). KV cache BUFFER unchanged — this "
        "is the A/B harness for quality/perf measurement; full memory "
        "savings need a separate cache-substitution patch."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    global _APPLIED, _ORIGINAL_FORWARD, _ORIGINAL_INIT
    if not _APPLIED:
        return False
    try:
        from vllm.model_executor.models import gemma4 as _g4
        if _ORIGINAL_FORWARD is not None:
            _g4.Gemma4Attention.forward = _ORIGINAL_FORWARD
        if _ORIGINAL_INIT is not None:
            _g4.Gemma4Attention.__init__ = _ORIGINAL_INIT
        _SIGNS_CACHE.clear()
        _APPLIED = False
        return True
    except ImportError:
        return False


__all__ = [
    "GENESIS_G4_19C_MARKER",
    "apply",
    "is_applied",
    "revert",
]
