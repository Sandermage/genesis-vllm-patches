# SPDX-License-Identifier: Apache-2.0
"""G4_19c — Round-trip K,V through G4-TurboQuant inside Gemma4Attention.

RETIRED 2026-05-29 — see `dispatcher/registry.py` G4_19C entry. This
module is preserved on disk for git-blame + operator rollback, but
`apply()` self-skips on the current pin. Do NOT enable the env flag
without re-reading the registry note and the §1.4 G4_19C Phase A+B
closure rationale.

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
ITERATION HISTORY
================================================================

  Iter 1 (commit 5e13cb8e): ``@torch._dynamo.disable`` on cold builders.
                            gb0098 fail under fullgraph.
  Iter 2 (commit 6ecc685d): ``register_buffer("_g4_19c_signs", ...)`` at
                            ``__init__`` time; forward reads buffer
                            directly. Closed the build-from-forward
                            blocker but left 8 other Dynamo blockers in
                            the class-level wrapped forward body
                            (env reads, config lookups, try/except,
                            kernel-resolution-per-call, log calls,
                            module-state mutation).
  Iter 3 (this commit):     ARCHITECTURAL shift. Every Python-side
                            decision is baked in at apply() / __init__
                            time as install-time constants. Each
                            instance gets EITHER the unmodified
                            original_forward (eager-pass — same as
                            "G4_19c never touched this layer") OR
                            the specialized ``_active_forward`` from
                            ``g4_19c_per_layer_forward.py`` (pure
                            tensor ops + one allow_in_graph kernel
                            call per K and V). Per-instance install
                            via ``types.MethodType`` — no class-level
                            forward monkeypatch.

See ``sndr_private/research/gemma4/analysis/
     G4_19C_FULLGRAPH_AUDIT_R_2026-05-23.md`` for the full audit.

================================================================
SERVER A/B RESULT 2026-05-17 — quality regression observed
================================================================

(Unchanged from iter-2 docstring — kept here as a deployment note.)

First end-to-end bench against live Gemma 4 31B AWQ + 256K context,
pack=uint32 wht=signs_only (3-bit Lloyd-Max, no real Hadamard):

  Baseline (G4_19c OFF)            G4_19c ON (uint32+signs)
  - "2+2?"        → "4" ✓         → "4! (Wait, it's 4!) ..." LOOPING
  - "primary cols" → "Red, blue,  → "//" BROKEN
                     yellow"
  - "WWII ended?"  → "1945" ✓     → "Historically, the** (Wait wait..." LOOPING

**Root cause hypothesis**: Lloyd-Max codebooks are calibrated for unit-
variance Gaussian marginals, but Gemma 4 K/V tensors AFTER q_norm +
k_norm + v_norm + RoPE have a different empirical distribution.

Default status of G4_19c: **OFF in production launcher** until quality
regression is resolved. The fullgraph fix (this iter) unblocks
re-enabling the wrapper for A/B benches; it does NOT solve the
quality regression itself.

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
at ``__init__`` and cache the device-resident sign tensor as a
non-persistent buffer on the module.

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
import types
from typing import Optional

log = logging.getLogger("genesis.turboquant.g4_19c")

GENESIS_G4_19C_MARKER = (
    "Genesis G4_19c attention K/V round-trip wrapper v3 "
    "(per-layer specialized forward via types.MethodType; "
    "allow_in_graph kernel entry; fullgraph-safe hot path)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_19C_ATTN_WRAP"
_ENV_DEBUG = "GENESIS_G4_19C_DEBUG"
_ENV_FORCE_ALL_LAYERS = "GENESIS_G4_19C_FORCE_ALL_LAYERS"

_APPLIED = False
_ORIGINAL_INIT = None
_ORIGINAL_FORWARD_REF = None  # captured at apply() for per-instance install
_FORCE_ALL_LAYERS = False     # frozen at apply() — read ONCE from env
_DEBUG = False                # frozen at apply() — read ONCE from env

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

    Retained for compatibility with prewarm / debug callers. NOT used
    on the hot path post-iter-3 — kernel resolution happens once at
    apply() time.
    """
    lt = getattr(config, "per_layer_types", None)
    if lt is not None and 0 <= layer_idx < len(lt):
        return config.bits_sliding if lt[layer_idx] == "sliding_attention" else config.bits_global
    return config.bits_global


_KERNEL_LOCK = threading.Lock()


def _resolve_kernels(config) -> tuple:
    """Return (write_fn, read_fn, kernel_name) for the active (pack, wht) mode.

    Called ONCE at apply() time post-iter-3 (was: every forward call
    in iter-1/iter-2). The result is stashed into the companion
    ``g4_19c_per_layer_forward`` module via ``setup()`` so the active
    forward can reach it without per-call resolution.
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
# seed_base, device-string). Populated eagerly at ``_wrapped_init``
# (and optionally ``prewarm_signs``); each Gemma4Attention instance
# carries its OWN copy as a non-persistent buffer ``self._g4_19c_signs``,
# so this cache is only used for de-dup / debug / prewarm. The forward
# hot path never reads from it.
_SIGNS_CACHE: dict[tuple, "object"] = {}


def _build_signs_torch(head_dim: int, layer_idx: int, seed_base: int):
    """CPU fp32 sign tensor matching numpy reference
    ``build_randomized_hadamard_seed`` (same seed → same signs).

    Called ONLY from eager paths (``_wrapped_init``, ``prewarm_signs``,
    direct unit-test callers). After iter-3 the active forward reads
    ``self._g4_19c_signs`` as a per-layer CUDA buffer attribute and
    never calls this function, so the numpy RNG / device transfer
    inside never enters a torch.compile fullgraph region.
    """
    import numpy as np
    import torch
    seed_raw = (seed_base ^ (0x9E3779B97F4A7C15 + layer_idx))
    rng = np.random.default_rng(seed_raw)
    bits = rng.choice([-1.0, 1.0], size=head_dim).astype(np.float32)
    return torch.from_numpy(bits)


def _get_or_build_signs(
    layer_idx: int, head_dim: int, seed_base: int, device,
    attn_layer=None,  # kept for API back-compat; ignored
):
    """Lookup-or-build the device-resident signs tensor from the
    process-global cache. **NOT called from any compile region after
    iter-2**.
    """
    del attn_layer
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


# ─── Per-instance install logic ─────────────────────────────────────────


def _decide_layer_active(self) -> bool:
    """Decide at __init__ time whether this layer should round-trip.

    All decisions use STATIC layer properties or apply-time constants —
    no per-call branches reach the compiled forward.
    """
    # Registry must have an active config.
    try:
        from .g4_19_config_registry import get_active_config
        config = get_active_config()
    except Exception:  # noqa: BLE001
        return False
    if config is None:
        return False

    # KV-shared layers re-use the previous layer's K — round-tripping
    # would corrupt the upstream shared K cache.
    if getattr(self, "is_kv_shared_layer", False):
        return False

    # Sliding-attention layers have a tiny KV budget (window=1024);
    # round-trip overhead is not justified vs storage savings.
    if getattr(self, "is_sliding", False) and not _FORCE_ALL_LAYERS:
        return False

    return True


def _attach_signs_buffer(self) -> None:
    """Build and attach ``self._g4_19c_signs`` as a non-persistent
    CUDA-resident buffer. Idempotent."""
    import torch
    from .g4_19_config_registry import get_active_config
    config = get_active_config()
    layer_idx = _extract_layer_idx(getattr(self, "prefix", ""))
    head_dim = getattr(self, "head_dim", 256)
    seed_base = getattr(config, "seed_base", 0xC0FFEE)

    if torch.cuda.is_available():
        device_str = f"cuda:{torch.cuda.current_device()}"
    else:
        device_str = "cpu"
    key = (layer_idx, head_dim, seed_base, device_str)
    cached = _SIGNS_CACHE.get(key)
    if cached is None:
        signs_cpu = _build_signs_torch(head_dim, layer_idx, seed_base)
        cached = signs_cpu.to(device_str)
        with _SIGNS_LOCK:
            _SIGNS_CACHE.setdefault(key, cached)

    if hasattr(self, "register_buffer") and callable(self.register_buffer):
        self.register_buffer("_g4_19c_signs", cached, persistent=False)
    else:
        self._g4_19c_signs = cached


def apply() -> tuple[str, str]:
    """Install K/V round-trip wrapper on Gemma4Attention.

    Iter-3 (Phase 7.G4.G4_19C-FULLGRAPH-AUDIT):
      • Resolves kernel pair + frozen env flags ONCE at apply() time.
      • Wraps ``Gemma4Attention.__init__`` at class level so every new
        instance gets a per-instance specialized forward bound to it.
      • Does NOT class-level monkeypatch ``Gemma4Attention.forward``
        — each instance carries its own bound forward via
        ``types.MethodType``.
    """
    global _APPLIED, _ORIGINAL_INIT, _ORIGINAL_FORWARD_REF
    global _FORCE_ALL_LAYERS, _DEBUG

    if not _env_enabled():
        return "skipped", (
            f"G4_19c disabled (set {_ENV_ENABLE}=1 to enable the K/V "
            "round-trip wrapper for A/B quality+perf benchmarking)"
        )

    # Need an active config from G4_19.
    try:
        from .g4_19_config_registry import is_active, get_active_config
        if not is_active():
            return "skipped", (
                "G4_19c needs G4_19 to have populated the config registry first "
                "(set GENESIS_ENABLE_G4_19_GEMMA4_TURBOQUANT_KV=1 + bring G4_19 "
                "into the apply chain before G4_19c)"
            )
        config = get_active_config()
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

    # Capture the ORIGINAL forward BEFORE any per-instance binding —
    # used as the eager-pass install for inactive layers and as the
    # baseline reference for revert().
    _ORIGINAL_FORWARD_REF = _g4.Gemma4Attention.forward

    # Freeze env flags at apply() time so the hot path never reads
    # os.environ.
    _FORCE_ALL_LAYERS = _env_truthy_local(_ENV_FORCE_ALL_LAYERS)
    _DEBUG = _env_debug()

    # Resolve the kernel pair ONCE and wire the companion module.
    write_fn, read_fn, kernel_name = _resolve_kernels(config)
    block_size = getattr(config, "block_size", 128)
    from . import g4_19c_per_layer_forward as _per_layer
    _per_layer.setup(write_fn, read_fn, block_size)

    # Wrap __init__ at class level so every newly-constructed
    # Gemma4Attention instance receives a per-instance forward.
    original_init = _g4.Gemma4Attention.__init__
    if getattr(original_init, "_genesis_g4_19c_init_wrapped", False):
        _APPLIED = True
        return "applied", (
            f"G4_19c init-wrap already installed (idempotent); "
            f"kernel={kernel_name}"
        )

    _ORIGINAL_INIT = original_init
    original_forward = _ORIGINAL_FORWARD_REF

    def _wrapped_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        do_roundtrip = _decide_layer_active(self)
        if do_roundtrip:
            try:
                _attach_signs_buffer(self)
            except Exception as e:  # noqa: BLE001
                log.warning(
                    "[G4_19c] sign pre-build at __init__ failed at "
                    "layer=%s (%r); installing eager-pass forward on "
                    "this layer (no round-trip, no Dynamo-region risk)",
                    getattr(self, "prefix", "?"), e,
                )
                do_roundtrip = False
        self._g4_19c_active = do_roundtrip

        specialized = _per_layer.make_per_layer_forward(
            do_roundtrip, original_forward,
        )
        # Per-instance bind. Each Gemma4Attention layer's `self.forward`
        # now points at either original_forward (eager-pass) or
        # _per_layer._active_forward (active hot path). Dynamo compiles
        # the bound method graph per instance — no class-level
        # branching.
        self.forward = types.MethodType(specialized, self)

    _wrapped_init._genesis_g4_19c_init_wrapped = True
    _wrapped_init.__wrapped__ = original_init
    _g4.Gemma4Attention.__init__ = _wrapped_init

    _APPLIED = True
    log.info(
        "[G4_19c] installed iter-3: per-instance specialized forward "
        "via types.MethodType, allow_in_graph kernel entry. "
        "kernel=%s force_all_layers=%s",
        kernel_name, _FORCE_ALL_LAYERS,
    )
    return "applied", (
        f"G4_19c iter-3 installed: per-layer specialized forward "
        f"(allow_in_graph kernel entry). kernel={kernel_name}. "
        "KV cache BUFFER unchanged — this is the A/B harness for "
        "quality/perf measurement; full memory savings need a separate "
        "cache-substitution patch."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    """Restore the original ``Gemma4Attention.__init__``. Per-instance
    forwards on already-constructed layers persist on those instances
    until they're garbage-collected (Gemma4Attention objects only live
    for the lifetime of one model load).
    """
    global _APPLIED, _ORIGINAL_INIT, _ORIGINAL_FORWARD_REF
    if not _APPLIED:
        return False
    try:
        from vllm.model_executor.models import gemma4 as _g4
        if _ORIGINAL_INIT is not None:
            _g4.Gemma4Attention.__init__ = _ORIGINAL_INIT
        _SIGNS_CACHE.clear()
        _APPLIED = False
        _ORIGINAL_INIT = None
        _ORIGINAL_FORWARD_REF = None
        return True
    except ImportError:
        return False


__all__ = [
    "GENESIS_G4_19C_MARKER",
    "apply",
    "is_applied",
    "revert",
    "prewarm_signs",
]
