# SPDX-License-Identifier: Apache-2.0
"""§6.H6 — known-trace catalog for the `sndr trace` CLI surface.

Plan §6.H of the unified development plan asks for an operator-facing
trace surface:

  * H6 `sndr trace list` (this commit)
  * H7 `sndr trace collect --container <name>`
  * H8 `sndr trace summarize <log-file>`
  * H9 `sndr support-bundle`

Genesis patches write a handful of diagnostic logs to ``/tmp/`` inside
the vLLM container when their `GENESIS_ENABLE_*` env flag is set. The
canonical list grew organically as patches landed; this module collects
them into a single registry so the CLI verbs above (and future audit
gates) can enumerate them without grepping the integrations tree.

Each ``TraceSpec`` records the on-disk path (always container-relative
because all known traces are written inside the runtime), the
emitting patch id, a one-line description, the activation env flag,
and a category for grouping. Adding a new trace = appending one
entry — no other code change required.

The container path convention is fixed at ``/tmp/genesis_*`` for now;
operators inspect them via ``docker exec <container> ls -la /tmp/`` or
``docker cp``. ``sndr trace collect`` will wrap that workflow in
plan-§6.H7.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


__all__ = [
    "TraceSpec",
    "TRACE_CATALOG",
    "TRACE_CATEGORIES",
    "find_by_id",
    "find_by_patch",
    "iter_by_category",
]


# Categories used by `sndr trace list` for grouping. The string values
# double as the user-visible labels.
TRACE_CATEGORIES = (
    "boot",          # boot-time apply / install logs
    "acceptance",    # spec-decode acceptance traces
    "kernel",        # Triton / CUDA kernel-firing traces
    "kv_write",      # KV cache write-path traces
    "routing",       # spec-decode + TQ route selection traces
    "mtp",           # MTP draft / verify lifecycle traces
    "oracle",        # oracle-acceptance probes (research traces)
    "tq_forward",    # TurboQuant forward-pass diagnostics
)


@dataclass(frozen=True)
class TraceSpec:
    """Single known diagnostic trace surface.

    Fields:
      id           — operator-facing short id; stable across edits.
      container_path — path on the container's FS where the trace lands.
      patch_id     — the registry id (e.g. ``PN248``) that emits this.
      enable_env   — env var that activates emission; ``None`` if the
                     trace is always written (boot logs).
      category     — one of ``TRACE_CATEGORIES``.
      description  — single line shown by ``sndr trace list``.
    """
    id: str
    container_path: str
    patch_id: str
    enable_env: Optional[str]
    category: str
    description: str


# Order = display order in `sndr trace list`. Group by category first,
# then chronological patch id within a category — matches how an
# operator typically scans the surface.
TRACE_CATALOG: tuple[TraceSpec, ...] = (
    # ── boot ────────────────────────────────────────────────────────
    TraceSpec(
        id="boot",
        container_path="/tmp/genesis_boot.log",
        patch_id="(launcher)",
        enable_env=None,
        category="boot",
        description=(
            "Genesis patch-apply boot log — captured by the launcher "
            "via `python3 -m sndr.apply 2>&1 | tee /tmp/"
            "genesis_boot.log` before `vllm serve` execs. Lists every "
            "patch's apply() result (applied / skipped / failed)."
        ),
    ),
    # ── acceptance ──────────────────────────────────────────────────
    TraceSpec(
        id="pn248_acceptance",
        container_path="/tmp/genesis_pn248_acceptance_trace.log",
        patch_id="PN248",
        enable_env="GENESIS_ENABLE_PN248_ACCEPTANCE_TRACE",
        category="acceptance",
        description=(
            "Per-step MTP acceptance trace — draft_token_ids vs "
            "target_argmax + accept/reject mask. Surfaces "
            "cross-quantization verifier-loop hypotheses (Hypothesis D)."
        ),
    ),
    TraceSpec(
        id="pn258_oracle",
        container_path="/tmp/genesis_pn258_oracle.txt",
        patch_id="PN258",
        enable_env="GENESIS_ENABLE_PN258_ORACLE_ACCEPTANCE",
        category="oracle",
        description=(
            "Pass 1 oracle output — one int per line; consumed by "
            "Pass 2 to gate ground-truth acceptance per draft slot."
        ),
    ),
    TraceSpec(
        id="pn258_oracle_trace",
        container_path="/tmp/genesis_pn258_oracle_trace.log",
        patch_id="PN258",
        enable_env="GENESIS_ENABLE_PN258_ORACLE_ACCEPTANCE",
        category="oracle",
        description=(
            "Pass 2 detailed oracle-acceptance trace — per-call "
            "acceptance breakdown when the oracle txt is loaded."
        ),
    ),
    # ── kernel ──────────────────────────────────────────────────────
    TraceSpec(
        id="pn260_kernel",
        container_path="/tmp/genesis_pn260_kernel_trace.log",
        patch_id="PN260",
        enable_env="GENESIS_ENABLE_PN260_TQ_KERNEL_TRACE",
        category="kernel",
        description=(
            "TurboQuant decode kernel firing trace — every kernel "
            "invocation with shape + decode-cache-hit + write-mode tag."
        ),
    ),
    TraceSpec(
        id="pn254_fire",
        container_path="/tmp/genesis_pn254_fire.log",
        patch_id="PN254",
        enable_env="GENESIS_ENABLE_PN254_TQ_FIRE_TRACE",
        category="kernel",
        description=(
            "PN254 kernel-fire trace — companion to PN260 in the "
            "PR42637 overlay tree; records call frequency + tail tokens."
        ),
    ),
    # ── kv_write ────────────────────────────────────────────────────
    TraceSpec(
        id="pn255_kv_write",
        container_path="/tmp/genesis_pn255_kv_write.log",
        patch_id="PN255",
        enable_env="GENESIS_ENABLE_PN255_KV_WRITE_TRACE",
        category="kv_write",
        description=(
            "KV-write-path trace — per-block write sites (4 emitters "
            "in turboquant_attn.py) with block-id + position + dtype."
        ),
    ),
    # ── routing ─────────────────────────────────────────────────────
    TraceSpec(
        id="pn256_route",
        container_path="/tmp/genesis_pn256_route.log",
        patch_id="PN256",
        enable_env="GENESIS_ENABLE_PN256_KPLUS1_RAW_KV",
        category="routing",
        description=(
            "K+1 raw-KV route trace — every dispatch into the K+1 "
            "branch (vs MTP draft-step branch) with batch shape + token "
            "indices."
        ),
    ),
    TraceSpec(
        id="pn261_tq_impl_init",
        container_path="/tmp/genesis_pn261_tq_impl_init.log",
        patch_id="PN261",
        enable_env="GENESIS_ENABLE_PN261_TQ_NATIVE_CACHE_ASSERT",
        category="routing",
        description=(
            "TurboQuant Impl-init audit trace — records every "
            "AttentionImpl construction with backend name + tier "
            "(turboquant_attn vs native FA)."
        ),
    ),
    # ── mtp ─────────────────────────────────────────────────────────
    TraceSpec(
        id="pn241_mtp",
        container_path="/tmp/genesis_pn241_mtp_trace.log",
        patch_id="PN241",
        enable_env="GENESIS_ENABLE_PN241_MTP_TRACE",
        category="mtp",
        description=(
            "MTP draft+verify lifecycle trace — per-call "
            "num_speculative_tokens / decode_buffer state / accept rate."
        ),
    ),
    # ── tq_forward ──────────────────────────────────────────────────
    TraceSpec(
        id="tq_forward",
        container_path="/tmp/genesis_tq_forward.log",
        patch_id="(turboquant_attn overlay)",
        enable_env="GENESIS_ENABLE_TQ_FORWARD_TRACE",
        category="tq_forward",
        description=(
            "TurboQuant forward-pass diagnostic — entry / exit per "
            "attention layer; large file, opt-in only for hot-path "
            "investigations."
        ),
    ),
)


def find_by_id(trace_id: str) -> Optional[TraceSpec]:
    """Return the catalog entry whose ``id`` matches exactly, else None."""
    for spec in TRACE_CATALOG:
        if spec.id == trace_id:
            return spec
    return None


def find_by_patch(patch_id: str) -> tuple[TraceSpec, ...]:
    """All catalog entries emitted by the given patch.

    A single patch may emit multiple traces (e.g. PN258 emits both
    ``pn258_oracle`` and ``pn258_oracle_trace``). Returns an empty
    tuple when no trace matches.
    """
    return tuple(spec for spec in TRACE_CATALOG if spec.patch_id == patch_id)


def iter_by_category() -> dict[str, tuple[TraceSpec, ...]]:
    """Group the catalog by category. Order of returned categories
    matches ``TRACE_CATEGORIES``; missing-from-catalog categories map
    to empty tuples so callers can iterate the canonical order without
    a guard."""
    out: dict[str, tuple[TraceSpec, ...]] = {}
    for cat in TRACE_CATEGORIES:
        out[cat] = tuple(s for s in TRACE_CATALOG if s.category == cat)
    return out
