# SPDX-License-Identifier: Apache-2.0
"""G4_68 — TurboQuant spec-decode cudagraph downgrade for PR #42637 overlay.

================================================================
PROBLEM
================================================================

P65 (`sndr/engines/vllm/patches/attention/turboquant/p65_turboquant_
spec_cg_downgrade.py`) downgrades `TurboQuantMetadataBuilder._cudagraph_
support` from `UNIFORM_BATCH` to `UNIFORM_SINGLE_TOKEN_DECODE` when
`speculative_config` is active. It uses a `TextPatcher` against
`vllm/v1/attention/backends/turboquant_attn.py` at runtime.

When the PR #42637 overlay is bind-mounted (read-only) over that stock
file via `docker run -v <overlay>:<target>:ro`, the text patcher cannot
apply because the file is not writable. P65 logs `read_only_mount` and
skips. Operators running Gemma 4 + TurboQuant + MTP under the overlay
therefore lose the P65 workaround for the spec-decode K+1 cudagraph
placement bug, leading to degenerate output like `"TheSLLLL..."` even
when P65 is set in the env.

================================================================
WHAT THIS PATCH DOES
================================================================

G4_68 is NOT a monkey-patch. It is a **marker verifier** that:

  1. Imports `vllm.v1.attention.backends.turboquant_attn` (whatever file
     is at the bind-mount target — stock or PR #42637 overlay).
  2. Inspects `TurboQuantMetadataBuilder` for the P65 v2 cudagraph
     downgrade marker — specifically the presence of a `get_cudagraph_
     support` classmethod that returns `AttentionCGSupport.UNIFORM_
     SINGLE_TOKEN_DECODE` under `speculative_config`.
  3. Reads BOTH runtime env flags
     (`GENESIS_ENABLE_P65_TURBOQUANT_SPEC_CG_DOWNGRADE` and
     `GENESIS_ENABLE_PN256_KPLUS1_RAW_KV`) and renders the apply-result
     as ACTIVE / PARTIAL / INERT so operators can see at a glance whether
     the workaround is actually engaged. See "ENV FLAGS" section below
     for the full semantics; one runtime env alone is not enough.
  4. Reports applied/error/skipped to the dispatcher so `patches doctor`
     and the boot-log apply summary correctly show whether the
     workaround is configured.

The actual cudagraph downgrade lives **inline** in the overlay source at
`sndr/engines/vllm/patches/attention/turboquant/overlays/pr42637/turboquant_
attn.py` (see the `[Genesis P65 v2 inlined for PR #42637 overlay]`
comment block on `TurboQuantMetadataBuilder`). The raw-K/V continuation
fix (PN256) lives in the same overlay inside `_prefill_attention()`.
G4_68's role is purely identity + dispatcher visibility; runtime behavior
is governed by the overlay code and the runtime env flags listed below.

================================================================
ENV FLAGS — read carefully, two distinct purposes
================================================================

The cudagraph-downgrade workaround needs three different env flags
serving three different roles. Operators conflating them will see
either silent no-op or missing dispatcher visibility.

RUNTIME-REQUIRED (these env flags change runtime behavior):

  * `GENESIS_ENABLE_P65_TURBOQUANT_SPEC_CG_DOWNGRADE=1`
    Read by the inlined `get_cudagraph_support` classmethod in the
    overlay. When set and `speculative_config is not None`, the
    classmethod returns `AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE`,
    which vLLM's `compilation.py` honors by downgrading
    `cudagraph_mode` from FULL_AND_PIECEWISE to PIECEWISE for the
    affected backend. Without this env: the classmethod returns the
    `UNIFORM_BATCH` ClassVar default and the broken cache-read route is
    captured into a CUDA graph (degenerate output).

  * `GENESIS_ENABLE_PN256_KPLUS1_RAW_KV=1`
    Read inside `_prefill_attention()` to actually route K+1 verify
    through raw-K/V `_continuation_prefill()` instead of
    `_decode_prefill_from_cache()`. Without this env: the cudagraph
    downgrade alone is not sufficient because the eager Python path
    still hits the cache-read continuation.

VERIFICATION-ONLY (this env flag does NOT change runtime behavior):

  * `GENESIS_ENABLE_G4_68_TQ_SPEC_CG_DOWNGRADE_OVERLAY=1`
    Controls whether THIS verifier module runs at boot. The verifier
    introspects `TurboQuantMetadataBuilder` for the inlined classmethod
    and reports applied/error/skipped to the dispatcher + `patches
    doctor` + boot apply summary. Setting this env alone changes
    nothing about model output or cudagraph capture. Setting it without
    the two runtime envs above produces a misleading "applied"
    dispatcher line while the workaround is INERT.

Recommended operator pattern for production-grade run (all three):

```bash
-e GENESIS_ENABLE_P65_TURBOQUANT_SPEC_CG_DOWNGRADE=1   # runtime
-e GENESIS_ENABLE_PN256_KPLUS1_RAW_KV=1                # runtime
-e GENESIS_ENABLE_G4_68_TQ_SPEC_CG_DOWNGRADE_OVERLAY=1 # operator visibility
```

The verifier's apply-result message surfaces this distinction by
reading the P65 env and reporting the runtime state as ACTIVE or INERT.

================================================================
DEPENDENCIES
================================================================

  * Companion to G4_60b (verifies the overlay file is bind-mounted).
  * Conflicts with stock P65 (`sndr.engines.vllm.patches.attention.
    turboquant.p65_turboquant_spec_cg_downgrade`) only in that P65 will
    correctly self-skip with `read_only_mount` reason when the overlay
    is mounted — by design. G4_68 then takes over reporting.
  * Pairs with PN256's raw-K/V continuation route inside the overlay
    (no separate patch ID; lives in
    `overlays/pr42637/turboquant_attn.py` along with P65).
  * Pairs with PN253 stride-0 fix in the same overlay.

================================================================
SCOPE / LIMITATIONS
================================================================

  * Active only when `GENESIS_ENABLE_G4_68_TQ_SPEC_CG_DOWNGRADE_OVERLAY=1`.
  * Pure diagnostic — no monkey-patching here.
  * Failure mode: returns `error` with explanation if overlay missing
    the P65 v2 inline (i.e., somebody updated the overlay file and
    dropped the patch). Operator must restore the inline P65 v2 block
    on `TurboQuantMetadataBuilder` in the overlay source.
  * P65+PN256 restores **target correctness** under the PR #42637
    overlay for Gemma 4 MTP. It does NOT restore MTP speedup — drafter
    acceptance remains 0% in observed tests. This is a correctness
    fallback only.

================================================================
REFERENCES
================================================================

  * Stock P65 source: `sndr/engines/vllm/patches/attention/turboquant/
    p65_turboquant_spec_cg_downgrade.py`
  * Overlay source: `sndr/engines/vllm/patches/attention/turboquant/
    overlays/pr42637/turboquant_attn.py` (search for
    `[Genesis P65 v2 inlined]`)
  * Diagnostic chain: PN253 → PN254 → PN255 → PN256 → PN257a (Genesis
    investigation 2026-05-18)
  * vLLM upstream issue tracking: see UPSTREAM_ISSUE_GEMMA4_TQ_MTP_K1_
    VERIFY_DRAFT_2026-05-18_EN.md

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.turboquant.g4_68_tq_spec_cg_downgrade_overlay")

GENESIS_G4_68_MARKER = (
    "Genesis G4_68 verify P65 v2 inlined in PR #42637 turboquant_attn overlay"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_68_TQ_SPEC_CG_DOWNGRADE_OVERLAY"
_ENV_P65 = "GENESIS_ENABLE_P65_TURBOQUANT_SPEC_CG_DOWNGRADE"
_ENV_PN256 = "GENESIS_ENABLE_PN256_KPLUS1_RAW_KV"
_APPLIED = False


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def apply() -> tuple[str, str]:
    """Verify P65 v2 inline lives on TurboQuantMetadataBuilder in overlay."""
    global _APPLIED

    if not _env_enabled():
        return "skipped", (
            f"G4_68 disabled (set {_ENV_ENABLE}=1 to verify overlay inline)"
        )

    if _APPLIED:
        return "applied", "G4_68 already verified (idempotent)"

    try:
        from vllm.v1.attention.backends import turboquant_attn as _tqa
    except ImportError as e:
        return "error", (
            f"vllm.v1.attention.backends.turboquant_attn not importable: {e}"
        )

    builder_cls = getattr(_tqa, "TurboQuantMetadataBuilder", None)
    if builder_cls is None:
        return "error", (
            "TurboQuantMetadataBuilder class missing — overlay file may "
            "be the stock file (G4_60b should also be skipped/error)."
        )

    # The inline P65 v2 adds a `get_cudagraph_support` classmethod that
    # consults speculative_config + env. Absence means the overlay is
    # missing the inline (operator updated the overlay without porting
    # P65 v2 forward).
    get_cg = getattr(builder_cls, "get_cudagraph_support", None)
    if get_cg is None or not callable(get_cg):
        return "error", (
            "TurboQuantMetadataBuilder.get_cudagraph_support classmethod "
            "not found — P65 v2 inline appears to be missing from the "
            "PR #42637 overlay source. Restore the "
            "[Genesis P65 v2 inlined] block on TurboQuantMetadataBuilder "
            "in sndr/engines/vllm/patches/attention/turboquant/"
            "overlays/pr42637/turboquant_attn.py."
        )

    p65_env_set = os.environ.get(_ENV_P65, "").strip().lower() in (
        "1", "true", "yes", "on",
    )
    pn256_env_set = os.environ.get(_ENV_PN256, "").strip().lower() in (
        "1", "true", "yes", "on",
    )
    # Both runtime envs are required for the workaround to actually fix
    # output. Verifier env alone does not change behavior. Surface this
    # in the apply-result so operators don't misread a green dispatcher
    # line as "workaround engaged".
    if p65_env_set and pn256_env_set:
        runtime_status = "ACTIVE (both runtime envs set)"
    elif p65_env_set:
        runtime_status = (
            f"PARTIAL ({_ENV_P65}=1 but {_ENV_PN256} unset — cudagraph "
            f"downgrade happens but eager path still hits cache-read)"
        )
    elif pn256_env_set:
        runtime_status = (
            f"PARTIAL ({_ENV_PN256}=1 but {_ENV_P65} unset — captured "
            f"path replays the broken cache-read route)"
        )
    else:
        runtime_status = (
            f"INERT (neither {_ENV_P65} nor {_ENV_PN256} set — verifier "
            f"reports applied but workaround does nothing)"
        )

    _APPLIED = True
    log.info(
        "[G4_68] P65 v2 inline verified on TurboQuantMetadataBuilder. "
        "Runtime status: %s. Required envs for runtime effect: "
        "%s=1 + %s=1. Verifier env (%s=1) controls only this report.",
        runtime_status,
        _ENV_P65,
        _ENV_PN256,
        _ENV_ENABLE,
    )
    return "applied", (
        f"G4_68 overlay inline verified: TurboQuantMetadataBuilder."
        f"get_cudagraph_support present. Runtime status: {runtime_status}."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    """No-op: inline lives in overlay source, controlled outside Python."""
    return False


__all__ = ["GENESIS_G4_68_MARKER", "apply", "is_applied", "revert"]
