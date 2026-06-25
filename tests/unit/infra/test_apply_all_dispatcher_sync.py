# SPDX-License-Identifier: Apache-2.0
"""Pin the apply_all.py ↔ dispatcher.py PATCH_REGISTRY sync contract.

For every patch that has an `@register_patch` + `apply_patch_<id>_*`
function in `apply_all.py`, there must be a corresponding entry in
`PATCH_REGISTRY` in `dispatcher.py` (and vice versa, with one
documented exception for P68/P69 sharing one apply function).

Why this matters: without this gate, the legacy P1–P46 patches drifted
out of the registry for an entire phase of development. New patches
can land in apply_all without dispatcher metadata (no env_flag, no
schema validation, invisible to `genesis explain` and `genesis list`).
This test catches that drift on every commit.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
# vllm/_genesis/patches/apply_all.py (now a back-compat shim) moved to
# sndr/apply/_per_patch_dispatch.py in the v12 refactor
# (vllm/sndr_core → sndr). Test parses the new canonical home.
APPLY_ALL = (
    REPO_ROOT / "sndr" / "apply" / "_per_patch_dispatch.py"
)


# Documented exceptions where one `apply_patch_*` function registers
# multiple registry entries (or vice versa). Update this set ONLY when
# adding a new intentional asymmetry — every entry here is a deviation
# from the 1:1 contract and should have a clear reason.
_KNOWN_REGISTRY_ONLY = frozenset({
    # 2026-06-23 (dev148 spec-driven additions): these patches are wired via
    # ``apply_module`` (spec-driven from inception), NOT the legacy
    # ``apply_patch_*`` convention — a registry entry with no apply_patch_*
    # function is correct here, not drift.
    "G4_83", "G4_84", "G4_85", "PN394", "PN398", "PN399", "PN400",
    # P68 and P69 share `apply_patch_68_long_ctx_tool_adherence` — both
    # patches modify the same long-context tool-adherence middleware so
    # they ship as one wiring function. Registry tracks them separately
    # for `genesis explain` / docs disambiguation.
    "P69",
    # PN40-classifier is a sub-component of PN40 sub-D — it is wired
    # inside `apply_patch_N40_dflash_omnibus()` (the PN40 omnibus apply
    # path). Registered separately so the dispatcher v2 validator does
    # not warn when scheduler.py reports `PN40-classifier` as applied,
    # but no standalone wiring function exists.
    "PN40-classifier",
    # P51 is a runtime layer-level TQ-active library guard living in
    # kernels/dequant_buffer.py — no env toggle, no apply_patch_* needed.
    # Registered for visibility in `genesis explain` and audit tooling.
    "P51",
    # P102 is a diagnostic-only spec-decode metadata module
    # (vllm/_genesis/spec_meta.py). Activated by direct call from spec
    # paths when GENESIS_ENABLE_P102=1, not via apply_all wiring.
    # Registered for `genesis explain` visibility.
    "P102",
    # PN60 is a preflight DX validator wired into compat/doctor.py — runs
    # BEFORE vLLM loads, not via apply_all. Registered for visibility +
    # operator search-ability in `genesis explain PN60`.
    "PN60",
    # PN61/PN62 are now WIRED (apply_patch_N61_*, apply_patch_N62_*)
    # — left commented here in case future refactor decouples them from
    # apply_all. As of 2026-05-05 they have full wiring + opt-in env flag
    # + class-rebind apply() with idempotency markers; cross-rig validation
    # pending an actual qwen3_vl checkpoint reachable from a Genesis test rig.
    # PN63 is a gpu_profile advisory rule (lives in gpu_profile.py
    # PATCH_RECOMMENDATIONS). Suggest-only, not a runtime patch — so no
    # apply_patch_* function exists by design.
    "PN63",
    # PN64 is a Marlin MoE per-SM tuning placeholder for SM 12.0. The
    # actual config entry lives in kernels/marlin_tuning.py table; the
    # registry entry is for `genesis explain` visibility. Real wiring
    # only matters when SM 12.0 hardware is detected at boot.
    "PN64",
    # Sprint 4 / 2026-05-09: PN16_V6 is a sub-variant of PN16's family
    # (V3/V5/V7/V8 pattern). Apply path is
    # `apply_patch_N16_v6_streaming_truncator` — the dispatcher-sync
    # regex extracts only "N16" prefix, so PN16_V6 looks registry-only.
    # The function exists in apply_all.py; no asymmetry in practice.
    "PN16_V6",
    # Sprint 2.6 v2 — renamed to PN122 in §12 cleanup. Kept here for
    # historical context; the canonical entry is now PN122 below.
    # PN122 trace tooling — wired through observability dispatch helper,
    # not via apply_patch_* function in apply_all.py.
    "PN122",
    # Wave 10 backports / experimental patches without standalone wiring
    # function in apply_all.py — each one ships its own apply via the
    # dispatcher's per-patch overlay loader (`integrations/<family>/<id>.py`
    # defines apply() directly). Registry entries exist for dispatcher
    # visibility and proof-artefact tracking.
    "PN104", "PN105", "PN106",         # offload tier (CPU)
    "PN110", "PN111",                  # kv_cache experimental
    "PN116", "PN118", "PN119",         # turboquant wave 10 backports
    "PN125", "PN126", "PN127", "PN128",  # warmup orchestrator series
    "PN129", "PN130",                  # JIT + TQ decode warmup
    "PN132", "PN133", "PN134",         # correctness backports
    "PN200", "PN201", "PN202", "PN203",  # streaming runtime
    "PN71", "PN73", "PN91", "PN92", "PN97",  # legacy backports retired/merged
    # Synthetic workspace token — coordinator only, no apply path.
    "SNDR_WORKSPACE_001",
    # ─── G4_* Gemma 4 model subsystem (R3 audit 2026-05-21 onwards) ───
    # All G4_* patches use the dispatcher overlay-loader pattern
    # (apply_module → integrations/<family>/<id>.py with own apply()),
    # same precedent as the PN104+ block above. Registry entries exist
    # for `genesis explain`, doc generation, and audit tooling.
    "G4_01", "G4_02", "G4_03", "G4_04", "G4_05", "G4_06", "G4_07", "G4_08",
    "G4_09", "G4_10", "G4_11", "G4_12", "G4_13", "G4_14", "G4_15", "G4_16",
    "G4_17", "G4_18", "G4_19", "G4_19B", "G4_19C", "G4_23", "G4_24", "G4_25",
    # G4_26 (DiffusionGemma TP>1 vocab-sharded soft-embed all-gather,
    # backport vllm#45774, 2026-06-17) — TextPatcher via the _G4_PATCHES
    # dispatcher tuple, same overlay-loader pattern; no apply_patch_* hook.
    "G4_26",
    "G4_31", "G4_32",
    # PR42637 overlay verifier series (10 _G4_60* patches)
    "G4_60A", "G4_60B", "G4_60C", "G4_60D", "G4_60E", "G4_60G",
    "G4_60H", "G4_60K", "G4_60L",
    "G4_61", "G4_62",
    "G4_67", "G4_68", "G4_69",
    # PN259B/C alloc + routing variants (G4_70 family)
    "G4_70", "G4_70B", "G4_70C",
    # DFlash drafter rerouting + Triton kernels (G4_71..G4_78)
    "G4_71", "G4_71B", "G4_72", "G4_73", "G4_74", "G4_75", "G4_76", "G4_78",
    # G4_79 (TQ supports_mm_prefix, 2026-06-11) + G4_80 (fp8_e5m2 KV for
    # weight-only checkpoints, vllm#45040, 50-PR sweep wave 1) — same
    # dispatcher overlay-loader pattern as the rest of the G4_* block
    # (apply_module with own apply(); no apply_patch_* wiring by design).
    # G4_79's absence here predated the sweep — backfilled 2026-06-11.
    "G4_79", "G4_80",
    # ─── Spec-decode telemetry / safety opt-ins (R3 audit 2026-05-21) ─
    # PN256/PN261/PN262/PN262B/PN271/PN275 use overlay loader. PN274 is
    # coordinator-only (apply_module=None) for operator visibility.
    # PN282 (STAGE-6-HARDENING.2C 2026-05-28) — non-dispatcher boot
    # coordinator, registry-only.
    # PN283 (Phase 10.5 2026-06-01) — sibling coordinator of PN282:
    # prometheus_client multiprocess-dir bootstrap. Same boot pattern
    # (boots directly from sndr_core/__init__.py), registry-only, no
    # apply_patch_* wiring.
    "PN256", "PN261", "PN262", "PN262B", "PN271", "PN274", "PN275",
    "PN282", "PN283",
    # PN288 (qwen3_coder finish_reason override) + PN289 (Genesis
    # process-info Prometheus gauge) — both 2026-05-30 spec-driven
    # additions that ship via dispatcher overlay-loader rather than
    # apply_patch_* in apply_all.py; the registry entries document the
    # patches for `genesis explain` + audit tooling. Phase 10.5
    # enterprise sweep 2026-06-01 — carve-out documented here.
    "PN288", "PN289",
    # SNDR_MTP_DYNAMIC_K_001 (Sandermage adaptive K MTP proposer — port
    # of vllm#26504 DynamicProposer) — wired via @register_patch in
    # _per_patch_dispatch.py rather than via apply_patch_* in
    # apply_all.py (the patch_id pattern doesn't match the
    # apply_patch_<id>_* regex because SNDR_* IDs use the engine-tier
    # namespace, not the canonical P[N]?\d+ form). Phase 10.5
    # enterprise sweep 2026-06-01 — carve-out documented here.
    "SNDR_MTP_DYNAMIC_K_001",
    # SNDR_EAGLE3_AUX_HIDDEN_001 — Genesis-original Phase 7 EAGLE-3
    # model-side preparation. Apply path uses @register_patch in
    # _per_patch_dispatch.py (compound SNDR_* ID, not P[N]?\d+ form).
    "SNDR_EAGLE3_AUX_HIDDEN_001",
    # v11.1.0/v11.2.0 md5+full-file PoC siblings (Phase 6 P3.1) — wired
    # via @register_patch in _per_patch_dispatch.py using compound IDs
    # (PN<base>_V2_MD5_<file>). The regex captures only the base
    # numeric prefix (PN118 / PN79) so the compound IDs look registry-
    # only. Each compound ID has its own apply_patch_n<n>_v2_md5_*
    # function under the dispatch helper module.
    "PN118_V2_MD5_WORKSPACE",
    "PN118_V2_MD5_TURBOQUANT_ATTN",
    "PN79_V2_MD5_CHUNK",
    "PN79_V2_MD5_CHUNK_DELTA_H",
    # Fix-wire companions with compound IDs (2026-06-04/08) — each HAS a
    # real apply function in _per_patch_dispatch.py, but the dispatcher-
    # sync regex captures only the numeric base, so the compound registry
    # ID looks registry-only (same PN16_V6 precedent above):
    #   P18B_TEXT → apply_patch_18b_text_kernel_literals (regex → "P18b")
    #   P23_WIRE  → apply_patch_23_wire_marlin_fp32_reduce (regex → "P23")
    #   P29_HEAL  → apply_patch_29_heal_qwen3coder_index (regex → "P29")
    "P18B_TEXT", "P23_WIRE", "P29_HEAL",
    # PN353B (TQ prefill CUDA-graph capture safety, vllm#43747 backport)
    # and PN357 (remapped greedy draft selection, vllm#43349 vendor) —
    # dispatcher overlay-loader pattern (`apply_module` points at
    # patches/<family>/<id>.py with its own apply()), same precedent as
    # the PN104+ block above. No apply_patch_* wiring by design.
    "PN353B", "PN357",
    # PN371 (deferred ref-pinned encoder-cache eviction, vllm#45199) +
    # PN373 (parallel_tool_calls explicit null != false, vllm#44955) —
    # 2026-06-11 50-PR sweep wave 1, spec-driven from inception (same
    # class as PN353B/PN357; wave siblings PN370/PN372/PN374/PN375 DID
    # get @register_patch parking-lot hooks and are not listed here).
    "PN371", "PN373",
    # 2026-06-13 50-PR sweep WAVE 2 — nine spec-driven-from-inception
    # vendors/blueprints (apply_module with own apply(), no
    # apply_patch_* hook; same class as PN371/PN373/G4_79/G4_80). The
    # tenth wave-2 patch PN377 DID get a legacy @register_patch
    # parking-lot hook (apply_patch_N377_*) so it is NOT listed here.
    "P88",     # prefix-cache stats retry de-dup (rewrite of vllm#45202)
    "PN358",   # FULL cudagraph forward-context refresh (vllm#44868)
    "PN376",   # fp8 modules_to_not_convert substring (vllm#44628)
    "PN378",   # recovered-token vocab-pad -inf mask (vllm#45060)
    "PN379",   # LoadConfig/DefaultModelLoader fail-fast (vllm#45196)
    "PN380",   # Qwen3.5/3.6 MTP pre-fused expert loader (vllm#44943)
    "PN381",   # allowed_token_ids spec-decode metadata (vllm#44742)
    "PN382",   # DecodeBenchConnector hybrid per-block fill (vllm#45080)
    "G4_81",   # TQ multi-query DIRECT decode routing (vllm#45144 blueprint)
    "PN383",   # KV-offload + MTP segfault gate (vllm#44784)
    # 2026-06-13 50-PR sweep BATCH-2 WAVE 1 — five spec-driven-from-
    # inception LIVE-bug vendors (apply_module with own apply(), no
    # apply_patch_* hook; same class as PN371/PN373/PN383). All opt-in.
    "PN384",   # Eagle/MTP prefix-cache prefill fix (vllm#44986)
    "PN385",   # forced-named empty-params tool schema -> object (vllm#45290)
    "PN386",   # required-tool streaming brace string-awareness (vllm#45389)
    "PN387",   # reject degenerate structured_outputs DoS guard (vllm#45346)
    "PN388",   # mamba-block-aligned intermediate prefill split (vllm#45477)
    # 2026-06-13 50-PR sweep BATCH-3 — four more spec-driven-from-inception
    # vendors (apply_module with own apply(), no apply_patch_* hook; same
    # class as PN383-PN388). All opt-in.
    "PN389",   # XGrammar grammar-compilation timeouts (vllm#45390)
    "PN390",   # streaming-LSE rejection sampler (vllm#45369)
    "PN391",   # /health/decode forward-progress watchdog (vllm#45453)
    "P89",     # reasoning_tokens in chat usage object (vllm#45471)
    "PN392",   # qwen3_coder streaming tool-call coalescing (dev491 fix)
    # G4_T1 (Gemma4 tool-parser PR #42006 vendor marker) — operator-
    # side bind-mount overlay; no apply_patch_* wiring by design
    # (registered only for `genesis explain` + audit visibility of the
    # mount state via the GENESIS_INFO_* INFO-semantic env flag).
    "G4_T1",
    # ─── Misc backports without per_patch_dispatch wiring ─────────────
    # P8 retired tombstone (kv_hybrid_reporting — registered for audit
    # trail only, retired lifecycle).
    "P8",
    # ─── 2026-06-17 session additions ─────────────────────────────────
    # G4_82 (TQ prefill SDPA head-dim) — G4_* dispatcher overlay-loader
    # pattern, same as the whole G4_* block above; no apply_patch_* hook.
    "G4_82",
    # PN252 (mrope prompt-embeds DoS guard), PN353A (TQ MetadataBuilder
    # workspace reserve — spec-only since §2.2.A 2026-06-16), PN517
    # (init-snapshot-before-NCCL): all dispatcher overlay-loader
    # (apply_module points at the module's own apply(); no apply_patch_*
    # hook), same class as the PN104+/PN353B blocks above.
    "PN252", "PN353A", "PN517",
    # PN-FP8MOE-KPAD (FP8-core backport of vllm#45703) DOES have a real
    # wiring hook (`apply_patch_pn_fp8moe_kpad_marlin_moe` in
    # _per_patch_dispatch.py), but the dispatcher-sync regex
    # `^apply_patch_([NM]?\d+...)` cannot capture the lowercase hyphenated
    # ID — same regex-mismatch class as SNDR_*/P18B_TEXT above. The hook
    # IS wired; this is a name-form carve-out, NOT missing wiring.
    # (Rename-to-PN<num> candidate to retire the hyphenated-ID friction
    # that already required a shadow.py _LEGACY_NAME_TO_PATCH_ID entry.)
    "PN-FP8MOE-KPAD",
    # ─── 2026-06-25 dev424 integration backlog ────────────────────────────
    # PN401 (TQ prefill continuation guard, vllm#46461), PN402 (sanitize
    # invalid MTP draft ids, vllm#46574), PN518 (INCConfig hybrid INT4+FP8
    # detect, vllm#46322), PN519 (SWA tile-loop first_allowed_key base,
    # vllm#46087): all spec-driven from inception (apply_module points at the
    # module's own apply() returning (status, reason); no apply_patch_* legacy
    # hook), same dispatcher overlay-loader class as the PN398/PN399/PN400
    # block above. Applied at legacy boot via _run_spec_only_supplement.
    "PN401", "PN402", "PN518", "PN519",
})

_KNOWN_APPLY_ONLY: frozenset[str] = frozenset({
    # 2026-06-19/20 consolidations: these patches were ABSORBED into a merged
    # module (P59/PN51 -> P61b; P61c/PN56 -> P64; PN29 -> PN298; PN369 -> P71)
    # and their PATCH_REGISTRY entries removed, but their apply_patch_* legacy
    # boot functions are retained in apply_all.py for boot continuity (mapped to
    # the merged primary in apply/shadow.py _LEGACY_NAME_TO_PATCH_ID).
    "P59", "PN51", "P61c", "PN56", "PN29", "PN369",
    # No documented exceptions yet — every apply_patch_* should have a
    # registry entry. Add an ID here with a comment if you intentionally
    # ship a wiring function without dispatcher metadata.
})


def _extract_apply_patch_ids() -> set[str]:
    """Parse apply_all.py and return all patch IDs from
    `apply_patch_<id>_*` function names. Format: prefix-letter `P` plus
    the captured ID, so `apply_patch_67b_*` → `P67b`,
    `apply_patch_N32_*` → `PN32`."""
    tree = ast.parse(APPLY_ALL.read_text(encoding="utf-8"))
    ids: set[str] = set()
    pattern = re.compile(r"^apply_patch_([NM]?\d+[a-zA-Z]?)(?:_|$)")
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith(
            "apply_patch_"
        ):
            m = pattern.match(node.name)
            if m:
                ids.add("P" + m.group(1))
    return ids


def _load_registry_ids() -> set[str]:
    """Load PATCH_REGISTRY from dispatcher.py without importing the
    full vllm package (so the test runs in CI without GPU)."""
    from sndr.dispatcher import PATCH_REGISTRY
    return set(PATCH_REGISTRY.keys())


@pytest.fixture(scope="module")
def apply_ids() -> set[str]:
    return _extract_apply_patch_ids()


@pytest.fixture(scope="module")
def registry_ids() -> set[str]:
    return _load_registry_ids()


def test_no_registry_entries_without_apply_function(
    apply_ids: set[str], registry_ids: set[str]
) -> None:
    """Every PATCH_REGISTRY entry must have a corresponding
    `apply_patch_<id>_*` function in apply_all.py — except for
    documented sharing cases like P68/P69."""
    registry_only = (registry_ids - apply_ids) - _KNOWN_REGISTRY_ONLY
    assert registry_only == set(), (
        f"PATCH_REGISTRY has {len(registry_only)} entries with no "
        f"apply_patch_* function: {sorted(registry_only)}\n"
        "Either add the wiring function in apply_all.py, or document "
        "the asymmetry in _KNOWN_REGISTRY_ONLY in this test."
    )


def test_no_apply_functions_without_registry_entry(
    apply_ids: set[str], registry_ids: set[str]
) -> None:
    """Every `apply_patch_<id>_*` function in apply_all.py must have a
    corresponding PATCH_REGISTRY entry. Without metadata the patch is
    invisible to `genesis explain` / schema validation / lifecycle
    audit / opt-in env discovery."""
    apply_only = (apply_ids - registry_ids) - _KNOWN_APPLY_ONLY
    assert apply_only == set(), (
        f"apply_all.py has {len(apply_only)} apply_patch_* functions "
        f"with no PATCH_REGISTRY entry: {sorted(apply_only)}\n"
        "Add an entry to PATCH_REGISTRY in dispatcher.py with at least "
        "title / env_flag / default_on / category / lifecycle, or "
        "document the asymmetry in _KNOWN_APPLY_ONLY in this test."
    )


def test_documented_exceptions_actually_present(
    registry_ids: set[str], apply_ids: set[str]
) -> None:
    """Sanity check: every ID in _KNOWN_REGISTRY_ONLY must really be in
    the registry but NOT in apply (else the exception is stale)."""
    for pid in _KNOWN_REGISTRY_ONLY:
        assert pid in registry_ids, (
            f"_KNOWN_REGISTRY_ONLY contains {pid!r} which is no longer "
            f"in PATCH_REGISTRY — remove the stale exception"
        )
        assert pid not in apply_ids, (
            f"_KNOWN_REGISTRY_ONLY contains {pid!r} but apply_all.py "
            f"now has an apply_patch_* function — remove the exception"
        )
    for pid in _KNOWN_APPLY_ONLY:
        assert pid in apply_ids, (
            f"_KNOWN_APPLY_ONLY contains {pid!r} which is no longer "
            f"in apply_all.py — remove the stale exception"
        )
        assert pid not in registry_ids, (
            f"_KNOWN_APPLY_ONLY contains {pid!r} but PATCH_REGISTRY "
            f"now has an entry — remove the exception"
        )
