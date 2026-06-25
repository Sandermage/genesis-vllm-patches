# SPDX-License-Identifier: Apache-2.0
"""CI-wide regression guard — no DRIFT entries in the legacy↔spec
divergence audit (v12.0.0 readiness gate).

Why this matters
----------------

The dispatcher migration plan (master plan §3 Phase 4, §13 follow-up #7)
targets v12.0.0 to flip the default apply path from legacy
`@register_patch(...)` iteration to spec-driven
`iter_patch_specs()` dispatch. For that flip to be safe, every patch
the legacy path applies must be either:

  (a) ALSO applied by spec-driven path (covered by an iter-yielded
      spec with apply_module set), OR
  (b) DELIBERATELY skipped by spec-driven path via the
      `GENESIS_LEGACY_*` env_flag policy (spec entry exists with
      apply_module=None and env_flag starts with GENESIS_LEGACY_).

Any legacy-only patch that doesn't fit (b) is `DRIFT` — a real
migration gap that would silently disable the patch on v12.0.0 flip.

v11.3.0 baseline (BUG #6 audit, this commit): 0 drift entries,
7 intentional legacy (P1, P17, P18b, P20, P23, P29, P32). The
audit script's `_diff_matrices` classifies them; this test pins
`drift_count == 0`.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Status: v11.3.0+ regression guard.
"""
from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]


def _import_audit_module():
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    import scripts.audit_legacy_vs_spec_driven_apply_matrix as A
    return A


def test_legacy_only_drift_zero():
    """v12.0.0 readiness invariant: no legacy-registered patch lacks
    EITHER a matching spec apply_module OR a policy-tagged spec
    informational entry (GENESIS_LEGACY_* env)."""
    audit = _import_audit_module()
    from sndr.dispatcher.registry import PATCH_REGISTRY
    legacy = audit._enumerate_legacy_path()
    spec = audit._enumerate_spec_driven_path(PATCH_REGISTRY)
    diff = audit._diff_matrices(legacy, spec, PATCH_REGISTRY)
    assert diff["legacy_only_drift_count"] == 0, (
        f"v12.0.0 readiness regression: {diff['legacy_only_drift_count']} "
        f"legacy-only entries lack a spec policy entry. On v12.0.0 flip "
        f"these patches would silently stop applying. "
        f"Drift IDs: {diff['legacy_only_drift_ids']}. "
        f"Fix by EITHER adding apply_module to the spec entry (real "
        f"migration) OR marking the spec entry with env_flag prefix "
        f"GENESIS_LEGACY_<ID> + apply_module=None + lifecycle=legacy "
        f"(intentional policy deferral)."
    )


def test_v12_0_safe_flag_true():
    """Pin v12_0_safe (drift-only flag) at True. order_divergent is
    tracked separately via v12_0_strict_order (False at v11.3.0
    baseline — see scripts/audit_legacy_vs_spec_driven_apply_matrix.py
    for the dict-insertion-order vs decorator-call-order analysis).
    """
    audit = _import_audit_module()
    from sndr.dispatcher.registry import PATCH_REGISTRY
    legacy = audit._enumerate_legacy_path()
    spec = audit._enumerate_spec_driven_path(PATCH_REGISTRY)
    diff = audit._diff_matrices(legacy, spec, PATCH_REGISTRY)
    assert diff["v12_0_safe"] is True, (
        f"v12_0_safe became False — drift introduced. "
        f"legacy_only_drift_count={diff['legacy_only_drift_count']}"
    )


def test_intentional_legacy_baseline():
    """Document the intentional legacy baseline. Any NEW entry
    here means a NEW patch was registered legacy-only — confirm it's
    intentional and update this baseline."""
    audit = _import_audit_module()
    from sndr.dispatcher.registry import PATCH_REGISTRY
    legacy = audit._enumerate_legacy_path()
    spec = audit._enumerate_spec_driven_path(PATCH_REGISTRY)
    diff = audit._diff_matrices(legacy, spec, PATCH_REGISTRY)
    # Baseline at v11.3.0 was 7 entries (P1, P17, P18b, P20, P23, P29,
    # P32). The 2026-06-04 fix-wire session shipped real patch files
    # for three of them (P18B_TEXT / P23_WIRE / P29_HEAL), so the
    # spec path now auto-derives apply_module for P18b / P23 / P29 —
    # they migrated out of legacy-only into BOTH-paths coverage (the
    # "removal = migrated, good" case below). 4 remain.
    intentional = set(diff["legacy_only_intentional_ids"])
    expected = {"P1", "P17", "P20", "P32"}
    assert intentional == expected, (
        f"Intentional legacy set changed:\n"
        f"  added: {sorted(intentional - expected)}\n"
        f"  removed: {sorted(expected - intentional)}\n"
        f"If intentional: update `expected` set in this test.\n"
        f"If a removal: that patch was migrated (good — verify "
        f"apply_module set + lifecycle != 'legacy')."
    )


def test_spec_only_truly_orphan_baseline():
    """v11.3.0 baseline for spec-only-truly-orphan patches — those
    that apply ONLY through the spec-driven path (SNDR_APPLY_VIA_SPECS=1).

    Why pin: an operator who enables one of these via env (e.g.
    `GENESIS_ENABLE_PN289_PROCESS_INFO=1`) without also setting
    `SNDR_APPLY_VIA_SPECS=1` gets a silent no-op. Bug class #5 + #7
    from the v11.3.0 audit-driven discovery sweep.

    Either:
      (a) add a legacy `@register_patch` hook in
          `sndr/apply/_per_patch_dispatch.py`, or
      (b) add a manual orchestration call in
          `sndr/plugin.py`, or
      (c) accept the orphan state — document in this baseline so a
          NEW orphan addition (not part of baseline) triggers review.

    Baseline at v11.3.0 (BUG #7 audit):
      - PN288: tool finish_reason override (not in any production YAML)
      - PN289: §6.H10 Prometheus process_info gauge (BUG #4 just-fixed
               tier=community; not in any production YAML)
      - PN40-classifier: DFlash sub-D workload classifier hook
        (env GENESIS_ENABLE_PN40_DFLASH_OMNIBUS — IS in production
        DFlash YAMLs, but the same env enables PN40 omnibus which
        has a legacy hook; classifier currently routes through there
        too — BUG #8 audit follow-up).

    v12 baseline expansion (accepted orphan state, option (c)):
      - The v11 import-time manual orchestration block
        (`vllm/sndr_core/__init__.py`, ~41 env-gated `mod.apply()`
        calls) was archived in commit 6bf9c04c
        (`sndr_private/archive/v11_vllm_sndr_core_shims/`) and NOT
        carried into `sndr/`. Of its modules, only G4_19/G4_19b kept
        manual orchestration (selective apply in `sndr/plugin.py`);
        the rest now apply ONLY via SNDR_APPLY_VIA_SPECS=1:
        G4_31, G4_32, G4_60A-L Gemma4 TQ overlays, G4_67, G4_69,
        G4_71-76 drafter patches, G4_78 (retired bridge), and the
        spec-decode probes PN262/PN262B/PN271. All are env-gated
        diagnostics/Gemma4-specific patches absent from production
        Qwen YAMLs — silent no-op risk accepted until they get spec
        hooks or retirement.
      - PN353B, PN357: June 2026 vendor-wave patches registered
        spec-only from birth (no legacy hook by design — they ride
        the v12 spec-driven path).
      - G4_80, PN371, PN373: 2026-06-11 50-PR sweep wave 1 patches
        registered spec-only from birth (same class as PN353B/PN357;
        their wave siblings PN370/PN372/PN374/PN375 DID get legacy
        parking-lot hooks). All opt-in env-gated, absent from
        production Qwen YAMLs today.

    NOTE (2026-06-11): the audit script's `spec_only_truly_orphan_ids`
    cap was raised 30 → 60 when this baseline crossed 30 entries —
    the test compares the FULL set and a 30-cap silently truncated it
    (first symptom: PN40-classifier "resolving" spuriously).
    """
    audit = _import_audit_module()
    from sndr.dispatcher.registry import PATCH_REGISTRY
    legacy = audit._enumerate_legacy_path()
    spec = audit._enumerate_spec_driven_path(PATCH_REGISTRY)
    diff = audit._diff_matrices(legacy, spec, PATCH_REGISTRY)
    expected = {
        # v11.3.0 baseline
        "PN288", "PN289", "PN40-classifier",
        # v12 — ex-manual-orchestration modules (archived third path)
        "G4_31", "G4_32", "G4_79",
        "G4_60A", "G4_60B", "G4_60C", "G4_60D", "G4_60E",
        "G4_60G", "G4_60H", "G4_60K", "G4_60L",
        "G4_67", "G4_69",
        "G4_71", "G4_71B", "G4_72", "G4_73", "G4_74", "G4_75",
        "G4_76", "G4_78",
        "PN262", "PN262B", "PN271",
        # June 2026 vendor wave — spec-only by design
        "PN353B", "PN357",
        # 2026-06-11 50-PR sweep wave 1 — spec-only by design
        "G4_80", "PN371", "PN373",
        # 2026-06-13 50-PR sweep wave 2 — nine spec-only-by-design
        # vendors/blueprints (apply_module set, no legacy hook). PN377
        # is the lone wave-2 patch with a legacy @register_patch hook
        # (apply_patch_N377_*), so it is NOT a spec-only orphan.
        "P88", "PN358", "PN376", "PN378", "PN379",
        "PN380", "PN381", "PN382", "G4_81", "PN383",
        # 2026-06-16 — G4_82 (TQ prefill SDPA fallback for head_dim>256,
        # Ampere FA2 256-cap vllm#38887): runtime monkey-patch, apply_module
        # set, no legacy hook — same spec-only-by-design class as G4_81.
        "G4_82",
        # 2026-06-13 50-PR sweep batch-2 wave 1 — five spec-only-by-design
        # LIVE-bug vendors (apply_module set, own apply(), no legacy hook;
        # same class as PN383).
        "PN384", "PN385", "PN386", "PN387", "PN388",
        # 2026-06-13 50-PR sweep batch-3 — four more spec-only-by-design
        # vendors (apply_module set, own apply(), no legacy hook; same
        # class as PN383-PN388).
        "PN389", "PN390", "PN391", "P89", "PN392",
        # 2026-06-14 PR-sweep wave-1 implementation — spec-driven from
        # inception (apply_module + own apply(), no legacy hook; applied
        # at legacy boot via _run_spec_only_supplement).
        "PN252",
        # PN517 (init snapshot before NCCL, worker family): spec-driven from
        # inception (apply_module + own apply() impl=full, no legacy hook).
        # It IS a genuine orphan — previously hidden by the audit's 60-id cap
        # (the sorted list truncated its tail). The cap was raised 60 -> 100
        # on 2026-06-23 when G4_85 pushed the count to 62, surfacing PN517.
        "PN517",
        # 2026-06-17 (§2.2.A spec-only conversion): PN353A converted to
        # spec-only (apply_module set, legacy @register_patch hook
        # removed); applied at legacy boot via _run_spec_only_supplement.
        "PN353A",
        # 2026-06-17 (0.23.1 pin-bump): PN398 backport of vllm#45100 (async
        # spec-decode accepted-counts race) — spec-driven from inception
        # (apply_module + own apply(), no legacy hook), default-off defensive.
        "PN398",
        # 2026-06-19 (dev148 TIER-1 audit): PN394 backport of MERGED vllm#46047
        # (qwen3 partial-param value `<` truncation) — spec-driven from
        # inception (apply_module + own apply(), no legacy hook), default-on
        # correctness fix.
        "PN394",
        # 2026-06-19 (dev148): PN399 backport of OPEN vllm#46067 (TurboQuant
        # decode-scratch fixed-buffer — fix CUDA IMA in FULL cudagraph) —
        # spec-driven from inception (apply_module + own apply(), no legacy
        # hook), default-OFF experimental belt-and-suspenders; composes with
        # PN118 (wraps its live decode output), requires_patches:[PN118].
        "PN399",
        # 2026-06-19 (dev148 P0 backport): PN400 backport of MERGED vllm#45656
        # (restore is_sym qzeros guard for symmetric AutoRound/GPTQ Marlin MoE;
        # fixes the vllm#43409 regression latent in dev148) — spec-driven from
        # inception (apply_module + own apply(), no legacy hook), default-OFF
        # pin-scoped correctness fix.
        "PN400",
        # 2026-06-23 (G4_85 LIVE re-target): TurboMind tensor-core int4
        # grouped-MoE kernel re-targeted from the orphaned moe_wna16.
        # MoeWNA16Method to the LIVE CompressedTensorsWNA16MoEMethod.apply —
        # spec-driven from inception (apply_module + own apply() returning
        # (status, reason), no legacy hook), default-OFF experimental.
        # Fires only on Marlin-ineligible int4 MoE; fail-open to the
        # original. Same spec-only-by-design class as PN398/PN399/PN400.
        "G4_85",
        # 2026-06-25 (dev424 TIER-1 backport): PN401 backport+improve of OPEN
        # vllm#46461 (TurboQuant prefill continuation guard — gate the
        # flash_attn fast path with `not _has_continuation` so a co-batched
        # continuation never drops its cached prefix K/V) — spec-driven from
        # inception (apply_module + own apply(), no legacy hook), default-OFF
        # experimental correctness fix; applied at legacy boot via
        # _run_spec_only_supplement. Same spec-only-by-design class as
        # PN398/PN399/PN400. Composes with P101/PN116/PN399 (disjoint anchors).
        "PN401",
        # 2026-06-25 (dev424 TIER-1 backport): PN402 backport+improve of OPEN
        # vllm#46574 (sanitize invalid -1/over-vocab MTP draft token ids
        # before batch prep on the new V1 gpu/model_runner path so a single
        # bad draft cannot CUDA-IMA-crash the engine) — spec-driven from
        # inception (apply_module + own apply(), no legacy hook), default-OFF
        # experimental stability fix; applied at legacy boot via
        # _run_spec_only_supplement. Same spec-only-by-design class as
        # PN398/PN399/PN400/PN401. Composes with PN378/PN361/PN133 (disjoint).
        "PN402",
        # 2026-06-25 (dev424 LATENT trap-closer): PN518 vendor of OPEN
        # vllm#46322 (INCConfig.maybe_update_config missing -> hybrid INT4+FP8
        # auto-round checkpoints silently skip FP8 attention/shared-expert
        # layers). Injects a detect-and-WARN maybe_update_config; STRICT NO-OP
        # when no FP8 layers present (the live 27B keeps linear_attn.in_proj
        # at bits=16; 35B is fp8 not inc) -> get_quant_method unperturbed.
        # spec-driven from inception (apply_module + own apply(), no legacy
        # hook), default-OFF latent guard; applied at legacy boot via
        # _run_spec_only_supplement. Same spec-only-by-design class as
        # PN398/PN399/PN400/PN401/PN402.
        "PN518",
        # 2026-06-25 (dev424 integration backlog): PN519 backport+improve of
        # OPEN vllm#46087 (start the SWA/chunked KV-tile loop at
        # first_allowed_key — compute_tile_loop_bounds returns tile_base and
        # both triton_unified_attention consumers offset seq_offset — drops the
        # redundant boundary tile per SWA request + kills the residue-dependent
        # online-softmax reduction-order non-determinism on Gemma4 sliding
        # layers). spec-driven from inception (apply_module + own apply(), no
        # legacy hook), default-OFF experimental kernel_perf fix; applied at
        # legacy boot via _run_spec_only_supplement. Same spec-only-by-design
        # class as PN398/PN399/PN400/PN401/PN402. Composes with PN351 (same
        # files, disjoint anchors). Runtime-inert on Qwen3.6 (FlashInfer/FA2).
        "PN519",
    }
    actual = set(diff["spec_only_truly_orphan_ids"])
    new_orphans = sorted(actual - expected)
    fixed_orphans = sorted(expected - actual)
    assert actual == expected, (
        f"spec_only_truly_orphan baseline changed:\n"
        f"  NEW orphans (review — add legacy hook or manual call): "
        f"{new_orphans}\n"
        f"  RESOLVED orphans (good — update this test's `expected`): "
        f"{fixed_orphans}"
    )


def test_no_apply_module_entry_must_have_legacy_env_or_known_lifecycle():
    """Policy invariant: a spec entry with apply_module=None and
    lifecycle='legacy' MUST use the GENESIS_LEGACY_* env_flag
    convention. This prevents an operator from setting
    `GENESIS_ENABLE_<X>=1` and expecting the patch to apply in
    spec-driven mode (it never will — the patch is legacy-path-only).

    Exemptions:
    - implementation_status='marker_only' / 'placeholder': advisory or
      preflight-doctor entries that don't apply via either path (e.g.
      PN60 quant arg validator, PN63 fp8_e5m2 advisory — they live in
      CLI preflight, not the apply loop).
    - implementation_status='scaffold': in-progress wiring.
    - lifecycle ∈ {retired, research, coordinator}: each has its own
      semantics for apply_module=None handling.
    """
    from sndr.dispatcher.registry import PATCH_REGISTRY
    EXEMPT_STATUS = {"marker_only", "placeholder", "scaffold"}
    violations: list[tuple[str, str, str]] = []
    for pid, meta in PATCH_REGISTRY.items():
        if not isinstance(meta, dict):
            continue
        if meta.get("apply_module") is not None:
            continue
        if meta.get("lifecycle") != "legacy":
            continue
        if meta.get("implementation_status") in EXEMPT_STATUS:
            continue
        env_flag = str(meta.get("env_flag", ""))
        if not env_flag.startswith("GENESIS_LEGACY_"):
            violations.append((pid, env_flag, str(meta.get("implementation_status", "-"))))
    assert not violations, (
        f"{len(violations)} spec entries with lifecycle='legacy' + "
        f"apply_module=None use a `GENESIS_ENABLE_*` env_flag instead "
        f"of the policy `GENESIS_LEGACY_*` prefix. An operator setting "
        f"that env flag would NOT enable the patch in spec-driven mode "
        f"(it always skips) — they'd get a silent no-op.\n\n"
        f"Either:\n"
        f"  (a) rename env_flag to GENESIS_LEGACY_<ID> (intentional "
        f"deferral — patch only applies via legacy path), OR\n"
        f"  (b) add an apply_module + change lifecycle to "
        f"`experimental`/`stable` (real migration), OR\n"
        f"  (c) set implementation_status='marker_only' if it's a "
        f"doctor/preflight advisory not a real apply patch.\n\n"
        f"Offenders:\n" + "\n".join(
            f"  - {pid}: env_flag={ef!r} status={st}"
            for pid, ef, st in violations
        )
    )
