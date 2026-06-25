"""Retire-impact / dependency-breakage detector — the bug class that slipped
through on dev148->dev301 (PN353A retired -> PN399 silently no-op'd -> -5.5% TPS
with genuine_drift=0).

Covers:
  * the detector FLAGS PN353A -> PN399 on the LIVE registry as the ONLY HIGH
    (perf + anchor break — name + emitted-symbol text),
  * the rig-audit refinement: composes_with / requires-only perf edges are
    MEDIUM (declared cooperation; dependent anchors vanilla upstream and still
    APPLIES — PN350/PN348/PN353B/PN365 booted clean on dev301; PN357 too),
  * a synthetic CLEAN case (retired patch, no dependents) flags nothing,
  * strong vs weak signals (anchor_text alone never reports an edge; anchor text
    = the retired patch's EMITTED symbol, not a bare sibling-id prose mention),
  * the anchor break is the LOAD-BEARING HIGH signal (survives requires_patches
    scrubbing),
  * perf-bearing classification (category OR title/credit text),
  * a retired patch depending on another retired patch is not a live edge.
"""
from dataclasses import dataclass

import pytest

from sndr.engines.vllm.retire_impact import (
    SEV_HIGH,
    SEV_MEDIUM,
    BreakEdge,
    detect_on_live_registry,
    detect_retire_impact,
    is_perf_bearing,
)


# ─── synthetic spec + registry fixtures ───────────────────────────────────

@dataclass
class FakeSpec:
    patch_id: str
    lifecycle: str = "experimental"
    category: str = "stability"
    title: str = ""
    default_on: bool = False
    requires_patches: tuple = ()
    apply_module: str = ""


def _reg(specs, composes=None, credits=None):
    """Build a minimal PATCH_REGISTRY dict from specs (+ optional overlays)."""
    composes = composes or {}
    credits = credits or {}
    return {
        s.patch_id: {
            "composes_with": list(composes.get(s.patch_id, ())),
            "credit": credits.get(s.patch_id, ""),
        }
        for s in specs
    }


def _no_source(_module):
    """source_reader stub — no anchor name/text signal (registry edges only)."""
    return None


# ─── live-registry acceptance test (the regression that happened) ──────────

def test_live_registry_flags_pn353a_breaks_pn399_high_perf():
    """ACCEPTANCE: on the CURRENT repo the detector must flag PN353A -> PN399
    as a HIGH (perf, anchor-break) breakage — the exact dev148->dev301
    regression, and the ONLY genuine breakage the dev301 rig audit found.

    HIGH is justified by the ANCHOR BREAK (not perf alone): PN399's
    apply-module names an anchor ``pn399_pn353a_decode_reserve_remove`` AND its
    anchor OLD text embeds ``_genesis_pn353a_torch`` — PN353A's EMITTED symbol.
    The anchor literally targets PN353A's emitted bytes, so when PN353A retired
    PN399's PN353A-form anchor went missing. This survives the obvious "just drop
    the requires_patches line" non-fix: even with PN353A scrubbed from PN399's
    declared edges, the anchor name+emitted-symbol signal still fires — see the
    resilience test below.

    The edge stays HIGH severity, but on dev424 it is now MITIGATED (PN399 carries
    a native-form C2 sibling not referencing PN353A — see
    ``test_live_registry_pn399_high_edge_is_mitigated``). This test pins the
    anchor-break DETECTION; the mitigation flag + detail are pinned separately.
    """
    report = detect_on_live_registry()
    edge = next(
        (e for e in report.edges
         if e.retired == "PN353A" and e.dependent == "PN399"),
        None,
    )
    assert edge is not None, "PN353A -> PN399 breakage not detected"
    assert edge.severity == SEV_HIGH
    assert edge.retired_reason == "retired"
    # the anchor BREAK: name (pn399_pn353a_decode_reserve_remove) AND emitted
    # symbol text (_genesis_pn353a_torch) both target PN353A's emitted bytes.
    assert "anchor_name" in edge.via
    assert "anchor_text" in edge.via
    assert "PN399" in edge.detail and "PN353A" in edge.detail
    # it is ranked among the HIGH edges
    assert edge in report.high


def test_live_registry_high_is_exactly_pn399_anchor_break():
    """RIG-AUDIT GROUND TRUTH (dev301, 2026-06): exactly ONE HIGH edge —
    PN353A -> PN399, the only dependent that physically anchor-broke. Every other
    flagged edge is a ``composes_with`` / declared-cooperation false-positive that
    the rig booted CLEAN (PN350/PN348/PN353B/PN365 APPLY on dev301; PN357 skips
    benignly via its own upstream-merge marker), so they must be MEDIUM."""
    report = detect_on_live_registry()
    high_ids = {(e.retired, e.dependent) for e in report.high}
    assert high_ids == {("PN353A", "PN399")}, (
        "expected exactly 1 HIGH (PN353A->PN399), got %s" % sorted(high_ids))
    assert len(report.high) == 1
    medium_ids = {(e.retired, e.dependent) for e in report.medium}
    # the four rig-proven composes_with-only false-positives are now MEDIUM.
    for retired, dep in [("PN54", "PN350"), ("PN108", "PN348"),
                         ("PN353A", "PN353B"), ("PN54", "PN365")]:
        assert (retired, dep) in medium_ids, (
            "%s->%s must be MEDIUM (rig-proven false-positive)" % (retired, dep))
    # PN357 (composes_with PN22 + a guarded anchor name, no emitted symbol) and
    # the G4_19C->G4_31 cooperation edge also stay MEDIUM.
    assert ("PN22", "PN357") in medium_ids
    assert ("G4_19C", "G4_31") in medium_ids
    # and none of those MEDIUM dependents leaked into HIGH.
    for pair in [("PN54", "PN350"), ("PN108", "PN348"), ("PN353A", "PN353B"),
                 ("PN54", "PN365"), ("PN22", "PN357"), ("G4_19C", "G4_31")]:
        assert pair not in high_ids


def test_live_registry_pn399_high_edge_is_mitigated():
    """APPLY-STATE-AWARE REFINEMENT (dev424, 2026-06): the PN353A->PN399 HIGH edge
    is HANDLED. PN399 carries TWO mutually-exclusive C2 sibling sub-patches — the
    PN353A-form ``pn399_pn353a_decode_reserve_remove`` (which anchor-breaks when
    PN353A retires) AND a native-form ``pn399_native_decode_reserve_remove`` that
    does NOT reference PN353A and applies on dev424 (vllm#44053 went native). The
    native sibling is a working fallback path independent of the retired patch, so
    the edge is MITIGATED: still surfaced as HIGH-tier, but flagged ``mitigated``
    so the gate does NOT false-fail (the operator already handled it)."""
    report = detect_on_live_registry()
    edge = next(
        (e for e in report.edges
         if e.retired == "PN353A" and e.dependent == "PN399"),
        None,
    )
    assert edge is not None, "PN353A -> PN399 breakage not detected"
    assert edge.severity == SEV_HIGH
    assert edge.mitigated is True, (
        "PN399 has a native-form sibling not referencing PN353A — must mitigate")
    # detail spells out the mitigation (native-form fallback), not a raw HIGH.
    assert "HIGH-MITIGATED" in edge.detail
    assert "alternative anchor" in edge.detail or "fallback" in edge.detail
    # it is still ranked among the HIGH edges (severity unchanged), but the
    # report exposes the mitigated subset distinctly.
    assert edge in report.high
    assert edge in report.high_mitigated
    assert edge not in report.high_unmitigated


def test_live_registry_no_unmitigated_high_on_current_repo():
    """RIG GROUND TRUTH (dev424): the ONLY HIGH edge (PN353A->PN399) is mitigated
    by PN399's native-form sibling, so there is NO genuinely-unmitigated HIGH on
    the current repo — the gate must PASS. A genuinely-unmitigated HIGH (a
    dependent whose ONLY path references the retired id) is the real regression
    class and is exercised by the synthetic tests below."""
    report = detect_on_live_registry()
    assert report.high_unmitigated == [], (
        "expected 0 unmitigated HIGH on dev424, got %s"
        % [(e.retired, e.dependent) for e in report.high_unmitigated])
    # the one HIGH that exists is the mitigated PN353A->PN399.
    assert {(e.retired, e.dependent) for e in report.high_mitigated} == {
        ("PN353A", "PN399")}


def test_live_registry_all_break_sources_are_actually_retired():
    """Every detected edge's SOURCE must be a genuinely retired/gated patch —
    the detector never fabricates a break from a live source."""
    from sndr.dispatcher.spec import iter_patch_specs

    lifecycles = {s.patch_id: s.lifecycle for s in iter_patch_specs()}
    report = detect_on_live_registry()
    assert report.edges, "expected >=1 live edge (PN353A->PN399 at minimum)"
    for e in report.edges:
        assert lifecycles.get(e.retired) in ("retired", "deprecated"), (
            "edge source %s is not retired" % e.retired)
        # a dependent is never itself retired (not a live regression)
        assert lifecycles.get(e.dependent) not in ("retired", "deprecated")


# ─── synthetic clean case: retired patch, no dependents ────────────────────

def test_retired_patch_with_no_dependents_flags_nothing():
    specs = [
        FakeSpec("PRET", lifecycle="retired"),
        FakeSpec("PLIVE", lifecycle="experimental"),  # references nothing
    ]
    report = detect_retire_impact(
        specs, registry=_reg(specs), source_reader=_no_source)
    assert report.edges == []
    assert report.high == [] and report.medium == []


# ─── synthetic break via requires_patches (HIGH perf) ──────────────────────

def test_requires_patches_perf_dependent_with_no_anchor_break_is_medium():
    """Rig-audit refinement (dev301 ground truth): a ``requires_patches`` edge
    with NO anchor break is DECLARED cooperation, not a physical anchor — the
    dependent anchors vanilla upstream and still APPLIES. So even a perf-bearing
    dependent is MEDIUM, not HIGH. (This is the evidence-backed downgrade that
    the rig proved on PN350/PN348/PN353B/PN365 — they booted clean on dev301.)"""
    specs = [
        FakeSpec("PRET", lifecycle="retired"),
        FakeSpec("PDEP", category="kernel_perf",
                 requires_patches=("PRET",)),
    ]
    report = detect_retire_impact(
        specs, registry=_reg(specs), source_reader=_no_source)
    assert len(report.edges) == 1
    e = report.edges[0]
    assert (e.retired, e.dependent, e.severity) == ("PRET", "PDEP", SEV_MEDIUM)
    assert e.via == ("requires_patches",)
    # detail spells out the declared-cooperation rationale.
    assert "declared cooperation" in e.detail
    assert "re-verify on bump" in e.detail


def test_composes_with_non_perf_dependent_is_medium():
    specs = [
        FakeSpec("PRET", lifecycle="deprecated"),
        FakeSpec("PDEP", category="structured_output", title="json grammar"),
    ]
    report = detect_retire_impact(
        specs, registry=_reg(specs, composes={"PDEP": ("PRET",)}),
        source_reader=_no_source)
    assert len(report.edges) == 1
    e = report.edges[0]
    assert e.severity == SEV_MEDIUM
    assert e.via == ("composes_with",)
    assert e.retired_reason == "deprecated"


# ─── version-gated-out source counts as retired-on-this-pin ─────────────────

def test_version_gated_out_source_breaks_dependent():
    specs = [
        FakeSpec("PGATE", lifecycle="experimental"),  # not retired by lifecycle
        FakeSpec("PDEP", category="perf_hotfix", requires_patches=("PGATE",)),
    ]
    report = detect_retire_impact(
        specs, registry=_reg(specs), gated_out={"PGATE"},
        source_reader=_no_source)
    assert len(report.edges) == 1
    assert report.edges[0].retired_reason == "version_gated"
    # a version-gated source still produces an edge; without an anchor break it
    # is MEDIUM (declared cooperation), per the rig-audit refinement.
    assert report.edges[0].severity == SEV_MEDIUM


# ─── strong vs weak signal: anchor_text alone never reports ────────────────

def test_anchor_text_alone_does_not_report_edge():
    """A dependent whose source merely MENTIONS a retired id (no declared
    dependency, no anchor name) must NOT be reported — avoids the noise of
    incidental sibling-id mentions."""
    specs = [
        FakeSpec("PRET", lifecycle="retired"),
        FakeSpec("PDEP", category="kernel_perf", apply_module="m.dep"),
    ]

    def src(_m):
        # mentions PRET in a code line, but no requires/composes and no name=
        return "    foo = call_into_PRET_output()\n"

    report = detect_retire_impact(
        specs, registry=_reg(specs), source_reader=src)
    assert report.edges == []


def test_anchor_text_enriches_an_existing_strong_edge():
    """anchor TEXT = the retired patch's EMITTED symbol (``_genesis_pret_*``),
    not a bare-id mention. With name + emitted-symbol text + perf this is a full
    anchor break -> HIGH (the PN399 class)."""
    specs = [
        FakeSpec("PRET", lifecycle="retired"),
        FakeSpec("PDEP", category="kernel_perf", apply_module="m.dep",
                 requires_patches=("PRET",)),
    ]

    def src(_m):
        return (
            '            name="pdep_pret_remove",\n'
            "    anchor = _genesis_pret_symbol\n"
        )

    report = detect_retire_impact(
        specs, registry=_reg(specs), source_reader=src)
    assert len(report.edges) == 1
    e = report.edges[0]
    via = e.via
    assert "requires_patches" in via
    assert "anchor_name" in via
    assert "anchor_text" in via
    # name + emitted-symbol text + perf -> a physical anchor break -> HIGH.
    assert e.severity == SEV_HIGH
    assert "physically" in e.detail and "PN399 class" in e.detail


def test_anchor_break_needs_emitted_symbol_not_bare_id_mention():
    """The rig-audit discriminator: a perf dependent that NAMES a guarded anchor
    referencing the retired id AND merely MENTIONS the bare retired id in its
    docstring (no ``_genesis_<id>_`` emitted symbol) is MEDIUM, not HIGH — this
    is the PN357 shape (dual-anchor with a vanilla Variant-B fallback). Only an
    anchor whose TEXT embeds the retired patch's EMITTED bytes is HIGH."""
    specs = [
        FakeSpec("PRET", lifecycle="retired"),
        FakeSpec("PDEP", category="kernel_perf", apply_module="m.dep",
                 requires_patches=("PRET",)),
    ]

    def src(_m):
        # a real guarded anchor NAME for PRET, plus a docstring prose mention of
        # the bare id — but NO _genesis_pret_ emitted symbol (no physical break).
        return (
            '    """Composes with PRET; swaps PRET fallback when present."""\n'
            '            name="pdep_swap_pret_fallback",\n'
            "    anchor = vanilla_class_body  # Variant B fallback\n"
        )

    report = detect_retire_impact(
        specs, registry=_reg(specs), source_reader=src)
    assert len(report.edges) == 1
    e = report.edges[0]
    assert "anchor_name" in e.via
    assert "anchor_text" not in e.via       # bare-id prose is NOT anchor text
    assert e.severity == SEV_MEDIUM
    assert "declared cooperation" in e.detail


def test_anchor_name_alone_is_a_strong_edge_signal_but_medium():
    """An anchor NAME referencing the retired id is a load-bearing EDGE signal
    on its own (reports even without a registry edge — the resilience that
    survives ``requires_patches`` scrubbing). But a name WITHOUT the retired
    patch's emitted-symbol anchor text is NOT a physical anchor break (it can be
    a guarded anchor with a vanilla fallback — the PN357 case), so it is MEDIUM,
    not HIGH. HIGH needs name AND emitted-symbol text (the PN399 class)."""
    specs = [
        FakeSpec("PRET", lifecycle="retired"),
        FakeSpec("PDEP", category="kernel_perf", apply_module="m.dep"),
    ]

    def src(_m):
        return '            name="pdep_pret_decode_reserve_remove",\n'

    report = detect_retire_impact(
        specs, registry=_reg(specs), source_reader=src)
    assert len(report.edges) == 1
    assert report.edges[0].via == ("anchor_name",)
    assert report.edges[0].severity == SEV_MEDIUM


def test_anchor_break_flags_high_even_when_requires_patches_scrubbed():
    """RESILIENCE (the PN399 class): the anchor break is the LOAD-BEARING signal.
    A maintainer who 'fixes' the warning by deleting the declared
    ``requires_patches`` / ``composes_with`` line does NOT silence the detector —
    the apply-module STILL names the anchor and embeds the retired patch's
    emitted symbol, so the HIGH still fires. The regression class survives the
    obvious non-fix, and so must the detector."""
    specs = [
        FakeSpec("PRET", lifecycle="retired"),
        # NO requires_patches, NO composes_with — the declared edges are scrubbed.
        FakeSpec("PDEP", category="kernel_perf", apply_module="m.dep"),
    ]

    def src(_m):
        # name + emitted-symbol anchor text -> a physical anchor break.
        return (
            '            name="pdep_pret_decode_reserve_remove",\n'
            "                _genesis_pret_torch.float32,\n"
        )

    report = detect_retire_impact(
        specs, registry=_reg(specs), source_reader=src)  # no composes overlay
    assert len(report.edges) == 1
    e = report.edges[0]
    assert e.via == ("anchor_name", "anchor_text")  # only anchor signals
    assert e.severity == SEV_HIGH
    assert "physically" in e.detail and "PN399 class" in e.detail


# ─── mitigation: an independent alternative anchor downgrades the gate ──────

def test_high_with_independent_alternative_anchor_is_mitigated():
    """The PN399 shape, synthesized: a perf dependent whose anchor BREAKS on the
    retired id (name + emitted symbol -> HIGH) BUT which also declares a sibling
    sub-patch whose name does NOT reference the retired id and whose anchor block
    does NOT reference it either (a working fallback path independent of the
    retired patch). The edge stays HIGH-tier but is flagged ``mitigated`` so the
    gate does not false-fail."""
    specs = [
        FakeSpec("PRET", lifecycle="retired"),
        FakeSpec("PDEP", category="kernel_perf", apply_module="m.dep",
                 requires_patches=("PRET",)),
    ]

    def src(_m):
        # sub-patch 1: the PRET-form anchor break (name + emitted symbol).
        # sub-patch 2: a NATIVE-form sibling that does NOT reference PRET — the
        # independent fallback (the mitigation).
        return (
            '            name="pdep_pret_decode_reserve_remove",\n'
            "                _genesis_pret_torch.float32,\n"
            '            name="pdep_native_decode_reserve_remove",\n'
            "                current_workspace_manager().get_simultaneous(\n"
        )

    report = detect_retire_impact(
        specs, registry=_reg(specs), source_reader=src)
    assert len(report.edges) == 1
    e = report.edges[0]
    assert e.severity == SEV_HIGH       # severity tier unchanged
    assert e.mitigated is True
    assert "HIGH-MITIGATED" in e.detail
    # exposed via the mitigated/unmitigated split.
    assert e in report.high and e in report.high_mitigated
    assert e not in report.high_unmitigated


def test_unmitigated_high_has_no_independent_alternative_anchor():
    """The real PN399-incident class: a perf dependent whose ONLY decode-reserve
    path references the retired id (no sibling sub-patch independent of it). When
    the retired patch goes native this dependent has NO fallback and physically
    no-ops — it is a genuinely-unmitigated HIGH and MUST still fail the gate."""
    specs = [
        FakeSpec("PRET", lifecycle="retired"),
        FakeSpec("PDEP", category="kernel_perf", apply_module="m.dep",
                 requires_patches=("PRET",)),
    ]

    def src(_m):
        # ONLY a PRET-form anchor break; no independent sibling sub-patch.
        return (
            '            name="pdep_pret_decode_reserve_remove",\n'
            "                _genesis_pret_torch.float32,\n"
        )

    report = detect_retire_impact(
        specs, registry=_reg(specs), source_reader=src)
    assert len(report.edges) == 1
    e = report.edges[0]
    assert e.severity == SEV_HIGH
    assert e.mitigated is False
    assert "HIGH-MITIGATED" not in e.detail
    assert "physically" in e.detail and "PN399 class" in e.detail
    assert e in report.high_unmitigated
    assert e not in report.high_mitigated


def test_mitigation_only_applies_to_high_edges():
    """A MEDIUM edge is never relabelled ``mitigated`` — the flag is meaningful
    only on a HIGH (anchor-break) edge. A declared-cooperation MEDIUM that happens
    to have sibling sub-patches stays a plain MEDIUM (no mitigated flag)."""
    specs = [
        FakeSpec("PRET", lifecycle="retired"),
        FakeSpec("PDEP", category="kernel_perf", apply_module="m.dep",
                 requires_patches=("PRET",)),
    ]

    def src(_m):
        # an anchor NAME referencing PRET but NO emitted-symbol text -> not an
        # anchor break -> MEDIUM. A sibling sub-patch exists but is irrelevant.
        return (
            '            name="pdep_pret_swap_fallback",\n'
            "    anchor = vanilla_body  # Variant B\n"
            '            name="pdep_other_subpatch",\n'
        )

    report = detect_retire_impact(
        specs, registry=_reg(specs), source_reader=src)
    assert len(report.edges) == 1
    e = report.edges[0]
    assert e.severity == SEV_MEDIUM
    assert e.mitigated is False
    assert "HIGH-MITIGATED" not in e.detail


# ─── a retired dependent is not a live regression ──────────────────────────

def test_retired_dependent_is_not_reported():
    specs = [
        FakeSpec("PRET", lifecycle="retired"),
        FakeSpec("PDEP", lifecycle="retired", category="kernel_perf",
                 requires_patches=("PRET",)),
    ]
    report = detect_retire_impact(
        specs, registry=_reg(specs), source_reader=_no_source)
    assert report.edges == []


# ─── id-boundary: PN35 must not match PN353A ───────────────────────────────

def test_id_reference_is_boundary_matched():
    specs = [
        FakeSpec("PN35", lifecycle="retired"),
        FakeSpec("PN353A", lifecycle="retired"),
        # depends on PN353A only; an anchor name with pn353a must NOT credit PN35
        FakeSpec("PDEP", category="kernel_perf", apply_module="m.dep",
                 requires_patches=("PN353A",)),
    ]

    def src(_m):
        return '            name="pdep_pn353a_remove",\n'

    report = detect_retire_impact(
        specs, registry=_reg(specs), source_reader=src)
    deps_by_source = {(e.retired, e.dependent) for e in report.edges}
    assert ("PN353A", "PDEP") in deps_by_source
    # PN35 must NOT be credited as a source for PDEP via the pn353a anchor name
    assert ("PN35", "PDEP") not in deps_by_source


# ─── perf-bearing classification ───────────────────────────────────────────

@pytest.mark.parametrize("category,title,expected", [
    ("kernel_perf", "", True),               # perf category
    ("memory_savings", "", True),
    ("stability", "cut boot overhead", True),  # text token (PN399 shape)
    ("stability", "fix +15.8% TPS re-tune", True),
    ("stability", "throughput win", True),
    ("structured_output", "json grammar compile", False),  # neither
    ("correctness", "fix wrong output", False),
])
def test_is_perf_bearing(category, title, expected):
    assert is_perf_bearing(FakeSpec("X", category=category, title=title)) is expected


def test_breakedge_to_dict_roundtrip():
    e = BreakEdge(
        retired="PRET", retired_reason="retired", dependent="PDEP",
        severity=SEV_HIGH, via=("requires_patches",),
        dependent_category="kernel_perf", dependent_lifecycle="experimental",
        dependent_default_on=False, detail="x", mitigated=True)
    d = e.to_dict()
    assert d["retired"] == "PRET" and d["severity"] == SEV_HIGH
    assert d["via"] == ("requires_patches",)
    assert d["mitigated"] is True
