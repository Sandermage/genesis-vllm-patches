"""Retire-impact / dependency-breakage detector — the bug class that slipped
through on dev148->dev301 (PN353A retired -> PN399 silently no-op'd -> -5.5% TPS
with genuine_drift=0).

Covers:
  * the detector FLAGS PN353A -> PN399 on the LIVE registry (HIGH, perf),
  * a synthetic CLEAN case (retired patch, no dependents) flags nothing,
  * strong vs weak signals (anchor_text alone never reports an edge),
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
    as a HIGH (perf) breakage — the exact dev148->dev301 regression.

    Robustness note: even after PN353A was scrubbed from PN399's
    ``requires_patches`` / ``composes_with`` (the dev301 re-anchor that removed
    the declared edges), PN399's apply-module STILL names an anchor
    ``pn399_pn353a_decode_reserve_remove`` and emits ``_genesis_pn353a_torch``
    in anchor TEXT — so the detector catches the dependency via the LOAD-BEARING
    anchor-name signal, not a declaration a maintainer can delete. That is the
    whole point: the regression class survives the obvious "just drop the
    requires_patches line" non-fix, and so must the detector.
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
    # the anchor-name signal is the load-bearing one (survives declaration
    # scrubbing): pn399_pn353a_decode_reserve_remove.
    assert "anchor_name" in edge.via
    assert "PN399" in edge.detail and "PN353A" in edge.detail
    # it is ranked among the HIGH edges
    assert edge in report.high


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

def test_requires_patches_perf_dependent_is_high():
    specs = [
        FakeSpec("PRET", lifecycle="retired"),
        FakeSpec("PDEP", category="kernel_perf",
                 requires_patches=("PRET",)),
    ]
    report = detect_retire_impact(
        specs, registry=_reg(specs), source_reader=_no_source)
    assert len(report.edges) == 1
    e = report.edges[0]
    assert (e.retired, e.dependent, e.severity) == ("PRET", "PDEP", SEV_HIGH)
    assert e.via == ("requires_patches",)


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
    assert report.edges[0].severity == SEV_HIGH


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
    via = report.edges[0].via
    assert "requires_patches" in via
    assert "anchor_name" in via
    assert "anchor_text" in via


def test_anchor_name_alone_is_a_strong_signal():
    """An anchor NAME referencing the retired id is load-bearing on its own
    (the PN399 ``pn399_pn353a_*`` case) — reports even without a registry edge."""
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
        dependent_default_on=False, detail="x")
    d = e.to_dict()
    assert d["retired"] == "PRET" and d["severity"] == SEV_HIGH
    assert d["via"] == ("requires_patches",)
