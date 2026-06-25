"""Retire-impact / dependency-breakage detector (anchor-SOT extension).

Static analysis over ``PATCH_REGISTRY``: when a patch is RETIRED (lifecycle
``retired``/``deprecated``) or version-gated OUT on the target pin, find every
OTHER patch that silently breaks because it depends on the retired one. The
dependency edge is one of:

  * ``requires_patches`` references the retired id      (hard dependency)
  * ``composes_with`` references the retired id         (soft dependency)
  * the dependent's anchor NAME references the retired id
    (e.g. PN399 sub-patch ``pn399_pn353a_decode_reserve_remove``)

These are the STRONG (load-bearing, explicitly declared) signals — at least one
must fire for an edge to be reported. A fourth, CORROBORATING signal is also
collected but never reports an edge on its own:

  * the dependent's anchor TEXT references a retired-specific symbol
    (e.g. ``_genesis_pn353a_torch`` emitted only by PN353A)

Anchor TEXT alone is too broad (a sibling-patch id is mentioned in many module
strings/comments without a real dependency). It only ENRICHES an edge that
already has a strong signal — confirming, e.g., that PN399's anchor literally
targets PN353A-emitted bytes.

The class of bug this catches (the dev148->dev301 regression that slipped
through): PN353A was retired (vllm#44053 went native). PN399 ``requires_patches``
includes PN353A and its anchor ``pn399_pn353a_decode_reserve_remove`` targets the
code PN353A modified. When PN353A retired, PN399's anchor went missing -> PN399
SKIPPED as a *benign* skip (NOT genuine anchor_drift) -> its decode-scratch perf
optimization no-op'd -> a real -5.5% TPS regression that no gate caught. The
anchor-SOT ``drift.rej.json`` showed ``genuine_drift=0`` (clean) while a perf
patch was silently dead.

Severity: a dependent that itself carries a PERF signal (category in the perf
set, OR title/credit mentions TPS / overhead / throughput / regression / ...) is
HIGH — a silent perf regression is exactly the landmine above. Everything else
is MEDIUM (a dependent that skips is still a behaviour change worth surfacing).

This module is PURE host code: it imports only the dispatcher spec layer (no
torch / no vLLM), so it runs where the manifest is built and is unit-testable
against a synthetic registry. The anchor name/text scan reads the dependent's
``apply_module`` source as plain text (best-effort; absent source degrades to
the registry-edge signal, never raises).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional

# Severity constants.
SEV_HIGH = "HIGH"      # PERF-tier dependent — silent perf regression risk
SEV_MEDIUM = "MEDIUM"  # behaviour-change dependent (correctness / skip)

# Categories that are intrinsically performance-bearing. A dependent in one of
# these is HIGH severity when broken. Kept in sync with dispatcher.spec
# VALID_CATEGORIES (the perf-flavoured subset).
PERF_CATEGORIES = frozenset({
    "kernel_perf",
    "perf_kernel",
    "perf_hotfix",
    "memory_savings",
    "memory_pool",
    "memory_hotfix",
    "ttft_warmup",
})

# Perf-signal tokens scanned in a dependent's title + credit. PN399 is the
# motivating case: its category is ``stability`` (not a perf category) yet it
# IS a perf optimization ("cut boot overhead", tied to a +15.8% TPS re-tune in
# the CHANGELOG). Category alone would miss it, so the text scan is the
# second, broader signal. Word-boundary matched to avoid false hits
# (e.g. "performance" matches "perf"; "overhead" is whole-word).
_PERF_SIGNAL_TOKENS = (
    "tps",
    "perf",            # perf / performance
    "throughput",
    "speedup",
    "faster",
    "latency",
    "overhead",
    "regression",
    "optimiz",         # optimize / optimization / optimise
    "no-op",
)
_PERF_SIGNAL_RE = re.compile(
    r"(?:%s)" % "|".join(re.escape(t) for t in _PERF_SIGNAL_TOKENS),
    re.IGNORECASE,
)


def is_perf_signal(category: str, title: str = "", credit: str = "") -> bool:
    """True iff (category, title, credit) carry a performance signal.

    The primitive: a perf ``category`` OR a perf-signal token in the title /
    credit text. PN399 is the motivating case — its category is ``stability``
    yet its credit says "cut boot overhead" (tied to a +15.8% TPS re-tune), so
    the text scan is essential; category alone would mis-classify it MEDIUM.
    """
    if str(category or "") in PERF_CATEGORIES:
        return True
    return bool(_PERF_SIGNAL_RE.search("%s %s" % (title or "", credit or "")))


def is_perf_bearing(spec: Any) -> bool:
    """``is_perf_signal`` over a spec-like object (``category`` / ``title`` /
    optional ``credit`` attributes). Convenience for callers holding a
    dispatcher ``PatchSpec`` (which lacks ``credit`` — read from registry meta
    by the caller and set on the object when available)."""
    return is_perf_signal(
        getattr(spec, "category", ""),
        getattr(spec, "title", "") or "",
        getattr(spec, "credit", "") or "",
    )


@dataclass(frozen=True)
class BreakEdge:
    """One ``retired X breaks dependent Y`` finding."""

    retired: str          # the retired / gated-out patch id
    retired_reason: str    # "retired" | "deprecated" | "version_gated"
    dependent: str         # the patch that breaks
    severity: str          # SEV_HIGH | SEV_MEDIUM
    via: tuple[str, ...]   # edge kinds: requires_patches/composes_with/anchor_name/anchor_text
    dependent_category: str
    dependent_lifecycle: str
    dependent_default_on: bool
    detail: str            # human one-liner

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetireImpactReport:
    """Ranked dependency-breakage findings (HIGH first, then by id)."""

    edges: list[BreakEdge] = field(default_factory=list)

    @property
    def high(self) -> list[BreakEdge]:
        return [e for e in self.edges if e.severity == SEV_HIGH]

    @property
    def medium(self) -> list[BreakEdge]:
        return [e for e in self.edges if e.severity == SEV_MEDIUM]

    def to_dict(self) -> dict:
        return {
            "high_count": len(self.high),
            "medium_count": len(self.medium),
            "edges": [e.to_dict() for e in self.edges],
        }


# ─── id-reference scanning ────────────────────────────────────────────────

def _id_token_re(patch_id: str) -> re.Pattern:
    """Word-/snake-boundary matcher for a patch id INSIDE identifiers.

    ``PN353A`` must match the anchor name ``pn399_pn353a_decode_reserve_remove``
    and the symbol ``_genesis_pn353a_torch`` (case-insensitive, surrounded by
    non-alphanumerics so ``PN35`` does NOT match ``PN353A`` and ``PN3`` does not
    match ``PN30``).
    """
    return re.compile(r"(?<![0-9A-Za-z])%s(?![0-9A-Za-z])" % re.escape(patch_id),
                      re.IGNORECASE)


def _read_module_source(apply_module: Optional[str]) -> Optional[str]:
    """Return the dependent's apply-module source as text (best-effort).

    Resolves the dotted module to a file via importlib's spec WITHOUT importing
    it (avoids torch / vLLM import side effects on the host). Returns None if the
    module / file can't be located — the caller degrades to registry-edge signal.
    """
    if not apply_module:
        return None
    try:
        import importlib.util

        spec = importlib.util.find_spec(apply_module)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        return None
    try:
        with open(spec.origin, encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return None


def _anchor_refs_retired(
    dependent_module: Optional[str],
    retired_id: str,
    *,
    source_reader=_read_module_source,
) -> tuple[bool, bool]:
    """(anchor_name_ref, anchor_text_ref) — does the dependent's apply-module
    reference ``retired_id`` in an anchor NAME (a ``TextPatch`` ``name=``) or in
    anchor TEXT (a retired-id-bearing symbol)?

    A pragmatic source scan, not an AST walk: the anchor-name signal is a
    snake-case occurrence of the retired id inside a ``name=`` line; the
    anchor-text signal is any other occurrence of the retired id in the source.
    Source is injected (``source_reader``) so tests run without the real file.
    """
    src = source_reader(dependent_module)
    if not src:
        return False, False
    id_re = _id_token_re(retired_id)
    name_ref = False
    text_ref = False
    for line in src.splitlines():
        if not id_re.search(line):
            continue
        stripped = line.strip()
        # comment lines describe the relationship — informative but not the
        # load-bearing anchor; skip so a doc-comment doesn't inflate signal.
        if stripped.startswith("#"):
            continue
        if "name=" in line or "name =" in line:
            name_ref = True
        else:
            text_ref = True
    return name_ref, text_ref


# ─── core detector ────────────────────────────────────────────────────────

def detect_retire_impact(
    specs: Iterable[Any],
    *,
    registry: Optional[dict[str, dict]] = None,
    gated_out: Optional[Iterable[str]] = None,
    source_reader=_read_module_source,
) -> RetireImpactReport:
    """Find every ``retired/gated X -> breaks dependent Y`` edge.

    Args:
        specs: iterable of dispatcher ``PatchSpec`` (``iter_patch_specs()``).
        registry: raw ``PATCH_REGISTRY`` dict, for ``composes_with`` /
            ``credit`` (not on ``PatchSpec``). Falls back to the live registry.
        gated_out: extra patch ids treated as "retired on this pin" because the
            target pin version-gates them OUT (their anchors are absent by
            design). Lets ``bump_preflight`` feed the per-pin gated set.
        source_reader: injectable module-source reader (tests).

    A dependent is only reported when it is NOT itself retired/gated-out: a
    retired patch depending on another retired patch is not a live regression.
    """
    if registry is None:
        from sndr.dispatcher.registry import PATCH_REGISTRY as registry  # noqa: N806

    specs = list(specs)
    by_id = {s.patch_id: s for s in specs}
    gated = set(gated_out or ())

    def _reason(pid: str, spec: Any) -> Optional[str]:
        lc = str(getattr(spec, "lifecycle", "")).lower()
        if lc in ("retired", "deprecated"):
            return lc
        if pid in gated:
            return "version_gated"
        return None

    retired_reasons = {
        s.patch_id: r for s in specs
        if (r := _reason(s.patch_id, s)) is not None
    }
    # version-gated ids that have no spec still count (defensive).
    for pid in gated:
        retired_reasons.setdefault(pid, "version_gated")

    edges: list[BreakEdge] = []
    for dep in specs:
        dep_id = dep.patch_id
        # A retired/gated dependent is not a live regression — skip it as a
        # dependent (it is still reported as a `retired` SOURCE above).
        if _reason(dep_id, dep) is not None:
            continue
        meta = registry.get(dep_id, {}) if isinstance(registry, dict) else {}
        req = set(getattr(dep, "requires_patches", ()) or ())
        comp = set(meta.get("composes_with") or ())
        credit = meta.get("credit", "")

        for retired_id, reason in retired_reasons.items():
            via: list[str] = []
            if retired_id in req:
                via.append("requires_patches")
            if retired_id in comp:
                via.append("composes_with")
            name_ref, text_ref = _anchor_refs_retired(
                getattr(dep, "apply_module", None), retired_id,
                source_reader=source_reader,
            )
            if name_ref:
                via.append("anchor_name")
            # STRONG signals declare a real dependency; report ONLY when one
            # fires. Anchor TEXT alone (a passing sibling-id mention) is too
            # broad to be a breakage on its own.
            if not via:
                continue
            # anchor_text is corroborating — appended after the strong gate so
            # it enriches an already-real edge without creating noise edges.
            if text_ref:
                via.append("anchor_text")

            perf = is_perf_signal(dep.category, dep.title, credit)
            severity = SEV_HIGH if perf else SEV_MEDIUM
            detail = _format_detail(retired_id, reason, dep_id, via, severity)
            edges.append(BreakEdge(
                retired=retired_id,
                retired_reason=reason,
                dependent=dep_id,
                severity=severity,
                via=tuple(via),
                dependent_category=str(dep.category),
                dependent_lifecycle=str(dep.lifecycle),
                dependent_default_on=bool(dep.default_on),
                detail=detail,
            ))

    edges.sort(key=lambda e: (e.severity != SEV_HIGH, e.retired, e.dependent))
    return RetireImpactReport(edges=edges)


def _format_detail(
    retired_id: str, reason: str, dep_id: str,
    via: list[str], severity: str,
) -> str:
    edges = []
    if "requires_patches" in via:
        edges.append("%s.requires_patches=[%s]" % (dep_id, retired_id))
    if "composes_with" in via:
        edges.append("%s.composes_with=[%s]" % (dep_id, retired_id))
    if "anchor_name" in via:
        edges.append("%s anchor '%s_%s_*'" % (
            dep_id, dep_id.lower(), retired_id.lower()))
    if "anchor_text" in via:
        edges.append("%s anchor text refs %s" % (dep_id, retired_id))
    risk = ("perf/correctness risk — its optimization NO-OPs"
            if severity == SEV_HIGH else "behaviour change")
    return ("retiring %s (%s) breaks dependent %s (%s) — %s will skip/no-op (%s)"
            % (retired_id, reason, dep_id, " / ".join(edges), dep_id, risk))


def detect_on_live_registry(
    gated_out: Optional[Iterable[str]] = None,
) -> RetireImpactReport:
    """Convenience: run the detector against the live ``PATCH_REGISTRY``."""
    from sndr.dispatcher.registry import PATCH_REGISTRY
    from sndr.dispatcher.spec import iter_patch_specs

    return detect_retire_impact(
        iter_patch_specs(), registry=PATCH_REGISTRY, gated_out=gated_out,
    )


__all__ = [
    "SEV_HIGH",
    "SEV_MEDIUM",
    "PERF_CATEGORIES",
    "BreakEdge",
    "RetireImpactReport",
    "is_perf_bearing",
    "detect_retire_impact",
    "detect_on_live_registry",
]
