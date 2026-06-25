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

Severity (rig-audit refined, dev301 ground truth 2026-06): only an ANCHOR-BREAK
edge warrants HIGH. The physical breakage mechanism is the dependent's TextPatch
anchor targeting the retired patch's EMITTED bytes — anchor NAME references the
retired id AND anchor TEXT embeds its ``_genesis_<id>_`` emitted symbol (the
PN399 class). Such a dependent silently no-ops when the retired patch goes
native, so a PERF-bearing anchor-break dependent is HIGH (the −5.5% TPS landmine
above). A ``requires_patches`` / ``composes_with`` edge with NO anchor break is a
DECLARED-cooperation / ordering hint — the dependent anchors vanilla upstream and
still APPLIES when a composed sibling retires (the dev301 rig booted PN350 /
PN348 / PN353B / PN365 clean after their sibling retired), so it is MEDIUM even
when the dependent is perf-bearing. A non-perf dependent is MEDIUM (a behaviour
change worth surfacing). Two severity levels — HIGH / MEDIUM.

Apply-state-aware MITIGATION (dev424 ground truth 2026-06): a HIGH anchor-break
edge carries an orthogonal ``mitigated`` flag (severity TIER is unchanged) when
the dependent ALSO declares an ALTERNATIVE anchor sub-patch independent of the
retired id — a working fallback that applies when the retired patch goes native.
PN399 is the motivating case: its C2 removal is PIN-SPLIT into two
mutually-exclusive ``required=False`` siblings — the PN353A-form
``pn399_pn353a_decode_reserve_remove`` (whose name + ``_genesis_pn353a_torch``
anchor text break when PN353A retires) AND the native-form
``pn399_native_decode_reserve_remove`` (referencing neither, targeting the
upstream-native ``_reserve_workspace`` body, applying on dev424). The native
sibling is the fallback, so PN353A->PN399 is HIGH but MITIGATED — already
handled. The bump-preflight gate then FAILS only on UNMITIGATED HIGH edges (a
dependent whose ONLY path references the retired id — the real PN399-incident
class), while still LISTING mitigated edges as HIGH-MITIGATED for visibility.
The mitigation proxy (``_has_independent_alternative_anchor``) is static and
pure: it splits the apply-module into ``TextPatch`` sub-patch regions and looks
for one whose NAME and anchor text reference neither the retired id nor its
emitted symbol.

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
    # SEV_HIGH only when the dependent is PERF-bearing AND anchor-breaks (its
    # anchor name+text target the retired patch's emitted bytes — the PN399
    # class that physically no-ops). Declared-cooperation edges
    # (requires_patches / composes_with with no anchor break) are SEV_MEDIUM.
    severity: str          # SEV_HIGH | SEV_MEDIUM
    via: tuple[str, ...]   # edge kinds: requires_patches/composes_with/anchor_name/anchor_text
    dependent_category: str
    dependent_lifecycle: str
    dependent_default_on: bool
    detail: str            # human one-liner
    # APPLY-STATE-AWARE flag (dev424): True iff this is a HIGH anchor-break edge
    # whose dependent ALSO declares an alternative anchor sub-patch that does NOT
    # reference the retired id — a working fallback path independent of the
    # retired patch (the PN399 native-form C2 sibling). The edge keeps its HIGH
    # SEVERITY tier (still surfaced as HIGH-MITIGATED), but a mitigated edge is
    # already handled, so the bump-preflight gate does NOT exit-1 on it. Only
    # ever True on a HIGH edge (meaningless on MEDIUM). Default False.
    mitigated: bool = False

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

    @property
    def high_mitigated(self) -> list[BreakEdge]:
        """HIGH edges already handled (dependent has an independent fallback)."""
        return [e for e in self.high if e.mitigated]

    @property
    def high_unmitigated(self) -> list[BreakEdge]:
        """HIGH edges with NO fallback — the genuine regression class the bump
        gate must fail on (a dependent whose only path references the retired
        id, the real PN399-incident class)."""
        return [e for e in self.high if not e.mitigated]

    def to_dict(self) -> dict:
        return {
            "high_count": len(self.high),
            "high_mitigated_count": len(self.high_mitigated),
            "high_unmitigated_count": len(self.high_unmitigated),
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


def _emitted_symbol_re(patch_id: str) -> re.Pattern:
    """Matcher for the EMITTED-SYMBOL a retired patch injects into source.

    Genesis text-patches name the host-bound symbols they emit with the
    ``_genesis_<patch_id>_...`` convention (e.g. PN353A emits
    ``_genesis_pn353a_torch``). A dependent whose anchor TEXT contains such a
    symbol is physically targeting the retired patch's EMITTED bytes — when the
    retired patch goes native the symbol is absent and the anchor no-ops. This
    is the load-bearing anchor-text signal; a bare sibling-id MENTION in a
    docstring (``Composes with PN54``) is NOT (it targets nothing).
    """
    return re.compile(r"_genesis_%s(?![0-9A-Za-z])" % re.escape(patch_id),
                      re.IGNORECASE)


def _anchor_refs_retired(
    dependent_module: Optional[str],
    retired_id: str,
    *,
    source_reader=_read_module_source,
) -> tuple[bool, bool]:
    """(anchor_name_ref, anchor_text_ref) — does the dependent's apply-module
    reference ``retired_id`` in an anchor NAME (a ``TextPatch`` ``name=``) or in
    anchor TEXT (a retired-patch-EMITTED ``_genesis_<id>_`` symbol)?

    A pragmatic source scan, not an AST walk:

      * anchor-NAME signal — a snake-case occurrence of the retired id inside a
        ``name=`` line (the ``TextPatch`` name physically targets the retired
        patch, e.g. ``pn399_pn353a_decode_reserve_remove``).
      * anchor-TEXT signal — an occurrence of the retired patch's EMITTED symbol
        (``_genesis_<retired_id>_...``) anywhere in the source. This is the
        retired-specific symbol the module docstring promises, NOT a bare
        sibling-id mention. A bare ``PN54`` in a docstring ("Composes with
        PN54") targets nothing and must NOT count as anchor text — those
        incidental prose mentions are exactly the false-positives the rig audit
        proved (PN348/PN350/PN365 APPLY cleanly when their composed sibling
        retires; only PN399's anchor, which embeds ``_genesis_pn353a_torch``,
        physically no-ops).

    Source is injected (``source_reader``) so tests run without the real file.
    """
    src = source_reader(dependent_module)
    if not src:
        return False, False
    id_re = _id_token_re(retired_id)
    sym_re = _emitted_symbol_re(retired_id)
    name_ref = False
    text_ref = False
    for line in src.splitlines():
        # anchor TEXT = the retired patch's emitted symbol embedded in the
        # dependent's anchor bytes (even inside comments — the symbol IS the
        # targeted byte; what matters is that the anchor literally contains it).
        if sym_re.search(line):
            text_ref = True
        if not id_re.search(line):
            continue
        stripped = line.strip()
        # comment lines describe the relationship — informative but not the
        # load-bearing anchor NAME; skip so a doc-comment doesn't inflate the
        # name signal (the emitted-symbol text scan above is unaffected).
        if stripped.startswith("#"):
            continue
        if "name=" in line or "name =" in line:
            name_ref = True
    return name_ref, text_ref


def _has_independent_alternative_anchor(
    dependent_module: Optional[str],
    retired_id: str,
    *,
    source_reader=_read_module_source,
) -> bool:
    """Does the dependent declare an ALTERNATIVE anchor sub-patch independent of
    the retired id — a working fallback path that does NOT break when the retired
    patch goes native?

    The mitigation proxy for the PN399 class. PN399's C2 removal is PIN-SPLIT into
    two mutually-exclusive ``required=False`` siblings: the PN353A-form
    ``pn399_pn353a_decode_reserve_remove`` (whose name + emitted-symbol anchor
    text reference the retired PN353A — it anchor-breaks and soft-skips when
    PN353A retires) AND the native-form ``pn399_native_decode_reserve_remove``
    (which references neither the bare retired id nor its ``_genesis_<id>_``
    emitted symbol — it targets the upstream-native ``_reserve_workspace`` body
    and APPLIES on dev424). The native sibling is the working fallback, so the
    anchor-break edge is already HANDLED.

    Static proxy (no AST walk; mirrors ``_anchor_refs_retired``): split the source
    into ``TextPatch`` sub-patch regions at each ``name=`` line, then look for at
    least one region whose NAME does NOT reference the retired id AND whose anchor
    text does NOT reference it (neither the bare id token nor the emitted symbol).
    Such a region is an alternative anchor not depending on the retired patch.

    A SINGLE-sub-patch dependent (e.g. a dependent whose ONLY decode-reserve path
    references the retired id — the real PN399-incident class) has no such region
    and is therefore NOT mitigated. Source is injected (tests run without the file).
    """
    src = source_reader(dependent_module)
    if not src:
        return False
    id_re = _id_token_re(retired_id)
    sym_re = _emitted_symbol_re(retired_id)

    lines = src.splitlines()
    # Index every sub-patch declaration line (a ``name=`` / ``name =``). Each
    # region runs from its name line up to (but not including) the next one.
    name_idx = [
        i for i, ln in enumerate(lines)
        if ("name=" in ln or "name =" in ln) and not ln.strip().startswith("#")
    ]
    if len(name_idx) < 2:
        # Fewer than two declared anchors -> no alternative path to fall back to.
        return False
    bounds = name_idx + [len(lines)]
    for k, start in enumerate(name_idx):
        end = bounds[k + 1]
        region = lines[start:end]
        name_line = lines[start]
        # An alternative sub-patch: its NAME must not reference the retired id,
        # and no line in its region may reference the retired id (bare token) or
        # its emitted symbol. That is a fallback anchor independent of the retired
        # patch.
        if id_re.search(name_line):
            continue
        if any(id_re.search(ln) or sym_re.search(ln) for ln in region):
            continue
        return True
    return False


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
            # Rig-audit refinement (dev301 ground truth): only an ANCHOR-BREAK
            # edge is a physical no-op. The dependent's TextPatch anchor targets
            # the retired patch's EMITTED bytes — its anchor NAME references the
            # retired id and its anchor TEXT embeds the ``_genesis_<id>_`` symbol
            # (the PN399 class). A bare ``composes_with`` / ``requires_patches``
            # edge is declared cooperation: the dependent anchors vanilla
            # upstream and still APPLIES when the sibling retires (PN350/PN348/
            # PN353B/PN365 booted clean on dev301), so it is MEDIUM even when
            # perf-bearing. (anchor_name without the emitted-symbol text — e.g.
            # PN357's guarded ``pn357_dflash_swap_pn22_fallback``, which has a
            # vanilla Variant-B fallback — is NOT an anchor break either.)
            anchor_break = ("anchor_name" in via) and ("anchor_text" in via)
            severity = SEV_HIGH if (perf and anchor_break) else SEV_MEDIUM
            # APPLY-STATE-AWARE downgrade (dev424): a HIGH anchor-break edge is
            # MITIGATED when the dependent ALSO declares an alternative anchor
            # sub-patch independent of the retired id (the PN399 native-form C2
            # sibling) — a working fallback that applies when the retired patch
            # goes native. The edge keeps its HIGH severity tier but the gate does
            # not fail on it. Only meaningful for HIGH (anchor-break) edges; a
            # MEDIUM declared-cooperation edge is never relabelled mitigated.
            mitigated = severity == SEV_HIGH and _has_independent_alternative_anchor(
                getattr(dep, "apply_module", None), retired_id,
                source_reader=source_reader,
            )
            detail = _format_detail(
                retired_id, reason, dep_id, via, severity, mitigated)
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
                mitigated=mitigated,
            ))

    edges.sort(key=lambda e: (e.severity != SEV_HIGH, e.retired, e.dependent))
    return RetireImpactReport(edges=edges)


def _format_detail(
    retired_id: str, reason: str, dep_id: str,
    via: list[str], severity: str, mitigated: bool = False,
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
    if mitigated:
        # HIGH anchor-break, but the dependent has an alternative anchor
        # sub-patch independent of the retired id (the PN399 native-form C2
        # sibling) — a working fallback path that applies when the retired
        # patch goes native. Already handled; the gate does not fail on it.
        risk = ("HIGH-MITIGATED: %s has an alternative anchor sub-patch not "
                "referencing %s (a working fallback path independent of the "
                "retired patch — the PN399 native-form sibling) that applies on "
                "the new pin; already handled, surfaced for visibility only"
                % (dep_id, retired_id))
    elif severity == SEV_HIGH:
        # anchor name+text target the retired patch's emitted bytes, and no
        # independent fallback anchor exists — the genuine no-op regression.
        risk = ("anchor targets the retired patch's emitted bytes — physically "
                "no-ops (the PN399 class)")
    else:
        # composes_with / requires only (or a guarded anchor with a vanilla
        # fallback): declared cooperation, dependent anchors vanilla upstream.
        risk = ("declared cooperation; dependent anchors vanilla upstream, so "
                "it likely still applies — re-verify on bump")
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
