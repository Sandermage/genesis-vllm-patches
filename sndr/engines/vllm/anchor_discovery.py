"""Shared anchor discovery — the single enumerator of "what to anchor".

This module owns the patch→patcher→anchor enumeration that was previously
private to ``tools/check_upstream_drift.py``. Both the drift-checker and the
per-pin manifest generator import from here, so there is exactly ONE place
that decides which patches/anchors exist (satisfies design requirement R1:
100% coverage of all anchor-bearing patches — no hand-typed subset).

Design: ``sndr/engines/vllm/anchor_discovery.py`` (lib) is imported by
``tools/check_upstream_drift.py`` (tool) and ``scripts/build_anchor_manifest.py``
(tool). Libs are never imported FROM tools — the dependency points one way.

Implements Phase 1 of the per-pin anchor source-of-truth design.
"""
from __future__ import annotations

import importlib
import inspect
from collections.abc import Iterator  # noqa: TC003
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AnchorTarget:
    """One anchor address. The atomic unit the per-pin manifest stores."""

    patch_id: str
    sub: str
    target_rel: str          # vllm-relative path, e.g. "model_executor/layers/fla/ops/chunk.py"
    anchor: str              # the byte-anchor text (old_text) searched in the target
    replacement: str | None
    required: bool
    # classification inputs (so an absent anchor can be split version_gated /
    # upstream_merged / genuine drift instead of all lumped as "drift"):
    vllm_version_range: tuple | None = None   # spec.applies_to.vllm_version_range
    upstream_merged_markers: tuple = ()          # sub-patch upstream_merged_markers
    # Patch lifecycle (spec.lifecycle: "retired" / "stable" / "research" / ...).
    # A retired patch's anchor legitimately no longer matches the dev source
    # (its code was superseded / absorbed upstream), so it must NOT be counted
    # as genuine anchor_drift. Carried so the manifest generator can route a
    # retired patch to STATUS_RETIRED instead of the re-anchor backlog.
    lifecycle: str | None = None


def iter_specs_with_apply_module() -> Iterator[Any]:
    """Yield every PatchSpec that has an on-disk ``apply_module``.

    No pre-filter on ``implementation_status`` — the per-module discovery step
    decides whether a module is buildable, and the version gate handles
    "doesn't apply at this pin". Up-front status filtering was a source of
    false-negatives (it dropped patches whose status didn't match a hardcoded
    set).
    """
    from sndr.dispatcher.spec import iter_patch_specs

    for spec in iter_patch_specs():
        if getattr(spec, "apply_module", None):
            yield spec


def _build_patcher_for_module(mod):
    """Return ``(patcher, note)``. Prefers the module-level ``_make_patcher``;
    falls back to the opt-in ``_make_patcher_for_drift`` shim for inline-builder
    patches (PN347 class). Returns ``(None, reason)`` when the module exposes no
    buildable text-patcher.

    Parameterized ``_make_patcher`` (e.g. P77 threshold, PN9 backend) is called
    with conservative defaults guessed from annotations — never feeding ``None``
    to a non-optional positional silently; if the guess fails the module is
    reported un-buildable rather than crashing to a false drift.

    (Moved verbatim from tools/check_upstream_drift.py at the Phase 1 extraction so
    both the drift-checker and the manifest generator share one builder.)
    """
    builder = getattr(mod, "_make_patcher", None)
    if builder is None:
        builder = getattr(mod, "_make_patcher_for_drift", None)
    if builder is None:
        return None, "no _make_patcher() or _make_patcher_for_drift() shim"
    return _call_builder(builder)


def _call_builder(builder):
    """Invoke a patcher-builder with conservative defaults guessed from its
    annotations (never feeds ``None`` to a non-optional positional silently).
    Returns ``(patcher, note)`` — ``(None, reason)`` when it can't build."""
    try:
        sig = inspect.signature(builder)
        kwargs: dict[str, Any] = {}
        for pname, p in sig.parameters.items():
            if p.default is not inspect.Parameter.empty:
                continue
            if p.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            ann = str(p.annotation)
            if "int" in ann:
                kwargs[pname] = 0
            elif "bool" in ann:
                kwargs[pname] = False
            elif "float" in ann:
                kwargs[pname] = 0.0
            elif "str" in ann and "Optional" not in ann and "None" not in ann:
                # A required str positional — empty string is safer than None
                # (None would break `f"..."` / `.lower()` call sites).
                kwargs[pname] = ""
            else:
                kwargs[pname] = None
    except (TypeError, ValueError):
        kwargs = {}

    try:
        patcher = builder(**kwargs) if kwargs else builder()
    except Exception as e:  # noqa: BLE001 — surface as un-buildable, not drift
        return None, f"builder raised: {e}"
    if patcher is None:
        return None, "builder returned None (target file absent at this pin)"
    return patcher, "ok"


# Builder-name convention: the canonical two, plus ANY module-level callable
# named `_make_*patcher*` (e.g. P58's _make_request_patcher /
# _make_scheduler_patcher for a multi-file patch). This is what recovers the
# ~65 anchor-bearing patches the singular discovery missed.
def _is_builder_name(name: str) -> bool:
    if name in ("_make_patcher", "_make_patcher_for_drift"):
        return True
    return name.startswith("_make_") and "patcher" in name


def discover_patchers(mod) -> list[tuple[Any, str]]:
    """Return ``[(patcher, note), ...]`` for EVERY patcher-builder in `mod`.

    A single-file patch has one builder; a multi-file patch (P58 class) has
    several `_make_*_patcher()` functions — the singular
    ``_build_patcher_for_module`` only ever saw the first canonical name and
    dropped the rest to ``needs_fixture`` despite real anchors. Builders that
    return None (target absent at this pin) are skipped. Order-stable by
    definition order in the module. Returns ``[]`` when the module exposes no
    builder (pure class-rebind wiring — handled by the binding resolver)."""
    builders: list[tuple[str, Any]] = []
    seen: set[int] = set()
    for name in dir(mod):
        if not _is_builder_name(name):
            continue
        fn = getattr(mod, name, None)
        if not callable(fn) or id(fn) in seen:
            continue
        seen.add(id(fn))
        builders.append((name, fn))
    # Stable order: definition order via __code__.co_firstlineno when available.
    def _order(item):
        code = getattr(item[1], "__code__", None)
        return getattr(code, "co_firstlineno", 1_000_000) if code else 1_000_000
    builders.sort(key=_order)

    out: list[tuple[Any, str]] = []
    for _name, fn in builders:
        # FP-2: a real runtime patcher-builder takes NO required args (apply()
        # calls it bare). A helper/fixture builder that requires args
        # (_make_patcher_for_target(cls), _make_*_for_fixture(...)) must be
        # skipped — feeding it guessed defaults fabricates a bogus patcher
        # (often target_file=None/"") that then wins worst-of aggregation and
        # cries wolf on an otherwise-clean patch.
        try:
            sig = inspect.signature(fn)
            if any(
                p.default is inspect.Parameter.empty
                and p.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
                for p in sig.parameters.values()
            ):
                continue
        except (TypeError, ValueError):
            pass
        patcher, note = _call_builder(fn)
        # Skip patchers with no usable target — a fabricated/empty patcher must
        # not enter aggregation (defensive belt for the arg-guess path).
        if patcher is not None and str(getattr(patcher, "target_file", "") or "").strip():
            out.append((patcher, note))
    return out


def _target_rel(target_file: str | None) -> str | None:
    """Map an absolute patcher target path to its vllm-relative form.

    ``/usr/local/.../site-packages/vllm/model_executor/layers/fla/ops/chunk.py``
    → ``model_executor/layers/fla/ops/chunk.py``. Splits on the LAST ``/vllm/``
    segment so a path containing 'vllm' elsewhere does not mis-strip.
    """
    if not target_file:
        return None
    s = str(target_file).replace("\\", "/")
    marker = "/vllm/"
    idx = s.rfind(marker)
    if idx == -1:
        return s
    return s[idx + len(marker):]


def iter_anchor_targets() -> Iterator[AnchorTarget]:
    """Enumerate every anchor address across ALL anchor-bearing patches (R1).

    For each spec with an ``apply_module``: import the module, build its
    TextPatcher, and yield one ``AnchorTarget`` per sub-patch that carries an
    ``anchor``. Import-wiring patches (PN287/PN392 class — no text anchors,
    they resolve classes) and un-buildable modules are skipped (they are not
    byte-anchor patches and are covered by the drift-checker's import-wiring
    path, not the per-pin anchor manifest).
    """
    for spec in iter_specs_with_apply_module():
        try:
            mod = importlib.import_module(spec.apply_module)
        except Exception:  # noqa: BLE001 — un-importable module: not an anchor
            continue
        patcher, _note = _build_patcher_for_module(mod)
        if patcher is None:
            continue
        target_rel = _target_rel(getattr(patcher, "target_file", None))
        if not target_rel:
            continue
        applies_to = getattr(spec, "applies_to", None) or {}
        vrange = applies_to.get("vllm_version_range")
        vrange_t = tuple(vrange) if isinstance(vrange, (list, tuple)) else (
            (vrange,) if vrange else None
        )
        # spec.lifecycle is the registry's lifecycle string (e.g. "retired").
        # Tagged onto every target so the manifest generator can classify a
        # retired patch's drifted anchor as STATUS_RETIRED, not anchor_drift.
        lifecycle = getattr(spec, "lifecycle", None)
        lifecycle = str(lifecycle).lower() if lifecycle else None
        for sp in getattr(patcher, "sub_patches", []) or []:
            anchor = getattr(sp, "anchor", None)
            if not anchor:
                continue
            yield AnchorTarget(
                patch_id=getattr(spec, "patch_id", "?"),
                sub=getattr(sp, "name", "?"),
                target_rel=target_rel,
                anchor=anchor,
                replacement=getattr(sp, "replacement", None),
                required=bool(getattr(sp, "required", False)),
                vllm_version_range=vrange_t,
                upstream_merged_markers=tuple(
                    getattr(sp, "upstream_merged_markers", []) or []
                ),
                lifecycle=lifecycle,
            )
