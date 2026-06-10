# SPDX-License-Identifier: Apache-2.0
"""M.1.1.T3 declarative retired-patch tests — schema + pilot invariants.

Covers ``_RETIRED_PATCHES`` table shape + ``RetiredPatchSpec`` dataclass
+ ``_retired_patch_handler`` + ``_register_retired_patches`` after the
T3.B-PILOT-OF-1 migration landed P8 as the first declarative entry.

History
───────
  M.1.1.T3.A   (commit f2886e4f) — scaffolding landed; ``_RETIRED_PATCHES``
                                   empty; ``dict[str, str]`` value type.
  M.1.1.T3.B-PILOT-OF-1            schema → ``dict[str, RetiredPatchSpec]``;
                                   P8 migrated as the only AST-confirmed
                                   pure retired stub. See
                                   ``sndr_private/planning/audits/
                                    M1_T3_SCOPE_REVISION_2026-05-27_RU.md``
                                   for the bulk-thesis correction.

T3.B-PILOT-OF-1 invariants
──────────────────────────
  * ``_RETIRED_PATCHES`` is exposed at module scope as a ``dict``.
  * Contains the P8 entry (1 declarative migration).
  * Each entry value is a frozen ``RetiredPatchSpec`` carrying name,
    wrapped_name, reason.
  * ``_retired_patch_handler(spec)`` returns a zero-arg callable that
    produces ``PatchResult(spec.name, "skipped", spec.reason)`` and
    has ``__name__ == spec.wrapped_name`` so the apply_registry.json
    snapshot's ``wrapped_name`` field stays byte-identical.
  * Synthetic-entry test still works: a monkeypatch'd ``RetiredPatchSpec``
    registers exactly one pair, with the registry restored on exit.
"""
from __future__ import annotations

import pytest


def _import_dispatch_module():
    """Lazy import so the apply package's heavy decorator chain is
    only walked once per test session."""
    from sndr.apply import _per_patch_dispatch as mod
    return mod


# ─── Schema invariants ────────────────────────────────────────────────────


class TestRetiredPatchesSchema:
    """The declarative table exists with the post-T3.B-PILOT schema:
    a ``dict[str, RetiredPatchSpec]`` carrying the migrated entries."""

    def test_retired_patches_dict_present(self):
        mod = _import_dispatch_module()
        assert hasattr(mod, "_RETIRED_PATCHES"), (
            "_RETIRED_PATCHES scaffolding missing from "
            "sndr.apply._per_patch_dispatch"
        )
        assert isinstance(mod._RETIRED_PATCHES, dict)

    def test_retired_patch_spec_dataclass_exposed(self):
        mod = _import_dispatch_module()
        assert hasattr(mod, "RetiredPatchSpec"), (
            "RetiredPatchSpec dataclass missing — schema regression"
        )
        spec = mod.RetiredPatchSpec(name="X", wrapped_name="y", reason="z")
        # Frozen — mutation raises FrozenInstanceError.
        with pytest.raises(Exception):
            spec.name = "Z"  # type: ignore[misc]

    def test_p8_pilot_entry_present(self):
        """T3.B-PILOT-OF-1 migrated P8. The declarative table must
        carry the exact name / wrapped_name / reason captured pre-
        migration; otherwise apply_registry.json drifts."""
        mod = _import_dispatch_module()
        key = "P8 KV hybrid reporting (per-token capacity)"
        assert key in mod._RETIRED_PATCHES, (
            "P8 pilot entry missing from _RETIRED_PATCHES — "
            "migration rollback would shift PATCH_REGISTRY order"
        )
        spec = mod._RETIRED_PATCHES[key]
        assert isinstance(spec, mod.RetiredPatchSpec)
        assert spec.name == key
        assert spec.wrapped_name == "apply_patch_8_kv_hybrid_reporting"
        assert spec.reason == "retired 2026-05-04 (upstream refactor superseded)"


# ─── _retired_patch_handler shape ─────────────────────────────────────────


class TestRetiredPatchHandler:
    """The handler factory builds a no-op apply() callable that
    matches the byte-identical behaviour of the hand-written stubs
    it replaces. ``__name__`` MUST be set to ``spec.wrapped_name`` so
    ``register_patch`` propagates it to the snapshot-visible
    ``__wrapped__.__name__`` attribute."""

    def test_returns_callable(self):
        mod = _import_dispatch_module()
        spec = mod.RetiredPatchSpec(name="X", wrapped_name="apply_x", reason="Y")
        handler = mod._retired_patch_handler(spec)
        assert callable(handler)

    def test_callable_name_set_to_wrapped_name(self):
        """The snapshot test reads ``fn.__wrapped__.__name__``;
        ``register_patch`` sets ``__wrapped__`` to the original fn —
        which here is the handler. Therefore ``handler.__name__``
        must equal ``spec.wrapped_name`` BEFORE registration."""
        mod = _import_dispatch_module()
        spec = mod.RetiredPatchSpec(
            name="X", wrapped_name="apply_patch_synthetic_x", reason="Y",
        )
        handler = mod._retired_patch_handler(spec)
        assert handler.__name__ == "apply_patch_synthetic_x"

    def test_callable_returns_patch_result(self):
        mod = _import_dispatch_module()
        from sndr.apply._state import PatchResult

        spec = mod.RetiredPatchSpec(
            name="TEST-PATCH name (synthetic)",
            wrapped_name="apply_patch_test_synthetic",
            reason="retired 2026-05-27 (synthetic test)",
        )
        handler = mod._retired_patch_handler(spec)
        result = handler()
        assert isinstance(result, PatchResult)
        assert result.name == "TEST-PATCH name (synthetic)"
        assert result.status == "skipped"
        assert result.reason == "retired 2026-05-27 (synthetic test)"

    def test_callable_emits_skipped_each_call(self):
        """Stateless: invoking twice produces the same shape both
        times. Catches a future bug where someone caches the
        PatchResult object or mutates it on the first call."""
        mod = _import_dispatch_module()
        spec = mod.RetiredPatchSpec(name="A", wrapped_name="apply_a", reason="B")
        handler = mod._retired_patch_handler(spec)
        first = handler()
        second = handler()
        assert first.status == "skipped"
        assert second.status == "skipped"
        assert first.name == second.name == "A"
        assert first.reason == second.reason == "B"


# ─── P8 migrated entry — direct invocation parity ─────────────────────────


class TestP8MigratedInvocation:
    """The migrated P8 entry, when invoked through PATCH_REGISTRY,
    must produce a PatchResult byte-identical to the pre-migration
    hand-written stub. This is the strongest single-stub guard against
    refactor drift in the declarative path."""

    def test_p8_invocation_matches_pre_migration(self):
        from sndr.apply._state import PATCH_REGISTRY, PatchResult

        # Find the P8 entry by decorator name (insertion order means
        # it's at position 0, but locate explicitly for robustness).
        p8_entries = [
            (i, name, fn) for i, (name, fn) in enumerate(PATCH_REGISTRY)
            if name == "P8 KV hybrid reporting (per-token capacity)"
        ]
        assert len(p8_entries) == 1, (
            f"P8 must appear exactly once in PATCH_REGISTRY; "
            f"found {len(p8_entries)} occurrence(s)"
        )
        pos, name, fn = p8_entries[0]

        # Position 0 in the pre-migration snapshot.
        assert pos == 0, (
            f"P8 expected at PATCH_REGISTRY position 0 (pre-migration), "
            f"got position {pos} — apply_registry.json byte-identity broken"
        )

        # wrapped_name visible through __wrapped__.__name__
        wrapped = getattr(fn, "__wrapped__", None)
        assert wrapped is not None
        assert wrapped.__name__ == "apply_patch_8_kv_hybrid_reporting"

        # Invocation parity — every field of the PatchResult matches
        # what the hand-written stub used to return.
        result = fn()
        assert isinstance(result, PatchResult)
        assert result.name == "P8 KV hybrid reporting (per-token capacity)"
        assert result.status == "skipped"
        assert result.reason == "retired 2026-05-04 (upstream refactor superseded)"


# ─── _register_retired_patches behaviour ─────────────────────────────────


class TestRegisterRetiredPatches:
    """The registration loop iterates ``_RETIRED_PATCHES`` and routes
    each entry through ``register_patch``. The synthetic-entry test
    uses monkeypatch so the live registry stays clean afterwards."""

    def test_idempotent_invocation(self):
        """Re-invoking ``_register_retired_patches()`` re-registers
        every entry — by design. This test captures the increment in
        ``PATCH_REGISTRY`` length and then trims the re-registrations
        so subsequent tests see byte-identical state.

        (T3.A's "idempotent on empty dict" test no longer applies
        because P8 is populated. The runtime guarantee is: the boot-
        time registration in this file fires exactly once at module
        load; re-invocation is an explicit test-only operation that
        must be cleaned up.)
        """
        mod = _import_dispatch_module()
        from sndr.apply import _state

        before_len = len(_state.PATCH_REGISTRY)
        n_entries = len(mod._RETIRED_PATCHES)
        try:
            mod._register_retired_patches()
            assert len(_state.PATCH_REGISTRY) == before_len + n_entries
        finally:
            if len(_state.PATCH_REGISTRY) > before_len:
                del _state.PATCH_REGISTRY[before_len:]

    def test_synthetic_entry_registers_one_pair(self, monkeypatch):
        """T3.B preview: when ``_RETIRED_PATCHES`` gains an additional
        entry, ``_register_retired_patches()`` appends one new
        ``(name, callable)`` tuple per entry to ``PATCH_REGISTRY``.

        Uses ``monkeypatch.setitem`` so the live dict gains the
        synthetic entry only inside this test; the appended
        registration tuples are removed before the test exits so
        subsequent test modules + the snapshot fixture see the
        canonical state.
        """
        mod = _import_dispatch_module()
        from sndr.apply import _state

        synthetic_name = "M.1.1.T3.synthetic test patch"
        synthetic_spec = mod.RetiredPatchSpec(
            name=synthetic_name,
            wrapped_name="apply_patch_synthetic_t3",
            reason="retired (synthetic test only)",
        )
        before_len = len(_state.PATCH_REGISTRY)
        n_existing = len(mod._RETIRED_PATCHES)

        monkeypatch.setitem(mod._RETIRED_PATCHES, synthetic_name, synthetic_spec)
        try:
            mod._register_retired_patches()
            # _register_retired_patches re-registers ALL entries, so the
            # registry grew by (n_existing + 1).
            assert len(_state.PATCH_REGISTRY) == before_len + n_existing + 1
            # The synthetic entry is the last one registered.
            name, fn = _state.PATCH_REGISTRY[-1]
            assert name == synthetic_name
            assert callable(fn)
            wrapped = getattr(fn, "__wrapped__", None)
            assert wrapped is not None
            assert wrapped.__name__ == "apply_patch_synthetic_t3"
            # Invoking the registered callable produces the expected result.
            result = fn()
            assert result.status == "skipped"
            assert result.name == synthetic_name
            assert result.reason == "retired (synthetic test only)"
        finally:
            # Always restore registry length so other tests +
            # snapshot fixtures see byte-identical state.
            if len(_state.PATCH_REGISTRY) > before_len:
                del _state.PATCH_REGISTRY[before_len:]
