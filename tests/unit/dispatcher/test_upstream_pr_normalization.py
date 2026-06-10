# SPDX-License-Identifier: Apache-2.0
"""CI-wide regression guard — `upstream_pr` field normalization.

Why this matters
----------------

The PATCH_REGISTRY `upstream_pr` field historically accepts:
  - int (canonical): `41043`
  - str int: `"41043"`
  - URL string: `"https://github.com/vllm-project/vllm/pull/40886"`
  - issue URL string: `".../issues/39407"`
  - None

Downstream consumers expect `Optional[int]`:
  - `PatchSpec.upstream_pr: Optional[int]` typed dataclass field
  - `scripts/audit_upstream_status.py` regex
    ``r'"upstream_pr"\\s*:\\s*(\\d+)'``
    — silently SKIPS URL forms (only matches integer)
  - `infer_source()` checks `isinstance(meta.get("upstream_pr"), int)`
    — returns wrong source for URL-form entries
  - PATCHES_AUTO.md doc generator builds clickable PR links from
    integer numbers; URL strings are emitted verbatim, breaking
    the table layout

v11.3.0 BUG #13 discovered: 28 G4_* patches use URL form (24 issue
URLs + 4 PR URLs). Their upstream merge state is invisible to the
weekly upstream audit gate. PR URLs are recoverable (extract the
integer) — issue URLs are NOT (they reference issues, not PRs;
separate semantic).

Fix: `sndr.dispatcher.spec.normalize_upstream_pr` parses
all forms to `Optional[int]`. `PatchSpec.upstream_pr` now uses the
normalizer. Issue URLs return None (correct: they're not PRs).

This test pins:
  1. Normalizer behavior across all input forms
  2. Every registry entry's `upstream_pr` resolves cleanly to
     `Optional[int]` after normalization
  3. Issue-URL entries flagged as advisory (semantic mismatch)

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Status: v11.3.0+ regression guard.
"""
from __future__ import annotations


def test_normalize_upstream_pr_handles_all_forms():
    """Unit test for the normalizer covering each accepted input form."""
    from sndr.dispatcher.spec import normalize_upstream_pr
    # int → int
    assert normalize_upstream_pr(41043) == 41043
    # str int → int
    assert normalize_upstream_pr("41043") == 41043
    assert normalize_upstream_pr("  41043  ") == 41043
    # PR URL → int
    assert normalize_upstream_pr(
        "https://github.com/vllm-project/vllm/pull/40886"
    ) == 40886
    # Issue URL → None (semantic distinction)
    assert normalize_upstream_pr(
        "https://github.com/vllm-project/vllm/issues/39407"
    ) is None
    # None → None
    assert normalize_upstream_pr(None) is None
    # Empty / garbage → None
    assert normalize_upstream_pr("") is None
    assert normalize_upstream_pr("   ") is None
    assert normalize_upstream_pr("not-a-number") is None
    # bool guard (Python bool is int subclass — should NOT pass through)
    assert normalize_upstream_pr(True) is None
    assert normalize_upstream_pr(False) is None


def test_every_registry_upstream_pr_normalizes_cleanly():
    """After normalization, every PATCH_REGISTRY `upstream_pr` value
    is either an int (PR number) or None. No exceptions."""
    from sndr.dispatcher.registry import PATCH_REGISTRY
    from sndr.dispatcher.spec import normalize_upstream_pr
    failures: list[tuple[str, object]] = []
    for pid, meta in PATCH_REGISTRY.items():
        if not isinstance(meta, dict):
            continue
        raw = meta.get("upstream_pr")
        try:
            norm = normalize_upstream_pr(raw)
        except Exception as e:
            failures.append((pid, f"raised {type(e).__name__}: {e}"))
            continue
        if norm is not None and not isinstance(norm, int):
            failures.append((pid, f"non-int result {norm!r}"))
    assert not failures, (
        f"{len(failures)} entries fail upstream_pr normalization:\n" +
        "\n".join(f"  - {pid}: {msg}" for pid, msg in failures)
    )


def test_patch_spec_upstream_pr_is_int_or_none():
    """After PatchSpec construction, every spec's upstream_pr is
    Optional[int] — never a URL string."""
    from sndr.dispatcher.spec import iter_patch_specs
    bad = []
    for spec in iter_patch_specs():
        up = spec.upstream_pr
        if up is not None and not isinstance(up, int):
            bad.append((spec.patch_id, up))
    assert not bad, (
        f"{len(bad)} PatchSpec.upstream_pr values are not "
        f"Optional[int] after normalization:\n" +
        "\n".join(f"  - {pid}: {val!r}" for pid, val in bad[:20])
    )


def test_advisory_issue_url_count_baseline():
    """Advisory baseline: 24 registry entries store an issue URL in
    `upstream_pr` (semantic mismatch — issue ≠ PR). These resolve to
    None after normalization (correct: they're not PRs). The advisory
    surfaces them so a future cleanup can either:
      (a) move to a separate `upstream_issue` field, OR
      (b) link the actual PR that fixes the issue (if any), OR
      (c) accept the None and rely on `credit` field for context.

    Baseline at v11.3.0: 24 issue-URL entries (all G4_* model-compat
    patches that link to upstream issues describing the model-specific
    bug, NOT to a tracking PR)."""
    from sndr.dispatcher.registry import PATCH_REGISTRY
    from sndr.dispatcher.spec import is_issue_url_not_pr
    issue_url_entries = [
        pid for pid, meta in PATCH_REGISTRY.items()
        if isinstance(meta, dict)
        and is_issue_url_not_pr(meta.get("upstream_pr"))
    ]
    # Surface (informational) the count — assert <= baseline so cleanup
    # progress is rewarded; > baseline → review the new addition.
    assert len(issue_url_entries) <= 30, (
        f"Issue-URL upstream_pr count grew to {len(issue_url_entries)} "
        f"(v11.3.0 baseline ~24, ceiling 30). New entries:\n" +
        "\n".join(f"  - {pid}" for pid in issue_url_entries[:30]) +
        f"\n\nIssue URLs are advisory-only: they resolve to None at "
        f"normalization. Use `related_upstream_prs` for tracking-by-"
        f"issue when an actual PR exists."
    )
