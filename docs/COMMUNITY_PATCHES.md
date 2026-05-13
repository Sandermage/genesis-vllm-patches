# Community patches — operator + contributor guide

**Status:** Phase 5 draft (`sndr community` SDK shipped, validator rules
R-1..R-7 enforced). Phase 7 finalizes documentation.

The community SDK lets outside contributors author patches as
self-contained plugins under `plugins/community/<author>/<patch-id>/`.
Each plugin describes itself with a `manifest.yaml` (PatchManifest
schema, see `vllm/sndr_core/model_configs/schema_v2.py`). The SDK
validates manifests, discovers them via filesystem + entry-points, and
scaffolds new patches from a template.

## Quickstart

```bash
# Scaffold a new draft patch
sndr community new-patch \
    --id PN999 \
    --author your_handle \
    --family spec_decode \
    --title "PN999 — short description of what it does"

# That creates:
#   plugins/community/your_handle/PN999/
#     manifest.yaml
#     patch.py            (apply() stub — replace with real logic)
#     tests/test_pn999.py

# Implement the apply() hook
$EDITOR plugins/community/your_handle/PN999/patch.py

# List every discoverable community patch
sndr community list

# Validate before pushing
sndr community validate

# Flip publish_state from draft → review when ready for promotion
$EDITOR plugins/community/your_handle/PN999/manifest.yaml
```

## CLI surface

| Command | Purpose |
|---|---|
| `sndr community list [--json]` | Enumerate filesystem + entry-point patches. |
| `sndr community validate [--json]` | Run release-tier validator (R-1..R-7). Exit 0 clean / 1 errors. |
| `sndr community new-patch --id ... --author ... --family ...` | Scaffold a working draft plugin tree. |

## Manifest fields

See `vllm/sndr_core/model_configs/schema_v2.py::PatchManifest` for the
authoritative dataclass. The most operator-visible fields:

| Field | Required | Notes |
|---|---|---|
| `id` | yes | Patch id (e.g. `PN999`, `P107_RETRY`). Unique across the registry. |
| `namespace` | yes | `community/<author>` for contributor patches. |
| `version` | yes | semver (`MAJOR.MINOR.PATCH`). Bump on each release. |
| `lifecycle` | default `community-test` | community-test → community-validated → promoted → retired |
| `implementation_status` | default `experimental` | experimental | beta | stable | deprecated | disabled |
| `publish_state` | default `draft` | draft → review → published → rejected (§6.6 no-stub gate) |
| `type` | default `runtime_hook` | runtime_hook | text_patch | composite |
| `env_flag` | when default_on | env var that toggles the patch at runtime |
| `default_on` | default false | If true: must be stable AND published (rule R-7) |
| `compatibility.min_vllm_pin` | optional | Earliest vllm pin where this patch is safe |
| `entry_points.apply` | required for runtime_hook | `module.path:callable_name` |
| `target_files` | required for text_patch | path + module + md5 anchor + pristine fixture |
| `tests_required` | release | Globs matching real test files (rule R-5) |
| `conflicts_with` / `requires_patches` | optional | Cross-references (rules R-2/R-3) |

## Validator rules

`sndr community validate` enforces seven release-tier rules on top of
the schema-level validation:

| Rule | Severity | What it catches |
|---|---|---|
| schema | error | Shape-level invariants from PatchManifest.validate() |
| R-1 | error | text_patch `context_md5` mismatch vs pristine fixture |
| R-2 | error | `requires_patches` references a patch id that doesn't exist |
| R-3 | warning | `conflicts_with` references a non-existent id (typo → silent no-op) |
| R-4 | error | `runtime_hook` entry_points.apply is not importable / not callable |
| R-5 | error | A `tests_required` glob matches no files |
| R-6 | error | `(namespace, id)` collision across the registry |
| R-7 | error | `default_on=True` patch is not implementation_status=stable AND publish_state=published |

Errors block release; warnings don't (but flag operator attention).

## Discovery: filesystem + entry-points

The SDK has two discovery paths:

1. **Filesystem** — walks `plugins/community/**/manifest.yaml`. The
   author/patch directory layout is `plugins/community/<author>/<id>/`.
   Directories whose name starts with `_` (e.g. `_template`,
   `_private`) are SKIPPED — that's how the reference template stays
   out of the release registry.

2. **Entry-points** — reads the `vllm.community_patches` setuptools
   entry-point group. Each entry resolves to a `PatchManifest`, a
   `dict`, or a `Path` to a YAML file. Useful for distributing patches
   as installable packages (`pip install your-vllm-patch-pack`).

`discover_all()` merges both, deduplicating by `(namespace, id)` —
filesystem wins on conflict (lets a local clone override an installed
package patch).

## Anchors (text_patch only)

Text patches must point at structural anchors, NEVER raw line numbers
(research lesson 7 — line numbers drift on every upstream rebase). The
manifest declares:

- `path` — vllm source file path
- `target_module` — dotted module
- `target_callable` — function/method name
- `context_md5` — md5 of the unchanged slice we anchor on
- `pristine_fixture` — path to the slice we anchor against
- `anchors[]` — per-operation context_before / context_after / what_we_do

The validator's R-1 rule re-computes the fixture's md5 and rejects
the patch if it drifted. When upstream changes, you must re-anchor
and re-bench.

## Promotion: draft → review → published

1. **draft** — work-in-progress. Never ships in release registry.
2. **review** — author thinks it's ready. PR opens; release gate runs
   the validator.
3. **published** — passes all validator rules + a maintainer review
   approves. Patch can be `default_on: true` once published.
4. **rejected** — failed review; reasoning captured in PR comments.

`publish_state` is orthogonal to `implementation_status`:

- A `published` patch can still be `experimental` (community wants to
  try it, but not yet default-on).
- A `stable` patch must be `published` if `default_on: true` (rule R-7).

## Roadmap context

- Phase 5 — community SDK MVP (this guide).
- Phase 6 — bench/log naming includes community patches (composed-key
  bench artefacts capture `enable: PN<id>` overrides).
- Phase 7 — full docs polish + CI gate (`make audit-configs` validates
  manifests, `make audit-community` runs release-tier validator).
- Phase 10 — long-tail patch integration uses this SDK for every new
  patch (canonical patches pass through the same pipeline).

## Reference

- `vllm/sndr_core/model_configs/schema_v2.py` — PatchManifest +
  PatchCompatibility + PatchTargetFile + PatchAnchor.
- `vllm/sndr_core/community/manifest.py` — load + path enumeration.
- `vllm/sndr_core/community/discovery.py` — filesystem + entry-points.
- `vllm/sndr_core/community/validator.py` — release-tier rules.
- `vllm/sndr_core/community/scaffold.py` — scaffold generator.
- `vllm/sndr_core/cli/community.py` — CLI surface.
- `plugins/community/_template/` — reference example layout.
