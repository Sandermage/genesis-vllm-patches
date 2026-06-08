# SPDX-License-Identifier: Apache-2.0
"""CONFIG-UX.3 — native `sndr preset` CLI (list / show / explain / recommend).

Replaces the legacy bridged stub. Renders the operator-product surface
defined by `PresetCard` (CONFIG-UX.1) on top of the typed loader
(`load_preset_def`). Composition output is reused from the existing
`compose()` path; this module only renders metadata + dry-run views.

Hard constraints (CONFIG_UX_R §5 + operator's CONFIG-UX.3 locked rules):

  - No torch import. CLI must work without a GPU.
  - Card data drives the surface; missing-card presets degrade gracefully
    (`list` shows them tagged "unannotated"; `recommend` skips them).
  - `recommend` honors `workload_deny`: a preset is excluded from results
    when the queried workload is in its deny list, even if its allow list
    is broad or empty.
  - Workload taxonomy is the frozen `KNOWN_WORKLOADS` tuple plus a
    `custom:<slug>` escape; recommend rejects other forms.

Output modes:
  - default: human-readable table / paragraph view (rustup-style)
  - --json:  machine-readable dump (used by tests + tooling)

Four leaves:

  sndr preset list [--family X] [--workload Y] [--hardware Z] [--mode M]
                   [--status S] [--json]
  sndr preset show <preset_id> [--field <dot.path>] [--json]
  sndr preset explain <preset_id> [--json]
  sndr preset recommend --workload <W> [--hardware H] [--concurrency N]
                        [--top N] [--json]
"""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Optional

from sndr.product_api import presets as preset_api

from . import _io


__all__ = [
    "add_argparser",
    "run_list",
    "run_show",
    "run_explain",
    "run_recommend",
]


# Status priority for `recommend` ranking — higher index = lower priority.
_STATUS_RANK: dict[str, int] = preset_api.STATUS_RANK


# ─── argparse registration ──────────────────────────────────────────────────


def add_argparser(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "preset",
        help="V2 preset surface — list/show/explain/recommend operator cards.",
        description=(
            "Operator-product view on top of V2 preset triplets (model + "
            "hardware + profile). Cards (`card:` in preset YAML) drive the "
            "list/show/explain/recommend surface; missing-card presets "
            "degrade gracefully. See CONFIG_UX_R §2 for the card schema."
        ),
    )
    sub = p.add_subparsers(dest="preset_cmd", required=True)

    # — list —
    p_list = sub.add_parser(
        "list",
        help="List presets in a table; filter by family / workload / hardware / mode / status.",
    )
    p_list.add_argument("--family", default=None,
                        help="Filter by card.routing_family.")
    p_list.add_argument("--workload", default=None,
                        help="Filter by workload_allow (intersection check).")
    p_list.add_argument("--hardware", default=None,
                        help="Filter by composed hardware id.")
    p_list.add_argument("--mode", default=None,
                        help="Filter by card.mode.")
    p_list.add_argument("--status", default=None,
                        help="Filter by card.status.")
    p_list.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON.")
    p_list.set_defaults(func=run_list)

    # — show —
    p_show = sub.add_parser(
        "show",
        help="Display card-formatted view of one preset.",
    )
    p_show.add_argument("preset_id", help="Preset alias id.")
    p_show.add_argument("--field", default=None,
                        help="Drill-down dot-path (e.g. 'card.evidence_refs.0.path').")
    p_show.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON.")
    p_show.set_defaults(func=run_show)

    # — explain —
    p_explain = sub.add_parser(
        "explain",
        help="Operator walkthrough: card narrative + composed runtime + fallback diff.",
    )
    p_explain.add_argument("preset_id", help="Preset alias id.")
    p_explain.add_argument("--json", action="store_true",
                           help="Emit machine-readable JSON.")
    p_explain.set_defaults(func=run_explain)

    # — recommend —
    p_recommend = sub.add_parser(
        "recommend",
        help="Inverse lookup: operator describes workload, CLI proposes presets.",
    )
    p_recommend.add_argument("--workload", required=True,
                             help=("Workload class (one of KNOWN_WORKLOADS or "
                                   "`custom:<slug>`)."))
    p_recommend.add_argument("--hardware", default=None,
                             help="Filter by composed hardware id.")
    p_recommend.add_argument("--concurrency", type=int, default=None,
                             help="Concurrency level — must fall within "
                                  "card.concurrency.[min..max].")
    p_recommend.add_argument("--top", type=int, default=5,
                             help="Return at most N ranked presets (default 5).")
    p_recommend.add_argument("--json", action="store_true",
                             help="Emit machine-readable JSON.")
    p_recommend.set_defaults(func=run_recommend)


# ─── Helpers ────────────────────────────────────────────────────────────────


def _load_corpus() -> list[tuple[str, Any]]:
    """Load all builtin presets as (alias_id, PresetDef) tuples.

    Lazy import keeps top-level torch-free check honest — `model_configs`
    has no torch dependency but the import chain through plugin loaders
    can be heavy at startup if vllm pulls anything in.
    """
    return list(preset_api.load_corpus())


def _compose_for(preset_id: str):
    """Compose to V1 ModelConfig (for `show` / `explain` hardware lookup).

    Lazy import + warning suppression mirror _load_corpus."""
    return preset_api.compose_for(preset_id)


def _hardware_id_of(preset_id: str, pd) -> Optional[str]:
    """Resolve the hardware id from preset triplet (without full compose)."""
    return pd.hardware


def _card_dict(card) -> dict:
    """Serialize a PresetCard dataclass to a plain dict (JSON-safe)."""
    return preset_api.card_to_dict(card)


def _preset_dict(alias_id: str, pd) -> dict:
    """Full JSON view: triplet + card."""
    return asdict(preset_api.preset_to_record(alias_id, pd))


def _drill(obj: Any, path: str) -> Any:
    """Walk a dot-path through nested dataclasses / lists / dicts.

    Path syntax: `card.evidence_refs.0.path` — attr → list-index → attr.
    Raises KeyError if the path can't be resolved (with the segment that
    failed) so the CLI can emit a precise error.
    """
    return preset_api.drill_field(obj, path)


# ─── list ───────────────────────────────────────────────────────────────────


def _passes_list_filters(alias_id: str, pd, args) -> bool:
    return preset_api.passes_list_filters(
        pd,
        family=args.family,
        workload=args.workload,
        hardware=args.hardware,
        mode=args.mode,
        status=args.status,
    )


def run_list(args) -> int:
    corpus = _load_corpus()
    matches = [
        (alias, pd) for alias, pd in corpus
        if _passes_list_filters(alias, pd, args)
    ]

    if args.json:
        payload = {
            "filters": {
                "family": args.family, "workload": args.workload,
                "hardware": args.hardware, "mode": args.mode,
                "status": args.status,
            },
            "matched": len(matches),
            "total": len(corpus),
            "presets": [
                {
                    "id": alias,
                    "model": pd.model,
                    "hardware": pd.hardware,
                    "profile": pd.profile,
                    "card": _card_dict(pd.card),
                    "has_card": pd.has_card(),
                }
                for alias, pd in matches
            ],
        }
        print(json.dumps(payload, indent=2, default=str))
        return 0

    if not matches:
        _io.warn("no presets match the given filters")
        return 0

    # Table view
    header = ("preset", "family", "mode", "status", "K", "conc", "title")
    rows = []
    for alias, pd in matches:
        card = pd.card
        if card is None:
            rows.append((alias, "—", "—", "(unannotated)", "—", "—", ""))
        else:
            conc_str = f"{card.concurrency.canonical}" if card.concurrency else "—"
            k_str = str(card.K) if card.K is not None else "—"
            rows.append((
                alias,
                card.routing_family or "—",
                card.mode or "—",
                card.status,
                k_str,
                conc_str,
                card.title[:60],
            ))

    # Column widths
    widths = [max(len(str(r[i])) for r in (rows + [header])) for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    print(fmt.format(*("─" * w for w in widths)))
    for r in rows:
        print(fmt.format(*r))
    print()
    _io.info(f"matched {len(matches)} / {len(corpus)} presets")
    return 0


# ─── show ──────────────────────────────────────────────────────────────────


def run_show(args) -> int:
    try:
        from sndr.model_configs.registry_v2 import load_preset_def
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            pd = load_preset_def(args.preset_id)
    except Exception as e:
        _io.error(f"preset {args.preset_id!r}: {e}")
        return 1

    payload = _preset_dict(args.preset_id, pd)

    if args.field:
        try:
            value = _drill(payload, args.field)
        except KeyError as e:
            _io.error(f"--field {args.field!r}: {e}")
            return 1
        if args.json:
            print(json.dumps(value, indent=2, default=str))
        else:
            print(value)
        return 0

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return 0

    # Human view — sectioned card render
    _render_card_human(args.preset_id, pd)
    return 0


def _render_card_human(alias_id: str, pd) -> None:
    print(f"\n  preset: {alias_id}")
    print("  composed from:")
    print(f"    model    = {pd.model}")
    print(f"    hardware = {pd.hardware}")
    print(f"    profile  = {pd.profile or '(none)'}")
    if pd.runtime:
        print(f"    runtime  = {pd.runtime}")

    card = pd.card
    if card is None:
        print("\n  (no card — CONFIG-UX.2b annotation pending)")
        return

    print(f"\n  {card.title}")
    print(f"    status:    {card.status}")
    if card.audience:
        print(f"    audience:  {card.audience}")
    if card.maturity:
        print(f"    maturity:  {card.maturity}")
    if card.mode:
        print(f"    mode:      {card.mode}")
    print("\n  Summary:")
    for line in str(card.summary).strip().splitlines():
        print(f"    {line}")

    if card.workload_allow:
        print("\n  Workload allow:")
        for w in card.workload_allow:
            print(f"    + {w}")
    if card.workload_deny:
        print("\n  Workload deny:")
        for w in card.workload_deny:
            print(f"    − {w}")

    if card.concurrency or card.K is not None or card.context:
        print("\n  Operating envelope:")
        if card.concurrency:
            print(f"    concurrency: min={card.concurrency.min} canonical="
                  f"{card.concurrency.canonical} max={card.concurrency.max}")
        if card.K is not None:
            print(f"    K (spec-decode draft tokens): {card.K}")
        if card.context:
            ctx = card.context
            print(f"    context: max_model_len={ctx.max_model_len}"
                  + (f" typical_in={ctx.typical_input_tokens}"
                     if ctx.typical_input_tokens else "")
                  + (f" typical_out={ctx.typical_output_tokens}"
                     if ctx.typical_output_tokens else ""))

    if card.routing_family or card.fallback_preset:
        print("\n  Routing:")
        if card.routing_family:
            tag = " (default for family)" if card.default_for_family else ""
            print(f"    family:   {card.routing_family}{tag}")
        if card.fallback_preset:
            print(f"    fallback: {card.fallback_preset}")

    if card.primary_metric or card.evidence_refs:
        print(f"\n  Evidence (visibility={card.evidence_visibility or 'unset'}):")
        if card.primary_metric:
            m = card.primary_metric
            print(f"    primary: {m.kind}={m.value} (source={m.source})")
        for ev in card.evidence_refs:
            vis = f" [{ev.visibility}]" if ev.visibility else ""
            note = f" — {ev.note}" if ev.note else ""
            print(f"    {ev.type}: {ev.path}{vis}{note}")

    if card.tradeoffs:
        print("\n  Tradeoffs:")
        for t in card.tradeoffs:
            print(f"    • {t}")

    if card.do_not_use:
        print("\n  Do not use:")
        for dnu in card.do_not_use:
            print(f"    ✗ {dnu.condition}")
            print(f"        — {dnu.reason}")

    print()


# ─── explain ───────────────────────────────────────────────────────────────


def run_explain(args) -> int:
    try:
        from sndr.model_configs.registry_v2 import load_preset_def
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            pd = load_preset_def(args.preset_id)
    except Exception as e:
        _io.error(f"preset {args.preset_id!r}: {e}")
        return 1

    # Compose to V1 ModelConfig for dry-run summary (this DOES touch
    # compose path; cli/preset.py reads composed cfg, doesn't mutate it).
    try:
        cfg = _compose_for(args.preset_id)
    except Exception as e:
        _io.error(f"compose failed for preset {args.preset_id!r}: {e}")
        return 1

    # Fallback diff (if card declares one).
    fallback_summary: Optional[dict] = None
    if pd.has_card() and pd.card.fallback_preset:
        try:
            fb_cfg = _compose_for(pd.card.fallback_preset)
            fallback_summary = _summarize_diff(cfg, fb_cfg, pd.card.fallback_preset)
        except Exception as e:  # pragma: no cover — defensive
            fallback_summary = {
                "fallback_preset": pd.card.fallback_preset,
                "error": f"compose failed: {e}",
            }

    payload = {
        "id": args.preset_id,
        "card": _card_dict(pd.card),
        "composed": _composed_summary(cfg),
        "fallback_diff": fallback_summary,
    }

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return 0

    _render_card_human(args.preset_id, pd)

    # Composed runtime
    print("\n  Composed runtime (dry-run):")
    cs = _composed_summary(cfg)
    for k in ("composed_key", "kv_cache_dtype", "max_model_len", "max_num_seqs",
              "gpu_memory_utilization", "spec_decode_method", "spec_decode_K",
              "enabled_patches_count"):
        if k in cs and cs[k] is not None:
            print(f"    {k:24s} {cs[k]}")

    if fallback_summary:
        if fallback_summary.get("error"):
            print(f"\n  Fallback {fallback_summary['fallback_preset']}: "
                  f"{fallback_summary['error']}")
        else:
            print(f"\n  Fallback diff vs `{fallback_summary['fallback_preset']}`:")
            for line in fallback_summary["diffs"]:
                print(f"    {line}")

    print()
    return 0


def _composed_summary(cfg) -> dict:
    """Pull a few fields out of the composed V1 ModelConfig for display."""
    return preset_api.composed_summary(cfg)


def _summarize_diff(cfg_a, cfg_b, fallback_id: str) -> dict:
    """Single-row delta between two composed configs (only the fields
    operators care about — sizing + spec-decode + KV)."""
    return preset_api.summarize_diff(cfg_a, cfg_b, fallback_id)


# ─── recommend ──────────────────────────────────────────────────────────────


def _passes_recommend_filters(
    alias_id: str, pd, *, workload: str,
    hardware: Optional[str], concurrency: Optional[int],
) -> bool:
    """Filter rules per CONFIG_UX_R §5.2.4 + operator safety amendment."""
    return preset_api.passes_recommend_filters(
        alias_id,
        pd,
        workload=workload,
        hardware=hardware,
        concurrency=concurrency,
    )


def _recommend_sort_key(alias_id: str, pd) -> tuple:
    return preset_api.recommend_sort_key(alias_id, pd)


def run_recommend(args) -> int:
    from sndr.model_configs.preset_schema import is_known_workload

    if not is_known_workload(args.workload):
        _io.error(
            f"--workload {args.workload!r} is not in KNOWN_WORKLOADS "
            f"and is not a valid `custom:<slug>` form. See "
            f"`KNOWN_WORKLOADS` in preset_schema.py."
        )
        return 1

    corpus = _load_corpus()
    matches = [
        (alias, pd) for alias, pd in corpus
        if _passes_recommend_filters(
            alias, pd,
            workload=args.workload,
            hardware=args.hardware,
            concurrency=args.concurrency,
        )
    ]
    matches.sort(key=lambda x: _recommend_sort_key(x[0], x[1]))
    matches = matches[: args.top]

    if args.json:
        payload = {
            "query": {
                "workload": args.workload,
                "hardware": args.hardware,
                "concurrency": args.concurrency,
                "top": args.top,
            },
            "results": [
                {
                    "id": alias,
                    "rank": i + 1,
                    "card": _card_dict(pd.card),
                }
                for i, (alias, pd) in enumerate(matches)
            ],
            "total_matches": len(matches),
        }
        print(json.dumps(payload, indent=2, default=str))
        return 0

    if not matches:
        _io.warn(f"no annotated preset matches workload={args.workload!r}"
                 + (f" hardware={args.hardware!r}" if args.hardware else "")
                 + (f" concurrency={args.concurrency}" if args.concurrency else ""))
        return 0

    print(f"\n  Recommended presets for workload={args.workload!r}"
          + (f" hardware={args.hardware!r}" if args.hardware else "")
          + (f" concurrency={args.concurrency}" if args.concurrency is not None else "")
          + ":\n")
    for i, (alias, pd) in enumerate(matches, start=1):
        card = pd.card
        metric = ""
        if card.primary_metric and card.primary_metric.value:
            metric = f"  {card.primary_metric.kind}={card.primary_metric.value}"
        default_tag = " ★default" if card.default_for_family else ""
        print(f"  {i}. {alias}  [{card.status}]{default_tag}{metric}")
        print(f"     {card.title}")
        if card.fallback_preset:
            print(f"     fallback: {card.fallback_preset}")
        print()
    return 0
