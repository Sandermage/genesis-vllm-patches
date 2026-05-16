# Project map — где что лежит

Один документ-навигатор для maintainer / contributor. Указывает на
canonical entry point для каждой задачи: который скрипт запустить,
где живёт реализация, где соответствующие тесты, где документация.

Этот файл — **публичный** (не `_internal/`), поэтому не содержит
operator paths, IPs или server-container names.

## Содержание

- [Quick start — что запускать в первую очередь](#quick-start)
- [scripts/ — каталог CLI-скриптов](#scripts)
- [vllm/sndr_core/ — пакет ядра](#vllmsndr_core)
- [tests/ — тестовые корни](#tests)
- [docs/ — публичная документация](#docs)
- [Makefile — release-tier audit gates](#makefile)

## Quick start

| Если хочется... | Используйте |
|---|---|
| Понять состояние установки | `sndr doctor` |
| Запустить preset | `sndr launch <key>` |
| Просмотреть patch registry | `sndr patches list` |
| Объяснить patch | `sndr patches explain <id>` |
| Прогнать static-check на все патчи | `sndr patches prove --all` |
| Проверить release-готовность | `make evidence` |
| Smoke-тест чистого clone | `make cold-install-smoke` |
| Создать community-patch скелет | `sndr community new-patch ...` |
| Inventory env-ключей | `sndr config-keys list` |

## scripts/

Все CLI-скрипты живут в `scripts/`. Конвенция: каждый имеет
docstring + `--help`, плюс exit-codes 0/1/2 (success/violation/error).

### Audit gates (release-tier)

Запускаются по одному либо через `make evidence`. Каждый — gating или
informational; список в `scripts/make_evidence.py::GATES`.

| Скрипт | Что проверяет | Severity |
|---|---|---|
| `audit_artifacts.py` | release-tier artefact policy (SBOM, constraints) | release-only |
| `audit_bench_methodology.py` | `bench_delta.methodology_sha` matches contract | gating |
| `audit_config_keys.py` | every YAML's Genesis/SNDR env keys in canonical registry | gating |
| `audit_configs.py` | every V2 preset alias composes cleanly | gating |
| `audit_engine_boundary.py` | only guarded `vllm.sndr_engine` imports in sndr_core | gating |
| `audit_evidence_freshness.py` | ledger ≤7 days OR contains HEAD sha | informational |
| `audit_launch_coverage.py` | every V2 hardware YAML covers canonical mounts | gating |
| `audit_model_baselines.py` | reference_metrics_ref points at existing JSON | gating |
| `audit_no_hardcoded_paths.py` | active config uses ${var}, no /home/USER paths | gating |
| `audit_no_new_v1.py` | top-level V1 YAML matches frozen baseline | gating |
| `audit_no_stub.py` | no bare `raise NotImplementedError` / `TODO`-with-name markers | gating |
| `audit_public_docs.py` | public docs boundary (no internal links, IPs, retired verbs) | gating |
| `audit_runtime_hook_ratchet.py` | stable runtime-hook patches have ≥2 production pins | gating |
| `audit_upstream_status.py` | upstream PR queue snapshot | informational |
| `audit_upstream_watchlist.py` | drift watch — newly-merged upstream PRs that affect our anchors | informational |
| `audit_v2_*.py` (15 scripts) | per-field V2 invariants — required fields, freshness, capability, etc. | gating (one is informational) |
| `docs_stale_scan.py` | forbidden tokens in public docs (retired verbs, old paths) | gating |
| `security_scan.py` | secrets, operator paths, private IPs across tracked files | informational |
| `make_evidence.py` | aggregate runner with structured ledger output | n/a (orchestrator) |

### Generators + sync

| Скрипт | Что генерирует |
|---|---|
| `generate_configs_md.py` | `docs/CONFIGS_AUTO.md` from PATCH_REGISTRY |
| `generate_patches_md.py` | `docs/PATCHES_AUTO.md` |
| `generate_sbom.py` | `release/SBOM.spdx.json` |
| `generate_trust_anchor.py` | Ed25519 keypair for license verification |
| `sync_readme_counters.py` | README.md badge / counter lines vs live registry |
| `build_anchor_manifest.py` | text-patch anchor manifest (per Stage 6 reorg) |
| `discover_apply_modules.py` | rebuilds `_apply_module_overlay.py` from legacy register |

### Operational scripts

| Скрипт | Назначение |
|---|---|
| `cold_install_smoke.sh` | non-destructive smoke from clean checkout (Phase 8a) |
| `check_dirty_state.py` | dirty-state policy (`--tier dev|audit|release`) |
| `check_doc_sync.py` | doc / code drift sanity |
| `lint_all_referents.py` | F822 — every `__all__` name resolves |
| `bench_v11_smoke.py` | server-side smoke bench |
| `fetch_models.sh` | curated model download wrapper |

### Bench / probe helpers

`scripts/genesis_*` — historical bench / quality / longbench runners.
`scripts/stress/` — multi-turn / cliff soak runners.
`scripts/git/pre-commit` — pre-commit hook installer.

### Archive

`scripts/_archive/` — retired scripts. Treat as historical, not
current. New work should not depend on these.

## vllm/sndr_core/

Single Python package, no namespace tricks.

### Top-level modules

| Module | Purpose |
|---|---|
| `__init__.py` | re-export public API + version |
| `version.py` | `__version__` constant |
| `env.py` | central enum of every Genesis/SNDR env flag |
| `brand.py` | SNDR vs Genesis brand string |
| `caveats.py` | known-host condition registry |
| `license.py` | DEV / community / engine license tier probe |
| `plugin.py` | vllm plugin entry point hook |
| `findings/` | external-finding tracker (vllm#XXXXX issue pipeline) |

### Subpackages

```text
sndr_core/
├── apply/           — text-patch + runtime-hook apply orchestrator
├── bundles/         — atomic feature bundles (multi-patch transactions)
├── cache/           — tier_manager, eviction_policies, PN95 runtime
├── cli/             — sndr <subcommand> implementations
├── community/       — community-patch SDK (manifest, validator, scaffold)
├── compat/          — schema validator, categories, self-test, deps
├── configs/         — internal config helpers
├── core/            — TextPatcher, MultiFilePatchTransaction, primitives
├── deps/            — host dependency inventory
├── detection/       — guards + model-class detection
├── dispatcher/      — PATCH_REGISTRY, decision policy, apply matrix
├── integrations/    — per-subsystem family directories (Stage 6 layout)
│   ├── attention/   (flash, gdn, turboquant)
│   ├── compile_safety/
│   ├── kernels/
│   ├── kv_cache/    — pn95_tier_aware_cache.py
│   ├── loader/
│   ├── lora/
│   ├── memory/
│   ├── middleware/
│   ├── moe/
│   ├── multimodal/
│   ├── observability/
│   ├── quantization/
│   ├── reasoning/
│   ├── scheduler/
│   ├── serving/
│   ├── spec_decode/
│   ├── tool_parsing/
│   └── worker/
├── kernels/         — pure CUDA/Triton kernel implementations
├── middleware/      — FastAPI/Starlette HTTP middleware
├── model_configs/   — V2 layered schema + composer + runtime container spec
├── runtime/         — redact, prealloc, runtime command spec
└── tests/           — legacy test residue (Stage-6 in progress)
```

### Where to find a patch

1. Browse `dispatcher/registry.py` — every patch has `apply_module`
   pointing at the implementation file.
2. Or `find vllm/sndr_core/integrations -name "<id>_*.py"`.
3. Or `sndr patches explain <id>` — prints the resolved path.

## tests/

| Path | What |
|---|---|
| `tests/unit/cache/` | TierManager / PN95 / eviction policies (~14 files, 175+ tests) |
| `tests/unit/community/` | community SDK + reference template |
| `tests/unit/compat/` | schema validator, self-test, plugin signature |
| `tests/unit/integrations/<family>/` | per-patch unit tests (mirrors integrations layout) |
| `tests/unit/scripts/` | tests for `scripts/audit_*.py` and friends |
| `tests/unit/cli/` | CLI subcommand tests |
| `tests/unit/dispatcher/` | dispatcher decision + matrix tests |
| `tests/unit/model_configs/` | V2 schema / composer tests |
| `tests/legacy/` | pre-Stage-6 tests still passing |
| `tests/soak/` | long-running stability tests |
| `tests/probes/` | live-probe scripts (require server) |
| `tests/bench/` | bench harnesses |

Run: `python3 -m pytest tests/unit -q` (~5700 collected, of which the
tracked-only subset is the release baseline; some tests in
`tests/unit/integrations/` skip on Mac without CUDA / cryptography).

## docs/

| File | Purpose |
|---|---|
| `README.md` (root) | public landing — what Genesis is + quickstart |
| `docs/INSTALL.md` | install playbook |
| `docs/DAY_1_CHECKLIST.md` | post-install verification steps |
| `docs/COMMANDS.md` | `sndr <verb>` reference |
| `docs/CONFIG_SYSTEM_V2.md` | layered config — model + hardware + profile + preset |
| `docs/MODEL_CONFIG_LAUNCHER.md` | `sndr launch` walkthrough |
| `docs/PATCHES.md` | high-level patch catalogue |
| `docs/TROUBLESHOOTING.md` | known hardware/software cliffs |
| `docs/PATCHES.md` | vLLM pin matrix + retired patches |
| `docs/COMMUNITY_PATCHES.md` | community SDK quickstart |
| `docs/PLUGINS.md` | plugin author guide |
| `docs/TROUBLESHOOTING.md` | R-001..R-008 recovery procedures |
| `docs/SELF_TEST.md` | `sndr self-test` semantics |
| `docs/BENCHMARK_GUIDE.md` | how to run our benches |
| `docs/PROJECT_MAP.md` | **this file** |

Internal planning, research, and audit notes live in a parallel,
gitignored documentation tree (operator-managed); they are not part
of the release artefact.

## Makefile

`make evidence` runs every gate; full list lives in
`scripts/make_evidence.py::GATES`. Individual gates: `make audit-X`
(autocomplete-friendly).

Convenience targets:

| Target | What |
|---|---|
| `make audit` | legacy aggregate (legacy-imports + public-paths + upstream + doc-sync) |
| `make evidence` | aggregate runner across every gate |
| `make cold-install-smoke` | Phase 8a smoke test |
| `make audit-dirty-state-dev` | dev-tier dirty-state check |
| `make audit-dirty-state-audit` | per-PR dirty-state gate |
| `make audit-dirty-state-release` | strict release dirty-state |
| `make audit-release-check-strict` | every patch must have bench-with-baseline proof |
