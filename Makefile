# Genesis vLLM Patches — operator shortcuts.
#
# Quick reference: `make help` lists every target with one-line description.
#
# Most common workflows:
#   make test          — run full test suite locally
#   make gates         — run the 4 CI gates (pin-gate, iron-rule, family, audit)
#   make audit         — run all 6 maintenance audit tools
#   make docs          — regenerate auto-docs (PATCHES_AUTO.md, CONFIGS_AUTO.md)
#   make precommit     — install + run pre-commit hooks on all files
#   make paths-env     — render canonical paths env file

.PHONY: help test gates audit docs precommit paths-env clean \
        test-pin-gate test-iron-rule test-family test-doc-sync \
        audit-upstream audit-yaml docs-check docs-write doctor \
        evidence evidence-release evidence-json

# Default target — show help.
.DEFAULT_GOAL := help

PYTHON ?= python3
PYTEST ?= $(PYTHON) -m pytest

help: ## Show this help message
	@echo "Genesis vLLM Patches — operator shortcuts"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "  See docs/CONTRIBUTING.md + Genesis_internal_docs/OPERATOR_RUNBOOK_*.md"

# ─── Tests ─────────────────────────────────────────────────────────────

test: ## Full pytest suite (all 1598+ tests)
	$(PYTEST) tests/ -q

test-pin-gate: ## Pin-gate adoption test (KNOWN_GOOD_VLLM_PINS drift)
	$(PYTEST) tests/unit/dispatcher/test_pin_gate.py -v

test-iron-rule: ## Iron-rule-#11 retire provenance meta-test
	$(PYTEST) tests/unit/dispatcher/test_iron_rule_11_enforcement.py -v

test-family: ## All 17 family contracts (~700 tests, covers 18/18 families)
	$(PYTEST) tests/unit/integrations/ -q

test-doc-sync: ## Doc-sync (patch counts consistent across 5 docs)
	$(PYTHON) scripts/check_doc_sync.py --strict

audit-phase3: ## Phase 3 relocation invariants (R1/R2/R3/R4)
	$(PYTHON) scripts/audit_phase3_relocation.py

audit-v2-runtime-pins: ## V2 runtime image + ModelDef pin harmonization (R-PIN-1..4)
	$(PYTHON) scripts/audit_v2_runtime_pins.py

gates: test-pin-gate test-iron-rule test-family test-doc-sync audit-phase3 audit-v2-runtime-pins ## Run all 6 CI gates fast-fail

# ─── Audits ────────────────────────────────────────────────────────────

audit-upstream-watchlist: ## Etap 5.1: validate UPSTREAM_WATCHLIST.yaml + emit PORT_CANDIDATE/WATCH report
	$(PYTHON) scripts/audit_upstream_watchlist.py

audit-upstream: audit-upstream-watchlist ## Audit PATCH_REGISTRY vs GitHub PR merge state (live)
	$(PYTHON) scripts/audit_upstream_status.py

audit-upstream-offline: audit-upstream-watchlist ## Audit registry sanity offline (no gh API)
	$(PYTHON) scripts/audit_upstream_status.py --skip-network

audit-yaml: ## YAML genesis_env vs docker inspect drift (env: YAML CONTAINER SSH_HOST)
	@echo "Usage: tools/audit_yaml_vs_runtime.sh <yaml> <container> [ssh_host]"
	@bash tools/audit_yaml_vs_runtime.sh \
		$${YAML:-vllm/sndr_core/model_configs/builtin/a5000-2x-27b-int4-tq-k8v4.yaml} \
		$${CONTAINER:-vllm-pn95-2xa5000} \
		$${SSH_HOST:-}

audit-legacy-imports: ## Forbid vllm.sndr_core.patches / vllm._genesis active imports
	@bash scripts/check_no_legacy_imports.sh

audit-public-paths: ## Etap 6.7: forbid private LAN IPs / home paths / usernames in public docs+code
	@echo "=== audit-public-paths ==="
	@bad=$$(rg -n "192\.168\.1\.10|/home/sander|sander@|User=sander" \
	    README.md docs/ scripts/ tools/ benchmarks/ vllm/ \
	    --glob '!sndr_private/**' \
	    --glob '!**/_archive/**' \
	    --glob '!**/_internal/**' \
	    --glob '!tests/integration/baselines/**' \
	    2>/dev/null || true); \
	if [ -n "$$bad" ]; then \
	    echo "$$bad"; \
	    echo ""; \
	    echo "✗ Private paths found in public files."; \
	    echo "  Replace LAN IPs with 127.0.0.1 / <your-host>,"; \
	    echo "  /home/sander with \$$HOME / <your-home>,"; \
	    echo "  sander@ with <user>@<host>."; \
	    exit 1; \
	fi
	@echo "✓ public-paths gate: clean"

audit-docs-stale: ## Supplement §3: forbid stale tokens (wiring/, _genesis, retired CLI verbs) in active public docs
	@$(PYTHON) scripts/docs_stale_scan.py

audit-dirty-state-dev: ## §6.3 gate #6: dirty-state policy — dev tier (informational)
	@$(PYTHON) scripts/check_dirty_state.py --tier dev

audit-dirty-state-audit: ## §6.3 gate #6: dirty-state policy — audit tier (per-PR gate)
	@$(PYTHON) scripts/check_dirty_state.py --tier audit

audit-dirty-state-release: ## §6.3 gate #6: dirty-state policy — release tier (strict)
	@$(PYTHON) scripts/check_dirty_state.py --tier release

audit-security: ## Phase 4.6 release gate — security_scan.py + (when --public-release) SBOM presence
	@$(PYTHON) scripts/security_scan.py

audit-security-release: ## Phase 4.6 strict release gate — security_scan.py --public-release
	@$(PYTHON) scripts/security_scan.py --public-release

audit-configs: ## Phase 7 gate: every V2 preset alias composes cleanly
	@$(PYTHON) scripts/audit_configs.py

audit-public-docs: ## Phase 7 / §6.10 gate: public docs boundary (no _internal links, IPs, paths, retired verbs)
	@$(PYTHON) scripts/audit_public_docs.py

audit-no-stub: ## §10.3 #2 / §10.5 no-stub gate: bare `raise NotImplementedError` / `TODO(...)` / sentinel pass in vllm/sndr_core
	@$(PYTHON) scripts/audit_no_stub.py

audit-schema-sync: ## P0-3 (audit 2026-05-14): patch_entry schemas (package + root mirror) byte-identical
	@$(PYTHON) scripts/audit_schema_sync.py

audit-patch-attribution: ## Phase A (2026-05-16): ModelDef.patches_attribution keys + role-presence consistency
	@$(PYTHON) scripts/audit_patch_attribution.py

audit-patch-plan-resolves: ## Phase D (2026-05-16): every V2 preset resolves cleanly under compat/safe/minimal
	@$(PYTHON) scripts/audit_patch_plan_resolves.py

audit-engine-boundary: ## §10.3 #5 engine boundary: only optional-discovery `vllm.sndr_engine` imports in sndr_core
	@$(PYTHON) scripts/audit_engine_boundary.py

audit-config-keys: ## §10.3 #4 / §6.7 canonical env-key registry: every committed YAML's Genesis/SNDR keys in canonical union
	@$(PYTHON) scripts/audit_config_keys.py

audit-evidence-freshness: ## §10.3 #3 evidence ledger freshness (operator-tier; skipped on CI when ledger absent)
	@$(PYTHON) scripts/audit_evidence_freshness.py

audit-license-anchor: ## P1-6: warn when development-only trust anchor still active
	@$(PYTHON) scripts/audit_license_anchor.py

audit-license-anchor-release: ## P1-6 strict: refuse release when development-only trust anchor still active
	@$(PYTHON) scripts/audit_license_anchor.py --release

audit-artifacts: ## Phase 7 / §6.11 gate: artefact storage policy (ledger, patch-proof, rollback playbook)
	@$(PYTHON) scripts/audit_artifacts.py

audit-artifacts-release: ## Phase 7 strict release gate: artefact storage + SBOM + constraints present
	@$(PYTHON) scripts/audit_artifacts.py --public-release

audit-community: ## Phase 7 gate: community SDK release-tier validator (R-1..R-7)
	@$(PYTHON) -m vllm.sndr_core.cli community validate

audit-no-new-v1: ## Phase 9 freeze gate: top-level builtin/*.yaml matches frozen baseline
	@$(PYTHON) scripts/audit_no_new_v1.py

audit-patches-prove: ## §6.8 R1 mitigation: dead-patch detector (lists patches without proof artefacts)
	@$(PYTHON) -m vllm.sndr_core.cli patches prove --dead-detect

audit-patches-prove-all: ## §6.8 release gate: run static checks on every PATCH_REGISTRY entry
	@$(PYTHON) -m vllm.sndr_core.cli patches prove --all --no-write

audit-proof-status: ## §6.8 read-side: bucket summary of every patch's proof-artefact state (informational)
	@$(PYTHON) -m vllm.sndr_core.cli patches proof-status

audit-release-check: ## §6.8 release-gate consumer — every patch must have a static proof (gating in make evidence --release)
	@$(PYTHON) -m vllm.sndr_core.cli patches release-check --mode require-static

audit-release-check-bench-attached: ## §6.8 ratchet 1: every patch must have at least one bench attachment (bridge to require-baseline)
	# Bridge between require-static (current public gate) and the strict
	# require-baseline below. Run this when promoting the default-on
	# subset of a production preset; not part of `make evidence --release`.
	# See docs/RELEASE_POLICY.md for the policy lifecycle.
	@$(PYTHON) -m vllm.sndr_core.cli patches release-check --mode require-bench

audit-release-check-baseline-optional: ## §6.8 ratchet 2: every patch must carry a bench_with_baseline proof (strict)
	# Informational by design: 0/169 entries have bench_with_baseline today.
	# Wiring this into `make evidence --release` as a hard gate would block
	# every release until operators re-bench all 169 entries on their rig.
	# Operators preparing a hardened deploy run this target directly after
	# the bench-attached ratchet above clears the default-on subset.
	# See docs/RELEASE_POLICY.md for the cutover procedure.
	@$(PYTHON) -m vllm.sndr_core.cli patches release-check --mode require-baseline

audit-model-baselines: ## Phase 7 supplement: every V2 model's reference_metrics_ref must point at an existing JSON file
	@$(PYTHON) scripts/audit_model_baselines.py

audit-launch-coverage: ## §4.2 V2 hardware schema: every V2 hardware YAML must cover canonical mount + env slots
	@$(PYTHON) scripts/audit_launch_coverage.py

config-v2-complete: ## §4.2 auto-completer: inject missing canonical mounts + env keys into V2 hardware YAMLs (check-only; pass ARGS=--write to rewrite)
	@$(PYTHON) scripts/config_v2_complete.py $(ARGS)

audit-v2-env-keys: ## §4.2 cross-layer env-key consistency: every Genesis/SNDR key across V2 model/profile/resolved-alias must be in canonical registry
	@$(PYTHON) scripts/audit_v2_env_keys.py

audit-bench-methodology: ## §6.8/§5 stale-bench detector: every bench_delta.methodology_sha must match current tools/bench_methodology.yaml SHA
	@$(PYTHON) scripts/audit_bench_methodology.py

audit-no-hardcoded-paths: ## §6.10 portability: active config (V1/V2 + compose) must use ${var} placeholders, no /home/USER or /Users/USER hardcoded paths
	@$(PYTHON) scripts/audit_no_hardcoded_paths.py

audit-v2-required-fields: ## §4.2 V2 schema: each V2 model/hardware/profile/preset YAML must carry the canonical required-field set
	@$(PYTHON) scripts/audit_v2_required_fields.py

audit-v2-freshness: ## §4.2 V2 model `last_validated` not older than 180 days (override via ARGS=--max-age-days N)
	@$(PYTHON) scripts/audit_v2_freshness.py $(ARGS)

audit-v2-id-consistency: ## §4.2 each V2 model/hardware/profile YAML's `id:` must equal its filename stem
	@$(PYTHON) scripts/audit_v2_id_consistency.py

audit-v2-license-coverage: ## §4.2/§6.10 each V2 model has SPDX-recognized `license:` + non-empty `maintainer:`
	@$(PYTHON) scripts/audit_v2_license_coverage.py

audit-v2-cross-reference: ## §4.2 every profile.parent_model + preset.{model,hardware,profile} ref resolves to a real file
	@$(PYTHON) scripts/audit_v2_cross_reference.py

audit-v2-vllm-pin-consistency: ## §4.2 model.versions.vllm_pin_required must equal baseline JSON's recorded vllm version
	@$(PYTHON) scripts/audit_v2_vllm_pin_consistency.py

audit-v2-patch-lifecycle: ## §4.2 V2 model.patches enabled lifecycle hygiene (retired patches require ALLOWED_RETIRED_PATCHES allowlist)
	@$(PYTHON) scripts/audit_v2_patch_lifecycle.py

audit-v2-hardware-sanity: ## §4.2 V2 hardware YAML numeric fields within sane bounds (cuda_capability_min, n_gpus, vram, gpu_memory_utilization, etc.)
	@$(PYTHON) scripts/audit_v2_hardware_sanity.py

audit-v2-patch-dependencies: ## §4.2 every enabled V2 patch's requires_patches/conflicts_with are satisfied
	@$(PYTHON) scripts/audit_v2_patch_dependencies.py

audit-v2-default-on-mismatch: ## §4.2 informational: surfaces explicit GENESIS_ENABLE_X='0' overrides on default_on=True patches
	@$(PYTHON) scripts/audit_v2_default_on_mismatch.py

audit-v2-capability-coverage: ## §4.2 every V2 model.capabilities string in frozen allowed set
	@$(PYTHON) scripts/audit_v2_capability_coverage.py

audit-v2-versions-pin-format: ## §4.2 V2 model.versions.{vllm_pin_required, genesis_pin_min} match canonical pin-format regex
	@$(PYTHON) scripts/audit_v2_versions_pin_format.py

audit-v2-quantization-coverage: ## §4.2 V2 model.quantization + dtype in frozen allowed set
	@$(PYTHON) scripts/audit_v2_quantization_coverage.py

audit-v2-context-length-sanity: ## §4.2 V2 hardware.sizing.max_model_len + max_num_batched_tokens within sane bounds + consistent
	@$(PYTHON) scripts/audit_v2_context_length_sanity.py

audit-v2-runtime-image-pin: ## §4.2 V2 hardware.runtime.docker.image_digest must be a canonical <repo>@sha256:<64-hex> pin
	@$(PYTHON) scripts/audit_v2_runtime_image_pin.py

audit-v2-network-port-consistency: ## §4.2 V2 hardware.runtime.docker network ports + shm_size + network name valid
	@$(PYTHON) scripts/audit_v2_network_port_consistency.py

audit-runtime-hook-ratchet: ## §4.2 P2.3: every stable patch declares stable_kind; runtime-hook kind requires production_validated_pins ≥2
	@$(PYTHON) scripts/audit_runtime_hook_ratchet.py

cold-install-smoke: ## Phase 8a: non-destructive smoke (CLI/imports/audit) — operator runs after `git clone` + `pip install -e .`
	@bash scripts/cold_install_smoke.sh $(ARGS)

audit-all-referents: ## §8 open item: F822 pure-Python gate — every `__all__` name must resolve
	@$(PYTHON) scripts/lint_all_referents.py

audit-readme-counters: ## §8 open item: README.md patch/V2 counters match live registry
	@$(PYTHON) scripts/sync_readme_counters.py --check

readme-sync: ## §8 open item: rewrite README.md counters to live registry values
	@$(PYTHON) scripts/sync_readme_counters.py

discover-apply-modules: ## Entry 12 follow-up: propose apply_module values for PATCH_REGISTRY metadata gap
	@$(PYTHON) scripts/discover_apply_modules.py

audit: audit-upstream-offline test-doc-sync audit-legacy-imports audit-public-paths ## Run audit suite (offline-safe subset)
	@echo "✓ Audit suite complete. Run 'make audit-upstream' for live gh check."

evidence: ## Phase 0 supplement: run every release gate, emit summary
	@$(PYTHON) scripts/make_evidence.py

evidence-release: ## Phase 0 supplement: aggregate + release-only gates (dirty-state, SBOM)
	@$(PYTHON) scripts/make_evidence.py --release

evidence-json: ## Phase 0 supplement: aggregate + JSON summary for CI
	@$(PYTHON) scripts/make_evidence.py --json

# ─── Docs ──────────────────────────────────────────────────────────────

docs-check: ## Verify PATCHES_AUTO.md + CONFIGS_AUTO.md in sync (CI gate)
	$(PYTHON) scripts/generate_patches_md.py --check
	$(PYTHON) scripts/generate_configs_md.py --check
	$(PYTHON) scripts/check_doc_sync.py --strict

docs-write: ## Regenerate auto-docs from registry + builtin YAMLs
	$(PYTHON) scripts/generate_patches_md.py
	$(PYTHON) scripts/generate_configs_md.py
	@echo "✓ Regenerated docs/PATCHES_AUTO.md + docs/CONFIGS_AUTO.md"

docs: docs-write docs-check ## Regenerate + verify auto-docs

# ─── Pre-commit ────────────────────────────────────────────────────────

precommit-install: ## Install pre-commit hooks (one-time per clone)
	$(PYTHON) -m pip install pre-commit
	pre-commit install

precommit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

# ─── Paths / env file ──────────────────────────────────────────────────

paths-env: ## Render canonical paths to ~/.genesis_paths.env
	$(PYTHON) scripts/emit_paths_env.py > ~/.genesis_paths.env
	@echo "✓ Wrote ~/.genesis_paths.env (source it in start-scripts)"

paths-print: ## Pretty-print all Genesis paths (debug)
	$(PYTHON) scripts/emit_paths_env.py --print

# ─── Maintenance ───────────────────────────────────────────────────────

clean: ## Remove __pycache__, .pytest_cache, .bak files
	find . -name "__pycache__" -type d -not -path "*/.git/*" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.bak" -not -path "*/.git/*" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache 2>/dev/null || true
	@echo "✓ Cleaned"

doctor: ## Run sndr doctor (genesis CLI health check)
	$(PYTHON) -m vllm.sndr_core.cli doctor

# ─── Integration (gated on GENESIS_INTEGRATION_ENDPOINT) ─────────────
#
# Все таргеты по умолчанию идут в localhost (порт 8101 для 27B, 8000
# для 35B). Для remote rig задавайте `HOST=http://<host>:<port>`.
# Жёстко вшитый LAN IP убран — operator-specific endpoint лежит только
# в `~/.sndr/host.yaml` (используйте `sndr host init`).

integration-27b: ## Run regression bounds against 27B PROD (env: HOST=<url>)
	@echo "Running integration regression bench against 27B PROD..."
	GENESIS_INTEGRATION_ENDPOINT=$${HOST:-http://127.0.0.1:8101/v1} \
	GENESIS_INTEGRATION_API_KEY=$${API_KEY:-genesis-local} \
	GENESIS_INTEGRATION_MODEL=$${MODEL:-qwen3.6-27b} \
	GENESIS_INTEGRATION_BASELINE=tests/integration/baselines/27b_v11_wave9.json \
	$(PYTEST) tests/integration/test_patch_regression_bounds.py -v

integration-35b: ## Run regression bounds against 35B PROD (env: HOST=<url>)
	@echo "Running integration regression bench against 35B PROD..."
	GENESIS_INTEGRATION_ENDPOINT=$${HOST:-http://127.0.0.1:8000/v1} \
	GENESIS_INTEGRATION_API_KEY=$${API_KEY:-genesis-local} \
	GENESIS_INTEGRATION_MODEL=$${MODEL:-qwen3.6-35b-a3b} \
	GENESIS_INTEGRATION_BASELINE=tests/integration/baselines/35b_v11_wave9.json \
	$(PYTEST) tests/integration/test_patch_regression_bounds.py -v

long-ctx-27b: ## Long-context smoke against 27B PROD (env: HOST, MAX_CTX)
	HOST=$${HOST:-http://127.0.0.1:8101} \
	MODEL=$${MODEL:-qwen3.6-27b} \
	START_CTX=$${START_CTX:-4096} \
	MAX_CTX=$${MAX_CTX:-131072} \
	bash tools/long_ctx_smoke.sh

long-ctx-35b: ## Long-context smoke against 35B PROD (env: HOST, MAX_CTX)
	HOST=$${HOST:-http://127.0.0.1:8000} \
	MODEL=$${MODEL:-qwen3.6-35b-a3b} \
	START_CTX=$${START_CTX:-4096} \
	MAX_CTX=$${MAX_CTX:-262144} \
	bash tools/long_ctx_smoke.sh

soak-1h-27b: ## 1h soak against 27B PROD (env: HOST)
	HOST=$${HOST:-http://127.0.0.1:8101} \
	MODEL=$${MODEL:-qwen3.6-27b} \
	DURATION_S=3600 \
	bash tools/soak.sh

soak-8h-27b: ## 8h soak against 27B PROD overnight (env: HOST)
	HOST=$${HOST:-http://127.0.0.1:8101} \
	MODEL=$${MODEL:-qwen3.6-27b} \
	DURATION_S=28800 \
	bash tools/soak.sh

smoke-content: ## OpenAI smoke — assert message.content not null (P0-3 audit gate)
	$(PYTHON) tools/openai_smoke.py \
		--host $${HOST:-http://127.0.0.1:8101} \
		--api-key $${API_KEY:-genesis-local} \
		--model $${MODEL:-qwen3.6-27b} \
		--max-tokens $${MAX_TOKENS:-128} \
		--enable-thinking $${ENABLE_THINKING:-false} \
		--assert-content

# ─── Aggregate ─────────────────────────────────────────────────────────

ci: gates docs-check audit-upstream-offline ## Run everything CI runs (no gh API call)
	@echo ""
	@echo "✓ All CI gates passed locally. Safe to push."
