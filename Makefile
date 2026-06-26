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
        audit-upstream audit-yaml preflight lint-drift-markers \
        tokenizer-fingerprint rebuild-pin audit-pin summarize-rej bump-preflight \
        docs-check docs-write doctor \
        evidence evidence-release evidence-json gui-build gui-lint test-gui-contract audit-i18n

# Default target — show help.
.DEFAULT_GOAL := help

PYTHON ?= python3
PYTEST ?= $(PYTHON) -m pytest

# GPU rig host for the optional SSH-driven maintainer targets below
# (rebuild-pin / audit-pin / audit-yaml). Override on the command line:
#   make rebuild-pin SSH_HOST=$(GENESIS_RIG_HOST)
# Defaults to a generic hostname so the public tree carries no internal IP.
GENESIS_RIG_HOST ?= rig.local
# Rig repo root ($(GENESIS_RIG_HOST):$(RIG_REPO)) — anchor-SoT regen target.
RIG_REPO ?= /tmp/genesis-consolidated

help: ## Show this help message
	@echo "Genesis vLLM Patches — operator shortcuts"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "  See docs/CONTRIBUTING.md + Genesis_internal_docs/OPERATOR_RUNBOOK_*.md"

# ─── Tests ─────────────────────────────────────────────────────────────

test: ## Full pytest suite (all 8400+ tests)
	$(PYTEST) tests/ -q

test-pin-gate: ## Pin-gate adoption test (KNOWN_GOOD_VLLM_PINS drift)
	$(PYTEST) tests/unit/dispatcher/test_pin_gate.py -v

test-iron-rule: ## Iron-rule-#11 retire provenance meta-test
	$(PYTEST) tests/unit/dispatcher/test_iron_rule_11_enforcement.py -v

test-family: ## All 23 family contracts (~2300 tests, covers 20/20 families)
	$(PYTEST) tests/unit/integrations/ -q

test-gui-contract: ## GUI↔backend route contract — every api.ts path has a daemon route (structural-compat drift gate)
	$(PYTEST) tests/unit/product_api/test_gui_contract.py -q

audit-i18n: ## GUI i18n coverage — every tr() string has a Russian translation (ratchet gate, baseline 0)
	$(PYTHON) scripts/audit_i18n.py

test-doc-sync: ## Doc-sync (patch counts consistent across 10 docs)
	$(PYTHON) scripts/check_doc_sync.py --strict

audit-phase3: ## Phase 3 relocation invariants (R1/R2/R3/R4)
	$(PYTHON) scripts/audit_phase3_relocation.py

audit-v2-runtime-pins: ## V2 runtime image + ModelDef pin harmonization (R-PIN-1..4)
	$(PYTHON) scripts/audit_v2_runtime_pins.py

audit-v2-modeldef-vs-hardware-pin: ## V2 ModelDef ↔ hardware canonical-pin drift (R-MD-HW-1/2, waiver-aware via pin_hold)
	$(PYTHON) scripts/audit_v2_modeldef_vs_hardware_pin.py

audit-ai-attribution: ## Attribution-label policy: forbid AI-agent Co-Authored-By / Generated-by / robot-emoji markers (§9.A.6)
	$(PYTHON) scripts/audit_ai_attribution.py

audit-links: ## Markdown link integrity: in-tree path existence (default); anchors opt-in via `--anchors` (§9.A.3)
	$(PYTHON) scripts/audit_links.py

audit-links-strict: ## Markdown link integrity STRICT: path + GitHub-slug anchor verification (operator-run; surfaces TOC drift)
	$(PYTHON) scripts/audit_links.py --anchors

audit-plan-supersession: ## Planning supersession consistency — filename-target verification + status:superseded targets (§9.A.5, operator-run; scans gitignored sndr_private/planning/)
	$(PYTHON) scripts/audit_plan_supersession.py

audit-repo-garbage: ## Preventive repo-garbage audit (merge leftovers, .DS_Store, editor backups, illegal chars, root scratchpads) (§9.A.2)
	$(PYTHON) scripts/audit_repo_garbage.py

audit-generated-links: ## Generated-doc link integrity — `[#NNNN](URL)` number/URL match + generator freshness (§9.A.4)
	$(PYTHON) scripts/audit_generated_links.py

audit-rig-divergence: ## Primary↔rig drift — local-only by default (--ssh-host + --allow-ssh for SSH mode); §9.A.9, operator-run
	$(PYTHON) scripts/audit_rig_divergence.py

audit-wheel-contents: ## Wheel-boundary invariants CLI surface — pyproject shape + canonical test-file presence (§9.A.1)
	$(PYTHON) scripts/audit_wheel_contents.py

audit-retire-eligibility: ## Retire-verdict distribution over PATCH_REGISTRY (offline by default); §9.A.16
	$(PYTHON) scripts/audit_retire_eligibility.py

audit-external-findings: ## External findings tracker validation — schema + cross-finding rules; offline; no-op on CI (§9.A.7)
	$(PYTHON) scripts/audit_external_findings.py

audit-external-findings-strict: ## External findings tracker strict — promotes F-4 staleness warnings to errors (operator preflight)
	$(PYTHON) scripts/audit_external_findings.py --strict-warnings

audit-shim-window: ## Historical-path compatibility shim integrity (E.1/E.2/E.3/E.4/E.5)
	$(PYTHON) scripts/audit_shim_window.py

audit-yaml-status-enum: ## Status: enum invariant — every builtin model YAML declares ✅/⚠️/🧪/👁️/⏸️/🗑️ + Caveats (club-3090 convention)
	$(PYTHON) scripts/audit_yaml_status_enum.py --strict

audit-pn59-cliff2b: ## PN59 streaming-GDN driver carries v7.72.5 Level 2 markers (Cliff 2b regression guard — club-3090 #22/#182)
	$(PYTHON) scripts/audit_pn59_cliff2b_markers.py --strict

audit-english-only: ## English-only-in-code rule (CLAUDE.md) — ratchet-down gate against baseline
	$(PYTHON) scripts/audit_english_only.py --check

audit-lifecycle-docstring-sync: ## Registry `lifecycle` vs docstring RETIRED/TOMBSTONED markers drift (catches PN108-class drift)
	$(PYTHON) scripts/audit_lifecycle_docstring_sync.py --strict

gates: test-pin-gate test-iron-rule test-family test-doc-sync test-gui-contract audit-i18n audit-phase3 audit-v2-runtime-pins audit-v2-modeldef-vs-hardware-pin audit-ai-attribution audit-links audit-repo-garbage audit-generated-links audit-wheel-contents audit-external-findings audit-shim-window audit-yaml-status-enum audit-pn59-cliff2b audit-english-only audit-override-policy-strict audit-lifecycle-docstring-sync ## Run all 21 CI gates fast-fail

# ─── Audits ────────────────────────────────────────────────────────────

audit-upstream-watchlist: ## Etap 5.1: validate UPSTREAM_WATCHLIST.yaml + emit PORT_CANDIDATE/WATCH report
	$(PYTHON) scripts/audit_upstream_watchlist.py

watchlist-check: ## PR-sweep `sweep:` section of upstream_watchlist.yaml — schema + pr uniqueness (2026-06-11 roadmap)
	$(PYTHON) tools/check_upstream_watchlist.py

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

preflight: ## Pin-bump preflight vs extracted candidate tree (env: CANDIDATE_ROOT [JSON_OUT]); see docs/PIN_BUMP_PLAYBOOK.md
	@test -n "$${CANDIDATE_ROOT}" || { \
		echo "Usage: make preflight CANDIDATE_ROOT=/tmp/candidate_pin/vllm [JSON_OUT=...]"; \
		echo "Extract first: tools/extract_candidate_tree.sh --image <ref> --rsync-to <dir> --py-only"; \
		exit 2; }
	$(PYTHON) tools/pin_preflight.py "$${CANDIDATE_ROOT}" \
		$${JSON_OUT:+--json-out "$${JSON_OUT}"}

lint-drift-markers: ## §6 self-collision lint: upstream_drift_markers vs the patch's own emitted text (env: CANDIDATE_ROOT [JSON_OUT]); allowlist: tools/lint_drift_markers_allowlist.txt
	@test -n "$${CANDIDATE_ROOT}" || { \
		echo "Usage: make lint-drift-markers CANDIDATE_ROOT=/tmp/candidate_pin/vllm [JSON_OUT=...]"; \
		echo "Extract first: tools/extract_candidate_tree.sh --image <ref> --rsync-to <dir> --py-only"; \
		exit 2; }
	$(PYTHON) tools/lint_drift_markers.py "$${CANDIDATE_ROOT}" \
		$${JSON_OUT:+--json-out "$${JSON_OUT}"}

tokenizer-fingerprint: ## Tokenizer-fingerprint gate, run in-container pre-bench on every pin bump (env: MODEL_PATH [PROMPTS_FILE] [JSON_OUT] [COMPARE]); see docs/PIN_BUMP_PLAYBOOK.md step 5b
	@test -n "$${MODEL_PATH}" || { \
		echo "Usage: make tokenizer-fingerprint MODEL_PATH=/models/<model> [PROMPTS_FILE=...] [JSON_OUT=...] [COMPARE=baseline.json]"; \
		echo "Runs in-container (needs transformers + the model tokenizer files)."; \
		exit 2; }
	$(PYTHON) tools/tokenizer_fingerprint.py --model-path "$${MODEL_PATH}" \
		$${PROMPTS_FILE:+--prompts-file "$${PROMPTS_FILE}"} \
		$${JSON_OUT:+--json-out "$${JSON_OUT}"} \
		$${COMPARE:+--compare "$${COMPARE}"}

rebuild-pin: ## Phase 4: regenerate the per-pin anchor source-of-truth on the rig, pull it back (env: SSH_HOST [CONTAINER] [IMAGE])
	@test -n "$${SSH_HOST}" || { \
		echo "Usage: make rebuild-pin SSH_HOST=<user@host> [CONTAINER=...] [IMAGE=...]"; \
		echo "Runs the proven 2-step: running-container discovery + bare-image pristine source."; \
		echo "Writes sndr/engines/vllm/pins/<pin>/anchors.json — review + commit the result."; \
		exit 2; }
	@echo "=== sync anchor-sot scripts + sndr to rig ==="
	rsync -a scripts/anchor_sot "$${SSH_HOST}:$(RIG_REPO)/scripts/"
	rsync -a sndr "$${SSH_HOST}:$(RIG_REPO)/"
	@echo "=== regenerate manifest on rig ==="
	ssh "$${SSH_HOST}" "CONTAINER=$${CONTAINER:-vllm-qwen3.6-35b-balanced-k3} IMAGE=$${IMAGE:-vllm/vllm-openai:nightly} REPO=$(RIG_REPO) bash $(RIG_REPO)/scripts/anchor_sot/rebuild_pin.sh"
	@echo "=== pull generated pins/ back to local repo ==="
	rsync -a "$${SSH_HOST}:$(RIG_REPO)/sndr/engines/vllm/pins/" sndr/engines/vllm/pins/
	@echo "rebuild-pin done — review + commit sndr/engines/vllm/pins/<pin>/anchors.json"

audit-pin: ## Phase 4: verify the committed per-pin manifest still matches a fresh rig regen (R2 drift gate; env: SSH_HOST [CONTAINER] [IMAGE])
	@test -n "$${SSH_HOST}" || { \
		echo "Usage: make audit-pin SSH_HOST=<user@host> [CONTAINER=...] [IMAGE=...]"; \
		echo "Regenerates from the live engine + diffs vs committed (ignoring timestamps)."; \
		exit 2; }
	@bash scripts/anchor_sot/audit_pin.sh "$${SSH_HOST}"

summarize-rej: ## Phase 4: human summary of pins/<pin>/drift.rej.json (counts by status + merge tri-state + retire-broken dependents); PIN=<dir> for one pin, else all
	@$(PYTHON) scripts/anchor_sot/summarize_rej.py $${PIN:+sndr/engines/vllm/pins/$${PIN}}

bump-preflight: ## Phase 4: bump gate — retire-impact + perf-landmine checklist between OLD/NEW pin manifests (OLD=<dir> NEW=<dir>); exits non-zero on a broken HIGH-severity perf dependent
	@test -n "$${OLD}" -a -n "$${NEW}" || { \
		echo "Usage: make bump-preflight OLD=sndr/engines/vllm/pins/<old> NEW=sndr/engines/vllm/pins/<new>"; \
		exit 2; }
	@$(PYTHON) scripts/anchor_sot/bump_preflight.py "$${OLD}" "$${NEW}"

audit-legacy-imports: ## Forbid vllm.sndr_core.patches / vllm._genesis active imports
	@bash scripts/check_no_legacy_imports.sh

audit-public-paths: ## Etap 6.7: forbid private LAN IPs / home paths / usernames in public docs+code
	@echo "=== audit-public-paths ==="
	@# grep is universally available; previous `rg` invocation silently
	@# returned empty (false-clean) on systems without ripgrep installed.
	@bad=$$(grep -rEn "(^|[^0-9])(10|172\.(1[6-9]|2[0-9]|3[01])|192\.168)\.[0-9]+\.[0-9]+|/home/sander|sander@|User=sander" \
	    README.md docs/ scripts/ tools/ benchmarks/ vllm/ \
	    --include='*.py' --include='*.sh' --include='*.md' \
	    --include='*.yaml' --include='*.yml' --include='*.json' \
	    --include='*.toml' --include='*.jinja' --include='*.txt' \
	    --exclude-dir='sndr_private' \
	    --exclude-dir='_archive' \
	    --exclude-dir='_internal' \
	    --exclude-dir='superpowers' \
	    --exclude-dir='baselines' \
	    --exclude-dir='__pycache__' \
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

audit-source-refs-in-docs: ## docs/*.md `vllm/sndr_core/<path>.{py,yaml}` token resolution (Phase 10.5 E-extension)
	@$(PYTHON) scripts/audit_source_refs_in_docs.py

audit-docs-refs-in-source: ## Reverse direction: Python source `docs/<name>.md` token resolution (Phase 10.5 D-extension)
	@$(PYTHON) scripts/audit_docs_refs_in_source.py

audit-phase3-relocation: ## §0.5: gemma4 relocation invariants (R1/R2/R3/R4 — shim boundary, canonical apply path, config-keys catalog)
	@$(PYTHON) scripts/audit_phase3_relocation.py

audit-anchor-fragility: ## Phase 3.1: TextPatcher anchor LOC fragility ratchet (informational; warn at >=25 lines)
	@$(PYTHON) scripts/audit_anchor_fragility.py

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

audit-private-namespace: ## P0.1 M.7: hard rule #27 — no `sndr_private` under `vllm/`; only repo-root `sndr_private/` allowed (gitignored)
	@$(PYTHON) scripts/audit_private_namespace.py

audit-config-catalog: ## CONFIG-UX.audit: preset card catalog (Stage 1 warnings; --strict for CI/release)
	@$(PYTHON) scripts/audit_config_catalog.py

audit-override-policy: ## CONFIG-UX.audit: profile OverridePolicy presence + shape (Stage 1 warnings)
	@$(PYTHON) scripts/audit_override_policy.py

audit-override-policy-strict: ## CONFIG-UX.4: profile OverridePolicy hard-enforcement (errors AND warnings fatal)
	@$(PYTHON) scripts/audit_override_policy.py --strict

audit-v1-migration: ## CONFIG-UX.4.1: V1 monolithic key migration bucket resolution (Stage 0/1 informational)
	@$(PYTHON) scripts/audit_v1_migration.py

audit-v1-sunset: ## §9.T V1 monolithic sunset countdown — stage readiness + blocker list (informational at default stage)
	@$(PYTHON) scripts/audit_v1_sunset.py

generate-config-catalog: ## CONFIG-UX.5.1: build derived catalog JSON from V2 YAML tree + baselines (no committed artifact yet)
	@$(PYTHON) scripts/generate_config_catalog.py

audit-config-catalog-fresh: ## CONFIG-UX.5.1: generated catalog determinism + redaction audit (informational at .5.1)
	@$(PYTHON) scripts/audit_generated_config_catalog.py

config-catalog: ## CONFIG-UX.5.2: discoverable alias for `sndr config-catalog build` (derived catalog UX sugar)
	@$(PYTHON) -m sndr.cli.legacy config-catalog build

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
	@$(PYTHON) -m sndr.cli.legacy community validate

audit-no-new-v1: ## Phase 9 freeze gate: top-level builtin/*.yaml matches frozen baseline
	@$(PYTHON) scripts/audit_no_new_v1.py

audit-patches-prove: ## §6.8 R1 mitigation: dead-patch detector (lists patches without proof artefacts)
	@$(PYTHON) -m sndr.cli.legacy patches prove --dead-detect

audit-patches-prove-all: ## §6.8 release gate: run static checks on every PATCH_REGISTRY entry
	@$(PYTHON) -m sndr.cli.legacy patches prove --all --no-write

audit-proof-status: ## §6.8 read-side: bucket summary of every patch's proof-artefact state (informational)
	@$(PYTHON) -m sndr.cli.legacy patches proof-status

audit-release-check: ## §6.8 release-gate consumer — every patch must have a static proof (gating in make evidence --release)
	@$(PYTHON) -m sndr.cli.legacy patches release-check --mode require-static

audit-release-check-bench-attached: ## §6.8 ratchet 1: every patch must have at least one bench attachment (bridge to require-baseline)
	# Bridge between require-static (current public gate) and the strict
	# require-baseline below. Run this when promoting the default-on
	# subset of a production preset; not part of `make evidence --release`.
	# See docs/RELEASE_POLICY.md for the policy lifecycle.
	@$(PYTHON) -m sndr.cli.legacy patches release-check --mode require-bench

audit-release-check-baseline-optional: ## §6.8 ratchet 2: every patch must carry a bench_with_baseline proof (strict)
	# Informational by design: 0/169 entries have bench_with_baseline today.
	# Wiring this into `make evidence --release` as a hard gate would block
	# every release until operators re-bench all 169 entries on their rig.
	# Operators preparing a hardened deploy run this target directly after
	# the bench-attached ratchet above clears the default-on subset.
	# See docs/RELEASE_POLICY.md for the cutover procedure.
	@$(PYTHON) -m sndr.cli.legacy patches release-check --mode require-baseline

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

audit-dispatcher-migration-readiness: ## v11.3.0 P3.4: dispatcher iter_patch_specs() readiness — 218/241 spec-ready, 0 real gaps
	@$(PYTHON) scripts/audit_dispatcher_migration_readiness.py --strict

audit-legacy-vs-spec-driven-apply-matrix: ## v11.3.0 P3.4: legacy vs spec-driven apply-matrix divergence report (informational — v12.0.0 unifies)
	@$(PYTHON) scripts/audit_legacy_vs_spec_driven_apply_matrix.py

audit-stale-vllm-version-ranges: ## v11.3.0 CLAUDE.md Class 5: stale vllm_version_range upper bounds (CRITICAL count must be 0)
	@$(PYTHON) scripts/audit_stale_vllm_version_ranges.py --strict

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

evidence-release: ## Phase 0 supplement: aggregate + release-only gates (dirty-state, SBOM); CONFIG-UX.4.2 sets Stage 2 strict for rollout audits
	@SNDR_V1_ROLLOUT_STAGE=2 $(PYTHON) scripts/make_evidence.py --release

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
	$(PYTHON) -m sndr.cli.legacy doctor

gui-build: ## Build the web UI and bundle it into the package for the daemon to serve
	cd gui/web && npm ci && npm run build
	rm -rf sndr/product_api/legacy/web_static
	cp -R gui/web/dist sndr/product_api/legacy/web_static
	@echo "✓ GUI built and copied to sndr/product_api/legacy/web_static (the path the daemon serves from)"
	@echo "  Run: $(PYTHON) -m sndr.cli gui-api  → serves UI + API on one port"

gui-lint: ## Typecheck + lint the GUI (tsc strict + eslint with a11y enforced as errors)
	cd gui/web && npx tsc -b && npx eslint .

gui-build-carbon: ## Build the new Carbon Control Center and bundle it for the modular API server (:8800)
	cd gui/web && npm ci --legacy-peer-deps && npm run build:carbon
	rm -rf sndr/product_api/web_static_carbon
	cp -R gui/web/dist sndr/product_api/web_static_carbon
	mv sndr/product_api/web_static_carbon/index.carbon.html sndr/product_api/web_static_carbon/index.html
	@echo "✓ Carbon GUI built and copied to sndr/product_api/web_static_carbon"
	@echo "  Run: uvicorn sndr.product_api.server:create_app --factory --port 8800  → serves Carbon UI + Envelope API"

# ─── Integration (gated on GENESIS_INTEGRATION_ENDPOINT) ─────────────
#
# All targets default to localhost (port 8101 for 27B, 8000 for 35B).
# For a remote rig, set `HOST=http://<host>:<port>`. The hardcoded LAN IP
# was removed — the operator-specific endpoint lives only in
# `~/.sndr/host.yaml` (use `sndr host init`).

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
