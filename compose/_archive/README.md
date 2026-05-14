# Archived compose files (v7.0-dev era)

These three files were the PROD-mirror INTEGRATION composes for VM 100
(`192.168.1.10`) during v7.0-dev. They reference the pre-v11 namespace
`vllm/_genesis/` which was removed in v11.0.0 (commit 776aa32b).

| File | Purpose at time |
|---|---|
| `docker-compose.integration.yml` | TQ k8v4 integration variant |
| `docker-compose.integration-awq.yml` | AWQ-quant integration variant |
| `docker-compose.integration-fp16kv.yml` | fp16 KV integration variant |

They are preserved here as **historical artefacts** — they document
the env-flag matrix that was active during the original integration
ladders, and they are referenced from a handful of bench reports.

**Do not run these.** They will fail with `ModuleNotFoundError:
vllm._genesis` on any v11+ install. The canonical replacement is
`sndr launch <preset>` (V2 layered model configs under
`vllm/sndr_core/model_configs/builtin/presets/`).

Audit `CURRENT_PROJECT_RECHECK_CLEANUP_ERRORS_2026-05-14_eaa44975_RU.md`
P1-1 documented why the migration happened.
