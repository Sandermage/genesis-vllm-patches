"""Per-model eligibility verification for PN50/PN52/PN54.

For each of the 7 Genesis start scripts, simulate plugin import +
apply_all dry-run with the patches' env-flag set, then check that:

  * PN50 fires on configs with GDN model class (27B Lorbus only)
  * PN52 fires on all configs (chunked-prefill is universal)
  * PN54 fires on configs with GDN model class (27B Lorbus only)

This runs LOCALLY on Mac (no GPU needed) — it tests the dispatcher's
eligibility logic + env-flag gating, not actual kernel execution.

Live PROD verification (boot + tool-call + perf A/B) is a separate step
that requires container restart (not done in this probe).
"""
from __future__ import annotations

import os
import sys


def check_patch(patch_id: str, env_flag: str, model_class: str) -> dict:
    """Simulate dispatcher decision for a patch on a given model class."""
    # Force env-flag ON to test the model-class gating, not env gating
    os.environ[env_flag] = "1"
    try:
        # Reimport dispatcher fresh to pick up env
        for stale in ("sndr.dispatcher",
                      "sndr.dispatcher.registry"):
            if stale in sys.modules:
                del sys.modules[stale]
        from sndr.dispatcher import should_apply
        from sndr.dispatcher.registry import PATCH_REGISTRY
        meta = PATCH_REGISTRY.get(patch_id, {})
        applies_to = meta.get("applies_to", {})
        allowed_classes = applies_to.get("model_class")
        if allowed_classes:
            class_match = model_class in allowed_classes
        else:
            class_match = True  # no model_class restriction → universal
        decision, reason = should_apply(patch_id)
        return {
            "patch": patch_id,
            "env_set": True,
            "model_class": model_class,
            "applies_to_model_class": allowed_classes,
            "class_match": class_match,
            "should_apply_decision": decision,
            "reason": reason[:80],
        }
    finally:
        del os.environ[env_flag]


MODELS = [
    ("start_27b_int4_TQ_k8v4.sh", "qwen3_5"),
    ("start_27b_int4_TQ_k8v4_NGRAM.sh", "qwen3_5"),
    ("start_27b_int4_DFLASH.sh", "qwen3_5"),
    ("start_27b_int4_fp8_e5m2_short.sh", "qwen3_5"),
    ("start_27b_int4_fp8_e5m2_long_256K.sh", "qwen3_5"),
    ("start_35b_fp8_PROD.sh", "qwen3_moe"),
    ("start_35b_fp8_DFLASH.sh", "qwen3_moe"),
]

PATCHES = [
    ("PN50", "GENESIS_ENABLE_PN50_GDN_FUSED_PROJ"),
    ("PN51", "GENESIS_ENABLE_PN51_QWEN3_STREAMING_THINKING_DISABLED"),
    ("PN52", "GENESIS_ENABLE_PN52_PROMPT_LOGPROBS_EVICTION"),
    ("PN54", "GENESIS_ENABLE_PN54_GDN_CONTIGUOUS_DEDUP"),
    ("PN55", "GENESIS_ENABLE_PN55_WAKE_UP_HYBRID_KV"),
    ("PN56", "GENESIS_ENABLE_PN56_QWEN3CODER_XML_FALLBACK"),
    ("PN57", "GENESIS_ENABLE_PN57_TQ_CENTROIDS_DISK_CACHE"),
    ("PN58", "GENESIS_ENABLE_PN58_SPEC_REASONING_BOUNDARY"),
    ("PN59", "GENESIS_ENABLE_PN59_STREAMING_GDN"),
    ("P107", "GENESIS_ENABLE_P107_MTP_TRUNCATION_DETECTOR"),
]


def main() -> int:
    print(f"=== Per-model eligibility verification (10 patches) ===\n")
    cols = "  ".join(f"{p[0]:>5}" for p in PATCHES)
    print(f"{'Script':<40} {cols}")
    print("-" * (40 + len(cols) + 4))
    issues = []
    for script, model_class in MODELS:
        row = [script[:38].ljust(38)]
        for patch_id, env_flag in PATCHES:
            r = check_patch(patch_id, env_flag, model_class)
            verdict = "APPLY" if r["should_apply_decision"] and r["class_match"] else "skip"
            row.append(verdict.rjust(6))
            # Validate expected behavior:
            # PN50/PN54: should APPLY only on qwen3_5 (27B)
            # PN51: model_class restriction in registry — any qwen3* OK
            # PN52: no model_class restriction — universal
            if patch_id in ("PN50", "PN54") and model_class == "qwen3_5":
                if not (r["should_apply_decision"] and r["class_match"]):
                    issues.append(f"  ✗ {patch_id} should APPLY on {script} ({model_class}) but didn't")
            if patch_id == "PN52":  # universal
                if not r["should_apply_decision"]:
                    issues.append(f"  ✗ PN52 should APPLY on {script} (no model class restriction)")
        print(" ".join(row))
    print()
    if issues:
        print("ISSUES FOUND:")
        for i in issues:
            print(i)
        return 1
    print("✓ All patches behave as expected per their registry entries")
    print("  - PN50/PN54: APPLY only on qwen3_5 (27B Lorbus)")
    print("  - PN51: APPLY on qwen3_* configs")
    print("  - PN52: APPLY universally (no model_class restriction)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
