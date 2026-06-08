#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gateway end-to-end smoke test — empirical proof that the multi-profile
architecture documented in docs/_internal/DEPLOYMENT_PROFILE_ROUTING_PLAN_2026-05-20.md
actually partitions workload classes between the chat-K=3 and structured-K=4
profile siblings without overlap.

Run:
    python3 tools/gateway_smoke_test.py            # gemma4-31b dense
    python3 tools/gateway_smoke_test.py --model 26b  # gemma4-26b MoE A4B

The test does NOT launch any vLLM upstream — it validates the routing
LOGIC directly via vllm.sndr_core.integrations.spec_decode.request_router
against the actual artifact JSONs in this repo. The end-to-end live
gateway flow (FastAPI proxy at port 8100 forwarding to two real vLLM
upstreams) requires two TP=2 launchers running simultaneously which
doesn't fit on a 2× A5000 rig (each launcher consumes both GPUs). This
script verifies that, GIVEN a request with a specific workload signal,
the request_router correctly selects the right profile per the
artifact's allowed/denied_workloads.

Exit code:
  0 = all routing assertions pass — multi-profile architecture is
      logically sound; an operator running both launchers behind the
      gateway will see the documented routing behavior.
  1 = at least one routing assertion failed — chat-K=3 + structured-K=4
      artifacts do not partition the workload-class space cleanly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sndr.engines.vllm.patches.spec_decode.functional_artifact import (
    find_matching,
)
from sndr.engines.vllm.patches.spec_decode.request_router import (
    select_profile,
)


MODEL_CONFIGS = {
    "31b": {
        "model_id": "gemma-4-31B-it-AWQ-4bit",
        "structured": ("gemma4-31b-tq-mtp-structured-k4", "71c874d7ffedae04"),
        "chat":       ("gemma4-31b-tq-mtp-chat-k3",      "aba0bb6b158f7632"),
        "fallback":   "gemma4-31b-tq-default",
    },
    "26b": {
        "model_id": "gemma-4-26B-A4B-it-AWQ",
        "structured": ("gemma4-26b-mtp-k4",          None),  # no artifact for K=4 on 26B
        "chat":       ("gemma4-26b-mtp-chat-k3",     "717c7edc75754aea"),
        "fallback":   "gemma4-26b-no-mtp",
    },
}


def _load(model_id: str, profile_name: str, config_hash: str | None):
    """Find artifact for (model, profile, hash). Returns None if hash is None."""
    if config_hash is None:
        return None
    art = find_matching(model_id, profile_name, config_hash)
    return art


def _check_routes_to(
    *, label: str, signal_request: dict, artifact, expected_accepted: bool,
    expected_profile: str, expected_workload_class: str | None,
    fallback_profile: str,
) -> tuple[bool, str]:
    """Run select_profile and assert the decision."""
    decision = select_profile(
        request=signal_request,
        artifact=artifact,
        fallback_profile=fallback_profile,
    )
    ok = (
        decision.accepted == expected_accepted
        and decision.profile == expected_profile
        and decision.workload_class == expected_workload_class
    )
    line = (
        f"  {'✓' if ok else '✗'} {label}: "
        f"accepted={decision.accepted} profile={decision.profile} "
        f"workload={decision.workload_class!r} reason={decision.reason[:80]}"
    )
    return ok, line


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="31b", choices=list(MODEL_CONFIGS.keys()))
    args = ap.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    model_id = cfg["model_id"]

    print(f"=== Gateway smoke test — {model_id} ===")
    print()

    structured_art = _load(model_id, *cfg["structured"])
    chat_art       = _load(model_id, *cfg["chat"])

    if cfg["structured"][1] is not None and structured_art is None:
        print(f"✗ structured artifact missing for {cfg['structured']!r}")
        return 1
    if chat_art is None:
        print(f"✗ chat artifact missing for {cfg['chat']!r}")
        return 1

    if structured_art is not None:
        print(f"  structured artifact: profile={structured_art.profile} "
              f"decision={structured_art.decision}")
        print(f"    allowed: {structured_art.allowed_workloads}")
        print(f"    denied : {structured_art.denied_workloads}")
    else:
        print(f"  structured artifact: (none registered for K=4 on this model)")
    print(f"  chat       artifact: profile={chat_art.profile} "
          f"decision={chat_art.decision}")
    print(f"    allowed: {chat_art.allowed_workloads}")
    print(f"    denied : {chat_art.denied_workloads}")
    print()

    # Signal samples — one per workload class + one unsignaled control.
    cases = [
        ("free_chat   (no signal)",   {},                                                       None),
        ("free_chat   (explicit tag)", {"extra_body": {"workload_class": "free_chat"}},        "free_chat"),
        ("code_gen    (explicit tag)", {"extra_body": {"workload_class": "code_gen"}},         "code_gen"),
        ("summarization (explicit tag)", {"extra_body": {"workload_class": "summarization"}},  "summarization"),
        ("structured_count (explicit tag)", {"extra_body": {"workload_class": "structured_count"}}, "structured_count"),
        ("tool_json   (response_format json_object)", {"response_format": {"type": "json_object"}}, "tool_json"),
        ("tool_json   (tool_choice required)", {"tool_choice": "required"}, "tool_json"),
    ]

    print("--- Route via STRUCTURED-role artifact (denies chat workloads) ---")
    all_ok = True
    for label, req, signaled_class in cases:
        if structured_art is None:
            continue
        expected_accepted = (
            signaled_class in (structured_art.allowed_workloads or [])
        )
        expected_profile = (
            structured_art.profile if expected_accepted
            else cfg["fallback"]
        )
        # workload_class is set in the decision only when the router saw a signal,
        # regardless of whether the artifact accepts it. No signal -> None.
        expected_wc = signaled_class if signaled_class else None
        ok, line = _check_routes_to(
            label=label,
            signal_request=req,
            artifact=structured_art,
            expected_accepted=expected_accepted,
            expected_profile=expected_profile,
            expected_workload_class=expected_wc,
            fallback_profile=cfg["fallback"],
        )
        print(line)
        all_ok = all_ok and ok

    print()
    print("--- Route via CHAT-role artifact (denies structured workloads) ---")
    for label, req, signaled_class in cases:
        expected_accepted = (
            signaled_class in (chat_art.allowed_workloads or [])
        )
        expected_profile = (
            chat_art.profile if expected_accepted else cfg["fallback"]
        )
        expected_wc = signaled_class if signaled_class else None
        ok, line = _check_routes_to(
            label=label,
            signal_request=req,
            artifact=chat_art,
            expected_accepted=expected_accepted,
            expected_profile=expected_profile,
            expected_workload_class=expected_wc,
            fallback_profile=cfg["fallback"],
        )
        print(line)
        all_ok = all_ok and ok

    # Verify the partition is clean: every class accepted by one artifact
    # is denied by the other (no overlap), and every class denied by both
    # would go to fallback (which is fine — but check for completeness).
    if structured_art is not None:
        print()
        print("--- Workload-class partition invariant ---")
        s_allowed = set(structured_art.allowed_workloads or [])
        c_allowed = set(chat_art.allowed_workloads or [])
        overlap = s_allowed & c_allowed
        if overlap:
            print(f"  ✗ OVERLAP between structured and chat allowed sets: {overlap}")
            all_ok = False
        else:
            print(f"  ✓ no overlap: structured ∩ chat = ∅")
            print(f"    structured allowed: {sorted(s_allowed)}")
            print(f"    chat       allowed: {sorted(c_allowed)}")
            uncovered = (
                set(structured_art.workload_classes or [])
                - s_allowed - c_allowed
            )
            if uncovered:
                print(f"  ⚠ workload classes covered by neither profile (fall back): {sorted(uncovered)}")
            else:
                print(f"  ✓ every workload class is in exactly one artifact's allowed set")

    print()
    print("=" * 64)
    if all_ok:
        print("✓ all routing assertions pass — multi-profile architecture works")
        return 0
    else:
        print("✗ at least one routing assertion failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
