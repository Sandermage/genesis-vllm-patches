#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tokenizer-fingerprint gate — sha256 over token-id sequences of a
canonical prompt set (pr-sweep-50 roadmap chunk 2 Theme E, upstream
vllm#45109 lesson).

Why this exists
───────────────
Upstream #45109 is a test-only PR: AWQ expected OUTPUTS changed because
the Transformers v5 tokenizer segments the same prompts differently.
Nothing to vendor — but the failure class is ours too: AWQ/AutoRound
checkpoints (Lorbus 27B, Gemma-4 AWQ) are exactly the affected class,
and a silent tokenizer-behavior change across a pin bump produces
output diffs that get misattributed to Genesis patches (hours of
misdirected bisection, iron-rule-#11 class).

The gate: fingerprint the model tokenizer BEFORE every post-bump bench
(PIN_BUMP_PLAYBOOK step 5b). Identical fingerprint -> tokenizer is not
the variable; any output diff is patch-attributable. Changed
fingerprint -> STOP, re-baseline expected outputs / check
tokenizer_class against the pin's
``_MODEL_TYPES_WITH_INCORRECT_TOKENIZER_CLASS`` hook first.

Usage (in-container on the rig; transformers is lazy-imported):

    python3 tools/tokenizer_fingerprint.py --model-path /models/<model> \
        [--prompts-file F] [--json-out OUT.json] [--compare BASELINE.json] \
        [--trust-remote-code]

    # make target:
    make tokenizer-fingerprint MODEL_PATH=/models/<model> [JSON_OUT=...]

Prompts file: ``.json`` -> list of strings OR list of {"id", "text"}
objects; anything else -> plain text, one prompt per line, literal
``\\n`` sequences unescaped, blank lines skipped. Without
``--prompts-file`` the embedded canonical set is used (versioned id —
fingerprints are only comparable within the same prompt set).

Exit codes:
    0 — fingerprint computed (and matches baseline when --compare)
    1 — --compare mismatch (tokenizer drift — re-baseline before bench)
    2 — invocation error (bad paths, tokenizer load failure,
        prompt-set mismatch in --compare)

Text output is timestamp-free and line-oriented so two runs diff
cleanly; --json-out carries the machine-readable report (the --compare
baseline format).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

CANONICAL_PROMPT_SET_ID = "genesis-canonical-v1"

# Canonical prompt classes chosen for tokenizer-drift sensitivity:
# merges/BPE boundaries move most around code, structured payloads,
# chat markup, whitespace runs and multilingual segmentation. The
# non-English STRINGS below are tokenizer-coverage DATA, not prose —
# multilingual vocab segmentation is exactly what shifts between
# tokenizer majors (the English-only rule governs comments, messages
# and identifiers, which stay English).
CANONICAL_PROMPTS: list[tuple[str, str]] = [
    ("ascii_basic", "The quick brown fox jumps over the lazy dog."),
    ("contractions", "It's the operator's job; we'd rather fail fast than debug prod."),
    ("numbers_units", "Throughput hit 211.5 TPS at 280000 context tokens (CV 3.02%)."),
    (
        "python_code",
        'def fingerprint(ids: list[int]) -> str:\n'
        '    return ",".join(map(str, ids))\n',
    ),
    (
        "json_tool_payload",
        '{"tool": "search", "arguments": {"query": "vllm pin bump", "k": 3}}',
    ),
    (
        "chat_markup",
        "<|im_start|>user\nCall the weather tool for the home rig<|im_end|>",
    ),
    (
        "xml_toolcall",
        "<tool_call><function=get_weather><parameter=city>Odessa"
        "</parameter></function></tool_call>",
    ),
    (
        "whitespace_mix",
        "indent\ttab  double-space\n    four-space\r\nwindows-newline trailing  ",
    ),
    (
        "unicode_punct",
        "Cafe naive resume — em-dash, ellipsis…, quotes “curly”, ‘single’.",
    ),
    ("multilingual_cyrillic", "Швидка перевірка токенізатора після оновлення pin."),
    ("multilingual_cjk", "模型升级后立即验证分词器的稳定性。"),
    ("emoji_symbols", "Rocket 🚀 + gauge 📈 = bench ✅ (≤ 1 min, ±0.0%)"),
    ("repetition_merge_stress", "ab" * 64),
    ("repeated_word_merge_stress", " ".join(["tokenization"] * 32)),
]


def prompt_fingerprint(token_ids: Sequence[int]) -> str:
    """sha256 hex over a token-id sequence.

    Length-prefixed, comma-joined decimal serialization: unambiguous
    (no concat collisions like [1,23] vs [12,3]) and stable across
    Python/json int formatting.
    """
    ids = [int(t) for t in token_ids]
    payload = f"{len(ids)}:" + ",".join(str(t) for t in ids)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_report(
    tokenizer: Any,
    prompts: Sequence[tuple[str, str]],
    prompt_set_id: str,
    model_path: str,
) -> dict:
    """Fingerprint every prompt through ``tokenizer.encode``.

    ``tokenizer`` only needs ``encode(text) -> sequence of ints`` — the
    deliberate seam that lets unit tests run on a tiny local stub while
    the rig leg uses the real AutoTokenizer. Plain ``encode`` (no
    add_special_tokens override) is intentional: the fingerprint must
    capture the FULL default encode behavior, special-token defaults
    included — those defaults are exactly what shifted in the
    Transformers v5 migration (#45109).
    """
    per_prompt = []
    for pid, text in prompts:
        ids = list(tokenizer.encode(text))
        per_prompt.append(
            {
                "id": pid,
                "sha256": prompt_fingerprint(ids),
                "num_tokens": len(ids),
            }
        )
    aggregate = hashlib.sha256(
        "\n".join(f"{e['id']}:{e['sha256']}" for e in per_prompt).encode("utf-8")
    ).hexdigest()
    return {
        "tool": "tokenizer_fingerprint",
        "schema_version": 1,
        "model_path": model_path,
        "prompt_set": prompt_set_id,
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "per_prompt": per_prompt,
        "aggregate_sha256": aggregate,
    }


def load_prompts_file(path: str) -> list[tuple[str, str]]:
    """Load prompts from a file (see module docstring for the format)."""
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    prompts: list[tuple[str, str]] = []
    if p.suffix == ".json":
        data = json.loads(raw)
        if not isinstance(data, list) or not data:
            raise ValueError(f"{path}: expected a non-empty JSON list")
        for i, item in enumerate(data, start=1):
            if isinstance(item, str):
                prompts.append((f"prompt_{i:03d}", item))
            elif isinstance(item, dict) and "id" in item and "text" in item:
                prompts.append((str(item["id"]), str(item["text"])))
            else:
                raise ValueError(
                    f"{path}: entry {i} must be a string or an "
                    f"object with 'id' and 'text'"
                )
    else:
        for i, line in enumerate(
            (ln for ln in raw.split("\n") if ln.strip()), start=1
        ):
            prompts.append((f"prompt_{i:03d}", line.replace("\\n", "\n")))
        if not prompts:
            raise ValueError(f"{path}: no prompts found")
    ids = [pid for pid, _ in prompts]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate prompt ids")
    return prompts


def load_tokenizer(model_path: str, trust_remote_code: bool = False) -> Any:
    """Load the model's tokenizer (rig leg — lazy transformers import)."""
    from transformers import AutoTokenizer  # heavy import, container-only

    return AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )


def render_text(report: dict) -> str:
    """Diffable line-oriented rendering (no timestamps by design)."""
    lines = [
        f"tokenizer_class={report['tokenizer_class']}",
        f"vocab_size={report['vocab_size']}",
        f"prompt_set={report['prompt_set']}",
        f"model_path={report['model_path']}",
    ]
    width = max(len(e["id"]) for e in report["per_prompt"])
    for e in report["per_prompt"]:
        lines.append(
            f"{e['id']:<{width}}  sha256={e['sha256']}  n={e['num_tokens']}"
        )
    lines.append(f"AGGREGATE sha256={report['aggregate_sha256']}")
    return "\n".join(lines)


def compare_reports(baseline: dict, current: dict) -> tuple[int, str]:
    """Compare a current report against a stored baseline.

    Returns (exit_code, human_text). Prompt-set mismatch is an
    invocation error (2) — fingerprints from different prompt sets are
    not comparable. Tokenizer drift is 1 with the drifted prompt
    classes named, so the operator sees WHICH segmentation class moved.
    """
    if baseline.get("prompt_set") != current.get("prompt_set"):
        return 2, (
            f"prompt-set mismatch: baseline={baseline.get('prompt_set')!r} "
            f"current={current.get('prompt_set')!r} — fingerprints are only "
            f"comparable within the same prompt set"
        )
    lines = []
    if baseline.get("tokenizer_class") != current.get("tokenizer_class"):
        lines.append(
            f"tokenizer_class drift: {baseline.get('tokenizer_class')} -> "
            f"{current.get('tokenizer_class')} (check the pin's "
            f"_MODEL_TYPES_WITH_INCORRECT_TOKENIZER_CLASS hook)"
        )
    base_by_id = {e["id"]: e for e in baseline.get("per_prompt", [])}
    cur_by_id = {e["id"]: e for e in current.get("per_prompt", [])}
    if set(base_by_id) != set(cur_by_id):
        return 2, (
            "per-prompt id sets differ between baseline and current — "
            "not the same prompt corpus"
        )
    drifted = [
        pid
        for pid in base_by_id
        if base_by_id[pid]["sha256"] != cur_by_id[pid]["sha256"]
    ]
    if baseline["aggregate_sha256"] == current["aggregate_sha256"] and not drifted:
        return 0, (
            f"MATCH aggregate sha256={current['aggregate_sha256']} — "
            f"tokenizer is not the variable; output diffs (if any) are "
            f"patch-attributable"
        )
    lines.append(
        f"MISMATCH aggregate {baseline['aggregate_sha256']} -> "
        f"{current['aggregate_sha256']}"
    )
    for pid in drifted:
        b, c = base_by_id[pid], cur_by_id[pid]
        lines.append(
            f"  drift {pid}: sha256 {b['sha256'][:16]}... -> "
            f"{c['sha256'][:16]}... n {b['num_tokens']} -> {c['num_tokens']}"
        )
    lines.append(
        "tokenizer drift detected — re-baseline expected outputs BEFORE "
        "benching; do NOT attribute output diffs to patches"
    )
    return 1, "\n".join(lines)


def main(
    argv: list[str] | None = None,
    tokenizer_loader: Callable[..., Any] = load_tokenizer,
) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Tokenizer-fingerprint gate: sha256 of token-id sequences for "
            "a canonical prompt set; diffable pre-bench on every pin bump."
        )
    )
    ap.add_argument("--model-path", required=True,
                    help="model directory (tokenizer files) or HF repo id")
    ap.add_argument("--prompts-file", default=None,
                    help="optional prompt corpus (.json list or text lines); "
                         "default: embedded canonical set")
    ap.add_argument("--json-out", default=None,
                    help="write the machine-readable report (baseline format)")
    ap.add_argument("--compare", default=None,
                    help="baseline JSON from a previous --json-out run; "
                         "exit 1 on fingerprint drift")
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="pass trust_remote_code=True to the tokenizer loader")
    args = ap.parse_args(argv)

    if args.prompts_file is not None:
        try:
            prompts = load_prompts_file(args.prompts_file)
        except (OSError, ValueError, json.JSONDecodeError) as e:
            print(f"error: cannot load prompts file: {e}", file=sys.stderr)
            return 2
        prompt_set_id = f"file:{Path(args.prompts_file).name}"
    else:
        prompts = CANONICAL_PROMPTS
        prompt_set_id = CANONICAL_PROMPT_SET_ID

    try:
        tokenizer = tokenizer_loader(
            args.model_path, trust_remote_code=args.trust_remote_code
        )
    except Exception as e:
        print(f"error: cannot load tokenizer for {args.model_path!r}: {e}",
              file=sys.stderr)
        return 2

    report = build_report(
        tokenizer, prompts, prompt_set_id=prompt_set_id,
        model_path=args.model_path,
    )
    print(render_text(report))

    if args.json_out:
        try:
            Path(args.json_out).write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError as e:
            print(f"error: cannot write {args.json_out}: {e}", file=sys.stderr)
            return 2

    if args.compare:
        try:
            baseline = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"error: cannot read baseline {args.compare}: {e}",
                  file=sys.stderr)
            return 2
        code, text = compare_reports(baseline, report)
        print(text)
        return code

    return 0


if __name__ == "__main__":
    sys.exit(main())
