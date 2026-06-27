# SPDX-License-Identifier: Apache-2.0
"""KL-divergence tail quality probe — the verification hole the needle ladder misses.

Why this exists
---------------
The Genesis verify-stress gate certifies needle-in-a-haystack recall (NIAH) and
HTTP-level stability, and our soak certifies multi-turn stability. None of them
certify *output-distribution* quality of a sub-8-bit KV cache. club-3090's
CLIFFS.md surfaced the gap directly: on the same prompts at 32K context, the
needle@32K recall is **100% across every KV cache mode (bf16 -> TQ k8v4 / turbo)**
— yet the **99.9-percentile KL divergence of the output token distribution falls
from 100% (parity with bf16) to ~54%**. The *median* token is fine; the *tail* is
where sub-8-bit KV silently breaks the low-probability-but-structurally-critical
tokens: a JSON brace, a closing quote, a tool-call argument boundary. A passing
needle ladder gives an operator false confidence that TQ k8v4 is tail-safe for
code / JSON / agentic workloads. It is not — and recall cannot see it, because
recalling 'crimson otter 42' never depends on the long tail of the distribution.

This probe closes that hole. Given two aligned per-token output distributions
over the *same* prompts —

  * a **reference** run (bf16 KV, the quality ceiling), and
  * a **candidate** run (e.g. TQ k8v4 / turbo KV) —

it computes the per-token KL divergence ``KL(P_ref || Q_cand)`` and reports the
**tail** of that distribution (99.9 / 99 / 95 percentile) plus mean/median. A
regression fires when the tail KL exceeds a threshold: the tail is the signal,
not the mean, exactly because the failure mode lives in the low-probability
tokens that the mean washes out.

What is offline-verifiable here (and what is NOT)
-------------------------------------------------
The MATH — per-token KL, the percentile tail, the threshold verdict — is pure and
unit-tested on synthetic distributions with analytically known KL (see
``tests/unit/quality_gate/test_kl_tail_probe.py``). No GPU, no model, no rig.

The MEASUREMENT — actually capturing the two per-token distributions from a live
engine at two KV dtypes over the same prompts — needs the rig and the served
model. That is a **rig-follow-up**, not done here. This module provides the
offline ``--from-captures ref.jsonl cand.jsonl`` path that consumes captures
produced ON the rig, and documents the capture contract below so the rig step is
unambiguous. We deliberately do NOT fabricate measured KL numbers; the 100%->54%
figure above is club-3090's reported observation (the motivation), not a Genesis
measurement, and is cited as such.

Capture contract (the rig-follow-up step)
-----------------------------------------
Each capture file is JSON-lines. One line per *generated token position*, an
object with at least::

    {"probs": [p0, p1, ...]}                  # full or top-k probability row, or
    {"logprobs": {"<tok>": lp, ...}}          # OpenAI-style logprob dict, or
    {"logits": [l0, l1, ...]}                 # raw logits (softmaxed here)

``ref.jsonl`` and ``cand.jsonl`` must be POSITIONALLY ALIGNED: line *i* of each is
the model's next-token distribution at the same decode position over the same
prompt+forced-prefix. The simplest rig capture is greedy/temperature-0 decode of
the *reference* run, then teacher-force the *candidate* run on the reference's
emitted token ids so both distributions are over an identical token sequence
(divergence is then purely the KV-dtype effect, not sampling drift). To capture
these from vLLM, request ``logprobs`` with a wide ``top_logprobs`` (or tap the
engine's pre-sample logits) at each position; write one row per position to each
file. ``compute_kl_tail_from_captures`` aligns by position and truncates to the
shorter file (logging the truncation) so an early-stop candidate is handled.

Provenance: motivated by club-3090's CLIFFS.md "needle != quality" observation
(github.com/noonghunna/club-3090, MIT). The Genesis contribution is turning that
observation into a runnable, unit-tested, thresholded gate wired to the same
quality-gate harness. See docs/QUALITY_GATE.md "KL-divergence tail probe".
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

# Numerical floor so a zero in the candidate distribution does not blow KL to
# +inf. KL(P||Q) has a log(P/Q) term; Q=0 where P>0 is "infinite surprise". We
# clamp Q (and P, symmetrically, for the renormalised top-k case) to EPS. This is
# the standard additive-smoothing guard; EPS is far below any meaningful prob.
_EPS = 1e-12


def kl_divergence(p: list[float], q: list[float]) -> float:
    """Discrete KL divergence ``D_KL(P || Q) = sum_i P(i) * log( P(i) / Q(i) )``.

    In nats (natural log). P is the reference (bf16) distribution, Q the
    candidate (e.g. TQ k8v4). Both are renormalised to sum to 1 first (so a
    top-k row that does not sum to exactly 1 is handled), then Q is floored at
    ``_EPS`` to keep the log finite where P has mass and Q is ~0 — that floored
    region is precisely the tail this probe is built to catch.

    Asymmetric on purpose: ``D_KL(P||Q)`` weights by the *reference* mass, so a
    candidate that drops probability on a token the reference considered likely
    is penalised — which is the JSON-brace / tool-arg failure we care about.

    Raises ValueError on a length mismatch or a non-normalisable (all-zero or
    negative-mass) input.
    """
    if len(p) != len(q):
        raise ValueError(f"distribution length mismatch: {len(p)} vs {len(q)}")
    if not p:
        raise ValueError("empty distribution")
    sp = math.fsum(p)
    sq = math.fsum(q)
    if sp <= 0 or sq <= 0:
        raise ValueError("distribution has non-positive total mass")
    kl = 0.0
    for pi_raw, qi_raw in zip(p, q, strict=True):
        if pi_raw < 0 or qi_raw < 0:
            raise ValueError("negative probability encountered")
        pi = pi_raw / sp
        if pi <= 0.0:
            continue  # 0 * log(0/q) == 0 by convention; no contribution.
        qi = max(qi_raw / sq, _EPS)
        kl += pi * math.log(pi / qi)
    # KL is non-negative; tiny negative values are floating-point noise.
    return max(kl, 0.0)


def _softmax(logits: list[float]) -> list[float]:
    """Numerically stable softmax (subtract max) — for ``logits`` capture rows."""
    if not logits:
        raise ValueError("empty logits row")
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = math.fsum(exps)
    return [e / s for e in exps]


def row_to_probs(row: dict[str, Any]) -> list[float]:
    """Normalise one capture row to a probability list.

    Accepts the three documented capture shapes (see module docstring):
      * ``{"probs": [...]}``    — taken as-is (renormalised downstream).
      * ``{"logits": [...]}``   — softmaxed here.
      * ``{"logprobs": {...}}`` — OpenAI-style {token: logprob}; exp() of the
        values (order is the dict's; alignment between ref/cand top-k sets is the
        caller's responsibility — for a teacher-forced capture the forced token
        is row 0 in both, which is what the tail metric keys on).
    """
    if "probs" in row:
        probs = row["probs"]
        if not isinstance(probs, list) or not probs:
            raise ValueError("`probs` must be a non-empty list")
        return [float(x) for x in probs]
    if "logits" in row:
        logits = row["logits"]
        if not isinstance(logits, list) or not logits:
            raise ValueError("`logits` must be a non-empty list")
        return _softmax([float(x) for x in logits])
    if "logprobs" in row:
        lp = row["logprobs"]
        if not isinstance(lp, dict) or not lp:
            raise ValueError("`logprobs` must be a non-empty object")
        return [math.exp(float(v)) for v in lp.values()]
    raise ValueError("capture row has none of: probs / logits / logprobs")


def percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolated percentile of an ASCENDING-sorted list.

    ``pct`` in [0, 100]. Matches numpy's default ('linear') interpolation so the
    99.9-pctile tail is computed the same way an analyst would reproduce it.
    Empty input -> 0.0 (a degenerate but well-defined floor).
    """
    if not sorted_values:
        return 0.0
    if pct <= 0:
        return sorted_values[0]
    if pct >= 100:
        return sorted_values[-1]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return sorted_values[lo]
    frac = rank - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


# The tail percentiles this probe reports. 99.9 is the headline (it is where
# club-3090's CLIFFS.md saw the 100%->54% collapse); 99 / 95 give the shape of
# the tail so an operator can see whether the regression is a single spike or a
# broad tail lift.
_TAIL_PCTS: tuple[float, ...] = (95.0, 99.0, 99.9)


@dataclass
class KLTailReport:
    """Structured outcome of a KL-tail comparison.

    ``tail`` maps each percentile (as a string key, e.g. ``"99.9"``) to the KL
    value at that percentile, in nats. ``regression`` is True iff the 99.9-pctile
    tail exceeds ``threshold`` — the gate verdict.
    """

    n_tokens: int
    mean_kl: float
    median_kl: float
    max_kl: float
    tail: dict[str, float]
    threshold: float
    headline_pct: float
    headline_kl: float
    regression: bool
    detail: str = ""
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_kl_tail(
    ref_dists: list[list[float]],
    cand_dists: list[list[float]],
    *,
    threshold: float = 0.10,
    headline_pct: float = 99.9,
) -> KLTailReport:
    """Compute the per-token KL tail between aligned reference + candidate runs.

    ``ref_dists`` / ``cand_dists`` are positionally aligned lists of per-token
    probability rows (line i of each = same decode position). They are truncated
    to the shorter length (a warning records the truncation) so an early-stopping
    candidate is handled gracefully.

    The verdict keys on the ``headline_pct`` (default 99.9) percentile of the
    per-token KL distribution: ``regression = headline_kl > threshold``. The tail
    — not the mean — is the signal, because the sub-8-bit-KV failure lives in the
    low-probability tokens (JSON braces, tool-arg boundaries) that the mean washes
    out. ``threshold`` is in nats; 0.10 nats ~= a 10% relative-likelihood drift on
    the reference's mass, a deliberately conservative default for an agentic /
    JSON gate (tune per workload on the rig).

    Pure / unit-tested. The CAPTURE of ``ref_dists`` / ``cand_dists`` from a live
    engine at two KV dtypes is the rig-follow-up; this function is the offline
    core.
    """
    warnings: list[str] = []
    n = min(len(ref_dists), len(cand_dists))
    if len(ref_dists) != len(cand_dists):
        warnings.append(
            f"length mismatch: ref={len(ref_dists)} cand={len(cand_dists)}; "
            f"truncated to {n} aligned positions (candidate likely early-stopped)."
        )
    if n == 0:
        return KLTailReport(
            n_tokens=0,
            mean_kl=0.0,
            median_kl=0.0,
            max_kl=0.0,
            tail={f"{p:g}": 0.0 for p in _TAIL_PCTS},
            threshold=threshold,
            headline_pct=headline_pct,
            headline_kl=0.0,
            regression=False,
            detail="no aligned token positions to compare",
            warnings=warnings or ["no aligned token positions"],
        )

    kls = [kl_divergence(ref_dists[i], cand_dists[i]) for i in range(n)]
    kls_sorted = sorted(kls)
    tail = {f"{p:g}": percentile(kls_sorted, p) for p in _TAIL_PCTS}
    headline_kl = percentile(kls_sorted, headline_pct)
    regression = headline_kl > threshold
    mean_kl = math.fsum(kls) / n
    median_kl = percentile(kls_sorted, 50.0)
    detail = (
        f"p{headline_pct:g} KL={headline_kl:.4f} nats "
        f"{'>' if regression else '<='} threshold {threshold:.4f} -> "
        f"{'TAIL REGRESSION' if regression else 'tail OK'} "
        f"(mean={mean_kl:.4f}, median={median_kl:.4f} over {n} tokens). "
        "Tail, not mean, is the signal: sub-8-bit KV breaks the low-prob tokens "
        "(JSON braces, tool-arg boundaries) the mean washes out."
    )
    return KLTailReport(
        n_tokens=n,
        mean_kl=mean_kl,
        median_kl=median_kl,
        max_kl=kls_sorted[-1],
        tail=tail,
        threshold=threshold,
        headline_pct=headline_pct,
        headline_kl=headline_kl,
        regression=regression,
        detail=detail,
        warnings=warnings,
    )


def _read_capture(path: str) -> list[list[float]]:
    """Read a JSON-lines capture file into a list of probability rows."""
    dists: list[list[float]] = []
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({e})") from e
            dists.append(row_to_probs(row))
    return dists


def compute_kl_tail_from_captures(
    ref_path: str,
    cand_path: str,
    *,
    threshold: float = 0.10,
    headline_pct: float = 99.9,
) -> KLTailReport:
    """Offline path: load two JSON-lines captures and compute the KL-tail report.

    This is the ``--from-captures`` entry point. The captures are produced ON the
    rig (see the module-docstring capture contract); this function consumes them
    host-side with no GPU. Positionally aligned, truncated to the shorter file.
    """
    ref = _read_capture(ref_path)
    cand = _read_capture(cand_path)
    return compute_kl_tail(
        ref, cand, threshold=threshold, headline_pct=headline_pct
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "KL-divergence tail quality probe: compare a bf16-KV reference run "
            "to a candidate (e.g. TQ k8v4) run over the SAME prompts and report "
            "the 99.9/99/95-percentile tail of the per-token output-distribution "
            "KL. Flags a TAIL REGRESSION when the 99.9-pctile exceeds --threshold. "
            "The needle ladder cannot see this (recall does not depend on the "
            "low-prob tail); this gate does."
        )
    )
    ap.add_argument(
        "--from-captures",
        nargs=2,
        metavar=("REF_JSONL", "CAND_JSONL"),
        required=True,
        help=(
            "two positionally-aligned JSON-lines captures: reference (bf16 KV) "
            "and candidate (e.g. TQ k8v4). Each line is one decode position with "
            "probs/logits/logprobs. Produced on the rig (see module docstring)."
        ),
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help="99.9-pctile KL (nats) above which a tail regression is flagged (default 0.10).",
    )
    ap.add_argument(
        "--headline-pct",
        type=float,
        default=99.9,
        help="tail percentile the verdict keys on (default 99.9).",
    )
    ap.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = ap.parse_args(argv)

    try:
        report = compute_kl_tail_from_captures(
            args.from_captures[0],
            args.from_captures[1],
            threshold=args.threshold,
            headline_pct=args.headline_pct,
        )
    except (OSError, ValueError) as e:
        sys.stderr.write(f"ERROR: {e}\n")
        return 2

    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print(report.detail)
        for w in report.warnings:
            print(f"  WARN: {w}")
        for pct in sorted(report.tail):
            print(f"  p{pct:>5}  KL = {report.tail[pct]:.4f} nats")
    # Exit 1 on a tail regression so a CI / rig driver can gate on it.
    return 1 if report.regression else 0


if __name__ == "__main__":
    sys.exit(main())
