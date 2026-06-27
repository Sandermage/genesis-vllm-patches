# SPDX-License-Identifier: Apache-2.0
"""Genesis quality-gate harness — engine-agnostic probe + soak + verdict core.

Public, runnable quality-standard harness for Genesis-served endpoints. The
bash drivers live in `scripts/verify_stress.sh` and `scripts/soak_continuous.sh`;
the unit-testable request-generation and verdict logic lives here.

Adapted and extended from club-3090's public test harness
(github.com/noonghunna/club-3090, MIT). See docs/QUALITY_GATE.md.
"""

from __future__ import annotations

__all__ = ["kl_tail_probe", "probes", "soak"]
