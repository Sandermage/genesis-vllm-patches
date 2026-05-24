# License policy

> Codifies how Genesis source files and built artefacts relate to
> the Apache-2.0 [`LICENSE`](../LICENSE) at the repo root and to the
> Ed25519-signed license gate in
> [`vllm/sndr_core/license.py`](../vllm/sndr_core/license.py).
> Companion to [`CORE_ENGINE_BOUNDARY.md`](CORE_ENGINE_BOUNDARY.md)
> and [`SPONSORS.md`](SPONSORS.md).

Genesis ships from a single source tree under one license contract,
but the contract has two distribution sides because of the
[core/engine/private boundary](CORE_ENGINE_BOUNDARY.md). This
document is the binding statement of which side governs what.

## TL;DR

| Artefact | License |
| --- | --- |
| Source files under `vllm/sndr_core/**` | Apache License 2.0 ([`LICENSE`](../LICENSE)) |
| Built wheel `vllm-sndr-core` | Apache License 2.0 |
| Source files under `vllm/sndr_engine/**` (currently empty) | Commercial — terms published with the wheel if/when released |
| Built wheel `vllm-sndr-engine` | Commercial — license gate via Ed25519-signed token |
| Maintainer-private archive at repo-root `sndr_private/` | Not distributed (gitignored). Internal to the maintainer; no public license applies because nothing is published. |
| Bench results, methodology, raw logs | Apache-2.0, same as code |
| Documentation under `docs/` and `README.md` | Apache-2.0 (same as code; doc text is not a separate license) |

## Apache-2.0 community tier

Every file under `vllm/sndr_core/`, every test, every script, every
YAML config, every doc, and every artefact produced by
`pip wheel` / `python -m build` of [`pyproject.toml`](../pyproject.toml)
is governed by the Apache-2.0 license in [`LICENSE`](../LICENSE).

Practical consequences:

- Anyone may use, modify, redistribute, and fork the community wheel
  for any purpose, commercial or otherwise, subject to the Apache-2.0
  attribution and notice requirements.
- All Genesis patches re-classified to community after the
  2026-05-08 strict-AND audit (P67/P67b/P67c, PN21..PN24, PN26, PN29,
  PN38, PN40, PN57, P82, PN16, PN65, and the legacy P* family) ship
  under Apache-2.0.
- Bench JSONs in [`tests/integration/baselines/`](../tests/integration/baselines/)
  and methodology in [`BENCHMARKS.md`](BENCHMARKS.md) are Apache-2.0.
  See [`SPONSORS.md`](SPONSORS.md#maintainer-commitments) for the
  written commitment that this never changes.
- SPDX header `# SPDX-License-Identifier: Apache-2.0` is required at
  the top of every new source file under `vllm/sndr_core/`.

The Apache-2.0 license carries an attribution obligation: every
upstream author and contributor is credited in
[`CREDITS.md`](CREDITS.md) and inside individual patch docstrings.

## Commercial engine tier

The namespace `vllm/sndr_engine/` is reserved for the future
commercial wheel `vllm-sndr-engine`, built from a separate
`pyproject-engine.toml` (not present in the repo until the
commercial product is released). After the 2026-05-08 strict-AND
audit the namespace is **reserved but empty** — no Genesis patch
currently meets the four conditions (no public PR link, no external
credit, no `upstream_pr` field, not byte-equivalent with upstream)
required to qualify as engine-tier. See
[`PATCHES.md`](PATCHES.md#engine-tier-the-strict-and-boundary).

If and when the engine wheel ships, the following terms apply:

- Source files under `vllm/sndr_engine/**` carry a commercial SPDX
  header (`# SPDX-License-Identifier: LicenseRef-SndrEngine-Commercial`
  or equivalent) and a license file specific to the commercial
  product.
- The community wheel `vllm-sndr-core` continues to be importable
  without any engine wheel installed. Engine code is reached only
  through optional-discovery imports (`engine_available()`), so a
  missing or unlicensed engine wheel never breaks the community
  install.
- The runtime license gate in [`vllm/sndr_core/license.py`](../vllm/sndr_core/license.py)
  refuses to enable `tier="engine"` patches unless an Ed25519-signed
  token is present. The check returns a structured `LicenseStatus`
  enum so operators see a specific failure mode (`UNSIGNED_TOKEN`,
  `BAD_SIGNATURE`, `BAD_PAYLOAD`, `EXPIRED`, `VERSION_MISMATCH`,
  `LICENSED`, `LICENSED_LEGACY`) rather than a generic deny.
- Legacy unsigned keys remain accepted only in dev / CI / transition
  windows behind `SNDR_ALLOW_LEGACY_LICENSE_KEYS=1`, producing a
  one-time warning and the `LICENSED_LEGACY` status. Production must
  use signed tokens.

The signed-token verification path needs the optional `cryptography`
dependency. Installed via the `license` extra:

```
pip install vllm-sndr-core[license]
```

This extra is purely for verification — it does **not** install the
commercial wheel or grant access to engine functionality. It only
enables the gate to verify a token the operator already holds.

## Maintainer-private archive

The repo-root `sndr_private/` directory is the maintainer's private
working space — planning, audits, abandoned experiments, run logs.
It is `gitignored`, never reaches the public repository, and is not
distributed in any wheel. Because nothing in it is published, no
public license applies. The audit script
`scripts/audit_private_namespace.py` (rule 3) verifies that this
directory remains gitignored on every pre-commit run.

The maintainer-private archive must never:

- Be moved under `vllm/` (would land it inside the packaged tree —
  blocked by hard rule #27, see
  [`CORE_ENGINE_BOUNDARY.md`](CORE_ENGINE_BOUNDARY.md#what-is-forbidden))
- Be committed to git (blocked by `.gitignore`)
- Be referenced from `pyproject.toml` package discovery (out of
  scope: discovery only includes `vllm.sndr_core*`)

## Source-of-truth for license claims

Anywhere a file, doc, or wheel claims a license, this document is
the canonical reference:

- `LICENSE` at repo root holds the verbatim Apache-2.0 text.
- `pyproject.toml` declares `license = {text = "Apache-2.0"}` for
  the community wheel.
- Every source file under `vllm/sndr_core/` carries
  `# SPDX-License-Identifier: Apache-2.0`.
- This file (`docs/LICENSE_POLICY.md`) is the narrative explanation
  of what the SPDX headers and `LICENSE` file mean across the
  three-zone boundary.

If the three sources ever appear to disagree, the order of
precedence is: (1) the verbatim `LICENSE` file, (2) the per-file
SPDX header, (3) this document. Disagreement is a bug — open an
issue.

## Contributor agreement

Genesis does not require a CLA. Contributions submitted via pull
request are accepted under the Apache-2.0 terms in `LICENSE` as
described in [`CONTRIBUTING.md`](CONTRIBUTING.md). Inbound
contributions to the engine namespace are not accepted from the
public (the namespace is for clean-room maintainer-original work
only, per the strict-AND rule).

## See also

- [`LICENSE`](../LICENSE) — verbatim Apache-2.0 text
- [`CORE_ENGINE_BOUNDARY.md`](CORE_ENGINE_BOUNDARY.md) — three-zone namespace policy
- [`SPONSORS.md`](SPONSORS.md) — sponsorship terms, including the
  explicit decoupling of sponsorship from the commercial engine wheel
- [`PATCHES.md`](PATCHES.md#engine-tier-the-strict-and-boundary) — strict-AND rule for tier=engine
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — contribution workflow and Apache-2.0 inbound terms
- [`CREDITS.md`](CREDITS.md) — upstream attribution
