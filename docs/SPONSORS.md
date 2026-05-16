# Sponsorship

Genesis vLLM Patches is developed and maintained by
[Sandermage (Aleksandr Barzov)](https://github.com/Sandermage) in
Odessa, Ukraine. The code is Apache-2.0 and will always remain
open; this page exists for people who have asked how to support
the work — both the maintainer's time and the project's
cross-platform test bench.

> Sponsorship is voluntary and carries no obligations on either
> side. Genesis has no premium tiers, no paywalled patches, and no
> priority queues — past and future contributions buy access to the
> same public docs and issue tracker as everyone else. What
> sponsorship does buy is **collective bandwidth**: more total
> hours spent on Genesis instead of on other paid commitments, plus
> more hardware to validate patches against.

## What sponsorship enables

Genesis is a one-maintainer project funded out of the maintainer's
own savings and developed alongside other commitments. Support
contributes to two concrete buckets — **time** and **hardware** —
and the project ratchets forward at whatever pace operators
collectively underwrite.

### Maintainer time

Working on Genesis is currently a part-time effort. Sponsorship
directly translates into the maintainer being able to reduce
unrelated paid commitments and spend that time instead on:

- Faster vLLM pin bumps with full deep-diff review of every
  upstream PR against existing patches (see iron rule #11 in
  [`CONTRIBUTING.md`](CONTRIBUTING.md)).
- More frequent bench cycles + longer soak runs to catch
  fragmentation-class regressions like Cliff 2b.
- Better-quality design docs and follow-up on the deferred
  architecture work (PN95 runtime split, GDN kernel fusion,
  WorkspaceFacade consolidation — see
  [`PATCH_DESIGNS.md`](PATCH_DESIGNS.md)).
- More responsive triage on community issues and faster review of
  community-submitted patches and configs.

This is the bucket that scales linearly with sponsorship — every
hour funded is an hour spent on Genesis instead of on something
else.

### Cross-platform test bench

Genesis is currently validated on the reference rig:
**2× NVIDIA RTX A5000** (Ampere SM 8.6, 24 GB each). Every patch
beyond that envelope — Hopper, Blackwell, RTX PRO 6000, H100,
Intel XPU, AMD ROCm CDNA — ships with defensive `applies_to` guards
but cannot be empirically validated against real silicon. Operators
on those platforms today get "graceful skip" semantics instead of
measured numbers.

Expanding the test bench would let the maintainer:

- Replace defensive guards with measured cross-platform baselines.
- Land hardware-specific patches (NVFP4 on Blackwell, native FP8 on
  Hopper, FlashInfer-MoE on H100, XPU / ROCm equivalents).
- Publish bench-attached proof artefacts for the wider patch
  registry (today only ~14 of 169 entries carry bench attachments —
  see [`RELEASE_POLICY.md`](RELEASE_POLICY.md)).

## How to support

### Financial channels

| Channel | Address |
| --- | --- |
| USDT (BEP-20) | `0x1E8C74aC4f37A201733D185b2293e9D69f305306` |
| USDT (TRC-20) | `TSyVYTA4PK22w3tZ7vgoc1itjXU5p4Vfks` |
| ETH (mainnet) | `0x1E8C74aC4f37A201733D185b2293e9D69f305306` |
| BTC | `bc1q9tau6xqgrv5jjgst63yjux550gslq6nm7y7q9f` |
| PayPal | `sander.odessa@gmail.com` |

### Hardware loan or donation

If you have a Hopper / Blackwell / RTX PRO 6000 / H100 / Intel XPU /
AMD ROCm card you can lend or donate to the project, write to
`sander.odessa@gmail.com` to discuss logistics. Loaned hardware is
returned when the validation cycle finishes; donated hardware
becomes part of the project's permanent test bench and shows up in
the acknowledgments below.

### Cross-rig bench reports

Bench JSONs from rigs not yet in
[`tests/integration/baselines/`](../tests/integration/baselines/)
are valuable contributions in their own right — no money required.
See [`BENCHMARKS.md`](BENCHMARKS.md) for the run-and-share guide.

## Maintainer commitments

- Everything Genesis ships stays under Apache-2.0, including bench
  results, methodology, and raw logs.
- Every upstream author and contributor is credited in
  [`CREDITS.md`](CREDITS.md) and inside individual patch docstrings.
- No functionality will ever be gated behind sponsorship,
  paywalls, or premium tiers.
- Support does not buy maintainer time, custom features, or
  priority on the issue tracker.

## Security contact

Security issues should NOT be reported in public issues. Email
`sander.odessa@gmail.com` with details — disclosures are
acknowledged within 72 hours and coordinated through standard
responsible-disclosure timelines.

## Acknowledgments

Past supporters and hardware sponsors will be listed here as
sponsorship arrives. The list is opt-in — contributors can request
attribution or stay anonymous. To opt in, mention the preferred
display name when you reach out.
