# Draft reply — noonghunna/qwen36-27b-single-3090#2 (radojko backward-compat ask)

**Status:** DRAFT for Sander review. Do NOT post until approved.

**Context:** trueimage opened #2 about missing `patch_genesis_unified.py` (deleted in v7.13 → v7.14 migration when monolith was split into modular `_genesis/` package). radojko commented 2026-04-27 13:55 asking for a backward-compat shim or migration documentation.

**Thread URL:** https://github.com/noonghunna/qwen36-27b-single-3090/issues/2

**Strategic:** Sander's repo is the source. Cheap goodwill: provide either a shim, OR a clearly-linked migration doc. Avoids same footgun for future v7.13 followers.

**Recommended approach:** publish a tiny `patch_genesis_unified.py` shim in Genesis repo root that's a one-liner deprecation notice + redirect to the new module. Plus add a "Migration from v7.13 monolith" section to README.

---

## English version (comment posted to noonghunna's issue OR our Genesis repo)

> @trueimage @radojko thanks for flagging — confirming this is on Genesis side, not noonghunna's. The v7.13 → v7.14 migration split the monolith `patch_genesis_unified.py` into the modular `vllm/_genesis/` package. The `_genesis` package is intentionally Python-importable (`import vllm._genesis.patches.apply_all`) which broke the old "drop a single file in" workflow some users had.
>
> Two fixes I'm queuing for the Genesis repo:
>
> **1. Compatibility shim** — I'll publish a tiny `patch_genesis_unified.py` at Genesis repo root that just imports + invokes the new modular `apply_all` so old launch scripts keep working:
>
> ```python
> # patch_genesis_unified.py — v7.14+ compatibility shim
> # The monolith was split into the modular vllm._genesis package.
> # This shim invokes the new apply_all for backward-compat.
> # New deployments should use:
> #     python3 -m vllm._genesis.patches.apply_all
> import warnings
> warnings.warn(
>     "patch_genesis_unified.py is deprecated since v7.14. "
>     "Use `python3 -m vllm._genesis.patches.apply_all` instead.",
>     DeprecationWarning, stacklevel=2,
> )
> from vllm._genesis.patches.apply_all import main
> if __name__ == "__main__":
>     main()
> ```
>
> **2. Migration section in README** — clear "v7.13 → v7.14+" subsection with the new launch command + env-flag mapping. Plus link to it from any Genesis-using repo via PR.
>
> Will push within 24h. If you have other v7.13 → v7.14 surprises (env vars renamed, scripts moved), please pile them in this thread or open an issue at [Sandermage/genesis-vllm-patches](https://github.com/Sandermage/genesis-vllm-patches/issues) so I can document them all in one go.
>
> *(Translated from Russian via AI assist; happy to clarify anything ambiguous.)*

---

## Russian version

> @trueimage @radojko спасибо за flag — подтверждаю это на Genesis side, не у noonghunna. v7.13 → v7.14 migration разбил monolith `patch_genesis_unified.py` на modular `vllm/_genesis/` package. Старый "drop a file in" workflow сломался.
>
> Два fix'а в очередь для Genesis repo:
>
> **1. Compat shim** — опубликую тоненький `patch_genesis_unified.py` в Genesis root что вызывает новый modular `apply_all` для backward-compat (см. EN version выше).
>
> **2. Migration section** — чёткий "v7.13 → v7.14+" subsection в README с новой launch command + env-flag mapping.
>
> Push в течение 24h. Если есть другие v7.13 → v7.14 surprises — пишите в этот thread или открывайте issue в Sandermage/genesis-vllm-patches.

---

## Implementation tasks (after approval)

1. Create `patch_genesis_unified.py` shim at Genesis repo root (Sander has authority)
2. Add "Migration from v7.13 monolith" section to Genesis README
3. Document env-flag mapping (any renamed vars between v7.13 monolith and modular)
4. Optionally tag a v7.13.LAST release for users who need monolith
5. Comment on noonghunna's #2 with link to fix
