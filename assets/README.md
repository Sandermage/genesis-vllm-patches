# `assets/` — Genesis brand assets

This directory holds binary brand assets that ship with the repository.
Keep contents here small and intentional: logos, hero images, badges,
favicons. Anything large or generated belongs elsewhere (`benchmarks/`
for plots, `docs/` for diagrams).

## Files

| File | Purpose | Used by |
|---|---|---|
| `logo.png` | Genesis Sea-Born Neural Beacon — 2816×1536 master, displayed at 1280px max width in README | README.md hero, GitHub social-preview, project banners |
| `logo-favicon.png` | 256×256 square crop for tab icons, badge generation, GitHub avatar | future docs site, GitHub social card |

## Versioning the logo

The master logo is the work of Sander (Barzov Aleksandr, Odessa).
Treat it as part of the project's identity: do not modify or recolor
without prior agreement.

If you replace `logo.png`, keep the same aspect ratio (≈ 1.83 :1) so
the GitHub README hero block does not need a height override.

## Adding new assets

1. Drop the file into `assets/`.
2. Reference it from a markdown doc using a repo-relative path:
   `![Alt text](assets/your-asset.png)`.
3. Add a one-line entry to the table above so future contributors
   can find it.
4. If the asset is over ~500 KB, evaluate whether it really belongs
   in git or should ship from a CDN / release artifact.
