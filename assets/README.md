# `assets/` — Genesis brand assets

This directory holds binary brand assets that ship with the repository.
Keep contents here small and intentional: logos, hero images, badges,
favicons. Anything large or generated belongs elsewhere (`benchmarks/`
for plots, `docs/` for diagrams).

## Files

| File | Purpose | Used by |
|---|---|---|
| `logo.png` | Genesis Sea-Born Neural Beacon — 2816×1536 master, displayed at 1280px max width in README | README.md hero, project banners |
| `logo-favicon.png` | 256×256 square crop for tab icons, badge generation, GitHub avatar | future docs site, GitHub social card |
| `og-card.png` | 1280×640 Open Graph / social-preview card (title + headline figures); regenerate with `python3 assets/charts/_og_card.py` | GitHub social preview, link unfurls (Reddit / HN / X / Slack) |

## Setting the GitHub social preview

`og-card.png` is the link-unfurl image. GitHub does **not** expose the social
preview via API — a maintainer must set it once in the UI:

**Settings → General → Social preview → Edit → Upload an image** → pick
`assets/og-card.png`. After that, any link to the repo unfurls with the card.

Regenerate the card after a headline-number change:

```bash
python3 assets/charts/_og_card.py   # writes assets/og-card.png (needs matplotlib)
```

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
