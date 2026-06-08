# Wide-screen (3440) visual audit — SNDR Control Center GUI

**Status:** audit only, no CSS changed yet. Awaiting sign-off on the per-section
plan below before edits.
**Date:** 2026-06-08 · branch `feat/v12-sndr-platform` (clean at `2ec1ee65`).
**Direction agreed:** *keep tiles full-width, enrich them so the width is earned*
(bigger value + sub-label + trailing context/sparkline), fixed **per-section** —
no global grid change (the last commit rolled a global experiment back).

## Method

- Rendered the **real production bundle** (`vite preview` of `dist/`) with the
  hermetic API mock (`e2e/fixtures.ts`) — same bundle the daemon serves.
- Forced viewport **3440×1440, dark theme, comfortable density** (the operator's
  workstation). Harness: `e2e/wide-shots.spec.ts` + `playwright.shots.config.ts`.
  Run: `npx playwright test --config playwright.shots.config.ts` → `/tmp/sndr-wide/*.png`.
- **Build health:** `npm run build` green (0 TS errors, 1654 modules). No
  functional breakage in the build/typecheck path.

### Data caveat (important)

The hermetic fixtures return **empty** shapes (all `0` / "no rows"). That
*exaggerates* sparseness: on the live daemon these tiles carry real counts and
the catalog tables/cards have rows. The **structural** stretching below is real
regardless of data, but judge final density against live data before signing off
each section.

## Root cause (one pattern, several sites)

Small KPI / stat strips lay out a handful of tiles with either
`repeat(N, minmax(0,1fr))` or `auto-fit minmax(small, 1fr)`. On 3440 the few
tiles stretch to ~440–570px each, while the tile's content (icon + label + a
2–3 char number, left-aligned in a `flex-column`) stays ~80px — so each tile is
mostly empty horizontal space. The 2-column **content** grids below the strips
(catalog/editor/detail panels) fill the width fine and are **not** the problem.

## Per-section findings (severity-ranked)

| # | Section | Offending element | CSS rule (line) | Markup | Severity |
|---|---------|-------------------|-----------------|--------|----------|
| 1 | **Doctor** | 4-tile summary (Healthy/Info/Warnings/Blocked) | `.doctor-stat-row` `repeat(4,1fr)` — [styles.css:1222](src/styles.css#L1222) | [doctor.tsx:29](src/sections/doctor.tsx#L29), also [setup-wizard.tsx:153](src/sections/setup-wizard.tsx#L153) | High — 4 tiles × ~440px, tiny "0" each |
| 2 | **Overview** | 6 KPI tiles (Presets/Models/Patches/Hosts/Doctor/Engine) | `.ov-kpis` `auto-fit minmax(112px,1fr)` — [styles.css:4044](src/styles.css#L4044) | [App.tsx:1873-1878](src/App.tsx#L1873) | High — 6 × ~565px |
| 3 | **Models** | 7 stat tiles (Models/Families/MoE/Dense/Tool-ready/Min VRAM/Selected) | `.model-summary-strip` `repeat(7,1fr)` — [styles.css:2459](src/styles.css#L2459) (base `.preset-summary-strip` [2060](src/styles.css#L2060)) | [models-workbench.tsx:79](src/sections/models-workbench.tsx#L79) | High |
| 4 | **Presets** | 6 stat tiles (Presets/Annotated/Missing card/Models/Hardware/Selected) | `.preset-summary-strip` `repeat(6,1fr)` — [styles.css:2060](src/styles.css#L2060) | [preset-views.tsx:184](src/sections/preset-views.tsx#L184) | High |
| 5 | **Launch Plan** | 5-tile metric strip (Presets/Models/Profiles/Product API/Metric) | `.metric-strip` `repeat(5,1fr)` — [styles.css:679](src/styles.css#L679) | [App.tsx:1126](src/App.tsx#L1126) | Med |
| 6 | **Planner** | 6 KV tiles (Weights/KV/Overhead/Total/Headroom/Max ctx) | `.kpi-strip` `auto-fit minmax(128px,1fr)` — [styles.css:4555](src/styles.css#L4555) | [Planner.tsx:161](src/Planner.tsx#L161) | Med |
| 7 | **Configs** | bottom status strip (Compatible/Status/Context/Sequences/KV/Spec) | `.config-status-strip` `auto-fit minmax(150px,1fr)` — [styles.css:1619](src/styles.css#L1619) | configs-workbench | Med |
| 8 | **Patches** | 4-tile registry summary cramped in left ~40%, right ~60% empty | `.patch-summary-grid` [1189](src/styles.css#L1189) inside `.patch-control-grid` `1.7fr / 1fr` [1280](src/styles.css#L1280) | patch-overview | Med — empty-data layout imbalance |

**Looks good already at 3440 (reference):** Virtualization, Routing, Benchmarks,
Hardware, Fleet — they fill width with wide content bands + at most one fact
strip, so no fix needed.

## Functional bug found (not layout)

**Containers section hard-crashes** to the error boundary on empty/partial data:
*"Cannot read properties of undefined (reading 'map')"*.

- Cause: [Containers.tsx:358](src/Containers.tsx#L358) — `df.types.map(...)` is
  reached whenever `df` is truthy, but the disk-usage payload's `types` (and
  `total_size`) are not guaranteed. The hermetic `systemDf` fixture has no
  `types` field, so `df.types` is `undefined` → throw.
- Fix (1 line, low-risk): guard the access — `(df.types ?? []).map(...)` and a
  fallback for `df.total_size`.
- **Unverified:** whether the live `/system/df` always returns `types`. The
  unguarded access is a latent robustness bug either way; every *other* section
  degrades to an empty state on missing data — Containers should too. Worth
  confirming against the live daemon before/after the fix.

## Proposed enrich-tiles pattern (the actual change, for review)

For each strip in the table, the fix is the same shape — **not** a width cap:

1. **Tile internals → horizontal.** Today the tile is a left-aligned
   `flex-direction: column` (icon, label, value), leaving the right ~70% empty.
   Switch to a horizontal layout: `icon | (value + label) | trailing slot`,
   `justify-content: space-between`, so a wide tile is filled.
2. **Trailing slot earns the width.** Put real context there — the existing
   `sub` line (OvKpi/Kpi already support `sub`), a delta/tone pill, or a small
   sparkline (the Containers `Sparkline` component already exists and can be
   reused). Where there's genuinely no second datum, widen typography (larger
   value) rather than padding empty space.
3. **Column ceiling.** Keep `auto-fit` but pair it with a sane `max` so a strip
   of 4–7 items doesn't each balloon past a comfortable card width — e.g.
   `repeat(auto-fit, minmax(180px, 260px))` + `justify-content: start`, so extra
   width becomes a small trailing margin, not 300px of dead tile.

Each section gets its own commit + a before/after 3440 screenshot pair for
sign-off, in this order (sharing the `.preset-summary-strip` base means #3 and #4
land together):

> Doctor → Overview → Models + Presets → Launch Plan → Planner → Configs → Patches
> Containers crash fix can go first (independent, functional).

## Open questions for sign-off

1. Sparkline-in-tile vs. sub-label-only — sparklines look best but only some
   tiles have a time series. OK to mix (spark where data exists, sub elsewhere)?
2. Should the Containers crash fix ship now (independent functional fix), ahead
   of the layout pass?
3. Confirm: no global `max-width`/centering on the workspace (per the prior
   rollback) — fill width, just denser. Assumed yes.
