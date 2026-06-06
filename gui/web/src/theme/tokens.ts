// SPDX-License-Identifier: Apache-2.0
/**
 * Design tokens for sndr Control Center.
 *
 * Layered on top of Carbon Design System g100 theme (dark, productivity-focused).
 * See: https://carbondesignsystem.com/elements/themes/overview/
 *
 * Engineering principle: every color, spacing, and typography decision lives
 * here as a named token. Component code references tokens, never literal
 * values. This makes theme overrides and brand changes trivial.
 */

export const tokens = {
  /** Brand accent colors layered on Carbon g100. */
  brand: {
    /** Primary blue — matches Carbon's blue 60. */
    primary: '#0f62fe',
    /** Lighter blue for hover states. */
    primaryHover: '#4589ff',
    /** Subtle accent used for "engine active" indicators. */
    accent: '#42be65',
  },

  /** Semantic colors for status indicators. */
  status: {
    /** Patch applied, drift OK, license valid. */
    success: '#24a148',
    /** Drift detected, license expiring soon. */
    warning: '#f1c21b',
    /** Patch failed, license expired, container crashed. */
    danger: '#fa4d56',
    /** Informational notification. */
    info: '#4589ff',
    /** Critical issue requiring immediate attention. */
    critical: '#ff832b',
  },

  /** Tier badge colors. */
  tier: {
    community: '#4589ff',
    engine: '#a56eff',
  },

  /** Lifecycle badge colors. */
  lifecycle: {
    stable: '#24a148',
    experimental: '#f1c21b',
    retired: '#8d8d8d',
  },

  /** Spacing scale (Carbon 8px base). */
  spacing: {
    xs: '4px',
    sm: '8px',
    md: '16px',
    lg: '24px',
    xl: '32px',
    xxl: '48px',
  },

  /** Typography (IBM Plex). */
  fonts: {
    sans: '"IBM Plex Sans", system-ui, sans-serif',
    mono: '"IBM Plex Mono", "SF Mono", Consolas, monospace',
    serif: '"IBM Plex Serif", Georgia, serif',
  },

  /** Animation timing. */
  motion: {
    fast: '70ms',
    moderate: '110ms',
    slow: '240ms',
    easing: 'cubic-bezier(0.2, 0, 0.38, 0.9)',
  },
} as const;

export type Tokens = typeof tokens;
