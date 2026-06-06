// SPDX-License-Identifier: Apache-2.0
/**
 * ThemeProvider — wraps the application in Carbon Design System's g100 theme
 * with sndr brand overrides applied via CSS tokens.
 *
 * Carbon's ``<Theme>`` component scopes its theme tokens to its subtree, so
 * the rest of the app can use ``<Theme>`` again for inverse-themed regions
 * (rare but supported).
 */
import type { ReactNode } from 'react';
import { Theme } from '@carbon/react';
import '@carbon/styles/css/styles.css';
import '@ibm/plex/css/ibm-plex.css';

interface ThemeProviderProps {
  children: ReactNode;
}

export function ThemeProvider({ children }: ThemeProviderProps): JSX.Element {
  return <Theme theme="g100">{children}</Theme>;
}
