// SPDX-License-Identifier: Apache-2.0
/**
 * main.carbon.tsx — entry point for the Carbon-based Control Center.
 *
 * Loaded when ``vite build --mode=carbon`` (see ``vite.config.ts``).
 * The legacy ``main.tsx`` continues to mount the old App.tsx for
 * operators on the v11 line until the swap is finalised.
 *
 * Provider stack (outside-in):
 *   <StrictMode>
 *   <QueryClientProvider>   — TanStack Query, shared cache
 *   <LinguiI18nProvider>    — Lingui locale, dynamic load
 *   <CarbonApp>             — Theme + Router + SideNav + Routes
 */
import React, { useEffect, useState } from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { I18nProvider } from '@lingui/react';
import { i18n, activateLocale, detectLocale } from './i18n';
import CarbonApp from './CarbonApp';
import './styles.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // 30s stale by default — operator-facing UI doesn't need
      // subsecond freshness for most surfaces.
      staleTime: 30_000,
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

function I18nBootstrap({ children }: { children: React.ReactNode }): JSX.Element | null {
  const [ready, setReady] = useState(false);
  useEffect(() => {
    activateLocale(detectLocale()).then(() => setReady(true));
  }, []);
  if (!ready) return null;
  return <I18nProvider i18n={i18n}>{children}</I18nProvider>;
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <I18nBootstrap>
        <CarbonApp />
      </I18nBootstrap>
    </QueryClientProvider>
  </React.StrictMode>,
);
