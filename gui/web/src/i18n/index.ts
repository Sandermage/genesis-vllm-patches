// SPDX-License-Identifier: Apache-2.0
/**
 * Lingui i18n setup for sndr Control Center.
 *
 * Lingui chosen over react-intl + i18next for:
 *   - Smaller bundle (~20kb vs 60kb)
 *   - ICU MessageFormat (handles Russian plurals correctly)
 *   - TypeScript-first
 *   - .po file format compatible with mainstream translator workflows
 *
 * Adding a new language:
 *   1. Create gui/web/src/i18n/locales/<code>/messages.po
 *   2. Extract strings: npm run i18n:extract
 *   3. Translate the .po file
 *   4. Compile: npm run i18n:compile
 *   5. Add code to SUPPORTED_LOCALES below
 */
import { i18n } from '@lingui/core';

export type LocaleCode = 'en' | 'ru';

export const SUPPORTED_LOCALES: Record<LocaleCode, { label: string; flag: string }> = {
  en: { label: 'English', flag: '🇬🇧' },
  ru: { label: 'Русский', flag: '🇷🇺' },
};

export const DEFAULT_LOCALE: LocaleCode = 'en';

/** Load and activate a locale. Dynamic import keeps initial bundle small. */
export async function activateLocale(code: LocaleCode): Promise<void> {
  const { messages } = await import(`./locales/${code}/messages.ts`);
  i18n.load(code, messages);
  i18n.activate(code);
  document.documentElement.lang = code;
}

/** Detect the user's preferred locale from browser settings + persisted preference. */
export function detectLocale(): LocaleCode {
  const stored = localStorage.getItem('sndr-locale');
  if (stored && stored in SUPPORTED_LOCALES) {
    return stored as LocaleCode;
  }
  const browser = navigator.language.slice(0, 2).toLowerCase();
  if (browser in SUPPORTED_LOCALES) {
    return browser as LocaleCode;
  }
  return DEFAULT_LOCALE;
}

/** Persist a user's locale choice. */
export function setLocale(code: LocaleCode): void {
  localStorage.setItem('sndr-locale', code);
  activateLocale(code);
}

export { i18n };
