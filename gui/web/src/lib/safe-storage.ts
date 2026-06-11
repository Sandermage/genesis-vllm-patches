// SPDX-License-Identifier: Apache-2.0
// localStorage access that never throws. `window.localStorage` raises on access
// in private-mode Safari, sandboxed iframes, and storage-disabled embeds — a
// bare get/set there would white-screen the app at module load or in an effect.
// These wrappers degrade to a no-op / null instead.

export function lsGet(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

export function lsSet(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    /* storage unavailable or over quota — ignore */
  }
}

export function lsRemove(key: string): void {
  try {
    window.localStorage.removeItem(key);
  } catch {
    /* storage unavailable — ignore */
  }
}
