// SPDX-License-Identifier: Apache-2.0
// Shell test: render the whole <App />, mock the daemon, and exercise the boot
// orchestration + SectionWorkspace tab routing in jsdom (under coverage). The
// hermetic Playwright spec proves the same against a real browser; this one
// drives the App-level state machine and section switching where v8 can measure
// it. Only the network `api` object is replaced — getApiBase/normalizeBaseUrl/
// hostLabel stay real so the shell wiring is genuinely exercised.
import { describe, it, expect, beforeAll, afterEach, vi, type Mock } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";

vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  const { makeApiMock } = await import("./test/api-fixtures");
  return { ...actual, api: makeApiMock(actual.api as unknown as Record<string, unknown>) };
});

// The mocked network object — same vi.fn instances the App calls (Proxy-cached).
import { api } from "./api";
const apiMock = api as unknown as Record<string, Mock>;

beforeAll(() => {
  // jsdom is missing a few browser APIs the shell touches on mount.
  if (typeof window.localStorage?.setItem !== "function") {
    const store = new Map<string, string>();
    const mem = {
      getItem: (k: string) => (store.has(k) ? store.get(k)! : null),
      setItem: (k: string, v: string) => void store.set(k, String(v)),
      removeItem: (k: string) => void store.delete(k),
      clear: () => store.clear(),
      key: (i: number) => Array.from(store.keys())[i] ?? null,
      get length() { return store.size; },
    };
    Object.defineProperty(window, "localStorage", { value: mem, configurable: true });
  }
  if (!("EventSource" in globalThis)) {
    class FakeEventSource {
      onmessage: ((e: MessageEvent) => void) | null = null;
      onerror: ((e: Event) => void) | null = null;
      addEventListener() {}
      removeEventListener() {}
      close() {}
    }
    (globalThis as Record<string, unknown>).EventSource = FakeEventSource;
  }
  if (!Element.prototype.scrollIntoView) Element.prototype.scrollIntoView = vi.fn();
  if (!window.matchMedia) {
    window.matchMedia = vi.fn().mockReturnValue({
      matches: false, media: "", addEventListener: vi.fn(), removeEventListener: vi.fn(),
      addListener: vi.fn(), removeListener: vi.fn(), onchange: null, dispatchEvent: () => false,
    }) as unknown as typeof window.matchMedia;
  }
});

afterEach(cleanup);

import App from "./App";

describe("App shell", () => {
  it("boots to the ready dashboard and renders the section nav", async () => {
    render(<App />);
    // The sidebar nav is static; the dashboard becomes ready once the mocked
    // boot fetches resolve. Wait for the API daemon card to leave "Connecting".
    expect(await screen.findByRole("navigation", { name: "SNDR sections" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Overview" })).toBeTruthy();
    await waitFor(() => expect(apiMock.overview).toHaveBeenCalled());
    await waitFor(() => expect(apiMock.launchPlan).toHaveBeenCalled());
  });

  it("routes to another section when its nav button is clicked", async () => {
    render(<App />);
    const patchesBtn = await screen.findByRole("button", { name: "Patches" });
    fireEvent.click(patchesBtn);
    // SectionWorkspace swaps content; the patches fetch fires on that route.
    await waitFor(() => expect(apiMock.patches).toHaveBeenCalled());
    expect(patchesBtn.className).toContain("active");
  });
});
