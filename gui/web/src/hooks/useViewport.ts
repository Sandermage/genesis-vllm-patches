// SPDX-License-Identifier: Apache-2.0
// Viewport/resolution detection. Reports the live window size plus a coarse
// "tier" so sections can adapt their layout to the actual screen — denser on a
// 3440 ultrawide, simpler on a laptop. The tier is also stamped on the shell as
// `data-viewport` so CSS can branch without prop-drilling.
import { useEffect, useState } from "react";

export type ViewportTier = "compact" | "standard" | "wide" | "ultra";

// Breakpoints chosen for this app's real targets: laptops (<1280), standard
// desktops (1280–1919), wide monitors (1920–2879) and the operator's 3440
// ultrawide (>=2880). Pure function so it is unit-testable.
export function tierForWidth(width: number): ViewportTier {
  if (width < 1280) return "compact";
  if (width < 1920) return "standard";
  if (width < 2880) return "wide";
  return "ultra";
}

export interface Viewport {
  width: number;
  height: number;
  tier: ViewportTier;
}

export function useViewport(): Viewport {
  const read = (): { width: number; height: number } =>
    typeof window === "undefined"
      ? { width: 1440, height: 900 }
      : { width: window.innerWidth, height: window.innerHeight };

  const [size, setSize] = useState(read);

  useEffect(() => {
    if (typeof window === "undefined") return;
    let raf = 0;
    const onResize = () => {
      // Coalesce a burst of resize events into one state update per frame.
      window.cancelAnimationFrame(raf);
      raf = window.requestAnimationFrame(() => setSize(read()));
    };
    window.addEventListener("resize", onResize);
    onResize();
    return () => {
      window.removeEventListener("resize", onResize);
      window.cancelAnimationFrame(raf);
    };
  }, []);

  return { ...size, tier: tierForWidth(size.width) };
}
