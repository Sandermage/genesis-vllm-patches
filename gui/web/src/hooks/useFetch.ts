// SPDX-License-Identifier: Apache-2.0
import { useEffect, useState } from "react";

export type FetchState = "idle" | "loading" | "ready" | "error";

// Shared fetch-on-mount state machine: loading/ready/error, AbortController-based
// cancellation, and a reload() for retry. Replaces the hand-rolled `cancelled`
// flag pattern that was duplicated across panels. The fetcher receives the
// AbortSignal so the in-flight HTTP request is cancelled on dep change/unmount.
export function useFetch<T>(
  fetcher: (signal: AbortSignal) => Promise<T>,
  deps: unknown[],
  opts: { enabled?: boolean } = {}
): { data: T | null; state: FetchState; error: string | null; reload: () => void } {
  const enabled = opts.enabled ?? true;
  const [data, setData] = useState<T | null>(null);
  const [state, setState] = useState<FetchState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);
  useEffect(() => {
    if (!enabled) {
      setData(null);
      setState("idle");
      setError(null);
      return;
    }
    const controller = new AbortController();
    setState("loading");
    setError(null);
    fetcher(controller.signal)
      .then((result) => {
        if (controller.signal.aborted) return;
        setData(result);
        setState("ready");
      })
      .catch((err) => {
        if (controller.signal.aborted) return;
        setData(null);
        setError(err instanceof Error ? err.message : String(err));
        setState("error");
      });
    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, enabled, nonce]);
  return { data, state, error, reload: () => setNonce((value) => value + 1) };
}
