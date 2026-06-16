// SPDX-License-Identifier: Apache-2.0
// Live backend event feed. In open (token-less) mode it subscribes to the
// daemon's native SSE stream; EventSource cannot send an Authorization header,
// so token-protected daemons fall back to authenticated polling. Either way the
// hook returns the most recent ~100 events for the activity panel.
import { useEffect, useState } from "react";
import { api, getApiToken, type BackendEvent } from "../api";

export function useLiveEvents(apiBase: string, enabled: boolean): BackendEvent[] {
  const [events, setEvents] = useState<BackendEvent[]>([]);
  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    const token = getApiToken();
    // Open mode: native SSE stream. EventSource cannot send an Authorization
    // header, so token-protected daemons fall back to authenticated polling.
    if (!token && typeof EventSource !== "undefined") {
      const source = new EventSource(`${apiBase}/api/v1/events`);
      source.addEventListener("snapshot", (event) => {
        try {
          const data = JSON.parse((event as MessageEvent).data);
          if (!cancelled) setEvents((data.events ?? []).slice(-100));
        } catch { /* ignore malformed frame */ }
      });
      source.addEventListener("event", (event) => {
        try {
          const item = JSON.parse((event as MessageEvent).data) as BackendEvent;
          if (!cancelled) setEvents((prev) => [...prev, item].slice(-100));
        } catch { /* ignore malformed frame */ }
      });
      source.onerror = () => { /* browser auto-reconnects */ };
      return () => { cancelled = true; source.close(); };
    }
    let cursor = 0;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const result = await api.eventsRecent(cursor);
        if (cancelled) return;
        if (result.events.length) {
          setEvents((prev) => [...prev, ...result.events].slice(-100));
          cursor = result.last_seq;
        }
      } catch { /* daemon may be briefly unreachable */ }
      if (!cancelled) timer = setTimeout(poll, 4000);
    };
    void poll();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [apiBase, enabled]);
  return events;
}
