// SPDX-License-Identifier: Apache-2.0
// Live running-model detection. Fetches /api/v1/engine/model — the served model
// of a running vLLM engine bridged to the SNDR catalog. With no host/port it
// targets the daemon's configured engine (the auto-discovered "what's running
// now"); pass host/port/hostId to inspect a specific engine (chat / clients).
// Shared, deduped cache: the top-bar chip, Overview KPI and Models badge all
// read the default-engine key, so one poll feeds them all.
import { useQuery } from "@tanstack/react-query";
import { api, type EngineModelDetail } from "../api";

export function useEngineModel(host?: string, port?: number, apiKey?: string, hostId?: string) {
  return useQuery<EngineModelDetail>({
    // The key must include an apiKey fingerprint: typing a valid key used to
    // leave the keyless 401/unreachable result cached (key change ≠ key in the
    // queryKey) until the 30 s interval fired. Length+tail avoids putting the
    // secret itself into the query cache key.
    queryKey: ["engine-model", host ?? "", port ?? 0, hostId ?? "", apiKey ? `${apiKey.length}:${apiKey.slice(-4)}` : ""],
    queryFn: () => api.engineModel(host || undefined, port, apiKey || undefined, hostId || undefined),
    staleTime: 15_000,
    refetchInterval: 30_000,
    // The route returns 200 with reachable:false when no engine answers, so a
    // down engine is data, not an error — no retry storm.
    retry: 0,
  });
}
