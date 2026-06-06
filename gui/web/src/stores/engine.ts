// SPDX-License-Identifier: Apache-2.0
/**
 * Engine selection store.
 *
 * Tracks which engine the operator is currently viewing in the Control Center.
 * Most resource views (patches, pins, drift, containers) are scoped to one
 * engine. The selection persists across navigation via Zustand.
 */
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type EngineName = 'vllm' | 'sglang';

interface EngineStore {
  /** Currently-selected engine. */
  selected: EngineName;

  /** Currently-selected pin within the engine. */
  pin: string | null;

  /** List of engines available for selection (loaded from API). */
  available: Array<{ name: EngineName; displayName: string; active: boolean }>;

  setEngine: (engine: EngineName) => void;
  setPin: (pin: string | null) => void;
  setAvailable: (engines: Array<{ name: EngineName; displayName: string; active: boolean }>) => void;
}

export const useEngineStore = create<EngineStore>()(
  persist(
    (set) => ({
      selected: 'vllm',
      pin: null,
      available: [
        { name: 'vllm', displayName: 'vLLM', active: true },
        { name: 'sglang', displayName: 'SGLang', active: false },
      ],
      setEngine: (engine) => set({ selected: engine, pin: null }),
      setPin: (pin) => set({ pin }),
      setAvailable: (engines) => set({ available: engines }),
    }),
    {
      name: 'sndr-engine-store',
      version: 1,
    },
  ),
);
