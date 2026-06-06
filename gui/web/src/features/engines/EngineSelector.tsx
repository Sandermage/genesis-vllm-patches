// SPDX-License-Identifier: Apache-2.0
/**
 * EngineSelector — top-bar dropdown for choosing the active engine.
 *
 * Carbon Dropdown component, themed with sndr tokens. Disabled engines
 * (e.g. sglang skeleton) appear in the list but cannot be selected.
 */
import { useEffect } from 'react';
import { Dropdown } from '@carbon/react';
import { useEngineStore, type EngineName } from '@/stores/engine';
import { listEngines } from './api';

interface DropdownItem {
  id: EngineName;
  label: string;
  active: boolean;
  disabled: boolean;
}

export function EngineSelector(): JSX.Element {
  const selected = useEngineStore((s) => s.selected);
  const available = useEngineStore((s) => s.available);
  const setEngine = useEngineStore((s) => s.setEngine);
  const setAvailable = useEngineStore((s) => s.setAvailable);

  useEffect(() => {
    // Sync available engines from API on mount.
    listEngines()
      .then((engines) => {
        setAvailable(
          engines
            .filter((e): e is typeof e & { name: EngineName } =>
              e.name === 'vllm' || e.name === 'sglang',
            )
            .map((e) => ({
              name: e.name,
              displayName: e.display_name,
              active: e.active,
            })),
        );
      })
      .catch(() => {
        // Fail silently — fall back to defaults from the store.
      });
  }, [setAvailable]);

  const items: DropdownItem[] = available.map((e) => ({
    id: e.name,
    label: e.displayName + (!e.active ? ' (unavailable)' : ''),
    active: e.active,
    disabled: !e.active,
  }));

  const selectedItem = items.find((i) => i.id === selected) ?? items[0];

  return (
    <Dropdown
      id="sndr-engine-selector"
      titleText="Engine"
      label="Select an engine"
      items={items}
      itemToString={(item: DropdownItem | null) => item?.label ?? ''}
      selectedItem={selectedItem}
      onChange={(evt: { selectedItem: DropdownItem | null }) => {
        if (evt.selectedItem && !evt.selectedItem.disabled) {
          setEngine(evt.selectedItem.id);
        }
      }}
      size="md"
    />
  );
}
