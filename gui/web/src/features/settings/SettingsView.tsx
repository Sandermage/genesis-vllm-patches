// SPDX-License-Identifier: Apache-2.0
import { useState } from 'react';
import {
  Dropdown, Toggle, Tile, TextInput, Button, InlineNotification,
  StructuredListWrapper, StructuredListHead, StructuredListBody,
  StructuredListRow, StructuredListCell,
} from '@carbon/react';
import { SUPPORTED_LOCALES, type LocaleCode, setLocale } from '@/i18n';
import { useEngineStore } from '@/stores/engine';

export function SettingsView(): JSX.Element {
  const engine = useEngineStore((s) => s.selected);
  const setEngine = useEngineStore((s) => s.setSelected);
  const [locale, setLocaleState] = useState<LocaleCode>(
    (localStorage.getItem('sndr-locale') as LocaleCode) ?? 'en'
  );
  const [apiBase, setApiBase] = useState<string>(
    localStorage.getItem('sndr-api-base') ?? ''
  );
  const [saved, setSaved] = useState(false);

  const handleSaveApiBase = () => {
    localStorage.setItem('sndr-api-base', apiBase);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div className="settings-view">
      <h2 className="cds--type-heading-04">Settings</h2>
      <Tile style={{ marginBottom: 16 }}>
        <h3 className="cds--type-heading-03">Locale</h3>
        <Dropdown
          id="settings-locale"
          titleText="Interface language"
          items={Object.entries(SUPPORTED_LOCALES).map(([code, info]) => ({
            id: code, label: `${info.flag} ${info.label}`, value: code,
          }))}
          selectedItem={{
            id: locale,
            label: `${SUPPORTED_LOCALES[locale].flag} ${SUPPORTED_LOCALES[locale].label}`,
            value: locale,
          }}
          onChange={({ selectedItem }) => {
            if (selectedItem) {
              const code = selectedItem.value as LocaleCode;
              setLocaleState(code);
              setLocale(code);
            }
          }}
        />
      </Tile>

      <Tile style={{ marginBottom: 16 }}>
        <h3 className="cds--type-heading-03">Active engine</h3>
        <Dropdown
          id="settings-engine"
          titleText="Engine"
          items={['vllm', 'sglang']}
          selectedItem={engine}
          onChange={({ selectedItem }) => selectedItem && setEngine(selectedItem)}
        />
      </Tile>

      <Tile style={{ marginBottom: 16 }}>
        <h3 className="cds--type-heading-03">API endpoint</h3>
        <TextInput
          id="settings-api-base"
          labelText="Override API base URL (empty = same-origin)"
          placeholder="https://sndr.lab.example.com"
          value={apiBase}
          onChange={(e) => setApiBase(e.target.value)}
        />
        <div style={{ marginTop: 8 }}>
          <Button kind="primary" size="md" onClick={handleSaveApiBase}>
            Save
          </Button>
        </div>
        {saved && (
          <InlineNotification kind="success" title="Saved" subtitle="Refresh to apply." hideCloseButton />
        )}
      </Tile>

      <Tile>
        <h3 className="cds--type-heading-03">About</h3>
        <StructuredListWrapper ariaLabel="About">
          <StructuredListBody>
            <StructuredListRow>
              <StructuredListCell>Version</StructuredListCell>
              <StructuredListCell>sndr-platform 12.0.0.dev0</StructuredListCell>
            </StructuredListRow>
            <StructuredListRow>
              <StructuredListCell>Theme</StructuredListCell>
              <StructuredListCell>Carbon g100</StructuredListCell>
            </StructuredListRow>
            <StructuredListRow>
              <StructuredListCell>i18n</StructuredListCell>
              <StructuredListCell>Lingui (en, ru)</StructuredListCell>
            </StructuredListRow>
          </StructuredListBody>
        </StructuredListWrapper>
      </Tile>
    </div>
  );
}

export default SettingsView;
