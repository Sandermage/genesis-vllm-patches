// SPDX-License-Identifier: Apache-2.0
import { useState } from 'react';
import {
  Tile, TextInput, Button, InlineNotification,
  StructuredListWrapper, StructuredListHead, StructuredListBody,
  StructuredListRow, StructuredListCell,
} from '@carbon/react';

/**
 * AuthView — operator-side API token + session management.
 *
 * v12.0 stub-real: stores a single bearer token in localStorage. The
 * full implementation in v12.1 will support multiple named tokens,
 * scopes, and rotation reminders.
 */
export function AuthView(): JSX.Element {
  const [token, setToken] = useState<string>(
    localStorage.getItem('sndr-api-token') ?? ''
  );
  const [saved, setSaved] = useState(false);

  const handleSave = () => {
    if (token) {
      localStorage.setItem('sndr-api-token', token);
    } else {
      localStorage.removeItem('sndr-api-token');
    }
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const handleClear = () => {
    setToken('');
    localStorage.removeItem('sndr-api-token');
  };

  const isAuthenticated = Boolean(localStorage.getItem('sndr-api-token'));

  return (
    <div className="auth-view">
      <h2 className="cds--type-heading-04">Auth</h2>
      <Tile style={{ marginBottom: 16 }}>
        <h3 className="cds--type-heading-03">Session status</h3>
        <StructuredListWrapper ariaLabel="Session">
          <StructuredListBody>
            <StructuredListRow>
              <StructuredListCell>Authenticated</StructuredListCell>
              <StructuredListCell>{isAuthenticated ? '✓ yes' : '— no'}</StructuredListCell>
            </StructuredListRow>
            <StructuredListRow>
              <StructuredListCell>Token preview</StructuredListCell>
              <StructuredListCell>
                {token ? `${token.slice(0, 8)}…${token.slice(-4)}` : '—'}
              </StructuredListCell>
            </StructuredListRow>
          </StructuredListBody>
        </StructuredListWrapper>
      </Tile>

      <Tile>
        <h3 className="cds--type-heading-03">API token</h3>
        <TextInput
          id="auth-token"
          type="password"
          labelText="Bearer token for the API"
          placeholder="sndr_…"
          value={token}
          onChange={(e) => setToken(e.target.value)}
        />
        <div style={{ marginTop: 12, display: 'flex', gap: 8 }}>
          <Button kind="primary" size="md" onClick={handleSave}>Save</Button>
          <Button kind="danger--tertiary" size="md" onClick={handleClear}>Clear</Button>
        </div>
        {saved && (
          <InlineNotification kind="success" title="Saved" hideCloseButton />
        )}
      </Tile>
    </div>
  );
}

export default AuthView;
