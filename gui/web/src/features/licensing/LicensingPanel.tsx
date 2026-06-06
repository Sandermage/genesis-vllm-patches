// SPDX-License-Identifier: Apache-2.0
/**
 * LicensingPanel — surfaces the current license tier status in the GUI.
 *
 * The component reads /api/v1/licensing/status (which carries only metadata,
 * never the token itself) and renders a Carbon notification + Tile with the
 * current state.
 */
import { useEffect, useState } from 'react';
import { InlineNotification, Tile, Tag, SkeletonText } from '@carbon/react';
import { getLicenseStatus, type LicenseStatus } from './api';

const STATUS_KIND: Record<string, 'success' | 'warning' | 'error' | 'info'> = {
  licensed: 'success',
  licensed_legacy: 'warning',
  expired: 'error',
  bad_signature: 'error',
  version_mismatch: 'error',
  no_key: 'info',
  no_package: 'info',
  unknown: 'info',
};

const STATUS_LABEL: Record<string, string> = {
  licensed: 'Licensed',
  licensed_legacy: 'Licensed (legacy signature)',
  expired: 'License expired',
  bad_signature: 'Invalid signature',
  version_mismatch: 'Version mismatch',
  no_key: 'No license key',
  no_package: 'Engine package not installed',
  unknown: 'Unknown',
};

export function LicensingPanel(): JSX.Element {
  const [status, setStatus] = useState<LicenseStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getLicenseStatus()
      .then((s) => {
        if (!cancelled) setStatus(s);
      })
      .catch((e) => {
        if (!cancelled) setError(String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (error) {
    return (
      <InlineNotification
        kind="error"
        title="License status unavailable"
        subtitle={error}
        hideCloseButton
      />
    );
  }

  if (!status) {
    return <SkeletonText paragraph lineCount={4} />;
  }

  const kind = STATUS_KIND[status.status] ?? 'info';
  const label = STATUS_LABEL[status.status] ?? status.status;

  return (
    <Tile>
      <h3>License status</h3>
      <InlineNotification
        kind={kind}
        title={label}
        subtitle={status.message ?? ''}
        hideCloseButton
        lowContrast
      />
      <dl className="sndr-license-detail">
        <dt>Engine package</dt>
        <dd>
          <Tag type={status.engine_package_installed ? 'green' : 'gray'}>
            {status.engine_package_installed ? 'installed' : 'not installed'}
          </Tag>
        </dd>
        <dt>Engine-tier patches available</dt>
        <dd>{status.engine_patches_available}</dd>
        {status.expires_at && (
          <>
            <dt>Expires</dt>
            <dd>
              {status.expires_at}
              {status.days_until_expiry !== null && (
                <> ({status.days_until_expiry} days)</>
              )}
            </dd>
          </>
        )}
        {status.customer_id_hash && (
          <>
            <dt>Customer (hash)</dt>
            <dd><code>{status.customer_id_hash}</code></dd>
          </>
        )}
      </dl>
    </Tile>
  );
}
