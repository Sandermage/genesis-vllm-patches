// SPDX-License-Identifier: Apache-2.0
/** Licensing API — typed wrappers over /api/v1/licensing. */
import { apiClient } from '@/api/client';

export type LicenseTierStatus =
  | 'licensed'
  | 'licensed_legacy'
  | 'expired'
  | 'bad_signature'
  | 'version_mismatch'
  | 'no_key'
  | 'no_package'
  | 'unknown';

export interface LicenseStatus {
  status: LicenseTierStatus;
  customer_id_hash: string | null;
  expires_at: string | null;
  days_until_expiry: number | null;
  engine_major: number | null;
  engine_package_installed: boolean;
  engine_patches_available: number;
  message: string | null;
}

export async function getLicenseStatus(): Promise<LicenseStatus> {
  const response = await apiClient.get<LicenseStatus>('/api/v1/licensing/status');
  return response.data;
}
