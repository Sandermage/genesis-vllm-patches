// SPDX-License-Identifier: Apache-2.0
/**
 * Overview aggregates data from multiple endpoints into a single view —
 * the operator-facing "what's the world look like" landing page.
 */
import { apiClient } from '@/api/client';
import type { FleetReport } from '@/features/fleet/api';
import type { PatchInventoryReport } from '@/features/patches/api';
import type { ContainerInventoryReport } from '@/features/containers/api';
import type { DoctorReport } from '@/features/doctor/api';
import type { EvidenceReport } from '@/features/evidence/api';

export interface OverviewSnapshot {
  fleet: FleetReport;
  patches: PatchInventoryReport;
  containers: ContainerInventoryReport;
  doctor: DoctorReport;
  evidence: EvidenceReport;
  api_version: string;
}

export async function getOverview(): Promise<OverviewSnapshot> {
  // Parallel fan-out: 5 calls, no sequential blocking.
  const [fleet, patches, containers, doctor, evidence] = await Promise.all([
    apiClient.get<FleetReport>('/api/v1/fleet'),
    apiClient.get<PatchInventoryReport>('/api/v1/patches/inventory'),
    apiClient.get<ContainerInventoryReport>('/api/v1/containers/inventory'),
    apiClient.get<DoctorReport>('/api/v1/doctor'),
    apiClient.get<EvidenceReport>('/api/v1/evidence'),
  ]);
  return {
    fleet: fleet.data,
    patches: patches.data,
    containers: containers.data,
    doctor: doctor.data,
    evidence: evidence.data,
    api_version: fleet.meta.api_version,
  };
}
