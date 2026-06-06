// SPDX-License-Identifier: Apache-2.0
import { Tile, Tag } from '@carbon/react';
import { DataView } from '@/components/DataView';
import { getFleetReport, type FleetReport } from './api';

function Stat({ label, value, tone }: { label: string; value: number; tone?: 'green' | 'red' | 'magenta' | 'gray' }) {
  return (
    <Tile className="fleet-stat">
      <p className="cds--type-helper-text-01">{label}</p>
      <p className="cds--type-heading-05">
        {tone ? <Tag type={tone}>{value}</Tag> : value}
      </p>
    </Tile>
  );
}

export function FleetView(): JSX.Element {
  return (
    <div className="fleet-view">
      <h2 className="cds--type-heading-04">Fleet</h2>
      <DataView<FleetReport> load={getFleetReport} errorTitle="Failed to load fleet">
        {(r) => (
          <div className="fleet-stats" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12 }}>
            <Stat label="Total hosts" value={r.total_hosts} />
            <Stat label="Online" value={r.online} tone="green" />
            <Stat label="Degraded" value={r.degraded} tone="magenta" />
            <Stat label="Offline" value={r.offline} tone="red" />
            <Stat label="Unknown" value={r.unknown} tone="gray" />
            <Stat label="GPUs" value={r.total_gpus} />
            <Stat label="VRAM (GiB)" value={r.total_vram_gib} />
          </div>
        )}
      </DataView>
    </div>
  );
}

export default FleetView;
