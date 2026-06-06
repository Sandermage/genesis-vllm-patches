// SPDX-License-Identifier: Apache-2.0
import { Tile, Tag, ProgressBar } from '@carbon/react';
import { DataView } from '@/components/DataView';
import { getOverview, type OverviewSnapshot } from './api';

function MetricTile({ title, primary, sub }: { title: string; primary: string | number; sub?: string }) {
  return (
    <Tile>
      <p className="cds--type-helper-text-01">{title}</p>
      <p className="cds--type-heading-04">{primary}</p>
      {sub && <p className="cds--type-helper-text-01">{sub}</p>}
    </Tile>
  );
}

export function OverviewView(): JSX.Element {
  return (
    <div className="overview-view">
      <h2 className="cds--type-heading-04">Overview</h2>
      <DataView<OverviewSnapshot> load={getOverview} errorTitle="Failed to load overview">
        {(s) => (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12 }}>
            <MetricTile
              title="Fleet"
              primary={`${s.fleet.online} / ${s.fleet.total_hosts} online`}
              sub={`${s.fleet.total_gpus} GPUs · ${s.fleet.total_vram_gib} GiB VRAM`}
            />
            <MetricTile
              title="Containers"
              primary={`${s.containers.by_state.running ?? 0} running`}
              sub={`Total ${s.containers.total}`}
            />
            <MetricTile
              title="Patches"
              primary={`${s.patches.enabled_now} live`}
              sub={`${s.patches.total} in registry · ${s.patches.active} active`}
            />
            <MetricTile
              title="Doctor"
              primary={s.doctor.ok ? <><Tag type="green">OK</Tag></> as any : <><Tag type="red">Issues</Tag></> as any}
              sub={`${s.doctor.findings.length} findings`}
            />
            <MetricTile
              title="Evidence"
              primary={`${s.evidence.gates_ok} / ${s.evidence.gates_total} OK`}
              sub={`${s.evidence.gates_fail} failing · ${s.evidence.gates_warning} warnings`}
            />
            <Tile>
              <p className="cds--type-helper-text-01">API version</p>
              <p className="cds--type-heading-05">{s.api_version}</p>
            </Tile>
          </div>
        )}
      </DataView>
    </div>
  );
}

export default OverviewView;
