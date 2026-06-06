// SPDX-License-Identifier: Apache-2.0
import { Tag, Tile, StructuredListWrapper, StructuredListHead, StructuredListBody, StructuredListRow, StructuredListCell } from '@carbon/react';
import { DataView } from '@/components/DataView';
import { runDoctor, type DoctorReport, type DoctorSeverity } from './api';

const SEV_TAG: Record<DoctorSeverity, 'green' | 'red' | 'magenta' | 'gray'> = {
  info: 'green', warning: 'magenta', error: 'red', critical: 'red',
};

export function DoctorView(): JSX.Element {
  return (
    <div className="doctor-view">
      <h2 className="cds--type-heading-04">Doctor</h2>
      <DataView<DoctorReport>
        load={runDoctor}
        errorTitle="Failed to run doctor"
        isEmpty={(d) => d.findings.length === 0}
        emptyTitle="No findings (everything green)"
      >
        {(report) => (
          <>
            <Tile>
              <p className="cds--type-body-01">
                {report.ok
                  ? <Tag type="green">All healthy</Tag>
                  : <Tag type="red">Issues found</Tag>}
                {' '} · Checked {new Date(report.checked_at).toLocaleString()}
              </p>
              <p className="cds--type-helper-text-01" style={{ marginTop: 8 }}>
                info: {report.counts.info ?? 0} ·
                warning: {report.counts.warning ?? 0} ·
                error: {report.counts.error ?? 0} ·
                critical: {report.counts.critical ?? 0}
              </p>
            </Tile>
            <StructuredListWrapper ariaLabel="Doctor findings">
              <StructuredListHead>
                <StructuredListRow head>
                  <StructuredListCell head>Severity</StructuredListCell>
                  <StructuredListCell head>Category</StructuredListCell>
                  <StructuredListCell head>Title</StructuredListCell>
                  <StructuredListCell head>Remediation</StructuredListCell>
                </StructuredListRow>
              </StructuredListHead>
              <StructuredListBody>
                {report.findings.map((f, i) => (
                  <StructuredListRow key={i}>
                    <StructuredListCell><Tag type={SEV_TAG[f.severity]}>{f.severity}</Tag></StructuredListCell>
                    <StructuredListCell>{f.category}</StructuredListCell>
                    <StructuredListCell>
                      <strong>{f.title}</strong>
                      {f.detail && <p className="cds--type-helper-text-01">{f.detail}</p>}
                    </StructuredListCell>
                    <StructuredListCell>{f.remediation ?? '—'}</StructuredListCell>
                  </StructuredListRow>
                ))}
              </StructuredListBody>
            </StructuredListWrapper>
          </>
        )}
      </DataView>
    </div>
  );
}

export default DoctorView;
