// SPDX-License-Identifier: Apache-2.0
import {
  DataTable, Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
  Tag, ProgressBar,
} from '@carbon/react';
import { DataView } from '@/components/DataView';
import { listJobs, type JobState, type JobSummary } from './api';

const STATE_TAG: Record<JobState, 'green' | 'red' | 'gray' | 'blue' | 'magenta'> = {
  queued: 'gray', running: 'blue', succeeded: 'green', failed: 'red', canceled: 'magenta',
};

const HEADERS = [
  { key: 'id', header: 'ID' }, { key: 'kind', header: 'Kind' },
  { key: 'state', header: 'State' }, { key: 'progress', header: 'Progress' },
  { key: 'started', header: 'Started' }, { key: 'summary', header: 'Summary' },
];

export function JobsView(): JSX.Element {
  return (
    <div className="jobs-view">
      <h2 className="cds--type-heading-04">Jobs</h2>
      <DataView<JobSummary[]>
        load={() => listJobs()}
        isEmpty={(d) => d.length === 0}
        emptyTitle="No jobs"
        errorTitle="Failed to load jobs"
        skeletonHeaders={HEADERS}
      >
        {(jobs) => {
          const rows = jobs.map((j) => ({
            id: j.id.slice(0, 8),
            kind: j.kind,
            state: <Tag type={STATE_TAG[j.state]}>{j.state}</Tag>,
            progress: <ProgressBar value={j.progress_pct} max={100} hideLabel size="small" />,
            started: j.started_at ? new Date(j.started_at).toLocaleString() : '—',
            summary: j.summary ?? '—',
          }));
          return (
            <DataTable rows={rows} headers={HEADERS}>
              {({ rows: dtRows, headers: dtHeaders, getHeaderProps, getRowProps, getTableProps }) => (
                <Table {...getTableProps()} size="md">
                  <TableHead>
                    <TableRow>
                      {dtHeaders.map((h: any) => (
                        <TableHeader {...getHeaderProps({ header: h })}>{h.header}</TableHeader>
                      ))}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {dtRows.map((row: any) => (
                      <TableRow {...getRowProps({ row })}>
                        {row.cells.map((cell: any) => (
                          <TableCell key={cell.id}>{cell.value}</TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </DataTable>
          );
        }}
      </DataView>
    </div>
  );
}

export default JobsView;
