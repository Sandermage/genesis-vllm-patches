// SPDX-License-Identifier: Apache-2.0
import {
  DataTable, Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
  Tag, Tile,
} from '@carbon/react';
import { DataView } from '@/components/DataView';
import { getEvidenceReport, type EvidenceReport, type EvidenceStatus } from './api';

const STATUS_TAG: Record<EvidenceStatus, 'green' | 'red' | 'magenta' | 'gray'> = {
  ok: 'green', warning: 'magenta', fail: 'red', skipped: 'gray',
};

const HEADERS = [
  { key: 'id', header: 'ID' },
  { key: 'name', header: 'Gate' },
  { key: 'status', header: 'Status' },
  { key: 'summary', header: 'Summary' },
  { key: 'last_run', header: 'Last run' },
];

export function EvidenceView(): JSX.Element {
  return (
    <div className="evidence-view">
      <h2 className="cds--type-heading-04">Evidence</h2>
      <DataView<EvidenceReport>
        load={getEvidenceReport}
        errorTitle="Failed to load evidence report"
        skeletonHeaders={HEADERS}
      >
        {(report) => {
          const rows = report.gates.map((g) => ({
            id: g.id,
            name: g.name,
            status: <Tag type={STATUS_TAG[g.status]}>{g.status}</Tag>,
            summary: g.summary ?? '—',
            last_run: g.last_run ? new Date(g.last_run).toLocaleString() : '—',
          }));
          return (
            <>
              <Tile>
                <p className="cds--type-helper-text-01">Release-readiness</p>
                <p className="cds--type-heading-05">
                  {report.gates_ok} / {report.gates_total} OK ·{' '}
                  {report.gates_fail > 0 && <Tag type="red">{report.gates_fail} failing</Tag>}{' '}
                  {report.gates_warning > 0 && <Tag type="magenta">{report.gates_warning} warnings</Tag>}{' '}
                  {report.gates_skipped > 0 && <Tag type="gray">{report.gates_skipped} skipped</Tag>}
                </p>
              </Tile>
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
            </>
          );
        }}
      </DataView>
    </div>
  );
}

export default EvidenceView;
