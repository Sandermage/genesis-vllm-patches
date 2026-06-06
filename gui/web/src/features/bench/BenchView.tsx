// SPDX-License-Identifier: Apache-2.0
import {
  DataTable, Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
  Tag,
} from '@carbon/react';
import { DataView } from '@/components/DataView';
import { listBenchRuns, type BenchOutcome, type BenchSummary } from './api';

const OUTCOME_TAG: Record<BenchOutcome, 'green' | 'red' | 'magenta' | 'gray' | 'blue'> = {
  success: 'green', regression: 'red', failed: 'red',
  pending: 'gray', running: 'blue',
};

const HEADERS = [
  { key: 'timestamp', header: 'When' },
  { key: 'model', header: 'Model' },
  { key: 'pin', header: 'Pin' },
  { key: 'wall_tps', header: 'wall TPS' },
  { key: 'tpot', header: 'TPOT ms' },
  { key: 'ttft', header: 'TTFT ms' },
  { key: 'cv', header: 'CV' },
  { key: 'outcome', header: 'Outcome' },
];

export function BenchView(): JSX.Element {
  return (
    <div className="bench-view">
      <h2 className="cds--type-heading-04">Bench</h2>
      <DataView<BenchSummary[]>
        load={() => listBenchRuns()}
        isEmpty={(d) => d.length === 0}
        emptyTitle="No bench runs yet"
        errorTitle="Failed to load bench runs"
        skeletonHeaders={HEADERS}
      >
        {(runs) => {
          const rows = runs.map((r) => ({
            id: r.id,
            timestamp: new Date(r.timestamp).toLocaleString(),
            model: r.model,
            pin: r.pin,
            wall_tps: r.wall_tps.toFixed(2),
            tpot: r.decode_tpot_ms.toFixed(2),
            ttft: r.ttft_ms.toFixed(0),
            cv: (r.cv * 100).toFixed(1) + '%',
            outcome: <Tag type={OUTCOME_TAG[r.outcome] ?? 'gray'}>{r.outcome}</Tag>,
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

export default BenchView;
