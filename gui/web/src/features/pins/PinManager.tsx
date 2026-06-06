// SPDX-License-Identifier: Apache-2.0
/**
 * PinManager — top-level view for managing engine pins.
 *
 * Shows current/previous/staging pins per engine with manifest and drift status.
 */
import { useEffect, useState } from 'react';
import {
  DataTable, Table, TableHead, TableHeader, TableRow,
  TableBody, TableCell, Tag, InlineNotification,
  DataTableSkeleton,
} from '@carbon/react';
import { useEngineStore } from '@/stores/engine';
import { listPins, type PinSummary } from './api';

const STATUS_TAG_TYPE: Record<string, 'green' | 'blue' | 'gray' | 'purple' | 'red'> = {
  current: 'green',
  previous: 'blue',
  staging: 'purple',
  deprecated: 'gray',
  retired: 'red',
};

const headers = [
  { key: 'pin', header: 'Pin' },
  { key: 'status', header: 'Status' },
  { key: 'full_version', header: 'Full version' },
  { key: 'has_manifest', header: 'Manifest' },
  { key: 'has_drift', header: 'Drift' },
  { key: 'bench_tps_last', header: 'Bench (TPS)' },
] as const;

export function PinManager(): JSX.Element {
  const engine = useEngineStore((s) => s.selected);
  const [pins, setPins] = useState<PinSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setPins(null);
    setError(null);
    listPins(engine)
      .then((p) => { if (!cancelled) setPins(p); })
      .catch((e) => { if (!cancelled) setError(String(e)); });
    return () => { cancelled = true; };
  }, [engine]);

  if (error) {
    return (
      <InlineNotification
        kind="error"
        title="Failed to load pins"
        subtitle={error}
        hideCloseButton
      />
    );
  }

  if (!pins) {
    return <DataTableSkeleton headers={headers.slice()} rowCount={3} />;
  }

  if (pins.length === 0) {
    return (
      <InlineNotification
        kind="info"
        title="No pins yet"
        subtitle={`No pin manifests have been generated for ${engine}. Use 'sndr manifest generate' to create one.`}
        hideCloseButton
      />
    );
  }

  const rows = pins.map((p) => ({
    id: p.pin,
    ...p,
    status: <Tag type={STATUS_TAG_TYPE[p.status] ?? 'gray'}>{p.status}</Tag>,
    has_manifest: p.has_manifest ? '✓' : '—',
    has_drift: p.has_drift
      ? <Tag type="red">drift</Tag>
      : <Tag type="green">ok</Tag>,
    bench_tps_last: p.bench_tps_last ?? '—',
  }));

  return (
    <DataTable rows={rows} headers={headers.slice()}>
      {({ rows: dtRows, headers: dtHeaders, getHeaderProps, getRowProps, getTableProps }) => (
        <Table {...getTableProps()}>
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
}
