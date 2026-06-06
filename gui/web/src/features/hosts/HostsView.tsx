// SPDX-License-Identifier: Apache-2.0
import { useEffect, useState } from 'react';
import {
  DataTable, Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
  Tag, Tile, InlineNotification, DataTableSkeleton,
} from '@carbon/react';
import { listHosts, type HostSummary, type HostStatus } from './api';

const STATUS_TAG: Record<HostStatus, 'green' | 'red' | 'gray' | 'magenta'> = {
  online: 'green', offline: 'red', degraded: 'magenta', unknown: 'gray',
};

const HEADERS = [
  { key: 'hostname', header: 'Hostname' },
  { key: 'status', header: 'Status' },
  { key: 'sndr_version', header: 'sndr' },
  { key: 'engine', header: 'Engine' },
  { key: 'gpus', header: 'GPUs' },
  { key: 'ram', header: 'RAM' },
] as const;

export function HostsView(): JSX.Element {
  const [hosts, setHosts] = useState<HostSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    listHosts()
      .then((rows) => { if (!cancelled) setHosts(rows); })
      .catch((e) => { if (!cancelled) setError(String(e)); });
    return () => { cancelled = true; };
  }, []);

  if (error) return <InlineNotification kind="error" title="Failed to load hosts" subtitle={error} hideCloseButton />;
  if (!hosts) return <DataTableSkeleton headers={HEADERS.slice()} rowCount={4} />;
  if (hosts.length === 0) {
    return <InlineNotification kind="info" title="No hosts in fleet" hideCloseButton />;
  }

  const rows = hosts.map((h) => ({
    id: h.hostname,
    hostname: h.hostname,
    status: <Tag type={STATUS_TAG[h.status] ?? 'gray'}>{h.status}</Tag>,
    sndr_version: h.sndr_version ?? '—',
    engine: h.active_engine ? `${h.active_engine} (${h.active_engine_pin ?? 'auto'})` : '—',
    gpus: h.hardware?.gpus.length ?? 0,
    ram: h.hardware ? `${h.hardware.ram_available_gib}/${h.hardware.ram_total_gib} GiB` : '—',
  }));

  return (
    <div className="hosts-view">
      <h2 className="cds--type-heading-04">Hosts</h2>
      <Tile className="hosts-summary">
        <p>Total: {hosts.length} · Online: {hosts.filter(h => h.status === 'online').length}</p>
      </Tile>
      <DataTable rows={rows} headers={HEADERS.slice()}>
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
    </div>
  );
}

export default HostsView;
