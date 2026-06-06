// SPDX-License-Identifier: Apache-2.0
import {
  DataTable, Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
  Tag, Tile,
} from '@carbon/react';
import { DataView } from '@/components/DataView';
import { listContainers, type ContainerState, type ContainerSummary } from './api';

const STATE_TAG: Record<ContainerState, 'green' | 'red' | 'gray' | 'magenta' | 'cyan' | 'purple'> = {
  running: 'green', exited: 'red', restarting: 'magenta',
  paused: 'cyan', created: 'purple', dead: 'red', unknown: 'gray',
};

const HEADERS = [
  { key: 'name', header: 'Name' },
  { key: 'state', header: 'State' },
  { key: 'image', header: 'Image' },
  { key: 'model', header: 'Model' },
  { key: 'engine', header: 'Engine' },
  { key: 'ports', header: 'Ports' },
];

export function ContainersView(): JSX.Element {
  return (
    <div className="containers-view">
      <h2 className="cds--type-heading-04">Containers</h2>
      <DataView<ContainerSummary[]>
        load={listContainers}
        isEmpty={(d) => d.length === 0}
        emptyTitle="No containers running"
        errorTitle="Failed to load containers"
        skeletonHeaders={HEADERS}
      >
        {(containers) => {
          const rows = containers.map((c) => ({
            id: c.name,
            name: c.name,
            state: <Tag type={STATE_TAG[c.state] ?? 'gray'}>{c.state}</Tag>,
            image: c.image.split('/').pop() ?? c.image,
            model: c.served_model_name ?? '—',
            engine: c.engine ? `${c.engine} (${c.engine_pin ?? '?'})` : '—',
            ports: c.ports.map((p) => `${p.host_port}→${p.container_port}`).join(', ') || '—',
          }));
          return (
            <>
              <Tile className="containers-summary">
                <p>{containers.length} containers · {containers.filter(c => c.state === 'running').length} running</p>
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

export default ContainersView;
