// SPDX-License-Identifier: Apache-2.0
import { useState } from 'react';
import {
  DataTable, Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
  Tabs, TabList, Tab, TabPanels, TabPanel, Tile,
} from '@carbon/react';
import { DataView } from '@/components/DataView';
import { getConfigCatalog, type ConfigCatalog } from './api';

const MODEL_HEADERS = [
  { key: 'id', header: 'ID' }, { key: 'family', header: 'Family' },
  { key: 'served_model_name', header: 'Served name' },
  { key: 'quant_format', header: 'Quant' },
  { key: 'kv_cache_dtype', header: 'KV dtype' },
  { key: 'spec_method', header: 'Spec' },
];

const HW_HEADERS = [
  { key: 'id', header: 'ID' }, { key: 'gpu', header: 'GPU' },
  { key: 'gpu_count', header: '#' }, { key: 'vram_per_gpu_gib', header: 'VRAM/GPU' },
  { key: 'cpu_cores', header: 'Cores' }, { key: 'ram_gib', header: 'RAM' },
];

const PROFILE_HEADERS = [
  { key: 'id', header: 'ID' }, { key: 'parent_model', header: 'Parent model' },
  { key: 'role', header: 'Role' },
];

const PRESET_HEADERS = [
  { key: 'id', header: 'ID' }, { key: 'parent_model', header: 'Parent' },
  { key: 'composed_key', header: 'Composed key' },
];

function renderTable(rows: any[], headers: typeof MODEL_HEADERS) {
  return (
    <DataTable rows={rows} headers={headers}>
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
                  <TableCell key={cell.id}>{cell.value ?? '—'}</TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </DataTable>
  );
}

export function ConfigsView(): JSX.Element {
  return (
    <div className="configs-view">
      <h2 className="cds--type-heading-04">Configs</h2>
      <DataView<ConfigCatalog> load={getConfigCatalog} errorTitle="Failed to load config catalog">
        {(catalog) => (
          <>
            <Tile>
              <p className="cds--type-helper-text-01">
                Models: {catalog.models.length} · Hardware: {catalog.hardware.length} ·
                Profiles: {catalog.profiles.length} · Presets: {catalog.presets.length}
              </p>
            </Tile>
            <Tabs>
              <TabList aria-label="Config catalog tabs">
                <Tab>Models</Tab>
                <Tab>Hardware</Tab>
                <Tab>Profiles</Tab>
                <Tab>Presets</Tab>
              </TabList>
              <TabPanels>
                <TabPanel>
                  {renderTable(catalog.models.map((m) => ({ ...m, id: m.id })), MODEL_HEADERS)}
                </TabPanel>
                <TabPanel>
                  {renderTable(catalog.hardware.map((h) => ({ ...h, id: h.id })), HW_HEADERS)}
                </TabPanel>
                <TabPanel>
                  {renderTable(catalog.profiles.map((p) => ({ ...p, id: p.id })), PROFILE_HEADERS)}
                </TabPanel>
                <TabPanel>
                  {renderTable(catalog.presets.map((p) => ({ ...p, id: p.id })), PRESET_HEADERS)}
                </TabPanel>
              </TabPanels>
            </Tabs>
          </>
        )}
      </DataView>
    </div>
  );
}

export default ConfigsView;
