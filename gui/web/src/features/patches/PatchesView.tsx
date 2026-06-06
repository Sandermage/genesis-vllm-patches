// SPDX-License-Identifier: Apache-2.0
/**
 * PatchesView — inventory of every patch known to the platform.
 *
 * Two regions:
 *   1. ``InventoryCard`` — aggregate counters (total / active / retired /
 *      enabled-now) sourced from /api/v1/patches/inventory.
 *   2. ``PatchTable`` — filterable Carbon DataTable sourced from
 *      /api/v1/patches with family / tier / lifecycle / enabled-only filters.
 *
 * Filter state lives in component state (URL sync deferred until React Router
 * lands in Phase 12). Carbon DataTable handles sort and column resize natively.
 */
import { useEffect, useState } from 'react';
import {
  DataTable, Table, TableHead, TableHeader, TableRow,
  TableBody, TableCell, Tag, InlineNotification,
  DataTableSkeleton, Tile, Dropdown, Toggle,
} from '@carbon/react';
import {
  getPatchInventory,
  listPatches,
  type PatchInventoryReport,
  type PatchLifecycle,
  type PatchSummary,
  type PatchTier,
} from './api';

const LIFECYCLE_TAG: Record<PatchLifecycle, 'green' | 'blue' | 'gray' | 'red'> = {
  active: 'green',
  experimental: 'blue',
  deprecated: 'gray',
  retired: 'red',
};

const TIER_TAG: Record<PatchTier, 'purple' | 'cyan'> = {
  community: 'cyan',
  engine: 'purple',
};

const HEADERS = [
  { key: 'id', header: 'ID' },
  { key: 'title', header: 'Title' },
  { key: 'family', header: 'Family' },
  { key: 'tier', header: 'Tier' },
  { key: 'lifecycle', header: 'Lifecycle' },
  { key: 'default_on', header: 'Default' },
  { key: 'enabled_now', header: 'Enabled' },
] as const;

interface Filters {
  family: string | undefined;
  tier: PatchTier | undefined;
  lifecycle: PatchLifecycle | undefined;
  enabledOnly: boolean;
}

const INITIAL_FILTERS: Filters = {
  family: undefined,
  tier: undefined,
  lifecycle: undefined,
  enabledOnly: false,
};

function InventoryCard({ report }: { report: PatchInventoryReport | null }): JSX.Element {
  if (!report) {
    return <Tile className="patches-inventory-card">Loading inventory...</Tile>;
  }
  return (
    <Tile className="patches-inventory-card">
      <h3 className="cds--type-heading-03">Inventory</h3>
      <dl className="patches-inventory-stats">
        <div><dt>Total</dt><dd>{report.total}</dd></div>
        <div><dt>Active</dt><dd>{report.active}</dd></div>
        <div><dt>Retired</dt><dd>{report.retired}</dd></div>
        <div><dt>Enabled now</dt><dd>{report.enabled_now}</dd></div>
        <div><dt>Engine tier</dt><dd>{report.by_tier.engine ?? 0}</dd></div>
      </dl>
    </Tile>
  );
}

interface PatchTableProps {
  patches: PatchSummary[] | null;
  error: string | null;
}

function PatchTable({ patches, error }: PatchTableProps): JSX.Element {
  if (error) {
    return (
      <InlineNotification
        kind="error"
        title="Failed to load patches"
        subtitle={error}
        hideCloseButton
      />
    );
  }

  if (!patches) {
    return <DataTableSkeleton headers={HEADERS.slice()} rowCount={6} />;
  }

  if (patches.length === 0) {
    return (
      <InlineNotification
        kind="info"
        title="No patches match the current filters"
        hideCloseButton
      />
    );
  }

  const rows = patches.map((p) => ({
    id: p.id,
    title: p.title,
    family: p.family,
    tier: <Tag type={TIER_TAG[p.tier]}>{p.tier}</Tag>,
    lifecycle: <Tag type={LIFECYCLE_TAG[p.lifecycle]}>{p.lifecycle}</Tag>,
    default_on: p.default_on ? 'yes' : 'no',
    enabled_now: p.enabled_now ? <Tag type="green">on</Tag> : <Tag type="gray">off</Tag>,
  }));

  return (
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
  );
}

interface FilterBarProps {
  filters: Filters;
  families: string[];
  onChange: (next: Filters) => void;
}

function FilterBar({ filters, families, onChange }: FilterBarProps): JSX.Element {
  const familyItems = ['(all)', ...families];
  const tierItems: Array<PatchTier | '(all)'> = ['(all)', 'community', 'engine'];
  const lifecycleItems: Array<PatchLifecycle | '(all)'> = [
    '(all)', 'active', 'experimental', 'deprecated', 'retired',
  ];

  return (
    <div className="patches-filter-bar">
      <Dropdown
        id="patches-filter-family"
        titleText="Family"
        label="(all)"
        items={familyItems}
        selectedItem={filters.family ?? '(all)'}
        onChange={({ selectedItem }) => onChange({
          ...filters,
          family: selectedItem === '(all)' ? undefined : (selectedItem ?? undefined),
        })}
      />
      <Dropdown
        id="patches-filter-tier"
        titleText="Tier"
        label="(all)"
        items={tierItems}
        selectedItem={filters.tier ?? '(all)'}
        onChange={({ selectedItem }) => onChange({
          ...filters,
          tier: selectedItem === '(all)' ? undefined : (selectedItem as PatchTier),
        })}
      />
      <Dropdown
        id="patches-filter-lifecycle"
        titleText="Lifecycle"
        label="(all)"
        items={lifecycleItems}
        selectedItem={filters.lifecycle ?? '(all)'}
        onChange={({ selectedItem }) => onChange({
          ...filters,
          lifecycle: selectedItem === '(all)' ? undefined : (selectedItem as PatchLifecycle),
        })}
      />
      <Toggle
        id="patches-filter-enabled-only"
        labelText="Enabled in live env only"
        toggled={filters.enabledOnly}
        onToggle={(checked: boolean) => onChange({ ...filters, enabledOnly: checked })}
      />
    </div>
  );
}

export function PatchesView(): JSX.Element {
  const [filters, setFilters] = useState<Filters>(INITIAL_FILTERS);
  const [patches, setPatches] = useState<PatchSummary[] | null>(null);
  const [inventory, setInventory] = useState<PatchInventoryReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getPatchInventory()
      .then((report) => { if (!cancelled) setInventory(report); })
      .catch((e) => { if (!cancelled) setError(String(e)); });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    let cancelled = false;
    setPatches(null);
    setError(null);
    listPatches({
      family: filters.family,
      tier: filters.tier,
      lifecycle: filters.lifecycle,
      enabledOnly: filters.enabledOnly,
    })
      .then((rows) => { if (!cancelled) setPatches(rows); })
      .catch((e) => { if (!cancelled) setError(String(e)); });
    return () => { cancelled = true; };
  }, [filters]);

  const families = inventory ? Object.keys(inventory.by_family).sort() : [];

  return (
    <div className="patches-view">
      <h2 className="cds--type-heading-04">Patches</h2>
      <InventoryCard report={inventory} />
      <FilterBar filters={filters} families={families} onChange={setFilters} />
      <PatchTable patches={patches} error={error} />
    </div>
  );
}

export default PatchesView;
