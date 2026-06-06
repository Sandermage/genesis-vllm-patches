// SPDX-License-Identifier: Apache-2.0
/**
 * CarbonApp — top-level shell of the Carbon-based Control Center.
 *
 * Layout:
 *   <Header>           — brand, engine selector, locale switcher
 *   <SideNav>          — feature navigation
 *   <main>             — routed feature view
 *
 * Side-nav items are grouped by domain:
 *   Overview / Live    — overview, fleet, hosts, containers
 *   Engines / Patches  — engines, pins, patches, drift
 *   Workloads          — bench, chat, jobs
 *   Health / Config    — doctor, evidence, configs
 *   Admin              — licensing, auth, settings
 */
import { useState, type ReactNode } from 'react';
import {
  BrowserRouter, Routes, Route, NavLink, Navigate, useLocation,
} from 'react-router-dom';
import {
  Header, HeaderName, HeaderGlobalAction, HeaderGlobalBar,
  SideNav, SideNavItems, SideNavLink, SideNavMenu, SideNavMenuItem,
  Theme, Tag,
} from '@carbon/react';
import {
  Dashboard, Network_3, Network_4, Container as ContainerIcon,
  Application, Catalog, ChartBar, Chat, JsonReference,
  Stethoscope, Certificate, Settings as SettingsIcon, User,
  WatsonHealthScalpel, Layers, AiStatusComplete,
} from '@carbon/icons-react';
import { OverviewView } from '@/features/overview';
import { FleetView } from '@/features/fleet';
import { HostsView } from '@/features/hosts';
import { ContainersView } from '@/features/containers';
import { EngineSelector } from '@/features/engines';
import { PinManager } from '@/features/pins';
import { PatchesView } from '@/features/patches';
import { DriftDashboard } from '@/features/drift';
import { BenchView } from '@/features/bench';
import { ChatView } from '@/features/chat';
import { JobsView } from '@/features/jobs';
import { DoctorView } from '@/features/doctor';
import { EvidenceView } from '@/features/evidence';
import { ConfigsView } from '@/features/configs';
import { LicensingPanel } from '@/features/licensing';
import { AuthView } from '@/features/auth';
import { SettingsView } from '@/features/settings';

interface NavGroup {
  label: string;
  icon: React.ElementType;
  items: Array<{ to: string; label: string }>;
}

const NAV_GROUPS: NavGroup[] = [
  {
    label: 'Live',
    icon: Dashboard,
    items: [
      { to: '/overview', label: 'Overview' },
      { to: '/fleet', label: 'Fleet' },
      { to: '/hosts', label: 'Hosts' },
      { to: '/containers', label: 'Containers' },
    ],
  },
  {
    label: 'Engines',
    icon: Network_3,
    items: [
      { to: '/engines', label: 'Engines' },
      { to: '/pins', label: 'Pins' },
      { to: '/patches', label: 'Patches' },
      { to: '/drift', label: 'Drift' },
    ],
  },
  {
    label: 'Workloads',
    icon: ChartBar,
    items: [
      { to: '/bench', label: 'Bench' },
      { to: '/chat', label: 'Chat' },
      { to: '/jobs', label: 'Jobs' },
    ],
  },
  {
    label: 'Health',
    icon: Stethoscope,
    items: [
      { to: '/doctor', label: 'Doctor' },
      { to: '/evidence', label: 'Evidence' },
      { to: '/configs', label: 'Configs' },
    ],
  },
  {
    label: 'Admin',
    icon: Certificate,
    items: [
      { to: '/licensing', label: 'Licensing' },
      { to: '/auth', label: 'Auth' },
      { to: '/settings', label: 'Settings' },
    ],
  },
];

function SideNavGroup({ group }: { group: NavGroup }): JSX.Element {
  const location = useLocation();
  const isActive = group.items.some((i) => location.pathname.startsWith(i.to));
  return (
    <SideNavMenu
      title={group.label}
      renderIcon={group.icon as any}
      defaultExpanded={isActive}
    >
      {group.items.map((item) => (
        <NavLink key={item.to} to={item.to} style={{ textDecoration: 'none' }}>
          {({ isActive: linkActive }) => (
            <SideNavMenuItem isActive={linkActive}>{item.label}</SideNavMenuItem>
          )}
        </NavLink>
      ))}
    </SideNavMenu>
  );
}

function CarbonShell({ children }: { children: ReactNode }): JSX.Element {
  return (
    <Theme theme="g100">
      <Header aria-label="sndr-platform Control Center">
        <HeaderName href="/overview" prefix="sndr">
          Control Center
        </HeaderName>
        <HeaderGlobalBar>
          <HeaderGlobalAction aria-label="Locale">
            <User size={20} />
          </HeaderGlobalAction>
        </HeaderGlobalBar>
        <SideNav aria-label="Side navigation" expanded isPersistent>
          <SideNavItems>
            {NAV_GROUPS.map((group) => (
              <SideNavGroup key={group.label} group={group} />
            ))}
          </SideNavItems>
        </SideNav>
      </Header>
      <main
        style={{
          marginLeft: 256,
          marginTop: 48,
          padding: 24,
          minHeight: 'calc(100vh - 48px)',
        }}
      >
        {children}
      </main>
    </Theme>
  );
}

export function CarbonApp(): JSX.Element {
  return (
    <BrowserRouter>
      <CarbonShell>
        <Routes>
          <Route path="/" element={<Navigate to="/overview" replace />} />
          <Route path="/overview" element={<OverviewView />} />
          <Route path="/fleet" element={<FleetView />} />
          <Route path="/hosts" element={<HostsView />} />
          <Route path="/containers" element={<ContainersView />} />
          <Route path="/engines" element={<EngineSelector />} />
          <Route path="/pins" element={<PinManager />} />
          <Route path="/patches" element={<PatchesView />} />
          <Route path="/drift" element={<DriftDashboard />} />
          <Route path="/bench" element={<BenchView />} />
          <Route path="/chat" element={<ChatView />} />
          <Route path="/jobs" element={<JobsView />} />
          <Route path="/doctor" element={<DoctorView />} />
          <Route path="/evidence" element={<EvidenceView />} />
          <Route path="/configs" element={<ConfigsView />} />
          <Route path="/licensing" element={<LicensingPanel />} />
          <Route path="/auth" element={<AuthView />} />
          <Route path="/settings" element={<SettingsView />} />
          <Route path="*" element={<Navigate to="/overview" replace />} />
        </Routes>
      </CarbonShell>
    </BrowserRouter>
  );
}

export default CarbonApp;
