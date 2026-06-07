// SPDX-License-Identifier: Apache-2.0
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, waitFor } from "@testing-library/react";

vi.mock("../api", () => ({
  api: {
    hostInventory: vi.fn().mockResolvedValue(null),
    fleetOverview: vi.fn().mockResolvedValue({ hosts: [] }),
    hostsReliability: vi.fn().mockResolvedValue({}),
    hostDelete: vi.fn(),
  },
}));
vi.mock("../components/toast", () => ({ toast: vi.fn() }));
import { HostsSection } from "./hosts-section";

afterEach(cleanup);

const props = {
  hostProfiles: [],
  environment: null,
  overview: null,
  runtimeTargets: [],
  apiBase: "http://127.0.0.1:8765",
  runtimeMode: "local" as const,
  onHostsRefresh: vi.fn(),
  onChatWithHost: vi.fn(),
  onAddServer: vi.fn(),
  focusHostId: null,
  onFocusConsumed: vi.fn(),
  onSetupNode: vi.fn(),
  onContainers: vi.fn(),
  onHardware: vi.fn(),
};

describe("HostsSection", () => {
  it("renders the hosts tab with the this-host card after inventory load", async () => {
    render(<HostsSection {...props} />);
    await waitFor(() => expect(screen.getByText(/This host/)).toBeTruthy());
  });
});
