// SPDX-License-Identifier: Apache-2.0
import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { UserPresetsPanel, ProfileDeltaPanel } from "./presets";

afterEach(cleanup);

describe("UserPresetsPanel", () => {
  it("renders an empty-state when no presets", () => {
    render(<UserPresetsPanel presets={null} />);
    expect(screen.getByText(/No operator-local presets yet/)).toBeTruthy();
  });

  it("renders one row per preset with model / profile", () => {
    render(
      <UserPresetsPanel
        presets={{ count: 1, presets: [{ id: "draft-a", model: "qwen3.6-27b", profile: "tq-k8v4" }] } as never}
      />
    );
    expect(screen.getByText("draft-a")).toBeTruthy();
    expect(screen.getByText("qwen3.6-27b / tq-k8v4")).toBeTruthy();
  });
});

describe("ProfileDeltaPanel", () => {
  it("summarizes enable/disable/override counts + lists disabled flags", () => {
    render(
      <ProfileDeltaPanel
        def={{
          id: "profile-x", status: "stable", role: "primary",
          patches_delta: { enable: { P1: "1", P2: "1" }, disable: ["P9"], override: { P3: "2" } },
          sizing_override: { max_model_len: 8192 },
        }}
      />
    );
    expect(screen.getByText("Profile delta: profile-x")).toBeTruthy();
    expect(screen.getByText(/stable · role primary/)).toBeTruthy();
    expect(screen.getByText("P9")).toBeTruthy();
  });
});
