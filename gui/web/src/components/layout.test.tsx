// SPDX-License-Identifier: Apache-2.0
import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { Box } from "lucide-react";
import { ModuleGrid, ModuleCard } from "./layout";

afterEach(cleanup);

describe("ModuleGrid", () => {
  it("renders children and applies an optional className", () => {
    const { container } = render(<ModuleGrid className="stretch-row"><span>child</span></ModuleGrid>);
    const section = container.querySelector("section.module-grid");
    expect(section).not.toBeNull();
    expect(section!.className).toContain("stretch-row");
    expect(screen.getByText("child")).toBeTruthy();
  });
});

describe("ModuleCard", () => {
  it("renders title/desc/children and the wide modifier", () => {
    const { container } = render(
      <ModuleCard title="Operations" icon={<Box />} desc="4 ops" wide>
        <p>body</p>
      </ModuleCard>
    );
    expect(screen.getByText("Operations")).toBeTruthy();
    expect(screen.getByText("4 ops")).toBeTruthy();
    expect(screen.getByText("body")).toBeTruthy();
    expect(container.querySelector("section.module-card")!.className).toContain("wide");
  });
});
