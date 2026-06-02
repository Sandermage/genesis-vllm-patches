import { test, expect } from "@playwright/test";

// A host card must be an action surface, not just info: "Chat with engine"
// drives the chat to that host:engine_port (+ key), and "Connect daemon"
// registers its daemon in the server switcher.
test("host card wires into chat and the server switcher", async ({ page }) => {
  const errors: string[] = [];
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });

  await page.goto("/");
  // Seed a host profile pointing engine→8199, daemon→8770, with an API key.
  await page.evaluate(async () => {
    localStorage.removeItem("sndr.chat.v1");
    localStorage.removeItem("sndr.gui.servers");
    await fetch("/api/v1/hosts", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label: "Wired Box", host: "127.0.0.1", engine_port: 8199, port: 8765, api_key: "genesis-local", role: "dev" }),
    });
  });
  await page.reload();

  await page.click('.side-nav button:has-text("Hosts")');
  await page.waitForSelector(".fleet-card");
  const card = page.locator(".fleet-card").filter({ hasText: "Wired Box" });
  await expect(card).toBeVisible();

  // 1) Chat with engine → chat prefilled with host/port/key, section switched.
  await card.locator('.primary-action:has-text("Chat with engine")').click();
  await expect(page.locator(".chat2")).toBeVisible();
  await expect(page.locator('.chat-field-port input')).toHaveValue("8199");
  await expect(page.locator('.chat-field:has(span:text-is("Host")) input')).toHaveValue("127.0.0.1");
  // The engine key is NOT pushed to the browser anymore — it's resolved
  // server-side from the host's encrypted secret via host_id, so the chat key
  // field stays empty (security: the raw key never reaches the client).
  expect(await page.locator('.chat-field-key input').inputValue()).toBe("");
  // No console errors up to here (engine may be down — that surfaces in-UI).
  expect(errors.filter((e) => !/ERR_CONNECTION_REFUSED|Failed to load resource/.test(e)), errors.join("\n")).toHaveLength(0);

  // 2) Connect daemon → probes for a real daemon (port 8765 is the live one),
  // then registers it + makes it the api base. (A dead port is refused — that
  // guard is what stops a GPU-only host from blanking the UI.)
  await page.click('.side-nav button:has-text("Hosts")');
  await page.waitForSelector(".fleet-card");
  await page.locator(".fleet-card").filter({ hasText: "Wired Box" }).locator('.ghost-button:has-text("Connect daemon")').click();
  await page.waitForTimeout(1200);
  // The probe passed (8765 is live) and the GUI re-pointed at that daemon.
  expect(await page.evaluate(() => localStorage.getItem("sndr.gui.apiBase"))).toBe("http://127.0.0.1:8765");
  // The daemon stayed reachable (no daemon-down banner).
  expect(await page.locator(".daemon-down").count()).toBe(0);

  // cleanup the seeded host
  await page.evaluate(async () => { await fetch("/api/v1/hosts/wired-box", { method: "DELETE" }); });
});
