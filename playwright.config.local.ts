import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright configuration for local testing with manual dev server
 * Usage: npx playwright test --config=playwright.config.local.ts
 */
export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: false,
  forbidOnly: false,
  retries: 1,
  workers: 1,
  timeout: 180000, // 3 minutes per test
  reporter: [
    ['list'],
    ['html', { outputFolder: 'playwright-report-local', open: 'never' }]
  ],
  use: {
    baseURL: 'http://localhost:5173',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    actionTimeout: 30000,
    navigationTimeout: 60000,
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  // No webServer - we start it manually
});
