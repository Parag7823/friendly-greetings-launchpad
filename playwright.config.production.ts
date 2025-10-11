import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright configuration for production testing
 * Tests against the deployed production environment
 */
export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: false,
  forbidOnly: true,
  retries: 2,
  workers: 1,
  timeout: 180000, // 3 minutes per test
  reporter: [
    ['list'],
    ['html', { outputFolder: 'playwright-report-production', open: 'never' }]
  ],
  use: {
    baseURL: 'https://friendly-greetings-launchpad-amey.onrender.com',
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
});  // No webServer needed - testing against deployed app
