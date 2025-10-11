import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright configuration for testing against production deployment
 * Usage: npx playwright test --config=playwright.config.production.ts
 */
export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: false, // Sequential for production testing
  forbidOnly: true,
  retries: 2, // Retry failed tests twice
  workers: 1, // Single worker for production
  timeout: 120000, // 2 minutes per test (production can be slower)
  reporter: [
    ['list'],
    ['html', { outputFolder: 'playwright-report-production' }]
  ],
  use: {
    // Production URL
    baseURL: 'https://friendly-greetings-launchpad-amey.onrender.com',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    // Longer timeouts for production
    actionTimeout: 30000,
    navigationTimeout: 60000,
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  // No webServer needed - testing against deployed app
});
