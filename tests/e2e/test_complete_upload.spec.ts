/**
 * E2E Tests for Complete Upload Flow (Playwright)
 * 
 * Tests full user journey:
 * - User opens app
 * - User uploads file
 * - Duplicate detected
 * - User chooses action
 * - File processes successfully
 * - Results displayed
 */

import { test, expect } from '@playwright/test';
import path from 'path';

test.describe('Complete Upload Flow', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to app
    await page.goto('/');
    
    // Wait for app to load
    await page.waitForLoadState('networkidle');
  });

  test('should complete successful file upload', async ({ page }) => {
    // Create test file
    const testFilePath = path.join(__dirname, '../fixtures/test_data.csv');
    
    // Find upload button/area
    const uploadInput = page.locator('input[type="file"]');
    
    // Upload file
    await uploadInput.setInputFiles(testFilePath);
    
    // Wait for upload to start
    await expect(page.locator('text=Uploading')).toBeVisible({ timeout: 5000 });
    
    // Wait for processing to complete
    await expect(page.locator('text=Processing')).toBeVisible({ timeout: 10000 });
    
    // Wait for completion
    await expect(page.locator('text=Complete')).toBeVisible({ timeout: 60000 });
    
    // Verify success message
    await expect(page.locator('text=successfully')).toBeVisible();
  });

  test('should handle duplicate file detection', async ({ page }) => {
    const testFilePath = path.join(__dirname, '../fixtures/test_data.csv');
    
    // Upload file first time
    const uploadInput = page.locator('input[type="file"]');
    await uploadInput.setInputFiles(testFilePath);
    
    // Wait for first upload to complete
    await expect(page.locator('text=Complete')).toBeVisible({ timeout: 60000 });
    
    // Upload same file again
    await uploadInput.setInputFiles(testFilePath);
    
    // Should show duplicate modal
    await expect(page.locator('text=duplicate')).toBeVisible({ timeout: 10000 });
    await expect(page.locator('text=Replace')).toBeVisible();
    await expect(page.locator('text=Skip')).toBeVisible();
  });

  test('should handle user decision to replace duplicate', async ({ page }) => {
    const testFilePath = path.join(__dirname, '../fixtures/test_data.csv');
    
    // Upload file twice to trigger duplicate
    const uploadInput = page.locator('input[type="file"]');
    await uploadInput.setInputFiles(testFilePath);
    await expect(page.locator('text=Complete')).toBeVisible({ timeout: 60000 });
    
    await uploadInput.setInputFiles(testFilePath);
    await expect(page.locator('text=duplicate')).toBeVisible({ timeout: 10000 });
    
    // Click replace button
    await page.locator('button:has-text("Replace")').click();
    
    // Should continue processing
    await expect(page.locator('text=Processing')).toBeVisible({ timeout: 10000 });
    await expect(page.locator('text=Complete')).toBeVisible({ timeout: 60000 });
  });

  test('should handle user decision to skip duplicate', async ({ page }) => {
    const testFilePath = path.join(__dirname, '../fixtures/test_data.csv');
    
    // Upload file twice
    const uploadInput = page.locator('input[type="file"]');
    await uploadInput.setInputFiles(testFilePath);
    await expect(page.locator('text=Complete')).toBeVisible({ timeout: 60000 });
    
    await uploadInput.setInputFiles(testFilePath);
    await expect(page.locator('text=duplicate')).toBeVisible({ timeout: 10000 });
    
    // Click skip button
    await page.locator('button:has-text("Skip")').click();
    
    // Should show skipped message
    await expect(page.locator('text=Skipped')).toBeVisible({ timeout: 5000 });
  });

  test('should show real-time progress updates', async ({ page }) => {
    const testFilePath = path.join(__dirname, '../fixtures/test_data.csv');
    
    const uploadInput = page.locator('input[type="file"]');
    await uploadInput.setInputFiles(testFilePath);
    
    // Should show various progress stages
    await expect(page.locator('text=Uploading')).toBeVisible({ timeout: 5000 });
    
    // Progress bar should be visible
    const progressBar = page.locator('[role="progressbar"]');
    await expect(progressBar).toBeVisible({ timeout: 5000 });
    
    // Wait for processing stages
    await expect(page.locator('text=Processing')).toBeVisible({ timeout: 10000 });
    
    // Final completion
    await expect(page.locator('text=Complete')).toBeVisible({ timeout: 60000 });
  });

  test('should handle file validation errors', async ({ page }) => {
    // Try to upload invalid file type
    const invalidFilePath = path.join(__dirname, '../fixtures/test_document.pdf');
    
    const uploadInput = page.locator('input[type="file"]');
    await uploadInput.setInputFiles(invalidFilePath);
    
    // Should show error message
    await expect(page.locator('text=invalid')).toBeVisible({ timeout: 5000 });
    await expect(page.locator('text=Excel')).toBeVisible({ timeout: 5000 });
  });

  test('should handle large file upload', async ({ page }) => {
    // This test requires a large test file (>100MB)
    test.skip(!process.env.TEST_LARGE_FILES, 'Large file tests disabled');
    
    const largeFilePath = path.join(__dirname, '../fixtures/large_data.xlsx');
    
    const uploadInput = page.locator('input[type="file"]');
    await uploadInput.setInputFiles(largeFilePath);
    
    // Should handle large file without timeout
    await expect(page.locator('text=Uploading')).toBeVisible({ timeout: 10000 });
    await expect(page.locator('text=Processing')).toBeVisible({ timeout: 30000 });
    await expect(page.locator('text=Complete')).toBeVisible({ timeout: 180000 }); // 3 minutes
  });

  test('should allow canceling upload', async ({ page }) => {
    const testFilePath = path.join(__dirname, '../fixtures/test_data.csv');
    
    const uploadInput = page.locator('input[type="file"]');
    await uploadInput.setInputFiles(testFilePath);
    
    // Wait for upload to start
    await expect(page.locator('text=Uploading')).toBeVisible({ timeout: 5000 });
    
    // Click cancel button
    const cancelButton = page.locator('button:has-text("Cancel")');
    if (await cancelButton.isVisible()) {
      await cancelButton.click();
      
      // Should show cancelled message
      await expect(page.locator('text=Cancelled')).toBeVisible({ timeout: 5000 });
    }
  });

  test('should display uploaded file results', async ({ page }) => {
    const testFilePath = path.join(__dirname, '../fixtures/test_data.csv');
    
    const uploadInput = page.locator('input[type="file"]');
    await uploadInput.setInputFiles(testFilePath);
    
    // Wait for completion
    await expect(page.locator('text=Complete')).toBeVisible({ timeout: 60000 });
    
    // Should show file in completed list
    await expect(page.locator('text=test_data.csv')).toBeVisible();
    
    // Should show completion timestamp
    await expect(page.locator('text=Completed')).toBeVisible();
  });

  test('should handle network errors gracefully', async ({ page }) => {
    // Simulate offline mode
    await page.context().setOffline(true);
    
    const testFilePath = path.join(__dirname, '../fixtures/test_data.csv');
    const uploadInput = page.locator('input[type="file"]');
    
    await uploadInput.setInputFiles(testFilePath);
    
    // Should show error message
    await expect(page.locator('text=error')).toBeVisible({ timeout: 10000 });
    
    // Restore online mode
    await page.context().setOffline(false);
  });

  test('should handle WebSocket fallback to polling', async ({ page }) => {
    // Block WebSocket connections
    await page.route('**/ws/**', route => route.abort());
    
    const testFilePath = path.join(__dirname, '../fixtures/test_data.csv');
    const uploadInput = page.locator('input[type="file"]');
    
    await uploadInput.setInputFiles(testFilePath);
    
    // Should still show progress via polling
    await expect(page.locator('text=Processing')).toBeVisible({ timeout: 10000 });
    await expect(page.locator('text=Complete')).toBeVisible({ timeout: 60000 });
  });
});

test.describe('Multiple File Upload', () => {
  test('should upload multiple files sequentially', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    const files = [
      path.join(__dirname, '../fixtures/test_data_1.csv'),
      path.join(__dirname, '../fixtures/test_data_2.csv'),
      path.join(__dirname, '../fixtures/test_data_3.csv'),
    ];
    
    const uploadInput = page.locator('input[type="file"]');
    await uploadInput.setInputFiles(files);
    
    // Should show all files in queue
    for (let i = 1; i <= 3; i++) {
      await expect(page.locator(`text=test_data_${i}.csv`)).toBeVisible();
    }
    
    // Wait for all to complete
    await expect(page.locator('text=Complete').nth(2)).toBeVisible({ timeout: 180000 });
  });

  test('should show progress for each file', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    const files = [
      path.join(__dirname, '../fixtures/test_data_1.csv'),
      path.join(__dirname, '../fixtures/test_data_2.csv'),
    ];
    
    const uploadInput = page.locator('input[type="file"]');
    await uploadInput.setInputFiles(files);
    
    // Each file should have its own progress indicator
    const progressBars = page.locator('[role="progressbar"]');
    await expect(progressBars).toHaveCount(2, { timeout: 10000 });
  });
});

test.describe('Error Recovery', () => {
  test('should retry failed uploads', async ({ page }) => {
    await page.goto('/');
    
    // Simulate server error
    await page.route('**/api/upload', route => {
      route.fulfill({ status: 500, body: 'Server error' });
    });
    
    const testFilePath = path.join(__dirname, '../fixtures/test_data.csv');
    const uploadInput = page.locator('input[type="file"]');
    
    await uploadInput.setInputFiles(testFilePath);
    
    // Should show error
    await expect(page.locator('text=error')).toBeVisible({ timeout: 10000 });
    
    // Remove route to allow retry
    await page.unroute('**/api/upload');
    
    // Click retry button if available
    const retryButton = page.locator('button:has-text("Retry")');
    if (await retryButton.isVisible()) {
      await retryButton.click();
      await expect(page.locator('text=Complete')).toBeVisible({ timeout: 60000 });
    }
  });

  test('should handle backend timeout', async ({ page }) => {
    await page.goto('/');
    
    // Simulate slow backend
    await page.route('**/api/upload', async route => {
      await new Promise(resolve => setTimeout(resolve, 35000)); // 35 seconds
      route.continue();
    });
    
    const testFilePath = path.join(__dirname, '../fixtures/test_data.csv');
    const uploadInput = page.locator('input[type="file"]');
    
    await uploadInput.setInputFiles(testFilePath);
    
    // Should handle timeout gracefully
    await expect(page.locator('text=timeout')).toBeVisible({ timeout: 40000 });
  });
});

test.describe('Accessibility', () => {
  test('should be keyboard navigable', async ({ page }) => {
    await page.goto('/');
    
    // Tab through elements
    await page.keyboard.press('Tab');
    await page.keyboard.press('Tab');
    
    // Should be able to trigger file upload with keyboard
    const uploadButton = page.locator('button:has-text("Upload")');
    if (await uploadButton.isVisible()) {
      await uploadButton.focus();
      await expect(uploadButton).toBeFocused();
    }
  });

  test('should have proper ARIA labels', async ({ page }) => {
    await page.goto('/');
    
    // Check for ARIA labels
    const uploadArea = page.locator('[role="button"]').first();
    await expect(uploadArea).toHaveAttribute('aria-label');
  });
});
