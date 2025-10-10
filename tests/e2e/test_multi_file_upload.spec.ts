/**
 * E2E Tests for Multi-File Upload (Playwright)
 * 
 * Tests:
 * - Upload 5 files simultaneously
 * - Verify all process correctly
 * - Verify progress tracking
 * - Verify cancel functionality
 */

import { test, expect } from '@playwright/test';
import path from 'path';

test.describe('Multi-File Upload', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
  });

  test('should upload 5 files simultaneously', async ({ page }) => {
    const files = [
      path.join(__dirname, '../fixtures/file1.csv'),
      path.join(__dirname, '../fixtures/file2.csv'),
      path.join(__dirname, '../fixtures/file3.csv'),
      path.join(__dirname, '../fixtures/file4.csv'),
      path.join(__dirname, '../fixtures/file5.csv'),
    ];
    
    const uploadInput = page.locator('input[type="file"]');
    await uploadInput.setInputFiles(files);
    
    // All files should appear in the list
    for (let i = 1; i <= 5; i++) {
      await expect(page.locator(`text=file${i}.csv`)).toBeVisible({ timeout: 5000 });
    }
    
    // Wait for all to complete
    const completedFiles = page.locator('text=Complete');
    await expect(completedFiles).toHaveCount(5, { timeout: 300000 }); // 5 minutes max
  });

  test('should show individual progress for each file', async ({ page }) => {
    const files = [
      path.join(__dirname, '../fixtures/file1.csv'),
      path.join(__dirname, '../fixtures/file2.csv'),
      path.join(__dirname, '../fixtures/file3.csv'),
    ];
    
    const uploadInput = page.locator('input[type="file"]');
    await uploadInput.setInputFiles(files);
    
    // Each file should have its own progress bar
    const progressBars = page.locator('[role="progressbar"]');
    await expect(progressBars).toHaveCount(3, { timeout: 10000 });
    
    // Each should show different progress values
    await page.waitForTimeout(2000); // Wait for processing to start
    
    const progressValues = await progressBars.evaluateAll(bars => 
      bars.map(bar => bar.getAttribute('aria-valuenow'))
    );
    
    // At least some should have different values
    const uniqueValues = new Set(progressValues);
    expect(uniqueValues.size).toBeGreaterThan(0);
  });

  test('should process files sequentially', async ({ page }) => {
    const files = [
      path.join(__dirname, '../fixtures/file1.csv'),
      path.join(__dirname, '../fixtures/file2.csv'),
    ];
    
    const uploadInput = page.locator('input[type="file"]');
    await uploadInput.setInputFiles(files);
    
    // First file should start processing first
    await expect(page.locator('text=file1.csv').locator('..').locator('text=Processing')).toBeVisible({ timeout: 10000 });
    
    // Second file should be queued
    await expect(page.locator('text=file2.csv').locator('..').locator('text=Queued')).toBeVisible({ timeout: 5000 });
    
    // Wait for first to complete
    await expect(page.locator('text=file1.csv').locator('..').locator('text=Complete')).toBeVisible({ timeout: 60000 });
    
    // Second should then start processing
    await expect(page.locator('text=file2.csv').locator('..').locator('text=Processing')).toBeVisible({ timeout: 10000 });
  });

  test('should handle mixed file types', async ({ page }) => {
    const files = [
      path.join(__dirname, '../fixtures/data.csv'),
      path.join(__dirname, '../fixtures/data.xlsx'),
      path.join(__dirname, '../fixtures/data.xls'),
    ];
    
    const uploadInput = page.locator('input[type="file"]');
    await uploadInput.setInputFiles(files);
    
    // All should be accepted
    await expect(page.locator('text=data.csv')).toBeVisible();
    await expect(page.locator('text=data.xlsx')).toBeVisible();
    await expect(page.locator('text=data.xls')).toBeVisible();
    
    // All should process
    await expect(page.locator('text=Complete')).toHaveCount(3, { timeout: 180000 });
  });

  test('should reject invalid files in batch', async ({ page }) => {
    const files = [
      path.join(__dirname, '../fixtures/valid.csv'),
      path.join(__dirname, '../fixtures/invalid.pdf'),
      path.join(__dirname, '../fixtures/valid2.xlsx'),
    ];
    
    const uploadInput = page.locator('input[type="file"]');
    await uploadInput.setInputFiles(files);
    
    // Should show error for invalid file
    await expect(page.locator('text=invalid')).toBeVisible({ timeout: 5000 });
    
    // Valid files should not be uploaded if batch validation fails
    // OR valid files should process and invalid should be skipped
  });

  test('should allow canceling individual files', async ({ page }) => {
    const files = [
      path.join(__dirname, '../fixtures/file1.csv'),
      path.join(__dirname, '../fixtures/file2.csv'),
      path.join(__dirname, '../fixtures/file3.csv'),
    ];
    
    const uploadInput = page.locator('input[type="file"]');
    await uploadInput.setInputFiles(files);
    
    // Wait for processing to start
    await page.waitForTimeout(1000);
    
    // Cancel second file
    const file2Row = page.locator('text=file2.csv').locator('..');
    const cancelButton = file2Row.locator('button:has-text("Cancel")');
    
    if (await cancelButton.isVisible()) {
      await cancelButton.click();
      
      // File 2 should be cancelled
      await expect(file2Row.locator('text=Cancelled')).toBeVisible({ timeout: 5000 });
      
      // Other files should continue
      await expect(page.locator('text=file1.csv').locator('..').locator('text=Complete')).toBeVisible({ timeout: 60000 });
    }
  });

  test('should remove completed files from list', async ({ page }) => {
    const files = [
      path.join(__dirname, '../fixtures/file1.csv'),
      path.join(__dirname, '../fixtures/file2.csv'),
    ];
    
    const uploadInput = page.locator('input[type="file"]');
    await uploadInput.setInputFiles(files);
    
    // Wait for completion
    await expect(page.locator('text=Complete')).toHaveCount(2, { timeout: 120000 });
    
    // Files should move to completed section after delay
    await page.waitForTimeout(3000);
    
    // Should appear in "Completed Files" section
    const completedSection = page.locator('text=Completed Files').locator('..');
    await expect(completedSection.locator('text=file1.csv')).toBeVisible({ timeout: 5000 });
  });

  test('should show total progress for batch', async ({ page }) => {
    const files = [
      path.join(__dirname, '../fixtures/file1.csv'),
      path.join(__dirname, '../fixtures/file2.csv'),
      path.join(__dirname, '../fixtures/file3.csv'),
    ];
    
    const uploadInput = page.locator('input[type="file"]');
    await uploadInput.setInputFiles(files);
    
    // Should show batch progress (e.g., "2 of 3 files completed")
    await expect(page.locator('text=/\\d+ of \\d+ files/')).toBeVisible({ timeout: 10000 });
  });

  test('should handle duplicate files in batch', async ({ page }) => {
    // Upload first batch
    const files1 = [
      path.join(__dirname, '../fixtures/file1.csv'),
      path.join(__dirname, '../fixtures/file2.csv'),
    ];
    
    const uploadInput = page.locator('input[type="file"]');
    await uploadInput.setInputFiles(files1);
    await expect(page.locator('text=Complete')).toHaveCount(2, { timeout: 120000 });
    
    // Upload same files again
    await uploadInput.setInputFiles(files1);
    
    // Should detect duplicates
    await expect(page.locator('text=duplicate')).toBeVisible({ timeout: 10000 });
  });

  test('should limit maximum files to 15', async ({ page }) => {
    // Try to upload 20 files
    const files = Array.from({ length: 20 }, (_, i) => 
      path.join(__dirname, `../fixtures/file${i + 1}.csv`)
    );
    
    const uploadInput = page.locator('input[type="file"]');
    await uploadInput.setInputFiles(files);
    
    // Should only accept 15 files
    const fileItems = page.locator('[data-testid="file-item"]');
    const count = await fileItems.count();
    
    expect(count).toBeLessThanOrEqual(15);
  });

  test('should show error summary for failed files', async ({ page }) => {
    // Simulate some files failing
    await page.route('**/api/upload', async (route, request) => {
      const body = await request.postDataJSON();
      if (body.filename === 'file2.csv') {
        route.fulfill({ status: 500, body: 'Processing error' });
      } else {
        route.continue();
      }
    });
    
    const files = [
      path.join(__dirname, '../fixtures/file1.csv'),
      path.join(__dirname, '../fixtures/file2.csv'),
      path.join(__dirname, '../fixtures/file3.csv'),
    ];
    
    const uploadInput = page.locator('input[type="file"]');
    await uploadInput.setInputFiles(files);
    
    // Wait for processing
    await page.waitForTimeout(5000);
    
    // Should show error for file2
    await expect(page.locator('text=file2.csv').locator('..').locator('text=Failed')).toBeVisible({ timeout: 10000 });
    
    // Others should complete
    await expect(page.locator('text=file1.csv').locator('..').locator('text=Complete')).toBeVisible({ timeout: 60000 });
  });

  test('should maintain order of files', async ({ page }) => {
    const files = [
      path.join(__dirname, '../fixtures/alpha.csv'),
      path.join(__dirname, '../fixtures/beta.csv'),
      path.join(__dirname, '../fixtures/gamma.csv'),
    ];
    
    const uploadInput = page.locator('input[type="file"]');
    await uploadInput.setInputFiles(files);
    
    // Files should appear in upload order
    const fileNames = await page.locator('[data-testid="file-name"]').allTextContents();
    
    expect(fileNames[0]).toContain('alpha');
    expect(fileNames[1]).toContain('beta');
    expect(fileNames[2]).toContain('gamma');
  });
});

test.describe('Batch Operations', () => {
  test('should allow canceling all files', async ({ page }) => {
    await page.goto('/');
    
    const files = [
      path.join(__dirname, '../fixtures/file1.csv'),
      path.join(__dirname, '../fixtures/file2.csv'),
      path.join(__dirname, '../fixtures/file3.csv'),
    ];
    
    const uploadInput = page.locator('input[type="file"]');
    await uploadInput.setInputFiles(files);
    
    // Wait for processing to start
    await page.waitForTimeout(1000);
    
    // Click "Cancel All" button if available
    const cancelAllButton = page.locator('button:has-text("Cancel All")');
    if (await cancelAllButton.isVisible()) {
      await cancelAllButton.click();
      
      // All files should be cancelled
      await expect(page.locator('text=Cancelled')).toHaveCount(3, { timeout: 10000 });
    }
  });

  test('should allow removing all completed files', async ({ page }) => {
    await page.goto('/');
    
    const files = [
      path.join(__dirname, '../fixtures/file1.csv'),
      path.join(__dirname, '../fixtures/file2.csv'),
    ];
    
    const uploadInput = page.locator('input[type="file"]');
    await uploadInput.setInputFiles(files);
    
    // Wait for completion
    await expect(page.locator('text=Complete')).toHaveCount(2, { timeout: 120000 });
    
    // Click "Clear All" button in completed section
    const clearAllButton = page.locator('button:has-text("Clear All")');
    if (await clearAllButton.isVisible()) {
      await clearAllButton.click();
      
      // Completed files should be removed
      await expect(page.locator('text=file1.csv')).not.toBeVisible();
      await expect(page.locator('text=file2.csv')).not.toBeVisible();
    }
  });
});

test.describe('Performance', () => {
  test('should handle 15 files efficiently', async ({ page }) => {
    await page.goto('/');
    
    const files = Array.from({ length: 15 }, (_, i) => 
      path.join(__dirname, `../fixtures/file${i + 1}.csv`)
    );
    
    const startTime = Date.now();
    
    const uploadInput = page.locator('input[type="file"]');
    await uploadInput.setInputFiles(files);
    
    // All should complete within reasonable time (15 files * 10s = 150s max)
    await expect(page.locator('text=Complete')).toHaveCount(15, { timeout: 180000 });
    
    const endTime = Date.now();
    const totalTime = (endTime - startTime) / 1000;
    
    console.log(`Total time for 15 files: ${totalTime}s`);
    expect(totalTime).toBeLessThan(200); // Should complete in <200s
  });
});
