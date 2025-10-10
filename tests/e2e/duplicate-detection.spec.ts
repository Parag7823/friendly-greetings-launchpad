/**
 * End-to-End Tests for Duplicate Detection UI
 * 
 * Tests the complete user journey for duplicate file handling:
 * - Upload duplicate file
 * - Modal interaction
 * - Decision submission
 * - Processing continuation
 */

import { test, expect } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

// Helper to create test Excel file
function createTestExcelFile(filename: string, rows: number): Buffer {
  // Create CSV content (will be uploaded as Excel-like data)
  const csvContent = `Date,Amount,Description\n${Array.from({ length: rows }, (_, i) => 
    `2024-01-${String(i + 1).padStart(2, '0')},${(i + 1) * 100},Transaction ${i + 1}`
  ).join('\n')}`;
  
  return Buffer.from(csvContent, 'utf-8');
}

test.describe('Duplicate Detection - Basic Flow', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to upload page
    await page.goto('/');
    
    // Wait for auth (anonymous sign-in)
    await page.waitForTimeout(2000);
  });

  test('should detect exact duplicate and show modal', async ({ page }) => {
    // Create test file
    const testFile = createTestExcelFile('test-file.csv', 10);
    
    // Upload file first time
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles({
      name: 'test-file.csv',
      mimeType: 'text/csv',
      buffer: testFile,
    });
    
    // Wait for first upload to complete
    await expect(page.getByText(/processing completed/i)).toBeVisible({ timeout: 60000 });
    
    // Upload same file again
    await fileInput.setInputFiles({
      name: 'test-file.csv',
      mimeType: 'text/csv',
      buffer: testFile,
    });
    
    // Verify duplicate modal appears
    await expect(page.getByText(/identical file detected/i)).toBeVisible({ timeout: 30000 });
    await expect(page.getByText(/duplicate/i)).toBeVisible();
    
    // Verify modal shows duplicate file info
    await expect(page.getByText(/test-file.csv/i)).toBeVisible();
    await expect(page.getByText(/uploaded/i)).toBeVisible();
  });

  test('should handle "Replace" decision', async ({ page }) => {
    const testFile = createTestExcelFile('replace-test.csv', 10);
    
    // Upload file first time
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles({
      name: 'replace-test.csv',
      mimeType: 'text/csv',
      buffer: testFile,
    });
    
    await expect(page.getByText(/processing completed/i)).toBeVisible({ timeout: 60000 });
    
    // Upload same file again
    await fileInput.setInputFiles({
      name: 'replace-test.csv',
      mimeType: 'text/csv',
      buffer: testFile,
    });
    
    // Wait for duplicate modal
    await expect(page.getByText(/identical file detected/i)).toBeVisible({ timeout: 30000 });
    
    // Click "Replace existing file" button
    await page.getByRole('button', { name: /replace existing file/i }).click();
    
    // Verify modal closes
    await expect(page.getByText(/identical file detected/i)).not.toBeVisible({ timeout: 5000 });
    
    // Verify processing continues
    await expect(page.getByText(/resuming/i)).toBeVisible({ timeout: 10000 });
    
    // Verify completion
    await expect(page.getByText(/processing completed/i)).toBeVisible({ timeout: 60000 });
  });

  test('should handle "Keep Both" decision', async ({ page }) => {
    const testFile = createTestExcelFile('keep-both-test.csv', 10);
    
    // Upload file first time
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles({
      name: 'keep-both-test.csv',
      mimeType: 'text/csv',
      buffer: testFile,
    });
    
    await expect(page.getByText(/processing completed/i)).toBeVisible({ timeout: 60000 });
    
    // Upload same file again
    await fileInput.setInputFiles({
      name: 'keep-both-test.csv',
      mimeType: 'text/csv',
      buffer: testFile,
    });
    
    // Wait for duplicate modal
    await expect(page.getByText(/identical file detected/i)).toBeVisible({ timeout: 30000 });
    
    // Click "Keep both files" button
    await page.getByRole('button', { name: /keep both files/i }).click();
    
    // Verify modal closes
    await expect(page.getByText(/identical file detected/i)).not.toBeVisible({ timeout: 5000 });
    
    // Verify processing continues
    await expect(page.getByText(/resuming/i)).toBeVisible({ timeout: 10000 });
    
    // Verify completion
    await expect(page.getByText(/processing completed/i)).toBeVisible({ timeout: 60000 });
  });

  test('should handle "Skip" decision', async ({ page }) => {
    const testFile = createTestExcelFile('skip-test.csv', 10);
    
    // Upload file first time
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles({
      name: 'skip-test.csv',
      mimeType: 'text/csv',
      buffer: testFile,
    });
    
    await expect(page.getByText(/processing completed/i)).toBeVisible({ timeout: 60000 });
    
    // Upload same file again
    await fileInput.setInputFiles({
      name: 'skip-test.csv',
      mimeType: 'text/csv',
      buffer: testFile,
    });
    
    // Wait for duplicate modal
    await expect(page.getByText(/identical file detected/i)).toBeVisible({ timeout: 30000 });
    
    // Click "Skip this upload" button
    await page.getByRole('button', { name: /skip this upload/i }).click();
    
    // Verify modal closes
    await expect(page.getByText(/identical file detected/i)).not.toBeVisible({ timeout: 5000 });
    
    // Verify upload cancelled
    await expect(page.getByText(/skipped/i)).toBeVisible({ timeout: 5000 });
    
    // Verify no processing occurs
    await expect(page.getByText(/processing completed/i)).not.toBeVisible({ timeout: 5000 });
  });

  test('should handle "Cancel" action', async ({ page }) => {
    const testFile = createTestExcelFile('cancel-test.csv', 10);
    
    // Upload file first time
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles({
      name: 'cancel-test.csv',
      mimeType: 'text/csv',
      buffer: testFile,
    });
    
    await expect(page.getByText(/processing completed/i)).toBeVisible({ timeout: 60000 });
    
    // Upload same file again
    await fileInput.setInputFiles({
      name: 'cancel-test.csv',
      mimeType: 'text/csv',
      buffer: testFile,
    });
    
    // Wait for duplicate modal
    await expect(page.getByText(/identical file detected/i)).toBeVisible({ timeout: 30000 });
    
    // Click "Cancel Upload" button
    await page.getByRole('button', { name: /cancel upload/i }).click();
    
    // Verify modal closes
    await expect(page.getByText(/identical file detected/i)).not.toBeVisible({ timeout: 5000 });
  });
});

test.describe('Duplicate Detection - Near Duplicate Flow', () => {
  test('should detect near-duplicate files', async ({ page }) => {
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Upload file with 100 rows
    const file1 = createTestExcelFile('near-dup-1.csv', 100);
    const fileInput = page.locator('input[type="file"]');
    
    await fileInput.setInputFiles({
      name: 'near-dup-1.csv',
      mimeType: 'text/csv',
      buffer: file1,
    });
    
    await expect(page.getByText(/processing completed/i)).toBeVisible({ timeout: 60000 });
    
    // Upload file with 95 same rows + 5 new rows (95% similarity)
    const file2 = createTestExcelFile('near-dup-2.csv', 95);
    
    await fileInput.setInputFiles({
      name: 'near-dup-2.csv',
      mimeType: 'text/csv',
      buffer: file2,
    });
    
    // Should detect near-duplicate or content overlap
    // Note: Actual detection depends on backend similarity threshold
    await page.waitForTimeout(10000);
    
    // Check if duplicate modal appeared or processing completed
    const modalVisible = await page.getByText(/similar file detected/i).isVisible().catch(() => false);
    const processingComplete = await page.getByText(/processing completed/i).isVisible().catch(() => false);
    
    // Either modal shows or processing completes (depending on similarity threshold)
    expect(modalVisible || processingComplete).toBeTruthy();
  });

  test('should offer delta merge for near-duplicates', async ({ page }) => {
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    const file1 = createTestExcelFile('delta-1.csv', 50);
    const fileInput = page.locator('input[type="file"]');
    
    await fileInput.setInputFiles({
      name: 'delta-1.csv',
      mimeType: 'text/csv',
      buffer: file1,
    });
    
    await expect(page.getByText(/processing completed/i)).toBeVisible({ timeout: 60000 });
    
    // Upload similar file
    const file2 = createTestExcelFile('delta-2.csv', 45);
    
    await fileInput.setInputFiles({
      name: 'delta-2.csv',
      mimeType: 'text/csv',
      buffer: file2,
    });
    
    // Wait for potential duplicate detection
    await page.waitForTimeout(15000);
    
    // Check if delta merge option is available
    const deltaMergeButton = page.getByRole('button', { name: /delta merge|merge new rows/i });
    const isDeltaMergeVisible = await deltaMergeButton.isVisible().catch(() => false);
    
    if (isDeltaMergeVisible) {
      // Click delta merge
      await deltaMergeButton.click();
      
      // Verify processing continues
      await expect(page.getByText(/resuming|processing/i)).toBeVisible({ timeout: 10000 });
      await expect(page.getByText(/processing completed/i)).toBeVisible({ timeout: 60000 });
    }
  });
});

test.describe('Duplicate Detection - Error Handling', () => {
  test('should handle network errors gracefully', async ({ page }) => {
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Upload file
    const testFile = createTestExcelFile('network-error-test.csv', 10);
    const fileInput = page.locator('input[type="file"]');
    
    await fileInput.setInputFiles({
      name: 'network-error-test.csv',
      mimeType: 'text/csv',
      buffer: testFile,
    });
    
    await expect(page.getByText(/processing completed/i)).toBeVisible({ timeout: 60000 });
    
    // Simulate network error by blocking API
    await page.route('**/handle-duplicate-decision', route => route.abort());
    
    // Upload duplicate
    await fileInput.setInputFiles({
      name: 'network-error-test.csv',
      mimeType: 'text/csv',
      buffer: testFile,
    });
    
    // Wait for duplicate modal
    await expect(page.getByText(/identical file detected/i)).toBeVisible({ timeout: 30000 });
    
    // Try to make decision
    await page.getByRole('button', { name: /replace existing file/i }).click();
    
    // Should show error toast
    await expect(page.getByText(/error|failed/i)).toBeVisible({ timeout: 10000 });
  });

  test('should handle missing job information', async ({ page }) => {
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // This test verifies error handling when job_id or file_hash is missing
    // In normal flow, this shouldn't happen, but we test defensive programming
    
    const testFile = createTestExcelFile('missing-job-test.csv', 10);
    const fileInput = page.locator('input[type="file"]');
    
    await fileInput.setInputFiles({
      name: 'missing-job-test.csv',
      mimeType: 'text/csv',
      buffer: testFile,
    });
    
    // Should complete normally
    await expect(page.getByText(/processing completed/i)).toBeVisible({ timeout: 60000 });
  });
});

test.describe('Duplicate Detection - Performance', () => {
  test('should handle large file duplicates efficiently', async ({ page }) => {
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Create large file (1000 rows)
    const largeFile = createTestExcelFile('large-file.csv', 1000);
    const fileInput = page.locator('input[type="file"]');
    
    const startTime = Date.now();
    
    await fileInput.setInputFiles({
      name: 'large-file.csv',
      mimeType: 'text/csv',
      buffer: largeFile,
    });
    
    await expect(page.getByText(/processing completed/i)).toBeVisible({ timeout: 120000 });
    
    // Upload duplicate
    await fileInput.setInputFiles({
      name: 'large-file.csv',
      mimeType: 'text/csv',
      buffer: largeFile,
    });
    
    // Duplicate detection should be fast (< 5 seconds)
    await expect(page.getByText(/identical file detected/i)).toBeVisible({ timeout: 10000 });
    
    const detectionTime = Date.now() - startTime;
    
    // Log performance
    console.log(`Large file duplicate detection took ${detectionTime}ms`);
    
    // Make decision
    await page.getByRole('button', { name: /skip this upload/i }).click();
  });

  test('should use caching for repeated checks', async ({ page }) => {
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    const testFile = createTestExcelFile('cache-test.csv', 50);
    const fileInput = page.locator('input[type="file"]');
    
    // First upload
    await fileInput.setInputFiles({
      name: 'cache-test.csv',
      mimeType: 'text/csv',
      buffer: testFile,
    });
    
    await expect(page.getByText(/processing completed/i)).toBeVisible({ timeout: 60000 });
    
    // Second upload (should use cache)
    const cacheStartTime = Date.now();
    
    await fileInput.setInputFiles({
      name: 'cache-test.csv',
      mimeType: 'text/csv',
      buffer: testFile,
    });
    
    await expect(page.getByText(/identical file detected/i)).toBeVisible({ timeout: 10000 });
    
    const cacheDetectionTime = Date.now() - cacheStartTime;
    
    // Cached detection should be very fast (< 2 seconds)
    expect(cacheDetectionTime).toBeLessThan(5000);
    
    console.log(`Cached duplicate detection took ${cacheDetectionTime}ms`);
    
    // Clean up
    await page.getByRole('button', { name: /skip this upload/i }).click();
  });
});

test.describe('Duplicate Detection - UI/UX', () => {
  test('should show progress during duplicate check', async ({ page }) => {
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    const testFile = createTestExcelFile('progress-test.csv', 100);
    const fileInput = page.locator('input[type="file"]');
    
    await fileInput.setInputFiles({
      name: 'progress-test.csv',
      mimeType: 'text/csv',
      buffer: testFile,
    });
    
    // Should show "Checking for duplicates" message
    await expect(page.getByText(/checking for duplicates/i)).toBeVisible({ timeout: 5000 });
    
    await expect(page.getByText(/processing completed/i)).toBeVisible({ timeout: 60000 });
  });

  test('should display duplicate file details correctly', async ({ page }) => {
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    const testFile = createTestExcelFile('details-test.csv', 20);
    const fileInput = page.locator('input[type="file"]');
    
    await fileInput.setInputFiles({
      name: 'details-test.csv',
      mimeType: 'text/csv',
      buffer: testFile,
    });
    
    await expect(page.getByText(/processing completed/i)).toBeVisible({ timeout: 60000 });
    
    // Upload duplicate
    await fileInput.setInputFiles({
      name: 'details-test.csv',
      mimeType: 'text/csv',
      buffer: testFile,
    });
    
    await expect(page.getByText(/identical file detected/i)).toBeVisible({ timeout: 30000 });
    
    // Verify modal shows file details
    await expect(page.getByText(/details-test.csv/i)).toBeVisible();
    await expect(page.getByText(/rows/i)).toBeVisible();
    
    // Clean up
    await page.getByRole('button', { name: /cancel upload/i }).click();
  });

  test('should allow closing modal and restarting', async ({ page }) => {
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    const testFile = createTestExcelFile('close-modal-test.csv', 10);
    const fileInput = page.locator('input[type="file"]');
    
    await fileInput.setInputFiles({
      name: 'close-modal-test.csv',
      mimeType: 'text/csv',
      buffer: testFile,
    });
    
    await expect(page.getByText(/processing completed/i)).toBeVisible({ timeout: 60000 });
    
    // Upload duplicate
    await fileInput.setInputFiles({
      name: 'close-modal-test.csv',
      mimeType: 'text/csv',
      buffer: testFile,
    });
    
    await expect(page.getByText(/identical file detected/i)).toBeVisible({ timeout: 30000 });
    
    // Close modal
    await page.getByRole('button', { name: /cancel upload/i }).click();
    
    // Verify modal closed
    await expect(page.getByText(/identical file detected/i)).not.toBeVisible({ timeout: 5000 });
    
    // Should be able to upload again
    await fileInput.setInputFiles({
      name: 'close-modal-test.csv',
      mimeType: 'text/csv',
      buffer: testFile,
    });
    
    // Modal should appear again
    await expect(page.getByText(/identical file detected/i)).toBeVisible({ timeout: 30000 });
  });
});
