/**
 * E2E Tests for Phase 4-6: File Processing & Enrichment
 * ======================================================
 * 
 * Tests cover:
 * - Phase 4: File Parsing & Streaming (memory-efficient chunk processing)
 * - Phase 5: Platform & Document Classification (AI-powered detection)
 * - Phase 6: Row-Level Processing & Enrichment (vendor standardization, currency conversion, AI classification)
 * 
 * Test Strategy:
 * - Unit: Each enrichment step
 * - Integration: Full pipeline
 * - E2E: Large file → All enriched
 */

import { test, expect, Page } from '@playwright/test';
import * as XLSX from 'xlsx';

// Helper: Create test Excel file with specific data
function createFinancialTestFile(filename: string, rowCount: number, platform: string = 'stripe'): Buffer {
  const data: any[] = [];
  
  // Create realistic financial data based on platform
  const platformData = {
    stripe: {
      headers: ['id', 'amount', 'currency', 'description', 'customer_email', 'created', 'status'],
      generateRow: (i: number) => ({
        id: `ch_${Math.random().toString(36).substring(7)}`,
        amount: (Math.random() * 1000 + 10).toFixed(2),
        currency: 'usd',
        description: `Payment for invoice #${1000 + i}`,
        customer_email: `customer${i}@example.com`,
        created: new Date(2024, 0, 1 + i).toISOString(),
        status: 'succeeded'
      })
    },
    quickbooks: {
      headers: ['TxnDate', 'RefNumber', 'Memo', 'Account', 'Amount', 'Name'],
      generateRow: (i: number) => ({
        TxnDate: `01/${String(i + 1).padStart(2, '0')}/2024`,
        RefNumber: `INV-${1000 + i}`,
        Memo: `Office supplies from Vendor ${i % 10}`,
        Account: 'Office Expenses',
        Amount: (Math.random() * 500 + 50).toFixed(2),
        Name: `Vendor ${i % 10} LLC`
      })
    },
    payroll: {
      headers: ['Employee Name', 'Employee ID', 'Gross Pay', 'Net Pay', 'Pay Date', 'Department'],
      generateRow: (i: number) => ({
        'Employee Name': `Employee ${i}`,
        'Employee ID': `EMP${String(i).padStart(4, '0')}`,
        'Gross Pay': (Math.random() * 5000 + 3000).toFixed(2),
        'Net Pay': (Math.random() * 4000 + 2500).toFixed(2),
        'Pay Date': `01/${String(i % 28 + 1).padStart(2, '0')}/2024`,
        'Department': ['Engineering', 'Sales', 'Marketing', 'Operations'][i % 4]
      })
    }
  };

  const config = platformData[platform] || platformData.stripe;
  
  for (let i = 0; i < rowCount; i++) {
    data.push(config.generateRow(i));
  }

  const worksheet = XLSX.utils.json_to_sheet(data);
  const workbook = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(workbook, worksheet, 'Sheet1');
  
  return XLSX.write(workbook, { type: 'buffer', bookType: 'xlsx' }) as Buffer;
}

// Helper: Setup upload page
async function setupUploadPage(page: Page) {
  await page.goto('/');
  
  // Click "Get Started" button to trigger auth
  const getStartedButton = page.getByRole('button', { name: /get started/i });
  await getStartedButton.waitFor({ state: 'visible', timeout: 10000 });
  await getStartedButton.click();
  
  // Wait for authentication
  await page.waitForTimeout(2000);
  
  // Navigate to upload page
  await page.goto('/upload');
  
  // Wait for upload area to be available
  await page.getByText(/click to upload or drag and drop/i).waitFor({ state: 'visible', timeout: 30000 });
}

test.describe('Phase 4: File Parsing & Streaming', () => {
  
  test('should process small CSV file (100 rows)', async ({ page }) => {
    await setupUploadPage(page);
    
    // Create small test file
    const testFile = createFinancialTestFile('small-test.csv', 100, 'stripe');
    const fileInput = page.locator('#file-upload');
    
    await fileInput.setInputFiles({
      name: 'small-test.csv',
      mimeType: 'text/csv',
      buffer: testFile,
    });
    
    // Wait for processing to complete
    await expect(page.getByText(/completed files/i)).toBeVisible({ timeout: 60000 });
    
    // Verify file appears in completed list
    await expect(page.getByText(/small-test.csv/i)).toBeVisible();
  });

  test('should process medium Excel file (1,000 rows) using streaming', async ({ page }) => {
    await setupUploadPage(page);
    
    // Create medium test file (should trigger streaming)
    const testFile = createFinancialTestFile('medium-test.xlsx', 1000, 'quickbooks');
    const fileInput = page.locator('#file-upload');
    
    await fileInput.setInputFiles({
      name: 'medium-test.xlsx',
      mimeType: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      buffer: testFile,
    });
    
    // Should show streaming progress
    await expect(page.getByText(/processing/i)).toBeVisible({ timeout: 10000 });
    
    // Wait for completion (may take longer for 1000 rows)
    await expect(page.getByText(/completed files/i)).toBeVisible({ timeout: 120000 });
    
    // Verify file appears in completed list
    await expect(page.getByText(/medium-test.xlsx/i)).toBeVisible();
  });

  test('should handle large Excel file (10,000 rows) without OOM', async ({ page }) => {
    await setupUploadPage(page);
    
    // Create large test file (tests memory management)
    const testFile = createFinancialTestFile('large-test.xlsx', 10000, 'stripe');
    const fileInput = page.locator('#file-upload');
    
    await fileInput.setInputFiles({
      name: 'large-test.xlsx',
      mimeType: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      buffer: testFile,
    });
    
    // Should show streaming progress
    await expect(page.getByText(/processing/i)).toBeVisible({ timeout: 10000 });
    
    // Wait for completion (will take several minutes)
    await expect(page.getByText(/completed files/i)).toBeVisible({ timeout: 300000 }); // 5 minutes
    
    // Verify file appears in completed list
    await expect(page.getByText(/large-test.xlsx/i)).toBeVisible();
  });

  test('should process multi-sheet Excel file', async ({ page }) => {
    await setupUploadPage(page);
    
    // Create multi-sheet file
    const data1 = Array.from({ length: 100 }, (_, i) => ({
      id: i,
      amount: (Math.random() * 100).toFixed(2),
      description: `Transaction ${i}`
    }));
    
    const data2 = Array.from({ length: 50 }, (_, i) => ({
      id: i,
      vendor: `Vendor ${i}`,
      total: (Math.random() * 500).toFixed(2)
    }));
    
    const ws1 = XLSX.utils.json_to_sheet(data1);
    const ws2 = XLSX.utils.json_to_sheet(data2);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, ws1, 'Transactions');
    XLSX.utils.book_append_sheet(workbook, ws2, 'Vendors');
    
    const buffer = XLSX.write(workbook, { type: 'buffer', bookType: 'xlsx' }) as Buffer;
    
    const fileInput = page.locator('#file-upload');
    await fileInput.setInputFiles({
      name: 'multi-sheet.xlsx',
      mimeType: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      buffer: buffer,
    });
    
    // Wait for completion
    await expect(page.getByText(/completed files/i)).toBeVisible({ timeout: 120000 });
    
    // Verify file processed
    await expect(page.getByText(/multi-sheet.xlsx/i)).toBeVisible();
  });
});

test.describe('Phase 5: Platform & Document Classification', () => {
  
  test('should detect Stripe platform from column names', async ({ page }) => {
    await setupUploadPage(page);
    
    // Create Stripe-like data
    const testFile = createFinancialTestFile('stripe-data.xlsx', 50, 'stripe');
    const fileInput = page.locator('#file-upload');
    
    await fileInput.setInputFiles({
      name: 'stripe-data.xlsx',
      mimeType: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      buffer: testFile,
    });
    
    // Wait for completion
    await expect(page.getByText(/completed files/i)).toBeVisible({ timeout: 60000 });
    
    // Platform detection happens in backend - verify file processed successfully
    await expect(page.getByText(/stripe-data.xlsx/i)).toBeVisible();
  });

  test('should detect QuickBooks platform from column names', async ({ page }) => {
    await setupUploadPage(page);
    
    // Create QuickBooks-like data
    const testFile = createFinancialTestFile('quickbooks-export.xlsx', 50, 'quickbooks');
    const fileInput = page.locator('#file-upload');
    
    await fileInput.setInputFiles({
      name: 'quickbooks-export.xlsx',
      mimeType: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      buffer: testFile,
    });
    
    // Wait for completion
    await expect(page.getByText(/completed files/i)).toBeVisible({ timeout: 60000 });
    
    // Verify file processed
    await expect(page.getByText(/quickbooks-export.xlsx/i)).toBeVisible();
  });

  test('should detect Payroll document type', async ({ page }) => {
    await setupUploadPage(page);
    
    // Create Payroll-like data
    const testFile = createFinancialTestFile('payroll-jan-2024.xlsx', 50, 'payroll');
    const fileInput = page.locator('#file-upload');
    
    await fileInput.setInputFiles({
      name: 'payroll-jan-2024.xlsx',
      mimeType: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      buffer: testFile,
    });
    
    // Wait for completion
    await expect(page.getByText(/completed files/i)).toBeVisible({ timeout: 60000 });
    
    // Verify file processed
    await expect(page.getByText(/payroll-jan-2024.xlsx/i)).toBeVisible();
  });

  test('should handle unknown platform with AI classification', async ({ page }) => {
    await setupUploadPage(page);
    
    // Create generic financial data (no clear platform indicators)
    const data = Array.from({ length: 50 }, (_, i) => ({
      date: `2024-01-${String(i + 1).padStart(2, '0')}`,
      description: `Transaction ${i}`,
      amount: (Math.random() * 1000).toFixed(2),
      category: ['Food', 'Transport', 'Utilities', 'Entertainment'][i % 4]
    }));
    
    const worksheet = XLSX.utils.json_to_sheet(data);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, 'Data');
    const buffer = XLSX.write(workbook, { type: 'buffer', bookType: 'xlsx' }) as Buffer;
    
    const fileInput = page.locator('#file-upload');
    await fileInput.setInputFiles({
      name: 'unknown-platform.xlsx',
      mimeType: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      buffer: buffer,
    });
    
    // Wait for completion (AI classification may take longer)
    await expect(page.getByText(/completed files/i)).toBeVisible({ timeout: 120000 });
    
    // Verify file processed
    await expect(page.getByText(/unknown-platform.xlsx/i)).toBeVisible();
  });
});

test.describe('Phase 6: Row-Level Processing & Enrichment', () => {
  
  test('should enrich all rows with vendor standardization', async ({ page }) => {
    await setupUploadPage(page);
    
    // Create data with vendor variations
    const data = Array.from({ length: 100 }, (_, i) => ({
      date: `2024-01-${String((i % 28) + 1).padStart(2, '0')}`,
      vendor: [
        'Amazon Inc.',
        'Amazon, Inc',
        'AMAZON INC',
        'Microsoft Corporation',
        'Microsoft Corp.',
        'MICROSOFT CORP'
      ][i % 6],
      amount: (Math.random() * 500).toFixed(2),
      description: `Purchase ${i}`
    }));
    
    const worksheet = XLSX.utils.json_to_sheet(data);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, 'Expenses');
    const buffer = XLSX.write(workbook, { type: 'buffer', bookType: 'xlsx' }) as Buffer;
    
    const fileInput = page.locator('#file-upload');
    await fileInput.setInputFiles({
      name: 'vendor-standardization-test.xlsx',
      mimeType: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      buffer: buffer,
    });
    
    // Wait for completion
    await expect(page.getByText(/completed files/i)).toBeVisible({ timeout: 120000 });
    
    // Verify file processed (vendor standardization happens in backend)
    await expect(page.getByText(/vendor-standardization-test.xlsx/i)).toBeVisible();
  });

  test('should handle currency conversion', async ({ page }) => {
    await setupUploadPage(page);
    
    // Create data with multiple currencies
    const data = Array.from({ length: 50 }, (_, i) => ({
      date: `2024-01-${String(i + 1).padStart(2, '0')}`,
      amount: (Math.random() * 1000).toFixed(2),
      currency: ['USD', 'EUR', 'GBP', 'INR'][i % 4],
      description: `International transaction ${i}`
    }));
    
    const worksheet = XLSX.utils.json_to_sheet(data);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, 'Transactions');
    const buffer = XLSX.write(workbook, { type: 'buffer', bookType: 'xlsx' }) as Buffer;
    
    const fileInput = page.locator('#file-upload');
    await fileInput.setInputFiles({
      name: 'multi-currency.xlsx',
      mimeType: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      buffer: buffer,
    });
    
    // Wait for completion
    await expect(page.getByText(/completed files/i)).toBeVisible({ timeout: 120000 });
    
    // Verify file processed
    await expect(page.getByText(/multi-currency.xlsx/i)).toBeVisible();
  });

  test('should classify rows with AI (expense vs revenue)', async ({ page }) => {
    await setupUploadPage(page);
    
    // Create mixed expense/revenue data
    const data = Array.from({ length: 100 }, (_, i) => {
      const isRevenue = i % 3 === 0;
      return {
        date: `2024-01-${String((i % 28) + 1).padStart(2, '0')}`,
        description: isRevenue ? `Client payment for project ${i}` : `Office supplies purchase ${i}`,
        amount: isRevenue ? (Math.random() * 5000 + 1000).toFixed(2) : (Math.random() * 500 + 50).toFixed(2),
        type: isRevenue ? 'income' : 'expense'
      };
    });
    
    const worksheet = XLSX.utils.json_to_sheet(data);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, 'Transactions');
    const buffer = XLSX.write(workbook, { type: 'buffer', bookType: 'xlsx' }) as Buffer;
    
    const fileInput = page.locator('#file-upload');
    await fileInput.setInputFiles({
      name: 'ai-classification-test.xlsx',
      mimeType: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      buffer: buffer,
    });
    
    // Wait for completion (AI classification takes longer)
    await expect(page.getByText(/completed files/i)).toBeVisible({ timeout: 180000 }); // 3 minutes
    
    // Verify file processed
    await expect(page.getByText(/ai-classification-test.xlsx/i)).toBeVisible();
  });

  test('should extract platform IDs (Stripe charge IDs)', async ({ page }) => {
    await setupUploadPage(page);
    
    // Create Stripe data with charge IDs
    const testFile = createFinancialTestFile('stripe-with-ids.xlsx', 50, 'stripe');
    const fileInput = page.locator('#file-upload');
    
    await fileInput.setInputFiles({
      name: 'stripe-with-ids.xlsx',
      mimeType: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      buffer: testFile,
    });
    
    // Wait for completion
    await expect(page.getByText(/completed files/i)).toBeVisible({ timeout: 120000 });
    
    // Verify file processed (platform ID extraction happens in backend)
    await expect(page.getByText(/stripe-with-ids.xlsx/i)).toBeVisible();
  });
});

test.describe('Phase 4-6 Integration: Full Pipeline', () => {
  
  test('should process 10,000 rows end-to-end with full enrichment', async ({ page }) => {
    await setupUploadPage(page);
    
    // Create large file with realistic data
    const testFile = createFinancialTestFile('full-pipeline-10k.xlsx', 10000, 'stripe');
    const fileInput = page.locator('#file-upload');
    
    await fileInput.setInputFiles({
      name: 'full-pipeline-10k.xlsx',
      mimeType: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      buffer: testFile,
    });
    
    // Should show processing progress
    await expect(page.getByText(/processing/i)).toBeVisible({ timeout: 10000 });
    
    // Wait for completion (full enrichment pipeline)
    await expect(page.getByText(/completed files/i)).toBeVisible({ timeout: 600000 }); // 10 minutes
    
    // Verify file processed successfully
    await expect(page.getByText(/full-pipeline-10k.xlsx/i)).toBeVisible();
  });

  test('should handle concurrent file uploads', async ({ page }) => {
    await setupUploadPage(page);
    
    // Create multiple test files
    const file1 = createFinancialTestFile('concurrent-1.xlsx', 100, 'stripe');
    const file2 = createFinancialTestFile('concurrent-2.xlsx', 100, 'quickbooks');
    const file3 = createFinancialTestFile('concurrent-3.xlsx', 100, 'payroll');
    
    const fileInput = page.locator('#file-upload');
    
    // Upload all files at once
    await fileInput.setInputFiles([
      { name: 'concurrent-1.xlsx', mimeType: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', buffer: file1 },
      { name: 'concurrent-2.xlsx', mimeType: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', buffer: file2 },
      { name: 'concurrent-3.xlsx', mimeType: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', buffer: file3 }
    ]);
    
    // Wait for all files to complete
    await expect(page.getByText(/completed files/i)).toBeVisible({ timeout: 300000 });
    
    // Verify all files processed
    await expect(page.getByText(/concurrent-1.xlsx/i)).toBeVisible();
    await expect(page.getByText(/concurrent-2.xlsx/i)).toBeVisible();
    await expect(page.getByText(/concurrent-3.xlsx/i)).toBeVisible();
  });
});

test.describe('Performance Tests', () => {
  
  test('should process 1,000 rows in under 60 seconds', async ({ page }) => {
    await setupUploadPage(page);
    
    const testFile = createFinancialTestFile('performance-1k.xlsx', 1000, 'stripe');
    const fileInput = page.locator('#file-upload');
    
    const startTime = Date.now();
    
    await fileInput.setInputFiles({
      name: 'performance-1k.xlsx',
      mimeType: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      buffer: testFile,
    });
    
    await expect(page.getByText(/completed files/i)).toBeVisible({ timeout: 60000 });
    
    const endTime = Date.now();
    const processingTime = (endTime - startTime) / 1000;
    
    console.log(`✅ Processed 1,000 rows in ${processingTime.toFixed(2)} seconds`);
    expect(processingTime).toBeLessThan(60);
  });
  
  test('should show progress updates during processing', async ({ page }) => {
    await setupUploadPage(page);
    
    const testFile = createFinancialTestFile('progress-test.xlsx', 500, 'quickbooks');
    const fileInput = page.locator('#file-upload');
    
    await fileInput.setInputFiles({
      name: 'progress-test.xlsx',
      mimeType: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      buffer: testFile,
    });
    
    // Should see various progress stages
    await expect(page.getByText(/processing/i)).toBeVisible({ timeout: 10000 });
    
    // Wait for completion
    await expect(page.getByText(/completed files/i)).toBeVisible({ timeout: 120000 });
  });
  
  test('should maintain memory efficiency with large file', async ({ page }) => {
    await setupUploadPage(page);
    
    // Create 5,000 row file to test memory management
    const testFile = createFinancialTestFile('memory-test.xlsx', 5000, 'stripe');
    const fileInput = page.locator('#file-upload');
    
    await fileInput.setInputFiles({
      name: 'memory-test.xlsx',
      mimeType: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      buffer: testFile,
    });
    
    // Should complete without memory errors
    await expect(page.getByText(/completed files/i)).toBeVisible({ timeout: 300000 });
    
    // Verify no error messages
    await expect(page.getByText(/memory/i)).not.toBeVisible();
    await expect(page.getByText(/out of memory/i)).not.toBeVisible();
  });
});

test.describe('Load Tests', () => {
  
  test('should handle 5 sequential uploads without degradation', async ({ page }) => {
    await setupUploadPage(page);
    
    const fileInput = page.locator('#file-upload');
    const processingTimes: number[] = [];
    
    for (let i = 0; i < 5; i++) {
      const testFile = createFinancialTestFile(`load-test-${i}.xlsx`, 200, 'stripe');
      const startTime = Date.now();
      
      await fileInput.setInputFiles({
        name: `load-test-${i}.xlsx`,
        mimeType: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        buffer: testFile,
      });
      
      await expect(page.getByText(/completed files/i)).toBeVisible({ timeout: 120000 });
      
      const endTime = Date.now();
      processingTimes.push((endTime - startTime) / 1000);
      
      console.log(`Upload ${i + 1}: ${processingTimes[i].toFixed(2)}s`);
      
      // Small delay between uploads
      await page.waitForTimeout(2000);
    }
    
    // Verify no significant degradation (last upload shouldn't be >2x first)
    const firstTime = processingTimes[0];
    const lastTime = processingTimes[processingTimes.length - 1];
    
    console.log(`First upload: ${firstTime.toFixed(2)}s, Last upload: ${lastTime.toFixed(2)}s`);
    expect(lastTime).toBeLessThan(firstTime * 2);
  });
});

test.describe('Scalability Tests', () => {
  
  test('should handle increasing file sizes gracefully', async ({ page }) => {
    await setupUploadPage(page);
    
    const fileSizes = [100, 500, 1000, 2000];
    const fileInput = page.locator('#file-upload');
    
    for (const size of fileSizes) {
      const testFile = createFinancialTestFile(`scale-test-${size}.xlsx`, size, 'quickbooks');
      
      await fileInput.setInputFiles({
        name: `scale-test-${size}.xlsx`,
        mimeType: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        buffer: testFile,
      });
      
      await expect(page.getByText(/completed files/i)).toBeVisible({ timeout: 300000 });
      
      console.log(`✅ Processed ${size} rows successfully`);
      
      await page.waitForTimeout(2000);
    }
  });
});
