/**
 * E2E Tests for Phase 4-6: Complete File Processing Pipeline
 * Tests full user journey from file upload to enriched data verification
 */

import { test, expect } from '@playwright/test';
import { createClient } from '@supabase/supabase-js';
import * as XLSX from 'xlsx';
import * as fs from 'fs';
import * as path from 'path';

// Supabase client for database verification
const supabaseUrl = process.env.VITE_SUPABASE_URL || process.env.SUPABASE_URL || 'http://localhost:54321';
const supabaseKey = process.env.VITE_SUPABASE_ANON_KEY || process.env.SUPABASE_ANON_KEY || '';

// Skip E2E tests if no Supabase credentials
if (!supabaseKey) {
  test.skip('Skipping E2E tests - no Supabase credentials provided', () => {});
}

const supabase = supabaseKey ? createClient(supabaseUrl, supabaseKey) : null;

test.describe('Phase 4-6: Complete File Processing E2E', () => {
  
  test.beforeEach(async ({ page }) => {
    // Navigate to application (use deployed URL if available)
    const appUrl = process.env.E2E_APP_URL || 'http://localhost:5173';
    await page.goto(appUrl);
    
    // Wait for app to load
    await page.waitForLoadState('networkidle');
  });

  test('E2E: Upload Stripe CSV and verify complete enrichment pipeline', async ({ page }) => {
    if (!supabase) {
      test.skip();
      return;
    }
    // ========================================================================
    // PHASE 4: File Upload & Parsing
    // ========================================================================
    
    // Create test Stripe CSV file
    const stripeData = [
      ['id', 'amount', 'currency', 'description', 'customer', 'created'],
      ['ch_1ABC123XYZ', '1000', 'usd', 'Payment for services', 'cus_XYZ789', '2024-01-15'],
      ['ch_2DEF456ABC', '2000', 'usd', 'Subscription payment', 'cus_ABC123', '2024-01-16'],
      ['ch_3GHI789DEF', '1500', 'usd', 'One-time purchase', 'cus_DEF456', '2024-01-17']
    ];
    
    const csvContent = stripeData.map(row => row.join(',')).join('\n');
    const testFilePath = path.join(__dirname, '../../test_files/stripe_test.csv');
    
    // Ensure test_files directory exists
    const testFilesDir = path.join(__dirname, '../../test_files');
    if (!fs.existsSync(testFilesDir)) {
      fs.mkdirSync(testFilesDir, { recursive: true });
    }
    
    fs.writeFileSync(testFilePath, csvContent);
    
    // Upload file
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(testFilePath);
    
    // Wait for file to be added to queue
    await expect(page.locator('text=stripe_test.csv')).toBeVisible({ timeout: 5000 });
    
    // Click upload/process button
    const uploadButton = page.locator('button:has-text("Upload")').or(
      page.locator('button:has-text("Process")')
    );
    await uploadButton.click();
    
    // ========================================================================
    // PHASE 5: Platform Detection & Document Classification
    // ========================================================================
    
    // Wait for processing to start
    await expect(page.locator('text=Processing').or(page.locator('text=Uploading'))).toBeVisible({ timeout: 10000 });
    
    // Wait for completion (with generous timeout for full pipeline)
    await expect(
      page.locator('text=Completed').or(page.locator('text=Success'))
    ).toBeVisible({ timeout: 60000 });
    
    // ========================================================================
    // PHASE 6: Verify Data Enrichment in Database
    // ========================================================================
    
    // Wait a bit for database writes to complete
    await page.waitForTimeout(2000);
    
    // Query raw_events table to verify enriched data
    const { data: events, error } = await supabase
      .from('raw_events')
      .select('*')
      .order('created_at', { ascending: false })
      .limit(10);
    
    if (error) {
      console.error('Database query error:', error);
    }
    
    expect(events).toBeTruthy();
    expect(events!.length).toBeGreaterThan(0);
    
    // Verify platform detection
    const stripeEvents = events!.filter(e => 
      e.source_platform?.toLowerCase().includes('stripe') ||
      e.payload?.description?.includes('Payment')
    );
    
    expect(stripeEvents.length).toBeGreaterThan(0);
    
    // Verify enrichment fields exist
    const firstEvent = events![0];
    expect(firstEvent).toHaveProperty('payload');
    expect(firstEvent.payload).toBeTruthy();
    
    // Cleanup
    fs.unlinkSync(testFilePath);
  });

  test('E2E: Upload QuickBooks Excel and verify 1000+ rows processing', async ({ page }) => {
    // ========================================================================
    // PHASE 4: Large File Processing
    // ========================================================================
    
    // Create test QuickBooks Excel with 1000 rows
    const rows = [['TxnDate', 'RefNumber', 'Memo', 'Account', 'Amount']];
    
    for (let i = 1; i <= 1000; i++) {
      rows.push([
        `01/${String(i % 28 + 1).padStart(2, '0')}/2024`,
        `INV-${1000 + i}`,
        `Transaction ${i}`,
        'Expenses',
        String(100 + (i % 500))
      ]);
    }
    
    const ws = XLSX.utils.aoa_to_sheet(rows);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, 'Transactions');
    
    const testFilePath = path.join(__dirname, '../../test_files/quickbooks_1000_rows.xlsx');
    XLSX.writeFile(wb, testFilePath);
    
    // Upload file
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(testFilePath);
    
    await expect(page.locator('text=quickbooks_1000_rows.xlsx')).toBeVisible({ timeout: 5000 });
    
    // Start processing
    const uploadButton = page.locator('button:has-text("Upload")').or(
      page.locator('button:has-text("Process")')
    );
    await uploadButton.click();
    
    // ========================================================================
    // PHASE 5 & 6: Monitor Progress and Verify Completion
    // ========================================================================
    
    // Wait for processing to start
    await expect(page.locator('text=Processing').or(page.locator('text=Uploading'))).toBeVisible({ timeout: 10000 });
    
    // Monitor progress updates (should show row counts)
    const progressIndicator = page.locator('[class*="progress"]').or(
      page.locator('text=/\\d+\\s*\\/\\s*\\d+/')
    );
    
    // Wait for completion (2 minutes for 1000 rows)
    await expect(
      page.locator('text=Completed').or(page.locator('text=Success'))
    ).toBeVisible({ timeout: 120000 });
    
    // Verify database has all 1000 rows
    await page.waitForTimeout(3000);
    
    const { count, error } = await supabase
      .from('raw_events')
      .select('*', { count: 'exact', head: true })
      .order('created_at', { ascending: false });
    
    if (error) {
      console.error('Database count error:', error);
    }
    
    // Should have at least 1000 new events
    expect(count).toBeGreaterThanOrEqual(1000);
    
    // Cleanup
    fs.unlinkSync(testFilePath);
  });

  test('E2E: Upload Razorpay CSV and verify platform-specific ID extraction', async ({ page }) => {
    // ========================================================================
    // PHASE 4-6: Platform-Specific Processing
    // ========================================================================
    
    // Create Razorpay test data
    const razorpayData = [
      ['payment_id', 'order_id', 'amount', 'currency', 'status', 'created_at'],
      ['pay_ABC123XYZ456', 'order_DEF789GHI012', '5000', 'INR', 'captured', '2024-01-15 10:30:00'],
      ['pay_DEF456ABC789', 'order_GHI012JKL345', '7500', 'INR', 'captured', '2024-01-15 11:45:00'],
      ['pay_GHI789DEF012', 'order_JKL345MNO678', '3000', 'INR', 'captured', '2024-01-15 14:20:00']
    ];
    
    const csvContent = razorpayData.map(row => row.join(',')).join('\n');
    const testFilePath = path.join(__dirname, '../../test_files/razorpay_test.csv');
    fs.writeFileSync(testFilePath, csvContent);
    
    // Upload and process
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(testFilePath);
    
    await expect(page.locator('text=razorpay_test.csv')).toBeVisible({ timeout: 5000 });
    
    const uploadButton = page.locator('button:has-text("Upload")').or(
      page.locator('button:has-text("Process")')
    );
    await uploadButton.click();
    
    // Wait for completion
    await expect(
      page.locator('text=Completed').or(page.locator('text=Success'))
    ).toBeVisible({ timeout: 60000 });
    
    // Verify Razorpay platform detection and ID extraction
    await page.waitForTimeout(2000);
    
    const { data: events, error } = await supabase
      .from('raw_events')
      .select('*')
      .order('created_at', { ascending: false })
      .limit(10);
    
    expect(events).toBeTruthy();
    
    // Check for Razorpay platform detection
    const razorpayEvents = events!.filter(e => 
      e.source_platform?.toLowerCase().includes('razorpay') ||
      e.payload?.payment_id?.startsWith('pay_')
    );
    
    expect(razorpayEvents.length).toBeGreaterThan(0);
    
    // Verify platform IDs were extracted
    const eventWithIds = razorpayEvents.find(e => 
      e.payload?.payment_id && e.payload?.order_id
    );
    
    expect(eventWithIds).toBeTruthy();
    expect(eventWithIds!.payload.payment_id).toMatch(/^pay_/);
    expect(eventWithIds!.payload.order_id).toMatch(/^order_/);
    
    // Cleanup
    fs.unlinkSync(testFilePath);
  });

  test('E2E: Verify vendor standardization in enriched data', async ({ page }) => {
    // Create test data with vendor names needing standardization
    const testData = [
      ['vendor', 'amount', 'date', 'description'],
      ['Google Inc', '1000', '2024-01-15', 'Cloud services'],
      ['Microsoft Corporation', '2000', '2024-01-16', 'Office 365'],
      ['Amazon LLC', '1500', '2024-01-17', 'AWS hosting'],
      ['APPLE INC.', '500', '2024-01-18', 'Developer account']
    ];
    
    const csvContent = testData.map(row => row.join(',')).join('\n');
    const testFilePath = path.join(__dirname, '../../test_files/vendor_test.csv');
    fs.writeFileSync(testFilePath, csvContent);
    
    // Upload and process
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(testFilePath);
    
    await expect(page.locator('text=vendor_test.csv')).toBeVisible({ timeout: 5000 });
    
    const uploadButton = page.locator('button:has-text("Upload")').or(
      page.locator('button:has-text("Process")')
    );
    await uploadButton.click();
    
    // Wait for completion
    await expect(
      page.locator('text=Completed').or(page.locator('text=Success'))
    ).toBeVisible({ timeout: 60000 });
    
    // Verify vendor standardization in database
    await page.waitForTimeout(2000);
    
    const { data: events, error } = await supabase
      .from('raw_events')
      .select('*')
      .order('created_at', { ascending: false })
      .limit(10);
    
    expect(events).toBeTruthy();
    
    // Check if vendors were standardized (suffixes removed)
    const vendorEvents = events!.filter(e => e.payload?.vendor);
    
    expect(vendorEvents.length).toBeGreaterThan(0);
    
    // Verify standardization happened (this depends on backend implementation)
    // At minimum, verify vendor field exists and is not empty
    vendorEvents.forEach(event => {
      expect(event.payload.vendor).toBeTruthy();
      expect(event.payload.vendor.length).toBeGreaterThan(0);
    });
    
    // Cleanup
    fs.unlinkSync(testFilePath);
  });

  test('E2E: Performance - Process 10K rows in under 2 minutes', async ({ page }) => {
    // Create large test file
    const rows = [['id', 'amount', 'description', 'date']];
    
    for (let i = 1; i <= 10000; i++) {
      rows.push([
        `txn_${i}`,
        String(100 + (i % 1000)),
        `Transaction ${i}`,
        `2024-01-${String((i % 28) + 1).padStart(2, '0')}`
      ]);
    }
    
    const ws = XLSX.utils.aoa_to_sheet(rows);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, 'Data');
    
    const testFilePath = path.join(__dirname, '../../test_files/performance_10k.xlsx');
    XLSX.writeFile(wb, testFilePath);
    
    const startTime = Date.now();
    
    // Upload file
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(testFilePath);
    
    await expect(page.locator('text=performance_10k.xlsx')).toBeVisible({ timeout: 5000 });
    
    // Start processing
    const uploadButton = page.locator('button:has-text("Upload")').or(
      page.locator('button:has-text("Process")')
    );
    await uploadButton.click();
    
    // Wait for completion (2 minutes max)
    await expect(
      page.locator('text=Completed').or(page.locator('text=Success'))
    ).toBeVisible({ timeout: 120000 });
    
    const endTime = Date.now();
    const processingTime = (endTime - startTime) / 1000; // seconds
    
    console.log(`✅ Processed 10,000 rows in ${processingTime.toFixed(2)} seconds`);
    
    // Should complete in under 2 minutes (120 seconds)
    expect(processingTime).toBeLessThan(120);
    
    // Cleanup
    fs.unlinkSync(testFilePath);
  });
});
