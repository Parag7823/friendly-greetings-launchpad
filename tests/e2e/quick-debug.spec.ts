import { test, expect } from '@playwright/test';

test('Debug: Check what page loads', async ({ page }) => {
  console.log('1. Going to home page...');
  await page.goto('/');
  await page.waitForTimeout(2000);
  
  console.log('2. Taking screenshot of home page...');
  await page.screenshot({ path: 'debug-home.png', fullPage: true });
  
  console.log('3. Looking for Get Started button...');
  const getStartedButton = page.getByRole('button', { name: /get started/i });
  const buttonVisible = await getStartedButton.isVisible().catch(() => false);
  console.log(`Get Started button visible: ${buttonVisible}`);
  
  if (buttonVisible) {
    console.log('4. Clicking Get Started...');
    await getStartedButton.click();
    await page.waitForTimeout(3000);
    
    console.log('5. Taking screenshot after auth...');
    await page.screenshot({ path: 'debug-after-auth.png', fullPage: true });
    
    console.log('6. Going to /upload...');
    await page.goto('/upload');
    await page.waitForTimeout(2000);
    
    console.log('7. Taking screenshot of upload page...');
    await page.screenshot({ path: 'debug-upload.png', fullPage: true });
    
    console.log('8. Looking for file input...');
    const fileInput = page.locator('#file-upload');
    const inputExists = await fileInput.count();
    console.log(`File input count: ${inputExists}`);
    
    if (inputExists > 0) {
      console.log('9. Trying to upload a test file...');
      const testFile = Buffer.from('Date,Amount\n2024-01-01,100', 'utf-8');
      await fileInput.setInputFiles({
        name: 'test.csv',
        mimeType: 'text/csv',
        buffer: testFile,
      });
      console.log('File uploaded successfully!');
      await page.waitForTimeout(3000);
      await page.screenshot({ path: 'debug-after-upload.png', fullPage: true });
    }
  }
  
  console.log('Debug test complete!');
});
