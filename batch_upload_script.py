#!/usr/bin/env python3
"""
Simple Batch Upload Script for Finley AI Testing
Uploads all 10 test files and runs comprehensive tests
"""

import requests
import time
import json
import os

# Configuration
BASE_URL = "https://friendly-greetings-launchpad.onrender.com"
TEST_FILES = [
    "company_bank_statements.csv",
    "company_invoices.csv", 
    "company_expenses.csv",
    "company_revenue.csv",
    "company_tax_records.csv",
    "company_accounts_receivable.csv",
    "comprehensive_payroll_data.csv",
    "comprehensive_vendor_payments.csv",
    "comprehensive_cash_flow.csv",
    "comprehensive_income_statement.csv"
]

def upload_file(filename):
    """Upload a single file to the system"""
    print(f"📤 Uploading: {filename}")
    
    file_path = f"test_files/{filename}"
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (filename, f, 'text/csv')}
            response = requests.post(f"{BASE_URL}/process_excel", files=files)
            
        if response.status_code == 200:
            result = response.json()
            job_id = result.get('job_id')
            print(f"✅ Uploaded successfully! Job ID: {job_id}")
            return job_id
        else:
            print(f"❌ Upload failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error uploading {filename}: {str(e)}")
        return None

def run_advanced_tests():
    """Run all the advanced feature tests"""
    print("\n🧪 Running Advanced Tests...")
    
    tests = [
        ("Currency Normalization", "/test-currency-normalization"),
        ("Vendor Standardization", "/test-vendor-standardization"),
        ("Platform ID Extraction", "/test-platform-id-extraction"),
        ("Data Enrichment", "/test-data-enrichment"),
        ("AI Relationship Detection", "/test-ai-relationship-detection/550e8400-e29b-41d4-a716-446655440000"),
        ("Relationship Discovery", "/test-relationship-discovery/550e8400-e29b-41d4-a716-446655440000"),
        ("Dynamic Platform Detection", "/test-dynamic-platform-detection"),
        ("Platform Learning", "/test-platform-learning/550e8400-e29b-41d4-a716-446655440000"),
        ("Flexible Relationship Engine", "/test-flexible-relationship-discovery/550e8400-e29b-41d4-a716-446655440000"),
        ("Relationship Pattern Learning", "/test-relationship-pattern-learning/550e8400-e29b-41d4-a716-446655440000")
    ]
    
    results = {}
    
    for test_name, endpoint in tests:
        print(f"🔍 Testing: {test_name}")
        try:
            response = requests.get(f"{BASE_URL}{endpoint}")
            if response.status_code == 200:
                print(f"✅ {test_name}: PASSED")
                results[test_name] = "PASSED"
            else:
                print(f"❌ {test_name}: FAILED ({response.status_code})")
                results[test_name] = "FAILED"
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {str(e)}")
            results[test_name] = "ERROR"
        
        time.sleep(1)  # Small delay between tests
    
    return results

def main():
    """Main function to run the complete batch test"""
    print("🚀 FINLEY AI BATCH TEST SUITE")
    print("=" * 50)
    
    # Step 1: Upload all files
    print("\n📁 STEP 1: Uploading All 10 Test Files")
    print("-" * 40)
    
    uploaded_jobs = []
    for i, filename in enumerate(TEST_FILES, 1):
        print(f"\n[{i}/10] Processing: {filename}")
        job_id = upload_file(filename)
        if job_id:
            uploaded_jobs.append(job_id)
        time.sleep(2)  # Delay between uploads
    
    print(f"\n✅ Successfully uploaded {len(uploaded_jobs)} out of {len(TEST_FILES)} files")
    
    # Step 2: Wait for processing
    print("\n⏳ STEP 2: Waiting for Processing to Complete")
    print("-" * 40)
    print("Waiting 30 seconds for all files to be processed...")
    time.sleep(30)
    
    # Step 3: Run advanced tests
    print("\n🧪 STEP 3: Running Advanced Feature Tests")
    print("-" * 40)
    
    test_results = run_advanced_tests()
    
    # Step 4: Summary
    print("\n📊 TEST SUMMARY")
    print("=" * 50)
    print(f"Files Uploaded: {len(uploaded_jobs)}/{len(TEST_FILES)}")
    print(f"Advanced Tests Run: {len(test_results)}")
    
    passed_tests = sum(1 for result in test_results.values() if result == "PASSED")
    failed_tests = sum(1 for result in test_results.values() if result == "FAILED")
    error_tests = sum(1 for result in test_results.values() if result == "ERROR")
    
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"⚠️  Errors: {error_tests}")
    
    if passed_tests == len(test_results):
        print("\n🎉 ALL TESTS PASSED! System is working perfectly!")
    else:
        print(f"\n⚠️  {failed_tests + error_tests} tests had issues. Check the logs above.")
    
    print("\n📋 Detailed Results:")
    for test_name, result in test_results.items():
        status = "✅" if result == "PASSED" else "❌" if result == "FAILED" else "⚠️"
        print(f"  {status} {test_name}: {result}")

if __name__ == "__main__":
    main() 