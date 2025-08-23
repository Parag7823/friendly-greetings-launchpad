# 🧪 STEP-BY-STEP TESTING GUIDE
## Complete Financial Data Processing System Testing

---

## 🚀 **QUICK START - READY TO TEST NOW!**

### **Prerequisites:**
- ✅ Your 10 CSV files ready
- ✅ Postman installed
- ✅ Backend running on `http://localhost:8000`
- ✅ User ID: `550e8400-e29b-41d4-a716-446655440000`

---

## 📋 **PHASE 1: DATABASE CLEANUP (5 minutes)**

### **Step 1: Execute Cleanup SQL**
1. **Open Supabase Dashboard** → SQL Editor
2. **Copy and paste this SQL:**

```sql
-- ⚠️ WARNING: This deletes ALL data for the test user
-- Based on actual database schema from migrations

-- Delete from main tables that actually exist
DELETE FROM public.raw_events WHERE user_id = '550e8400-e29b-41d4-a716-446655440000';
DELETE FROM public.raw_records WHERE user_id = '550e8400-e29b-41d4-a716-446655440000';
DELETE FROM public.metrics WHERE user_id = '550e8400-e29b-41d4-a716-446655440000';
DELETE FROM public.ingestion_jobs WHERE user_id = '550e8400-e29b-41d4-a716-446655440000';

-- Verify cleanup - check the main tables
SELECT
    'raw_events' as table_name, COUNT(*) as remaining_records
FROM public.raw_events WHERE user_id = '550e8400-e29b-41d4-a716-446655440000'
UNION ALL
SELECT
    'raw_records' as table_name, COUNT(*) as remaining_records
FROM public.raw_records WHERE user_id = '550e8400-e29b-41d4-a716-446655440000'
UNION ALL
SELECT
    'metrics' as table_name, COUNT(*) as remaining_records
FROM public.metrics WHERE user_id = '550e8400-e29b-41d4-a716-446655440000';

-- Expected result: All counts should be 0
```

3. **Click "Run"**
4. **Expected Result:** `remaining_events: 0`

### **Step 2: Verify Clean State**
1. **Open Postman**
2. **Import:** `Complete_Finley_AI_Test_Collection.json`
3. **Run:** "5. Debug User Data"
4. **Expected Result:** `test_user_exists: false, total_events: 0`

---

## 📤 **PHASE 2: FRESH FILE UPLOAD (15 minutes)**

### **Upload Your 10 Files:**

**For each file, use this Postman request:**

1. **Method:** `POST`
2. **URL:** `http://localhost:8000/upload-and-process`
3. **Body:** `form-data`
   - `file`: [Select your CSV file]
   - `user_id`: `550e8400-e29b-41d4-a716-446655440000`

### **Upload Order (recommended):**
1. `company_invoices.csv`
2. `comprehensive_vendor_payments.csv`
3. `company_revenue.csv`
4. `comprehensive_cash_flow.csv`
5. `company_expenses.csv`
6. `company_bank_statements.csv`
7. `comprehensive_payroll_data.csv`
8. `company_accounts_receivable.csv`
9. `company_tax_filings.csv`
10. `company_assets.csv`

### **After Each Upload:**
- ✅ **Check Status:** Response should be `200 OK`
- ✅ **Check Message:** Should say "processing completed"
- ✅ **Check Events:** Run "6. Test Raw Events" to see event count increase

### **Final Verification:**
- **Run:** "5. Debug User Data"
- **Expected:** `total_events > 100`, `total_files: 10`, `test_user_exists: true`

---

## 🧪 **PHASE 3: COMPREHENSIVE TESTING (30 minutes)**

### **Testing Sequence:**

#### **A. System Health (2 minutes)**
Run these tests in order:
1. ✅ **Health Check** - Should return `status: "ok"`
2. ✅ **Simple Test** - Should load all dependencies
3. ✅ **Database Test** - Should connect to Supabase
4. ✅ **Environment Debug** - Should show configured environment

#### **B. Data Verification (3 minutes)**
5. ✅ **Debug User Data** - Should show your uploaded data
6. ✅ **Test Raw Events** - Should show events from all 10 files

#### **C. AI Features (10 minutes)**
7. ✅ **Platform Detection** - Should identify payment platforms
8. ✅ **AI Row Classification** - Should classify transaction types
9. ✅ **Dynamic Platform Detection** - Should discover new patterns
10. ✅ **Platform Learning** - Should learn from your data
11. ✅ **Platform Discovery** - Should find platform patterns
12. ✅ **Batch Processing** - Should handle large datasets

#### **D. Data Enrichment (8 minutes)**
13. ✅ **Currency Normalization** - Should convert currencies
14. ✅ **Vendor Standardization** - Should clean vendor names
15. ✅ **Platform ID Extraction** - Should extract platform IDs
16. ✅ **Complete Data Enrichment** - Should enrich all data
17. ✅ **Enrichment Stats** - Should show enrichment metrics
18. ✅ **Vendor Search** - Should find vendors in your data
19. ✅ **Currency Summary** - Should summarize currencies

#### **E. Entity Resolution (3 minutes)**
20. ✅ **Entity Resolution Test** - Should match similar entities
21. ✅ **Entity Search** - Should find entities by name
22. ✅ **Entity Stats** - Should show resolution statistics

#### **F. Relationship Detection (4 minutes)**
23. ✅ **Debug Cross-File Data** - Should show file relationships
24. ✅ **Enhanced Relationship Detection** - Should find cross-file relationships
25. ✅ **Cross-File Relationships** - Should detect file connections

---

## 📊 **PHASE 4: RESULTS VALIDATION**

### **Success Criteria:**

#### **System Health:**
- ✅ All 4 endpoints return `200 OK`
- ✅ No dependency errors
- ✅ Database connection successful

#### **Data Processing:**
- ✅ All 10 files uploaded successfully
- ✅ Total events > 100 (depends on your file sizes)
- ✅ All files appear in debug data

#### **AI Features:**
- ✅ Platform detection finds 3+ platforms
- ✅ Row classification works without errors
- ✅ Dynamic detection discovers patterns

#### **Data Enrichment:**
- ✅ Currency normalization handles multiple currencies
- ✅ Vendor names are standardized
- ✅ Platform IDs are extracted

#### **Entity Resolution:**
- ✅ Similar entities are matched
- ✅ Search returns relevant results

#### **Relationship Detection:**
- ✅ Cross-file relationships found (target: 5+)
- ✅ Enhanced detection works
- ✅ Debug shows file analysis ready

---

## 🔍 **TROUBLESHOOTING GUIDE**

### **Common Issues:**

#### **"No data found for relationship analysis"**
- ❌ **Problem:** Wrong user ID
- ✅ **Solution:** Run "Debug User Data" and use the suggested user ID

#### **"Environment variables not configured"**
- ❌ **Problem:** Missing API keys
- ✅ **Solution:** Check `.env` file has `OPENAI_API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY`

#### **Upload fails with 500 error**
- ❌ **Problem:** File format or size issue
- ✅ **Solution:** Check file is valid CSV, under 10MB

#### **No relationships found**
- ❌ **Problem:** Data doesn't have matching patterns
- ✅ **Solution:** Check "Debug Cross-File Data" to see what's available

#### **Platform detection returns empty**
- ❌ **Problem:** Data doesn't contain recognizable platform patterns
- ✅ **Solution:** This is normal if your data doesn't have Stripe/Razorpay/etc patterns

---

## 🎯 **EXPECTED RESULTS SUMMARY**

### **What Should Work:**
- ✅ **File Upload:** All 10 files process successfully
- ✅ **Data Storage:** Events stored in database
- ✅ **Basic Features:** Health, database, environment checks
- ✅ **Data Enrichment:** Currency, vendor, platform processing

### **What Might Have Limited Results:**
- ⚠️ **Platform Detection:** Depends on your data having platform-specific patterns
- ⚠️ **Cross-File Relationships:** Depends on your files having matching data
- ⚠️ **Entity Resolution:** Depends on having similar entity names

### **What Indicates Success:**
- ✅ **No 500 errors** on core endpoints
- ✅ **Data appears** in debug endpoints
- ✅ **Processing completes** without crashes
- ✅ **Some relationships found** (even if just within-file)

---

## 🚀 **READY TO START?**

1. **Clean database** (Phase 1)
2. **Upload files** (Phase 2)  
3. **Run tests** (Phase 3)
4. **Analyze results** (Phase 4)

**Total Time:** ~50 minutes for complete testing

**Let's begin!** 🎉
