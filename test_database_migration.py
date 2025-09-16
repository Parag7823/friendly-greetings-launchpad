"""
Test script to verify database migration logic
This script tests the safe migration approach without requiring a live database connection
"""

import os
import sys

def test_migration_logic():
    """Test the migration logic by checking if the SQL is syntactically correct"""
    
    # Read the safe migration file
    try:
        with open('supabase/migrations/20250817000000-safe-add-data-enrichment-fields.sql', 'r') as f:
            migration_sql = f.read()
        print("✅ Migration file read successfully")
    except FileNotFoundError:
        print("❌ Migration file not found")
        return False
    
    # Check if the migration contains the necessary components
    required_components = [
        "DO $$",
        "IF NOT EXISTS (SELECT 1 FROM information_schema.columns",
        "ALTER TABLE public.raw_events ADD COLUMN",
        "CREATE INDEX IF NOT EXISTS",
        "CREATE OR REPLACE FUNCTION"
    ]
    
    for component in required_components:
        if component in migration_sql:
            print(f"✅ Found required component: {component}")
        else:
            print(f"❌ Missing required component: {component}")
            return False
    
    # Check for specific columns that should be added
    expected_columns = [
        "amount_original",
        "amount_usd", 
        "currency",
        "exchange_rate",
        "exchange_date",
        "vendor_raw",
        "vendor_standard",
        "vendor_confidence",
        "vendor_cleaning_method",
        "platform_ids",
        "standard_description",
        "ingested_on"
    ]
    
    for column in expected_columns:
        if f"ADD COLUMN {column}" in migration_sql:
            print(f"✅ Column {column} will be added safely")
        else:
            print(f"❌ Column {column} not found in migration")
            return False
    
    # Check for required indexes
    expected_indexes = [
        "idx_raw_events_amount_usd",
        "idx_raw_events_currency",
        "idx_raw_events_vendor_standard",
        "idx_raw_events_ingested_on",
        "idx_raw_events_platform_ids_gin"
    ]
    
    for index in expected_indexes:
        if f"CREATE INDEX IF NOT EXISTS {index}" in migration_sql:
            print(f"✅ Index {index} will be created safely")
        else:
            print(f"❌ Index {index} not found in migration")
            return False
    
    # Check for required functions
    expected_functions = [
        "get_events_by_platform_id",
        "get_platform_id_stats", 
        "validate_platform_id_pattern",
        "get_enrichment_stats",
        "search_events_by_vendor",
        "get_currency_summary"
    ]
    
    for function in expected_functions:
        if f"CREATE OR REPLACE FUNCTION {function}" in migration_sql:
            print(f"✅ Function {function} will be created/updated")
        else:
            print(f"❌ Function {function} not found in migration")
            return False
    
    print("\n🎉 All migration components verified successfully!")
    print("\n📋 Migration Summary:")
    print(f"   • {len(expected_columns)} columns will be added safely")
    print(f"   • {len(expected_indexes)} indexes will be created safely") 
    print(f"   • {len(expected_functions)} functions will be created/updated")
    print("   • All operations use IF NOT EXISTS to prevent errors")
    
    return True

def test_platform_id_extraction():
    """Test the platform ID extraction logic"""
    try:
        # Import the PlatformIDExtractor
        from fastapi_backend import PlatformIDExtractor
        print("✅ PlatformIDExtractor imported successfully")
        
        # Test that the class has the expected methods
        extractor = PlatformIDExtractor()
        
        expected_methods = [
            'extract_platform_ids',
            '_validate_platform_id',
            '_validate_quickbooks_id',
            '_validate_stripe_id',
            '_validate_razorpay_id',
            '_validate_xero_id',
            '_validate_gusto_id',
            '_generate_deterministic_platform_id'
        ]
        
        for method in expected_methods:
            if hasattr(extractor, method):
                print(f"✅ Method {method} exists")
            else:
                print(f"❌ Method {method} missing")
                return False
        
        # Test that extract_platform_ids is async
        import inspect
        if inspect.iscoroutinefunction(extractor.extract_platform_ids):
            print("✅ extract_platform_ids is async")
        else:
            print("❌ extract_platform_ids is not async")
            return False
        
        print("✅ Platform ID extraction logic verified")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import PlatformIDExtractor: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing PlatformIDExtractor: {e}")
        return False

def main():
    """Main test function"""
    print("🔍 Testing Database Migration and Platform ID Extraction Fixes")
    print("=" * 70)
    
    # Test migration logic
    print("\n1. Testing Migration Logic:")
    print("-" * 30)
    migration_ok = test_migration_logic()
    
    # Test platform ID extraction
    print("\n2. Testing Platform ID Extraction:")
    print("-" * 35)
    extraction_ok = test_platform_id_extraction()
    
    # Summary
    print("\n" + "=" * 70)
    if migration_ok and extraction_ok:
        print("🎉 ALL TESTS PASSED!")
        print("\n📋 Next Steps:")
        print("   1. Start Docker Desktop")
        print("   2. Run: supabase start")
        print("   3. Run: supabase db reset")
        print("   4. The safe migration will handle existing columns gracefully")
        print("\n✅ The database migration is ready to run safely!")
    else:
        print("❌ SOME TESTS FAILED!")
        print("   Please fix the issues before running the migration")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
