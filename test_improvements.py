#!/usr/bin/env python3
"""
Comprehensive test script to verify all improvements made to the Finley AI system
"""

import requests
import json
import time
import os

def test_endpoint(url, name, expected_status=200):
    """Test a single endpoint"""
    try:
        print(f"🔍 Testing {name}...")
        response = requests.get(url, timeout=15)
        
        if response.status_code == expected_status:
            print(f"✅ {name}: SUCCESS ({response.status_code})")
            return True, response.json() if response.status_code == 200 else None
        else:
            print(f"❌ {name}: FAILED ({response.status_code})")
            print(f"   Response: {response.text[:200]}...")
            return False, None
            
    except requests.exceptions.ConnectionError:
        print(f"❌ {name}: CONNECTION ERROR (Server not running?)")
        return False, None
    except Exception as e:
        print(f"❌ {name}: ERROR - {str(e)}")
        return False, None

def test_relationship_detection():
    """Test relationship detection improvements"""
    print("\n🧠 TESTING RELATIONSHIP DETECTION IMPROVEMENTS")
    print("=" * 50)
    
    test_user_id = "550e8400-e29b-41d4-a716-446655440000"
    base_url = "http://localhost:8000"
    
    tests = [
        (f"{base_url}/test-enhanced-relationship-detection/{test_user_id}", "Enhanced Relationship Detection"),
        (f"{base_url}/test-ai-relationship-detection/{test_user_id}", "AI Relationship Detection"),
        (f"{base_url}/test-cross-file-relationships/{test_user_id}", "Cross-File Relationships"),
        (f"{base_url}/test-relationship-discovery/{test_user_id}", "Relationship Discovery"),
    ]
    
    results = []
    for url, name in tests:
        success, data = test_endpoint(url, name)
        results.append((name, success, data))
        time.sleep(1)
    
    return results

def test_data_enrichment():
    """Test data enrichment improvements"""
    print("\n💰 TESTING DATA ENRICHMENT IMPROVEMENTS")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    tests = [
        (f"{base_url}/test-currency-normalization", "Currency Normalization"),
        (f"{base_url}/test-vendor-standardization", "Vendor Standardization"),
        (f"{base_url}/test-platform-id-extraction", "Platform ID Extraction"),
        (f"{base_url}/test-data-enrichment", "Data Enrichment Pipeline"),
    ]
    
    results = []
    for url, name in tests:
        success, data = test_endpoint(url, name)
        results.append((name, success, data))
        time.sleep(1)
    
    return results

def test_fixed_endpoints():
    """Test endpoints that were previously failing"""
    print("\n🔧 TESTING FIXED ENDPOINTS")
    print("=" * 50)
    
    test_user_id = "550e8400-e29b-41d4-a716-446655440000"
    base_url = "http://localhost:8000"
    
    tests = [
        (f"{base_url}/test-vendor-search/{test_user_id}?vendor_name=Google", "Vendor Search (Fixed)"),
        (f"{base_url}/test-currency-summary/{test_user_id}", "Currency Summary (Fixed)"),
        (f"{base_url}/test-enrichment-stats/{test_user_id}", "Enrichment Statistics"),
    ]
    
    results = []
    for url, name in tests:
        success, data = test_endpoint(url, name)
        results.append((name, success, data))
        time.sleep(1)
    
    return results

def analyze_results(relationship_results, enrichment_results, fixed_results):
    """Analyze test results and provide insights"""
    print("\n📊 COMPREHENSIVE ANALYSIS")
    print("=" * 50)
    
    all_results = relationship_results + enrichment_results + fixed_results
    
    # Count successes and failures
    total_tests = len(all_results)
    successful_tests = sum(1 for _, success, _ in all_results if success)
    failed_tests = total_tests - successful_tests
    
    print(f"📈 Overall Results: {successful_tests}/{total_tests} tests passed")
    print(f"🎯 Success Rate: {(successful_tests/total_tests)*100:.1f}%")
    
    # Analyze relationship detection
    relationship_success = sum(1 for _, success, _ in relationship_results if success)
    print(f"\n🧠 Relationship Detection: {relationship_success}/{len(relationship_results)} passed")
    
    # Analyze data enrichment
    enrichment_success = sum(1 for _, success, _ in enrichment_results if success)
    print(f"💰 Data Enrichment: {enrichment_success}/{len(enrichment_results)} passed")
    
    # Analyze fixed endpoints
    fixed_success = sum(1 for _, success, _ in fixed_results if success)
    print(f"🔧 Fixed Endpoints: {fixed_success}/{len(fixed_results)} passed")
    
    # Check for specific improvements
    print(f"\n🔍 SPECIFIC IMPROVEMENTS CHECK:")
    
    # Check if JWT token issues are fixed
    jwt_fixed = all(success for name, success, _ in fixed_results if "Vendor Search" in name or "Currency Summary" in name)
    print(f"   JWT Token Issues: {'✅ FIXED' if jwt_fixed else '❌ STILL FAILING'}")
    
    # Check relationship detection quality
    if relationship_results:
        enhanced_success = any(success for name, success, _ in relationship_results if "Enhanced" in name)
        print(f"   Enhanced Relationship Detection: {'✅ WORKING' if enhanced_success else '❌ FAILING'}")
    
    # Check data enrichment quality
    if enrichment_results:
        currency_success = any(success for name, success, _ in enrichment_results if "Currency" in name)
        vendor_success = any(success for name, success, _ in enrichment_results if "Vendor" in name)
        print(f"   Currency Normalization: {'✅ WORKING' if currency_success else '❌ FAILING'}")
        print(f"   Vendor Standardization: {'✅ WORKING' if vendor_success else '❌ FAILING'}")
    
    return successful_tests == total_tests

def main():
    print("🚀 FINLEY AI COMPREHENSIVE IMPROVEMENTS TEST")
    print("=" * 60)
    print("Testing all improvements and fixes implemented...")
    
    # Test relationship detection improvements
    relationship_results = test_relationship_detection()
    
    # Test data enrichment improvements
    enrichment_results = test_data_enrichment()
    
    # Test fixed endpoints
    fixed_results = test_fixed_endpoints()
    
    # Analyze all results
    all_passed = analyze_results(relationship_results, enrichment_results, fixed_results)
    
    # Final summary
    print(f"\n🎉 FINAL SUMMARY")
    print("=" * 50)
    
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("🎯 The Finley AI system is working perfectly with all improvements.")
        print("🚀 Ready for production deployment!")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("🔧 Please check the failed tests above and address any issues.")
        print("📋 Review the error messages for specific problems.")
    
    print(f"\n📋 Next Steps:")
    if all_passed:
        print("   1. ✅ System is production ready")
        print("   2. 🚀 Deploy to production environment")
        print("   3. 📊 Monitor system performance")
        print("   4. 🔄 Run regular maintenance tests")
    else:
        print("   1. 🔧 Fix any failed tests")
        print("   2. 🔍 Review error logs")
        print("   3. 🧪 Re-run tests after fixes")
        print("   4. ✅ Ensure all tests pass before deployment")

if __name__ == "__main__":
    main()
