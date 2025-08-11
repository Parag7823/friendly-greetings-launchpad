#!/usr/bin/env python3
"""
Quick test script to verify endpoints are working
"""

import requests
import json
import time

def test_endpoint(url, name):
    """Test a single endpoint"""
    try:
        print(f"🔍 Testing {name}...")
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            print(f"✅ {name}: SUCCESS (200)")
            return True
        else:
            print(f"❌ {name}: FAILED ({response.status_code})")
            print(f"   Response: {response.text[:200]}...")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ {name}: CONNECTION ERROR (Server not running?)")
        return False
    except Exception as e:
        print(f"❌ {name}: ERROR - {str(e)}")
        return False

def main():
    print("🧪 Finley AI Endpoint Test")
    print("=" * 40)
    
    # Test user ID (replace with your actual test user ID)
    test_user_id = "550e8400-e29b-41d4-a716-446655440000"
    base_url = "http://localhost:8000"
    
    # List of endpoints to test
    endpoints = [
        (f"{base_url}/test-cross-file-relationships/{test_user_id}", "Cross-File Relationships"),
        (f"{base_url}/test-enhanced-relationship-detection/{test_user_id}", "Enhanced Relationship Detection"),
        (f"{base_url}/test-ai-relationship-detection/{test_user_id}", "AI Relationship Detection"),
        (f"{base_url}/health", "Health Check"),
    ]
    
    print(f"🌐 Testing against: {base_url}")
    print(f"👤 Test User ID: {test_user_id}")
    print()
    
    # Test each endpoint
    results = []
    for url, name in endpoints:
        result = test_endpoint(url, name)
        results.append((name, result))
        time.sleep(1)  # Small delay between requests
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 TEST SUMMARY")
    print("=" * 40)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\n🎯 Results: {passed}/{total} endpoints working")
    
    if passed == total:
        print("🎉 All endpoints are working! The 404 error should be resolved.")
    else:
        print("⚠️  Some endpoints are still failing. Check the server logs.")

if __name__ == "__main__":
    main()
