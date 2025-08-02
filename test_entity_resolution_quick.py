#!/usr/bin/env python3
"""
Quick Entity Resolution Test
Validates the Entity Resolution system is working before running full tests
"""

import requests
import json

def quick_entity_resolution_test():
    """Quick test of Entity Resolution system"""
    
    url = "https://friendly-greetings-launchpad.onrender.com/test-entity-resolution"
    
    try:
        print("🔍 Quick Entity Resolution Test...")
        print("=" * 50)
        
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status: {response.status_code}")
            print(f"📊 Message: {data.get('message', 'N/A')}")
            print(f"🔢 Total Tests: {data.get('total_tests', 0)}")
            print(f"✅ Successful Tests: {data.get('successful_tests', 0)}")
            print(f"❌ Failed Tests: {data.get('failed_tests', 0)}")
            
            if data.get('successful_tests', 0) > 0:
                print("\n✅ Entity Resolution system is working!")
                print("🎯 Ready to run full Postman collection")
                return True
            else:
                print("\n❌ Entity Resolution system has issues")
                print("🔧 Check database migration and API endpoints")
                return False
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return False

def test_entity_search():
    """Test entity search functionality"""
    
    url = "https://friendly-greetings-launchpad.onrender.com/test-entity-search/test-user-123"
    params = {'search_term': 'Abhishek', 'entity_type': 'employee'}
    
    try:
        print("\n🔍 Testing Entity Search...")
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Entity Search working")
            print(f"📊 Total Results: {data.get('total_results', 0)}")
            return True
        else:
            print(f"❌ Entity Search failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Entity Search error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Finley AI - Quick Entity Resolution Test")
    print("=" * 50)
    
    # Test Entity Resolution
    er_working = quick_entity_resolution_test()
    
    # Test Entity Search
    es_working = test_entity_search()
    
    print("\n" + "=" * 50)
    if er_working and es_working:
        print("✅ All quick tests passed!")
        print("🎯 Ready to run full Postman collection")
        print("📋 Import 'Finley_AI_Complete_Test_Collection.json' into Postman")
        print("📁 Use test files from 'test_files/' directory")
    else:
        print("❌ Some quick tests failed")
        print("🔧 Check database migration and API setup")
        print("📋 Apply Entity Resolution migration SQL first")
    
    print("\n📖 See 'COMPREHENSIVE_TEST_GUIDE.md' for full testing instructions") 