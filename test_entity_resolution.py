#!/usr/bin/env python3
"""
Test script for Entity Resolution System
Tests cross-platform entity matching and resolution
"""

import requests
import json

def test_entity_resolution():
    """Test the Entity Resolution system"""
    
    url = "https://friendly-greetings-launchpad.onrender.com/test-entity-resolution"
    
    try:
        print("🔍 Testing Entity Resolution System...")
        print("=" * 60)
        
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status: {response.status_code}")
            print(f"📊 Message: {data.get('message', 'N/A')}")
            print(f"🔢 Total Tests: {data.get('total_tests', 0)}")
            print(f"✅ Successful Tests: {data.get('successful_tests', 0)}")
            print(f"❌ Failed Tests: {data.get('failed_tests', 0)}")
            print()
            
            # Display test results
            for i, test_result in enumerate(data.get('test_results', []), 1):
                print(f"📋 Test {i}: {test_result['test_case']}")
                print(f"   Description: {test_result['description']}")
                print(f"   Platform: {test_result['platform']}")
                print(f"   Success: {'✅' if test_result['success'] else '❌'}")
                
                if test_result['success']:
                    resolution = test_result.get('resolution_result', {})
                    print(f"   🤖 Resolution Results:")
                    print(f"      Total Resolved: {resolution.get('total_resolved', 0)}")
                    print(f"      Total Attempted: {resolution.get('total_attempted', 0)}")
                    
                    resolved_entities = resolution.get('resolved_entities', {})
                    for entity_type, entities in resolved_entities.items():
                        if entities:
                            print(f"      {entity_type.title()}: {len(entities)} entities")
                            for entity in entities[:3]:  # Show first 3
                                print(f"         - {entity['name']} → {entity['resolved_name']}")
                else:
                    print(f"   ❌ Error: {test_result.get('error', 'Unknown error')}")
                
                print()
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")

def test_entity_search():
    """Test entity search functionality"""
    
    url = "https://friendly-greetings-launchpad.onrender.com/test-entity-search/test-user-123"
    params = {
        'search_term': 'Abhishek',
        'entity_type': 'employee'
    }
    
    try:
        print("🔍 Testing Entity Search...")
        print("=" * 60)
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status: {response.status_code}")
            print(f"📊 Message: {data.get('message', 'N/A')}")
            print(f"🔍 Search Term: {data.get('search_term', 'N/A')}")
            print(f"🏷️ Entity Type: {data.get('entity_type', 'N/A')}")
            print(f"👤 User ID: {data.get('user_id', 'N/A')}")
            print(f"📊 Total Results: {data.get('total_results', 0)}")
            print()
            
            # Display search results
            for i, result in enumerate(data.get('results', []), 1):
                print(f"📋 Result {i}:")
                print(f"   ID: {result.get('id', 'N/A')}")
                print(f"   Entity Type: {result.get('entity_type', 'N/A')}")
                print(f"   Canonical Name: {result.get('canonical_name', 'N/A')}")
                print(f"   Aliases: {result.get('aliases', [])}")
                print(f"   Email: {result.get('email', 'N/A')}")
                print(f"   Platform Sources: {result.get('platform_sources', [])}")
                print(f"   Confidence: {result.get('confidence_score', 'N/A')}")
                print(f"   Similarity: {result.get('similarity_score', 'N/A')}")
                print()
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")

def test_entity_stats():
    """Test entity resolution statistics"""
    
    url = "https://friendly-greetings-launchpad.onrender.com/test-entity-stats/test-user-123"
    
    try:
        print("📊 Testing Entity Resolution Statistics...")
        print("=" * 60)
        
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status: {response.status_code}")
            print(f"📊 Message: {data.get('message', 'N/A')}")
            print(f"👤 User ID: {data.get('user_id', 'N/A')}")
            print(f"✅ Success: {data.get('success', False)}")
            print()
            
            stats = data.get('stats', {})
            if stats:
                print(f"📈 Entity Resolution Statistics:")
                print(f"   Total Entities: {stats.get('total_entities', 0)}")
                print(f"   Employees: {stats.get('employees_count', 0)}")
                print(f"   Vendors: {stats.get('vendors_count', 0)}")
                print(f"   Customers: {stats.get('customers_count', 0)}")
                print(f"   Projects: {stats.get('projects_count', 0)}")
                print(f"   Total Matches: {stats.get('total_matches', 0)}")
                print(f"   Exact Matches: {stats.get('exact_matches', 0)}")
                print(f"   Fuzzy Matches: {stats.get('fuzzy_matches', 0)}")
                print(f"   Email Matches: {stats.get('email_matches', 0)}")
                print(f"   Bank Matches: {stats.get('bank_matches', 0)}")
                print(f"   New Entities: {stats.get('new_entities', 0)}")
                print(f"   Avg Confidence: {stats.get('avg_confidence', 0)}")
            else:
                print("❌ No statistics available")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")

if __name__ == "__main__":
    print("🚀 Finley AI - Entity Resolution System Test")
    print("=" * 60)
    print()
    
    # Test entity resolution
    test_entity_resolution()
    print()
    
    # Test entity search
    test_entity_search()
    print()
    
    # Test entity stats
    test_entity_stats()
    print()
    
    print("✅ Entity Resolution testing completed!") 