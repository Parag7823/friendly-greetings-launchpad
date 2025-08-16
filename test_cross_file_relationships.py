#!/usr/bin/env python3
"""
Test script for cross-file relationship detection
Validates that relationships are found between different uploaded files
"""

import asyncio
import os
import requests
import json
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"
TEST_USER_ID = "550e8400-e29b-41d4-a716-446655440000"

def test_endpoint(endpoint_name: str, url: str) -> dict:
    """Test a single endpoint and return results"""
    print(f"\n🧪 Testing: {endpoint_name}")
    print(f"📡 URL: {url}")
    
    try:
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Status: {response.status_code}")
            
            # Extract key metrics
            relationships = result.get('relationships', [])
            if isinstance(relationships, dict):
                relationships = relationships.get('relationships', [])
            
            total_relationships = len(relationships) if relationships else result.get('total_relationships', 0)
            cross_file_count = result.get('cross_file_relationships', 0)
            within_file_count = result.get('within_file_relationships', 0)
            
            print(f"📊 Total Relationships: {total_relationships}")
            print(f"📁 Cross-File: {cross_file_count}")
            print(f"📄 Within-File: {within_file_count}")
            
            # Show sample relationships
            if relationships and len(relationships) > 0:
                print(f"\n🔗 SAMPLE RELATIONSHIPS:")
                for i, rel in enumerate(relationships[:3]):  # Show first 3
                    source_file = rel.get('source_file', 'unknown')
                    target_file = rel.get('target_file', 'unknown')
                    rel_type = rel.get('relationship_type', 'unknown')
                    confidence = rel.get('confidence_score', 0)
                    method = rel.get('detection_method', 'unknown')
                    
                    print(f"  {i+1}. {source_file} → {target_file}")
                    print(f"     Type: {rel_type}")
                    print(f"     Confidence: {confidence:.3f}")
                    print(f"     Method: {method}")
            else:
                print("❌ No relationships found")
            
            return result
            
        else:
            print(f"❌ Status: {response.status_code}")
            print(f"❌ Error: {response.text}")
            return {"error": f"HTTP {response.status_code}", "details": response.text}
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return {"error": str(e)}

def test_debug_data() -> dict:
    """Test the debug endpoint to see what data is available"""
    print("\n" + "="*60)
    print("🔍 DEBUGGING AVAILABLE DATA")
    print("="*60)
    
    url = f"{BASE_URL}/debug-cross-file-data/{TEST_USER_ID}"
    result = test_endpoint("Debug Cross-File Data", url)
    
    if 'files' in result:
        print(f"\n📁 AVAILABLE FILES:")
        for file_info in result['files']:
            filename = file_info.get('filename', 'unknown')
            event_count = file_info.get('event_count', 0)
            print(f"  📄 {filename}: {event_count} events")
    
    if 'potential_cross_file_relationships' in result:
        print(f"\n🔗 POTENTIAL CROSS-FILE RELATIONSHIPS:")
        for potential in result['potential_cross_file_relationships']:
            source = potential.get('source_file')
            target = potential.get('target_file')
            can_analyze = potential.get('can_analyze', False)
            source_events = potential.get('source_events', 0)
            target_events = potential.get('target_events', 0)
            
            status = "✅ READY" if can_analyze else "❌ MISSING"
            print(f"  {status} {source} ({source_events}) ↔ {target} ({target_events})")
    
    return result

def test_all_relationship_endpoints():
    """Test all relationship detection endpoints"""
    print("🚀 COMPREHENSIVE CROSS-FILE RELATIONSHIP TESTING")
    print("="*60)
    
    # First, debug what data is available
    debug_result = test_debug_data()
    
    # Test endpoints in order
    endpoints = [
        ("Enhanced Relationship Detection", f"{BASE_URL}/test-enhanced-relationship-detection/{TEST_USER_ID}"),
        ("Cross-File Relationships", f"{BASE_URL}/test-cross-file-relationships/{TEST_USER_ID}"),
        ("AI Relationship Detection", f"{BASE_URL}/test-ai-relationship-detection/{TEST_USER_ID}"),
        ("Relationship Discovery", f"{BASE_URL}/test-relationship-discovery/{TEST_USER_ID}"),
        ("AI Relationship Scoring", f"{BASE_URL}/test-ai-relationship-scoring/{TEST_USER_ID}"),
        ("Relationship Validation", f"{BASE_URL}/test-relationship-validation/{TEST_USER_ID}")
    ]
    
    results = {}
    
    for name, url in endpoints:
        print("\n" + "="*60)
        print(f"🧪 TESTING: {name.upper()}")
        print("="*60)
        
        result = test_endpoint(name, url)
        results[name] = result
        
        # Add delay between tests
        import time
        time.sleep(2)
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY REPORT")
    print("="*60)
    
    total_files = debug_result.get('total_files', 0)
    total_events = debug_result.get('total_events', 0)
    analysis_ready = debug_result.get('analysis_ready', 0)
    
    print(f"📁 Total Files: {total_files}")
    print(f"📊 Total Events: {total_events}")
    print(f"🔗 Cross-File Patterns Ready: {analysis_ready}")
    
    print(f"\n🧪 ENDPOINT RESULTS:")
    for name, result in results.items():
        if 'error' in result:
            print(f"  ❌ {name}: ERROR - {result['error']}")
        else:
            relationships = result.get('relationships', [])
            if isinstance(relationships, dict):
                relationships = relationships.get('relationships', [])
            
            total_rels = len(relationships) if relationships else result.get('total_relationships', 0)
            cross_file = result.get('cross_file_relationships', 0)
            
            if total_rels > 0:
                print(f"  ✅ {name}: {total_rels} relationships ({cross_file} cross-file)")
            else:
                print(f"  ⚠️  {name}: No relationships found")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if total_files < 2:
        print("  📁 Upload more files to enable cross-file relationship detection")
    elif analysis_ready == 0:
        print("  🔧 No matching file patterns found - check file naming conventions")
    elif all('error' in result for result in results.values()):
        print("  🚨 All endpoints failed - check server logs and configuration")
    else:
        successful_tests = [name for name, result in results.items() if 'error' not in result]
        if successful_tests:
            print(f"  ✅ Working endpoints: {', '.join(successful_tests)}")
        
        failed_tests = [name for name, result in results.items() if 'error' in result]
        if failed_tests:
            print(f"  ❌ Failed endpoints: {', '.join(failed_tests)}")

if __name__ == "__main__":
    test_all_relationship_endpoints()
