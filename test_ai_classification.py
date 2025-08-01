#!/usr/bin/env python3
"""
Test script for AI-powered row classification
Demonstrates semantic understanding and entity extraction
"""

import requests
import json

def test_ai_row_classification():
    """Test the AI row classification endpoint"""
    
    url = "https://friendly-greetings-launchpad.onrender.com/test-ai-row-classification"
    
    try:
        print("🧠 Testing AI-Powered Row Classification...")
        print("=" * 60)
        
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status: {response.status_code}")
            print(f"📊 Message: {data.get('message', 'N/A')}")
            print(f"🔢 Total Tests: {data.get('total_tests', 0)}")
            print()
            
            # Display test results
            for i, test_result in enumerate(data.get('test_results', []), 1):
                print(f"📋 Test {i}: {test_result['test_case']}")
                print(f"   Description: {test_result['description']}")
                print(f"   Row Data: {test_result['row_data']}")
                
                ai_class = test_result.get('ai_classification', {})
                print(f"   🤖 AI Classification:")
                print(f"      Row Type: {ai_class.get('row_type', 'N/A')}")
                print(f"      Category: {ai_class.get('category', 'N/A')}")
                print(f"      Subcategory: {ai_class.get('subcategory', 'N/A')}")
                print(f"      Confidence: {ai_class.get('confidence', 'N/A')}")
                print(f"      Description: {ai_class.get('description', 'N/A')}")
                print(f"      Reasoning: {ai_class.get('reasoning', 'N/A')}")
                
                # Show extracted entities
                entities = ai_class.get('entities', {})
                if entities:
                    print(f"      👥 Entities:")
                    for entity_type, entity_list in entities.items():
                        if entity_list:
                            print(f"         {entity_type}: {entity_list}")
                
                # Show relationships
                relationships = ai_class.get('relationships', {})
                if relationships:
                    print(f"      🔗 Relationships:")
                    for rel_type, rel_id in relationships.items():
                        if rel_id:
                            print(f"         {rel_type}: {rel_id}")
                
                print()
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")

def test_platform_detection():
    """Test the enhanced platform detection"""
    
    url = "https://friendly-greetings-launchpad.onrender.com/test-platform-detection"
    
    try:
        print("🔍 Testing Enhanced Platform Detection...")
        print("=" * 60)
        
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status: {response.status_code}")
            print(f"📊 Message: {data.get('message', 'N/A')}")
            print(f"🔢 Total Tests: {data.get('total_tests', 0)}")
            print()
            
            # Display platform detection results
            for i, test_result in enumerate(data.get('test_results', []), 1):
                print(f"📋 Test {i}: {test_result['test_case']}")
                print(f"   Filename: {test_result['filename']}")
                
                platform_info = test_result.get('platform_detection', {})
                print(f"   🎯 Platform Detection:")
                print(f"      Platform: {platform_info.get('platform', 'N/A')}")
                print(f"      Confidence: {platform_info.get('confidence', 'N/A')}")
                print(f"      Description: {platform_info.get('description', 'N/A')}")
                print(f"      Reasoning: {platform_info.get('reasoning', 'N/A')}")
                
                # Show AI row classification
                ai_class = test_result.get('ai_row_classification', {})
                if ai_class:
                    print(f"   🤖 AI Row Classification:")
                    print(f"      Row Type: {ai_class.get('row_type', 'N/A')}")
                    print(f"      Category: {ai_class.get('category', 'N/A')}")
                    print(f"      Confidence: {ai_class.get('confidence', 'N/A')}")
                
                print()
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")

if __name__ == "__main__":
    print("🚀 Finley AI - Enhanced Row Classification Test")
    print("=" * 60)
    print()
    
    # Test AI row classification
    test_ai_row_classification()
    print()
    
    # Test platform detection
    test_platform_detection()
    print()
    
    print("✅ Testing completed!") 