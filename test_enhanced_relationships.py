#!/usr/bin/env python3
"""
Test script for Enhanced Relationship Detection

This script tests the enhanced relationship detection system to ensure it:
1. Actually finds relationships between events
2. Handles cross-file relationships
3. Handles within-file relationships
4. Provides proper scoring and validation
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Dict, Any

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_relationship_detector import EnhancedRelationshipDetector
from openai import AsyncOpenAI
from supabase import create_client

async def test_enhanced_relationship_detection():
    """Test the enhanced relationship detection system"""
    
    print("🔍 Testing Enhanced Relationship Detection System")
    print("=" * 60)
    
    try:
        # Initialize OpenAI client
        openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Initialize Supabase client
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if not supabase_url or not supabase_key:
            print("❌ Supabase credentials not configured")
            return False
        
        supabase = create_client(supabase_url, supabase_key)
        
        # Test user ID
        user_id = "550e8400-e29b-41d4-a716-446655440000"
        
        print(f"📊 Testing with user ID: {user_id}")
        
        # Initialize Enhanced Relationship Detector
        enhanced_detector = EnhancedRelationshipDetector(openai_client, supabase)
        
        # Detect relationships
        print("🔍 Detecting relationships...")
        result = await enhanced_detector.detect_all_relationships(user_id)
        
        # Analyze results
        print("\n📈 RESULTS ANALYSIS:")
        print("-" * 40)
        
        if result.get("error"):
            print(f"❌ Error: {result['error']}")
            return False
        
        total_relationships = result.get("total_relationships", 0)
        cross_file_relationships = result.get("cross_file_relationships", 0)
        within_file_relationships = result.get("within_file_relationships", 0)
        processing_stats = result.get("processing_stats", {})
        
        print(f"✅ Total Relationships Found: {total_relationships}")
        print(f"📁 Cross-File Relationships: {cross_file_relationships}")
        print(f"📄 Within-File Relationships: {within_file_relationships}")
        print(f"📊 Total Events Processed: {processing_stats.get('total_events', 0)}")
        print(f"📁 Files Analyzed: {processing_stats.get('files_analyzed', 0)}")
        
        # Check relationship types
        relationship_types = processing_stats.get('relationship_types_found', [])
        if relationship_types:
            print(f"🔗 Relationship Types Found: {', '.join(relationship_types)}")
        
        # Show sample relationships
        relationships = result.get("relationships", [])
        if relationships:
            print(f"\n🔗 SAMPLE RELATIONSHIPS:")
            print("-" * 40)
            
            for i, rel in enumerate(relationships[:5]):  # Show first 5
                print(f"Relationship {i+1}:")
                print(f"  Type: {rel.get('relationship_type')}")
                print(f"  Confidence: {rel.get('confidence_score', 0):.3f}")
                print(f"  Method: {rel.get('detection_method')}")
                print(f"  Source: {rel.get('source_file')}")
                print(f"  Target: {rel.get('target_file')}")
                print(f"  Reasoning: {rel.get('reasoning', 'N/A')}")
                print()
        
        # Validate the results
        print("🔍 VALIDATION CHECKS:")
        print("-" * 40)
        
        # Check 1: Are relationships actually found?
        if total_relationships > 0:
            print("✅ PASS: Relationships are being detected")
        else:
            print("❌ FAIL: No relationships detected")
            return False
        
        # Check 2: Are confidence scores valid?
        valid_scores = True
        for rel in relationships:
            score = rel.get('confidence_score', 0)
            if not (0.0 <= score <= 1.0):
                valid_scores = False
                break
        
        if valid_scores:
            print("✅ PASS: All confidence scores are valid (0.0-1.0)")
        else:
            print("❌ FAIL: Invalid confidence scores found")
            return False
        
        # Check 3: Are relationship types diverse?
        if len(relationship_types) > 1:
            print("✅ PASS: Multiple relationship types detected")
        else:
            print("⚠️  WARNING: Only one relationship type detected")
        
        # Check 4: Are both cross-file and within-file relationships found?
        if cross_file_relationships > 0 and within_file_relationships > 0:
            print("✅ PASS: Both cross-file and within-file relationships detected")
        elif cross_file_relationships > 0:
            print("✅ PASS: Cross-file relationships detected")
        elif within_file_relationships > 0:
            print("✅ PASS: Within-file relationships detected")
        else:
            print("❌ FAIL: No relationships detected in either category")
            return False
        
        print("\n🎉 ENHANCED RELATIONSHIP DETECTION TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_relationship_scoring():
    """Test the relationship scoring system"""
    
    print("\n🔍 Testing Relationship Scoring System")
    print("=" * 60)
    
    try:
        # Initialize clients
        openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        supabase = create_client(supabase_url, supabase_key)
        
        # Initialize detector
        detector = EnhancedRelationshipDetector(openai_client, supabase)
        
        # Create test events
        test_source = {
            'id': 'test-source-1',
            'payload': {
                'amount': 1000.0,
                'description': 'Invoice payment to ABC Corp',
                'date': '2024-01-15',
                'entities': {'vendor': ['ABC Corp'], 'invoice_id': ['INV-001']}
            },
            'source_filename': 'company_invoices.csv'
        }
        
        test_target = {
            'id': 'test-target-1',
            'payload': {
                'amount': 1000.0,
                'description': 'Payment to ABC Corp for invoice',
                'date': '2024-01-15',
                'entities': {'vendor': ['ABC Corp'], 'payment_id': ['PAY-001']}
            },
            'source_filename': 'comprehensive_vendor_payments.csv'
        }
        
        # Test scoring
        score = await detector._calculate_relationship_score(
            test_source, test_target, 'invoice_to_payment'
        )
        
        print(f"📊 Test Relationship Score: {score:.3f}")
        
        if 0.0 <= score <= 1.0:
            print("✅ PASS: Relationship scoring works correctly")
            return True
        else:
            print("❌ FAIL: Invalid relationship score")
            return False
            
    except Exception as e:
        print(f"❌ Scoring test failed: {e}")
        return False

async def main():
    """Main test function"""
    
    print("🚀 Starting Enhanced Relationship Detection Tests")
    print("=" * 60)
    
    # Test 1: Full relationship detection
    test1_passed = await test_enhanced_relationship_detection()
    
    # Test 2: Relationship scoring
    test2_passed = await test_relationship_scoring()
    
    # Summary
    print("\n📋 TEST SUMMARY:")
    print("=" * 60)
    print(f"🔍 Full Relationship Detection: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"📊 Relationship Scoring: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED! Enhanced relationship detection is working correctly.")
        return True
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 