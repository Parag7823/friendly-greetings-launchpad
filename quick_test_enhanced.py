#!/usr/bin/env python3
"""
Quick Test for Enhanced Relationship Detection

This script quickly tests the enhanced relationship detection system
to verify it's working with the user's uploaded data.
"""

import asyncio
import os
import sys
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_relationship_detector import EnhancedRelationshipDetector
from openai import AsyncOpenAI
from supabase import create_client

async def quick_test():
    """Quick test of the enhanced relationship detection"""
    
    print("🚀 Quick Test: Enhanced Relationship Detection")
    print("=" * 50)
    
    try:
        # Initialize clients
        openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if not supabase_url or not supabase_key:
            print("❌ Missing Supabase credentials")
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
        
        # Display results
        print("\n📈 RESULTS:")
        print("-" * 30)
        
        if result.get("error"):
            print(f"❌ Error: {result['error']}")
            return False
        
        total_relationships = result.get("total_relationships", 0)
        cross_file_relationships = result.get("cross_file_relationships", 0)
        within_file_relationships = result.get("within_file_relationships", 0)
        
        print(f"✅ Total Relationships: {total_relationships}")
        print(f"📁 Cross-File: {cross_file_relationships}")
        print(f"📄 Within-File: {within_file_relationships}")
        
        # Show sample relationships
        relationships = result.get("relationships", [])
        if relationships:
            print(f"\n🔗 SAMPLE RELATIONSHIPS:")
            print("-" * 30)
            
            for i, rel in enumerate(relationships[:3]):  # Show first 3
                print(f"Relationship {i+1}:")
                print(f"  Type: {rel.get('relationship_type')}")
                print(f"  Confidence: {rel.get('confidence_score', 0):.3f}")
                print(f"  Method: {rel.get('detection_method')}")
                print()
        
        # Success criteria
        if total_relationships > 0:
            print("🎉 SUCCESS: Relationships detected!")
            return True
        else:
            print("❌ FAILED: No relationships detected")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(quick_test())
    sys.exit(0 if success else 1) 