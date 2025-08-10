"""
Test script for optimized relationship detection
"""

import asyncio
import os
from optimized_relationship_detector import OptimizedAIRelationshipDetector
from openai import OpenAI
from supabase import create_client

async def test_optimized_relationship_detection():
    """Test the optimized relationship detector"""
    
    # Initialize clients
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
    
    if not supabase_url or not supabase_key:
        print("❌ Supabase credentials not configured")
        return
    
    supabase = create_client(supabase_url, supabase_key)
    
    # Initialize optimized detector
    detector = OptimizedAIRelationshipDetector(openai_client, supabase)
    
    # Test user ID
    user_id = "550e8400-e29b-41d4-a716-446655440000"
    
    print("🚀 Testing Optimized AI Relationship Detection...")
    print(f"User ID: {user_id}")
    
    try:
        # Run the optimized detection
        result = await detector.detect_all_relationships(user_id)
        
        print("✅ Optimized Relationship Detection Completed!")
        print(f"Total Relationships: {result.get('total_relationships', 0)}")
        print(f"Relationship Types: {result.get('relationship_types', [])}")
        print(f"Processing Stats: {result.get('processing_stats', {})}")
        print(f"Message: {result.get('message', '')}")
        
        # Show sample relationships
        relationships = result.get('relationships', [])
        if relationships:
            print(f"\n📊 Sample Relationships (showing first 5):")
            for i, rel in enumerate(relationships[:5]):
                print(f"  {i+1}. {rel.get('relationship_type')} - Score: {rel.get('confidence_score', 0):.2f}")
                print(f"     Source: {rel.get('source_event_id')[:8]}...")
                print(f"     Target: {rel.get('target_event_id')[:8]}...")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(test_optimized_relationship_detection()) 