#!/usr/bin/env python3
"""
Debug script to test advanced processing steps
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from supabase import create_client, Client

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def debug_processing_steps():
    """Debug the advanced processing steps"""
    
    # Initialize Supabase client
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    
    if not supabase_url or not supabase_key:
        print("❌ Missing Supabase credentials")
        return
    
    supabase: Client = create_client(supabase_url, supabase_key)
    
    # Test 1: Check if we have any raw_events
    print("🔍 Testing raw_events...")
    try:
        events_result = supabase.table('raw_events').select('*').limit(5).execute()
        print(f"✅ Found {len(events_result.data)} raw_events")
        
        if events_result.data:
            # Show sample event structure
            sample_event = events_result.data[0]
            print(f"📋 Sample event keys: {list(sample_event.keys())}")
            print(f"📋 Sample payload keys: {list(sample_event.get('payload', {}).keys())}")
            
            # Check for vendor/merchant data
            payload = sample_event.get('payload', {})
            vendor_fields = ['vendor_raw', 'vendor', 'merchant', 'payee', 'description']
            found_vendors = []
            for field in vendor_fields:
                if field in payload and payload[field]:
                    found_vendors.append(f"{field}: {payload[field]}")
            
            if found_vendors:
                print(f"✅ Found vendor data: {found_vendors}")
            else:
                print("❌ No vendor/merchant data found in events")
        else:
            print("❌ No raw_events found")
            
    except Exception as e:
        print(f"❌ Error checking raw_events: {e}")
    
    # Test 2: Check entity extraction
    print("\n🔍 Testing entity extraction...")
    try:
        # Get a sample user_id
        events_result = supabase.table('raw_events').select('user_id').limit(1).execute()
        if events_result.data:
            user_id = events_result.data[0]['user_id']
            file_id = events_result.data[0].get('file_id')
            
            # Simulate entity extraction
            events = supabase.table('raw_events').select('*').eq('user_id', user_id).eq('file_id', file_id).execute()
            
            entities = []
            entity_map = {}
            
            for event in events.data:
                payload = event.get('payload', {})
                vendor_raw = payload.get('vendor_raw') or payload.get('vendor') or payload.get('merchant')
                
                if vendor_raw and vendor_raw not in entity_map:
                    entity = {
                        'entity_type': 'vendor',
                        'canonical_name': vendor_raw,
                        'aliases': [vendor_raw],
                        'email': payload.get('email'),
                        'phone': payload.get('phone'),
                        'bank_account': payload.get('bank_account'),
                        'platform_sources': [event.get('source_platform', 'unknown')],
                        'source_files': [event.get('source_filename', '')],
                        'confidence_score': 0.8
                    }
                    entities.append(entity)
                    entity_map[vendor_raw] = entity
            
            print(f"✅ Extracted {len(entities)} entities")
            if entities:
                print(f"📋 Sample entity: {entities[0]['canonical_name']}")
            else:
                print("❌ No entities extracted - check if events have vendor/merchant data")
                
    except Exception as e:
        print(f"❌ Error in entity extraction: {e}")
    
    # Test 3: Check relationship detection
    print("\n🔍 Testing relationship detection...")
    try:
        from enhanced_relationship_detector import EnhancedRelationshipDetector
        from openai import AsyncOpenAI
        
        openai_key = os.environ.get('OPENAI_API_KEY')
        if not openai_key:
            print("❌ OPENAI_API_KEY not set - relationship detection will fail")
        else:
            openai_client = AsyncOpenAI(api_key=openai_key)
            relationship_detector = EnhancedRelationshipDetector(openai_client, supabase)
            
            # Get a sample user_id
            events_result = supabase.table('raw_events').select('user_id').limit(1).execute()
            if events_result.data:
                user_id = events_result.data[0]['user_id']
                relationship_results = await relationship_detector.detect_all_relationships(user_id)
                print(f"✅ Relationship detection completed: {relationship_results.get('total_relationships', 0)} relationships found")
            else:
                print("❌ No events found for relationship detection")
                
    except Exception as e:
        print(f"❌ Error in relationship detection: {e}")
    
    # Test 4: Check table existence
    print("\n🔍 Testing table existence...")
    tables_to_check = [
        'normalized_entities',
        'entity_matches', 
        'relationship_instances',
        'cross_platform_relationships',
        'processing_transactions',
        'platform_patterns',
        'discovered_platforms',
        'metrics'
    ]
    
    for table in tables_to_check:
        try:
            result = supabase.table(table).select('id').limit(1).execute()
            print(f"✅ {table}: exists")
        except Exception as e:
            print(f"❌ {table}: {e}")
    
    print("\n🎯 Summary:")
    print("1. Check if raw_events have vendor/merchant data")
    print("2. Verify OPENAI_API_KEY is set")
    print("3. Ensure all required tables exist")
    print("4. Check backend logs for specific error messages")

if __name__ == "__main__":
    asyncio.run(debug_processing_steps())
