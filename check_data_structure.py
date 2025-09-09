#!/usr/bin/env python3
"""
Check the structure of uploaded data to see what fields are available
"""

import os
import sys
import json
from supabase import create_client, Client

def check_data_structure():
    """Check what data structure we have in raw_events"""
    
    # You'll need to set these manually or get them from your environment
    # Replace with your actual Supabase credentials
    SUPABASE_URL = "YOUR_SUPABASE_URL_HERE"
    SUPABASE_KEY = "YOUR_SUPABASE_SERVICE_KEY_HERE"
    
    print("🔍 Checking data structure in raw_events...")
    print("📝 Note: Replace SUPABASE_URL and SUPABASE_KEY with your actual values")
    print("📝 Or set them as environment variables: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY")
    
    if SUPABASE_URL == "YOUR_SUPABASE_URL_HERE":
        print("\n❌ Please update the script with your Supabase credentials")
        print("   Or set environment variables and run: python check_data_structure.py")
        return
    
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Get sample events
        events_result = supabase.table('raw_events').select('*').limit(3).execute()
        
        if not events_result.data:
            print("❌ No raw_events found")
            return
        
        print(f"✅ Found {len(events_result.data)} sample events")
        
        # Analyze the structure
        for i, event in enumerate(events_result.data):
            print(f"\n📋 Event {i+1}:")
            print(f"   ID: {event.get('id')}")
            print(f"   Provider: {event.get('provider')}")
            print(f"   Kind: {event.get('kind')}")
            print(f"   Source Platform: {event.get('source_platform')}")
            print(f"   Category: {event.get('category')}")
            print(f"   Source Filename: {event.get('source_filename')}")
            
            # Check payload structure
            payload = event.get('payload', {})
            print(f"   Payload keys: {list(payload.keys())}")
            
            # Look for vendor/merchant fields
            vendor_fields = ['vendor_raw', 'vendor', 'merchant', 'payee', 'description', 'amount', 'date']
            found_fields = {}
            for field in vendor_fields:
                if field in payload:
                    found_fields[field] = payload[field]
            
            if found_fields:
                print(f"   Found vendor fields: {found_fields}")
            else:
                print(f"   ❌ No vendor/merchant fields found")
                print(f"   Full payload: {json.dumps(payload, indent=2, default=str)}")
        
        # Check what the entity extraction is looking for
        print(f"\n🔍 Entity extraction looks for these fields:")
        print(f"   - payload.vendor_raw")
        print(f"   - payload.vendor") 
        print(f"   - payload.merchant")
        print(f"   - payload.email")
        print(f"   - payload.phone")
        print(f"   - payload.bank_account")
        
        # Check if we have any entities that could be extracted
        all_events = supabase.table('raw_events').select('payload').execute()
        vendor_count = 0
        for event in all_events.data:
            payload = event.get('payload', {})
            if any(field in payload for field in ['vendor_raw', 'vendor', 'merchant']):
                vendor_count += 1
        
        print(f"\n📊 Summary:")
        print(f"   Total events: {len(all_events.data)}")
        print(f"   Events with vendor data: {vendor_count}")
        
        if vendor_count == 0:
            print(f"\n❌ PROBLEM FOUND: No events have vendor/merchant data!")
            print(f"   This is why normalized_entities and entity_matches are empty.")
            print(f"   The entity extraction requires vendor/merchant fields in the payload.")
            print(f"\n💡 Solutions:")
            print(f"   1. Upload a file with vendor/merchant/payee columns")
            print(f"   2. Check if the AI classification is extracting vendor data properly")
            print(f"   3. Verify the platform detection is working correctly")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_data_structure()
