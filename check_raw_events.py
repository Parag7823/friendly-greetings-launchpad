#!/usr/bin/env python3
"""
Check what's actually in the raw_events to see why entity extraction is failing
"""

import os
import json
from supabase import create_client, Client

def check_raw_events():
    """Check what's in raw_events"""
    
    # You need to set these environment variables or replace with your values
    SUPABASE_URL = os.environ.get("SUPABASE_URL")
    SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("❌ Please set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables")
        print("   Or replace the values in this script")
        return
    
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Get sample events
        events_result = supabase.table('raw_events').select('*').limit(3).execute()
        
        if not events_result.data:
            print("❌ No raw_events found")
            return
        
        print(f"✅ Found {len(events_result.data)} sample events")
        
        for i, event in enumerate(events_result.data):
            print(f"\n📋 Event {i+1}:")
            print(f"   ID: {event.get('id')}")
            print(f"   Provider: {event.get('provider')}")
            print(f"   Kind: {event.get('kind')}")
            print(f"   Category: {event.get('category')}")
            print(f"   Source Platform: {event.get('source_platform')}")
            print(f"   Source Filename: {event.get('source_filename')}")
            
            # Check payload structure
            payload = event.get('payload', {})
            print(f"   Payload keys: {list(payload.keys())}")
            
            # Show the actual payload content
            print(f"   Payload content:")
            for key, value in payload.items():
                if isinstance(value, (str, int, float)) and len(str(value)) < 100:
                    print(f"     {key}: {value}")
                else:
                    print(f"     {key}: [complex data - {type(value).__name__}]")
            
            # Check if we have vendor-like data
            vendor_like_fields = []
            for key, value in payload.items():
                if any(term in key.lower() for term in ['vendor', 'merchant', 'client', 'customer', 'payee', 'name']):
                    vendor_like_fields.append(f"{key}: {value}")
            
            if vendor_like_fields:
                print(f"   🎯 Vendor-like fields found: {vendor_like_fields}")
            else:
                print(f"   ❌ No vendor-like fields found")
        
        # Check what the entity extraction is looking for
        print(f"\n🔍 Entity extraction looks for these specific fields:")
        print(f"   - payload.vendor_raw")
        print(f"   - payload.vendor") 
        print(f"   - payload.merchant")
        print(f"   - payload.email")
        print(f"   - payload.phone")
        print(f"   - payload.bank_account")
        
        print(f"\n💡 The problem is likely that the AI classification isn't extracting")
        print(f"   vendor data into these specific field names, even though the")
        print(f"   source files have 'Client Name' and 'Vendor Name' columns.")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_raw_events()
