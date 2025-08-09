#!/usr/bin/env python3
"""
Script to apply the entity resolution over-merging fix
"""

import os
import sys
import asyncio
from supabase import create_client, Client

# Configuration
SUPABASE_URL = os.getenv('SUPABASE_URL', 'https://gnrbafqifucxlaihtyuv.supabase.co')
SUPABASE_SERVICE_KEY = os.getenv('SUPABASE_SERVICE_KEY')

def read_sql_file(filename: str) -> str:
    """Read SQL file content"""
    with open(filename, 'r') as f:
        return f.read()

async def apply_migration(supabase: Client):
    """Apply the entity resolution fix migration"""
    print("🔄 Applying entity resolution fix migration...")
    
    try:
        # Read and execute the migration
        migration_sql = read_sql_file('supabase/migrations/20250810000000-fix-entity-resolution-over-merging.sql')
        
        # Execute the migration
        result = supabase.rpc('exec_sql', {'sql': migration_sql}).execute()
        
        if result.data:
            print("✅ Migration applied successfully!")
        else:
            print("❌ Migration failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error applying migration: {e}")
        return False
    
    return True

async def run_cleanup(supabase: Client):
    """Run the entity cleanup script"""
    print("🧹 Running entity cleanup...")
    
    try:
        # Read and execute the cleanup script
        cleanup_sql = read_sql_file('fix_overmerged_entities.sql')
        
        # Execute the cleanup
        result = supabase.rpc('exec_sql', {'sql': cleanup_sql}).execute()
        
        if result.data:
            print("✅ Cleanup completed successfully!")
            print("📊 Cleanup results:")
            for row in result.data:
                print(f"  - {row['action']}: {row['details']}")
        else:
            print("❌ Cleanup failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error running cleanup: {e}")
        return False
    
    return True

async def verify_fix(supabase: Client):
    """Verify that the fix worked"""
    print("🔍 Verifying entity resolution fix...")
    
    try:
        # Check entity counts
        result = supabase.table('normalized_entities').select('entity_type, canonical_name, aliases').execute()
        
        if result.data:
            print("📊 Current entities:")
            for entity in result.data:
                alias_count = len(entity['aliases']) if entity['aliases'] else 0
                print(f"  - {entity['entity_type']}: {entity['canonical_name']} ({alias_count} aliases)")
            
            # Check for over-merged entities
            over_merged = [e for e in result.data if len(e['aliases']) > 10 if e['aliases'] else 0]
            if over_merged:
                print(f"⚠️  Found {len(over_merged)} potentially over-merged entities")
            else:
                print("✅ No over-merged entities found!")
                
        else:
            print("❌ Could not verify entities!")
            return False
            
    except Exception as e:
        print(f"❌ Error verifying fix: {e}")
        return False
    
    return True

async def main():
    """Main function"""
    if not SUPABASE_SERVICE_KEY:
        print("❌ SUPABASE_SERVICE_KEY environment variable not set!")
        sys.exit(1)
    
    # Initialize Supabase client
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    
    print("🚀 Starting entity resolution fix...")
    
    # Apply migration
    if not await apply_migration(supabase):
        print("❌ Failed to apply migration!")
        sys.exit(1)
    
    # Run cleanup
    if not await run_cleanup(supabase):
        print("❌ Failed to run cleanup!")
        sys.exit(1)
    
    # Verify fix
    if not await verify_fix(supabase):
        print("❌ Fix verification failed!")
        sys.exit(1)
    
    print("🎉 Entity resolution fix completed successfully!")
    print("📝 Next steps:")
    print("  1. Re-run your entity resolution tests")
    print("  2. Verify that entities are no longer over-merged")
    print("  3. Test with new data to ensure the fix works")

if __name__ == "__main__":
    asyncio.run(main()) 