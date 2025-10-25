-- Migration: Add RLS Policies for EXISTING Tables Only
-- Date: 2025-10-25 14:00:01
-- Purpose: Enable Row Level Security only for tables that actually exist

-- ============================================================================
-- HELPER FUNCTION: Enable RLS only if table exists
-- ============================================================================

DO $$
DECLARE
    table_record RECORD;
    tables_to_secure TEXT[] := ARRAY[
        'raw_records',
        'ingestion_jobs', 
        'raw_events',
        'normalized_entities',
        'entity_matches',
        'processing_transactions',
        'processing_locks',
        'external_items',
        'user_connections',
        'ai_classification_cache',
        'duplicate_detection_cache',
        'platform_detections',
        'document_classifications',
        'processing_metrics',
        'user_activity_logs',
        'chat_sessions',
        'chat_messages',
        'temporal_patterns',
        'relationship_instances',
        'cross_platform_relationships',
        'relationship_patterns'
    ];
    table_name TEXT;
BEGIN
    FOREACH table_name IN ARRAY tables_to_secure
    LOOP
        -- Check if table exists
        IF EXISTS (
            SELECT 1 FROM pg_tables 
            WHERE schemaname = 'public' AND tablename = table_name
        ) THEN
            -- Enable RLS
            EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY', table_name);
            RAISE NOTICE 'Enabled RLS for: %', table_name;
            
            -- Drop existing policies if they exist
            EXECUTE format('DROP POLICY IF EXISTS "service_role_all_%s" ON public.%I', table_name, table_name);
            EXECUTE format('DROP POLICY IF EXISTS "users_own_%s" ON public.%I', table_name, table_name);
            
            -- Create service_role policy
            EXECUTE format(
                'CREATE POLICY "service_role_all_%s" ON public.%I FOR ALL USING (auth.role() = ''service_role'')',
                table_name, table_name
            );
            
            -- Create user policy
            EXECUTE format(
                'CREATE POLICY "users_own_%s" ON public.%I FOR ALL USING (auth.uid() = user_id)',
                table_name, table_name
            );
            
            -- Grant permissions
            EXECUTE format('GRANT SELECT, INSERT, UPDATE, DELETE ON public.%I TO authenticated', table_name);
            
            RAISE NOTICE 'Created policies for: %', table_name;
        ELSE
            RAISE NOTICE 'Table does not exist, skipping: %', table_name;
        END IF;
    END LOOP;
END $$;

-- ============================================================================
-- VERIFICATION
-- ============================================================================

-- Show which tables now have RLS enabled
SELECT 
    tablename,
    rowsecurity as rls_enabled,
    (SELECT COUNT(*) FROM pg_policies 
     WHERE schemaname = 'public' AND tablename = pt.tablename) as policy_count
FROM pg_tables pt
WHERE schemaname = 'public'
  AND rowsecurity = true
ORDER BY tablename;
