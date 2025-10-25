-- Migration: Add RLS Policies for ALL 39 Tables (FINAL VERSION)
-- Date: 2025-10-25 14:00:04
-- Purpose: Enable Row Level Security for all existing tables with user_id

-- ============================================================================
-- APPLY RLS TO ALL 39 TABLES
-- ============================================================================

DO $$
DECLARE
    tables_with_user_id TEXT[] := ARRAY[
        'causal_relationships',
        'chat_messages',
        'counterfactual_analyses',
        'cross_platform_relationships',
        'detection_log',
        'discovered_platforms',
        'enriched_relationships',
        'entity_matches',
        'error_logs',
        'event_delta_logs',
        'external_items',
        'field_mappings',
        'ingestion_jobs',
        'l3_ap_open',
        'l3_ar_open',
        'l3_burn_rate',
        'l3_cash_positions',
        'l3_events_ready',
        'l3_revenue_velocity',
        'metrics',
        'normalized_entities',
        'performance_metrics',
        'platform_patterns',
        'predicted_relationships',
        'processing_locks',
        'processing_transactions',
        'raw_events',
        'raw_records',
        'relationship_instances',
        'relationship_patterns',
        'resolution_log',
        'root_cause_analyses',
        'seasonal_patterns',
        'sync_cursors',
        'sync_runs',
        'temporal_anomalies',
        'temporal_patterns',
        'user_connections',
        'webhook_events'
    ];
    table_name TEXT;
    tables_without_user_id TEXT[] := ARRAY[
        'connectors'
    ];
BEGIN
    -- Process tables WITH user_id column
    FOREACH table_name IN ARRAY tables_with_user_id
    LOOP
        -- Check if table exists
        IF EXISTS (
            SELECT 1 FROM pg_tables 
            WHERE schemaname = 'public' AND tablename = table_name
        ) THEN
            -- Enable RLS
            EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY', table_name);
            
            -- Drop existing policies
            EXECUTE format('DROP POLICY IF EXISTS "service_role_all_%s" ON public.%I', 
                          table_name, table_name);
            EXECUTE format('DROP POLICY IF EXISTS "users_own_%s" ON public.%I', 
                          table_name, table_name);
            
            -- Create service_role policy (full access)
            EXECUTE format(
                'CREATE POLICY "service_role_all_%s" ON public.%I FOR ALL USING (auth.role() = ''service_role'')',
                table_name, table_name
            );
            
            -- Create user policy (user_id based)
            EXECUTE format(
                'CREATE POLICY "users_own_%s" ON public.%I FOR ALL USING (auth.uid() = user_id)',
                table_name, table_name
            );
            
            -- Grant permissions
            EXECUTE format('GRANT SELECT, INSERT, UPDATE, DELETE ON public.%I TO authenticated', table_name);
            
            RAISE NOTICE 'Applied RLS to: % (with user_id)', table_name;
        ELSE
            RAISE NOTICE 'Table does not exist, skipping: %', table_name;
        END IF;
    END LOOP;
    
    -- Process tables WITHOUT user_id column (allow all authenticated users)
    FOREACH table_name IN ARRAY tables_without_user_id
    LOOP
        IF EXISTS (
            SELECT 1 FROM pg_tables 
            WHERE schemaname = 'public' AND tablename = table_name
        ) THEN
            -- Enable RLS
            EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY', table_name);
            
            -- Drop existing policies
            EXECUTE format('DROP POLICY IF EXISTS "service_role_all_%s" ON public.%I', 
                          table_name, table_name);
            EXECUTE format('DROP POLICY IF EXISTS "authenticated_access_%s" ON public.%I', 
                          table_name, table_name);
            
            -- Create service_role policy
            EXECUTE format(
                'CREATE POLICY "service_role_all_%s" ON public.%I FOR ALL USING (auth.role() = ''service_role'')',
                table_name, table_name
            );
            
            -- Create authenticated policy (all authenticated users)
            EXECUTE format(
                'CREATE POLICY "authenticated_access_%s" ON public.%I FOR ALL USING (auth.role() = ''authenticated'')',
                table_name, table_name
            );
            
            -- Grant permissions
            EXECUTE format('GRANT SELECT, INSERT, UPDATE, DELETE ON public.%I TO authenticated', table_name);
            
            RAISE NOTICE 'Applied RLS to: % (no user_id, authenticated access)', table_name;
        ELSE
            RAISE NOTICE 'Table does not exist, skipping: %', table_name;
        END IF;
    END LOOP;
    
    RAISE NOTICE '✅ RLS policies applied successfully to all tables!';
END $$;

-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================

-- 1. Show RLS status for all tables
SELECT 
    tablename,
    rowsecurity as rls_enabled,
    (SELECT COUNT(*) FROM pg_policies 
     WHERE schemaname = 'public' AND tablename = pt.tablename) as policy_count
FROM pg_tables pt
WHERE schemaname = 'public'
ORDER BY tablename;

-- 2. Show all policies created
SELECT 
    tablename,
    policyname,
    CASE 
        WHEN policyname LIKE 'service_role%' THEN '🔧 Service Role (Full Access)'
        WHEN policyname LIKE 'users_own%' THEN '👤 User-Specific (user_id)'
        WHEN policyname LIKE 'authenticated%' THEN '🔓 All Authenticated Users'
        ELSE '❓ Other'
    END as policy_type
FROM pg_policies
WHERE schemaname = 'public'
ORDER BY tablename, policyname;

-- 3. Count tables by RLS status
SELECT 
    CASE 
        WHEN rowsecurity THEN 'RLS Enabled'
        ELSE 'RLS Disabled'
    END as status,
    COUNT(*) as table_count
FROM pg_tables
WHERE schemaname = 'public'
GROUP BY rowsecurity;

-- 4. Show tables with their policy counts
SELECT 
    pt.tablename,
    COUNT(pp.policyname) as policy_count,
    string_agg(pp.policyname, ', ' ORDER BY pp.policyname) as policies
FROM pg_tables pt
LEFT JOIN pg_policies pp ON pp.schemaname = 'public' AND pp.tablename = pt.tablename
WHERE pt.schemaname = 'public'
GROUP BY pt.tablename
ORDER BY pt.tablename;
