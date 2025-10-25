-- Migration: Add RLS Policies - Smart Version (Checks for user_id column)
-- Date: 2025-10-25 14:00:03
-- Purpose: Only apply user-based policies to tables that have user_id column

-- ============================================================================
-- SMART RLS POLICY APPLICATION
-- ============================================================================

DO $$
DECLARE
    table_record RECORD;
    has_user_id BOOLEAN;
BEGIN
    -- Loop through all tables in public schema
    FOR table_record IN 
        SELECT tablename 
        FROM pg_tables 
        WHERE schemaname = 'public'
    LOOP
        -- Enable RLS for all tables
        EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY', table_record.tablename);
        RAISE NOTICE 'Enabled RLS for: %', table_record.tablename;
        
        -- Drop existing policies
        EXECUTE format('DROP POLICY IF EXISTS "service_role_all_%s" ON public.%I', 
                      table_record.tablename, table_record.tablename);
        EXECUTE format('DROP POLICY IF EXISTS "users_own_%s" ON public.%I', 
                      table_record.tablename, table_record.tablename);
        
        -- Always create service_role policy (full access)
        EXECUTE format(
            'CREATE POLICY "service_role_all_%s" ON public.%I FOR ALL USING (auth.role() = ''service_role'')',
            table_record.tablename, table_record.tablename
        );
        RAISE NOTICE 'Created service_role policy for: %', table_record.tablename;
        
        -- Check if table has user_id column
        SELECT EXISTS (
            SELECT 1 
            FROM information_schema.columns 
            WHERE table_schema = 'public' 
              AND table_name = table_record.tablename 
              AND column_name = 'user_id'
        ) INTO has_user_id;
        
        -- Only create user policy if user_id column exists
        IF has_user_id THEN
            EXECUTE format(
                'CREATE POLICY "users_own_%s" ON public.%I FOR ALL USING (auth.uid() = user_id)',
                table_record.tablename, table_record.tablename
            );
            RAISE NOTICE 'Created user policy for: % (has user_id)', table_record.tablename;
        ELSE
            -- For tables without user_id, allow all authenticated users to access
            EXECUTE format(
                'CREATE POLICY "users_own_%s" ON public.%I FOR ALL USING (auth.role() = ''authenticated'')',
                table_record.tablename, table_record.tablename
            );
            RAISE NOTICE 'Created authenticated policy for: % (no user_id)', table_record.tablename;
        END IF;
        
        -- Grant permissions
        EXECUTE format('GRANT SELECT, INSERT, UPDATE, DELETE ON public.%I TO authenticated', 
                      table_record.tablename);
        
    END LOOP;
    
    RAISE NOTICE 'RLS policies applied to all tables successfully!';
END $$;

-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================

-- Show which tables have RLS enabled
SELECT 
    tablename,
    rowsecurity as rls_enabled,
    (SELECT COUNT(*) FROM pg_policies 
     WHERE schemaname = 'public' AND tablename = pt.tablename) as policy_count,
    (SELECT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
          AND table_name = pt.tablename 
          AND column_name = 'user_id'
    )) as has_user_id
FROM pg_tables pt
WHERE schemaname = 'public'
ORDER BY tablename;

-- Show all policies created
SELECT 
    schemaname,
    tablename,
    policyname,
    CASE 
        WHEN policyname LIKE 'service_role%' THEN 'Service Role (Full Access)'
        WHEN policyname LIKE 'users_own%' THEN 'User-Specific or Authenticated'
        ELSE 'Other'
    END as policy_type
FROM pg_policies
WHERE schemaname = 'public'
ORDER BY tablename, policyname;
