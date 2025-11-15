-- Migration: Delete metrics table and all references
-- Date: 2025-11-15
-- Purpose: Remove unused metrics table and clean up all references

-- Drop all policies first
DROP POLICY IF EXISTS "service_role_all_metrics" ON public.metrics;
DROP POLICY IF EXISTS "users_own_metrics" ON public.metrics;

-- Drop all indexes
DROP INDEX IF EXISTS idx_metrics_user_id;
DROP INDEX IF EXISTS idx_metrics_record_id;
DROP INDEX IF EXISTS idx_metrics_metric_type;
DROP INDEX IF EXISTS idx_metrics_date_recorded;
DROP INDEX IF EXISTS idx_metrics_transaction_id;

-- Drop all functions that reference metrics
DROP FUNCTION IF EXISTS get_metrics_summary(UUID);
DROP FUNCTION IF EXISTS get_metrics_by_type(UUID, TEXT);
DROP FUNCTION IF EXISTS calculate_metrics_aggregates(UUID);

-- Drop all triggers
DROP TRIGGER IF EXISTS trigger_update_metrics_updated_at ON public.metrics;

-- Remove foreign key constraints from other tables that reference metrics
-- (Check if any exist first)
DO $$
DECLARE
    constraint_record RECORD;
BEGIN
    -- Find all foreign key constraints that reference metrics table
    FOR constraint_record IN
        SELECT 
            tc.table_name,
            tc.constraint_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu 
            ON tc.constraint_name = kcu.constraint_name
        JOIN information_schema.constraint_column_usage ccu 
            ON ccu.constraint_name = tc.constraint_name
        WHERE tc.constraint_type = 'FOREIGN KEY'
        AND ccu.table_name = 'metrics'
    LOOP
        EXECUTE format('ALTER TABLE %I DROP CONSTRAINT IF EXISTS %I', 
                      constraint_record.table_name, 
                      constraint_record.constraint_name);
        RAISE NOTICE 'Dropped foreign key constraint % from table %', 
                     constraint_record.constraint_name, 
                     constraint_record.table_name;
    END LOOP;
END $$;

-- Drop the metrics table
DROP TABLE IF EXISTS public.metrics CASCADE;

-- Remove any views that depend on metrics
DROP VIEW IF EXISTS public.metrics_summary CASCADE;
DROP VIEW IF EXISTS public.user_metrics_overview CASCADE;

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'Migration 20251115000002: Successfully deleted metrics table and all references';
    RAISE NOTICE 'Removed: table, policies, indexes, functions, triggers, constraints, views';
END $$;
