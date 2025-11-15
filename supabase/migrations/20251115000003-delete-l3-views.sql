-- Migration: Delete L3 views and related functions
-- Date: 2025-11-15
-- Purpose: Remove unused L3 gating, core, and velocity views

-- Drop all L3 views in dependency order
DROP VIEW IF EXISTS public.l3_revenue_velocity CASCADE;
DROP VIEW IF EXISTS public.l3_ar_open CASCADE;
DROP VIEW IF EXISTS public.l3_ap_open CASCADE;
DROP VIEW IF EXISTS public.l3_burn_rate CASCADE;
DROP VIEW IF EXISTS public.l3_cash_positions CASCADE;
DROP VIEW IF EXISTS public.l3_events_ready CASCADE;

-- Drop related functions
DROP FUNCTION IF EXISTS public.get_l3_events_ready(UUID, INTEGER, INTEGER, TEXT, TEXT);
DROP FUNCTION IF EXISTS public.get_l3_revenue_velocity(UUID);
DROP FUNCTION IF EXISTS public.get_l3_ar_summary(UUID);
DROP FUNCTION IF EXISTS public.get_l3_ap_summary(UUID);
DROP FUNCTION IF EXISTS public.get_l3_burn_rate(UUID);
DROP FUNCTION IF EXISTS public.get_l3_cash_positions(UUID);

-- Remove any policies that might reference these views
DROP POLICY IF EXISTS "l3_events_ready_policy" ON public.l3_events_ready;
DROP POLICY IF EXISTS "l3_revenue_velocity_policy" ON public.l3_revenue_velocity;

-- Remove any triggers related to L3 views
DROP TRIGGER IF EXISTS trigger_l3_events_ready_update ON public.raw_events;
DROP TRIGGER IF EXISTS trigger_l3_revenue_velocity_update ON public.raw_events;

-- Remove any indexes that were specifically created for L3 views
DROP INDEX IF EXISTS idx_raw_events_l3_ready;
DROP INDEX IF EXISTS idx_raw_events_l3_revenue;
DROP INDEX IF EXISTS idx_processing_transactions_l3;

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'Migration 20251115000003: Successfully deleted all L3 views and related objects';
    RAISE NOTICE 'Removed: l3_events_ready, l3_revenue_velocity, l3_ar_open, l3_ap_open, l3_burn_rate, l3_cash_positions';
    RAISE NOTICE 'Removed: All related functions, policies, triggers, and indexes';
END $$;
