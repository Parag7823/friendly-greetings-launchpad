-- Comprehensive RLS fix for backend service role access
-- This migration ensures the backend can access all necessary tables

-- Fix raw_events RLS policies
DROP POLICY IF EXISTS "Users can view their own events" ON public.raw_events;
DROP POLICY IF EXISTS "Users can insert their own events" ON public.raw_events;
DROP POLICY IF EXISTS "Users can update their own events" ON public.raw_events;

-- Create comprehensive policies for raw_events
CREATE POLICY "raw_events_select_policy" ON public.raw_events
    FOR SELECT USING (
        auth.uid() = user_id OR 
        auth.role() = 'service_role' OR
        auth.uid() IS NULL
    );

CREATE POLICY "raw_events_insert_policy" ON public.raw_events
    FOR INSERT WITH CHECK (
        auth.uid() = user_id OR 
        auth.role() = 'service_role' OR
        auth.uid() IS NULL
    );

CREATE POLICY "raw_events_update_policy" ON public.raw_events
    FOR UPDATE USING (
        auth.uid() = user_id OR 
        auth.role() = 'service_role' OR
        auth.uid() IS NULL
    );

CREATE POLICY "raw_events_delete_policy" ON public.raw_events
    FOR DELETE USING (
        auth.uid() = user_id OR 
        auth.role() = 'service_role'
    );

-- Fix raw_records RLS policies
DROP POLICY IF EXISTS "Users can view their own records" ON public.raw_records;
DROP POLICY IF EXISTS "Users can insert their own records" ON public.raw_records;
DROP POLICY IF EXISTS "Users can update their own records" ON public.raw_records;

-- Create comprehensive policies for raw_records
CREATE POLICY "raw_records_select_policy" ON public.raw_records
    FOR SELECT USING (
        auth.uid() = user_id OR 
        auth.role() = 'service_role' OR
        auth.uid() IS NULL
    );

CREATE POLICY "raw_records_insert_policy" ON public.raw_records
    FOR INSERT WITH CHECK (
        auth.uid() = user_id OR 
        auth.role() = 'service_role' OR
        auth.uid() IS NULL
    );

CREATE POLICY "raw_records_update_policy" ON public.raw_records
    FOR UPDATE USING (
        auth.uid() = user_id OR 
        auth.role() = 'service_role' OR
        auth.uid() IS NULL
    );

CREATE POLICY "raw_records_delete_policy" ON public.raw_records
    FOR DELETE USING (
        auth.uid() = user_id OR 
        auth.role() = 'service_role'
    );

-- Fix ingestion_jobs RLS policies
DROP POLICY IF EXISTS "Users can view their own jobs" ON public.ingestion_jobs;
DROP POLICY IF EXISTS "Users can insert their own jobs" ON public.ingestion_jobs;
DROP POLICY IF EXISTS "Users can update their own jobs" ON public.ingestion_jobs;

-- Create comprehensive policies for ingestion_jobs
CREATE POLICY "ingestion_jobs_select_policy" ON public.ingestion_jobs
    FOR SELECT USING (
        auth.uid() = user_id OR 
        auth.role() = 'service_role' OR
        auth.uid() IS NULL
    );

CREATE POLICY "ingestion_jobs_insert_policy" ON public.ingestion_jobs
    FOR INSERT WITH CHECK (
        auth.uid() = user_id OR 
        auth.role() = 'service_role' OR
        auth.uid() IS NULL
    );

CREATE POLICY "ingestion_jobs_update_policy" ON public.ingestion_jobs
    FOR UPDATE USING (
        auth.uid() = user_id OR 
        auth.role() = 'service_role' OR
        auth.uid() IS NULL
    );

CREATE POLICY "ingestion_jobs_delete_policy" ON public.ingestion_jobs
    FOR DELETE USING (
        auth.uid() = user_id OR 
        auth.role() = 'service_role'
    );

-- Also ensure metrics table has proper policies
DROP POLICY IF EXISTS "Users can view their own metrics" ON public.metrics;
DROP POLICY IF EXISTS "Users can insert their own metrics" ON public.metrics;
DROP POLICY IF EXISTS "Users can update their own metrics" ON public.metrics;

-- Create comprehensive policies for metrics
CREATE POLICY "metrics_select_policy" ON public.metrics
    FOR SELECT USING (
        auth.uid() = user_id OR 
        auth.role() = 'service_role' OR
        auth.uid() IS NULL
    );

CREATE POLICY "metrics_insert_policy" ON public.metrics
    FOR INSERT WITH CHECK (
        auth.uid() = user_id OR 
        auth.role() = 'service_role' OR
        auth.uid() IS NULL
    );

CREATE POLICY "metrics_update_policy" ON public.metrics
    FOR UPDATE USING (
        auth.uid() = user_id OR 
        auth.role() = 'service_role' OR
        auth.uid() IS NULL
    );

CREATE POLICY "metrics_delete_policy" ON public.metrics
    FOR DELETE USING (
        auth.uid() = user_id OR 
        auth.role() = 'service_role'
    ); 