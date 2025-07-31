-- Simple RLS bypass for service role operations
-- This migration allows the backend to operate without RLS restrictions

-- For raw_events table
DROP POLICY IF EXISTS "raw_events_select_policy" ON public.raw_events;
DROP POLICY IF EXISTS "raw_events_insert_policy" ON public.raw_events;
DROP POLICY IF EXISTS "raw_events_update_policy" ON public.raw_events;
DROP POLICY IF EXISTS "raw_events_delete_policy" ON public.raw_events;

-- Create simple policies that allow service role full access
CREATE POLICY "raw_events_service_access" ON public.raw_events
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "raw_events_user_access" ON public.raw_events
    FOR ALL USING (auth.uid() = user_id);

-- For raw_records table
DROP POLICY IF EXISTS "raw_records_select_policy" ON public.raw_records;
DROP POLICY IF EXISTS "raw_records_insert_policy" ON public.raw_records;
DROP POLICY IF EXISTS "raw_records_update_policy" ON public.raw_records;
DROP POLICY IF EXISTS "raw_records_delete_policy" ON public.raw_records;

-- Create simple policies that allow service role full access
CREATE POLICY "raw_records_service_access" ON public.raw_records
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "raw_records_user_access" ON public.raw_records
    FOR ALL USING (auth.uid() = user_id);

-- For ingestion_jobs table
DROP POLICY IF EXISTS "ingestion_jobs_select_policy" ON public.ingestion_jobs;
DROP POLICY IF EXISTS "ingestion_jobs_insert_policy" ON public.ingestion_jobs;
DROP POLICY IF EXISTS "ingestion_jobs_update_policy" ON public.ingestion_jobs;
DROP POLICY IF EXISTS "ingestion_jobs_delete_policy" ON public.ingestion_jobs;

-- Create simple policies that allow service role full access
CREATE POLICY "ingestion_jobs_service_access" ON public.ingestion_jobs
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "ingestion_jobs_user_access" ON public.ingestion_jobs
    FOR ALL USING (auth.uid() = user_id);

-- For metrics table
DROP POLICY IF EXISTS "metrics_select_policy" ON public.metrics;
DROP POLICY IF EXISTS "metrics_insert_policy" ON public.metrics;
DROP POLICY IF EXISTS "metrics_update_policy" ON public.metrics;
DROP POLICY IF EXISTS "metrics_delete_policy" ON public.metrics;

-- Create simple policies that allow service role full access
CREATE POLICY "metrics_service_access" ON public.metrics
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "metrics_user_access" ON public.metrics
    FOR ALL USING (auth.uid() = user_id); 