-- Refine RLS policies to restrict anonymous write access

-- Drop existing policies for raw_events
DROP POLICY IF EXISTS "raw_events_select_policy" ON public.raw_events;
DROP POLICY IF EXISTS "raw_events_insert_policy" ON public.raw_events;
DROP POLICY IF EXISTS "raw_events_update_policy" ON public.raw_events;
DROP POLICY IF EXISTS "raw_events_delete_policy" ON public.raw_events;

-- Create refined policies for raw_events
CREATE POLICY "raw_events_select_policy" ON public.raw_events
    FOR SELECT USING (
        auth.uid() = user_id OR
        auth.role() = 'service_role'
    );

CREATE POLICY "raw_events_insert_policy" ON public.raw_events
    FOR INSERT WITH CHECK (
        auth.uid() = user_id OR
        auth.role() = 'service_role'
    );

CREATE POLICY "raw_events_update_policy" ON public.raw_events
    FOR UPDATE USING (
        auth.uid() = user_id OR
        auth.role() = 'service_role'
    );

CREATE POLICY "raw_events_delete_policy" ON public.raw_events
    FOR DELETE USING (
        auth.uid() = user_id OR
        auth.role() = 'service_role'
    );

-- Drop existing policies for raw_records
DROP POLICY IF EXISTS "raw_records_select_policy" ON public.raw_records;
DROP POLICY IF EXISTS "raw_records_insert_policy" ON public.raw_records;
DROP POLICY IF EXISTS "raw_records_update_policy" ON public.raw_records;
DROP POLICY IF EXISTS "raw_records_delete_policy" ON public.raw_records;

-- Create refined policies for raw_records
CREATE POLICY "raw_records_select_policy" ON public.raw_records
    FOR SELECT USING (
        auth.uid() = user_id OR
        auth.role() = 'service_role'
    );

CREATE POLICY "raw_records_insert_policy" ON public.raw_records
    FOR INSERT WITH CHECK (
        auth.uid() = user_id OR
        auth.role() = 'service_role'
    );

CREATE POLICY "raw_records_update_policy" ON public.raw_records
    FOR UPDATE USING (
        auth.uid() = user_id OR
        auth.role() = 'service_role'
    );

CREATE POLICY "raw_records_delete_policy" ON public.raw_records
    FOR DELETE USING (
        auth.uid() = user_id OR
        auth.role() = 'service_role'
    );

-- Drop existing policies for ingestion_jobs
DROP POLICY IF EXISTS "ingestion_jobs_select_policy" ON public.ingestion_jobs;
DROP POLICY IF EXISTS "ingestion_jobs_insert_policy" ON public.ingestion_jobs;
DROP POLICY IF EXISTS "ingestion_jobs_update_policy" ON public.ingestion_jobs;
DROP POLICY IF EXISTS "ingestion_jobs_delete_policy" ON public.ingestion_jobs;

-- Create refined policies for ingestion_jobs
CREATE POLICY "ingestion_jobs_select_policy" ON public.ingestion_jobs
    FOR SELECT USING (
        auth.uid() = user_id OR
        auth.role() = 'service_role'
    );

CREATE POLICY "ingestion_jobs_insert_policy" ON public.ingestion_jobs
    FOR INSERT WITH CHECK (
        auth.uid() = user_id OR
        auth.role() = 'service_role'
    );

CREATE POLICY "ingestion_jobs_update_policy" ON public.ingestion_jobs
    FOR UPDATE USING (
        auth.uid() = user_id OR
        auth.role() = 'service_role'
    );

CREATE POLICY "ingestion_jobs_delete_policy" ON public.ingestion_jobs
    FOR DELETE USING (
        auth.uid() = user_id OR
        auth.role() = 'service_role'
    );

-- Drop existing policies for metrics
DROP POLICY IF EXISTS "metrics_select_policy" ON public.metrics;
DROP POLICY IF EXISTS "metrics_insert_policy" ON public.metrics;
DROP POLICY IF EXISTS "metrics_update_policy" ON public.metrics;
DROP POLICY IF EXISTS "metrics_delete_policy" ON public.metrics;

-- Create refined policies for metrics
CREATE POLICY "metrics_select_policy" ON public.metrics
    FOR SELECT USING (
        auth.uid() = user_id OR
        auth.role() = 'service_role'
    );

CREATE POLICY "metrics_insert_policy" ON public.metrics
    FOR INSERT WITH CHECK (
        auth.uid() = user_id OR
        auth.role() = 'service_role'
    );

CREATE POLICY "metrics_update_policy" ON public.metrics
    FOR UPDATE USING (
        auth.uid() = user_id OR
        auth.role() = 'service_role'
    );

CREATE POLICY "metrics_delete_policy" ON public.metrics
    FOR DELETE USING (
        auth.uid() = user_id OR
        auth.role() = 'service_role'
    );
