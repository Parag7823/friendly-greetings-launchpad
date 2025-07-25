-- Disable RLS on all public schema tables only

-- Disable RLS on ingestion_jobs table
ALTER TABLE public.ingestion_jobs DISABLE ROW LEVEL SECURITY;

-- Disable RLS on integration_test_logs table
ALTER TABLE public.integration_test_logs DISABLE ROW LEVEL SECURITY;

-- Disable RLS on metrics table
ALTER TABLE public.metrics DISABLE ROW LEVEL SECURITY;

-- Disable RLS on raw_records table
ALTER TABLE public.raw_records DISABLE ROW LEVEL SECURITY;