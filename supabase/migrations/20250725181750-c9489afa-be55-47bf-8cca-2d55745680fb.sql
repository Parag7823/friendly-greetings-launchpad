-- Disable RLS on all tables and drop existing policies

-- Disable RLS on ingestion_jobs table
ALTER TABLE public.ingestion_jobs DISABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users can insert their own jobs" ON public.ingestion_jobs;
DROP POLICY IF EXISTS "Users can update their own jobs" ON public.ingestion_jobs;
DROP POLICY IF EXISTS "Users can view their own jobs" ON public.ingestion_jobs;

-- Disable RLS on integration_test_logs table
ALTER TABLE public.integration_test_logs DISABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Anyone can insert test logs" ON public.integration_test_logs;
DROP POLICY IF EXISTS "Anyone can read test logs" ON public.integration_test_logs;

-- Disable RLS on metrics table
ALTER TABLE public.metrics DISABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users can insert their own metrics" ON public.metrics;
DROP POLICY IF EXISTS "Users can view their own metrics" ON public.metrics;

-- Disable RLS on raw_records table
ALTER TABLE public.raw_records DISABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Allow integration test inserts" ON public.raw_records;
DROP POLICY IF EXISTS "Users can insert their own records" ON public.raw_records;
DROP POLICY IF EXISTS "Users can update their own records" ON public.raw_records;
DROP POLICY IF EXISTS "Users can view their own records" ON public.raw_records;

-- Disable RLS on storage.objects and drop storage policies
ALTER TABLE storage.objects DISABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Allow authenticated uploads to finely-upload" ON storage.objects;
DROP POLICY IF EXISTS "Allow authenticated access to finely-upload" ON storage.objects;