-- Enable RLS on key user-scoped tables to match existing policies
ALTER TABLE public.ingestion_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.raw_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.raw_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.metrics ENABLE ROW LEVEL SECURITY;
-- Test/logs table also has permissive policies; enable RLS to satisfy linter
ALTER TABLE public.integration_test_logs ENABLE ROW LEVEL SECURITY;