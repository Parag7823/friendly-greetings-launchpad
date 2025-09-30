-- Migration: ARQ DLQ support - job_failures table
-- Date: 2025-09-29 22:09:00

-- 1) Dead Letter Queue table for exhausted ARQ retries
CREATE TABLE IF NOT EXISTS public.job_failures (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    provider TEXT NOT NULL,
    user_id UUID,
    connection_id TEXT,
    correlation_id TEXT,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    error TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Useful indexes for querying and dashboards
CREATE INDEX IF NOT EXISTS idx_job_failures_provider_created_at ON public.job_failures(provider, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_job_failures_user_id_created_at ON public.job_failures(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_job_failures_connection_id_created_at ON public.job_failures(connection_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_job_failures_correlation_id ON public.job_failures(correlation_id);

-- Enable RLS
ALTER TABLE public.job_failures ENABLE ROW LEVEL SECURITY;

-- Owner-only access policy (service role bypasses RLS)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies WHERE schemaname = 'public' AND tablename = 'job_failures' AND policyname = 'job_failures_owner'
    ) THEN
        CREATE POLICY "job_failures_owner" ON public.job_failures
            FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);
    END IF;
END$$;
