-- Migration: Add Debug Logs Table
-- Date: 2025-01-27
-- Purpose: Store detailed debug/reasoning data for developer introspection

-- Debug logs table for storing AI reasoning and processing details
CREATE TABLE IF NOT EXISTS public.debug_logs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    job_id UUID REFERENCES public.ingestion_jobs(id) ON DELETE CASCADE,
    user_id UUID NOT NULL,
    stage TEXT NOT NULL, -- upload, excel, platform, classification, row_processing, entity_resolution, relationships
    component TEXT, -- specific component name (e.g., "UniversalPlatformDetector")
    data JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_debug_logs_job_id ON public.debug_logs(job_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_debug_logs_user_stage ON public.debug_logs(user_id, stage, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_debug_logs_component ON public.debug_logs(component, created_at DESC);

-- RLS policies
ALTER TABLE public.debug_logs ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "service_role_all_debug_logs" ON public.debug_logs;
CREATE POLICY "service_role_all_debug_logs" ON public.debug_logs
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_debug_logs" ON public.debug_logs;
CREATE POLICY "users_own_debug_logs" ON public.debug_logs
    FOR ALL USING (auth.uid() = user_id);

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON public.debug_logs TO authenticated;

-- Comments
COMMENT ON TABLE public.debug_logs IS 'Stores detailed debug and reasoning data for developer introspection';
COMMENT ON COLUMN public.debug_logs.stage IS 'Processing stage: upload, excel, platform, classification, row_processing, entity_resolution, relationships';
COMMENT ON COLUMN public.debug_logs.data IS 'Detailed debug data including AI reasoning, confidence scores, indicators';
COMMENT ON COLUMN public.debug_logs.metadata IS 'Additional metadata like processing time, errors, warnings';
