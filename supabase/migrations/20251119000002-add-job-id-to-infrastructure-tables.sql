-- Migration: Add job_id to infrastructure tables for unified tracking
-- Date: 2025-11-19
-- Purpose: Enable job-based tracking and cleanup for connectors and processing_locks

-- Add job_id to connectors table
ALTER TABLE IF EXISTS public.connectors
ADD COLUMN IF NOT EXISTS job_id UUID REFERENCES public.ingestion_jobs(id) ON DELETE CASCADE;

-- Add job_id to processing_locks table
ALTER TABLE IF EXISTS public.processing_locks
ADD COLUMN IF NOT EXISTS job_id UUID REFERENCES public.ingestion_jobs(id) ON DELETE CASCADE;

-- Create indexes for efficient job-based queries
CREATE INDEX IF NOT EXISTS idx_connectors_job_id ON public.connectors(job_id) WHERE job_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_processing_locks_job_id ON public.processing_locks(job_id) WHERE job_id IS NOT NULL;

-- Add comments explaining the purpose
COMMENT ON COLUMN public.connectors.job_id IS 'Links to ingestion_jobs for unified tracking and deletion';
COMMENT ON COLUMN public.processing_locks.job_id IS 'Links to ingestion_jobs for unified tracking and deletion';
