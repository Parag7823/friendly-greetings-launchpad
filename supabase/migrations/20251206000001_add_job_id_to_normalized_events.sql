-- Migration: Add missing job_id column to normalized_events table
-- Date: December 6, 2025
-- Purpose: Link normalized events back to ingestion jobs for cleanup and tracking

-- Add missing job_id column to normalized_events table
ALTER TABLE public.normalized_events
ADD COLUMN IF NOT EXISTS job_id UUID REFERENCES public.ingestion_jobs(id) ON DELETE CASCADE;

-- Backfill job_id from raw_events (via raw_event_id)
UPDATE public.normalized_events ne
SET job_id = re.job_id
FROM public.raw_events re
WHERE ne.raw_event_id = re.id
AND ne.job_id IS NULL;

-- Create index on job_id for query performance
CREATE INDEX IF NOT EXISTS idx_normalized_events_job_id_fk 
ON public.normalized_events(job_id) WHERE job_id IS NOT NULL;

-- Create composite index for common queries
CREATE INDEX IF NOT EXISTS idx_normalized_events_user_job_id 
ON public.normalized_events(user_id, job_id) WHERE job_id IS NOT NULL;

-- Add comment for documentation
COMMENT ON COLUMN public.normalized_events.job_id IS 'Links to ingestion_jobs for unified tracking and cascade deletion. Enables cleanup of normalized events when job is deleted.';
