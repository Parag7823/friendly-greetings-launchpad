-- Migration: Add job_id to entity and platform tables for unified tracking
-- Date: 2025-11-19
-- Purpose: Enable job-based cleanup and tracking for entity resolution and platform discovery

-- Add job_id to entity_matches table
ALTER TABLE IF EXISTS public.entity_matches
ADD COLUMN IF NOT EXISTS job_id UUID REFERENCES public.ingestion_jobs(id) ON DELETE CASCADE;

-- Add job_id to platform_patterns table
ALTER TABLE IF EXISTS public.platform_patterns
ADD COLUMN IF NOT EXISTS job_id UUID REFERENCES public.ingestion_jobs(id) ON DELETE CASCADE;

-- Add job_id to discovered_platforms table
ALTER TABLE IF EXISTS public.discovered_platforms
ADD COLUMN IF NOT EXISTS job_id UUID REFERENCES public.ingestion_jobs(id) ON DELETE CASCADE;

-- Create indexes for efficient job-based queries
CREATE INDEX IF NOT EXISTS idx_entity_matches_job_id ON public.entity_matches(job_id) WHERE job_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_platform_patterns_job_id ON public.platform_patterns(job_id) WHERE job_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_discovered_platforms_job_id ON public.discovered_platforms(job_id) WHERE job_id IS NOT NULL;

-- Add comments explaining the purpose
COMMENT ON COLUMN public.entity_matches.job_id IS 'Links to ingestion_jobs for unified tracking and deletion';
COMMENT ON COLUMN public.platform_patterns.job_id IS 'Links to ingestion_jobs for unified tracking and deletion';
COMMENT ON COLUMN public.discovered_platforms.job_id IS 'Links to ingestion_jobs for unified tracking and deletion';
