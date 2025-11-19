-- Migration: Expand ingestion_jobs schema with missing tracking columns
-- Date: 2025-11-19
-- Purpose: Add all columns needed for comprehensive job tracking and retry management

-- Add retry and scheduling columns
ALTER TABLE IF EXISTS public.ingestion_jobs
ADD COLUMN IF NOT EXISTS priority VARCHAR(20) DEFAULT 'normal' CHECK (priority IN ('low', 'normal', 'high', 'critical')),
ADD COLUMN IF NOT EXISTS file_size BIGINT,
ADD COLUMN IF NOT EXISTS retry_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS max_retries INTEGER DEFAULT 3,
ADD COLUMN IF NOT EXISTS last_retry_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS next_retry_at TIMESTAMP WITH TIME ZONE;

-- Add processing metadata columns
ALTER TABLE IF EXISTS public.ingestion_jobs
ADD COLUMN IF NOT EXISTS filename TEXT,
ADD COLUMN IF NOT EXISTS progress_percentage INTEGER DEFAULT 0 CHECK (progress_percentage >= 0 AND progress_percentage <= 100),
ADD COLUMN IF NOT EXISTS status_message TEXT,
ADD COLUMN IF NOT EXISTS error_details JSONB DEFAULT '{}',
ADD COLUMN IF NOT EXISTS stream_offset BIGINT DEFAULT 0;

-- Add processing statistics columns
ALTER TABLE IF EXISTS public.ingestion_jobs
ADD COLUMN IF NOT EXISTS extracted_rows INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS duplicate_status VARCHAR(50) CHECK (duplicate_status IN ('none', 'exact', 'near', 'content', 'merged')),
ADD COLUMN IF NOT EXISTS ai_detection_summary JSONB DEFAULT '{}',
ADD COLUMN IF NOT EXISTS errors_json JSONB DEFAULT '[]',
ADD COLUMN IF NOT EXISTS final_output JSONB DEFAULT '{}';

-- Add processing stage tracking
ALTER TABLE IF EXISTS public.ingestion_jobs
ADD COLUMN IF NOT EXISTS processing_stage VARCHAR(100),
ADD COLUMN IF NOT EXISTS total_rows INTEGER DEFAULT 0;

-- Create indexes for new columns
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_priority ON public.ingestion_jobs(priority) WHERE priority != 'normal';
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_retry_count ON public.ingestion_jobs(retry_count) WHERE retry_count > 0;
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_next_retry ON public.ingestion_jobs(next_retry_at) WHERE next_retry_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_filename ON public.ingestion_jobs(filename) WHERE filename IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_progress ON public.ingestion_jobs(progress_percentage) WHERE progress_percentage < 100;
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_duplicate_status ON public.ingestion_jobs(duplicate_status) WHERE duplicate_status != 'none';
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_processing_stage ON public.ingestion_jobs(processing_stage) WHERE processing_stage IS NOT NULL;

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status_retry ON public.ingestion_jobs(status, next_retry_at) WHERE status = 'failed' AND next_retry_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_user_progress ON public.ingestion_jobs(user_id, progress_percentage) WHERE progress_percentage < 100;

-- Add comments for documentation
COMMENT ON COLUMN public.ingestion_jobs.priority IS 'Job priority level for scheduling (low, normal, high, critical)';
COMMENT ON COLUMN public.ingestion_jobs.file_size IS 'Size of the ingested file in bytes';
COMMENT ON COLUMN public.ingestion_jobs.retry_count IS 'Number of retry attempts made';
COMMENT ON COLUMN public.ingestion_jobs.max_retries IS 'Maximum number of retry attempts allowed';
COMMENT ON COLUMN public.ingestion_jobs.last_retry_at IS 'Timestamp of last retry attempt';
COMMENT ON COLUMN public.ingestion_jobs.next_retry_at IS 'Scheduled time for next retry';
COMMENT ON COLUMN public.ingestion_jobs.filename IS 'Original filename of the ingested file';
COMMENT ON COLUMN public.ingestion_jobs.progress_percentage IS 'Processing progress as percentage (0-100)';
COMMENT ON COLUMN public.ingestion_jobs.status_message IS 'Human-readable status message';
COMMENT ON COLUMN public.ingestion_jobs.error_details IS 'Detailed error information in JSON format';
COMMENT ON COLUMN public.ingestion_jobs.stream_offset IS 'Current offset in streaming file processing';
COMMENT ON COLUMN public.ingestion_jobs.extracted_rows IS 'Number of rows successfully extracted';
COMMENT ON COLUMN public.ingestion_jobs.duplicate_status IS 'Duplicate detection status (none, exact, near, content, merged)';
COMMENT ON COLUMN public.ingestion_jobs.ai_detection_summary IS 'Summary of AI detection results';
COMMENT ON COLUMN public.ingestion_jobs.errors_json IS 'Array of errors encountered during processing';
COMMENT ON COLUMN public.ingestion_jobs.final_output IS 'Final output/result of the job';
COMMENT ON COLUMN public.ingestion_jobs.processing_stage IS 'Current processing stage (e.g., field_detection, platform_detection, normalization)';
COMMENT ON COLUMN public.ingestion_jobs.total_rows IS 'Total number of rows in the file';
