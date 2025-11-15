-- Migration: Complete fix for ingestion_jobs table
-- Date: 2025-11-14
-- Purpose: Add missing backend fields and fix status values for core pipeline

-- Add missing fields that backend expects
ALTER TABLE public.ingestion_jobs 
ADD COLUMN IF NOT EXISTS stream_offset BIGINT DEFAULT 0,
ADD COLUMN IF NOT EXISTS extracted_rows INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS duplicate_status TEXT,
ADD COLUMN IF NOT EXISTS ai_detection_summary JSONB DEFAULT '{}',
ADD COLUMN IF NOT EXISTS errors_json JSONB DEFAULT '[]',
ADD COLUMN IF NOT EXISTS final_output JSONB DEFAULT '{}',
ADD COLUMN IF NOT EXISTS processing_stage TEXT DEFAULT 'init',
ADD COLUMN IF NOT EXISTS total_rows INTEGER DEFAULT 0;

-- Drop existing status constraint and add correct one with all backend values
ALTER TABLE public.ingestion_jobs DROP CONSTRAINT IF EXISTS ingestion_jobs_status_check;
ALTER TABLE public.ingestion_jobs 
ADD CONSTRAINT ingestion_jobs_status_check 
CHECK (status IN (
    'queued', 'running', 'completed', 'failed',  -- Original values
    'starting', 'processing', 'waiting_user_decision', 'cancelled'  -- Missing backend values
));

-- Add constraint for processing_stage
ALTER TABLE public.ingestion_jobs 
ADD CONSTRAINT ingestion_jobs_processing_stage_check 
CHECK (processing_stage IN (
    'init', 'streaming', 'extracting', 'classifying', 
    'normalizing', 'deduplicating', 'completed', 'failed'
));

-- Add constraint for duplicate_status
ALTER TABLE public.ingestion_jobs 
ADD CONSTRAINT ingestion_jobs_duplicate_status_check 
CHECK (duplicate_status IS NULL OR duplicate_status IN (
    'none', 'detected', 'resolved', 'skip', 'replace', 'keep_both', 'delta_merge'
));

-- Create indexes for new fields
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_processing_stage ON public.ingestion_jobs(processing_stage);
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_duplicate_status ON public.ingestion_jobs(duplicate_status) WHERE duplicate_status IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_extracted_rows ON public.ingestion_jobs(extracted_rows) WHERE extracted_rows > 0;
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_total_rows ON public.ingestion_jobs(total_rows) WHERE total_rows > 0;

-- Add comments for documentation
COMMENT ON COLUMN public.ingestion_jobs.stream_offset IS 'Current offset in streaming processing (bytes)';
COMMENT ON COLUMN public.ingestion_jobs.extracted_rows IS 'Number of rows extracted from file';
COMMENT ON COLUMN public.ingestion_jobs.duplicate_status IS 'Status of duplicate detection and resolution';
COMMENT ON COLUMN public.ingestion_jobs.ai_detection_summary IS 'Summary of AI detection results and metadata';
COMMENT ON COLUMN public.ingestion_jobs.errors_json IS 'Structured error log as JSON array';
COMMENT ON COLUMN public.ingestion_jobs.final_output IS 'Final processing results and statistics';
COMMENT ON COLUMN public.ingestion_jobs.processing_stage IS 'Current stage in the processing pipeline';
COMMENT ON COLUMN public.ingestion_jobs.total_rows IS 'Total number of rows to process';

-- Update existing records to have default values
UPDATE public.ingestion_jobs 
SET 
    stream_offset = 0,
    extracted_rows = 0,
    processing_stage = 'completed',
    total_rows = 0
WHERE stream_offset IS NULL 
   OR extracted_rows IS NULL 
   OR processing_stage IS NULL 
   OR total_rows IS NULL;

-- Log migration completion
DO $$
BEGIN
    RAISE NOTICE 'Migration 20251114000001: Fixed ingestion_jobs table schema';
    RAISE NOTICE 'Added fields: stream_offset, extracted_rows, duplicate_status, ai_detection_summary, errors_json, final_output, processing_stage, total_rows';
    RAISE NOTICE 'Updated status constraint to include: starting, processing, waiting_user_decision, cancelled';
    RAISE NOTICE 'Added processing_stage constraint with pipeline stages';
END $$;
