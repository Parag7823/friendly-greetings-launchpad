-- Migration: Expand raw_records schema with duplicate tracking columns
-- Date: 2025-11-19
-- Purpose: Add duplicate decision tracking and integrity verification columns

-- Add duplicate decision tracking columns
ALTER TABLE IF EXISTS public.raw_records
ADD COLUMN IF NOT EXISTS duplicate_decision VARCHAR(50) CHECK (duplicate_decision IN ('none', 'skip', 'merge', 'replace', 'keep_both')),
ADD COLUMN IF NOT EXISTS duplicate_of UUID REFERENCES public.raw_records(id) ON DELETE SET NULL,
ADD COLUMN IF NOT EXISTS decision_timestamp TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS decision_metadata JSONB DEFAULT '{}';

-- Add duplicate flag
ALTER TABLE IF EXISTS public.raw_records
ADD COLUMN IF NOT EXISTS is_duplicate BOOLEAN DEFAULT FALSE;

-- Add integrity verification columns
ALTER TABLE IF EXISTS public.raw_records
ADD COLUMN IF NOT EXISTS file_hash_verified BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS integrity_check_at TIMESTAMP WITH TIME ZONE;

-- Add classification status column
ALTER TABLE IF EXISTS public.raw_records
ADD COLUMN IF NOT EXISTS classification_status VARCHAR(50) CHECK (classification_status IN ('pending', 'classified', 'failed', 'manual_review'));

-- Create indexes for duplicate tracking
CREATE INDEX IF NOT EXISTS idx_raw_records_duplicate_of ON public.raw_records(duplicate_of) WHERE duplicate_of IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_raw_records_is_duplicate ON public.raw_records(is_duplicate) WHERE is_duplicate = TRUE;
CREATE INDEX IF NOT EXISTS idx_raw_records_duplicate_decision ON public.raw_records(duplicate_decision) WHERE duplicate_decision != 'none';
CREATE INDEX IF NOT EXISTS idx_raw_records_decision_timestamp ON public.raw_records(decision_timestamp) WHERE decision_timestamp IS NOT NULL;

-- Create indexes for integrity verification
CREATE INDEX IF NOT EXISTS idx_raw_records_file_hash_verified ON public.raw_records(file_hash_verified) WHERE file_hash_verified = FALSE;
CREATE INDEX IF NOT EXISTS idx_raw_records_integrity_check ON public.raw_records(integrity_check_at) WHERE integrity_check_at IS NOT NULL;

-- Create indexes for classification
CREATE INDEX IF NOT EXISTS idx_raw_records_classification_status ON public.raw_records(classification_status) WHERE classification_status IN ('pending', 'manual_review');

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_raw_records_user_duplicate ON public.raw_records(user_id, is_duplicate) WHERE is_duplicate = TRUE;
CREATE INDEX IF NOT EXISTS idx_raw_records_file_duplicate ON public.raw_records(file_id, duplicate_decision) WHERE duplicate_decision IS NOT NULL;

-- Add comments for documentation
COMMENT ON COLUMN public.raw_records.duplicate_decision IS 'Decision made for duplicate handling (skip, merge, replace, keep_both)';
COMMENT ON COLUMN public.raw_records.duplicate_of IS 'Reference to the original record if this is a duplicate';
COMMENT ON COLUMN public.raw_records.decision_timestamp IS 'When the duplicate decision was made';
COMMENT ON COLUMN public.raw_records.decision_metadata IS 'Metadata about the duplicate decision (reason, confidence, etc.)';
COMMENT ON COLUMN public.raw_records.is_duplicate IS 'Flag indicating if this record is a duplicate';
COMMENT ON COLUMN public.raw_records.file_hash_verified IS 'Whether the file hash has been verified';
COMMENT ON COLUMN public.raw_records.integrity_check_at IS 'When the last integrity check was performed';
COMMENT ON COLUMN public.raw_records.classification_status IS 'Status of document classification (pending, classified, failed, manual_review)';
