-- Migration: Fix Race Condition in Duplicate Detection
-- Date: 2025-01-24
-- Purpose: Add unique constraint to prevent concurrent uploads from bypassing duplicate check

-- ============================================================================
-- CRITICAL FIX #1: Add unique constraint on (user_id, file_hash)
-- ============================================================================

-- This prevents race conditions where multiple concurrent uploads of the same file
-- can both pass the duplicate check and insert duplicate records

-- First, clean up any existing duplicates (keep the oldest record for each user+hash)
WITH duplicates AS (
    SELECT 
        id,
        user_id,
        file_hash,
        ROW_NUMBER() OVER (PARTITION BY user_id, file_hash ORDER BY created_at ASC) as rn
    FROM raw_records
    WHERE file_hash IS NOT NULL
)
DELETE FROM raw_records
WHERE id IN (
    SELECT id FROM duplicates WHERE rn > 1
);

-- Now add the unique constraint
ALTER TABLE raw_records 
ADD CONSTRAINT unique_user_file_hash 
UNIQUE (user_id, file_hash);

-- Add index for better performance on duplicate checks
CREATE INDEX IF NOT EXISTS idx_raw_records_user_hash_created 
ON raw_records(user_id, file_hash, created_at DESC) 
WHERE file_hash IS NOT NULL;

-- ============================================================================
-- CRITICAL FIX #4: Add columns for duplicate decision persistence
-- ============================================================================

-- Add columns to store duplicate decisions in raw_records table
-- This ensures decisions are persisted where duplicate checks query

ALTER TABLE raw_records 
ADD COLUMN IF NOT EXISTS duplicate_decision TEXT 
CHECK (duplicate_decision IN ('skip', 'replace', 'keep_both', 'delta_merge'));

ALTER TABLE raw_records 
ADD COLUMN IF NOT EXISTS duplicate_of UUID REFERENCES raw_records(id) ON DELETE SET NULL;

ALTER TABLE raw_records 
ADD COLUMN IF NOT EXISTS decision_timestamp TIMESTAMPTZ;

ALTER TABLE raw_records 
ADD COLUMN IF NOT EXISTS decision_metadata JSONB DEFAULT '{}';

-- Add index for querying duplicate decisions
CREATE INDEX IF NOT EXISTS idx_raw_records_duplicate_decision 
ON raw_records(user_id, duplicate_decision) 
WHERE duplicate_decision IS NOT NULL;

-- Add comments for documentation
COMMENT ON COLUMN raw_records.duplicate_decision IS 'User decision when duplicate detected: skip, replace, keep_both, or delta_merge';
COMMENT ON COLUMN raw_records.duplicate_of IS 'Reference to the original file if this is a duplicate';
COMMENT ON COLUMN raw_records.decision_timestamp IS 'When the duplicate decision was made';
COMMENT ON COLUMN raw_records.decision_metadata IS 'Additional metadata about the duplicate decision';

COMMENT ON CONSTRAINT unique_user_file_hash ON raw_records IS 'Prevents race conditions in duplicate detection by enforcing uniqueness at database level';
