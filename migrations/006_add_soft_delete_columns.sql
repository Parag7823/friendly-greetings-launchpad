-- Migration: 006_add_soft_delete_columns.sql
-- Purpose: Add soft-delete infrastructure to support graph cache invalidation
-- Date: 2025-11-21
-- Risk Level: LOW (adds columns with DEFAULT false, no data loss)

-- ============================================================================
-- FIX #3: Add soft-delete columns to support graph desynchronization fix
-- ============================================================================

-- Step 1: Add is_deleted column to normalized_entities
ALTER TABLE normalized_entities 
ADD COLUMN IF NOT EXISTS is_deleted boolean DEFAULT false;

-- Step 2: Add is_deleted column to relationship_instances
ALTER TABLE relationship_instances 
ADD COLUMN IF NOT EXISTS is_deleted boolean DEFAULT false;

-- Step 3: Add is_deleted column to raw_events
ALTER TABLE raw_events 
ADD COLUMN IF NOT EXISTS is_deleted boolean DEFAULT false;

-- ============================================================================
-- CREATE INDEXES FOR PERFORMANCE
-- ============================================================================

-- Step 4: Create index for soft-deleted normalized_entities queries
CREATE INDEX IF NOT EXISTS idx_normalized_entities_is_deleted 
ON normalized_entities(user_id, is_deleted) 
WHERE is_deleted = true;

-- Step 5: Create index for soft-deleted relationship_instances queries
CREATE INDEX IF NOT EXISTS idx_relationship_instances_is_deleted 
ON relationship_instances(user_id, is_deleted) 
WHERE is_deleted = true;

-- Step 6: Create index for soft-deleted raw_events queries
CREATE INDEX IF NOT EXISTS idx_raw_events_is_deleted 
ON raw_events(user_id, is_deleted) 
WHERE is_deleted = true;

-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================

-- Verify columns were added
-- SELECT column_name, data_type, column_default 
-- FROM information_schema.columns 
-- WHERE table_name IN ('normalized_entities', 'relationship_instances', 'raw_events')
-- AND column_name = 'is_deleted';

-- Verify indexes were created
-- SELECT indexname FROM pg_indexes 
-- WHERE tablename IN ('normalized_entities', 'relationship_instances', 'raw_events')
-- AND indexname LIKE '%is_deleted%';

-- ============================================================================
-- NOTES
-- ============================================================================

-- This migration enables soft-delete functionality:
-- 1. Instead of hard-deleting records, set is_deleted = true
-- 2. Graph incremental_update can detect soft-deleted records
-- 3. Allows cache invalidation without losing audit trail
-- 4. Backward compatible: is_deleted defaults to false for all existing records

-- The indexes ensure efficient queries for:
-- - Finding soft-deleted records by user
-- - Incremental graph updates
-- - Audit trail queries

-- No data migration needed:
-- - All existing records have is_deleted = false
-- - No existing data is affected
-- - Queries that don't check is_deleted will continue to work
