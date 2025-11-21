-- Migration: 005_add_unique_constraint_relationship_patterns.sql
-- Purpose: Add unique constraint to prevent race condition in pattern creation
-- Date: 2025-11-21
-- Risk Level: LOW (adds constraint, no data loss)

-- ============================================================================
-- FIX #2: Add unique constraint to relationship_patterns
-- ============================================================================

-- Step 1: Add unique constraint on (user_id, relationship_type)
-- This prevents duplicate patterns and enables UPSERT operation
ALTER TABLE relationship_patterns 
ADD CONSTRAINT unique_pattern_per_user_type 
UNIQUE (user_id, relationship_type);

-- Step 2: Create index for performance (if not already indexed)
CREATE INDEX IF NOT EXISTS idx_relationship_patterns_user_type 
ON relationship_patterns(user_id, relationship_type);

-- ============================================================================
-- VERIFICATION
-- ============================================================================

-- Verify constraint was created
-- SELECT constraint_name FROM information_schema.table_constraints 
-- WHERE table_name = 'relationship_patterns' AND constraint_type = 'UNIQUE';

-- ============================================================================
-- NOTES
-- ============================================================================

-- This migration enables the UPSERT operation in enhanced_relationship_detector.py
-- to prevent race conditions when multiple workers try to create the same pattern
-- simultaneously.

-- The unique constraint ensures that:
-- 1. Only one pattern per (user_id, relationship_type) combination
-- 2. Concurrent inserts will use ON CONFLICT DO UPDATE instead of failing
-- 3. No duplicate patterns are created
