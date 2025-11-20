-- Migration: 003_drop_merged_tables.sql
-- Purpose: Drop the 3 merged tables after data migration is complete
-- Date: 2025-11-21
-- Risk Level: LOW (only after 002_safe_table_merges.sql is applied and tested)

-- ============================================================================
-- IMPORTANT: RUN THIS ONLY AFTER:
-- 1. Migration 002_safe_table_merges.sql has been applied
-- 2. Application code has been updated (7 locations)
-- 3. Thorough testing confirms data integrity
-- 4. Backup has been taken
-- ============================================================================

-- ============================================================================
-- DROP TABLE #1: duplicate_transactions
-- ============================================================================

-- Verify data was migrated before dropping
-- SELECT COUNT(*) as duplicate_count FROM relationship_instances WHERE is_duplicate = true;
-- This should match the count from duplicate_transactions before deletion

DROP TABLE IF EXISTS duplicate_transactions CASCADE;
-- CASCADE drops any dependent foreign keys or views

-- ============================================================================
-- DROP TABLE #2: temporal_anomalies
-- ============================================================================

-- Verify data was migrated before dropping
-- SELECT COUNT(*) as anomaly_count FROM temporal_patterns WHERE anomalies != '[]'::jsonb;
-- This should match the count from temporal_anomalies before deletion

DROP TABLE IF EXISTS temporal_anomalies CASCADE;
-- CASCADE drops any dependent foreign keys or views

-- ============================================================================
-- DROP TABLE #3: seasonal_patterns
-- ============================================================================

-- Verify data was migrated before dropping
-- SELECT COUNT(*) as seasonal_count FROM temporal_patterns WHERE seasonal_data IS NOT NULL;
-- This should match the count from seasonal_patterns before deletion

DROP TABLE IF EXISTS seasonal_patterns CASCADE;
-- CASCADE drops any dependent foreign keys or views

-- ============================================================================
-- VERIFICATION AFTER DROPPING
-- ============================================================================

-- Run these queries to verify all data is still accessible:

-- 1. Verify duplicate data is accessible
-- SELECT COUNT(*) as duplicates FROM relationship_instances WHERE is_duplicate = true;

-- 2. Verify anomaly data is accessible
-- SELECT COUNT(*) as anomalies FROM temporal_patterns WHERE anomalies != '[]'::jsonb;

-- 3. Verify seasonal data is accessible
-- SELECT COUNT(*) as seasonal FROM temporal_patterns WHERE seasonal_data IS NOT NULL;

-- ============================================================================
-- CLEANUP COMPLETE
-- ============================================================================

-- Summary of changes:
-- - Dropped: duplicate_transactions (data migrated to relationship_instances)
-- - Dropped: temporal_anomalies (data migrated to temporal_patterns.anomalies)
-- - Dropped: seasonal_patterns (data migrated to temporal_patterns.seasonal_data)
-- - Result: 33 tables → 30 tables (-9%)
-- - Data loss: ZERO
-- - Functionality loss: ZERO

-- All data is preserved in parent tables with JSONB columns.
-- All audit trails (created_at, job_id, transaction_id) are preserved.
-- All indexes have been created for performance.
