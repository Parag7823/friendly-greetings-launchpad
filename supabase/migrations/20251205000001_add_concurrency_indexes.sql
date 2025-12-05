-- ✅ FIX: Add critical indexes for 50 concurrent users
-- These indexes optimize queries that run on every file upload and duplicate check

-- Index 1: processing_locks lookup (used on every upload)
-- Current: Full table scan on every lock acquisition
-- Impact: 50 concurrent uploads = 50 simultaneous table scans
-- Fix: Index on (user_id, resource_id, status) for fast lock lookups
CREATE INDEX IF NOT EXISTS idx_processing_locks_user_resource 
ON processing_locks(user_id, resource_id, status)
WHERE status = 'active';

-- Index 2: raw_records duplicate detection (used during file processing)
-- Current: Full table scan checking file_hash
-- Impact: With 10K files, duplicate check takes 5-10 seconds
-- Fix: Index on (user_id, file_hash, status) for fast duplicate lookups
CREATE INDEX IF NOT EXISTS idx_raw_records_hash_lookup 
ON raw_records(user_id, file_hash, status)
WHERE is_duplicate = false;

-- Index 3: ingestion_jobs dashboard loading
-- Current: Sorts entire table by created_at
-- Impact: Dashboard loads slowly with many jobs
-- Fix: Index on (user_id, status, created_at DESC) for fast dashboard queries
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_user_status_date 
ON ingestion_jobs(user_id, status, created_at DESC);

-- Index 4: external_items connector sync
-- Current: Filter by user_connection_id requires full scan
-- Impact: Connector sync queries are slow
-- Fix: Index on (user_connection_id, status, created_at DESC) for fast connector queries
CREATE INDEX IF NOT EXISTS idx_external_items_connection_status 
ON external_items(user_connection_id, status, created_at DESC);

-- Index 5: raw_events query optimization
-- Current: Queries by user_id and job_id are slow
-- Impact: Event retrieval during processing is slow
-- Fix: Index on (user_id, job_id, status) for fast event queries
CREATE INDEX IF NOT EXISTS idx_raw_events_user_job 
ON raw_events(user_id, job_id, status);

-- Index 6: Covering index for common duplicate check query
-- Current: Duplicate check requires multiple table lookups
-- Impact: Duplicate detection is slow
-- Fix: Covering index includes all columns needed for duplicate check
CREATE INDEX IF NOT EXISTS idx_raw_records_duplicate_check 
ON raw_records(user_id, file_hash, status) 
INCLUDE (id, file_name, created_at, metadata);

-- Verify indexes are created
-- Run: SELECT * FROM pg_indexes WHERE schemaname = 'public' AND tablename IN ('processing_locks', 'raw_records', 'ingestion_jobs', 'external_items', 'raw_events');
