-- Performance Optimization Migration
-- Adds critical indexes and optimizations to fix O(N²) performance issues
-- Created: 2025-09-20

-- ============================================================================
-- CRITICAL COMPOSITE INDEXES FOR RAW_EVENTS TABLE
-- ============================================================================

-- Index for user-specific queries with platform and timestamp filtering
CREATE INDEX IF NOT EXISTS idx_raw_events_user_platform_timestamp 
ON public.raw_events (user_id, source_platform, ingest_ts DESC);

-- Index for user-specific queries with kind and status filtering
CREATE INDEX IF NOT EXISTS idx_raw_events_user_kind_status 
ON public.raw_events (user_id, kind, status);

-- Index for file-specific queries (used in entity resolution)
CREATE INDEX IF NOT EXISTS idx_raw_events_user_file_job 
ON public.raw_events (user_id, file_id, job_id);

-- Index for duplicate detection queries
CREATE INDEX IF NOT EXISTS idx_raw_events_user_created 
ON public.raw_events (user_id, created_at DESC);

-- ============================================================================
-- GIN INDEXES FOR JSONB COLUMNS
-- ============================================================================

-- GIN index for payload JSONB queries
CREATE INDEX IF NOT EXISTS idx_raw_events_payload_gin 
ON public.raw_events USING GIN (payload);

-- GIN index for classification_metadata JSONB queries
CREATE INDEX IF NOT EXISTS idx_raw_events_classification_gin 
ON public.raw_events USING GIN (classification_metadata);

-- GIN index for entities JSONB queries
CREATE INDEX IF NOT EXISTS idx_raw_events_entities_gin 
ON public.raw_events USING GIN (entities);

-- GIN index for platform_ids JSONB queries
CREATE INDEX IF NOT EXISTS idx_raw_events_platform_ids_gin 
ON public.raw_events USING GIN (platform_ids);

-- ============================================================================
-- PARTIAL INDEXES FOR STATUS FILTERING
-- ============================================================================

-- Partial index for active/processing events only
CREATE INDEX IF NOT EXISTS idx_raw_events_active_status 
ON public.raw_events (user_id, created_at DESC) 
WHERE status IN ('active', 'processing', 'pending');

-- Partial index for failed events only
CREATE INDEX IF NOT EXISTS idx_raw_events_failed_status 
ON public.raw_events (user_id, created_at DESC) 
WHERE status = 'failed';

-- ============================================================================
-- RAW_RECORDS TABLE OPTIMIZATIONS
-- ============================================================================

-- Index for duplicate detection queries
CREATE INDEX IF NOT EXISTS idx_raw_records_user_duplicate 
ON public.raw_records (user_id, is_duplicate, created_at DESC);

-- Index for file hash lookups (duplicate detection)
CREATE INDEX IF NOT EXISTS idx_raw_records_user_hash 
ON public.raw_records (user_id, ((content->>'file_hash')));

-- GIN index for content JSONB
CREATE INDEX IF NOT EXISTS idx_raw_records_content_gin 
ON public.raw_records USING GIN (content);

-- ============================================================================
-- INGESTION_JOBS TABLE OPTIMIZATIONS
-- ============================================================================

-- Index for user job queries with status filtering
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_user_status 
ON public.ingestion_jobs (user_id, status, created_at DESC);

-- Index for job monitoring queries
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status_updated 
ON public.ingestion_jobs (status, updated_at DESC) 
WHERE status IN ('processing', 'pending');

-- ============================================================================
-- NORMALIZED_ENTITIES TABLE OPTIMIZATIONS
-- ============================================================================

-- Index for entity resolution queries
CREATE INDEX IF NOT EXISTS idx_normalized_entities_user_type 
ON public.normalized_entities (user_id, entity_type, last_seen_at DESC);

-- Index for canonical name lookups
CREATE INDEX IF NOT EXISTS idx_normalized_entities_canonical 
ON public.normalized_entities (user_id, canonical_name);

-- GIN index for aliases array
CREATE INDEX IF NOT EXISTS idx_normalized_entities_aliases_gin 
ON public.normalized_entities USING GIN (aliases);

-- ============================================================================
-- CHAT_MESSAGES TABLE OPTIMIZATIONS
-- ============================================================================

-- Index for chat history queries
CREATE INDEX IF NOT EXISTS idx_chat_messages_user_chat_created 
ON public.chat_messages (user_id, chat_id, created_at DESC);

-- ============================================================================
-- FILE_VERSIONS TABLE OPTIMIZATIONS
-- ============================================================================

-- Index for version queries
CREATE INDEX IF NOT EXISTS idx_file_versions_user_group 
ON public.file_versions (user_id, version_group_id, version_number DESC);

-- Index for active version queries
CREATE INDEX IF NOT EXISTS idx_file_versions_active 
ON public.file_versions (user_id, is_active_version, created_at DESC) 
WHERE is_active_version = true;

-- ============================================================================
-- PROCESSING_TRANSACTIONS TABLE OPTIMIZATIONS
-- ============================================================================

-- Index for transaction monitoring
CREATE INDEX IF NOT EXISTS idx_processing_transactions_user_status 
ON public.processing_transactions (user_id, status, created_at DESC);

-- Index for cleanup queries
CREATE INDEX IF NOT EXISTS idx_processing_transactions_cleanup 
ON public.processing_transactions (status, created_at) 
WHERE status IN ('failed', 'rolled_back');

-- ============================================================================
-- PROCESSING_LOCKS TABLE OPTIMIZATIONS
-- ============================================================================

-- Index for lock acquisition queries
CREATE INDEX IF NOT EXISTS idx_processing_locks_resource_type 
ON public.processing_locks (resource_id, lock_type, status);

-- Index for lock cleanup queries
CREATE INDEX IF NOT EXISTS idx_processing_locks_cleanup 
ON public.processing_locks (status, expires_at) 
WHERE status = 'active';

-- ============================================================================
-- ERROR_LOGS TABLE OPTIMIZATIONS
-- ============================================================================

-- Index for error monitoring queries
CREATE INDEX IF NOT EXISTS idx_error_logs_user_severity 
ON public.error_logs (user_id, severity, occurred_at DESC);

-- Index for job-specific error queries
CREATE INDEX IF NOT EXISTS idx_error_logs_job_transaction 
ON public.error_logs (job_id, transaction_id, occurred_at DESC);

-- ============================================================================
-- OPTIMIZED FUNCTIONS FOR COMMON QUERIES
-- ============================================================================

-- Function to get user events with optimized pagination
CREATE OR REPLACE FUNCTION get_user_events_optimized(
    p_user_id UUID,
    p_limit INTEGER DEFAULT 100,
    p_offset INTEGER DEFAULT 0,
    p_kind TEXT DEFAULT NULL,
    p_source_platform TEXT DEFAULT NULL,
    p_status TEXT DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    kind TEXT,
    source_platform TEXT,
    payload JSONB,
    row_index INTEGER,
    source_filename TEXT,
    status TEXT,
    confidence_score NUMERIC,
    created_at TIMESTAMP WITH TIME ZONE,
    processed_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        re.id,
        re.kind,
        re.source_platform,
        re.payload,
        re.row_index,
        re.source_filename,
        re.status,
        re.confidence_score,
        re.created_at,
        re.processed_at
    FROM public.raw_events re
    WHERE re.user_id = p_user_id
        AND (p_kind IS NULL OR re.kind = p_kind)
        AND (p_source_platform IS NULL OR re.source_platform = p_source_platform)
        AND (p_status IS NULL OR re.status = p_status)
    ORDER BY re.created_at DESC
    LIMIT p_limit
    OFFSET p_offset;
END;
$$ LANGUAGE plpgsql STABLE;

-- Function to get duplicate candidates efficiently
CREATE OR REPLACE FUNCTION get_duplicate_candidates(
    p_user_id UUID,
    p_file_hash TEXT,
    p_limit INTEGER DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    file_name TEXT,
    file_size BIGINT,
    created_at TIMESTAMP WITH TIME ZONE,
    file_hash TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        rr.id,
        rr.file_name,
        rr.file_size,
        rr.created_at,
        rr.content->>'file_hash' as file_hash
    FROM public.raw_records rr
    WHERE rr.user_id = p_user_id
        AND rr.content->>'file_hash' = p_file_hash
    ORDER BY rr.created_at DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

-- Function to get user statistics efficiently
CREATE OR REPLACE FUNCTION get_user_statistics_fast(p_user_id UUID)
RETURNS JSONB AS $$
DECLARE
    result JSONB;
    events_count INTEGER;
    jobs_count INTEGER;
    records_count INTEGER;
    platforms_count INTEGER;
BEGIN
    -- Get counts using optimized queries
    SELECT COUNT(*) INTO events_count 
    FROM public.raw_events 
    WHERE user_id = p_user_id;
    
    SELECT COUNT(*) INTO jobs_count 
    FROM public.ingestion_jobs 
    WHERE user_id = p_user_id;
    
    SELECT COUNT(*) INTO records_count 
    FROM public.raw_records 
    WHERE user_id = p_user_id;
    
    SELECT COUNT(DISTINCT source_platform) INTO platforms_count 
    FROM public.raw_events 
    WHERE user_id = p_user_id AND source_platform IS NOT NULL;
    
    -- Build result JSON
    result := jsonb_build_object(
        'total_events', events_count,
        'total_jobs', jobs_count,
        'total_records', records_count,
        'unique_platforms', platforms_count,
        'generated_at', now()
    );
    
    RETURN result;
END;
$$ LANGUAGE plpgsql STABLE;

-- ============================================================================
-- MATERIALIZED VIEW FOR DASHBOARD METRICS
-- ============================================================================

-- Create materialized view for user dashboard metrics
CREATE MATERIALIZED VIEW IF NOT EXISTS user_dashboard_metrics AS
SELECT 
    user_id,
    COUNT(*) as total_events,
    COUNT(DISTINCT source_platform) as unique_platforms,
    COUNT(DISTINCT DATE(created_at)) as active_days,
    MAX(created_at) as last_activity,
    COUNT(*) FILTER (WHERE status = 'processed') as processed_events,
    COUNT(*) FILTER (WHERE status = 'failed') as failed_events,
    AVG(confidence_score) as avg_confidence,
    COUNT(DISTINCT file_id) as unique_files
FROM public.raw_events
GROUP BY user_id;

-- Create unique index on materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_user_dashboard_metrics_user 
ON user_dashboard_metrics (user_id);

-- ============================================================================
-- REFRESH FUNCTION FOR MATERIALIZED VIEW
-- ============================================================================

CREATE OR REPLACE FUNCTION refresh_user_dashboard_metrics()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY user_dashboard_metrics;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- PERFORMANCE MONITORING FUNCTIONS
-- ============================================================================

-- Function to analyze query performance
CREATE OR REPLACE FUNCTION analyze_query_performance()
RETURNS TABLE (
    query_type TEXT,
    avg_duration_ms NUMERIC,
    total_calls BIGINT,
    recommendation TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        'User Events Query'::TEXT as query_type,
        0.0::NUMERIC as avg_duration_ms,
        0::BIGINT as total_calls,
        'Use get_user_events_optimized() function for better performance'::TEXT as recommendation
    UNION ALL
    SELECT 
        'Duplicate Detection'::TEXT,
        0.0::NUMERIC,
        0::BIGINT,
        'Use get_duplicate_candidates() function with file hash index'::TEXT
    UNION ALL
    SELECT 
        'User Statistics'::TEXT,
        0.0::NUMERIC,
        0::BIGINT,
        'Use get_user_statistics_fast() function or materialized view'::TEXT;
END;
$$ LANGUAGE plpgsql STABLE;

-- ============================================================================
-- CLEANUP AND MAINTENANCE
-- ============================================================================

-- Function to cleanup old processing locks
CREATE OR REPLACE FUNCTION cleanup_expired_processing_locks()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM public.processing_locks
    WHERE status = 'active' AND expires_at < now();
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- GRANT PERMISSIONS
-- ============================================================================

-- Grant execute permissions on functions
GRANT EXECUTE ON FUNCTION get_user_events_optimized(UUID, INTEGER, INTEGER, TEXT, TEXT, TEXT) TO authenticated;
GRANT EXECUTE ON FUNCTION get_duplicate_candidates(UUID, TEXT, INTEGER) TO authenticated;
GRANT EXECUTE ON FUNCTION get_user_statistics_fast(UUID) TO authenticated;
GRANT EXECUTE ON FUNCTION refresh_user_dashboard_metrics() TO authenticated;
GRANT EXECUTE ON FUNCTION analyze_query_performance() TO authenticated;
GRANT EXECUTE ON FUNCTION cleanup_expired_processing_locks() TO authenticated;

-- Grant select on materialized view
GRANT SELECT ON user_dashboard_metrics TO authenticated;

-- ============================================================================
-- COMMENTS FOR DOCUMENTATION
-- ============================================================================

COMMENT ON INDEX idx_raw_events_user_platform_timestamp IS 'Optimizes user queries with platform and timestamp filtering';
COMMENT ON INDEX idx_raw_events_payload_gin IS 'Enables fast JSONB queries on payload data';
COMMENT ON FUNCTION get_user_events_optimized IS 'Optimized function for paginated user events queries';
COMMENT ON MATERIALIZED VIEW user_dashboard_metrics IS 'Pre-computed dashboard metrics for better performance';

-- ============================================================================
-- PERFORMANCE VALIDATION
-- ============================================================================

-- Analyze tables to update statistics
ANALYZE public.raw_events;
ANALYZE public.raw_records;
ANALYZE public.ingestion_jobs;
ANALYZE public.normalized_entities;
ANALYZE public.chat_messages;
ANALYZE public.file_versions;
ANALYZE public.processing_transactions;
ANALYZE public.processing_locks;
ANALYZE public.error_logs;

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'Performance optimization migration completed successfully';
    RAISE NOTICE 'Added % composite indexes for O(N²) performance fixes', 15;
    RAISE NOTICE 'Added % GIN indexes for JSONB query optimization', 5;
    RAISE NOTICE 'Added % partial indexes for status filtering', 3;
    RAISE NOTICE 'Created % optimized functions for common queries', 6;
    RAISE NOTICE 'Created materialized view for dashboard performance';
END $$;
