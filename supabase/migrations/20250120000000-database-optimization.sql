-- Database Optimization Migration
-- This migration adds indexes, optimizes queries, and improves performance
-- Date: 2025-01-20

-- ============================================================================
-- PERFORMANCE INDEXES
-- ============================================================================

-- Raw Events Table - Most frequently queried table
CREATE INDEX IF NOT EXISTS idx_raw_events_user_id_created_at ON public.raw_events(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_raw_events_user_id_status ON public.raw_events(user_id, status);
CREATE INDEX IF NOT EXISTS idx_raw_events_user_id_kind ON public.raw_events(user_id, kind);
CREATE INDEX IF NOT EXISTS idx_raw_events_user_id_source_platform ON public.raw_events(user_id, source_platform);
CREATE INDEX IF NOT EXISTS idx_raw_events_file_id_status ON public.raw_events(file_id, status);
CREATE INDEX IF NOT EXISTS idx_raw_events_job_id_status ON public.raw_events(job_id, status);
CREATE INDEX IF NOT EXISTS idx_raw_events_processed_at ON public.raw_events(processed_at) WHERE processed_at IS NOT NULL;

-- Raw Records Table
CREATE INDEX IF NOT EXISTS idx_raw_records_user_id_status ON public.raw_records(user_id, status);
CREATE INDEX IF NOT EXISTS idx_raw_records_user_id_created_at ON public.raw_records(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_raw_records_classification_status ON public.raw_records(classification_status);

-- Ingestion Jobs Table
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_user_id_status ON public.ingestion_jobs(user_id, status);
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_user_id_created_at ON public.ingestion_jobs(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status_created_at ON public.ingestion_jobs(status, created_at DESC);

-- Metrics Table
CREATE INDEX IF NOT EXISTS idx_metrics_user_id_metric_type ON public.metrics(user_id, metric_type);
CREATE INDEX IF NOT EXISTS idx_metrics_user_id_date_recorded ON public.metrics(user_id, date_recorded DESC);
CREATE INDEX IF NOT EXISTS idx_metrics_record_id ON public.metrics(record_id);

-- Normalized Entities Table
CREATE INDEX IF NOT EXISTS idx_normalized_entities_user_id_entity_type ON public.normalized_entities(user_id, entity_type);
CREATE INDEX IF NOT EXISTS idx_normalized_entities_user_id_canonical_name ON public.normalized_entities(user_id, canonical_name);
CREATE INDEX IF NOT EXISTS idx_normalized_entities_email ON public.normalized_entities(email) WHERE email IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_normalized_entities_phone ON public.normalized_entities(phone) WHERE phone IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_normalized_entities_tax_id ON public.normalized_entities(tax_id) WHERE tax_id IS NOT NULL;

-- Entity Matches Table
CREATE INDEX IF NOT EXISTS idx_entity_matches_user_id ON public.entity_matches(user_id);
CREATE INDEX IF NOT EXISTS idx_entity_matches_source_entity_id ON public.entity_matches(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_matches_target_entity_id ON public.entity_matches(target_entity_id);

-- File Versions Table
CREATE INDEX IF NOT EXISTS idx_file_versions_user_id ON public.file_versions(user_id);
CREATE INDEX IF NOT EXISTS idx_file_versions_version_group_id ON public.file_versions(version_group_id);
CREATE INDEX IF NOT EXISTS idx_file_versions_file_hash ON public.file_versions(file_hash);
CREATE INDEX IF NOT EXISTS idx_file_versions_is_active_version ON public.file_versions(is_active_version) WHERE is_active_version = true;

-- Cross Platform Relationships Table
CREATE INDEX IF NOT EXISTS idx_cross_platform_relationships_user_id ON public.cross_platform_relationships(user_id);
CREATE INDEX IF NOT EXISTS idx_cross_platform_relationships_source_event_id ON public.cross_platform_relationships(source_event_id);
CREATE INDEX IF NOT EXISTS idx_cross_platform_relationships_target_event_id ON public.cross_platform_relationships(target_event_id);
CREATE INDEX IF NOT EXISTS idx_cross_platform_relationships_relationship_type ON public.cross_platform_relationships(relationship_type);

-- Chat Messages Table
CREATE INDEX IF NOT EXISTS idx_chat_messages_user_id_created_at ON public.chat_messages(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_messages_chat_id_created_at ON public.chat_messages(chat_id, created_at DESC);

-- ============================================================================
-- OPTIMIZED DATABASE FUNCTIONS
-- ============================================================================

-- Function to get user events with pagination and filtering
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
    confidence_score DECIMAL(3,2),
    created_at TIMESTAMP WITH TIME ZONE
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
        re.created_at
    FROM public.raw_events re
    WHERE re.user_id = p_user_id
        AND (p_kind IS NULL OR re.kind = p_kind)
        AND (p_source_platform IS NULL OR re.source_platform = p_source_platform)
        AND (p_status IS NULL OR re.status = p_status)
    ORDER BY re.created_at DESC
    LIMIT p_limit
    OFFSET p_offset;
END;
$$ LANGUAGE plpgsql;

-- Function to get user metrics with aggregation
CREATE OR REPLACE FUNCTION get_user_metrics_optimized(
    p_user_id UUID,
    p_metric_type TEXT DEFAULT NULL,
    p_start_date DATE DEFAULT NULL,
    p_end_date DATE DEFAULT NULL
)
RETURNS TABLE (
    metric_type TEXT,
    category TEXT,
    total_amount DECIMAL(15,2),
    avg_amount DECIMAL(15,2),
    count_records BIGINT,
    avg_confidence DECIMAL(3,2)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        m.metric_type,
        m.category,
        SUM(m.amount) as total_amount,
        AVG(m.amount) as avg_amount,
        COUNT(*) as count_records,
        AVG(m.confidence_score) as avg_confidence
    FROM public.metrics m
    WHERE m.user_id = p_user_id
        AND (p_metric_type IS NULL OR m.metric_type = p_metric_type)
        AND (p_start_date IS NULL OR m.date_recorded >= p_start_date)
        AND (p_end_date IS NULL OR m.date_recorded <= p_end_date)
    GROUP BY m.metric_type, m.category
    ORDER BY total_amount DESC;
END;
$$ LANGUAGE plpgsql;

-- Function to get entity resolution statistics
CREATE OR REPLACE FUNCTION get_entity_resolution_stats(
    p_user_id UUID
)
RETURNS TABLE (
    entity_type TEXT,
    total_entities BIGINT,
    avg_confidence DECIMAL(3,2),
    platform_count BIGINT,
    last_updated TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ne.entity_type,
        COUNT(*) as total_entities,
        AVG(ne.confidence_score) as avg_confidence,
        COUNT(DISTINCT unnest(ne.platform_sources)) as platform_count,
        MAX(ne.updated_at) as last_updated
    FROM public.normalized_entities ne
    WHERE ne.user_id = p_user_id
    GROUP BY ne.entity_type
    ORDER BY total_entities DESC;
END;
$$ LANGUAGE plpgsql;

-- Function to get processing job statistics
CREATE OR REPLACE FUNCTION get_processing_job_stats(
    p_user_id UUID,
    p_days_back INTEGER DEFAULT 30
)
RETURNS TABLE (
    job_type TEXT,
    status TEXT,
    count_jobs BIGINT,
    avg_processing_time INTERVAL,
    success_rate DECIMAL(5,2)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ij.job_type,
        ij.status,
        COUNT(*) as count_jobs,
        AVG(ij.completed_at - ij.created_at) as avg_processing_time,
        CASE 
            WHEN COUNT(*) > 0 THEN 
                (COUNT(*) FILTER (WHERE ij.status = 'completed')::DECIMAL / COUNT(*)::DECIMAL) * 100
            ELSE 0
        END as success_rate
    FROM public.ingestion_jobs ij
    WHERE ij.user_id = p_user_id
        AND ij.created_at >= NOW() - INTERVAL '1 day' * p_days_back
    GROUP BY ij.job_type, ij.status
    ORDER BY count_jobs DESC;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- MATERIALIZED VIEWS FOR COMPLEX QUERIES
-- ============================================================================

-- Materialized view for user dashboard statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS user_dashboard_stats AS
SELECT 
    re.user_id,
    COUNT(DISTINCT re.file_id) as total_files,
    COUNT(*) as total_events,
    COUNT(DISTINCT re.source_platform) as unique_platforms,
    COUNT(DISTINCT re.kind) as unique_event_types,
    AVG(re.confidence_score) as avg_confidence,
    MAX(re.created_at) as last_activity,
    COUNT(*) FILTER (WHERE re.status = 'processed') as processed_events,
    COUNT(*) FILTER (WHERE re.status = 'failed') as failed_events
FROM public.raw_events re
GROUP BY re.user_id;

-- Create index on materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_user_dashboard_stats_user_id ON user_dashboard_stats(user_id);

-- Materialized view for platform usage statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS platform_usage_stats AS
SELECT 
    re.user_id,
    re.source_platform,
    COUNT(*) as event_count,
    COUNT(DISTINCT re.file_id) as file_count,
    AVG(re.confidence_score) as avg_confidence,
    MIN(re.created_at) as first_seen,
    MAX(re.created_at) as last_seen
FROM public.raw_events re
WHERE re.source_platform IS NOT NULL
GROUP BY re.user_id, re.source_platform;

-- Create index on materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_platform_usage_stats_user_platform ON platform_usage_stats(user_id, source_platform);

-- ============================================================================
-- REFRESH FUNCTIONS FOR MATERIALIZED VIEWS
-- ============================================================================

-- Function to refresh all materialized views
CREATE OR REPLACE FUNCTION refresh_all_materialized_views()
RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY user_dashboard_stats;
    REFRESH MATERIALIZED VIEW CONCURRENTLY platform_usage_stats;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- QUERY OPTIMIZATION HINTS
-- ============================================================================

-- Add comments for query optimization guidance
COMMENT ON TABLE public.raw_events IS 'Main events table - use pagination and specific column selection for large datasets';
COMMENT ON INDEX idx_raw_events_user_id_created_at IS 'Optimized for user activity queries with time-based ordering';
COMMENT ON INDEX idx_raw_events_user_id_status IS 'Optimized for filtering events by processing status';
COMMENT ON INDEX idx_raw_events_user_id_kind IS 'Optimized for filtering events by type (payroll, revenue, etc.)';

-- ============================================================================
-- PERFORMANCE MONITORING
-- ============================================================================

-- Function to get slow query statistics
CREATE OR REPLACE FUNCTION get_slow_query_stats()
RETURNS TABLE (
    query_text TEXT,
    avg_duration INTERVAL,
    call_count BIGINT,
    total_duration INTERVAL
) AS $$
BEGIN
    -- This would typically query pg_stat_statements if enabled
    -- For now, return empty result
    RETURN QUERY
    SELECT 
        'Query monitoring not enabled'::TEXT,
        '0 seconds'::INTERVAL,
        0::BIGINT,
        '0 seconds'::INTERVAL
    WHERE FALSE;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- CLEANUP AND MAINTENANCE
-- ============================================================================

-- Function to clean up old processed events (optional - for data retention)
CREATE OR REPLACE FUNCTION cleanup_old_processed_events(
    p_user_id UUID,
    p_days_to_keep INTEGER DEFAULT 90
)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM public.raw_events 
    WHERE user_id = p_user_id 
        AND status = 'processed' 
        AND created_at < NOW() - INTERVAL '1 day' * p_days_to_keep;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- GRANTS AND PERMISSIONS
-- ============================================================================

-- Grant execute permissions on functions to authenticated users
GRANT EXECUTE ON FUNCTION get_user_events_optimized TO authenticated;
GRANT EXECUTE ON FUNCTION get_user_metrics_optimized TO authenticated;
GRANT EXECUTE ON FUNCTION get_entity_resolution_stats TO authenticated;
GRANT EXECUTE ON FUNCTION get_processing_job_stats TO authenticated;
GRANT EXECUTE ON FUNCTION refresh_all_materialized_views TO authenticated;
GRANT EXECUTE ON FUNCTION cleanup_old_processed_events TO authenticated;

-- Grant select permissions on materialized views
GRANT SELECT ON user_dashboard_stats TO authenticated;
GRANT SELECT ON platform_usage_stats TO authenticated;
