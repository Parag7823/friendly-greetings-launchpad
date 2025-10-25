-- Migration: Fix Temporal Pattern SQL Errors
-- Date: 2025-10-25
-- Issues Fixed:
-- 1. Ambiguous column reference "source_event_id" in predict_missing_relationships
-- 2. Type mismatch (numeric vs double precision) in detect_temporal_anomalies
-- 3. Missing _store_computed_metrics method (handled in Python code)

-- ============================================================================
-- FIX 1: Fix ambiguous column reference in predict_missing_relationships
-- ============================================================================

CREATE OR REPLACE FUNCTION predict_missing_relationships(
    p_user_id UUID,
    p_lookback_days INTEGER DEFAULT 180
)
RETURNS TABLE(
    source_event_id UUID,
    predicted_target_type VARCHAR(100),
    relationship_type VARCHAR(100),
    expected_date TIMESTAMP WITH TIME ZONE,
    confidence_score FLOAT,
    days_overdue INTEGER
) AS $$
BEGIN
    RETURN QUERY
    WITH source_events AS (
        -- Find events that should have follow-up relationships
        SELECT 
            se.id as event_id,
            se.document_type,
            se.source_ts,
            tp.relationship_type,
            tp.avg_days_between,
            tp.std_dev_days,
            tp.confidence_score
        FROM public.raw_events se
        CROSS JOIN public.temporal_patterns tp
        WHERE se.user_id = p_user_id
            AND tp.user_id = p_user_id
            AND se.source_ts > NOW() - INTERVAL '1 day' * p_lookback_days
            AND tp.sample_count >= 3
    ),
    existing_relationships AS (
        -- Find which events already have relationships
        -- FIX: Use table alias to avoid ambiguous column reference
        SELECT DISTINCT ri.source_event_id
        FROM public.relationship_instances ri
        WHERE ri.user_id = p_user_id
    )
    SELECT 
        se.event_id as source_event_id,
        'unknown'::VARCHAR(100) as predicted_target_type,
        se.relationship_type,
        (se.source_ts + INTERVAL '1 day' * se.avg_days_between) as expected_date,
        se.confidence_score,
        EXTRACT(DAY FROM NOW() - (se.source_ts + INTERVAL '1 day' * se.avg_days_between))::INTEGER as days_overdue
    FROM source_events se
    WHERE se.event_id NOT IN (SELECT source_event_id FROM existing_relationships)
        AND (se.source_ts + INTERVAL '1 day' * se.avg_days_between) < NOW()
    ORDER BY days_overdue DESC;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================================
-- FIX 2: Fix type mismatch in detect_temporal_anomalies
-- ============================================================================

CREATE OR REPLACE FUNCTION detect_temporal_anomalies(
    p_user_id UUID,
    p_threshold_std_dev FLOAT DEFAULT 2.0
)
RETURNS TABLE(
    relationship_id UUID,
    relationship_type VARCHAR(100),
    expected_days DOUBLE PRECISION,  -- Changed from FLOAT to DOUBLE PRECISION
    actual_days DOUBLE PRECISION,    -- Changed from FLOAT to DOUBLE PRECISION
    deviation_days DOUBLE PRECISION, -- Changed from FLOAT to DOUBLE PRECISION
    severity VARCHAR(50)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ri.id as relationship_id,
        ri.relationship_type,
        tp.avg_days_between::DOUBLE PRECISION as expected_days,  -- Explicit cast
        (EXTRACT(EPOCH FROM (te.source_ts - se.source_ts)) / 86400.0)::DOUBLE PRECISION as actual_days,  -- Explicit cast
        ABS((EXTRACT(EPOCH FROM (te.source_ts - se.source_ts)) / 86400.0) - tp.avg_days_between)::DOUBLE PRECISION as deviation_days,  -- Explicit cast
        CASE 
            WHEN ABS((EXTRACT(EPOCH FROM (te.source_ts - se.source_ts)) / 86400.0) - tp.avg_days_between) > tp.std_dev_days * 3 THEN 'critical'::VARCHAR(50)
            WHEN ABS((EXTRACT(EPOCH FROM (te.source_ts - se.source_ts)) / 86400.0) - tp.avg_days_between) > tp.std_dev_days * 2 THEN 'high'::VARCHAR(50)
            WHEN ABS((EXTRACT(EPOCH FROM (te.source_ts - se.source_ts)) / 86400.0) - tp.avg_days_between) > tp.std_dev_days THEN 'medium'::VARCHAR(50)
            ELSE 'low'::VARCHAR(50)
        END as severity
    FROM public.relationship_instances ri
    JOIN public.raw_events se ON ri.source_event_id = se.id
    JOIN public.raw_events te ON ri.target_event_id = te.id
    JOIN public.temporal_patterns tp ON tp.relationship_type = ri.relationship_type AND tp.user_id = ri.user_id
    WHERE ri.user_id = p_user_id
        AND tp.std_dev_days > 0
        AND ABS((EXTRACT(EPOCH FROM (te.source_ts - se.source_ts)) / 86400.0) - tp.avg_days_between) > tp.std_dev_days * p_threshold_std_dev
    ORDER BY deviation_days DESC;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================================
-- VERIFICATION
-- ============================================================================

-- Test predict_missing_relationships (should not error)
-- SELECT * FROM predict_missing_relationships('00000000-0000-0000-0000-000000000000'::UUID) LIMIT 1;

-- Test detect_temporal_anomalies (should not error)
-- SELECT * FROM detect_temporal_anomalies('00000000-0000-0000-0000-000000000000'::UUID) LIMIT 1;

-- Add comments
COMMENT ON FUNCTION predict_missing_relationships IS 'Fixed: Resolves ambiguous column reference for source_event_id';
COMMENT ON FUNCTION detect_temporal_anomalies IS 'Fixed: Returns DOUBLE PRECISION types to match expected function signature';
