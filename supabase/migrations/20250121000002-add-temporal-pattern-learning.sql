-- Migration: Add temporal pattern learning tables
-- Date: 2025-01-21
-- Purpose: Enable temporal pattern analysis, seasonal detection, and relationship prediction

-- ============================================================================
-- 1. TEMPORAL PATTERNS TABLE
-- ============================================================================
-- Stores learned temporal patterns for relationship types
CREATE TABLE IF NOT EXISTS public.temporal_patterns (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL,
    relationship_type VARCHAR(100) NOT NULL,
    
    -- Timing statistics
    avg_days_between FLOAT NOT NULL,
    std_dev_days FLOAT DEFAULT 0.0,
    min_days FLOAT DEFAULT 0.0,
    max_days FLOAT DEFAULT 0.0,
    median_days FLOAT DEFAULT 0.0,
    
    -- Pattern confidence
    sample_count INTEGER DEFAULT 0,
    confidence_score FLOAT DEFAULT 0.0,
    
    -- Seasonal patterns
    has_seasonal_pattern BOOLEAN DEFAULT FALSE,
    seasonal_period_days INTEGER,
    seasonal_amplitude FLOAT,
    
    -- Pattern metadata
    pattern_description TEXT,
    learned_from_relationship_ids JSONB DEFAULT '[]',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_validated_at TIMESTAMP WITH TIME ZONE,
    
    UNIQUE(user_id, relationship_type)
);

-- ============================================================================
-- 2. PREDICTED RELATIONSHIPS TABLE
-- ============================================================================
-- Stores predicted/expected relationships that haven't occurred yet
CREATE TABLE IF NOT EXISTS public.predicted_relationships (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL,
    
    -- Prediction details
    source_event_id UUID NOT NULL,
    predicted_target_type VARCHAR(100) NOT NULL,
    relationship_type VARCHAR(100) NOT NULL,
    
    -- Timing prediction
    expected_date TIMESTAMP WITH TIME ZONE NOT NULL,
    expected_date_range_start TIMESTAMP WITH TIME ZONE,
    expected_date_range_end TIMESTAMP WITH TIME ZONE,
    days_until_expected INTEGER,
    
    -- Prediction confidence
    confidence_score FLOAT DEFAULT 0.0,
    prediction_method VARCHAR(100),
    
    -- Status tracking
    status VARCHAR(50) DEFAULT 'pending' CHECK (status IN ('pending', 'fulfilled', 'overdue', 'cancelled')),
    fulfilled_by_event_id UUID,
    fulfilled_at TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    prediction_reasoning TEXT,
    temporal_pattern_id UUID,
    metadata JSONB DEFAULT '{}',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    FOREIGN KEY (source_event_id) REFERENCES public.raw_events(id) ON DELETE CASCADE,
    FOREIGN KEY (fulfilled_by_event_id) REFERENCES public.raw_events(id) ON DELETE SET NULL,
    FOREIGN KEY (temporal_pattern_id) REFERENCES public.temporal_patterns(id) ON DELETE SET NULL
);

-- ============================================================================
-- 3. TEMPORAL ANOMALIES TABLE
-- ============================================================================
-- Stores detected temporal anomalies (relationships that deviate from patterns)
CREATE TABLE IF NOT EXISTS public.temporal_anomalies (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL,
    relationship_id UUID NOT NULL,
    
    -- Anomaly details
    anomaly_type VARCHAR(100) NOT NULL,
    expected_days FLOAT NOT NULL,
    actual_days FLOAT NOT NULL,
    deviation_days FLOAT NOT NULL,
    deviation_percentage FLOAT NOT NULL,
    
    -- Severity
    severity VARCHAR(50) CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    anomaly_score FLOAT DEFAULT 0.0,
    
    -- Context
    temporal_pattern_id UUID,
    anomaly_description TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    FOREIGN KEY (relationship_id) REFERENCES public.relationship_instances(id) ON DELETE CASCADE,
    FOREIGN KEY (temporal_pattern_id) REFERENCES public.temporal_patterns(id) ON DELETE SET NULL
);

-- ============================================================================
-- 4. SEASONAL PATTERNS TABLE
-- ============================================================================
-- Stores detected seasonal patterns in financial events
CREATE TABLE IF NOT EXISTS public.seasonal_patterns (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL,
    
    -- Pattern identification
    pattern_name VARCHAR(255) NOT NULL,
    pattern_type VARCHAR(100) NOT NULL,
    event_type VARCHAR(100),
    
    -- Seasonal characteristics
    period_days INTEGER NOT NULL,
    amplitude FLOAT DEFAULT 0.0,
    phase_offset_days INTEGER DEFAULT 0,
    
    -- Statistical validation
    confidence_score FLOAT DEFAULT 0.0,
    p_value FLOAT,
    sample_count INTEGER DEFAULT 0,
    
    -- Pattern description
    description TEXT,
    detected_cycles JSONB DEFAULT '[]',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(user_id, pattern_name)
);

-- ============================================================================
-- INDEXES
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_temporal_patterns_user ON public.temporal_patterns(user_id);
CREATE INDEX IF NOT EXISTS idx_temporal_patterns_type ON public.temporal_patterns(relationship_type);
CREATE INDEX IF NOT EXISTS idx_temporal_patterns_confidence ON public.temporal_patterns(confidence_score DESC);

CREATE INDEX IF NOT EXISTS idx_predicted_relationships_user ON public.predicted_relationships(user_id);
CREATE INDEX IF NOT EXISTS idx_predicted_relationships_source ON public.predicted_relationships(source_event_id);
CREATE INDEX IF NOT EXISTS idx_predicted_relationships_status ON public.predicted_relationships(status);
CREATE INDEX IF NOT EXISTS idx_predicted_relationships_expected_date ON public.predicted_relationships(expected_date);

CREATE INDEX IF NOT EXISTS idx_temporal_anomalies_user ON public.temporal_anomalies(user_id);
CREATE INDEX IF NOT EXISTS idx_temporal_anomalies_relationship ON public.temporal_anomalies(relationship_id);
CREATE INDEX IF NOT EXISTS idx_temporal_anomalies_severity ON public.temporal_anomalies(severity);

CREATE INDEX IF NOT EXISTS idx_seasonal_patterns_user ON public.seasonal_patterns(user_id);
CREATE INDEX IF NOT EXISTS idx_seasonal_patterns_type ON public.seasonal_patterns(pattern_type);

-- ============================================================================
-- POSTGRESQL FUNCTIONS
-- ============================================================================

-- Function to learn temporal pattern from relationships
CREATE OR REPLACE FUNCTION learn_temporal_pattern(
    p_user_id UUID,
    p_relationship_type VARCHAR(100)
)
RETURNS JSONB AS $$
DECLARE
    pattern_stats RECORD;
    result JSONB;
BEGIN
    -- Calculate timing statistics for relationship type
    SELECT 
        COUNT(*) as sample_count,
        AVG(EXTRACT(EPOCH FROM (te.source_ts - se.source_ts)) / 86400.0) as avg_days,
        STDDEV(EXTRACT(EPOCH FROM (te.source_ts - se.source_ts)) / 86400.0) as std_dev,
        MIN(EXTRACT(EPOCH FROM (te.source_ts - se.source_ts)) / 86400.0) as min_days,
        MAX(EXTRACT(EPOCH FROM (te.source_ts - se.source_ts)) / 86400.0) as max_days,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (te.source_ts - se.source_ts)) / 86400.0) as median_days
    INTO pattern_stats
    FROM public.relationship_instances ri
    JOIN public.raw_events se ON ri.source_event_id = se.id
    JOIN public.raw_events te ON ri.target_event_id = te.id
    WHERE ri.user_id = p_user_id
        AND ri.relationship_type = p_relationship_type
        AND te.source_ts > se.source_ts;
    
    IF pattern_stats.sample_count < 3 THEN
        RETURN json_build_object('error', 'Insufficient data for pattern learning (minimum 3 samples required)');
    END IF;
    
    -- Calculate confidence score based on sample size and consistency
    result := json_build_object(
        'relationship_type', p_relationship_type,
        'avg_days_between', ROUND(pattern_stats.avg_days::numeric, 2),
        'std_dev_days', ROUND(COALESCE(pattern_stats.std_dev, 0)::numeric, 2),
        'min_days', ROUND(pattern_stats.min_days::numeric, 2),
        'max_days', ROUND(pattern_stats.max_days::numeric, 2),
        'median_days', ROUND(pattern_stats.median_days::numeric, 2),
        'sample_count', pattern_stats.sample_count,
        'confidence_score', ROUND(LEAST(1.0, pattern_stats.sample_count / 10.0)::numeric, 2),
        'pattern_description', 
            CASE 
                WHEN pattern_stats.avg_days < 7 THEN 'Immediate (< 1 week)'
                WHEN pattern_stats.avg_days < 30 THEN 'Short-term (< 1 month)'
                WHEN pattern_stats.avg_days < 90 THEN 'Medium-term (1-3 months)'
                ELSE 'Long-term (> 3 months)'
            END
    );
    
    RETURN result;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to predict missing relationships
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
        SELECT DISTINCT source_event_id
        FROM public.relationship_instances
        WHERE user_id = p_user_id
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

-- Function to detect temporal anomalies
CREATE OR REPLACE FUNCTION detect_temporal_anomalies(
    p_user_id UUID,
    p_threshold_std_dev FLOAT DEFAULT 2.0
)
RETURNS TABLE(
    relationship_id UUID,
    relationship_type VARCHAR(100),
    expected_days FLOAT,
    actual_days FLOAT,
    deviation_days FLOAT,
    severity VARCHAR(50)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ri.id as relationship_id,
        ri.relationship_type,
        tp.avg_days_between as expected_days,
        EXTRACT(EPOCH FROM (te.source_ts - se.source_ts)) / 86400.0 as actual_days,
        ABS(EXTRACT(EPOCH FROM (te.source_ts - se.source_ts)) / 86400.0 - tp.avg_days_between) as deviation_days,
        CASE 
            WHEN ABS(EXTRACT(EPOCH FROM (te.source_ts - se.source_ts)) / 86400.0 - tp.avg_days_between) > tp.std_dev_days * 3 THEN 'critical'::VARCHAR(50)
            WHEN ABS(EXTRACT(EPOCH FROM (te.source_ts - se.source_ts)) / 86400.0 - tp.avg_days_between) > tp.std_dev_days * 2 THEN 'high'::VARCHAR(50)
            WHEN ABS(EXTRACT(EPOCH FROM (te.source_ts - se.source_ts)) / 86400.0 - tp.avg_days_between) > tp.std_dev_days THEN 'medium'::VARCHAR(50)
            ELSE 'low'::VARCHAR(50)
        END as severity
    FROM public.relationship_instances ri
    JOIN public.raw_events se ON ri.source_event_id = se.id
    JOIN public.raw_events te ON ri.target_event_id = te.id
    JOIN public.temporal_patterns tp ON tp.relationship_type = ri.relationship_type AND tp.user_id = ri.user_id
    WHERE ri.user_id = p_user_id
        AND tp.std_dev_days > 0
        AND ABS(EXTRACT(EPOCH FROM (te.source_ts - se.source_ts)) / 86400.0 - tp.avg_days_between) > tp.std_dev_days * p_threshold_std_dev
    ORDER BY deviation_days DESC;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get temporal pattern statistics
CREATE OR REPLACE FUNCTION get_temporal_pattern_stats(p_user_id UUID)
RETURNS JSON AS $$
DECLARE
    result JSON;
BEGIN
    SELECT json_build_object(
        'total_patterns', COUNT(*),
        'avg_confidence', ROUND(AVG(confidence_score)::numeric, 3),
        'patterns_by_type', (
            SELECT json_object_agg(relationship_type, count)
            FROM (
                SELECT relationship_type, COUNT(*) as count
                FROM public.temporal_patterns
                WHERE user_id = p_user_id
                GROUP BY relationship_type
            ) t
        ),
        'seasonal_patterns_count', COUNT(*) FILTER (WHERE has_seasonal_pattern = TRUE),
        'total_predictions', (
            SELECT COUNT(*) FROM public.predicted_relationships WHERE user_id = p_user_id
        ),
        'pending_predictions', (
            SELECT COUNT(*) FROM public.predicted_relationships WHERE user_id = p_user_id AND status = 'pending'
        ),
        'overdue_predictions', (
            SELECT COUNT(*) FROM public.predicted_relationships WHERE user_id = p_user_id AND status = 'overdue'
        ),
        'total_anomalies', (
            SELECT COUNT(*) FROM public.temporal_anomalies WHERE user_id = p_user_id
        )
    ) INTO result
    FROM public.temporal_patterns
    WHERE user_id = p_user_id;
    
    RETURN COALESCE(result, '{}'::json);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_temporal_pattern_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_temporal_pattern_updated_at ON public.temporal_patterns;
CREATE TRIGGER trigger_update_temporal_pattern_updated_at
    BEFORE UPDATE ON public.temporal_patterns
    FOR EACH ROW
    EXECUTE FUNCTION update_temporal_pattern_updated_at();

DROP TRIGGER IF EXISTS trigger_update_predicted_relationship_updated_at ON public.predicted_relationships;
CREATE TRIGGER trigger_update_predicted_relationship_updated_at
    BEFORE UPDATE ON public.predicted_relationships
    FOR EACH ROW
    EXECUTE FUNCTION update_temporal_pattern_updated_at();

-- Grant permissions
GRANT SELECT, INSERT, UPDATE ON public.temporal_patterns TO authenticated;
GRANT SELECT, INSERT, UPDATE ON public.predicted_relationships TO authenticated;
GRANT SELECT, INSERT ON public.temporal_anomalies TO authenticated;
GRANT SELECT, INSERT, UPDATE ON public.seasonal_patterns TO authenticated;

GRANT EXECUTE ON FUNCTION learn_temporal_pattern TO authenticated;
GRANT EXECUTE ON FUNCTION predict_missing_relationships TO authenticated;
GRANT EXECUTE ON FUNCTION detect_temporal_anomalies TO authenticated;
GRANT EXECUTE ON FUNCTION get_temporal_pattern_stats TO authenticated;

-- Add comments
COMMENT ON TABLE public.temporal_patterns IS 'Learned temporal patterns for relationship types (e.g., invoices paid in 30±5 days)';
COMMENT ON TABLE public.predicted_relationships IS 'Predicted/expected relationships that should occur based on learned patterns';
COMMENT ON TABLE public.temporal_anomalies IS 'Detected anomalies where relationships deviate significantly from learned patterns';
COMMENT ON TABLE public.seasonal_patterns IS 'Detected seasonal patterns and cycles in financial events';

COMMENT ON FUNCTION learn_temporal_pattern IS 'Learns timing patterns for a relationship type from historical data';
COMMENT ON FUNCTION predict_missing_relationships IS 'Predicts relationships that should exist but are missing';
COMMENT ON FUNCTION detect_temporal_anomalies IS 'Detects relationships with unusual timing compared to learned patterns';
