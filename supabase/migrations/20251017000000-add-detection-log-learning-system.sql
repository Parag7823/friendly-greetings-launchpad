-- Migration: Add detection_log table for persistent learning system
-- Date: 2025-10-17
-- Purpose: Enable the learning system to persist detection history and improve over time

-- Create detection_log table for storing platform and document detection history
CREATE TABLE IF NOT EXISTS public.detection_log (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    
    -- Detection metadata
    detection_id TEXT NOT NULL,
    detection_type TEXT NOT NULL CHECK (detection_type IN ('platform', 'document')),
    
    -- Detection results
    detected_value TEXT NOT NULL, -- platform name or document type
    confidence NUMERIC(5, 4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    method TEXT NOT NULL, -- 'pattern_match', 'ai_classification', 'hybrid', etc.
    
    -- Context information
    indicators JSONB DEFAULT '[]'::jsonb, -- Array of indicators that led to detection
    payload_keys TEXT[] DEFAULT ARRAY[]::TEXT[], -- Keys present in the payload
    filename TEXT,
    
    -- Timestamps
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    
    -- Additional metadata
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_detection_log_user_id ON public.detection_log(user_id);
CREATE INDEX IF NOT EXISTS idx_detection_log_detection_type ON public.detection_log(detection_type);
CREATE INDEX IF NOT EXISTS idx_detection_log_detected_value ON public.detection_log(detected_value);
CREATE INDEX IF NOT EXISTS idx_detection_log_detected_at ON public.detection_log(detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_detection_log_confidence ON public.detection_log(confidence);

-- Composite index for common queries
CREATE INDEX IF NOT EXISTS idx_detection_log_user_type_value ON public.detection_log(user_id, detection_type, detected_value);

-- GIN index for JSONB columns
CREATE INDEX IF NOT EXISTS idx_detection_log_indicators ON public.detection_log USING GIN (indicators);
CREATE INDEX IF NOT EXISTS idx_detection_log_metadata ON public.detection_log USING GIN (metadata);

-- Enable Row Level Security
ALTER TABLE public.detection_log ENABLE ROW LEVEL SECURITY;

-- RLS Policies
DROP POLICY IF EXISTS "service_role_all_detection_log" ON public.detection_log;
CREATE POLICY "service_role_all_detection_log" ON public.detection_log
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_detection_log" ON public.detection_log;
CREATE POLICY "users_own_detection_log" ON public.detection_log
    FOR ALL USING (auth.uid() = user_id);

-- Function to analyze detection patterns and update platform_patterns table
CREATE OR REPLACE FUNCTION analyze_detection_patterns(
    p_user_id UUID,
    p_detection_type TEXT,
    p_min_occurrences INTEGER DEFAULT 5,
    p_min_confidence NUMERIC DEFAULT 0.8
)
RETURNS TABLE(
    detected_value TEXT,
    occurrence_count BIGINT,
    avg_confidence NUMERIC,
    common_indicators JSONB,
    suggested_pattern JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        dl.detected_value,
        COUNT(*) as occurrence_count,
        AVG(dl.confidence) as avg_confidence,
        jsonb_agg(DISTINCT indicator) FILTER (WHERE indicator IS NOT NULL) as common_indicators,
        jsonb_build_object(
            'value', dl.detected_value,
            'confidence_threshold', AVG(dl.confidence),
            'indicators', jsonb_agg(DISTINCT indicator) FILTER (WHERE indicator IS NOT NULL),
            'sample_count', COUNT(*)
        ) as suggested_pattern
    FROM public.detection_log dl,
         LATERAL jsonb_array_elements_text(dl.indicators) as indicator
    WHERE dl.user_id = p_user_id
      AND dl.detection_type = p_detection_type
      AND dl.confidence >= p_min_confidence
    GROUP BY dl.detected_value
    HAVING COUNT(*) >= p_min_occurrences
    ORDER BY occurrence_count DESC, avg_confidence DESC;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get detection statistics for a user
CREATE OR REPLACE FUNCTION get_detection_statistics(p_user_id UUID)
RETURNS JSONB AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'total_detections', COUNT(*),
        'platform_detections', COUNT(*) FILTER (WHERE detection_type = 'platform'),
        'document_detections', COUNT(*) FILTER (WHERE detection_type = 'document'),
        'avg_confidence', AVG(confidence),
        'high_confidence_rate', 
            COUNT(*) FILTER (WHERE confidence >= 0.9)::NUMERIC / NULLIF(COUNT(*), 0),
        'unique_platforms', 
            COUNT(DISTINCT detected_value) FILTER (WHERE detection_type = 'platform'),
        'unique_documents', 
            COUNT(DISTINCT detected_value) FILTER (WHERE detection_type = 'document'),
        'detection_methods', (
            SELECT jsonb_object_agg(method, method_count)
            FROM (
                SELECT method, COUNT(*) as method_count
                FROM public.detection_log
                WHERE user_id = p_user_id
                GROUP BY method
            ) methods
        )
    ) INTO result
    FROM public.detection_log
    WHERE user_id = p_user_id;
    
    RETURN COALESCE(result, '{}'::jsonb);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to cleanup old detection logs (keep last 90 days)
CREATE OR REPLACE FUNCTION cleanup_old_detection_logs(days_to_keep INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM public.detection_log
    WHERE detected_at < (now() - INTERVAL '1 day' * days_to_keep);
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger to update platform_patterns based on detection_log
CREATE OR REPLACE FUNCTION auto_update_platform_patterns()
RETURNS TRIGGER AS $$
BEGIN
    -- Only process high-confidence platform detections
    IF NEW.detection_type = 'platform' AND NEW.confidence >= 0.85 THEN
        -- Check if pattern exists and update or insert
        INSERT INTO public.platform_patterns (
            user_id,
            platform,
            pattern_type,
            pattern_data,
            confidence_score,
            usage_count,
            last_used_at
        )
        VALUES (
            NEW.user_id,
            NEW.detected_value,
            'learned_from_detection',
            jsonb_build_object(
                'indicators', NEW.indicators,
                'method', NEW.method,
                'detection_id', NEW.detection_id
            ),
            NEW.confidence,
            1,
            NEW.detected_at
        )
        ON CONFLICT (user_id, platform, pattern_type) 
        DO UPDATE SET
            usage_count = platform_patterns.usage_count + 1,
            last_used_at = NEW.detected_at,
            confidence_score = (platform_patterns.confidence_score + NEW.confidence) / 2,
            pattern_data = platform_patterns.pattern_data || NEW.indicators;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS trigger_auto_update_platform_patterns ON public.detection_log;
CREATE TRIGGER trigger_auto_update_platform_patterns
    AFTER INSERT ON public.detection_log
    FOR EACH ROW
    EXECUTE FUNCTION auto_update_platform_patterns();

-- Grant necessary permissions
GRANT SELECT, INSERT ON public.detection_log TO authenticated;
GRANT EXECUTE ON FUNCTION analyze_detection_patterns TO authenticated;
GRANT EXECUTE ON FUNCTION get_detection_statistics TO authenticated;

-- Add comment
COMMENT ON TABLE public.detection_log IS 'Stores detection history for platform and document classification learning system';
COMMENT ON FUNCTION analyze_detection_patterns IS 'Analyzes detection patterns to suggest improvements to platform_patterns table';
COMMENT ON FUNCTION get_detection_statistics IS 'Returns detection statistics for a user';
COMMENT ON FUNCTION cleanup_old_detection_logs IS 'Cleans up old detection logs to maintain performance';
