-- Migration: Add JSONB structure validation for platform_patterns
-- Date: 2025-11-15
-- Purpose: Standardize platform_patterns JSONB structure and add validation

-- Create validation function for platform pattern structure
CREATE OR REPLACE FUNCTION validate_platform_pattern_structure(pattern_data JSONB)
RETURNS BOOLEAN AS $$
BEGIN
    -- Check required fields exist
    RETURN (
        pattern_data ? 'pattern_type' AND
        pattern_data ? 'confidence_score' AND  
        pattern_data ? 'detection_method' AND
        pattern_data ? 'pattern_data' AND
        -- Validate data types
        jsonb_typeof(pattern_data->'pattern_type') = 'string' AND
        jsonb_typeof(pattern_data->'confidence_score') = 'number' AND
        jsonb_typeof(pattern_data->'detection_method') = 'string' AND
        jsonb_typeof(pattern_data->'pattern_data') = 'object' AND
        -- Validate confidence_score range
        (pattern_data->>'confidence_score')::DECIMAL >= 0 AND
        (pattern_data->>'confidence_score')::DECIMAL <= 1
    );
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Add constraint to enforce JSONB structure
ALTER TABLE public.platform_patterns 
ADD CONSTRAINT valid_pattern_structure 
CHECK (validate_platform_pattern_structure(patterns));

-- Add GIN index for efficient JSONB queries
CREATE INDEX IF NOT EXISTS idx_platform_patterns_gin 
ON public.platform_patterns USING GIN (patterns);

-- Add expression indexes for common queries
CREATE INDEX IF NOT EXISTS idx_platform_patterns_pattern_type 
ON public.platform_patterns ((patterns->>'pattern_type'));

CREATE INDEX IF NOT EXISTS idx_platform_patterns_confidence_score 
ON public.platform_patterns ((patterns->>'confidence_score'));

CREATE INDEX IF NOT EXISTS idx_platform_patterns_detection_method 
ON public.platform_patterns ((patterns->>'detection_method'));

-- Update existing function to handle new structure
CREATE OR REPLACE FUNCTION get_platform_patterns_structured(p_user_id UUID)
RETURNS TABLE (
    platform VARCHAR(100),
    pattern_type TEXT,
    confidence_score DECIMAL(3,2),
    detection_method TEXT,
    pattern_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        pp.platform,
        (pp.patterns->>'pattern_type')::TEXT as pattern_type,
        (pp.patterns->>'confidence_score')::DECIMAL(3,2) as confidence_score,
        (pp.patterns->>'detection_method')::TEXT as detection_method,
        pp.patterns->'pattern_data' as pattern_data,
        pp.created_at,
        pp.updated_at
    FROM public.platform_patterns pp
    WHERE pp.user_id = p_user_id
    ORDER BY pp.updated_at DESC;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Grant permissions
GRANT EXECUTE ON FUNCTION get_platform_patterns_structured(UUID) TO authenticated;

-- Add comments for documentation
COMMENT ON FUNCTION validate_platform_pattern_structure(JSONB) IS 'Validates platform pattern JSONB structure has required fields and correct types';
COMMENT ON FUNCTION get_platform_patterns_structured(UUID) IS 'Returns platform patterns with structured access to JSONB fields';

-- Log migration completion
DO $$
BEGIN
    RAISE NOTICE 'Migration 20251115000001: Added JSONB structure validation for platform_patterns';
    RAISE NOTICE 'Added constraint: valid_pattern_structure';
    RAISE NOTICE 'Added GIN and expression indexes for efficient queries';
    RAISE NOTICE 'Created structured access function: get_platform_patterns_structured';
END $$;
