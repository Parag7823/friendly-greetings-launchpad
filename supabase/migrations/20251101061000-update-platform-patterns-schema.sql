-- Migration: align platform_patterns schema with backend storage fields
-- Date: 2025-11-01 06:10:00

-- Ensure new metadata columns exist
ALTER TABLE public.platform_patterns
    ADD COLUMN IF NOT EXISTS pattern_type VARCHAR(100) DEFAULT 'column_structure',
    ADD COLUMN IF NOT EXISTS pattern_data JSONB DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS confidence_score NUMERIC(5,4) DEFAULT 0.0,
    ADD COLUMN IF NOT EXISTS detection_method VARCHAR(100) DEFAULT 'unknown';

-- Backfill pattern_data from legacy `patterns` column when available
UPDATE public.platform_patterns
SET pattern_data = COALESCE(pattern_data, patterns)
WHERE patterns IS NOT NULL
  AND (pattern_data IS NULL OR pattern_data = '{}'::jsonb);

COMMENT ON COLUMN public.platform_patterns.pattern_type IS 'Type/category of learned platform pattern (e.g., column_structure).';
COMMENT ON COLUMN public.platform_patterns.pattern_data IS 'Detailed pattern metadata captured during ingestion.';
COMMENT ON COLUMN public.platform_patterns.confidence_score IS 'Confidence score for the learned pattern.';
COMMENT ON COLUMN public.platform_patterns.detection_method IS 'Mechanism used to learn the pattern (ai_analysis, heuristic, etc.).';
