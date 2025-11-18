-- Migration: Add semantic, temporal, and causal metadata columns to normalized_events
-- Date: 2025-11-18
-- Purpose: Connect semantic/temporal/causal analysis outputs to normalized_events table

-- Add columns for semantic relationship evidence
ALTER TABLE public.normalized_events
ADD COLUMN IF NOT EXISTS relationship_evidence JSONB DEFAULT '{}',
ADD COLUMN IF NOT EXISTS relationship_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS semantic_links JSONB DEFAULT '[]';

-- Add columns for temporal pattern metadata
ALTER TABLE public.normalized_events
ADD COLUMN IF NOT EXISTS temporal_patterns JSONB DEFAULT '[]',
ADD COLUMN IF NOT EXISTS temporal_cycle_metadata JSONB DEFAULT '{}',
ADD COLUMN IF NOT EXISTS temporal_confidence DECIMAL(3,2) CHECK (temporal_confidence >= 0 AND temporal_confidence <= 1);

-- Add columns for causal analysis
ALTER TABLE public.normalized_events
ADD COLUMN IF NOT EXISTS causal_weight DECIMAL(5,3) DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS causal_links JSONB DEFAULT '[]',
ADD COLUMN IF NOT EXISTS causal_reasoning TEXT;

-- Add column for pattern prediction metadata
ALTER TABLE public.normalized_events
ADD COLUMN IF NOT EXISTS pattern_used_for_prediction TEXT,
ADD COLUMN IF NOT EXISTS prediction_confidence DECIMAL(3,2) CHECK (prediction_confidence >= 0 AND prediction_confidence <= 1);

-- Create indexes for new columns
CREATE INDEX IF NOT EXISTS idx_normalized_events_relationship_count ON public.normalized_events(relationship_count) WHERE relationship_count > 0;
CREATE INDEX IF NOT EXISTS idx_normalized_events_temporal_confidence ON public.normalized_events(temporal_confidence) WHERE temporal_confidence IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_normalized_events_causal_weight ON public.normalized_events(causal_weight) WHERE causal_weight > 0;
CREATE INDEX IF NOT EXISTS idx_normalized_events_pattern_used ON public.normalized_events(pattern_used_for_prediction) WHERE pattern_used_for_prediction IS NOT NULL;

-- GIN indexes for JSONB columns
CREATE INDEX IF NOT EXISTS idx_normalized_events_relationship_evidence_gin ON public.normalized_events USING GIN (relationship_evidence);
CREATE INDEX IF NOT EXISTS idx_normalized_events_semantic_links_gin ON public.normalized_events USING GIN (semantic_links);
CREATE INDEX IF NOT EXISTS idx_normalized_events_temporal_patterns_gin ON public.normalized_events USING GIN (temporal_patterns);
CREATE INDEX IF NOT EXISTS idx_normalized_events_causal_links_gin ON public.normalized_events USING GIN (causal_links);

-- Add comments for documentation
COMMENT ON COLUMN public.normalized_events.relationship_evidence IS 'Evidence and reasoning for detected relationships';
COMMENT ON COLUMN public.normalized_events.relationship_count IS 'Number of relationships this event participates in';
COMMENT ON COLUMN public.normalized_events.semantic_links IS 'Array of semantic relationship links';
COMMENT ON COLUMN public.normalized_events.temporal_patterns IS 'Array of detected temporal patterns';
COMMENT ON COLUMN public.normalized_events.temporal_cycle_metadata IS 'Metadata about temporal cycles and seasonality';
COMMENT ON COLUMN public.normalized_events.temporal_confidence IS 'Confidence in temporal pattern detection';
COMMENT ON COLUMN public.normalized_events.causal_weight IS 'Weight of causal influence (0-1)';
COMMENT ON COLUMN public.normalized_events.causal_links IS 'Array of causal relationship links';
COMMENT ON COLUMN public.normalized_events.causal_reasoning IS 'Reasoning behind causal analysis';
COMMENT ON COLUMN public.normalized_events.pattern_used_for_prediction IS 'Name of pattern used for prediction';
COMMENT ON COLUMN public.normalized_events.prediction_confidence IS 'Confidence in prediction based on pattern';

-- Log migration completion
DO $$
BEGIN
    RAISE NOTICE 'Migration 20251118000001: Added semantic/temporal/causal metadata columns to normalized_events';
    RAISE NOTICE 'New columns: relationship_evidence, temporal_patterns, causal_weight, pattern_used_for_prediction';
    RAISE NOTICE 'This enables full connection of analytics engines to normalized events';
END $$;
