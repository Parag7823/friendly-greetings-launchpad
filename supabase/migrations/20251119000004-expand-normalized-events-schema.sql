-- Migration: Expand normalized_events schema with enrichment and graph columns
-- Date: 2025-11-19
-- Purpose: Add missing enrichment columns for relationship, temporal, and causal tracking

-- Add relationship tracking columns
ALTER TABLE IF EXISTS public.normalized_events
ADD COLUMN IF NOT EXISTS relationship_evidence JSONB DEFAULT '{}',
ADD COLUMN IF NOT EXISTS relationship_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS semantic_links JSONB DEFAULT '[]';

-- Add temporal pattern columns
ALTER TABLE IF EXISTS public.normalized_events
ADD COLUMN IF NOT EXISTS temporal_patterns JSONB DEFAULT '[]',
ADD COLUMN IF NOT EXISTS temporal_cycle_metadata JSONB DEFAULT '{}',
ADD COLUMN IF NOT EXISTS temporal_confidence DECIMAL(3,2) CHECK (temporal_confidence >= 0 AND temporal_confidence <= 1);

-- Add causal analysis columns
ALTER TABLE IF EXISTS public.normalized_events
ADD COLUMN IF NOT EXISTS causal_weight DECIMAL(5,4) DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS causal_links JSONB DEFAULT '[]',
ADD COLUMN IF NOT EXISTS causal_reasoning JSONB DEFAULT '{}';

-- Add prediction tracking columns
ALTER TABLE IF EXISTS public.normalized_events
ADD COLUMN IF NOT EXISTS pattern_used_for_prediction TEXT,
ADD COLUMN IF NOT EXISTS prediction_confidence DECIMAL(3,2) CHECK (prediction_confidence >= 0 AND prediction_confidence <= 1);

-- Add job_id for unified tracking
ALTER TABLE IF EXISTS public.normalized_events
ADD COLUMN IF NOT EXISTS job_id UUID REFERENCES public.ingestion_jobs(id) ON DELETE CASCADE;

-- Create indexes for new columns
CREATE INDEX IF NOT EXISTS idx_normalized_events_relationship_count ON public.normalized_events(relationship_count) WHERE relationship_count > 0;
CREATE INDEX IF NOT EXISTS idx_normalized_events_temporal_confidence ON public.normalized_events(temporal_confidence) WHERE temporal_confidence IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_normalized_events_causal_weight ON public.normalized_events(causal_weight) WHERE causal_weight > 0;
CREATE INDEX IF NOT EXISTS idx_normalized_events_pattern_used ON public.normalized_events(pattern_used_for_prediction) WHERE pattern_used_for_prediction IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_normalized_events_job_id ON public.normalized_events(job_id) WHERE job_id IS NOT NULL;

-- GIN indexes for new JSONB columns
CREATE INDEX IF NOT EXISTS idx_normalized_events_semantic_links_gin ON public.normalized_events USING GIN (semantic_links);
CREATE INDEX IF NOT EXISTS idx_normalized_events_temporal_patterns_gin ON public.normalized_events USING GIN (temporal_patterns);
CREATE INDEX IF NOT EXISTS idx_normalized_events_causal_links_gin ON public.normalized_events USING GIN (causal_links);
CREATE INDEX IF NOT EXISTS idx_normalized_events_relationship_evidence_gin ON public.normalized_events USING GIN (relationship_evidence);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_normalized_events_user_job ON public.normalized_events(user_id, job_id) WHERE job_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_normalized_events_causal_temporal ON public.normalized_events(causal_weight, temporal_confidence) WHERE causal_weight > 0 AND temporal_confidence IS NOT NULL;

-- Add comments for documentation
COMMENT ON COLUMN public.normalized_events.relationship_evidence IS 'Evidence for detected relationships with other events';
COMMENT ON COLUMN public.normalized_events.relationship_count IS 'Number of relationships this event participates in';
COMMENT ON COLUMN public.normalized_events.semantic_links IS 'Semantic relationships and connections to other events';
COMMENT ON COLUMN public.normalized_events.temporal_patterns IS 'Detected temporal patterns this event participates in';
COMMENT ON COLUMN public.normalized_events.temporal_cycle_metadata IS 'Metadata about temporal cycles (daily, weekly, monthly, etc.)';
COMMENT ON COLUMN public.normalized_events.temporal_confidence IS 'Confidence in temporal pattern detection (0.0-1.0)';
COMMENT ON COLUMN public.normalized_events.causal_weight IS 'Weight in causal relationships (0.0-1.0)';
COMMENT ON COLUMN public.normalized_events.causal_links IS 'Causal relationships this event participates in';
COMMENT ON COLUMN public.normalized_events.causal_reasoning IS 'Reasoning for causal relationship determination';
COMMENT ON COLUMN public.normalized_events.pattern_used_for_prediction IS 'Pattern ID used to predict this event';
COMMENT ON COLUMN public.normalized_events.prediction_confidence IS 'Confidence of prediction that led to this event (0.0-1.0)';
COMMENT ON COLUMN public.normalized_events.job_id IS 'Links to ingestion_jobs for unified tracking and deletion';
