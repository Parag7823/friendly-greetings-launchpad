-- Migration: Align relationship prediction schema with backend expectations
-- Date: 2025-11-01 09:00:00

-- 1. Extend predicted_relationships schema for new analytics fields
ALTER TABLE public.predicted_relationships
    ADD COLUMN IF NOT EXISTS pattern_id UUID,
    ADD COLUMN IF NOT EXISTS transaction_id UUID,
    ADD COLUMN IF NOT EXISTS source_entity_id UUID,
    ADD COLUMN IF NOT EXISTS target_entity_id UUID,
    ADD COLUMN IF NOT EXISTS predicted_relationship_type VARCHAR(100),
    ADD COLUMN IF NOT EXISTS prediction_basis JSONB DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS predicted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();

-- Allow pattern-based predictions that may not have a concrete event yet
ALTER TABLE public.predicted_relationships
    ALTER COLUMN source_event_id DROP NOT NULL,
    ALTER COLUMN predicted_target_type DROP NOT NULL,
    ALTER COLUMN expected_date DROP NOT NULL,
    ALTER COLUMN expected_date SET DEFAULT NOW();

-- Attach foreign keys for the new references if they are missing
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'predicted_relationships_pattern_fk'
    ) THEN
        ALTER TABLE public.predicted_relationships
            ADD CONSTRAINT predicted_relationships_pattern_fk
            FOREIGN KEY (pattern_id)
            REFERENCES public.relationship_patterns(id)
            ON DELETE SET NULL;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'predicted_relationships_transaction_fk'
    ) THEN
        ALTER TABLE public.predicted_relationships
            ADD CONSTRAINT predicted_relationships_transaction_fk
            FOREIGN KEY (transaction_id)
            REFERENCES public.processing_transactions(id)
            ON DELETE SET NULL;
    END IF;
END
$$;

-- Helpful indexes for the new analytics columns
CREATE INDEX IF NOT EXISTS idx_predicted_relationships_pattern_id
    ON public.predicted_relationships(pattern_id);

CREATE INDEX IF NOT EXISTS idx_predicted_relationships_transaction
    ON public.predicted_relationships(transaction_id);

-- Document the new fields
COMMENT ON COLUMN public.predicted_relationships.pattern_id IS 'Relationship pattern that produced this prediction.';
COMMENT ON COLUMN public.predicted_relationships.transaction_id IS 'Processing transaction associated with this prediction.';
COMMENT ON COLUMN public.predicted_relationships.source_entity_id IS 'Resolved entity assumed to originate the future relationship.';
COMMENT ON COLUMN public.predicted_relationships.target_entity_id IS 'Resolved entity expected to participate in the predicted relationship.';
COMMENT ON COLUMN public.predicted_relationships.predicted_relationship_type IS 'Relationship archetype predicted by analytics.';
COMMENT ON COLUMN public.predicted_relationships.prediction_basis IS 'Supporting metadata used to generate the prediction.';
COMMENT ON COLUMN public.predicted_relationships.predicted_at IS 'Timestamp when the prediction was recorded.';

-- 2. Add computed_at to metrics for downstream analytics freshness
ALTER TABLE public.metrics
    ADD COLUMN IF NOT EXISTS computed_at TIMESTAMPTZ DEFAULT NOW();

COMMENT ON COLUMN public.metrics.computed_at IS 'Timestamp when the metric was computed by the processing pipeline.';
