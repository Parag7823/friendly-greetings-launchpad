-- Migration: Add missing fields to raw_events table for backend compatibility
-- Date: 2025-11-14
-- Purpose: Fix schema drift - add fields that backend expects but are missing from schema

-- Add missing AI and processing fields to raw_events
ALTER TABLE public.raw_events 
ADD COLUMN IF NOT EXISTS ai_confidence DECIMAL(3,2) CHECK (ai_confidence >= 0 AND ai_confidence <= 1),
ADD COLUMN IF NOT EXISTS ai_reasoning TEXT,
ADD COLUMN IF NOT EXISTS source_ts TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS relationship_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS last_relationship_check TIMESTAMP WITH TIME ZONE;

-- Create indexes for new fields
CREATE INDEX IF NOT EXISTS idx_raw_events_ai_confidence ON public.raw_events(ai_confidence) WHERE ai_confidence IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_raw_events_source_ts ON public.raw_events(source_ts) WHERE source_ts IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_raw_events_relationship_count ON public.raw_events(relationship_count) WHERE relationship_count > 0;

-- Add comments to explain the columns
COMMENT ON COLUMN public.raw_events.ai_confidence IS 'AI classification confidence score (0.0 to 1.0)';
COMMENT ON COLUMN public.raw_events.ai_reasoning IS 'AI reasoning/explanation for classification decisions';
COMMENT ON COLUMN public.raw_events.source_ts IS 'Original timestamp from the source data/transaction';
COMMENT ON COLUMN public.raw_events.relationship_count IS 'Number of relationships this event participates in';
COMMENT ON COLUMN public.raw_events.last_relationship_check IS 'Last time relationship count was updated';

-- Update existing records to have default values where appropriate
UPDATE public.raw_events
SET relationship_count = 0
WHERE relationship_count IS NULL;

-- Log migration completion
DO $$
BEGIN
    RAISE NOTICE 'Migration 20251114000000: Added missing AI and processing fields to raw_events table';
    RAISE NOTICE 'Added columns: ai_confidence, ai_reasoning, source_ts, relationship_count, last_relationship_check';
    RAISE NOTICE 'Created indexes for performance optimization';
END $$;
