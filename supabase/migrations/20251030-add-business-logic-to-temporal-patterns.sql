-- Migration: Add business_logic column to temporal_patterns table
-- Date: 2025-10-30
-- Problem: temporal_pattern_learner.py tries to insert business_logic field but column doesn't exist
-- Solution: Add business_logic column to store business logic pattern descriptions

ALTER TABLE public.temporal_patterns 
ADD COLUMN IF NOT EXISTS business_logic VARCHAR(255);

-- Add comment for documentation
COMMENT ON COLUMN public.temporal_patterns.business_logic IS 'Business logic pattern description (e.g., "Predictable pattern based on X historical occurrences")';

-- Create index for business logic queries
CREATE INDEX IF NOT EXISTS idx_temporal_patterns_business_logic ON public.temporal_patterns(business_logic);
