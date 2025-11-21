-- Migration: Add Prophet forecast columns to temporal_patterns
-- Date: 2025-11-21
-- Purpose: Store Prophet forecast data directly in the temporal_patterns table to avoid a separate table.

-- Add forecast columns to temporal_patterns table
ALTER TABLE public.temporal_patterns 
ADD COLUMN IF NOT EXISTS forecast_data JSONB DEFAULT '[]',
ADD COLUMN IF NOT EXISTS forecast_generated_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS forecast_expires_at TIMESTAMP WITH TIME ZONE;

-- Add index for expiration checks
CREATE INDEX IF NOT EXISTS idx_temporal_patterns_forecast_expires ON public.temporal_patterns(forecast_expires_at);

-- Comment on columns
COMMENT ON COLUMN public.temporal_patterns.forecast_data IS 'Prophet forecast results (predicted values, confidence intervals)';
COMMENT ON COLUMN public.temporal_patterns.forecast_generated_at IS 'Timestamp when the forecast was last generated';
COMMENT ON COLUMN public.temporal_patterns.forecast_expires_at IS 'Timestamp when the forecast should be regenerated';
