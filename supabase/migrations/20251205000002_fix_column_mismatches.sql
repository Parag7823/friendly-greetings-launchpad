-- ✅ ISSUE #9 FIX: Standardize column usage across tables
-- Fixes inconsistencies between backend code and database schema

-- FIX 1: ingestion_jobs - Remove unused progress_percentage, use progress
-- Backend uses 'progress' (integer 0-100), not 'progress_percentage'
ALTER TABLE public.ingestion_jobs 
DROP COLUMN IF EXISTS progress_percentage;

-- Ensure progress column has correct type and default
ALTER TABLE public.ingestion_jobs 
ALTER COLUMN progress SET DEFAULT 0,
ALTER COLUMN progress SET NOT NULL;

-- Add index for progress queries
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_progress 
ON public.ingestion_jobs(user_id, status, progress) 
WHERE status IN ('processing', 'queued');

-- FIX 2: raw_events - Standardize date columns
-- Problem: Some code writes source_ts, others write transaction_date
-- Solution: Use transaction_date as primary, make source_ts an alias
-- Add migration to backfill transaction_date from source_ts
UPDATE public.raw_events 
SET transaction_date = source_ts::date 
WHERE transaction_date IS NULL AND source_ts IS NOT NULL;

-- Add trigger to auto-populate transaction_date from source_ts
CREATE OR REPLACE FUNCTION public.sync_transaction_date()
RETURNS TRIGGER AS $$
BEGIN
  IF NEW.transaction_date IS NULL AND NEW.source_ts IS NOT NULL THEN
    NEW.transaction_date := NEW.source_ts::date;
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_sync_transaction_date ON public.raw_events;
CREATE TRIGGER trg_sync_transaction_date
BEFORE INSERT OR UPDATE ON public.raw_events
FOR EACH ROW
EXECUTE FUNCTION public.sync_transaction_date();

-- Add index for date range queries
CREATE INDEX IF NOT EXISTS idx_raw_events_transaction_date 
ON public.raw_events(user_id, transaction_date DESC) 
WHERE transaction_date IS NOT NULL;

-- FIX 3: raw_records - Remove unused duplicate_decision column
-- This column is written to processing_transactions.metadata instead
ALTER TABLE public.raw_records 
DROP COLUMN IF EXISTS duplicate_decision;

-- FIX 4: Add missing columns that code expects
-- Backend writes to these but they might not exist in all environments
-- Ensure processing_transactions has all tracking columns
ALTER TABLE public.processing_transactions 
ADD COLUMN IF NOT EXISTS inserted_ids JSONB DEFAULT '{}',
ADD COLUMN IF NOT EXISTS updated_ids JSONB DEFAULT '{}',
ADD COLUMN IF NOT EXISTS rollback_data JSONB DEFAULT '{}',
ADD COLUMN IF NOT EXISTS start_time TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS end_time TIMESTAMPTZ;

-- Ensure ingestion_jobs has source column for tracking upload vs integration
ALTER TABLE public.ingestion_jobs 
ADD COLUMN IF NOT EXISTS source VARCHAR(50) DEFAULT 'manual';

-- Add check constraint for valid source values
ALTER TABLE public.ingestion_jobs 
DROP CONSTRAINT IF EXISTS chk_ingestion_jobs_source;
ALTER TABLE public.ingestion_jobs 
ADD CONSTRAINT chk_ingestion_jobs_source 
CHECK (source IN ('manual', 'integration', 'connector_gmail', 'connector_quickbooks', 'connector_xero', 'connector_stripe', 'connector_razorpay', 'connector_dropbox', 'connector_google_drive'));

-- FIX 5: Add composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_raw_events_user_job_status 
ON public.raw_events(user_id, job_id, status);

CREATE INDEX IF NOT EXISTS idx_normalized_events_user_date 
ON public.normalized_events(user_id, normalized_at DESC);

-- Add comment for documentation
COMMENT ON COLUMN public.ingestion_jobs.progress IS 'Processing progress: 0-100 (integer, not percentage)';
COMMENT ON COLUMN public.raw_events.transaction_date IS 'Transaction date (auto-synced from source_ts)';
COMMENT ON COLUMN public.processing_transactions.inserted_ids IS 'Track inserted record IDs for rollback capability';
COMMENT ON COLUMN public.processing_transactions.start_time IS 'Transaction start timestamp for monitoring';
