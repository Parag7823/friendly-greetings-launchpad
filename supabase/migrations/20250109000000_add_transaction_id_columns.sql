-- Migration: Add transaction_id column to all tables for complete rollback capability
-- Issue: Transaction manager can only rollback 4 tables, but file processing writes to 9+ tables
-- Solution: Add transaction_id to all tables involved in file processing operations

-- Add transaction_id to tables that don't have it yet
ALTER TABLE IF EXISTS public.raw_records 
ADD COLUMN IF NOT EXISTS transaction_id UUID REFERENCES public.processing_transactions(id) ON DELETE CASCADE;

ALTER TABLE IF EXISTS public.ingestion_jobs 
ADD COLUMN IF NOT EXISTS transaction_id UUID REFERENCES public.processing_transactions(id) ON DELETE CASCADE;

ALTER TABLE IF EXISTS public.metrics 
ADD COLUMN IF NOT EXISTS transaction_id UUID REFERENCES public.processing_transactions(id) ON DELETE CASCADE;

ALTER TABLE IF EXISTS public.platform_patterns 
ADD COLUMN IF NOT EXISTS transaction_id UUID REFERENCES public.processing_transactions(id) ON DELETE CASCADE;

ALTER TABLE IF EXISTS public.discovered_platforms 
ADD COLUMN IF NOT EXISTS transaction_id UUID REFERENCES public.processing_transactions(id) ON DELETE CASCADE;

ALTER TABLE IF EXISTS public.debug_logs 
ADD COLUMN IF NOT EXISTS transaction_id UUID REFERENCES public.processing_transactions(id) ON DELETE CASCADE;

ALTER TABLE IF EXISTS public.field_mappings 
ADD COLUMN IF NOT EXISTS transaction_id UUID REFERENCES public.processing_transactions(id) ON DELETE CASCADE;

ALTER TABLE IF EXISTS public.detection_log 
ADD COLUMN IF NOT EXISTS transaction_id UUID REFERENCES public.processing_transactions(id) ON DELETE CASCADE;

ALTER TABLE IF EXISTS public.resolution_log 
ADD COLUMN IF NOT EXISTS transaction_id UUID REFERENCES public.processing_transactions(id) ON DELETE CASCADE;

-- Create indexes for efficient transaction-based queries
CREATE INDEX IF NOT EXISTS idx_raw_records_transaction_id ON public.raw_records(transaction_id) WHERE transaction_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_transaction_id ON public.ingestion_jobs(transaction_id) WHERE transaction_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_metrics_transaction_id ON public.metrics(transaction_id) WHERE transaction_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_platform_patterns_transaction_id ON public.platform_patterns(transaction_id) WHERE transaction_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_discovered_platforms_transaction_id ON public.discovered_platforms(transaction_id) WHERE transaction_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_debug_logs_transaction_id ON public.debug_logs(transaction_id) WHERE transaction_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_field_mappings_transaction_id ON public.field_mappings(transaction_id) WHERE transaction_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_detection_log_transaction_id ON public.detection_log(transaction_id) WHERE transaction_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_resolution_log_transaction_id ON public.resolution_log(transaction_id) WHERE transaction_id IS NOT NULL;

-- Add comments explaining the purpose
COMMENT ON COLUMN public.raw_records.transaction_id IS 'Links to processing_transactions for atomic rollback capability';
COMMENT ON COLUMN public.ingestion_jobs.transaction_id IS 'Links to processing_transactions for atomic rollback capability';
COMMENT ON COLUMN public.metrics.transaction_id IS 'Links to processing_transactions for atomic rollback capability';
COMMENT ON COLUMN public.platform_patterns.transaction_id IS 'Links to processing_transactions for atomic rollback capability';
COMMENT ON COLUMN public.discovered_platforms.transaction_id IS 'Links to processing_transactions for atomic rollback capability';
COMMENT ON COLUMN public.debug_logs.transaction_id IS 'Links to processing_transactions for atomic rollback capability';
COMMENT ON COLUMN public.field_mappings.transaction_id IS 'Links to processing_transactions for atomic rollback capability';
COMMENT ON COLUMN public.detection_log.transaction_id IS 'Links to processing_transactions for atomic rollback capability';
COMMENT ON COLUMN public.resolution_log.transaction_id IS 'Links to processing_transactions for atomic rollback capability';
