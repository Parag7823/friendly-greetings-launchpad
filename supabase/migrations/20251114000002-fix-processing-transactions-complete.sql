-- Migration: Complete fix for processing_transactions table
-- Date: 2025-11-14
-- Purpose: Add missing backend fields and fix status values for transaction management

-- Add missing fields that backend expects
ALTER TABLE public.processing_transactions 
ADD COLUMN IF NOT EXISTS job_id UUID,
ADD COLUMN IF NOT EXISTS file_id UUID,
ADD COLUMN IF NOT EXISTS rollback_data JSONB DEFAULT '{}',
ADD COLUMN IF NOT EXISTS inserted_ids JSONB DEFAULT '[]',
ADD COLUMN IF NOT EXISTS updated_ids JSONB DEFAULT '[]',
ADD COLUMN IF NOT EXISTS start_time TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS end_time TIMESTAMP WITH TIME ZONE;

-- Add foreign key constraints for job_id and file_id
ALTER TABLE public.processing_transactions 
ADD CONSTRAINT fk_processing_transactions_job_id 
FOREIGN KEY (job_id) REFERENCES public.ingestion_jobs(id) ON DELETE SET NULL;

ALTER TABLE public.processing_transactions 
ADD CONSTRAINT fk_processing_transactions_file_id 
FOREIGN KEY (file_id) REFERENCES public.raw_records(id) ON DELETE SET NULL;

-- Drop existing status constraint and add correct one with backend values
ALTER TABLE public.processing_transactions DROP CONSTRAINT IF EXISTS processing_transactions_status_check;
ALTER TABLE public.processing_transactions 
ADD CONSTRAINT processing_transactions_status_check 
CHECK (status IN (
    'pending', 'active', 'committed', 'rolled_back', 'failed'
));

-- Create indexes for new fields
CREATE INDEX IF NOT EXISTS idx_processing_transactions_job_id ON public.processing_transactions(job_id) WHERE job_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_processing_transactions_file_id ON public.processing_transactions(file_id) WHERE file_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_processing_transactions_start_time ON public.processing_transactions(start_time) WHERE start_time IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_processing_transactions_end_time ON public.processing_transactions(end_time) WHERE end_time IS NOT NULL;

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_processing_transactions_job_status ON public.processing_transactions(job_id, status) WHERE job_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_processing_transactions_file_status ON public.processing_transactions(file_id, status) WHERE file_id IS NOT NULL;

-- Add comments for documentation
COMMENT ON COLUMN public.processing_transactions.job_id IS 'Reference to ingestion_jobs for job-based rollback';
COMMENT ON COLUMN public.processing_transactions.file_id IS 'Reference to raw_records for file-based rollback';
COMMENT ON COLUMN public.processing_transactions.rollback_data IS 'Structured data needed for transaction rollback';
COMMENT ON COLUMN public.processing_transactions.inserted_ids IS 'Array of record IDs inserted in this transaction';
COMMENT ON COLUMN public.processing_transactions.updated_ids IS 'Array of record IDs updated in this transaction';
COMMENT ON COLUMN public.processing_transactions.start_time IS 'Transaction start time (for monitoring/UI)';
COMMENT ON COLUMN public.processing_transactions.end_time IS 'Transaction end time (for monitoring/UI)';

-- Create function to automatically set start_time and end_time based on started_at
CREATE OR REPLACE FUNCTION sync_transaction_times()
RETURNS TRIGGER AS $$
BEGIN
    -- Set start_time when started_at is set
    IF NEW.started_at IS NOT NULL AND (OLD.started_at IS NULL OR NEW.started_at != OLD.started_at) THEN
        NEW.start_time = NEW.started_at;
    END IF;
    
    -- Set end_time when transaction completes
    IF NEW.status IN ('committed', 'rolled_back', 'failed') THEN
        IF NEW.committed_at IS NOT NULL THEN
            NEW.end_time = NEW.committed_at;
        ELSIF NEW.rolled_back_at IS NOT NULL THEN
            NEW.end_time = NEW.rolled_back_at;
        ELSIF NEW.failed_at IS NOT NULL THEN
            NEW.end_time = NEW.failed_at;
        ELSE
            NEW.end_time = NOW();
        END IF;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to automatically sync times
DROP TRIGGER IF EXISTS trigger_sync_transaction_times ON public.processing_transactions;
CREATE TRIGGER trigger_sync_transaction_times
    BEFORE INSERT OR UPDATE ON public.processing_transactions
    FOR EACH ROW
    EXECUTE FUNCTION sync_transaction_times();

-- Update existing records to sync times
UPDATE public.processing_transactions 
SET 
    start_time = started_at,
    end_time = COALESCE(committed_at, rolled_back_at, failed_at)
WHERE start_time IS NULL AND started_at IS NOT NULL;

-- Create function to get transaction rollback data
CREATE OR REPLACE FUNCTION get_transaction_rollback_data(p_transaction_id UUID)
RETURNS TABLE(
    table_name TEXT,
    record_ids UUID[],
    operation_type TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        'raw_events'::TEXT as table_name,
        ARRAY_AGG(id) as record_ids,
        'DELETE'::TEXT as operation_type
    FROM public.raw_events 
    WHERE transaction_id = p_transaction_id
    
    UNION ALL
    
    SELECT 
        'raw_records'::TEXT as table_name,
        ARRAY_AGG(id) as record_ids,
        'DELETE'::TEXT as operation_type
    FROM public.raw_records 
    WHERE transaction_id = p_transaction_id
    
    UNION ALL
    
    SELECT 
        'normalized_entities'::TEXT as table_name,
        ARRAY_AGG(id) as record_ids,
        'DELETE'::TEXT as operation_type
    FROM public.normalized_entities 
    WHERE transaction_id = p_transaction_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Log migration completion
DO $$
BEGIN
    RAISE NOTICE 'Migration 20251114000002: Fixed processing_transactions table schema';
    RAISE NOTICE 'Added fields: job_id, file_id, rollback_data, inserted_ids, updated_ids, start_time, end_time';
    RAISE NOTICE 'Updated status constraint to include: pending (instead of just active)';
    RAISE NOTICE 'Created automatic time sync trigger and rollback data function';
END $$;
