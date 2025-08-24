-- Add processing_transactions table for transaction management
-- This enables proper rollback and cleanup of partial data during file processing

CREATE TABLE public.processing_transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    
    -- Transaction metadata
    status TEXT NOT NULL CHECK (status IN ('active', 'committed', 'rolled_back', 'failed')),
    operation_type TEXT NOT NULL, -- 'file_processing', 'batch_processing', etc.
    
    -- Timestamps
    started_at TIMESTAMP WITH TIME ZONE NOT NULL,
    committed_at TIMESTAMP WITH TIME ZONE,
    rolled_back_at TIMESTAMP WITH TIME ZONE,
    failed_at TIMESTAMP WITH TIME ZONE,
    
    -- Error details for failed/rolled back transactions
    error_details TEXT,
    
    -- Processing metadata
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Indexes for performance
CREATE INDEX idx_processing_transactions_user_id ON public.processing_transactions(user_id);
CREATE INDEX idx_processing_transactions_status ON public.processing_transactions(status);
CREATE INDEX idx_processing_transactions_started_at ON public.processing_transactions(started_at);
CREATE INDEX idx_processing_transactions_operation_type ON public.processing_transactions(operation_type);

-- Composite indexes for common queries
CREATE INDEX idx_processing_transactions_user_status ON public.processing_transactions(user_id, status);
CREATE INDEX idx_processing_transactions_type_status ON public.processing_transactions(operation_type, status);

-- Enable Row Level Security
ALTER TABLE public.processing_transactions ENABLE ROW LEVEL SECURITY;

-- RLS Policies for processing_transactions
CREATE POLICY "_service_access_processing_transactions" ON public.processing_transactions
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "_user_access_processing_transactions" ON public.processing_transactions
    FOR ALL USING (auth.uid() = user_id);

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_processing_transactions_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_processing_transactions_updated_at
    BEFORE UPDATE ON public.processing_transactions
    FOR EACH ROW
    EXECUTE FUNCTION update_processing_transactions_updated_at();

-- Add transaction_id column to existing tables for transaction linking
ALTER TABLE public.raw_records ADD COLUMN IF NOT EXISTS transaction_id UUID REFERENCES public.processing_transactions(id);
ALTER TABLE public.raw_events ADD COLUMN IF NOT EXISTS transaction_id UUID REFERENCES public.processing_transactions(id);
ALTER TABLE public.ingestion_jobs ADD COLUMN IF NOT EXISTS transaction_id UUID REFERENCES public.processing_transactions(id);
ALTER TABLE public.normalized_entities ADD COLUMN IF NOT EXISTS transaction_id UUID REFERENCES public.processing_transactions(id);
ALTER TABLE public.platform_patterns ADD COLUMN IF NOT EXISTS transaction_id UUID REFERENCES public.processing_transactions(id);
ALTER TABLE public.relationship_instances ADD COLUMN IF NOT EXISTS transaction_id UUID REFERENCES public.processing_transactions(id);
ALTER TABLE public.discovered_platforms ADD COLUMN IF NOT EXISTS transaction_id UUID REFERENCES public.processing_transactions(id);
ALTER TABLE public.metrics ADD COLUMN IF NOT EXISTS transaction_id UUID REFERENCES public.processing_transactions(id);

-- Indexes for transaction_id columns
CREATE INDEX IF NOT EXISTS idx_raw_records_transaction_id ON public.raw_records(transaction_id);
CREATE INDEX IF NOT EXISTS idx_raw_events_transaction_id ON public.raw_events(transaction_id);
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_transaction_id ON public.ingestion_jobs(transaction_id);
CREATE INDEX IF NOT EXISTS idx_normalized_entities_transaction_id ON public.normalized_entities(transaction_id);
CREATE INDEX IF NOT EXISTS idx_platform_patterns_transaction_id ON public.platform_patterns(transaction_id);
CREATE INDEX IF NOT EXISTS idx_relationship_instances_transaction_id ON public.relationship_instances(transaction_id);
CREATE INDEX IF NOT EXISTS idx_discovered_platforms_transaction_id ON public.discovered_platforms(transaction_id);
CREATE INDEX IF NOT EXISTS idx_metrics_transaction_id ON public.metrics(transaction_id);

-- Function to get transaction statistics
CREATE OR REPLACE FUNCTION get_transaction_stats(user_uuid UUID)
RETURNS TABLE(
    total_transactions BIGINT,
    active_transactions BIGINT,
    committed_transactions BIGINT,
    rolled_back_transactions BIGINT,
    failed_transactions BIGINT,
    avg_processing_time_seconds NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*) as total_transactions,
        COUNT(*) FILTER (WHERE status = 'active') as active_transactions,
        COUNT(*) FILTER (WHERE status = 'committed') as committed_transactions,
        COUNT(*) FILTER (WHERE status = 'rolled_back') as rolled_back_transactions,
        COUNT(*) FILTER (WHERE status = 'failed') as failed_transactions,
        AVG(EXTRACT(EPOCH FROM (COALESCE(committed_at, rolled_back_at, failed_at) - started_at))) as avg_processing_time_seconds
    FROM public.processing_transactions
    WHERE user_id = user_uuid;
END;
$$ LANGUAGE plpgsql;

-- Function to cleanup old transactions
CREATE OR REPLACE FUNCTION cleanup_old_transactions(days_to_keep INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM public.processing_transactions 
    WHERE started_at < (now() - INTERVAL '1 day' * days_to_keep)
    AND status IN ('committed', 'rolled_back', 'failed');
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;
