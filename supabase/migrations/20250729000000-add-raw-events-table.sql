-- Create raw_events table for storing individual line items from uploaded files
-- This enables proper streaming and processing of financial data at the row level

CREATE TABLE public.raw_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    file_id UUID REFERENCES public.raw_records(id) ON DELETE CASCADE,
    job_id UUID REFERENCES public.ingestion_jobs(id) ON DELETE CASCADE,
    
    -- Event metadata
    provider TEXT NOT NULL, -- 'excel-upload', 'csv-upload', etc.
    kind TEXT NOT NULL, -- 'payroll_row', 'revenue_row', 'expense_row', etc.
    source_platform TEXT, -- 'gusto', 'razorpay', 'quickbooks', etc.
    
    -- Row data and context
    payload JSONB NOT NULL, -- The actual row data
    row_index INTEGER NOT NULL, -- Position in the original file
    sheet_name TEXT, -- Which sheet this row came from
    
    -- File context
    source_filename TEXT NOT NULL,
    uploader UUID REFERENCES auth.users(id),
    
    -- Processing metadata
    ingest_ts TIMESTAMP WITH TIME ZONE DEFAULT now(),
    processed_at TIMESTAMP WITH TIME ZONE,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'processed', 'failed', 'skipped')),
    error_message TEXT,
    
    -- Confidence and classification
    confidence_score DECIMAL(3,2) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    classification_metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Indexes for performance
CREATE INDEX idx_raw_events_user_id ON public.raw_events(user_id);
CREATE INDEX idx_raw_events_file_id ON public.raw_events(file_id);
CREATE INDEX idx_raw_events_job_id ON public.raw_events(job_id);
CREATE INDEX idx_raw_events_provider ON public.raw_events(provider);
CREATE INDEX idx_raw_events_kind ON public.raw_events(kind);
CREATE INDEX idx_raw_events_source_platform ON public.raw_events(source_platform);
CREATE INDEX idx_raw_events_status ON public.raw_events(status);
CREATE INDEX idx_raw_events_created_at ON public.raw_events(created_at);
CREATE INDEX idx_raw_events_ingest_ts ON public.raw_events(ingest_ts);

-- Composite indexes for common queries
CREATE INDEX idx_raw_events_user_status ON public.raw_events(user_id, status);
CREATE INDEX idx_raw_events_file_status ON public.raw_events(file_id, status);
CREATE INDEX idx_raw_events_platform_kind ON public.raw_events(source_platform, kind);

-- Enable Row Level Security
ALTER TABLE public.raw_events ENABLE ROW LEVEL SECURITY;

-- RLS Policies for raw_events
CREATE POLICY "Users can view their own events" ON public.raw_events
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own events" ON public.raw_events
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own events" ON public.raw_events
    FOR UPDATE USING (auth.uid() = user_id);

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_raw_events_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_raw_events_updated_at
    BEFORE UPDATE ON public.raw_events
    FOR EACH ROW
    EXECUTE FUNCTION update_raw_events_updated_at();

-- Add a function to get event statistics
CREATE OR REPLACE FUNCTION get_raw_events_stats(user_uuid UUID)
RETURNS TABLE(
    total_events BIGINT,
    processed_events BIGINT,
    failed_events BIGINT,
    pending_events BIGINT,
    unique_files BIGINT,
    unique_platforms TEXT[]
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*) as total_events,
        COUNT(*) FILTER (WHERE status = 'processed') as processed_events,
        COUNT(*) FILTER (WHERE status = 'failed') as failed_events,
        COUNT(*) FILTER (WHERE status = 'pending') as pending_events,
        COUNT(DISTINCT file_id) as unique_files,
        ARRAY_AGG(DISTINCT source_platform) FILTER (WHERE source_platform IS NOT NULL) as unique_platforms
    FROM public.raw_events
    WHERE user_id = user_uuid;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER; 