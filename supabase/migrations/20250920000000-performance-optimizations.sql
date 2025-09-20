-- Performance Optimizations Migration
-- Adds advanced indexes, partitioning, and query optimization features
-- Created: 2025-01-20

-- ============================================================================
-- ADVANCED INDEXING STRATEGY
-- ============================================================================

-- Enhanced composite indexes for common query patterns
-- Note: Removed CONCURRENTLY as it cannot run inside transactions
CREATE INDEX IF NOT EXISTS idx_raw_events_user_date_optimized 
ON public.raw_events(user_id, ingest_ts DESC) 
WHERE status = 'processed';

CREATE INDEX IF NOT EXISTS idx_raw_events_platform_kind_optimized 
ON public.raw_events(source_platform, kind, ingest_ts DESC) 
WHERE status = 'processed';

CREATE INDEX IF NOT EXISTS idx_raw_events_user_platform_date 
ON public.raw_events(user_id, source_platform, ingest_ts DESC) 
WHERE status = 'processed';

-- Job processing optimization indexes
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status_priority 
ON public.ingestion_jobs(status, priority, created_at) 
WHERE status IN ('pending', 'processing');

CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_user_status_date 
ON public.ingestion_jobs(user_id, status, created_at DESC);

-- Add priority column to ingestion_jobs if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'ingestion_jobs' AND column_name = 'priority') THEN
        ALTER TABLE public.ingestion_jobs 
        ADD COLUMN priority TEXT DEFAULT 'normal' 
        CHECK (priority IN ('high', 'normal', 'low'));
    END IF;
END $$;

-- Add file_size column for priority calculation
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'ingestion_jobs' AND column_name = 'file_size') THEN
        ALTER TABLE public.ingestion_jobs 
        ADD COLUMN file_size BIGINT DEFAULT 0;
    END IF;
END $$;

-- Add retry tracking columns
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'ingestion_jobs' AND column_name = 'retry_count') THEN
        ALTER TABLE public.ingestion_jobs 
        ADD COLUMN retry_count INTEGER DEFAULT 0;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'ingestion_jobs' AND column_name = 'max_retries') THEN
        ALTER TABLE public.ingestion_jobs 
        ADD COLUMN max_retries INTEGER DEFAULT 3;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'ingestion_jobs' AND column_name = 'last_retry_at') THEN
        ALTER TABLE public.ingestion_jobs 
        ADD COLUMN last_retry_at TIMESTAMP WITH TIME ZONE;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'ingestion_jobs' AND column_name = 'next_retry_at') THEN
        ALTER TABLE public.ingestion_jobs 
        ADD COLUMN next_retry_at TIMESTAMP WITH TIME ZONE;
    END IF;
END $$;

-- Performance monitoring indexes
CREATE INDEX IF NOT EXISTS idx_processing_transactions_job_status 
ON public.processing_transactions(job_id, status, started_at DESC);

-- ============================================================================
-- QUERY OPTIMIZATION FUNCTIONS
-- ============================================================================

-- Function to automatically set job priority based on file size
CREATE OR REPLACE FUNCTION set_job_priority()
RETURNS TRIGGER AS $$
BEGIN
    -- Set priority based on file size
    IF NEW.file_size IS NOT NULL THEN
        IF NEW.file_size < 10485760 THEN -- 10MB
            NEW.priority = 'high';
        ELSIF NEW.file_size < 104857600 THEN -- 100MB
            NEW.priority = 'normal';
        ELSE
            NEW.priority = 'low';
        END IF;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for automatic priority setting
DROP TRIGGER IF EXISTS trigger_set_job_priority ON public.ingestion_jobs;
CREATE TRIGGER trigger_set_job_priority
    BEFORE INSERT OR UPDATE ON public.ingestion_jobs
    FOR EACH ROW
    EXECUTE FUNCTION set_job_priority();

-- ============================================================================
-- ADVANCED ANALYTICS FUNCTIONS
-- ============================================================================

-- Enhanced statistics function with performance metrics
CREATE OR REPLACE FUNCTION get_enhanced_job_stats(user_uuid UUID, days_back INTEGER DEFAULT 30)
RETURNS TABLE(
    total_jobs BIGINT,
    completed_jobs BIGINT,
    failed_jobs BIGINT,
    pending_jobs BIGINT,
    processing_jobs BIGINT,
    avg_processing_time_minutes NUMERIC,
    total_files_processed BIGINT,
    total_events_processed BIGINT,
    success_rate NUMERIC,
    retry_rate NUMERIC,
    priority_distribution JSONB
) AS $$
BEGIN
    RETURN QUERY
    WITH job_stats AS (
        SELECT 
            j.*,
            EXTRACT(EPOCH FROM (j.completed_at - j.created_at))/60 as processing_minutes,
            COALESCE(re.event_count, 0) as events_processed
        FROM public.ingestion_jobs j
        LEFT JOIN (
            SELECT job_id, COUNT(*) as event_count
            FROM public.raw_events
            GROUP BY job_id
        ) re ON j.id = re.job_id
        WHERE j.user_id = user_uuid 
        AND j.created_at >= CURRENT_DATE - (days_back || ' days')::INTERVAL
    )
    SELECT 
        COUNT(*) as total_jobs,
        COUNT(*) FILTER (WHERE status = 'completed') as completed_jobs,
        COUNT(*) FILTER (WHERE status = 'failed') as failed_jobs,
        COUNT(*) FILTER (WHERE status = 'pending') as pending_jobs,
        COUNT(*) FILTER (WHERE status = 'processing') as processing_jobs,
        ROUND(AVG(processing_minutes) FILTER (WHERE status = 'completed'), 2) as avg_processing_time_minutes,
        COUNT(*) FILTER (WHERE status = 'completed') as total_files_processed,
        COALESCE(SUM(events_processed), 0) as total_events_processed,
        ROUND(
            (COUNT(*) FILTER (WHERE status = 'completed')::NUMERIC / NULLIF(COUNT(*), 0)) * 100, 
            2
        ) as success_rate,
        ROUND(
            (COUNT(*) FILTER (WHERE retry_count > 0)::NUMERIC / NULLIF(COUNT(*), 0)) * 100, 
            2
        ) as retry_rate,
        jsonb_build_object(
            'high', COUNT(*) FILTER (WHERE priority = 'high'),
            'normal', COUNT(*) FILTER (WHERE priority = 'normal'),
            'low', COUNT(*) FILTER (WHERE priority = 'low')
        ) as priority_distribution
    FROM job_stats;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get next job for processing (with priority queue)
CREATE OR REPLACE FUNCTION get_next_job_for_processing()
RETURNS TABLE(
    job_id UUID,
    user_id UUID,
    filename TEXT,
    priority TEXT,
    file_size BIGINT,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        j.id as job_id,
        j.user_id,
        j.filename,
        j.priority,
        j.file_size,
        j.created_at
    FROM public.ingestion_jobs j
    WHERE j.status = 'pending'
    AND (j.retry_count < j.max_retries OR j.retry_count IS NULL)
    ORDER BY 
        CASE j.priority 
            WHEN 'high' THEN 1
            WHEN 'normal' THEN 2
            WHEN 'low' THEN 3
            ELSE 4
        END,
        j.created_at ASC
    LIMIT 1
    FOR UPDATE SKIP LOCKED;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================================
-- PERFORMANCE MONITORING
-- ============================================================================

-- Create performance monitoring table
CREATE TABLE IF NOT EXISTS public.performance_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    metric_name TEXT NOT NULL,
    metric_value NUMERIC NOT NULL,
    metric_unit TEXT,
    job_id UUID REFERENCES public.ingestion_jobs(id),
    user_id UUID REFERENCES auth.users(id),
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    metadata JSONB DEFAULT '{}'
);

-- Index for performance metrics
CREATE INDEX IF NOT EXISTS idx_performance_metrics_name_date 
ON public.performance_metrics(metric_name, recorded_at DESC);

CREATE INDEX IF NOT EXISTS idx_performance_metrics_job 
ON public.performance_metrics(job_id, recorded_at DESC);

-- Function to record performance metrics
CREATE OR REPLACE FUNCTION record_performance_metric(
    p_metric_name TEXT,
    p_metric_value NUMERIC,
    p_metric_unit TEXT DEFAULT NULL,
    p_job_id UUID DEFAULT NULL,
    p_user_id UUID DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'
)
RETURNS UUID AS $$
DECLARE
    metric_id UUID;
BEGIN
    INSERT INTO public.performance_metrics (
        metric_name, metric_value, metric_unit, job_id, user_id, metadata
    ) VALUES (
        p_metric_name, p_metric_value, p_metric_unit, p_job_id, p_user_id, p_metadata
    ) RETURNING id INTO metric_id;
    
    RETURN metric_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================================
-- COMMENTS AND DOCUMENTATION
-- ============================================================================

COMMENT ON INDEX idx_raw_events_user_date_optimized IS 'Optimized index for user-specific date-range queries on processed events';
COMMENT ON INDEX idx_raw_events_platform_kind_optimized IS 'Optimized index for platform and transaction type analysis';
COMMENT ON INDEX idx_ingestion_jobs_status_priority IS 'Priority queue index for job processing optimization';
COMMENT ON FUNCTION get_enhanced_job_stats(UUID, INTEGER) IS 'Comprehensive job statistics with performance metrics';
COMMENT ON FUNCTION get_next_job_for_processing() IS 'Priority-based job queue with retry logic and locking';
COMMENT ON TABLE public.performance_metrics IS 'System performance monitoring and metrics collection';

-- Grant necessary permissions
GRANT SELECT, INSERT, UPDATE ON public.performance_metrics TO authenticated;
GRANT USAGE ON SCHEMA public TO authenticated;
