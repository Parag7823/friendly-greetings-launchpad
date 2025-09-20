-- Critical Fixes Support Migration
-- Adds tables and functions to support transaction management, error recovery, and atomic operations
-- Created: 2025-09-20

-- ============================================================================
-- PROCESSING LOCKS TABLE FOR ATOMIC OPERATIONS
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.processing_locks (
    id TEXT PRIMARY KEY,
    lock_type TEXT NOT NULL CHECK (lock_type IN ('duplicate_detection', 'transaction', 'file_processing')),
    resource_id TEXT NOT NULL,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    acquired_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'expired', 'released')),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Indexes for processing locks
CREATE INDEX IF NOT EXISTS idx_processing_locks_resource ON public.processing_locks(resource_id, lock_type);
CREATE INDEX IF NOT EXISTS idx_processing_locks_user ON public.processing_locks(user_id);
CREATE INDEX IF NOT EXISTS idx_processing_locks_expires ON public.processing_locks(expires_at);
CREATE INDEX IF NOT EXISTS idx_processing_locks_status ON public.processing_locks(status);

-- Enable RLS for processing locks
ALTER TABLE public.processing_locks ENABLE ROW LEVEL SECURITY;

-- RLS policies for processing locks
CREATE POLICY "Service role can manage all locks" ON public.processing_locks
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Users can view their own locks" ON public.processing_locks
    FOR SELECT USING (auth.uid() = user_id);

-- ============================================================================
-- ERROR LOGS TABLE FOR ERROR RECOVERY
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.error_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    job_id UUID REFERENCES public.ingestion_jobs(id) ON DELETE SET NULL,
    transaction_id UUID REFERENCES public.processing_transactions(id) ON DELETE SET NULL,
    operation_type TEXT NOT NULL,
    error_message TEXT NOT NULL,
    error_details JSONB DEFAULT '{}',
    severity TEXT NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    occurred_at TIMESTAMP WITH TIME ZONE NOT NULL,
    recovery_attempts INTEGER DEFAULT 0,
    recovery_completed BOOLEAN DEFAULT false,
    recovery_success BOOLEAN DEFAULT false,
    recovery_action TEXT,
    recovery_details JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Indexes for error logs
CREATE INDEX IF NOT EXISTS idx_error_logs_user ON public.error_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_error_logs_job ON public.error_logs(job_id);
CREATE INDEX IF NOT EXISTS idx_error_logs_transaction ON public.error_logs(transaction_id);
CREATE INDEX IF NOT EXISTS idx_error_logs_severity ON public.error_logs(severity);
CREATE INDEX IF NOT EXISTS idx_error_logs_occurred_at ON public.error_logs(occurred_at);
CREATE INDEX IF NOT EXISTS idx_error_logs_recovery ON public.error_logs(recovery_completed, recovery_success);

-- Enable RLS for error logs
ALTER TABLE public.error_logs ENABLE ROW LEVEL SECURITY;

-- RLS policies for error logs
CREATE POLICY "Service role can manage all error logs" ON public.error_logs
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Users can view their own error logs" ON public.error_logs
    FOR SELECT USING (auth.uid() = user_id);

-- ============================================================================
-- ENHANCED FUNCTIONS FOR DATA CONSISTENCY
-- ============================================================================

-- Function to find orphaned events
CREATE OR REPLACE FUNCTION find_orphaned_events(p_user_id UUID, p_cutoff_time TIMESTAMP WITH TIME ZONE)
RETURNS TABLE(
    id UUID,
    job_id UUID,
    file_id UUID,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        re.id,
        re.job_id,
        re.file_id,
        re.created_at
    FROM public.raw_events re
    LEFT JOIN public.ingestion_jobs ij ON re.job_id = ij.id
    LEFT JOIN public.raw_records rr ON re.file_id = rr.id
    WHERE re.user_id = p_user_id
    AND re.created_at < p_cutoff_time
    AND (
        ij.id IS NULL OR 
        ij.status IN ('failed', 'cancelled') OR
        rr.id IS NULL
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to find orphaned records
CREATE OR REPLACE FUNCTION find_orphaned_records(p_user_id UUID, p_cutoff_time TIMESTAMP WITH TIME ZONE)
RETURNS TABLE(
    id UUID,
    file_name TEXT,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        rr.id,
        rr.file_name,
        rr.created_at
    FROM public.raw_records rr
    LEFT JOIN public.ingestion_jobs ij ON rr.id = ij.file_id
    WHERE rr.user_id = p_user_id
    AND rr.created_at < p_cutoff_time
    AND (
        ij.id IS NULL OR 
        ij.status IN ('failed', 'cancelled')
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to validate event-record consistency
CREATE OR REPLACE FUNCTION validate_event_record_consistency(p_user_id UUID)
RETURNS TABLE(
    id UUID,
    job_id UUID,
    file_id UUID
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        re.id,
        re.job_id,
        re.file_id
    FROM public.raw_events re
    LEFT JOIN public.raw_records rr ON re.file_id = rr.id
    WHERE re.user_id = p_user_id
    AND rr.id IS NULL;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to validate transaction consistency
CREATE OR REPLACE FUNCTION validate_transaction_consistency(p_user_id UUID)
RETURNS TABLE(
    id UUID,
    status TEXT,
    committed_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        pt.id,
        pt.status,
        pt.committed_at
    FROM public.processing_transactions pt
    LEFT JOIN public.raw_events re ON pt.id = re.transaction_id
    LEFT JOIN public.raw_records rr ON pt.id = rr.transaction_id
    WHERE pt.user_id = p_user_id
    AND pt.status = 'committed'
    AND re.id IS NULL
    AND rr.id IS NULL;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to cleanup expired locks automatically
CREATE OR REPLACE FUNCTION cleanup_expired_locks()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM public.processing_locks 
    WHERE expires_at < now()
    AND status = 'active';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get system health metrics
CREATE OR REPLACE FUNCTION get_system_health_metrics()
RETURNS TABLE(
    metric_name TEXT,
    metric_value NUMERIC,
    metric_unit TEXT,
    recorded_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        'active_jobs'::TEXT as metric_name,
        COUNT(*)::NUMERIC as metric_value,
        'count'::TEXT as metric_unit,
        now() as recorded_at
    FROM public.ingestion_jobs 
    WHERE status IN ('processing', 'queued')
    
    UNION ALL
    
    SELECT 
        'active_locks'::TEXT as metric_name,
        COUNT(*)::NUMERIC as metric_value,
        'count'::TEXT as metric_unit,
        now() as recorded_at
    FROM public.processing_locks 
    WHERE status = 'active' AND expires_at > now()
    
    UNION ALL
    
    SELECT 
        'active_transactions'::TEXT as metric_name,
        COUNT(*)::NUMERIC as metric_value,
        'count'::TEXT as metric_unit,
        now() as recorded_at
    FROM public.processing_transactions 
    WHERE status = 'active'
    
    UNION ALL
    
    SELECT 
        'recent_errors'::TEXT as metric_name,
        COUNT(*)::NUMERIC as metric_value,
        'count'::TEXT as metric_unit,
        now() as recorded_at
    FROM public.error_logs 
    WHERE occurred_at > now() - INTERVAL '1 hour'
    
    UNION ALL
    
    SELECT 
        'critical_errors'::TEXT as metric_name,
        COUNT(*)::NUMERIC as metric_value,
        'count'::TEXT as metric_unit,
        now() as recorded_at
    FROM public.error_logs 
    WHERE severity = 'critical' 
    AND occurred_at > now() - INTERVAL '24 hours';
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================================
-- TRIGGERS FOR AUTOMATIC MAINTENANCE
-- ============================================================================

-- Trigger to update updated_at timestamp on error logs
CREATE OR REPLACE FUNCTION update_error_logs_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_error_logs_updated_at
    BEFORE UPDATE ON public.error_logs
    FOR EACH ROW
    EXECUTE FUNCTION update_error_logs_updated_at();

-- ============================================================================
-- ENHANCED INDEXES FOR PERFORMANCE
-- ============================================================================

-- Additional indexes for better performance with the new systems
CREATE INDEX IF NOT EXISTS idx_raw_events_transaction_user 
ON public.raw_events(transaction_id, user_id) 
WHERE transaction_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_raw_records_transaction_user 
ON public.raw_records(transaction_id, user_id) 
WHERE transaction_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status_user_date 
ON public.ingestion_jobs(status, user_id, created_at DESC);

-- Partial index for active processing transactions
CREATE INDEX IF NOT EXISTS idx_processing_transactions_active 
ON public.processing_transactions(user_id, started_at) 
WHERE status = 'active';

-- ============================================================================
-- GRANT PERMISSIONS
-- ============================================================================

-- Grant necessary permissions for the new tables and functions
GRANT SELECT, INSERT, UPDATE, DELETE ON public.processing_locks TO authenticated;
GRANT SELECT, INSERT, UPDATE ON public.error_logs TO authenticated;

GRANT EXECUTE ON FUNCTION find_orphaned_events(UUID, TIMESTAMP WITH TIME ZONE) TO authenticated;
GRANT EXECUTE ON FUNCTION find_orphaned_records(UUID, TIMESTAMP WITH TIME ZONE) TO authenticated;
GRANT EXECUTE ON FUNCTION validate_event_record_consistency(UUID) TO authenticated;
GRANT EXECUTE ON FUNCTION validate_transaction_consistency(UUID) TO authenticated;
GRANT EXECUTE ON FUNCTION cleanup_expired_locks() TO authenticated;
GRANT EXECUTE ON FUNCTION get_system_health_metrics() TO authenticated;

-- ============================================================================
-- COMMENTS FOR DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE public.processing_locks IS 'Database-level locks for atomic operations and race condition prevention';
COMMENT ON TABLE public.error_logs IS 'Comprehensive error logging and recovery tracking';

COMMENT ON FUNCTION find_orphaned_events(UUID, TIMESTAMP WITH TIME ZONE) IS 'Find events without valid job or record references';
COMMENT ON FUNCTION find_orphaned_records(UUID, TIMESTAMP WITH TIME ZONE) IS 'Find records without valid job references';
COMMENT ON FUNCTION validate_event_record_consistency(UUID) IS 'Validate consistency between events and records';
COMMENT ON FUNCTION validate_transaction_consistency(UUID) IS 'Validate transaction data consistency';
COMMENT ON FUNCTION cleanup_expired_locks() IS 'Automatically cleanup expired processing locks';
COMMENT ON FUNCTION get_system_health_metrics() IS 'Get real-time system health and performance metrics';
