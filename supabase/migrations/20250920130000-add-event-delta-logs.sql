-- Event Delta Logs Table Migration
-- Supports delta merge tracking and audit trail for duplicate detection

-- ============================================================================
-- EVENT DELTA LOGS TABLE
-- ============================================================================

-- Create event_delta_logs table for tracking delta merge operations
CREATE TABLE IF NOT EXISTS public.event_delta_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    
    -- File references
    existing_file_id UUID REFERENCES public.raw_records(id) ON DELETE CASCADE NOT NULL,
    new_file_id UUID REFERENCES public.raw_records(id) ON DELETE CASCADE NOT NULL,
    
    -- Delta merge summary
    delta_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- Expected structure:
    -- {
    --   "merged_events": 50,
    --   "existing_events": 100,
    --   "new_record_id": "uuid",
    --   "existing_file_id": "uuid"
    -- }
    
    -- Event mapping (which new events map to existing events)
    events_included JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- Expected structure: {"new_event_id": "existing_event_id", ...}
    
    -- Metadata
    merge_type TEXT CHECK (merge_type IN ('append', 'intelligent', 'new_only')) DEFAULT 'intelligent',
    rows_added INTEGER DEFAULT 0,
    rows_skipped INTEGER DEFAULT 0,
    confidence_score DECIMAL(3,2),
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    
    -- Constraints
    CHECK (existing_file_id != new_file_id)
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_event_delta_logs_user_id 
ON public.event_delta_logs(user_id);

CREATE INDEX IF NOT EXISTS idx_event_delta_logs_existing_file 
ON public.event_delta_logs(existing_file_id);

CREATE INDEX IF NOT EXISTS idx_event_delta_logs_new_file 
ON public.event_delta_logs(new_file_id);

CREATE INDEX IF NOT EXISTS idx_event_delta_logs_created_at 
ON public.event_delta_logs(created_at DESC);

-- Composite index for user-specific queries
CREATE INDEX IF NOT EXISTS idx_event_delta_logs_user_created 
ON public.event_delta_logs(user_id, created_at DESC);

-- ============================================================================
-- ROW LEVEL SECURITY
-- ============================================================================

ALTER TABLE public.event_delta_logs ENABLE ROW LEVEL SECURITY;

-- Users can only see their own delta logs
CREATE POLICY "event_delta_logs_select_policy" ON public.event_delta_logs
    FOR SELECT USING (auth.uid() = user_id OR auth.role() = 'service_role');

-- Service role can insert delta logs
CREATE POLICY "event_delta_logs_insert_policy" ON public.event_delta_logs
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

-- No updates or deletes allowed (audit trail)
-- Only service role can delete for cleanup
CREATE POLICY "event_delta_logs_delete_policy" ON public.event_delta_logs
    FOR DELETE USING (auth.role() = 'service_role');

-- ============================================================================
-- HELPER FUNCTION: Get Delta Merge History
-- ============================================================================

CREATE OR REPLACE FUNCTION get_delta_merge_history(p_user_id UUID, p_file_id UUID)
RETURNS TABLE (
    merge_id UUID,
    merge_date TIMESTAMP WITH TIME ZONE,
    source_file_name TEXT,
    target_file_name TEXT,
    rows_added INTEGER,
    rows_skipped INTEGER,
    merge_type TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        edl.id AS merge_id,
        edl.created_at AS merge_date,
        rr_new.file_name AS source_file_name,
        rr_existing.file_name AS target_file_name,
        edl.rows_added,
        edl.rows_skipped,
        edl.merge_type
    FROM public.event_delta_logs edl
    JOIN public.raw_records rr_existing ON edl.existing_file_id = rr_existing.id
    JOIN public.raw_records rr_new ON edl.new_file_id = rr_new.id
    WHERE edl.user_id = p_user_id
      AND (edl.existing_file_id = p_file_id OR edl.new_file_id = p_file_id)
    ORDER BY edl.created_at DESC;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Grant execute permission
GRANT EXECUTE ON FUNCTION get_delta_merge_history(UUID, UUID) TO authenticated;
GRANT EXECUTE ON FUNCTION get_delta_merge_history(UUID, UUID) TO service_role;

-- ============================================================================
-- COMMENTS FOR DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE public.event_delta_logs IS 'Tracks delta merge operations for duplicate file handling';
COMMENT ON COLUMN public.event_delta_logs.delta_summary IS 'JSON summary of merge operation including counts and metadata';
COMMENT ON COLUMN public.event_delta_logs.events_included IS 'Mapping of new event IDs to existing event IDs for deduplication';
COMMENT ON FUNCTION get_delta_merge_history(UUID, UUID) IS 'Retrieves delta merge history for a specific file';
