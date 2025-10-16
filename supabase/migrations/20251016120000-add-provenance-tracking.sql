-- Migration: Add Complete Provenance Tracking (Row Hash + Lineage Path)
-- Date: 2025-10-16
-- Purpose: Enable tamper detection and full transformation chain visibility

-- ============================================================================
-- PART 1: Add Provenance Columns to raw_events
-- ============================================================================

-- Add row_hash for tamper detection (SHA256 of original row data)
ALTER TABLE public.raw_events 
ADD COLUMN IF NOT EXISTS row_hash TEXT;

-- Add lineage_path for full transformation chain tracking
ALTER TABLE public.raw_events 
ADD COLUMN IF NOT EXISTS lineage_path JSONB DEFAULT '[]'::jsonb;

-- Add created_by for accountability (user or system agent)
ALTER TABLE public.raw_events 
ADD COLUMN IF NOT EXISTS created_by TEXT;

-- Add modified_at and modified_by for audit trail
ALTER TABLE public.raw_events 
ADD COLUMN IF NOT EXISTS modified_at TIMESTAMP WITH TIME ZONE;

ALTER TABLE public.raw_events 
ADD COLUMN IF NOT EXISTS modified_by TEXT;

-- ============================================================================
-- PART 2: Create Indexes for Provenance Queries
-- ============================================================================

-- Index for tamper detection queries
CREATE INDEX IF NOT EXISTS idx_raw_events_row_hash 
ON public.raw_events(row_hash) 
WHERE row_hash IS NOT NULL;

-- GIN index for lineage path queries
CREATE INDEX IF NOT EXISTS idx_raw_events_lineage_path 
ON public.raw_events USING GIN(lineage_path);

-- Index for audit trail queries
CREATE INDEX IF NOT EXISTS idx_raw_events_created_by 
ON public.raw_events(user_id, created_by);

-- ============================================================================
-- PART 3: Add Comments for Documentation
-- ============================================================================

COMMENT ON COLUMN public.raw_events.row_hash IS 
'SHA256 hash of original row data for tamper detection. Format: sha256(source_filename||row_index||original_payload)';

COMMENT ON COLUMN public.raw_events.lineage_path IS 
'Full transformation chain from raw data to final event. Array of transformation steps with timestamps, operations, and confidence scores.
Example: [
  {"step": "ingestion", "timestamp": "2025-10-16T10:00:00Z", "source": "file_upload", "operation": "raw_extract"},
  {"step": "classification", "timestamp": "2025-10-16T10:00:05Z", "operation": "ai_classify", "confidence": 0.95},
  {"step": "enrichment", "timestamp": "2025-10-16T10:00:10Z", "operation": "currency_normalize", "from": "INR", "to": "USD"},
  {"step": "entity_resolution", "timestamp": "2025-10-16T10:00:15Z", "operation": "entity_match", "entity_id": "uuid", "confidence": 0.92}
]';

COMMENT ON COLUMN public.raw_events.created_by IS 
'User ID or system agent that created this event. Examples: "user:uuid", "system:excel_processor", "system:quickbooks_sync"';

-- ============================================================================
-- PART 4: Helper Function - Calculate Row Hash
-- ============================================================================

CREATE OR REPLACE FUNCTION calculate_row_hash(
    p_source_filename TEXT,
    p_row_index INTEGER,
    p_payload JSONB
)
RETURNS TEXT
LANGUAGE plpgsql
IMMUTABLE
AS $$
DECLARE
    v_hash_input TEXT;
BEGIN
    -- Concatenate source identifiers with canonical payload representation
    v_hash_input := COALESCE(p_source_filename, '') || '||' || 
                    COALESCE(p_row_index::TEXT, '') || '||' || 
                    COALESCE(p_payload::TEXT, '{}');
    
    -- Return SHA256 hash
    RETURN encode(digest(v_hash_input, 'sha256'), 'hex');
END;
$$;

COMMENT ON FUNCTION calculate_row_hash IS 
'Calculates SHA256 hash of row data for tamper detection. Combines source filename, row index, and original payload.';

-- ============================================================================
-- PART 5: Helper Function - Build Lineage Path
-- ============================================================================

CREATE OR REPLACE FUNCTION build_lineage_step(
    p_step TEXT,
    p_operation TEXT,
    p_metadata JSONB DEFAULT '{}'::jsonb
)
RETURNS JSONB
LANGUAGE plpgsql
IMMUTABLE
AS $$
BEGIN
    RETURN jsonb_build_object(
        'step', p_step,
        'timestamp', now(),
        'operation', p_operation,
        'metadata', p_metadata
    );
END;
$$;

COMMENT ON FUNCTION build_lineage_step IS 
'Builds a single lineage step with timestamp and metadata. Used to construct lineage_path array.';

-- ============================================================================
-- PART 6: Helper Function - Append Lineage Step
-- ============================================================================

CREATE OR REPLACE FUNCTION append_lineage_step(
    p_existing_path JSONB,
    p_step TEXT,
    p_operation TEXT,
    p_metadata JSONB DEFAULT '{}'::jsonb
)
RETURNS JSONB
LANGUAGE plpgsql
IMMUTABLE
AS $$
DECLARE
    v_new_step JSONB;
BEGIN
    -- Build new step
    v_new_step := build_lineage_step(p_step, p_operation, p_metadata);
    
    -- Append to existing path
    IF p_existing_path IS NULL OR jsonb_array_length(p_existing_path) = 0 THEN
        RETURN jsonb_build_array(v_new_step);
    ELSE
        RETURN p_existing_path || v_new_step;
    END IF;
END;
$$;

COMMENT ON FUNCTION append_lineage_step IS 
'Appends a new transformation step to an existing lineage path. Maintains chronological order.';

-- ============================================================================
-- PART 7: Helper Function - Get Full Provenance
-- ============================================================================

CREATE OR REPLACE FUNCTION get_event_provenance(
    p_user_id UUID,
    p_event_id UUID
)
RETURNS TABLE(
    event_id UUID,
    row_hash TEXT,
    lineage_path JSONB,
    source_info JSONB,
    transformation_summary JSONB,
    entity_links JSONB,
    audit_trail JSONB
)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        re.id as event_id,
        re.row_hash,
        re.lineage_path,
        
        -- Source information
        jsonb_build_object(
            'source_filename', re.source_filename,
            'source_platform', re.source_platform,
            'row_index', re.row_index,
            'sheet_name', re.sheet_name,
            'file_id', re.file_id,
            'job_id', re.job_id,
            'transaction_id', re.transaction_id,
            'ingest_ts', re.ingest_ts,
            'original_payload', re.payload
        ) as source_info,
        
        -- Transformation summary
        jsonb_build_object(
            'classification', jsonb_build_object(
                'kind', re.kind,
                'category', re.category,
                'subcategory', re.subcategory,
                'confidence', re.confidence_score
            ),
            'enrichment', jsonb_build_object(
                'vendor_original', re.payload->>'vendor',
                'vendor_standardized', re.vendor_standard,
                'amount_original', re.amount_original,
                'amount_usd', re.amount_usd,
                'currency', re.currency,
                'platform_ids', re.platform_ids
            ),
            'status', re.status,
            'processed_at', re.processed_at
        ) as transformation_summary,
        
        -- Entity links
        re.entities as entity_links,
        
        -- Audit trail
        jsonb_build_object(
            'created_at', re.created_at,
            'created_by', re.created_by,
            'modified_at', re.modified_at,
            'modified_by', re.modified_by,
            'uploader', re.uploader
        ) as audit_trail
        
    FROM public.raw_events re
    WHERE re.user_id = p_user_id
    AND re.id = p_event_id;
END;
$$;

COMMENT ON FUNCTION get_event_provenance IS 
'Returns complete provenance information for a single event, including source, transformations, entities, and audit trail. Powers "Ask Why" functionality.';

-- ============================================================================
-- PART 8: Helper Function - Verify Row Integrity
-- ============================================================================

CREATE OR REPLACE FUNCTION verify_row_integrity(
    p_user_id UUID,
    p_event_id UUID,
    p_expected_hash TEXT
)
RETURNS TABLE(
    is_valid BOOLEAN,
    stored_hash TEXT,
    expected_hash TEXT,
    message TEXT
)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    v_stored_hash TEXT;
    v_recalculated_hash TEXT;
    v_source_filename TEXT;
    v_row_index INTEGER;
    v_payload JSONB;
BEGIN
    -- Get event data
    SELECT re.row_hash, re.source_filename, re.row_index, re.payload
    INTO v_stored_hash, v_source_filename, v_row_index, v_payload
    FROM public.raw_events re
    WHERE re.user_id = p_user_id AND re.id = p_event_id;
    
    -- Recalculate hash
    v_recalculated_hash := calculate_row_hash(v_source_filename, v_row_index, v_payload);
    
    -- Compare hashes
    IF v_stored_hash = v_recalculated_hash AND v_stored_hash = p_expected_hash THEN
        RETURN QUERY SELECT TRUE, v_stored_hash, p_expected_hash, 'Row integrity verified - no tampering detected'::TEXT;
    ELSIF v_stored_hash = v_recalculated_hash THEN
        RETURN QUERY SELECT FALSE, v_stored_hash, p_expected_hash, 'Hash mismatch - expected hash does not match stored hash'::TEXT;
    ELSE
        RETURN QUERY SELECT FALSE, v_stored_hash, p_expected_hash, 'CRITICAL: Row data has been modified - tampering detected'::TEXT;
    END IF;
END;
$$;

COMMENT ON FUNCTION verify_row_integrity IS 
'Verifies row integrity by comparing stored hash with recalculated hash. Detects tampering or data corruption.';

-- ============================================================================
-- PART 9: Helper Function - Get Lineage Summary
-- ============================================================================

CREATE OR REPLACE FUNCTION get_lineage_summary(
    p_user_id UUID,
    p_event_id UUID
)
RETURNS TABLE(
    step_number INTEGER,
    step_name TEXT,
    operation TEXT,
    step_timestamp TIMESTAMP WITH TIME ZONE,
    metadata JSONB
)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        (row_number() OVER ())::INTEGER as step_number,
        (step_data->>'step')::TEXT as step_name,
        (step_data->>'operation')::TEXT as operation,
        (step_data->>'timestamp')::TIMESTAMP WITH TIME ZONE as step_timestamp,
        (step_data->'metadata')::JSONB as metadata
    FROM public.raw_events re,
         jsonb_array_elements(re.lineage_path) as step_data
    WHERE re.user_id = p_user_id
    AND re.id = p_event_id
    ORDER BY step_number;
END;
$$;

COMMENT ON FUNCTION get_lineage_summary IS 
'Returns lineage path as a table for easy querying. Shows transformation chain step by step.';

-- ============================================================================
-- PART 10: RLS Policies for Provenance Functions
-- ============================================================================

-- Grant execute permissions to authenticated users
GRANT EXECUTE ON FUNCTION calculate_row_hash TO authenticated;
GRANT EXECUTE ON FUNCTION build_lineage_step TO authenticated;
GRANT EXECUTE ON FUNCTION append_lineage_step TO authenticated;
GRANT EXECUTE ON FUNCTION get_event_provenance TO authenticated;
GRANT EXECUTE ON FUNCTION verify_row_integrity TO authenticated;
GRANT EXECUTE ON FUNCTION get_lineage_summary TO authenticated;

-- ============================================================================
-- PART 11: Trigger - Auto-update modified_at
-- ============================================================================

CREATE OR REPLACE FUNCTION update_modified_timestamp()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.modified_at := now();
    RETURN NEW;
END;
$$;

CREATE TRIGGER trigger_raw_events_modified_at
    BEFORE UPDATE ON public.raw_events
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_timestamp();

COMMENT ON TRIGGER trigger_raw_events_modified_at ON public.raw_events IS 
'Automatically updates modified_at timestamp when row is updated. Part of audit trail.';

-- ============================================================================
-- PART 12: Add Provenance to Other Tables
-- ============================================================================

-- Add row_hash to raw_records for file-level integrity
ALTER TABLE public.raw_records 
ADD COLUMN IF NOT EXISTS file_hash_verified BOOLEAN DEFAULT FALSE;

ALTER TABLE public.raw_records 
ADD COLUMN IF NOT EXISTS integrity_check_at TIMESTAMP WITH TIME ZONE;

COMMENT ON COLUMN public.raw_records.file_hash_verified IS 
'Whether file hash has been verified against stored hash. Used for tamper detection.';

-- Add lineage to normalized_entities
ALTER TABLE public.normalized_entities 
ADD COLUMN IF NOT EXISTS lineage_path JSONB DEFAULT '[]'::jsonb;

COMMENT ON COLUMN public.normalized_entities.lineage_path IS 
'Lineage path showing how this entity was created and merged from various sources.';

-- ============================================================================
-- END OF MIGRATION
-- ============================================================================
