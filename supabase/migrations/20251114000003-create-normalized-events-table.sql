-- Migration: Create normalized_events table for financial row normalization
-- Date: 2025-11-14
-- Purpose: Create the missing normalized_events table that backend expects (separate from normalized_entities)

-- Create normalized_events table for financial row-level canonicalization
CREATE TABLE public.normalized_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    
    -- Source event reference
    raw_event_id UUID REFERENCES public.raw_events(id) ON DELETE CASCADE,
    
    -- Normalized data (what backend writes)
    normalized_payload JSONB NOT NULL DEFAULT '{}',
    resolved_entities JSONB DEFAULT '{}',
    final_platform JSONB DEFAULT '{}',
    confidence_scores JSONB DEFAULT '{}',
    
    -- Duplicate detection and grouping
    duplicate_group_id UUID,
    duplicate_hash TEXT,
    
    -- Document classification
    document_type TEXT,
    document_subtype TEXT,
    
    -- Delta-merge metadata
    merge_strategy TEXT CHECK (merge_strategy IN ('replace', 'merge', 'keep_both', 'skip')),
    previous_versions JSONB DEFAULT '[]',
    version_number INTEGER DEFAULT 1,
    
    -- Processing metadata
    normalization_method TEXT DEFAULT 'ai_enhanced',
    normalization_confidence DECIMAL(3,2) CHECK (normalization_confidence >= 0 AND normalization_confidence <= 1),
    requires_review BOOLEAN DEFAULT FALSE,
    review_reason TEXT,
    
    -- Platform and classification
    platform_label TEXT,
    semantic_confidence DECIMAL(3,2) CHECK (semantic_confidence >= 0 AND semantic_confidence <= 1),
    
    -- Linking and relationships
    transaction_id UUID REFERENCES public.processing_transactions(id) ON DELETE CASCADE,
    
    -- Timestamps
    normalized_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_normalized_events_user_id ON public.normalized_events(user_id);
CREATE INDEX idx_normalized_events_raw_event_id ON public.normalized_events(raw_event_id);
CREATE INDEX idx_normalized_events_duplicate_group_id ON public.normalized_events(duplicate_group_id) WHERE duplicate_group_id IS NOT NULL;
CREATE INDEX idx_normalized_events_duplicate_hash ON public.normalized_events(duplicate_hash) WHERE duplicate_hash IS NOT NULL;
CREATE INDEX idx_normalized_events_document_type ON public.normalized_events(document_type) WHERE document_type IS NOT NULL;
CREATE INDEX idx_normalized_events_platform_label ON public.normalized_events(platform_label) WHERE platform_label IS NOT NULL;
CREATE INDEX idx_normalized_events_transaction_id ON public.normalized_events(transaction_id) WHERE transaction_id IS NOT NULL;
CREATE INDEX idx_normalized_events_normalized_at ON public.normalized_events(normalized_at);

-- GIN indexes for JSONB fields
CREATE INDEX idx_normalized_events_normalized_payload_gin ON public.normalized_events USING GIN (normalized_payload);
CREATE INDEX idx_normalized_events_resolved_entities_gin ON public.normalized_events USING GIN (resolved_entities);
CREATE INDEX idx_normalized_events_final_platform_gin ON public.normalized_events USING GIN (final_platform);
CREATE INDEX idx_normalized_events_confidence_scores_gin ON public.normalized_events USING GIN (confidence_scores);

-- Composite indexes for common queries
CREATE INDEX idx_normalized_events_user_document_type ON public.normalized_events(user_id, document_type) WHERE document_type IS NOT NULL;
CREATE INDEX idx_normalized_events_user_platform ON public.normalized_events(user_id, platform_label) WHERE platform_label IS NOT NULL;
CREATE INDEX idx_normalized_events_duplicate_detection ON public.normalized_events(user_id, duplicate_hash) WHERE duplicate_hash IS NOT NULL;

-- Enable Row Level Security
ALTER TABLE public.normalized_events ENABLE ROW LEVEL SECURITY;

-- RLS Policies
CREATE POLICY "service_role_all_normalized_events" ON public.normalized_events
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "users_own_normalized_events" ON public.normalized_events
    FOR ALL USING (auth.uid() = user_id);

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_normalized_events_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_normalized_events_updated_at
    BEFORE UPDATE ON public.normalized_events
    FOR EACH ROW
    EXECUTE FUNCTION update_normalized_events_updated_at();

-- Function to get normalization statistics
CREATE OR REPLACE FUNCTION get_normalization_stats(user_uuid UUID)
RETURNS TABLE(
    total_normalized BIGINT,
    by_document_type JSONB,
    by_platform JSONB,
    avg_confidence DECIMAL(3,2),
    requires_review_count BIGINT,
    duplicate_groups BIGINT
) AS $$
DECLARE
    doc_type_breakdown JSONB;
    platform_breakdown JSONB;
BEGIN
    -- Get document type breakdown
    SELECT jsonb_object_agg(document_type, count) INTO doc_type_breakdown
    FROM (
        SELECT document_type, COUNT(*) as count
        FROM public.normalized_events
        WHERE user_id = user_uuid AND document_type IS NOT NULL
        GROUP BY document_type
    ) doc_counts;
    
    -- Get platform breakdown
    SELECT jsonb_object_agg(platform_label, count) INTO platform_breakdown
    FROM (
        SELECT platform_label, COUNT(*) as count
        FROM public.normalized_events
        WHERE user_id = user_uuid AND platform_label IS NOT NULL
        GROUP BY platform_label
    ) platform_counts;
    
    RETURN QUERY
    SELECT 
        COUNT(*) as total_normalized,
        COALESCE(doc_type_breakdown, '{}'::jsonb) as by_document_type,
        COALESCE(platform_breakdown, '{}'::jsonb) as by_platform,
        AVG(normalization_confidence) as avg_confidence,
        COUNT(*) FILTER (WHERE requires_review = true) as requires_review_count,
        COUNT(DISTINCT duplicate_group_id) FILTER (WHERE duplicate_group_id IS NOT NULL) as duplicate_groups
    FROM public.normalized_events
    WHERE user_id = user_uuid;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to search normalized events
CREATE OR REPLACE FUNCTION search_normalized_events(
    user_uuid UUID,
    search_text TEXT DEFAULT NULL,
    document_type_filter TEXT DEFAULT NULL,
    platform_filter TEXT DEFAULT NULL,
    min_confidence DECIMAL DEFAULT 0.0
)
RETURNS TABLE(
    id UUID,
    raw_event_id UUID,
    normalized_payload JSONB,
    resolved_entities JSONB,
    document_type TEXT,
    platform_label TEXT,
    normalization_confidence DECIMAL(3,2),
    requires_review BOOLEAN,
    normalized_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ne.id,
        ne.raw_event_id,
        ne.normalized_payload,
        ne.resolved_entities,
        ne.document_type,
        ne.platform_label,
        ne.normalization_confidence,
        ne.requires_review,
        ne.normalized_at
    FROM public.normalized_events ne
    WHERE ne.user_id = user_uuid
    AND (search_text IS NULL OR ne.normalized_payload::text ILIKE '%' || search_text || '%')
    AND (document_type_filter IS NULL OR ne.document_type = document_type_filter)
    AND (platform_filter IS NULL OR ne.platform_label = platform_filter)
    AND (ne.normalization_confidence IS NULL OR ne.normalization_confidence >= min_confidence)
    ORDER BY ne.normalized_at DESC;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON public.normalized_events TO authenticated;

-- Add comments for documentation
COMMENT ON TABLE public.normalized_events IS 'Normalized financial events after AI processing and canonicalization';
COMMENT ON COLUMN public.normalized_events.normalized_payload IS 'Canonicalized event data with standardized fields';
COMMENT ON COLUMN public.normalized_events.resolved_entities IS 'Entities resolved and linked to canonical forms';
COMMENT ON COLUMN public.normalized_events.final_platform IS 'Final platform determination with confidence';
COMMENT ON COLUMN public.normalized_events.confidence_scores IS 'Multi-dimensional confidence scores for normalization';
COMMENT ON COLUMN public.normalized_events.duplicate_group_id IS 'Groups duplicate events together';
COMMENT ON COLUMN public.normalized_events.merge_strategy IS 'Strategy used for handling duplicates';
COMMENT ON COLUMN public.normalized_events.previous_versions IS 'Array of previous versions for delta tracking';

-- Log migration completion
DO $$
BEGIN
    RAISE NOTICE 'Migration 20251114000003: Created normalized_events table';
    RAISE NOTICE 'Table includes all backend-expected fields: normalized_payload, resolved_entities, final_platform, confidence_scores, duplicate_group_id, document_type, merge_strategy, previous_versions';
    RAISE NOTICE 'Created indexes, RLS policies, and utility functions';
    RAISE NOTICE 'This is separate from normalized_entities (entity resolution system)';
END $$;
