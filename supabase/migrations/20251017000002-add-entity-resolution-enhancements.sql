-- Migration: Entity Resolution Enhancements
-- Date: 2025-10-17
-- Purpose: Add pg_trgm fuzzy matching, phonetic matching, and resolution logging

-- Enable required PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- Trigram-based text similarity
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;  -- Phonetic matching (soundex, metaphone)

-- Add phonetic columns to normalized_entities table
ALTER TABLE public.normalized_entities 
ADD COLUMN IF NOT EXISTS canonical_name_soundex TEXT,
ADD COLUMN IF NOT EXISTS canonical_name_metaphone TEXT,
ADD COLUMN IF NOT EXISTS canonical_name_dmetaphone TEXT;

-- Create function to update phonetic columns
CREATE OR REPLACE FUNCTION update_entity_phonetic_columns()
RETURNS TRIGGER AS $$
BEGIN
    -- Generate phonetic representations
    NEW.canonical_name_soundex = soundex(NEW.canonical_name);
    NEW.canonical_name_metaphone = metaphone(NEW.canonical_name, 8);
    NEW.canonical_name_dmetaphone = dmetaphone(NEW.canonical_name);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to automatically update phonetic columns
DROP TRIGGER IF EXISTS trigger_update_entity_phonetic ON public.normalized_entities;
CREATE TRIGGER trigger_update_entity_phonetic
    BEFORE INSERT OR UPDATE OF canonical_name ON public.normalized_entities
    FOR EACH ROW
    EXECUTE FUNCTION update_entity_phonetic_columns();

-- Backfill phonetic columns for existing entities
UPDATE public.normalized_entities
SET 
    canonical_name_soundex = soundex(canonical_name),
    canonical_name_metaphone = metaphone(canonical_name, 8),
    canonical_name_dmetaphone = dmetaphone(canonical_name)
WHERE canonical_name_soundex IS NULL;

-- Create GIN indexes for efficient trigram-based fuzzy matching
CREATE INDEX IF NOT EXISTS idx_normalized_entities_canonical_name_trgm 
ON public.normalized_entities USING GIN (canonical_name gin_trgm_ops);

-- Note: aliases is a text[] array, cannot use gin_trgm_ops directly
-- Instead, we'll search aliases using array operations in the query
-- If needed, we can create a GIN index for array containment
CREATE INDEX IF NOT EXISTS idx_normalized_entities_aliases_gin 
ON public.normalized_entities USING GIN (aliases);

-- Create indexes for phonetic matching
CREATE INDEX IF NOT EXISTS idx_normalized_entities_soundex 
ON public.normalized_entities(canonical_name_soundex);

CREATE INDEX IF NOT EXISTS idx_normalized_entities_metaphone 
ON public.normalized_entities(canonical_name_metaphone);

CREATE INDEX IF NOT EXISTS idx_normalized_entities_dmetaphone 
ON public.normalized_entities(canonical_name_dmetaphone);

-- Create composite index for user_id + entity_type (common query pattern)
CREATE INDEX IF NOT EXISTS idx_normalized_entities_user_type 
ON public.normalized_entities(user_id, entity_type);

-- Create resolution_log table for learning system
CREATE TABLE IF NOT EXISTS public.resolution_log (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    
    -- Resolution details
    resolution_id TEXT NOT NULL,
    entity_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    platform TEXT,
    
    -- Resolution outcome
    resolved_entity_id UUID REFERENCES public.normalized_entities(id) ON DELETE SET NULL,
    resolved_name TEXT,
    resolution_method TEXT NOT NULL CHECK (resolution_method IN ('exact_match', 'fuzzy_match', 'new_entity', 'user_correction', 'ai_match')),
    confidence NUMERIC(5, 4) CHECK (confidence >= 0 AND confidence <= 1),
    
    -- Similarity scores
    name_similarity NUMERIC(5, 4),
    identifier_similarity NUMERIC(5, 4),
    phonetic_match BOOLEAN DEFAULT FALSE,
    
    -- Source information
    source_file TEXT,
    row_id TEXT,
    identifiers JSONB DEFAULT '{}'::jsonb,
    
    -- User feedback
    user_corrected BOOLEAN DEFAULT FALSE,
    correction_timestamp TIMESTAMP WITH TIME ZONE,
    correct_entity_id UUID REFERENCES public.normalized_entities(id) ON DELETE SET NULL,
    
    -- Performance tracking
    processing_time_ms INTEGER,
    cache_hit BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    resolved_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    
    -- Additional metadata
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Indexes for resolution_log
CREATE INDEX IF NOT EXISTS idx_resolution_log_user_id ON public.resolution_log(user_id);
CREATE INDEX IF NOT EXISTS idx_resolution_log_entity_type ON public.resolution_log(entity_type);
CREATE INDEX IF NOT EXISTS idx_resolution_log_resolved_entity ON public.resolution_log(resolved_entity_id);
CREATE INDEX IF NOT EXISTS idx_resolution_log_resolution_method ON public.resolution_log(resolution_method);
CREATE INDEX IF NOT EXISTS idx_resolution_log_confidence ON public.resolution_log(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_resolution_log_user_corrected ON public.resolution_log(user_corrected) WHERE user_corrected = TRUE;
CREATE INDEX IF NOT EXISTS idx_resolution_log_resolved_at ON public.resolution_log(resolved_at DESC);

-- Composite index for common queries
CREATE INDEX IF NOT EXISTS idx_resolution_log_user_entity_type 
ON public.resolution_log(user_id, entity_type, resolved_at DESC);

-- GIN index for JSONB columns
CREATE INDEX IF NOT EXISTS idx_resolution_log_identifiers ON public.resolution_log USING GIN (identifiers);
CREATE INDEX IF NOT EXISTS idx_resolution_log_metadata ON public.resolution_log USING GIN (metadata);

-- Enable Row Level Security
ALTER TABLE public.resolution_log ENABLE ROW LEVEL SECURITY;

-- RLS Policies for resolution_log
DROP POLICY IF EXISTS "service_role_all_resolution_log" ON public.resolution_log;
CREATE POLICY "service_role_all_resolution_log" ON public.resolution_log
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_resolution_log" ON public.resolution_log;
CREATE POLICY "users_own_resolution_log" ON public.resolution_log
    FOR ALL USING (auth.uid() = user_id);

-- Function to find fuzzy matches using pg_trgm
CREATE OR REPLACE FUNCTION find_fuzzy_entity_matches(
    p_user_id UUID,
    p_entity_name TEXT,
    p_entity_type TEXT,
    p_similarity_threshold NUMERIC DEFAULT 0.7,
    p_max_results INTEGER DEFAULT 10
)
RETURNS TABLE(
    entity_id UUID,
    canonical_name TEXT,
    similarity_score NUMERIC,
    match_type TEXT,
    email TEXT,
    tax_id TEXT,
    bank_account TEXT,
    phone TEXT
) AS $$
BEGIN
    -- Use pg_trgm similarity for efficient fuzzy matching
    RETURN QUERY
    SELECT 
        ne.id AS entity_id,
        ne.canonical_name,
        similarity(ne.canonical_name, p_entity_name)::NUMERIC AS similarity_score,  -- Cast real to numeric
        CASE 
            WHEN ne.canonical_name = p_entity_name THEN 'exact'
            WHEN similarity(ne.canonical_name, p_entity_name) > 0.9 THEN 'very_high'
            WHEN similarity(ne.canonical_name, p_entity_name) > 0.8 THEN 'high'
            ELSE 'moderate'
        END AS match_type,
        ne.email,
        ne.tax_id,
        ne.bank_account,
        ne.phone
    FROM public.normalized_entities ne
    WHERE ne.user_id = p_user_id
      AND ne.entity_type = p_entity_type
      AND similarity(ne.canonical_name, p_entity_name) >= p_similarity_threshold
    ORDER BY similarity_score DESC
    LIMIT p_max_results;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to find phonetic matches
CREATE OR REPLACE FUNCTION find_phonetic_entity_matches(
    p_user_id UUID,
    p_entity_name TEXT,
    p_entity_type TEXT,
    p_max_results INTEGER DEFAULT 10
)
RETURNS TABLE(
    entity_id UUID,
    canonical_name TEXT,
    match_method TEXT,
    email TEXT,
    tax_id TEXT,
    bank_account TEXT,
    phone TEXT
) AS $$
DECLARE
    v_soundex TEXT;
    v_metaphone TEXT;
    v_dmetaphone TEXT;
BEGIN
    -- Generate phonetic representations of search term
    v_soundex := soundex(p_entity_name);
    v_metaphone := metaphone(p_entity_name, 8);
    v_dmetaphone := dmetaphone(p_entity_name);
    
    -- Find entities with matching phonetic representations
    RETURN QUERY
    SELECT 
        ne.id AS entity_id,
        ne.canonical_name,
        CASE 
            WHEN ne.canonical_name_soundex = v_soundex THEN 'soundex'
            WHEN ne.canonical_name_metaphone = v_metaphone THEN 'metaphone'
            WHEN ne.canonical_name_dmetaphone = v_dmetaphone THEN 'dmetaphone'
            ELSE 'unknown'
        END AS match_method,
        ne.email,
        ne.tax_id,
        ne.bank_account,
        ne.phone
    FROM public.normalized_entities ne
    WHERE ne.user_id = p_user_id
      AND ne.entity_type = p_entity_type
      AND (
          ne.canonical_name_soundex = v_soundex
          OR ne.canonical_name_metaphone = v_metaphone
          OR ne.canonical_name_dmetaphone = v_dmetaphone
      )
    LIMIT p_max_results;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to record user correction
CREATE OR REPLACE FUNCTION record_entity_correction(
    p_resolution_log_id UUID,
    p_correct_entity_id UUID
)
RETURNS VOID AS $$
BEGIN
    UPDATE public.resolution_log
    SET 
        user_corrected = TRUE,
        correct_entity_id = p_correct_entity_id,
        correction_timestamp = now()
    WHERE id = p_resolution_log_id;
    
    -- Update entity confidence based on correction
    UPDATE public.normalized_entities
    SET 
        confidence_score = LEAST(confidence_score + 0.05, 1.0),
        updated_at = now()
    WHERE id = p_correct_entity_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get resolution statistics
CREATE OR REPLACE FUNCTION get_resolution_statistics(
    p_user_id UUID,
    p_entity_type TEXT DEFAULT NULL,
    p_days INTEGER DEFAULT 30
)
RETURNS TABLE(
    total_resolutions BIGINT,
    exact_matches BIGINT,
    fuzzy_matches BIGINT,
    new_entities BIGINT,
    user_corrections BIGINT,
    avg_confidence NUMERIC,
    avg_processing_time_ms NUMERIC,
    cache_hit_rate NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::BIGINT AS total_resolutions,
        COUNT(*) FILTER (WHERE resolution_method = 'exact_match')::BIGINT AS exact_matches,
        COUNT(*) FILTER (WHERE resolution_method = 'fuzzy_match')::BIGINT AS fuzzy_matches,
        COUNT(*) FILTER (WHERE resolution_method = 'new_entity')::BIGINT AS new_entities,
        COUNT(*) FILTER (WHERE user_corrected = TRUE)::BIGINT AS user_corrections,
        AVG(confidence) AS avg_confidence,
        AVG(processing_time_ms) AS avg_processing_time_ms,
        (COUNT(*) FILTER (WHERE cache_hit = TRUE)::NUMERIC / NULLIF(COUNT(*), 0)) AS cache_hit_rate
    FROM public.resolution_log
    WHERE user_id = p_user_id
      AND (p_entity_type IS NULL OR entity_type = p_entity_type)
      AND resolved_at >= now() - (p_days || ' days')::INTERVAL;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to analyze resolution patterns for learning
CREATE OR REPLACE FUNCTION analyze_resolution_patterns(
    p_user_id UUID,
    p_entity_type TEXT DEFAULT NULL
)
RETURNS TABLE(
    pattern_type TEXT,
    pattern_value TEXT,
    occurrence_count BIGINT,
    avg_confidence NUMERIC,
    success_rate NUMERIC
) AS $$
BEGIN
    -- Analyze common resolution patterns
    RETURN QUERY
    SELECT 
        'resolution_method' AS pattern_type,
        resolution_method AS pattern_value,
        COUNT(*)::BIGINT AS occurrence_count,
        AVG(confidence) AS avg_confidence,
        (COUNT(*) FILTER (WHERE NOT user_corrected)::NUMERIC / NULLIF(COUNT(*), 0)) AS success_rate
    FROM public.resolution_log
    WHERE user_id = p_user_id
      AND (p_entity_type IS NULL OR entity_type = p_entity_type)
    GROUP BY resolution_method
    
    UNION ALL
    
    -- Analyze phonetic match patterns
    SELECT 
        'phonetic_match' AS pattern_type,
        CASE WHEN phonetic_match THEN 'yes' ELSE 'no' END AS pattern_value,
        COUNT(*)::BIGINT AS occurrence_count,
        AVG(confidence) AS avg_confidence,
        (COUNT(*) FILTER (WHERE NOT user_corrected)::NUMERIC / NULLIF(COUNT(*), 0)) AS success_rate
    FROM public.resolution_log
    WHERE user_id = p_user_id
      AND (p_entity_type IS NULL OR entity_type = p_entity_type)
    GROUP BY phonetic_match
    
    ORDER BY occurrence_count DESC;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Grant necessary permissions
GRANT SELECT, INSERT, UPDATE ON public.resolution_log TO authenticated;
GRANT EXECUTE ON FUNCTION find_fuzzy_entity_matches TO authenticated;
GRANT EXECUTE ON FUNCTION find_phonetic_entity_matches TO authenticated;
GRANT EXECUTE ON FUNCTION record_entity_correction TO authenticated;
GRANT EXECUTE ON FUNCTION get_resolution_statistics TO authenticated;
GRANT EXECUTE ON FUNCTION analyze_resolution_patterns TO authenticated;

-- Function to find cross-document relationships efficiently using database JOINs
CREATE OR REPLACE FUNCTION find_cross_document_relationships(
    p_user_id UUID,
    p_source_document_type TEXT,
    p_target_document_type TEXT,
    p_relationship_type TEXT,
    p_max_results INTEGER DEFAULT 1000,
    p_amount_tolerance NUMERIC DEFAULT 5.0,
    p_date_range_days INTEGER DEFAULT 30
)
RETURNS TABLE(
    source_event_id UUID,
    target_event_id UUID,
    relationship_type TEXT,
    confidence NUMERIC,
    amount_match BOOLEAN,
    date_match BOOLEAN,
    entity_match BOOLEAN,
    metadata JSONB
) AS $$
BEGIN
    -- Efficient database-level JOIN to find relationships
    -- This replaces O(N²) Python loops with optimized SQL
    RETURN QUERY
    SELECT 
        source.id AS source_event_id,
        target.id AS target_event_id,
        p_relationship_type AS relationship_type,
        -- Calculate confidence score based on matches
        (
            CASE WHEN ABS(source.amount_usd - target.amount_usd) <= p_amount_tolerance THEN 0.4 ELSE 0.0 END +
            CASE WHEN ABS(EXTRACT(EPOCH FROM (source.source_ts - target.source_ts)) / 86400) <= p_date_range_days THEN 0.3 ELSE 0.0 END +
            CASE WHEN source.vendor_standard = target.vendor_standard AND source.vendor_standard IS NOT NULL THEN 0.3 ELSE 0.0 END
        ) AS confidence,
        ABS(source.amount_usd - target.amount_usd) <= p_amount_tolerance AS amount_match,
        ABS(EXTRACT(EPOCH FROM (source.source_ts - target.source_ts)) / 86400) <= p_date_range_days AS date_match,
        (source.vendor_standard = target.vendor_standard AND source.vendor_standard IS NOT NULL) AS entity_match,
        jsonb_build_object(
            'source_amount', source.amount_usd,
            'target_amount', target.amount_usd,
            'amount_diff', ABS(source.amount_usd - target.amount_usd),
            'source_date', source.source_ts,
            'target_date', target.source_ts,
            'date_diff_days', ABS(EXTRACT(EPOCH FROM (source.source_ts - target.source_ts)) / 86400),
            'source_vendor', source.vendor_standard,
            'target_vendor', target.vendor_standard
        ) AS metadata
    FROM public.raw_events source
    INNER JOIN public.raw_events target ON (
        source.user_id = target.user_id
        AND source.id != target.id
        AND source.document_type = p_source_document_type
        AND target.document_type = p_target_document_type
        -- Efficient filters to reduce JOIN size
        AND ABS(source.amount_usd - target.amount_usd) <= p_amount_tolerance
        AND ABS(EXTRACT(EPOCH FROM (source.source_ts - target.source_ts)) / 86400) <= p_date_range_days
    )
    WHERE source.user_id = p_user_id
      AND (
          -- At least 2 of 3 criteria must match for a relationship
          (ABS(source.amount_usd - target.amount_usd) <= p_amount_tolerance AND 
           ABS(EXTRACT(EPOCH FROM (source.source_ts - target.source_ts)) / 86400) <= p_date_range_days)
          OR
          (ABS(source.amount_usd - target.amount_usd) <= p_amount_tolerance AND 
           source.vendor_standard = target.vendor_standard AND source.vendor_standard IS NOT NULL)
          OR
          (ABS(EXTRACT(EPOCH FROM (source.source_ts - target.source_ts)) / 86400) <= p_date_range_days AND
           source.vendor_standard = target.vendor_standard AND source.vendor_standard IS NOT NULL)
      )
    ORDER BY confidence DESC
    LIMIT p_max_results;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to find within-document relationships efficiently
CREATE OR REPLACE FUNCTION find_within_document_relationships(
    p_user_id UUID,
    p_file_id UUID,
    p_relationship_type TEXT,
    p_max_results INTEGER DEFAULT 1000
)
RETURNS TABLE(
    source_event_id UUID,
    target_event_id UUID,
    relationship_type TEXT,
    confidence NUMERIC,
    metadata JSONB
) AS $$
BEGIN
    -- Find relationships within the same file using efficient self-JOIN
    RETURN QUERY
    SELECT 
        e1.id AS source_event_id,
        e2.id AS target_event_id,
        p_relationship_type AS relationship_type,
        -- Calculate confidence based on relationship patterns
        CASE 
            WHEN e1.category = 'revenue' AND e2.category = 'expense' THEN 0.7
            WHEN e1.vendor_standard = e2.vendor_standard AND e1.vendor_standard IS NOT NULL THEN 0.8
            WHEN ABS(e1.amount_usd - e2.amount_usd) < 1.0 THEN 0.6
            ELSE 0.5
        END AS confidence,
        jsonb_build_object(
            'source_category', e1.category,
            'target_category', e2.category,
            'source_amount', e1.amount_usd,
            'target_amount', e2.amount_usd,
            'shared_vendor', e1.vendor_standard = e2.vendor_standard,
            'vendor_name', e1.vendor_standard
        ) AS metadata
    FROM public.raw_events e1
    INNER JOIN public.raw_events e2 ON (
        e1.user_id = e2.user_id
        AND e1.file_id = e2.file_id
        AND e1.id != e2.id
        AND (
            -- Different categories (e.g., revenue paired with expense)
            (e1.category != e2.category AND e1.category IS NOT NULL AND e2.category IS NOT NULL)
            OR
            -- Same vendor
            (e1.vendor_standard = e2.vendor_standard AND e1.vendor_standard IS NOT NULL)
            OR
            -- Similar amounts
            (ABS(e1.amount_usd - e2.amount_usd) < 1.0)
        )
    )
    WHERE e1.user_id = p_user_id
      AND e1.file_id = p_file_id
    ORDER BY confidence DESC
    LIMIT p_max_results;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get relationship statistics
CREATE OR REPLACE FUNCTION get_relationship_statistics(
    p_user_id UUID,
    p_days INTEGER DEFAULT 30
)
RETURNS TABLE(
    total_relationships BIGINT,
    cross_file_relationships BIGINT,
    within_file_relationships BIGINT,
    by_type JSONB,
    avg_confidence NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::BIGINT AS total_relationships,
        COUNT(*) FILTER (WHERE source_file_id != target_file_id)::BIGINT AS cross_file_relationships,
        COUNT(*) FILTER (WHERE source_file_id = target_file_id)::BIGINT AS within_file_relationships,
        jsonb_object_agg(relationship_type, type_count) AS by_type,
        AVG(confidence) AS avg_confidence
    FROM (
        SELECT 
            ri.relationship_type,
            ri.confidence,
            se.file_id AS source_file_id,
            te.file_id AS target_file_id,
            COUNT(*) OVER (PARTITION BY ri.relationship_type) AS type_count
        FROM public.relationship_instances ri
        JOIN public.raw_events se ON ri.source_event_id = se.id
        JOIN public.raw_events te ON ri.target_event_id = te.id
        WHERE ri.user_id = p_user_id
          AND ri.created_at >= now() - (p_days || ' days')::INTERVAL
    ) stats
    GROUP BY ();
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Grant permissions
GRANT EXECUTE ON FUNCTION find_cross_document_relationships TO authenticated;
GRANT EXECUTE ON FUNCTION find_within_document_relationships TO authenticated;
GRANT EXECUTE ON FUNCTION get_relationship_statistics TO authenticated;

-- Add comments
COMMENT ON TABLE public.resolution_log IS 'Stores entity resolution history for learning and analytics';
COMMENT ON FUNCTION find_fuzzy_entity_matches IS 'Efficiently finds fuzzy entity matches using pg_trgm trigram similarity';
COMMENT ON FUNCTION find_phonetic_entity_matches IS 'Finds entities with phonetically similar names using soundex/metaphone';
COMMENT ON FUNCTION record_entity_correction IS 'Records user correction for entity resolution';
COMMENT ON FUNCTION get_resolution_statistics IS 'Returns resolution statistics for a user';
COMMENT ON FUNCTION analyze_resolution_patterns IS 'Analyzes resolution patterns for learning and optimization';
COMMENT ON FUNCTION find_cross_document_relationships IS 'Efficiently finds relationships between documents using database JOINs (replaces O(N²) Python loops)';
COMMENT ON FUNCTION find_within_document_relationships IS 'Efficiently finds relationships within a document using self-JOIN';
COMMENT ON FUNCTION get_relationship_statistics IS 'Returns relationship detection statistics for a user';
