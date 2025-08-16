-- Comprehensive Duplicate Detection System Migration
-- Phase 1: Basic duplicate detection with hash-based checking
-- Phase 2: Near-duplicate detection with content similarity
-- Phase 3: Intelligent version selection and recommendations

-- ============================================================================
-- PHASE 1: BASIC DUPLICATE DETECTION
-- ============================================================================

-- Add file_hash column to raw_records if it doesn't exist (it's currently in JSONB)
-- Extract file_hash from content JSONB to dedicated column for better indexing
ALTER TABLE public.raw_records 
ADD COLUMN IF NOT EXISTS file_hash TEXT;

-- Create index on file_hash for fast duplicate detection
CREATE INDEX IF NOT EXISTS idx_raw_records_file_hash 
ON public.raw_records(file_hash) WHERE file_hash IS NOT NULL;

-- Create composite index for user-specific duplicate detection
CREATE INDEX IF NOT EXISTS idx_raw_records_user_hash 
ON public.raw_records(user_id, file_hash) WHERE file_hash IS NOT NULL;

-- Add unique constraint to prevent exact duplicates (optional - can be enforced at app level)
-- ALTER TABLE public.raw_records 
-- ADD CONSTRAINT unique_user_file_hash UNIQUE (user_id, file_hash);

-- ============================================================================
-- PHASE 2: FILE VERSIONING AND NEAR-DUPLICATE DETECTION
-- ============================================================================

-- Create file_versions table to track file relationships and versions
CREATE TABLE IF NOT EXISTS public.file_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    
    -- Version group identification
    version_group_id UUID NOT NULL, -- Groups related file versions together
    version_number INTEGER NOT NULL DEFAULT 1,
    is_active_version BOOLEAN DEFAULT true,
    
    -- File identification
    file_id UUID REFERENCES public.raw_records(id) ON DELETE CASCADE,
    file_hash TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    normalized_filename TEXT NOT NULL, -- Filename without version indicators
    
    -- Content analysis
    total_rows INTEGER DEFAULT 0,
    total_columns INTEGER DEFAULT 0,
    column_names TEXT[] DEFAULT '{}',
    content_fingerprint TEXT, -- Hash of column structure + sample data
    
    -- Version detection metadata
    detected_version_pattern TEXT, -- 'v1', 'v2', 'final', 'draft', etc.
    filename_similarity_score DECIMAL(3,2), -- Similarity to other files in group
    content_similarity_score DECIMAL(3,2), -- Content similarity to other versions
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    
    -- Constraints
    UNIQUE(version_group_id, version_number),
    UNIQUE(file_id) -- Each file can only be in one version group
);

-- Create file_similarity_analysis table for storing comparison results
CREATE TABLE IF NOT EXISTS public.file_similarity_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    
    -- Files being compared
    source_file_id UUID REFERENCES public.raw_records(id) ON DELETE CASCADE,
    target_file_id UUID REFERENCES public.raw_records(id) ON DELETE CASCADE,
    
    -- Similarity metrics
    filename_similarity DECIMAL(3,2) NOT NULL,
    content_similarity DECIMAL(3,2) NOT NULL,
    structure_similarity DECIMAL(3,2) NOT NULL, -- Column names/types similarity
    row_overlap_percentage DECIMAL(3,2) NOT NULL, -- % of rows that are similar
    
    -- Analysis details
    similar_rows_count INTEGER DEFAULT 0,
    total_rows_compared INTEGER DEFAULT 0,
    matching_columns TEXT[] DEFAULT '{}',
    differing_columns TEXT[] DEFAULT '{}',
    
    -- Relationship determination
    relationship_type TEXT CHECK (relationship_type IN ('identical', 'version', 'similar', 'unrelated')),
    confidence_score DECIMAL(3,2) NOT NULL,
    analysis_reason TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    
    -- Constraints
    UNIQUE(source_file_id, target_file_id),
    CHECK (source_file_id != target_file_id)
);

-- ============================================================================
-- PHASE 3: INTELLIGENT VERSION RECOMMENDATIONS
-- ============================================================================

-- Create version_recommendations table for storing AI-generated recommendations
CREATE TABLE IF NOT EXISTS public.version_recommendations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    
    -- Version group being analyzed
    version_group_id UUID NOT NULL,
    recommended_version_id UUID REFERENCES public.file_versions(id) ON DELETE CASCADE,
    
    -- Recommendation analysis
    recommendation_type TEXT CHECK (recommendation_type IN ('most_complete', 'most_recent', 'best_quality', 'manual_review')),
    confidence_score DECIMAL(3,2) NOT NULL,
    reasoning TEXT NOT NULL,
    
    -- Comparison metrics
    completeness_scores JSONB, -- {file_id: score} for each version
    recency_scores JSONB, -- {file_id: score} for each version  
    quality_scores JSONB, -- {file_id: score} for each version
    
    -- User interaction
    user_accepted BOOLEAN DEFAULT NULL, -- NULL = pending, true = accepted, false = rejected
    user_feedback TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    
    -- Constraints
    UNIQUE(version_group_id) -- One recommendation per version group
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- File versions indexes
CREATE INDEX IF NOT EXISTS idx_file_versions_user_id ON public.file_versions(user_id);
CREATE INDEX IF NOT EXISTS idx_file_versions_group_id ON public.file_versions(version_group_id);
CREATE INDEX IF NOT EXISTS idx_file_versions_active ON public.file_versions(is_active_version) WHERE is_active_version = true;
CREATE INDEX IF NOT EXISTS idx_file_versions_filename ON public.file_versions(normalized_filename);

-- File similarity indexes
CREATE INDEX IF NOT EXISTS idx_file_similarity_user_id ON public.file_similarity_analysis(user_id);
CREATE INDEX IF NOT EXISTS idx_file_similarity_source ON public.file_similarity_analysis(source_file_id);
CREATE INDEX IF NOT EXISTS idx_file_similarity_target ON public.file_similarity_analysis(target_file_id);
CREATE INDEX IF NOT EXISTS idx_file_similarity_relationship ON public.file_similarity_analysis(relationship_type);

-- Version recommendations indexes
CREATE INDEX IF NOT EXISTS idx_version_recommendations_user_id ON public.version_recommendations(user_id);
CREATE INDEX IF NOT EXISTS idx_version_recommendations_group ON public.version_recommendations(version_group_id);
CREATE INDEX IF NOT EXISTS idx_version_recommendations_pending ON public.version_recommendations(user_accepted) WHERE user_accepted IS NULL;

-- ============================================================================
-- ROW LEVEL SECURITY POLICIES
-- ============================================================================

-- Enable RLS on new tables
ALTER TABLE public.file_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.file_similarity_analysis ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.version_recommendations ENABLE ROW LEVEL SECURITY;

-- File versions policies
CREATE POLICY "file_versions_select_policy" ON public.file_versions
    FOR SELECT USING (auth.uid() = user_id OR auth.role() = 'service_role');

CREATE POLICY "file_versions_insert_policy" ON public.file_versions
    FOR INSERT WITH CHECK (auth.uid() = user_id OR auth.role() = 'service_role');

CREATE POLICY "file_versions_update_policy" ON public.file_versions
    FOR UPDATE USING (auth.uid() = user_id OR auth.role() = 'service_role');

-- File similarity policies
CREATE POLICY "file_similarity_select_policy" ON public.file_similarity_analysis
    FOR SELECT USING (auth.uid() = user_id OR auth.role() = 'service_role');

CREATE POLICY "file_similarity_insert_policy" ON public.file_similarity_analysis
    FOR INSERT WITH CHECK (auth.uid() = user_id OR auth.role() = 'service_role');

-- Version recommendations policies
CREATE POLICY "version_recommendations_select_policy" ON public.version_recommendations
    FOR SELECT USING (auth.uid() = user_id OR auth.role() = 'service_role');

CREATE POLICY "version_recommendations_insert_policy" ON public.version_recommendations
    FOR INSERT WITH CHECK (auth.uid() = user_id OR auth.role() = 'service_role');

CREATE POLICY "version_recommendations_update_policy" ON public.version_recommendations
    FOR UPDATE USING (auth.uid() = user_id OR auth.role() = 'service_role');

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to normalize filename for version detection
CREATE OR REPLACE FUNCTION normalize_filename(filename TEXT)
RETURNS TEXT AS $$
BEGIN
    -- Remove common version patterns and normalize
    RETURN regexp_replace(
        regexp_replace(
            regexp_replace(
                lower(trim(filename)),
                '_v\d+|_version\d+|_final|_draft|_copy|\(\d+\)', '', 'g'
            ),
            '\s+', '_', 'g'
        ),
        '_+', '_', 'g'
    );
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to extract version pattern from filename
CREATE OR REPLACE FUNCTION extract_version_pattern(filename TEXT)
RETURNS TEXT AS $$
DECLARE
    version_match TEXT;
BEGIN
    -- Look for common version patterns
    SELECT INTO version_match
        CASE 
            WHEN filename ~* '_v(\d+)' THEN regexp_replace(filename, '.*(_v\d+).*', '\1', 'i')
            WHEN filename ~* '_version(\d+)' THEN regexp_replace(filename, '.*(_version\d+).*', '\1', 'i')
            WHEN filename ~* '_final' THEN 'final'
            WHEN filename ~* '_draft' THEN 'draft'
            WHEN filename ~* '_copy' THEN 'copy'
            WHEN filename ~* '\((\d+)\)' THEN regexp_replace(filename, '.*\((\d+)\).*', '(\1)')
            ELSE 'v1'
        END;
    
    RETURN COALESCE(version_match, 'v1');
END;
$$ LANGUAGE plpgsql IMMUTABLE;
