-- Migration: Add embeddings to temporal_patterns and causal_relationships tables
-- Date: 2025-12-09
-- Purpose: Enable semantic grouping, similarity search, and AI-powered pattern discovery

-- ============================================================================
-- 1. ADD EMBEDDING COLUMNS
-- ============================================================================

-- Add embedding column to temporal_patterns
ALTER TABLE public.temporal_patterns 
ADD COLUMN IF NOT EXISTS pattern_embedding vector(1024);

-- Add embedding column to causal_relationships
ALTER TABLE public.causal_relationships 
ADD COLUMN IF NOT EXISTS causal_embedding vector(1024);

-- ============================================================================
-- 2. CREATE VECTOR INDEXES (ivfflat for fast similarity search)
-- ============================================================================

-- Index for temporal pattern embeddings
CREATE INDEX IF NOT EXISTS idx_temporal_pattern_embedding ON public.temporal_patterns 
USING ivfflat (pattern_embedding vector_cosine_ops)
WITH (lists = 100);

-- Index for causal relationship embeddings
CREATE INDEX IF NOT EXISTS idx_causal_embedding ON public.causal_relationships 
USING ivfflat (causal_embedding vector_cosine_ops)
WITH (lists = 100);

-- ============================================================================
-- 3. SIMILARITY SEARCH FUNCTIONS
-- ============================================================================

-- Function to find similar temporal patterns using vector similarity
CREATE OR REPLACE FUNCTION find_similar_temporal_patterns(
    p_user_id UUID,
    query_embedding vector(1024),
    similarity_threshold FLOAT DEFAULT 0.7,
    max_results INTEGER DEFAULT 10
)
RETURNS TABLE(
    id UUID,
    relationship_type VARCHAR(100),
    pattern_description TEXT,
    confidence_score FLOAT,
    similarity_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        tp.id,
        tp.relationship_type,
        tp.pattern_description,
        tp.confidence_score,
        1 - (tp.pattern_embedding <=> query_embedding) as similarity_score
    FROM public.temporal_patterns tp
    WHERE tp.user_id = p_user_id
        AND tp.pattern_embedding IS NOT NULL
        AND 1 - (tp.pattern_embedding <=> query_embedding) >= similarity_threshold
    ORDER BY tp.pattern_embedding <=> query_embedding
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to find similar causal relationships using vector similarity
CREATE OR REPLACE FUNCTION find_similar_causal_relationships(
    p_user_id UUID,
    query_embedding vector(1024),
    similarity_threshold FLOAT DEFAULT 0.7,
    max_results INTEGER DEFAULT 10
)
RETURNS TABLE(
    id UUID,
    relationship_id UUID,
    causal_score FLOAT,
    causal_direction VARCHAR(50),
    similarity_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        cr.id,
        cr.relationship_id,
        cr.causal_score,
        cr.causal_direction,
        1 - (cr.causal_embedding <=> query_embedding) as similarity_score
    FROM public.causal_relationships cr
    WHERE cr.user_id = p_user_id
        AND cr.causal_embedding IS NOT NULL
        AND 1 - (cr.causal_embedding <=> query_embedding) >= similarity_threshold
    ORDER BY cr.causal_embedding <=> query_embedding
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================================
-- 4. GRANT PERMISSIONS
-- ============================================================================

GRANT EXECUTE ON FUNCTION find_similar_temporal_patterns TO authenticated;
GRANT EXECUTE ON FUNCTION find_similar_causal_relationships TO authenticated;

-- ============================================================================
-- 5. ADD COMMENTS
-- ============================================================================

COMMENT ON COLUMN public.temporal_patterns.pattern_embedding IS 'Vector embedding (1024 dimensions from BGE bge-large-en-v1.5) for semantic similarity search of patterns';
COMMENT ON COLUMN public.causal_relationships.causal_embedding IS 'Vector embedding (1024 dimensions from BGE bge-large-en-v1.5) for semantic similarity search of causal explanations';

COMMENT ON FUNCTION find_similar_temporal_patterns IS 'Find temporal patterns semantically similar to a query embedding using cosine similarity';
COMMENT ON FUNCTION find_similar_causal_relationships IS 'Find causal relationships semantically similar to a query embedding using cosine similarity';
