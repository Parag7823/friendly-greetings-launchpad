-- Migration: Fix embedding dimensions to match BGE model (1024 instead of 1536)
-- Date: 2025-10-30
-- Problem: BGE embeddings produce 1024 dimensions, but column was defined as 1536 (OpenAI standard)
-- Solution: Change vector dimension to 1024 to match BGE embeddings

-- Step 1: Drop the old index that depends on the vector column
DROP INDEX IF EXISTS idx_relationship_embedding;

-- Step 2: Drop the old vector column
ALTER TABLE public.relationship_instances 
DROP COLUMN IF EXISTS relationship_embedding;

-- Step 3: Create new vector column with correct dimensions (1024 for BGE)
ALTER TABLE public.relationship_instances 
ADD COLUMN relationship_embedding vector(1024);

-- Step 4: Recreate the index with correct dimensions
CREATE INDEX IF NOT EXISTS idx_relationship_embedding ON public.relationship_instances 
USING ivfflat (relationship_embedding vector_cosine_ops)
WITH (lists = 100);

-- Step 5: Update the search function to use 1024 dimensions
CREATE OR REPLACE FUNCTION search_similar_relationships(
    query_embedding vector(1024),
    similarity_threshold FLOAT DEFAULT 0.7,
    max_results INTEGER DEFAULT 10,
    user_id_param UUID DEFAULT NULL
)
RETURNS TABLE(
    id UUID,
    source_event_id UUID,
    target_event_id UUID,
    relationship_type VARCHAR(100),
    semantic_description TEXT,
    temporal_causality VARCHAR(50),
    business_logic VARCHAR(100),
    confidence_score FLOAT,
    similarity_score FLOAT,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ri.id,
        ri.source_event_id,
        ri.target_event_id,
        ri.relationship_type,
        ri.semantic_description,
        ri.temporal_causality,
        ri.business_logic,
        ri.confidence_score,
        1 - (ri.relationship_embedding <=> query_embedding) as similarity_score,
        ri.created_at
    FROM public.relationship_instances ri
    WHERE ri.relationship_embedding IS NOT NULL
        AND (user_id_param IS NULL OR ri.user_id = user_id_param)
        AND 1 - (ri.relationship_embedding <=> query_embedding) >= similarity_threshold
    ORDER BY ri.relationship_embedding <=> query_embedding
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Step 6: Update documentation comment
COMMENT ON COLUMN public.relationship_instances.relationship_embedding IS 'Vector embedding for semantic similarity search (1024 dimensions from BGE bge-large-en-v1.5)';
