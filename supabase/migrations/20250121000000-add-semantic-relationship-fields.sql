-- Migration: Add semantic relationship fields to relationship_instances
-- Date: 2025-01-21
-- Purpose: Enhance relationship detection with AI-powered semantic understanding

-- Enable pgvector extension for relationship embeddings (if not already enabled)
CREATE EXTENSION IF NOT EXISTS vector;

-- Add semantic fields to relationship_instances table
ALTER TABLE public.relationship_instances 
ADD COLUMN IF NOT EXISTS semantic_description TEXT,
ADD COLUMN IF NOT EXISTS temporal_causality VARCHAR(50) 
    CHECK (temporal_causality IN ('source_causes_target', 'target_causes_source', 'bidirectional', 'correlation_only')),
ADD COLUMN IF NOT EXISTS business_logic VARCHAR(100)
    CHECK (business_logic IN ('standard_payment_flow', 'revenue_recognition', 'expense_reimbursement', 
                               'payroll_processing', 'tax_withholding', 'asset_depreciation', 
                               'loan_repayment', 'refund_processing', 'recurring_billing', 'unknown')),
ADD COLUMN IF NOT EXISTS relationship_embedding vector(1536),  -- OpenAI text-embedding-3-small dimension
ADD COLUMN IF NOT EXISTS key_factors JSONB DEFAULT '[]',
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();

-- Add metadata column if it doesn't exist (for backward compatibility)
ALTER TABLE public.relationship_instances 
ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}';

-- Create index for semantic similarity search using vector
CREATE INDEX IF NOT EXISTS idx_relationship_embedding ON public.relationship_instances 
USING ivfflat (relationship_embedding vector_cosine_ops)
WITH (lists = 100);

-- Create indexes for semantic fields
CREATE INDEX IF NOT EXISTS idx_relationship_temporal_causality ON public.relationship_instances(temporal_causality);
CREATE INDEX IF NOT EXISTS idx_relationship_business_logic ON public.relationship_instances(business_logic);
CREATE INDEX IF NOT EXISTS idx_relationship_semantic_description ON public.relationship_instances 
USING gin(to_tsvector('english', semantic_description));

-- Create function to search relationships by semantic similarity
CREATE OR REPLACE FUNCTION search_similar_relationships(
    query_embedding vector(1536),
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

-- Create function to get semantic relationship statistics
CREATE OR REPLACE FUNCTION get_semantic_relationship_stats(user_id_param UUID)
RETURNS JSON AS $$
DECLARE
    result JSON;
BEGIN
    SELECT json_build_object(
        'total_relationships', COUNT(*),
        'relationships_with_semantics', COUNT(*) FILTER (WHERE semantic_description IS NOT NULL),
        'semantic_coverage_percent', 
            ROUND(100.0 * COUNT(*) FILTER (WHERE semantic_description IS NOT NULL) / NULLIF(COUNT(*), 0), 2),
        'temporal_causality_distribution', (
            SELECT json_object_agg(temporal_causality, count)
            FROM (
                SELECT temporal_causality, COUNT(*) as count
                FROM public.relationship_instances
                WHERE user_id = user_id_param AND temporal_causality IS NOT NULL
                GROUP BY temporal_causality
            ) t
        ),
        'business_logic_distribution', (
            SELECT json_object_agg(business_logic, count)
            FROM (
                SELECT business_logic, COUNT(*) as count
                FROM public.relationship_instances
                WHERE user_id = user_id_param AND business_logic IS NOT NULL
                GROUP BY business_logic
            ) t
        ),
        'avg_confidence', ROUND(AVG(confidence_score)::numeric, 3),
        'causal_relationships', COUNT(*) FILTER (WHERE temporal_causality IN ('source_causes_target', 'target_causes_source', 'bidirectional')),
        'correlation_only', COUNT(*) FILTER (WHERE temporal_causality = 'correlation_only')
    ) INTO result
    FROM public.relationship_instances
    WHERE user_id = user_id_param;
    
    RETURN COALESCE(result, '{}'::json);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create function to find causal chains (multi-hop causal relationships)
CREATE OR REPLACE FUNCTION find_causal_chain(
    start_event_id UUID,
    max_depth INTEGER DEFAULT 5,
    user_id_param UUID DEFAULT NULL
)
RETURNS TABLE(
    chain_depth INTEGER,
    event_path UUID[],
    relationship_path UUID[],
    total_confidence FLOAT,
    chain_description TEXT
) AS $$
WITH RECURSIVE causal_chain AS (
    -- Base case: direct causal relationships from start event
    SELECT 
        1 as depth,
        ARRAY[ri.source_event_id, ri.target_event_id] as event_path,
        ARRAY[ri.id] as relationship_path,
        ri.confidence_score as total_confidence,
        ri.semantic_description as chain_description
    FROM public.relationship_instances ri
    WHERE ri.source_event_id = start_event_id
        AND ri.temporal_causality IN ('source_causes_target', 'bidirectional')
        AND (user_id_param IS NULL OR ri.user_id = user_id_param)
    
    UNION ALL
    
    -- Recursive case: extend chain with next causal relationship
    SELECT 
        cc.depth + 1,
        cc.event_path || ri.target_event_id,
        cc.relationship_path || ri.id,
        cc.total_confidence * ri.confidence_score,
        cc.chain_description || ' → ' || ri.semantic_description
    FROM causal_chain cc
    JOIN public.relationship_instances ri ON ri.source_event_id = cc.event_path[array_length(cc.event_path, 1)]
    WHERE cc.depth < max_depth
        AND ri.temporal_causality IN ('source_causes_target', 'bidirectional')
        AND NOT (ri.target_event_id = ANY(cc.event_path))  -- Prevent cycles
        AND (user_id_param IS NULL OR ri.user_id = user_id_param)
)
SELECT 
    depth as chain_depth,
    event_path,
    relationship_path,
    total_confidence,
    chain_description
FROM causal_chain
ORDER BY depth, total_confidence DESC;
$$ LANGUAGE sql SECURITY DEFINER;

-- Create trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_relationship_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_relationship_updated_at ON public.relationship_instances;
CREATE TRIGGER trigger_update_relationship_updated_at
    BEFORE UPDATE ON public.relationship_instances
    FOR EACH ROW
    EXECUTE FUNCTION update_relationship_updated_at();

-- Create view for enriched relationships with semantic data
CREATE OR REPLACE VIEW enriched_relationships AS
SELECT 
    ri.id,
    ri.user_id,
    ri.source_event_id,
    ri.target_event_id,
    ri.relationship_type,
    ri.confidence_score,
    ri.detection_method,
    ri.semantic_description,
    ri.temporal_causality,
    ri.business_logic,
    ri.reasoning,
    ri.key_factors,
    ri.metadata,
    ri.created_at,
    ri.updated_at,
    -- Source event details
    se.source_platform as source_platform,
    se.document_type as source_document_type,
    se.amount_usd as source_amount,
    se.source_ts as source_date,
    se.vendor_standard as source_vendor,
    -- Target event details
    te.source_platform as target_platform,
    te.document_type as target_document_type,
    te.amount_usd as target_amount,
    te.source_ts as target_date,
    te.vendor_standard as target_vendor,
    -- Computed fields
    CASE 
        WHEN ri.temporal_causality IN ('source_causes_target', 'target_causes_source', 'bidirectional') 
        THEN true 
        ELSE false 
    END as is_causal,
    EXTRACT(EPOCH FROM (te.source_ts - se.source_ts)) / 86400 as days_between
FROM public.relationship_instances ri
LEFT JOIN public.raw_events se ON ri.source_event_id = se.id
LEFT JOIN public.raw_events te ON ri.target_event_id = te.id;

-- Grant permissions
GRANT SELECT ON enriched_relationships TO authenticated;
GRANT EXECUTE ON FUNCTION search_similar_relationships TO authenticated;
GRANT EXECUTE ON FUNCTION get_semantic_relationship_stats TO authenticated;
GRANT EXECUTE ON FUNCTION find_causal_chain TO authenticated;

-- Add comments for documentation
COMMENT ON COLUMN public.relationship_instances.semantic_description IS 'Natural language description of the relationship generated by AI';
COMMENT ON COLUMN public.relationship_instances.temporal_causality IS 'Type of temporal causality: source_causes_target, target_causes_source, bidirectional, or correlation_only';
COMMENT ON COLUMN public.relationship_instances.business_logic IS 'Business logic pattern this relationship follows';
COMMENT ON COLUMN public.relationship_instances.relationship_embedding IS 'Vector embedding for semantic similarity search (1536 dimensions from text-embedding-3-small)';
COMMENT ON COLUMN public.relationship_instances.key_factors IS 'JSON array of key factors that support this relationship';

COMMENT ON FUNCTION search_similar_relationships IS 'Find semantically similar relationships using vector similarity search';
COMMENT ON FUNCTION get_semantic_relationship_stats IS 'Get comprehensive statistics about semantic relationship coverage and distribution';
COMMENT ON FUNCTION find_causal_chain IS 'Find causal chains (multi-hop causal relationships) starting from a given event';

-- Create indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_relationship_instances_user_confidence ON public.relationship_instances(user_id, confidence_score DESC);
CREATE INDEX IF NOT EXISTS idx_relationship_instances_created_at ON public.relationship_instances(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_relationship_instances_source_target ON public.relationship_instances(source_event_id, target_event_id);
