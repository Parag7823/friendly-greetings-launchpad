-- Migration: Create Materialized View for Enriched Relationships
-- Purpose: Solve N+10 graph build problem by joining enrichment tables in one SQL query
-- Impact: Reduces multiple separate DB queries to 1, eliminates connection pool exhaustion
-- Date: 2025-11-20
-- Schema: Based on actual Supabase schema verification

-- Create materialized view that joins relationship_instances with enrichment tables
CREATE MATERIALIZED VIEW IF NOT EXISTS view_enriched_relationships AS
SELECT
    -- Core relationship data
    ri.id,
    ri.user_id,
    ri.source_event_id,
    ri.target_event_id,
    ri.relationship_type,
    ri.confidence_score,
    ri.detection_method,
    ri.reasoning,
    ri.created_at,
    ri.updated_at,
    ri.pattern_id,
    ri.semantic_description,
    ri.temporal_causality,
    ri.business_logic,
    ri.key_factors,
    ri.metadata,
    ri.relationship_embedding,
    
    -- Layer 1: Causal Intelligence (from causal_relationships)
    COALESCE(cr.causal_score, 0.0) as causal_strength,
    COALESCE(cr.causal_direction, 'none') as causal_direction,
    
    -- Layer 2: Temporal Intelligence (from temporal_patterns)
    COALESCE(tp.id, NULL) as temporal_pattern_id,
    COALESCE(tp.confidence_score, 0.0) as recurrence_score,
    COALESCE(CONCAT('Every ', ROUND(tp.avg_days_between::numeric, 1), ' days'), 'none') as recurrence_frequency,
    NULL::TIMESTAMP WITH TIME ZONE as last_occurrence,
    NULL::TIMESTAMP WITH TIME ZONE as next_predicted_occurrence,
    
    -- Layer 3: Seasonal Intelligence (from temporal_patterns seasonal fields)
    COALESCE(tp.seasonal_period_days, 0) as seasonal_pattern_id,
    COALESCE(tp.seasonal_amplitude, 0.0) as seasonal_strength,
    '[]'::jsonb as seasonal_months,
    
    -- Layer 4: Pattern Intelligence (from relationship_patterns)
    COALESCE(rp.id, NULL) as pattern_id_enriched,
    0.0 as pattern_confidence,
    COALESCE(rp.relationship_type, '') as pattern_name,
    
    -- Layer 5: Cross-Platform Intelligence (from cross_platform_relationships)
    COALESCE(cpr.id, NULL) as cross_platform_id,
    COALESCE(cpr.source_platform, '')::text as platform_sources,
    
    -- Layer 6: Prediction Intelligence (from predicted_relationships)
    COALESCE(pr.id, NULL) as predicted_relationship_id,
    COALESCE(pr.confidence_score, 0.0) as prediction_confidence,
    COALESCE(pr.prediction_reasoning, '') as prediction_reason,
    
    -- Layer 7: Root Cause Intelligence (from root_cause_analyses)
    COALESCE(rca.id, NULL) as root_cause_id,
    COALESCE(rca.root_cause_description, '') as root_cause_analysis,
    
    -- Layer 8: Change Tracking (no direct relationship - placeholder)
    NULL::UUID as delta_log_id,
    'none' as change_type,
    
    -- Layer 9: Fraud Detection (placeholder - duplicate_transactions table not yet created)
    NULL::UUID as duplicate_transaction_id,
    FALSE as is_duplicate,
    0.0 as duplicate_confidence
    
FROM relationship_instances ri

-- Layer 1: Causal relationships (join on relationship_id)
LEFT JOIN causal_relationships cr 
    ON ri.id = cr.relationship_id 
    AND ri.user_id = cr.user_id

-- Layer 2: Temporal patterns (join on relationship_type)
LEFT JOIN temporal_patterns tp 
    ON ri.relationship_type = tp.relationship_type 
    AND ri.user_id = tp.user_id

-- Layer 4: Relationship patterns (join on pattern_id from relationship_instances)
LEFT JOIN relationship_patterns rp 
    ON ri.pattern_id = rp.id 
    AND ri.user_id = rp.user_id

-- Layer 5: Cross-platform relationships (join on source_event_id and target_event_id)
LEFT JOIN cross_platform_relationships cpr 
    ON ri.source_event_id = cpr.source_event_id 
    AND ri.target_event_id = cpr.target_event_id
    AND ri.user_id = cpr.user_id

-- Layer 6: Predicted relationships (join on source_event_id)
LEFT JOIN predicted_relationships pr 
    ON ri.source_event_id = pr.source_event_id 
    AND ri.user_id = pr.user_id
    AND pr.status IN ('pending', 'fulfilled')

-- Layer 7: Root cause analyses (join on source_event_id as problem_event_id)
LEFT JOIN root_cause_analyses rca 
    ON ri.source_event_id = rca.problem_event_id 
    AND ri.user_id = rca.user_id

-- Layer 9: Duplicate transactions join removed (table not yet created)
-- When duplicate_transactions table is added, uncomment:
-- LEFT JOIN duplicate_transactions dt 
--     ON ri.source_event_id = dt.from_event_id 
--     AND ri.target_event_id = dt.to_event_id
--     AND ri.user_id = dt.user_id

WHERE TRUE;

-- Create UNIQUE index on materialized view (required for concurrent refresh)
CREATE UNIQUE INDEX IF NOT EXISTS idx_enriched_rel_id_unique 
ON view_enriched_relationships (id);

-- Create additional indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_enriched_rel_user_id 
ON view_enriched_relationships (user_id);

CREATE INDEX IF NOT EXISTS idx_enriched_rel_created_at 
ON view_enriched_relationships (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_enriched_rel_source_target 
ON view_enriched_relationships (user_id, source_event_id, target_event_id);

-- Refresh the materialized view (now possible with unique index)
REFRESH MATERIALIZED VIEW CONCURRENTLY view_enriched_relationships;

-- Optional: Set up automatic refresh on schedule (requires pg_cron extension)
-- SELECT cron.schedule('refresh_enriched_relationships', '*/5 * * * *', 'REFRESH MATERIALIZED VIEW CONCURRENTLY view_enriched_relationships');
