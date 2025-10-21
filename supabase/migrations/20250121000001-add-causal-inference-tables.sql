-- Migration: Add causal inference tables for Bradford Hill criteria and causal analysis
-- Date: 2025-01-21
-- Purpose: Enable causal relationship detection, root cause analysis, and counterfactual analysis

-- ============================================================================
-- 1. CAUSAL RELATIONSHIPS TABLE
-- ============================================================================
-- Stores causal analysis results using Bradford Hill criteria
CREATE TABLE IF NOT EXISTS public.causal_relationships (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL,
    relationship_id UUID NOT NULL,
    
    -- Bradford Hill Criteria Scores (0.0 - 1.0)
    temporal_precedence_score FLOAT DEFAULT 0.0,
    strength_score FLOAT DEFAULT 0.0,
    consistency_score FLOAT DEFAULT 0.0,
    specificity_score FLOAT DEFAULT 0.0,
    dose_response_score FLOAT DEFAULT 0.0,
    plausibility_score FLOAT DEFAULT 0.0,
    
    -- Overall causal score (average of all criteria)
    causal_score FLOAT DEFAULT 0.0,
    
    -- Causal determination
    is_causal BOOLEAN DEFAULT FALSE,
    causal_direction VARCHAR(50) CHECK (causal_direction IN ('source_to_target', 'target_to_source', 'bidirectional', 'none')),
    
    -- Metadata
    criteria_details JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Foreign key
    FOREIGN KEY (relationship_id) REFERENCES public.relationship_instances(id) ON DELETE CASCADE,
    
    -- Unique constraint
    UNIQUE(relationship_id)
);

-- Indexes for causal_relationships
CREATE INDEX IF NOT EXISTS idx_causal_relationships_user ON public.causal_relationships(user_id);
CREATE INDEX IF NOT EXISTS idx_causal_relationships_score ON public.causal_relationships(causal_score DESC);
CREATE INDEX IF NOT EXISTS idx_causal_relationships_is_causal ON public.causal_relationships(is_causal) WHERE is_causal = TRUE;
CREATE INDEX IF NOT EXISTS idx_causal_relationships_relationship ON public.causal_relationships(relationship_id);

-- ============================================================================
-- 2. ROOT CAUSE ANALYSES TABLE
-- ============================================================================
-- Stores root cause analysis results for problem events
CREATE TABLE IF NOT EXISTS public.root_cause_analyses (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL,
    problem_event_id UUID NOT NULL,
    root_event_id UUID NOT NULL,
    
    -- Causal path from root to problem
    causal_path JSONB NOT NULL DEFAULT '[]',
    path_length INTEGER DEFAULT 0,
    
    -- Impact analysis
    total_impact_usd FLOAT DEFAULT 0.0,
    affected_event_count INTEGER DEFAULT 0,
    affected_event_ids JSONB DEFAULT '[]',
    
    -- Root cause details
    root_cause_type VARCHAR(100),
    root_cause_description TEXT,
    confidence_score FLOAT DEFAULT 0.0,
    
    -- Metadata
    analysis_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Foreign keys
    FOREIGN KEY (problem_event_id) REFERENCES public.raw_events(id) ON DELETE CASCADE,
    FOREIGN KEY (root_event_id) REFERENCES public.raw_events(id) ON DELETE CASCADE
);

-- Indexes for root_cause_analyses
CREATE INDEX IF NOT EXISTS idx_root_cause_analyses_user ON public.root_cause_analyses(user_id);
CREATE INDEX IF NOT EXISTS idx_root_cause_analyses_problem ON public.root_cause_analyses(problem_event_id);
CREATE INDEX IF NOT EXISTS idx_root_cause_analyses_root ON public.root_cause_analyses(root_event_id);
CREATE INDEX IF NOT EXISTS idx_root_cause_analyses_path_length ON public.root_cause_analyses(path_length);
CREATE INDEX IF NOT EXISTS idx_root_cause_analyses_impact ON public.root_cause_analyses(total_impact_usd DESC);

-- ============================================================================
-- 3. COUNTERFACTUAL ANALYSES TABLE
-- ============================================================================
-- Stores "what-if" scenario analysis results
CREATE TABLE IF NOT EXISTS public.counterfactual_analyses (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL,
    
    -- Intervention details
    intervention_event_id UUID NOT NULL,
    intervention_type VARCHAR(100) NOT NULL,
    original_value JSONB NOT NULL,
    counterfactual_value JSONB NOT NULL,
    
    -- Impact analysis
    affected_events JSONB DEFAULT '[]',
    total_impact_delta_usd FLOAT DEFAULT 0.0,
    affected_event_count INTEGER DEFAULT 0,
    
    -- Scenario description
    scenario_description TEXT,
    scenario_name VARCHAR(255),
    
    -- Metadata
    analysis_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Foreign key
    FOREIGN KEY (intervention_event_id) REFERENCES public.raw_events(id) ON DELETE CASCADE
);

-- Indexes for counterfactual_analyses
CREATE INDEX IF NOT EXISTS idx_counterfactual_analyses_user ON public.counterfactual_analyses(user_id);
CREATE INDEX IF NOT EXISTS idx_counterfactual_analyses_intervention ON public.counterfactual_analyses(intervention_event_id);
CREATE INDEX IF NOT EXISTS idx_counterfactual_analyses_type ON public.counterfactual_analyses(intervention_type);
CREATE INDEX IF NOT EXISTS idx_counterfactual_analyses_created ON public.counterfactual_analyses(created_at DESC);

-- ============================================================================
-- 4. POSTGRESQL FUNCTIONS
-- ============================================================================

-- Function to calculate Bradford Hill criteria scores
CREATE OR REPLACE FUNCTION calculate_bradford_hill_scores(
    p_relationship_id UUID,
    p_source_event_id UUID,
    p_target_event_id UUID,
    p_user_id UUID
)
RETURNS JSONB AS $$
DECLARE
    source_event RECORD;
    target_event RECORD;
    relationship RECORD;
    scores JSONB;
    temporal_score FLOAT := 0.0;
    strength_score FLOAT := 0.0;
    consistency_score FLOAT := 0.0;
    specificity_score FLOAT := 0.0;
    dose_response_score FLOAT := 0.0;
    plausibility_score FLOAT := 0.0;
    time_diff_days FLOAT;
    similar_count INTEGER;
BEGIN
    -- Fetch events and relationship
    SELECT * INTO source_event FROM public.raw_events WHERE id = p_source_event_id AND user_id = p_user_id;
    SELECT * INTO target_event FROM public.raw_events WHERE id = p_target_event_id AND user_id = p_user_id;
    SELECT * INTO relationship FROM public.relationship_instances WHERE id = p_relationship_id;
    
    IF source_event IS NULL OR target_event IS NULL OR relationship IS NULL THEN
        RETURN json_build_object('error', 'Events or relationship not found');
    END IF;
    
    -- 1. Temporal Precedence: Source must come before target
    time_diff_days := EXTRACT(EPOCH FROM (target_event.source_ts - source_event.source_ts)) / 86400.0;
    IF time_diff_days > 0 THEN
        temporal_score := LEAST(1.0, time_diff_days / 90.0); -- Max score if 90+ days apart
    ELSE
        temporal_score := 0.0;
    END IF;
    
    -- 2. Strength of Association: Use existing confidence score
    strength_score := COALESCE(relationship.confidence_score, 0.0);
    
    -- 3. Consistency: Count similar relationship patterns
    SELECT COUNT(*) INTO similar_count
    FROM public.relationship_instances
    WHERE user_id = p_user_id
        AND relationship_type = relationship.relationship_type
        AND confidence_score >= 0.7;
    
    consistency_score := LEAST(1.0, similar_count / 10.0); -- Max score if 10+ similar patterns
    
    -- 4. Specificity: Check if one-to-one mapping
    SELECT COUNT(*) INTO similar_count
    FROM public.relationship_instances
    WHERE user_id = p_user_id
        AND (source_event_id = p_source_event_id OR target_event_id = p_target_event_id);
    
    specificity_score := CASE 
        WHEN similar_count = 1 THEN 1.0
        WHEN similar_count <= 3 THEN 0.7
        ELSE 0.3
    END;
    
    -- 5. Dose-Response: Amount correlation
    IF source_event.amount_usd IS NOT NULL AND target_event.amount_usd IS NOT NULL THEN
        IF source_event.amount_usd > 0 THEN
            dose_response_score := 1.0 - ABS(source_event.amount_usd - target_event.amount_usd) / GREATEST(source_event.amount_usd, target_event.amount_usd);
            dose_response_score := GREATEST(0.0, dose_response_score);
        END IF;
    END IF;
    
    -- 6. Plausibility: Use business logic and temporal causality from semantic analysis
    plausibility_score := CASE
        WHEN relationship.temporal_causality = 'source_causes_target' THEN 1.0
        WHEN relationship.temporal_causality = 'bidirectional' THEN 0.8
        WHEN relationship.temporal_causality = 'target_causes_source' THEN 0.3
        ELSE 0.5
    END;
    
    -- Build result
    scores := json_build_object(
        'temporal_precedence_score', temporal_score,
        'strength_score', strength_score,
        'consistency_score', consistency_score,
        'specificity_score', specificity_score,
        'dose_response_score', dose_response_score,
        'plausibility_score', plausibility_score,
        'causal_score', (temporal_score + strength_score + consistency_score + specificity_score + dose_response_score + plausibility_score) / 6.0,
        'time_diff_days', time_diff_days,
        'similar_pattern_count', similar_count
    );
    
    RETURN scores;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to find root causes for a problem event
CREATE OR REPLACE FUNCTION find_root_causes(
    p_problem_event_id UUID,
    p_user_id UUID,
    p_max_depth INTEGER DEFAULT 10
)
RETURNS TABLE(
    root_event_id UUID,
    causal_path JSONB,
    path_length INTEGER,
    total_causal_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    WITH RECURSIVE causal_chain AS (
        -- Base case: direct causes of problem event
        SELECT 
            ri.source_event_id as current_event_id,
            ri.target_event_id as next_event_id,
            ARRAY[ri.source_event_id, ri.target_event_id] as path,
            1 as depth,
            cr.causal_score as score
        FROM public.relationship_instances ri
        JOIN public.causal_relationships cr ON cr.relationship_id = ri.id
        WHERE ri.target_event_id = p_problem_event_id
            AND ri.user_id = p_user_id
            AND cr.is_causal = TRUE
        
        UNION ALL
        
        -- Recursive case: trace backwards
        SELECT 
            ri.source_event_id,
            ri.target_event_id,
            cc.path || ri.source_event_id,
            cc.depth + 1,
            cc.score * cr.causal_score
        FROM causal_chain cc
        JOIN public.relationship_instances ri ON ri.target_event_id = cc.current_event_id
        JOIN public.causal_relationships cr ON cr.relationship_id = ri.id
        WHERE cc.depth < p_max_depth
            AND ri.user_id = p_user_id
            AND cr.is_causal = TRUE
            AND NOT (ri.source_event_id = ANY(cc.path)) -- Prevent cycles
    )
    SELECT 
        cc.current_event_id as root_event_id,
        array_to_json(cc.path)::jsonb as causal_path,
        cc.depth as path_length,
        cc.score as total_causal_score
    FROM causal_chain cc
    WHERE NOT EXISTS (
        -- Find events with no incoming causal edges (true roots)
        SELECT 1 FROM public.relationship_instances ri2
        JOIN public.causal_relationships cr2 ON cr2.relationship_id = ri2.id
        WHERE ri2.target_event_id = cc.current_event_id
            AND ri2.user_id = p_user_id
            AND cr2.is_causal = TRUE
    )
    ORDER BY cc.score DESC, cc.depth ASC;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get causal graph statistics
CREATE OR REPLACE FUNCTION get_causal_graph_stats(p_user_id UUID)
RETURNS JSON AS $$
DECLARE
    result JSON;
BEGIN
    SELECT json_build_object(
        'total_relationships', COUNT(*),
        'causal_relationships', COUNT(*) FILTER (WHERE is_causal = TRUE),
        'causal_percentage', ROUND(100.0 * COUNT(*) FILTER (WHERE is_causal = TRUE) / NULLIF(COUNT(*), 0), 2),
        'avg_causal_score', ROUND(AVG(causal_score)::numeric, 3),
        'causal_direction_distribution', (
            SELECT json_object_agg(causal_direction, count)
            FROM (
                SELECT causal_direction, COUNT(*) as count
                FROM public.causal_relationships
                WHERE user_id = p_user_id AND is_causal = TRUE
                GROUP BY causal_direction
            ) t
        ),
        'avg_bradford_hill_scores', (
            SELECT json_build_object(
                'temporal_precedence', ROUND(AVG(temporal_precedence_score)::numeric, 3),
                'strength', ROUND(AVG(strength_score)::numeric, 3),
                'consistency', ROUND(AVG(consistency_score)::numeric, 3),
                'specificity', ROUND(AVG(specificity_score)::numeric, 3),
                'dose_response', ROUND(AVG(dose_response_score)::numeric, 3),
                'plausibility', ROUND(AVG(plausibility_score)::numeric, 3)
            )
            FROM public.causal_relationships
            WHERE user_id = p_user_id AND is_causal = TRUE
        )
    ) INTO result
    FROM public.causal_relationships
    WHERE user_id = p_user_id;
    
    RETURN COALESCE(result, '{}'::json);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_causal_relationship_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_causal_relationship_updated_at ON public.causal_relationships;
CREATE TRIGGER trigger_update_causal_relationship_updated_at
    BEFORE UPDATE ON public.causal_relationships
    FOR EACH ROW
    EXECUTE FUNCTION update_causal_relationship_updated_at();

-- Grant permissions
GRANT SELECT, INSERT, UPDATE ON public.causal_relationships TO authenticated;
GRANT SELECT, INSERT, UPDATE ON public.root_cause_analyses TO authenticated;
GRANT SELECT, INSERT, UPDATE ON public.counterfactual_analyses TO authenticated;
GRANT EXECUTE ON FUNCTION calculate_bradford_hill_scores TO authenticated;
GRANT EXECUTE ON FUNCTION find_root_causes TO authenticated;
GRANT EXECUTE ON FUNCTION get_causal_graph_stats TO authenticated;

-- Add comments for documentation
COMMENT ON TABLE public.causal_relationships IS 'Stores causal analysis results using Bradford Hill criteria for determining cause-effect relationships';
COMMENT ON TABLE public.root_cause_analyses IS 'Stores root cause analysis results tracing problems back to their original causes';
COMMENT ON TABLE public.counterfactual_analyses IS 'Stores what-if scenario analysis results for simulating interventions';

COMMENT ON COLUMN public.causal_relationships.causal_score IS 'Overall causal score (0.0-1.0) averaged from all Bradford Hill criteria';
COMMENT ON COLUMN public.causal_relationships.is_causal IS 'TRUE if causal_score >= 0.7, indicating strong causal relationship';
COMMENT ON COLUMN public.root_cause_analyses.causal_path IS 'JSON array of event IDs showing the causal chain from root to problem';
COMMENT ON COLUMN public.counterfactual_analyses.affected_events IS 'JSON array of events affected by the counterfactual intervention';

COMMENT ON FUNCTION calculate_bradford_hill_scores IS 'Calculates Bradford Hill criteria scores for a relationship to determine causality';
COMMENT ON FUNCTION find_root_causes IS 'Finds root causes of a problem event by traversing the causal graph backwards';
COMMENT ON FUNCTION get_causal_graph_stats IS 'Returns comprehensive statistics about the causal graph for a user';
