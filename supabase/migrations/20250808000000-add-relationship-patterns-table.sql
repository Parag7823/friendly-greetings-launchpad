-- Migration: Add relationship patterns table for flexible relationship engine
-- Date: 2025-08-08

-- Create relationship_patterns table for storing learned relationship patterns
CREATE TABLE IF NOT EXISTS public.relationship_patterns (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL,
    relationship_type VARCHAR(100) NOT NULL,
    pattern_data JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, relationship_type)
);

-- Create cross_platform_relationships table for storing cross-platform relationship mappings
CREATE TABLE IF NOT EXISTS public.cross_platform_relationships (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL,
    source_event_id UUID NOT NULL,
    target_event_id UUID NOT NULL,
    relationship_type VARCHAR(100) NOT NULL,
    source_platform VARCHAR(100),
    target_platform VARCHAR(100),
    platform_compatibility VARCHAR(50),
    confidence_score FLOAT DEFAULT 0.5,
    detection_method VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    FOREIGN KEY (source_event_id) REFERENCES public.raw_events(id) ON DELETE CASCADE,
    FOREIGN KEY (target_event_id) REFERENCES public.raw_events(id) ON DELETE CASCADE
);

-- Create relationship_instances table for storing detected relationship instances
CREATE TABLE IF NOT EXISTS public.relationship_instances (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL,
    source_event_id UUID NOT NULL,
    target_event_id UUID NOT NULL,
    relationship_type VARCHAR(100) NOT NULL,
    confidence_score FLOAT DEFAULT 0.5,
    detection_method VARCHAR(100),
    pattern_id UUID,
    reasoning TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    FOREIGN KEY (source_event_id) REFERENCES public.raw_events(id) ON DELETE CASCADE,
    FOREIGN KEY (target_event_id) REFERENCES public.raw_events(id) ON DELETE CASCADE,
    FOREIGN KEY (pattern_id) REFERENCES public.relationship_patterns(id) ON DELETE SET NULL
);

-- Add indexes for better performance
CREATE INDEX IF NOT EXISTS idx_relationship_patterns_user_type ON public.relationship_patterns(user_id, relationship_type);
CREATE INDEX IF NOT EXISTS idx_cross_platform_relationships_user ON public.cross_platform_relationships(user_id);
CREATE INDEX IF NOT EXISTS idx_cross_platform_relationships_platforms ON public.cross_platform_relationships(source_platform, target_platform);
CREATE INDEX IF NOT EXISTS idx_relationship_instances_user ON public.relationship_instances(user_id);
CREATE INDEX IF NOT EXISTS idx_relationship_instances_type ON public.relationship_instances(relationship_type);

-- Add columns to raw_events for relationship tracking
ALTER TABLE public.raw_events 
ADD COLUMN IF NOT EXISTS relationship_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS last_relationship_check TIMESTAMP WITH TIME ZONE;

-- Create function to update relationship count
CREATE OR REPLACE FUNCTION update_relationship_count()
RETURNS TRIGGER AS $$
BEGIN
    -- Update source event count
    UPDATE public.raw_events 
    SET relationship_count = (
        SELECT COUNT(*) 
        FROM public.relationship_instances 
        WHERE source_event_id = NEW.source_event_id OR target_event_id = NEW.source_event_id
    )
    WHERE id = NEW.source_event_id;
    
    -- Update target event count
    UPDATE public.raw_events 
    SET relationship_count = (
        SELECT COUNT(*) 
        FROM public.relationship_instances 
        WHERE source_event_id = NEW.target_event_id OR target_event_id = NEW.target_event_id
    )
    WHERE id = NEW.target_event_id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create trigger to automatically update relationship counts
DROP TRIGGER IF EXISTS trigger_update_relationship_count ON public.relationship_instances;
CREATE TRIGGER trigger_update_relationship_count
    AFTER INSERT OR DELETE ON public.relationship_instances
    FOR EACH ROW
    EXECUTE FUNCTION update_relationship_count();

-- Create function to get relationship statistics
CREATE OR REPLACE FUNCTION get_relationship_stats(user_id_param UUID)
RETURNS JSON AS $$
DECLARE
    result JSON;
BEGIN
    SELECT json_build_object(
        'total_relationships', COUNT(*),
        'relationships_by_type', (
            SELECT json_object_agg(relationship_type, count)
            FROM (
                SELECT relationship_type, COUNT(*) as count
                FROM public.relationship_instances
                WHERE user_id = user_id_param
                GROUP BY relationship_type
            ) t
        ),
        'cross_platform_relationships', (
            SELECT COUNT(*)
            FROM public.cross_platform_relationships
            WHERE user_id = user_id_param
        ),
        'learned_patterns', (
            SELECT COUNT(*)
            FROM public.relationship_patterns
            WHERE user_id = user_id_param
        ),
        'platform_compatibility', (
            SELECT json_object_agg(platform_compatibility, count)
            FROM (
                SELECT platform_compatibility, COUNT(*) as count
                FROM public.cross_platform_relationships
                WHERE user_id = user_id_param
                GROUP BY platform_compatibility
            ) t
        )
    ) INTO result
    FROM public.relationship_instances
    WHERE user_id = user_id_param;
    
    RETURN COALESCE(result, '{}'::json);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create function to search relationships
CREATE OR REPLACE FUNCTION search_relationships(
    user_id_param UUID,
    search_term TEXT DEFAULT NULL,
    relationship_type_param TEXT DEFAULT NULL,
    min_confidence FLOAT DEFAULT 0.0
)
RETURNS TABLE(
    id UUID,
    source_event_id UUID,
    target_event_id UUID,
    relationship_type TEXT,
    confidence_score FLOAT,
    detection_method TEXT,
    reasoning TEXT,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ri.id,
        ri.source_event_id,
        ri.target_event_id,
        ri.relationship_type,
        ri.confidence_score,
        ri.detection_method,
        ri.reasoning,
        ri.created_at
    FROM public.relationship_instances ri
    WHERE ri.user_id = user_id_param
        AND (search_term IS NULL OR ri.reasoning ILIKE '%' || search_term || '%')
        AND (relationship_type_param IS NULL OR ri.relationship_type = relationship_type_param)
        AND ri.confidence_score >= min_confidence
    ORDER BY ri.confidence_score DESC, ri.created_at DESC;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER; 