-- Migration: Add RLS Policies for Relationship Tables
-- Date: 2025-10-25 13:00:00
-- Purpose: Enable Row Level Security for relationship tables to allow users to see their data

-- ============================================================================
-- Enable RLS on relationship tables
-- ============================================================================

ALTER TABLE public.relationship_instances ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.relationship_patterns ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.cross_platform_relationships ENABLE ROW LEVEL SECURITY;

-- ============================================================================
-- RLS Policies for relationship_instances
-- ============================================================================

-- Service role has full access
DROP POLICY IF EXISTS "service_role_all_relationship_instances" ON public.relationship_instances;
CREATE POLICY "service_role_all_relationship_instances" ON public.relationship_instances
    FOR ALL USING (auth.role() = 'service_role');

-- Users can view and manage their own relationships
DROP POLICY IF EXISTS "users_own_relationship_instances" ON public.relationship_instances;
CREATE POLICY "users_own_relationship_instances" ON public.relationship_instances
    FOR ALL USING (auth.uid() = user_id);

-- ============================================================================
-- RLS Policies for relationship_patterns
-- ============================================================================

-- Service role has full access
DROP POLICY IF EXISTS "service_role_all_relationship_patterns" ON public.relationship_patterns;
CREATE POLICY "service_role_all_relationship_patterns" ON public.relationship_patterns
    FOR ALL USING (auth.role() = 'service_role');

-- Users can view and manage their own patterns
DROP POLICY IF EXISTS "users_own_relationship_patterns" ON public.relationship_patterns;
CREATE POLICY "users_own_relationship_patterns" ON public.relationship_patterns
    FOR ALL USING (auth.uid() = user_id);

-- ============================================================================
-- RLS Policies for cross_platform_relationships
-- ============================================================================

-- Service role has full access
DROP POLICY IF EXISTS "service_role_all_cross_platform_relationships" ON public.cross_platform_relationships;
CREATE POLICY "service_role_all_cross_platform_relationships" ON public.cross_platform_relationships
    FOR ALL USING (auth.role() = 'service_role');

-- Users can view and manage their own cross-platform relationships
DROP POLICY IF EXISTS "users_own_cross_platform_relationships" ON public.cross_platform_relationships;
CREATE POLICY "users_own_cross_platform_relationships" ON public.cross_platform_relationships
    FOR ALL USING (auth.uid() = user_id);

-- ============================================================================
-- Grant permissions
-- ============================================================================

GRANT SELECT, INSERT, UPDATE, DELETE ON public.relationship_instances TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.relationship_patterns TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.cross_platform_relationships TO authenticated;

-- ============================================================================
-- Comments
-- ============================================================================

COMMENT ON POLICY "service_role_all_relationship_instances" ON public.relationship_instances 
IS 'Service role has full access to all relationship instances';

COMMENT ON POLICY "users_own_relationship_instances" ON public.relationship_instances 
IS 'Users can only access their own relationship instances';

COMMENT ON POLICY "service_role_all_relationship_patterns" ON public.relationship_patterns 
IS 'Service role has full access to all relationship patterns';

COMMENT ON POLICY "users_own_relationship_patterns" ON public.relationship_patterns 
IS 'Users can only access their own relationship patterns';

COMMENT ON POLICY "service_role_all_cross_platform_relationships" ON public.cross_platform_relationships 
IS 'Service role has full access to all cross-platform relationships';

COMMENT ON POLICY "users_own_cross_platform_relationships" ON public.cross_platform_relationships 
IS 'Users can only access their own cross-platform relationships';
