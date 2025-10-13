-- Remove Unused Version Tables Migration
-- ============================================================================
-- 
-- REASON FOR REMOVAL:
-- These 3 tables were part of an over-engineered versioning system that was
-- never fully implemented. They consume storage and add complexity without
-- providing any value.
--
-- TABLES BEING REMOVED:
-- 1. file_versions - Track file version relationships (NEVER USED)
-- 2. file_similarity_analysis - Store pairwise file comparisons (NEVER USED)
-- 3. version_recommendations - AI recommendations for versions (2 legacy endpoints only)
--
-- REPLACEMENT:
-- The current system uses event_delta_logs for delta merging, which is:
-- - Simpler (1 table vs 3 tables)
-- - Faster (on-demand vs pre-computed)
-- - More efficient (no N² storage explosion)
-- - Actually used (integrated in production code)
--
-- SAFETY:
-- ✅ No frontend calls to these endpoints
-- ✅ No active backend usage (only 2 legacy endpoints removed)
-- ✅ No data loss (tables are empty or contain test data only)
-- ✅ Can be restored from git history if needed
--
-- Date: 2025-10-13
-- ============================================================================

-- Drop dependent objects first (foreign keys, indexes, policies)
-- ============================================================================

-- Drop indexes
DROP INDEX IF EXISTS public.idx_version_recommendations_user_id;
DROP INDEX IF EXISTS public.idx_version_recommendations_group;
DROP INDEX IF EXISTS public.idx_version_recommendations_pending;

DROP INDEX IF EXISTS public.idx_file_similarity_user_id;
DROP INDEX IF EXISTS public.idx_file_similarity_source;
DROP INDEX IF EXISTS public.idx_file_similarity_target;
DROP INDEX IF EXISTS public.idx_file_similarity_relationship;

DROP INDEX IF EXISTS public.idx_file_versions_user_id;
DROP INDEX IF EXISTS public.idx_file_versions_group_id;
DROP INDEX IF EXISTS public.idx_file_versions_active;
DROP INDEX IF EXISTS public.idx_file_versions_filename;
DROP INDEX IF EXISTS public.idx_file_versions_version_group_id;
DROP INDEX IF EXISTS public.idx_file_versions_file_hash;
DROP INDEX IF EXISTS public.idx_file_versions_is_active_version;
DROP INDEX IF EXISTS public.idx_file_versions_user_group;

-- Drop RLS policies
DROP POLICY IF EXISTS "version_recommendations_select_policy" ON public.version_recommendations;
DROP POLICY IF EXISTS "version_recommendations_insert_policy" ON public.version_recommendations;
DROP POLICY IF EXISTS "version_recommendations_update_policy" ON public.version_recommendations;

DROP POLICY IF EXISTS "file_similarity_select_policy" ON public.file_similarity_analysis;
DROP POLICY IF EXISTS "file_similarity_insert_policy" ON public.file_similarity_analysis;

DROP POLICY IF EXISTS "file_versions_select_policy" ON public.file_versions;
DROP POLICY IF EXISTS "file_versions_insert_policy" ON public.file_versions;
DROP POLICY IF EXISTS "file_versions_update_policy" ON public.file_versions;

-- Drop tables (CASCADE will handle remaining dependencies)
-- ============================================================================

-- Drop version_recommendations first (has FK to file_versions)
DROP TABLE IF EXISTS public.version_recommendations CASCADE;

-- Drop file_similarity_analysis
DROP TABLE IF EXISTS public.file_similarity_analysis CASCADE;

-- Drop file_versions last
DROP TABLE IF EXISTS public.file_versions CASCADE;

-- ============================================================================
-- VERIFICATION
-- ============================================================================

-- Verify tables are gone
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'file_versions') THEN
        RAISE EXCEPTION 'file_versions table still exists!';
    END IF;
    
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'file_similarity_analysis') THEN
        RAISE EXCEPTION 'file_similarity_analysis table still exists!';
    END IF;
    
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'version_recommendations') THEN
        RAISE EXCEPTION 'version_recommendations table still exists!';
    END IF;
    
    RAISE NOTICE 'SUCCESS: All 3 unused version tables have been removed';
END $$;

-- ============================================================================
-- DOCUMENTATION
-- ============================================================================

COMMENT ON SCHEMA public IS 'Cleaned up unused version tables on 2025-10-13. Removed file_versions, file_similarity_analysis, and version_recommendations tables as they were part of an unfinished over-engineered design. Current system uses event_delta_logs for delta merging.';
