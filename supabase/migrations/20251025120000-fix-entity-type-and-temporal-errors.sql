-- Migration: Fix Entity Type Check Constraint and Temporal SQL Errors
-- Date: 2025-10-25 12:00:00
-- Issues Fixed:
-- 1. Entity type 'projects' not allowed in normalized_entities table
-- 2. Ambiguous column reference in predict_missing_relationships (already fixed but verifying)

-- ============================================================================
-- FIX 1: Update entity_type CHECK constraint to include 'projects'
-- ============================================================================

-- Drop the existing-- Add CHECK constraint for entity_type with both singular and plural forms
-- CRITICAL FIX: Support both singular AND plural for ALL entity types
ALTER TABLE public.normalized_entities 
DROP CONSTRAINT IF EXISTS normalized_entities_entity_type_check;

ALTER TABLE public.normalized_entities 
ADD CONSTRAINT normalized_entities_entity_type_check 
CHECK (entity_type IN (
    'employee', 'employees',
    'vendor', 'vendors', 
    'customer', 'customers',
    'project', 'projects'
));

-- Note: We allow both 'project' and 'projects' for backward compatibility
-- The code may use either variant, so we support both

-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================

-- Verify the constraint is updated
-- SELECT conname, pg_get_constraintdef(oid) 
-- FROM pg_constraint 
-- WHERE conrelid = 'public.normalized_entities'::regclass 
-- AND conname = 'normalized_entities_entity_type_check';

-- Test entity creation with 'projects' type (should succeed now)
-- SELECT find_or_create_entity(
--     '93071a7e-aa3e-450f-8f72-c3254a67a917'::UUID,
--     'Test Project',
--     'projects',
--     'test_platform',
--     NULL, NULL, NULL, NULL, 'test.csv'
-- );

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON CONSTRAINT normalized_entities_entity_type_check ON public.normalized_entities 
IS 'Allows entity types (both singular and plural): employee/employees, vendor/vendors, customer/customers, project/projects. Both forms supported for compatibility.';
