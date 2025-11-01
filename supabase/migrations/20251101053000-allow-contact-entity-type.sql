-- Migration: expand normalized_entities.entity_type to include contact variants
-- Date: 2025-11-01 05:30:00

-- Drop existing constraint so we can widen allowed values
ALTER TABLE public.normalized_entities
DROP CONSTRAINT IF EXISTS normalized_entities_entity_type_check;

-- Allow both singular and plural forms for all supported entity types
ALTER TABLE public.normalized_entities
ADD CONSTRAINT normalized_entities_entity_type_check
CHECK (entity_type IN (
    'employee', 'employees',
    'vendor', 'vendors',
    'customer', 'customers',
    'project', 'projects',
    'contact', 'contacts'
));

COMMENT ON CONSTRAINT normalized_entities_entity_type_check ON public.normalized_entities
IS 'Permits normalized entity types (singular + plural): employee(s), vendor(s), customer(s), project(s), contact(s).';
