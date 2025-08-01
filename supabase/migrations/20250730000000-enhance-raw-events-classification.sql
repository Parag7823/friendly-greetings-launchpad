-- Enhance raw_events table with AI-powered classification fields
-- This adds support for semantic understanding and entity extraction

-- Add new classification fields
ALTER TABLE public.raw_events 
ADD COLUMN category TEXT, -- 'payroll', 'revenue', 'expense', 'investment', 'tax', 'other'
ADD COLUMN subcategory TEXT, -- 'employee_salary', 'office_rent', 'client_payment', etc.
ADD COLUMN entities JSONB DEFAULT '{}', -- Extracted entities: employees, vendors, customers, projects
ADD COLUMN relationships JSONB DEFAULT '{}'; -- Mapped relationships to internal IDs

-- Add indexes for new fields
CREATE INDEX idx_raw_events_category ON public.raw_events(category);
CREATE INDEX idx_raw_events_subcategory ON public.raw_events(subcategory);
CREATE INDEX idx_raw_events_category_kind ON public.raw_events(category, kind);

-- Update the statistics function to include category breakdown
CREATE OR REPLACE FUNCTION get_raw_events_stats(user_uuid UUID)
RETURNS TABLE(
    total_events BIGINT,
    processed_events BIGINT,
    failed_events BIGINT,
    pending_events BIGINT,
    unique_files BIGINT,
    unique_platforms TEXT[],
    category_breakdown JSONB,
    kind_breakdown JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*) as total_events,
        COUNT(*) FILTER (WHERE status = 'processed') as processed_events,
        COUNT(*) FILTER (WHERE status = 'failed') as failed_events,
        COUNT(*) FILTER (WHERE status = 'pending') as pending_events,
        COUNT(DISTINCT file_id) as unique_files,
        ARRAY_AGG(DISTINCT source_platform) FILTER (WHERE source_platform IS NOT NULL) as unique_platforms,
        jsonb_object_agg(
            COALESCE(category, 'unknown'), 
            COUNT(*)
        ) FILTER (WHERE category IS NOT NULL) as category_breakdown,
        jsonb_object_agg(
            COALESCE(kind, 'unknown'), 
            COUNT(*)
        ) FILTER (WHERE kind IS NOT NULL) as kind_breakdown
    FROM public.raw_events
    WHERE user_id = user_uuid;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Add a function to get entity statistics
CREATE OR REPLACE FUNCTION get_entity_stats(user_uuid UUID)
RETURNS TABLE(
    total_employees BIGINT,
    total_vendors BIGINT,
    total_customers BIGINT,
    total_projects BIGINT,
    employee_names TEXT[],
    vendor_names TEXT[],
    customer_names TEXT[],
    project_names TEXT[]
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(DISTINCT jsonb_array_elements_text(entities->'employees')) as total_employees,
        COUNT(DISTINCT jsonb_array_elements_text(entities->'vendors')) as total_vendors,
        COUNT(DISTINCT jsonb_array_elements_text(entities->'customers')) as total_customers,
        COUNT(DISTINCT jsonb_array_elements_text(entities->'projects')) as total_projects,
        ARRAY_AGG(DISTINCT jsonb_array_elements_text(entities->'employees')) FILTER (WHERE entities->'employees' IS NOT NULL) as employee_names,
        ARRAY_AGG(DISTINCT jsonb_array_elements_text(entities->'vendors')) FILTER (WHERE entities->'vendors' IS NOT NULL) as vendor_names,
        ARRAY_AGG(DISTINCT jsonb_array_elements_text(entities->'customers')) FILTER (WHERE entities->'customers' IS NOT NULL) as customer_names,
        ARRAY_AGG(DISTINCT jsonb_array_elements_text(entities->'projects')) FILTER (WHERE entities->'projects' IS NOT NULL) as project_names
    FROM public.raw_events
    WHERE user_id = user_uuid;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Add a function to search events by entity
CREATE OR REPLACE FUNCTION search_events_by_entity(user_uuid UUID, entity_type TEXT, entity_name TEXT)
RETURNS TABLE(
    id UUID,
    kind TEXT,
    category TEXT,
    subcategory TEXT,
    source_platform TEXT,
    payload JSONB,
    classification_metadata JSONB,
    entities JSONB,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        re.id,
        re.kind,
        re.category,
        re.subcategory,
        re.source_platform,
        re.payload,
        re.classification_metadata,
        re.entities,
        re.created_at
    FROM public.raw_events re
    WHERE re.user_id = user_uuid
    AND re.entities->entity_type ? entity_name;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER; 