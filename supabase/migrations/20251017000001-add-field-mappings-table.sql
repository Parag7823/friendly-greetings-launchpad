-- Migration: Add field_mappings table for smart field mapping
-- Date: 2025-10-17
-- Purpose: Enable adaptive field mapping based on user data and feedback

-- Create field_mappings table for storing user-specific and learned column mappings
CREATE TABLE IF NOT EXISTS public.field_mappings (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    
    -- Mapping information
    source_column TEXT NOT NULL, -- Original column name from user's file
    target_field TEXT NOT NULL, -- Standardized field name (e.g., 'amount', 'vendor', 'date')
    
    -- Context
    platform TEXT, -- Platform this mapping applies to (null = global)
    document_type TEXT, -- Document type this mapping applies to (null = global)
    filename_pattern TEXT, -- Filename pattern this mapping applies to (null = global)
    
    -- Confidence and source
    confidence NUMERIC(5, 4) DEFAULT 0.8 CHECK (confidence >= 0 AND confidence <= 1),
    mapping_source TEXT NOT NULL CHECK (mapping_source IN ('user_feedback', 'ai_learned', 'pattern_match', 'manual')),
    
    -- Usage tracking
    usage_count INTEGER DEFAULT 1,
    success_count INTEGER DEFAULT 0,
    last_used_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    
    -- Additional metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Ensure unique mappings per user/context
    UNIQUE(user_id, source_column, platform, document_type)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_field_mappings_user_id ON public.field_mappings(user_id);
CREATE INDEX IF NOT EXISTS idx_field_mappings_source_column ON public.field_mappings(source_column);
CREATE INDEX IF NOT EXISTS idx_field_mappings_target_field ON public.field_mappings(target_field);
CREATE INDEX IF NOT EXISTS idx_field_mappings_platform ON public.field_mappings(platform);
CREATE INDEX IF NOT EXISTS idx_field_mappings_confidence ON public.field_mappings(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_field_mappings_last_used ON public.field_mappings(last_used_at DESC);

-- Composite index for lookup queries
CREATE INDEX IF NOT EXISTS idx_field_mappings_lookup ON public.field_mappings(user_id, source_column, platform);

-- GIN index for JSONB metadata
CREATE INDEX IF NOT EXISTS idx_field_mappings_metadata ON public.field_mappings USING GIN (metadata);

-- Enable Row Level Security
ALTER TABLE public.field_mappings ENABLE ROW LEVEL SECURITY;

-- RLS Policies
DROP POLICY IF EXISTS "service_role_all_field_mappings" ON public.field_mappings;
CREATE POLICY "service_role_all_field_mappings" ON public.field_mappings
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_field_mappings" ON public.field_mappings;
CREATE POLICY "users_own_field_mappings" ON public.field_mappings
    FOR ALL USING (auth.uid() = user_id);

-- Function to get field mapping for a column
CREATE OR REPLACE FUNCTION get_field_mapping(
    p_user_id UUID,
    p_source_column TEXT,
    p_platform TEXT DEFAULT NULL,
    p_document_type TEXT DEFAULT NULL
)
RETURNS TABLE(
    target_field TEXT,
    confidence NUMERIC,
    mapping_source TEXT,
    metadata JSONB
) AS $$
BEGIN
    -- Try exact match first (with platform and document_type)
    RETURN QUERY
    SELECT 
        fm.target_field,
        fm.confidence,
        fm.mapping_source,
        fm.metadata
    FROM public.field_mappings fm
    WHERE fm.user_id = p_user_id
      AND LOWER(fm.source_column) = LOWER(p_source_column)
      AND (fm.platform = p_platform OR fm.platform IS NULL)
      AND (fm.document_type = p_document_type OR fm.document_type IS NULL)
    ORDER BY 
        -- Prioritize exact context matches
        CASE WHEN fm.platform = p_platform AND fm.document_type = p_document_type THEN 1
             WHEN fm.platform = p_platform THEN 2
             WHEN fm.document_type = p_document_type THEN 3
             ELSE 4 END,
        fm.confidence DESC,
        fm.usage_count DESC
    LIMIT 1;
    
    -- If no exact match, return null
    IF NOT FOUND THEN
        RETURN;
    END IF;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to upsert field mapping (create or update)
CREATE OR REPLACE FUNCTION upsert_field_mapping(
    p_user_id UUID,
    p_source_column TEXT,
    p_target_field TEXT,
    p_platform TEXT DEFAULT NULL,
    p_document_type TEXT DEFAULT NULL,
    p_confidence NUMERIC DEFAULT 0.8,
    p_mapping_source TEXT DEFAULT 'user_feedback',
    p_metadata JSONB DEFAULT '{}'::jsonb
)
RETURNS UUID AS $$
DECLARE
    v_mapping_id UUID;
BEGIN
    INSERT INTO public.field_mappings (
        user_id,
        source_column,
        target_field,
        platform,
        document_type,
        confidence,
        mapping_source,
        usage_count,
        success_count,
        last_used_at,
        metadata
    )
    VALUES (
        p_user_id,
        p_source_column,
        p_target_field,
        p_platform,
        p_document_type,
        p_confidence,
        p_mapping_source,
        1,
        0,
        now(),
        p_metadata
    )
    ON CONFLICT (user_id, source_column, platform, document_type)
    DO UPDATE SET
        target_field = EXCLUDED.target_field,
        confidence = (field_mappings.confidence + EXCLUDED.confidence) / 2, -- Average confidence
        usage_count = field_mappings.usage_count + 1,
        last_used_at = now(),
        updated_at = now(),
        metadata = field_mappings.metadata || EXCLUDED.metadata
    RETURNING id INTO v_mapping_id;
    
    RETURN v_mapping_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to record successful mapping usage
CREATE OR REPLACE FUNCTION record_mapping_success(p_mapping_id UUID)
RETURNS VOID AS $$
BEGIN
    UPDATE public.field_mappings
    SET 
        success_count = success_count + 1,
        last_used_at = now(),
        updated_at = now()
    WHERE id = p_mapping_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get all mappings for a user
CREATE OR REPLACE FUNCTION get_user_field_mappings(
    p_user_id UUID,
    p_platform TEXT DEFAULT NULL
)
RETURNS TABLE(
    id UUID,
    source_column TEXT,
    target_field TEXT,
    platform TEXT,
    document_type TEXT,
    confidence NUMERIC,
    mapping_source TEXT,
    usage_count INTEGER,
    success_count INTEGER,
    last_used_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        fm.id,
        fm.source_column,
        fm.target_field,
        fm.platform,
        fm.document_type,
        fm.confidence,
        fm.mapping_source,
        fm.usage_count,
        fm.success_count,
        fm.last_used_at
    FROM public.field_mappings fm
    WHERE fm.user_id = p_user_id
      AND (p_platform IS NULL OR fm.platform = p_platform OR fm.platform IS NULL)
    ORDER BY fm.confidence DESC, fm.usage_count DESC;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_field_mappings_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_field_mappings_updated_at ON public.field_mappings;
CREATE TRIGGER trigger_update_field_mappings_updated_at
    BEFORE UPDATE ON public.field_mappings
    FOR EACH ROW
    EXECUTE FUNCTION update_field_mappings_updated_at();

-- Grant necessary permissions
GRANT SELECT, INSERT, UPDATE ON public.field_mappings TO authenticated;
GRANT EXECUTE ON FUNCTION get_field_mapping TO authenticated;
GRANT EXECUTE ON FUNCTION upsert_field_mapping TO authenticated;
GRANT EXECUTE ON FUNCTION record_mapping_success TO authenticated;
GRANT EXECUTE ON FUNCTION get_user_field_mappings TO authenticated;

-- Add comment
COMMENT ON TABLE public.field_mappings IS 'Stores user-specific and learned field mappings for adaptive data extraction';
COMMENT ON FUNCTION get_field_mapping IS 'Retrieves the best field mapping for a given source column';
COMMENT ON FUNCTION upsert_field_mapping IS 'Creates or updates a field mapping';
COMMENT ON FUNCTION record_mapping_success IS 'Records successful usage of a field mapping';
COMMENT ON FUNCTION get_user_field_mappings IS 'Retrieves all field mappings for a user';
