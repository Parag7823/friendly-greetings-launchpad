-- Migration: Add document_type column to raw_events table
-- Date: 2025-01-15
-- Purpose: FIX #2 - Save document classification from Phase 3 to row-level events

-- Add document_type and document_confidence columns to raw_events
ALTER TABLE public.raw_events 
ADD COLUMN IF NOT EXISTS document_type TEXT,
ADD COLUMN IF NOT EXISTS document_confidence DECIMAL(3,2);

-- Create index for document_type queries
CREATE INDEX IF NOT EXISTS idx_raw_events_document_type ON public.raw_events(document_type);

-- Create composite index for document type filtering
CREATE INDEX IF NOT EXISTS idx_raw_events_user_document_type ON public.raw_events(user_id, document_type) 
WHERE document_type IS NOT NULL;

-- Add comment to explain the column
COMMENT ON COLUMN public.raw_events.document_type IS 'Document type classification from Phase 3 (e.g., payroll, invoice, bank_statement, financial_data)';
COMMENT ON COLUMN public.raw_events.document_confidence IS 'Confidence score for document type classification (0.0 to 1.0)';

-- Update existing records to have default document_type from classification_metadata
UPDATE public.raw_events
SET document_type = 
    CASE 
        WHEN classification_metadata->>'platform_detection' IS NOT NULL 
        THEN COALESCE(
            (classification_metadata->'platform_detection'->>'document_type'),
            'financial_data'
        )
        ELSE 'financial_data'
    END,
    document_confidence = 
    CASE 
        WHEN classification_metadata->>'platform_detection' IS NOT NULL 
        THEN COALESCE(
            (classification_metadata->'platform_detection'->>'document_confidence')::DECIMAL,
            0.8
        )
        ELSE 0.8
    END
WHERE document_type IS NULL;

-- Create function to search events by document type
CREATE OR REPLACE FUNCTION search_events_by_document_type(
    user_uuid UUID,
    doc_type TEXT DEFAULT NULL,
    min_confidence DECIMAL DEFAULT 0.0
)
RETURNS TABLE(
    id UUID,
    kind TEXT,
    category TEXT,
    subcategory TEXT,
    document_type TEXT,
    document_confidence DECIMAL(3,2),
    source_platform TEXT,
    amount_usd DECIMAL(15,2),
    vendor_standard TEXT,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        re.id,
        re.kind,
        re.category,
        re.subcategory,
        re.document_type,
        re.document_confidence,
        re.source_platform,
        re.amount_usd,
        re.vendor_standard,
        re.created_at
    FROM public.raw_events re
    WHERE re.user_id = user_uuid
    AND (doc_type IS NULL OR re.document_type = doc_type)
    AND (re.document_confidence IS NULL OR re.document_confidence >= min_confidence)
    ORDER BY re.created_at DESC;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create function to get document type statistics
CREATE OR REPLACE FUNCTION get_document_type_stats(user_uuid UUID)
RETURNS TABLE(
    document_type TEXT,
    total_count BIGINT,
    avg_confidence DECIMAL(3,2),
    total_amount_usd DECIMAL(15,2),
    unique_platforms TEXT[]
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        re.document_type,
        COUNT(*) as total_count,
        AVG(re.document_confidence) as avg_confidence,
        SUM(re.amount_usd) as total_amount_usd,
        ARRAY_AGG(DISTINCT re.source_platform) FILTER (WHERE re.source_platform IS NOT NULL) as unique_platforms
    FROM public.raw_events re
    WHERE re.user_id = user_uuid
    AND re.document_type IS NOT NULL
    GROUP BY re.document_type
    ORDER BY total_count DESC;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Log migration completion
DO $$
BEGIN
    RAISE NOTICE 'Migration 20250115000000: Added document_type and document_confidence columns to raw_events table';
    RAISE NOTICE 'Created indexes: idx_raw_events_document_type, idx_raw_events_user_document_type';
    RAISE NOTICE 'Created functions: search_events_by_document_type, get_document_type_stats';
END $$;
