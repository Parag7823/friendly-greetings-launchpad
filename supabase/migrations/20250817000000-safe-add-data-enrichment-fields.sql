-- Safe migration to add data enrichment fields to raw_events table
-- This migration safely adds fields for currency normalization, vendor standardization, and platform ID extraction
-- It checks if columns exist before adding them to prevent errors

-- Add enrichment fields to raw_events table (only if they don't exist)
DO $$
BEGIN
    -- Add amount_original column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'raw_events' 
                   AND column_name = 'amount_original' 
                   AND table_schema = 'public') THEN
        ALTER TABLE public.raw_events ADD COLUMN amount_original DECIMAL(15,2);
    END IF;
    
    -- Add amount_usd column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'raw_events' 
                   AND column_name = 'amount_usd' 
                   AND table_schema = 'public') THEN
        ALTER TABLE public.raw_events ADD COLUMN amount_usd DECIMAL(15,2);
    END IF;
    
    -- Add currency column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'raw_events' 
                   AND column_name = 'currency' 
                   AND table_schema = 'public') THEN
        ALTER TABLE public.raw_events ADD COLUMN currency TEXT DEFAULT 'USD';
    END IF;
    
    -- Add exchange_rate column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'raw_events' 
                   AND column_name = 'exchange_rate' 
                   AND table_schema = 'public') THEN
        ALTER TABLE public.raw_events ADD COLUMN exchange_rate DECIMAL(10,6);
    END IF;
    
    -- Add exchange_date column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'raw_events' 
                   AND column_name = 'exchange_date' 
                   AND table_schema = 'public') THEN
        ALTER TABLE public.raw_events ADD COLUMN exchange_date DATE;
    END IF;
    
    -- Add vendor_raw column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'raw_events' 
                   AND column_name = 'vendor_raw' 
                   AND table_schema = 'public') THEN
        ALTER TABLE public.raw_events ADD COLUMN vendor_raw TEXT;
    END IF;
    
    -- Add vendor_standard column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'raw_events' 
                   AND column_name = 'vendor_standard' 
                   AND table_schema = 'public') THEN
        ALTER TABLE public.raw_events ADD COLUMN vendor_standard TEXT;
    END IF;
    
    -- Add vendor_confidence column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'raw_events' 
                   AND column_name = 'vendor_confidence' 
                   AND table_schema = 'public') THEN
        ALTER TABLE public.raw_events ADD COLUMN vendor_confidence DECIMAL(3,2);
    END IF;
    
    -- Add vendor_cleaning_method column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'raw_events' 
                   AND column_name = 'vendor_cleaning_method' 
                   AND table_schema = 'public') THEN
        ALTER TABLE public.raw_events ADD COLUMN vendor_cleaning_method TEXT;
    END IF;
    
    -- Add platform_ids column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'raw_events' 
                   AND column_name = 'platform_ids' 
                   AND table_schema = 'public') THEN
        ALTER TABLE public.raw_events ADD COLUMN platform_ids JSONB DEFAULT '{}';
    END IF;
    
    -- Add standard_description column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'raw_events' 
                   AND column_name = 'standard_description' 
                   AND table_schema = 'public') THEN
        ALTER TABLE public.raw_events ADD COLUMN standard_description TEXT;
    END IF;
    
    -- Add ingested_on column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'raw_events' 
                   AND column_name = 'ingested_on' 
                   AND table_schema = 'public') THEN
        ALTER TABLE public.raw_events ADD COLUMN ingested_on TIMESTAMP WITH TIME ZONE DEFAULT now();
    END IF;
END $$;

-- Add indexes for new fields (only if they don't exist)
CREATE INDEX IF NOT EXISTS idx_raw_events_amount_usd ON public.raw_events(amount_usd);
CREATE INDEX IF NOT EXISTS idx_raw_events_currency ON public.raw_events(currency);
CREATE INDEX IF NOT EXISTS idx_raw_events_vendor_standard ON public.raw_events(vendor_standard);
CREATE INDEX IF NOT EXISTS idx_raw_events_ingested_on ON public.raw_events(ingested_on);

-- CRITICAL: Add GIN index for platform_ids JSONB column for efficient querying (only if it doesn't exist)
CREATE INDEX IF NOT EXISTS idx_raw_events_platform_ids_gin ON public.raw_events USING GIN (platform_ids);

-- Add composite indexes for common query patterns (only if they don't exist)
CREATE INDEX IF NOT EXISTS idx_raw_events_user_platform_ids ON public.raw_events(user_id, platform_ids) WHERE platform_ids != '{}';
CREATE INDEX IF NOT EXISTS idx_raw_events_platform_confidence ON public.raw_events(source_platform, confidence_score) WHERE confidence_score > 0.5;

-- Create specialized functions for platform ID operations (replace if they exist)
CREATE OR REPLACE FUNCTION get_events_by_platform_id(
    p_user_id UUID,
    p_platform TEXT,
    p_id_type TEXT,
    p_id_value TEXT
)
RETURNS TABLE (
    id UUID,
    kind TEXT,
    source_platform TEXT,
    payload JSONB,
    platform_ids JSONB,
    confidence_score DECIMAL(3,2),
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        re.id,
        re.kind,
        re.source_platform,
        re.payload,
        re.platform_ids,
        re.confidence_score,
        re.created_at
    FROM public.raw_events re
    WHERE re.user_id = p_user_id
        AND re.platform_ids ? p_id_type
        AND re.platform_ids->>p_id_type = p_id_value
        AND (p_platform IS NULL OR re.source_platform = p_platform)
    ORDER BY re.created_at DESC;
END;
$$ LANGUAGE plpgsql;

-- Function to get platform ID statistics
CREATE OR REPLACE FUNCTION get_platform_id_stats(p_user_id UUID)
RETURNS TABLE (
    platform TEXT,
    id_type TEXT,
    total_count BIGINT,
    unique_count BIGINT,
    avg_confidence DECIMAL(3,2),
    most_common_ids JSONB
) AS $$
BEGIN
    RETURN QUERY
    WITH platform_id_data AS (
        SELECT 
            re.source_platform as platform,
            jsonb_object_keys(re.platform_ids) as id_type,
            re.platform_ids,
            re.confidence_score
        FROM public.raw_events re
        WHERE re.user_id = p_user_id
            AND re.platform_ids != '{}'
    ),
    aggregated_data AS (
        SELECT 
            platform,
            id_type,
            COUNT(*) as total_count,
            COUNT(DISTINCT platform_ids->>id_type) as unique_count,
            AVG(confidence_score) as avg_confidence,
            jsonb_agg(DISTINCT platform_ids->>id_type) as all_ids
        FROM platform_id_data
        GROUP BY platform, id_type
    )
    SELECT 
        platform,
        id_type,
        total_count,
        unique_count,
        avg_confidence,
        jsonb_agg(DISTINCT all_ids) as most_common_ids
    FROM aggregated_data
    GROUP BY platform, id_type, total_count, unique_count, avg_confidence
    ORDER BY total_count DESC;
END;
$$ LANGUAGE plpgsql;

-- Function to validate platform ID patterns
CREATE OR REPLACE FUNCTION validate_platform_id_pattern(
    p_id_value TEXT,
    p_platform TEXT,
    p_id_type TEXT
)
RETURNS JSONB AS $$
DECLARE
    validation_result JSONB;
BEGIN
    validation_result := jsonb_build_object(
        'is_valid', true,
        'reason', 'Valid ID format',
        'validation_method', 'format_check',
        'warnings', '[]'::jsonb
    );
    
    -- Basic validation
    IF p_id_value IS NULL OR trim(p_id_value) = '' THEN
        validation_result := jsonb_build_object(
            'is_valid', false,
            'reason', 'Empty or null ID value',
            'validation_method', 'basic_check'
        );
        RETURN validation_result;
    END IF;
    
    -- Length validation
    IF length(trim(p_id_value)) < 1 OR length(trim(p_id_value)) > 50 THEN
        validation_result := jsonb_build_object(
            'is_valid', false,
            'reason', format('ID length invalid: %s (must be 1-50 characters)', length(trim(p_id_value))),
            'validation_method', 'length_check'
        );
        RETURN validation_result;
    END IF;
    
    -- Platform-specific validation
    IF p_platform = 'quickbooks' THEN
        IF NOT trim(p_id_value) ~ '^(?:TXN-?|INV-?|VEN-?|CUST-?|BILL-?|PAY-?|ACC-?|CLASS-?|ITEM-?|JE-?)?\d{1,8}$' THEN
            validation_result := jsonb_build_object(
                'is_valid', false,
                'reason', 'Invalid QuickBooks ID format',
                'validation_method', 'platform_specific'
            );
            RETURN validation_result;
        END IF;
    ELSIF p_platform = 'stripe' THEN
        IF NOT trim(p_id_value) ~ '^(ch_|pi_|cus_|in_)[a-zA-Z0-9]{14,24}$' THEN
            validation_result := jsonb_build_object(
                'is_valid', false,
                'reason', 'Invalid Stripe ID format',
                'validation_method', 'platform_specific'
            );
            RETURN validation_result;
        END IF;
    ELSIF p_platform = 'razorpay' THEN
        IF NOT trim(p_id_value) ~ '^(pay_|order_|rfnd_|setl_)[a-zA-Z0-9]{14}$' THEN
            validation_result := jsonb_build_object(
                'is_valid', false,
                'reason', 'Invalid Razorpay ID format',
                'validation_method', 'platform_specific'
            );
            RETURN validation_result;
        END IF;
    END IF;
    
    RETURN validation_result;
END;
$$ LANGUAGE plpgsql;

-- Create function to get enrichment statistics
CREATE OR REPLACE FUNCTION get_enrichment_stats(user_uuid UUID)
RETURNS TABLE(
    total_events BIGINT,
    events_with_currency_conversion BIGINT,
    events_with_vendor_standardization BIGINT,
    events_with_platform_ids BIGINT,
    total_amount_usd DECIMAL(15,2),
    currency_breakdown JSONB,
    vendor_standardization_accuracy DECIMAL(3,2),
    avg_exchange_rate DECIMAL(10,6)
) AS $$
DECLARE
    currency_breakdown_result JSONB;
BEGIN
    -- Get currency breakdown separately to avoid nested aggregates
    SELECT jsonb_object_agg(currency, count) INTO currency_breakdown_result
    FROM (
        SELECT currency, COUNT(*) as count
        FROM public.raw_events
        WHERE user_id = user_uuid AND currency IS NOT NULL
        GROUP BY currency
    ) currency_counts;
    
    RETURN QUERY
    SELECT 
        COUNT(*) as total_events,
        COUNT(*) FILTER (WHERE amount_usd IS NOT NULL AND amount_usd != amount_original) as events_with_currency_conversion,
        COUNT(*) FILTER (WHERE vendor_standard IS NOT NULL AND vendor_standard != vendor_raw) as events_with_vendor_standardization,
        COUNT(*) FILTER (WHERE platform_ids != '{}') as events_with_platform_ids,
        COALESCE(SUM(amount_usd), 0) as total_amount_usd,
        COALESCE(currency_breakdown_result, '{}'::jsonb) as currency_breakdown,
        AVG(vendor_confidence) FILTER (WHERE vendor_confidence IS NOT NULL) as vendor_standardization_accuracy,
        AVG(exchange_rate) FILTER (WHERE exchange_rate IS NOT NULL) as avg_exchange_rate
    FROM public.raw_events
    WHERE user_id = user_uuid;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create function to search by vendor (standardized)
CREATE OR REPLACE FUNCTION search_events_by_vendor(user_uuid UUID, vendor_name TEXT)
RETURNS TABLE(
    id UUID,
    kind TEXT,
    category TEXT,
    subcategory TEXT,
    amount_usd DECIMAL(15,2),
    currency TEXT,
    vendor_standard TEXT,
    platform TEXT,
    standard_description TEXT,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        re.id,
        re.kind,
        re.category,
        re.subcategory,
        re.amount_usd,
        re.currency,
        re.vendor_standard,
        re.source_platform,
        re.standard_description,
        re.created_at
    FROM public.raw_events re
    WHERE re.user_id = user_uuid
    AND re.vendor_standard ILIKE '%' || vendor_name || '%'
    ORDER BY re.created_at DESC;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create function to get currency conversion summary
CREATE OR REPLACE FUNCTION get_currency_summary(user_uuid UUID)
RETURNS TABLE(
    currency TEXT,
    total_original_amount DECIMAL(15,2),
    total_usd_amount DECIMAL(15,2),
    transaction_count BIGINT,
    avg_exchange_rate DECIMAL(10,6)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        re.currency,
        SUM(re.amount_original) as total_original_amount,
        SUM(re.amount_usd) as total_usd_amount,
        COUNT(*) as transaction_count,
        AVG(re.exchange_rate) as avg_exchange_rate
    FROM public.raw_events re
    WHERE re.user_id = user_uuid
    AND re.currency IS NOT NULL
    GROUP BY re.currency
    ORDER BY total_usd_amount DESC;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
