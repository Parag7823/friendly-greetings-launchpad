-- Add data enrichment fields to raw_events table
-- This migration adds fields for currency normalization, vendor standardization, and platform ID extraction

-- Add enrichment fields to raw_events table
ALTER TABLE public.raw_events 
ADD COLUMN amount_original DECIMAL(15,2),
ADD COLUMN amount_usd DECIMAL(15,2),
ADD COLUMN currency TEXT DEFAULT 'USD',
ADD COLUMN exchange_rate DECIMAL(10,6),
ADD COLUMN exchange_date DATE,
ADD COLUMN vendor_raw TEXT,
ADD COLUMN vendor_standard TEXT,
ADD COLUMN vendor_confidence DECIMAL(3,2),
ADD COLUMN vendor_cleaning_method TEXT,
ADD COLUMN platform_ids JSONB DEFAULT '{}',
ADD COLUMN standard_description TEXT,
ADD COLUMN ingested_on TIMESTAMP WITH TIME ZONE DEFAULT now();

-- Add indexes for new fields
CREATE INDEX idx_raw_events_amount_usd ON public.raw_events(amount_usd);
CREATE INDEX idx_raw_events_currency ON public.raw_events(currency);
CREATE INDEX idx_raw_events_vendor_standard ON public.raw_events(vendor_standard);
CREATE INDEX idx_raw_events_ingested_on ON public.raw_events(ingested_on);

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