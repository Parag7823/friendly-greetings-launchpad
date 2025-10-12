-- Add Performance Indexes for Enrichment Fields
-- Created: 2025-10-12
-- Purpose: Optimize queries on enriched data fields

-- ============================================================================
-- VENDOR ANALYSIS INDEXES
-- ============================================================================

-- Index for vendor standardization queries
CREATE INDEX IF NOT EXISTS idx_raw_events_vendor_standard 
ON public.raw_events(user_id, vendor_standard) 
WHERE vendor_standard IS NOT NULL;

-- Index for vendor confidence filtering
CREATE INDEX IF NOT EXISTS idx_raw_events_vendor_confidence 
ON public.raw_events(user_id, vendor_confidence DESC) 
WHERE vendor_confidence IS NOT NULL;

-- ============================================================================
-- AMOUNT & CURRENCY INDEXES
-- ============================================================================

-- Index for amount range queries (most common query pattern)
CREATE INDEX IF NOT EXISTS idx_raw_events_amount_usd 
ON public.raw_events(user_id, amount_usd DESC) 
WHERE amount_usd IS NOT NULL;

-- Index for currency-specific queries
CREATE INDEX IF NOT EXISTS idx_raw_events_currency 
ON public.raw_events(user_id, currency) 
WHERE currency IS NOT NULL;

-- Composite index for platform + currency analysis
CREATE INDEX IF NOT EXISTS idx_raw_events_platform_currency 
ON public.raw_events(source_platform, currency, amount_usd DESC);

-- ============================================================================
-- PLATFORM IDS JSONB INDEXES
-- ============================================================================

-- GIN index for platform_ids JSONB field (enables fast JSON queries)
CREATE INDEX IF NOT EXISTS idx_raw_events_platform_ids_gin 
ON public.raw_events USING GIN (platform_ids);

-- ============================================================================
-- CLASSIFICATION METADATA INDEXES
-- ============================================================================

-- GIN index for classification_metadata JSONB
CREATE INDEX IF NOT EXISTS idx_raw_events_classification_metadata_gin 
ON public.raw_events USING GIN (classification_metadata);

-- GIN index for entities JSONB
CREATE INDEX IF NOT EXISTS idx_raw_events_entities_gin 
ON public.raw_events USING GIN (entities);

-- GIN index for relationships JSONB
CREATE INDEX IF NOT EXISTS idx_raw_events_relationships_gin 
ON public.raw_events USING GIN (relationships);

-- ============================================================================
-- COMPOSITE INDEXES FOR COMMON QUERY PATTERNS
-- ============================================================================

-- Index for vendor + amount analysis
CREATE INDEX IF NOT EXISTS idx_raw_events_vendor_amount 
ON public.raw_events(user_id, vendor_standard, amount_usd DESC) 
WHERE vendor_standard IS NOT NULL AND amount_usd IS NOT NULL;

-- Index for platform + vendor analysis
CREATE INDEX IF NOT EXISTS idx_raw_events_platform_vendor 
ON public.raw_events(user_id, source_platform, vendor_standard) 
WHERE source_platform IS NOT NULL AND vendor_standard IS NOT NULL;

-- Index for time-series amount queries
CREATE INDEX IF NOT EXISTS idx_raw_events_time_amount 
ON public.raw_events(user_id, ingest_ts DESC, amount_usd DESC);

-- Index for category-based filtering
CREATE INDEX IF NOT EXISTS idx_raw_events_category 
ON public.raw_events(user_id, category, subcategory) 
WHERE category IS NOT NULL;

-- ============================================================================
-- PARTIAL INDEXES FOR STATUS FILTERING
-- ============================================================================

-- Index for processed events only (most common filter)
CREATE INDEX IF NOT EXISTS idx_raw_events_processed 
ON public.raw_events(user_id, ingest_ts DESC) 
WHERE status = 'processed';

-- Index for failed events (for error analysis)
CREATE INDEX IF NOT EXISTS idx_raw_events_failed 
ON public.raw_events(user_id, ingest_ts DESC) 
WHERE status = 'failed';

-- ============================================================================
-- COVERING INDEXES FOR DASHBOARD QUERIES
-- ============================================================================

-- Covering index for vendor summary (includes all needed columns)
CREATE INDEX IF NOT EXISTS idx_raw_events_vendor_summary 
ON public.raw_events(user_id, vendor_standard, amount_usd, currency, ingest_ts) 
WHERE vendor_standard IS NOT NULL AND status = 'processed';

-- Covering index for platform summary
CREATE INDEX IF NOT EXISTS idx_raw_events_platform_summary 
ON public.raw_events(user_id, source_platform, amount_usd, currency, ingest_ts) 
WHERE source_platform IS NOT NULL AND status = 'processed';

-- ============================================================================
-- ANALYZE TABLES FOR QUERY PLANNER
-- ============================================================================

-- Update statistics for query planner
ANALYZE public.raw_events;

-- ============================================================================
-- VERIFICATION QUERY
-- ============================================================================

-- Verify indexes were created
DO $$
DECLARE
    index_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO index_count
    FROM pg_indexes
    WHERE schemaname = 'public'
    AND tablename = 'raw_events'
    AND indexname LIKE 'idx_raw_events_%';
    
    RAISE NOTICE '✅ Created % performance indexes on raw_events table', index_count;
END $$;
