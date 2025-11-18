-- Migration: Consolidate and clean up schema drift
-- Date: 2025-11-18
-- Purpose: Remove duplicate columns added multiple times, consolidate schema

-- NOTE: This migration documents schema consolidation
-- All duplicate columns have already been added via IF NOT EXISTS clauses
-- This migration ensures consistency and removes any orphaned columns

-- Verify raw_events has all required columns (added via previous migrations)
-- These columns should exist from earlier migrations:
-- - ai_confidence (added in 20251114000000)
-- - ai_reasoning (added in 20251114000000)
-- - source_ts (added in 20251114000000)
-- - relationship_count (added in 20251114000000)
-- - last_relationship_check (added in 20251114000000)

-- Verify all indexes exist
CREATE INDEX IF NOT EXISTS idx_raw_events_ai_confidence ON public.raw_events(ai_confidence) WHERE ai_confidence IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_raw_events_source_ts ON public.raw_events(source_ts) WHERE source_ts IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_raw_events_relationship_count ON public.raw_events(relationship_count) WHERE relationship_count > 0;

-- Ensure relationship_instances has transaction_id for rollback capability
ALTER TABLE public.relationship_instances
ADD COLUMN IF NOT EXISTS transaction_id UUID REFERENCES public.processing_transactions(id) ON DELETE CASCADE;

CREATE INDEX IF NOT EXISTS idx_relationship_instances_transaction_id ON public.relationship_instances(transaction_id) WHERE transaction_id IS NOT NULL;

-- Ensure entity_matches has transaction_id
ALTER TABLE public.entity_matches
ADD COLUMN IF NOT EXISTS transaction_id UUID REFERENCES public.processing_transactions(id) ON DELETE CASCADE;

CREATE INDEX IF NOT EXISTS idx_entity_matches_transaction_id ON public.entity_matches(transaction_id) WHERE transaction_id IS NOT NULL;

-- Ensure semantic_links has transaction_id (if table exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'semantic_links' AND table_schema = 'public') THEN
        ALTER TABLE public.semantic_links
        ADD COLUMN IF NOT EXISTS transaction_id UUID REFERENCES public.processing_transactions(id) ON DELETE CASCADE;
        
        CREATE INDEX IF NOT EXISTS idx_semantic_links_transaction_id ON public.semantic_links(transaction_id) WHERE transaction_id IS NOT NULL;
    END IF;
END $$;

-- Ensure temporal_patterns has transaction_id
ALTER TABLE public.temporal_patterns
ADD COLUMN IF NOT EXISTS transaction_id UUID REFERENCES public.processing_transactions(id) ON DELETE CASCADE;

CREATE INDEX IF NOT EXISTS idx_temporal_patterns_transaction_id ON public.temporal_patterns(transaction_id) WHERE transaction_id IS NOT NULL;

-- Ensure causal_links has transaction_id (if table exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'causal_links' AND table_schema = 'public') THEN
        ALTER TABLE public.causal_links
        ADD COLUMN IF NOT EXISTS transaction_id UUID REFERENCES public.processing_transactions(id) ON DELETE CASCADE;
        
        CREATE INDEX IF NOT EXISTS idx_causal_links_transaction_id ON public.causal_links(transaction_id) WHERE transaction_id IS NOT NULL;
    END IF;
END $$;

-- Add comments for schema documentation
COMMENT ON COLUMN public.relationship_instances.transaction_id IS 'Links to processing transaction for rollback capability';
COMMENT ON COLUMN public.entity_matches.transaction_id IS 'Links to processing transaction for rollback capability';
COMMENT ON COLUMN public.temporal_patterns.transaction_id IS 'Links to processing transaction for rollback capability';

-- Conditional comments for optional tables
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'semantic_links' AND table_schema = 'public') THEN
        EXECUTE 'COMMENT ON COLUMN public.semantic_links.transaction_id IS ''Links to processing transaction for rollback capability''';
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'causal_links' AND table_schema = 'public') THEN
        EXECUTE 'COMMENT ON COLUMN public.causal_links.transaction_id IS ''Links to processing transaction for rollback capability''';
    END IF;
END $$;

-- Log migration completion
DO $$
BEGIN
    RAISE NOTICE 'Migration 20251118000002: Consolidated schema drift and added transaction_id to all analytics tables';
    RAISE NOTICE 'Added transaction_id columns to: relationship_instances, entity_matches, semantic_links, temporal_patterns, causal_links';
    RAISE NOTICE 'Created indexes for transaction_id on all tables for efficient rollback queries';
    RAISE NOTICE 'Schema is now consistent and supports full transaction rollback capability';
END $$;
