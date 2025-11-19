-- Migration: Expand raw_events schema with missing financial and metadata columns
-- Date: 2025-11-19
-- Purpose: Add all columns needed for comprehensive financial event tracking

-- Add financial amount columns
ALTER TABLE IF EXISTS public.raw_events
ADD COLUMN IF NOT EXISTS category TEXT,
ADD COLUMN IF NOT EXISTS subcategory TEXT,
ADD COLUMN IF NOT EXISTS amount_original DECIMAL(15,2),
ADD COLUMN IF NOT EXISTS amount_usd DECIMAL(15,2),
ADD COLUMN IF NOT EXISTS currency VARCHAR(3),
ADD COLUMN IF NOT EXISTS exchange_rate DECIMAL(10,6),
ADD COLUMN IF NOT EXISTS exchange_date DATE;

-- Add vendor/entity columns
ALTER TABLE IF EXISTS public.raw_events
ADD COLUMN IF NOT EXISTS vendor_raw TEXT,
ADD COLUMN IF NOT EXISTS vendor_standard TEXT,
ADD COLUMN IF NOT EXISTS vendor_confidence DECIMAL(3,2) CHECK (vendor_confidence >= 0 AND vendor_confidence <= 1),
ADD COLUMN IF NOT EXISTS vendor_cleaning_method VARCHAR(100),
ADD COLUMN IF NOT EXISTS vendor_canonical_id UUID,
ADD COLUMN IF NOT EXISTS vendor_verified BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS vendor_alternatives TEXT[];

-- Add entity and relationship tracking
ALTER TABLE IF EXISTS public.raw_events
ADD COLUMN IF NOT EXISTS entities JSONB DEFAULT '{}',
ADD COLUMN IF NOT EXISTS relationships JSONB DEFAULT '{}',
ADD COLUMN IF NOT EXISTS platform_ids TEXT[];

-- Add classification and standardization
ALTER TABLE IF EXISTS public.raw_events
ADD COLUMN IF NOT EXISTS standard_description TEXT,
ADD COLUMN IF NOT EXISTS document_type TEXT,
ADD COLUMN IF NOT EXISTS document_confidence DECIMAL(3,2) CHECK (document_confidence >= 0 AND document_confidence <= 1);

-- Add transaction metadata
ALTER TABLE IF EXISTS public.raw_events
ADD COLUMN IF NOT EXISTS transaction_type VARCHAR(50),
ADD COLUMN IF NOT EXISTS amount_direction VARCHAR(10) CHECK (amount_direction IN ('debit', 'credit', 'neutral')),
ADD COLUMN IF NOT EXISTS amount_signed_usd DECIMAL(15,2),
ADD COLUMN IF NOT EXISTS affects_cash BOOLEAN DEFAULT TRUE,
ADD COLUMN IF NOT EXISTS source_ts TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS ingested_ts TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS processed_ts TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS transaction_date DATE,
ADD COLUMN IF NOT EXISTS exchange_rate_date DATE;

-- Add validation and quality flags
ALTER TABLE IF EXISTS public.raw_events
ADD COLUMN IF NOT EXISTS validation_flags JSONB DEFAULT '{}',
ADD COLUMN IF NOT EXISTS is_valid BOOLEAN DEFAULT TRUE,
ADD COLUMN IF NOT EXISTS overall_confidence DECIMAL(3,2) CHECK (overall_confidence >= 0 AND overall_confidence <= 1),
ADD COLUMN IF NOT EXISTS requires_review BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS review_reason TEXT,
ADD COLUMN IF NOT EXISTS review_priority VARCHAR(20) CHECK (review_priority IN ('low', 'medium', 'high', 'critical')),
ADD COLUMN IF NOT EXISTS accuracy_enhanced BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS accuracy_version INTEGER DEFAULT 1;

-- Add audit and lineage columns
ALTER TABLE IF EXISTS public.raw_events
ADD COLUMN IF NOT EXISTS row_hash TEXT,
ADD COLUMN IF NOT EXISTS lineage_path TEXT,
ADD COLUMN IF NOT EXISTS created_by UUID REFERENCES auth.users(id) ON DELETE SET NULL,
ADD COLUMN IF NOT EXISTS modified_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS modified_by UUID REFERENCES auth.users(id) ON DELETE SET NULL;

-- Add AI enrichment columns
ALTER TABLE IF EXISTS public.raw_events
ADD COLUMN IF NOT EXISTS ai_confidence DECIMAL(3,2) CHECK (ai_confidence >= 0 AND ai_confidence <= 1),
ADD COLUMN IF NOT EXISTS ai_reasoning JSONB DEFAULT '{}';

-- Add relationship tracking columns (already in schema but ensure they exist)
ALTER TABLE IF EXISTS public.raw_events
ADD COLUMN IF NOT EXISTS ingested_on DATE,
ADD COLUMN IF NOT EXISTS relationship_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS last_relationship_check TIMESTAMP WITH TIME ZONE;

-- Create indexes for new columns
CREATE INDEX IF NOT EXISTS idx_raw_events_category ON public.raw_events(category) WHERE category IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_raw_events_vendor_standard ON public.raw_events(vendor_standard) WHERE vendor_standard IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_raw_events_vendor_canonical_id ON public.raw_events(vendor_canonical_id) WHERE vendor_canonical_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_raw_events_amount_usd ON public.raw_events(amount_usd) WHERE amount_usd IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_raw_events_document_type ON public.raw_events(document_type) WHERE document_type IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_raw_events_transaction_type ON public.raw_events(transaction_type) WHERE transaction_type IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_raw_events_transaction_date ON public.raw_events(transaction_date) WHERE transaction_date IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_raw_events_requires_review ON public.raw_events(requires_review) WHERE requires_review = TRUE;
CREATE INDEX IF NOT EXISTS idx_raw_events_is_valid ON public.raw_events(is_valid) WHERE is_valid = FALSE;
CREATE INDEX IF NOT EXISTS idx_raw_events_row_hash ON public.raw_events(row_hash) WHERE row_hash IS NOT NULL;

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_raw_events_vendor_date ON public.raw_events(vendor_standard, transaction_date) WHERE vendor_standard IS NOT NULL AND transaction_date IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_raw_events_category_amount ON public.raw_events(category, amount_usd) WHERE category IS NOT NULL AND amount_usd IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_raw_events_document_review ON public.raw_events(document_type, requires_review) WHERE document_type IS NOT NULL;

-- Add comments for documentation
COMMENT ON COLUMN public.raw_events.category IS 'Financial transaction category (e.g., payroll, revenue, expense)';
COMMENT ON COLUMN public.raw_events.amount_usd IS 'Amount in USD (converted from original currency)';
COMMENT ON COLUMN public.raw_events.vendor_standard IS 'Standardized vendor name after cleaning';
COMMENT ON COLUMN public.raw_events.entities IS 'Extracted entities (employees, vendors, customers, projects)';
COMMENT ON COLUMN public.raw_events.relationships IS 'Detected relationships with other events';
COMMENT ON COLUMN public.raw_events.document_type IS 'Type of financial document (invoice, receipt, payroll, etc.)';
COMMENT ON COLUMN public.raw_events.ai_confidence IS 'Confidence score from AI enrichment (0.0-1.0)';
COMMENT ON COLUMN public.raw_events.ai_reasoning IS 'AI reasoning and explanation for enrichment decisions';
