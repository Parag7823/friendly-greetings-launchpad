-- Migration: Add transaction_id to cross_platform_relationships for proper cleanup
-- Date: 2025-10-16
-- FIX #9: Enable proper cleanup of cross-platform relationships on transaction rollback

-- Add transaction_id column to cross_platform_relationships
ALTER TABLE public.cross_platform_relationships 
ADD COLUMN IF NOT EXISTS transaction_id UUID REFERENCES public.processing_transactions(id) ON DELETE CASCADE;

-- Create index for efficient cleanup queries
CREATE INDEX IF NOT EXISTS idx_cross_platform_relationships_transaction_id 
ON public.cross_platform_relationships(transaction_id);

-- Add comment for documentation
COMMENT ON COLUMN public.cross_platform_relationships.transaction_id IS 'FIX #9: Links cross-platform relationships to transactions for atomic rollback and cleanup';
