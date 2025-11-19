-- Migration: Add transaction_id to resolution_log for transaction tracking
-- Date: 2025-11-19
-- Purpose: Enable transaction-based rollback and tracking for entity resolution operations

-- Add transaction_id column
ALTER TABLE IF EXISTS public.resolution_log
ADD COLUMN IF NOT EXISTS transaction_id UUID REFERENCES public.processing_transactions(id) ON DELETE SET NULL;

-- Create index for transaction tracking
CREATE INDEX IF NOT EXISTS idx_resolution_log_transaction_id ON public.resolution_log(transaction_id) WHERE transaction_id IS NOT NULL;

-- Composite index for user and transaction queries
CREATE INDEX IF NOT EXISTS idx_resolution_log_user_transaction ON public.resolution_log(user_id, transaction_id) WHERE transaction_id IS NOT NULL;

-- Add comment for documentation
COMMENT ON COLUMN public.resolution_log.transaction_id IS 'Links to processing_transactions for atomic operations and rollback capability';
