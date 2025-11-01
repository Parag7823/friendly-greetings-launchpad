-- Migration: Add transaction tracking to entity_matches
-- Date: 2025-11-01 08:30:00
-- Purpose: Align entity_matches schema with backend expectations (transaction_id usage)

ALTER TABLE public.entity_matches
    ADD COLUMN IF NOT EXISTS transaction_id UUID;

-- Optional: maintain referential traceability to processing_transactions when available
ALTER TABLE public.entity_matches
    ADD CONSTRAINT entity_matches_transaction_fk
    FOREIGN KEY (transaction_id)
    REFERENCES public.processing_transactions(id)
    ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_entity_matches_transaction_id
    ON public.entity_matches(transaction_id);
