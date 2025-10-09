-- Migration: Add Layer 3 gating view (l3_events_ready)
-- Date: 2025-10-09

-- Transaction-gated view: only includes processed events that are either
-- outside of any transaction or belong to a committed processing transaction.
CREATE OR REPLACE VIEW public.l3_events_ready AS
SELECT re.*
FROM public.raw_events re
LEFT JOIN public.processing_transactions pt
  ON re.transaction_id = pt.id
WHERE re.status = 'processed'
  AND (
    re.transaction_id IS NULL
    OR pt.status = 'committed'
  );

-- Helper function: paginated access to gated events for a user
-- Returns the raw_events row type for ease of consumption
CREATE OR REPLACE FUNCTION public.get_l3_events_ready(
    p_user_id UUID,
    p_limit INTEGER DEFAULT 100,
    p_offset INTEGER DEFAULT 0,
    p_kind TEXT DEFAULT NULL,
    p_source_platform TEXT DEFAULT NULL
)
RETURNS SETOF public.raw_events
LANGUAGE sql
STABLE
AS $$
    SELECT re.*
    FROM public.l3_events_ready re
    WHERE re.user_id = p_user_id
      AND (p_kind IS NULL OR re.kind = p_kind)
      AND (p_source_platform IS NULL OR re.source_platform = p_source_platform)
    ORDER BY re.created_at DESC
    LIMIT GREATEST(p_limit, 0)
    OFFSET GREATEST(p_offset, 0)
$$;

-- Permissions: allow authenticated role to read the view and call the function
GRANT SELECT ON public.l3_events_ready TO authenticated;
GRANT EXECUTE ON FUNCTION public.get_l3_events_ready(UUID, INTEGER, INTEGER, TEXT, TEXT) TO authenticated;

-- Documentation
COMMENT ON VIEW public.l3_events_ready IS 'Transaction-gated view of processed raw_events; only committed batches are included';
COMMENT ON FUNCTION public.get_l3_events_ready(UUID, INTEGER, INTEGER, TEXT, TEXT) IS 'Paginated access to l3_events_ready for a given user';
