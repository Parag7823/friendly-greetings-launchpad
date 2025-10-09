-- Migration: Add L3 core KPI views (cash positions, burn rate)
-- Date: 2025-10-09

-- View: l3_cash_positions
-- Primary source: latest statement-like events with parsed closing/available/statement balance
-- Fallback: 90-day net cash flow (revenue - expense) by inferred account reference
CREATE OR REPLACE VIEW public.l3_cash_positions AS
WITH base AS (
  SELECT * FROM public.l3_events_ready
),
stmt AS (
  SELECT
    re.user_id,
    COALESCE(
      re.payload->>'account_number',
      re.payload->>'account_id',
      re.classification_metadata->>'account_id',
      re.platform_ids->>'account_id',
      re.source_platform,
      'unknown'
    ) AS account_ref,
    re.id,
    re.created_at,
    -- Parse numeric balance from common keys
    CAST(
      NULLIF(
        regexp_replace(
          COALESCE(
            re.payload->>'closing_balance',
            re.payload->>'available_balance',
            re.payload->>'statement_balance'
          ), '[^0-9\.-]', '', 'g'
        ), ''
      ) AS NUMERIC
    ) AS parsed_balance
  FROM base re
  WHERE (
    re.classification_metadata->>'document_type' ILIKE '%statement%'
    OR re.kind = 'bank_statement'
  )
),
stmt_latest AS (
  SELECT DISTINCT ON (user_id, account_ref)
    user_id, account_ref, id, created_at, parsed_balance
  FROM stmt
  WHERE parsed_balance IS NOT NULL
  ORDER BY user_id, account_ref, created_at DESC, id
),
flows AS (
  SELECT
    re.user_id,
    COALESCE(
      re.payload->>'account_number',
      re.payload->>'account_id',
      re.classification_metadata->>'account_id',
      re.platform_ids->>'account_id',
      re.source_platform,
      'unknown'
    ) AS account_ref,
    SUM(CASE WHEN re.category = 'revenue' THEN COALESCE(re.amount_usd, 0) ELSE 0 END)
      - SUM(CASE WHEN re.category = 'expense' THEN COALESCE(re.amount_usd, 0) ELSE 0 END) AS net_flow_usd,
    array_agg(re.id) AS evidence_ids,
    MAX(re.created_at) AS as_of
  FROM base re
  WHERE re.created_at >= (now() - INTERVAL '90 days')
  GROUP BY re.user_id, account_ref
)
SELECT
  COALESCE(sl.user_id, f.user_id) AS user_id,
  COALESCE(sl.account_ref, f.account_ref) AS account_ref,
  sl.parsed_balance AS balance_from_statements_usd,
  f.net_flow_usd AS fallback_balance_usd,
  COALESCE(sl.parsed_balance, f.net_flow_usd) AS balance_usd,
  CASE WHEN sl.id IS NOT NULL THEN ARRAY[sl.id] ELSE f.evidence_ids END AS evidence_ids,
  COALESCE(sl.created_at, f.as_of, now()) AS as_of
FROM stmt_latest sl
FULL OUTER JOIN flows f
  ON sl.user_id = f.user_id AND sl.account_ref = f.account_ref;

GRANT SELECT ON public.l3_cash_positions TO authenticated;
COMMENT ON VIEW public.l3_cash_positions IS 'Cash on hand per account_ref; prefers latest statement balance, falls back to 90-day net flows. Includes evidence_ids.';

-- View: l3_burn_rate (30/60/90-day windows)
CREATE OR REPLACE VIEW public.l3_burn_rate AS
WITH base AS (
  SELECT * FROM public.l3_events_ready
),
windowed AS (
  SELECT re.user_id, 30::INT AS window_days, re.id, re.subcategory, re.amount_usd, re.created_at
  FROM base re
  WHERE re.category = 'expense' AND re.created_at >= (now() - INTERVAL '30 days')
  UNION ALL
  SELECT re.user_id, 60::INT AS window_days, re.id, re.subcategory, re.amount_usd, re.created_at
  FROM base re
  WHERE re.category = 'expense' AND re.created_at >= (now() - INTERVAL '60 days')
  UNION ALL
  SELECT re.user_id, 90::INT AS window_days, re.id, re.subcategory, re.amount_usd, re.created_at
  FROM base re
  WHERE re.category = 'expense' AND re.created_at >= (now() - INTERVAL '90 days')
),
rolled AS (
  SELECT
    user_id,
    window_days,
    SUM(CASE WHEN (subcategory IS NULL OR subcategory NOT IN ('capital_purchase','asset_purchase','capex'))
             THEN COALESCE(amount_usd, 0) ELSE 0 END) AS opex_usd,
    array_agg(id) AS evidence_ids
  FROM windowed
  GROUP BY user_id, window_days
)
SELECT
  user_id,
  window_days,
  opex_usd,
  ROUND((CASE WHEN window_days > 0 THEN (opex_usd / window_days) * 30.0 ELSE NULL END)::NUMERIC, 2) AS burn_rate_usd_per_month,
  evidence_ids
FROM rolled;

GRANT SELECT ON public.l3_burn_rate TO authenticated;
COMMENT ON VIEW public.l3_burn_rate IS 'Operating expense over 30/60/90 days with monthly burn rate projection. Excludes common capex subcategories. Includes evidence_ids.';
