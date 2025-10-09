-- Migration: Add L3 views for revenue velocity and AR/AP open balances
-- Date: 2025-10-09

-- View: l3_revenue_velocity (7/30-day inflow averages + simple trend)
CREATE OR REPLACE VIEW public.l3_revenue_velocity AS
WITH base AS (
  SELECT * FROM public.l3_events_ready
),
cur7 AS (
  SELECT user_id,
         7::INT AS window_days,
         SUM(COALESCE(amount_usd, 0)) AS total_inflow_usd,
         array_agg(id) AS evidence_ids
  FROM base
  WHERE category = 'revenue' AND created_at >= (now() - INTERVAL '7 days')
  GROUP BY user_id
),
prev7 AS (
  SELECT user_id,
         SUM(COALESCE(amount_usd, 0)) AS total_inflow_usd_prev
  FROM base
  WHERE category = 'revenue'
    AND created_at >= (now() - INTERVAL '14 days')
    AND created_at <  (now() - INTERVAL '7 days')
  GROUP BY user_id
),
win30 AS (
  SELECT user_id,
         30::INT AS window_days,
         SUM(COALESCE(amount_usd, 0)) AS total_inflow_usd,
         array_agg(id) AS evidence_ids
  FROM base
  WHERE category = 'revenue' AND created_at >= (now() - INTERVAL '30 days')
  GROUP BY user_id
)
SELECT
  -- 7-day row
  c.user_id,
  7::INT AS window_days,
  c.total_inflow_usd,
  ROUND((c.total_inflow_usd / 7.0)::NUMERIC, 2) AS avg_daily_inflow_usd,
  CASE 
    WHEN p.total_inflow_usd_prev IS NULL OR p.total_inflow_usd_prev = 0 THEN NULL
    ELSE ROUND(((c.total_inflow_usd - p.total_inflow_usd_prev) / p.total_inflow_usd_prev * 100.0)::NUMERIC, 2)
  END AS trend_percent_vs_prev7,
  CASE 
    WHEN p.total_inflow_usd_prev IS NULL OR p.total_inflow_usd_prev = 0 THEN 'unknown'
    WHEN (c.total_inflow_usd - p.total_inflow_usd_prev) / p.total_inflow_usd_prev > 0.05 THEN 'up'
    WHEN (c.total_inflow_usd - p.total_inflow_usd_prev) / p.total_inflow_usd_prev < -0.05 THEN 'down'
    ELSE 'flat'
  END AS trend_label,
  c.evidence_ids,
  now() AS as_of
FROM cur7 c
LEFT JOIN prev7 p USING (user_id)
UNION ALL
SELECT
  w.user_id,
  30::INT AS window_days,
  w.total_inflow_usd,
  ROUND((w.total_inflow_usd / 30.0)::NUMERIC, 2) AS avg_daily_inflow_usd,
  NULL::NUMERIC AS trend_percent_vs_prev7,
  NULL::TEXT AS trend_label,
  w.evidence_ids,
  now() AS as_of
FROM win30 w;

GRANT SELECT ON public.l3_revenue_velocity TO authenticated;
COMMENT ON VIEW public.l3_revenue_velocity IS 'Revenue inflow velocity over 7/30 days with simple 7-day trend; includes evidence_ids.';


-- View: l3_ar_open (open Accounts Receivable invoices)
CREATE OR REPLACE VIEW public.l3_ar_open AS
WITH base AS (
  SELECT * FROM public.l3_events_ready
),
inv AS (
  SELECT
    re.user_id,
    re.id AS invoice_id,
    COALESCE(
      (re.entities->'customers'->>0),
      re.payload->>'customer',
      re.payload->>'client',
      'unknown'
    ) AS customer,
    COALESCE((re.payload->>'invoice_date')::date, re.created_at::date) AS issued_date,
    (re.payload->>'due_date')::date AS due_date,
    COALESCE(
      re.amount_usd,
      CAST(NULLIF(regexp_replace(re.payload->>'total_amount', '[^0-9\.-]', '', 'g'), '') AS NUMERIC)
    ) AS gross_amount_usd
  FROM base re
  WHERE (
    re.classification_metadata->>'document_type' ILIKE '%invoice%'
    OR (re.kind ILIKE '%invoice%' OR re.subcategory ILIKE '%invoice%')
  )
),
payments AS (
  SELECT
    ri.user_id,
    CASE WHEN ri.source_event_id = i.invoice_id THEN ri.target_event_id ELSE ri.source_event_id END AS payment_event_id
  FROM public.relationship_instances ri
  JOIN inv i ON (ri.source_event_id = i.invoice_id OR ri.target_event_id = i.invoice_id)
),
pay_sums AS (
  SELECT
    i.user_id,
    i.invoice_id,
    COALESCE(SUM(CASE WHEN p_re.category = 'expense' THEN 0 ELSE COALESCE(p_re.amount_usd,0) END), 0) AS paid_amount_usd,
    array_agg(p_re.id) AS payment_evidence
  FROM inv i
  LEFT JOIN payments p ON p.user_id = i.user_id
    AND (p.payment_event_id IS NOT NULL)
  LEFT JOIN public.l3_events_ready p_re ON p_re.id = p.payment_event_id
  GROUP BY i.user_id, i.invoice_id
)
SELECT
  i.user_id,
  i.invoice_id,
  i.customer,
  i.issued_date,
  i.due_date,
  GREATEST(COALESCE(i.gross_amount_usd,0) - COALESCE(ps.paid_amount_usd,0), 0) AS open_amount_usd,
  (now()::date - COALESCE(i.due_date, i.issued_date)) AS days_outstanding,
  ARRAY[i.invoice_id] || COALESCE(ps.payment_evidence, '{}') AS evidence_ids
FROM inv i
LEFT JOIN pay_sums ps ON ps.user_id = i.user_id AND ps.invoice_id = i.invoice_id
WHERE COALESCE(i.gross_amount_usd,0) > COALESCE(ps.paid_amount_usd,0);

GRANT SELECT ON public.l3_ar_open TO authenticated;
COMMENT ON VIEW public.l3_ar_open IS 'Open AR: invoices minus linked payments via relationship_instances; includes evidence_ids.';


-- View: l3_ap_open (open Accounts Payable bills)
CREATE OR REPLACE VIEW public.l3_ap_open AS
WITH base AS (
  SELECT * FROM public.l3_events_ready
),
bills AS (
  SELECT
    re.user_id,
    re.id AS bill_id,
    COALESCE(
      (re.entities->'vendors'->>0),
      re.vendor_standard,
      re.payload->>'vendor',
      'unknown'
    ) AS vendor,
    COALESCE((re.payload->>'bill_date')::date, re.created_at::date) AS issued_date,
    (re.payload->>'due_date')::date AS due_date,
    COALESCE(
      re.amount_usd,
      CAST(NULLIF(regexp_replace(re.payload->>'total_amount', '[^0-9\.-]', '', 'g'), '') AS NUMERIC)
    ) AS gross_amount_usd
  FROM base re
  WHERE (
    re.classification_metadata->>'document_type' ILIKE '%bill%'
    OR (re.kind ILIKE '%bill%' OR re.subcategory ILIKE '%bill%')
    OR (re.category = 'expense' AND (re.subcategory IS NULL OR re.subcategory NOT IN ('refund','reversal')))
  )
),
payouts AS (
  SELECT
    ri.user_id,
    CASE WHEN ri.source_event_id = b.bill_id THEN ri.target_event_id ELSE ri.source_event_id END AS payout_event_id
  FROM public.relationship_instances ri
  JOIN bills b ON (ri.source_event_id = b.bill_id OR ri.target_event_id = b.bill_id)
),
payout_sums AS (
  SELECT
    b.user_id,
    b.bill_id,
    COALESCE(SUM(CASE WHEN p_re.category = 'expense' THEN COALESCE(p_re.amount_usd,0) ELSE 0 END), 0) AS paid_amount_usd,
    array_agg(p_re.id) AS payout_evidence
  FROM bills b
  LEFT JOIN payouts p ON p.user_id = b.user_id AND (p.payout_event_id IS NOT NULL)
  LEFT JOIN public.l3_events_ready p_re ON p_re.id = p.payout_event_id
  GROUP BY b.user_id, b.bill_id
)
SELECT
  b.user_id,
  b.bill_id,
  b.vendor,
  b.issued_date,
  b.due_date,
  GREATEST(COALESCE(b.gross_amount_usd,0) - COALESCE(po.paid_amount_usd,0), 0) AS open_amount_usd,
  (now()::date - COALESCE(b.due_date, b.issued_date)) AS days_outstanding,
  ARRAY[b.bill_id] || COALESCE(po.payout_evidence, '{}') AS evidence_ids
FROM bills b
LEFT JOIN payout_sums po ON po.user_id = b.user_id AND po.bill_id = b.bill_id
WHERE COALESCE(b.gross_amount_usd,0) > COALESCE(po.paid_amount_usd,0);

GRANT SELECT ON public.l3_ap_open TO authenticated;
COMMENT ON VIEW public.l3_ap_open IS 'Open AP: bills/expenses minus linked payouts via relationship_instances; includes evidence_ids.';
