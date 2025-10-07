-- Backfill Universal Finance Schema from existing legacy tables
-- Date: 2025-10-07 19:30:00 +05:30

BEGIN;

-- 1) Backfill universal_records from raw_events (idempotent)
INSERT INTO public.universal_records (
  user_id, record_type, source_event_id, universal, created_at, updated_at
)
SELECT
  e.user_id,
  COALESCE(e.classification_metadata->>'category', e.kind, 'transaction') AS record_type,
  e.id AS source_event_id,
  jsonb_strip_nulls(
    jsonb_build_object(
      'user_id', e.user_id::text,
      'record_type', COALESCE(e.classification_metadata->>'category', e.kind, 'transaction'),
      'amount', jsonb_build_object(
        'value', NULLIF(e.payload->>'amount_usd','')::numeric,
        'currency', NULLIF(e.payload->>'currency','')
      ),
      'dates', jsonb_build_object(
        'issued', NULLIF(e.payload->>'date',''),
        'posted', NULLIF(e.created_at::text,''),
        'due', NULLIF(e.payload->>'due_date','')
      ),
      'identifiers', jsonb_build_object(
        'invoice_id', NULLIF(e.payload->>'invoice_id',''),
        'payment_id', NULLIF(e.payload->>'payment_id',''),
        'external_id', NULLIF(e.payload->>'id',''),
        'reference', NULLIF(e.payload->>'reference','')
      ),
      'classification', jsonb_build_object(
        'document_type', COALESCE(e.classification_metadata->>'subcategory', e.payload->>'row_type'),
        'platform', COALESCE(e.classification_metadata->>'platform', e.source_platform),
        'confidence', e.confidence_score
      ),
      'source', jsonb_build_object(
        'platform', e.source_platform,
        'filename', e.source_filename,
        'row_index', e.row_index,
        'event_id', e.id::text,
        'file_id', NULLIF(e.file_id::text,''),
        'ingest_ts', e.ingest_ts
      ),
      'context', jsonb_build_object(
        'description', COALESCE(e.payload->>'standard_description', e.payload->>'description','')
      ),
      'metadata', jsonb_build_object(
        'schema_version', 'v1',
        'mapping', 'backfill_from_raw_events'
      )
    )
  ) AS universal,
  NOW(),
  NOW()
FROM public.raw_events e
LEFT JOIN public.universal_records ur ON ur.source_event_id = e.id
WHERE ur.id IS NULL;

-- 2) Backfill universal_parties from normalized_entities (idempotent)
INSERT INTO public.universal_parties (
  user_id, entity_type, canonical_name, aliases, identifiers, match_history, confidence_score, created_at, updated_at
)
SELECT
  ne.user_id,
  COALESCE(ne.entity_type,'vendor'),
  ne.canonical_name,
  COALESCE(ne.aliases, '{}')::text[],
  jsonb_strip_nulls(jsonb_build_object(
    'email', ne.email,
    'phone', ne.phone,
    'bank_account', ne.bank_account,
    'tax_id', ne.tax_id
  )),
  '[]'::jsonb,
  ne.confidence_score,
  NOW(),
  NOW()
FROM public.normalized_entities ne
LEFT JOIN public.universal_parties up
  ON up.user_id = ne.user_id AND up.canonical_name = ne.canonical_name AND up.entity_type = COALESCE(ne.entity_type,'vendor')
WHERE up.id IS NULL;

-- 3) Backfill universal_relationships from relationship_instances (idempotent)
INSERT INTO public.universal_relationships (
  user_id, source_record_id, target_record_id, relationship_type, confidence_score, detection_method, reasoning, created_at
)
SELECT
  r.user_id,
  urs.id AS source_record_id,
  urt.id AS target_record_id,
  r.relationship_type,
  r.confidence_score,
  r.detection_method,
  r.reasoning,
  NOW()
FROM public.relationship_instances r
JOIN public.universal_records urs ON urs.source_event_id = r.source_event_id
JOIN public.universal_records urt ON urt.source_event_id = r.target_event_id
LEFT JOIN public.universal_relationships ur
  ON ur.user_id = r.user_id
 AND ur.source_record_id = urs.id
 AND ur.target_record_id = urt.id
 AND ur.relationship_type = r.relationship_type
WHERE ur.id IS NULL;

COMMIT;
