-- Migration: Add Universal Finance Schema core tables (v1)
-- Date: 2025-10-07 17:15:00 +05:30

BEGIN;

-- Ensure pgcrypto for gen_random_uuid
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- 1) universal_records: canonical record per financial row/transaction
CREATE TABLE IF NOT EXISTS public.universal_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    record_type TEXT,
    source_event_id UUID, -- optional backref to raw_events(id)
    universal JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_universal_records_user_created 
  ON public.universal_records(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_universal_records_record_type 
  ON public.universal_records(record_type);
CREATE INDEX IF NOT EXISTS idx_universal_records_source_event 
  ON public.universal_records(source_event_id);
CREATE INDEX IF NOT EXISTS idx_universal_records_universal_gin 
  ON public.universal_records USING GIN (universal);

ALTER TABLE public.universal_records ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies WHERE schemaname='public' AND tablename='universal_records' AND policyname='universal_records_service_access'
  ) THEN
    CREATE POLICY "universal_records_service_access" ON public.universal_records
      FOR ALL USING (auth.role() = 'service_role') WITH CHECK (auth.role() = 'service_role');
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies WHERE schemaname='public' AND tablename='universal_records' AND policyname='universal_records_user_access'
  ) THEN
    CREATE POLICY "universal_records_user_access" ON public.universal_records
      FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);
  END IF;
END$$;

-- 2) universal_parties: canonical entities (vendors, customers, employees)
CREATE TABLE IF NOT EXISTS public.universal_parties (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    entity_type TEXT NOT NULL, -- vendor | customer | employee | bank | tax | other
    canonical_name TEXT NOT NULL,
    aliases TEXT[] DEFAULT '{}',
    identifiers JSONB DEFAULT '{}'::jsonb,
    match_history JSONB DEFAULT '[]'::jsonb,
    confidence_score DECIMAL(3,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_universal_parties_user_type 
  ON public.universal_parties(user_id, entity_type, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_universal_parties_canonical 
  ON public.universal_parties(user_id, canonical_name);
CREATE INDEX IF NOT EXISTS idx_universal_parties_aliases_gin 
  ON public.universal_parties USING GIN (aliases);

ALTER TABLE public.universal_parties ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies WHERE schemaname='public' AND tablename='universal_parties' AND policyname='universal_parties_service_access'
  ) THEN
    CREATE POLICY "universal_parties_service_access" ON public.universal_parties
      FOR ALL USING (auth.role() = 'service_role') WITH CHECK (auth.role() = 'service_role');
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies WHERE schemaname='public' AND tablename='universal_parties' AND policyname='universal_parties_user_access'
  ) THEN
    CREATE POLICY "universal_parties_user_access" ON public.universal_parties
      FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);
  END IF;
END$$;

-- 3) universal_relationships: links between universal_records
CREATE TABLE IF NOT EXISTS public.universal_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    source_record_id UUID NOT NULL REFERENCES public.universal_records(id) ON DELETE CASCADE,
    target_record_id UUID NOT NULL REFERENCES public.universal_records(id) ON DELETE CASCADE,
    relationship_type TEXT NOT NULL,
    confidence_score DECIMAL(3,2),
    detection_method TEXT,
    reasoning TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_universal_relationships_user_type 
  ON public.universal_relationships(user_id, relationship_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_universal_relationships_src_tgt 
  ON public.universal_relationships(source_record_id, target_record_id);

ALTER TABLE public.universal_relationships ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies WHERE schemaname='public' AND tablename='universal_relationships' AND policyname='universal_relationships_service_access'
  ) THEN
    CREATE POLICY "universal_relationships_service_access" ON public.universal_relationships
      FOR ALL USING (auth.role() = 'service_role') WITH CHECK (auth.role() = 'service_role');
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies WHERE schemaname='public' AND tablename='universal_relationships' AND policyname='universal_relationships_user_access'
  ) THEN
    CREATE POLICY "universal_relationships_user_access" ON public.universal_relationships
      FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);
  END IF;
END$$;

COMMIT;
