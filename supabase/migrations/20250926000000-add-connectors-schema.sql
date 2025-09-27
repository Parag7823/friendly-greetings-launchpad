-- Migration: Add connectors & ingestion schema (Nango-backed)
-- Date: 2025-09-26

-- Enable required extension
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- 1) Connectors (catalog of supported providers via Nango)
CREATE TABLE IF NOT EXISTS public.connectors (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    provider TEXT NOT NULL, -- e.g., gmail, gdrive, dropbox, slack, quickbooks, xero, stripe, plaid, outlook
    integration_id TEXT NOT NULL, -- Nango integration id
    auth_type TEXT NOT NULL DEFAULT 'OAUTH2',
    scopes JSONB NOT NULL DEFAULT '[]'::jsonb,
    endpoints_needed JSONB NOT NULL DEFAULT '[]'::jsonb,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_connectors_provider ON public.connectors(provider);

-- 2) User connections (a user's authenticated connection to a provider)
CREATE TABLE IF NOT EXISTS public.user_connections (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL,
    connector_id UUID NOT NULL REFERENCES public.connectors(id) ON DELETE CASCADE,
    nango_connection_id TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'active', -- active | expired | revoked
    sync_mode TEXT NOT NULL DEFAULT 'pull', -- pull | webhook
    last_synced_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_connections_user_id ON public.user_connections(user_id);
CREATE INDEX IF NOT EXISTS idx_user_connections_connector ON public.user_connections(connector_id);

-- 3) Sync runs (each execution of a sync job)
CREATE TABLE IF NOT EXISTS public.sync_runs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL,
    user_connection_id UUID NOT NULL REFERENCES public.user_connections(id) ON DELETE CASCADE,
    type TEXT NOT NULL, -- historical | incremental | manual
    status TEXT NOT NULL DEFAULT 'queued', -- queued | running | succeeded | failed | partial
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    finished_at TIMESTAMP WITH TIME ZONE,
    stats JSONB NOT NULL DEFAULT '{}'::jsonb, -- { records_fetched, actions_used, bytes }
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_sync_runs_user_connection ON public.sync_runs(user_connection_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sync_runs_user_id ON public.sync_runs(user_id);

-- 4) Sync cursors (bookmark per resource)
CREATE TABLE IF NOT EXISTS public.sync_cursors (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL,
    user_connection_id UUID NOT NULL REFERENCES public.user_connections(id) ON DELETE CASCADE,
    resource TEXT NOT NULL, -- emails | files | txns
    cursor_type TEXT NOT NULL, -- time | page | history_id
    value TEXT NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE (user_connection_id, resource, cursor_type)
);

CREATE INDEX IF NOT EXISTS idx_sync_cursors_user_connection ON public.sync_cursors(user_connection_id);
CREATE INDEX IF NOT EXISTS idx_sync_cursors_user_id ON public.sync_cursors(user_id);

-- 5) Webhook events (raw payload audit & replay)
CREATE TABLE IF NOT EXISTS public.webhook_events (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL,
    user_connection_id UUID REFERENCES public.user_connections(id) ON DELETE SET NULL,
    event_type TEXT,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    signature_valid BOOLEAN NOT NULL DEFAULT FALSE,
    received_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processed_at TIMESTAMP WITH TIME ZONE,
    status TEXT NOT NULL DEFAULT 'queued', -- queued | processed | failed
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_webhook_events_user_connection ON public.webhook_events(user_connection_id, received_at DESC);
CREATE INDEX IF NOT EXISTS idx_webhook_events_user_id ON public.webhook_events(user_id);

-- 6) External items (normalized envelope for fetched items)
CREATE TABLE IF NOT EXISTS public.external_items (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL,
    user_connection_id UUID NOT NULL REFERENCES public.user_connections(id) ON DELETE CASCADE,
    provider_id TEXT NOT NULL, -- e.g., gmail messageId / fileId
    kind TEXT NOT NULL, -- email | file | txn
    source_ts TIMESTAMP WITH TIME ZONE,
    hash TEXT,
    storage_path TEXT, -- Supabase storage path for saved attachment/file
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    relevance_score DOUBLE PRECISION,
    status TEXT NOT NULL DEFAULT 'fetched', -- fetched | stored | queued | processed | skipped
    error TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE (user_connection_id, provider_id)
);

CREATE INDEX IF NOT EXISTS idx_external_items_user_connection ON public.external_items(user_connection_id, status);
CREATE INDEX IF NOT EXISTS idx_external_items_hash ON public.external_items(hash);
CREATE INDEX IF NOT EXISTS idx_external_items_user_id ON public.external_items(user_id);

-- RLS policies
ALTER TABLE public.connectors ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_connections ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.sync_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.sync_cursors ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.webhook_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.external_items ENABLE ROW LEVEL SECURITY;

-- Connectors can be readable (catalog) by all authenticated users
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies WHERE schemaname = 'public' AND tablename = 'connectors' AND policyname = 'connectors_select_all'
    ) THEN
        CREATE POLICY "connectors_select_all" ON public.connectors
            FOR SELECT USING (true);
    END IF;
END$$;

-- User-scoped tables: only owner can read/write
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies WHERE schemaname = 'public' AND tablename = 'user_connections' AND policyname = 'user_connections_owner'
    ) THEN
        CREATE POLICY "user_connections_owner" ON public.user_connections
            FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_policies WHERE schemaname = 'public' AND tablename = 'sync_runs' AND policyname = 'sync_runs_owner'
    ) THEN
        CREATE POLICY "sync_runs_owner" ON public.sync_runs
            FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_policies WHERE schemaname = 'public' AND tablename = 'sync_cursors' AND policyname = 'sync_cursors_owner'
    ) THEN
        CREATE POLICY "sync_cursors_owner" ON public.sync_cursors
            FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_policies WHERE schemaname = 'public' AND tablename = 'webhook_events' AND policyname = 'webhook_events_owner'
    ) THEN
        CREATE POLICY "webhook_events_owner" ON public.webhook_events
            FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_policies WHERE schemaname = 'public' AND tablename = 'external_items' AND policyname = 'external_items_owner'
    ) THEN
        CREATE POLICY "external_items_owner" ON public.external_items
            FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);
    END IF;
END$$;
