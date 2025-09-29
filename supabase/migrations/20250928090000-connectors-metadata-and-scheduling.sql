-- Add metadata & scheduling fields for connectors and user connections

-- Add metadata JSONB to connectors and user_connections
ALTER TABLE public.connectors 
ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb;

ALTER TABLE public.user_connections 
ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb,
ADD COLUMN IF NOT EXISTS provider_account_id TEXT,
ADD COLUMN IF NOT EXISTS sync_frequency_minutes INTEGER DEFAULT 60;

-- Indexes for scheduling and lookups
CREATE INDEX IF NOT EXISTS idx_user_connections_status ON public.user_connections(status);
CREATE INDEX IF NOT EXISTS idx_user_connections_next_sync ON public.user_connections(sync_frequency_minutes);

-- Webhook idempotency support
ALTER TABLE public.webhook_events 
ADD COLUMN IF NOT EXISTS event_id TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS idx_webhook_events_event_id ON public.webhook_events(event_id) WHERE event_id IS NOT NULL;

-- Ensure external_items uniqueness is enforced (idempotency)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes WHERE schemaname = 'public' AND indexname = 'external_items_user_conn_provider_unique') THEN
        CREATE UNIQUE INDEX external_items_user_conn_provider_unique 
        ON public.external_items(user_connection_id, provider_id);
    END IF;
END $$;
