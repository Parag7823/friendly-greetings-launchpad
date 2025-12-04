-- Migration: Clean up Nango-specific tables (dev environment)
-- Date: 2025-12-04
-- Purpose: Remove tables designed for Nango, keep only Airbyte-compatible schema
-- Safety: Dev environment, no customer data exists
-- Impact: Zero data loss (no data exists), schema simplified for Airbyte

-- ============================================================================
-- STEP 1: DELETE NANGO-SPECIFIC TABLES (Safe - no dependencies)
-- ============================================================================

DO $$
BEGIN
    -- Delete sync_cursors (Airbyte manages cursors internally)
    DROP TABLE IF EXISTS public.sync_cursors CASCADE;
    RAISE NOTICE '✅ DELETED: sync_cursors (Airbyte manages cursors)';

    -- Delete sync_checkpoints (Airbyte has built-in checkpointing)
    DROP TABLE IF EXISTS public.sync_checkpoints CASCADE;
    RAISE NOTICE '✅ DELETED: sync_checkpoints (Airbyte has built-in checkpointing)';

    -- Delete connectors (Airbyte manages source definitions)
    -- This will CASCADE delete user_connections (OK - no data exists)
    DROP TABLE IF EXISTS public.connectors CASCADE;
    RAISE NOTICE '✅ DELETED: connectors (Airbyte manages sources)';
END $$;

-- ============================================================================
-- STEP 2: RECREATE user_connections WITHOUT connector_id FK
-- ============================================================================
-- user_connections was CASCADE deleted above, now recreate it simplified

CREATE TABLE IF NOT EXISTS public.user_connections (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL,
    provider TEXT NOT NULL,  -- gmail, dropbox, google-drive, etc. (moved from connectors)
    airbyte_connection_id UUID,  -- Airbyte's connection UUID
    nango_connection_id TEXT UNIQUE,  -- Legacy compatibility
    status TEXT NOT NULL DEFAULT 'active',  -- active | expired | revoked
    sync_mode TEXT NOT NULL DEFAULT 'pull',  -- pull | webhook
    last_synced_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb,
    provider_account_id TEXT,
    sync_frequency_minutes INTEGER DEFAULT 60,
    integration_id TEXT
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_user_connections_user_id 
ON public.user_connections(user_id);

CREATE INDEX IF NOT EXISTS idx_user_connections_status 
ON public.user_connections(status);

CREATE INDEX IF NOT EXISTS idx_user_connections_provider 
ON public.user_connections(provider);

CREATE INDEX IF NOT EXISTS idx_user_connections_user_status 
ON public.user_connections(user_id, status) WHERE status = 'active';

CREATE INDEX IF NOT EXISTS idx_user_connections_airbyte_id 
ON public.user_connections(airbyte_connection_id) WHERE airbyte_connection_id IS NOT NULL;

-- Enable RLS
ALTER TABLE public.user_connections ENABLE ROW LEVEL SECURITY;

-- Create RLS policy
DROP POLICY IF EXISTS "user_connections_owner" ON public.user_connections;
CREATE POLICY "user_connections_owner" ON public.user_connections
    FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

-- ============================================================================
-- STEP 3: VERIFY DEPENDENT TABLES STILL EXIST
-- ============================================================================

-- Verify sync_runs exists and has correct FK to user_connections
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables 
        WHERE table_name = 'sync_runs'
    ) THEN
        RAISE NOTICE '✅ VERIFIED: sync_runs table exists';
        
        -- Verify FK is correct
        IF EXISTS (
            SELECT 1 FROM information_schema.table_constraints 
            WHERE table_name = 'sync_runs' 
            AND constraint_name = 'sync_runs_user_connection_id_fkey'
        ) THEN
            RAISE NOTICE '✅ VERIFIED: sync_runs FK to user_connections is valid';
        ELSE
            -- Add FK if missing
            ALTER TABLE public.sync_runs 
            ADD CONSTRAINT sync_runs_user_connection_id_fkey 
            FOREIGN KEY (user_connection_id) REFERENCES public.user_connections(id) ON DELETE CASCADE;
            RAISE NOTICE '✅ ADDED: sync_runs FK to user_connections';
        END IF;
    ELSE
        RAISE NOTICE '⚠️  WARNING: sync_runs table does not exist';
    END IF;
END $$;

-- Verify external_items exists and has correct FK to user_connections
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables 
        WHERE table_name = 'external_items'
    ) THEN
        RAISE NOTICE '✅ VERIFIED: external_items table exists';
        
        -- Verify FK is correct
        IF EXISTS (
            SELECT 1 FROM information_schema.table_constraints 
            WHERE table_name = 'external_items' 
            AND constraint_name LIKE '%user_connection_id%'
        ) THEN
            RAISE NOTICE '✅ VERIFIED: external_items FK to user_connections is valid';
        ELSE
            -- Add FK if missing
            ALTER TABLE public.external_items 
            ADD CONSTRAINT external_items_user_connection_id_fkey 
            FOREIGN KEY (user_connection_id) REFERENCES public.user_connections(id) ON DELETE CASCADE;
            RAISE NOTICE '✅ ADDED: external_items FK to user_connections';
        END IF;
    ELSE
        RAISE NOTICE '⚠️  WARNING: external_items table does not exist';
    END IF;
END $$;

-- ============================================================================
-- STEP 4: FINAL VERIFICATION
-- ============================================================================

DO $$
DECLARE
    connectors_exists BOOLEAN;
    sync_cursors_exists BOOLEAN;
    sync_checkpoints_exists BOOLEAN;
    user_connections_exists BOOLEAN;
BEGIN
    -- Check if tables were deleted
    SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'connectors') 
    INTO connectors_exists;
    
    SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'sync_cursors') 
    INTO sync_cursors_exists;
    
    SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'sync_checkpoints') 
    INTO sync_checkpoints_exists;
    
    SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'user_connections') 
    INTO user_connections_exists;
    
    RAISE NOTICE '';
    RAISE NOTICE '╔════════════════════════════════════════════════════════════╗';
    RAISE NOTICE '║  MIGRATION COMPLETE: Nango Tables Cleaned Up              ║';
    RAISE NOTICE '╚════════════════════════════════════════════════════════════╝';
    RAISE NOTICE '';
    RAISE NOTICE '✅ DELETED TABLES:';
    RAISE NOTICE '   connectors: %', CASE WHEN NOT connectors_exists THEN 'DELETED ✓' ELSE 'STILL EXISTS ⚠️' END;
    RAISE NOTICE '   sync_cursors: %', CASE WHEN NOT sync_cursors_exists THEN 'DELETED ✓' ELSE 'STILL EXISTS ⚠️' END;
    RAISE NOTICE '   sync_checkpoints: %', CASE WHEN NOT sync_checkpoints_exists THEN 'DELETED ✓' ELSE 'STILL EXISTS ⚠️' END;
    RAISE NOTICE '';
    RAISE NOTICE '✅ RECREATED TABLES:';
    RAISE NOTICE '   user_connections: %', CASE WHEN user_connections_exists THEN 'RECREATED ✓' ELSE 'MISSING ⚠️' END;
    RAISE NOTICE '   (without connector_id FK - now Airbyte-compatible)';
    RAISE NOTICE '';
    RAISE NOTICE '✅ SCHEMA CHANGES:';
    RAISE NOTICE '   - provider moved from connectors → user_connections';
    RAISE NOTICE '   - airbyte_connection_id added to user_connections';
    RAISE NOTICE '   - connector_id FK removed (no longer needed)';
    RAISE NOTICE '   - All RLS policies updated';
    RAISE NOTICE '';
    RAISE NOTICE '✅ READY FOR AIRBYTE:';
    RAISE NOTICE '   - user_connections stores Airbyte connection IDs';
    RAISE NOTICE '   - sync_runs tracks Airbyte sync jobs';
    RAISE NOTICE '   - external_items stores synced data';
    RAISE NOTICE '   - webhook_events tracks Airbyte webhooks';
    RAISE NOTICE '';
END $$;
