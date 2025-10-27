-- Migration: Add Performance Indexes
-- Date: 2025-01-27
-- Purpose: Fix N+1 query problems and improve query performance
-- Note: Uses DO blocks to safely create indexes only if tables exist

-- Index for ingestion_jobs table (frequently queried by user_id and status)
DO $$ 
BEGIN
    IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'ingestion_jobs') THEN
        CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_user_status ON ingestion_jobs(user_id, status);
        CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_created_at ON ingestion_jobs(created_at DESC);
        RAISE NOTICE 'Created indexes for ingestion_jobs';
    END IF;
END $$;

-- Index for user_connections (queried by user and connector)
DO $$ 
BEGIN
    IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'user_connections') THEN
        CREATE INDEX IF NOT EXISTS idx_user_connections_user_connector ON user_connections(user_id, connector_id);
        RAISE NOTICE 'Created indexes for user_connections';
    END IF;
END $$;

-- Index for financial_transactions (frequently filtered by user and date)
DO $$ 
BEGIN
    IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'financial_transactions') THEN
        CREATE INDEX IF NOT EXISTS idx_financial_transactions_user_date ON financial_transactions(user_id, transaction_date DESC);
        CREATE INDEX IF NOT EXISTS idx_financial_transactions_vendor ON financial_transactions(user_id, vendor_name);
        RAISE NOTICE 'Created indexes for financial_transactions';
    END IF;
END $$;

-- Index for webhook_events (for retry processing)
DO $$ 
BEGIN
    IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'webhook_events') THEN
        CREATE INDEX IF NOT EXISTS idx_webhook_events_status ON webhook_events(status, received_at DESC)
        WHERE status IN ('queued', 'failed');
        RAISE NOTICE 'Created indexes for webhook_events';
    END IF;
END $$;

-- Index for chat_messages (for chat history queries)
DO $$ 
BEGIN
    IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'chat_messages') THEN
        CREATE INDEX IF NOT EXISTS idx_chat_messages_user_chat ON chat_messages(user_id, chat_id, created_at DESC);
        RAISE NOTICE 'Created indexes for chat_messages';
    END IF;
END $$;

-- Composite index for connector sync history
DO $$ 
BEGIN
    IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'connector_sync_history') THEN
        CREATE INDEX IF NOT EXISTS idx_connector_sync_history_connection ON connector_sync_history(connection_id, started_at DESC);
        RAISE NOTICE 'Created indexes for connector_sync_history';
    END IF;
END $$;

-- Add comments on indexes (only if they exist)
DO $$ 
BEGIN
    IF EXISTS (SELECT FROM pg_indexes WHERE indexname = 'idx_ingestion_jobs_user_status') THEN
        COMMENT ON INDEX idx_ingestion_jobs_user_status IS 'Improves queries filtering jobs by user and status';
    END IF;
    IF EXISTS (SELECT FROM pg_indexes WHERE indexname = 'idx_ingestion_jobs_created_at') THEN
        COMMENT ON INDEX idx_ingestion_jobs_created_at IS 'Speeds up recent jobs queries with ORDER BY created_at';
    END IF;
    IF EXISTS (SELECT FROM pg_indexes WHERE indexname = 'idx_user_connections_user_connector') THEN
        COMMENT ON INDEX idx_user_connections_user_connector IS 'Optimizes connection lookups by user and connector';
    END IF;
    IF EXISTS (SELECT FROM pg_indexes WHERE indexname = 'idx_financial_transactions_user_date') THEN
        COMMENT ON INDEX idx_financial_transactions_user_date IS 'Accelerates transaction queries with date filtering';
    END IF;
    IF EXISTS (SELECT FROM pg_indexes WHERE indexname = 'idx_financial_transactions_vendor') THEN
        COMMENT ON INDEX idx_financial_transactions_vendor IS 'Speeds up vendor analysis queries';
    END IF;
    IF EXISTS (SELECT FROM pg_indexes WHERE indexname = 'idx_webhook_events_status') THEN
        COMMENT ON INDEX idx_webhook_events_status IS 'Improves webhook retry processing queries';
    END IF;
    IF EXISTS (SELECT FROM pg_indexes WHERE indexname = 'idx_chat_messages_user_chat') THEN
        COMMENT ON INDEX idx_chat_messages_user_chat IS 'Optimizes chat history retrieval';
    END IF;
    IF EXISTS (SELECT FROM pg_indexes WHERE indexname = 'idx_connector_sync_history_connection') THEN
        COMMENT ON INDEX idx_connector_sync_history_connection IS 'Speeds up sync history queries';
    END IF;
END $$;
