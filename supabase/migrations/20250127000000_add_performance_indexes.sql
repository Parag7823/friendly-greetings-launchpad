-- Migration: Add Performance Indexes
-- Date: 2025-01-27
-- Purpose: Fix N+1 query problems and improve query performance

-- Index for ingestion_jobs table (frequently queried by user_id and status)
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_user_status 
ON ingestion_jobs(user_id, status);

-- Index for ingestion_jobs created_at (for ordering recent jobs)
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_created_at 
ON ingestion_jobs(created_at DESC);

-- Index for connector_connections (queried by user and provider)
CREATE INDEX IF NOT EXISTS idx_connector_connections_user_provider 
ON connector_connections(user_id, provider);

-- Index for financial_transactions (frequently filtered by user and date)
CREATE INDEX IF NOT EXISTS idx_financial_transactions_user_date 
ON financial_transactions(user_id, transaction_date DESC);

-- Index for financial_transactions vendor queries
CREATE INDEX IF NOT EXISTS idx_financial_transactions_vendor 
ON financial_transactions(user_id, vendor_name);

-- Index for webhook_events (for retry processing)
CREATE INDEX IF NOT EXISTS idx_webhook_events_status 
ON webhook_events(status, created_at DESC)
WHERE status IN ('pending', 'retry_pending');

-- Index for chat_messages (for chat history queries)
CREATE INDEX IF NOT EXISTS idx_chat_messages_user_chat 
ON chat_messages(user_id, chat_id, created_at DESC);

-- Composite index for connector sync history
CREATE INDEX IF NOT EXISTS idx_connector_sync_history_connection 
ON connector_sync_history(connection_id, started_at DESC);

-- Comment on indexes for documentation
COMMENT ON INDEX idx_ingestion_jobs_user_status IS 'Improves queries filtering jobs by user and status';
COMMENT ON INDEX idx_ingestion_jobs_created_at IS 'Speeds up recent jobs queries with ORDER BY created_at';
COMMENT ON INDEX idx_connector_connections_user_provider IS 'Optimizes connection lookups by user and provider';
COMMENT ON INDEX idx_financial_transactions_user_date IS 'Accelerates transaction queries with date filtering';
COMMENT ON INDEX idx_financial_transactions_vendor IS 'Speeds up vendor analysis queries';
COMMENT ON INDEX idx_webhook_events_status IS 'Improves webhook retry processing queries';
COMMENT ON INDEX idx_chat_messages_user_chat IS 'Optimizes chat history retrieval';
COMMENT ON INDEX idx_connector_sync_history_connection IS 'Speeds up sync history queries';
