-- Migration: Add Comprehensive RLS Policies for All Tables
-- Date: 2025-10-25 14:00:00
-- Purpose: Enable Row Level Security for all user data tables so users can see their data

-- ============================================================================
-- ENABLE RLS ON ALL USER DATA TABLES
-- ============================================================================

-- Core data tables
ALTER TABLE public.raw_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ingestion_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.raw_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.normalized_entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.entity_matches ENABLE ROW LEVEL SECURITY;

-- Processing and transaction tables
ALTER TABLE public.processing_transactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.processing_locks ENABLE ROW LEVEL SECURITY;

-- External integrations
ALTER TABLE public.external_items ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_connections ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.sync_logs ENABLE ROW LEVEL SECURITY;

-- Cache tables
ALTER TABLE public.ai_classification_cache ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.duplicate_detection_cache ENABLE ROW LEVEL SECURITY;

-- Platform and document classification
ALTER TABLE public.platform_detections ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.document_classifications ENABLE ROW LEVEL SECURITY;

-- Metrics and analytics
ALTER TABLE public.processing_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_activity_logs ENABLE ROW LEVEL SECURITY;

-- Chat tables
ALTER TABLE public.chat_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_messages ENABLE ROW LEVEL SECURITY;

-- Temporal patterns
ALTER TABLE public.temporal_patterns ENABLE ROW LEVEL SECURITY;

-- ============================================================================
-- RLS POLICIES: raw_records
-- ============================================================================

DROP POLICY IF EXISTS "service_role_all_raw_records" ON public.raw_records;
CREATE POLICY "service_role_all_raw_records" ON public.raw_records
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_raw_records" ON public.raw_records;
CREATE POLICY "users_own_raw_records" ON public.raw_records
    FOR ALL USING (auth.uid() = user_id);

-- ============================================================================
-- RLS POLICIES: ingestion_jobs
-- ============================================================================

DROP POLICY IF EXISTS "service_role_all_ingestion_jobs" ON public.ingestion_jobs;
CREATE POLICY "service_role_all_ingestion_jobs" ON public.ingestion_jobs
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_ingestion_jobs" ON public.ingestion_jobs;
CREATE POLICY "users_own_ingestion_jobs" ON public.ingestion_jobs
    FOR ALL USING (auth.uid() = user_id);

-- ============================================================================
-- RLS POLICIES: raw_events
-- ============================================================================

DROP POLICY IF EXISTS "service_role_all_raw_events" ON public.raw_events;
CREATE POLICY "service_role_all_raw_events" ON public.raw_events
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_raw_events" ON public.raw_events;
CREATE POLICY "users_own_raw_events" ON public.raw_events
    FOR ALL USING (auth.uid() = user_id);

-- ============================================================================
-- RLS POLICIES: normalized_entities
-- ============================================================================

DROP POLICY IF EXISTS "service_role_all_normalized_entities" ON public.normalized_entities;
CREATE POLICY "service_role_all_normalized_entities" ON public.normalized_entities
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_normalized_entities" ON public.normalized_entities;
CREATE POLICY "users_own_normalized_entities" ON public.normalized_entities
    FOR ALL USING (auth.uid() = user_id);

-- ============================================================================
-- RLS POLICIES: entity_matches
-- ============================================================================

DROP POLICY IF EXISTS "service_role_all_entity_matches" ON public.entity_matches;
CREATE POLICY "service_role_all_entity_matches" ON public.entity_matches
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_entity_matches" ON public.entity_matches;
CREATE POLICY "users_own_entity_matches" ON public.entity_matches
    FOR ALL USING (auth.uid() = user_id);

-- ============================================================================
-- RLS POLICIES: processing_transactions
-- ============================================================================

DROP POLICY IF EXISTS "service_role_all_processing_transactions" ON public.processing_transactions;
CREATE POLICY "service_role_all_processing_transactions" ON public.processing_transactions
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_processing_transactions" ON public.processing_transactions;
CREATE POLICY "users_own_processing_transactions" ON public.processing_transactions
    FOR ALL USING (auth.uid() = user_id);

-- ============================================================================
-- RLS POLICIES: processing_locks
-- ============================================================================

DROP POLICY IF EXISTS "service_role_all_processing_locks" ON public.processing_locks;
CREATE POLICY "service_role_all_processing_locks" ON public.processing_locks
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_processing_locks" ON public.processing_locks;
CREATE POLICY "users_own_processing_locks" ON public.processing_locks
    FOR ALL USING (auth.uid() = user_id);

-- ============================================================================
-- RLS POLICIES: external_items
-- ============================================================================

DROP POLICY IF EXISTS "service_role_all_external_items" ON public.external_items;
CREATE POLICY "service_role_all_external_items" ON public.external_items
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_external_items" ON public.external_items;
CREATE POLICY "users_own_external_items" ON public.external_items
    FOR ALL USING (auth.uid() = user_id);

-- ============================================================================
-- RLS POLICIES: user_connections
-- ============================================================================

DROP POLICY IF EXISTS "service_role_all_user_connections" ON public.user_connections;
CREATE POLICY "service_role_all_user_connections" ON public.user_connections
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_user_connections" ON public.user_connections;
CREATE POLICY "users_own_user_connections" ON public.user_connections
    FOR ALL USING (auth.uid() = user_id);

-- ============================================================================
-- RLS POLICIES: sync_logs
-- ============================================================================

DROP POLICY IF EXISTS "service_role_all_sync_logs" ON public.sync_logs;
CREATE POLICY "service_role_all_sync_logs" ON public.sync_logs
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_sync_logs" ON public.sync_logs;
CREATE POLICY "users_own_sync_logs" ON public.sync_logs
    FOR ALL USING (auth.uid() = user_id);

-- ============================================================================
-- RLS POLICIES: ai_classification_cache
-- ============================================================================

DROP POLICY IF EXISTS "service_role_all_ai_classification_cache" ON public.ai_classification_cache;
CREATE POLICY "service_role_all_ai_classification_cache" ON public.ai_classification_cache
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_ai_classification_cache" ON public.ai_classification_cache;
CREATE POLICY "users_own_ai_classification_cache" ON public.ai_classification_cache
    FOR ALL USING (auth.uid() = user_id);

-- ============================================================================
-- RLS POLICIES: duplicate_detection_cache
-- ============================================================================

DROP POLICY IF EXISTS "service_role_all_duplicate_detection_cache" ON public.duplicate_detection_cache;
CREATE POLICY "service_role_all_duplicate_detection_cache" ON public.duplicate_detection_cache
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_duplicate_detection_cache" ON public.duplicate_detection_cache;
CREATE POLICY "users_own_duplicate_detection_cache" ON public.duplicate_detection_cache
    FOR ALL USING (auth.uid() = user_id);

-- ============================================================================
-- RLS POLICIES: platform_detections
-- ============================================================================

DROP POLICY IF EXISTS "service_role_all_platform_detections" ON public.platform_detections;
CREATE POLICY "service_role_all_platform_detections" ON public.platform_detections
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_platform_detections" ON public.platform_detections;
CREATE POLICY "users_own_platform_detections" ON public.platform_detections
    FOR ALL USING (auth.uid() = user_id);

-- ============================================================================
-- RLS POLICIES: document_classifications
-- ============================================================================

DROP POLICY IF EXISTS "service_role_all_document_classifications" ON public.document_classifications;
CREATE POLICY "service_role_all_document_classifications" ON public.document_classifications
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_document_classifications" ON public.document_classifications;
CREATE POLICY "users_own_document_classifications" ON public.document_classifications
    FOR ALL USING (auth.uid() = user_id);

-- ============================================================================
-- RLS POLICIES: processing_metrics
-- ============================================================================

DROP POLICY IF EXISTS "service_role_all_processing_metrics" ON public.processing_metrics;
CREATE POLICY "service_role_all_processing_metrics" ON public.processing_metrics
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_processing_metrics" ON public.processing_metrics;
CREATE POLICY "users_own_processing_metrics" ON public.processing_metrics
    FOR ALL USING (auth.uid() = user_id);

-- ============================================================================
-- RLS POLICIES: user_activity_logs
-- ============================================================================

DROP POLICY IF EXISTS "service_role_all_user_activity_logs" ON public.user_activity_logs;
CREATE POLICY "service_role_all_user_activity_logs" ON public.user_activity_logs
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_user_activity_logs" ON public.user_activity_logs;
CREATE POLICY "users_own_user_activity_logs" ON public.user_activity_logs
    FOR ALL USING (auth.uid() = user_id);

-- ============================================================================
-- RLS POLICIES: chat_sessions
-- ============================================================================

DROP POLICY IF EXISTS "service_role_all_chat_sessions" ON public.chat_sessions;
CREATE POLICY "service_role_all_chat_sessions" ON public.chat_sessions
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_chat_sessions" ON public.chat_sessions;
CREATE POLICY "users_own_chat_sessions" ON public.chat_sessions
    FOR ALL USING (auth.uid() = user_id);

-- ============================================================================
-- RLS POLICIES: chat_messages
-- ============================================================================

DROP POLICY IF EXISTS "service_role_all_chat_messages" ON public.chat_messages;
CREATE POLICY "service_role_all_chat_messages" ON public.chat_messages
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_chat_messages" ON public.chat_messages;
CREATE POLICY "users_own_chat_messages" ON public.chat_messages
    FOR ALL USING (auth.uid() = user_id);

-- ============================================================================
-- RLS POLICIES: temporal_patterns
-- ============================================================================

DROP POLICY IF EXISTS "service_role_all_temporal_patterns" ON public.temporal_patterns;
CREATE POLICY "service_role_all_temporal_patterns" ON public.temporal_patterns
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_temporal_patterns" ON public.temporal_patterns;
CREATE POLICY "users_own_temporal_patterns" ON public.temporal_patterns
    FOR ALL USING (auth.uid() = user_id);

-- ============================================================================
-- GRANT PERMISSIONS
-- ============================================================================

GRANT SELECT, INSERT, UPDATE, DELETE ON public.raw_records TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.ingestion_jobs TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.raw_events TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.normalized_entities TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.entity_matches TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.processing_transactions TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.processing_locks TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.external_items TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.user_connections TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.sync_logs TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.ai_classification_cache TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.duplicate_detection_cache TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.platform_detections TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.document_classifications TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.processing_metrics TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.user_activity_logs TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.chat_sessions TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.chat_messages TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.temporal_patterns TO authenticated;

-- ============================================================================
-- VERIFICATION QUERY
-- ============================================================================

-- Run this to verify all tables have RLS enabled:
-- SELECT 
--     schemaname,
--     tablename,
--     rowsecurity as rls_enabled
-- FROM pg_tables
-- WHERE schemaname = 'public'
-- ORDER BY tablename;
