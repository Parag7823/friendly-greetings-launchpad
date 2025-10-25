-- Migration: Add RLS Policies for ALL ACTUAL EXISTING Tables
-- Date: 2025-10-25 14:00:02
-- Purpose: Enable Row Level Security for all 33 existing tables

-- ============================================================================
-- ENABLE RLS ON ALL EXISTING TABLES
-- ============================================================================

-- Core data tables
ALTER TABLE public.raw_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ingestion_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.raw_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.normalized_entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.entity_matches ENABLE ROW LEVEL SECURITY;

-- Processing tables
ALTER TABLE public.processing_transactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.processing_locks ENABLE ROW LEVEL SECURITY;

-- External integrations
ALTER TABLE public.external_items ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_connections ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.connectors ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.sync_cursors ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.sync_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.webhook_events ENABLE ROW LEVEL SECURITY;

-- Relationship tables
ALTER TABLE public.relationship_instances ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.cross_platform_relationships ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.relationship_patterns ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.causal_relationships ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.predicted_relationships ENABLE ROW LEVEL SECURITY;

-- Pattern and analysis tables
ALTER TABLE public.temporal_patterns ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.temporal_anomalies ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.seasonal_patterns ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.platform_patterns ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.discovered_platforms ENABLE ROW LEVEL SECURITY;

-- Advanced analysis tables
ALTER TABLE public.counterfactual_analyses ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.root_cause_analyses ENABLE ROW LEVEL SECURITY;

-- Logging and metrics tables
ALTER TABLE public.detection_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.resolution_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.event_delta_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.error_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.performance_metrics ENABLE ROW LEVEL SECURITY;

-- Field mappings and chat
ALTER TABLE public.field_mappings ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_messages ENABLE ROW LEVEL SECURITY;

-- ============================================================================
-- RLS POLICIES: Core Data Tables
-- ============================================================================

-- raw_records
DROP POLICY IF EXISTS "service_role_all_raw_records" ON public.raw_records;
CREATE POLICY "service_role_all_raw_records" ON public.raw_records
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_raw_records" ON public.raw_records;
CREATE POLICY "users_own_raw_records" ON public.raw_records
    FOR ALL USING (auth.uid() = user_id);

-- ingestion_jobs
DROP POLICY IF EXISTS "service_role_all_ingestion_jobs" ON public.ingestion_jobs;
CREATE POLICY "service_role_all_ingestion_jobs" ON public.ingestion_jobs
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_ingestion_jobs" ON public.ingestion_jobs;
CREATE POLICY "users_own_ingestion_jobs" ON public.ingestion_jobs
    FOR ALL USING (auth.uid() = user_id);

-- raw_events
DROP POLICY IF EXISTS "service_role_all_raw_events" ON public.raw_events;
CREATE POLICY "service_role_all_raw_events" ON public.raw_events
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_raw_events" ON public.raw_events;
CREATE POLICY "users_own_raw_events" ON public.raw_events
    FOR ALL USING (auth.uid() = user_id);

-- normalized_entities
DROP POLICY IF EXISTS "service_role_all_normalized_entities" ON public.normalized_entities;
CREATE POLICY "service_role_all_normalized_entities" ON public.normalized_entities
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_normalized_entities" ON public.normalized_entities;
CREATE POLICY "users_own_normalized_entities" ON public.normalized_entities
    FOR ALL USING (auth.uid() = user_id);

-- entity_matches
DROP POLICY IF EXISTS "service_role_all_entity_matches" ON public.entity_matches;
CREATE POLICY "service_role_all_entity_matches" ON public.entity_matches
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_entity_matches" ON public.entity_matches;
CREATE POLICY "users_own_entity_matches" ON public.entity_matches
    FOR ALL USING (auth.uid() = user_id);

-- ============================================================================
-- RLS POLICIES: Processing Tables
-- ============================================================================

-- processing_transactions
DROP POLICY IF EXISTS "service_role_all_processing_transactions" ON public.processing_transactions;
CREATE POLICY "service_role_all_processing_transactions" ON public.processing_transactions
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_processing_transactions" ON public.processing_transactions;
CREATE POLICY "users_own_processing_transactions" ON public.processing_transactions
    FOR ALL USING (auth.uid() = user_id);

-- processing_locks
DROP POLICY IF EXISTS "service_role_all_processing_locks" ON public.processing_locks;
CREATE POLICY "service_role_all_processing_locks" ON public.processing_locks
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_processing_locks" ON public.processing_locks;
CREATE POLICY "users_own_processing_locks" ON public.processing_locks
    FOR ALL USING (auth.uid() = user_id);

-- ============================================================================
-- RLS POLICIES: External Integrations
-- ============================================================================

-- external_items
DROP POLICY IF EXISTS "service_role_all_external_items" ON public.external_items;
CREATE POLICY "service_role_all_external_items" ON public.external_items
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_external_items" ON public.external_items;
CREATE POLICY "users_own_external_items" ON public.external_items
    FOR ALL USING (auth.uid() = user_id);

-- user_connections
DROP POLICY IF EXISTS "service_role_all_user_connections" ON public.user_connections;
CREATE POLICY "service_role_all_user_connections" ON public.user_connections
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_user_connections" ON public.user_connections;
CREATE POLICY "users_own_user_connections" ON public.user_connections
    FOR ALL USING (auth.uid() = user_id);

-- connectors
DROP POLICY IF EXISTS "service_role_all_connectors" ON public.connectors;
CREATE POLICY "service_role_all_connectors" ON public.connectors
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_connectors" ON public.connectors;
CREATE POLICY "users_own_connectors" ON public.connectors
    FOR ALL USING (auth.uid() = user_id);

-- sync_cursors
DROP POLICY IF EXISTS "service_role_all_sync_cursors" ON public.sync_cursors;
CREATE POLICY "service_role_all_sync_cursors" ON public.sync_cursors
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_sync_cursors" ON public.sync_cursors;
CREATE POLICY "users_own_sync_cursors" ON public.sync_cursors
    FOR ALL USING (auth.uid() = user_id);

-- sync_runs
DROP POLICY IF EXISTS "service_role_all_sync_runs" ON public.sync_runs;
CREATE POLICY "service_role_all_sync_runs" ON public.sync_runs
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_sync_runs" ON public.sync_runs;
CREATE POLICY "users_own_sync_runs" ON public.sync_runs
    FOR ALL USING (auth.uid() = user_id);

-- webhook_events
DROP POLICY IF EXISTS "service_role_all_webhook_events" ON public.webhook_events;
CREATE POLICY "service_role_all_webhook_events" ON public.webhook_events
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_webhook_events" ON public.webhook_events;
CREATE POLICY "users_own_webhook_events" ON public.webhook_events
    FOR ALL USING (auth.uid() = user_id);

-- ============================================================================
-- RLS POLICIES: Relationship Tables
-- ============================================================================

-- relationship_instances
DROP POLICY IF EXISTS "service_role_all_relationship_instances" ON public.relationship_instances;
CREATE POLICY "service_role_all_relationship_instances" ON public.relationship_instances
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_relationship_instances" ON public.relationship_instances;
CREATE POLICY "users_own_relationship_instances" ON public.relationship_instances
    FOR ALL USING (auth.uid() = user_id);

-- cross_platform_relationships
DROP POLICY IF EXISTS "service_role_all_cross_platform_relationships" ON public.cross_platform_relationships;
CREATE POLICY "service_role_all_cross_platform_relationships" ON public.cross_platform_relationships
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_cross_platform_relationships" ON public.cross_platform_relationships;
CREATE POLICY "users_own_cross_platform_relationships" ON public.cross_platform_relationships
    FOR ALL USING (auth.uid() = user_id);

-- relationship_patterns
DROP POLICY IF EXISTS "service_role_all_relationship_patterns" ON public.relationship_patterns;
CREATE POLICY "service_role_all_relationship_patterns" ON public.relationship_patterns
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_relationship_patterns" ON public.relationship_patterns;
CREATE POLICY "users_own_relationship_patterns" ON public.relationship_patterns
    FOR ALL USING (auth.uid() = user_id);

-- causal_relationships
DROP POLICY IF EXISTS "service_role_all_causal_relationships" ON public.causal_relationships;
CREATE POLICY "service_role_all_causal_relationships" ON public.causal_relationships
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_causal_relationships" ON public.causal_relationships;
CREATE POLICY "users_own_causal_relationships" ON public.causal_relationships
    FOR ALL USING (auth.uid() = user_id);

-- predicted_relationships
DROP POLICY IF EXISTS "service_role_all_predicted_relationships" ON public.predicted_relationships;
CREATE POLICY "service_role_all_predicted_relationships" ON public.predicted_relationships
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_predicted_relationships" ON public.predicted_relationships;
CREATE POLICY "users_own_predicted_relationships" ON public.predicted_relationships
    FOR ALL USING (auth.uid() = user_id);

-- ============================================================================
-- RLS POLICIES: Pattern and Analysis Tables
-- ============================================================================

-- temporal_patterns
DROP POLICY IF EXISTS "service_role_all_temporal_patterns" ON public.temporal_patterns;
CREATE POLICY "service_role_all_temporal_patterns" ON public.temporal_patterns
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_temporal_patterns" ON public.temporal_patterns;
CREATE POLICY "users_own_temporal_patterns" ON public.temporal_patterns
    FOR ALL USING (auth.uid() = user_id);

-- temporal_anomalies
DROP POLICY IF EXISTS "service_role_all_temporal_anomalies" ON public.temporal_anomalies;
CREATE POLICY "service_role_all_temporal_anomalies" ON public.temporal_anomalies
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_temporal_anomalies" ON public.temporal_anomalies;
CREATE POLICY "users_own_temporal_anomalies" ON public.temporal_anomalies
    FOR ALL USING (auth.uid() = user_id);

-- seasonal_patterns
DROP POLICY IF EXISTS "service_role_all_seasonal_patterns" ON public.seasonal_patterns;
CREATE POLICY "service_role_all_seasonal_patterns" ON public.seasonal_patterns
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_seasonal_patterns" ON public.seasonal_patterns;
CREATE POLICY "users_own_seasonal_patterns" ON public.seasonal_patterns
    FOR ALL USING (auth.uid() = user_id);

-- platform_patterns
DROP POLICY IF EXISTS "service_role_all_platform_patterns" ON public.platform_patterns;
CREATE POLICY "service_role_all_platform_patterns" ON public.platform_patterns
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_platform_patterns" ON public.platform_patterns;
CREATE POLICY "users_own_platform_patterns" ON public.platform_patterns
    FOR ALL USING (auth.uid() = user_id);

-- discovered_platforms
DROP POLICY IF EXISTS "service_role_all_discovered_platforms" ON public.discovered_platforms;
CREATE POLICY "service_role_all_discovered_platforms" ON public.discovered_platforms
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_discovered_platforms" ON public.discovered_platforms;
CREATE POLICY "users_own_discovered_platforms" ON public.discovered_platforms
    FOR ALL USING (auth.uid() = user_id);

-- ============================================================================
-- RLS POLICIES: Advanced Analysis Tables
-- ============================================================================

-- counterfactual_analyses
DROP POLICY IF EXISTS "service_role_all_counterfactual_analyses" ON public.counterfactual_analyses;
CREATE POLICY "service_role_all_counterfactual_analyses" ON public.counterfactual_analyses
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_counterfactual_analyses" ON public.counterfactual_analyses;
CREATE POLICY "users_own_counterfactual_analyses" ON public.counterfactual_analyses
    FOR ALL USING (auth.uid() = user_id);

-- root_cause_analyses
DROP POLICY IF EXISTS "service_role_all_root_cause_analyses" ON public.root_cause_analyses;
CREATE POLICY "service_role_all_root_cause_analyses" ON public.root_cause_analyses
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_root_cause_analyses" ON public.root_cause_analyses;
CREATE POLICY "users_own_root_cause_analyses" ON public.root_cause_analyses
    FOR ALL USING (auth.uid() = user_id);

-- ============================================================================
-- RLS POLICIES: Logging and Metrics Tables
-- ============================================================================

-- detection_log
DROP POLICY IF EXISTS "service_role_all_detection_log" ON public.detection_log;
CREATE POLICY "service_role_all_detection_log" ON public.detection_log
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_detection_log" ON public.detection_log;
CREATE POLICY "users_own_detection_log" ON public.detection_log
    FOR ALL USING (auth.uid() = user_id);

-- resolution_log
DROP POLICY IF EXISTS "service_role_all_resolution_log" ON public.resolution_log;
CREATE POLICY "service_role_all_resolution_log" ON public.resolution_log
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_resolution_log" ON public.resolution_log;
CREATE POLICY "users_own_resolution_log" ON public.resolution_log
    FOR ALL USING (auth.uid() = user_id);

-- event_delta_logs
DROP POLICY IF EXISTS "service_role_all_event_delta_logs" ON public.event_delta_logs;
CREATE POLICY "service_role_all_event_delta_logs" ON public.event_delta_logs
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_event_delta_logs" ON public.event_delta_logs;
CREATE POLICY "users_own_event_delta_logs" ON public.event_delta_logs
    FOR ALL USING (auth.uid() = user_id);

-- error_logs
DROP POLICY IF EXISTS "service_role_all_error_logs" ON public.error_logs;
CREATE POLICY "service_role_all_error_logs" ON public.error_logs
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_error_logs" ON public.error_logs;
CREATE POLICY "users_own_error_logs" ON public.error_logs
    FOR ALL USING (auth.uid() = user_id);

-- metrics
DROP POLICY IF EXISTS "service_role_all_metrics" ON public.metrics;
CREATE POLICY "service_role_all_metrics" ON public.metrics
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_metrics" ON public.metrics;
CREATE POLICY "users_own_metrics" ON public.metrics
    FOR ALL USING (auth.uid() = user_id);

-- performance_metrics
DROP POLICY IF EXISTS "service_role_all_performance_metrics" ON public.performance_metrics;
CREATE POLICY "service_role_all_performance_metrics" ON public.performance_metrics
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_performance_metrics" ON public.performance_metrics;
CREATE POLICY "users_own_performance_metrics" ON public.performance_metrics
    FOR ALL USING (auth.uid() = user_id);

-- ============================================================================
-- RLS POLICIES: Field Mappings and Chat
-- ============================================================================

-- field_mappings
DROP POLICY IF EXISTS "service_role_all_field_mappings" ON public.field_mappings;
CREATE POLICY "service_role_all_field_mappings" ON public.field_mappings
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_field_mappings" ON public.field_mappings;
CREATE POLICY "users_own_field_mappings" ON public.field_mappings
    FOR ALL USING (auth.uid() = user_id);

-- chat_messages
DROP POLICY IF EXISTS "service_role_all_chat_messages" ON public.chat_messages;
CREATE POLICY "service_role_all_chat_messages" ON public.chat_messages
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "users_own_chat_messages" ON public.chat_messages;
CREATE POLICY "users_own_chat_messages" ON public.chat_messages
    FOR ALL USING (auth.uid() = user_id);

-- ============================================================================
-- GRANT PERMISSIONS TO AUTHENTICATED USERS
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
GRANT SELECT, INSERT, UPDATE, DELETE ON public.connectors TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.sync_cursors TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.sync_runs TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.webhook_events TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.relationship_instances TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.cross_platform_relationships TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.relationship_patterns TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.causal_relationships TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.predicted_relationships TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.temporal_patterns TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.temporal_anomalies TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.seasonal_patterns TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.platform_patterns TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.discovered_platforms TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.counterfactual_analyses TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.root_cause_analyses TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.detection_log TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.resolution_log TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.event_delta_logs TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.error_logs TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.metrics TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.performance_metrics TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.field_mappings TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.chat_messages TO authenticated;

-- ============================================================================
-- VERIFICATION QUERY
-- ============================================================================

SELECT 
    tablename,
    rowsecurity as rls_enabled,
    (SELECT COUNT(*) FROM pg_policies 
     WHERE schemaname = 'public' AND tablename = pt.tablename) as policy_count
FROM pg_tables pt
WHERE schemaname = 'public'
ORDER BY tablename;
