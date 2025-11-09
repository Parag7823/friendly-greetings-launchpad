-- Migration: Add job_id to all tables for unified tracking
-- Issue: Tables are fragmented between job_id and transaction_id, making it impossible to delete all data for a file
-- Solution: Add job_id to ALL tables so we can query/delete by either job_id OR transaction_id

-- Add job_id to tables that only have transaction_id
ALTER TABLE IF EXISTS public.normalized_entities 
ADD COLUMN IF NOT EXISTS job_id UUID REFERENCES public.ingestion_jobs(id) ON DELETE CASCADE;

ALTER TABLE IF EXISTS public.entity_matches 
ADD COLUMN IF NOT EXISTS job_id UUID REFERENCES public.ingestion_jobs(id) ON DELETE CASCADE;

ALTER TABLE IF EXISTS public.platform_patterns 
ADD COLUMN IF NOT EXISTS job_id UUID REFERENCES public.ingestion_jobs(id) ON DELETE CASCADE;

ALTER TABLE IF EXISTS public.metrics 
ADD COLUMN IF NOT EXISTS job_id UUID REFERENCES public.ingestion_jobs(id) ON DELETE CASCADE;

ALTER TABLE IF EXISTS public.cross_platform_relationships 
ADD COLUMN IF NOT EXISTS job_id UUID REFERENCES public.ingestion_jobs(id) ON DELETE CASCADE;

ALTER TABLE IF EXISTS public.relationship_instances 
ADD COLUMN IF NOT EXISTS job_id UUID REFERENCES public.ingestion_jobs(id) ON DELETE CASCADE;

ALTER TABLE IF EXISTS public.predicted_relationships 
ADD COLUMN IF NOT EXISTS job_id UUID REFERENCES public.ingestion_jobs(id) ON DELETE CASCADE;

ALTER TABLE IF EXISTS public.causal_relationships 
ADD COLUMN IF NOT EXISTS job_id UUID REFERENCES public.ingestion_jobs(id) ON DELETE CASCADE;

ALTER TABLE IF EXISTS public.temporal_patterns 
ADD COLUMN IF NOT EXISTS job_id UUID REFERENCES public.ingestion_jobs(id) ON DELETE CASCADE;

ALTER TABLE IF EXISTS public.seasonal_patterns 
ADD COLUMN IF NOT EXISTS job_id UUID REFERENCES public.ingestion_jobs(id) ON DELETE CASCADE;

ALTER TABLE IF EXISTS public.temporal_anomalies 
ADD COLUMN IF NOT EXISTS job_id UUID REFERENCES public.ingestion_jobs(id) ON DELETE CASCADE;

ALTER TABLE IF EXISTS public.root_cause_analyses 
ADD COLUMN IF NOT EXISTS job_id UUID REFERENCES public.ingestion_jobs(id) ON DELETE CASCADE;

ALTER TABLE IF EXISTS public.counterfactual_analyses 
ADD COLUMN IF NOT EXISTS job_id UUID REFERENCES public.ingestion_jobs(id) ON DELETE CASCADE;

ALTER TABLE IF EXISTS public.duplicate_transactions 
ADD COLUMN IF NOT EXISTS job_id UUID REFERENCES public.ingestion_jobs(id) ON DELETE CASCADE;

ALTER TABLE IF EXISTS public.event_delta_logs 
ADD COLUMN IF NOT EXISTS job_id UUID REFERENCES public.ingestion_jobs(id) ON DELETE CASCADE;

-- Create indexes for efficient job-based queries
CREATE INDEX IF NOT EXISTS idx_normalized_entities_job_id ON public.normalized_entities(job_id) WHERE job_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_entity_matches_job_id ON public.entity_matches(job_id) WHERE job_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_platform_patterns_job_id ON public.platform_patterns(job_id) WHERE job_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_metrics_job_id ON public.metrics(job_id) WHERE job_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_cross_platform_relationships_job_id ON public.cross_platform_relationships(job_id) WHERE job_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_relationship_instances_job_id ON public.relationship_instances(job_id) WHERE job_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_predicted_relationships_job_id ON public.predicted_relationships(job_id) WHERE job_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_causal_relationships_job_id ON public.causal_relationships(job_id) WHERE job_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_temporal_patterns_job_id ON public.temporal_patterns(job_id) WHERE job_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_seasonal_patterns_job_id ON public.seasonal_patterns(job_id) WHERE job_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_temporal_anomalies_job_id ON public.temporal_anomalies(job_id) WHERE job_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_root_cause_analyses_job_id ON public.root_cause_analyses(job_id) WHERE job_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_counterfactual_analyses_job_id ON public.counterfactual_analyses(job_id) WHERE job_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_duplicate_transactions_job_id ON public.duplicate_transactions(job_id) WHERE job_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_event_delta_logs_job_id ON public.event_delta_logs(job_id) WHERE job_id IS NOT NULL;

-- Add comments explaining the purpose
COMMENT ON COLUMN public.normalized_entities.job_id IS 'Links to ingestion_jobs for unified tracking and deletion';
COMMENT ON COLUMN public.entity_matches.job_id IS 'Links to ingestion_jobs for unified tracking and deletion';
COMMENT ON COLUMN public.platform_patterns.job_id IS 'Links to ingestion_jobs for unified tracking and deletion';
COMMENT ON COLUMN public.metrics.job_id IS 'Links to ingestion_jobs for unified tracking and deletion';
COMMENT ON COLUMN public.cross_platform_relationships.job_id IS 'Links to ingestion_jobs for unified tracking and deletion';
COMMENT ON COLUMN public.relationship_instances.job_id IS 'Links to ingestion_jobs for unified tracking and deletion';
COMMENT ON COLUMN public.predicted_relationships.job_id IS 'Links to ingestion_jobs for unified tracking and deletion';
COMMENT ON COLUMN public.causal_relationships.job_id IS 'Links to ingestion_jobs for unified tracking and deletion';
COMMENT ON COLUMN public.temporal_patterns.job_id IS 'Links to ingestion_jobs for unified tracking and deletion';
COMMENT ON COLUMN public.seasonal_patterns.job_id IS 'Links to ingestion_jobs for unified tracking and deletion';
COMMENT ON COLUMN public.temporal_anomalies.job_id IS 'Links to ingestion_jobs for unified tracking and deletion';
COMMENT ON COLUMN public.root_cause_analyses.job_id IS 'Links to ingestion_jobs for unified tracking and deletion';
COMMENT ON COLUMN public.counterfactual_analyses.job_id IS 'Links to ingestion_jobs for unified tracking and deletion';
COMMENT ON COLUMN public.duplicate_transactions.job_id IS 'Links to ingestion_jobs for unified tracking and deletion';
COMMENT ON COLUMN public.event_delta_logs.job_id IS 'Links to ingestion_jobs for unified tracking and deletion';
