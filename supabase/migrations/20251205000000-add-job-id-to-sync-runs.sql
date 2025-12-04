-- ============================================================================
-- CRITICAL BUG FIX #1: Add job_id column to sync_runs table
-- ============================================================================
-- Issue: Webhook handler queries sync_runs by job_id, but column doesn't exist
-- Impact: Airbyte webhooks fail to update sync status
-- Solution: Add job_id column to track Airbyte job IDs

-- Add job_id column to sync_runs
ALTER TABLE public.sync_runs 
ADD COLUMN IF NOT EXISTS job_id TEXT UNIQUE;

-- Add index for webhook lookups by job_id
CREATE INDEX IF NOT EXISTS idx_sync_runs_job_id ON public.sync_runs(job_id);

-- Add comment explaining the column
COMMENT ON COLUMN public.sync_runs.job_id IS 
'Airbyte job ID from trigger_sync response. Used to match webhook events to sync runs.';
