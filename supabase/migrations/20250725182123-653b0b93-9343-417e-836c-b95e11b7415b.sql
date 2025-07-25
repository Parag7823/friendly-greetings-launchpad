-- Remove any check constraints on the ingestion_jobs table
ALTER TABLE public.ingestion_jobs DROP CONSTRAINT IF EXISTS ingestion_jobs_status_check;

-- Also check for any other constraints that might be causing issues
-- Let's see what constraints exist
SELECT conname, contype, pg_get_constraintdef(oid) 
FROM pg_constraint 
WHERE conrelid = 'public.ingestion_jobs'::regclass;