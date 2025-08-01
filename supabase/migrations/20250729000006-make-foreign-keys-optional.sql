-- Make foreign key constraints optional for raw_events table
-- This allows the backend to work even if job_id or file_id references don't exist

-- Drop existing foreign key constraints
ALTER TABLE public.raw_events DROP CONSTRAINT IF EXISTS raw_events_job_id_fkey;
ALTER TABLE public.raw_events DROP CONSTRAINT IF EXISTS raw_events_file_id_fkey;

-- Recreate foreign key constraints as optional (ON DELETE SET NULL)
ALTER TABLE public.raw_events 
ADD CONSTRAINT raw_events_job_id_fkey 
FOREIGN KEY (job_id) REFERENCES public.ingestion_jobs(id) ON DELETE SET NULL;

ALTER TABLE public.raw_events 
ADD CONSTRAINT raw_events_file_id_fkey 
FOREIGN KEY (file_id) REFERENCES public.raw_records(id) ON DELETE SET NULL;

-- Also make the columns nullable if they aren't already
ALTER TABLE public.raw_events ALTER COLUMN job_id DROP NOT NULL;
ALTER TABLE public.raw_events ALTER COLUMN file_id DROP NOT NULL; 