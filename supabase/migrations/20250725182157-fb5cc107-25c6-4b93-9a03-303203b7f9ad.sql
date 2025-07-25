-- Remove the foreign key constraint to auth.users
ALTER TABLE public.ingestion_jobs DROP CONSTRAINT IF EXISTS ingestion_jobs_user_id_fkey;

-- Remove the foreign key constraint to raw_records as well to simplify
ALTER TABLE public.ingestion_jobs DROP CONSTRAINT IF EXISTS ingestion_jobs_record_id_fkey;