-- ERROR #3 FIX: Add source column to ingestion_jobs table
-- Tracks file source (quickbooks, dropbox, google-drive, manual)
-- Required for FileCard source badges and file tracking

-- Add source column with default value
ALTER TABLE public.ingestion_jobs 
ADD COLUMN IF NOT EXISTS source VARCHAR(50) DEFAULT 'manual';

-- Add index for efficient filtering by source
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_source ON public.ingestion_jobs(source);

-- Add comment for documentation
COMMENT ON COLUMN public.ingestion_jobs.source IS 'Source of the file: quickbooks, dropbox, google-drive, manual, etc.';

-- Update existing rows to have source if null
UPDATE public.ingestion_jobs 
SET source = 'manual' 
WHERE source IS NULL;

-- Make column NOT NULL after setting defaults
ALTER TABLE public.ingestion_jobs 
ALTER COLUMN source SET NOT NULL;
