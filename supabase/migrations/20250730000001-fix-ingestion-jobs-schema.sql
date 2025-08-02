-- Fix ingestion_jobs table schema to match code expectations
-- Add missing file_id column and update table structure

-- Add file_id column to ingestion_jobs table
ALTER TABLE public.ingestion_jobs 
ADD COLUMN file_id UUID REFERENCES public.raw_records(id) ON DELETE SET NULL;

-- Add created_at and updated_at columns if they don't exist
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'ingestion_jobs' AND column_name = 'created_at') THEN
        ALTER TABLE public.ingestion_jobs ADD COLUMN created_at TIMESTAMP WITH TIME ZONE DEFAULT now();
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'ingestion_jobs' AND column_name = 'updated_at') THEN
        ALTER TABLE public.ingestion_jobs ADD COLUMN updated_at TIMESTAMP WITH TIME ZONE DEFAULT now();
    END IF;
END $$;

-- Create trigger for updated_at if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_ingestion_jobs_updated_at') THEN
        CREATE TRIGGER update_ingestion_jobs_updated_at
            BEFORE UPDATE ON public.ingestion_jobs
            FOR EACH ROW
            EXECUTE FUNCTION public.update_updated_at_column();
    END IF;
END $$;

-- Add index for file_id
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_file_id ON public.ingestion_jobs(file_id);

-- Update RLS policies to allow service role access
DROP POLICY IF EXISTS "Users can view their own jobs" ON public.ingestion_jobs;
DROP POLICY IF EXISTS "Users can insert their own jobs" ON public.ingestion_jobs;
DROP POLICY IF EXISTS "Users can update their own jobs" ON public.ingestion_jobs;

-- Create new policies that allow service role access
CREATE POLICY "_service_access" ON public.ingestion_jobs
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "_user_access" ON public.ingestion_jobs
    FOR ALL USING (auth.uid() = user_id);

-- Ensure the table has the correct structure
COMMENT ON TABLE public.ingestion_jobs IS 'Tracks async processing jobs for file ingestion';
COMMENT ON COLUMN public.ingestion_jobs.file_id IS 'Reference to the file being processed';
COMMENT ON COLUMN public.ingestion_jobs.user_id IS 'User who owns this job';
COMMENT ON COLUMN public.ingestion_jobs.status IS 'Current status of the job'; 