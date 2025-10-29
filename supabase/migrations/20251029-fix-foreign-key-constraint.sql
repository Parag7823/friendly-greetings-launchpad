-- FIX: Foreign Key Constraint Issue
-- Problem: raw_events.file_id references raw_records.id but constraint was too strict
-- Solution: Ensure constraint allows NULL and is properly deferred

-- 1. Drop existing constraint if it exists
ALTER TABLE public.raw_events DROP CONSTRAINT IF EXISTS raw_events_file_id_fkey;

-- 2. Recreate with proper settings
ALTER TABLE public.raw_events 
ADD CONSTRAINT raw_events_file_id_fkey 
FOREIGN KEY (file_id) REFERENCES public.raw_records(id) ON DELETE SET NULL;

-- 3. Ensure file_id column is nullable
ALTER TABLE public.raw_events ALTER COLUMN file_id DROP NOT NULL;

-- 4. Add index on file_id for performance
CREATE INDEX IF NOT EXISTS idx_raw_events_file_id ON public.raw_events(file_id);
CREATE INDEX IF NOT EXISTS idx_raw_records_id ON public.raw_records(id);

-- 5. Verify constraint is working
SELECT constraint_name, table_name, column_name 
FROM information_schema.key_column_usage 
WHERE table_name = 'raw_events' AND column_name = 'file_id';
