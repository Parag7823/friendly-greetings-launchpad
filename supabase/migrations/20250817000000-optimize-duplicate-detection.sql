-- Optimize duplicate detection by adding a normalized_filename column and using pg_trgm for similarity search

-- Step 1: Add the normalized_filename column to raw_records
ALTER TABLE public.raw_records
ADD COLUMN IF NOT EXISTS normalized_filename TEXT;

-- Step 2: Enable the pg_trgm extension for trigram similarity
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Step 3: Create an index on the new column using GiST for trigram operations
CREATE INDEX IF NOT EXISTS idx_raw_records_normalized_filename_gist ON public.raw_records USING gist (normalized_filename gist_trgm_ops);

-- Step 4: Create a function to normalize filenames (mirroring the Python logic)
CREATE OR REPLACE FUNCTION normalize_filename(filename TEXT)
RETURNS TEXT AS $$
BEGIN
    -- Remove common version patterns and normalize
    RETURN regexp_replace(
        regexp_replace(
            regexp_replace(
                lower(trim(filename)),
                '\.[^.]+$', '' -- remove extension
            ),
            '_v\d+|_version\d+|_final|_draft|_copy|\(\d+\)|_\d+$', '', 'g'
        ),
        '\s+', '_', 'g'
    );
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Step 5: Backfill the normalized_filename for existing records
UPDATE public.raw_records
SET normalized_filename = normalize_filename(file_name)
WHERE normalized_filename IS NULL;

-- Step 6: Create the database function to find similar files
CREATE OR REPLACE FUNCTION find_similar_files_by_name(p_user_id UUID, p_normalized_filename TEXT)
RETURNS TABLE (
    id UUID,
    file_name TEXT,
    file_hash TEXT,
    created_at TIMESTAMP WITH TIME ZONE,
    content JSONB,
    similarity REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        r.id,
        r.file_name,
        r.file_hash,
        r.created_at,
        r.content,
        similarity(r.normalized_filename, p_normalized_filename)::REAL AS similarity
    FROM
        public.raw_records r
    WHERE
        r.user_id = p_user_id
        AND r.normalized_filename % p_normalized_filename
    ORDER BY
        similarity DESC
    LIMIT 10;
END;
$$ LANGUAGE plpgsql;
