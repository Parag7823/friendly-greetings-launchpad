-- Add infrastructure for partial duplicate detection using row hashes and Jaccard similarity

-- Step 1: Create the file_row_hashes table
CREATE TABLE IF NOT EXISTS public.file_row_hashes (
    id BIGSERIAL PRIMARY KEY,
    file_id UUID NOT NULL REFERENCES public.raw_records(id) ON DELETE CASCADE,
    row_hash TEXT NOT NULL
);

-- Step 2: Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_file_row_hashes_file_id ON public.file_row_hashes(file_id);
CREATE INDEX IF NOT EXISTS idx_file_row_hashes_row_hash ON public.file_row_hashes(row_hash);

-- Step 3: Enable RLS on the new table
ALTER TABLE public.file_row_hashes ENABLE ROW LEVEL SECURITY;

-- Step 4: Add RLS policies for the new table
CREATE POLICY "Allow service role full access on file_row_hashes" 
ON public.file_row_hashes FOR ALL
USING (auth.role() = 'service_role');

CREATE POLICY "Users can manage their own file_row_hashes" 
ON public.file_row_hashes FOR ALL
USING (EXISTS (
    SELECT 1 FROM public.raw_records
    WHERE id = file_id AND user_id = auth.uid()
));

-- Step 5: Create a function to calculate Jaccard similarity between two files
CREATE OR REPLACE FUNCTION jaccard_similarity(p_file_id_1 UUID, p_file_id_2 UUID)
RETURNS REAL AS $$
DECLARE
    intersection_count INT;
    union_count INT;
    total_count_1 INT;
    total_count_2 INT;
BEGIN
    -- Count hashes in the first file
    SELECT COUNT(*) INTO total_count_1 FROM public.file_row_hashes WHERE file_id = p_file_id_1;

    -- Count hashes in the second file
    SELECT COUNT(*) INTO total_count_2 FROM public.file_row_hashes WHERE file_id = p_file_id_2;

    -- Calculate the intersection of row hashes
    SELECT COUNT(*)
    INTO intersection_count
    FROM (
        SELECT row_hash FROM public.file_row_hashes WHERE file_id = p_file_id_1
        INTERSECT
        SELECT row_hash FROM public.file_row_hashes WHERE file_id = p_file_id_2
    ) AS intersection;

    -- Calculate the union count
    union_count := total_count_1 + total_count_2 - intersection_count;

    -- Avoid division by zero
    IF union_count = 0 THEN
        RETURN 0;
    END IF;

    -- Return Jaccard similarity
    RETURN intersection_count::REAL / union_count::REAL;
END;
$$ LANGUAGE plpgsql;
