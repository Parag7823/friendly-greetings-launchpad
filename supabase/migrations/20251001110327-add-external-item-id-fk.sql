-- Add external_item_id to raw_records and create FK to external_items(id)
-- Run-time: 2025-10-01 11:03:27 +05:30

BEGIN;

-- 1) Add column if it doesn't exist
ALTER TABLE public.raw_records
ADD COLUMN IF NOT EXISTS external_item_id UUID;

-- 2) Create index for faster joins/lookups
CREATE INDEX IF NOT EXISTS idx_raw_records_external_item_id
  ON public.raw_records(external_item_id);

-- 3) Add foreign key constraint if not already present
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.table_constraints tc
        WHERE tc.constraint_type = 'FOREIGN KEY'
          AND tc.table_schema = 'public'
          AND tc.table_name = 'raw_records'
          AND tc.constraint_name = 'fk_raw_records_external_item'
    ) THEN
        ALTER TABLE public.raw_records
        ADD CONSTRAINT fk_raw_records_external_item
        FOREIGN KEY (external_item_id)
        REFERENCES public.external_items(id)
        ON DELETE SET NULL;
    END IF;
END
$$;

COMMIT;
