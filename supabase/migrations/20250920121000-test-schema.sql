-- Test Schema Validation
-- Quick test to ensure all referenced columns exist before running the main migration

-- Test if raw_events table exists and has expected columns
DO $$
BEGIN
    -- Check if raw_events table exists
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables 
                   WHERE table_name = 'raw_events' AND table_schema = 'public') THEN
        RAISE EXCEPTION 'raw_events table does not exist';
    END IF;
    
    -- Check critical columns exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'raw_events' AND column_name = 'user_id' AND table_schema = 'public') THEN
        RAISE EXCEPTION 'raw_events.user_id column does not exist';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'raw_events' AND column_name = 'source_platform' AND table_schema = 'public') THEN
        RAISE EXCEPTION 'raw_events.source_platform column does not exist';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'raw_events' AND column_name = 'ingest_ts' AND table_schema = 'public') THEN
        RAISE EXCEPTION 'raw_events.ingest_ts column does not exist';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'raw_events' AND column_name = 'payload' AND table_schema = 'public') THEN
        RAISE EXCEPTION 'raw_events.payload column does not exist';
    END IF;
    
    -- Check if raw_records table exists
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables 
                   WHERE table_name = 'raw_records' AND table_schema = 'public') THEN
        RAISE EXCEPTION 'raw_records table does not exist';
    END IF;
    
    -- Check critical columns exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'raw_records' AND column_name = 'user_id' AND table_schema = 'public') THEN
        RAISE EXCEPTION 'raw_records.user_id column does not exist';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'raw_records' AND column_name = 'content' AND table_schema = 'public') THEN
        RAISE EXCEPTION 'raw_records.content column does not exist';
    END IF;
    
    RAISE NOTICE 'Schema validation passed! All required tables and columns exist.';
END $$;
