-- Debug and fix service role access issues
-- This migration helps us understand what's happening with authentication

-- Create a function to check current authentication context
CREATE OR REPLACE FUNCTION check_auth_context()
RETURNS TABLE(
    current_user_id UUID,
    current_role TEXT,
    is_authenticated BOOLEAN,
    is_service_role BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        auth.uid() as current_user_id,
        auth.role() as current_role,
        auth.uid() IS NOT NULL as is_authenticated,
        auth.role() = 'service_role' as is_service_role;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create a function to test raw_events insertion with service role
CREATE OR REPLACE FUNCTION test_raw_events_insert(
    test_user_id UUID,
    test_file_id UUID,
    test_job_id UUID
)
RETURNS BOOLEAN AS $$
DECLARE
    insert_result RECORD;
BEGIN
    -- Try to insert a test record
    INSERT INTO public.raw_events (
        user_id, file_id, job_id, provider, kind, source_platform,
        payload, row_index, sheet_name, source_filename, uploader,
        status, confidence_score
    ) VALUES (
        test_user_id, test_file_id, test_job_id, 'test', 'test_row', 'test',
        '{"test": "data"}'::jsonb, 0, 'test_sheet', 'test_file.xlsx', test_user_id,
        'pending', 0.5
    ) RETURNING id INTO insert_result;
    
    -- If we get here, the insert was successful
    RETURN TRUE;
    
EXCEPTION WHEN OTHERS THEN
    -- Log the error details
    RAISE NOTICE 'Insert failed: %', SQLERRM;
    RETURN FALSE;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Grant execute permissions
GRANT EXECUTE ON FUNCTION check_auth_context() TO service_role;
GRANT EXECUTE ON FUNCTION test_raw_events_insert(UUID, UUID, UUID) TO service_role; 