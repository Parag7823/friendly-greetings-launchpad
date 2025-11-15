-- Migration: Delete unnecessary tables and test data
-- Date: 2025-11-15
-- Purpose: Remove unused chat_messages, test tables, and integration test artifacts

-- Drop chat_messages table and all references
DROP POLICY IF EXISTS "service_role_all_chat_messages" ON public.chat_messages;
DROP POLICY IF EXISTS "users_own_chat_messages" ON public.chat_messages;
DROP POLICY IF EXISTS "Users can view their own chat messages" ON public.chat_messages;
DROP POLICY IF EXISTS "Users can insert their own chat messages" ON public.chat_messages;
DROP POLICY IF EXISTS "Users can update their own chat messages" ON public.chat_messages;
DROP POLICY IF EXISTS "Users can delete their own chat messages" ON public.chat_messages;

DROP TRIGGER IF EXISTS update_chat_messages_updated_at ON public.chat_messages;

DROP INDEX IF EXISTS idx_chat_messages_user_id;
DROP INDEX IF EXISTS idx_chat_messages_chat_id;
DROP INDEX IF EXISTS idx_chat_messages_created_at;
DROP INDEX IF EXISTS idx_chat_messages_user_chat;
DROP INDEX IF EXISTS idx_chat_messages_user_chat_created;

DROP TABLE IF EXISTS public.chat_messages CASCADE;

-- Drop chat_sessions table if it exists
DROP POLICY IF EXISTS "service_role_all_chat_sessions" ON public.chat_sessions;
DROP POLICY IF EXISTS "users_own_chat_sessions" ON public.chat_sessions;
DROP TABLE IF EXISTS public.chat_sessions CASCADE;

-- Drop integration test tables
DROP POLICY IF EXISTS "Allow integration tests" ON public.integration_test_logs;
DROP TABLE IF EXISTS public.integration_test_logs CASCADE;

-- Remove test user data (keep the auth.users entry but clean up related data)
-- Note: We don't delete from auth.users as that's managed by Supabase Auth

-- Drop any test-specific functions
DROP FUNCTION IF EXISTS create_test_chat_message(UUID, TEXT, TEXT);
DROP FUNCTION IF EXISTS get_chat_history(UUID, TEXT);
DROP FUNCTION IF EXISTS cleanup_test_data();

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'Migration 20251115000004: Successfully deleted unnecessary tables';
    RAISE NOTICE 'Removed: chat_messages, chat_sessions, integration_test_logs';
    RAISE NOTICE 'Removed: All related policies, triggers, indexes, and functions';
END $$;
