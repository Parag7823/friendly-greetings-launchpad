-- Fix RLS policies for raw_events table to allow service role operations
-- This migration updates the RLS policies to work with both authenticated users and service role

-- Drop existing policies
DROP POLICY IF EXISTS "Users can view their own events" ON public.raw_events;
DROP POLICY IF EXISTS "Users can insert their own events" ON public.raw_events;
DROP POLICY IF EXISTS "Users can update their own events" ON public.raw_events;

-- Create new policies that work with both authenticated users and service role
CREATE POLICY "Users can view their own events" ON public.raw_events
    FOR SELECT USING (
        auth.uid() = user_id OR 
        auth.role() = 'service_role'
    );

CREATE POLICY "Users can insert their own events" ON public.raw_events
    FOR INSERT WITH CHECK (
        auth.uid() = user_id OR 
        auth.role() = 'service_role'
    );

CREATE POLICY "Users can update their own events" ON public.raw_events
    FOR UPDATE USING (
        auth.uid() = user_id OR 
        auth.role() = 'service_role'
    );

-- Add a policy for service role to delete (for cleanup operations)
CREATE POLICY "Service role can delete events" ON public.raw_events
    FOR DELETE USING (auth.role() = 'service_role'); 