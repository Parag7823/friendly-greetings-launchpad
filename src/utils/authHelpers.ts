import { supabase } from '@/integrations/supabase/client';

/**
 * Centralized session token retrieval
 * Prevents duplication across ChatInterface, DataSourcesPanel, FastAPIProcessor
 */
export const getSessionToken = async (): Promise<string | null> => {
  try {
    const { data: { session } } = await supabase.auth.getSession();
    return session?.access_token ?? null;
  } catch (error) {
    console.error('Failed to get session token:', error);
    return null;
  }
};

/**
 * Get both user and session in one call
 * Useful for components that need both
 */
export const getUserAndSession = async () => {
  try {
    const { data: { user } } = await supabase.auth.getUser();
    const { data: { session } } = await supabase.auth.getSession();
    return { user, session };
  } catch (error) {
    console.error('Failed to get user and session:', error);
    return { user: null, session: null };
  }
};
