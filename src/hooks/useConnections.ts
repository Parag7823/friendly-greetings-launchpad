import { useQuery, useQueryClient } from '@tanstack/react-query';
import { supabase } from '@/integrations/supabase/client';
import { config } from '@/config';
import { useAuth } from '@/components/AuthProvider';

interface Connection {
  connection_id: string;
  integration_id: string;
  provider: string;
  status: string;
  last_synced_at: string | null;
  created_at: string;
}

/**
 * Shared hook for fetching user connections
 * Prevents duplicate polling and provides single source of truth
 */
export const useConnections = () => {
  const { user } = useAuth();
  const queryClient = useQueryClient();

  return useQuery({
    queryKey: ['connections', user?.id],
    queryFn: async (): Promise<Connection[]> => {
      if (!user?.id) return [];

      const { data: sessionData } = await supabase.auth.getSession();
      const sessionToken = sessionData?.session?.access_token;

      const response = await fetch(`${config.apiUrl}/api/connectors/user-connections`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: user.id,
          session_token: sessionToken
        })
      });

      if (!response.ok) {
        throw new Error('Failed to fetch connections');
      }

      const data = await response.json();
      const connections = data.connections || [];

      return connections.map((connection: Connection) => ({
        ...connection,
        provider: connection.provider || connection.integration_id,
        integration_id: mapIntegrationId(connection.integration_id),
      }));
    },
    enabled: !!user?.id,
    refetchInterval: 300000, // BUG #7 FIX: Changed from 30s to 300s (5 min). Reduces DB load from 4,000 to 200 req/min for 1000 users
    staleTime: 60000, // Consider data fresh for 60 seconds (connections don't change frequently)
    retry: 2,
  });
};

const mapIntegrationId = (integrationId?: string) => {
  switch (integrationId) {
    case 'google-mail':
      return 'gmail';
    default:
      return integrationId;
  }
};

/**
 * Hook to manually refresh connections
 */
export const useRefreshConnections = () => {
  const { user } = useAuth();
  const queryClient = useQueryClient();

  return () => {
    queryClient.invalidateQueries({ queryKey: ['connections', user?.id] });
  };
};
