import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { FileSpreadsheet, Mail, Database, BookOpen, DollarSign, FileText, HardDrive, Cloud, CreditCard, Wallet, Banknote } from "lucide-react";
import IntegrationCard from "@/components/IntegrationCard";
import { useAuth } from "@/components/AuthProvider";
import { config } from "@/config";
import { useToast } from "@/hooks/use-toast";

interface Provider {
  provider: string;
  display_name: string;
  integration_id: string;
  auth_type: string;
  scopes: string[];
  endpoints: string[];
  category?: string;
}

interface UserConnection {
  connection_id: string;
  integration_id: string;
  status: string;
  last_synced_at: string | null;
  created_at: string;
}

const Integrations = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const { toast } = useToast();
  const [providers, setProviders] = useState<Provider[]>([]);
  const [connections, setConnections] = useState<UserConnection[]>([]);
  const [loading, setLoading] = useState(true);
  const [connectingProvider, setConnectingProvider] = useState<string | null>(null);

  useEffect(() => {
    document.title = "Integrations — Finley AI";
    if (user) {
      fetchProvidersAndConnections();
    }
  }, [user]);

  const fetchProvidersAndConnections = async () => {
    try {
      setLoading(true);
      const token = await (window as any).getAuthToken?.();
      
      // Fetch available providers
      const providersResp = await fetch(`${config.apiUrl}/api/connectors/providers`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token && { 'Authorization': `Bearer ${token}` })
        },
        body: JSON.stringify({
          user_id: user?.id || '',
          session_token: token
        })
      });
      
      if (providersResp.ok) {
        const providersData = await providersResp.json();
        setProviders(providersData.providers || []);
      }

      // Fetch user connections
      const connectionsResp = await fetch(`${config.apiUrl}/api/connectors/user-connections`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token && { 'Authorization': `Bearer ${token}` })
        },
        body: JSON.stringify({
          user_id: user?.id || '',
          session_token: token
        })
      });
      
      if (connectionsResp.ok) {
        const connectionsData = await connectionsResp.json();
        setConnections(connectionsData.connections || []);
      }
    } catch (error) {
      console.error('Failed to fetch integrations:', error);
      toast({
        title: "Error",
        description: "Failed to load integrations. Please try again.",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };

  const handleConnect = async (provider: Provider) => {
    try {
      setConnectingProvider(provider.provider);
      const token = await (window as any).getAuthToken?.();
      
      const resp = await fetch(`${config.apiUrl}/api/connectors/initiate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token && { 'Authorization': `Bearer ${token}` })
        },
        body: JSON.stringify({
          provider: provider.provider,
          user_id: user?.id || '',
          session_token: token
        })
      });

      if (!resp.ok) {
        throw new Error('Failed to initiate connection');
      }

      const data = await resp.json();
      const session = data?.connect_session || {};
      const url = session.connect_url || session.url || session.authorization_url || session.hosted_url;
      
      if (!url) {
        console.error('No authorization URL in response:', data);
        throw new Error('No authorization URL received from server');
      }
      
      // Open Nango Connect UI in new window
      const connectWindow = window.open(url as string, '_blank', 'width=600,height=800,noopener,noreferrer');
      
      if (!connectWindow) {
        throw new Error('Failed to open authorization window. Please allow popups for this site.');
      }
      
      toast({
        title: "Connection Initiated",
        description: `Opening ${provider.display_name} authorization window...`,
      });
      
      // Refresh connections after a delay to check if connection was successful
      setTimeout(() => {
        fetchProvidersAndConnections();
      }, 3000);
    } catch (error) {
      console.error('Connection error:', error);
      toast({
        title: "Connection Failed",
        description: `Failed to connect to ${provider.display_name}. Please try again.`,
        variant: "destructive"
      });
    } finally {
      setConnectingProvider(null);
    }
  };

  const handleManageConnection = (connection: UserConnection) => {
    navigate(`/sync-history/${connection.connection_id}`);
  };

  const handleExcelAction = () => {
    navigate("/");
    setTimeout(() => {
      window.dispatchEvent(new Event("open-excel-upload"));
    }, 0);
  };

  const getProviderIcon = (provider: string) => {
    const iconMap: Record<string, JSX.Element> = {
      'google-mail': <Mail className="w-8 h-8 text-red-600" />,
      'zoho-mail': <Mail className="w-8 h-8 text-blue-600" />,
      'dropbox': <Cloud className="w-8 h-8 text-blue-500" />,
      'google-drive': <HardDrive className="w-8 h-8 text-yellow-600" />,
      'zoho-books': <BookOpen className="w-8 h-8 text-orange-600" />,
      'quickbooks-sandbox': <DollarSign className="w-8 h-8 text-green-600" />,
      'xero': <FileText className="w-8 h-8 text-blue-700" />,
      'stripe': <CreditCard className="w-8 h-8 text-purple-600" />,
      'razorpay': <Wallet className="w-8 h-8 text-blue-800" />,
      'paypal': <Banknote className="w-8 h-8 text-blue-600" />,
    };
    return iconMap[provider] || <Database className="w-8 h-8 text-gray-600" />;
  };

  const isConnected = (integrationId: string) => {
    return connections.some(conn => conn.integration_id === integrationId && conn.status === 'active');
  };

  const getConnection = (integrationId: string) => {
    return connections.find(conn => conn.integration_id === integrationId);
  };

  if (loading) {
    return (
      <main className="w-full p-6">
        <div className="text-center text-muted-foreground">Loading integrations...</div>
      </main>
    );
  }

  return (
    <main className="w-full">
      {connections.length === 0 && (
        <div className="px-6 pt-6">
          <div className="finley-card rounded-md border border-border bg-card text-card-foreground p-4">
            Connect integrations to automate your financial workflows.
          </div>
        </div>
      )}

      {connections.length > 0 && (
        <section className="p-6">
          <h2 className="text-lg font-semibold mb-4">Connected Integrations</h2>
          <div className="space-y-3">
            {connections.map((conn) => {
              const provider = providers.find(p => p.integration_id === conn.integration_id);
              if (!provider) return null;
              
              return (
                <IntegrationCard
                  key={conn.connection_id}
                  variant="list"
                  icon={getProviderIcon(provider.provider)}
                  title={provider.display_name}
                  description={`Connected ${conn.last_synced_at ? `• Last synced: ${new Date(conn.last_synced_at).toLocaleDateString()}` : ''}`}
                  actionLabel="Manage"
                  onAction={() => handleManageConnection(conn)}
                  statusLabel={conn.status === 'active' ? 'Active' : conn.status}
                />
              );
            })}
          </div>
        </section>
      )}

      {/* Payment Gateways Section */}
      <section className="p-6">
        <h2 className="text-lg font-semibold mb-4">Payment Gateways</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {providers.filter(p => p.category === 'payment').map((provider) => {
            const connected = isConnected(provider.integration_id);
            const connection = getConnection(provider.integration_id);
            
            return (
              <IntegrationCard
                key={provider.provider}
                icon={getProviderIcon(provider.provider)}
                title={provider.display_name}
                description={`Connect your ${provider.display_name} account to sync payment data automatically.`}
                actionLabel={connected ? "Manage" : "Connect"}
                onAction={() => connected && connection ? handleManageConnection(connection) : handleConnect(provider)}
                disabled={connectingProvider === provider.provider}
                statusLabel={connected ? "Connected" : undefined}
              />
            );
          })}
        </div>
      </section>

      {/* Accounting Platforms Section */}
      <section className="p-6">
        <h2 className="text-lg font-semibold mb-4">Accounting Platforms</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {providers.filter(p => p.category === 'accounting').map((provider) => {
            const connected = isConnected(provider.integration_id);
            const connection = getConnection(provider.integration_id);
            
            return (
              <IntegrationCard
                key={provider.provider}
                icon={getProviderIcon(provider.provider)}
                title={provider.display_name}
                description={`Connect your ${provider.display_name} account to sync accounting data automatically.`}
                actionLabel={connected ? "Manage" : "Connect"}
                onAction={() => connected && connection ? handleManageConnection(connection) : handleConnect(provider)}
                disabled={connectingProvider === provider.provider}
                statusLabel={connected ? "Connected" : undefined}
              />
            );
          })}
        </div>
      </section>

      {/* Cloud & Storage Section */}
      <section className="p-6">
        <h2 className="text-lg font-semibold mb-4">Cloud & Storage</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          <IntegrationCard
            icon={<FileSpreadsheet className="w-8 h-8 text-green-600" aria-hidden />}
            title="Excel Integration"
            description="Upload and process spreadsheets directly from Excel or CSV files."
            actionLabel="Upload File"
            onAction={handleExcelAction}
          />

          {providers.filter(p => p.category === 'storage' || p.category === 'email').map((provider) => {
            const connected = isConnected(provider.integration_id);
            const connection = getConnection(provider.integration_id);
            
            return (
              <IntegrationCard
                key={provider.provider}
                icon={getProviderIcon(provider.provider)}
                title={provider.display_name}
                description={`Connect your ${provider.display_name} account to sync files and data automatically.`}
                actionLabel={connected ? "Manage" : "Connect"}
                onAction={() => connected && connection ? handleManageConnection(connection) : handleConnect(provider)}
                disabled={connectingProvider === provider.provider}
                statusLabel={connected ? "Connected" : undefined}
              />
            );
          })}
        </div>
      </section>
    </main>
  );
};

export default Integrations;
