import { useState, useEffect } from 'react';
import { FileSpreadsheet, Plug, RefreshCw, CheckCircle2, AlertCircle, ChevronDown, ChevronRight, X, Loader2, Mail, HardDrive, Calculator, CreditCard } from 'lucide-react';
import { Card } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { useAuth } from './AuthProvider';
import { supabase } from '@/integrations/supabase/client';
import { config } from '@/config';
import { useToast } from './ui/use-toast';
import { motion, AnimatePresence } from 'framer-motion';
import { useFastAPIProcessor } from './FastAPIProcessor';

interface DataSourcesPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

interface Integration {
  id: string;
  name: string;
  provider: string;
  description: string;
  icon: React.ReactNode;
  category: string;
}

interface Connection {
  connection_id: string;
  integration_id: string;
  provider: string;
  status: string;
  last_synced_at: string | null;
  created_at: string;
}

// All verified working integrations from backend
const INTEGRATIONS: Integration[] = [
  // Accounting
  { 
    id: 'quickbooks-sandbox', 
    name: 'QuickBooks', 
    provider: 'quickbooks-sandbox',
    description: 'Sync invoices, bills, and payments',
    icon: <Calculator className="w-5 h-5" />,
    category: 'accounting'
  },
  { 
    id: 'xero', 
    name: 'Xero', 
    provider: 'xero',
    description: 'Connect your Xero accounting data',
    icon: <Calculator className="w-5 h-5" />,
    category: 'accounting'
  },
  { 
    id: 'zoho-books', 
    name: 'Zoho Books', 
    provider: 'zoho-books',
    description: 'Import Zoho Books transactions',
    icon: <Calculator className="w-5 h-5" />,
    category: 'accounting'
  },
  
  // Cloud Storage
  { 
    id: 'google-drive', 
    name: 'Google Drive', 
    provider: 'google-drive',
    description: 'Access files from Google Drive',
    icon: <HardDrive className="w-5 h-5" />,
    category: 'storage'
  },
  { 
    id: 'dropbox', 
    name: 'Dropbox', 
    provider: 'dropbox',
    description: 'Sync files from Dropbox',
    icon: <HardDrive className="w-5 h-5" />,
    category: 'storage'
  },
  
  // Email
  { 
    id: 'google-mail', 
    name: 'Gmail', 
    provider: 'google-mail',
    description: 'Extract attachments from Gmail',
    icon: <Mail className="w-5 h-5" />,
    category: 'email'
  },
  { 
    id: 'zoho-mail', 
    name: 'Zoho Mail', 
    provider: 'zoho-mail',
    description: 'Get attachments from Zoho Mail',
    icon: <Mail className="w-5 h-5" />,
    category: 'email'
  },
  
  // Payment
  { 
    id: 'stripe', 
    name: 'Stripe', 
    provider: 'stripe',
    description: 'Import Stripe transactions',
    icon: <CreditCard className="w-5 h-5" />,
    category: 'payment'
  },
  { 
    id: 'razorpay', 
    name: 'Razorpay', 
    provider: 'razorpay',
    description: 'Sync Razorpay payments',
    icon: <CreditCard className="w-5 h-5" />,
    category: 'payment'
  },
];

const CATEGORY_INFO = {
  accounting: { name: 'Accounting Platforms', icon: <Calculator className="w-4 h-4" /> },
  storage: { name: 'Cloud Storage', icon: <HardDrive className="w-4 h-4" /> },
  email: { name: 'Email Platforms', icon: <Mail className="w-4 h-4" /> },
  payment: { name: 'Payment Processors', icon: <CreditCard className="w-4 h-4" /> },
};

export const DataSourcesPanel = ({ isOpen, onClose }: DataSourcesPanelProps) => {
  const { user } = useAuth();
  const { toast } = useToast();
  const { processFileWithFastAPI } = useFastAPIProcessor();
  
  const [uploadedFiles, setUploadedFiles] = useState<any[]>([]);
  const [connections, setConnections] = useState<Connection[]>([]);
  const [loading, setLoading] = useState(false);
  const [syncing, setSyncing] = useState<string | null>(null);
  const [connecting, setConnecting] = useState<string | null>(null);
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    new Set(['accounting', 'storage', 'email', 'payment'])
  );

  // Load connections
  useEffect(() => {
    const loadConnections = async () => {
      if (!user?.id) return;
      try {
        setLoading(true);
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

        if (response.ok) {
          const data = await response.json();
          setConnections(data.connections || []);
        }
      } catch (e) {
        console.error('Failed to load connections', e);
      } finally {
        setLoading(false);
      }
    };

    if (isOpen) {
      loadConnections();
      const interval = setInterval(loadConnections, 30000); // Refresh every 30s
      return () => clearInterval(interval);
    }
  }, [user?.id, isOpen]);

  // Load uploaded files with real-time polling
  useEffect(() => {
    const loadFiles = async () => {
      if (!user?.id) return;
      try {
        const { data, error } = await supabase
          .from('ingestion_jobs')
          .select('id, filename, status, created_at, progress')
          .eq('user_id', user.id)
          .order('created_at', { ascending: false })
          .limit(20);

        if (!error && data) {
          setUploadedFiles(data);
        }
      } catch (e) {
        console.error('Failed to load files', e);
      }
    };

    if (isOpen) {
      loadFiles(); // Initial load
      const interval = setInterval(loadFiles, 3000); // Poll every 3 seconds for real-time updates
      return () => clearInterval(interval);
    }
  }, [user?.id, isOpen]);

  const handleConnect = async (provider: string) => {
    try {
      setConnecting(provider);
      const { data: sessionData } = await supabase.auth.getSession();
      const sessionToken = sessionData?.session?.access_token;

      const response = await fetch(`${config.apiUrl}/api/connectors/initiate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider,
          user_id: user?.id,
          session_token: sessionToken
        })
      });

      if (!response.ok) {
        throw new Error('Failed to initiate connection');
      }

      const data = await response.json();
      const connectUrl = data?.connect_session?.url || data?.connect_session?.connect_url;

      if (connectUrl) {
        window.open(connectUrl, '_blank', 'noopener,noreferrer');
        toast({
          title: 'Connection Started',
          description: 'Complete the authorization in the popup window'
        });
      } else {
        throw new Error('No authorization URL returned');
      }
    } catch (e) {
      console.error('Connect failed', e);
      toast({
        title: 'Connection Failed',
        description: 'Unable to start connection. Please try again.',
        variant: 'destructive'
      });
    } finally {
      setConnecting(null);
    }
  };

  const handleSync = async (connectionId: string, integrationId: string) => {
    try {
      setSyncing(connectionId);
      const { data: sessionData } = await supabase.auth.getSession();
      const sessionToken = sessionData?.session?.access_token;

      const response = await fetch(`${config.apiUrl}/api/connectors/sync`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: user?.id,
          connection_id: connectionId,
          integration_id: integrationId,
          mode: 'incremental',
          session_token: sessionToken
        })
      });

      if (!response.ok) {
        throw new Error('Sync failed');
      }

      toast({
        title: 'Sync Started',
        description: 'Your data is being synchronized'
      });
    } catch (e) {
      console.error('Sync failed', e);
      toast({
        title: 'Sync Failed',
        description: 'Unable to sync. Please try again.',
        variant: 'destructive'
      });
    } finally {
      setSyncing(null);
    }
  };

  const toggleCategory = (category: string) => {
    setExpandedCategories(prev => {
      const next = new Set(prev);
      if (next.has(category)) {
        next.delete(category);
      } else {
        next.add(category);
      }
      return next;
    });
  };

  const isConnected = (provider: string) => {
    return connections.some(c => c.integration_id === provider || c.provider === provider);
  };

  const getConnection = (provider: string) => {
    return connections.find(c => c.integration_id === provider || c.provider === provider);
  };

  const integrationsByCategory = INTEGRATIONS.reduce((acc, integration) => {
    if (!acc[integration.category]) {
      acc[integration.category] = [];
    }
    acc[integration.category].push(integration);
    return acc;
  }, {} as Record<string, Integration[]>);

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ x: '100%' }}
        animate={{ x: 0 }}
        exit={{ x: '100%' }}
        transition={{ type: 'spring', damping: 25, stiffness: 200 }}
        className="fixed right-0 top-0 h-full w-full md:w-[500px] bg-background border-l border-border shadow-2xl z-50 flex flex-col"
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border">
          <div className="flex items-center gap-2">
            <Plug className="w-5 h-5 text-primary" />
            <h2 className="text-lg font-semibold">Data Sources</h2>
          </div>
          <Button variant="ghost" size="sm" onClick={onClose}>
            <X className="w-4 h-4" />
          </Button>
        </div>

        <div className="flex-1 overflow-y-auto">
          <div className="p-4 space-y-6">
            {/* Unified Uploaded Files Section - Shows all files with their current state */}
            <div>
              <div className="flex items-center gap-2 mb-3">
                <FileSpreadsheet className="w-4 h-4 text-muted-foreground" />
                <h3 className="text-sm font-medium">Uploaded Files</h3>
                {uploadedFiles.length > 0 && (
                  <Badge variant="outline" className="text-[10px]">
                    {uploadedFiles.length}
                  </Badge>
                )}
              </div>
              {uploadedFiles.length === 0 ? (
                <p className="text-xs text-muted-foreground">No files uploaded yet</p>
              ) : (
                <div className="space-y-2">
                  {uploadedFiles.map((file) => {
                    const isProcessing = file.status === 'processing' || file.status === 'pending';
                    const isCompleted = file.status === 'completed';
                    const isFailed = file.status === 'failed';
                    const progress = file.progress || 0;
                    
                    return (
                      <div 
                        key={file.id} 
                        className="flex items-center justify-between p-3 bg-muted/50 rounded-md border border-border hover:bg-muted/70 transition-colors"
                      >
                        <div className="flex-1 min-w-0 space-y-1">
                          <div className="flex items-center gap-2">
                            <p className="text-xs font-medium truncate">{file.filename}</p>
                            {isProcessing && (
                              <Loader2 className="w-3 h-3 animate-spin text-primary flex-shrink-0" />
                            )}
                            {isCompleted && (
                              <CheckCircle2 className="w-3 h-3 text-green-500 flex-shrink-0" />
                            )}
                            {isFailed && (
                              <AlertCircle className="w-3 h-3 text-destructive flex-shrink-0" />
                            )}
                          </div>
                          
                          <div className="flex items-center gap-2">
                            <p className="text-[10px] text-muted-foreground">
                              {new Date(file.created_at).toLocaleString()}
                            </p>
                            {isProcessing && progress > 0 && (
                              <span className="text-[10px] text-primary font-medium">
                                {progress}%
                              </span>
                            )}
                          </div>
                          
                          {/* Progress bar for processing files */}
                          {isProcessing && (
                            <div className="w-full h-1 bg-muted rounded-full overflow-hidden">
                              <div 
                                className="h-full bg-primary transition-all duration-300"
                                style={{ width: `${progress}%` }}
                              />
                            </div>
                          )}
                        </div>
                        
                        <div className="ml-3 flex-shrink-0">
                          {isProcessing && (
                            <Badge variant="secondary" className="text-[10px]">
                              Processing...
                            </Badge>
                          )}
                          {isCompleted && (
                            <Badge variant="default" className="text-[10px] bg-green-500/10 text-green-600 border-green-500/20">
                              ✓ Completed
                            </Badge>
                          )}
                          {isFailed && (
                            <Badge variant="destructive" className="text-[10px]">
                              Failed
                            </Badge>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Integrations Section */}
            <div>
              <div className="flex items-center gap-2 mb-3">
                <Plug className="w-4 h-4 text-muted-foreground" />
                <h3 className="text-sm font-medium">Integrations</h3>
              </div>

              {loading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
                </div>
              ) : (
                <div className="space-y-4">
                  {Object.entries(integrationsByCategory).map(([category, integrations]) => {
                    const categoryInfo = CATEGORY_INFO[category as keyof typeof CATEGORY_INFO];
                    const isExpanded = expandedCategories.has(category);

                    return (
                      <div key={category} className="border border-border rounded-lg overflow-hidden">
                        {/* Category Header */}
                        <button
                          onClick={() => toggleCategory(category)}
                          className="w-full flex items-center justify-between p-3 bg-muted/30 hover:bg-muted/50 transition-colors"
                        >
                          <div className="flex items-center gap-2">
                            {categoryInfo.icon}
                            <span className="text-sm font-medium">{categoryInfo.name}</span>
                            <Badge variant="outline" className="text-[10px]">
                              {integrations.length}
                            </Badge>
                          </div>
                          {isExpanded ? (
                            <ChevronDown className="w-4 h-4 text-muted-foreground" />
                          ) : (
                            <ChevronRight className="w-4 h-4 text-muted-foreground" />
                          )}
                        </button>

                        {/* Category Content */}
                        <AnimatePresence>
                          {isExpanded && (
                            <motion.div
                              initial={{ height: 0 }}
                              animate={{ height: 'auto' }}
                              exit={{ height: 0 }}
                              transition={{ duration: 0.2 }}
                              className="overflow-hidden"
                            >
                              <div className="p-3 space-y-2 bg-background">
                                {integrations.map((integration) => {
                                  const connection = getConnection(integration.provider);
                                  const connected = !!connection;

                                  return (
                                    <div
                                      key={integration.id}
                                      className="flex items-center justify-between p-3 border border-border rounded-md hover:bg-muted/30 transition-colors"
                                    >
                                      <div className="flex items-center gap-3 flex-1 min-w-0">
                                        <div className="text-muted-foreground">
                                          {integration.icon}
                                        </div>
                                        <div className="flex-1 min-w-0">
                                          <div className="flex items-center gap-2">
                                            <p className="text-sm font-medium">{integration.name}</p>
                                            {connected && (
                                              <Badge variant="default" className="text-[10px]">
                                                <CheckCircle2 className="w-3 h-3 mr-1" />
                                                Connected
                                              </Badge>
                                            )}
                                          </div>
                                          <p className="text-xs text-muted-foreground truncate">
                                            {integration.description}
                                          </p>
                                          {connected && connection.last_synced_at && (
                                            <p className="text-[10px] text-muted-foreground mt-1">
                                              Last synced: {new Date(connection.last_synced_at).toLocaleString()}
                                            </p>
                                          )}
                                        </div>
                                      </div>
                                      <div className="flex items-center gap-2">
                                        {connected ? (
                                          <Button
                                            size="sm"
                                            variant="outline"
                                            onClick={() => handleSync(connection.connection_id, connection.integration_id)}
                                            disabled={syncing === connection.connection_id}
                                            className="text-xs"
                                          >
                                            {syncing === connection.connection_id ? (
                                              <>
                                                <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                                                Syncing
                                              </>
                                            ) : (
                                              <>
                                                <RefreshCw className="w-3 h-3 mr-1" />
                                                Sync
                                              </>
                                            )}
                                          </Button>
                                        ) : (
                                          <Button
                                            size="sm"
                                            onClick={() => handleConnect(integration.provider)}
                                            disabled={connecting === integration.provider}
                                            className="text-xs"
                                          >
                                            {connecting === integration.provider ? (
                                              <>
                                                <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                                                Connecting
                                              </>
                                            ) : (
                                              'Connect'
                                            )}
                                          </Button>
                                        )}
                                      </div>
                                    </div>
                                  );
                                })}
                              </div>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
};
