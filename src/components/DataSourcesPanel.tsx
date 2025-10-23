import { useState, useEffect } from 'react';
import { FileSpreadsheet, Plug, RefreshCw, CheckCircle2, AlertCircle, ChevronDown, ChevronRight, X, Loader2, Mail, HardDrive, Calculator, CreditCard, Trash2 } from 'lucide-react';
import gmailLogo from "@/assets/logos/gmail.svg";
import zohoMailLogo from "@/assets/logos/zoho-mail.svg";
import zohoLogo from "@/assets/logos/zoho.svg";
import quickbooksLogo from "@/assets/logos/quickbooks.svg";
import xeroLogo from "@/assets/logos/xero.svg";
import stripeLogo from "@/assets/logos/stripe.svg";
import razorpayLogo from "@/assets/logos/razorpay.svg";
import googleDriveLogo from "@/assets/logos/google-drive.svg";
import dropboxLogo from "@/assets/logos/dropbox.svg";
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
    icon: <img src={quickbooksLogo} alt="QuickBooks" className="w-5 h-5 object-contain" />,
    category: 'accounting'
  },
  { 
    id: 'xero', 
    name: 'Xero', 
    provider: 'xero',
    description: 'Connect your Xero accounting data',
    icon: <img src={xeroLogo} alt="Xero" className="w-5 h-5 object-contain" />,
    category: 'accounting'
  },
  { 
    id: 'zoho-books', 
    name: 'Zoho Books', 
    provider: 'zoho-books',
    description: 'Import Zoho Books transactions',
    icon: <img src={zohoLogo} alt="Zoho Books" className="w-5 h-5 object-contain" />,
    category: 'accounting'
  },
  
  // Data Sources (Cloud Storage + Email)
  { 
    id: 'google-drive', 
    name: 'Google Drive', 
    provider: 'google-drive',
    description: 'Access files from Google Drive',
    icon: <img src={googleDriveLogo} alt="Google Drive" className="w-5 h-5 object-contain" />,
    category: 'data-sources'
  },
  { 
    id: 'dropbox', 
    name: 'Dropbox', 
    provider: 'dropbox',
    description: 'Sync files from Dropbox',
    icon: <img src={dropboxLogo} alt="Dropbox" className="w-5 h-5 object-contain" />,
    category: 'data-sources'
  },
  { 
    id: 'google-mail', 
    name: 'Gmail', 
    provider: 'google-mail',
    description: 'Extract attachments from Gmail',
    icon: <img src={gmailLogo} alt="Gmail" className="w-5 h-5 object-contain" />,
    category: 'data-sources'
  },
  { 
    id: 'zoho-mail', 
    name: 'Zoho Mail', 
    provider: 'zoho-mail',
    description: 'Get attachments from Zoho Mail',
    icon: <img src={zohoMailLogo} alt="Zoho Mail" className="w-5 h-5 object-contain" />,
    category: 'data-sources'
  },
  
  // Payment
  { 
    id: 'stripe', 
    name: 'Stripe', 
    provider: 'stripe',
    description: 'Import Stripe transactions',
    icon: <img src={stripeLogo} alt="Stripe" className="w-5 h-5 object-contain" />,
    category: 'payment'
  },
  { 
    id: 'razorpay', 
    name: 'Razorpay', 
    provider: 'razorpay',
    description: 'Sync Razorpay payments',
    icon: <img src={razorpayLogo} alt="Razorpay" className="w-5 h-5 object-contain" />,
    category: 'payment'
  },
];

const CATEGORY_INFO = {
  accounting: { name: 'Accounting Platforms', icon: <Calculator className="w-4 h-4" /> },
  'data-sources': { name: 'Cloud & Email', icon: <HardDrive className="w-4 h-4" /> },
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
    new Set(['accounting', 'data-sources', 'payment'])
  );

  // Load connections
  useEffect(() => {
    let isInitialLoad = true;
    
    const loadConnections = async (showLoading = false) => {
      if (!user?.id) return;
      try {
        // Only show loading spinner on initial load
        if (showLoading) {
          setLoading(true);
        }
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
        } else {
          console.error('Failed to load connections: HTTP', response.status);
        }
      } catch (e) {
        console.error('Failed to load connections', e);
      } finally {
        // ALWAYS set loading to false, even on error
        if (showLoading) {
          setLoading(false);
        }
      }
    };

    if (isOpen && user?.id) {
      // Initial load with loading spinner
      loadConnections(true);
      
      // Poll every 30 seconds for updates (no loading spinner)
      const interval = setInterval(() => loadConnections(false), 30000);
      
      // Refresh when window regains focus (user returns from Nango popup)
      const handleFocus = () => {
        loadConnections(false);
      };
      window.addEventListener('focus', handleFocus);
      
      return () => {
        clearInterval(interval);
        window.removeEventListener('focus', handleFocus);
      };
    }
  }, [user?.id, isOpen]);

  // Load uploaded files with real-time polling
  useEffect(() => {
    const loadFiles = async () => {
      if (!user?.id) return;
      try {
        const { data, error } = await supabase
          .from('ingestion_jobs')
          .select('id, filename, status, created_at, progress, error_message')
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

    if (isOpen && user?.id) {
      loadFiles(); // Initial load immediately
      const interval = setInterval(loadFiles, 2000); // Poll every 2 seconds for real-time updates
      
      // Listen for file upload events and actually process the files
      const handleFileUpload = async (event: any) => {
        const files = event.detail?.files;
        if (files && files.length > 0) {
          // Show toast for batch upload
          toast({
            title: 'Uploading Files',
            description: `Processing ${files.length} file${files.length > 1 ? 's' : ''}...`
          });
          
          // Process all files in parallel (not sequentially)
          const uploadPromises = Array.from(files).map(async (file: File) => {
            try {
              await processFileWithFastAPI(
                file,
                undefined, // No custom prompt
                (progress) => {
                  // Progress callback
                  console.log(`Processing ${file.name}:`, progress);
                  
                  // Refresh file list on any progress update to show all files
                  loadFiles();
                  
                  if (progress.status === 'completed') {
                    toast({
                      title: 'File Completed',
                      description: `${file.name} processed successfully`
                    });
                  } else if (progress.status === 'error') {
                    toast({
                      title: 'Upload Failed',
                      description: `${file.name} failed to process`,
                      variant: 'destructive'
                    });
                  }
                }
              );
            } catch (error) {
              console.error(`File upload error for ${file.name}:`, error);
              toast({
                title: 'Upload Failed',
                description: `${file.name} failed to process`,
                variant: 'destructive'
              });
            }
          });
          
          // Wait for all uploads to complete
          await Promise.allSettled(uploadPromises);
          
          // Final refresh
          loadFiles();
        }
      };
      window.addEventListener('files-selected-for-upload', handleFileUpload as EventListener);
      
      return () => {
        clearInterval(interval);
        window.removeEventListener('files-selected-for-upload', handleFileUpload);
      };
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
        // Open popup and monitor when it closes
        const popup = window.open(connectUrl, '_blank', 'width=600,height=700,noopener,noreferrer');
        
        toast({
          title: 'Connection Started',
          description: 'Complete the authorization in the popup window'
        });
        
        // Poll to detect when popup closes, then verify connection
        if (popup) {
          const pollTimer = setInterval(() => {
            if (popup.closed) {
              clearInterval(pollTimer);
              // Verify connection after popup closes (creates record if webhook failed)
              setTimeout(async () => {
                const { data: sessionData } = await supabase.auth.getSession();
                const sessionToken = sessionData?.session?.access_token;
                
                // Call verify endpoint to ensure connection is saved
                try {
                  console.log('Verifying connection for provider:', provider);
                  const verifyResponse = await fetch(`${config.apiUrl}/api/connectors/verify-connection`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                      user_id: user?.id,
                      provider: provider,
                      session_token: sessionToken
                    })
                  });
                  
                  if (verifyResponse.ok) {
                    console.log('Connection verified successfully');
                    toast({
                      title: 'Connected!',
                      description: `${integration.name} connected successfully`
                    });
                  } else {
                    const error = await verifyResponse.text();
                    console.error('Verify failed:', error);
                  }
                } catch (e) {
                  console.error('Failed to verify connection:', e);
                }
                
                // Refresh connections list
                const response = await fetch(`${config.apiUrl}/api/connectors/user-connections`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({
                    user_id: user?.id,
                    session_token: sessionToken
                  })
                });
                if (response.ok) {
                  const data = await response.json();
                  setConnections(data.connections || []);
                  console.log('Connections refreshed:', data.connections);
                }
              }, 3000); // Wait 3 seconds for Nango to process
            }
          }, 500);
        }
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

  const handleDisconnect = async (connectionId: string, integrationName: string) => {
    try {
      const { data: sessionData } = await supabase.auth.getSession();
      const sessionToken = sessionData?.session?.access_token;

      const response = await fetch(`${config.apiUrl}/api/connectors/disconnect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: user?.id,
          connection_id: connectionId,
          session_token: sessionToken
        })
      });

      if (!response.ok) {
        throw new Error('Disconnect failed');
      }

      // Refresh connections immediately
      const connectionsResponse = await fetch(`${config.apiUrl}/api/connectors/user-connections`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: user?.id,
          session_token: sessionToken
        })
      });

      if (connectionsResponse.ok) {
        const data = await connectionsResponse.json();
        setConnections(data.connections || []);
      }

      toast({
        title: 'Disconnected',
        description: `${integrationName} has been disconnected`
      });
    } catch (e) {
      console.error('Disconnect failed', e);
      toast({
        title: 'Disconnect Failed',
        description: 'Unable to disconnect. Please try again.',
        variant: 'destructive'
      });
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

  const handleDeleteFile = async (fileId: string, filename: string) => {
    if (!confirm(`Are you sure you want to delete "${filename}"? This action cannot be undone.`)) {
      return;
    }

    try {
      // Delete from database
      const { error } = await supabase
        .from('ingestion_jobs')
        .delete()
        .eq('id', fileId)
        .eq('user_id', user?.id);

      if (error) throw error;

      // Update local state
      setUploadedFiles(prev => prev.filter(f => f.id !== fileId));

      toast({
        title: 'File Deleted',
        description: `"${filename}" has been deleted successfully.`
      });
    } catch (e) {
      console.error('Failed to delete file', e);
      toast({
        title: 'Delete Failed',
        description: 'Unable to delete file. Please try again.',
        variant: 'destructive'
      });
    }
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
            <h2 className="text-base font-semibold">Data Sources</h2>
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
                        
                        <div className="ml-3 flex items-center gap-2 flex-shrink-0">
                          <div>
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
                              <div className="flex flex-col items-end gap-1">
                                <Badge variant="destructive" className="text-[10px]">
                                  Failed
                                </Badge>
                                {file.error_message && (
                                  <p className="text-[9px] text-destructive/70 max-w-[150px] text-right truncate" title={file.error_message}>
                                    {file.error_message}
                                  </p>
                                )}
                              </div>
                            )}
                          </div>
                          
                          {/* Delete button */}
                          <button
                            onClick={() => handleDeleteFile(file.id, file.filename)}
                            className="p-1 hover:bg-destructive/10 rounded transition-colors group"
                            title="Delete file"
                          >
                            <Trash2 className="w-3.5 h-3.5 text-muted-foreground group-hover:text-destructive transition-colors" />
                          </button>
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
                                      className={`flex items-center justify-between p-3 border rounded-md hover:bg-muted/30 transition-all ${
                                        connected 
                                          ? 'border-2 border-emerald-500/50 bg-emerald-500/5 shadow-sm shadow-emerald-500/20' 
                                          : 'border-border'
                                      }`}
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
                                          <>
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
                                            <Button
                                              size="sm"
                                              variant="ghost"
                                              onClick={() => handleDisconnect(connection.connection_id, integration.name)}
                                              className="text-xs h-9 w-9 p-0 hover:bg-destructive/10 hover:text-destructive"
                                              title="Disconnect"
                                            >
                                              <X className="w-4 h-4" />
                                            </Button>
                                          </>
                                        ) : (
                                          <div className="relative">
                                            {/* Animated gradient border - colors shift smoothly */}
                                            <div className="absolute -inset-0.5 bg-gradient-to-r from-primary via-purple-500 to-primary rounded-full opacity-50 blur-[2px] animate-gradient" />
                                            
                                            <Button
                                              size="sm"
                                              onClick={() => handleConnect(integration.provider)}
                                              disabled={connecting === integration.provider}
                                              className="relative h-9 px-4 rounded-full bg-black text-white text-xs font-medium hover:bg-black/90 transition-colors disabled:opacity-50"
                                            >
                                              {connecting === integration.provider ? (
                                                <>
                                                  <Loader2 className="w-3 h-3 mr-1.5 animate-spin" />
                                                  Connecting
                                                </>
                                              ) : (
                                                'Connect'
                                              )}
                                            </Button>
                                          </div>
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
