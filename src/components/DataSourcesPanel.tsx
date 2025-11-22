import React, { useState, useEffect, useRef, useCallback } from 'react';
import { FileSpreadsheet, Plug, RefreshCw, CheckCircle2, AlertCircle, ChevronDown, ChevronRight, X, Loader2, Mail, HardDrive, Calculator, CreditCard, Trash2, Plus, Eye, GripVertical } from 'lucide-react';
import { useConnections, useRefreshConnections } from '@/hooks/useConnections';
import gmailLogo from "@/assets/logos/gmail.svg";
import zohoMailLogo from "@/assets/logos/zoho-mail.svg";
import zohoLogo from "@/assets/logos/zoho.svg";
import quickbooksLogo from "@/assets/logos/quickbooks.svg";
import xeroLogo from "@/assets/logos/xero.svg";
import stripeLogo from "@/assets/logos/stripe.svg";
import razorpayLogo from "@/assets/logos/razorpay.svg";
import paypalLogo from "@/assets/logos/paypal.svg";
import googleDriveLogo from "@/assets/logos/google-drive.svg";
import dropboxLogo from "@/assets/logos/dropbox.svg";
import { Card } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { StarBorder } from './ui/star-border';
import { ToastAction } from './ui/toast';
import { useAuth } from './AuthProvider';
import { supabase } from '@/integrations/supabase/client';
import { config } from '@/config';
import { useToast } from './ui/use-toast';
// Removed framer-motion - not needed in 3-panel layout
import { useFastAPIProcessor } from './FastAPIProcessor';

interface DataSourcesPanelProps {
  isOpen: boolean;
  onClose: () => void;
  onFilePreview?: (fileId: string, filename: string, fileData: any) => void;
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
  {
    id: 'paypal',
    name: 'PayPal',
    provider: 'paypal',
    description: 'Sync PayPal transactions',
    icon: <img src={paypalLogo} alt="PayPal" className="w-5 h-5 object-contain" />,
    category: 'payment'
  },
];

const CATEGORY_INFO = {
  accounting: { name: 'Accounting Platforms', icon: <Calculator className="w-4 h-4" /> },
  'data-sources': { name: 'Cloud & Email', icon: <HardDrive className="w-4 h-4" /> },
  payment: { name: 'Payment Processors', icon: <CreditCard className="w-4 h-4" /> },
};

export const DataSourcesPanel = ({ isOpen, onClose, onFilePreview }: DataSourcesPanelProps) => {
  const { user } = useAuth();
  const { toast } = useToast();
  const { processFileWithFastAPI } = useFastAPIProcessor();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [uploadedFiles, setUploadedFiles] = useState<any[]>([]);

  // IMPROVEMENT: Use shared hook for connections (prevents duplicate polling)
  const { data: connections = [], isLoading: loading, refetch: refetchConnections } = useConnections();
  const refreshConnections = useRefreshConnections();
  const [syncing, setSyncing] = useState<string | null>(null);
  const [connecting, setConnecting] = useState<string | null>(null);
  const [verifying, setVerifying] = useState<string | null>(null); // Track which provider is being verified
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    new Set(['accounting', 'data-sources', 'payment'])
  );

  const handlePlusClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    const customEvent = new CustomEvent('files-selected-for-upload', {
      detail: { files }
    });

    window.dispatchEvent(customEvent);
    event.target.value = '';
  };

  const toggleCategory = (category: string) => {
    setExpandedCategories((prev) => {
      const next = new Set(prev);
      if (next.has(category)) {
        next.delete(category);
      } else {
        next.add(category);
      }
      return next;
    });
  };

  const getConnection = (provider: string) => {
    return connections.find(
      (connection) =>
        connection.provider === provider || connection.integration_id === provider
    );
  };

  const getSessionToken = useCallback(async () => {
    const { data: sessionData } = await supabase.auth.getSession();
    return sessionData?.session?.access_token;
  }, []);

  const handleConnect = async (provider: string) => {
    if (!user?.id) return;

    setConnecting(provider);
    try {
      const sessionToken = await getSessionToken();
      if (!sessionToken) {
        throw new Error('Unable to authenticate request. Please sign in again.');
      }

      const response = await fetch(`${config.apiUrl}/api/connectors/initiate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          provider,
          user_id: user.id,
          session_token: sessionToken
        })
      });

      if (!response.ok) {
        let detail: any = undefined;
        try {
          detail = await response.json();
        } catch (err) {
          detail = undefined;
        }
        const message = detail?.detail?.message || detail?.detail || detail?.error || `Failed to start connection for ${provider}`;
        throw new Error(message);
      }

      const data = await response.json();
      const connectUrl = data?.connect_session?.connect_url || data?.connect_session?.url;

      if (connectUrl) {
        window.open(connectUrl, '_blank', 'width=600,height=800');

        toast({
          title: 'Connecting...',
          description: 'Complete the flow in the newly opened window.'
        });

        setVerifying(provider);

        // FIX: Add 60-second timeout to clear "Connecting..." state if verification never completes
        setTimeout(() => {
          setVerifying((current) => {
            if (current === provider) {
              toast({
                title: 'Connection timeout',
                description: 'Please try connecting again or check if the popup was blocked.',
                variant: 'destructive'
              });
              return null;
            }
            return current;
          });
        }, 60000); // 60 seconds
      } else {
        throw new Error('No connection URL received from server');
      }
    } catch (error: any) {
      console.error('Connection error:', error);
      toast({
        title: 'Connection failed',
        description: error?.message || 'Unable to start connection.',
        variant: 'destructive'
      });
      setVerifying(null); // Clear verifying state on error
    } finally {
      setConnecting(null);
    }
  };

  const handleDisconnect = async (connectionId: string, integrationName: string, provider: string) => {
    if (!user?.id) return;

    try {
      const sessionToken = await getSessionToken();
      if (!sessionToken) {
        throw new Error('Unable to authenticate request. Please sign in again.');
      }

      const response = await fetch(`${config.apiUrl}/api/connectors/disconnect`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          user_id: user.id,
          connection_id: connectionId,
          provider,
          session_token: sessionToken
        })
      });

      if (!response.ok) {
        let detail: any = undefined;
        try {
          detail = await response.json();
        } catch (err) {
          detail = undefined;
        }
        const message = detail?.detail?.message || detail?.detail || 'Failed to disconnect';
        throw new Error(message);
      }

      toast({
        title: `${integrationName} disconnected`,
        description: 'The integration has been successfully disconnected.'
      });
      refreshConnections();
    } catch (error: any) {
      console.error('Disconnect error:', error);
      toast({
        title: 'Disconnect failed',
        description: error?.message || 'Unable to disconnect integration.',
        variant: 'destructive'
      });
    }
  };

  const handleSync = async (connectionId: string, integrationId: string | undefined, provider: string) => {
    if (!user?.id) return;

    setSyncing(connectionId);
    try {
      const sessionToken = await getSessionToken();
      if (!sessionToken) {
        throw new Error('Unable to authenticate request. Please sign in again.');
      }

      const response = await fetch(`${config.apiUrl}/api/connectors/sync`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          user_id: user.id,
          connection_id: connectionId,
          integration_id: integrationId || provider,
          mode: 'historical',
          session_token: sessionToken
        })
      });

      if (!response.ok) {
        let detail: any = undefined;
        try {
          detail = await response.json();
        } catch (err) {
          detail = undefined;
        }
        const message = detail?.detail?.message || detail?.detail || 'Failed to start sync';
        throw new Error(message);
      }

      toast({
        title: 'Sync started',
        description: 'We’ll notify you once the sync completes.'
      });
      await refreshConnections();
    } catch (error: any) {
      console.error('Sync error:', error);
      toast({
        title: 'Sync failed',
        description: error?.message || 'Unable to start sync.',
        variant: 'destructive'
      });
    } finally {
      setSyncing(null);
    }
  };

  const verifyConnector = useCallback(async (provider: string) => {
    if (!user?.id || !provider) return;

    try {
      const sessionToken = await getSessionToken();
      if (!sessionToken) {
        throw new Error('Unable to authenticate request. Please sign in again.');
      }

      const response = await fetch(`${config.apiUrl}/api/connectors/verify-connection`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          user_id: user.id,
          provider,
          session_token: sessionToken
        })
      });

      if (!response.ok) {
        // Verification can fail if the user closes the popup early; log but don't block.
        let detail: any = undefined;
        try {
          detail = await response.json();
        } catch (err) {
          detail = undefined;
        }
        const message = detail?.detail?.message || detail?.detail || null;
        if (message) {
          console.warn('Connector verification warning:', message);
        }
      } else {
        toast({
          title: 'Connection verified',
          description: 'Your integration is ready to use.'
        });
      }
    } catch (error) {
      console.warn('Connector verification error:', error);
    } finally {
      setVerifying(null);
      refreshConnections();
    }
  }, [getSessionToken, refreshConnections, toast, user?.id]);

  const handleDeleteFile = async (fileId: string, filename: string) => {
    if (!user?.id) return;

    try {
      // CRITICAL FIX: Use correct backend endpoint /api/files/{job_id}
      const response = await fetch(`${config.apiUrl}/api/files/${fileId}?user_id=${user.id}`, {
        method: 'DELETE'
      });

      if (!response.ok) {
        throw new Error('Failed to delete file');
      }

      toast({
        title: 'File deleted',
        description: `${filename} has been removed.`
      });

      setUploadedFiles((prev) => prev.filter((file) => file.id !== fileId));
    } catch (error: any) {
      console.error('Delete file error:', error);
      toast({
        title: 'Delete failed',
        description: error?.message || 'Unable to delete file.',
        variant: 'destructive'
      });
    }
  };



  // IMPROVEMENT: Removed manual connection loading - now handled by useConnections hook
  // Refresh when window regains focus (user returns from Nango popup)
  useEffect(() => {
    if (!isOpen || !user?.id) return;

    const handleFocus = () => {
      if (verifying) {
        verifyConnector(verifying);
      } else {
        refreshConnections();
      }
    };

    window.addEventListener('focus', handleFocus);

    return () => {
      window.removeEventListener('focus', handleFocus);
    };
  }, [user?.id, isOpen, refreshConnections, verifyConnector, verifying]);

  // Load uploaded files with real-time polling
  useEffect(() => {
    const loadFiles = async () => {
      if (!user?.id) return;
      try {
        console.log('🔄 Loading uploaded files for user:', user.id);
        const { data, error } = await supabase
          .from('ingestion_jobs')
          .select('id, filename, status, created_at, progress, error_message')
          .eq('user_id', user.id)
          .order('created_at', { ascending: false })
          .limit(20);

        if (error) {
          console.error('❌ Error loading files:', error);
        } else {
          console.log('✅ Loaded files:', data?.length || 0, 'files');
          console.log('Files data:', data);
          setUploadedFiles(data || []);
        }
      } catch (e) {
        console.error('❌ Failed to load files:', e);
      }
    };

    if (isOpen && user?.id) {
      loadFiles(); // Initial load immediately
      // CRITICAL FIX: Reduce polling frequency to prevent database overload
      // With 1000 concurrent users, 2-second polling = 30,000 req/min
      // Changed to 30 seconds = 2,000 req/min (15x reduction)
      // TODO: Replace with WebSocket subscriptions for true real-time updates
      const interval = setInterval(loadFiles, 30000); // Poll every 30 seconds

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

  // ...

  const integrationsByCategory = INTEGRATIONS.reduce((acc, integration) => {
    if (!acc[integration.category]) {
      acc[integration.category] = [];
    }
    acc[integration.category].push(integration);
    return acc;
  }, {} as Record<string, Integration[]>);

  return (
    <div className="h-full w-full finley-dynamic-bg flex flex-col">
      {/* Header - NO X button */}
      <div className="flex items-center gap-2 p-3 border-b border-border">
        <Plug className="w-4 h-4 text-primary" />
        <h2 className="text-xs font-semibold">Data Sources</h2>
      </div>

      <div className="flex-1 overflow-y-auto">
        <div className="p-4 space-y-6">
          {/* Unified Uploaded Files Section - Shows all files with their current state */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <FileSpreadsheet className="w-4 h-4 text-muted-foreground" />
                <h3 className="text-xs font-medium">Uploaded Files</h3>
                {uploadedFiles.length > 0 && (
                  <Badge variant="outline" className="text-[8px]">
                    {uploadedFiles.length}
                  </Badge>
                )}
              </div>
              <div className="flex items-center gap-1">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handlePlusClick}
                  className="h-7 px-2"
                  title="Upload Files"
                >
                  <Plus className="w-3 h-3" />
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={async () => {
                    if (!user?.id) return;
                    try {
                      const { data, error } = await supabase
                        .from('ingestion_jobs')
                        .select('id, filename, status, created_at, progress, error_message')
                        .eq('user_id', user.id)
                        .order('created_at', { ascending: false })
                        .limit(20);
                      if (!error && data) {
                        setUploadedFiles(data || []);
                        toast({
                          title: 'Refreshed',
                          description: `Found ${data.length} files`
                        });
                      }
                    } catch (e) {
                      console.error('Refresh failed:', e);
                    }
                  }}
                  className="h-7 px-2"
                  title="Refresh Files"
                >
                  <RefreshCw className="w-3 h-3" />
                </Button>
              </div>
            </div>

            {/* Hidden file input */}
            <input
              ref={fileInputRef}
              type="file"
              accept=".xlsx,.xls,.csv"
              multiple
              onChange={handleFileSelect}
              className="hidden"
            />
            {loading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
              </div>
            ) : uploadedFiles.length === 0 ? (
              <div className="text-center py-8 space-y-2">
                <FileSpreadsheet className="w-8 h-8 mx-auto text-muted-foreground/50" />
                <p className="text-[10px] text-muted-foreground">No files uploaded yet</p>
                <p className="text-[8px] text-muted-foreground/70">Upload files to see them here</p>
              </div>
            ) : (
              <div className="max-h-[400px] overflow-y-auto space-y-2 pr-2 scrollbar-thin">
                {uploadedFiles.map((file) => {
                  const isProcessing = file.status === 'processing' || file.status === 'pending';
                  const isCompleted = file.status === 'completed';
                  const isFailed = file.status === 'failed';
                  const progress = file.progress || 0;

                  return (
                    <div
                      key={file.id}
                      className="flex items-center justify-between p-3 rounded-md border finley-dynamic-bg hover:bg-muted/20 transition-colors group cursor-pointer"
                      onClick={() => onFilePreview?.(file.id, file.filename || file.id, file)}
                    >
                      <div className="flex-1 min-w-0 space-y-1">
                        <div className="flex items-center gap-2">
                          <p className="text-xs font-medium truncate text-foreground">
                            {file.filename || file.id || 'Unnamed File'}
                          </p>
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
                          <p className="text-[8px] text-muted-foreground">
                            {new Date(file.created_at).toLocaleString()}
                          </p>
                          {isProcessing && progress > 0 && (
                            <span className="text-[8px] text-primary font-medium">
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
                        {/* Preview button */}
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onFilePreview?.(file.id, file.filename || file.id, file);
                          }}
                          className="p-1.5 hover:bg-primary/10 rounded transition-colors opacity-0 group-hover:opacity-100"
                          title="Preview file"
                        >
                          <Eye className="w-3.5 h-3.5 text-primary" />
                        </button>

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
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDeleteFile(file.id, file.filename || file.id);
                          }}
                          className="p-1 hover:bg-destructive/10 rounded transition-colors opacity-0 group-hover:opacity-100"
                          title="Delete file"
                        >
                          <Trash2 className="w-3.5 h-3.5 text-muted-foreground hover:text-destructive transition-colors" />
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
                          <span className="text-xs font-medium">{categoryInfo.name}</span>
                          <Badge variant="outline" className="text-[8px]">
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
                      {isExpanded && (
                        <div className="p-3 space-y-2 bg-background">
                          {integrations.map((integration) => {
                            const connection = getConnection(integration.provider);
                            const connected = !!connection;

                            return (
                              <div
                                key={integration.id}
                                className={`flex items-center justify-between p-3 border rounded-md transition-all ${connected
                                    ? 'border-2 border-emerald-500/60 bg-gradient-to-r from-emerald-500/10 via-green-500/5 to-emerald-500/10 shadow-md shadow-emerald-500/20 hover:shadow-emerald-500/30'
                                    : 'finley-dynamic-bg hover:bg-muted/20'
                                  }`}
                              >
                                <div className="flex items-center gap-3 flex-1 min-w-0">
                                  <div className="text-muted-foreground">
                                    {integration.icon}
                                  </div>
                                  <div className="flex-1 min-w-0">
                                    <div className="flex items-center gap-2">
                                      <p className="text-[10px] font-medium">{integration.name}</p>
                                      {connected && (
                                        <Badge
                                          variant="default"
                                          className="text-[8px] bg-gradient-to-r from-emerald-500 to-green-600 text-white border-0 shadow-sm"
                                        >
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
                                        onClick={() => handleSync(connection.connection_id, connection.integration_id, connection.provider)}
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
                                        onClick={() => handleDisconnect(connection.connection_id, integration.name, integration.provider)}
                                        className="text-xs h-9 w-9 p-0 hover:bg-destructive/10 hover:text-destructive"
                                        title="Disconnect"
                                      >
                                        <X className="w-4 h-4" />
                                      </Button>
                                    </>
                                  ) : verifying === integration.provider ? (
                                    <Button
                                      size="sm"
                                      variant="outline"
                                      disabled
                                      className="text-xs h-9 px-4"
                                    >
                                      <Loader2 className="w-3 h-3 mr-1.5 animate-spin" />
                                      Verifying...
                                    </Button>
                                  ) : (
                                    <StarBorder
                                      as="button"
                                      onClick={() => handleConnect(integration.provider)}
                                      disabled={connecting === integration.provider}
                                      speed="5s"
                                      className="h-9 px-4 text-xs font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                      {connecting === integration.provider ? (
                                        <div className="flex items-center gap-1.5">
                                          <Loader2 className="w-3 h-3 animate-spin" />
                                          Connecting
                                        </div>
                                      ) : (
                                        'Connect'
                                      )}
                                    </StarBorder>
                                  )}
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
