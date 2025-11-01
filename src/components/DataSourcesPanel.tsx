import React, { useState, useEffect, useRef } from 'react';
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
import { AnimatePresence, motion } from 'framer-motion';
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

  const handleConnect = async (provider: string) => {
    if (!user?.id) return;

    setConnecting(provider);
    try {
      const response = await fetch(`${config.apiUrl}/integrations/${provider}/connect`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ user_id: user.id })
      });

      if (!response.ok) {
        throw new Error(`Failed to start connection for ${provider}`);
      }

      const data = await response.json();
      const { auth_url } = data;

      if (auth_url) {
        window.open(auth_url, '_blank', 'width=600,height=800');
      }

      toast({
        title: 'Connecting...',
        description: 'Complete the flow in the newly opened window.'
      });
    } catch (error: any) {
      console.error('Connection error:', error);
      toast({
        title: 'Connection failed',
        description: error?.message || 'Unable to start connection.',
        variant: 'destructive'
      });
    } finally {
      setConnecting(null);
      refreshConnections();
    }
  };

  const handleDisconnect = async (connectionId: string, integrationName: string) => {
    if (!user?.id) return;

    try {
      const response = await fetch(`${config.apiUrl}/integrations/disconnect`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          user_id: user.id,
          connection_id: connectionId
        })
      });

      if (!response.ok) {
        throw new Error('Failed to disconnect');
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

  const handleSync = async (connectionId: string, integrationId: string) => {
    if (!user?.id) return;

    setSyncing(connectionId);
    try {
      const response = await fetch(`${config.apiUrl}/integrations/${integrationId}/sync`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          user_id: user.id,
          connection_id: connectionId
        })
      });

      if (!response.ok) {
        throw new Error('Failed to start sync');
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

  const handleDeleteFile = async (fileId: string, filename: string) => {
    if (!user?.id) return;

    try {
      const response = await fetch(`${config.apiUrl}/ingestion/${fileId}`, {
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
  
  // IMPROVEMENT: Use shared hook for connections (prevents duplicate polling)
  const { data: connections = [], isLoading: loading, refetch: refetchConnections } = useConnections();
  const refreshConnections = useRefreshConnections();
  const [syncing, setSyncing] = useState<string | null>(null);
  const [connecting, setConnecting] = useState<string | null>(null);
  const [verifying, setVerifying] = useState<string | null>(null); // Track which provider is being verified
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    new Set(['accounting', 'data-sources', 'payment'])
  );

  // IMPROVEMENT: Removed manual connection loading - now handled by useConnections hook
  // Refresh when window regains focus (user returns from Nango popup)
  useEffect(() => {
    if (!isOpen || !user?.id) return;
    
    const handleFocus = () => {
      refreshConnections();
    };
    window.addEventListener('focus', handleFocus);
    
    return () => {
      window.removeEventListener('focus', handleFocus);
    };
  }, [user?.id, isOpen, refreshConnections]);

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

  // ...

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
        className="fixed right-0 top-0 h-full w-full md:w-[500px] finley-dynamic-bg border-l border-border shadow-2xl z-50 flex flex-col"
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
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <FileSpreadsheet className="w-4 h-4 text-muted-foreground" />
                  <h3 className="text-sm font-medium">Uploaded Files</h3>
                  {uploadedFiles.length > 0 && (
                    <Badge variant="outline" className="text-[10px]">
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
                  <p className="text-xs text-muted-foreground">No files uploaded yet</p>
                  <p className="text-[10px] text-muted-foreground/70">Upload files to see them here</p>
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
                            <p className="text-sm font-medium truncate text-foreground">
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
                          {/* Preview button */}
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              onFilePreview?.(file.id, file.filename || file.id);
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
                        {isExpanded && (
                          <div className="p-3 space-y-2 bg-background">
                            {integrations.map((integration) => {
                              const connection = getConnection(integration.provider);
                              const connected = !!connection;

                              return (
                                <div
                                  key={integration.id}
                                  className={`flex items-center justify-between p-3 border rounded-md transition-all ${
                                    connected 
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
                                        <p className="text-sm font-medium">{integration.name}</p>
                                        {connected && (
                                          <Badge 
                                            variant="default" 
                                            className="text-[10px] bg-gradient-to-r from-emerald-500 to-green-600 text-white border-0 shadow-sm"
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
      </motion.div>
    </AnimatePresence>
  );
};
