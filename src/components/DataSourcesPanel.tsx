import { useState, useEffect } from 'react';
import { FileSpreadsheet, Plug, RefreshCw, CheckCircle2, AlertCircle, ChevronRight, X, Loader2 } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
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

interface UploadingFile {
  id: string;
  name: string;
  status: 'uploading' | 'processing' | 'completed' | 'error';
  progress: number;
  timestamp: Date;
  error?: string;
}

export const DataSourcesPanel = ({ isOpen, onClose }: DataSourcesPanelProps) => {
  const { user } = useAuth();
  const { toast } = useToast();
  const { processFileWithFastAPI } = useFastAPIProcessor();
  const [uploadedFiles, setUploadedFiles] = useState<any[]>([]);
  const [uploadingFiles, setUploadingFiles] = useState<UploadingFile[]>([]);
  const [connections, setConnections] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [syncing, setSyncing] = useState<string | null>(null);

  // Load uploaded files
  useEffect(() => {
    const loadFiles = async () => {
      if (!user?.id) return;
      try {
        const { data, error } = await supabase
          .from('ingestion_jobs')
          .select('id, filename, status, created_at, progress')
          .eq('user_id', user.id)
          .order('created_at', { ascending: false })
          .limit(10);

        if (!error && data) {
          setUploadedFiles(data);
        }
      } catch (e) {
        console.error('Failed to load files', e);
      }
    };

    if (isOpen) {
      loadFiles();
    }
  }, [user?.id, isOpen]);

  // Load connections
  useEffect(() => {
    const loadConnections = async () => {
      if (!user?.id) return;
      try {
        setLoading(true);
        const { data: sessionData } = await supabase.auth.getSession();
        const sessionToken = sessionData?.session?.access_token;

        const response = await fetch(`${config.apiUrl}/api/connectors/list`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: user.id,
            session_token: sessionToken
          })
        });

        if (response.ok) {
          const data = await response.json();
          setConnections(data?.connections || []);
        }
      } catch (e) {
        console.error('Failed to load connections', e);
      } finally {
        setLoading(false);
      }
    };

    if (isOpen) {
      loadConnections();
    }
  }, [user?.id, isOpen]);

  // Listen for file upload events from ChatInterface
  useEffect(() => {
    const handleFilesSelected = async (event: CustomEvent) => {
      const files: File[] = event.detail.files;
      
      // Add files to uploading state immediately
      const newUploadingFiles: UploadingFile[] = files.map((file, idx) => ({
        id: `${Date.now()}-${idx}`,
        name: file.name,
        status: 'uploading',
        progress: 0,
        timestamp: new Date()
      }));
      
      setUploadingFiles(prev => [...prev, ...newUploadingFiles]);
      
      // Process each file with FastAPI WebSocket integration
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const fileId = newUploadingFiles[i].id;
        
        try {
          await processFileWithFastAPI(file, undefined, (progress) => {
            // Real-time WebSocket progress updates
            setUploadingFiles(prev => prev.map(f => 
              f.id === fileId 
                ? { 
                    ...f, 
                    status: progress.step === 'completed' ? 'completed' : 'processing',
                    progress: progress.progress || 0
                  }
                : f
            ));
          });
          
          // Mark as completed
          setUploadingFiles(prev => prev.map(f => 
            f.id === fileId ? { ...f, status: 'completed', progress: 100 } : f
          ));
          
          // Reload uploaded files list
          const { data } = await supabase
            .from('ingestion_jobs')
            .select('id, filename, status, created_at, progress')
            .eq('user_id', user?.id)
            .order('created_at', { ascending: false })
            .limit(10);
          
          if (data) setUploadedFiles(data);
          
        } catch (error) {
          setUploadingFiles(prev => prev.map(f => 
            f.id === fileId 
              ? { ...f, status: 'error', error: 'Processing failed' }
              : f
          ));
        }
      }
    };
    
    window.addEventListener('files-selected-for-upload', handleFilesSelected as EventListener);
    return () => window.removeEventListener('files-selected-for-upload', handleFilesSelected as EventListener);
  }, [user?.id, processFileWithFastAPI]);

  const handleSync = async (provider: string, connectionId: string) => {
    try {
      setSyncing(connectionId);
      const { data: sessionData } = await supabase.auth.getSession();
      const sessionToken = sessionData?.session?.access_token;

      const response = await fetch(`${config.apiUrl}/sync/${provider}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: user?.id,
          connection_id: connectionId,
          session_token: sessionToken
        })
      });

      if (response.ok) {
        toast({
          title: 'Sync Started',
          description: `Syncing data from ${provider}...`
        });
      } else {
        throw new Error('Sync failed');
      }
    } catch (e) {
      toast({
        title: 'Sync Failed',
        description: 'Failed to start sync. Please try again.',
        variant: 'destructive'
      });
    } finally {
      setSyncing(null);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle2 className="w-4 h-4 text-green-500" />;
      case 'failed':
        return <AlertCircle className="w-4 h-4 text-red-500" />;
      case 'processing':
        return <RefreshCw className="w-4 h-4 text-blue-500 animate-spin" />;
      default:
        return <RefreshCw className="w-4 h-4 text-muted-foreground" />;
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Overlay for mobile */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-[#1a1a1a]/50 z-40 lg:hidden"
            onClick={onClose}
          />

          {/* Panel */}
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 20, stiffness: 300 }}
            className="fixed right-0 top-0 h-full w-80 bg-background border-l border-border z-50 overflow-y-auto"
          >
            <div className="p-4 space-y-4">
              {/* Header */}
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold">Data Sources</h2>
                <Button variant="ghost" size="icon" onClick={onClose}>
                  <X className="w-4 h-4" />
                </Button>
              </div>

              {/* Currently Uploading Files - Real-time WebSocket Progress */}
              {uploadingFiles.length > 0 && (
                <Card className="border-primary/50">
                  <CardHeader className="pb-3">
                    <div className="flex items-center gap-2">
                      <Loader2 className="w-4 h-4 text-primary animate-spin" />
                      <CardTitle className="text-sm">Processing Files</CardTitle>
                    </div>
                    <CardDescription className="text-xs">
                      Real-time WebSocket updates
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    {uploadingFiles.map((file) => (
                      <div
                        key={file.id}
                        className="p-3 rounded-md bg-primary/5 border border-primary/20"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2 flex-1 min-w-0">
                            {file.status === 'completed' ? (
                              <CheckCircle2 className="w-4 h-4 text-green-500" />
                            ) : file.status === 'error' ? (
                              <AlertCircle className="w-4 h-4 text-red-500" />
                            ) : (
                              <Loader2 className="w-4 h-4 text-primary animate-spin" />
                            )}
                            <div className="flex-1 min-w-0">
                              <p className="text-xs font-medium truncate">{file.name}</p>
                              <p className="text-[10px] text-muted-foreground">
                                {formatDate(file.timestamp.toISOString())}
                              </p>
                            </div>
                          </div>
                          <Badge 
                            variant={file.status === 'completed' ? 'default' : 'secondary'} 
                            className="text-[10px]"
                          >
                            {file.progress}%
                          </Badge>
                        </div>
                        {/* Progress Bar */}
                        <div className="w-full bg-muted rounded-full h-1.5">
                          <div 
                            className={`h-1.5 rounded-full transition-all duration-300 ${
                              file.status === 'completed' ? 'bg-green-500' : 
                              file.status === 'error' ? 'bg-red-500' : 'bg-primary'
                            }`}
                            style={{ width: `${file.progress}%` }}
                          />
                        </div>
                        {file.error && (
                          <p className="text-[10px] text-red-500 mt-1">{file.error}</p>
                        )}
                      </div>
                    ))}
                  </CardContent>
                </Card>
              )}

              {/* Section A: Uploaded Files - Fixed Height with Independent Scroll */}
              <Card className="flex flex-col h-[250px]">
                <CardHeader className="pb-3 flex-shrink-0">
                  <div className="flex items-center gap-2">
                    <FileSpreadsheet className="w-4 h-4 text-muted-foreground" />
                    <CardTitle className="text-sm">Uploaded Files</CardTitle>
                  </div>
                  <CardDescription className="text-xs">
                    Recent file uploads
                  </CardDescription>
                </CardHeader>
                <CardContent className="flex-1 overflow-y-auto space-y-2 min-h-0">
                  {uploadedFiles.length === 0 ? (
                    <p className="text-xs text-muted-foreground">No files uploaded yet</p>
                  ) : (
                    uploadedFiles.map((file) => (
                      <div
                        key={file.id}
                        className="flex items-center justify-between p-2 rounded-md bg-muted/50 hover:bg-muted transition-colors"
                      >
                        <div className="flex items-center gap-2 flex-1 min-w-0">
                          {getStatusIcon(file.status)}
                          <div className="flex-1 min-w-0">
                            <p className="text-xs font-medium truncate">{file.filename}</p>
                            <p className="text-[10px] text-muted-foreground">
                              {formatDate(file.created_at)}
                            </p>
                          </div>
                        </div>
                        {file.status === 'processing' && (
                          <Badge variant="secondary" className="text-[10px]">
                            {file.progress || 0}%
                          </Badge>
                        )}
                      </div>
                    ))
                  )}
                </CardContent>
              </Card>

              {/* Section B: Integrations - Fixed Height with Independent Scroll */}
              <Card className="flex flex-col h-[280px]">
                <CardHeader className="pb-3 flex-shrink-0">
                  <div className="flex items-center gap-2">
                    <Plug className="w-4 h-4 text-muted-foreground" />
                    <CardTitle className="text-sm">Integrations</CardTitle>
                  </div>
                  <CardDescription className="text-xs">
                    Connected data sources
                  </CardDescription>
                </CardHeader>
                <CardContent className="flex-1 overflow-y-auto space-y-2 min-h-0">
                  {loading ? (
                    <p className="text-xs text-muted-foreground">Loading...</p>
                  ) : connections.length === 0 ? (
                    <p className="text-xs text-muted-foreground">No integrations connected</p>
                  ) : (
                    connections.map((conn) => (
                      <div
                        key={conn.id}
                        className="flex items-center justify-between p-2 rounded-md bg-muted/50 hover:bg-muted transition-colors"
                      >
                        <div className="flex items-center gap-2 flex-1">
                          <CheckCircle2 className="w-4 h-4 text-green-500" />
                          <div className="flex-1">
                            <p className="text-xs font-medium capitalize">{conn.provider}</p>
                            <p className="text-[10px] text-muted-foreground">
                              {conn.last_synced_at ? `Synced ${formatDate(conn.last_synced_at)}` : 'Never synced'}
                            </p>
                          </div>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleSync(conn.provider, conn.id)}
                          disabled={syncing === conn.id}
                        >
                          <RefreshCw className={`w-3 h-3 ${syncing === conn.id ? 'animate-spin' : ''}`} />
                        </Button>
                      </div>
                    ))
                  )}
                </CardContent>
              </Card>

              {/* Section C: Quick Stats - Fixed Height, No Scroll */}
              <Card className="h-[140px]">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm">Quick Stats</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex justify-between text-xs">
                    <span className="text-muted-foreground">Total Files</span>
                    <span className="font-medium">{uploadedFiles.length}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-muted-foreground">Integrations</span>
                    <span className="font-medium">{connections.length}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-muted-foreground">Total Sources</span>
                    <span className="font-medium">{uploadedFiles.length + connections.length}</span>
                  </div>
                </CardContent>
              </Card>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};
