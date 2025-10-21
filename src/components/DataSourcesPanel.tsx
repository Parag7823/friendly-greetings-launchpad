import { useState, useEffect } from 'react';
import { FileSpreadsheet, Plug, RefreshCw, CheckCircle2, AlertCircle, ChevronRight, X } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { useAuth } from './AuthProvider';
import { supabase } from '@/integrations/supabase/client';
import { config } from '@/config';
import { useToast } from './ui/use-toast';
import { motion, AnimatePresence } from 'framer-motion';

interface DataSourcesPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

export const DataSourcesPanel = ({ isOpen, onClose }: DataSourcesPanelProps) => {
  const { user } = useAuth();
  const { toast } = useToast();
  const [uploadedFiles, setUploadedFiles] = useState<any[]>([]);
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

              {/* Uploaded Files */}
              <Card>
                <CardHeader className="pb-3">
                  <div className="flex items-center gap-2">
                    <FileSpreadsheet className="w-4 h-4 text-muted-foreground" />
                    <CardTitle className="text-sm">Uploaded Files</CardTitle>
                  </div>
                  <CardDescription className="text-xs">
                    Recent file uploads
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-2">
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

              {/* Connected Integrations */}
              <Card>
                <CardHeader className="pb-3">
                  <div className="flex items-center gap-2">
                    <Plug className="w-4 h-4 text-muted-foreground" />
                    <CardTitle className="text-sm">Integrations</CardTitle>
                  </div>
                  <CardDescription className="text-xs">
                    Connected data sources
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-2">
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

              {/* Quick Stats */}
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm">Quick Stats</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
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
