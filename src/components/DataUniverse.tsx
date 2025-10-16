import { useState, useEffect } from 'react';
import { 
  Database, 
  Upload, 
  Bell, 
  ChevronDown, 
  ChevronRight,
  Plug,
  FileSpreadsheet,
  Mail,
  Cloud,
  CheckCircle2,
  AlertCircle,
  Clock,
  Loader2
} from 'lucide-react';
import { useAuth } from './AuthProvider';
import { supabase } from '@/integrations/supabase/client';
import { Badge } from './ui/badge';
import { ScrollArea } from './ui/scroll-area';
import { Separator } from './ui/separator';
import { cn } from '@/lib/utils';

interface Connector {
  connection_id: string;
  integration_id: string;
  status: string;
  last_synced_at: string | null;
}

interface UploadedFile {
  id: string;
  filename: string;
  upload_timestamp: string;
  status: string;
  total_rows: number;
}

interface Notification {
  id: string;
  type: 'success' | 'error' | 'info';
  message: string;
  timestamp: string;
}

export const DataUniverse = () => {
  const { user } = useAuth();
  const [connectorsExpanded, setConnectorsExpanded] = useState(true);
  const [uploadsExpanded, setUploadsExpanded] = useState(true);
  const [notificationsExpanded, setNotificationsExpanded] = useState(false);
  
  const [connectors, setConnectors] = useState<Connector[]>([]);
  const [uploads, setUploads] = useState<UploadedFile[]>([]);
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [loading, setLoading] = useState(true);

  // Load connectors
  useEffect(() => {
    const loadConnectors = async () => {
      if (!user?.id) return;
      
      try {
        const { data: sessionData } = await supabase.auth.getSession();
        const sessionToken = sessionData?.session?.access_token;
        
        const resp = await fetch('/api/connectors/user-connections', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            user_id: user.id, 
            session_token: sessionToken 
          }),
        });
        
        if (resp.ok) {
          const data = await resp.json();
          setConnectors(Array.isArray(data?.connections) ? data.connections : []);
        }
      } catch (error) {
        console.error('Failed to load connectors:', error);
      }
    };

    loadConnectors();
    const interval = setInterval(loadConnectors, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, [user?.id]);

  // Load uploaded files
  useEffect(() => {
    const loadUploads = async () => {
      if (!user?.id) return;
      
      try {
        const { data, error } = await supabase
          .from('uploaded_files')
          .select('id, filename, upload_timestamp, status, total_rows')
          .eq('user_id', user.id)
          .order('upload_timestamp', { ascending: false })
          .limit(10);
        
        if (!error && data) {
          setUploads(data);
        }
      } catch (error) {
        console.error('Failed to load uploads:', error);
      } finally {
        setLoading(false);
      }
    };

    loadUploads();
    const interval = setInterval(loadUploads, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, [user?.id]);

  // Mock notifications (replace with real notification system)
  useEffect(() => {
    setNotifications([
      {
        id: '1',
        type: 'success',
        message: 'QuickBooks sync completed successfully',
        timestamp: new Date().toISOString(),
      },
      {
        id: '2',
        type: 'info',
        message: 'New file uploaded: invoice_2025.xlsx',
        timestamp: new Date().toISOString(),
      },
    ]);
  }, []);

  const getConnectorIcon = (integrationId: string) => {
    if (integrationId.includes('mail')) return <Mail className="h-4 w-4" />;
    if (integrationId.includes('drive') || integrationId.includes('dropbox')) return <Cloud className="h-4 w-4" />;
    return <Plug className="h-4 w-4" />;
  };

  const getStatusBadge = (status: string) => {
    const statusMap: Record<string, { variant: 'default' | 'secondary' | 'destructive' | 'outline', icon: React.ReactNode }> = {
      'active': { variant: 'default', icon: <CheckCircle2 className="h-3 w-3" /> },
      'syncing': { variant: 'secondary', icon: <Loader2 className="h-3 w-3 animate-spin" /> },
      'error': { variant: 'destructive', icon: <AlertCircle className="h-3 w-3" /> },
      'pending': { variant: 'outline', icon: <Clock className="h-3 w-3" /> },
    };
    
    const config = statusMap[status] || statusMap['pending'];
    
    return (
      <Badge variant={config.variant} className="flex items-center gap-1 text-xs">
        {config.icon}
        {status}
      </Badge>
    );
  };

  const getConnectorName = (integrationId: string) => {
    const nameMap: Record<string, string> = {
      'google-mail': 'Gmail',
      'zoho-mail': 'Zoho Mail',
      'dropbox': 'Dropbox',
      'google-drive': 'Google Drive',
      'quickbooks': 'QuickBooks',
      'quickbooks-sandbox': 'QuickBooks (Sandbox)',
      'xero': 'Xero',
    };
    return nameMap[integrationId] || integrationId;
  };

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-card">
      {/* Header */}
      <div className="p-4 border-b border-border">
        <div className="flex items-center gap-2">
          <Database className="h-5 w-5 text-primary" />
          <h2 className="text-sm font-semibold text-foreground">Data Universe</h2>
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          All your connected data sources
        </p>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-2 space-y-1">
          {/* Connectors Section */}
          <div>
            <button
              onClick={() => setConnectorsExpanded(!connectorsExpanded)}
              className="w-full flex items-center justify-between p-2 rounded-md hover:bg-muted/50 transition-colors"
            >
              <div className="flex items-center gap-2">
                {connectorsExpanded ? (
                  <ChevronDown className="h-4 w-4 text-muted-foreground" />
                ) : (
                  <ChevronRight className="h-4 w-4 text-muted-foreground" />
                )}
                <Plug className="h-4 w-4 text-primary" />
                <span className="text-sm font-medium text-foreground">Connectors</span>
                <Badge variant="secondary" className="text-xs">
                  {connectors.length}
                </Badge>
              </div>
            </button>

            {connectorsExpanded && (
              <div className="ml-6 mt-1 space-y-1">
                {connectors.length === 0 ? (
                  <div className="p-3 text-xs text-muted-foreground">
                    No connectors yet. Connect your first data source!
                  </div>
                ) : (
                  connectors.map((connector) => (
                    <div
                      key={connector.connection_id}
                      className="p-2 rounded-md hover:bg-muted/50 transition-colors cursor-pointer group"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2 flex-1 min-w-0">
                          {getConnectorIcon(connector.integration_id)}
                          <span className="text-xs font-medium text-foreground truncate">
                            {getConnectorName(connector.integration_id)}
                          </span>
                        </div>
                        {getStatusBadge(connector.status)}
                      </div>
                      {connector.last_synced_at && (
                        <div className="text-xs text-muted-foreground mt-1 ml-6">
                          Last sync: {new Date(connector.last_synced_at).toLocaleString()}
                        </div>
                      )}
                    </div>
                  ))
                )}
              </div>
            )}
          </div>

          <Separator className="my-2" />

          {/* Uploads Section */}
          <div>
            <button
              onClick={() => setUploadsExpanded(!uploadsExpanded)}
              className="w-full flex items-center justify-between p-2 rounded-md hover:bg-muted/50 transition-colors"
            >
              <div className="flex items-center gap-2">
                {uploadsExpanded ? (
                  <ChevronDown className="h-4 w-4 text-muted-foreground" />
                ) : (
                  <ChevronRight className="h-4 w-4 text-muted-foreground" />
                )}
                <Upload className="h-4 w-4 text-primary" />
                <span className="text-sm font-medium text-foreground">Uploads</span>
                <Badge variant="secondary" className="text-xs">
                  {uploads.length}
                </Badge>
              </div>
            </button>

            {uploadsExpanded && (
              <div className="ml-6 mt-1 space-y-1">
                {uploads.length === 0 ? (
                  <div className="p-3 text-xs text-muted-foreground">
                    No files uploaded yet. Upload your first file!
                  </div>
                ) : (
                  uploads.map((upload) => (
                    <div
                      key={upload.id}
                      className="p-2 rounded-md hover:bg-muted/50 transition-colors cursor-pointer group"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2 flex-1 min-w-0">
                          <FileSpreadsheet className="h-4 w-4 text-primary flex-shrink-0" />
                          <span className="text-xs font-medium text-foreground truncate">
                            {upload.filename}
                          </span>
                        </div>
                        {getStatusBadge(upload.status)}
                      </div>
                      <div className="text-xs text-muted-foreground mt-1 ml-6">
                        {upload.total_rows} rows • {new Date(upload.upload_timestamp).toLocaleDateString()}
                      </div>
                    </div>
                  ))
                )}
              </div>
            )}
          </div>

          <Separator className="my-2" />

          {/* Notifications Section */}
          <div>
            <button
              onClick={() => setNotificationsExpanded(!notificationsExpanded)}
              className="w-full flex items-center justify-between p-2 rounded-md hover:bg-muted/50 transition-colors"
            >
              <div className="flex items-center gap-2">
                {notificationsExpanded ? (
                  <ChevronDown className="h-4 w-4 text-muted-foreground" />
                ) : (
                  <ChevronRight className="h-4 w-4 text-muted-foreground" />
                )}
                <Bell className="h-4 w-4 text-primary" />
                <span className="text-sm font-medium text-foreground">Notifications</span>
                {notifications.length > 0 && (
                  <Badge variant="destructive" className="text-xs">
                    {notifications.length}
                  </Badge>
                )}
              </div>
            </button>

            {notificationsExpanded && (
              <div className="ml-6 mt-1 space-y-1">
                {notifications.length === 0 ? (
                  <div className="p-3 text-xs text-muted-foreground">
                    No new notifications
                  </div>
                ) : (
                  notifications.map((notification) => (
                    <div
                      key={notification.id}
                      className={cn(
                        "p-2 rounded-md transition-colors cursor-pointer",
                        notification.type === 'error' && "bg-destructive/10 hover:bg-destructive/20",
                        notification.type === 'success' && "bg-primary/10 hover:bg-primary/20",
                        notification.type === 'info' && "hover:bg-muted/50"
                      )}
                    >
                      <div className="flex items-start gap-2">
                        {notification.type === 'success' && <CheckCircle2 className="h-4 w-4 text-primary flex-shrink-0 mt-0.5" />}
                        {notification.type === 'error' && <AlertCircle className="h-4 w-4 text-destructive flex-shrink-0 mt-0.5" />}
                        {notification.type === 'info' && <Bell className="h-4 w-4 text-muted-foreground flex-shrink-0 mt-0.5" />}
                        <div className="flex-1 min-w-0">
                          <p className="text-xs text-foreground">{notification.message}</p>
                          <p className="text-xs text-muted-foreground mt-1">
                            {new Date(notification.timestamp).toLocaleTimeString()}
                          </p>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            )}
          </div>
        </div>
      </ScrollArea>
    </div>
  );
};
