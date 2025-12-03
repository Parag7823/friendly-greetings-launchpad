import { useState, useEffect, useMemo } from 'react';
import { X, FileSpreadsheet, Info, Loader2, CheckCircle2, AlertCircle, Database } from 'lucide-react';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Card } from './ui/card';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip';
import { motion, AnimatePresence } from 'framer-motion';
import { supabase } from '@/integrations/supabase/client';
import { useAuth } from './AuthProvider';
import { cn } from '@/lib/utils';
import { getFileIcon } from '@/utils/fileHelpers';
import { ChatHistoryModal } from './ChatHistoryModal';

// Ag-Grid Imports
import { AgGridReact } from 'ag-grid-react';
import { ColDef, ModuleRegistry } from 'ag-grid-community';
import { ClientSideRowModelModule } from 'ag-grid-community';
import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-quartz.css';
import '@/styles/ag-grid-finley-theme.css';

// Register Ag-Grid modules
ModuleRegistry.registerModules([ClientSideRowModelModule]);

interface OpenFile {
  id: string;
  filename: string;
  status: string;
  progress: number;
  created_at: string;
  platform?: string;
  events_count?: number;
  error_message?: string;
  file_path?: string;
  storage_path?: string;
}

interface TabbedFilePreviewProps {
  openFiles: OpenFile[];
  activeFileId: string | null;
  onFileSelect: (fileId: string) => void;
  onFileClose: (fileId: string) => void;
  onToggleDataSources?: () => void;
  isDataSourcesCollapsed?: boolean;
}

export const TabbedFilePreview = ({
  openFiles,
  activeFileId,
  onFileSelect,
  onFileClose,
  onToggleDataSources,
  isDataSourcesCollapsed = true
}: TabbedFilePreviewProps) => {
  const { user } = useAuth();
  const [fileContent, setFileContent] = useState<Record<string, any[]>>({});
  const [loadingContent, setLoadingContent] = useState<Record<string, boolean>>({});
  const [showMetadata, setShowMetadata] = useState<Record<string, boolean>>({});
  const [showChatHistory, setShowChatHistory] = useState(false);
  const [chatTitle, setChatTitle] = useState('New Chat');

  const activeFile = openFiles.find(f => f.id === activeFileId);

  // Listen for chat title updates from ChatInterface
  useEffect(() => {
    const handleChatTitleUpdate = (event: CustomEvent) => {
      const { title } = event.detail;
      if (title) {
        setChatTitle(title);
      }
    };

    window.addEventListener('chat-title-updated', handleChatTitleUpdate as EventListener);
    return () => window.removeEventListener('chat-title-updated', handleChatTitleUpdate as EventListener);
  }, []);

  // FIX #14: Cleanup file content cache when file is closed to prevent memory leaks
  useEffect(() => {
    return () => {
      // Clear content for files that are no longer in openFiles
      setFileContent(prev => {
        const activeFileIds = new Set(openFiles.map(f => f.id));
        const cleaned = { ...prev };
        Object.keys(cleaned).forEach(fileId => {
          if (!activeFileIds.has(fileId)) {
            delete cleaned[fileId];
          }
        });
        return cleaned;
      });
      
      // Clear loading state for closed files
      setLoadingContent(prev => {
        const activeFileIds = new Set(openFiles.map(f => f.id));
        const cleaned = { ...prev };
        Object.keys(cleaned).forEach(fileId => {
          if (!activeFileIds.has(fileId)) {
            delete cleaned[fileId];
          }
        });
        return cleaned;
      });
    };
  }, [openFiles]);

  // Load raw file content for active file
  useEffect(() => {
    if (!activeFileId || loadingContent[activeFileId] || fileContent[activeFileId]) return;

    const loadFileContent = async () => {
      setLoadingContent(prev => ({ ...prev, [activeFileId]: true }));
      try {
        // BUG #8 FIX: Load only first 100 rows initially for performance
        // AG-Grid will request more rows as user scrolls (server-side pagination)
        // This reduces initial memory usage from 50,000 rows to 100 rows
        const { data: events, error } = await supabase
          .from('raw_events')
          .select('*')
          .eq('file_id', activeFileId)
          .order('source_ts', { ascending: true })
          .limit(100); // Load first 100 rows only

        if (error) {
          console.error('Failed to load events:', error);
          return;
        }

        if (!events || events.length === 0) {
          console.log('No events found for file:', activeFileId);
          return;
        }

        // Convert events to table rows
        const rows = events.map(event => {
          const row: any = {};
          // Extract key fields for display
          if (event.ingest_ts) row['Date'] = new Date(event.ingest_ts).toLocaleDateString();
          if (event.vendor_standard) row['Vendor'] = event.vendor_standard;
          if (event.amount_usd !== null) row['Amount'] = Number(event.amount_usd); // Keep as number for sorting
          if (event.kind) row['Type'] = event.kind;
          if (event.source_platform) row['Platform'] = event.source_platform;

          // Add payload fields
          if (event.payload) {
            Object.keys(event.payload).forEach(key => {
              if (!row[key]) {
                row[key] = event.payload[key];
              }
            });
          }

          return row;
        });

        setFileContent(prev => ({ ...prev, [activeFileId]: rows }));
      } catch (error) {
        console.error('Failed to load file content:', error);
      } finally {
        setLoadingContent(prev => ({ ...prev, [activeFileId]: false }));
      }
    };

    loadFileContent();
  }, [activeFileId]);

  // Dynamic Column Definitions
  const columnDefs = useMemo<ColDef[]>(() => {
    if (!activeFileId || !fileContent[activeFileId] || fileContent[activeFileId].length === 0) {
      return [];
    }

    const firstRow = fileContent[activeFileId][0];
    return Object.keys(firstRow).map(key => {
      const isAmount = key === 'Amount';
      const isDate = key === 'Date';

      return {
        field: key,
        headerName: key,
        sortable: true,
        filter: true,
        resizable: true,
        flex: 1,
        minWidth: 100,
        // Specific formatting
        valueFormatter: isAmount ? (params: any) => {
          return params.value ? `$${params.value.toFixed(2)}` : '';
        } : undefined,
        cellClass: isAmount ? 'text-right font-mono' : undefined,
        headerClass: isAmount ? 'ag-right-aligned-header' : undefined,
      };
    });
  }, [activeFileId, fileContent]);

  const defaultColDef = useMemo(() => ({
    sortable: true,
    filter: true,
    resizable: true,
  }), []);


  const getStatusBadge = (file: OpenFile) => {
    const isProcessing = file.status === 'processing' || file.status === 'pending';
    const isCompleted = file.status === 'completed';
    const isFailed = file.status === 'failed';

    if (isProcessing) {
      return (
        <Badge variant="secondary" className="text-[10px] ml-2">
          <Loader2 className="w-3 h-3 mr-1 animate-spin" />
          {file.progress}%
        </Badge>
      );
    }
    if (isCompleted) {
      return (
        <Badge variant="default" className="text-[10px] ml-2 bg-green-500/10 text-green-600 border-green-500/20">
          <CheckCircle2 className="w-3 h-3" />
        </Badge>
      );
    }
    if (isFailed) {
      return (
        <Badge variant="destructive" className="text-[10px] ml-2">
          <AlertCircle className="w-3 h-3" />
        </Badge>
      );
    }
    return null;
  };

  if (openFiles.length === 0) {
    return (
      <div className="h-full flex flex-col">
        {/* Header Bar - Aligned with Chat "New Chat" bar */}
        <div className="flex items-start justify-between px-4 py-2 bg-background border-b border-border/20">
          {/* Left side - Reserved for future features/routing */}
          <div className="flex items-center gap-2">
            {/* Feature icons will be added here */}
          </div>
          {/* Right side - Data Sources Button */}
          <div className="flex items-start gap-2">
            <Button
              onClick={onToggleDataSources}
              className="h-7 px-3 rounded-md bg-primary/10 hover:bg-primary/20 text-white border border-primary/30 text-xs font-medium transition-colors"
              title={isDataSourcesCollapsed ? "Open Data Sources" : "Close Data Sources"}
            >
              <Database className="w-3.5 h-3.5 mr-1.5" />
              Data Sources
            </Button>
          </div>
        </div>
        
        {/* File Preview Container */}
        <div className="flex-1 flex flex-col bg-background overflow-hidden">
          {/* Empty State Content */}
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center space-y-3 p-8">
            <h3 className="text-2xl font-semibold text-foreground">Finance Workspace</h3>
          </div>
        </div>
        </div>

        {/* Chat History Modal */}
        <ChatHistoryModal
          isOpen={showChatHistory}
          onClose={() => setShowChatHistory(false)}
          onSelectChat={(chatId, title) => {
            setChatTitle(title);
            window.dispatchEvent(new CustomEvent('chat-selected', {
              detail: { chatId }
            }));
          }}
        />
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header Bar - Aligned with Chat "New Chat" bar */}
      <div className="flex items-start justify-between px-4 py-2 bg-background border-b border-border/20">
        {/* Left side - Reserved for future features/routing */}
        <div className="flex items-center gap-2">
          {/* Feature icons will be added here */}
        </div>
        {/* Right side - Data Sources Button */}
        <div className="flex items-start gap-2">
          <Button
            onClick={onToggleDataSources}
            className="h-7 px-3 rounded-md bg-primary/10 hover:bg-primary/20 text-white border border-primary/30 text-xs font-medium transition-colors"
            title={isDataSourcesCollapsed ? "Open Data Sources" : "Close Data Sources"}
          >
            <Database className="w-3.5 h-3.5 mr-1.5" />
            Data Sources
          </Button>
        </div>
      </div>
      
      {/* File Preview Container */}
      <div className="flex-1 flex flex-col bg-background overflow-hidden">
        {/* File Toolbar Header */}
        <div className="bg-background/80 backdrop-blur-sm px-4 py-3">
        <div className="flex items-center justify-between">
          {/* File Name Display */}
          <div className="flex items-center gap-3 min-w-0">
            <FileSpreadsheet className="w-4 h-4 text-primary flex-shrink-0" />
            <div className="min-w-0">
              <h2 className="text-[10px] font-medium text-foreground truncate">
                {activeFile?.filename || 'File Preview'}
              </h2>
              {activeFile && (
                <p className="text-[9px] text-muted-foreground">
                  {activeFile.events_count ? `${activeFile.events_count} events` : 'Loading...'}
                </p>
              )}
            </div>
          </div>
          
          {/* Action Buttons Placeholder - Will be populated later */}
          <div className="flex items-center gap-2 flex-shrink-0">
            {/* Download, Share, Refresh buttons will go here */}
          </div>
        </div>
      </div>

      {/* Tabs Header */}
      <div className="bg-background/50 backdrop-blur-sm">
        <div className="flex items-center overflow-x-auto scrollbar-thin scrollbar-thumb-border scrollbar-track-transparent">
          {openFiles.map((file) => {
            const Icon = getFileIcon(file.filename);
            const isActive = file.id === activeFileId;

            return (
              <motion.div
                key={file.id}
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className={cn(
                  "flex items-center gap-2 px-4 py-2.5 border-r border-border cursor-pointer group relative",
                  "hover:bg-muted/50 transition-colors min-w-[180px] max-w-[240px]",
                  isActive && "bg-muted/70 border-b-2 border-b-primary"
                )}
                onClick={() => onFileSelect(file.id)}
              >
                <Icon className={cn("w-4 h-4 flex-shrink-0", isActive ? "text-primary" : "text-muted-foreground")} />
                <span className={cn(
                  "text-sm truncate flex-1",
                  isActive ? "font-medium text-foreground" : "text-muted-foreground"
                )}>
                  {file.filename}
                </span>

                {getStatusBadge(file)}

                {/* Info Icon - Shows metadata on hover */}
                <TooltipProvider delayDuration={0}>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-5 w-5 p-0 opacity-0 group-hover:opacity-100 transition-opacity"
                        onClick={(e) => {
                          e.stopPropagation();
                          setShowMetadata(prev => ({ ...prev, [file.id]: !prev[file.id] }));
                        }}
                      >
                        <Info className="w-3 h-3" />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent side="bottom" className="max-w-xs">
                      <div className="space-y-1 text-xs">
                        <div className="flex justify-between gap-4">
                          <span className="text-muted-foreground">Uploaded:</span>
                          <span className="font-medium">{new Date(file.created_at).toLocaleString()}</span>
                        </div>
                        {file.platform && (
                          <div className="flex justify-between gap-4">
                            <span className="text-muted-foreground">Platform:</span>
                            <span className="font-medium">{file.platform}</span>
                          </div>
                        )}
                        {file.events_count && (
                          <div className="flex justify-between gap-4">
                            <span className="text-muted-foreground">Events:</span>
                            <span className="font-medium">{file.events_count}</span>
                          </div>
                        )}
                      </div>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>

                {/* Close Button */}
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-5 w-5 p-0 opacity-0 group-hover:opacity-100 transition-opacity hover:bg-destructive/10 hover:text-destructive"
                  onClick={(e) => {
                    e.stopPropagation();
                    onFileClose(file.id);
                  }}
                >
                  <X className="w-3 h-3" />
                </Button>
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* File Content */}
      <div className="flex-1 overflow-hidden bg-slate-800/20">
        <AnimatePresence mode="wait">
          {activeFile && (
            <motion.div
              key={activeFile.id}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.2 }}
              className="h-full flex flex-col"
            >
              {/* Error Message */}
              {activeFile.status === 'failed' && activeFile.error_message && (
                <div className="p-4">
                  <Card className="p-4 bg-destructive/10 border-destructive/20">
                    <div className="flex items-start gap-3">
                      <AlertCircle className="w-5 h-5 text-destructive flex-shrink-0 mt-0.5" />
                      <div>
                        <p className="text-sm font-medium text-destructive mb-1">Processing Failed</p>
                        <p className="text-xs text-destructive/80">{activeFile.error_message}</p>
                      </div>
                    </div>
                  </Card>
                </div>
              )}

              {/* Ag-Grid Table */}
              {loadingContent[activeFile.id] ? (
                <div className="flex items-center justify-center h-full">
                  <Loader2 className="w-8 h-8 animate-spin text-primary" />
                </div>
              ) : fileContent[activeFile.id] && fileContent[activeFile.id].length > 0 ? (
                <div className="h-full w-full ag-theme-quartz-dark">
                  <AgGridReact
                    rowData={fileContent[activeFile.id]}
                    columnDefs={columnDefs}
                    defaultColDef={defaultColDef}
                    pagination={true}
                    paginationPageSize={50}
                    rowSelection="multiple"
                    animateRows={true}
                  />
                </div>
              ) : (
                <div className="flex items-center justify-center h-full p-12">
                  <Card className="p-12 text-center bg-transparent border-none shadow-none">
                    <FileSpreadsheet className="w-16 h-16 mx-auto text-muted-foreground/30 mb-4" />
                    <h3 className="text-sm font-semibold mb-2">No file content</h3>
                    <p className="text-xs text-muted-foreground">Unable to load file content</p>
                  </Card>
                </div>
              )
              }
            </motion.div>
          )}
        </AnimatePresence>
      </div>
      </div>

      {/* Chat History Modal */}
      <ChatHistoryModal
        isOpen={showChatHistory}
        onClose={() => setShowChatHistory(false)}
        onSelectChat={(chatId, title) => {
          setChatTitle(title);
          // Dispatch event to switch chat in ChatInterface
          window.dispatchEvent(new CustomEvent('chat-selected', {
            detail: { chatId }
          }));
        }}
      />
    </div>
  );
};
