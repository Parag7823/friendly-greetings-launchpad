import { useState, useEffect } from 'react';
import { X, FileSpreadsheet, FileText, File, Info, Download, Loader2, CheckCircle2, AlertCircle } from 'lucide-react';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Card } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip';
import { motion, AnimatePresence } from 'framer-motion';
import { supabase } from '@/integrations/supabase/client';
import { useAuth } from './AuthProvider';
import { cn } from '@/lib/utils';

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
}

export const TabbedFilePreview = ({ 
  openFiles, 
  activeFileId, 
  onFileSelect, 
  onFileClose 
}: TabbedFilePreviewProps) => {
  const { user } = useAuth();
  const [fileContent, setFileContent] = useState<Record<string, any[]>>({});
  const [loadingContent, setLoadingContent] = useState<Record<string, boolean>>({});
  const [showMetadata, setShowMetadata] = useState<Record<string, boolean>>({});

  const activeFile = openFiles.find(f => f.id === activeFileId);

  // Load raw file content for active file
  useEffect(() => {
    if (!activeFileId || loadingContent[activeFileId] || fileContent[activeFileId]) return;

    const loadFileContent = async () => {
      setLoadingContent(prev => ({ ...prev, [activeFileId]: true }));
      try {
        // Fetch file metadata to get storage path
        const { data: fileData, error: fileError } = await supabase
          .from('ingestion_jobs')
          .select('file_path, storage_path')
          .eq('id', activeFileId)
          .single();

        if (fileError || !fileData?.storage_path) {
          console.error('Failed to get file path:', fileError);
          return;
        }

        // Download file from Supabase Storage
        const { data: fileBlob, error: downloadError } = await supabase.storage
          .from('uploaded-files')
          .download(fileData.storage_path);

        if (downloadError || !fileBlob) {
          console.error('Failed to download file:', downloadError);
          return;
        }

        // Parse CSV content
        const text = await fileBlob.text();
        const lines = text.split('\n').filter(line => line.trim());
        
        if (lines.length === 0) return;

        // Parse CSV (simple parser - handles basic CSV)
        const headers = lines[0].split(',').map(h => h.trim());
        const rows = lines.slice(1, 101).map(line => { // Limit to 100 rows for performance
          const values = line.split(',').map(v => v.trim());
          const row: any = {};
          headers.forEach((header, index) => {
            row[header] = values[index] || '';
          });
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

  const getFileIcon = (filename: string) => {
    if (!filename) return File;
    const ext = filename.split('.').pop()?.toLowerCase();
    if (ext === 'csv' || ext === 'xlsx' || ext === 'xls') return FileSpreadsheet;
    if (ext === 'txt' || ext === 'md') return FileText;
    return File;
  };

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
      <div className="h-full flex items-center justify-center finley-dynamic-bg">
        <div className="text-center space-y-3 p-8">
          <FileSpreadsheet className="w-16 h-16 mx-auto text-muted-foreground/30" />
          <div>
            <h3 className="text-lg font-semibold text-foreground">No files open</h3>
            <p className="text-sm text-muted-foreground mt-1">
              Click a file in Data Sources to view it here
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col finley-dynamic-bg">
      {/* Tabs Header */}
      <div className="border-b border-border bg-background/50 backdrop-blur-sm">
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
      <div className="flex-1 overflow-hidden">
        <AnimatePresence mode="wait">
          {activeFile && (
            <motion.div
              key={activeFile.id}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.2 }}
              className="h-full overflow-y-auto p-4"
            >
              {/* Error Message */}
              {activeFile.status === 'failed' && activeFile.error_message && (
                <Card className="p-4 bg-destructive/10 border-destructive/20 mb-4">
                  <div className="flex items-start gap-3">
                    <AlertCircle className="w-5 h-5 text-destructive flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="text-sm font-medium text-destructive mb-1">Processing Failed</p>
                      <p className="text-xs text-destructive/80">{activeFile.error_message}</p>
                    </div>
                  </div>
                </Card>
              )}

              {/* File Content Table */}
              {loadingContent[activeFile.id] ? (
                  <div className="flex items-center justify-center py-12">
                    <Loader2 className="w-8 h-8 animate-spin text-primary" />
                  </div>
                ) : fileContent[activeFile.id] && fileContent[activeFile.id].length > 0 ? (
                  <div className="border border-border rounded-lg overflow-auto max-h-[600px]">
                    <table className="w-full text-[10px]">
                      <thead className="bg-muted/50 sticky top-0">
                        <tr>
                          <th className="px-2 py-1.5 text-left font-medium text-muted-foreground border-r border-border w-10">#</th>
                          {Object.keys(fileContent[activeFile.id][0] || {}).map((header) => (
                            <th key={header} className="px-2 py-1.5 text-left font-medium border-r border-border whitespace-nowrap">
                              {header}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="font-mono text-[9px]">
                        {fileContent[activeFile.id].map((row, index) => (
                          <tr key={index} className="border-t border-border hover:bg-muted/20">
                            <td className="px-2 py-1 text-muted-foreground border-r border-border text-right">{index + 1}</td>
                            {Object.values(row).map((value: any, colIndex) => (
                              <td key={colIndex} className="px-2 py-1 border-r border-border whitespace-nowrap">
                                {value}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <Card className="p-12 text-center">
                    <FileSpreadsheet className="w-16 h-16 mx-auto text-muted-foreground/30 mb-4" />
                    <h3 className="text-[10px] font-semibold mb-2">No file content</h3>
                    <p className="text-[8px] text-muted-foreground">Unable to load file content</p>
                  </Card>
                )
              }
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};
