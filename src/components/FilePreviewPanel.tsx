import { useState, useEffect } from 'react';
import { X, FileSpreadsheet, Calendar, Clock, CheckCircle2, AlertCircle, Loader2, Download } from 'lucide-react';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Card } from './ui/card';
import { motion, AnimatePresence } from 'framer-motion';
import { supabase } from '@/integrations/supabase/client';
import { useAuth } from './AuthProvider';

interface FilePreviewPanelProps {
  fileId: string | null;
  filename: string;
  isOpen: boolean;
  onClose: () => void;
}

interface FileDetails {
  id: string;
  filename?: string;
  status: string;
  progress: number;
  created_at: string;
  error_message?: string;
  file_size?: number;
  events_count?: number;
  platform?: string;
}

export const FilePreviewPanel = ({ fileId, filename, isOpen, onClose }: FilePreviewPanelProps) => {
  const { user } = useAuth();
  const [fileDetails, setFileDetails] = useState<FileDetails | null>(null);
  const [loading, setLoading] = useState(false);
  const [events, setEvents] = useState<any[]>([]);

  useEffect(() => {
    if (!fileId || !isOpen) return;

    const loadFileDetails = async () => {
      setLoading(true);
      try {
        // Load file details
        const { data: fileData, error: fileError } = await supabase
          .from('ingestion_jobs')
          .select('*')
          .eq('id', fileId)
          .single();

        if (fileError) throw fileError;
        setFileDetails({
          ...fileData,
          filename: fileData.filename || filename || 'Unknown File'
        });

        // Load events for this file
        const { data: eventsData, error: eventsError } = await supabase
          .from('raw_events')
          .select('id, payload, kind, source_platform, created_at')
          .eq('file_id', fileId)
          .order('created_at', { ascending: false })
          .limit(10);

        if (!eventsError && eventsData) {
          setEvents(eventsData);
        }
      } catch (error) {
        console.error('Failed to load file details:', error);
      } finally {
        setLoading(false);
      }
    };

    loadFileDetails();

    // Poll for updates if file is processing
    const interval = setInterval(() => {
      if (fileDetails?.status === 'processing' || fileDetails?.status === 'pending') {
        loadFileDetails();
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [fileId, isOpen, fileDetails?.status]);

  if (!isOpen || !fileId) return null;

  const isProcessing = fileDetails?.status === 'processing' || fileDetails?.status === 'pending';
  const isCompleted = fileDetails?.status === 'completed';
  const isFailed = fileDetails?.status === 'failed';

  return (
    <AnimatePresence>
      <motion.div
        initial={{ x: '100%' }}
        animate={{ x: 0 }}
        exit={{ x: '100%' }}
        transition={{ type: 'spring', damping: 25, stiffness: 200 }}
        className="fixed right-0 top-0 h-full w-full md:w-[600px] finley-dynamic-bg border-l border-border shadow-2xl z-40 flex flex-col"
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border">
          <div className="flex items-center gap-3 flex-1 min-w-0">
            <FileSpreadsheet className="w-5 h-5 text-primary flex-shrink-0" />
            <div className="flex-1 min-w-0">
              <h2 className="text-base font-semibold truncate">{filename}</h2>
              {fileDetails && (
                <div className="flex items-center gap-2 mt-1">
                  {isProcessing && (
                    <Badge variant="secondary" className="text-[10px]">
                      <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                      Processing...
                    </Badge>
                  )}
                  {isCompleted && (
                    <Badge variant="default" className="text-[10px] bg-green-500/10 text-green-600 border-green-500/20">
                      <CheckCircle2 className="w-3 h-3 mr-1" />
                      Completed
                    </Badge>
                  )}
                  {isFailed && (
                    <Badge variant="destructive" className="text-[10px]">
                      <AlertCircle className="w-3 h-3 mr-1" />
                      Failed
                    </Badge>
                  )}
                </div>
              )}
            </div>
          </div>
          <Button variant="ghost" size="sm" onClick={onClose}>
            <X className="w-4 h-4" />
          </Button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="w-8 h-8 animate-spin text-primary" />
            </div>
          ) : fileDetails ? (
            <>
              {/* File Info Card */}
              <Card className="p-4 space-y-3">
                <h3 className="text-sm font-semibold text-foreground">File Information</h3>
                
                <div className="space-y-2 text-sm">
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground flex items-center gap-2">
                      <Calendar className="w-4 h-4" />
                      Uploaded
                    </span>
                    <span className="font-medium">
                      {new Date(fileDetails.created_at).toLocaleDateString()}
                    </span>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground flex items-center gap-2">
                      <Clock className="w-4 h-4" />
                      Time
                    </span>
                    <span className="font-medium">
                      {new Date(fileDetails.created_at).toLocaleTimeString()}
                    </span>
                  </div>

                  {fileDetails.platform && (
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Platform</span>
                      <Badge variant="outline" className="text-[10px]">
                        {fileDetails.platform}
                      </Badge>
                    </div>
                  )}

                  {events.length > 0 && (
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Events Extracted</span>
                      <Badge variant="default" className="text-[10px]">
                        {events.length}
                      </Badge>
                    </div>
                  )}
                </div>

                {/* Progress Bar */}
                {isProcessing && fileDetails.progress > 0 && (
                  <div className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-muted-foreground">Progress</span>
                      <span className="font-medium text-primary">{fileDetails.progress}%</span>
                    </div>
                    <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-primary transition-all duration-300"
                        style={{ width: `${fileDetails.progress}%` }}
                      />
                    </div>
                  </div>
                )}

                {/* Error Message */}
                {isFailed && fileDetails.error_message && (
                  <div className="p-3 bg-destructive/10 border border-destructive/20 rounded-md">
                    <p className="text-xs text-destructive font-medium mb-1">Error</p>
                    <p className="text-xs text-destructive/80">{fileDetails.error_message}</p>
                  </div>
                )}
              </Card>

              {/* Events Preview */}
              {events.length > 0 && (
                <Card className="p-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-foreground">Recent Events</h3>
                    <Badge variant="outline" className="text-[10px]">
                      {events.length} shown
                    </Badge>
                  </div>
                  
                  <div className="space-y-2 max-h-[400px] overflow-y-auto">
                    {events.map((event, index) => (
                      <div 
                        key={event.id}
                        className="p-3 border border-border rounded-md hover:bg-muted/20 transition-colors"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <Badge variant="outline" className="text-[10px]">
                            {event.kind || 'Unknown'}
                          </Badge>
                          <span className="text-[10px] text-muted-foreground">
                            {new Date(event.created_at).toLocaleTimeString()}
                          </span>
                        </div>
                        
                        {event.payload && (
                          <div className="space-y-1">
                            {Object.entries(event.payload).slice(0, 3).map(([key, value]) => (
                              <div key={key} className="flex items-start gap-2 text-xs">
                                <span className="text-muted-foreground min-w-[80px] truncate">
                                  {key}:
                                </span>
                                <span className="font-medium text-foreground truncate flex-1">
                                  {String(value)}
                                </span>
                              </div>
                            ))}
                            {Object.keys(event.payload).length > 3 && (
                              <p className="text-[10px] text-muted-foreground italic">
                                +{Object.keys(event.payload).length - 3} more fields
                              </p>
                            )}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </Card>
              )}

              {/* No Events Message */}
              {events.length === 0 && isCompleted && (
                <Card className="p-8 text-center">
                  <FileSpreadsheet className="w-12 h-12 mx-auto text-muted-foreground/50 mb-3" />
                  <p className="text-sm text-muted-foreground">No events extracted yet</p>
                  <p className="text-xs text-muted-foreground/70 mt-1">
                    Events will appear here once processing is complete
                  </p>
                </Card>
              )}
            </>
          ) : (
            <Card className="p-8 text-center">
              <AlertCircle className="w-12 h-12 mx-auto text-destructive/50 mb-3" />
              <p className="text-sm text-muted-foreground">Failed to load file details</p>
            </Card>
          )}
        </div>
      </motion.div>
    </AnimatePresence>
  );
};
