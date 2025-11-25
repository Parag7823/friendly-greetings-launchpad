import React, { useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Sheet, SheetContent, SheetHeader, SheetTitle } from './ui/sheet';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Pause, CheckCircle, AlertCircle, Loader2, Clock } from 'lucide-react';
import { Progress } from './ui/progress';
import { useFileStatusStore } from '@/stores/useFileStatusStore';
import { useWebSocket } from '@/contexts/WebSocketContext';

/**
 * FileStatusSheet Component
 * 
 * Slide-over panel showing real-time file processing status with:
 * - Step-by-step progress visualization
 * - Animated status indicators
 * - Overall progress bar
 * - Friendly status messages from backend
 * 
 * Integrates with:
 * - useFileStatusStore (Zustand) for state
 * - WebSocket updates from backend
 * - Real-time progress tracking
 */

interface FileStatusSheetProps {
  fileId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export const FileStatusSheet: React.FC<FileStatusSheetProps> = ({
  fileId,
  open,
  onOpenChange,
}) => {
  const status = useFileStatusStore((state) => state.getStatus(fileId));
  const { emit, isConnected } = useWebSocket();
  const [isPaused, setIsPaused] = React.useState(false);

  // ERROR #5 FIX: Use shared WebSocket context instead of creating new hook
  const handlePause = useCallback(() => {
    if (!isConnected || !fileId) return;
    
    try {
      emit('pause_processing', { fileId });
      setIsPaused(true);
    } catch (error) {
      console.error('Failed to pause processing:', error);
    }
  }, [isConnected, fileId, emit]);

  if (!status) {
    return (
      <Sheet open={open} onOpenChange={onOpenChange}>
        <SheetContent side="right" className="w-96">
          <SheetHeader>
            <SheetTitle>Processing Status</SheetTitle>
          </SheetHeader>
          <div className="flex items-center justify-center h-32">
            <p className="text-sm text-muted-foreground">No status available</p>
          </div>
        </SheetContent>
      </Sheet>
    );
  }

  const steps = status.steps || [];
  const overallProgress = status.overallProgress || 0;
  const isComplete = status.completedAt !== undefined;
  const hasError = status.error !== undefined;

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="w-96 space-y-4">
        {/* Header */}
        <SheetHeader>
          <div className="flex items-center justify-between gap-2">
            <SheetTitle>Processing Status</SheetTitle>
            {!isComplete && !hasError && !isPaused && (
              <Button
                size="sm"
                variant="outline"
                onClick={handlePause}
                className="gap-1"
                title="Pause file processing"
              >
                <Pause className="w-3.5 h-3.5" />
                <span className="text-xs">Pause</span>
              </Button>
            )}
            {isPaused && (
              <span className="text-xs font-medium text-amber-600">⏸ Paused</span>
            )}
            {isComplete && !hasError && (
              <span className="text-xs font-normal text-green-600">✓ Complete</span>
            )}
            {hasError && (
              <span className="text-xs font-normal text-destructive">✗ Failed</span>
            )}
          </div>
        </SheetHeader>

        {/* Filename */}
        <div className="space-y-1">
          <p className="text-xs text-muted-foreground">File</p>
          <p className="text-sm font-medium truncate" title={status.filename}>
            {status.filename || 'Unknown File'}
          </p>
        </div>

        {/* Overall Progress */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <p className="text-xs text-muted-foreground">Overall Progress</p>
            <span className="text-sm font-semibold text-primary">
              {Math.round(overallProgress)}%
            </span>
          </div>
          <Progress value={overallProgress} className="h-2" />
        </div>

        {/* Divider */}
        <div className="h-px bg-border" />

        {/* Steps Timeline */}
        <div className="space-y-3">
          <p className="text-xs font-semibold text-muted-foreground">Processing Steps</p>

          <AnimatePresence mode="popLayout">
            {steps.length === 0 ? (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="text-center py-4"
              >
                <p className="text-xs text-muted-foreground">Waiting to start...</p>
              </motion.div>
            ) : (
              <div className="space-y-2 max-h-[400px] overflow-y-auto pr-2">
                {steps.map((step, index) => {
                  const isCurrentStep = index === steps.length - 1 && !isComplete;
                  const isCompleted = step.status === 'complete';
                  const isFailed = step.status === 'error';

                  return (
                    <motion.div
                      key={`${step.step}-${index}`}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: 20 }}
                      transition={{ duration: 0.3 }}
                      className="flex items-start gap-3 p-2 rounded-md hover:bg-muted/50 transition-colors"
                    >
                      {/* Status Icon */}
                      <div className="flex-shrink-0 mt-0.5">
                        {isCompleted ? (
                          <motion.div
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            transition={{ type: 'spring', stiffness: 200 }}
                          >
                            <CheckCircle className="w-4 h-4 text-green-500" />
                          </motion.div>
                        ) : isFailed ? (
                          <AlertCircle className="w-4 h-4 text-destructive" />
                        ) : isCurrentStep ? (
                          <motion.div
                            animate={{ rotate: 360 }}
                            transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                          >
                            <Loader2 className="w-4 h-4 text-primary" />
                          </motion.div>
                        ) : (
                          <Clock className="w-4 h-4 text-muted-foreground/50" />
                        )}
                      </div>

                      {/* Step Info */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <p className="text-xs font-medium text-foreground">
                            {step.step.replace(/_/g, ' ').charAt(0).toUpperCase() +
                              step.step.replace(/_/g, ' ').slice(1)}
                          </p>
                          {step.progress !== undefined && (
                            <span className="text-xs text-muted-foreground">
                              {step.progress}%
                            </span>
                          )}
                        </div>

                        {/* Friendly Message */}
                        <p className="text-xs text-muted-foreground mt-0.5 line-clamp-2">
                          {step.message}
                        </p>

                        {/* Timestamp */}
                        {step.timestamp && (
                          <p className="text-[10px] text-muted-foreground/50 mt-1">
                            {new Date(step.timestamp).toLocaleTimeString()}
                          </p>
                        )}
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            )}
          </AnimatePresence>
        </div>

        {/* Error Display */}
        {hasError && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="p-3 rounded-md bg-destructive/10 border border-destructive/20"
          >
            <p className="text-xs font-semibold text-destructive mb-1">Error</p>
            <p className="text-xs text-destructive/80">{status.error}</p>
          </motion.div>
        )}

        {/* Completion Info */}
        {isComplete && !hasError && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="p-3 rounded-md bg-green-500/10 border border-green-500/20"
          >
            <p className="text-xs font-semibold text-green-600">
              ✓ Processing Complete
            </p>
            <p className="text-xs text-green-600/80 mt-1">
              Your file has been successfully processed and is ready for analysis.
            </p>
          </motion.div>
        )}

        {/* Metadata */}
        {status.startedAt && (
          <div className="text-[10px] text-muted-foreground space-y-1 pt-2 border-t">
            <p>Started: {new Date(status.startedAt).toLocaleString()}</p>
            {status.completedAt && (
              <p>
                Completed: {new Date(status.completedAt).toLocaleString()}
              </p>
            )}
          </div>
        )}
      </SheetContent>
    </Sheet>
  );
};
