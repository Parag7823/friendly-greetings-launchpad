import React from 'react';
import { 
  CheckCircle, 
  AlertCircle, 
  Loader2, 
  X, 
  FileSpreadsheet, 
  Trash2,
  AlertTriangle
} from 'lucide-react';
import { Progress } from './ui/progress';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { cn } from '@/lib/utils';

export type FileStatus = 'queued' | 'uploading' | 'processing' | 'completed' | 'failed' | 'cancelled';

export interface FileRowData {
  id: string;
  file: File;
  status: FileStatus;
  progress: number;
  currentStep?: string;
  error?: string;
  jobId?: string;
  analysisResults?: any;
  sheetProgress?: {
    currentSheet: string;
    sheetsCompleted: number;
    totalSheets: number;
  };
}

interface FileRowProps {
  fileData: FileRowData;
  onCancel: (fileId: string) => void;
  onRemove: (fileId: string) => void;
  className?: string;
}

export const FileRow: React.FC<FileRowProps> = ({
  fileData,
  onCancel,
  onRemove,
  className
}) => {
  const {
    id,
    file,
    status,
    progress,
    currentStep,
    error,
    sheetProgress
  } = fileData;

  const getStatusIcon = () => {
    switch (status) {
      case 'queued':
        return <div className="w-4 h-4 rounded-full bg-muted-foreground/30" />;
      case 'uploading':
      case 'processing':
        return <Loader2 className="w-4 h-4 text-finley-accent animate-spin" />;
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'failed':
        return <AlertCircle className="w-4 h-4 text-destructive" />;
      case 'cancelled':
        return <AlertTriangle className="w-4 h-4 text-yellow-500" />;
      default:
        return <Loader2 className="w-4 h-4 text-finley-accent animate-spin" />;
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'queued':
        return 'bg-muted-foreground/20 text-muted-foreground';
      case 'uploading':
      case 'processing':
        return 'bg-finley-accent/20 text-finley-accent';
      case 'completed':
        return 'bg-green-500/20 text-green-500';
      case 'failed':
        return 'bg-destructive/20 text-destructive';
      case 'cancelled':
        return 'bg-yellow-500/20 text-yellow-500';
      default:
        return 'bg-muted-foreground/20 text-muted-foreground';
    }
  };

  const getStatusText = () => {
    switch (status) {
      case 'queued':
        return 'Queued';
      case 'uploading':
        return 'Uploading...';
      case 'processing':
        return 'Processing...';
      case 'completed':
        return 'Completed';
      case 'failed':
        return 'Failed';
      case 'cancelled':
        return 'Cancelled';
      default:
        return 'Unknown';
    }
  };

  const canCancel = status === 'uploading' || status === 'processing';
  const canRemove = status === 'completed' || status === 'failed' || status === 'cancelled';

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className={cn(
      "group relative flex items-center gap-4 p-4 rounded-lg border transition-all duration-200",
      "hover:bg-muted/30 hover:border-finley-accent/30",
      "bg-card border-border",
      className
    )}>
      {/* File Icon */}
      <div className="flex-shrink-0">
        <FileSpreadsheet className="w-5 h-5 text-finley-accent" />
      </div>

      {/* File Info */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <h4 className="text-sm font-medium text-foreground truncate">
            {file.name}
          </h4>
          <Badge 
            variant="secondary" 
            className={cn("text-xs px-2 py-0.5", getStatusColor())}
          >
            {getStatusText()}
          </Badge>
        </div>
        
        <div className="flex items-center gap-2 text-xs text-muted-foreground mb-2">
          <span>{formatFileSize(file.size)}</span>
          {sheetProgress && (
            <>
              <span>•</span>
              <span>{sheetProgress.sheetsCompleted}/{sheetProgress.totalSheets} sheets</span>
            </>
          )}
        </div>

        {/* Progress Bar */}
        {(status === 'uploading' || status === 'processing') && (
          <div className="space-y-1">
            <Progress 
              value={progress} 
              className="h-1.5 bg-muted/50"
            />
            <div className="flex items-center justify-between text-xs">
              <span className="text-muted-foreground">
                {currentStep || 'Processing...'}
              </span>
              <span className="text-finley-accent font-medium">
                {Math.round(progress)}%
              </span>
            </div>
            {sheetProgress?.currentSheet && (
              <div className="text-xs text-finley-accent">
                Processing: {sheetProgress.currentSheet}
              </div>
            )}
          </div>
        )}

        {/* Error Message */}
        {/* Status Messages (Error, Warning, Info) */}
        {error && (
          <div className="mt-2 p-2 rounded-md bg-destructive/10 border border-destructive/20">
            <div className="flex items-start gap-2">
              <AlertCircle className="w-4 h-4 text-destructive flex-shrink-0 mt-0.5" />
              <p className="text-xs text-destructive">{error}</p>
            </div>
          </div>
        )}
        {currentStep && currentStep.toLowerCase().includes('warning') && (
          <div className="mt-2 p-2 rounded-md bg-yellow-500/10 border border-yellow-500/20">
            <div className="flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 text-yellow-500 flex-shrink-0 mt-0.5" />
              <p className="text-xs text-yellow-600">{currentStep}</p>
            </div>
          </div>
        )}
      </div>

      {/* Status Icon */}
      <div className="flex-shrink-0">
        {getStatusIcon()}
      </div>

      {/* Action Button */}
      <div className="flex-shrink-0">
        {canCancel && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onCancel(id)}
            className="h-8 w-8 p-0 text-muted-foreground hover:text-destructive hover:bg-destructive/10"
          >
            <X className="w-4 h-4" />
          </Button>
        )}
        
        {canRemove && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onRemove(id)}
            className="h-8 w-8 p-0 text-muted-foreground hover:text-destructive hover:bg-destructive/10"
          >
            <Trash2 className="w-4 h-4" />
          </Button>
        )}
      </div>
    </div>
  );
};
