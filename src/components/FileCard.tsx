import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Eye, Trash2, Loader2, CheckCircle2, AlertCircle, Cloud, Upload, Building2 } from 'lucide-react';
import { Badge } from './ui/badge';
import { Button } from './ui/button';

/**
 * FileCard Component
 * 
 * Displays a single financial document with:
 * - Source badge (QuickBooks, Dropbox, Manual Upload, etc.)
 * - Processing status indicator
 * - Hover state with action buttons
 * - Real-time progress tracking
 */

interface FileCardProps {
  file: {
    id: string;
    filename: string;
    status: 'processing' | 'pending' | 'completed' | 'failed';
    created_at: string;
    progress?: number;
    error_message?: string;
    source?: 'quickbooks' | 'dropbox' | 'google-drive' | 'manual' | string;
  };
  onPreview: (fileId: string, filename: string, file: any) => void;
  onDelete: (fileId: string, filename: string) => void;
  onViewProgress: (fileId: string) => void;
}

const sourceIcons: Record<string, React.ReactNode> = {
  'quickbooks': <Building2 className="w-3 h-3" />,
  'dropbox': <Cloud className="w-3 h-3" />,
  'google-drive': <Cloud className="w-3 h-3" />,
  'manual': <Upload className="w-3 h-3" />,
};

const sourceLabels: Record<string, string> = {
  'quickbooks': 'QuickBooks',
  'dropbox': 'Dropbox',
  'google-drive': 'Google Drive',
  'manual': 'Manual Upload',
};

export const FileCard: React.FC<FileCardProps> = ({
  file,
  onPreview,
  onDelete,
  onViewProgress,
}) => {
  const [isHovered, setIsHovered] = useState(false);

  const isProcessing = file.status === 'processing' || file.status === 'pending';
  const isCompleted = file.status === 'completed';
  const isFailed = file.status === 'failed';
  const progress = file.progress || 0;

  const source = file.source || 'manual';
  const sourceIcon = sourceIcons[source] || sourceIcons['manual'];
  const sourceLabel = sourceLabels[source] || source;

  return (
    <motion.div
      className="relative p-3 rounded-md border finley-dynamic-bg hover:bg-muted/20 transition-colors group cursor-pointer"
      onHoverStart={() => setIsHovered(true)}
      onHoverEnd={() => setIsHovered(false)}
      whileHover={{ y: -2 }}
      onClick={() => onPreview(file.id, file.filename || file.id, file)}
    >
      {/* Main Content */}
      <div className="flex items-center justify-between gap-3">
        {/* Left: File Info */}
        <div className="flex-1 min-w-0 space-y-1">
          {/* Filename + Status Icon */}
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

          {/* Date + Progress */}
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

          {/* Progress Bar */}
          {isProcessing && (
            <div className="w-full h-1 bg-muted rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-primary"
                initial={{ width: 0 }}
                animate={{ width: `${progress}%` }}
                transition={{ duration: 0.3 }}
              />
            </div>
          )}

          {/* Error Message */}
          {isFailed && file.error_message && (
            <p className="text-[8px] text-destructive/70 truncate" title={file.error_message}>
              {file.error_message}
            </p>
          )}
        </div>

        {/* Right: Badges + Actions */}
        <div className="flex items-center gap-2 flex-shrink-0">
          {/* Source Badge */}
          <Badge variant="outline" className="gap-1 text-[9px] whitespace-nowrap">
            {sourceIcon}
            <span>{sourceLabel}</span>
          </Badge>

          {/* Status Badge */}
          {isProcessing && (
            <Badge variant="secondary" className="text-[10px]">
              Processing...
            </Badge>
          )}
          {isCompleted && (
            <Badge variant="default" className="text-[10px] bg-green-500/10 text-green-600 border-green-500/20">
              ✓ Done
            </Badge>
          )}
          {isFailed && (
            <Badge variant="destructive" className="text-[10px]">
              Failed
            </Badge>
          )}
        </div>
      </div>

      {/* Hover Actions */}
      <motion.div
        className="absolute top-2 right-2 flex items-center gap-1"
        initial={{ opacity: 0, x: 10 }}
        animate={isHovered ? { opacity: 1, x: 0 } : { opacity: 0, x: 10 }}
        transition={{ duration: 0.2 }}
      >
        {/* View Progress Button */}
        <Button
          size="sm"
          variant="ghost"
          className="h-7 px-2 text-xs"
          onClick={(e) => {
            e.stopPropagation();
            onViewProgress(file.id);
          }}
          title="View detailed progress"
        >
          <Eye className="w-3.5 h-3.5 mr-1" />
          Progress
        </Button>

        {/* Delete Button */}
        <Button
          size="sm"
          variant="ghost"
          className="h-7 px-2"
          onClick={(e) => {
            e.stopPropagation();
            onDelete(file.id, file.filename || file.id);
          }}
          title="Delete file"
        >
          <Trash2 className="w-3.5 h-3.5 text-muted-foreground hover:text-destructive transition-colors" />
        </Button>
      </motion.div>
    </motion.div>
  );
};
