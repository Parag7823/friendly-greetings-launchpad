import React from 'react';
import { FileRow, FileRowData } from './FileRow';
import { cn } from '@/lib/utils';

interface FileListProps {
  files: FileRowData[];
  onCancel: (fileId: string) => void;
  onRemove: (fileId: string) => void;
  className?: string;
  maxHeight?: string;
}

export const FileList: React.FC<FileListProps> = ({
  files,
  onCancel,
  onRemove,
  className,
  maxHeight = "max-h-96"
}) => {
  if (files.length === 0) {
    return null;
  }

  return (
    <div className={cn("space-y-3", className)}>
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-foreground">
          Processing Files ({files.length})
        </h3>
        <div className="text-sm text-muted-foreground">
          {files.filter(f => f.status === 'completed').length} completed
        </div>
      </div>
      
      <div className={cn(
        "overflow-y-auto space-y-2 pr-2",
        "scrollbar-thin scrollbar-thumb-finley-accent/20 scrollbar-track-transparent",
        "hover:scrollbar-thumb-finley-accent/30 scroll-smooth",
        maxHeight
      )}>
        {files.map((fileData) => (
          <FileRow
            key={fileData.id}
            fileData={fileData}
            onCancel={onCancel}
            onRemove={onRemove}
          />
        ))}
      </div>
      
      {/* Scroll indicator */}
      {files.length > 4 && (
        <div className="absolute bottom-0 left-0 right-0 h-8 bg-gradient-to-t from-background to-transparent pointer-events-none" />
      )}
    </div>
  );
};
