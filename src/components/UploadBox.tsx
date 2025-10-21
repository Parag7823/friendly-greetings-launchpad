import React, { useCallback, useState } from 'react';
import { Upload, FileSpreadsheet, Plus } from 'lucide-react';
import { cn } from '@/lib/utils';

interface UploadBoxProps {
  onFilesSelected: (files: FileList) => void;
  disabled?: boolean;
  className?: string;
}

export const UploadBox: React.FC<UploadBoxProps> = ({
  onFilesSelected,
  disabled = false,
  className
}) => {
  const [isDragOver, setIsDragOver] = useState(false);

  const handleFileSelect = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      onFilesSelected(files);
    }
  }, [onFilesSelected]);

  const handleDrop = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragOver(false);
    
    if (disabled) return;
    
    const files = event.dataTransfer.files;
    if (files && files.length > 0) {
      onFilesSelected(files);
    }
  }, [onFilesSelected, disabled]);

  const handleDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    if (!disabled) {
      setIsDragOver(true);
    }
  }, [disabled]);

  const handleDragLeave = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragOver(false);
  }, []);

  const handleClick = useCallback(() => {
    if (!disabled) {
      document.getElementById('file-upload-input')?.click();
    }
  }, [disabled]);

  return (
    <div
      className={cn(
        "relative w-full",
        className
      )}
    >
      <input
        id="file-upload-input"
        type="file"
        accept=".xlsx,.xls,.csv,.pdf,.png,.jpg,.jpeg,.gif,.webp,.bmp,.tiff,.tif"
        multiple
        onChange={handleFileSelect}
        className="hidden"
        disabled={disabled}
      />
      
      <div
        className={cn(
          "group relative cursor-pointer rounded-lg border-2 border-dashed transition-all duration-200",
          "hover:border-finley-accent/50 hover:bg-finley-accent/5",
          "focus-within:ring-2 focus-within:ring-finley-accent/20 focus-within:border-finley-accent",
          isDragOver && "border-finley-accent bg-finley-accent/10 scale-[1.02]",
          disabled && "opacity-50 cursor-not-allowed",
          "bg-muted/20 border-muted-foreground/30"
        )}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={handleClick}
      >
        <div className="flex flex-col items-center justify-center p-6 text-center">
          <div className="relative mb-3">
            <div className={cn(
              "w-12 h-12 rounded-full flex items-center justify-center transition-all duration-200",
              "bg-finley-accent/10 group-hover:bg-finley-accent/20",
              isDragOver && "bg-finley-accent/30 scale-110"
            )}>
              <Upload className={cn(
                "w-6 h-6 transition-colors duration-200",
                "text-finley-accent group-hover:text-finley-accent/80"
              )} />
            </div>
            <div className={cn(
              "absolute -top-1 -right-1 w-6 h-6 rounded-full flex items-center justify-center",
              "bg-background border-2 border-finley-accent/30",
              "opacity-0 group-hover:opacity-100 transition-opacity duration-200"
            )}>
              <Plus className="w-3 h-3 text-finley-accent" />
            </div>
          </div>
          
          <div className="space-y-1">
            <p className="text-sm font-medium text-foreground">
              {isDragOver ? "Drop files here" : "Upload Financial Documents"}
            </p>
            <p className="text-xs text-muted-foreground">
              Drag & drop or click to browse
            </p>
            <div className="flex items-center justify-center gap-2 text-xs text-muted-foreground">
              <FileSpreadsheet className="w-3 h-3" />
              <span>Excel, CSV, PDF, Images • Max 500MB each</span>
            </div>
          </div>
        </div>
        
        {/* Subtle background pattern */}
        <div className="absolute inset-0 rounded-lg bg-gradient-to-br from-finley-accent/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none" />
      </div>
    </div>
  );
};
