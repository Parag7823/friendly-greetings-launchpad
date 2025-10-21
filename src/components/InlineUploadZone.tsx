import { Upload, FileSpreadsheet } from 'lucide-react';
import { useState, useCallback } from 'react';
import { Button } from './ui/button';
import { Card } from './ui/card';

interface InlineUploadZoneProps {
  onFilesSelected: (files: File[]) => void;
  disabled?: boolean;
}

export const InlineUploadZone = ({ onFilesSelected, disabled = false }: InlineUploadZoneProps) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) {
      setIsDragging(true);
    }
  }, [disabled]);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (disabled) return;

    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      onFilesSelected(files);
    }
  }, [disabled, onFilesSelected]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) {
      onFilesSelected(files);
    }
    // Reset input
    e.target.value = '';
  }, [onFilesSelected]);

  return (
    <Card
      className={`
        border-2 border-dashed transition-all duration-200
        ${isDragging ? 'border-primary bg-primary/5' : 'border-border bg-card'}
        ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer hover:border-primary/50'}
      `}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <div className="p-6">
        <div className="flex flex-col items-center justify-center space-y-3 text-center">
          <div className={`
            w-12 h-12 rounded-full flex items-center justify-center
            ${isDragging ? 'bg-primary/20' : 'bg-muted'}
          `}>
            <Upload className={`w-6 h-6 ${isDragging ? 'text-primary' : 'text-muted-foreground'}`} />
          </div>
          
          <div>
            <h3 className="text-sm font-medium text-foreground mb-1">
              Upload Financial Documents
            </h3>
            <p className="text-xs text-muted-foreground">
              Drag & drop files here, or click to browse
            </p>
          </div>

          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <FileSpreadsheet className="w-4 h-4" />
            <span>Excel, CSV, PDF supported • Max 500MB</span>
          </div>

          <input
            type="file"
            id="inline-file-input"
            className="hidden"
            multiple
            accept=".xlsx,.xls,.csv,.pdf,.png,.jpg,.jpeg,.gif,.webp,.bmp,.tiff,.tif"
            onChange={handleFileInput}
            disabled={disabled}
          />

          <Button
            variant="outline"
            size="sm"
            onClick={() => document.getElementById('inline-file-input')?.click()}
            disabled={disabled}
          >
            <Upload className="w-4 h-4 mr-2" />
            Browse Files
          </Button>
        </div>
      </div>
    </Card>
  );
};
