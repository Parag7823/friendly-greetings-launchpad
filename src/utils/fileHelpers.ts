import { File, FileSpreadsheet, FileText } from 'lucide-react';

/**
 * Get appropriate icon component for file type
 * Extracted from TabbedFilePreview to prevent duplication
 */
export const getFileIcon = (filename: string) => {
  if (!filename) return File;
  const ext = filename.split('.').pop()?.toLowerCase();
  if (ext === 'csv' || ext === 'xlsx' || ext === 'xls') return FileSpreadsheet;
  if (ext === 'txt' || ext === 'md') return FileText;
  return File;
};

/**
 * File validation utility
 * Extracted from EnhancedFileUpload to prevent duplication
 */
export const validateFile = (file: File): { isValid: boolean; error?: string } => {
  const validTypes = [
    // Spreadsheets
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/vnd.ms-excel',
    'text/csv',
    // PDFs
    'application/pdf',
    // Images
    'image/png',
    'image/jpeg',
    'image/jpg',
    'image/gif',
    'image/webp',
    'image/bmp',
    'image/tiff'
  ];
  
  const validExtensions = /\.(xlsx|xls|csv|pdf|png|jpg|jpeg|gif|webp|bmp|tiff|tif)$/i;
  
  if (!validTypes.includes(file.type) && !file.name.match(validExtensions)) {
    return {
      isValid: false,
      error: 'Please upload Excel (.xlsx, .xls), CSV, PDF, or image files (PNG, JPG, GIF, WebP, BMP, TIFF).'
    };
  }
  
  const maxSize = 500 * 1024 * 1024; // 500MB (matches backend limit)
  if (file.size > maxSize) {
    return {
      isValid: false,
      error: 'File size must be less than 500MB.'
    };
  }
  
  return { isValid: true };
};
