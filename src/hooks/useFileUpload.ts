import { useState } from 'react';
import { useToast } from '@/components/ui/use-toast';
import { useFastAPIProcessor } from '@/components/FastAPIProcessor';

interface UploadProgress {
  fileId: string;
  fileName: string;
  progress: number;
  status: 'pending' | 'uploading' | 'processing' | 'completed' | 'error';
  error?: string;
}

/**
 * Shared hook for file upload logic
 * Prevents code duplication between ChatInterface and DataSourcesPanel
 */
export const useFileUpload = () => {
  const { toast } = useToast();
  const { processFileWithFastAPI } = useFastAPIProcessor();
  const [uploadProgress, setUploadProgress] = useState<Map<string, UploadProgress>>(new Map());
  const [isUploading, setIsUploading] = useState(false);

  const uploadFiles = async (
    files: File[],
    onComplete?: (results: { successful: number; failed: number }) => void,
    onFileComplete?: (file: File, success: boolean) => void
  ) => {
    if (files.length === 0) return;

    setIsUploading(true);

    // Initialize progress for all files
    const progressMap = new Map<string, UploadProgress>();
    files.forEach((file, index) => {
      const fileId = `${Date.now()}-${index}-${file.name}`;
      progressMap.set(fileId, {
        fileId,
        fileName: file.name,
        progress: 0,
        status: 'pending'
      });
    });
    setUploadProgress(progressMap);

    toast({
      title: 'Uploading Files',
      description: `Processing ${files.length} file(s)...`
    });

    // Process all files in parallel
    const uploadPromises = Array.from(progressMap.entries()).map(async ([fileId, fileProgress]) => {
      const file = files.find(f => f.name === fileProgress.fileName);
      if (!file) return { status: 'failed', file: fileProgress.fileName };

      try {
        // Update status to uploading
        setUploadProgress(prev => {
          const updated = new Map(prev);
          updated.set(fileId, { ...fileProgress, status: 'uploading' });
          return updated;
        });

        await processFileWithFastAPI(
          file,
          undefined,
          (progress) => {
            // Update progress
            setUploadProgress(prev => {
              const updated = new Map(prev);
              updated.set(fileId, {
                ...fileProgress,
                progress: progress.progress,
                status: progress.status === 'completed' ? 'completed' : 'processing'
              });
              return updated;
            });
          }
        );

        // Mark as completed
        setUploadProgress(prev => {
          const updated = new Map(prev);
          updated.set(fileId, { ...fileProgress, status: 'completed', progress: 100 });
          return updated;
        });

        onFileComplete?.(file, true);
        return { status: 'success', file: file.name };
      } catch (error) {
        console.error('File upload failed:', error);
        
        // Mark as error
        setUploadProgress(prev => {
          const updated = new Map(prev);
          updated.set(fileId, {
            ...fileProgress,
            status: 'error',
            error: error instanceof Error ? error.message : 'Upload failed'
          });
          return updated;
        });

        onFileComplete?.(file, false);
        return { status: 'failed', file: file.name, error };
      }
    });

    // Wait for all uploads
    const results = await Promise.allSettled(uploadPromises);
    
    // Count successes and failures
    const successful = results.filter(r => r.status === 'fulfilled' && (r.value as any).status === 'success').length;
    const failed = results.filter(r => r.status === 'rejected' || (r.status === 'fulfilled' && (r.value as any).status === 'failed')).length;

    // Show final toast
    toast({
      title: 'Upload Complete',
      description: `${successful} file(s) uploaded successfully${failed > 0 ? `, ${failed} failed` : ''}`,
      variant: failed > 0 ? 'destructive' : 'default'
    });

    setIsUploading(false);
    onComplete?.({ successful, failed });

    // Clear progress after 2 seconds
    setTimeout(() => {
      setUploadProgress(new Map());
    }, 2000);
  };

  return {
    uploadFiles,
    uploadProgress,
    isUploading
  };
};
