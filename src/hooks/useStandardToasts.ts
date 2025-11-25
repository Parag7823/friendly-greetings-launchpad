import { useToast } from '@/hooks/use-toast';

/**
 * Centralized toast patterns to prevent duplication
 * Extracted from multiple components using similar toast messages
 */
export const useStandardToasts = () => {
  const { toast } = useToast();

  return {
    uploadStarted: (fileCount: number) => {
      toast({
        title: 'Uploading Files',
        description: `Processing ${fileCount} file${fileCount > 1 ? 's' : ''}...`
      });
    },

    uploadSuccess: (successful: number, failed: number) => {
      toast({
        title: 'Upload Complete',
        description: `${successful} file(s) uploaded successfully${failed > 0 ? `, ${failed} failed` : ''}`,
        variant: failed > 0 ? 'destructive' : 'default'
      });
    },

    uploadFailed: (fileName: string, reason?: string) => {
      toast({
        variant: 'destructive',
        title: 'Upload Failed',
        description: reason || `Failed to upload ${fileName}`
      });
    },

    fileValidationFailed: (invalidFiles: string[]) => {
      toast({
        variant: 'destructive',
        title: 'Upload Failed',
        description: `Invalid files: ${invalidFiles.join(', ')}`
      });
    },

    processingComplete: (fileName: string) => {
      toast({
        title: 'Processing Complete',
        description: `${fileName} processed successfully`
      });
    },

    processingFailed: (fileName: string) => {
      toast({
        title: 'Processing Failed',
        description: `Unable to process ${fileName}`,
        variant: 'destructive'
      });
    },

    uploadCancelled: () => {
      toast({
        title: 'Upload Cancelled',
        description: 'File upload has been cancelled'
      });
    },

    cancelFailed: () => {
      toast({
        variant: 'destructive',
        title: 'Cancel Failed',
        description: 'Could not cancel upload. Please try again.'
      });
    },

    duplicateDetected: () => {
      toast({
        title: 'Duplicate Detected',
        description: 'This file has already been uploaded.'
      });
    },

    authenticationRequired: () => {
      toast({
        variant: 'destructive',
        title: 'Authentication Required',
        description: 'Unable to authenticate request. Please sign in again.'
      });
    },

    error: (title: string, description: string) => {
      toast({
        variant: 'destructive',
        title,
        description
      });
    },

    info: (title: string, description: string) => {
      toast({
        title,
        description
      });
    }
  };
};
