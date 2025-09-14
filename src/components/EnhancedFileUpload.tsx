import React, { useState, useCallback, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { UploadBox } from './UploadBox';
import { FileList, FileRowData } from './FileList';
import { useToast } from '@/hooks/use-toast';
import { useFastAPIProcessor } from './FastAPIProcessor';
import { DuplicateDetectionModal } from './DuplicateDetectionModal';
import { useAuth } from './AuthProvider';

interface UploadedFile {
  id: string;
  name: string;
  uploadedAt: Date;
  analysisResults?: any;
  sheets?: any[];
}

export const EnhancedFileUpload: React.FC = () => {
  const { user } = useAuth();
  const [files, setFiles] = useState<FileRowData[]>([]);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);

  // Duplicate detection state
  const [duplicateModal, setDuplicateModal] = useState({
    isOpen: false,
    phase: 'basic_duplicate' as 'basic_duplicate' | 'versions_detected' | 'similar_files',
    duplicateInfo: null as any,
    versionCandidates: null as any,
    recommendation: null as any,
    currentJobId: null as string | null,
    currentFileHash: null as string | null
  });

  const { toast } = useToast();
  const { processFileWithFastAPI } = useFastAPIProcessor();

  const validateFile = (file: File): { isValid: boolean; error?: string } => {
    const validTypes = [
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      'application/vnd.ms-excel',
      'text/csv'
    ];
    
    if (!validTypes.includes(file.type) && !file.name.match(/\.(xlsx|xls|csv)$/i)) {
      return {
        isValid: false,
        error: 'Please upload a valid Excel file (.xlsx, .xls) or CSV file.'
      };
    }
    
    const maxSize = 50 * 1024 * 1024; // 50MB
    if (file.size > maxSize) {
      return {
        isValid: false,
        error: 'File size must be less than 50MB.'
      };
    }
    
    return { isValid: true };
  };

  const processFile = async (file: File, fileId: string, customPrompt?: string) => {
    try {
      const result = await processFileWithFastAPI(file, customPrompt, (progress) => {
        // Handle duplicate detection progress
        if (progress.step === 'duplicate_detected') {
          // Show duplicate modal
          setDuplicateModal(prev => ({
            ...prev,
            isOpen: true,
            phase: 'basic_duplicate',
            duplicateInfo: {
              message: 'Duplicate file detected!',
              filename: file.name,
              recommendation: 'replace_or_skip'
            },
            currentJobId: null,
            currentFileHash: null
          }));
        }
        
        setFiles(prev => prev.map(f => 
          f.id === fileId 
            ? {
                ...f,
                currentStep: progress.message,
                progress: progress.progress,
                sheetProgress: progress.sheetProgress,
                status: progress.progress === 100 ? 'completed' : 'processing'
              }
            : f
        ));
      }, (jobId) => {
        // Store job ID for cancel functionality
        setFiles(prev => prev.map(f => 
          f.id === fileId 
            ? { ...f, jobId }
            : f
        ));
      });

      // Check if result indicates duplicate detection
      if (result.status === 'duplicate_detected') {
        // Show duplicate modal with actual duplicate information
        setDuplicateModal(prev => ({
          ...prev,
          isOpen: true,
          phase: 'basic_duplicate',
          duplicateInfo: {
            message: result.message || 'Duplicate file detected!',
            filename: file.name,
            recommendation: result.duplicate_analysis?.recommendation || 'replace_or_skip',
            duplicateFiles: result.duplicate_analysis?.duplicate_files || []
          },
          currentJobId: result.job_id,
          currentFileHash: null
        }));
        
        // Don't proceed with normal completion
        return result;
      }

      // Move to completed state only if not a duplicate
      setFiles(prev => prev.map(f => 
        f.id === fileId 
          ? { ...f, status: 'completed' as const, progress: 100 }
          : f
      ));

      // Move to uploaded files after a delay
      setTimeout(() => {
        setFiles(prev => prev.filter(f => f.id !== fileId));
        setUploadedFiles(prev => [...prev, {
          id: `${file.name}-${Date.now()}`,
          name: file.name,
          uploadedAt: new Date(),
          analysisResults: result,
          sheets: result.sheets || []
        }]);
      }, 2000);

      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Processing failed';
      setFiles(prev => prev.map(f => 
        f.id === fileId 
          ? { 
              ...f, 
              status: 'failed' as const, 
              error: errorMessage 
            }
          : f
      ));
      throw error;
    }
  };

  const handleFilesSelected = useCallback(async (fileList: FileList) => {
    const fileArray = Array.from(fileList).slice(0, 15); // Limit to 15 files

    // Validate all files first
    const validations = fileArray.map(file => ({
      file,
      validation: validateFile(file)
    }));

    const invalidFiles = validations.filter(v => !v.validation.isValid);
    if (invalidFiles.length > 0) {
      toast({
        variant: "destructive",
        title: "Upload Failed",
        description: `Invalid files: ${invalidFiles.map(f => f.file.name).join(', ')}`
      });
      return;
    }

    // Prepare file entries
    const now = Date.now();
    const fileEntries = fileArray.map((file, i) => ({
      id: `${now}-${i}-${file.name}`,
      file
    }));

    // Initialize file states
    const initialFileStates: FileRowData[] = fileEntries.map(({ id, file }) => ({
      id,
      file,
      status: 'queued' as const,
      progress: 0,
      currentStep: 'Queued for processing...'
    }));

    setFiles(prev => [...prev, ...initialFileStates]);
    setIsProcessing(true);

    // Process files sequentially for better control
    for (let i = 0; i < fileEntries.length; i++) {
      const { id, file } = fileEntries[i];
      
      // Update status to uploading
      setFiles(prev => prev.map(f => 
        f.id === id 
          ? { ...f, status: 'uploading' as const, currentStep: 'Uploading file...' }
          : f
      ));

      try {
        await processFile(file, id);
      } catch (error) {
        console.error(`Error processing file ${file.name}:`, error);
      }
    }

    setIsProcessing(false);
  }, [toast]);

  const handleCancel = useCallback(async (fileId: string) => {
    const fileData = files.find(f => f.id === fileId);
    if (!fileData || !fileData.jobId) return;

    try {
      const response = await fetch(`https://friendly-greetings-launchpad.onrender.com/cancel-upload/${fileData.jobId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      });

      if (response.ok) {
        setFiles(prev => prev.map(f => 
          f.id === fileId 
            ? { ...f, status: 'cancelled' as const, currentStep: 'Cancelled by user' }
            : f
        ));
        
        toast({
          title: "Upload Cancelled",
          description: "File upload has been cancelled"
        });
      } else {
        throw new Error('Failed to cancel upload');
      }
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Cancel Failed",
        description: "Could not cancel upload. Please try again."
      });
    }
  }, [files, toast]);

  const handleRemove = useCallback((fileId: string) => {
    setFiles(prev => prev.filter(f => f.id !== fileId));
    setUploadedFiles(prev => prev.filter(f => f.id !== fileId));
  }, []);

  // Duplicate detection handlers
  const handleDuplicateDecision = async (decision: 'replace' | 'keep_both' | 'skip') => {
    if (!duplicateModal.currentJobId || !duplicateModal.currentFileHash) {
      toast({
        title: "Error",
        description: "Missing job information for duplicate decision",
        variant: "destructive"
      });
      return;
    }

    try {
      const response = await fetch('/api/handle-duplicate-decision', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          job_id: duplicateModal.currentJobId,
          user_id: user?.id || 'anonymous',
          decision: decision,
          file_hash: duplicateModal.currentFileHash
        })
      });

      if (response.ok) {
        setDuplicateModal(prev => ({ ...prev, isOpen: false }));
        toast({
          title: "Decision Processed",
          description: `File will be processed with decision: ${decision}`,
        });
      } else {
        throw new Error('Failed to process duplicate decision');
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to process duplicate decision",
        variant: "destructive"
      });
    }
  };

  const handleVersionRecommendationFeedback = async (accepted: boolean, feedback?: string) => {
    if (!duplicateModal.recommendation) return;

    try {
      const response = await fetch('/api/version-recommendation-feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          recommendation_id: duplicateModal.recommendation.id,
          user_id: user?.id || 'anonymous',
          accepted: accepted,
          feedback: feedback
        })
      });

      if (response.ok) {
        setDuplicateModal(prev => ({ ...prev, isOpen: false }));
        toast({
          title: accepted ? "Recommendation Accepted" : "Feedback Submitted",
          description: accepted
            ? "Processing will continue with the recommended version"
            : "Thank you for your feedback.",
        });
      } else {
        throw new Error('Failed to submit feedback');
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to submit feedback",
        variant: "destructive"
      });
    }
  };

  return (
    <div className="space-y-6">
      {/* Compact Upload Section */}
      <Card>
        <CardHeader className="pb-4">
          <CardTitle className="text-lg">Upload Financial Documents</CardTitle>
          <CardDescription>
            Upload Excel or CSV files for AI-powered financial analysis
          </CardDescription>
        </CardHeader>
        <CardContent>
          <UploadBox
            onFilesSelected={handleFilesSelected}
            disabled={isProcessing}
            className="max-w-2xl mx-auto"
          />
        </CardContent>
      </Card>

      {/* File Processing List */}
      {files.length > 0 && (
        <Card>
          <CardContent className="p-6">
            <FileList
              files={files}
              onCancel={handleCancel}
              onRemove={handleRemove}
              maxHeight="max-h-96"
            />
          </CardContent>
        </Card>
      )}

      {/* Uploaded Files Summary */}
      {uploadedFiles.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Completed Files ({uploadedFiles.length})</CardTitle>
            <CardDescription>
              Successfully processed files ready for analysis
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-3">
              {uploadedFiles.map((file) => (
                <div
                  key={file.id}
                  className="flex items-center justify-between p-3 rounded-lg bg-muted/30 border"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-full bg-green-500/20 flex items-center justify-center">
                      <span className="text-green-500 text-sm font-medium">✓</span>
                    </div>
                    <div>
                      <p className="font-medium text-sm">{file.name}</p>
                      <p className="text-xs text-muted-foreground">
                        Completed {file.uploadedAt.toLocaleString()}
                        {file.sheets && ` • ${file.sheets.length} sheets`}
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={() => handleRemove(file.id)}
                    className="text-muted-foreground hover:text-destructive transition-colors"
                  >
                    Remove
                  </button>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Duplicate Detection Modal */}
      <DuplicateDetectionModal
        isOpen={duplicateModal.isOpen}
        onClose={() => setDuplicateModal(prev => ({ ...prev, isOpen: false }))}
        duplicateInfo={duplicateModal.duplicateInfo}
        versionCandidates={duplicateModal.versionCandidates}
        recommendation={duplicateModal.recommendation}
        onDecision={handleDuplicateDecision}
        onVersionAccept={handleVersionRecommendationFeedback}
        phase={duplicateModal.phase}
      />
    </div>
  );
};
