import React, { useState, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Upload, FileText, CheckCircle, XCircle, Loader2, AlertCircle, X } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { supabase } from '@/integrations/supabase/client';
import { useFastAPIProcessor } from './FastAPIProcessor';
import { DuplicateDetectionModal } from './DuplicateDetectionModal';
import { config } from '@/config';
import { useAuth } from './AuthProvider';

interface FileRowData {
  id: string;
  file: File;
  status: 'pending' | 'uploading' | 'processing' | 'completed' | 'error';
  progress: number;
  error?: string;
}

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
    duplicateInfo: undefined as any,
    versionCandidates: undefined as any,
    recommendation: undefined as any,
    phase: 'basic_duplicate' as 'basic_duplicate' | 'versions_detected' | 'similar_files',
    currentJobId: null as string | null,
    currentFileHash: null as string | null,
    currentFileId: null as string | null,
    currentExistingFileId: null as string | null,
    deltaAnalysis: undefined as any,
    error: null as string | null
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
    
    const maxSize = 500 * 1024 * 1024; // 500MB (matches backend limit)
    if (file.size > maxSize) {
      return {
        isValid: false,
        error: 'File size must be less than 500MB.'
      };
    }
    
    return { isValid: true };
  };

  const processFile = async (file: File, fileId: string, customPrompt?: string) => {
    try {
      const result = await processFileWithFastAPI(file, customPrompt, (progress) => {
        // Handle duplicate detection progress
        if ([
          'duplicate_detected',
          'duplicate_found',
          'near_duplicate_found',
          'content_duplicate_found',
          'delta_analysis_complete'
        ].includes(progress.step)) {
          // Extract duplicate info from WebSocket extras for accurate modal context
          const extra: any = (progress as any).extra || {};
          let duplicateFiles: any[] = [];
          let existingId: string | null = null;
          if (extra.duplicate_info?.duplicate_files?.length) {
            duplicateFiles = extra.duplicate_info.duplicate_files;
            existingId = duplicateFiles[0]?.id || null;
          } else if (extra.near_duplicate_info?.duplicate_files?.length) {
            duplicateFiles = extra.near_duplicate_info.duplicate_files;
            existingId = duplicateFiles[0]?.id || null;
          } else if (extra.content_duplicate_info?.overlapping_files?.length) {
            duplicateFiles = extra.content_duplicate_info.overlapping_files;
            existingId = duplicateFiles[0]?.id || null;
          }

          // Normalize items for UI expectations
          const normalizedFiles = (duplicateFiles || []).map((f: any) => ({
            id: f.id,
            filename: f.filename || f.file_name || 'Unknown',
            uploaded_at: f.uploaded_at || new Date().toISOString(),
            status: f.status || 'unknown',
            total_rows: f.total_rows || 0
          }));

          // Show duplicate modal populated from WS context
          setDuplicateModal(prev => ({
            ...prev,
            isOpen: true,
            phase: 'basic_duplicate',
            duplicateInfo: {
              message: progress.message || 'Potential duplicate detected!',
              filename: file.name,
              recommendation: 'replace_or_skip',
              duplicate_files: normalizedFiles
            },
            // currentJobId and currentFileHash are set in onJobId earlier
            currentExistingFileId: existingId,
            deltaAnalysis: extra.delta_analysis || prev.deltaAnalysis
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
      }, (jobId, fileHash) => {
        // Store job ID for cancel functionality
        setFiles(prev => prev.map(f => 
          f.id === fileId 
            ? { ...f, jobId }
            : f
        ));

        // Pre-fill duplicate modal context so decisions can be sent instantly when modal opens via WebSocket
        setDuplicateModal(prev => ({
          ...prev,
          currentJobId: jobId,
          currentFileHash: fileHash || prev.currentFileHash
        }));
      });

      // Check if result indicates any duplicate flow requiring user decision
      if (result.requires_user_decision) {
        // Show duplicate modal with actual duplicate information
        setDuplicateModal(prev => ({
          ...prev,
          isOpen: true,
          phase: 'basic_duplicate',
          duplicateInfo: {
            message: result.message || 'Duplicate or similar file detected!',
            filename: file.name,
            recommendation: result.duplicate_analysis?.recommendation || 'replace_or_skip',
            duplicateFiles: result.duplicate_analysis?.duplicate_files || []
          },
          currentJobId: result.job_id,
          currentFileHash: result.file_hash || null,
          currentStoragePath: result.storage_path || null,
          currentFileName: result.file_name || file.name,
          currentFileId: fileId,
          currentExistingFileId: (result as any).existing_file_id || null,
          deltaAnalysis: (result as any).delta_analysis || null
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
      const response = await fetch(`${config.apiUrl}/cancel-upload/${fileData.jobId}`, {
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
  const handleDuplicateDecision = async (decision: 'replace' | 'keep_both' | 'skip' | 'delta_merge') => {
    if (!duplicateModal.currentJobId || !duplicateModal.currentFileHash) {
      toast({
        title: "Error",
        description: "Missing job information for duplicate decision",
        variant: "destructive"
      });
      return;
    }

    try {
      // Fetch session token for backend security validation
      const { data: { session } } = await supabase.auth.getSession();

      // Always notify backend of decision (skip, replace, keep_both)
      const response = await fetch(`${config.apiUrl}/handle-duplicate-decision`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          job_id: duplicateModal.currentJobId,
          user_id: user?.id || 'anonymous',
          decision: decision,
          file_hash: duplicateModal.currentFileHash,
          session_token: session?.access_token,
          ...(decision === 'delta_merge' && duplicateModal.currentExistingFileId
            ? { existing_file_id: duplicateModal.currentExistingFileId }
            : {})
        })
      });
      if (!response.ok) throw new Error('Failed to process duplicate decision');

      // Close modal
      setDuplicateModal(prev => ({ ...prev, isOpen: false }));

      // If skip, mark UI as cancelled and remove from list
      if (decision === 'skip') {
        const fileRowId = duplicateModal.currentFileId as string | null;
        if (fileRowId) {
          setFiles(prev => prev.map(f => 
            f.id === fileRowId 
              ? { ...f, status: 'cancelled' as const, currentStep: 'Skipped due to duplicate', progress: 100 }
              : f
          ));
          // Remove after a short delay for UX
          setTimeout(() => {
            setFiles(prev => prev.filter(f => f.id !== fileRowId));
          }, 1200);
        }
        toast({ title: 'Upload Skipped', description: 'Duplicate file was skipped.' });
        return;
      }

      // For replace/keep_both, poll job status as backend resumes processing
      const jobId = duplicateModal.currentJobId as string;
      const fileRowId = duplicateModal.currentFileId as string | null;

      if (fileRowId) {
        // Optimistic UI update
        setFiles(prev => prev.map(f => 
          f.id === fileRowId 
            ? { ...f, status: 'processing' as const, currentStep: 'Resuming after duplicate decision...', progress: Math.max(25, f.progress || 0) }
            : f
        ));

        // Poll loop
        const maxAttempts = 120;
        let attempts = 0;
        while (attempts < maxAttempts) {
          try {
            const res = await fetch(`${config.apiUrl}/job-status/${jobId}`);
            if (res.ok) {
              const data = await res.json();
              const status = data.status;
              const progress = data.progress ?? 0;
              const message = data.message || 'Processing...';
              setFiles(prev => prev.map(f => 
                f.id === fileRowId 
                  ? { ...f, currentStep: message, progress: progress, status: status === 'completed' ? 'completed' as const : status === 'failed' ? 'failed' as const : status === 'cancelled' ? 'cancelled' as const : 'processing' as const }
                  : f
              ));
              if (status === 'completed') {
                toast({ title: 'Processing Complete', description: 'File processed successfully after duplicate resolution.' });
                break;
              }
              if (status === 'failed') {
                toast({ variant: 'destructive', title: 'Processing Failed', description: data.error || 'Unknown error' });
                break;
              }
              if (status === 'cancelled') break;
            }
          } catch {
            // ignore and retry
          }
          await new Promise(r => setTimeout(r, 1500));
          attempts++;
        }
        if (attempts >= maxAttempts) {
          toast({ variant: 'destructive', title: 'Timeout', description: 'Processing timed out after resume.' });
        }
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Failed to process duplicate decision";
      
      // Show error in modal
      setDuplicateModal(prev => ({ 
        ...prev, 
        error: errorMessage 
      }));
      
      toast({
        title: "Error",
        description: errorMessage,
        variant: "destructive"
      });
    }
  };

  // Expose a helper to cancel from modal's footer button
  const handleModalCancel = useCallback(async () => {
    try {
      const jobId = duplicateModal.currentJobId;
      const fileRowId = duplicateModal.currentFileId as string | null;
      if (!jobId) {
        setDuplicateModal(prev => ({ ...prev, isOpen: false }));
        return;
      }
      const res = await fetch(`${config.apiUrl}/cancel-upload/${jobId}`, { method: 'POST', headers: { 'Content-Type': 'application/json' } });
      // Optimistic UI regardless
      if (fileRowId) {
        setFiles(prev => prev.map(f => 
          f.id === fileRowId 
            ? { ...f, status: 'cancelled' as const, currentStep: 'Cancelled by user', progress: 100 }
            : f
        ));
        setTimeout(() => setFiles(prev => prev.filter(f => f.id !== fileRowId)), 1200);
      }
      setDuplicateModal(prev => ({ ...prev, isOpen: false }));
      if (!res.ok) throw new Error('Failed to cancel upload');
      toast({ title: 'Upload Cancelled', description: 'File upload has been cancelled' });
    } catch (e) {
      toast({ variant: 'destructive', title: 'Cancel Failed', description: 'Could not cancel upload. Please try again.' });
    }
  }, [duplicateModal, toast]);

  const handleVersionRecommendationFeedback = async (accepted: boolean, feedback?: string) => {
    if (!duplicateModal.recommendation) return;

    try {
      const { data: { session } } = await supabase.auth.getSession();
      const response = await fetch(`${config.apiUrl}/version-recommendation-feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          recommendation_id: duplicateModal.recommendation.id,
          user_id: user?.id || 'anonymous',
          accepted: accepted,
          feedback: feedback,
          session_token: session?.access_token
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
          <div 
            className={`border-2 border-dashed rounded-lg p-8 text-center hover:border-primary transition-colors ${isProcessing ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
            onClick={() => !isProcessing && document.getElementById('file-upload')?.click()}
          >
            <Upload className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
            <p className="text-sm text-muted-foreground mb-2">
              Click to upload or drag and drop
            </p>
            <p className="text-xs text-muted-foreground">
              Excel (.xlsx, .xls) or CSV files up to 500MB
            </p>
            <input
              id="file-upload"
              type="file"
              multiple
              accept=".xlsx,.xls,.csv"
              onChange={(e) => e.target.files && handleFilesSelected(Array.from(e.target.files))}
              className="hidden"
              disabled={isProcessing}
            />
          </div>
        </CardContent>
      </Card>

      {/* File Processing List */}
      {files.length > 0 && (
        <Card>
          <CardContent className="p-6">
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {files.map((fileData) => (
                <div key={fileData.id} className="flex items-center justify-between p-3 bg-muted rounded-lg">
                  <div className="flex items-center gap-3 flex-1">
                    <FileText className="h-5 w-5" />
                    <div className="flex-1">
                      <p className="text-sm font-medium">{fileData.file.name}</p>
                      <p className="text-xs text-muted-foreground">{fileData.status}</p>
                    </div>
                  </div>
                  {fileData.status === 'completed' && <CheckCircle className="h-5 w-5 text-green-500" />}
                  {fileData.status === 'error' && <XCircle className="h-5 w-5 text-red-500" />}
                  {fileData.status === 'processing' && <Loader2 className="h-5 w-5 animate-spin" />}
                  <button
                    onClick={() => handleRemove(fileData.id)}
                    className="ml-2 p-1 hover:bg-background rounded"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
              ))}
            </div>
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
        onClose={handleModalCancel}
        duplicateInfo={duplicateModal.duplicateInfo}
        versionCandidates={duplicateModal.versionCandidates}
        recommendation={duplicateModal.recommendation}
        onDecision={handleDuplicateDecision}
        onVersionAccept={handleVersionRecommendationFeedback}
        phase={duplicateModal.phase}
        deltaAnalysis={duplicateModal.deltaAnalysis}
        error={duplicateModal.error}
      />
    </div>
  );
};
