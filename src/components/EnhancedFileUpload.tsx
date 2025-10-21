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
  status: 'pending' | 'uploading' | 'processing' | 'completed' | 'error' | 'failed' | 'queued' | 'cancelled';
  progress: number;
  error?: string;
  currentStep?: string;
  sheetProgress?: any;
  jobId?: string;
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

  // FIX #5: Consolidated duplicate modal state for better maintainability
  interface DuplicateModalState {
    isOpen: boolean;
    phase: 'basic_duplicate' | 'versions_detected' | 'similar_files';
    context: {
      jobId: string | null;
      fileHash: string | null;
      fileId: string | null;
      existingFileId: string | null;
    };
    data: {
      duplicateInfo?: any;
      versionCandidates?: any;
      recommendation?: any;
      deltaAnalysis?: any;
    };
    error: string | null;
  }

  const [duplicateModal, setDuplicateModal] = useState<DuplicateModalState>({
    isOpen: false,
    phase: 'basic_duplicate',
    context: {
      jobId: null,
      fileHash: null,
      fileId: null,
      existingFileId: null,
    },
    data: {},
    error: null,
  });

  const { toast } = useToast();
  const { processFileWithFastAPI } = useFastAPIProcessor();

  // FIX #3: Expand file type validation to match backend capabilities
  const validateFile = (file: File): { isValid: boolean; error?: string } => {
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

          // FIX #4: Show consolidated duplicate modal
          setDuplicateModal(prev => ({
            ...prev,
            isOpen: true,
            phase: 'basic_duplicate',
            context: {
              ...prev.context,
              existingFileId: existingId,
            },
            data: {
              ...prev.data,
              duplicateInfo: {
                message: progress.message || 'Potential duplicate detected!',
                filename: file.name,
                recommendation: 'replace_or_skip',
                duplicate_files: normalizedFiles
              },
              deltaAnalysis: extra.delta_analysis || prev.data.deltaAnalysis
            }
          }));
        }
        
        // Update file state with progress and enrichment details
        const enrichmentDetails = (progress as any).enrichment_details;
        let detailedMessage = progress.message;
        
        // Add enrichment stats to message if available
        if (enrichmentDetails && progress.step === 'enrichment') {
          detailedMessage = `${progress.message}\n✓ ${enrichmentDetails.vendors_standardized || 0} vendors standardized\n✓ ${enrichmentDetails.platform_ids_extracted || 0} IDs extracted\n✓ ${enrichmentDetails.amounts_normalized || 0} amounts normalized`;
        }
        
        setFiles(prev => prev.map(f => 
          f.id === fileId 
            ? { 
                ...f, 
                status: progress.step === 'completed' ? 'completed' : 'processing',
                progress: progress.progress,
                currentStep: detailedMessage,
                sheetProgress: progress.sheetProgress
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

        // Pre-fill duplicate modal context
        setDuplicateModal(prev => ({
          ...prev,
          context: {
            ...prev.context,
            jobId,
            fileHash: fileHash || prev.context.fileHash
          }
        }));
      });

      // Check if result indicates any duplicate flow requiring user decision
      if (result.requires_user_decision) {
        // FIX #4: Show consolidated duplicate modal
        setDuplicateModal(prev => ({
          ...prev,
          isOpen: true,
          phase: 'basic_duplicate',
          context: {
            jobId: result.job_id || null,
            fileHash: result.file_hash || null,
            fileId: fileId,
            existingFileId: (result as any).existing_file_id || null,
          },
          data: {
            duplicateInfo: {
              message: result.message || 'Duplicate or similar file detected!',
              filename: file.name,
              recommendation: result.duplicate_analysis?.recommendation || 'replace_or_skip',
              duplicateFiles: result.duplicate_analysis?.duplicate_files || []
            },
            deltaAnalysis: (result as any).delta_analysis || null
          }
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

  const handleFilesSelected = useCallback(async (fileList: FileList | File[]) => {
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

    // CONSUMER EXPERIENCE: Process up to 3 files concurrently for faster bulk uploads
    const MAX_CONCURRENT = 3;
    const processQueue = async () => {
      const queue = [...fileEntries];
      const processing = new Map<string, Promise<void>>();
      
      while (queue.length > 0 || processing.size > 0) {
        // Start new files while under concurrent limit
        while (processing.size < MAX_CONCURRENT && queue.length > 0) {
          const entry = queue.shift()!;
          const { id, file } = entry;
          
          // Update status to uploading
          setFiles(prev => prev.map(f => 
            f.id === id 
              ? { ...f, status: 'uploading' as const, currentStep: 'Uploading file...' }
              : f
          ));
          
          // Start processing with proper cleanup
          const processPromise = processFile(file, id)
            .then(() => {}) // Convert to Promise<void> for Map type consistency
            .catch(error => {
              console.error(`Error processing file ${file.name}:`, error);
            })
            .finally(() => {
              // Remove from processing map when done
              processing.delete(id);
            });
          
          processing.set(id, processPromise);
        }
        
        // Wait for at least one to complete
        if (processing.size > 0) {
          await Promise.race(Array.from(processing.values()));
        }
      }
    };
    
    await processQueue();
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
    if (!duplicateModal.context.jobId || !duplicateModal.context.fileHash) {
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
          job_id: duplicateModal.context.jobId,
          user_id: user?.id || 'anonymous',
          decision: decision,
          file_hash: duplicateModal.context.fileHash,
          session_token: session?.access_token,
          ...(decision === 'delta_merge' && duplicateModal.context.existingFileId
            ? { existing_file_id: duplicateModal.context.existingFileId }
            : {})
        })
      });
      if (!response.ok) throw new Error('Failed to process duplicate decision');

      // Close modal
      setDuplicateModal(prev => ({ ...prev, isOpen: false }));

      // If skip, mark UI as cancelled and remove from list
      if (decision === 'skip') {
        const fileRowId = duplicateModal.context.fileId as string | null;
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
      const jobId = duplicateModal.context.jobId as string;
      const fileRowId = duplicateModal.context.fileId as string | null;

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
      const jobId = duplicateModal.context.jobId;
      const fileRowId = duplicateModal.context.fileId as string | null;
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
    if (!duplicateModal.data.recommendation) return;

    try {
      const { data: { session } } = await supabase.auth.getSession();
      const response = await fetch(`${config.apiUrl}/version-recommendation-feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          recommendation_id: duplicateModal.data.recommendation.id,
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
            className={`bg-[#0A0A0A] border-2 border-dashed border-white/10 rounded-lg p-8 text-center hover:border-white/20 transition-colors ${isProcessing ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
            onClick={() => !isProcessing && document.getElementById('file-upload')?.click()}
          >
            <Upload className="mx-auto h-12 w-12 text-white/70 mb-4" />
            <p className="text-sm text-white mb-2">
              Click to upload or drag and drop
            </p>
            <p className="text-xs text-white/60">
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
                <div key={fileData.id} className="flex items-center justify-between p-3 bg-[#0A0A0A] border border-white/10 rounded-lg">
                  <div className="flex items-center gap-3 flex-1">
                    <FileText className="h-5 w-5 text-white/70" />
                    <div className="flex-1">
                      <p className="text-sm font-medium text-white">{fileData.file.name}</p>
                      <p className="text-xs text-white/60">{fileData.status}</p>
                    </div>
                  </div>
                  {fileData.status === 'completed' && <CheckCircle className="h-5 w-5 text-green-500" />}
                  {fileData.status === 'error' && <XCircle className="h-5 w-5 text-red-500" />}
                  {fileData.status === 'processing' && <Loader2 className="h-5 w-5 animate-spin text-white" />}
                  <button
                    onClick={() => handleRemove(fileData.id)}
                    className="ml-2 p-1 hover:bg-white/10 rounded text-white/70"
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
                  className="flex items-center justify-between p-3 rounded-lg bg-[#0A0A0A] border border-white/10"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-full bg-green-500/20 flex items-center justify-center">
                      <span className="text-green-500 text-sm font-medium">✓</span>
                    </div>
                    <div>
                      <p className="font-medium text-sm text-white">{file.name}</p>
                      <p className="text-xs text-white/60">
                        Completed {file.uploadedAt.toLocaleString()}
                        {file.sheets && ` • ${file.sheets.length} sheets`}
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={() => handleRemove(file.id)}
                    className="text-white/60 hover:text-red-500 transition-colors"
                  >
                    Remove
                  </button>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* FIX #4: Duplicate Detection Modal with consolidated state */}
      <DuplicateDetectionModal
        isOpen={duplicateModal.isOpen}
        onClose={handleModalCancel}
        duplicateInfo={duplicateModal.data.duplicateInfo}
        versionCandidates={duplicateModal.data.versionCandidates}
        recommendation={duplicateModal.data.recommendation}
        onDecision={handleDuplicateDecision}
        onVersionAccept={handleVersionRecommendationFeedback}
        phase={duplicateModal.phase}
        deltaAnalysis={duplicateModal.data.deltaAnalysis}
        error={duplicateModal.error}
      />
    </div>
  );
};
