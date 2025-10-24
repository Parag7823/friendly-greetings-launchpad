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

interface EnhancedFileUploadProps {
  initialFiles?: File[];
  onUploadComplete?: () => void;
}

export const EnhancedFileUpload: React.FC<EnhancedFileUploadProps> = ({ initialFiles, onUploadComplete }) => {
  const { user } = useAuth();
  const [files, setFiles] = useState<FileRowData[]>([]);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);

  // FIX #5: Consolidated duplicate modal state for better maintainability
  interface DuplicateModalState {
    isOpen: boolean;
    phase: 'basic_duplicate'; // Only phase used - versions_detected was never implemented
    context: {
      jobId: string | null;
      fileHash: string | null;
      fileId: string | null;
      existingFileId: string | null;
    };
    data: {
      duplicateInfo?: any;
      deltaAnalysis?: any;
    };
    error: string | null;
  }

  // FIX #5: Use Map to track duplicate modals per file to prevent state corruption
  const [duplicateModals, setDuplicateModals] = useState<Map<string, DuplicateModalState>>(new Map());
  
  // Helper to get current active modal (first open one)
  const getActiveModal = (): [string | null, DuplicateModalState | null] => {
    for (const [fileId, modal] of duplicateModals.entries()) {
      if (modal.isOpen) {
        return [fileId, modal];
      }
    }
    return [null, null];
  };
  
  // FIX ISSUE #6: Use useMemo to ensure activeModal updates when duplicateModals changes
  // This prevents stale closure and ensures modal shows correct file information
  const [activeModalFileId, activeModal] = React.useMemo(() => getActiveModal(), [duplicateModals]);

  const { toast } = useToast();
  const { processFileWithFastAPI } = useFastAPIProcessor();

  // Auto-process initialFiles when provided
  React.useEffect(() => {
    if (initialFiles && initialFiles.length > 0) {
      handleFilesSelected(initialFiles);
    }
  }, [initialFiles]);

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

          // FIX #5: Store duplicate modal per file to prevent corruption
          setDuplicateModals(prev => {
            const newMap = new Map(prev);
            const existingModal = newMap.get(fileId) || {
              isOpen: false,
              phase: 'basic_duplicate',
              context: { jobId: null, fileHash: null, fileId: null, existingFileId: null },
              data: {},
              error: null
            };
            
            newMap.set(fileId, {
              ...existingModal,
              isOpen: true,
              phase: 'basic_duplicate',
              context: {
                ...existingModal.context,
                fileId: fileId,
                existingFileId: existingId,
              },
              data: {
                ...existingModal.data,
                duplicateInfo: {
                  message: progress.message || 'Potential duplicate detected!',
                  filename: file.name,
                  recommendation: 'replace_or_skip',
                  duplicate_files: normalizedFiles,
                  // CRITICAL FIX: Pass duplicate_type and similarity_score from backend
                  duplicate_type: extra.duplicate_info?.duplicate_type || extra.near_duplicate_info?.duplicate_type || 'exact',
                  similarity_score: extra.duplicate_info?.similarity_score || extra.near_duplicate_info?.similarity_score || 1.0
                },
                deltaAnalysis: extra.delta_analysis ? { delta_analysis: extra.delta_analysis } : existingModal.data.deltaAnalysis
              }
            });
            
            return newMap;
          });
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

        // FIX #5: Pre-fill duplicate modal context in Map
        setDuplicateModals(prev => {
          const newMap = new Map(prev);
          const existing = newMap.get(fileId) || {
            isOpen: false,
            phase: 'basic_duplicate',
            context: { jobId: null, fileHash: null, fileId: null, existingFileId: null },
            data: {},
            error: null
          };
          newMap.set(fileId, {
            ...existing,
            context: {
              ...existing.context,
              jobId,
              fileHash: fileHash || existing.context.fileHash
            }
          });
          return newMap;
        });
      });

      // Check if result indicates any duplicate flow requiring user decision
      if (result.requires_user_decision) {
        // FIX #5: Show consolidated duplicate modal in Map
        setDuplicateModals(prev => {
          const newMap = new Map(prev);
          newMap.set(fileId, {
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
                duplicate_files: result.duplicate_analysis?.duplicate_files || [],
                // CRITICAL FIX: Pass duplicate_type and similarity_score from backend response
                duplicate_type: result.duplicate_analysis?.duplicate_type || (result as any).duplicate_type || 'exact',
                similarity_score: result.duplicate_analysis?.similarity_score || (result as any).similarity_score || 1.0
              },
              deltaAnalysis: (result as any).delta_analysis ? { delta_analysis: (result as any).delta_analysis } : null
            },
            error: null
          });
          return newMap;
        });
        
        // Don't proceed with normal completion
        return result;
      }

      // Move to completed state only if not a duplicate
      setFiles(prev => prev.map(f => 
        f.id === fileId 
          ? { ...f, status: 'completed' as const, progress: 100, currentStep: '✅ Processing complete!' }
          : f
      ));

      // CRITICAL FIX: Add to uploaded files immediately (not after delay)
      // This ensures files show up in "Uploaded Files" section right away
      const uploadedFile = {
        id: result.file_id || `${file.name}-${Date.now()}`,
        name: file.name,
        uploadedAt: new Date(),
        analysisResults: result,
        sheets: result.sheets || []
      };
      
      setUploadedFiles(prev => {
        // Prevent duplicates
        const exists = prev.some(f => f.id === uploadedFile.id);
        if (exists) return prev;
        return [...prev, uploadedFile];
      });

      // Remove from processing list after a short delay (for visual feedback)
      setTimeout(() => {
        setFiles(prev => prev.filter(f => f.id !== fileId));
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

  // FIX #1: In-memory lock for duplicate checking to prevent race conditions
  const duplicateCheckLock = React.useRef<Set<string>>(new Set());

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

    // FIX #1: CRITICAL - Sequential duplicate checking with in-memory lock
    // Process files concurrently BUT with sequential duplicate checking
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
          
          // FIX #1: Wrap processing with duplicate check lock
          const processPromise = (async () => {
            // Calculate file hash BEFORE acquiring lock
            const fileBuffer = await file.arrayBuffer();
            const hashBuffer = await crypto.subtle.digest('SHA-256', fileBuffer);
            const hashArray = Array.from(new Uint8Array(hashBuffer));
            const fileHash = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
            
            // FIX #1: CRITICAL - Acquire lock for this file hash
            // Wait if another file with same hash is being checked
            while (duplicateCheckLock.current.has(fileHash)) {
              await new Promise(resolve => setTimeout(resolve, 100));
            }
            
            // Acquire lock
            duplicateCheckLock.current.add(fileHash);
            
            try {
              // Now process file with lock held
              await processFile(file, id);
            } finally {
              // Always release lock
              duplicateCheckLock.current.delete(fileHash);
            }
          })();
          
          const wrappedPromise = processPromise
            .then(() => {}) // Convert to Promise<void> for Map type consistency
            .catch(error => {
              console.error(`Error processing file ${file.name}:`, error);
            })
            .finally(() => {
              // Remove from processing map when done
              processing.delete(id);
            });
          
          processing.set(id, wrappedPromise);
        }
        
        // Wait for at least one to complete
        if (processing.size > 0) {
          await Promise.race(Array.from(processing.values()));
        }
      }
    };
    
    await processQueue();
    setIsProcessing(false);
    
    // Call completion callback if provided
    if (onUploadComplete) {
      onUploadComplete();
    }
  }, [toast, onUploadComplete]);

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

  // FIX #5: Duplicate detection handlers using Map-based state
  const handleDuplicateDecision = async (decision: 'replace' | 'keep_both' | 'skip' | 'delta_merge') => {
    if (!activeModal || !activeModalFileId) {
      toast({
        title: "Error",
        description: "No active duplicate modal",
        variant: "destructive"
      });
      return;
    }
    
    if (!activeModal.context.jobId || !activeModal.context.fileHash) {
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
          ...(session?.access_token && { 'Authorization': `Bearer ${session.access_token}` })
        },
        body: JSON.stringify({
          job_id: activeModal.context.jobId,
          user_id: user?.id || 'anonymous',
          decision: decision,
          file_hash: activeModal.context.fileHash,
          session_token: session?.access_token,
          ...(decision === 'delta_merge' && activeModal.context.existingFileId
            ? { existing_file_id: activeModal.context.existingFileId }
            : {})
        })
      });
      if (!response.ok) throw new Error('Failed to process duplicate decision');

      // FIX #5: Close modal in Map
      setDuplicateModals(prev => {
        const newMap = new Map(prev);
        newMap.delete(activeModalFileId);
        return newMap;
      });

      // If skip, mark UI as cancelled and remove from list
      if (decision === 'skip') {
        const fileRowId = activeModal.context.fileId as string | null;
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
      const jobId = activeModal.context.jobId as string;
      const fileRowId = activeModal.context.fileId as string | null;

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
      
      // FIX #5: Show error in modal Map
      setDuplicateModals(prev => {
        const newMap = new Map(prev);
        const existing = newMap.get(activeModalFileId);
        if (existing) {
          newMap.set(activeModalFileId, { ...existing, error: errorMessage });
        }
        return newMap;
      });
      
      toast({
        title: "Error",
        description: errorMessage,
        variant: "destructive"
      });
    }
  };

  // FIX #5: Expose a helper to cancel from modal's footer button
  const handleModalCancel = useCallback(async () => {
    if (!activeModal || !activeModalFileId) return;
    
    try {
      const jobId = activeModal.context.jobId;
      const fileRowId = activeModal.context.fileId as string | null;
      if (!jobId) {
        setDuplicateModals(prev => {
          const newMap = new Map(prev);
          newMap.delete(activeModalFileId);
          return newMap;
        });
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
      setDuplicateModals(prev => {
        const newMap = new Map(prev);
        newMap.delete(activeModalFileId);
        return newMap;
      });
      if (!res.ok) throw new Error('Failed to cancel upload');
      toast({ title: 'Upload Cancelled', description: 'File upload has been cancelled' });
    } catch (e) {
      toast({ variant: 'destructive', title: 'Cancel Failed', description: 'Could not cancel upload. Please try again.' });
    }
  }, [activeModal, activeModalFileId, duplicateModals, toast]);

  // REMOVED: handleVersionRecommendationFeedback() function
  // This was dead code for the deprecated version_recommendations system
  // Backend endpoint /version-recommendation-feedback was removed in migration 20251013000000

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
            className={`bg-[#1a1a1a] border-2 border-dashed border-white/10 rounded-lg p-8 text-center hover:border-white/20 transition-colors ${isProcessing ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
            onClick={() => !isProcessing && document.getElementById('file-upload')?.click()}
          >
            <Upload className="mx-auto h-12 w-12 text-white/70 mb-4" />
            <p className="text-sm text-white mb-2">
              Click to upload or drag and drop
            </p>
            <p className="text-xs text-white/60">
              Excel, CSV, PDF, or image files up to 500MB
            </p>
            <input
              id="file-upload"
              type="file"
              multiple
              accept=".xlsx,.xls,.csv,.pdf,.png,.jpg,.jpeg,.gif,.webp,.bmp,.tiff,.tif"
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
                <div key={fileData.id} className="flex items-center justify-between p-3 bg-[#1a1a1a] border border-white/10 rounded-lg">
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
                  className="flex items-center justify-between p-3 rounded-lg bg-[#1a1a1a] border border-white/10"
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

      {/* FIX #5: Duplicate Detection Modal using Map-based state */}
      {activeModal && (
        <DuplicateDetectionModal
          isOpen={activeModal.isOpen}
          onClose={handleModalCancel}
          duplicateInfo={activeModal.data.duplicateInfo}
          onDecision={handleDuplicateDecision}
          phase={activeModal.phase}
          deltaAnalysis={activeModal.data.deltaAnalysis}
          error={activeModal.error}
        />
      )}
    </div>
  );
};
