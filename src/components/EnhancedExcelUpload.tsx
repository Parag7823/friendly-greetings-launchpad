import { useState, useCallback, useEffect } from 'react';
import { Upload, CheckCircle, AlertCircle, Loader2, Settings, X, FileSpreadsheet, Zap } from 'lucide-react';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/hooks/use-toast';
import { FastAPIProcessor, useFastAPIProcessor } from './FastAPIProcessor';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { DuplicateDetectionModal } from './DuplicateDetectionModal';
interface FileUploadState {
  id: string;
  file: File;
  status: 'processing' | 'success' | 'error';
  progress: number;
  currentStep?: string;
  error?: string;
  analysisResults?: any;
  sheetProgress?: {
    currentSheet: string;
    sheetsCompleted: number;
    totalSheets: number;
  };
}
interface UploadState {
  files: FileUploadState[];
  uploadedFiles: {
    id: string;
    name: string;
    uploadedAt: Date;
    analysisResults?: any;
    sheets?: any[];
  }[];
  showConfig: boolean;
}
export const EnhancedExcelUpload = () => {
  const [uploadState, setUploadState] = useState<UploadState>({
    files: [],
    uploadedFiles: [],
    showConfig: false
  });
  const [apiKey, setApiKey] = useState<string>('');
  const [processingMode, setProcessingMode] = useState<'basic' | 'fastapi'>('fastapi');
  const [selectedFile, setSelectedFile] = useState<string | null>(null);

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

  const {
    toast
  } = useToast();
  const {
    processFileWithFastAPI: processWithFastAPI
  } = useFastAPIProcessor();

  // Load API key from localStorage on component mount
  useEffect(() => {
    const storedKey = localStorage.getItem('openai_api_key');
    if (storedKey) {
      setApiKey(storedKey);
    }
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
          user_id: 'current-user-id', // Replace with actual user ID
          decision: decision,
          file_hash: duplicateModal.currentFileHash
        })
      });

      if (response.ok) {
        setDuplicateModal(prev => ({ ...prev, isOpen: false }));

        if (decision === 'skip') {
          toast({
            title: "Upload Cancelled",
            description: "File upload was cancelled as requested",
          });
        } else {
          toast({
            title: "Decision Processed",
            description: `File will be processed with decision: ${decision}`,
          });
        }
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
    if (!duplicateModal.recommendation) {
      return;
    }

    try {
      const response = await fetch('/api/version-recommendation-feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          recommendation_id: duplicateModal.recommendation.id,
          user_id: 'current-user-id', // Replace with actual user ID
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
            : "Thank you for your feedback. It will help improve our recommendations.",
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
  const validateFile = (file: File): {
    isValid: boolean;
    error?: string;
  } => {
    const validTypes = ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/vnd.ms-excel', 'text/csv'];
    if (!validTypes.includes(file.type) && !file.name.match(/\.(xlsx|xls|csv)$/i)) {
      return {
        isValid: false,
        error: 'Please upload a valid Excel file (.xlsx, .xls) or CSV file.'
      };
    }
    const maxSize = 50 * 1024 * 1024; // Increased to 50MB for FastAPI
    if (file.size > maxSize) {
      return {
        isValid: false,
        error: 'File size must be less than 50MB.'
      };
    }
    return {
      isValid: true
    };
  };
const processFileEnhanced = async (file: File, fileId: string, customPrompt?: string) => {
  try {
    const result = await processWithFastAPI(file, customPrompt, progress => {
      setUploadState(prev => ({
        ...prev,
        files: prev.files.map(f => f.id === fileId ? {
          ...f,
          currentStep: progress.message,
          progress: progress.progress,
          sheetProgress: progress.sheetProgress
        } : f)
      }));
    });

    // Move file to uploaded files list and remove from processing
    setUploadState(prev => ({
      ...prev,
      files: prev.files.filter(f => f.id !== fileId),
      uploadedFiles: [...prev.uploadedFiles, {
        id: `${file.name}-${Date.now()}`,
        name: file.name,
        uploadedAt: new Date(),
        analysisResults: result,
        sheets: result.sheets || []
      }]
    }));
    return result;
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'FastAPI processing failed';
    setUploadState(prev => ({
      ...prev,
      files: prev.files.map(f => f.id === fileId ? {
        ...f,
        status: 'error' as const,
        error: errorMessage
      } : f)
    }));
    throw error;
  }
};
const handleMultipleFileUpload = useCallback(async (files: FileList, customPrompt?: string) => {
  const fileArray = Array.from(files).slice(0, 15); // Limit to 15 files max

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

  // Prepare file entries with stable IDs
  const now = Date.now();
  const fileEntries = fileArray.map((file, i) => ({
    id: `${now}-${i}-${file.name}`,
    file
  }));

  // Initialize file states
  const initialFileStates: FileUploadState[] = fileEntries.map(({ id, file }) => ({
    id,
    file,
    status: 'processing' as const,
    progress: 0,
    currentStep: 'Preparing for advanced analysis...'
  }));
  setUploadState(prev => ({
    ...prev,
    files: [...prev.files, ...initialFileStates]
  }));

  // Process files one by one for better progress tracking
  for (let i = 0; i < fileEntries.length; i++) {
    try {
      await processFileEnhanced(fileEntries[i].file, fileEntries[i].id, customPrompt);
    } catch (error) {
      console.error(`Error processing file ${fileEntries[i].file.name}:`, error);
    }
  }
}, [toast]);
  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      handleMultipleFileUpload(files);
    }
  };
  const handleDrop = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    const files = event.dataTransfer.files;
    if (files && files.length > 0) {
      handleMultipleFileUpload(files);
    }
  }, [handleMultipleFileUpload]);
  const handleDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
  }, []);
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'processing':
        return <Loader2 className="w-4 h-4 text-finley-accent animate-spin" />;
      case 'success':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'error':
        return <AlertCircle className="w-4 h-4 text-destructive" />;
      default:
        return <Loader2 className="w-4 h-4 text-finley-accent animate-spin" />;
    }
  };
  const selectedFileData = uploadState.uploadedFiles.find(f => f.id === selectedFile);
  return <div className="space-y-6">
      {/* Processing Mode Selector */}
      <Card>
        
        
      </Card>

      {/* File Upload Zone */}
      <Card>
        <CardContent className="p-6 flex justify-center">
          <input type="file" id="excel-upload" accept=".xlsx,.xls,.csv" multiple onChange={handleFileSelect} className="hidden" />
          
          <div className="cursor-pointer w-full mx-auto border border-border rounded-md p-8 transition-colors" onDrop={handleDrop} onDragOver={handleDragOver} onClick={() => document.getElementById('excel-upload')?.click()}>
            <div className="text-center">
              <div className="mb-4">
                <Upload className="w-8 h-8 text-finley-accent mx-auto mb-2" />
                <FileSpreadsheet className="w-6 h-6 text-finley-accent/60 mx-auto" />
              </div>
              <p className="text-lg font-medium text-foreground mb-2">Upload Financial Documents</p>
              <p className="text-sm text-muted-foreground mb-2">
                Drag & drop or click to browse (up to 15 files)
              </p>
              <p className="text-xs text-muted-foreground">.xlsx, .xls, .csv • Max 50MB per file </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* File Processing Progress */}
      {uploadState.files.length > 0 && <Card>
          <CardHeader>
            <CardTitle className="text-base">Processing Files</CardTitle>
            <CardDescription>Advanced AI analysis in progress</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {uploadState.files.map((fileState, index) => <div key={index} className="space-y-3 p-4 bg-muted/30 rounded-lg">
                <div className="flex items-center justify-between">
                <div className="flex items-center gap-3 min-w-0 flex-1">
                    {getStatusIcon(fileState.status)}
                    <div className="min-w-0">
                      <div className="font-medium text-sm truncate">{fileState.file.name}</div>
                      <div className="text-xs text-muted-foreground">
                        {Math.round(fileState.file.size / 1024 / 1024 * 100) / 100} MB
                      </div>
                    </div>

                  </div>
                  <div className="text-right">
                    <div className="text-sm font-medium">{fileState.progress}%</div>
                    {fileState.sheetProgress && <div className="text-xs text-muted-foreground">
                        {fileState.sheetProgress.sheetsCompleted}/{fileState.sheetProgress.totalSheets} sheets
                      </div>}
                  </div>
                </div>
                
                <Progress value={fileState.progress} className="h-2" />
                
                <div className="text-xs text-muted-foreground">
                  {fileState.currentStep}
                  {fileState.sheetProgress && <span className="ml-2 text-finley-accent">
                      Processing: {fileState.sheetProgress.currentSheet}
                    </span>}
                </div>
                
                {fileState.error && <div className="text-xs text-destructive bg-destructive/10 p-2 rounded">
                    {fileState.error}
                  </div>}
              </div>)}
          </CardContent>
        </Card>}

      {/* Results and Analysis Interface */}
      {uploadState.uploadedFiles.length > 0 && <Tabs defaultValue="files" className="space-y-4">
          <TabsList className="grid grid-cols-1 w-full">
            <TabsTrigger value="files">Uploaded Files ({uploadState.uploadedFiles.length})</TabsTrigger>
          </TabsList>

          <TabsContent value="files" className="space-y-4">
            <div className="relative">
              <div className="max-h-96 overflow-y-auto pr-2 space-y-3 scrollbar-thin scrollbar-thumb-finley-accent/20 scrollbar-track-transparent hover:scrollbar-thumb-finley-accent/30 scroll-smooth">
                {uploadState.uploadedFiles.map(file => <Card key={file.id} className={`cursor-pointer transition-colors ${selectedFile === file.id ? 'ring-2 ring-finley-accent bg-finley-accent/5' : 'hover:bg-muted/50'}`} onClick={() => setSelectedFile(selectedFile === file.id ? null : file.id)}>
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <FileSpreadsheet className="w-6 h-6 text-finley-accent" />
                          <div className="flex-1">
                            <div className="font-medium">{file.name}</div>
                            <div className="text-sm text-muted-foreground">
                              Uploaded {file.uploadedAt.toLocaleString()}
                              {file.sheets && ` • ${file.sheets.length} sheets detected`}
                            </div>
                            {/* Show processing results inline */}
                            {file.analysisResults && (
                              <div className="mt-2 space-y-1">
                                <div className="flex items-center gap-2 text-xs">
                                  <Badge variant="outline" className="text-xs">
                                    {file.analysisResults.documentType || 'Financial Data'}
                                  </Badge>
                                  {file.analysisResults.processing_stats && (
                                    <span className="text-muted-foreground">
                                      {file.analysisResults.processing_stats.events_created || 0} events processed
                                    </span>
                                  )}
                                </div>
                                {file.analysisResults.relationship_analysis && (
                                  <div className="text-xs text-muted-foreground">
                                    {file.analysisResults.relationship_analysis.total_relationships || 0} relationships found
                                  </div>
                                )}
                                {file.analysisResults.platform_details && (
                                  <div className="text-xs text-muted-foreground">
                                    Platform: {file.analysisResults.platform_details.name} 
                                    ({Math.round(file.analysisResults.platform_details.detection_confidence * 100)}% confidence)
                                  </div>
                                )}
                              </div>
                            )}
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <CheckCircle className="w-5 h-5 text-green-500" />
                        </div>
                      </div>
                    </CardContent>
                  </Card>)}
              </div>
              {/* Scroll indicator - only show if there are more than 4 files */}
              {uploadState.uploadedFiles.length > 4 && (
                <div className="absolute bottom-0 left-0 right-0 h-8 bg-gradient-to-t from-background to-transparent pointer-events-none" />
              )}
            </div>
          </TabsContent>

        </Tabs>}

      {/* Empty State */}
      {uploadState.files.length === 0 && uploadState.uploadedFiles.length === 0 && <Card>

        </Card>}

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
    </div>;
};