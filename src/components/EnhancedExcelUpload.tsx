import { useState, useCallback, useEffect } from 'react';
import { Upload, CheckCircle, AlertCircle, Loader2, Settings, X, FileSpreadsheet, Zap } from 'lucide-react';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/hooks/use-toast';
import { FastAPIProcessor, useFastAPIProcessor } from './FastAPIProcessor';
import { SheetPreview } from './SheetPreview';
import { CustomPromptInterface } from './CustomPromptInterface';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';

interface FileUploadState {
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
  const { toast } = useToast();
  const { processFileWithFastAPI: processWithFastAPI } = useFastAPIProcessor();

  // Load API key from localStorage on component mount
  useEffect(() => {
    const storedKey = localStorage.getItem('openai_api_key');
    if (storedKey) {
      setApiKey(storedKey);
    }
  }, []);

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

    const maxSize = 50 * 1024 * 1024; // Increased to 50MB for FastAPI
    if (file.size > maxSize) {
      return {
        isValid: false,
        error: 'File size must be less than 50MB.'
      };
    }

    return { isValid: true };
  };

  const processFileEnhanced = async (file: File, fileIndex: number, customPrompt?: string) => {
    try {
      const result = await processWithFastAPI(
        file,
        customPrompt,
        (progress) => {
          setUploadState(prev => ({
            ...prev,
            files: prev.files.map((f, i) => 
              i === fileIndex 
                ? { 
                    ...f, 
                    currentStep: progress.message, 
                    progress: progress.progress,
                    sheetProgress: progress.sheetProgress
                  }
                : f
            )
          }));
        }
      );

      // Move file to uploaded files list and remove from processing
      setUploadState(prev => ({
        ...prev,
        files: prev.files.filter((_, i) => i !== fileIndex),
        uploadedFiles: [
          ...prev.uploadedFiles,
          {
            id: `${file.name}-${Date.now()}`,
            name: file.name,
            uploadedAt: new Date(),
            analysisResults: result,
            sheets: result.sheets || []
          }
        ]
      }));

      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'FastAPI processing failed';
      
      setUploadState(prev => ({
        ...prev,
        files: prev.files.map((f, i) => 
          i === fileIndex 
            ? { ...f, status: 'error' as const, error: errorMessage }
            : f
        )
      }));
      
      throw error;
    }
  };

  const handleMultipleFileUpload = useCallback(async (files: FileList, customPrompt?: string) => {
    const fileArray = Array.from(files).slice(0, 5); // Limit to 5 files for FastAPI
    
    // Validate all files first
    const validations = fileArray.map(file => ({ file, validation: validateFile(file) }));
    const invalidFiles = validations.filter(v => !v.validation.isValid);
    
    if (invalidFiles.length > 0) {
      toast({
        variant: "destructive",
        title: "Upload Failed",
        description: `Invalid files: ${invalidFiles.map(f => f.file.name).join(', ')}`
      });
      return;
    }

    // Initialize file states
    const baseIndex = uploadState.files.length;
    const initialFileStates: FileUploadState[] = fileArray.map(file => ({
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
    for (let i = 0; i < fileArray.length; i++) {
      try {
        await processFileEnhanced(fileArray[i], baseIndex + i, customPrompt);
      } catch (error) {
        console.error(`Error processing file ${fileArray[i].name}:`, error);
      }
    }
  }, [uploadState.files.length, toast]);

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

  const handleCustomPromptSubmit = (prompt: string) => {
    if (uploadState.files.length > 0) {
      toast({
        title: "Custom Analysis Started",
        description: `Reprocessing ${uploadState.files.length} file(s) with your custom prompt`
      });
      // Re-process current files with custom prompt
      const fileList = new DataTransfer();
      uploadState.files.forEach(f => fileList.items.add(f.file));
      handleMultipleFileUpload(fileList.files, prompt);
    } else {
      toast({
        variant: "destructive",
        title: "No Files to Analyze",
        description: "Please upload files first, then apply custom prompts"
      });
    }
  };

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

  return (
    <div className="space-y-6">
      {/* Processing Mode Selector */}
      <Card>
        <CardHeader className="pb-4">
          <CardTitle className="flex items-center gap-2">
            <Zap className="w-5 h-5 text-finley-accent" />
            Advanced AI Processing
          </CardTitle>
          <CardDescription>
            FastAPI integration provides sophisticated sheet-by-sheet analysis, financial statement recognition, and custom prompt processing
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4">
            <Badge variant="default" className="bg-finley-accent">
              <Zap className="w-3 h-3 mr-1" />
              FastAPI Enhanced
            </Badge>
            <div className="text-sm text-muted-foreground">
              • Advanced sheet recognition • Custom prompts • Real-time progress • 50MB limit
            </div>
          </div>
        </CardContent>
      </Card>

      {/* File Upload Zone */}
      <Card>
        <CardContent className="p-6">
          <input
            type="file"
            id="excel-upload"
            accept=".xlsx,.xls,.csv"
            multiple
            onChange={handleFileSelect}
            className="hidden"
          />
          
          <div
            className="upload-zone cursor-pointer border-2 border-dashed border-finley-accent/30 hover:border-finley-accent/50 rounded-lg p-8 transition-colors"
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onClick={() => document.getElementById('excel-upload')?.click()}
          >
            <div className="text-center">
              <div className="mb-4">
                <Upload className="w-8 h-8 text-finley-accent mx-auto mb-2" />
                <FileSpreadsheet className="w-6 h-6 text-finley-accent/60 mx-auto" />
              </div>
              <p className="text-lg font-medium text-foreground mb-2">Upload Financial Documents</p>
              <p className="text-sm text-muted-foreground mb-2">
                Drag & drop or click to browse (up to 5 files)
              </p>
              <p className="text-xs text-muted-foreground">
                .xlsx, .xls, .csv • Max 50MB per file • FastAPI enhanced analysis
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* File Processing Progress */}
      {uploadState.files.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Processing Files</CardTitle>
            <CardDescription>Advanced AI analysis in progress</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {uploadState.files.map((fileState, index) => (
              <div key={index} className="space-y-3 p-4 bg-muted/30 rounded-lg">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    {getStatusIcon(fileState.status)}
                    <div>
                      <div className="font-medium text-sm">{fileState.file.name}</div>
                      <div className="text-xs text-muted-foreground">
                        {Math.round(fileState.file.size / 1024 / 1024 * 100) / 100} MB
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-medium">{fileState.progress}%</div>
                    {fileState.sheetProgress && (
                      <div className="text-xs text-muted-foreground">
                        {fileState.sheetProgress.sheetsCompleted}/{fileState.sheetProgress.totalSheets} sheets
                      </div>
                    )}
                  </div>
                </div>
                
                <Progress value={fileState.progress} className="h-2" />
                
                <div className="text-xs text-muted-foreground">
                  {fileState.currentStep}
                  {fileState.sheetProgress && (
                    <span className="ml-2 text-finley-accent">
                      Processing: {fileState.sheetProgress.currentSheet}
                    </span>
                  )}
                </div>
                
                {fileState.error && (
                  <div className="text-xs text-destructive bg-destructive/10 p-2 rounded">
                    {fileState.error}
                  </div>
                )}
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {/* Results and Analysis Interface */}
      {uploadState.uploadedFiles.length > 0 && (
        <Tabs defaultValue="files" className="space-y-4">
          <TabsList className="grid grid-cols-3 w-full">
            <TabsTrigger value="files">Uploaded Files ({uploadState.uploadedFiles.length})</TabsTrigger>
            <TabsTrigger value="analysis">Sheet Analysis</TabsTrigger>
            <TabsTrigger value="prompts">Custom Prompts</TabsTrigger>
          </TabsList>

          <TabsContent value="files" className="space-y-4">
            <div className="grid gap-4">
              {uploadState.uploadedFiles.map((file) => (
                <Card 
                  key={file.id} 
                  className={`cursor-pointer transition-colors ${
                    selectedFile === file.id ? 'ring-2 ring-finley-accent bg-finley-accent/5' : 'hover:bg-muted/50'
                  }`}
                  onClick={() => setSelectedFile(selectedFile === file.id ? null : file.id)}
                >
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <FileSpreadsheet className="w-6 h-6 text-finley-accent" />
                        <div>
                          <div className="font-medium">{file.name}</div>
                          <div className="text-sm text-muted-foreground">
                            Uploaded {file.uploadedAt.toLocaleString()}
                            {file.sheets && ` • ${file.sheets.length} sheets detected`}
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        {file.analysisResults && (
                          <Badge variant="secondary">
                            {file.analysisResults.documentType || 'Analyzed'}
                          </Badge>
                        )}
                        <CheckCircle className="w-5 h-5 text-green-500" />
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="analysis">
            {selectedFileData?.sheets ? (
              <SheetPreview 
                sheets={selectedFileData.sheets}
                onSheetSelect={(sheetName) => {
                  toast({
                    title: "Sheet Selected",
                    description: `Focus analysis on ${sheetName}`
                  });
                }}
                onPromptSuggestion={(prompt) => {
                  toast({
                    title: "Prompt Applied",
                    description: prompt
                  });
                }}
              />
            ) : (
              <Card>
                <CardContent className="p-8 text-center">
                  <FileSpreadsheet className="w-12 h-12 mx-auto mb-4 text-muted-foreground/50" />
                  <p className="text-muted-foreground">
                    Select a file from the "Uploaded Files" tab to view detailed sheet analysis
                  </p>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="prompts">
            <CustomPromptInterface
              onSubmit={handleCustomPromptSubmit}
              isProcessing={uploadState.files.length > 0}
              documentType={selectedFileData?.analysisResults?.documentType}
              suggestedPrompts={selectedFileData?.analysisResults?.customPromptSuggestions || []}
              detectedSheets={selectedFileData?.sheets?.map(s => s.name) || []}
            />
          </TabsContent>
        </Tabs>
      )}

      {/* Empty State */}
      {uploadState.files.length === 0 && uploadState.uploadedFiles.length === 0 && (
        <Card>
          <CardContent className="p-8 text-center">
            <Upload className="w-16 h-16 mx-auto mb-4 text-muted-foreground/50" />
            <h3 className="text-lg font-medium mb-2">Ready for Financial Analysis</h3>
            <p className="text-muted-foreground mb-4">
              Upload your Excel files to begin advanced AI-powered financial analysis with FastAPI
            </p>
            <div className="flex justify-center gap-4 text-sm text-muted-foreground">
              <div className="flex items-center gap-1">
                <CheckCircle className="w-4 h-4 text-green-500" />
                Sheet Recognition
              </div>
              <div className="flex items-center gap-1">
                <CheckCircle className="w-4 h-4 text-green-500" />
                Custom Prompts
              </div>
              <div className="flex items-center gap-1">
                <CheckCircle className="w-4 h-4 text-green-500" />
                Real-time Progress
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};