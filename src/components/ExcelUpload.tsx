import { useState, useCallback, useEffect } from 'react';
import { Upload, CheckCircle, AlertCircle, Loader2, Settings, X } from 'lucide-react';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/hooks/use-toast';
import { ExcelProcessor, ProcessingProgress, ProcessingResult } from '@/lib/excelProcessor';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';

interface FileUploadState {
  file: File;
  status: 'processing' | 'success' | 'error';
  progress: number;
  currentStep?: string;
  error?: string;
  analysisResults?: ProcessingResult;
}

interface UploadState {
  files: FileUploadState[];
  uploadedFiles: { id: string; name: string; uploadedAt: Date; analysisResults?: ProcessingResult }[];
  showConfig: boolean;
}

export const ExcelUpload = () => {
  const [uploadState, setUploadState] = useState<UploadState>({
    files: [],
    uploadedFiles: [],
    showConfig: false
  });
  const [apiKey, setApiKey] = useState<string>('');
  const { toast } = useToast();

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

    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
      return {
        isValid: false,
        error: 'File size must be less than 10MB.'
      };
    }

    return { isValid: true };
  };

  const saveJobToDatabase = async (file: File, result: ProcessingResult): Promise<void> => {
    try {
      console.log('🔍 Starting saveJobToDatabase...');
      const { data: { user } } = await supabase.auth.getUser();
      
      console.log('👤 User authentication status:', { 
        isAuthenticated: !!user, 
        userId: user?.id 
      });
      
      if (!user) {
        console.error('❌ User not authenticated - cannot save to database');
        return;
      }

      // First, save raw record with file content
      console.log('💾 Attempting to save raw record...');
      const rawRecordData = {
        user_id: user.id,
        source: 'excel_upload',
        file_name: file.name,
        file_size: file.size,
        content: {
          document_type: result.documentType,
          insights: result.insights,
          metrics: result.metrics,
          summary: result.summary,
          file_metadata: {
            name: file.name,
            size: file.size,
            type: file.type,
            last_modified: file.lastModified
          }
        },
        status: 'completed',
        classification_status: 'completed'
      };
      
      console.log('📄 Raw record data:', rawRecordData);
      
      const { data: rawRecord, error: rawError } = await supabase
        .from('raw_records')
        .insert(rawRecordData)
        .select()
        .single();

      if (rawError) {
        console.error('❌ Failed to save raw record:', rawError);
        console.error('Error details:', JSON.stringify(rawError, null, 2));
        throw rawError;
      }
      
      console.log('✅ Raw record saved successfully:', rawRecord);

      // Then save job with reference to raw record
      console.log('📊 Attempting to save ingestion job...');
      const { error: jobError } = await supabase
        .from('ingestion_jobs')
        .insert({
          job_type: 'excel_analysis',
          user_id: user.id,
          record_id: rawRecord.id,
          status: 'completed',
          progress: 100,
          result: {
            file_name: file.name,
            file_size: file.size,
            document_type: result.documentType,
            insights: result.insights,
            metrics: result.metrics,
            summary: result.summary,
            processed_at: new Date().toISOString()
          }
        });

      if (jobError) {
        console.error('⚠️ Failed to save job to database:', jobError);
      }
    } catch (error) {
      console.warn('Database save error:', error);
    }
  };

  const processFile = async (file: File, fileIndex: number) => {
    try {
      const storedApiKey = localStorage.getItem('openai_api_key') || apiKey;
      const processor = new ExcelProcessor(storedApiKey);
      
      processor.setProgressCallback((progress: ProcessingProgress) => {
        setUploadState(prev => ({
          ...prev,
          files: prev.files.map((f, i) => 
            i === fileIndex 
              ? { ...f, currentStep: progress.message, progress: progress.progress }
              : f
          )
        }));
      });

      const result = await processor.processFile(file);
      
      // Save to database in background
      await saveJobToDatabase(file, result);

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
            analysisResults: result
          }
        ]
      }));

      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      
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

  const handleMultipleFileUpload = useCallback(async (files: FileList) => {
    const fileArray = Array.from(files).slice(0, 10); // Limit to 10 files
    
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

    // Debug API key storage
    const storedApiKey = localStorage.getItem('openai_api_key');
    const currentApiKey = apiKey;
    
    console.log('Debug API Key Info:', {
      storedApiKey: storedApiKey ? 'Present' : 'Not found',
      currentApiKey: currentApiKey ? 'Present' : 'Not found',
      finalKey: storedApiKey || currentApiKey
    });
    
    // Check if API key is available
    const finalApiKey = storedApiKey || currentApiKey;
    if (!finalApiKey) {
      setUploadState(prev => ({ ...prev, showConfig: true }));
      toast({
        variant: "destructive",
        title: "API Key Required",
        description: "Please configure your OpenAI API key for enhanced analysis."
      });
      return;
    }

    // Initialize file states
    const baseIndex = uploadState.files.length;
    const initialFileStates: FileUploadState[] = fileArray.map(file => ({
      file,
      status: 'processing' as const,
      progress: 0,
      currentStep: 'Initializing...'
    }));

    setUploadState(prev => ({
      ...prev,
      files: [...prev.files, ...initialFileStates]
    }));

    // Process all files in parallel
    const promises = fileArray.map((file, index) => 
      processFile(file, baseIndex + index)
    );

    try {
      const results = await Promise.allSettled(promises);
      
      // Show success message for completed files
      const successCount = results.filter(r => r.status === 'fulfilled').length;
      
      if (successCount > 0) {
        toast({
          title: "Analysis Complete",
          description: `Successfully processed ${successCount} file(s). Data stored in database.`
        });
      }

      if (successCount < fileArray.length) {
        toast({
          variant: "destructive",
          title: "Some Files Failed",
          description: `${fileArray.length - successCount} file(s) failed to process.`
        });
      }
    } catch (error) {
      console.error('Error processing files:', error);
    }
  }, [uploadState.files.length, toast, apiKey]);

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

  const removeFile = (fileIndex: number) => {
    setUploadState(prev => ({
      ...prev,
      files: prev.files.filter((_, i) => i !== fileIndex)
    }));
  };

  const resetUpload = () => {
    setUploadState({
      files: [],
      uploadedFiles: [],
      showConfig: false
    });
  };

  const deleteUploadedFile = async (fileId: string) => {
    try {
      const file = uploadState.uploadedFiles.find(f => f.id === fileId);
      if (!file) return;

      // Delete from database
      const { data: { user } } = await supabase.auth.getUser();
      
      const { error } = await supabase
        .from('ingestion_jobs')
        .delete()
        .eq('user_id', user?.id)
        .eq('job_type', 'excel_analysis')
        .like('result->file_name', `%${file.name}%`);

      if (error) {
        console.warn('Failed to delete from database:', error);
      }

      // Remove from state
      setUploadState(prev => ({
        ...prev,
        uploadedFiles: prev.uploadedFiles.filter(f => f.id !== fileId)
      }));

      toast({
        title: "File deleted successfully",
        description: `${file.name} has been removed.`
      });
    } catch (error) {
      console.error('Error deleting file:', error);
      toast({
        variant: "destructive",
        title: "Delete Failed",
        description: "Failed to delete the file. Please try again."
      });
    }
  };

  const viewStoredData = async () => {
    try {
      const { data: { user } } = await supabase.auth.getUser();
      
      const { data, error } = await supabase
        .from('ingestion_jobs')
        .select('*')
        .eq('user_id', user?.id)
        .eq('job_type', 'excel_analysis')
        .order('created_at', { ascending: false });

      if (error) {
        toast({
          variant: "destructive",
          title: "Database Error",
          description: "Failed to fetch stored data."
        });
        return;
      }

      toast({
        title: "Database Query Complete",
        description: `Found ${data?.length || 0} stored analysis records. Check console for details.`
      });
      
      console.log('Stored analysis data:', data);
    } catch (error) {
      console.error('Error fetching data:', error);
      toast({
        variant: "destructive",
        title: "Query Failed",
        description: "Unable to retrieve stored data."
      });
    }
  };

  const handleApiKeySubmit = () => {
    if (apiKey.trim()) {
      localStorage.setItem('openai_api_key', apiKey.trim());
      setUploadState(prev => ({ ...prev, showConfig: false }));
      toast({
        title: "API Key Saved",
        description: "OpenAI API key has been configured for enhanced analysis."
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

  if (uploadState.showConfig) {
    return (
      <Card className="w-full max-w-md mx-auto">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="w-5 h-5" />
            Configure AI Analysis
          </CardTitle>
          <CardDescription>
            Enter your OpenAI API key for enhanced financial analysis
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <label className="text-sm font-medium mb-2 block">OpenAI API Key</label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="sk-..."
              className="w-full px-3 py-2 border border-border rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
            />
            <p className="text-xs text-muted-foreground mt-1">
              Your key is stored locally and used only for analysis
            </p>
          </div>
          <div className="flex gap-2">
            <Button onClick={handleApiKeySubmit} className="flex-1">
              Save & Continue
            </Button>
            <Button variant="outline" onClick={() => setUploadState(prev => ({ ...prev, showConfig: false }))}>
              Skip
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <input
        type="file"
        id="excel-upload"
        accept=".xlsx,.xls,.csv"
        multiple
        onChange={handleFileSelect}
        className="hidden"
      />
      
      <div
        className="upload-zone cursor-pointer"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onClick={() => document.getElementById('excel-upload')?.click()}
      >
        <div className="text-center">
          <Upload className="w-6 h-6 text-finley-accent mx-auto mb-2" />
          <p className="text-sm font-medium text-foreground mb-1">Upload Excel</p>
          <p className="text-xs text-muted-foreground">
            Drag & drop or click to browse (up to 10 files)
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            .xlsx, .xls, .csv (max 10MB each)
          </p>
        </div>
      </div>

      {/* File Progress List */}
      {uploadState.files.length > 0 && (
        <div className="space-y-2 max-h-60 overflow-y-auto">
          {uploadState.files.map((fileState, index) => (
            <div key={index} className="bg-muted/50 p-3 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  {getStatusIcon(fileState.status)}
                  <span className="text-sm font-medium truncate max-w-32">
                    {fileState.file.name}
                  </span>
                </div>
                <button
                  onClick={() => removeFile(index)}
                  className="text-muted-foreground hover:text-foreground"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
              
              {fileState.status === 'processing' && (
                <>
                  <div className="w-full bg-muted rounded-full h-1.5 mb-1">
                    <div 
                      className="bg-primary h-1.5 rounded-full transition-all duration-300"
                      style={{ width: `${fileState.progress}%` }}
                    />
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {fileState.currentStep}
                  </p>
                </>
              )}
              
              {fileState.status === 'error' && (
                <p className="text-xs text-destructive">
                  {fileState.error}
                </p>
              )}
              
              {fileState.status === 'success' && (
                <p className="text-xs text-green-600">
                  Analysis complete
                </p>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Uploaded Files List */}
      {uploadState.uploadedFiles.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-foreground">Uploaded Files</h4>
          <div className="space-y-2 max-h-40 overflow-y-auto">
            {uploadState.uploadedFiles.map((file) => (
              <div key={file.id} className="bg-finley-accent/10 p-3 rounded-lg border border-finley-accent/20">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-finley-accent" />
                    <div>
                      <span className="text-sm font-medium text-foreground">
                        {file.name}
                      </span>
                      <p className="text-xs text-muted-foreground">
                        Uploaded {file.uploadedAt.toLocaleDateString()} at {file.uploadedAt.toLocaleTimeString()}
                      </p>
                      {file.analysisResults && (
                        <p className="text-xs text-finley-accent font-medium">
                          {file.analysisResults.documentType || 'Financial Document'} • Stored in Database
                        </p>
                      )}
                    </div>
                  </div>
                  <button
                    onClick={() => deleteUploadedFile(file.id)}
                    className="text-muted-foreground hover:text-destructive transition-colors"
                    title="Delete file"
                  >
                    🗑️
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      
      
      <div className="mt-3 text-center space-y-2">
        <button
          onClick={() => setUploadState(prev => ({ ...prev, showConfig: true }))}
          className="text-xs text-finley-accent hover:underline flex items-center gap-1 mx-auto"
        >
          <Settings className="w-3 h-3" />
          Configure AI for enhanced analysis
        </button>
        
        {uploadState.uploadedFiles.length > 0 && (
          <button
            onClick={viewStoredData}
            className="text-xs text-finley-accent hover:underline flex items-center gap-1 mx-auto"
          >
            🔍 View Stored Database Records
          </button>
        )}
        
        {uploadState.files.length > 0 && (
          <button
            onClick={resetUpload}
            className="text-xs text-muted-foreground hover:text-foreground underline"
          >
            Clear all files
          </button>
        )}
      </div>
    </div>
  );
};