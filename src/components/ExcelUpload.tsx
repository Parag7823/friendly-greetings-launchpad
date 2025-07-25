import { useState, useCallback } from 'react';
import { Upload, CheckCircle, AlertCircle, Loader2, Brain, FileText, TrendingUp, Settings } from 'lucide-react';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/hooks/use-toast';
import { ExcelProcessor, ProcessingProgress, ProcessingResult } from '@/lib/excelProcessor';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';

interface UploadState {
  status: 'idle' | 'processing' | 'success' | 'error' | 'config';
  progress: number;
  error?: string;
  fileName?: string;
  currentStep?: string;
  analysisResults?: ProcessingResult;
}

export const ExcelUpload = () => {
  const [uploadState, setUploadState] = useState<UploadState>({
    status: 'idle',
    progress: 0
  });
  const [apiKey, setApiKey] = useState<string>('');
  const [showConfig, setShowConfig] = useState<boolean>(false);
  const { toast } = useToast();

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
      const { data: { user } } = await supabase.auth.getUser();
      
      const { error } = await supabase
        .from('ingestion_jobs')
        .insert({
          job_type: 'excel_analysis',
          user_id: user?.id,
          record_id: null,
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

      if (error) {
        console.warn('Failed to save job to database:', error);
      }
    } catch (error) {
      console.warn('Database save error:', error);
    }
  };

  const handleFileUpload = useCallback(async (file: File) => {
    const validation = validateFile(file);
    if (!validation.isValid) {
      setUploadState({
        status: 'error',
        progress: 0,
        error: validation.error
      });
      toast({
        variant: "destructive",
        title: "Upload Failed",
        description: validation.error
      });
      return;
    }

    setUploadState({
      status: 'processing',
      progress: 0,
      fileName: file.name,
      currentStep: 'Initializing...'
    });

    try {
      // Get API key from localStorage or state
      const storedApiKey = localStorage.getItem('openai_api_key') || apiKey;
      
      const processor = new ExcelProcessor(storedApiKey);
      
      processor.setProgressCallback((progress: ProcessingProgress) => {
        setUploadState(prev => ({
          ...prev,
          currentStep: progress.message,
          progress: progress.progress
        }));
      });

      const result = await processor.processFile(file);
      
      // Save to database in background
      saveJobToDatabase(file, result);

      setUploadState({
        status: 'success',
        progress: 100,
        fileName: file.name,
        analysisResults: result
      });

      toast({
        title: "Analysis Complete",
        description: `${file.name} has been successfully analyzed by Finley AI.`
      });
      
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      setUploadState({
        status: 'error',
        progress: 0,
        error: errorMessage
      });
      
      toast({
        variant: "destructive",
        title: "Analysis Failed",
        description: errorMessage
      });
    }
  }, [toast, apiKey]);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      handleFileUpload(file);
    }
  };

  const handleDrop = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    
    const file = event.dataTransfer.files[0];
    if (file) {
      handleFileUpload(file);
    }
  }, [handleFileUpload]);

  const handleDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
  }, []);

  const resetUpload = () => {
    setUploadState({
      status: 'idle',
      progress: 0
    });
  };

  const handleApiKeySubmit = () => {
    if (apiKey.trim()) {
      localStorage.setItem('openai_api_key', apiKey.trim());
      setShowConfig(false);
      toast({
        title: "API Key Saved",
        description: "OpenAI API key has been configured for enhanced analysis."
      });
    }
  };

  const getStepIcon = (status: string) => {
    switch (status) {
      case 'processing':
        return <Brain className="w-8 h-8 text-finley-accent mx-auto mb-2 animate-pulse" />;
      case 'success':
        return <CheckCircle className="w-8 h-8 text-green-500 mx-auto mb-2" />;
      case 'error':
        return <AlertCircle className="w-8 h-8 text-destructive mx-auto mb-2" />;
      default:
        return <Loader2 className="w-8 h-8 text-finley-accent mx-auto mb-2 animate-spin" />;
    }
  };

  if (showConfig) {
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
            <Button variant="outline" onClick={() => setShowConfig(false)}>
              Skip
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (uploadState.status === 'success' && uploadState.analysisResults) {
    const results = uploadState.analysisResults;
    return (
      <div className="space-y-4">
        <div className="text-center p-4">
          <CheckCircle className="w-8 h-8 text-green-500 mx-auto mb-2" />
          <p className="text-sm font-medium text-foreground">Analysis Complete</p>
          <p className="text-xs text-muted-foreground mb-2">{uploadState.fileName}</p>
          <div className="text-xs text-finley-accent bg-accent/20 rounded-lg p-2 border border-border mb-3">
            <TrendingUp className="w-4 h-4 inline mr-1" />
            {results.documentType}
          </div>
          <button
            onClick={resetUpload}
            className="finley-button-outline text-xs px-3 py-1"
          >
            Upload Another
          </button>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Analysis Results</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {results.summary && (
              <div>
                <h4 className="font-medium mb-2">Summary</h4>
                <p className="text-sm text-muted-foreground">{results.summary}</p>
              </div>
            )}

            {results.metrics && results.metrics.length > 0 && (
              <div>
                <h4 className="font-medium mb-2">Key Metrics</h4>
                <div className="grid grid-cols-2 gap-2">
                  {results.metrics.map((metric, index) => (
                    <div key={index} className="bg-muted/50 p-2 rounded">
                      <p className="text-xs text-muted-foreground">{metric.label}</p>
                      <p className="font-medium">{metric.value}</p>
                      {metric.change && (
                        <p className="text-xs text-finley-accent">{metric.change}</p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {results.insights && results.insights.length > 0 && (
              <div>
                <h4 className="font-medium mb-2">Insights</h4>
                <ul className="space-y-1">
                  {results.insights.map((insight, index) => (
                    <li key={index} className="text-sm text-muted-foreground">
                      • {insight}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    );
  }

  if (uploadState.status === 'error') {
    return (
      <div className="text-center p-4">
        <AlertCircle className="w-8 h-8 text-destructive mx-auto mb-2" />
        <p className="text-sm font-medium text-destructive">Analysis Failed</p>
        <p className="text-xs text-muted-foreground mb-3">{uploadState.error}</p>
        <div className="flex gap-2 justify-center">
          <button
            onClick={resetUpload}
            className="finley-button text-xs px-3 py-1"
          >
            Try Again
          </button>
          <button
            onClick={() => setShowConfig(true)}
            className="finley-button-outline text-xs px-3 py-1"
          >
            Configure AI
          </button>
        </div>
      </div>
    );
  }

  if (uploadState.status === 'processing') {
    return (
      <div className="text-center p-4">
        {getStepIcon(uploadState.status)}
        <p className="text-sm font-medium text-foreground">Analyzing Document...</p>
        <p className="text-xs text-muted-foreground mb-2">{uploadState.fileName}</p>
        <div className="w-full bg-muted rounded-full h-1.5 mb-2">
          <div 
            className="bg-primary h-1.5 rounded-full transition-all duration-300"
            style={{ width: `${uploadState.progress}%` }}
          />
        </div>
        <p className="text-xs text-finley-accent font-medium mb-1">{uploadState.progress}%</p>
        {uploadState.currentStep && (
          <p className="text-xs text-muted-foreground italic">
            {uploadState.currentStep}
          </p>
        )}
      </div>
    );
  }

  return (
    <div>
      <input
        type="file"
        id="excel-upload"
        accept=".xlsx,.xls,.csv"
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
            Drag & drop or click to browse
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            .xlsx, .xls, .csv (max 10MB)
          </p>
        </div>
      </div>
      
      <div className="mt-3 text-center">
        <button
          onClick={() => setShowConfig(true)}
          className="text-xs text-finley-accent hover:underline flex items-center gap-1 mx-auto"
        >
          <Settings className="w-3 h-3" />
          Configure AI for enhanced analysis
        </button>
      </div>
    </div>
  );
};