
import { useState, useCallback } from 'react';
import { Upload, CheckCircle, AlertCircle, Loader2, Brain, FileText, TrendingUp } from 'lucide-react';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/hooks/use-toast';

interface UploadState {
  status: 'idle' | 'uploading' | 'processing' | 'success' | 'error';
  progress: number;
  error?: string;
  fileName?: string;
  currentStep?: string;
  analysisResults?: any;
}

interface ProcessingMessage {
  step: string;
  message: string;
  progress: number;
  icon?: string;
}

export const ExcelUpload = () => {
  const [uploadState, setUploadState] = useState<UploadState>({
    status: 'idle',
    progress: 0
  });
  const [ws, setWs] = useState<WebSocket | null>(null);
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

  const setupWebSocket = (jobId: string) => {
    const websocket = new WebSocket(`wss://ai-financial-backend-production.up.railway.app/ws/${jobId}`);
    
    websocket.onopen = () => {
      console.log('WebSocket connected for job:', jobId);
    };

    websocket.onmessage = (event) => {
      try {
        const data: ProcessingMessage = JSON.parse(event.data);
        console.log('WebSocket message:', data);
        
        setUploadState(prev => ({
          ...prev,
          currentStep: data.message,
          progress: data.progress
        }));

        if (data.step === 'completed') {
          setUploadState(prev => ({
            ...prev,
            status: 'success',
            progress: 100,
            analysisResults: data
          }));
          websocket.close();
        } else if (data.step === 'error') {
          setUploadState(prev => ({
            ...prev,
            status: 'error',
            error: data.message
          }));
          websocket.close();
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
      setUploadState(prev => ({
        ...prev,
        status: 'error',
        error: 'Connection error during processing'
      }));
    };

    websocket.onclose = () => {
      console.log('WebSocket connection closed');
    };

    setWs(websocket);
    return websocket;
  };

  const uploadToSupabase = async (file: File): Promise<string> => {
    const fileName = `${Date.now()}-${file.name}`;
    const { data, error } = await supabase.storage
      .from('finley-uploads')
      .upload(fileName, file);

    if (error) {
      throw new Error(`Storage upload failed: ${error.message}`);
    }

    return data.path;
  };

  const createIngestionJob = async (file: File, storagePath: string): Promise<string> => {
    const { data, error } = await supabase
      .from('ingestion_jobs')
      .insert({
        job_type: 'excel_analysis',
        record_id: null,
        status: 'queued',
        progress: 0,
        result: {
          storage_path: storagePath,
          file_name: file.name,
          file_size: file.size
        }
      })
      .select('id')
      .single();

    if (error) {
      throw new Error(`Failed to create job: ${error.message}`);
    }

    return data.id;
  };

  const triggerFastAPIProcessing = async (jobId: string, storagePath: string, fileName: string) => {
    // Replace this URL with your actual Railway deployment URL
    const response = await fetch('https://YOUR_RAILWAY_URL.railway.app/process-excel', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        job_id: jobId,
        storage_path: storagePath,
        file_name: fileName,
        supabase_url: 'https://gnrbafqifucxlaihtyuv.supabase.co',
        supabase_key: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImducmJhZnFpZnVjeGxhaWh0eXV2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTMxMTM5OTksImV4cCI6MjA2ODY4OTk5OX0.Lb5Fuu1ktYuPKBgx0Oxla9SXot-TWI-bPhsML9EkRwE'
      })
    });

    if (!response.ok) {
      throw new Error(`FastAPI processing failed: ${response.statusText}`);
    }

    return response.json();
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
      status: 'uploading',
      progress: 0,
      fileName: file.name,
      currentStep: 'Uploading file to secure storage...'
    });

    try {
      // Step 1: Upload to Supabase Storage
      const storagePath = await uploadToSupabase(file);
      setUploadState(prev => ({ ...prev, progress: 25, currentStep: 'Creating analysis job...' }));
      
      // Step 2: Create ingestion job
      const jobId = await createIngestionJob(file, storagePath);
      setUploadState(prev => ({ ...prev, progress: 50, currentStep: 'Connecting to AI analysis engine...' }));
      
      // Step 3: Setup WebSocket for real-time updates
      setupWebSocket(jobId);
      setUploadState(prev => ({ ...prev, status: 'processing', progress: 60, currentStep: 'Starting intelligent analysis...' }));
      
      // Step 4: Trigger FastAPI processing
      await triggerFastAPIProcessing(jobId, storagePath, file.name);
      
      toast({
        title: "Analysis Started",
        description: `${file.name} is being analyzed by Finley AI. You'll see real-time progress updates.`
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
        title: "Upload Failed",
        description: errorMessage
      });
    }
  }, [toast]);

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
    if (ws) {
      ws.close();
      setWs(null);
    }
    setUploadState({
      status: 'idle',
      progress: 0
    });
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

  if (uploadState.status === 'success') {
    return (
      <div className="text-center p-4">
        <CheckCircle className="w-8 h-8 text-green-500 mx-auto mb-2" />
        <p className="text-sm font-medium text-foreground">Analysis Complete</p>
        <p className="text-xs text-muted-foreground mb-2">{uploadState.fileName}</p>
        <div className="text-xs text-finley-accent bg-accent/20 rounded-lg p-2 border border-border mb-3">
          <TrendingUp className="w-4 h-4 inline mr-1" />
          Ready for intelligent insights
        </div>
        <button
          onClick={resetUpload}
          className="finley-button-outline text-xs px-3 py-1"
        >
          Upload Another
        </button>
      </div>
    );
  }

  if (uploadState.status === 'error') {
    return (
      <div className="text-center p-4">
        <AlertCircle className="w-8 h-8 text-destructive mx-auto mb-2" />
        <p className="text-sm font-medium text-destructive">Analysis Failed</p>
        <p className="text-xs text-muted-foreground mb-3">{uploadState.error}</p>
        <button
          onClick={resetUpload}
          className="finley-button text-xs px-3 py-1"
        >
          Try Again
        </button>
      </div>
    );
  }

  if (uploadState.status === 'uploading' || uploadState.status === 'processing') {
    return (
      <div className="text-center p-4">
        {getStepIcon(uploadState.status)}
        <p className="text-sm font-medium text-foreground">
          {uploadState.status === 'processing' ? 'Analyzing Document...' : 'Uploading...'}
        </p>
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
    </div>
  );
};
