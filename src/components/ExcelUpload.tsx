import { useState, useCallback } from 'react';
import { Upload, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/hooks/use-toast';

interface UploadState {
  status: 'idle' | 'uploading' | 'success' | 'error';
  progress: number;
  error?: string;
  fileName?: string;
}

export const ExcelUpload = () => {
  const [uploadState, setUploadState] = useState<UploadState>({
    status: 'idle',
    progress: 0
  });
  const { toast } = useToast();

  const validateFile = (file: File): { isValid: boolean; error?: string } => {
    // Check file type
    const validTypes = [
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', // .xlsx
      'application/vnd.ms-excel', // .xls
      'text/csv' // .csv
    ];
    
    if (!validTypes.includes(file.type) && !file.name.match(/\.(xlsx|xls|csv)$/i)) {
      return {
        isValid: false,
        error: 'Please upload a valid Excel file (.xlsx, .xls) or CSV file.'
      };
    }

    // Check file size (max 10MB)
    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
      return {
        isValid: false,
        error: 'File size must be less than 10MB.'
      };
    }

    return { isValid: true };
  };

  const uploadToSupabase = async (file: File): Promise<string> => {
    // Upload file to Supabase Storage
    const fileName = `${Date.now()}-${file.name}`;
    const { data, error } = await supabase.storage
      .from('finley-uploads')
      .upload(fileName, file);

    if (error) {
      throw new Error(`Storage upload failed: ${error.message}`);
    }

    return data.path;
  };

  const saveToDatabase = async (file: File, storagePath: string) => {
    // Save record to raw_records table
    const { error } = await supabase
      .from('raw_records')
      .insert({
        source: 'excel_upload',
        file_name: file.name,
        file_size: file.size,
        content: { storage_path: storagePath },
        metadata: {
          file_type: file.type,
          uploaded_at: new Date().toISOString()
        },
        status: 'uploaded'
      });

    if (error) {
      throw new Error(`Database save failed: ${error.message}`);
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
      status: 'uploading',
      progress: 0,
      fileName: file.name
    });

    try {
      // Simulate progress updates
      setUploadState(prev => ({ ...prev, progress: 25 }));
      
      // Upload to Supabase Storage
      const storagePath = await uploadToSupabase(file);
      setUploadState(prev => ({ ...prev, progress: 75 }));
      
      // Save to database
      await saveToDatabase(file, storagePath);
      setUploadState(prev => ({ ...prev, progress: 100 }));
      
      // Success state
      setUploadState({
        status: 'success',
        progress: 100,
        fileName: file.name
      });

      toast({
        title: "Upload Successful",
        description: `${file.name} has been uploaded and is ready for analysis.`
      });

      // TODO: Trigger FastAPI processing
      console.log('File uploaded successfully. Ready to send to FastAPI for processing.');
      
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
    setUploadState({
      status: 'idle',
      progress: 0
    });
  };

  if (uploadState.status === 'success') {
    return (
      <div className="text-center p-4">
        <CheckCircle className="w-8 h-8 text-green-500 mx-auto mb-2" />
        <p className="text-sm font-medium text-foreground">Upload Complete</p>
        <p className="text-xs text-muted-foreground mb-3">{uploadState.fileName}</p>
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
        <p className="text-sm font-medium text-destructive">Upload Failed</p>
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

  if (uploadState.status === 'uploading') {
    return (
      <div className="text-center p-4">
        <Loader2 className="w-8 h-8 text-finley-accent mx-auto mb-2 animate-spin" />
        <p className="text-sm font-medium text-foreground">Uploading...</p>
        <p className="text-xs text-muted-foreground mb-2">{uploadState.fileName}</p>
        <div className="w-full bg-muted rounded-full h-1.5 mb-2">
          <div 
            className="bg-primary h-1.5 rounded-full transition-all duration-300"
            style={{ width: `${uploadState.progress}%` }}
          />
        </div>
        <p className="text-xs text-muted-foreground">{uploadState.progress}%</p>
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