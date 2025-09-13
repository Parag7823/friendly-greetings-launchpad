import { useState, useCallback, useRef } from 'react';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/hooks/use-toast';
import { FileProcessingEvent } from '@/types/database';

interface SheetMetadata {
  name: string;
  type: 'balance_sheet' | 'income_statement' | 'cash_flow' | 'budget' | 'general';
  rowCount: number;
  columnCount: number;
  keyColumns: string[];
  hasNumbers: boolean;
  preview: any[][];
  confidenceScore: number;
  detectedPeriod?: string;
  suggestedPrompts?: string[];
}

interface FastAPIProcessingResult {
  documentType: string;
  insights: any;
  metrics: any;
  summary: string;
  sheets: SheetMetadata[];
  customPromptSuggestions: string[];
  processingTime: number;
}

interface FastAPIProcessingProgress {
  step: string;
  message: string;
  progress: number;
  sheetProgress?: {
    currentSheet: string;
    sheetsCompleted: number;
    totalSheets: number;
  };
}

export class FastAPIProcessor {
  private apiUrl: string;
  private progressCallback?: (progress: FastAPIProcessingProgress) => void;

  constructor() {
    // Use Render FastAPI URL
    this.apiUrl = "https://friendly-greetings-launchpad.onrender.com";
  }

  setProgressCallback(callback: (progress: FastAPIProcessingProgress) => void) {
    this.progressCallback = callback;
  }

  private updateProgress(step: string, message: string, progress: number, sheetProgress?: any) {
    if (this.progressCallback) {
      this.progressCallback({
        step,
        message,
        progress,
        sheetProgress
      });
    }
  }

  private async calculateFileHash(fileBuffer: ArrayBuffer): Promise<string> {
    const hashBuffer = await crypto.subtle.digest('SHA-256', fileBuffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  }

  private async checkForDuplicates(userId: string, fileHash: string, fileName: string): Promise<{
    is_duplicate: boolean;
    duplicate_files?: any[];
    latest_duplicate?: any;
    recommendation?: string;
    message?: string;
    error?: string;
  }> {
    try {
      // Check if file with same hash exists
      const { data: existingFiles, error } = await supabase
        .from('raw_records')
        .select('id, file_name, created_at, content')
        .eq('user_id', userId)
        .eq('file_hash', fileHash) as any;

      if (error) {
        console.error('Error checking for duplicates:', error);
        return { is_duplicate: false, error: error.message };
      }

      if (existingFiles && existingFiles.length > 0) {
        const duplicateFiles = existingFiles.map(file => ({
          id: file.id,
          filename: file.file_name,
          uploaded_at: file.created_at,
          total_rows: (file.content as any)?.total_rows || 0
        }));

        const latestDuplicate = duplicateFiles.reduce((latest, current) => 
          new Date(current.uploaded_at) > new Date(latest.uploaded_at) ? current : latest
        );

        return {
          is_duplicate: true,
          duplicate_files: duplicateFiles,
          latest_duplicate: latestDuplicate,
          recommendation: 'replace_or_skip',
          message: `Identical file '${latestDuplicate.filename}' was uploaded on ${latestDuplicate.uploaded_at.split('T')[0]}. Do you want to replace it or skip this upload?`
        };
      }

      return { is_duplicate: false };
    } catch (error) {
      console.error('Duplicate check failed:', error);
      return { is_duplicate: false, error: error instanceof Error ? error.message : 'Unknown error' };
    }
  }

  private setupWebSocketConnection(jobId: string): Promise<any> {
    return new Promise((resolve, reject) => {
      const wsUrl = `wss://friendly-greetings-launchpad.onrender.com/ws/${jobId}`;
      console.log(`Connecting to WebSocket: ${wsUrl}`);
      
      const ws = new WebSocket(wsUrl);
      let timeoutId: NodeJS.Timeout;

      // Set a timeout for the connection
      timeoutId = setTimeout(() => {
        ws.close();
        reject(new Error('WebSocket connection timeout'));
      }, 10000); // 10 second timeout

      ws.onopen = () => {
        console.log('WebSocket connected for job:', jobId);
        clearTimeout(timeoutId);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('WebSocket progress update:', data);

          if (data.step === 'duplicate_detected') {
            this.progressCallback?.(data);
          } else if (data.step === 'partial_duplicate_detected') {
            this.progressCallback?.(data);
          } else if (data.status === 'completed') {
            clearTimeout(timeoutId);
            ws.close();
            resolve(data.result || data);
          } else if (data.status === 'error') {
            clearTimeout(timeoutId);
            ws.close();
            reject(new Error(data.error || 'Processing failed'));
          } else {
            this.updateProgress(
              data.step || 'processing',
              data.message || 'Processing...',
              data.progress || 0,
              data.sheetProgress
            );
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        clearTimeout(timeoutId);
        reject(new Error('WebSocket connection failed - will use polling fallback'));
      };

      ws.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason);
        clearTimeout(timeoutId);
        if (event.code !== 1000 && event.reason !== 'Processing completed') {
          reject(new Error('WebSocket connection closed unexpectedly'));
        }
      };
    });
  }

  private async pollForResults(jobId: string, initialResponse: any): Promise<any> {
    const maxAttempts = 60; // 5 minutes max (5 seconds * 60)
    let attempts = 0;
    
    while (attempts < maxAttempts) {
      try {
        // Check job status
        const statusResponse = await fetch(`${this.apiUrl}/job-status/${jobId}`);
        if (statusResponse.ok) {
          const statusData = await statusResponse.json();
          
          if (statusData.status === 'completed') {
            this.updateProgress('complete', 'Processing completed!', 100);
            return statusData.result || statusData;
          } else if (statusData.status === 'failed') {
            throw new Error(statusData.error || 'Processing failed');
          } else if (statusData.status === 'cancelled') {
            throw new Error('Processing was cancelled');
          }
          
          // Update progress if available
          if (statusData.progress !== undefined) {
            this.updateProgress('processing', statusData.message || 'Processing...', statusData.progress);
          }
        }
        
        // Wait 5 seconds before next poll (faster polling)
        await new Promise(resolve => setTimeout(resolve, 5000));
        attempts++;
        
      } catch (error) {
        console.error('Polling error:', error);
        attempts++;
        await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds on error
      }
    }
    
    // If we get here, polling timed out
    throw new Error('Processing timeout - please try again');
  }

  async processFileWithFastAPI(
    file: File, 
    customPrompt?: string,
    onProgress?: (progress: any) => void,
    onJobId?: (jobId: string) => void
  ): Promise<FastAPIProcessingResult> {
    let jobData: any = null;
    
    try {
      this.updateProgress('upload', 'Checking for duplicates...', 5);
      
      // Get user first
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        throw new Error('User authentication required');
      }

      // Calculate file hash for duplicate detection
      const fileBuffer = await file.arrayBuffer();
      const fileHash = await this.calculateFileHash(fileBuffer);

      // Check for duplicates
      const duplicateCheck = await this.checkForDuplicates(user.id, fileHash, file.name);
      
      if (duplicateCheck.is_duplicate) {
        this.updateProgress('duplicate_detected', 'Duplicate file detected!', 15);
        // Emit duplicate detection event through progress callback
        if (onProgress) {
          onProgress({
            step: 'duplicate_detected',
            message: 'Duplicate file detected!',
            progress: 15,
            duplicate_info: duplicateCheck.duplicate_files,
            job_id: null,
            file_hash: fileHash
          });
        }
        // For now, continue processing - in real implementation, wait for user decision
      }

      this.updateProgress('upload', 'Uploading file...', 20);
      
      // Create FormData
      const formData = new FormData();
      formData.append('file', file);
      formData.append('user_id', user.id);
      formData.append('file_hash', fileHash);
      
      if (customPrompt) {
        formData.append('custom_prompt', customPrompt);
      }

      // Upload file
      const uploadResponse = await fetch(`${this.apiUrl}/upload`, {
        method: 'POST',
        body: formData
      });

      if (!uploadResponse.ok) {
        const errorData = await uploadResponse.json();
        throw new Error(errorData.detail || 'Upload failed');
      }

      jobData = await uploadResponse.json();
      const jobId = jobData.job_id;
      
      if (onJobId) {
        onJobId(jobId);
      }

      this.updateProgress('processing', 'Processing file...', 25);

      // Try WebSocket connection first, fall back to polling
      try {
        const result = await this.setupWebSocketConnection(jobId);
        return result;
      } catch (wsError) {
        console.log('WebSocket failed, falling back to polling:', wsError);
        return await this.pollForResults(jobId, jobData);
      }

    } catch (error) {
      console.error('File processing error:', error);
      throw error;
    }
  }

  getSheetInsights(sheets: SheetMetadata[]): any {
    if (!sheets || sheets.length === 0) {
      return {
        totalSheets: 0,
        sheetTypes: {},
        recommendations: []
      };
    }

    const sheetTypes = sheets.reduce((acc, sheet) => {
      acc[sheet.type] = (acc[sheet.type] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    const recommendations = [];
    
    // Add recommendations based on sheet types
    if (sheetTypes.balance_sheet && sheetTypes.income_statement) {
      recommendations.push("Consider analyzing financial ratios across balance sheet and income statement");
    }
    
    if (sheetTypes.cash_flow) {
      recommendations.push("Cash flow analysis available for liquidity insights");
    }

    return {
      totalSheets: sheets.length,
      sheetTypes,
      recommendations,
      averageConfidence: sheets.reduce((sum, sheet) => sum + sheet.confidenceScore, 0) / sheets.length
    };
  }
}

export const useFastAPIProcessor = () => {
  const { toast } = useToast();
  const processorRef = useRef(new FastAPIProcessor());

  const processFileWithFastAPI = useCallback(
    async (
      file: File,
      customPrompt?: string,
      onProgress?: (progress: any) => void,
      onJobId?: (jobId: string) => void
    ) => {
      const processor = processorRef.current;
      processor.setProgressCallback(onProgress || (() => {}));

      try {
        const result = await processor.processFileWithFastAPI(file, customPrompt, onProgress, onJobId);
        toast({ title: "Success", description: "File processed successfully." });
        return result;
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred';
        toast({ title: "Error", description: errorMessage, variant: "destructive" });
        throw error;
      }
    },
    [toast]
  );

  return {
    processFileWithFastAPI,
    getSheetInsights: processorRef.current.getSheetInsights.bind(processorRef.current),
  };
};
