import { useState, useCallback } from 'react';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/hooks/use-toast';

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
    // Use Railway FastAPI URL
    this.apiUrl = "https://ai-financial-backend-production.up.railway.app";
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

  async processFile(
    file: File, 
    customPrompt?: string
  ): Promise<FastAPIProcessingResult> {
    try {
      this.updateProgress('upload', 'Uploading file to secure processing...', 10);
      
      // Upload file to Supabase storage first
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        throw new Error('User authentication required');
      }

      const fileName = `${user.id}/${Date.now()}-${file.name}`;
      const { data: uploadData, error: uploadError } = await supabase.storage
        .from('finley-uploads')
        .upload(fileName, file);

      if (uploadError) {
        throw new Error(`Upload failed: ${uploadError.message}`);
      }

      this.updateProgress('processing', 'Initializing advanced AI analysis...', 20);

      // Create processing job in database
      const { data: jobData, error: jobError } = await supabase
        .from('ingestion_jobs')
        .insert({
          job_type: 'fastapi_excel_analysis',
          user_id: user.id,
          status: 'processing',
          progress: 20
        })
        .select()
        .single();

      if (jobError) {
        throw new Error(`Job creation failed: ${jobError.message}`);
      }

      this.updateProgress('analysis', 'Connecting to FastAPI for deep analysis...', 30);

      // Connect to WebSocket for real-time updates
      const ws = new WebSocket(`wss://ai-financial-backend-production.up.railway.app/ws/${jobData.id}`);
      
      return new Promise((resolve, reject) => {
        let lastProgress = 30;

        ws.onopen = async () => {
          try {
            // Send processing request to FastAPI
            const response = await fetch(`${this.apiUrl}/process-excel`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                job_id: jobData.id,
                storage_path: fileName,
                file_name: file.name,
                supabase_url: "https://gnrbafqifucxlaihtyuv.supabase.co",
                supabase_key: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImducmJhZnFpZnVjeGxhaWh0eXV2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTMxMTM5OTksImV4cCI6MjA2ODY4OTk5OX0.Lb5Fuu1ktYuPKBgx0Oxla9SXot-TWI-bPhsML9EkRwE",
                custom_prompt: customPrompt
              }),
            });

            if (!response.ok) {
              throw new Error(`FastAPI request failed: ${response.statusText}`);
            }
          } catch (error) {
            reject(error);
          }
        };

        ws.onmessage = (event) => {
          try {
            const update = JSON.parse(event.data);
            
            if (update.type === 'progress') {
              lastProgress = Math.max(lastProgress, update.progress || 30);
              
              // Enhanced progress messages based on FastAPI feedback
              let enhancedMessage = update.message;
              if (update.sheet_info) {
                enhancedMessage = `Analyzing ${update.sheet_info.name} (${update.sheet_info.type})`;
              }
              
              this.updateProgress(
                update.step || 'processing',
                enhancedMessage,
                lastProgress,
                update.sheet_progress
              );
            } else if (update.type === 'sheet_analysis') {
              this.updateProgress(
                'analysis',
                `Found ${update.sheet_count} sheets: ${update.sheet_types.join(', ')}`,
                60
              );
            } else if (update.type === 'complete') {
              ws.close();
              
              // Enhanced result with sophisticated analysis
              const resultData = update.result as any;
              const result: FastAPIProcessingResult = {
                documentType: resultData.document_type || 'unknown',
                insights: resultData.insights || {},
                metrics: resultData.metrics || {},
                summary: resultData.summary || '',
                sheets: resultData.sheets || [],
                customPromptSuggestions: resultData.suggested_prompts || [],
                processingTime: resultData.processing_time || 0
              };
              
              resolve(result);
            } else if (update.type === 'error') {
              ws.close();
              reject(new Error(update.message));
            }
          } catch (error) {
            console.error('WebSocket message parsing error:', error);
          }
        };

        ws.onerror = (error) => {
          reject(new Error('WebSocket connection failed'));
        };

        ws.onclose = () => {
          // If closed without completion, check job status
          setTimeout(async () => {
            try {
              const { data: job } = await supabase
                .from('ingestion_jobs')
                .select('*')
                .eq('id', jobData.id)
                .single();

              if (job?.status === 'completed' && job.result) {
                const jobResult = job.result as any;
                resolve({
                  documentType: jobResult.document_type || 'unknown',
                  insights: jobResult.insights || {},
                  metrics: jobResult.metrics || {},
                  summary: jobResult.summary || '',
                  sheets: jobResult.sheets || [],
                  customPromptSuggestions: jobResult.suggested_prompts || [],
                  processingTime: jobResult.processing_time || 0
                });
              }
            } catch (error) {
              console.error('Error checking job status:', error);
            }
          }, 1000);
        };

        // Timeout after 5 minutes
        setTimeout(() => {
          ws.close();
          reject(new Error('Processing timeout - please try again'));
        }, 300000);
      });

    } catch (error) {
      throw new Error(`FastAPI processing failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  // Get sheet-specific insights
  async getSheetInsights(sheetName: string, jobId: string): Promise<any> {
    try {
      const response = await fetch(`${this.apiUrl}/sheet-insights`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          job_id: jobId,
          sheet_name: sheetName
        }),
      });

      if (!response.ok) {
        throw new Error(`Sheet insights request failed: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      throw new Error(`Failed to get sheet insights: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }
}

export const useFastAPIProcessor = () => {
  const [processor] = useState(() => new FastAPIProcessor());
  const { toast } = useToast();

  const processFileWithFastAPI = useCallback(async (
    file: File,
    customPrompt?: string,
    onProgress?: (progress: FastAPIProcessingProgress) => void
  ) => {
    try {
      if (onProgress) {
        processor.setProgressCallback(onProgress);
      }

      const result = await processor.processFile(file, customPrompt);
      
      toast({
        title: "Analysis Complete",
        description: `Processed ${result.sheets.length} sheets with FastAPI intelligence`
      });

      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      toast({
        variant: "destructive", 
        title: "FastAPI Processing Failed",
        description: errorMessage
      });
      throw error;
    }
  }, [processor, toast]);

  return {
    processFileWithFastAPI,
    getSheetInsights: processor.getSheetInsights.bind(processor)
  };
};