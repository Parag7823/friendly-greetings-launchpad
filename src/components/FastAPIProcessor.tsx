import { useState, useCallback } from 'react';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/hooks/use-toast';
import { FileProcessingEvent } from '@/types/database';
import { config } from '@/config';
import { UnifiedErrorHandler, ErrorSeverity, ErrorSource } from '@/utils/errorHandler';

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
  status?: string;
  duplicate_analysis?: any;
  job_id?: string;
  file_hash?: string;
  storage_path?: string;
  file_name?: string;
  requires_user_decision?: boolean;
  message?: string;
  existing_file_id?: string;
  file_id?: string;
  delta_analysis?: any;
}

interface FastAPIProcessingProgress {
  step: string;
  message: string;
  progress: number;
  status?: 'processing' | 'completed' | 'error' | 'failed';
  sheetProgress?: {
    currentSheet: string;
    sheetsCompleted: number;
    totalSheets: number;
  };
  // When duplicates are detected via WebSocket, backend attaches rich context.
  // We forward it via this optional field so the UI can show a modal immediately.
  extra?: any;
  requires_user_decision?: boolean;
}

export class FastAPIProcessor {
  private apiUrl: string;
  private progressCallback?: (progress: FastAPIProcessingProgress) => void;

  constructor() {
    // Use centralized config for API URL
    this.apiUrl = config.apiUrl;
  }

  setProgressCallback(callback: (progress: FastAPIProcessingProgress) => void) {
    this.progressCallback = callback;
  }

  private updateProgress(step: string, message: string, progress: number, sheetProgress?: any, extra?: any) {
    if (this.progressCallback) {
      this.progressCallback({
        step,
        message,
        progress,
        sheetProgress,
        extra,
        requires_user_decision: extra?.requires_user_decision
      });
    }
  }

  private async calculateFileHash(fileBuffer: ArrayBuffer): Promise<string> {
    const hashBuffer = await crypto.subtle.digest('SHA-256', fileBuffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  }

  // CRITICAL FIX: Removed checkForDuplicates method
  // Duplicate detection is now handled exclusively by backend /process-excel endpoint
  // This eliminates redundant API calls and ensures single source of truth

  // ISSUE #1 FIX: Removed raw WebSocket implementation
  // Now using Socket.IO via useFileStatusSocket hook (initialized in FinleyLayout)
  // Benefits:
  // - Single shared WebSocket connection
  // - Automatic reconnection with exponential backoff
  // - Built-in heartbeat/ping-pong
  // - Room-based messaging
  // - Consistent with rest of app

  /**
   * Wait for job completion using Socket.IO events
   * Polls the job status endpoint as fallback if Socket.IO doesn't deliver results
   */
  private async waitForJobCompletion(jobId: string): Promise<any> {
    // Maximum wait time: 25 minutes (same as backend processing timeout)
    const maxWaitTime = 25 * 60 * 1000;
    const startTime = Date.now();
    // ISSUE #3 FIX: Increase polling interval from 1.5s to 5s
    // Reduces requests from 1000 to 300 per file (70% reduction)
    // Socket.IO events should arrive first anyway, polling is just fallback
    const pollInterval = 5000; // 5 seconds

    while (Date.now() - startTime < maxWaitTime) {
      try {
        // Poll job status as fallback (Socket.IO events should arrive first)
        const response = await fetch(`${this.apiUrl}/job-status/${jobId}`);
        if (!response.ok) {
          await new Promise(resolve => setTimeout(resolve, pollInterval));
          continue;
        }

        const statusData = await response.json();

        if (statusData.status === 'completed') {
          this.updateProgress('complete', 'Processing completed!', 100);
          return statusData.result || statusData;
        } else if (statusData.status === 'failed') {
          throw new Error(statusData.error || 'Processing failed');
        } else if (statusData.status === 'cancelled') {
          throw new Error('Processing was cancelled');
        }

        // Show progress during polling
        if (statusData.progress !== undefined) {
          const timeElapsed = Math.floor((Date.now() - startTime) / 1000);
          const progressMessage = statusData.message || `Processing... (${timeElapsed}s elapsed)`;
          this.updateProgress('processing', progressMessage, statusData.progress);
        }

        await new Promise(resolve => setTimeout(resolve, pollInterval));
      } catch (error) {
        console.error('Error checking job status:', error);
        await new Promise(resolve => setTimeout(resolve, pollInterval));
      }
    }

    throw new Error(
      'Processing is taking longer than expected. The job may still be running in the background. Please refresh the page or try again later.'
    );
  }

  async processFile(
    file: File, 
    customPrompt?: string,
    userId?: string,
    onJobId?: (jobId: string, fileHash: string) => void
  ): Promise<FastAPIProcessingResult> {
    let jobData: any = null;
    const startTime = performance.now(); // Track processing start time
    
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
      
      // Get session for consistent token passing
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) {
        throw new Error('User session required');
      }
      
      // CRITICAL FIX: Remove redundant frontend duplicate check
      // Backend /process-excel endpoint handles duplicate detection comprehensively
      // This eliminates double API call and ensures single source of truth
      
      const fileName = `${user.id}/${Date.now()}-${file.name}`;
      const { data: uploadData, error: uploadError } = await supabase.storage
        .from('finely-upload')
        .upload(fileName, file);

      if (uploadError) {
        throw new Error(`Upload failed: ${uploadError.message}`);
      }

      this.updateProgress('processing', 'Initializing advanced AI analysis...', 10);

      // Create processing job in database
      const { data: jobResult, error: jobError } = await supabase
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

      jobData = jobResult; // Assign to outer scope variable
      
      // Notify about job ID for cancel functionality
      // Note: file_hash will be computed by backend and returned in response
      if (onJobId) {
        onJobId(jobData.id, undefined);  // Hash computed by backend
      }

      this.updateProgress('analysis', 'Processing with FastAPI backend...', 30);

      // Prepare request for FastAPI backend (session already fetched above)
      const requestBody = {
        job_id: jobData.id,
        storage_path: fileName,
        file_name: file.name,
        user_id: user.id,
        file_hash: fileHash,
        session_token: session.access_token,  // Backend reads from body for validation
        endpoint: 'process-excel'  // Add endpoint for security validation
      };

      // Start FastAPI backend processing and connect to WebSocket for real-time updates
      try {
        // Add JWT token to API request headers for Supabase operations
        const headers: Record<string, string> = {
          'Content-Type': 'application/json',
        };
        
        if (session?.access_token) {
          headers['Authorization'] = `Bearer ${session.access_token}`;
        }
        
        // Start the processing job
        const response = await fetch(`${this.apiUrl}/process-excel`, {
          method: 'POST',
          headers,
          body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
          throw new Error(`Backend processing failed: ${response.statusText}`);
        }

        const initialResponse = await response.json();
        // FastAPI processing started successfully

        // Check if duplicate was detected
        if (initialResponse.status === 'duplicate_detected') {
          this.updateProgress('duplicate_detected', 'Duplicate file detected!', 20);
          
          // Return duplicate information for user decision
          return {
            status: 'duplicate_detected',
            duplicate_analysis: initialResponse.duplicate_analysis,
            job_id: initialResponse.job_id,
            requires_user_decision: true,
            message: initialResponse.message,
            // Return empty result structure for consistency
            documentType: 'duplicate',
            insights: {},
            metrics: {},
            summary: 'Duplicate file detected',
            sheets: [],
            customPromptSuggestions: [],
            processingTime: 0,
            file_hash: fileHash,
            storage_path: fileName,
            file_name: file.name,
            existing_file_id: initialResponse.existing_file_id || null,
            delta_analysis: initialResponse.delta_analysis
          };
        }

        // Handle near/content duplicate statuses that also require a user decision
        if (initialResponse.status === 'near_duplicate_detected' || initialResponse.status === 'content_duplicate_detected') {
          this.updateProgress('duplicate_detected', 'Potential duplicate detected!', 20);
          const analysis =
            initialResponse.duplicate_analysis ??
            initialResponse.near_duplicate_analysis ??
            initialResponse.content_duplicate_analysis;
          return {
            status: initialResponse.status,
            duplicate_analysis: analysis,
            delta_analysis: initialResponse.delta_analysis,
            job_id: initialResponse.job_id,
            requires_user_decision: true,
            message: initialResponse.message,
            // Return empty result structure for consistency
            documentType: 'duplicate',
            insights: {},
            metrics: {},
            summary: 'Duplicate or similar file detected',
            sheets: [],
            customPromptSuggestions: [],
            processingTime: 0,
            file_hash: fileHash,
            storage_path: fileName,
            file_name: file.name,
            existing_file_id: initialResponse.existing_file_id || null
          };
        }

        // ISSUE #1 FIX: Use Socket.IO for real-time updates (initialized globally in FinleyLayout)
        // Socket.IO handles:
        // - Connection management
        // - Automatic reconnection with exponential backoff
        // - Heartbeat/ping-pong
        // - No polling needed
        this.updateProgress('websocket', 'Waiting for real-time updates via Socket.IO...', 35);
        
        // Wait for job completion via Socket.IO events
        // The backend emits 'job_complete' when processing finishes
        // This is handled by useFileStatusSocket hook which updates Zustand store
        const backendResult = await this.waitForJobCompletion(jobData.id);
        
        // Parse backend results into our format
        const sheets: SheetMetadata[] = [];
        const insights = backendResult.results || {};
        const metrics = insights.summary_stats || {};
        const summary = insights.analysis || 'Analysis completed successfully.';

        // Convert backend results to our sheet format
        if (backendResult.results?.processing_stats) {
          const stats = backendResult.results.processing_stats;
          sheets.push({
            name: 'Processed Data',
            type: 'general',
            rowCount: stats.total_rows_processed || 0,
            columnCount: 0,
            keyColumns: [],
            hasNumbers: true,
            preview: [],
            confidenceScore: stats.platform_confidence || 0.5,
            detectedPeriod: 'Current',
            suggestedPrompts: []
          });
        }

        this.updateProgress('complete', 'Analysis complete!', 100);
        // Do not write to raw_records/raw_events/metrics here; backend has already persisted
        // results atomically inside its transaction. Avoid double-writes from the frontend.

        const result: FastAPIProcessingResult = {
          documentType: backendResult.results?.document_type || 'unknown',
          insights: backendResult.results || {},
          metrics: backendResult.results?.summary_stats || {},
          summary: backendResult.results?.analysis || 'Analysis completed successfully.',
          sheets,
          customPromptSuggestions: [],
          processingTime: performance.now() - startTime
        };

        return result;

      } catch (backendError) {
        // MISMATCH FIX #3: Use unified error handler
        UnifiedErrorHandler.handle({
          message: backendError instanceof Error ? backendError.message : 'Backend processing failed',
          severity: ErrorSeverity.HIGH,
          source: ErrorSource.BACKEND,
          jobId: jobData?.id,
          retryable: true
        });
        
        // REMOVED: Incomplete fallback processing that doesn't run enrichment, entity resolution, or relationship detection
        // This ensures data quality consistency - all files must be processed by the backend
        this.updateProgress('error', 'Backend processing failed. Please try again.', 0);
        
        // Update job status to failed
        if (jobData?.id) {
          await supabase
            .from('ingestion_jobs')
            .update({
              status: 'failed',
              progress: 0,
              error_message: backendError instanceof Error ? backendError.message : 'Backend processing failed'
            })
            .eq('id', jobData.id);
        }
        
        throw new Error('Backend processing failed. Please try again later.');
      }

    } catch (error) {
      // Handle outer try-catch errors
      if (jobData?.id) {
        await supabase
          .from('ingestion_jobs')
          .update({
            status: 'failed',
            error_details: error instanceof Error ? error.message : 'Unknown error'
          })
          .eq('id', jobData.id);
      }
      
      throw new Error(`FastAPI processing failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  private detectSheetType(sheetName: string, data: any[]): 'balance_sheet' | 'income_statement' | 'cash_flow' | 'budget' | 'general' {
    const name = sheetName.toLowerCase();
    if (name.includes('balance') || name.includes('bs')) return 'balance_sheet';
    if (name.includes('income') || name.includes('profit') || name.includes('p&l') || name.includes('pnl')) return 'income_statement';
    if (name.includes('cash') || name.includes('flow')) return 'cash_flow';
    if (name.includes('budget') || name.includes('forecast')) return 'budget';
    return 'general';
  }

  private extractKeyColumns(data: any[]): string[] {
    if (!data || data.length === 0) return [];
    const headers = data[0] as any[];
    return headers?.filter(h => typeof h === 'string').slice(0, 5) || [];
  }

  private hasNumericData(data: any[]): boolean {
    return data.some(row => 
      Array.isArray(row) && row.some(cell => typeof cell === 'number')
    );
  }

  private detectPeriod(data: any[]): string {
    // Simple period detection based on common patterns
    const headers = data[0] as any[];
    if (headers?.some((h: any) => typeof h === 'string' && /202[0-9]/.test(h))) {
      return 'Annual';
    }
    if (headers?.some((h: any) => typeof h === 'string' && /(q[1-4]|quarter)/i.test(h))) {
      return 'Quarterly';
    }
    return 'Monthly';
  }

  private generatePrompts(type: string, sheetName: string): string[] {
    const prompts = {
      balance_sheet: [
        "What is my current ratio?",
        "How leveraged is my company?",
        "What are my biggest assets?"
      ],
      income_statement: [
        "What is my profit margin?",
        "How are my expenses trending?",
        "What is my biggest cost driver?"
      ],
      cash_flow: [
        "How is my cash position?",
        "What is my burn rate?",
        "Where is cash being used?"
      ],
      budget: [
        "Am I on track with my budget?",
        "Where am I over/under budget?",
        "What are my budget variances?"
      ]
    };
    return prompts[type as keyof typeof prompts] || [
      `Analyze the ${sheetName} data`,
      `What insights can you find in ${sheetName}?`,
      `Summarize the key metrics in ${sheetName}`
    ];
  }

  private generateInsights(sheets: SheetMetadata[]): any {
    return {
      totalSheets: sheets.length,
      detectedTypes: sheets.map(s => s.type),
      dataQuality: sheets.reduce((acc, s) => acc + s.confidenceScore, 0) / sheets.length,
      keyFindings: sheets.map(s => `${s.name}: ${s.type} with ${s.rowCount} rows`)
    };
  }

  private calculateMetrics(sheets: SheetMetadata[]): any {
    return {
      totalRows: sheets.reduce((acc, s) => acc + s.rowCount, 0),
      averageColumns: sheets.reduce((acc, s) => acc + s.columnCount, 0) / sheets.length,
      sheetsWithNumbers: sheets.filter(s => s.hasNumbers).length,
      coverage: sheets.filter(s => s.confidenceScore > 0.8).length / sheets.length
    };
  }

  private generateSummary(sheets: SheetMetadata[], insights: any): string {
    const types = [...new Set(sheets.map(s => s.type))];
    return `Analyzed ${sheets.length} sheets containing ${types.join(', ')}. ` +
           `Found ${insights.totalRows} total rows of financial data with ` +
           `${Math.round(insights.dataQuality * 100)}% confidence in data quality.`;
  }

  private detectDocumentType(sheets: SheetMetadata[]): string {
    const types = sheets.map(s => s.type);
    if (types.includes('balance_sheet') && types.includes('income_statement')) {
      return 'financial_statements';
    }
    if (types.includes('budget')) return 'budget_analysis';
    if (types.includes('cash_flow')) return 'cash_flow_analysis';
    return 'financial_data';
  }

  private extractSheetMetrics(sheet: SheetMetadata, recordId: string): any[] {
    const metrics: any[] = [];
    
    if (!sheet.preview || sheet.preview.length < 2) return metrics;
    
    const headers = sheet.preview[0] as string[];
    const dataRows = sheet.preview.slice(1);
    
    // Look for financial columns
    const amountColumns = headers
      .map((header, index) => ({ header: header?.toLowerCase() || '', index }))
      .filter(col => 
        col.header.includes('amount') || 
        col.header.includes('revenue') || 
        col.header.includes('expense') || 
        col.header.includes('profit') || 
        col.header.includes('cost') ||
        col.header.includes('price') ||
        col.header.includes('total')
      );
    
    // Extract metrics from each data row
    dataRows.forEach((row: any[], rowIndex) => {
      amountColumns.forEach(col => {
        const value = row[col.index];
        if (typeof value === 'number' && value > 0) {
          metrics.push({
            category: this.getCategoryFromHeader(col.header),
            subcategory: col.header,
            amount: value,
            date_recorded: new Date().toISOString().split('T')[0]
          });
        }
      });
    });
    
    return metrics;
  }

  private getCategoryFromHeader(header: string): string {
    if (header.includes('revenue') || header.includes('income')) return 'revenue';
    if (header.includes('expense') || header.includes('cost')) return 'expense';
    if (header.includes('profit')) return 'profit';
    if (header.includes('asset')) return 'asset';
    if (header.includes('liability')) return 'liability';
    return 'other';
  }

  private generateCustomPrompts(sheets: SheetMetadata[]): string[] {
    const types = [...new Set(sheets.map(s => s.type))];
    return [
      "Analyze my financial health",
      "What are the key trends in my data?",
      "How can I improve my financial position?",
      ...types.map(type => `Analyze my ${type.replace('_', ' ')}`),
      "Compare performance across periods"
    ];
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
    onProgress?: (progress: FastAPIProcessingProgress) => void,
    onJobId?: (jobId: string, fileHash: string) => void
  ) => {
    try {
      if (onProgress) {
        processor.setProgressCallback(onProgress);
      }

      toast({
        title: "Processing Started",
        description: "Real-time WebSocket updates enabled"
      });

      const result = await processor.processFile(file, customPrompt, undefined, onJobId);

      if (result?.requires_user_decision) {
        toast({
          title: "Action Required",
          description: result.message || "Duplicate detected. Please choose how to proceed."
        });
        return result;
      }

      toast({
        title: "Analysis Complete",
        description: `Processed ${result.sheets.length} sheets with real-time updates`
      });

      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      toast({
        variant: "destructive", 
        title: "Processing Failed",
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
