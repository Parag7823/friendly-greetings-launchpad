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

  // FIX #8: Move isCleanedUp outside to prevent multiple cleanup calls
  private wsCleanupFlags = new Map<string, boolean>();

  private async connectWebSocket(jobId: string): Promise<any> {
    return new Promise((resolve, reject) => {
      // Build WS URL from centralized config
      const wsUrl = `${config.wsUrl}/ws/${jobId}`;
      // Connecting to WebSocket for real-time updates
      
      const ws = new WebSocket(wsUrl);
      let timeoutId: NodeJS.Timeout;
      
      // FIX #8: Use Map-based flag to prevent multiple cleanup calls
      this.wsCleanupFlags.set(jobId, false);

      // Cleanup function to prevent memory leaks
      const cleanup = () => {
        // FIX #8: Check flag from Map instead of local variable
        if (this.wsCleanupFlags.get(jobId)) return;
        this.wsCleanupFlags.set(jobId, true);
        
        clearTimeout(timeoutId);
        if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
          ws.close();
        }
        
        // FIX #8: Clean up flag after a delay to prevent immediate reuse
        setTimeout(() => {
          this.wsCleanupFlags.delete(jobId);
        }, 1000);
      };

      // Set a timeout for the connection (60 seconds for better large file handling)
      timeoutId = setTimeout(() => {
        cleanup();
        reject(new Error('WebSocket connection timeout'));
      }, 60000); // 60 second timeout for large files

      ws.onopen = () => {
        // WebSocket connected successfully
        clearTimeout(timeoutId);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          // Process WebSocket progress update

          if (data.status === 'completed') {
            cleanup();
            resolve(data.result || data);
          } else if (data.status === 'error') {
            cleanup();
            reject(new Error(data.error || 'Processing failed'));
          } else {
            // Progress update
            const extra: any = {
              duplicate_info: data.duplicate_info,
              near_duplicate_info: data.near_duplicate_info,
              content_duplicate_info: data.content_duplicate_info,
              delta_analysis: data.delta_analysis,
              requires_user_decision: data.requires_user_decision,
            };
            this.updateProgress(
              data.step || 'processing',
              data.message || 'Processing...',
              data.progress || 0,
              data.sheetProgress,
              extra
            );
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onerror = (error) => {
        // MISMATCH FIX #3: Use unified error handler
        UnifiedErrorHandler.handle({
          message: 'WebSocket connection failed - using polling fallback',
          severity: ErrorSeverity.MEDIUM,
          source: ErrorSource.WEBSOCKET,
          retryable: true
        });
        cleanup();
        reject(new Error('WebSocket connection failed - will use polling fallback'));
      };

      ws.onclose = (event) => {
        // CRITICAL FIX: Implement reconnection logic before falling back to polling
        cleanup();
        // FIX #8: Check cleanup flag from Map instead of local variable
        if (event.code !== 1000 && event.reason !== 'Processing completed' && !this.wsCleanupFlags.get(jobId)) {
          // Try to reconnect before giving up
          this.attemptWebSocketReconnection(jobId, resolve, reject, 0);
        }
      };
    });
  }

  private async attemptWebSocketReconnection(
    jobId: string, 
    resolve: (value: any) => void, 
    reject: (reason?: any) => void,
    attemptNumber: number
  ): Promise<void> {
    /**
     * CRITICAL FIX: Exponential backoff reconnection strategy.
     * Prevents permanent downgrade to polling after temporary network issues.
     * 
     * Reconnection attempts: 1s, 2s, 4s, 8s, 16s (max 5 attempts)
     * Only falls back to polling after all attempts fail.
     */
    const MAX_RECONNECT_ATTEMPTS = config.websocket.reconnectAttempts;
    const BASE_DELAY = config.websocket.reconnectBaseDelay;
    
    if (attemptNumber >= MAX_RECONNECT_ATTEMPTS) {
      console.warn(`WebSocket reconnection failed after ${MAX_RECONNECT_ATTEMPTS} attempts - falling back to polling`);
      reject(new Error('WebSocket connection closed unexpectedly - will use polling fallback'));
      return;
    }
    
    // Calculate exponential backoff delay
    const delay = BASE_DELAY * Math.pow(2, attemptNumber);
    console.log(`Attempting WebSocket reconnection ${attemptNumber + 1}/${MAX_RECONNECT_ATTEMPTS} in ${delay}ms...`);
    
    await new Promise(resolve => setTimeout(resolve, delay));
    
    try {
      // Attempt to reconnect
      const result = await this.connectWebSocket(jobId);
      resolve(result);
    } catch (error) {
      // Reconnection failed, try again
      console.warn(`WebSocket reconnection attempt ${attemptNumber + 1} failed:`, error);
      this.attemptWebSocketReconnection(jobId, resolve, reject, attemptNumber + 1);
    }
  }

  private async pollForResults(jobId: string, initialResponse: any): Promise<any> {
    // FIX ISSUE #15: Reduce timeout from 5 minutes to 2 minutes for better UX
    const maxAttempts = 80; // 2 minutes max (1.5 seconds * 80 = 120 seconds)
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
          
          // FIX ISSUE #15: Show progress indication during polling
          if (statusData.progress !== undefined) {
            const timeElapsed = Math.floor((attempts * config.websocket.pollingInterval) / 1000);
            const progressMessage = statusData.message || `Processing... (${timeElapsed}s elapsed)`;
            this.updateProgress('processing', progressMessage, statusData.progress);
          }
        }
        
        // CRITICAL FIX: Use configurable polling interval from config
        await new Promise(resolve => setTimeout(resolve, config.websocket.pollingInterval));
        attempts++;
        
      } catch (error) {
        console.error('Polling error:', error);
        attempts++;
        await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds on error
      }
    }
    
    // FIX ISSUE #15: Better timeout message with suggestion
    throw new Error('Processing is taking longer than expected. The job may still be running in the background. Please refresh the page or try again later.');
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
      if (onJobId) {
        onJobId(jobData.id, fileHash);
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

        // Connect to WebSocket for real-time progress updates
        this.updateProgress('websocket', 'Connecting to real-time updates...', 35);
        
        let backendResult: any;
        try {
          // Try WebSocket connection with timeout (60 seconds)
          backendResult = await Promise.race([
            this.connectWebSocket(jobData.id),
            new Promise((_, reject) => 
              setTimeout(() => reject(new Error('WebSocket timeout')), 60000) // 60 second timeout for large files
            )
          ]);
          // WebSocket processing completed successfully
        } catch (websocketError) {
          // WebSocket connection failed, using polling fallback
          
          // Fallback: Poll for results instead of WebSocket
          this.updateProgress('polling', 'Using polling fallback for updates...', 40);
          backendResult = await this.pollForResults(jobData.id, initialResponse);
        }
        
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

  private extractMetricsFromStats(stats: any, recordId: string, userId: string): any[] {
    const metrics: any[] = [];
    
    if (!stats || typeof stats !== 'object') return metrics;
    
    // Extract metrics from summary statistics
    Object.entries(stats).forEach(([key, value]) => {
      if (typeof value === 'number' && value > 0) {
        metrics.push({
          category: this.getCategoryFromStatKey(key),
          subcategory: key,
          amount: value,
          date_recorded: new Date().toISOString().split('T')[0],
          metric_type: 'backend_extracted',
          user_id: userId,
          record_id: recordId
        });
      }
    });
    
    return metrics;
  }

  private getCategoryFromStatKey(key: string): string {
    const lowerKey = key.toLowerCase();
    if (lowerKey.includes('revenue') || lowerKey.includes('income') || lowerKey.includes('sales')) return 'revenue';
    if (lowerKey.includes('expense') || lowerKey.includes('cost') || lowerKey.includes('expense')) return 'expense';
    if (lowerKey.includes('profit') || lowerKey.includes('net')) return 'profit';
    if (lowerKey.includes('asset') || lowerKey.includes('cash')) return 'asset';
    if (lowerKey.includes('liability') || lowerKey.includes('debt')) return 'liability';
    return 'other';
  }

  private createRawEventsFromBackend(
    processedEvents: any[],
    jobId: string,
    userId: string,
    fileName: string,
    fileId: string
  ): any[] {
    const rawEvents: any[] = [];

    processedEvents.forEach((event, index) => {
      rawEvents.push({
        user_id: userId,
        file_id: fileId,
        job_id: jobId,
        provider: 'fastapi_backend',
        kind: event.kind || 'financial_transaction',
        source_platform: event.source_platform || 'unknown',
        payload: event.payload || event,
        row_index: event.row_index || index,
        sheet_name: event.sheet_name || 'Unknown',
        source_filename: fileName,
        status: 'processed',
        confidence_score: event.confidence_score || 0.8,
        classification_metadata: event.classification_metadata || {},
        category: event.category || 'financial_data',
        subcategory: event.subcategory || 'transaction',
        entities: event.entities || {},
        relationships: event.relationships || {}
      });
    });

    return rawEvents;
  }

  private createRawEventsFromLocal(
    sheets: SheetMetadata[],
    jobId: string,
    userId: string,
    fileName: string,
    fileId: string
  ): any[] {
    const rawEvents: any[] = [];

    sheets.forEach(sheet => {
      if (sheet.preview && sheet.preview.length > 1) {
        const headers = sheet.preview[0] as string[];
        const dataRows = sheet.preview.slice(1);

        dataRows.forEach((row: any[], rowIndex) => {
          // Only create events for rows with meaningful data
          if (row.some(cell => cell !== null && cell !== undefined && cell !== '')) {
            const rowObject: any = {};
            headers.forEach((header, colIndex) => {
              if (header && row[colIndex] !== undefined) {
                rowObject[header] = row[colIndex];
              }
            });

            rawEvents.push({
              user_id: userId,
              file_id: fileId,
              job_id: jobId,
              provider: 'local_processing',
              kind: 'financial_row',
              source_platform: 'excel_upload',
              payload: rowObject,
              row_index: rowIndex + 1,
              sheet_name: sheet.name,
              source_filename: fileName,
              status: 'processed',
              confidence_score: sheet.confidenceScore,
              classification_metadata: {
                sheet_type: sheet.type,
                detected_period: sheet.detectedPeriod,
                has_numbers: sheet.hasNumbers
              },
              category: this.getRowCategory(rowObject, headers),
              subcategory: sheet.type,
              entities: this.extractRowEntities(rowObject, headers),
              relationships: {}
            });
          }
        });
      }
    });

    return rawEvents;
  }

  private getRowCategory(rowData: any, headers: string[]): string {
    const headerStr = headers.join(' ').toLowerCase();
    const dataStr = Object.values(rowData).join(' ').toLowerCase();
    
    if (headerStr.includes('revenue') || headerStr.includes('income') || dataStr.includes('revenue')) return 'revenue';
    if (headerStr.includes('expense') || headerStr.includes('cost') || dataStr.includes('expense')) return 'expense';
    if (headerStr.includes('asset') || dataStr.includes('asset')) return 'asset';
    if (headerStr.includes('liability') || dataStr.includes('liability')) return 'liability';
    if (headerStr.includes('cash') || headerStr.includes('payment')) return 'cash_flow';
    return 'financial_data';
  }

  private extractRowEntities(rowData: any, headers: string[]): any {
    const entities: any = {};
    
    // Look for common entity patterns
    Object.entries(rowData).forEach(([key, value]) => {
      const keyLower = key.toLowerCase();
      const valueStr = String(value);
      
      if (keyLower.includes('vendor') || keyLower.includes('supplier')) {
        entities.vendors = entities.vendors || [];
        if (valueStr && valueStr.trim()) entities.vendors.push(valueStr.trim());
      }
      if (keyLower.includes('customer') || keyLower.includes('client')) {
        entities.customers = entities.customers || [];
        if (valueStr && valueStr.trim()) entities.customers.push(valueStr.trim());
      }
      if (keyLower.includes('employee') || keyLower.includes('staff')) {
        entities.employees = entities.employees || [];
        if (valueStr && valueStr.trim()) entities.employees.push(valueStr.trim());
      }
      if (keyLower.includes('project') || keyLower.includes('job')) {
        entities.projects = entities.projects || [];
        if (valueStr && valueStr.trim()) entities.projects.push(valueStr.trim());
      }
    });

    return entities;
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
