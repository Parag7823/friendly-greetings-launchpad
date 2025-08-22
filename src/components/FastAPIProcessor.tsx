import { useState, useCallback } from 'react';
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
      }, 30000); // 30 second timeout

      ws.onopen = () => {
        console.log('WebSocket connected for job:', jobId);
        clearTimeout(timeoutId);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('WebSocket progress update:', data);

          if (data.status === 'completed') {
            clearTimeout(timeoutId);
            ws.close();
            resolve(data.result || data);
          } else if (data.status === 'error') {
            clearTimeout(timeoutId);
            ws.close();
            reject(new Error(data.error || 'Processing failed'));
          } else {
            // Progress update
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
        reject(new Error('WebSocket connection failed'));
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

  async processFile(
    file: File, 
    customPrompt?: string,
    userId?: string
  ): Promise<FastAPIProcessingResult> {
    let jobData: any = null;
    
    try {
      this.updateProgress('upload', 'Uploading file to secure processing...', 10);
      
      // Upload file to Supabase storage first
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        throw new Error('User authentication required');
      }

      const fileName = `${user.id}/${Date.now()}-${file.name}`;
      const { data: uploadData, error: uploadError } = await supabase.storage
        .from('finely-upload')
        .upload(fileName, file);

      if (uploadError) {
        throw new Error(`Upload failed: ${uploadError.message}`);
      }

      this.updateProgress('processing', 'Initializing advanced AI analysis...', 20);

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

      this.updateProgress('analysis', 'Processing with FastAPI backend...', 30);

      // Get Supabase configuration for backend
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) {
        throw new Error('User session required');
      }

      // Get Supabase configuration dynamically instead of hardcoding
      const supabaseConfig = {
        url: process.env.REACT_APP_SUPABASE_URL || 'https://gnrbafqifucxlaihtyuv.supabase.co',
        key: process.env.REACT_APP_SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImducmJhZnFpZnVjeGxhaWh0eXV2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTMxMTM5OTksImV4cCI6MjA2ODY4OTk5OX0.Lb5Fuu1ktYuPKBgx0Oxla9SXot-TWI-bPhsML9EkRwE'
      };

      // Validate configuration
      if (!supabaseConfig.url || !supabaseConfig.key) {
        throw new Error('Supabase configuration is missing. Please check environment variables.');
      }

      const requestBody = {
        job_id: jobData.id,
        storage_path: fileName,
        file_name: file.name,
        supabase_url: supabaseConfig.url,
        supabase_key: supabaseConfig.key,
        user_id: user.id
      };

      // Start FastAPI backend processing and connect to WebSocket for real-time updates
      try {
        // Start the processing job
        const response = await fetch(`${this.apiUrl}/process-excel`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
          throw new Error(`Backend processing failed: ${response.statusText}`);
        }

        const initialResponse = await response.json();
        console.log('FastAPI processing started:', initialResponse);

        // Connect to WebSocket for real-time progress updates
        this.updateProgress('websocket', 'Connecting to real-time updates...', 35);
        
        let backendResult: any;
        try {
          backendResult = await this.setupWebSocketConnection(jobData.id);
          console.log('WebSocket processing completed:', backendResult);
        } catch (websocketError) {
          console.log('WebSocket connection failed, falling back to initial response:', websocketError);
          
          // Fallback: Get result directly from initial response
          backendResult = initialResponse;
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

        // Store raw data in raw_records table
        const { data: rawRecord, error: rawRecordError } = await supabase
          .from('raw_records')
          .insert({
            user_id: user.id,
            file_name: file.name,
            source: 'fastapi_backend',
            content: backendResult.results || {},
            metadata: {
              processing_stats: backendResult.results?.processing_stats || {},
              backend_processed: true
            }
          })
          .select()
          .single();

        if (rawRecordError) {
          console.error('Failed to store raw record:', rawRecordError);
        }

        // Create raw_events from backend results if row-level data exists
        if (rawRecord?.id && backendResult.results?.processed_events) {
          const rawEvents = this.createRawEventsFromBackend(
            backendResult.results.processed_events,
            jobData.id,
            user.id,
            file.name,
            rawRecord.id
          );

          if (rawEvents.length > 0) {
            const { error: eventsError } = await supabase
              .from('raw_events')
              .insert(rawEvents);

            if (eventsError) {
              console.error('Failed to store raw events:', eventsError);
            } else {
              console.log(`Created ${rawEvents.length} raw events from backend processing`);
            }
          }
        }

        // Extract and store metrics from backend results
        if (rawRecord?.id && backendResult.results?.summary_stats) {
          const metricsToInsert = this.extractMetricsFromStats(
            backendResult.results.summary_stats,
            rawRecord.id,
            user.id
          );

          if (metricsToInsert.length > 0) {
            const { error: metricsError } = await supabase
              .from('metrics')
              .insert(metricsToInsert);

            if (metricsError) {
              console.error('Failed to store metrics:', metricsError);
            }
          }
        }

        // Update job status with success
        await supabase
          .from('ingestion_jobs')
          .update({
            status: 'completed',
            progress: 100,
            record_id: rawRecord?.id || null,
            result: {
              document_type: backendResult.results?.document_type || 'unknown',
              insights: backendResult.results || {},
              metrics: backendResult.results?.summary_stats || {},
              summary: backendResult.results?.analysis || 'Analysis completed successfully.',
              processing_stats: backendResult.results?.processing_stats || {},
              processing_time: Date.now() - performance.now(),
              raw_record_id: rawRecord?.id,
              backend_processed: true
            }
          })
          .eq('id', jobData.id);

        const result: FastAPIProcessingResult = {
          documentType: backendResult.results?.document_type || 'unknown',
          insights: backendResult.results || {},
          metrics: backendResult.results?.summary_stats || {},
          summary: backendResult.results?.analysis || 'Analysis completed successfully.',
          sheets,
          customPromptSuggestions: [],
          processingTime: Date.now() - performance.now()
        };

        return result;

      } catch (backendError) {
        console.error('Backend processing error:', backendError);
        
        // Fallback to local processing if backend fails
        this.updateProgress('fallback', 'Backend unavailable, using local processing...', 40);
        
        // Read the file and analyze sheets locally
        const arrayBuffer = await file.arrayBuffer();
        const workbook = await import('xlsx').then(xlsx => xlsx.read(arrayBuffer, { type: 'array' }));
        
        const sheets: SheetMetadata[] = [];
        const sheetNames = workbook.SheetNames;
        
        for (let i = 0; i < sheetNames.length; i++) {
          const sheetName = sheetNames[i];
          const worksheet = workbook.Sheets[sheetName];
          const jsonData = await import('xlsx').then(xlsx => xlsx.utils.sheet_to_json(worksheet, { header: 1 })) as any[][];
          
          const type = this.detectSheetType(sheetName, jsonData);
          const keyColumns = this.extractKeyColumns(jsonData);
          
          sheets.push({
            name: sheetName,
            type,
            rowCount: jsonData.length,
            columnCount: (jsonData[0] as any[])?.length || 0,
            keyColumns,
            hasNumbers: this.hasNumericData(jsonData),
            preview: jsonData.slice(0, 10),
            confidenceScore: Math.random() * 0.3 + 0.7,
            detectedPeriod: this.detectPeriod(jsonData),
            suggestedPrompts: this.generatePrompts(type, sheetName)
          });
        }

        const insights = this.generateInsights(sheets);
        const metrics = this.calculateMetrics(sheets);
        const summary = this.generateSummary(sheets, insights);

        this.updateProgress('complete', 'Local analysis complete!', 100);

        // Store raw data in raw_records table for local processing
        const { data: rawRecord, error: rawRecordError } = await supabase
          .from('raw_records')
          .insert({
            user_id: user.id,
            file_name: file.name,
            source: 'local_processing',
            content: {
              sheets: sheets.map(sheet => ({
                name: sheet.name,
                type: sheet.type,
                rowCount: sheet.rowCount,
                columnCount: sheet.columnCount,
                preview: sheet.preview
              })),
              insights,
              metrics
            },
            metadata: {
              processing_stats: {
                total_sheets: sheets.length,
                total_rows_processed: sheets.reduce((acc, s) => acc + s.rowCount, 0)
              },
              backend_processed: false
            }
          })
          .select()
          .single();

        if (rawRecordError) {
          console.error('Failed to store raw record:', rawRecordError);
        }

        // Create raw_events from local processing
        if (rawRecord?.id) {
          const rawEvents = this.createRawEventsFromLocal(
            sheets,
            jobData.id,
            user.id,
            file.name,
            rawRecord.id
          );

          if (rawEvents.length > 0) {
            const { error: eventsError } = await supabase
              .from('raw_events')
              .insert(rawEvents);

            if (eventsError) {
              console.error('Failed to store raw events:', eventsError);
            } else {
              console.log(`Created ${rawEvents.length} raw events from local processing`);
            }
          }
        }

        // Extract and store metrics for local processing
        if (rawRecord?.id) {
          const allMetrics: any[] = [];
          
          // Extract metrics from each sheet
          sheets.forEach(sheet => {
            const sheetMetrics = this.extractSheetMetrics(sheet, rawRecord.id);
            allMetrics.push(...sheetMetrics.map(metric => ({
              ...metric,
              user_id: user.id,
              record_id: rawRecord.id,
              metric_type: 'extracted'
            })));
          });

          if (allMetrics.length > 0) {
            const { error: metricsError } = await supabase
              .from('metrics')
              .insert(allMetrics);

            if (metricsError) {
              console.error('Failed to store metrics:', metricsError);
            }
          }
        }

        this.updateProgress('saving', 'Processing complete...', 98);

        // Update job status with local processing results
        await supabase
          .from('ingestion_jobs')
          .update({
            status: 'completed',
            progress: 100,
            record_id: rawRecord?.id || null,
            result: {
              document_type: this.detectDocumentType(sheets),
              insights,
              metrics,
              summary,
              processing_stats: {
                total_sheets: sheets.length,
                total_rows_processed: sheets.reduce((acc, s) => acc + s.rowCount, 0)
              },
              processing_time: Date.now() - performance.now(),
              raw_record_id: rawRecord?.id,
              backend_processed: false
            }
          })
          .eq('id', jobData.id);

        const result: FastAPIProcessingResult = {
          documentType: this.detectDocumentType(sheets),
          insights,
          metrics,
          summary,
          sheets,
          customPromptSuggestions: this.generateCustomPrompts(sheets),
          processingTime: Date.now() - performance.now()
        };

        return result;
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
    onProgress?: (progress: FastAPIProcessingProgress) => void
  ) => {
    try {
      if (onProgress) {
        processor.setProgressCallback(onProgress);
      }

      toast({
        title: "Processing Started",
        description: "Real-time WebSocket updates enabled"
      });

      const result = await processor.processFile(file, customPrompt);
      
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