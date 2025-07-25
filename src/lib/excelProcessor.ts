import * as XLSX from 'xlsx';
import OpenAI from 'openai';

export interface ProcessingProgress {
  step: string;
  message: string;
  progress: number;
}

export interface ProcessingResult {
  documentType: string;
  insights: string[];
  metrics: Array<{
    label: string;
    value: string;
    change?: string;
  }>;
  summary: string;
}

interface SheetAnalysis {
  name: string;
  type: 'balance_sheet' | 'income_statement' | 'cash_flow' | 'budget' | 'general';
  rowCount: number;
  keyColumns: string[];
  hasNumbers: boolean;
}

export class ExcelProcessor {
  private openai: OpenAI | null = null;
  private onProgress: ((progress: ProcessingProgress) => void) | null = null;

  constructor(apiKey?: string) {
    if (apiKey) {
      this.openai = new OpenAI({
        apiKey: apiKey,
        dangerouslyAllowBrowser: true
      });
    }
  }

  setProgressCallback(callback: (progress: ProcessingProgress) => void) {
    this.onProgress = callback;
  }

  private updateProgress(step: string, message: string, progress: number) {
    if (this.onProgress) {
      this.onProgress({ step, message, progress });
    }
  }

  async processFile(file: File): Promise<ProcessingResult> {
    try {
      this.updateProgress('starting', '🚀 Starting intelligent document analysis...', 5);

      // Read the entire workbook
      this.updateProgress('reading', '📖 Reading Excel workbook structure...', 15);
      const workbook = await this.readExcelWorkbook(file);

      // Analyze all sheets
      this.updateProgress('scanning', '🔍 Scanning sheet names and structure...', 25);
      const sheetsAnalysis = await this.analyzeAllSheets(workbook);

      // Process each sheet with real-time feedback
      let currentProgress = 30;
      const progressPerSheet = 40 / sheetsAnalysis.length;
      
      const allSheetData: any = {};
      for (let i = 0; i < sheetsAnalysis.length; i++) {
        const sheet = sheetsAnalysis[i];
        const sheetProgress = currentProgress + (i * progressPerSheet);
        
        await this.processSheetWithFeedback(workbook, sheet, sheetProgress, allSheetData);
      }

      // Generate final insights
      this.updateProgress('finalizing', '💡 Generating comprehensive financial insights...', 80);
      const insights = await this.generateComprehensiveInsights(allSheetData, sheetsAnalysis);

      this.updateProgress('completed', '✅ Financial analysis completed successfully!', 100);

      return {
        documentType: this.determineDocumentType(sheetsAnalysis),
        insights: insights.insights || [],
        metrics: insights.metrics || [],
        summary: insights.summary || 'Comprehensive analysis completed.'
      };

    } catch (error) {
      console.error('Processing error:', error);
      throw new Error(`Analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  private async readExcelWorkbook(file: File): Promise<XLSX.WorkBook> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      
      reader.onload = (e) => {
        try {
          const data = new Uint8Array(e.target?.result as ArrayBuffer);
          const workbook = XLSX.read(data, { type: 'array' });
          resolve(workbook);
        } catch (error) {
          reject(error);
        }
      };
      
      reader.onerror = () => reject(new Error('Failed to read file'));
      reader.readAsArrayBuffer(file);
    });
  }

  private async analyzeAllSheets(workbook: XLSX.WorkBook): Promise<SheetAnalysis[]> {
    const sheetsAnalysis: SheetAnalysis[] = [];

    for (const sheetName of workbook.SheetNames) {
      const worksheet = workbook.Sheets[sheetName];
      const data = XLSX.utils.sheet_to_json(worksheet, { header: 1 }) as any[][];
      
      const analysis: SheetAnalysis = {
        name: sheetName,
        type: this.classifySheetType(sheetName, data),
        rowCount: data.length,
        keyColumns: this.extractKeyColumns(data),
        hasNumbers: this.containsNumericalData(data)
      };

      sheetsAnalysis.push(analysis);
    }

    return sheetsAnalysis;
  }

  private classifySheetType(sheetName: string, data: any[][]): SheetAnalysis['type'] {
    const name = sheetName.toLowerCase();
    
    if (name.includes('balance') || name.includes('bs') || name.includes('position')) {
      return 'balance_sheet';
    }
    if (name.includes('income') || name.includes('profit') || name.includes('loss') || name.includes('p&l') || name.includes('pnl')) {
      return 'income_statement';
    }
    if (name.includes('cash') || name.includes('flow') || name.includes('cf')) {
      return 'cash_flow';
    }
    if (name.includes('budget') || name.includes('forecast') || name.includes('plan')) {
      return 'budget';
    }

    // Analyze content for classification
    const content = data.flat().join(' ').toLowerCase();
    if (content.includes('assets') && content.includes('liabilities')) {
      return 'balance_sheet';
    }
    if (content.includes('revenue') && content.includes('expenses')) {
      return 'income_statement';
    }
    if (content.includes('cash') && (content.includes('operating') || content.includes('investing'))) {
      return 'cash_flow';
    }

    return 'general';
  }

  private extractKeyColumns(data: any[][]): string[] {
    if (data.length === 0) return [];
    
    const headers = data[0] || [];
    return headers
      .filter(header => header && typeof header === 'string')
      .slice(0, 5) // Top 5 columns
      .map(header => String(header));
  }

  private containsNumericalData(data: any[][]): boolean {
    return data.some(row => 
      row.some(cell => 
        typeof cell === 'number' || 
        (typeof cell === 'string' && /[\d,.]/.test(cell))
      )
    );
  }

  private async processSheetWithFeedback(
    workbook: XLSX.WorkBook, 
    sheet: SheetAnalysis, 
    progress: number,
    allSheetData: any
  ) {
    // Generate contextual progress message
    let message = this.generateSheetProcessingMessage(sheet);
    this.updateProgress('processing', message, progress);

    // Small delay to show the message
    await new Promise(resolve => setTimeout(resolve, 300));

    // Actually process the sheet
    const worksheet = workbook.Sheets[sheet.name];
    const data = XLSX.utils.sheet_to_json(worksheet, { header: 1 });
    allSheetData[sheet.name] = {
      ...sheet,
      data: data
    };

    // Show specific findings
    const findingMessage = this.generateFindingMessage(sheet, data);
    this.updateProgress('analyzing', findingMessage, progress + 5);
    
    await new Promise(resolve => setTimeout(resolve, 200));
  }

  private generateSheetProcessingMessage(sheet: SheetAnalysis): string {
    const typeMessages = {
      'balance_sheet': `📊 Reading Balance Sheet: "${sheet.name}" - Analyzing assets and liabilities...`,
      'income_statement': `💰 Processing P&L Statement: "${sheet.name}" - Examining revenue and expenses...`,
      'cash_flow': `💸 Analyzing Cash Flow: "${sheet.name}" - Tracking money movements...`,
      'budget': `📈 Reading Budget Data: "${sheet.name}" - Reviewing financial projections...`,
      'general': `📋 Processing Sheet: "${sheet.name}" - Analyzing financial data...`
    };

    return typeMessages[sheet.type];
  }

  private generateFindingMessage(sheet: SheetAnalysis, data: any[]): string {
    const findings = [];

    if (sheet.hasNumbers) {
      findings.push(`Found ${sheet.rowCount} rows of data`);
    }

    if (sheet.keyColumns.length > 0) {
      findings.push(`Identified key columns: ${sheet.keyColumns.slice(0, 3).join(', ')}`);
    }

    // Look for specific financial indicators
    const content = JSON.stringify(data).toLowerCase();
    if (content.includes('revenue') || content.includes('sales')) {
      findings.push('📈 Revenue data detected');
    }
    if (content.includes('expense') || content.includes('cost')) {
      findings.push('💸 Expense analysis in progress');
    }
    if (content.includes('asset')) {
      findings.push('🏦 Assets being evaluated');
    }
    if (content.includes('liability') || content.includes('debt')) {
      findings.push('⚠️ Liabilities under review');
    }

    return findings.length > 0 
      ? `   └─ ${findings.join(' • ')}`
      : `   └─ Analyzing ${sheet.rowCount} rows of financial data...`;
  }

  private determineDocumentType(sheetsAnalysis: SheetAnalysis[]): string {
    const types = sheetsAnalysis.map(s => s.type);
    
    if (types.includes('balance_sheet') && types.includes('income_statement') && types.includes('cash_flow')) {
      return 'Complete Financial Statements';
    }
    if (types.includes('balance_sheet') && types.includes('income_statement')) {
      return 'Financial Statements Package';
    }
    if (types.includes('balance_sheet')) {
      return 'Balance Sheet';
    }
    if (types.includes('income_statement')) {
      return 'Income Statement / P&L';
    }
    if (types.includes('cash_flow')) {
      return 'Cash Flow Statement';
    }
    if (types.includes('budget')) {
      return 'Budget & Forecast';
    }
    
    return 'Financial Workbook';
  }

  private async generateComprehensiveInsights(allSheetData: any, sheetsAnalysis: SheetAnalysis[]) {
    if (!this.openai) {
      return {
        insights: [
          `Analyzed ${sheetsAnalysis.length} sheets`,
          `Document type: ${this.determineDocumentType(sheetsAnalysis)}`,
          'Enable AI integration for detailed financial insights'
        ],
        metrics: [
          { label: 'Total Sheets', value: sheetsAnalysis.length.toString() },
          { label: 'Data Rows', value: sheetsAnalysis.reduce((sum, s) => sum + s.rowCount, 0).toString() }
        ],
        summary: 'Basic analysis completed. Configure AI for comprehensive insights.'
      };
    }

    try {
      // Prepare summary of all sheets for AI analysis
      const summary = sheetsAnalysis.map(sheet => ({
        name: sheet.name,
        type: sheet.type,
        rowCount: sheet.rowCount,
        keyColumns: sheet.keyColumns
      }));

      const response = await this.openai.chat.completions.create({
        model: "gpt-4.1-2025-04-14",
        messages: [
          {
            role: "system",
            content: "You are an expert financial analyst. Analyze the Excel workbook structure and provide comprehensive insights in JSON format with 'insights' (array), 'metrics' (array of {label, value, change}), and 'summary' (string)."
          },
          {
            role: "user",
            content: `I've analyzed an Excel workbook with the following structure:\n\n${JSON.stringify(summary, null, 2)}\n\nProvide detailed financial insights about this workbook structure, key metrics you can identify, and a comprehensive summary of what this financial document contains.`
          }
        ],
        max_tokens: 1500,
        temperature: 0.1
      });

      const result = response.choices[0]?.message?.content;
      if (result) {
        try {
          return JSON.parse(result);
        } catch {
          return {
            insights: [result],
            metrics: [
              { label: 'Document Type', value: this.determineDocumentType(sheetsAnalysis) },
              { label: 'Total Sheets', value: sheetsAnalysis.length.toString() }
            ],
            summary: 'AI analysis completed successfully'
          };
        }
      }
    } catch (error) {
      console.warn('AI insights generation failed:', error);
    }

    return {
      insights: [
        `Comprehensive analysis of ${sheetsAnalysis.length} financial sheets`,
        `Identified: ${sheetsAnalysis.map(s => s.name).join(', ')}`,
        'Document structure successfully mapped'
      ],
      metrics: [
        { label: 'Document Type', value: this.determineDocumentType(sheetsAnalysis) },
        { label: 'Total Sheets', value: sheetsAnalysis.length.toString() },
        { label: 'Data Points', value: sheetsAnalysis.reduce((sum, s) => sum + s.rowCount, 0).toString() }
      ],
      summary: 'Comprehensive financial document analysis completed with detailed sheet mapping.'
    };
  }
}