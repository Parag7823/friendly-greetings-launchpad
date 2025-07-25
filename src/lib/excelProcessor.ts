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
      this.updateProgress('starting', '🚀 Starting intelligent analysis...', 10);

      // Read the file
      this.updateProgress('reading', '📖 Reading and parsing your document...', 20);
      const data = await this.readExcelFile(file);

      // Analyze document structure
      this.updateProgress('analyzing', '🧠 Analyzing document structure with AI...', 40);
      const documentAnalysis = await this.analyzeDocument(data, file.name);

      // Generate insights
      this.updateProgress('insights', '💡 Generating intelligent financial insights...', 70);
      const insights = await this.generateInsights(data, documentAnalysis);

      this.updateProgress('completed', '✅ Analysis completed successfully!', 100);

      return {
        documentType: documentAnalysis.type || 'Financial Document',
        insights: insights.insights || [],
        metrics: insights.metrics || [],
        summary: insights.summary || 'Analysis completed successfully.'
      };

    } catch (error) {
      console.error('Processing error:', error);
      throw new Error(`Processing failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  private async readExcelFile(file: File): Promise<any[][]> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      
      reader.onload = (e) => {
        try {
          const data = new Uint8Array(e.target?.result as ArrayBuffer);
          const workbook = XLSX.read(data, { type: 'array' });
          
          // Get the first worksheet
          const firstSheetName = workbook.SheetNames[0];
          const worksheet = workbook.Sheets[firstSheetName];
          
          // Convert to array of arrays
          const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });
          resolve(jsonData as any[][]);
        } catch (error) {
          reject(error);
        }
      };
      
      reader.onerror = () => reject(new Error('Failed to read file'));
      reader.readAsArrayBuffer(file);
    });
  }

  private async analyzeDocument(data: any[][], filename: string) {
    if (!this.openai) {
      // Fallback analysis without AI
      return {
        type: 'Financial Document',
        structure: 'Standard spreadsheet format'
      };
    }

    try {
      // Get first 10 rows for analysis
      const sampleData = data.slice(0, 10).map(row => row.slice(0, 10));
      
      const response = await this.openai.chat.completions.create({
        model: "gpt-3.5-turbo",
        messages: [
          {
            role: "system",
            content: "You are a financial analyst. Analyze the spreadsheet data and identify the document type."
          },
          {
            role: "user",
            content: `Analyze this spreadsheet data from file "${filename}":\n\n${JSON.stringify(sampleData, null, 2)}\n\nWhat type of financial document is this? Return a JSON with "type" and "structure" fields.`
          }
        ],
        max_tokens: 500,
        temperature: 0.1
      });

      const result = response.choices[0]?.message?.content;
      if (result) {
        try {
          return JSON.parse(result);
        } catch {
          return { type: result, structure: 'AI analyzed' };
        }
      }
    } catch (error) {
      console.warn('AI analysis failed, using fallback:', error);
    }

    return {
      type: 'Financial Document',
      structure: 'Standard spreadsheet format'
    };
  }

  private async generateInsights(data: any[][], documentAnalysis: any) {
    if (!this.openai) {
      // Fallback insights without AI
      return {
        insights: [
          'Document contains financial data',
          `Spreadsheet has ${data.length} rows and ${data[0]?.length || 0} columns`,
          'Further analysis requires AI integration'
        ],
        metrics: [
          { label: 'Total Rows', value: data.length.toString() },
          { label: 'Total Columns', value: (data[0]?.length || 0).toString() }
        ],
        summary: 'Basic analysis completed. Enable AI for detailed insights.'
      };
    }

    try {
      // Get sample of data for analysis
      const sampleData = data.slice(0, 20);
      
      const response = await this.openai.chat.completions.create({
        model: "gpt-3.5-turbo",
        messages: [
          {
            role: "system",
            content: "You are a financial analyst. Provide insights about the financial data in JSON format with 'insights' (array), 'metrics' (array of {label, value, change}), and 'summary' (string)."
          },
          {
            role: "user",
            content: `Analyze this ${documentAnalysis.type} data:\n\n${JSON.stringify(sampleData, null, 2)}\n\nProvide financial insights, key metrics, and a summary.`
          }
        ],
        max_tokens: 1000,
        temperature: 0.1
      });

      const result = response.choices[0]?.message?.content;
      if (result) {
        try {
          return JSON.parse(result);
        } catch {
          return {
            insights: [result],
            metrics: [],
            summary: 'AI analysis completed'
          };
        }
      }
    } catch (error) {
      console.warn('AI insights generation failed:', error);
    }

    return {
      insights: ['Analysis completed with limited AI access'],
      metrics: [
        { label: 'Data Points', value: (data.length * (data[0]?.length || 0)).toString() }
      ],
      summary: 'Analysis completed. For detailed insights, ensure proper AI configuration.'
    };
  }
}