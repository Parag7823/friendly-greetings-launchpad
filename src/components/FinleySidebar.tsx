import { FileSpreadsheet, Upload } from 'lucide-react';
import { ExcelUpload } from './ExcelUpload';

export const FinleySidebar = () => {
  return (
    <div className="finley-sidebar flex flex-col h-full p-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-foreground tracking-tight">
          Finley AI
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          Intelligent Financial Analyst
        </p>
      </div>
      
      {/* Integrations Section */}
      <div className="flex-1">
        <h2 className="text-lg font-semibold text-foreground mb-4">
          Integrations
        </h2>
        
        {/* Excel Integration Card */}
        <div className="finley-card p-4 hover-lift hover-glow">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-8 h-8 bg-green-600 rounded-lg flex items-center justify-center">
              <FileSpreadsheet className="w-4 h-4 text-white" />
            </div>
            <div>
              <h3 className="font-medium text-foreground">Excel</h3>
              <p className="text-xs text-muted-foreground">
                Upload spreadsheets
              </p>
            </div>
          </div>
          
          {/* Upload Component */}
          <ExcelUpload />
          
          {/* Inspirational Tagline */}
          <p className="text-xs text-muted-foreground mt-3 italic leading-relaxed">
            "Excel is where data begins — Finley turns it into intelligence."
          </p>
        </div>
      </div>
      
      {/* Footer */}
      <div className="mt-auto pt-6 border-t border-border">
        <p className="text-xs text-muted-foreground text-center">
          Step 1: Data Ingestion Layer
        </p>
      </div>
    </div>
  );
};