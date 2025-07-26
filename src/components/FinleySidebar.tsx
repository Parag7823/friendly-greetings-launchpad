import { useState } from 'react';
import { FileSpreadsheet, ChevronDown, ChevronRight, Settings } from 'lucide-react';
import { EnhancedExcelUpload } from './EnhancedExcelUpload';
export const FinleySidebar = () => {
  const [isIntegrationsOpen, setIsIntegrationsOpen] = useState(false);
  return <div className="finley-sidebar flex flex-col h-full p-6 overflow-y-auto">
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
        <div className="space-y-2">
          {/* Collapsible Integrations Menu */}
          <button onClick={() => setIsIntegrationsOpen(!isIntegrationsOpen)} className="w-full flex items-center justify-between p-3 text-left hover:bg-muted/50 rounded-lg transition-colors">
            <div className="flex items-center gap-2">
              <Settings className="w-5 h-5 text-muted-foreground" />
              <span className="font-medium text-foreground">Integrations</span>
            </div>
            {isIntegrationsOpen ? <ChevronDown className="w-4 h-4 text-muted-foreground" /> : <ChevronRight className="w-4 h-4 text-muted-foreground" />}
          </button>
          
          {/* Excel Integration - Only show when open */}
          {isIntegrationsOpen && <div className="ml-4 pl-4 border-l border-border">
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
                <EnhancedExcelUpload />
                
                {/* Inspirational Tagline */}
                <p className="text-xs text-muted-foreground mt-3 italic leading-relaxed">
                  "Excel is where data begins — Finley turns it into intelligence."
                </p>
              </div>
            </div>}
        </div>
      </div>
      
      {/* Footer */}
      <div className="mt-auto pt-6 border-t border-border">
        
      </div>
    </div>;
};