import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { Badge } from './ui/badge';
import { 
  Sparkles, 
  Send, 
  RefreshCw, 
  Lightbulb,
  Target,
  TrendingUp,
  AlertTriangle,
  FileText
} from 'lucide-react';

interface CustomPromptInterfaceProps {
  onSubmit: (prompt: string) => void;
  isProcessing?: boolean;
  documentType?: string;
  suggestedPrompts?: string[];
  detectedSheets?: string[];
}

const FINANCE_PROMPT_TEMPLATES = {
  analysis: [
    "Analyze the key financial ratios and trends in this document",
    "What are the main financial strengths and weaknesses shown?",
    "Identify any unusual patterns or anomalies in the data",
    "Compare performance across different time periods"
  ],
  insights: [
    "What strategic insights can you derive from this financial data?",
    "Identify the top 3 areas that need immediate attention",
    "What opportunities for cost optimization do you see?",
    "How does cash flow look and what are the main drivers?"
  ],
  forecasting: [
    "Based on this data, what trends do you predict for next quarter?",
    "What assumptions would you make for budgeting purposes?",
    "Identify seasonal patterns and their business implications",
    "What scenarios should we plan for based on this data?"
  ],
  compliance: [
    "Are there any red flags from a financial compliance perspective?",
    "What additional reporting requirements might apply?",
    "Review for potential audit considerations",
    "Assess internal control implications"
  ]
};

const DOCUMENT_SPECIFIC_PROMPTS = {
  balance_sheet: [
    "Analyze the debt-to-equity ratio and capital structure",
    "Review working capital components and liquidity position",
    "Assess asset utilization and efficiency",
    "Evaluate financial stability and solvency"
  ],
  income_statement: [
    "Break down revenue streams and growth patterns",
    "Analyze expense categories and cost structure",
    "Calculate profit margins and profitability trends",
    "Identify key performance drivers"
  ],
  cash_flow: [
    "Evaluate operating cash flow quality and sustainability",
    "Analyze capital expenditure patterns and investments",
    "Review financing activities and debt management",
    "Assess free cash flow generation"
  ],
  budget: [
    "Compare actual vs. budget variances",
    "Identify areas of over/under performance",
    "Analyze budget accuracy and forecasting quality",
    "Recommend budget adjustments or reforecasting"
  ]
};

export const CustomPromptInterface = ({
  onSubmit,
  isProcessing = false,
  documentType,
  suggestedPrompts = [],
  detectedSheets = []
}: CustomPromptInterfaceProps) => {
  const [customPrompt, setCustomPrompt] = useState('');
  const [activeCategory, setActiveCategory] = useState('analysis');
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Generate contextual prompts based on document type
  const getContextualPrompts = () => {
    const prompts = [...suggestedPrompts];
    
    if (documentType && DOCUMENT_SPECIFIC_PROMPTS[documentType as keyof typeof DOCUMENT_SPECIFIC_PROMPTS]) {
      prompts.push(...DOCUMENT_SPECIFIC_PROMPTS[documentType as keyof typeof DOCUMENT_SPECIFIC_PROMPTS]);
    }
    
    // Add sheet-specific prompts
    detectedSheets.forEach(sheet => {
      const sheetType = sheet.toLowerCase();
      if (sheetType.includes('balance')) {
        prompts.push(`Focus analysis on the ${sheet} sheet specifically`);
      } else if (sheetType.includes('income') || sheetType.includes('p&l')) {
        prompts.push(`Deep dive into revenue and expense trends in ${sheet}`);
      } else if (sheetType.includes('cash')) {
        prompts.push(`Analyze cash flow patterns in the ${sheet} sheet`);
      }
    });
    
    return Array.from(new Set(prompts)); // Remove duplicates
  };

  const handleSubmit = () => {
    if (customPrompt.trim()) {
      onSubmit(customPrompt.trim());
      setCustomPrompt('');
    }
  };

  const handleSuggestedPrompt = (prompt: string) => {
    setCustomPrompt(prompt);
  };

  const getPromptsByCategory = (category: string) => {
    return FINANCE_PROMPT_TEMPLATES[category as keyof typeof FINANCE_PROMPT_TEMPLATES] || [];
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'analysis':
        return <TrendingUp className="w-4 h-4" />;
      case 'insights':
        return <Lightbulb className="w-4 h-4" />;
      case 'forecasting':
        return <Target className="w-4 h-4" />;
      case 'compliance':
        return <AlertTriangle className="w-4 h-4" />;
      default:
        return <FileText className="w-4 h-4" />;
    }
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-finley-accent" />
          Custom Analysis Prompt
        </CardTitle>
        <CardDescription>
          Guide the AI analysis with specific questions or focus areas
        </CardDescription>
      </CardHeader>
      
      <CardContent className="space-y-6">
        {/* Quick Prompt Categories */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-medium">Quick Prompts</h4>
            <Button 
              variant="ghost" 
              size="sm"
              onClick={() => setShowAdvanced(!showAdvanced)}
            >
              {showAdvanced ? 'Basic' : 'Advanced'}
            </Button>
          </div>
          
          <div className="flex flex-wrap gap-2">
            {Object.keys(FINANCE_PROMPT_TEMPLATES).map((category) => (
              <Button
                key={category}
                variant={activeCategory === category ? "default" : "outline"}
                size="sm"
                onClick={() => setActiveCategory(category)}
                className="capitalize"
              >
                {getCategoryIcon(category)}
                {category}
              </Button>
            ))}
          </div>
        </div>

        {/* Suggested Prompts Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          {getPromptsByCategory(activeCategory).map((prompt, index) => (
            <Button
              key={index}
              variant="ghost"
              className="justify-start h-auto p-3 text-left hover:bg-finley-accent/5 hover:border-finley-accent/20 border border-transparent"
              onClick={() => handleSuggestedPrompt(prompt)}
            >
              <div className="text-sm">{prompt}</div>
            </Button>
          ))}
        </div>

        {/* Contextual Prompts */}
        {showAdvanced && getContextualPrompts().length > 0 && (
          <div className="space-y-3">
            <h4 className="text-sm font-medium">Document-Specific Suggestions</h4>
            <div className="space-y-2 max-h-40 overflow-y-auto">
              {getContextualPrompts().slice(0, 6).map((prompt, index) => (
                <Button
                  key={index}
                  variant="ghost"
                  size="sm"
                  className="justify-start h-auto p-2 text-left w-full hover:bg-muted/50"
                  onClick={() => handleSuggestedPrompt(prompt)}
                >
                  <div className="text-xs text-muted-foreground truncate">
                    {prompt}
                  </div>
                </Button>
              ))}
            </div>
          </div>
        )}

        {/* Custom Prompt Input */}
        <div className="space-y-3">
          <h4 className="text-sm font-medium">Custom Question</h4>
          <div className="space-y-3">
            <Textarea
              placeholder="Ask anything about your financial data... e.g., 'What are the main cost drivers affecting profitability?' or 'Analyze working capital trends over the past quarters'"
              value={customPrompt}
              onChange={(e) => setCustomPrompt(e.target.value)}
              className="min-h-20 resize-none"
              disabled={isProcessing}
            />
            
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {documentType && (
                  <Badge variant="secondary" className="text-xs">
                    <FileText className="w-3 h-3 mr-1" />
                    {documentType.replace('_', ' ')}
                  </Badge>
                )}
                {detectedSheets.length > 0 && (
                  <Badge variant="outline" className="text-xs">
                    {detectedSheets.length} sheets
                  </Badge>
                )}
              </div>
              
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCustomPrompt('')}
                  disabled={!customPrompt || isProcessing}
                >
                  <RefreshCw className="w-4 h-4" />
                  Clear
                </Button>
                <Button
                  onClick={handleSubmit}
                  disabled={!customPrompt.trim() || isProcessing}
                  className="min-w-20"
                >
                  {isProcessing ? (
                    <RefreshCw className="w-4 h-4 animate-spin" />
                  ) : (
                    <>
                      <Send className="w-4 h-4 mr-2" />
                      Analyze
                    </>
                  )}
                </Button>
              </div>
            </div>
          </div>
        </div>

        {/* Pro Tips */}
        <div className="bg-muted/30 p-3 rounded-lg">
          <div className="flex items-start gap-2">
            <Lightbulb className="w-4 h-4 text-finley-accent mt-0.5 flex-shrink-0" />
            <div className="text-xs text-muted-foreground">
              <strong>Pro tips:</strong> Be specific about time periods, metrics, or business questions. 
              Ask for comparisons, trends, or actionable insights. The AI can analyze ratios, 
              identify patterns, and provide strategic recommendations.
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};