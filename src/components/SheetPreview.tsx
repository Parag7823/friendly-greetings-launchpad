import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { 
  FileSpreadsheet, 
  TrendingUp, 
  DollarSign, 
  Calendar,
  Eye,
  EyeOff,
  ChevronDown,
  ChevronRight,
  AlertTriangle,
  CheckCircle
} from 'lucide-react';

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

interface SheetPreviewProps {
  sheets: SheetMetadata[];
  onSheetSelect?: (sheetName: string) => void;
  onPromptSuggestion?: (prompt: string) => void;
  selectedSheet?: string;
}

const getSheetTypeIcon = (type: string) => {
  switch (type) {
    case 'balance_sheet':
      return <FileSpreadsheet className="w-4 h-4 text-blue-500" />;
    case 'income_statement':
      return <TrendingUp className="w-4 h-4 text-green-500" />;
    case 'cash_flow':
      return <DollarSign className="w-4 h-4 text-purple-500" />;
    case 'budget':
      return <Calendar className="w-4 h-4 text-orange-500" />;
    default:
      return <FileSpreadsheet className="w-4 h-4 text-gray-500" />;
  }
};

const getSheetTypeColor = (type: string) => {
  switch (type) {
    case 'balance_sheet':
      return 'bg-blue-50 text-blue-700 border-blue-200';
    case 'income_statement':
      return 'bg-green-50 text-green-700 border-green-200';
    case 'cash_flow':
      return 'bg-purple-50 text-purple-700 border-purple-200';
    case 'budget':
      return 'bg-orange-50 text-orange-700 border-orange-200';
    default:
      return 'bg-gray-50 text-gray-700 border-gray-200';
  }
};

const getConfidenceIndicator = (score: number) => {
  if (score >= 0.8) {
    return <CheckCircle className="w-4 h-4 text-green-500" />;
  } else if (score >= 0.6) {
    return <AlertTriangle className="w-4 h-4 text-yellow-500" />;
  } else {
    return <AlertTriangle className="w-4 h-4 text-red-500" />;
  }
};

const formatSheetType = (type: string) => {
  return type.split('_').map(word => 
    word.charAt(0).toUpperCase() + word.slice(1)
  ).join(' ');
};

export const SheetPreview = ({ 
  sheets, 
  onSheetSelect, 
  onPromptSuggestion,
  selectedSheet 
}: SheetPreviewProps) => {
  const [expandedSheets, setExpandedSheets] = useState<Set<string>>(new Set());
  const [activeTab, setActiveTab] = useState('overview');

  const toggleSheetExpansion = (sheetName: string) => {
    const newExpanded = new Set(expandedSheets);
    if (newExpanded.has(sheetName)) {
      newExpanded.delete(sheetName);
    } else {
      newExpanded.add(sheetName);
    }
    setExpandedSheets(newExpanded);
  };

  const financialSheets = sheets.filter(s => s.type !== 'general');
  const otherSheets = sheets.filter(s => s.type === 'general');

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <FileSpreadsheet className="w-5 h-5 text-finley-accent" />
          Document Structure Analysis
        </CardTitle>
        <CardDescription>
          {sheets.length} sheets detected • {financialSheets.length} financial statements identified
        </CardDescription>
      </CardHeader>
      
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid grid-cols-3 w-full">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="financial">Financial Sheets</TabsTrigger>
            <TabsTrigger value="preview">Data Preview</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center p-3 bg-muted/50 rounded-lg">
                <div className="text-2xl font-bold text-finley-accent">{sheets.length}</div>
                <div className="text-sm text-muted-foreground">Total Sheets</div>
              </div>
              <div className="text-center p-3 bg-muted/50 rounded-lg">
                <div className="text-2xl font-bold text-green-600">{financialSheets.length}</div>
                <div className="text-sm text-muted-foreground">Financial Statements</div>
              </div>
              <div className="text-center p-3 bg-muted/50 rounded-lg">
                <div className="text-2xl font-bold text-blue-600">
                  {Math.round(sheets.reduce((acc, s) => acc + s.confidenceScore, 0) / sheets.length * 100)}%
                </div>
                <div className="text-sm text-muted-foreground">Avg Confidence</div>
              </div>
              <div className="text-center p-3 bg-muted/50 rounded-lg">
                <div className="text-2xl font-bold text-purple-600">
                  {sheets.reduce((acc, s) => acc + s.rowCount, 0).toLocaleString()}
                </div>
                <div className="text-sm text-muted-foreground">Total Rows</div>
              </div>
            </div>

            <div className="space-y-2">
              <h4 className="font-medium text-sm">Quick Insights</h4>
              {sheets.map((sheet, index) => (
                <div 
                  key={sheet.name}
                  className="flex items-center justify-between p-3 border rounded-lg hover:bg-muted/50 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    {getSheetTypeIcon(sheet.type)}
                    <div>
                      <div className="font-medium text-sm">{sheet.name}</div>
                      <div className="text-xs text-muted-foreground">
                        {sheet.rowCount.toLocaleString()} rows • {sheet.columnCount} columns
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {getConfidenceIndicator(sheet.confidenceScore)}
                    <Badge 
                      variant="outline" 
                      className={`text-xs ${getSheetTypeColor(sheet.type)}`}
                    >
                      {formatSheetType(sheet.type)}
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="financial" className="space-y-4">
            {financialSheets.length > 0 ? (
              <div className="space-y-3">
                {financialSheets.map((sheet) => (
                  <Card key={sheet.name} className="border-l-4 border-l-finley-accent">
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3">
                          {getSheetTypeIcon(sheet.type)}
                          <div>
                            <h4 className="font-medium">{sheet.name}</h4>
                            <p className="text-sm text-muted-foreground">
                              {formatSheetType(sheet.type)}
                              {sheet.detectedPeriod && ` • ${sheet.detectedPeriod}`}
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="text-right">
                            <div className="text-sm font-medium">
                              {Math.round(sheet.confidenceScore * 100)}% confident
                            </div>
                            <div className="text-xs text-muted-foreground">
                              {sheet.hasNumbers ? 'Numerical data detected' : 'Text-based'}
                            </div>
                          </div>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => toggleSheetExpansion(sheet.name)}
                          >
                            {expandedSheets.has(sheet.name) ? (
                              <ChevronDown className="w-4 h-4" />
                            ) : (
                              <ChevronRight className="w-4 h-4" />
                            )}
                          </Button>
                        </div>
                      </div>

                      {expandedSheets.has(sheet.name) && (
                        <div className="space-y-3 pt-3 border-t">
                          <div>
                            <h5 className="text-sm font-medium mb-2">Key Columns</h5>
                            <div className="flex flex-wrap gap-1">
                              {sheet.keyColumns.map((col, index) => (
                                <Badge key={index} variant="secondary" className="text-xs">
                                  {col}
                                </Badge>
                              ))}
                            </div>
                          </div>
                          
                          {sheet.suggestedPrompts && sheet.suggestedPrompts.length > 0 && (
                            <div>
                              <h5 className="text-sm font-medium mb-2">Suggested Analysis</h5>
                              <div className="space-y-1">
                                {sheet.suggestedPrompts.slice(0, 3).map((prompt, index) => (
                                  <Button
                                    key={index}
                                    variant="ghost"
                                    size="sm"
                                    className="justify-start h-auto p-2 text-left"
                                    onClick={() => onPromptSuggestion?.(prompt)}
                                  >
                                    <div className="text-xs text-muted-foreground">
                                      "{prompt}"
                                    </div>
                                  </Button>
                                ))}
                              </div>
                            </div>
                          )}
                          
                          {onSheetSelect && (
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => onSheetSelect(sheet.name)}
                              className="w-full"
                            >
                              Analyze This Sheet
                            </Button>
                          )}
                        </div>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <FileSpreadsheet className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p>No financial statements detected</p>
                <p className="text-sm">The uploaded file may contain general data or require manual review</p>
              </div>
            )}
          </TabsContent>

          <TabsContent value="preview" className="space-y-4">
            {sheets.map((sheet) => (
              <Card key={sheet.name}>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center gap-2">
                    {getSheetTypeIcon(sheet.type)}
                    {sheet.name}
                  </CardTitle>
                  <CardDescription>
                    Preview of first 5 rows • {sheet.rowCount} total rows
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {sheet.preview && sheet.preview.length > 0 ? (
                    <div className="overflow-x-auto">
                      <table className="w-full text-xs border-collapse">
                        <thead>
                          <tr className="border-b bg-muted/50">
                            {sheet.preview[0]?.map((header: any, index: number) => (
                              <th key={index} className="text-left p-2 font-medium">
                                {header || `Column ${index + 1}`}
                              </th>
                            )) || []}
                          </tr>
                        </thead>
                        <tbody>
                          {sheet.preview.slice(1, 6).map((row: any[], rowIndex: number) => (
                            <tr key={rowIndex} className="border-b hover:bg-muted/25">
                              {row.map((cell: any, cellIndex: number) => (
                                <td key={cellIndex} className="p-2 truncate max-w-32">
                                  {cell !== null && cell !== undefined ? String(cell) : '—'}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <div className="text-center py-4 text-muted-foreground">
                      <Eye className="w-8 h-8 mx-auto mb-2 opacity-50" />
                      <p>No preview available for this sheet</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};