import React, { useState } from 'react';
import { AlertTriangle, FileText, Calendar, BarChart3, CheckCircle, XCircle, Info, TrendingUp, Database, Sparkles, ArrowRight } from 'lucide-react';
import { Progress } from './ui/progress';
import { Badge } from './ui/badge';
import { Alert, AlertDescription, AlertTitle } from './ui/alert';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table';

interface DuplicateFile {
  id: string;
  filename: string;
  uploaded_at: string;
  status: string;
  total_rows: number;
}

interface DuplicateInfo {
  is_duplicate: boolean;
  duplicate_files: DuplicateFile[];
  latest_duplicate?: DuplicateFile;
  message: string;
  duplicate_type?: 'exact' | 'near' | 'content';
  similarity_score?: number;
  recommendation?: 'skip' | 'replace' | 'merge' | 'keep_both';
}

// REMOVED: VersionCandidate and VersionRecommendation interfaces
// These were part of the deprecated version_recommendations system
// that was removed in migration 20251013000000-remove-unused-version-tables.sql
// The backend never populates these fields, making this dead code.

interface DeltaAnalysis {
  delta_analysis?: {
    new_rows: number;
    existing_rows: number;
    modified_rows?: number;
    confidence: number;
    recommendation?: string;
    sample_new_rows?: Array<Record<string, any>>;
  };
}

interface DuplicateDetectionModalProps {
  isOpen: boolean;
  onClose: () => void;
  duplicateInfo?: DuplicateInfo;
  onDecision: (decision: 'replace' | 'keep_both' | 'skip' | 'delta_merge') => void;
  phase: 'basic_duplicate';
  deltaAnalysis?: DeltaAnalysis;
  error?: string | null;
}

export const DuplicateDetectionModal: React.FC<DuplicateDetectionModalProps> = ({
  isOpen,
  onClose,
  duplicateInfo,
  onDecision,
  phase,
  deltaAnalysis,
  error
}) => {

  if (!isOpen) return null;

  // Extract duplicate type and similarity score
  const duplicateType = duplicateInfo?.duplicate_type || 'exact';
  const similarityScore = duplicateInfo?.similarity_score || 1.0;
  const backendRecommendation = duplicateInfo?.recommendation;
  const delta = deltaAnalysis?.delta_analysis;
  const totalRows = (delta?.new_rows || 0) + (delta?.existing_rows || 0);

  // Determine smart recommendation based on duplicate type
  const getSmartRecommendation = (): { action: string; reason: string } => {
    if (duplicateType === 'exact') {
      return {
        action: 'skip',
        reason: 'This file is 100% identical to the existing one. No need to process it again.'
      };
    }
    if (duplicateType === 'near' && similarityScore >= 0.95) {
      return {
        action: 'replace',
        reason: `Files are ${Math.round(similarityScore * 100)}% similar. This appears to be an updated version.`
      };
    }
    if (duplicateType === 'content' && delta && delta.new_rows > 0) {
      return {
        action: 'delta_merge',
        reason: `Found ${delta.new_rows} new rows. Merge them to keep your data up-to-date.`
      };
    }
    return {
      action: backendRecommendation || 'skip',
      reason: 'Review the options below and choose what works best for you.'
    };
  };

  const recommendation = getSmartRecommendation();

  const renderBasicDuplicate = () => (
    <div className="space-y-6 animate-fade-in">
      {/* ENHANCED: Dynamic Header Based on Duplicate Type */}
      <div className="flex items-center justify-between space-x-4 p-4 bg-gradient-to-r from-white/5 to-white/10 rounded-xl border border-white/20">
        <div className="flex items-center space-x-4">
          <div className="p-3 bg-white/10 rounded-lg">
            {duplicateType === 'exact' && <AlertTriangle className="h-7 w-7 text-red-400" />}
            {duplicateType === 'near' && <TrendingUp className="h-7 w-7 text-yellow-400" />}
            {duplicateType === 'content' && <Database className="h-7 w-7 text-blue-400" />}
          </div>
          <div>
            <h3 className="text-xl font-bold text-white">
              {duplicateType === 'exact' && 'Identical File Detected'}
              {duplicateType === 'near' && 'Similar File Found'}
              {duplicateType === 'content' && 'Overlapping Data Detected'}
            </h3>
            <p className="text-sm text-muted-foreground">
              {duplicateType === 'exact' && '100% match with existing file'}
              {duplicateType === 'near' && `${Math.round(similarityScore * 100)}% similarity detected`}
              {duplicateType === 'content' && 'Same data structure with possible new rows'}
            </p>
          </div>
        </div>
        
        {/* Duplicate Type Badge */}
        <Badge 
          variant={duplicateType === 'exact' ? 'destructive' : duplicateType === 'near' ? 'default' : 'secondary'}
          className="text-xs font-bold px-3 py-1"
        >
          {duplicateType === 'exact' && '100% IDENTICAL'}
          {duplicateType === 'near' && `${Math.round(similarityScore * 100)}% SIMILAR`}
          {duplicateType === 'content' && 'SAME STRUCTURE'}
        </Badge>
      </div>

      {/* ENHANCEMENT: Similarity Score Visualization (for near duplicates) */}
      {duplicateType === 'near' && (
        <Card className="bg-card/50 border-border">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-white">Similarity Analysis</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex justify-between items-center text-sm">
              <span className="text-muted-foreground">Match Score</span>
              <span className="text-2xl font-bold text-white">{Math.round(similarityScore * 100)}%</span>
            </div>
            <Progress value={similarityScore * 100} className="h-3" />
            <p className="text-xs text-muted-foreground">
              {similarityScore >= 0.95 && '✨ Almost identical - likely the same file with minor edits'}
              {similarityScore >= 0.85 && similarityScore < 0.95 && '📝 Very similar - possibly an updated version'}
              {similarityScore < 0.85 && '⚠️ Somewhat similar - review carefully before replacing'}
            </p>
          </CardContent>
        </Card>
      )}

      {duplicateInfo && (
        <div className="bg-card border border-border rounded-xl p-5 space-y-4">
          <p className="text-white font-medium">{duplicateInfo.message}</p>
          
          <div className="space-y-3">
            {duplicateInfo.duplicate_files.map((file) => (
              <div key={file.id} className="flex items-center justify-between bg-background/50 rounded-lg p-4 border border-border hover:border-muted-foreground/30 transition-all">
                <div className="flex items-center space-x-3">
                  <div className="p-2 bg-muted rounded-lg">
                    <FileText className="h-5 w-5 text-muted-foreground" />
                  </div>
                  <div>
                    <p className="font-semibold text-white">{file.filename}</p>
                    <p className="text-sm text-muted-foreground">
                      Uploaded {new Date(file.uploaded_at).toLocaleDateString()} • {file.total_rows} rows
                    </p>
                  </div>
                </div>
                <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                  file.status === 'completed' ? 'bg-green-500/20 text-green-400' : 'bg-muted text-muted-foreground'
                }`}>
                  {file.status}
                </span>
              </div>
            ))}
          </div>

          {/* ENHANCED: Delta Analysis Visualization */}
          {delta && delta.new_rows !== undefined && (
            <Card className="mt-4 bg-gradient-to-br from-blue-500/10 to-purple-500/10 border-blue-500/30">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <BarChart3 className="h-5 w-5" />
                  Delta Analysis Results
                </CardTitle>
                <CardDescription>
                  Intelligent comparison of new vs existing data
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Visual Breakdown Cards */}
                <div className="grid grid-cols-3 gap-3">
                  <Card className="bg-green-500/10 border-green-500/30">
                    <CardContent className="pt-4 text-center">
                      <div className="text-3xl font-bold text-green-400">{delta.new_rows}</div>
                      <div className="text-xs text-muted-foreground mt-1">New Rows</div>
                    </CardContent>
                  </Card>
                  
                  <Card className="bg-blue-500/10 border-blue-500/30">
                    <CardContent className="pt-4 text-center">
                      <div className="text-3xl font-bold text-blue-400">{delta.existing_rows}</div>
                      <div className="text-xs text-muted-foreground mt-1">Existing</div>
                    </CardContent>
                  </Card>
                  
                  <Card className="bg-purple-500/10 border-purple-500/30">
                    <CardContent className="pt-4 text-center">
                      <div className="text-3xl font-bold text-purple-400">{Math.round(delta.confidence * 100)}%</div>
                      <div className="text-xs text-muted-foreground mt-1">Confidence</div>
                    </CardContent>
                  </Card>
                </div>

                {/* Visual Stacked Bar */}
                {totalRows > 0 && (
                  <div className="space-y-2">
                    <div className="flex h-8 rounded-lg overflow-hidden">
                      <div 
                        className="bg-green-500 flex items-center justify-center text-xs font-bold text-white"
                        style={{width: `${(delta.new_rows / totalRows) * 100}%`}}
                      >
                        {delta.new_rows > 0 && `${Math.round((delta.new_rows / totalRows) * 100)}%`}
                      </div>
                      <div 
                        className="bg-blue-500 flex items-center justify-center text-xs font-bold text-white"
                        style={{width: `${(delta.existing_rows / totalRows) * 100}%`}}
                      >
                        {delta.existing_rows > 0 && `${Math.round((delta.existing_rows / totalRows) * 100)}%`}
                      </div>
                    </div>
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>← New Data</span>
                      <span>Existing Data →</span>
                    </div>
                  </div>
                )}

                {/* Merge Preview */}
                {delta.sample_new_rows && delta.sample_new_rows.length > 0 && (
                  <div className="space-y-2">
                    <h4 className="text-sm font-semibold text-white flex items-center gap-2">
                      <Sparkles className="h-4 w-4 text-yellow-400" />
                      Preview of New Rows
                    </h4>
                    <div className="bg-background/50 rounded-lg border border-border overflow-hidden">
                      <Table>
                        <TableHeader>
                          <TableRow className="border-border">
                            {Object.keys(delta.sample_new_rows[0]).slice(0, 4).map((key) => (
                              <TableHead key={key} className="text-xs text-muted-foreground">
                                {key}
                              </TableHead>
                            ))}
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {delta.sample_new_rows.slice(0, 3).map((row, idx) => (
                            <TableRow key={idx} className="border-border bg-green-500/5">
                              {Object.values(row).slice(0, 4).map((val: any, i) => (
                                <TableCell key={i} className="text-xs text-white">
                                  {val !== null && val !== undefined ? String(val).slice(0, 30) : '-'}
                                </TableCell>
                              ))}
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </div>
                    <p className="text-xs text-muted-foreground text-center">
                      Showing {Math.min(3, delta.sample_new_rows.length)} of {delta.new_rows} new rows
                    </p>
                  </div>
                )}

                {/* What Will Happen Alert */}
                <Alert className="bg-blue-500/10 border-blue-500/30">
                  <Info className="h-4 w-4 text-blue-400" />
                  <AlertTitle className="text-white">What will happen?</AlertTitle>
                  <AlertDescription className="text-muted-foreground">
                    We'll add <span className="font-bold text-white">{delta.new_rows}</span> new transactions to your existing{' '}
                    <span className="font-bold text-white">{delta.existing_rows}</span> records. Duplicate rows will be automatically skipped.
                  </AlertDescription>
                </Alert>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4">
          <div className="flex items-center space-x-3">
            <XCircle className="h-5 w-5 text-red-400" />
            <div>
              <p className="font-semibold text-red-400">Error Processing Decision</p>
              <p className="text-sm text-red-300 mt-1">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* ENHANCED: Smart Recommendation Alert */}
      <Alert className="bg-gradient-to-r from-blue-500/10 to-purple-500/10 border-blue-500/30">
        <Sparkles className="h-5 w-5 text-blue-400" />
        <AlertTitle className="text-white font-bold">💡 Smart Recommendation</AlertTitle>
        <AlertDescription className="text-muted-foreground">
          {recommendation.reason}
        </AlertDescription>
      </Alert>

      <div className="space-y-4">
        <h4 className="font-semibold text-white text-lg">What would you like to do?</h4>
        
        {/* CONDITIONAL: Show Skip for Exact Duplicates (Recommended) */}
        {duplicateType === 'exact' && (
          <>
            <button
              onClick={() => onDecision('skip')}
              data-testid="skip-button"
              className="group w-full flex items-center justify-between p-5 border-2 border-green-500/50 bg-green-500/10 rounded-xl hover:border-green-500/70 hover:bg-green-500/20 transition-all duration-200 hover-lift"
            >
              <div className="flex items-center space-x-4">
                <div className="p-2 bg-green-500/20 rounded-lg group-hover:bg-green-500/30 transition-colors">
                  <CheckCircle className="h-6 w-6 text-green-400" />
                </div>
                <div className="text-left">
                  <p className="font-semibold text-white text-base flex items-center gap-2">
                    Skip this upload
                    <Badge variant="secondary" className="text-[10px]">RECOMMENDED</Badge>
                  </p>
                  <p className="text-sm text-muted-foreground">File is identical - no need to process again</p>
                </div>
              </div>
              <ArrowRight className="h-5 w-5 text-green-400" />
            </button>

            <button
              onClick={() => onDecision('replace')}
              data-testid="replace-button"
              className="group w-full flex items-center justify-between p-5 border-2 border-white/20 rounded-xl hover:border-white/40 hover:bg-white/5 transition-all duration-200 hover-lift"
            >
              <div className="flex items-center space-x-4">
                <div className="p-2 bg-white/10 rounded-lg group-hover:bg-white/20 transition-colors">
                  <CheckCircle className="h-6 w-6 text-white" />
                </div>
                <div className="text-left">
                  <p className="font-semibold text-white text-base">Replace anyway</p>
                  <p className="text-sm text-muted-foreground">Archive old and reprocess (not recommended)</p>
                </div>
              </div>
            </button>
          </>
        )}

        {/* CONDITIONAL: Show Replace for Near Duplicates (Recommended for high similarity) */}
        {duplicateType === 'near' && (
          <>
            {similarityScore >= 0.95 ? (
              <button
                onClick={() => onDecision('replace')}
                data-testid="replace-button"
                className="group w-full flex items-center justify-between p-5 border-2 border-green-500/50 bg-green-500/10 rounded-xl hover:border-green-500/70 hover:bg-green-500/20 transition-all duration-200 hover-lift"
              >
                <div className="flex items-center space-x-4">
                  <div className="p-2 bg-green-500/20 rounded-lg group-hover:bg-green-500/30 transition-colors">
                    <CheckCircle className="h-6 w-6 text-green-400" />
                  </div>
                  <div className="text-left">
                    <p className="font-semibold text-white text-base flex items-center gap-2">
                      Replace existing file
                      <Badge variant="secondary" className="text-[10px]">RECOMMENDED</Badge>
                    </p>
                    <p className="text-sm text-muted-foreground">Update with this newer version</p>
                  </div>
                </div>
                <ArrowRight className="h-5 w-5 text-green-400" />
              </button>
            ) : (
              <button
                onClick={() => onDecision('replace')}
                data-testid="replace-button"
                className="group w-full flex items-center justify-between p-5 border-2 border-white/20 rounded-xl hover:border-white/40 hover:bg-white/5 transition-all duration-200 hover-lift"
              >
                <div className="flex items-center space-x-4">
                  <div className="p-2 bg-white/10 rounded-lg group-hover:bg-white/20 transition-colors">
                    <CheckCircle className="h-6 w-6 text-white" />
                  </div>
                  <div className="text-left">
                    <p className="font-semibold text-white text-base">Replace existing file</p>
                    <p className="text-sm text-muted-foreground">Archive old and process this one</p>
                  </div>
                </div>
              </button>
            )}

            <button
              onClick={() => onDecision('keep_both')}
              data-testid="keep-both-button"
              className="group w-full flex items-center justify-between p-5 border-2 border-white/20 rounded-xl hover:border-white/40 hover:bg-white/5 transition-all duration-200 hover-lift"
            >
              <div className="flex items-center space-x-4">
                <div className="p-2 bg-white/10 rounded-lg group-hover:bg-white/20 transition-colors">
                  <Info className="h-6 w-6 text-white" />
                </div>
                <div className="text-left">
                  <p className="font-semibold text-white text-base">Keep both files</p>
                  <p className="text-sm text-muted-foreground">Process alongside existing data</p>
                </div>
              </div>
            </button>

            <button
              onClick={() => onDecision('skip')}
              data-testid="skip-button"
              className="group w-full flex items-center justify-between p-5 border-2 border-white/10 rounded-xl hover:border-white/20 hover:bg-white/5 transition-all duration-200 hover-lift"
            >
              <div className="flex items-center space-x-4">
                <div className="p-2 bg-white/5 rounded-lg group-hover:bg-white/10 transition-colors">
                  <XCircle className="h-6 w-6 text-muted-foreground" />
                </div>
                <div className="text-left">
                  <p className="font-semibold text-white text-base">Skip this upload</p>
                  <p className="text-sm text-muted-foreground">Cancel and keep existing file</p>
                </div>
              </div>
            </button>
          </>
        )}

        {/* CONDITIONAL: Show Delta Merge for Content Duplicates (Recommended) */}
        {duplicateType === 'content' && delta && delta.new_rows > 0 && (
          <>
            <button
              onClick={() => onDecision('delta_merge')}
              data-testid="delta-merge-button"
              className="group w-full flex items-center justify-between p-5 border-2 border-green-500/50 bg-green-500/10 rounded-xl hover:border-green-500/70 hover:bg-green-500/20 transition-all duration-200 hover-lift"
            >
              <div className="flex items-center space-x-4">
                <div className="p-2 bg-green-500/20 rounded-lg group-hover:bg-green-500/30 transition-colors">
                  <BarChart3 className="h-6 w-6 text-green-400" />
                </div>
                <div className="text-left">
                  <p className="font-semibold text-white text-base flex items-center gap-2">
                    Merge {delta.new_rows} new rows
                    <Badge variant="secondary" className="text-[10px]">RECOMMENDED</Badge>
                  </p>
                  <p className="text-sm text-muted-foreground">Smart merge - only add new data</p>
                </div>
              </div>
              <ArrowRight className="h-5 w-5 text-green-400" />
            </button>

            <button
              onClick={() => onDecision('replace')}
              data-testid="replace-button"
              className="group w-full flex items-center justify-between p-5 border-2 border-white/20 rounded-xl hover:border-white/40 hover:bg-white/5 transition-all duration-200 hover-lift"
            >
              <div className="flex items-center space-x-4">
                <div className="p-2 bg-white/10 rounded-lg group-hover:bg-white/20 transition-colors">
                  <CheckCircle className="h-6 w-6 text-white" />
                </div>
                <div className="text-left">
                  <p className="font-semibold text-white text-base">Replace existing file</p>
                  <p className="text-sm text-muted-foreground">Delete old data and start fresh</p>
                </div>
              </div>
            </button>

            <button
              onClick={() => onDecision('keep_both')}
              data-testid="keep-both-button"
              className="group w-full flex items-center justify-between p-5 border-2 border-white/20 rounded-xl hover:border-white/40 hover:bg-white/5 transition-all duration-200 hover-lift"
            >
              <div className="flex items-center space-x-4">
                <div className="p-2 bg-white/10 rounded-lg group-hover:bg-white/20 transition-colors">
                  <Info className="h-6 w-6 text-white" />
                </div>
                <div className="text-left">
                  <p className="font-semibold text-white text-base">Keep both files</p>
                  <p className="text-sm text-muted-foreground">Process as separate datasets</p>
                </div>
              </div>
            </button>

            <button
              onClick={() => onDecision('skip')}
              data-testid="skip-button"
              className="group w-full flex items-center justify-between p-5 border-2 border-white/10 rounded-xl hover:border-white/20 hover:bg-white/5 transition-all duration-200 hover-lift"
            >
              <div className="flex items-center space-x-4">
                <div className="p-2 bg-white/5 rounded-lg group-hover:bg-white/10 transition-colors">
                  <XCircle className="h-6 w-6 text-muted-foreground" />
                </div>
                <div className="text-left">
                  <p className="font-semibold text-white text-base">Skip this upload</p>
                  <p className="text-sm text-muted-foreground">Cancel and keep existing file</p>
                </div>
              </div>
            </button>
          </>
        )}

        {/* FALLBACK: Show all options if no specific type */}
        {!duplicateType && (
          <>
            <button
              onClick={() => onDecision('replace')}
              data-testid="replace-button"
              className="group w-full flex items-center justify-between p-5 border-2 border-white/20 rounded-xl hover:border-white/40 hover:bg-white/5 transition-all duration-200 hover-lift"
            >
              <div className="flex items-center space-x-4">
                <div className="p-2 bg-white/10 rounded-lg group-hover:bg-white/20 transition-colors">
                  <CheckCircle className="h-6 w-6 text-white" />
                </div>
                <div className="text-left">
                  <p className="font-semibold text-white text-base">Replace existing file</p>
                  <p className="text-sm text-muted-foreground">Archive the old version and process this new one</p>
                </div>
              </div>
            </button>

            <button
              onClick={() => onDecision('keep_both')}
              data-testid="keep-both-button"
              className="group w-full flex items-center justify-between p-5 border-2 border-white/20 rounded-xl hover:border-white/40 hover:bg-white/5 transition-all duration-200 hover-lift"
            >
              <div className="flex items-center space-x-4">
                <div className="p-2 bg-white/10 rounded-lg group-hover:bg-white/20 transition-colors">
                  <Info className="h-6 w-6 text-white" />
                </div>
                <div className="text-left">
                  <p className="font-semibold text-white text-base">Keep both files</p>
                  <p className="text-sm text-muted-foreground">Process this file alongside the existing one</p>
                </div>
              </div>
            </button>

            <button
              onClick={() => onDecision('skip')}
              data-testid="skip-button"
              className="group w-full flex items-center justify-between p-5 border-2 border-white/10 rounded-xl hover:border-white/20 hover:bg-white/5 transition-all duration-200 hover-lift"
            >
              <div className="flex items-center space-x-4">
                <div className="p-2 bg-white/5 rounded-lg group-hover:bg-white/10 transition-colors">
                  <XCircle className="h-6 w-6 text-muted-foreground" />
                </div>
                <div className="text-left">
                  <p className="font-semibold text-white text-base">Skip this upload</p>
                  <p className="text-sm text-muted-foreground">Cancel processing and keep the existing file</p>
                </div>
              </div>
            </button>
          </>
        )}
      </div>
    </div>
  );

  // REMOVED: renderVersionDetection() function (105 lines)
  // This was dead code for the deprecated version_recommendations system
  // The phase was always 'basic_duplicate' and never 'versions_detected'

  return (
    <div className="fixed inset-0 bg-[#1a1a1a]/80 backdrop-blur-sm flex items-center justify-center z-50 animate-fade-in">
      <div className="bg-card border border-border rounded-2xl shadow-2xl max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto animate-scale-in">
        <div className="p-8">
          {renderBasicDuplicate()}
          
          <div className="mt-8 pt-6 border-t border-border">
            <button
              onClick={onClose}
              data-testid="cancel-upload-button"
              className="w-full px-6 py-3 border-2 border-border text-white rounded-xl hover:bg-muted hover:border-muted-foreground/30 transition-all duration-200 font-medium"
            >
              Cancel Upload
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
