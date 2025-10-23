import React, { useState } from 'react';
import { AlertTriangle, FileText, Calendar, BarChart3, CheckCircle, XCircle, Info } from 'lucide-react';

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
}

// REMOVED: VersionCandidate and VersionRecommendation interfaces
// These were part of the deprecated version_recommendations system
// that was removed in migration 20251013000000-remove-unused-version-tables.sql
// The backend never populates these fields, making this dead code.

interface DuplicateDetectionModalProps {
  isOpen: boolean;
  onClose: () => void;
  duplicateInfo?: DuplicateInfo;
  onDecision: (decision: 'replace' | 'keep_both' | 'skip' | 'delta_merge') => void;
  // phase is always 'basic_duplicate' - other phases were never implemented
  phase: 'basic_duplicate';
  deltaAnalysis?: any;
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

  const renderBasicDuplicate = () => (
    <div className="space-y-6 animate-fade-in">
      {/* FIX #6: Header with Finley AI branding - Premium Black with White accents */}
      <div className="flex items-center space-x-4 p-4 bg-gradient-to-r from-white/5 to-white/10 rounded-xl border border-white/20">
        <div className="p-3 bg-white/10 rounded-lg">
          <AlertTriangle className="h-7 w-7 text-white" />
        </div>
        <div>
          <h3 className="text-xl font-bold text-white">Identical File Detected</h3>
          <p className="text-sm text-muted-foreground">This exact file has been uploaded before</p>
        </div>
      </div>

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

          {/* FIX ISSUE #8: Delta Analysis Display - Backend wraps in { delta_analysis: {...} } */}
          {deltaAnalysis?.delta_analysis?.new_rows !== undefined && (
            <div className="mt-4 bg-background/50 rounded-lg p-4 border border-border">
              <p className="font-semibold text-white mb-2">Delta Analysis</p>
              <div className="flex items-center space-x-4 text-sm">
                <span className="text-muted-foreground">
                  New rows: <span className="font-bold text-white">{deltaAnalysis.delta_analysis.new_rows}</span>
                </span>
                <span className="text-muted-foreground">•</span>
                <span className="text-muted-foreground">
                  Existing: <span className="font-bold text-white">{deltaAnalysis.delta_analysis.existing_rows}</span>
                </span>
                <span className="text-muted-foreground">•</span>
                <span className="text-muted-foreground">
                  Confidence: <span className="font-bold text-white">{Math.round((deltaAnalysis.delta_analysis.confidence || 0) * 100)}%</span>
                </span>
              </div>
            </div>
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

      <div className="space-y-4">
        <h4 className="font-semibold text-white text-lg">What would you like to do?</h4>
        
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

        {/* Delta merge option when we have analysis or similar/near duplicate */}
        <button
          onClick={() => onDecision('delta_merge')}
          data-testid="delta-merge-button"
          className="group w-full flex items-center justify-between p-5 border-2 border-white/30 rounded-xl hover:border-white/50 hover:bg-white/10 transition-all duration-200 hover-lift bg-white/5"
        >
          <div className="flex items-center space-x-4">
            <div className="p-2 bg-white/20 rounded-lg group-hover:bg-white/30 transition-colors">
              <BarChart3 className="h-6 w-6 text-white" />
            </div>
            <div className="text-left">
              <p className="font-semibold text-white text-base">Merge new rows (delta)</p>
              <p className="text-sm text-muted-foreground">Only append rows not present in the existing file</p>
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
