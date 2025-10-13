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

interface VersionCandidate {
  id: string;
  filename: string;
  file_hash: string;
  created_at: string;
  total_rows: number;
  total_columns: number;
}

interface VersionRecommendation {
  recommended_file_id: string;
  recommended_version: {
    filename: string;
    row_count: number;
    column_count: number;
    overall_score: number;
    is_most_recent: boolean;
  };
  reasoning: string;
  confidence: number;
}

interface DuplicateDetectionModalProps {
  isOpen: boolean;
  onClose: () => void;
  duplicateInfo?: DuplicateInfo;
  versionCandidates?: VersionCandidate[];
  recommendation?: VersionRecommendation;
  onDecision: (decision: 'replace' | 'keep_both' | 'skip' | 'delta_merge') => void;
  onVersionAccept: (accepted: boolean, feedback?: string) => void;
  phase: 'basic_duplicate' | 'versions_detected' | 'similar_files';
  deltaAnalysis?: any;
  error?: string | null;
}

export const DuplicateDetectionModal: React.FC<DuplicateDetectionModalProps> = ({
  isOpen,
  onClose,
  duplicateInfo,
  versionCandidates,
  recommendation,
  onDecision,
  onVersionAccept,
  phase,
  deltaAnalysis,
  error
}) => {
  const [feedback, setFeedback] = useState('');
  const [showFeedback, setShowFeedback] = useState(false);

  if (!isOpen) return null;

  const renderBasicDuplicate = () => (
    <div className="space-y-6 animate-fade-in">
      {/* Header with modern gradient */}
      <div className="flex items-center space-x-4 p-4 bg-gradient-to-r from-amber-500/10 to-orange-500/10 rounded-xl border border-amber-500/20">
        <div className="p-3 bg-amber-500/20 rounded-lg">
          <AlertTriangle className="h-7 w-7 text-amber-400" />
        </div>
        <div>
          <h3 className="text-xl font-bold text-white">Identical File Detected</h3>
          <p className="text-sm text-gray-400">This exact file has been uploaded before</p>
        </div>
      </div>

      {duplicateInfo && (
        <div className="bg-card border border-border rounded-xl p-5 space-y-4">
          <p className="text-amber-400 font-medium">{duplicateInfo.message}</p>
          
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

          {/* Delta Analysis Display - Backend sends unwrapped delta_analysis object */}
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
          className="group w-full flex items-center justify-between p-5 border-2 border-border rounded-xl hover:border-green-500/50 hover:bg-green-500/5 transition-all duration-200 hover-lift"
        >
          <div className="flex items-center space-x-4">
            <div className="p-2 bg-green-500/20 rounded-lg group-hover:bg-green-500/30 transition-colors">
              <CheckCircle className="h-6 w-6 text-green-400" />
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
          className="group w-full flex items-center justify-between p-5 border-2 border-border rounded-xl hover:border-blue-500/50 hover:bg-blue-500/5 transition-all duration-200 hover-lift"
        >
          <div className="flex items-center space-x-4">
            <div className="p-2 bg-blue-500/20 rounded-lg group-hover:bg-blue-500/30 transition-colors">
              <Info className="h-6 w-6 text-blue-400" />
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
          className="group w-full flex items-center justify-between p-5 border-2 border-blue-500/30 rounded-xl hover:border-blue-500/60 hover:bg-blue-500/10 transition-all duration-200 hover-lift bg-blue-500/5"
        >
          <div className="flex items-center space-x-4">
            <div className="p-2 bg-blue-500/30 rounded-lg group-hover:bg-blue-500/40 transition-colors">
              <BarChart3 className="h-6 w-6 text-blue-300" />
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
          className="group w-full flex items-center justify-between p-5 border-2 border-border rounded-xl hover:border-red-500/50 hover:bg-red-500/5 transition-all duration-200 hover-lift"
        >
          <div className="flex items-center space-x-4">
            <div className="p-2 bg-red-500/20 rounded-lg group-hover:bg-red-500/30 transition-colors">
              <XCircle className="h-6 w-6 text-red-400" />
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

  const renderVersionDetection = () => (
    <div className="space-y-6">
      <div className="flex items-center space-x-3">
        <BarChart3 className="h-8 w-8 text-blue-500" />
        <div>
          <h3 className="text-lg font-semibold text-gray-900">Multiple Versions Detected</h3>
          <p className="text-sm text-gray-600">We found similar files that appear to be different versions</p>
        </div>
      </div>

      {versionCandidates && (
        <div className="space-y-3">
          <h4 className="font-medium text-gray-900">Version Candidates:</h4>
          {versionCandidates.map((candidate) => (
            <div key={candidate.id} className="flex items-center justify-between bg-gray-50 rounded p-3 border">
              <div className="flex items-center space-x-3">
                <FileText className="h-5 w-5 text-gray-400" />
                <div>
                  <p className="font-medium text-gray-900">{candidate.filename}</p>
                  <p className="text-sm text-gray-500">
                    {candidate.total_rows} rows • {candidate.total_columns} columns • 
                    {new Date(candidate.created_at).toLocaleDateString()}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {recommendation && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-start space-x-3">
            <CheckCircle className="h-6 w-6 text-blue-500 mt-0.5" />
            <div className="flex-1">
              <h4 className="font-medium text-blue-900 mb-2">AI Recommendation</h4>
              <p className="text-blue-800 mb-3">{recommendation.reasoning}</p>
              
              <div className="bg-white rounded p-3 border border-blue-200">
                <p className="font-medium text-gray-900">{recommendation.recommended_version.filename}</p>
                <div className="flex items-center space-x-4 mt-1 text-sm text-gray-600">
                  <span>{recommendation.recommended_version.row_count} rows</span>
                  <span>{recommendation.recommended_version.column_count} columns</span>
                  <span className="flex items-center space-x-1">
                    <BarChart3 className="h-4 w-4" />
                    <span>{Math.round(recommendation.confidence * 100)}% confidence</span>
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="space-y-3">
        <h4 className="font-medium text-gray-900">Do you accept this recommendation?</h4>
        
        <div className="flex space-x-3">
          <button
            onClick={() => onVersionAccept(true)}
            className="flex-1 flex items-center justify-center space-x-2 px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <CheckCircle className="h-5 w-5" />
            <span>Accept Recommendation</span>
          </button>
          
          <button
            onClick={() => setShowFeedback(true)}
            className="flex-1 flex items-center justify-center space-x-2 px-4 py-3 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
          >
            <XCircle className="h-5 w-5" />
            <span>Provide Feedback</span>
          </button>
        </div>

        {showFeedback && (
          <div className="space-y-3 p-4 bg-gray-50 rounded-lg">
            <label className="block text-sm font-medium text-gray-700">
              Why don't you agree with this recommendation?
            </label>
            <textarea
              value={feedback}
              onChange={(e) => setFeedback(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              rows={3}
              placeholder="Your feedback helps us improve our recommendations..."
            />
            <div className="flex space-x-2">
              <button
                onClick={() => onVersionAccept(false, feedback)}
                className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition-colors"
              >
                Submit Feedback
              </button>
              <button
                onClick={() => setShowFeedback(false)}
                className="px-4 py-2 border border-gray-300 text-gray-700 rounded hover:bg-gray-50 transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 animate-fade-in">
      <div className="bg-card border border-border rounded-2xl shadow-2xl max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto animate-scale-in">
        <div className="p-8">
          {phase === 'basic_duplicate' && renderBasicDuplicate()}
          {phase === 'versions_detected' && renderVersionDetection()}
          
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
