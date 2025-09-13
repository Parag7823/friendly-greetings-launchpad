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
  similarity?: number;
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
  onDecision: (decision: 'replace' | 'keep_both' | 'skip') => void;
  onVersionAccept: (accepted: boolean, feedback?: string) => void;
  phase: 'basic_duplicate' | 'versions_detected' | 'similar_files';
}

export const DuplicateDetectionModal: React.FC<DuplicateDetectionModalProps> = ({
  isOpen,
  onClose,
  duplicateInfo,
  versionCandidates,
  recommendation,
  onDecision,
  onVersionAccept,
  phase
}) => {
  const [feedback, setFeedback] = useState('');
  const [showFeedback, setShowFeedback] = useState(false);

  if (!isOpen) return null;

  const renderBasicDuplicate = () => (
    <div className="space-y-6">
      <div className="flex items-center space-x-3">
        <AlertTriangle className="h-8 w-8 text-amber-500" />
        <div>
          <h3 className="text-lg font-semibold text-gray-900">Identical File Detected</h3>
          <p className="text-sm text-gray-600">This exact file has been uploaded before</p>
        </div>
      </div>

      {duplicateInfo && (
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
          <p className="text-amber-800 mb-3">{duplicateInfo.message}</p>
          
          <div className="space-y-2">
            {duplicateInfo.duplicate_files.map((file) => (
              <div key={file.id} className="flex items-center justify-between bg-white rounded p-3 border">
                <div className="flex items-center space-x-3">
                  <FileText className="h-5 w-5 text-gray-400" />
                  <div>
                    <p className="font-medium text-gray-900">{file.filename}</p>
                    <p className="text-sm text-gray-500">
                      Uploaded {new Date(file.uploaded_at).toLocaleDateString()} • {file.total_rows} rows
                    </p>
                  </div>
                </div>
                <span className={`px-2 py-1 rounded text-xs font-medium ${
                  file.status === 'completed' ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
                }`}>
                  {file.status}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="space-y-3">
        <h4 className="font-medium text-gray-900">What would you like to do?</h4>
        
        <button
          onClick={() => onDecision('replace')}
          className="w-full flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
        >
          <div className="flex items-center space-x-3">
            <CheckCircle className="h-5 w-5 text-green-500" />
            <div className="text-left">
              <p className="font-medium text-gray-900">Replace existing file</p>
              <p className="text-sm text-gray-600">Archive the old version and process this new one</p>
            </div>
          </div>
        </button>

        <button
          onClick={() => onDecision('keep_both')}
          className="w-full flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
        >
          <div className="flex items-center space-x-3">
            <Info className="h-5 w-5 text-blue-500" />
            <div className="text-left">
              <p className="font-medium text-gray-900">Keep both files</p>
              <p className="text-sm text-gray-600">Process this file alongside the existing one</p>
            </div>
          </div>
        </button>

        <button
          onClick={() => onDecision('skip')}
          className="w-full flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
        >
          <div className="flex items-center space-x-3">
            <XCircle className="h-5 w-5 text-red-500" />
            <div className="text-left">
              <p className="font-medium text-gray-900">Skip this upload</p>
              <p className="text-sm text-gray-600">Cancel processing and keep the existing file</p>
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
                    {candidate.similarity && ` • ${Math.round(candidate.similarity * 100)}% content overlap`}
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
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          {phase === 'basic_duplicate' && renderBasicDuplicate()}
          {phase === 'versions_detected' && renderVersionDetection()}
          
          <div className="mt-6 pt-4 border-t border-gray-200">
            <button
              onClick={onClose}
              className="w-full px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
            >
              Cancel Upload
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
