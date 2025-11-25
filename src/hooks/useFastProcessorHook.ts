/**
 * useFastProcessorHook - DEPRECATED
 * 
 * This hook is no longer used. File processing now uses:
 * - Socket.IO for real-time updates (initialized in FinleyLayout)
 * - Fallback polling in FastAPIProcessor.waitForJobCompletion()
 * 
 * Kept for backward compatibility but can be removed in future cleanup.
 * 
 * ISSUE #2 FIX: Removed custom polling and WebSocket logic
 * - Polling is now handled in FastAPIProcessor.waitForJobCompletion()
 * - WebSocket is handled by useFileStatusSocket (Socket.IO)
 * - This hook is no longer needed
 */

import { useCallback, useRef, useState } from 'react';
import { config } from '@/config';
import { UnifiedErrorHandler, ErrorSeverity, ErrorSource } from '@/utils/errorHandler';

export interface ProcessingProgress {
  step: string;
  message: string;
  progress: number;
  status?: 'processing' | 'completed' | 'error' | 'failed';
  sheetProgress?: {
    currentSheet: string;
    sheetsCompleted: number;
    totalSheets: number;
  };
  extra?: any;
  requires_user_decision?: boolean;
}

export interface ProcessingResult {
  status: string;
  result?: any;
  error?: string;
  [key: string]: any;
}

export const useFastProcessorHook = () => {
  // ISSUE #2 FIX: Simplified hook - no longer handles WebSocket or polling
  // These are now handled by:
  // - useFileStatusSocket (Socket.IO) in FinleyLayout
  // - FastAPIProcessor.waitForJobCompletion() (polling fallback)
  
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentProgress, setCurrentProgress] = useState<ProcessingProgress | null>(null);

  // Stub methods for backward compatibility (no-op)
  const connectWebSocket = useCallback(async (jobId: string): Promise<ProcessingResult> => {
    throw new Error('connectWebSocket is deprecated. Use Socket.IO via useFileStatusSocket instead.');
  }, []);

  const pollForResults = useCallback(async (jobId: string, apiUrl: string): Promise<ProcessingResult> => {
    throw new Error('pollForResults is deprecated. Use FastAPIProcessor.waitForJobCompletion instead.');
  }, []);

  const processWithFallback = useCallback(async (jobId: string, apiUrl: string): Promise<ProcessingResult> => {
    throw new Error('processWithFallback is deprecated. Use FastAPIProcessor.processFile instead.');
  }, []);

  return {
    isProcessing,
    currentProgress,
    processWithFallback,
    connectWebSocket,
    pollForResults,
  };
};
