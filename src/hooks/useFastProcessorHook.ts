/**
 * useFastProcessorHook - Unified FastAPI Processing Hook
 * 
 * Consolidates all FastAPI processing logic including:
 * - WebSocket connection and reconnection
 * - Polling fallback with exponential backoff
 * - Progress tracking
 * - Error handling
 * 
 * Fixes anti-pattern of direct polling in FastAPIProcessor.tsx
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
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentProgress, setCurrentProgress] = useState<ProcessingProgress | null>(null);
  const wsCleanupFlags = useRef<Map<string, boolean>>(new Map());

  /**
   * Connect to WebSocket for real-time processing updates
   */
  const connectWebSocket = useCallback(
    (jobId: string): Promise<ProcessingResult> => {
      return new Promise((resolve, reject) => {
        const wsUrl = `${config.wsUrl}/ws/${jobId}`;
        const ws = new WebSocket(wsUrl);
        let timeoutId: NodeJS.Timeout;

        // Initialize cleanup flag
        wsCleanupFlags.current.set(jobId, false);

        const cleanup = () => {
          if (wsCleanupFlags.current.get(jobId)) return;
          wsCleanupFlags.current.set(jobId, true);

          clearTimeout(timeoutId);
          if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
            ws.close();
          }

          setTimeout(() => {
            wsCleanupFlags.current.delete(jobId);
          }, 1000);
        };

        // Set connection timeout (60 seconds for large files)
        timeoutId = setTimeout(() => {
          cleanup();
          reject(new Error('WebSocket connection timeout'));
        }, 60000);

        ws.onopen = () => {
          clearTimeout(timeoutId);
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);

            if (data.status === 'completed') {
              cleanup();
              resolve(data.result || data);
            } else if (data.status === 'error') {
              cleanup();
              reject(new Error(data.error || 'Processing failed'));
            } else {
              // Progress update
              const extra: any = {
                duplicate_info: data.duplicate_info,
                near_duplicate_info: data.near_duplicate_info,
                content_duplicate_info: data.content_duplicate_info,
                delta_analysis: data.delta_analysis,
                requires_user_decision: data.requires_user_decision,
              };
              setCurrentProgress({
                step: data.step || 'processing',
                message: data.message || 'Processing...',
                progress: data.progress || 0,
                sheetProgress: data.sheetProgress,
                extra,
                requires_user_decision: data.requires_user_decision,
              });
            }
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
          }
        };

        ws.onerror = (error) => {
          UnifiedErrorHandler.handle({
            message: 'WebSocket connection failed - using polling fallback',
            severity: ErrorSeverity.MEDIUM,
            source: ErrorSource.WEBSOCKET,
            retryable: true,
          });
          cleanup();
          reject(new Error('WebSocket connection failed - will use polling fallback'));
        };

        ws.onclose = (event) => {
          cleanup();
          if (event.code !== 1000 && event.reason !== 'Processing completed' && !wsCleanupFlags.current.get(jobId)) {
            attemptWebSocketReconnection(jobId, resolve, reject, 0);
          }
        };
      });
    },
    []
  );

  /**
   * Attempt WebSocket reconnection with exponential backoff
   */
  const attemptWebSocketReconnection = useCallback(
    async (
      jobId: string,
      resolve: (value: ProcessingResult) => void,
      reject: (reason?: any) => void,
      attemptNumber: number
    ) => {
      const MAX_RECONNECT_ATTEMPTS = config.websocket.reconnectAttempts;
      const BASE_DELAY = config.websocket.reconnectBaseDelay;

      if (attemptNumber >= MAX_RECONNECT_ATTEMPTS) {
        console.warn(`WebSocket reconnection failed after ${MAX_RECONNECT_ATTEMPTS} attempts - falling back to polling`);
        reject(new Error('WebSocket connection closed unexpectedly - will use polling fallback'));
        return;
      }

      const delay = BASE_DELAY * Math.pow(2, attemptNumber);
      console.log(`Attempting WebSocket reconnection ${attemptNumber + 1}/${MAX_RECONNECT_ATTEMPTS} in ${delay}ms...`);

      await new Promise((res) => setTimeout(res, delay));

      try {
        const result = await connectWebSocket(jobId);
        resolve(result);
      } catch (error) {
        console.warn(`WebSocket reconnection attempt ${attemptNumber + 1} failed:`, error);
        attemptWebSocketReconnection(jobId, resolve, reject, attemptNumber + 1);
      }
    },
    [connectWebSocket]
  );

  /**
   * Poll for job status as fallback when WebSocket fails
   * CRITICAL FIX: Consolidated polling logic from FastAPIProcessor.tsx
   */
  const pollForResults = useCallback(
    async (jobId: string, apiUrl: string): Promise<ProcessingResult> => {
      // Account for queue time + processing time
      // ARQ worker function_timeout = 900 seconds (15 minutes for processing)
      // But jobs can wait in queue before processing starts
      // Increased to 25 minutes total (15 min processing + 10 min queue buffer)
      const maxAttempts = 1000; // 25 minutes max (1.5 seconds * 1000 = 1500 seconds)
      let attempts = 0;

      while (attempts < maxAttempts) {
        try {
          // Check job status
          const statusResponse = await fetch(`${apiUrl}/job-status/${jobId}`);
          if (statusResponse.ok) {
            const statusData = await statusResponse.json();

            if (statusData.status === 'completed') {
              setCurrentProgress({
                step: 'complete',
                message: 'Processing completed!',
                progress: 100,
              });
              return statusData.result || statusData;
            } else if (statusData.status === 'failed') {
              throw new Error(statusData.error || 'Processing failed');
            } else if (statusData.status === 'cancelled') {
              throw new Error('Processing was cancelled');
            }

            // Show progress indication during polling
            if (statusData.progress !== undefined) {
              const timeElapsed = Math.floor((attempts * config.websocket.pollingInterval) / 1000);
              const progressMessage = statusData.message || `Processing... (${timeElapsed}s elapsed)`;
              setCurrentProgress({
                step: 'processing',
                message: progressMessage,
                progress: statusData.progress,
              });
            }
          }

          // Use configurable polling interval from config
          await new Promise((res) => setTimeout(res, config.websocket.pollingInterval));
          attempts++;
        } catch (error) {
          console.error('Polling error:', error);
          attempts++;
          await new Promise((res) => setTimeout(res, 2000)); // Wait 2 seconds on error
        }
      }

      throw new Error(
        'Processing is taking longer than expected. The job may still be running in the background. Please refresh the page or try again later.'
      );
    },
    []
  );

  /**
   * Main processing function that handles WebSocket with polling fallback
   */
  const processWithFallback = useCallback(
    async (jobId: string, apiUrl: string): Promise<ProcessingResult> => {
      setIsProcessing(true);
      try {
        setCurrentProgress({
          step: 'websocket',
          message: 'Connecting to real-time updates...',
          progress: 35,
        });

        let result: ProcessingResult;
        try {
          // Try WebSocket connection with timeout (60 seconds)
          result = await Promise.race([
            connectWebSocket(jobId),
            new Promise<ProcessingResult>((_, reject) =>
              setTimeout(() => reject(new Error('WebSocket timeout')), 60000)
            ),
          ]);
        } catch (wsError) {
          console.warn('WebSocket failed, falling back to polling:', wsError);
          setCurrentProgress({
            step: 'polling',
            message: 'Using polling for updates...',
            progress: 40,
          });

          // Fall back to polling
          result = await pollForResults(jobId, apiUrl);
        }

        return result;
      } finally {
        setIsProcessing(false);
      }
    },
    [connectWebSocket, pollForResults]
  );

  return {
    isProcessing,
    currentProgress,
    processWithFallback,
    connectWebSocket,
    pollForResults,
  };
};
