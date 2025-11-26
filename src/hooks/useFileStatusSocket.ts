import { useEffect, useRef, useCallback, useState } from 'react';
import { io, Socket } from 'socket.io-client';
import { useFileStatusStore, ProcessingStep } from '@/stores/useFileStatusStore';
import { config } from '@/config';

/**
 * WebSocket Hook for File Processing Status
 * 
 * Connects to backend Socket.IO server and listens for real-time file processing updates.
 * Automatically updates the global file status store.
 * 
 * Features:
 * - Auto-connect/disconnect on mount/unmount
 * - Handles connection errors gracefully
 * - Supports multiple concurrent file uploads
 * - Automatic reconnection with exponential backoff
 * - FIX: Proper connection status state management (connecting|connected|failed|polling)
 */
export const useFileStatusSocket = (userId?: string, sessionToken?: string) => {
  const socketRef = useRef<Socket | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttemptsRef = useRef(config.websocket.reconnectAttempts);
  
  // FIX: Add connection status state (not just boolean)
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'failed' | 'polling'>('connecting');
  
  const addStep = useFileStatusStore((state) => state.addStep);
  const updateProgress = useFileStatusStore((state) => state.updateProgress);
  const setError = useFileStatusStore((state) => state.setError);
  const markComplete = useFileStatusStore((state) => state.markComplete);

  /**
   * Handle incoming file progress update from backend
   */
  const handleFileProgress = useCallback(
    (data: any) => {
      const {
        job_id,
        step,
        message,
        progress,
        status,
        timestamp,
        ...extra
      } = data;

      // ISSUE #2 FIX: Standardize on job_id (backend now includes it in payload)
      // Removed fileId/jobId fallback - use only job_id for consistency
      if (!job_id) {
        console.warn('Received progress update without job_id', data);
        return;
      }
      const id = job_id;

      // Create processing step
      const processingStep: ProcessingStep = {
        step,
        message,
        status: status || 'in_progress',
        timestamp: timestamp ? new Date(timestamp).getTime() : Date.now(),
        progress: progress || 0,
        extra: Object.keys(extra).length > 0 ? extra : undefined,
      };

      // Update store
      addStep(id, processingStep);
      updateProgress(id, progress || 0);

      // Log for debugging
      console.log(`📊 File Progress [${id}]: ${step} - ${message} (${progress}%)`);
    },
    [addStep, updateProgress]
  );

  /**
   * Handle job completion
   */
  const handleJobComplete = useCallback(
    (data: any) => {
      const { fileId, jobId, result } = data;
      const id = fileId || jobId;

      if (!id) return;

      markComplete(id);
      console.log(`✅ File Processing Complete [${id}]`, result);
    },
    [markComplete]
  );

  /**
   * Handle job error
   */
  const handleJobError = useCallback(
    (data: any) => {
      const { fileId, jobId, error, message } = data;
      const id = fileId || jobId;

      if (!id) return;

      const errorMessage = error || message || 'Unknown error';
      setError(id, errorMessage);
      console.error(`❌ File Processing Error [${id}]:`, errorMessage);
    },
    [setError]
  );

  /**
   * Initialize WebSocket connection
   */
  useEffect(() => {
    if (!userId || !sessionToken) {
      console.warn('useFileStatusSocket: Missing userId or sessionToken');
      return;
    }

    try {
      // Build WebSocket URL with auth parameters
      const wsUrl = config.wsUrl || `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}`;
      
      // Create Socket.IO client with auth
      const socket = io(wsUrl, {
        query: {
          user_id: userId,
          session_token: sessionToken,
        },
        reconnection: true,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 5000,
        reconnectionAttempts: 5,
        transports: ['websocket', 'polling'],
      });

      // Connection event handlers
      socket.on('connect', () => {
        console.log('✅ File Status WebSocket Connected');
        setConnectionStatus('connected');
        reconnectAttemptsRef.current = 0; // Reset attempts on successful connection
      });

      socket.on('disconnect', (reason) => {
        console.log(`⚠️ File Status WebSocket Disconnected: ${reason}`);
        // If disconnected due to server error, mark as failed after max attempts
        if (reason === 'io server disconnect' || reason === 'transport close') {
          setConnectionStatus('failed');
        }
      });

      socket.on('connect_error', (error) => {
        console.error('❌ File Status WebSocket Connection Error:', error);
        reconnectAttemptsRef.current += 1;
        
        // FIX: After max reconnect attempts, give up and use polling
        if (reconnectAttemptsRef.current >= maxReconnectAttemptsRef.current) {
          console.warn(`❌ WebSocket reconnection failed after ${maxReconnectAttemptsRef.current} attempts - falling back to polling`);
          setConnectionStatus('polling');
          // Emit event so UI can show "Using polling" instead of "Reconnecting..."
          window.dispatchEvent(new CustomEvent('connection_failed_using_polling', { detail: { attempts: reconnectAttemptsRef.current } }));
        } else {
          setConnectionStatus('connecting');
        }
      });

      // File processing event handlers
      // BUG #1 FIX: Use only 'job_update' (backend standardized on this)
      // Removed duplicate 'file_progress' listener - was redundant
      socket.on('job_update', (data) => {
        handleFileProgress(data);
        // ERROR #4 FIX: Emit window events for polling replacement
        window.dispatchEvent(new CustomEvent('job_update', { detail: data }));
      });
      socket.on('job_complete', (data) => {
        handleJobComplete(data);
        // Emit event for DataSourcesPanel to refresh
        window.dispatchEvent(new CustomEvent('job_complete', { detail: data }));
      });
      socket.on('job_error', (data) => {
        handleJobError(data);
        window.dispatchEvent(new CustomEvent('job_error', { detail: data }));
      });

      socketRef.current = socket;

      // Cleanup on unmount
      return () => {
        socket.off('file_progress', handleFileProgress);
        socket.off('job_update', handleFileProgress);
        socket.off('job_complete', handleJobComplete);
        socket.off('job_error', handleJobError);
        socket.disconnect();
      };
    } catch (error) {
      console.error('Failed to initialize file status WebSocket:', error);
    }
  }, [userId, sessionToken, handleFileProgress, handleJobComplete, handleJobError]);

  /**
   * Emit event to backend (if needed)
   */
  const emit = useCallback(
    (event: string, data: any) => {
      if (socketRef.current?.connected) {
        socketRef.current.emit(event, data);
      } else {
        console.warn(`Cannot emit '${event}': WebSocket not connected`);
      }
    },
    []
  );

  /**
   * Check if WebSocket is connected
   */
  const isConnected = connectionStatus === 'connected';

  return {
    isConnected,
    connectionStatus, // FIX: Expose full status for UI
    emit,
    socket: socketRef.current,
  };
};
