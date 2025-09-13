import { useCallback, useRef, useEffect } from 'react';
import { useToast } from '@/hooks/use-toast';

interface WebSocketProgressData {
  step: string;
  message: string;
  progress: number;
  sheetProgress?: {
    currentSheet: string;
    sheetsCompleted: number;
    totalSheets: number;
  };
  status?: 'processing' | 'completed' | 'error';
  error?: string;
}

interface UseWebSocketProgressOptions {
  onProgress?: (data: WebSocketProgressData) => void;
  onComplete?: (data: any) => void;
  onError?: (error: string) => void;
}

export const useWebSocketProgress = (options: UseWebSocketProgressOptions = {}) => {
  const { toast } = useToast();
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttempts = 5;

  const connectWebSocket = useCallback((jobId: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return; // Already connected
    }

    const wsUrl = `wss://friendly-greetings-launchpad.onrender.com/ws/${jobId}`;
    console.log(`Connecting to WebSocket: ${wsUrl}`);

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected for job:', jobId);
      reconnectAttemptsRef.current = 0;
      
      toast({
        title: "Connected",
        description: "Real-time updates enabled",
      });
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('WebSocket message received:', data);

        if (data.status === 'completed') {
          options.onComplete?.(data);
          ws.close();
        } else if (data.status === 'error') {
          options.onError?.(data.error || 'Processing failed');
          ws.close();
        } else {
          // Progress update
          options.onProgress?.(data);
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      toast({
        title: "Connection Error",
        description: "Real-time updates interrupted",
        variant: "destructive",
      });
    };

    ws.onclose = (event) => {
      console.log('WebSocket closed:', event.code, event.reason);
      
      // Attempt reconnection if not manually closed and within retry limit
      if (event.code !== 1000 && reconnectAttemptsRef.current < maxReconnectAttempts) {
        const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 10000);
        console.log(`Attempting to reconnect in ${delay}ms...`);
        
        reconnectTimeoutRef.current = setTimeout(() => {
          reconnectAttemptsRef.current++;
          connectWebSocket(jobId);
        }, delay);
      }
    };

    return ws;
  }, [options, toast]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    if (wsRef.current) {
      wsRef.current.close(1000, 'Manual disconnect');
      wsRef.current = null;
    }
    
    reconnectAttemptsRef.current = 0;
  }, []);

  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    connectWebSocket,
    disconnect,
    isConnected: wsRef.current?.readyState === WebSocket.OPEN
  };
};