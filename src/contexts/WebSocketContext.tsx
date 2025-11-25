import React, { createContext, useContext } from 'react';
import { Socket } from 'socket.io-client';

interface WebSocketContextType {
  socket: Socket | null;
  isConnected: boolean;
  emit: (event: string, data?: any) => void;
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

export const WebSocketProvider: React.FC<{ children: React.ReactNode; socket: Socket | null; isConnected: boolean }> = ({
  children,
  socket,
  isConnected,
}) => {
  const emit = (event: string, data?: any) => {
    // FIX #11: Check proper socket state (not just connected flag)
    // Socket can be in states: connecting (0), open/connected (1), closing (2), closed (3)
    if (socket && socket.connected && socket.readyState === 1) {
      socket.emit(event, data);
    } else {
      const state = socket?.readyState ?? -1;
      const stateNames = { 0: 'connecting', 1: 'open', 2: 'closing', 3: 'closed' };
      const stateName = stateNames[state as keyof typeof stateNames] || 'unknown';
      console.warn(`WebSocket not ready (state: ${stateName}), cannot emit ${event}`);
    }
  };

  return (
    <WebSocketContext.Provider value={{ socket, isConnected, emit }}>
      {children}
    </WebSocketContext.Provider>
  );
};

export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within WebSocketProvider');
  }
  return context;
};
