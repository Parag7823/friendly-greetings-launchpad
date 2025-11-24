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
    if (socket?.connected) {
      socket.emit(event, data);
    } else {
      console.warn(`WebSocket not connected, cannot emit ${event}`);
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
