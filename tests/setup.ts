import { expect, afterEach, vi } from 'vitest';
import { cleanup } from '@testing-library/react';
import '@testing-library/jest-dom';

// Cleanup after each test
afterEach(() => {
  cleanup();
});

// Mock crypto.subtle for hash calculations (if not already available)
if (!global.crypto?.subtle) {
  Object.defineProperty(global, 'crypto', {
    value: {
      subtle: {
        digest: async (algorithm: string, data: ArrayBuffer) => {
          // Simple mock implementation
          const hash = new Uint8Array(32); // 32 bytes for SHA-256
          for (let i = 0; i < 32; i++) {
            hash[i] = Math.floor(Math.random() * 256);
          }
          return hash.buffer;
        },
      },
    },
    writable: true,
    configurable: true,
  });
}

// Mock WebSocket
global.WebSocket = class MockWebSocket {
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;

  constructor(public url: string) {
    setTimeout(() => {
      if (this.onopen) {
        this.onopen(new Event('open'));
      }
    }, 0);
  }

  send(data: string) {
    // Mock send
  }

  close() {
    if (this.onclose) {
      this.onclose(new CloseEvent('close', { code: 1000 }));
    }
  }
} as any;
